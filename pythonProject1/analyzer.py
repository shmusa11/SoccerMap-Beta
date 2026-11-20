# football_analyzer/analyzer.py

import json
from collections import defaultdict, deque, Counter
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from inference import get_model
from tqdm import tqdm
import torch

from data_models import PlayerData, BallData, FrameAnalytics
from team_classifier import EnhancedTeamClassifier
from pitch_mapper import DynamicPitchMapper
from player_manager import OptimizedPlayerManager


class OptimizedFootballAnalyzer:
    def __init__(self, roboflow_api_key=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not roboflow_api_key: raise ValueError("Roboflow API key is required.")

        field_model_id = "football-field-mdvk6/1"

        print(f"Loading player model...")
        self.player_model = get_model(model_id="soccer-player-ball-tracker-bechj/4", api_key=roboflow_api_key)
        print(f"Loading universal field model: {field_model_id}")
        self.field_model = get_model(model_id=field_model_id, api_key=roboflow_api_key)

        self.tracker = sv.ByteTrack(frame_rate=30, track_activation_threshold=0.25)
        self.team_classifier = EnhancedTeamClassifier(device=self.device)
        self.pitch_mapper = DynamicPitchMapper()
        self.pitch_mapper.set_field_model(self.field_model)
        self.player_manager = OptimizedPlayerManager(self.pitch_mapper)
        self.frame_analytics = []
        self.setup_annotators()

    def setup_annotators(self):
        team_colors = [sv.Color.from_hex("#FF0000"), sv.Color.from_hex("#FFFF00"),
                       sv.Color(r=128, g=128, b=128)]
        self.color_palette = sv.ColorPalette(colors=team_colors)
        self.player_annotators = {
            "box": sv.BoundingBoxAnnotator(color=self.color_palette),
            "label": sv.LabelAnnotator(color=self.color_palette, text_color=sv.Color.BLACK,
                                       text_position=sv.Position.TOP_CENTER),
        }
        self.ball_annotator = sv.EllipseAnnotator(color=sv.Color.GREEN)

    def process_frame(self, frame):
        bgr, hsv = self.team_classifier.preprocess_frame(frame)
        detections = self.detect_players_and_ball(bgr)
        tracked_detections = self.tracker.update_with_detections(detections)
        is_map_stable = self.pitch_mapper.update_homography(frame)
        self.player_manager.update(bgr, tracked_detections, self.team_classifier, hsv)
        players = self.player_manager.get_player_data_list()
        ball_detections = tracked_detections[tracked_detections.class_id == 0]
        ball = None
        if len(ball_detections) > 0:
            best_idx = np.argmax(ball_detections.confidence)
            best_ball = ball_detections[best_idx:best_idx + 1]
            bbox_coords = best_ball.xyxy[0]
            position = ((bbox_coords[0] + bbox_coords[2]) / 2, bbox_coords[3])
            ball = BallData(id=0, bbox=tuple(bbox_coords.astype(int)), position=position,
                            confidence=best_ball.confidence[0])
        if is_map_stable:
            player_positions = [p.position for p in players if p.position]
            if player_positions:
                coords = self.pitch_mapper.transform_to_pitch(player_positions)
                for player, coord in zip(players, coords):
                    if coord is not None: player.field_position = tuple(coord)
            if ball and ball.position:
                ball_coord = self.pitch_mapper.transform_to_pitch([ball.position])
                if len(ball_coord) > 0: ball.field_position = tuple(ball_coord[0])

        # --- NEW: Get landmark data for visualization ---
        landmark_positions = self.pitch_mapper.last_used_image_points if is_map_stable else None

        return FrameAnalytics(len(self.frame_analytics), players, ball, None, {},
                              detected_landmarks_screen_positions=landmark_positions)

    # ... Other functions are unchanged ...
    def detect_players_and_ball(self, frame):
        result = self.player_model.infer(frame, confidence=0.15)[0]
        detections = sv.Detections.from_inference(result)
        detections.class_id[detections.class_id == 1] = 2
        return detections

    def train_team_classifier_and_assign_squads(self, video_path):
        cap = cv2.VideoCapture(video_path)
        for _ in range(40):
            ret, frame = cap.read()
            if not ret: break
            bgr, hsv = self.team_classifier.preprocess_frame(frame)
            detections = self.detect_players_and_ball(bgr)
            tracked_detections = self.tracker.update_with_detections(detections)
            player_detections = tracked_detections[tracked_detections.class_id == 2]
            if len(player_detections) > 12:
                print(f"Found setup frame with {len(player_detections)} players.")
                if self.team_classifier.fit_automatically(frame, player_detections, hsv):
                    self.player_manager.initial_assignment(frame, player_detections, self.team_classifier, hsv)
                    cap.release()
                    return True
        cap.release()
        print("Setup failed: Could not classify teams.")
        return False

    def analyze_video(self, video_path, max_frames=None, output_path='combined_output.mp4'):
        if not self.train_team_classifier_and_assign_squads(video_path):
            return None
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames: total = min(total, max_frames)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (w, h))
        for _ in tqdm(range(total), desc="Analyzing Video"):
            ret, frame = cap.read()
            if not ret: break
            analytics = self.process_frame(frame)
            self.frame_analytics.append(analytics)
            annotated_frame = self._annotate_frame(frame.copy(), analytics)
            diagnostic_frame = self._draw_field_diagnostics(annotated_frame)
            final_frame = self.pitch_mapper.get_field_overlay(diagnostic_frame)
            pitch_view = self.draw_2d_pitch_view(analytics)
            combined_frame = self._create_combined_frame(final_frame, pitch_view, w, h)
            writer.write(combined_frame)
        writer.release()
        cap.release()
        print(f"Analysis complete! Output saved to {output_path}")
        return self.frame_analytics

    def _draw_field_diagnostics(self, frame):
        if self.pitch_mapper.last_raw_predictions:
            for pred in self.pitch_mapper.last_raw_predictions:
                x1, y1 = int(pred.x - pred.width / 2), int(pred.y - pred.height / 2)
                x2, y2 = int(pred.x + pred.width / 2), int(pred.y + pred.height / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                label = f"ID: {pred.class_name} ({pred.confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        return frame

    def _annotate_frame(self, frame, analytics):
        if analytics.players:
            player_detections = sv.Detections(
                xyxy=np.array([p.bbox for p in analytics.players]),
                class_id=np.array([p.team if p.team in [0, 1] else 2 for p in analytics.players])
            )
            labels = [f"T{p.team} #{p.id}" for p in analytics.players]
            frame = self.player_annotators["box"].annotate(frame, player_detections)
            frame = self.player_annotators["label"].annotate(frame, player_detections, labels=labels)
        if analytics.ball:
            ball_det = sv.Detections(
                xyxy=np.array([analytics.ball.bbox]),
                class_id=np.array([0])
            )
            frame = self.ball_annotator.annotate(frame, ball_det)
        return frame

    def _create_combined_frame(self, main_frame, pitch_view, w, h):
        pv_h, pv_w = pitch_view.shape[:2]
        new_w = int(w * 0.3)
        new_h = int(pv_h * (new_w / pv_w))
        resized_pitch = cv2.resize(pitch_view, (new_w, new_h))
        y_off, x_off = 20, w - new_w - 20
        main_frame[y_off:y_off + new_h, x_off:x_off + new_w] = resized_pitch
        return main_frame

    def draw_2d_pitch_view(self, analytics):
        w, h = 420, 272
        pitch_img = np.full((h, w, 3), (0, 120, 0), dtype=np.uint8)
        self._draw_pitch_markings(pitch_img, w, h)
        self._draw_ground_truth_landmarks(pitch_img, w, h)

        if self.pitch_mapper.is_tracking_stable():
            # --- NEW: Draw the translated detected landmarks ---
            if analytics.detected_landmarks_screen_positions:
                transformed_landmarks = self.pitch_mapper.transform_to_pitch(
                    analytics.detected_landmarks_screen_positions)
                for point in transformed_landmarks:
                    x = int(point[0] * (w / self.pitch_mapper.pitch_width))
                    y = int(point[1] * (h / self.pitch_mapper.pitch_height))
                    if 0 <= x < w and 0 <= y < h:
                        # Draw as cyan circles to compare against magenta ground-truth
                        cv2.circle(pitch_img, (x, y), 6, (255, 255, 0), -1)  # BGR for Cyan

            # Draw players
            team_colors = {0: (0, 0, 255), 1: (0, 255, 255)}  # BGR: Red, Yellow
            for p in analytics.players:
                if p.field_position:
                    x = int(p.field_position[0] * (w / self.pitch_mapper.pitch_width))
                    y = int(p.field_position[1] * (h / self.pitch_mapper.pitch_height))
                    if 0 <= x < w and 0 <= y < h:
                        cv2.circle(pitch_img, (x, y), 8, team_colors.get(p.team, (128, 128, 128)), -1)
                        cv2.circle(pitch_img, (x, y), 8, (0, 0, 0), 1)

                        label = f"{p.id}"
                        font_scale = 0.6;
                        font_thickness = 2
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                        text_x = x - text_size[0] // 2;
                        text_y = y - 12
                        cv2.putText(pitch_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            # Draw ball
            if analytics.ball and analytics.ball.field_position:
                x = int(analytics.ball.field_position[0] * (w / self.pitch_mapper.pitch_width))
                y = int(analytics.ball.field_position[1] * (h / self.pitch_mapper.pitch_height))
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(pitch_img, (x, y), 6, (255, 255, 255), -1)
        else:
            cv2.putText(pitch_img, "MAP OFFLINE", (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2)
        return pitch_img

    def _draw_ground_truth_landmarks(self, img, w, h):
        for name, (mx, my, mz) in self.pitch_mapper.PITCH_3D_MODEL.items():
            px = int(mx * (w / self.pitch_mapper.pitch_width))
            py = int(my * (h / self.pitch_mapper.pitch_height))
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(img, (px, py), 4, (255, 0, 255), -1)  # BGR for Magenta

    def _draw_pitch_markings(self, img, w, h, color=(255, 255, 255)):
        sx, sy = w / 105.0, h / 68.0
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 1)
        cv2.line(img, (w // 2, 0), (w // 2, h), color, 1)
        cv2.circle(img, (w // 2, h // 2), int(9.15 * sy), color, 1)
        cv2.rectangle(img, (0, int((h - 40.32 * sy) / 2)), (int(16.5 * sx), int((h + 40.32 * sy) / 2)), color, 1)
        cv2.rectangle(img, (w - int(16.5 * sx), int((h - 40.32 * sy) / 2)), (w, int((h + 40.32 * sy) / 2)), color, 1)

    def save_analytics(self, filename="football_analytics.json"):
        print(f"Analytics would be saved to {filename}")

    def generate_match_report(self):
        return "Match report would be generated here."