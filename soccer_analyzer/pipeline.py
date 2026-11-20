"""
Main analysis pipeline.

Orchestrates: detection → tracking → team classification → pitch mapping → visualization.
Uses manual calibration for pitch mapping (no field landmark model).
"""

import numpy as np
import cv2
import supervision as sv
from inference import get_model
from tqdm import tqdm

from soccer_analyzer import config
from soccer_analyzer.models import BallData, FrameAnalytics
from soccer_analyzer.team_classifier import TeamClassifier
from soccer_analyzer.calibration.mapper import PitchMapper
from soccer_analyzer.player_manager import PlayerManager


class FootballAnalyzer:
    def __init__(
        self,
        calibration_path=None,
        detector=None,
        roboflow_api_key=None,
        team_a_color=None,
        team_b_color=None,
        smooth=False,
    ):
        api_key = roboflow_api_key or config.ROBOFLOW_API_KEY
        if not api_key:
            raise ValueError(
                "Roboflow API key required. Set ROBOFLOW_API_KEY env var or pass to constructor."
            )

        # Only one model needed now (no field landmark model)
        print("Loading player detection model...")
        self.player_model = get_model(model_id=config.PLAYER_MODEL_ID, api_key=api_key)

        self.tracker = sv.ByteTrack(
            frame_rate=config.TRACKER_FRAME_RATE,
            track_activation_threshold=config.TRACKER_ACTIVATION_THRESHOLD,
        )
        self.team_classifier = TeamClassifier(team_a_color, team_b_color)
        # Automatic calibration. If a calibration JSON is given, use it as a bridge
        # (CachedHomographyDetector); otherwise the mapper falls back to the automatic
        # PnLCalib detector. See soccer_analyzer/calibration/.
        self.pitch_mapper = PitchMapper(
            detector=detector, calibration_path=calibration_path, smooth=smooth
        )
        self.player_manager = PlayerManager()
        self.frame_analytics = []
        self._setup_annotators()

    def _setup_annotators(self):
        team_colors = [
            sv.Color.from_hex("#FF0000"),  # Team 0
            sv.Color.from_hex("#FFFF00"),  # Team 1
            sv.Color(r=128, g=128, b=128),  # Unknown
        ]
        palette = sv.ColorPalette(colors=team_colors)
        self.box_annotator = sv.BoundingBoxAnnotator(color=palette)
        self.label_annotator = sv.LabelAnnotator(
            color=palette,
            text_color=sv.Color.BLACK,
            text_position=sv.Position.TOP_CENTER,
        )
        self.ball_annotator = sv.EllipseAnnotator(color=sv.Color.GREEN)

    # --- Core pipeline ---

    def _detect(self, frame):
        """Detect players and ball. Remap goalkeeper to player class."""
        result = self.player_model.infer(frame, confidence=config.PLAYER_CONFIDENCE)[0]
        detections = sv.Detections.from_inference(result)
        detections.class_id[detections.class_id == 1] = 2  # goalkeeper -> player
        return detections

    def process_frame(self, frame):
        """Process a single frame through the full pipeline."""
        bgr, lab = self.team_classifier.preprocess_frame(frame)

        # Detect and track
        detections = self._detect(bgr)
        tracked = self.tracker.update_with_detections(detections)

        # Update pitch mapping (optical flow)
        self.pitch_mapper.update(frame)

        # Update player identities and teams
        self.player_manager.update(bgr, tracked, self.team_classifier, lab)
        players = self.player_manager.get_active_players()

        # Ball detection
        ball = self._get_ball(tracked)

        # Map positions to pitch coordinates
        self._map_to_pitch(players, ball)

        return FrameAnalytics(
            frame_number=len(self.frame_analytics),
            players=players,
            ball=ball,
            team_possession=None,
        )

    def _get_ball(self, tracked):
        """Extract the best ball detection."""
        ball_dets = tracked[tracked.class_id == 0]
        if len(ball_dets) == 0:
            return None
        best = np.argmax(ball_dets.confidence)
        bbox = ball_dets.xyxy[best]
        position = ((bbox[0] + bbox[2]) / 2, bbox[3])
        return BallData(
            id=0,
            bbox=tuple(bbox.astype(int)),
            position=position,
            confidence=float(ball_dets.confidence[best]),
        )

    def _map_to_pitch(self, players, ball):
        """Transform screen positions to pitch coordinates."""
        positions = [p.position for p in players if p.position]
        if positions:
            pitch_coords = self.pitch_mapper.transform(positions)
            for player, coord in zip(players, pitch_coords):
                if self.pitch_mapper.is_position_on_pitch(coord):
                    player.field_position = tuple(coord)

        if ball and ball.position:
            ball_coords = self.pitch_mapper.transform([ball.position])
            if len(ball_coords) > 0 and self.pitch_mapper.is_position_on_pitch(ball_coords[0]):
                ball.field_position = tuple(ball_coords[0])

    def _setup_teams(self, video_path, setup_frames=60):
        """Auto-detect team colors from early frames (if not manually set)."""
        if self.team_classifier.is_fitted:
            print("Team colors set manually, skipping auto-detection.")
            return True

        print("Auto-detecting team colors...")
        cap = cv2.VideoCapture(video_path)
        for _ in range(setup_frames):
            ret, frame = cap.read()
            if not ret:
                break
            bgr, lab = self.team_classifier.preprocess_frame(frame)
            detections = self._detect(bgr)
            tracked = self.tracker.update_with_detections(detections)
            player_dets = tracked[tracked.class_id == 2]
            if len(player_dets) >= 10:
                if self.team_classifier.fit_auto(frame, player_dets, lab):
                    cap.release()
                    return True
        cap.release()
        print("Warning: Could not auto-detect teams. All players will be 'unknown'.")
        return False

    def analyze_video(self, video_path, max_frames=None, output_path="output/analysis.mp4"):
        """Run the full analysis pipeline on a video."""
        self._setup_teams(video_path)

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        for _ in tqdm(range(total), desc="Analyzing"):
            ret, frame = cap.read()
            if not ret:
                break

            analytics = self.process_frame(frame)
            self.frame_analytics.append(analytics)

            vis = self._draw_frame(frame.copy(), analytics, w, h)
            writer.write(vis)

        writer.release()
        cap.release()
        print(f"Output saved to {output_path}")
        return self.frame_analytics

    # --- Visualization ---

    def _draw_frame(self, frame, analytics, w, h):
        """Draw annotations and pitch map on the frame."""
        # Player boxes and labels
        if analytics.players:
            det = sv.Detections(
                xyxy=np.array([p.bbox for p in analytics.players]),
                class_id=np.array([p.team if p.team in (0, 1) else 2 for p in analytics.players]),
            )
            team_labels = {0: "A", 1: "B"}
            labels = [f"{team_labels.get(p.team, '?')} #{p.id}" for p in analytics.players]
            frame = self.box_annotator.annotate(frame, det)
            frame = self.label_annotator.annotate(frame, det, labels=labels)

        # Ball
        if analytics.ball:
            ball_det = sv.Detections(
                xyxy=np.array([analytics.ball.bbox]),
                class_id=np.array([0]),
            )
            frame = self.ball_annotator.annotate(frame, ball_det)

        # Status overlay
        text, color = self.pitch_mapper.get_status_text()
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Pitch minimap
        pitch_view = self._draw_pitch_view(analytics)
        frame = self._overlay_minimap(frame, pitch_view, w, h)

        return frame

    def _draw_pitch_view(self, analytics):
        """Render the 2D pitch map."""
        pw, ph = 420, 272
        pitch = np.full((ph, pw, 3), (0, 100, 0), dtype=np.uint8)
        sx = pw / self.pitch_mapper.pitch_width
        sy = ph / self.pitch_mapper.pitch_height

        # Pitch markings
        white = (255, 255, 255)
        cv2.rectangle(pitch, (0, 0), (pw - 1, ph - 1), white, 1)
        cv2.line(pitch, (pw // 2, 0), (pw // 2, ph), white, 1)
        cv2.circle(pitch, (pw // 2, ph // 2), int(9.15 * sy), white, 1)
        # Penalty areas
        cv2.rectangle(
            pitch,
            (0, int((ph - 40.32 * sy) / 2)),
            (int(16.5 * sx), int((ph + 40.32 * sy) / 2)),
            white,
            1,
        )
        cv2.rectangle(
            pitch,
            (pw - int(16.5 * sx), int((ph - 40.32 * sy) / 2)),
            (pw, int((ph + 40.32 * sy) / 2)),
            white,
            1,
        )
        # Goal areas
        cv2.rectangle(
            pitch,
            (0, int((ph - 18.32 * sy) / 2)),
            (int(5.5 * sx), int((ph + 18.32 * sy) / 2)),
            white,
            1,
        )
        cv2.rectangle(
            pitch,
            (pw - int(5.5 * sx), int((ph - 18.32 * sy) / 2)),
            (pw, int((ph + 18.32 * sy) / 2)),
            white,
            1,
        )
        # Center dot
        cv2.circle(pitch, (pw // 2, ph // 2), 3, white, -1)

        # Players
        team_colors = {0: (0, 0, 255), 1: (0, 255, 255), None: (128, 128, 128)}
        for p in analytics.players:
            if p.field_position:
                x = int(p.field_position[0] * sx)
                y = int(p.field_position[1] * sy)
                if 0 <= x < pw and 0 <= y < ph:
                    color = team_colors.get(p.team, (128, 128, 128))
                    cv2.circle(pitch, (x, y), 7, color, -1)
                    cv2.circle(pitch, (x, y), 7, (0, 0, 0), 1)

        # Ball
        if analytics.ball and analytics.ball.field_position:
            x = int(analytics.ball.field_position[0] * sx)
            y = int(analytics.ball.field_position[1] * sy)
            if 0 <= x < pw and 0 <= y < ph:
                cv2.circle(pitch, (x, y), 5, (255, 255, 255), -1)
                cv2.circle(pitch, (x, y), 5, (0, 0, 0), 1)

        return pitch

    def _overlay_minimap(self, frame, pitch_view, w, h):
        """Overlay the pitch minimap on the top-right of the frame."""
        pv_h, pv_w = pitch_view.shape[:2]
        new_w = int(w * 0.28)
        new_h = int(pv_h * (new_w / pv_w))
        resized = cv2.resize(pitch_view, (new_w, new_h))

        # Semi-transparent background
        x_off = w - new_w - 15
        y_off = 15
        roi = frame[y_off : y_off + new_h, x_off : x_off + new_w]
        blended = cv2.addWeighted(roi, 0.3, resized, 0.7, 0)
        frame[y_off : y_off + new_h, x_off : x_off + new_w] = blended

        # Border
        cv2.rectangle(frame, (x_off, y_off), (x_off + new_w, y_off + new_h), (255, 255, 255), 2)

        return frame
