# football_analyzer/pitch_mapper.py

import cv2
import numpy as np
from collections import deque


class DynamicPitchMapper:
    """
    Final stabilized version using 3D-to-2D solvePnP with robust orientation locking
    and aggressive vector smoothing to prevent flipping and jitter.
    """

    def __init__(self, pitch_dims=(105, 68), smoothing_window=10):  # Increased smoothing
        self.pitch_width, self.pitch_height = pitch_dims
        self.field_model = None
        self.stable_homography = None
        self.min_landmarks_for_lock = 4
        self.last_matched_points_count = 0
        self.last_raw_predictions = []
        self.last_used_image_points = []

        self.rvec_history = deque(maxlen=smoothing_window)
        self.tvec_history = deque(maxlen=smoothing_window)

        self.PITCH_3D_MODEL = {
            'c': [52.5, 34.0, 0], '14': [43.35, 34.0, 0], '15': [61.65, 34.0, 0],
            '16': [52.5, 0.0, 0], '13': [52.5, 68.0, 0], '3': [88.5, 13.84, 0],
            '4': [88.5, 54.16, 0], '6': [99.5, 24.84, 0], '7': [99.5, 43.16, 0],
            '10': [94.0, 34.0, 0], '1': [16.5, 13.84, 0], '2': [16.5, 54.16, 0],
            '5': [5.5, 24.84, 0], '8': [5.5, 43.16, 0], '9': [11.0, 34.0, 0],
            '11': [0.0, 0.0, 0], '12': [105.0, 0.0, 0],
        }

        self.pitch_3d_corners = np.array([
            [0, 0, 0], [self.pitch_width, 0, 0],
            [self.pitch_width, self.pitch_height, 0], [0, self.pitch_height, 0]
        ], dtype=np.float32)

        self.pitch_2d_map_corners = np.array([
            [0, 0], [self.pitch_width, 0],
            [self.pitch_width, self.pitch_height], [0, self.pitch_height]
        ], dtype=np.float32)

        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))

    def _estimate_camera_intrinsics(self, frame_shape):
        height, width = frame_shape
        focal_length = 1500
        optical_center = (width / 2, height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0, optical_center[0]],
            [0, focal_length, optical_center[1]],
            [0, 0, 1]
        ], dtype="double")
        print(f"Camera intrinsics set with realistic focal length: {focal_length}")

    def set_field_model(self, field_model):
        self.field_model = field_model
        print("solvePnP Mapper with Orientation Lock initialized.")

    def update_homography(self, frame):
        if self.camera_matrix is None:
            self._estimate_camera_intrinsics(frame.shape[:2])

        try:
            result = self.field_model.infer(frame, confidence=0.6, overlap=0.5)[0]
            raw_predictions = result.predictions
        except Exception:
            raw_predictions = []

        h, _ = frame.shape[:2]
        y_cutoff = h * 0.85
        self.last_raw_predictions = [p for p in raw_predictions if p.y < y_cutoff]

        image_points = [(p.x, p.y) for p in self.last_raw_predictions if str(p.class_name) in self.PITCH_3D_MODEL]
        object_points = [self.PITCH_3D_MODEL[str(p.class_name)] for p in self.last_raw_predictions if
                         str(p.class_name) in self.PITCH_3D_MODEL]
        self.last_matched_points_count = len(image_points)

        if self.last_matched_points_count >= self.min_landmarks_for_lock:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    np.array(object_points, dtype=np.float32), np.array(image_points, dtype=np.float32),
                    self.camera_matrix, self.dist_coeffs
                )
                if success:
                    self.rvec_history.append(rvec)
                    self.tvec_history.append(tvec)
                    self.last_used_image_points = image_points
                else:
                    self.rvec_history.clear()
                    self.tvec_history.clear()
            except Exception:
                self.rvec_history.clear()
                self.tvec_history.clear()
        else:
            self.rvec_history.clear()
            self.tvec_history.clear()

        if len(self.rvec_history) > 0:
            smoothed_rvec = np.mean(self.rvec_history, axis=0)
            smoothed_tvec = np.mean(self.tvec_history, axis=0)

            projected_corners, _ = cv2.projectPoints(
                self.pitch_3d_corners, smoothed_rvec, smoothed_tvec, self.camera_matrix, self.dist_coeffs
            )
            homography, _ = cv2.findHomography(
                projected_corners, self.pitch_2d_map_corners
            )

            # --- ORIENTATION LOCK / SANITY CHECK ---
            if homography is not None:
                # We expect landmark '9' (left penalty spot) to be on the left side of the map (x < 52.5)
                # Find landmark '9' in the original detections if it exists
                anchor_point_screen = None
                for p in self.last_raw_predictions:
                    if str(p.class_name) == '9':
                        anchor_point_screen = (p.x, p.y)
                        break

                if anchor_point_screen:
                    transformed_anchor = self.transform_point(anchor_point_screen, homography)
                    # If the left-side anchor point appears on the right side of the map, the map has flipped.
                    if transformed_anchor is not None and transformed_anchor[0] > self.pitch_width / 2:
                        # Create an inversion matrix and apply it
                        inversion_matrix = np.array([[-1, 0, self.pitch_width], [0, 1, 0], [0, 0, 1]])
                        homography = inversion_matrix @ homography

            self.stable_homography = homography
            return True
        else:
            self.stable_homography = None
            self.last_used_image_points = []
            return False

    def is_tracking_stable(self):
        return self.stable_homography is not None

    def transform_point(self, point, homography):
        """Helper to transform a single point."""
        point_np = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_np, homography)
        return transformed[0][0] if transformed is not None else None

    def transform_to_pitch(self, frame_points):
        if not self.is_tracking_stable() or not frame_points: return np.array([])
        points_np = np.array(frame_points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points_np, self.stable_homography)
        return transformed.reshape(-1, 2) if transformed is not None else np.array([])

    def get_field_overlay(self, frame):
        status, color = (f"Tracking Stable ({self.last_matched_points_count} Landmarks)", (0, 255, 0)) \
            if self.is_tracking_stable() else (
            f"Searching... ({self.last_matched_points_count}/{self.min_landmarks_for_lock} Landmarks)", (0, 0, 255))
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return frame