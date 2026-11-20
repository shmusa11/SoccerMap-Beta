"""
Pitch mapping via manual calibration + optical flow tracking.

Loads a calibration file (created by calibrate.py) that maps known pixel
positions to pitch coordinates. Computes a homography from these correspondences.
As the camera pans, optical flow tracks the frame-to-frame motion and updates
the homography accordingly.

No ML models, no solvePnP, no camera intrinsics. Just geometry.
"""

import json
import cv2
import numpy as np


class PitchMapper:
    def __init__(self, calibration_path):
        with open(calibration_path) as f:
            cal = json.load(f)

        self.pitch_width = cal["pitch_width"]
        self.pitch_height = cal["pitch_height"]

        # Compute homography from calibration points
        if "homography" in cal:
            self.base_homography = np.array(cal["homography"], dtype=np.float64)
        else:
            frame_pts = np.array([p["frame"] for p in cal["points"]], dtype=np.float32)
            pitch_pts = np.array([p["pitch"] for p in cal["points"]], dtype=np.float32)
            self.base_homography, _ = cv2.findHomography(frame_pts, pitch_pts, cv2.RANSAC, 5.0)

        self.current_homography = self.base_homography.copy()

        # Optical flow state
        self._prev_gray = None
        self._prev_features = None

        # Track cumulative drift for diagnostics
        self._frames_since_calibration = 0

    def update(self, frame):
        """Update the homography using optical flow from the previous frame.

        Call this once per frame BEFORE calling transform().
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (
            self._prev_gray is not None
            and self._prev_features is not None
            and len(self._prev_features) >= 10
        ):
            # Track features from previous frame to current frame
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray,
                gray,
                self._prev_features,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )

            if new_pts is not None and status is not None:
                good = status.ravel() == 1
                if np.sum(good) >= 10:
                    prev_good = self._prev_features[good]
                    curr_good = new_pts[good]

                    # G maps prev_screen -> curr_screen
                    G, mask = cv2.findHomography(prev_good, curr_good, cv2.RANSAC, 5.0)
                    if G is not None:
                        try:
                            G_inv = np.linalg.inv(G)
                            self.current_homography = self.current_homography @ G_inv
                        except np.linalg.LinAlgError:
                            pass

        # Refresh tracked features for next frame
        self._prev_gray = gray
        self._prev_features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=20,
        )
        self._frames_since_calibration += 1

    def transform(self, screen_points):
        """Transform screen (x, y) coordinates to pitch coordinates (meters).

        Args:
            screen_points: list of (x, y) tuples in pixel coordinates

        Returns:
            numpy array of shape (N, 2) with pitch coordinates, or empty array
        """
        if not screen_points:
            return np.array([])
        pts = np.array(screen_points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.current_homography)
        return transformed.reshape(-1, 2) if transformed is not None else np.array([])

    def is_position_on_pitch(self, pitch_point, margin=5.0):
        """Check if a pitch coordinate is within the pitch bounds (with margin)."""
        x, y = pitch_point
        return (
            -margin <= x <= self.pitch_width + margin and -margin <= y <= self.pitch_height + margin
        )

    def get_status_text(self):
        """Return (text, bgr_color) for overlay."""
        return (f"Calibrated (frame {self._frames_since_calibration})", (0, 255, 0))
