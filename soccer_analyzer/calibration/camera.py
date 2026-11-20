"""Conservative reference-view checks for fixed-camera calibration reuse.

Compare directly with the calibrated image, never with the preceding frame: incremental
comparisons would hide accumulated pan or zoom. Low texture and uncertain matches fail closed.
"""

from __future__ import annotations

import cv2
import numpy as np


def camera_gray(frame: np.ndarray, width: int = 480) -> np.ndarray:
    """A bounded grayscale reference; input is an OpenCV BGR frame."""
    height, source_width = frame.shape[:2]
    return cv2.cvtColor(
        cv2.resize(frame, (width, max(1, round(height * width / source_width)))), cv2.COLOR_BGR2GRAY
    )


class CameraReference:
    """Verify a stationary view using distributed background features and robust optical flow."""

    def __init__(self, frame: np.ndarray, tolerance_px: float = 1.5):
        self.size = frame.shape[:2]
        self.gray = camera_gray(frame)
        self.tolerance = tolerance_px  # pixels in the 480-pixel-wide reference
        self.points = cv2.goodFeaturesToTrack(
            self.gray, maxCorners=160, qualityLevel=0.015, minDistance=12, blockSize=7
        )

    def matches(self, frame: np.ndarray) -> bool:
        """True only with spatially distributed, consistent support for the same view."""
        if frame.shape[:2] != self.size or self.points is None or len(self.points) < 12:
            return False
        gray = camera_gray(frame)
        forward, status, _ = cv2.calcOpticalFlowPyrLK(
            self.gray, gray, self.points, None, winSize=(21, 21), maxLevel=3
        )
        if forward is None:
            return False
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.gray, forward, None, winSize=(21, 21), maxLevel=3
        )
        if back is None:
            return False
        good = (
            (status.ravel() == 1)
            & (back_status.ravel() == 1)
            & (np.linalg.norm(self.points - back, axis=2).ravel() < 0.8)
        )
        source, target = self.points[good].reshape(-1, 2), forward[good].reshape(-1, 2)
        if len(source) < 12:
            return False
        affine, mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=1.5,
            maxIters=300,
            confidence=0.99,
        )
        if affine is None or mask is None:
            return False
        inliers = source[mask.ravel().astype(bool)]
        h, w = gray.shape
        if len(inliers) < 12 or len(inliers) / len(source) < 0.65:
            return False
        if np.ptp(inliers[:, 0]) < w * 0.35 or np.ptp(inliers[:, 1]) < h * 0.2:
            return False
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        shifted = corners @ affine[:, :2].T + affine[:, 2]
        if np.max(np.linalg.norm(shifted - corners, axis=1)) > self.tolerance:
            return False
        # Require patch agreement as well as motion: unrelated scenes sometimes fit a tiny
        # consensus of repetitive grass/stands. Brightness shifts under 20 levels are tolerated.
        residual = cv2.absdiff(self.gray, gray)
        return float(np.median(residual)) < 20 and float(np.mean(residual < 35)) > 0.7


def calibrate_from_reference(
    video_path: str,
    reference: np.ndarray,
    matrix: np.ndarray,
    *,
    stride: int,
    max_frames: int,
    progress=None,
    start_frame: int = 0,
) -> tuple[dict, dict]:
    """Check each sampled source frame against a user-reviewed setup without field inference."""
    from soccer_analyzer.video import open_capture, frame_window, seek_capture

    if stride <= 0 or max_frames <= 0:
        raise ValueError("stride and max_frames must be positive")
    guard = CameraReference(reference)
    result = {}
    with open_capture(video_path) as cap:
        start, stop = frame_window(cap.get(cv2.CAP_PROP_FRAME_COUNT), max_frames, start_frame)
        seek_capture(cap, start)
        for index in range(start, stop):
            if not cap.grab():
                break
            if (index - start) % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            result[index] = matrix if guard.matches(frame) else None
            if progress and len(result) % 10 == 0:
                progress((index - start + 1) / (stop - start))
    return result, {
        "frames": len(result),
        "covered": sum(h is not None for h in result.values()),
        "inference_calls": 0,
        "reused": sum(h is not None for h in result.values()),
        "mode": "saved_camera",
    }
