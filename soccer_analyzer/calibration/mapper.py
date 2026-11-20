"""
PitchMapper — turns per-keyframe field registration into a usable image->pitch mapping.

Phase A behaviour:
  * runs the FieldDetector on keyframes (every `keyframe_interval` frames);
  * holds the last good homography between keyframes;
  * exposes a confidence that decays the longer we go without a fresh detection;
  * is a drop-in replacement for the old soccer_analyzer.pitch_mapper.PitchMapper
    (same method names: update / transform / is_position_on_pitch / get_status_text,
    and attributes pitch_width / pitch_height) so pipeline.py needs minimal changes.

Phase B replaces the "hold last good" logic with the GTSAM factor-graph smoother (fusing
detections + frame-to-frame correspondences) for drift-free, covariance-aware poses.
Phase C adds mosaic relocalization. The interface here is designed to absorb both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH
from soccer_analyzer.calibration.field_detector import (
    FieldDetector,
    CachedHomographyDetector,
    PnLCalibFieldDetector,
)


class _FlowTracker:
    """Frame-to-frame camera motion via Lucas-Kanade optical flow on background features.

    Returns G, the image->image homography from the previous frame to the current one.
    RANSAC rejects moving players, so the dominant (field/background) motion is recovered.
    Operates on a downscaled gray image for speed, returning G in FULL-RES coordinates.
    """

    def __init__(
        self, target_w: int = 960, max_corners: int = 500, quality: float = 0.01, min_dist: int = 16
    ):
        self.target_w = target_w
        self.max_corners = max_corners
        self.quality = quality
        self.min_dist = min_dist
        self._prev = None  # downscaled gray
        self._pts = None
        self._scale = 1.0  # full -> small

    def _prep(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        s = min(1.0, self.target_w / w)
        small = cv2.resize(frame_bgr, (int(w * s), int(h * s))) if s < 1.0 else frame_bgr
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), s

    def reset(self, frame_bgr):
        gray, s = self._prep(frame_bgr)
        self._scale = s
        self._prev = gray
        self._pts = cv2.goodFeaturesToTrack(gray, self.max_corners, self.quality, self.min_dist)

    def step(self, frame_bgr) -> Optional[np.ndarray]:
        gray, s = self._prep(frame_bgr)
        G = None
        if self._prev is not None and self._pts is not None and len(self._pts) >= 12:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                self._prev,
                gray,
                self._pts,
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            if nxt is not None and st is not None:
                good = st.ravel() == 1
                if int(good.sum()) >= 12:
                    g_small, _ = cv2.findHomography(self._pts[good], nxt[good], cv2.RANSAC, 3.0)
                    if g_small is not None:
                        # conjugate small-space G to full-res: G_full = inv(S) G_small S
                        S = np.diag([s, s, 1.0])
                        G = np.linalg.inv(S) @ g_small @ S
        self._prev = gray
        self._pts = cv2.goodFeaturesToTrack(gray, self.max_corners, self.quality, self.min_dist)
        return G


@dataclass
class FrameCalibration:
    frame_idx: int
    homography: np.ndarray  # 3x3 image(px) -> pitch(m)
    confidence: float  # 0..1
    source: str  # "detect" | "held" | "none"
    n_landmarks: int
    frames_since_detection: int
    reproj_err: Optional[float] = None  # detector-reported reprojection error (px)


class PitchMapper:
    def __init__(
        self,
        detector: Optional[FieldDetector] = None,
        calibration_path: Optional[str] = None,
        pitch: PitchModel = STANDARD_PITCH,
        keyframe_interval: int = 10,
        max_hold_frames: int = 120,
        smooth: bool = False,
        outlier_px: float = 250.0,
        reacquire_after: int = 3,
        trust_reproj_px: float = 8.0,
    ):
        """Resolve a detector:
        explicit `detector`         -> use it
        else `calibration_path`     -> CachedHomographyDetector (bridge)
        else                        -> PnLCalibFieldDetector (automatic; needs setup)
        """
        if keyframe_interval <= 0 or max_hold_frames <= 0:
            raise ValueError("keyframe_interval and max_hold_frames must be positive")
        if detector is not None:
            self.detector = detector
        elif calibration_path:
            self.detector = CachedHomographyDetector(calibration_path, pitch)
        else:
            self.detector = PnLCalibFieldDetector(pitch=pitch)

        self.pitch = pitch
        self.pitch_width = pitch.length  # keep old attribute names for the minimap
        self.pitch_height = pitch.width
        self.keyframe_interval = keyframe_interval
        self.max_hold_frames = max_hold_frames
        self.smooth = smooth
        self.outlier_px = outlier_px
        self.reacquire_after = reacquire_after
        self.trust_reproj_px = trust_reproj_px
        self._flow = _FlowTracker() if smooth else None

        self._frame_counter = -1
        self._current: Optional[FrameCalibration] = None
        self._frames_since_detection = 10**9
        self._rejections = 0
        self.last_observation = None  # most recent valid FieldObservation (for validation)
        # cache of pitch keypoints (meters) for the consistency check
        self._pitch_pts = np.array(list(pitch.keypoints.values()), dtype=np.float64)

    # ---- per-frame update --------------------------------------------------------

    def update(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> FrameCalibration:
        """Call once per frame BEFORE transform().

        Phase B pipeline (smooth=True):
          1. PREDICT: propagate the current homography through frame-to-frame optical flow.
          2. DETECT (keyframes): run the field detector.
          3. GATE: accept the detection only if it agrees with the prediction; reject gross
             outliers (bad detections / scene cuts) until `reacquire_after` consecutive
             rejections force a re-acquisition. Between keyframes, use the flow prediction.
        With smooth=False this degrades to "detect on keyframes, hold in between".
        """
        self._frame_counter += 1
        idx = self._frame_counter if frame_idx is None else frame_idx

        # 1. PREDICT (flow propagation)
        predicted_H = None
        if (
            self.smooth
            and self._flow is not None
            and self._current is not None
            and self._current.source != "none"
        ):
            G = self._flow.step(frame)
            if G is not None and np.all(np.isfinite(G)):
                try:
                    predicted_H = self._current.homography @ np.linalg.inv(G)
                except np.linalg.LinAlgError:
                    predicted_H = None

        # 2. DETECT (keyframes, or whenever we have no track)
        is_keyframe = idx % self.keyframe_interval == 0
        obs = None
        if is_keyframe or self._current is None or self._current.source == "none":
            try:
                obs = self.detector.detect(frame, idx)
            except NotImplementedError:
                raise
            except Exception:
                obs = None
            if obs is not None and not obs.is_valid:
                obs = None

        # 3. GATE the detection against the prediction. A detection with low reprojection
        # error is TRUSTED unconditionally (it's more reliable than flow, which drifts on
        # dark/low-texture footage). Only borderline detections are sanity-checked against
        # the flow prediction.
        if obs is not None:
            trusted = obs.reproj_err is not None and obs.reproj_err <= self.trust_reproj_px
            accept = True
            if not trusted and predicted_H is not None and self._rejections < self.reacquire_after:
                if self._disagreement_px(obs.homography, predicted_H) > self.outlier_px:
                    accept = False
                    self._rejections += 1
            if accept:
                self._rejections = 0
                self._frames_since_detection = 0
                self.last_observation = obs
                if self.smooth and self._flow is not None:
                    self._flow.reset(frame)  # re-anchor flow at the accepted detection
                self._current = FrameCalibration(
                    frame_idx=idx,
                    homography=obs.homography,
                    confidence=self._confidence(obs.n_landmarks, 0),
                    source="detect",
                    n_landmarks=obs.n_landmarks,
                    frames_since_detection=0,
                    reproj_err=obs.reproj_err,
                )
                return self._current

        # No accepted detection: use the flow prediction, else hold, else go offline.
        self._frames_since_detection += 1
        if self._current is not None and self._frames_since_detection <= self.max_hold_frames:
            if predicted_H is not None:
                H, source = predicted_H, "flow"
            else:
                H, source = self._current.homography, "held"
            n = self._current.n_landmarks
            self._current = FrameCalibration(
                frame_idx=idx,
                homography=H,
                confidence=self._confidence(n, self._frames_since_detection),
                source=source,
                n_landmarks=n,
                frames_since_detection=self._frames_since_detection,
                reproj_err=self._current.reproj_err,
            )
        else:
            self._current = FrameCalibration(
                frame_idx=idx,
                homography=np.eye(3),
                confidence=0.0,
                source="none",
                n_landmarks=0,
                frames_since_detection=self._frames_since_detection,
            )
        return self._current

    def _disagreement_px(self, H1: np.ndarray, H2: np.ndarray) -> float:
        """Mean pixel displacement of the pitch keypoints projected (pitch->image) by two
        homographies. Used to reject detections inconsistent with the predicted pose."""
        try:
            i1, i2 = np.linalg.inv(H1), np.linalg.inv(H2)
        except np.linalg.LinAlgError:
            return float("inf")
        p = np.hstack([self._pitch_pts, np.ones((len(self._pitch_pts), 1))])
        a = (i1 @ p.T).T
        b = (i2 @ p.T).T
        aw, bw = a[:, 2:3], b[:, 2:3]
        ok = (np.abs(aw[:, 0]) > 1e-6) & (np.abs(bw[:, 0]) > 1e-6)
        if not ok.any():
            return float("inf")
        a2 = a[ok, :2] / aw[ok]
        b2 = b[ok, :2] / bw[ok]
        return float(np.median(np.linalg.norm(a2 - b2, axis=1)))

    def _confidence(self, n_landmarks: int, frames_since: int) -> float:
        if n_landmarks <= 0:
            return 0.0
        base = min(1.0, n_landmarks / 8.0)  # 8+ landmarks -> full
        decay = max(0.0, 1.0 - frames_since / float(self.max_hold_frames))
        return float(base * decay)

    # ---- mapping -----------------------------------------------------------------

    @property
    def homography(self) -> Optional[np.ndarray]:
        return None if self._current is None else self._current.homography

    def transform(self, screen_points) -> np.ndarray:
        """image (x, y) px -> pitch (x, y) meters. Drop-in compatible with the old mapper
        (returns an (N, 2) array, or empty if no valid calibration)."""
        if not len(screen_points) or self._current is None or self._current.source == "none":
            return np.array([])
        import cv2

        pts = np.array(screen_points, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self._current.homography)
        return out.reshape(-1, 2) if out is not None else np.array([])

    def transform_with_uncertainty(self, screen_points) -> Tuple[np.ndarray, np.ndarray]:
        """Same as transform() but also returns a coarse per-point 1-sigma (meters).

        Phase A uncertainty is a placeholder derived from confidence (low confidence ->
        large sigma); Phase B replaces it with the propagated pose covariance."""
        coords = self.transform(screen_points)
        if not len(coords):
            return coords, np.array([])
        conf = self._current.confidence if self._current else 0.0
        sigma_m = 1.0 + 9.0 * (1.0 - conf)  # ~1 m at full conf, ~10 m at zero
        sigma = np.full_like(coords, sigma_m)
        return coords, sigma

    def is_position_on_pitch(self, pitch_point, margin: float = 5.0) -> bool:
        x, y = pitch_point
        return (
            -margin <= x <= self.pitch_width + margin and -margin <= y <= self.pitch_height + margin
        )

    # ---- status ------------------------------------------------------------------

    def current_confidence(self) -> float:
        return 0.0 if self._current is None else self._current.confidence

    def get_status_text(self):
        if self._current is None or self._current.source == "none":
            return ("Calibration: OFFLINE", (0, 0, 255))
        c = self._current
        color = (0, 255, 0) if c.confidence > 0.5 else (0, 200, 255)
        return (
            f"Calib {c.source} conf={c.confidence:.2f} "
            f"kp={c.n_landmarks} age={c.frames_since_detection}",
            color,
        )
