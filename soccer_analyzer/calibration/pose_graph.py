"""
Robust fixed-camera pose smoothing (the real fix for wrong-but-low-reproj frames).

Per-frame calibration can be confidently WRONG: with few keypoints PnLCalib fits them at
<1px while matching them to the wrong pitch landmarks (a wrong CORRESPONDENCE). Reprojection
error can't catch that. But the camera is FIXED, so the true rotation trajectory is smooth
and consistent — the wrong frames are outliers. This module pools every keyframe's rotation,
fits a smooth trajectory that REJECTS outliers (quality-weighted, robust), and rebuilds every
frame's homography from the fused (locked) intrinsics+position + the smoothed rotation.

So the many correct frames outvote the occasional wrong one, and the whole clip becomes
consistent instead of "great on some frames, garbage on others".

STATUS (2026-06-18): correct in principle, but BLOCKED by detection quality on amateur
footage. Diagnostic showed PnLCalib's per-frame camera POSITION varies by 400-1000m on the
night / multi-sport-line clips (a fixed camera should be ~0), i.e. the calibrations are
mostly WRONG, not just noisy. A robust smoother needs a consistent majority to lock onto;
there isn't one. The real fix is better DETECTION (classical circle/line fallback or a
fine-tuned detector), after which this smoother becomes useful.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from soccer_analyzer.calibration.field_detector import projection_from_cam_params
from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH


def _homography_from(cam_params: dict, pitch: PitchModel) -> Optional[np.ndarray]:
    P = projection_from_cam_params({"cam_params": cam_params})
    Hwi = P[:, [0, 1, 3]]
    try:
        Hiw = np.linalg.inv(Hwi)
    except np.linalg.LinAlgError:
        return None
    A = np.array([[1.0, 0, pitch.length / 2.0], [0, 1.0, pitch.width / 2.0], [0, 0, 1.0]])
    H = A @ Hiw
    if abs(H[2, 2]) < 1e-12 or not np.all(np.isfinite(H)):
        return None
    return H / H[2, 2]


class PoseSmoother:
    def __init__(self, fusion, pitch: PitchModel = STANDARD_PITCH):
        self.fusion = fusion
        self.pitch = pitch
        self.ready = False
        self.idxs = None
        self._rots = None  # scipy Rotation, smoothed, per keyframe

    def fit(
        self,
        observations: List[Tuple[int, np.ndarray, float, int]],
        window: int = 12,
        reject_deg: float = 7.0,
        min_kp: int = 6,
    ):
        """observations: list of (frame_idx, rotation_3x3, reproj_px, n_keypoints)."""
        from scipy.spatial.transform import Rotation, Slerp

        if not getattr(self.fusion, "ready", False):
            return self
        obs = [o for o in observations if o[1] is not None and np.all(np.isfinite(o[1]))]
        if len(obs) < 4:
            return self
        obs.sort(key=lambda o: o[0])
        idxs = np.array([o[0] for o in obs], float)
        rotvecs = Rotation.from_matrix(np.array([o[1] for o in obs])).as_rotvec()
        reproj = np.array([o[2] if o[2] else 12.0 for o in obs])
        nkp = np.array([o[3] for o in obs], float)
        # quality weight: trust low-reproj, many-keypoint frames
        w = (1.0 / (1.0 + reproj)) * np.clip(nkp / min_kp, 0.0, 1.0)
        w = np.maximum(w, 1e-3)

        smoothed = np.zeros_like(rotvecs)
        n = len(obs)
        for i in range(n):
            lo, hi = max(0, i - window), min(n, i + window + 1)
            wl = w[lo:hi].copy()
            seg = rotvecs[lo:hi]
            # robust weighted mean with two reject passes (drop window outliers)
            mean = np.average(seg, axis=0, weights=wl)
            for _ in range(2):
                R_seg = Rotation.from_rotvec(seg)
                R_mean = Rotation.from_rotvec(mean)
                d = np.degrees((R_mean.inv() * R_seg).magnitude())
                keep = d < (reject_deg * 2.5)
                if keep.sum() >= 2:
                    wl2 = wl * keep
                    if wl2.sum() > 1e-6:
                        mean = np.average(seg, axis=0, weights=wl2)
            smoothed[i] = mean

        self.idxs = idxs
        self._rots = Rotation.from_rotvec(smoothed)
        self._slerp = Slerp(idxs, self._rots)
        self.ready = True
        self._n_outliers = self._count_outliers(rotvecs, smoothed, reject_deg)
        return self

    @staticmethod
    def _count_outliers(raw, smoothed, reject_deg):
        from scipy.spatial.transform import Rotation

        d = np.degrees(
            (Rotation.from_rotvec(smoothed).inv() * Rotation.from_rotvec(raw)).magnitude()
        )
        return int((d > reject_deg).sum())

    def homography_at(self, frame_idx: int) -> Optional[np.ndarray]:
        if not self.ready:
            return None
        fi = float(np.clip(frame_idx, self.idxs[0], self.idxs[-1]))
        R = self._slerp([fi])[0].as_matrix()
        cp = {
            "x_focal_length": self.fusion.focal,
            "y_focal_length": self.fusion.focal,
            "principal_point": list(self.fusion.principal),
            "position_meters": list(self.fusion.position),
            "rotation_matrix": R,
        }
        return _homography_from(cp, self.pitch)

    def summary(self) -> dict:
        if not self.ready:
            return {"ready": False}
        return {
            "ready": True,
            "keyframes": len(self.idxs),
            "outliers_rejected": getattr(self, "_n_outliers", 0),
        }


def collect_and_smooth(
    detector, video_path: str, pitch: PitchModel = STANDARD_PITCH, n_frames: int = 120
) -> PoseSmoother:
    """Pre-pass: sample frames, collect per-keyframe rotations, fit the robust smoother."""
    import cv2

    sm = PoseSmoother(detector.fusion, pitch)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return sm
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_frames
    idxs = np.linspace(0, max(0, total - 1), min(n_frames, total)).astype(int)
    obs = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        try:
            o = detector.detect(frame, int(fi))
        except Exception:
            o = None
        if o is not None and o.cam_params is not None:
            obs.append(
                (int(fi), np.array(o.cam_params["rotation_matrix"]), o.reproj_err, o.n_landmarks)
            )
    cap.release()
    return sm.fit(obs)
