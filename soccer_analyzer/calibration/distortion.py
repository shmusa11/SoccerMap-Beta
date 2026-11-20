"""
Automatic lens-distortion estimation + correction (Stage 0, per video).

Different videos have different lenses, so distortion is estimated automatically PER CLIP
from the footage itself — no presets. Method = the classic "plumb-line" idea specialised to
soccer: the pitch lines are known to be straight, so we find the single radial-distortion
coefficient that, after undistortion, makes the imaged field lines straightest. Evidence is
aggregated over many frames and many lines for robustness, and the result is GATED: if there
isn't enough evidence or undistortion doesn't measurably straighten the lines, we return a
no-op (k=0) so a low-distortion camera is never made worse.

The estimated distortion is locked once and frames are undistorted up front, so every
downstream stage (field registration, player detection, mapping, overlay) is consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH


@dataclass
class DistortionModel:
    K: np.ndarray  # 3x3 intrinsics used for the distortion model
    dist: np.ndarray  # OpenCV (k1, k2, p1, p2)
    size: Tuple[int, int]  # (w, h)
    n_samples: int = 0
    improvement: float = 0.0  # fractional reduction in line-straightness residual
    source: str = "none"  # "estimated" | "none"

    @property
    def enabled(self) -> bool:
        return float(np.sum(np.abs(self.dist[:2]))) > 1e-7


class Undistorter:
    """Undistort frames (and points) with a locked distortion model. No-op when disabled."""

    def __init__(self, model: DistortionModel):
        self.model = model
        self.enabled = model.enabled
        if self.enabled:
            w, h = model.size
            self.newK, _ = cv2.getOptimalNewCameraMatrix(model.K, model.dist, (w, h), 0)
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                model.K, model.dist, None, self.newK, (w, h), cv2.CV_16SC2
            )
        else:
            self.newK = model.K

    def frame(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR) if self.enabled else img

    def points(self, pts: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return pts
        p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
        return cv2.undistortPoints(p, self.model.K, self.model.dist, P=self.newK).reshape(-1, 2)


# ---- estimation ----------------------------------------------------------------------


def _line_points_in_image(H_inv, p0, p1, w, h, step=2.0):
    """Sample a pitch line (meters) and map to image (straight under a pinhole homography)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(8, int(np.linalg.norm(p1 - p0) / step))
    pm = np.array([p0 + t * (p1 - p0) for t in np.linspace(0, 1, n)])
    ph = np.hstack([pm, np.ones((len(pm), 1))])
    q = (H_inv @ ph.T).T
    out = []
    for x, y, wq in q:
        if wq > 1e-6:
            u, v = x / wq, y / wq
            if 0 <= u < w and 0 <= v < h:
                out.append((u, v))
    return np.array(out) if out else np.empty((0, 2))


def _snap_to_white_on_grass(hsv, pts, normal, band=26):
    """For each reprojected line point, search along the line normal for the nearest
    white-on-grass pixel (bright, low-sat, with green neighbours). Returns snapped image
    points (only the ones that found clean evidence)."""
    h, w = hsv.shape[:2]
    snapped = []
    nx, ny = normal
    for u, v in pts:
        best = None
        for d in range(-band, band + 1):
            su, sv = int(round(u + d * nx)), int(round(v + d * ny))
            if not (0 <= su < w and 0 <= sv < h):
                continue
            H_, S_, V_ = hsv[sv, su]
            if V_ > 140 and S_ < 70:  # white-ish line pixel
                # require grass (green) nearby on the normal -> a line painted on the field
                gu, gv = int(round(u + (d + 8) * nx)), int(round(v + (d + 8) * ny))
                if 0 <= gu < w and 0 <= gv < h:
                    hg, sg, vg = hsv[gv, gu]
                    if 35 <= hg <= 90 and sg > 40:
                        if best is None or abs(d) < abs(best[0]):
                            best = (d, su, sv)
        if best is not None:
            snapped.append((best[1], best[2]))
    return np.array(snapped) if snapped else np.empty((0, 2))


def _straightness_cost(line_sets, K, dist):
    """Total RMS perpendicular residual of each line after undistorting its points."""
    total, count = 0.0, 0
    for pts in line_sets:
        if len(pts) < 6:
            continue
        u = cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float32), K, dist, P=K).reshape(
            -1, 2
        )
        c = u.mean(0)
        uu = u - c
        # principal direction via SVD; perpendicular residual = projection on the minor axis
        _, _, V = np.linalg.svd(uu, full_matrices=False)
        perp = uu @ V[-1]
        total += float(np.sum(perp**2))
        count += len(u)
    return np.sqrt(total / count) if count else float("inf")


def estimate_distortion(
    detector,
    video_path: str,
    pitch: PitchModel = STANDARD_PITCH,
    n_frames: int = 30,
    k1_range=(-0.40, 0.20),
    n_steps: int = 25,
    min_samples: int = 200,
    min_improvement: float = 0.12,
    focal: Optional[float] = None,
) -> DistortionModel:
    """Estimate radial distortion by straightening the imaged pitch lines over many frames.

    Returns a DistortionModel; .enabled is False (no-op) unless there is enough evidence AND
    undistortion measurably straightens the lines.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_frames * 20
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idxs = np.linspace(0, max(0, total - 1), n_frames).astype(int)
    min_span = 0.25 * max(w, h)  # only keep lines that span a good fraction of the frame

    line_sets: List[np.ndarray] = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        try:
            lines = detector.detect_line_points(frame)  # robust trained-detector line ridges
        except Exception:
            continue
        for pts in lines:
            if len(pts) < 20:
                continue
            uu = pts - pts.mean(0)
            # PCA: primary singular value ~ line length; keep only long lines (best signal)
            s = np.linalg.svd(uu, compute_uv=False)
            if s[0] / np.sqrt(len(pts)) * 2.0 >= min_span:
                line_sets.append(pts)
    cap.release()

    n_samples = int(sum(len(s) for s in line_sets))
    K = np.array(
        [[focal or max(w, h), 0, w / 2.0], [0, focal or max(w, h), h / 2.0], [0, 0, 1.0]],
        dtype=np.float64,
    )

    if n_samples < min_samples or len(line_sets) < 4:
        return DistortionModel(
            K=K, dist=np.zeros(4), size=(w, h), n_samples=n_samples, source="none"
        )

    base = _straightness_cost(line_sets, K, np.zeros(4))
    best_k1, best_cost = 0.0, base
    for k1 in np.linspace(k1_range[0], k1_range[1], n_steps):
        c = _straightness_cost(line_sets, K, np.array([k1, 0, 0, 0], float))
        if c < best_cost:
            best_cost, best_k1 = c, float(k1)

    improvement = (base - best_cost) / base if base > 0 else 0.0
    if improvement < min_improvement or abs(best_k1) < 1e-3:
        return DistortionModel(
            K=K,
            dist=np.zeros(4),
            size=(w, h),
            n_samples=n_samples,
            improvement=improvement,
            source="none",
        )
    return DistortionModel(
        K=K,
        dist=np.array([best_k1, 0, 0, 0], float),
        size=(w, h),
        n_samples=n_samples,
        improvement=improvement,
        source="estimated",
    )
