"""Keypoint-anchored calibration — break the symmetry the lines alone can't.

On amateur/night footage PnLCalib's HOMOGRAPHY voting is garbage (camera position jumps
100s of metres frame-to-frame), but its raw KEYPOINT detections carry their fixed pitch
INDEX — i.e. a labelled correspondence (this pixel == that named pitch point). A handful of
labelled points is exactly what line-support optimisation lacks: it tells you WHICH penalty
box / WHICH half you're looking at, killing the 2-fold pitch symmetry that makes line-support
ambiguous on partial views.

So we build our OWN homography from those labelled correspondences (RANSAC, so a few mislabels
don't sink it). Because the correspondences pin the geometry, the result can't shear or flip
the way a free line-support solve can — it lands in the correct basin. line_support then just
grades it. The camera is fixed, so we keep the best-graded anchor and propagate it.

PnLCalib keypoint world coords (utils/utils_keypointsWC.py, 57 pts, 1-based) live in the SAME
105x68 corner-origin frame as our PitchModel, so for a different field size we just rescale.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH

# PnLCalib's 57 keypoint world coordinates (metres, standard 105x68, corner origin).
KP_WORLD = [
    [0.0, 0.0],
    [52.5, 0.0],
    [105.0, 0.0],
    [0.0, 13.84],
    [16.5, 13.84],
    [88.5, 13.84],
    [105.0, 13.84],
    [0.0, 24.84],
    [5.5, 24.84],
    [99.5, 24.84],
    [105.0, 24.84],
    [0.0, 30.34],
    [0.0, 30.34],
    [105.0, 30.34],
    [105.0, 30.34],
    [0.0, 37.66],
    [0.0, 37.66],
    [105.0, 37.66],
    [105.0, 37.66],
    [0.0, 43.16],
    [5.5, 43.16],
    [99.5, 43.16],
    [105.0, 43.16],
    [0.0, 54.16],
    [16.5, 54.16],
    [88.5, 54.16],
    [105.0, 54.16],
    [0.0, 68.0],
    [52.5, 68.0],
    [105.0, 68.0],
    [16.5, 26.68],
    [52.5, 24.85],
    [88.5, 26.68],
    [16.5, 41.31],
    [52.5, 43.15],
    [88.5, 41.31],
    [19.99, 32.29],
    [43.68, 31.53],
    [61.31, 31.53],
    [85.0, 32.29],
    [19.99, 35.7],
    [43.68, 36.46],
    [61.31, 36.46],
    [85.0, 35.7],
    [11.0, 34.0],
    [16.5, 34.0],
    [20.15, 34.0],
    [46.03, 27.53],
    [58.97, 27.53],
    [43.35, 34.0],
    [52.5, 34.0],
    [61.5, 34.0],
    [46.03, 40.47],
    [58.97, 40.47],
    [84.85, 34.0],
    [88.5, 34.0],
    [94.0, 34.0],
]
# PnLCalib masks these indices (duplicate / unreliable channels).
MASKED_KP = {12, 15, 16, 19}
_STD_L, _STD_W = 105.0, 68.0


def keypoint_correspondences(
    observation, pitch: PitchModel = STANDARD_PITCH
) -> Tuple[np.ndarray, np.ndarray]:
    """(img_px Nx2, pitch_m Nx2) from a FieldObservation's labelled keypoints, rescaled to the
    given field size."""
    img, world = [], []
    sx, sy = pitch.length / _STD_L, pitch.width / _STD_W
    for name, (u, v) in (getattr(observation, "keypoints", None) or {}).items():
        try:
            idx = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if idx in MASKED_KP or idx < 1 or idx > len(KP_WORLD):
            continue
        wx, wy = KP_WORLD[idx - 1]
        img.append([u, v])
        world.append([wx * sx, wy * sy])
    return np.array(img, np.float32), np.array(world, np.float32)


def homography_plausible(
    H: np.ndarray,
    pitch: PitchModel = STANDARD_PITCH,
    frame_size: Optional[Tuple[int, int]] = None,
) -> bool:
    """Geometric sanity check that line_support + spread cannot provide.

    With `frame_size` (pixels): H must describe a real pinhole camera above the pitch plane
    (`camera_consistent` after sign normalisation). This rejects shears, folds and mirrored fits
    while accepting the sign ambiguity of H and sideline views where a far pitch corner lies
    behind the camera.

    Without `frame_size` (legacy behaviour, kept for the multi-sport path): the pitch rectangle
    must map to a convex, front-facing quad. That rule rejects a correct H whose sign is flipped
    and normal views of one half of the pitch, so new callers should pass the frame size."""
    if H is None or not np.all(np.isfinite(H)):
        return False
    if frame_size is not None:
        from soccer_analyzer.calibration.homography import camera_consistent, normalize_sign

        return camera_consistent(normalize_sign(H, frame_size), frame_size)
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return False
    L, W = pitch.length, pitch.width
    corners = np.array([[0, 0, 1], [L, 0, 1], [L, W, 1], [0, W, 1]], float)
    q = (Hi @ corners.T).T
    if np.any(q[:, 2] <= 1e-6):  # a pitch corner folds behind the camera
        return False
    p = q[:, :2] / q[:, 2:3]
    # convexity: all consecutive edge cross-products share one sign
    cross = []
    for i in range(4):
        a = p[(i + 1) % 4] - p[i]
        b = p[(i + 2) % 4] - p[(i + 1) % 4]
        cross.append(a[0] * b[1] - a[1] * b[0])
    cross = np.array(cross)
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
    # not a degenerate needle: shortest side at least 4% of the longest (rejects extreme skew
    # while still allowing oblique grazing views of a partially-visible field)
    sides = [np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)]
    return min(sides) > 0.04 * max(sides)


def _general_position(pts: np.ndarray, min_ratio: float = 0.10, min_extent: float = 1e-3) -> bool:
    """True if the points span 2D (not collinear). A homography fit from near-collinear points
    is degenerate — it can place the field anywhere along the missing direction, which is how the
    wrong-LOCATION fits (that still score line_support on dense football clutter) slip through."""
    if len(pts) < 4:
        return False
    c = pts - pts.mean(0)
    sv = np.linalg.svd(c, compute_uv=False)
    return sv[-1] > min_extent and (sv[-1] / sv[0]) > min_ratio


def keypoint_homography(
    observation, pitch: PitchModel = STANDARD_PITCH, ransac_px: float = 6.0, min_inliers: int = 5
) -> Tuple[Optional[np.ndarray], int]:
    """RANSAC image(px)->pitch(m) homography from labelled keypoints. Returns (H, n_inliers).

    Requires >=min_inliers correspondences in GENERAL POSITION (not collinear) in both image and
    world — collinear keypoints (common on these partial/oblique views, e.g. several points along
    one goal line) under-determine the homography and produce confident wrong-location fits."""
    img, world = keypoint_correspondences(observation, pitch)
    if len(img) < min_inliers:
        return None, 0
    # OpenCV measures RANSAC residuals in DESTINATION units. Fit into pixels,
    # then invert, so ransac_px does not accidentally permit metre-scale errors.
    inverse, mask = cv2.findHomography(world, img, cv2.RANSAC, ransac_px)
    if inverse is None or not np.all(np.isfinite(inverse)) or mask is None:
        return None, 0
    inl = mask.ravel().astype(bool)
    n = int(inl.sum())
    if n < min_inliers:
        return None, n
    if not _general_position(img[inl]) or not _general_position(world[inl]):
        return None, n
    try:
        H = np.linalg.inv(inverse)
    except np.linalg.LinAlgError:
        return None, n
    return H, n
