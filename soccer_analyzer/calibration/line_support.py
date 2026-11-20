"""
Programmatic line evidence — the cheap "what a human sees" check that verifies the CNN.

The turf is a smooth, near-constant background; the painted lines (and people) pop out as
local high-contrast structure. A top-hat filter extracts exactly that ("brighter than the
local surroundings"), independent of overall brightness, so it works at night.

We use it two ways:
  * line_support(H): score how well a calibration's reprojected pitch markings land on the
    REAL extracted line pixels. A correct calibration scores high; a wrong-but-low-reproj
    one scores low — catching the failures reprojection error cannot.
  * (future) snap reprojected lines to the extracted pixels to refine accuracy.
"""

from __future__ import annotations


import cv2
import numpy as np

from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH


def line_mask(bgr: np.ndarray, kernel: int = 25, sat_max: int = 90) -> np.ndarray:
    """Binary mask of painted-line pixels via white top-hat (lines = thin bright structures
    that pop out of the smooth turf). Robust to night footage (local contrast, not absolute)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.5, (8, 8)).apply(gray)  # consistent night-enhance for all callers
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    t = max(12, int(np.mean(th) + 2.5 * np.std(th)))
    mask = ((th > t) & (sat < sat_max)).astype(np.uint8) * 255
    return cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )


def line_distance_field(bgr: np.ndarray) -> np.ndarray:
    """Distance (px) from every pixel to the nearest extracted line pixel."""
    m = line_mask(bgr)
    if (m > 0).sum() < 50:
        return None
    return cv2.distanceTransform(255 - m, cv2.DIST_L2, 3)


def _keep_line_components(raw: np.ndarray) -> np.ndarray:
    """Keep thin/elongated components (lines, arcs); drop filled blobs (players, jerseys, track
    surfaces, scoreboards)."""
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    out = np.zeros_like(raw)
    big = 0.05 * raw.size
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 8 or area > big:
            continue
        fill = area / (bw * bh + 1e-6)
        elong = max(bw, bh) / max(1, min(bw, bh))
        if elong >= 4 or fill < 0.30:
            out[lbl == i] = 255
    return out


def red_line_mask(bgr: np.ndarray) -> np.ndarray:
    """Reddish/orange painted LINES (R clearly above G, then a line-shape filter), e.g. soccer
    lines on a multi-sport field where the football yard lines are white. The shape filter drops
    red jerseys / red running tracks / buildings (blobs) so this fires on coloured-line venues,
    not on any field that happens to have red things. Lets us verify against the ACTUAL soccer
    markings instead of the white clutter that fools plain line_support."""
    b, g, r = cv2.split(bgr.astype(np.int16))
    m = (np.clip(r - g, 0, 255) > 45).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return _keep_line_components(m)


def red_line_distance_field(bgr: np.ndarray, min_px: int = 6000):
    """Distance field of reddish soccer lines, plus the red-pixel count. Returns (df, n_px).
    df is None unless there are enough red-line pixels to indicate a coloured-soccer-line venue
    (so the red gate only activates on multi-sport fields, never on normal white-line pitches)."""
    m = red_line_mask(bgr)
    n = int((m > 0).sum())
    if n < min_px:
        return None, n
    return cv2.distanceTransform(255 - m, cv2.DIST_L2, 3), n


def line_support(
    homography: np.ndarray,
    dist_field: np.ndarray,
    pitch: PitchModel = STANDARD_PITCH,
    radius: float = 6.0,
    step: float = 0.5,
) -> float:
    """Fraction of the reprojected pitch markings that land on a real line pixel.

    Reproject the pitch lines+circles into the image via inv(H); a sample is 'supported' if
    the nearest extracted line pixel is within `radius`. Returns 0..1 over in-frame samples.
    A geometrically-correct calibration scores high even when reprojection error is low-but-
    wrong, because it checks against the ACTUAL painted lines, not the CNN's own keypoints."""
    if homography is None or dist_field is None or not np.all(np.isfinite(homography)):
        return 0.0
    try:
        H_inv = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return 0.0
    h, w = dist_field.shape
    pts = pitch.sample_markings(step)
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    q = (H_inv @ ph.T).T
    ok = np.abs(q[:, 2]) > 1e-6
    u = q[ok, 0] / q[ok, 2]
    v = q[ok, 1] / q[ok, 2]
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if inb.sum() < 20:
        return 0.0
    d = dist_field[v[inb].astype(int), u[inb].astype(int)]
    return float((d <= radius).mean())
