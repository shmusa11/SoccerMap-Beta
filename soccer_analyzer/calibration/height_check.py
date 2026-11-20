"""Advisory player-height check for a calibrated pinhole camera.

This assumes square pixels, zero skew, a principal point at the image center and upright
players. Cropping, lens distortion and crouching can violate those assumptions. The check
can warn about scale; it cannot certify a field or detect a mirrored/translated mapping.
"""

from __future__ import annotations

import numpy as np


def camera_from_homography(matrix: np.ndarray, image_size: tuple[int, int]) -> np.ndarray | None:
    """Infer a 3x4 pitch-metres-to-image camera, or None when focal length is ambiguous."""
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    width, height = image_size
    centered = np.array([[1, 0, -width / 2], [0, 1, -height / 2], [0, 0, 1]]) @ inverse
    a, b = centered[:, 0], centered[:, 1]
    # Rotation columns must be orthogonal and equal in length after K^-1.
    coefficients = np.array([a[:2] @ b[:2], a[:2] @ a[:2] - b[:2] @ b[:2]])
    target = np.array([-a[2] * b[2], b[2] ** 2 - a[2] ** 2])
    denominator = coefficients @ coefficients
    if denominator < 1e-16:
        return None
    inverse_focal_sq = float(coefficients @ target / denominator)
    if inverse_focal_sq <= 0 or not np.isfinite(inverse_focal_sq):
        return None
    focal = 1 / np.sqrt(inverse_focal_sq)
    if not width * 0.3 <= focal <= width * 5:
        return None
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    plane = np.linalg.solve(intrinsics, inverse)
    scale = (np.linalg.norm(plane[:, 0]) + np.linalg.norm(plane[:, 1])) / 2
    if not np.isfinite(scale) or scale <= 1e-12:
        return None
    plane /= scale
    if (
        abs(plane[:, 0] @ plane[:, 1]) > 0.1
        or abs(np.linalg.norm(plane[:, 0]) - np.linalg.norm(plane[:, 1])) > 0.1
    ):
        return None
    vertical = np.cross(plane[:, 0], plane[:, 1])
    vertical /= np.linalg.norm(vertical)
    # Choose the side of the field plane containing the camera as positive height.
    if vertical @ plane[:, 2] > 0:
        vertical *= -1
    return intrinsics @ np.column_stack([plane[:, 0], plane[:, 1], vertical, plane[:, 2]])


def estimate_height(
    camera: np.ndarray, foot_m: tuple[float, float], head_px: tuple[float, float]
) -> float | None:
    """Least-squares upright height in metres; callers must exclude interpolated/partial boxes."""
    ground = camera @ np.array([*foot_m, 0, 1])
    vertical = camera[:, 2]
    head = np.asarray(head_px, np.float64)
    coefficients = head * vertical[2] - vertical[:2]
    target = ground[:2] - head * ground[2]
    denominator = coefficients @ coefficients
    if denominator < 1e-12:
        return None
    height = float(coefficients @ target / denominator)
    if not np.isfinite(height) or height <= 0:
        return None
    projected = ground + height * vertical
    if projected[2] <= 1e-8:
        return None
    # Heads displaced far sideways are not upright evidence (occlusion or another person).
    residual = np.linalg.norm(projected[:2] / projected[2] - head)
    return height if residual < 12 else None


def summarize_heights(heights: list[float]) -> dict:
    """Report a broad scale warning only with at least 20 usable upright estimates."""
    usable = [h for h in heights if np.isfinite(h) and h > 0]
    if len(usable) < 20:
        return {"state": "unavailable", "samples": len(usable), "median_m": None}
    median = float(np.median(usable))
    fraction = float(np.mean([(h < 1.1 or h > 2.4) for h in usable]))
    return {
        "state": "review" if fraction > 0.65 else "no_warning",
        "samples": len(usable),
        "median_m": round(median, 2),
        "outlier_fraction": round(fraction, 3),
    }
