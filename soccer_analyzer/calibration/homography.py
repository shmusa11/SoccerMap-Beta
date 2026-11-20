"""Sign normalisation and a camera-model plausibility test for image->pitch homographies.

A homography H and -H describe the same projective mapping, but every consumer that asks "is
this pitch point in front of the camera?" reads the SIGN of the homogeneous w coordinate. A
solver may return either sign, so a correct calibration could be rejected as "folded behind the
camera" or drawn as nothing. `normalize_sign` picks the sign under which the ground visible at
the bottom of the frame lies in front of the camera.

`camera_consistency` asks whether H could come from a real pinhole camera (square pixels,
principal point near the image centre) above the pitch plane. For such a camera,
inv(H) = s * K [r1 r2 t], so after removing K the first two columns must be orthogonal and of
equal length. Shears, folds and mirrored fits violate this whichever pitch corners are visible,
unlike the old "all four pitch corners in front of the camera" rule, which rejected normal
sideline views of one half of the pitch yet accepted a convex shear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def normalize_sign(H: np.ndarray, frame_size: Tuple[int, int], samples: int = 9) -> np.ndarray:
    """Return +H or -H so that pixels along the bottom of the frame map with positive w.

    H maps image pixels to pitch metres. The bottom image row of an upright camera views the
    ground in front of it; under the returned matrix those pixels have (H @ [u, v, 1])[2] > 0,
    so pitch points they see also have positive w under inv(H).
    """
    w, h = frame_size
    us = np.linspace(0.1 * w, 0.9 * w, samples)
    pix = np.stack([us, np.full(samples, h - 1.0), np.ones(samples)])
    return -H if float(np.sum(np.sign((H @ pix)[2]))) < 0 else H


@dataclass(frozen=True)
class CameraFit:
    """Pinhole interpretation of a homography (focal in pixels, height in pitch metres)."""

    focal_px: float  # least-squares focal length; nan when not positive
    column_cos: float  # |cos| of the angle between the recovered rotation columns (0 = ideal)
    norm_ratio: float  # longer / shorter rotation column length (1 = ideal)
    height_m: float = float("nan")  # camera height above the pitch plane; negative = mirrored


def camera_fit(H: np.ndarray, frame_size: Tuple[int, int]) -> Optional[CameraFit]:
    """Recover the focal length and height implied by H and how far H is from a rigid camera.

    H maps image pixels to pitch metres and should already be sign-normalised (see
    `normalize_sign`) for the height to be meaningful. Returns None for a singular or non-finite
    H. Pixel coordinates are centred on the image and scaled by the frame width so the two
    constraints are comparably conditioned.
    """
    if H is None or not np.all(np.isfinite(H)):
        return None
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    w, h = frame_size
    T = np.array([[1.0 / w, 0.0, -0.5], [0.0, 1.0 / w, -0.5 * h / w], [0.0, 0.0, 1.0]])
    A = T @ Hi
    A = A / np.linalg.norm(A)
    a1, a2 = A[:, 0], A[:, 1]
    # With K = diag(f, f, 1) (f in frame widths) and u = 1/f^2, the rotation columns are
    # (ax*sqrt(u), ay*sqrt(u), az). Orthogonality and equal length are linear in u.
    p = np.array([a1[0] * a2[0] + a1[1] * a2[1], a1[0] ** 2 + a1[1] ** 2 - a2[0] ** 2 - a2[1] ** 2])
    q = np.array([a1[2] * a2[2], a1[2] ** 2 - a2[2] ** 2])
    if float(p @ p) < 1e-18:
        return None
    u = -float(p @ q) / float(p @ p)
    if u <= 0:
        return CameraFit(float("nan"), 1.0, float("inf"))
    s = np.sqrt(u)
    r1 = np.array([a1[0] * s, a1[1] * s, a1[2]])
    r2 = np.array([a2[0] * s, a2[1] * s, a2[2]])
    n1, n2 = float(np.linalg.norm(r1)), float(np.linalg.norm(r2))
    if n1 < 1e-12 or n2 < 1e-12:
        return None
    cos = abs(float(r1 @ r2)) / (n1 * n2)
    # With visible points at positive w the overall scale is positive, so r3 = r1 x r2 is the
    # proper rotation's third column and the camera centre is C = -R^T t. Pitch x runs along the
    # length and y towards the near touchline, so the right-handed z = x cross y points INTO the
    # ground: the height above the pitch is -C_z = r3 . t (negative for a mirrored fit).
    lam = 0.5 * (n1 + n2)
    t = np.array([A[0, 2] * s, A[1, 2] * s, A[2, 2]]) / lam
    r3 = np.cross(r1 / lam, r2 / lam)
    return CameraFit(float(w / s), cos, max(n1, n2) / min(n1, n2), float(r3 @ t))


def decompose_camera(
    H: np.ndarray, frame_size: Tuple[int, int], pitch_length: float, pitch_width: float
) -> Optional[dict]:
    """Pinhole camera parameters implied by an image->pitch homography, or None.

    Returns PnLCalib-style parameters (the convention `ls_calibrator.homography_from_rvec`
    rebuilds homographies from): world origin at the pitch centre, z into the ground, so a camera
    above the pitch has a negative z position. The principal point is the image centre and the
    rotation is the nearest proper rotation to the recovered columns.
    """
    Hn = normalize_sign(H, frame_size)
    fit = camera_fit(Hn, frame_size)
    if fit is None or not np.isfinite(fit.focal_px):
        return None
    w, h = frame_size
    f = fit.focal_px
    Q = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
    centre = np.array([[1.0, 0.0, pitch_length / 2.0], [0.0, 1.0, pitch_width / 2.0], [0, 0, 1.0]])
    try:
        M = np.linalg.inv(Q) @ np.linalg.inv(Hn) @ centre
    except np.linalg.LinAlgError:
        return None
    s = 0.5 * (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    if s <= 0:
        return None
    r1, r2, t = M[:, 0] / s, M[:, 1] / s, M[:, 2] / s
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    position = -R.T @ t
    return {
        "x_focal_length": float(f),
        "y_focal_length": float(f),
        "principal_point": [w / 2.0, h / 2.0],
        "position_meters": [float(v) for v in position],
        "rotation_matrix": R,
    }


def camera_consistent(
    H: np.ndarray,
    frame_size: Tuple[int, int],
    max_cos: float = 0.2,
    max_ratio: float = 1.35,
    focal_range: Tuple[float, float] = (0.25, 40.0),
    height_range_m: Tuple[float, float] = (0.5, 200.0),
) -> bool:
    """True if a sign-normalised H is close to a real pinhole camera above the pitch plane.

    focal_range is in frame widths (0.25 = ultra-wide, 40 = long telephoto). The angle and
    ratio tolerances absorb lens distortion and an off-centre principal point. A camera below
    the pitch plane means a mirrored fit. Sensitivity to shear depends on the viewpoint: on a
    steep oblique view a mild shear can be absorbed by a different focal length (on a synthetic
    half-pitch view a 0.6 shear passes and 0.8 fails), so this is a plausibility filter that
    complements the line-support check rather than a verifier on its own.
    """
    fit = camera_fit(H, frame_size)
    if fit is None or not np.isfinite(fit.focal_px):
        return False
    w = frame_size[0]
    lo, hi = focal_range
    return (
        fit.column_cos <= max_cos
        and fit.norm_ratio <= max_ratio
        and lo * w <= fit.focal_px <= hi * w
        and height_range_m[0] <= fit.height_m <= height_range_m[1]
    )


def camera_inside_pitch(
    H: np.ndarray,
    frame_size: Tuple[int, int],
    pitch_length: float,
    pitch_width: float,
    margin_m: float = 3.0,
    max_height_m: float = 30.0,
) -> bool:
    """True if H puts a low camera (< max_height_m) on the pitch or within margin_m of it.

    Match footage is filmed from beside the field, so a fit placing the camera on the pitch is
    almost always a wrong decomposition: on a bardNight frame a fit placing the camera 1.3 m
    inside the near touchline scored higher line support yet tilted the halfway line by up to
    175 px near the camera, while the fit with the camera 12 m behind the touchline was within
    40 px. Benches and stands sit several metres back, hence the margin. High cameras (drones)
    are not penalised."""
    cam = decompose_camera(H, frame_size, pitch_length, pitch_width)
    if cam is None:
        return False
    x, y, z = cam["position_meters"]
    x, y, height = x + pitch_length / 2.0, y + pitch_width / 2.0, -z
    return (
        height < max_height_m
        and -margin_m < x < pitch_length + margin_m
        and -margin_m < y < pitch_width + margin_m
    )
