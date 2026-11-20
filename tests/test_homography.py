import cv2
import numpy as np
import pytest

from soccer_analyzer.calibration.clip_calibrator import select_calibration
from soccer_analyzer.calibration.homography import camera_consistent, camera_fit, normalize_sign
from soccer_analyzer.calibration.keypoint_calib import homography_plausible
from soccer_analyzer.calibration.mapper import FrameCalibration
from soccer_analyzer.calibration.pitch_model import STANDARD_PITCH
from soccer_analyzer.calibration.visualize import draw_calibration_overlay

SIZE = (960, 540)


def camera_homography(position, target, focal_widths, size=SIZE):
    """image(px) -> pitch(m) homography of a pinhole camera at `position` (x, y, height) looking
    at `target`. Pitch x runs along the length, y towards the near touchline."""
    w, h = size
    c = np.asarray(position, float)
    forward = np.asarray(target, float) - c
    forward /= np.linalg.norm(forward)
    right = np.cross([0.0, 0.0, 1.0], forward)
    right /= np.linalg.norm(right)
    down = np.cross(right, forward)
    R = np.stack([right, down, forward])
    K = np.array([[focal_widths * w, 0, w / 2], [0, focal_widths * w, h / 2], [0, 0, 1.0]])
    return np.linalg.inv(K @ np.column_stack([R[:, 0], R[:, 1], -R @ c]))


CENTRE_VIEW = camera_homography((52.5, 83, 12), (52.5, 30, 0), 1.3)
# Panned to the left half from close to the touchline: pitch corner (105, 68) is behind the camera.
HALF_VIEW = camera_homography((52.5, 75, 6), (8, 30, 0), 0.9)


def test_camera_fit_recovers_focal_and_height():
    fit = camera_fit(CENTRE_VIEW, SIZE)
    assert fit.focal_px == pytest.approx(1.3 * SIZE[0], rel=1e-6)
    assert fit.height_m == pytest.approx(12.0, rel=1e-6)
    assert fit.column_cos == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("H", [CENTRE_VIEW, HALF_VIEW])
def test_sign_flipped_and_half_pitch_views_are_plausible(H):
    # Regression: the corner rule rejected both -H (same mapping) and a camera panned to one half.
    for candidate in (H, -H):
        assert homography_plausible(candidate, STANDARD_PITCH, frame_size=SIZE)
        assert camera_fit(normalize_sign(candidate, SIZE), SIZE).height_m > 0
    assert not homography_plausible(HALF_VIEW, STANDARD_PITCH)  # documents the legacy behaviour


def test_shear_mirror_and_degenerate_fits_are_rejected():
    shear = np.array([[1, 0.3, 0], [0, 1, 0], [0, 0, 1.0]]) @ CENTRE_VIEW
    mirror = np.array([[-1, 0, 105], [0, 1, 0], [0, 0, 1.0]]) @ CENTRE_VIEW
    assert homography_plausible(shear, STANDARD_PITCH)  # legacy rule misses the shear
    for H in (shear, mirror, np.zeros((3, 3)), np.full((3, 3), np.nan)):
        assert not homography_plausible(H, STANDARD_PITCH, frame_size=SIZE)
    assert not camera_consistent(normalize_sign(mirror, SIZE), SIZE)


def _line_distance_field(H):
    """Distance field of the pitch markings as this camera would see them."""
    mask = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    pts = STANDARD_PITCH.sample_markings(0.25)
    img = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), np.linalg.inv(H))
    for u, v in img.reshape(-1, 2):
        if 0 <= u < SIZE[0] and 0 <= v < SIZE[1]:
            mask[int(v), int(u)] = 255
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)


def test_select_calibration_returns_verified_positive_sign_fit():
    df = _line_distance_field(HALF_VIEW)
    # On this steep view a mild shear is absorbable by another focal length (0.6 still passes
    # as a different camera); 1.0 is not.
    shear = np.array([[1, 1.0, 0], [0, 1, 0], [0, 0, 1.0]]) @ HALF_VIEW
    pick = select_calibration([("shear", shear), ("flipped", -HALF_VIEW)], df, SIZE, STANDARD_PITCH)
    assert pick is not None
    H, support, tag = pick
    assert tag == "flipped" and support > 0.9
    np.testing.assert_allclose(H, HALF_VIEW)  # sign-normalised
    assert select_calibration([("shear", shear)], df, SIZE, STANDARD_PITCH) is None
    assert select_calibration([("ok", HALF_VIEW)], None, SIZE, STANDARD_PITCH) is None


def test_overlay_draws_identically_for_either_sign():
    frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    drawn = []
    for H in (HALF_VIEW, -HALF_VIEW):
        calib = FrameCalibration(0, H, 1.0, "detect", 0, 0)
        out = draw_calibration_overlay(frame, calib, fade_by_confidence=False)
        drawn.append(int((out[80:, :, 1] > 150).sum()))  # below the HUD bar
    assert drawn[0] > 500 and drawn[0] == drawn[1]


def test_selection_prefers_camera_beside_the_field():
    from soccer_analyzer.calibration.homography import camera_inside_pitch

    inside = camera_homography((50, 69, 8), (20, 25, 0), 0.9)  # 1 m behind the touchline
    drone = camera_homography((52.5, 34, 60), (52.5, 20, 0), 0.9)
    assert camera_inside_pitch(inside, SIZE, 105, 68)
    assert not camera_inside_pitch(HALF_VIEW, SIZE, 105, 68)
    assert not camera_inside_pitch(drone, SIZE, 105, 68)  # high cameras are allowed over the pitch
    # The lines are drawn from the inside camera, so it has the higher line support; the
    # camera beside the field must still win while it passes the support floor.
    df = _line_distance_field(inside)
    beside_support = select_calibration([("b", HALF_VIEW)], df, SIZE, STANDARD_PITCH, ls_floor=0.0)[
        1
    ]
    inside_support = select_calibration([("i", inside)], df, SIZE, STANDARD_PITCH, ls_floor=0.0)[1]
    assert inside_support > beside_support
    pick = select_calibration(
        [("inside", inside), ("beside", HALF_VIEW)], df, SIZE, STANDARD_PITCH, ls_floor=0.0
    )
    assert pick[2] == "beside"
