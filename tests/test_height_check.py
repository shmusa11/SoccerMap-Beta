import numpy as np
import pytest

from soccer_analyzer.calibration.height_check import (
    camera_from_homography,
    estimate_height,
    summarize_heights,
)


def test_upright_height_from_a_known_camera():
    intrinsics = np.array([[900, 0, 960], [0, 900, 540], [0, 0, 1]])
    # Tilted camera with an orthonormal rotation, looking down at the pitch.
    angle = np.deg2rad(55)
    rotation = np.array(
        [[1, 0, 0], [0, -np.cos(angle), -np.sin(angle)], [0, np.sin(angle), -np.cos(angle)]]
    )
    translation = np.array([-52.5, 25, 70])
    camera = intrinsics @ np.column_stack([rotation, translation])
    plane = camera[:, [0, 1, 3]]
    estimated = camera_from_homography(np.linalg.inv(plane), (1920, 1080))
    assert estimated is not None
    for x, y in [(30, 20), (65, 40), (90, 55)]:
        head = camera @ [x, y, 1.8, 1]
        actual = estimate_height(estimated, (x, y), head[:2] / head[2])
        assert actual == pytest.approx(1.8, abs=0.01)


def test_ambiguous_affine_view_and_small_samples_are_unavailable():
    assert camera_from_homography(np.eye(3), (640, 450)) is None
    assert camera_from_homography(np.zeros((3, 3)), (640, 450)) is None
    assert summarize_heights([1.8] * 19)["state"] == "unavailable"
    assert summarize_heights([1.8] * 30)["state"] == "no_warning"
    assert summarize_heights([4.0] * 30)["state"] == "review"
