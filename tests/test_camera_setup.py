"""Model-free camera setup geometry and reference-reuse regression tests."""

import cv2
import numpy as np
import pytest

from soccer_analyzer.calibration.camera import CameraReference, calibrate_from_reference
from soccer_analyzer.calibration.manual import landmarks, solve_manual
from soccer_analyzer.calibration.pitch_model import PitchModel


def field_points():
    return [
        {"landmark": name, "x": x, "y": y}
        for name, x, y in (
            ("left_top", 60, 50),
            ("right_top", 580, 50),
            ("right_bottom", 580, 400),
            ("left_bottom", 60, 400),
        )
    ]


def textured_frame():
    rng = np.random.default_rng(42)
    frame = np.full((450, 640, 3), (40, 120, 50), np.uint8)
    for _ in range(180):
        x, y = rng.integers([10, 10], [630, 440])
        color = tuple(int(c) for c in rng.integers(30, 240, 3))
        cv2.rectangle(frame, (x, y), (x + 5, y + 5), color, -1)
    return frame


def test_manual_four_points_are_order_independent_and_metric():
    points = field_points()[::-1]
    matrix = solve_manual(points, PitchModel(), (640, 450))
    image = np.array([[[p["x"], p["y"]] for p in points]], np.float32)
    actual = cv2.perspectiveTransform(image, matrix)[0]
    np.testing.assert_allclose(actual, [[0, 68], [105, 68], [105, 0], [0, 0]], atol=1e-5)


def test_custom_pitch_keeps_regulation_box_dimensions():
    points = {p["id"]: p for p in landmarks(PitchModel(100, 60))}
    assert points["left_penalty_top_inner"]["x"] == 16.5
    assert points["left_penalty_top_inner"]["y"] == pytest.approx(9.84)


@pytest.mark.parametrize(
    "problem", ["duplicate", "outside", "nonfinite", "collinear", "folded", "missing"]
)
def test_manual_rejects_ambiguous_geometry(problem):
    points = field_points()
    if problem == "duplicate":
        points[1]["landmark"] = points[0]["landmark"]
    elif problem == "outside":
        points[0]["x"] = 640
    elif problem == "nonfinite":
        points[0]["x"] = float("nan")
    elif problem == "collinear":
        for point in points[:3]:
            point["y"] = 50
    elif problem == "folded":
        points[1]["landmark"], points[2]["landmark"] = points[2]["landmark"], points[1]["landmark"]
    else:
        points.pop()
    with pytest.raises(ValueError):
        solve_manual(points, PitchModel(), (640, 450))


def test_camera_check_allows_foreground_motion_but_rejects_pan_zoom_and_blank():
    reference = textured_frame()
    guard = CameraReference(reference)
    assert guard.matches(reference.copy())
    player = reference.copy()
    cv2.rectangle(player, (250, 220), (265, 260), (0, 0, 220), -1)
    assert guard.matches(player)
    pan = cv2.warpAffine(reference, np.array([[1, 0, 15], [0, 1, 0]], np.float32), (640, 450))
    assert not guard.matches(pan)
    zoom = cv2.warpAffine(reference, cv2.getRotationMatrix2D((320, 225), 0, 1.05), (640, 450))
    assert not guard.matches(zoom)
    assert not guard.matches(np.zeros_like(reference))
    assert not CameraReference(np.zeros_like(reference)).matches(np.zeros_like(reference))
    assert not guard.matches(reference[:200])


def test_reference_reuse_keeps_rejected_source_frames_unavailable(tmp_path):
    reference = textured_frame()
    path = tmp_path / "camera.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 450))
    for i in range(9):
        writer.write(reference if i < 3 or i >= 6 else np.zeros_like(reference))
    writer.release()
    result, stats = calibrate_from_reference(
        str(path), reference, np.eye(3), stride=3, max_frames=9
    )
    assert list(result) == [0, 3, 6]
    assert result[0] is not None and result[3] is None and result[6] is not None
    assert stats["covered"] == 2 and stats["inference_calls"] == 0
    window, _ = calibrate_from_reference(
        str(path), reference, np.eye(3), stride=2, start_frame=3, max_frames=6
    )
    assert list(window) == [3, 5, 7]
    assert window[3] is None and window[5] is None and window[7] is not None


def test_automatic_calibration_reuses_verified_reference_without_more_inference(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from soccer_analyzer.calibration.clip_calibrator import calibrate_clip_auto

    reference = textured_frame()
    path = tmp_path / "static.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 450))
    for index in range(9):
        writer.write(np.zeros_like(reference) if index == 3 else reference)
    writer.release()
    monkeypatch.setattr(
        "soccer_analyzer.calibration.line_support.line_distance_field", lambda f: np.ones((2, 2))
    )
    monkeypatch.setattr("soccer_analyzer.calibration.line_support.line_support", lambda *a, **k: 1)
    monkeypatch.setattr("soccer_analyzer.calibration.ls_calibrator.spread_score", lambda *a: 1)
    monkeypatch.setattr(
        "soccer_analyzer.calibration.keypoint_calib.homography_plausible", lambda *a, **k: True
    )
    monkeypatch.setattr(
        "soccer_analyzer.calibration.keypoint_calib.keypoint_homography", lambda *a: (None, 0)
    )
    detector = Mock()

    def detect(frame, index):
        if index == 3:
            raise RuntimeError("unusable observation")
        return SimpleNamespace(homography=np.eye(3), reproj_err=None)

    detector.detect.side_effect = detect
    mappings, stats = calibrate_clip_auto(detector, str(path), stride=1, max_frames=9)
    assert detector.detect.call_count == 2  # First anchor and the explicitly rejected blank.
    assert stats["reused"] == 7
    assert stats["inference_calls"] == 2  # Include the failed attempt in cost counters.
    assert mappings[3] is None and mappings[4] is not None
    detector.reset_mock()
    window, stats = calibrate_clip_auto(detector, str(path), start_frame=2, stride=1, max_frames=4)
    assert list(window) == [2, 3, 4, 5]
    assert window[3] is None and detector.detect.call_count == 2
