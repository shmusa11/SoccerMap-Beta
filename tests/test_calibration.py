import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from soccer_analyzer.calibration.clip_calibrator import _propagate_static, calibrate_clip_auto
from soccer_analyzer.calibration.field_detector import CachedHomographyDetector, FieldObservation
from soccer_analyzer.calibration.keypoint_calib import KP_WORLD, keypoint_homography
from soccer_analyzer.calibration.lookup import CalibrationLookup
from soccer_analyzer.calibration.mapper import PitchMapper


def test_rejected_sample_cannot_borrow_neighbor():
    lookup = CalibrationLookup({0: np.eye(3), 2: None, 4: np.eye(3)})
    assert lookup.at(2) is None
    assert lookup.at(3) is None
    assert lookup.at(20) is None
    assert lookup.coverage(5, 2) == (2, 3)


def test_empty_lookup():
    assert CalibrationLookup({}).coverage(9, 3) == (0, 3)


def test_no_propagation_across_camera_motion():
    entries = [
        dict(idx=0, Hv=np.eye(3), motion=0),
        dict(idx=2, Hv=None, motion=1),
        dict(idx=4, Hv=None, motion=5),
    ]
    result, stats = _propagate_static(entries, max_gap=90)
    assert result[2] is not None
    assert result[4] is None
    assert stats["covered"] == 2


@pytest.mark.parametrize("matrix", [None, np.zeros((3, 3)), np.eye(2), np.full((3, 3), np.nan)])
def test_invalid_field_observation(matrix):
    assert not FieldObservation(0, matrix).is_valid


def test_cached_detector_rejects_singular_matrix(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"homography": np.zeros((3, 3)).tolist()}))
    with pytest.raises(ValueError, match="invertible"):
        CachedHomographyDetector(path)


def test_mapper_injection_and_expiration():
    detector = Mock()
    detector.detect.side_effect = [FieldObservation(0, np.eye(3), n_landmarks=8), None, None, None]
    mapper = PitchMapper(detector=detector, keyframe_interval=1, max_hold_frames=2)
    frame = np.zeros((100, 100, 3), np.uint8)
    assert mapper.update(frame).source == "detect"
    np.testing.assert_allclose(mapper.transform([(5, 8)]), [[5, 8]])
    assert mapper.update(frame).source == "held"
    mapper.update(frame)
    assert mapper.update(frame).source == "none"
    assert mapper.transform([(5, 8)]).size == 0


def test_ransac_threshold_is_pixels_not_metres():
    indices = [1, 2, 3, 5, 6, 25, 26, 28, 29, 30, 45, 51, 57]
    points = {f"pnl_{i}": tuple(np.array(KP_WORLD[i - 1]) * 10 + [50, 40]) for i in indices}
    # 30 pixels = 3 metres. This must not pass a 6-pixel RANSAC threshold.
    points["pnl_51"] = tuple(np.array(points["pnl_51"]) + [30, 0])
    matrix, count = keypoint_homography(SimpleNamespace(keypoints=points), ransac_px=6)
    assert count == len(indices) - 1
    np.testing.assert_allclose(
        matrix / matrix[2, 2], [[0.1, 0, -5], [0, 0.1, -4], [0, 0, 1]], atol=1e-5
    )


def test_calibration_honors_limit_when_frame_count_unknown():
    cap = Mock()
    cap.get.return_value = 0
    cap.grab.return_value = True
    cap.retrieve.return_value = (True, np.zeros((20, 20, 3), np.uint8))
    detector = Mock()
    detector.detect.return_value = None
    with patch("soccer_analyzer.video.cv2.VideoCapture", return_value=cap):
        result, stats = calibrate_clip_auto(detector, "clip", stride=2, max_frames=5)
    assert list(result) == [0, 2, 4]
    assert stats["frames"] == 3
    assert cap.grab.call_count == 5
    assert cap.retrieve.call_count == 3  # Unsampled frames need no BGR conversion.
    cap.release.assert_called_once()
