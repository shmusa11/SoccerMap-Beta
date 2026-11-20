from contextlib import nullcontext
from unittest.mock import Mock

import numpy as np
import pytest

from soccer_analyzer.calibration.field_detector import PnLCalibFieldDetector


class FakeTensor:
    """Small CPU tensor double for testing adapter orchestration without PyTorch."""

    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def permute(self, *axes):
        return FakeTensor(self.array.transpose(axes))

    def float(self):
        return self

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.array, axis))

    def __truediv__(self, divisor):
        return self

    def to(self, device):
        return self


def backend(params):
    camera = Mock()
    camera.heuristic_voting.return_value = params
    torch = Mock()
    torch.from_numpy.side_effect = FakeTensor
    torch.no_grad.side_effect = nullcontext
    keypoints = {i: {"x": 0.1 * i, "y": 0.1 * i} for i in range(1, 6)}
    return {
        "torch": torch,
        "device": "cpu",
        "resize": Mock(side_effect=lambda _: FakeTensor(np.zeros((1, 3, 540, 960)))),
        "model": Mock(return_value=np.zeros((1, 2, 4, 4))),
        "model_l": Mock(return_value=np.zeros((1, 2, 4, 4))),
        "kp_from_hm": Mock(return_value=[]),
        "kp_from_hm_l": Mock(return_value=[]),
        "coords_to_dict": Mock(return_value=[{}]),
        "complete_keypoints": Mock(return_value=(keypoints, {})),
        "FramebyFrameCalib": Mock(return_value=camera),
    }


@pytest.mark.parametrize(
    "params,matrix", [(None, None), ({"rep_err": 100}, np.eye(3)), ({"rep_err": 1}, None)]
)
def test_keypoints_survive_failed_camera_fit(params, matrix):
    detector = PnLCalibFieldDetector()
    detector._loaded = backend(params)
    detector._homography_from_params = Mock(return_value=matrix)
    observation = detector.detect(np.zeros((540, 960, 3), np.uint8), 7)
    assert observation.homography is None
    assert len(observation.keypoints) == 5
    assert observation.keypoints["pnl_1"] == (96, 54)
    assert observation.frame_idx == 7


def test_resize_checks_height_even_when_width_matches():
    detector = PnLCalibFieldDetector()
    detector._loaded = backend(None)
    detector.detect(np.zeros((720, 960, 3), np.uint8), 0)
    detector._loaded["resize"].assert_called_once()


def test_default_weight_paths_follow_root():
    detector = PnLCalibFieldDetector(pnlcalib_root="/example/PnLCalib")
    assert detector.weights_kp == "/example/PnLCalib/SV_kp"
    assert detector.weights_line == "/example/PnLCalib/SV_lines"
