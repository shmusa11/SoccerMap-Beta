from unittest.mock import patch

import cv2
import numpy as np

from soccer_analyzer.calibration.mapper import FrameCalibration
from soccer_analyzer.calibration.visualize import draw_calibration_overlay, draw_topdown
from soccer_analyzer.calibration.pitch_model import STANDARD_PITCH
from soccer_analyzer.review_render import draw_track_box, review_caption


def test_overlay_legend_does_not_claim_landmarks_without_observations():
    calib = FrameCalibration(0, np.eye(3), 1, "detect", 0, 0)
    with patch("soccer_analyzer.calibration.visualize.cv2.putText", wraps=cv2.putText) as draw:
        draw_calibration_overlay(np.zeros((440, 640, 3), np.uint8), calib, fade_by_confidence=False)
    labels = [call.args[1] for call in draw.call_args_list]
    assert any("not all markings were observed" in label for label in labels)
    for call in draw.call_args_list:
        _, text, origin, font, scale, _, thickness, _ = call.args
        width = cv2.getTextSize(text, font, scale, thickness)[0][0]
        assert origin[0] + width <= 640


def test_review_caption_preserves_footage_and_never_invents_confidence():
    source = np.full((100, 160, 3), 80, np.uint8)
    for valid in (True, False):
        with patch("soccer_analyzer.review_render.cv2.putText", wraps=cv2.putText) as draw:
            result = review_caption(source, 240, valid)
        assert np.array_equal(result[:100], source)
        labels = [call.args[1] for call in draw.call_args_list]
        assert sum("Source frame 240" in label for label in labels) == 1
        assert not any("conf" in label.lower() for label in labels)
        assert ("Map unavailable" in labels[0]) is (not valid)
        for call in draw.call_args_list:
            _, text, origin, font, scale, _, thickness, _ = call.args
            assert origin[0] + cv2.getTextSize(text, font, scale, thickness)[0][0] <= 160


def test_interpolated_boxes_and_map_positions_are_visually_distinct():
    observed = np.zeros((100, 100, 3), np.uint8)
    interpolated = observed.copy()
    draw_track_box(observed, (10, 10, 90, 90), (255, 255, 255), True)
    draw_track_box(interpolated, (10, 10, 90, 90), (255, 255, 255), False)
    assert observed[10, 50].any() and not interpolated[10, 50].any()
    assert interpolated[10, 10].any()
    args = dict(pitch=STANDARD_PITCH, size=(900, 620), ball=(80, 50))
    baseline = draw_topdown(STANDARD_PITCH)
    observed = draw_topdown(**args, players=[(25, 20, 0)])
    interpolated = draw_topdown(**args, players=[(25, 20, 0, False)], ball_measured=False)
    for x, y in ((25, 20), (80, 50)):
        # draw_topdown keeps a 42 px margin so off-pitch throw-in takers fit beside the lines.
        px, py = int(42 + x * (900 - 84) / 105), int(42 + y * (620 - 84) / 68)
        assert np.array_equal(interpolated[py, px], baseline[py, px])
        assert not np.array_equal(observed[py, px], baseline[py, px])


def test_topdown_draws_a_player_just_off_the_touchline():
    """A throw-in taker 1.5 m outside the touchline is drawn beside the line, not dropped."""
    from soccer_analyzer.calibration.pitch_model import PitchModel
    from soccer_analyzer.calibration.visualize import draw_topdown

    pitch = PitchModel(105, 68)
    empty = draw_topdown(pitch)
    off = draw_topdown(pitch, players=[(50.0, -1.5, 0)])
    far = draw_topdown(pitch, players=[(50.0, -20.0, 0)])  # a spectator well beyond the band
    changed = np.argwhere((off != empty).any(axis=2))
    assert len(changed) > 0
    assert changed[:, 0].max() < 42  # all in the margin above the top touchline
    assert not (far != empty).any()
