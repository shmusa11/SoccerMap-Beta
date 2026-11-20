from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from soccer_analyzer.match_processor import BallPathSolver, MatchData, MatchProcessor


def test_ball_path_marks_interpolation_and_respects_gaps():
    result = BallPathSolver(min_nodes=2).solve({0: [(10, 20, 0.9)], 2: [(12, 20, 0.9)]})
    assert result == {0: (10, 20, True), 1: (11, 20, False), 2: (12, 20, True)}
    assert (
        BallPathSolver(min_nodes=2, gap_frames=1).solve({0: [(10, 20, 0.9)], 10: [(12, 20, 0.9)]})
        == {}
    )


def test_track_gaps_only_fill_processed_frames():
    data = MatchData(
        frames=[0, 3, 6],
        people={0: [(2, (0, 0, 10, 10), (5, 10))], 6: [(2, (6, 0, 16, 10), (11, 10))]},
    )
    MatchProcessor._interpolate_tracks(data, max_gap_frames=6)
    assert data.people[3] == [(2, (3, 0, 13, 10), (8, 10))]
    assert 1 not in data.people


def fake_detector():
    detector = Mock()
    detector._enhance.side_effect = lambda f: f
    detector.has_soccer = False
    detector.conf, detector.people_imgsz, detector.device = 0.1, 1920, "cpu"
    detector._yolo.predict.side_effect = lambda frames, **kwargs: [
        SimpleNamespace(boxes=[]) for _ in frames
    ]
    return detector


def test_sweep_limit_batch_tail_and_reuse():
    cap = Mock()
    cap.get.return_value = 0  # A stream without frame-count metadata.
    cap.grab.return_value = True
    cap.retrieve.return_value = (True, np.zeros((40, 40, 3), np.uint8))
    detector = fake_detector()
    processor = MatchProcessor(detector, yolo_batch=2)
    with patch("soccer_analyzer.video.cv2.VideoCapture", return_value=cap):
        data = processor.sweep("clip", stride=2, max_frames=5)
        again = processor.sweep("clip", stride=2, max_frames=5)
    assert data.frames == again.frames == [0, 2, 4]
    assert data.stats["processed"] == 3
    assert cap.grab.call_count == 10
    assert cap.retrieve.call_count == 6
    assert data.stats["images_retrieved"] == 3
    assert data.stats["source_frames_advanced"] == 5
    assert cap.release.call_count == 2
    assert detector._yolo.predict.call_count == 4  # Full batch + tail, twice.


def test_sweep_releases_video_on_model_failure():
    cap = Mock()
    cap.get.return_value = 5
    cap.grab.return_value = True
    cap.retrieve.return_value = (True, np.zeros((40, 40, 3), np.uint8))
    detector = fake_detector()
    detector._yolo.predict.side_effect = RuntimeError("inference failed")
    processor = MatchProcessor(detector)
    with patch("soccer_analyzer.video.cv2.VideoCapture", return_value=cap):
        with pytest.raises(RuntimeError, match="inference failed"):
            processor.sweep("clip")
    cap.release.assert_called_once()


def test_sweep_discards_ball_triples_around_a_cut_between_samples():
    cap = Mock()
    cap.get.return_value = 0
    cap.grab.return_value = True
    cap.retrieve.side_effect = [
        (True, np.full((40, 40, 3), 0 if i <= 6 else 255, np.uint8)) for i in range(13)
    ]
    ball = Mock()
    ball.detect.return_value = [(10, 10, 0.9)]
    processor = MatchProcessor(fake_detector(), wasb=ball, yolo_batch=2)
    with patch("soccer_analyzer.video.cv2.VideoCapture", return_value=cap):
        data = processor.sweep("clip", stride=3, max_frames=13)
    assert data.cuts == [9]  # Actual cut is at raw frame 7, between sampled frames 6 and 9.
    assert ball.detect.call_count == 3  # Candidate centers 3, 6, 9.
    assert data.stats["ball_frames_with_cands"] == 1  # Retain only the triple centered at 3.


def test_later_sweep_keeps_raw_indices_and_consecutive_ball_inputs(tmp_path):
    import cv2

    source = tmp_path / "window.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (40, 40))
    for index in range(20):
        writer.write(np.full((40, 40, 3), index * 10, np.uint8))
    writer.release()
    ball = Mock()
    ball.detect.return_value = [(10, 10, 0.9)]
    data = MatchProcessor(fake_detector(), wasb=ball).sweep(
        str(source), start_frame=5, max_frames=9, stride=3
    )
    assert data.frames == [5, 8, 11]
    assert ball.detect.call_count == 2
    for call, expected in zip(ball.detect.call_args_list, ([70, 80, 90], [100, 110, 120])):
        np.testing.assert_allclose([np.mean(f) for f in call.args[0]], expected, atol=2)


@pytest.mark.parametrize("stride", [1, 3, 5])
@pytest.mark.parametrize("ball_every", [1, 2, 4])
def test_sparse_retrieval_preserves_every_scheduled_raw_ball_triple(tmp_path, stride, ball_every):
    import cv2

    source = tmp_path / "sparse.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (40, 40))
    for index in range(40):
        writer.write(np.full((40, 40, 3), index * 5, np.uint8))
    writer.release()
    ball = Mock()
    ball.detect.return_value = []
    detector = fake_detector()
    data = MatchProcessor(detector, wasb=ball, ball_every=ball_every).sweep(
        str(source), start_frame=5, max_frames=29, stride=stride
    )
    samples = list(range(5, 34, stride))
    centers = [f for f in range(5, 34, stride * ball_every) if 5 < f < 33]
    assert data.frames == samples
    assert data.stats["ball_scans"] == len(centers)
    needed = set(samples) | {center + offset for center in centers for offset in (-1, 0, 1)}
    assert data.stats["images_retrieved"] == len(needed)
    assert data.stats["source_frames_advanced"] == 29
    for call, center in zip(ball.detect.call_args_list, centers):
        np.testing.assert_allclose(
            [np.mean(frame) for frame in call.args[0]],
            [(center + offset) * 5 for offset in (-1, 0, 1)],
            atol=2,
        )


@pytest.mark.parametrize("kwargs", [{"yolo_batch": 0}, {"officials_every": 0}, {"ball_every": 0}])
def test_invalid_cadences(kwargs):
    with pytest.raises(ValueError):
        MatchProcessor(fake_detector(), **kwargs)


def test_ball_solver_recovers_after_long_occlusion_without_bridging():
    candidates = {f: [(100 + f, 100, 0.9)] for f in [0, 2, 4, 6, 100, 102, 104, 106]}
    solved = BallPathSolver(gap_frames=10).solve(candidates)
    assert solved[0][2] and solved[106][2]
    assert 50 not in solved
    assert solved[103][2] is False


def test_ball_solver_never_bridges_scene_cut():
    candidates = {f: [(10 + f, 20, 0.9)] for f in [0, 2, 4, 6, 8, 10, 12, 14]}
    solved = BallPathSolver().solve(candidates, cuts=[8])
    assert 7 not in solved
    assert solved[6][2] and solved[8][2]


def test_ball_solver_drops_nonfinite_and_out_of_range_candidates():
    candidates = {0: [(float("nan"), 3, 0.9), (3, 3, 2)], 1: [(2, 3, 0.9)], 2: [(3, 3, 0.9)]}
    assert set(BallPathSolver(min_nodes=2).solve(candidates)) == {1, 2}


def test_interpolation_is_labelled_and_stops_at_cuts_and_large_jumps():
    data = MatchData(
        frames=[0, 3, 6],
        people={0: [(2, (0, 0, 10, 10), (5, 10))], 6: [(2, (6, 0, 16, 10), (11, 10))]},
    )
    MatchProcessor._interpolate_tracks(data, 6)
    assert data.interpolated == {(3, 2)}
    for cuts, box in [([3], (6, 0, 16, 10)), ([], (100, 0, 110, 10))]:
        data = MatchData(
            frames=[0, 3, 6],
            people={0: [(2, (0, 0, 10, 10), (5, 10))], 6: [(2, box, (box[0] + 5, 10))]},
            cuts=cuts,
        )
        MatchProcessor._interpolate_tracks(data, 6)
        assert not data.people.get(3)
