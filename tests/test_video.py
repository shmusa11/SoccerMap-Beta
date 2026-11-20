from unittest.mock import Mock, patch

import pytest

from soccer_analyzer.video import (
    frame_limit,
    frame_window,
    seek_capture,
    open_capture,
    open_writer,
    sampled_fps,
)


@pytest.mark.parametrize(
    "reported,limit,expected",
    [(0, 9, 9), (100, 9, 9), (5, 9, 5), (0, None, 0), (float("nan"), 7, 7)],
)
def test_frame_limit(reported, limit, expected):
    assert frame_limit(reported, limit) == expected


@pytest.mark.parametrize(
    "fps,stride,expected",
    [(60, 3, 20), (29.97, 3, 9.99), (24, 10, 2.4), (0, 3, 10), (float("nan"), 3, 10)],
)
def test_preserve_playback_speed(fps, stride, expected):
    assert sampled_fps(fps, stride) == pytest.approx(expected)


def test_invalid_limits():
    with pytest.raises(ValueError):
        frame_limit(20, 0)
    with pytest.raises(ValueError):
        sampled_fps(30, 0)


def test_capture_released_after_error():
    cap = Mock()
    with patch("soccer_analyzer.video.cv2.VideoCapture", return_value=cap):
        with pytest.raises(ValueError), open_capture("clip"):
            raise ValueError("detector failed")
    cap.release.assert_called_once()


def test_unreadable_capture_released():
    cap = Mock()
    cap.isOpened.return_value = False
    with patch("soccer_analyzer.video.cv2.VideoCapture", return_value=cap):
        with pytest.raises(RuntimeError, match="Could not open"):
            with open_capture("clip"):
                pytest.fail("unreadable input must not be yielded")
    cap.release.assert_called_once()


def test_writer_fallback_releases_failed_encoder():
    first, second = Mock(), Mock()
    first.isOpened.return_value = False
    with patch("soccer_analyzer.video.cv2.VideoWriter", side_effect=[first, second]):
        assert open_writer("out.mp4", 20, (640, 480)) is second
    first.release.assert_called_once()
    second.release.assert_not_called()


def test_both_encoders_fail():
    writer = Mock()
    writer.isOpened.return_value = False
    with patch("soccer_analyzer.video.cv2.VideoWriter", return_value=writer):
        with pytest.raises(RuntimeError, match="encoder"):
            open_writer("out.mp4", 20, (640, 480))
    assert writer.release.call_count == 2


@pytest.mark.parametrize(
    "count,limit,start,expected",
    [
        (100, 30, 50, (50, 80)),
        (100, 30, 90, (90, 100)),
        (0, 30, 50, (50, 80)),
        (0, None, 50, (50, None)),
        (100, None, 50, (50, 100)),
    ],
)
def test_raw_frame_windows(count, limit, start, expected):
    assert frame_window(count, limit, start) == expected


@pytest.mark.parametrize("start", [-1, True, 1.5, 10])
def test_invalid_or_empty_frame_window(start):
    with pytest.raises(ValueError):
        frame_window(10, 5, start)


def test_seek_refuses_to_mislabel_a_frame_and_can_rewind():
    cap = Mock()
    cap.set.return_value = True
    cap.get.return_value = 9
    with pytest.raises(ValueError, match="accurately"):
        seek_capture(cap, 10)
    cap.get.return_value = 10
    seek_capture(cap, 10)
    cap.get.return_value = 0
    seek_capture(cap, 0, fresh=False)
    assert cap.set.call_args.args[1] == 0


def write_indexed_video(path, n=60, size=(320, 240), fps=30.0):
    """Synthetic clip whose frame i shows i as 8 black/white blocks (survives compression)."""
    import cv2
    import numpy as np

    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur((rng.random((h, w + 4 * n)) * 255).astype(np.uint8), (5, 5), 0)
    for i in range(n):
        g = texture[:, 2 * i : 2 * i + w].copy()  # slow pan, so camera motion is measurable
        for bit in range(8):
            g[:24, bit * 32 : bit * 32 + 24] = 255 if (i >> bit) & 1 else 0
        writer.write(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    writer.release()
    return path


def frame_index(frame):
    """Decode the index written by `write_indexed_video`."""
    return sum(
        1 << bit for bit in range(8) if frame[4:20, bit * 32 + 4 : bit * 32 + 20].mean() > 127
    )


def test_skip_frames_counts_from_the_start(tmp_path):
    from soccer_analyzer.video import skip_frames

    path = write_indexed_video(tmp_path / "idx.avi")
    with open_capture(path) as cap:
        assert skip_frames(cap, 37) == 37
        ok, frame = cap.read()
        assert ok and frame_index(frame) == 37
        assert skip_frames(cap, 0) == 0
        assert skip_frames(cap, 1000) == 60 - 38  # stops at the end of the clip
    with pytest.raises(ValueError):
        skip_frames(Mock(), -1)


def test_prefetch_frames_matches_a_sequential_loop(tmp_path):
    from soccer_analyzer.video import prefetch_frames, skip_frames

    path = write_indexed_video(tmp_path / "idx.avi")
    with open_capture(path) as cap:
        skip_frames(cap, 5)
        with prefetch_frames(cap, 5, 40, lambda i: i % 3 == 0, depth=2, extra=frame_index) as it:
            got = list(it)
    assert [i for i, _, _ in got] == list(range(5, 40))
    for i, frame, aux in got:
        if i % 3 == 0:
            assert frame_index(frame) == i and aux == i
        else:
            assert frame is None and aux is None
    with open_capture(path) as cap:  # stop=None runs to the end of the clip
        with prefetch_frames(cap, 0, None, lambda i: True) as it:
            assert [i for i, _, _ in it] == list(range(60))


def test_prefetch_frames_stops_early_and_reraises(tmp_path):
    from soccer_analyzer.video import prefetch_frames

    path = write_indexed_video(tmp_path / "idx.avi")
    with open_capture(path) as cap:
        with prefetch_frames(cap, 0, None, lambda i: True, depth=1) as it:
            assert next(it)[0] == 0  # leaving early stops and joins the decode thread

    def boom(frame):
        raise RuntimeError("bad frame")

    with open_capture(path) as cap:
        with pytest.raises(RuntimeError, match="bad frame"):
            with prefetch_frames(cap, 0, None, lambda i: True, extra=boom) as it:
                list(it)
