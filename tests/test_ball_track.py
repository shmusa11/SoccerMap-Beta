import cv2
import numpy as np
import pytest

from soccer_analyzer import ball_track as bt
from soccer_analyzer.fast_pipeline import StageTimer, _BallSweep, _Work


def _ball_scene(T=30, seed=0, ball_p=0.6, clutter_p=0.05):
    rng = np.random.default_rng(seed)
    frames = list(range(0, 2 * T, 2))
    truth = {f: (100.0 + 6 * i, 300.0 + 2 * i) for i, f in enumerate(frames)}
    cands = {}
    for f in frames:
        clutter = np.c_[rng.uniform(0, 1280, (3, 2)), np.full(3, clutter_p)]
        ball = [[*truth[f], ball_p]] if ball_p else []
        cands[f] = np.vstack([np.array(ball).reshape(-1, 3), clutter])
    return frames, truth, cands


def test_dp_follows_a_moving_ball_through_clutter():
    frames, truth, cands = _ball_scene()
    track = bt.track_ball2(frames, cands, None, fps=15, scale=np.full(len(frames), 30.0))
    hits = [
        track[f] is not None and np.hypot(track[f][0] - x, track[f][1] - y) < 2
        for f, (x, y) in truth.items()
    ]
    assert np.mean(hits) > 0.9


def test_dp_reports_no_ball_when_only_weak_clutter_exists():
    frames, _, cands = _ball_scene(ball_p=0.0, clutter_p=0.02)
    track = bt.track_ball2(frames, cands, None, fps=15, scale=np.full(len(frames), 30.0))
    assert all(v is None for v in track.values())


def test_motion_blobs_are_size_filtered_against_person_height_and_overlay():
    boxes = np.array([[0, 0, 20, 160], [100, 0, 120, 160], [200, 0, 220, 160]], float)  # 160 px
    # x, y, row, w, h, area, peak/threshold, dark, scale: expected ball ~20 px across
    ball = [500, 400, 400, 20, 20, 300, 4.0, 0, 1.0]
    huge = [700, 400, 400, 200, 120, 18000, 4.0, 0, 1.0]
    on_graphic = [50, 30, 30, 20, 20, 300, 4.0, 0, 1.0]
    overlay = np.zeros((720, 1280), bool)
    overlay[:60, :120] = True
    out = bt.motion_candidates_from_blobs(np.array([ball, huge, on_graphic]), boxes, overlay)
    assert out.shape == (1, 6) and tuple(out[0, :2]) == (500, 400)


def test_vlm_anchors_ask_once_in_a_batch_and_skip_night_frames():
    day = np.full((100, 200, 3), 120, np.uint8)
    night = np.zeros((100, 200, 3), np.uint8)
    frames = list(range(0, 60, 2))
    track = {f: (150.0, 50.0, 0.5) for f in frames}
    asked = []

    def ask(questions):
        asked.append([(f, box) for f, _, box in questions])
        return [{"ball": [10.0, 5.0]} for _ in questions]

    anchors, n = bt.vlm_anchors(
        frames, track, lambda f: night if f == 30 else day, ask, every=5, max_queries=20
    )
    assert len(asked) == 1 and n == 5 and 30 not in anchors  # 6 checkpoints, one at night
    f, box = asked[0][0]
    assert anchors[f]["xy"] == (box[0] + 10.0, box[1] + 5.0)


def test_duplicate_frames_are_detected():
    rng = np.random.default_rng(1)
    a = (rng.random((270, 480)) * 255).astype(np.uint8)
    b = a.copy()
    b[100:140, 100:180] = 255 - b[100:140, 100:180]
    assert bt.is_duplicate(a, a.copy()) and not bt.is_duplicate(b, a)


class _FakeWasb:
    def __init__(self):
        self.trios = []

    def peaks(self, trio):
        self.trios.append([int(t[0, 0, 0]) for t in trio])
        return [(1.0, 2.0, 0.5)]


@pytest.mark.parametrize("workers", [0, 2])
def test_ball_sweep_uses_neighbouring_and_distinct_frames(monkeypatch, workers):
    """Render frames 2 and 4; raw frame 3 repeats frame 2 (screen recording). WASB sees the raw
    trio; the motion difference uses the previous / next DISTINCT frames. Same result whether
    the motion blobs run inline or on worker threads."""
    seen = {}

    def fake_blobs(cur, prev, nxt):
        seen[int(cur[0, 0, 0])] = (int(prev[0, 0, 0]), int(nxt[0, 0, 0]))
        return np.zeros((0, 9), np.float32)

    monkeypatch.setattr(bt, "motion_blobs", fake_blobs)
    rng = np.random.default_rng(2)
    raw = []
    for i in range(7):
        f = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
        f[0, 0] = i * 10  # pixel (0, 0) encodes the frame index
        raw.append(f)
    raw[3] = raw[2].copy()  # duplicate
    wasb = _FakeWasb()
    T = StageTimer()
    with _Work(workers, T) as work:
        sweep = _BallSweep(wasb, T, start_frame=0, vlm_every=1, work=work)
        for i, f in enumerate(raw):
            sweep.push(i, f, is_render=i in (2, 4))
        sweep.finish()
    assert sweep.duplicates == 1
    assert seen[20] == (10, 40)  # frame 2: prev distinct 1, next distinct 4 (3 repeats 2)
    assert seen[40] == (20, 50)  # frame 4: its previous run [2, 3] ends at 3, which shows 2
    assert wasb.trios == [[10, 20, 20], [20, 40, 50]]  # raw trios: frame 3 is a copy of frame 2
    assert set(sweep.wasb_c) == {2, 4} and set(sweep.jpgs) == {2, 4}
    assert cv2.imdecode(sweep.jpgs[2], cv2.IMREAD_COLOR).shape == (48, 64, 3)
