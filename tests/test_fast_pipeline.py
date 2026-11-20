import cv2
import numpy as np
import pytest

from soccer_analyzer.fast_pipeline import (
    FastConfig,
    frame_motion,
    match_boxes,
    mean_displacement,
    two_means,
    vote_labels,
)
from soccer_analyzer.vision_oracle import OraclePerson


def test_config_rejects_bad_rates():
    with pytest.raises(ValueError):
        FastConfig(render_hz=0)
    with pytest.raises(ValueError):
        FastConfig(detect_every=0)


def test_frame_motion_recovers_a_pan():
    rng = np.random.default_rng(0)
    big = cv2.GaussianBlur((rng.random((700, 1300)) * 255).astype(np.uint8), (5, 5), 0)
    a, b = big[50:590, 100:1060], big[50:590, 112:1072]  # camera panned 12 px to the right
    G = frame_motion(a, b, scale=1.0)
    np.testing.assert_allclose(G @ [500, 300, 1] / (G @ [500, 300, 1])[2], [488, 300, 1], atol=0.5)
    assert mean_displacement(G, (960, 540)) == pytest.approx(12 / np.hypot(960, 540), rel=0.05)
    assert frame_motion(a, np.zeros_like(a), 1.0) is None  # cut / nothing to track


def test_match_boxes_is_one_to_one_by_iou():
    tracked = [(1, (0, 0, 10, 20), None), (2, (100, 0, 110, 20), None), (-1, (0, 0, 10, 20), None)]
    boxes = [(101, 1, 111, 21), (1, 1, 11, 21), (500, 500, 510, 520)]
    assert match_boxes(tracked, boxes) == {2: 0, 1: 1}


def _p(box, role="player", team=None):
    return OraclePerson(box, role, team)


def test_vote_labels_roles_teams_and_offfield():
    # Two keyframes: tracks 1/2 are players, 3 the referee, 4 a spectator far from play (missed
    # by the oracle at both keyframes) and 5 a player the oracle skipped next to track 1.
    people_at = {
        f: [
            (1, (100, 100, 120, 150), None),
            (2, (300, 100, 320, 150), None),
            (3, (500, 100, 520, 150), None),
            (4, (900, 10, 910, 30), None),
            (5, (125, 100, 145, 150), None),
        ]
        for f in (0, 60)
    }
    oracle = [
        _p((100, 100, 120, 150), team="white"),
        _p((300, 100, 320, 150), team="red"),
        _p((500, 100, 520, 150), role="referee", team="yellow"),
    ]
    keyframes = {0: oracle, 60: list(oracle)}
    kind, team, offfield, names = vote_labels(keyframes, people_at)
    assert names == {0: "red", 1: "white"}  # red kit drawn red, never blue
    assert team[1] == 1 and team[2] == 0 and kind[3] == "referee" and 3 not in team
    assert offfield == {4}  # 5 sat beside an oracle player: ambiguous, not off-field


def test_two_means_separates_clusters():
    X = np.array([[0, 0], [0.1, 0], [0, 0.2], [5, 5], [5.1, 4.9]], float)
    c = two_means(X)
    assert sorted(np.round(c[:, 0]).tolist()) == [0.0, 5.0]
    assert two_means(X[:1]) is None


class _NoPeople:
    """SoccerDetector stand-in: records the frames YOLO would see and finds nobody."""

    people_imgsz, device = 640, "cpu"

    def __init__(self):
        self.seen = []
        self._yolo = self

    def predict(self, frames, **kw):
        from test_video import frame_index

        class _Empty:
            boxes = []

        self.seen += [frame_index(f) for f in frames]
        return [_Empty() for _ in frames]


@pytest.mark.parametrize(
    "cache_mb,workers,cache", [(400, 3, True), (0, 0, False), (0.05, 3, False)]
)
def test_run_and_render_use_original_frame_indices_when_seeking_is_broken(
    tmp_path, monkeypatch, cache_mb, workers, cache
):
    # Regression: cap.set(CAP_PROP_POS_FRAMES, 352) returned true frame 372 on a 4K clip. Here
    # every seek lands 5 frames late; the pipeline must count frames instead of seeking. The
    # renderer reads the pass-1 frame cache, or decodes again without it (cache off, or the
    # window is longer than the cache budget: 0.05 MB holds 5 of these 20 frames).
    from contextlib import contextmanager

    import soccer_analyzer.fast_pipeline as fp
    from test_video import frame_index, write_indexed_video

    path = write_indexed_video(tmp_path / "idx.avi", n=60)

    class BadSeek:
        def __init__(self, cap):
            self.cap = cap

        def __getattr__(self, name):
            return getattr(self.cap, name)

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                value += 5
            return self.cap.set(prop, value)

    real_open = fp.open_capture

    @contextmanager
    def bad_open(p):
        with real_open(p) as cap:
            yield BadSeek(cap)

    monkeypatch.setattr(fp, "open_capture", bad_open)
    people = _NoPeople()
    cfg = FastConfig(multisport_fallback=False, render_cache_mb=cache_mb, cpu_workers=workers)
    pipe = fp.FastPipeline(None, people, config=cfg)
    anchored = []
    monkeypatch.setattr(
        pipe,
        "_anchor",
        lambda frame, idx: (anchored.append((idx, frame_index(frame))), (None, None))[1],
    )
    result = pipe.run(str(path), max_frames=40, start_frame=10)
    assert result.render_frames[0] == 10 and result.render_frames[-1] == 48
    assert anchored and all(claimed == shown for claimed, shown in anchored)
    assert people.seen == result.detect_frames == list(range(12, 50, 4))
    assert all(result.md.people[f] == [] for f in result.render_frames)
    assert sorted(result.render_cache) == (result.render_frames if cache else [])

    drawn = []
    monkeypatch.setattr(
        fp,
        "open_writer",
        lambda *a: type(
            "W", (), {"write": lambda s, c: drawn.append(c), "release": lambda s: None}
        )(),
    )
    monkeypatch.setattr(fp, "compose", lambda overlay, top, out_width: overlay)
    info = pipe.render(str(path), result, str(tmp_path / "out.mp4"))
    assert info["rendered_frames"] == len(result.render_frames)
    assert [frame_index(c) for c in drawn] == result.render_frames


def test_labels_hide_tracks_that_never_step_onto_the_pitch():
    from soccer_analyzer.fast_pipeline import FastPipeline
    from soccer_analyzer.match_processor import MatchData

    pipe = FastPipeline(None, None)
    H = np.diag([0.1, 0.1, 1.0])  # 10 px per metre, image origin = pitch corner
    frames = list(range(0, 80, 2))
    md = MatchData(frames=frames)
    for i, f in enumerate(frames):
        md.people[f] = [
            (1, (500, 280, 520, 340), (510, 340)),  # on the pitch: (51, 34) m
            (2, (500, -160, 520, -100), (510, -100)),  # 10 m off all clip; oracle says player
            (3, (300, -120, 320, -60), (310, -60)),  # spectator, never labelled
            # a player whose calibrated frames are on the pitch; uncalibrated ones never count
            (4, (700, 300, 720, 360) if i < 30 else (700, -900, 720, -840), None),
            # throw-in: 1.5 m outside the far touchline, on the pitch 2 s later
            (5, (600, -25, 620, -15) if i < 30 else (600, 40, 620, 50), None),
        ]
    for f, p in md.people.items():
        md.people[f] = [(t, b, ((b[0] + b[2]) / 2, b[3])) for t, b, _ in p]
    calib = {f: H for f in frames[:30]}
    calib.update({f: (H if f >= 60 else None) for f in frames[30:]})
    kf = {0: [_p((500, 280, 520, 340), team="red"), _p((500, -160, 520, -100), team="red")]}
    kind, team, off, _, vis, ai = pipe._labels(md, kf, {}, calib, fps=30.0)
    # round 3: an oracle "player" label no longer keeps a track that is never on the pitch on
    # calibrated frames (only referees are protected); the taker stays on every frame
    assert off == {2, 3} and ai == {}
    assert kind.get(1) == kind.get(2) == "player"
    assert all(5 in vis.visible[f] and 1 in vis.visible[f] for f in frames)
    assert vis.pitch_xy[0][5][1] < 0  # drawn beside the touchline on the 2D map
    assert all(4 in vis.visible[f] for f in frames)


def test_labels_ask_the_decisions_api_when_an_unlabelled_track_lacks_calibration():
    from soccer_analyzer.fast_pipeline import FastPipeline
    from soccer_analyzer.match_processor import MatchData

    asked = []

    class Oracle:
        def decide(self, images, questions, text=None, max_width=1920):
            asked.append(len(questions))
            return {"answers": {q["name"]: {"probability": 0.05} for q in questions}}

    pipe = FastPipeline(None, None, oracle=Oracle())
    frames = list(range(0, 60, 2))
    md = MatchData(frames=frames)
    for f in frames:
        md.people[f] = [
            (1, (500, 280, 520, 340), (510, 340)),  # oracle-labelled player
            (2, (100, 100, 120, 160), (110, 160)),  # unlabelled -> p 0.05 -> hidden
        ]
    kf = {0: [_p((500, 280, 520, 340), team="red")]}
    jpg = cv2.imencode(".jpg", np.zeros((360, 640, 3), np.uint8))[1]
    calib = {f: None for f in frames}
    _, _, off, _, vis, ai = pipe._labels(
        md, kf, {}, calib, fps=30.0, oracle_frames={0: jpg, 30: jpg}
    )
    # the oracle label wins over the AI; every track is asked (answers depend on the mosaic)
    assert off == {2} and vis.status[2] == "ai-hide" and vis.status[1] == "uncalibrated"
    assert ai["asked"] == 2 and asked == [8]  # 2 tracks x 2 tiles x 2 predicates
    # no track whose status the answer could change: no request
    md2 = MatchData(frames=frames, people={f: md.people[f][:1] for f in frames})
    asked.clear()
    _, _, _, _, _, ai = pipe._labels(md2, kf, {}, calib, fps=30.0, oracle_frames={0: jpg})
    assert ai == {} and asked == []
    # render-size frames: boxes are scaled into them
    _, _, off, _, _, ai = pipe._labels(
        md, kf, {}, calib, fps=30.0, oracle_frames={0: jpg}, frame_scale=0.5
    )
    assert off == {2} and ai["tiles"] == 2
    pipe.cfg.onfield_ai = False
    _, _, off, _, _, ai = pipe._labels(md, kf, {}, calib, fps=30.0, oracle_frames={0: jpg})
    assert off == set() and ai == {}


def test_reported_ball_points_never_fall_below_the_gate(monkeypatch):
    from types import SimpleNamespace

    from soccer_analyzer import ball_track as bt
    from soccer_analyzer.fast_pipeline import FastConfig, FastPipeline

    frames = [0, 2, 4, 6]
    # a reported segment (conf 0.3) whose bridged frame carries 0.5 x a weak segment's conf
    track = {0: (10.0, 10.0, 0.3), 2: (12.0, 10.0, 0.04), 4: None, 6: (16.0, 10.0, 0.05)}
    monkeypatch.setattr(bt, "track_ball2", lambda *a, **k: dict(track))
    sweep = SimpleNamespace(wasb_c={}, blobs={}, grays=[], jpgs={}, night_frames=0, duplicates=0)
    pipe = FastPipeline(None, None, config=FastConfig(ball_vlm=False, ball_gate_offline=0.05))
    raw = {f: (np.zeros((0, 4)), np.zeros(0)) for f in frames}
    out, info = pipe._track_ball(
        sweep, frames, {f: np.eye(3) for f in frames}, raw, 15.0, (640, 360)
    )
    assert sorted(out) == [0, 6] and info["gate"] == 0.05


class _MemoOracle:
    """Vision-oracle stand-in with a per-video memo. Markings answers name one landmark (a real
    answer, but one point cannot fix a camera: the field is never confirmed). mode "budget": the
    budget allows `n_ok` markings calls, then submit returns None (as `_reserve` does); mode
    "error": every call fails (result None)."""

    def __init__(self, failed_windows=0, mode=None, n_ok=0):
        from concurrent.futures import Future

        self._Future = Future
        self.memo = {"markings_failed_windows": failed_windows} if failed_windows else {}
        self.asked = []
        self.spent_usd = 0.0
        self.mode, self.n_ok = mode, n_ok

    def _done(self, r):
        f = self._Future()
        f.set_result(r)
        return f

    def memo_get(self, name):
        return self.memo.get(name)

    def memo_set(self, name, value):
        self.memo[name] = value

    def submit_markings(self, idx, frame, alt=False):
        from soccer_analyzer.vision_oracle import OracleResult

        if self.mode == "budget" and len(self.asked) >= self.n_ok:
            return None
        self.asked.append(idx)
        if self.mode == "error":
            return self._done(None)
        marks = [("centre", 40.0, 30.0, 0.9)]
        return self._done(OracleResult(idx, "markings", markings=marks, line_colour="white"))

    def submit_people(self, idx, frame):
        return None

    def submit_landmarks(self, idx, frame):
        return None


def _memo_run(tmp_path, monkeypatch, oracle, max_keys=4, joint=None, multisport=None):
    import soccer_analyzer.fast_pipeline as fp
    from test_video import write_indexed_video

    path = write_indexed_video(tmp_path / "idx.avi", n=60)
    cfg = FastConfig(
        multisport_fallback=multisport is not None,
        render_cache_mb=0,
        cpu_workers=0,
        ball_vlm=False,
        onfield_ai=False,
        markings_max_keys=max_keys,
        markings_candidate_s=0.2,
    )
    pipe = fp.FastPipeline(None, _NoPeople(), oracle=oracle, config=cfg)
    monkeypatch.setattr(pipe, "_anchor", lambda frame, idx: (None, None))
    if joint is not None:
        monkeypatch.setattr(pipe, "_joint_calibration", joint)
    if multisport is not None:
        monkeypatch.setattr(pipe, "_multisport", multisport)
    return pipe.run(str(path), max_frames=60)


@pytest.mark.parametrize("failed_before", [0, 2])
def test_unconfirmable_video_stops_paying_for_markings(tmp_path, monkeypatch, failed_before):
    # A window whose field stays unconfirmed after the whole keyframe budget (>= markings_max_keys
    # answers naming markings, no call short) is noted per video; after `markings_fail_windows`
    # such windows, later windows ask no markings at all (the per-match cost of an
    # uncalibratable camera setup is bounded, not linear in match length).
    oracle = _MemoOracle(failed_before)
    result = _memo_run(tmp_path, monkeypatch, oracle)
    mk = result.stats["field_markings"]
    assert all(h is None for h in result.calib.values())
    if failed_before:
        assert oracle.asked == [] and "skipped" in mk
        assert oracle.memo["markings_failed_windows"] == 2
    else:
        assert len(oracle.asked) == 4 and mk.get("exhausted") and mk["calls_short"] == 0
        assert oracle.memo["markings_failed_windows"] == 1


def test_memo_skipped_markings_never_fall_back_to_multisport(tmp_path, monkeypatch):
    # A window whose markings the per-video memo skipped has no answers to check against, but
    # earlier windows' markings were answered and never confirmed: that is marking evidence, so
    # the colour-agnostic fallback (6-11 m wrong on the multi-sport clip) must not run.
    calls = []

    def multisport(video_path, total, calib, source):
        calls.append(video_path)
        return {f: np.eye(3) for f in calib}, {f: "multisport" for f in calib}, {}

    oracle = _MemoOracle(2)
    result = _memo_run(tmp_path, monkeypatch, oracle, multisport=multisport)
    assert oracle.asked == [] and calls == []
    assert all(h is None for h in result.calib.values())


@pytest.mark.parametrize("mode,n_ok", [("budget", 0), ("budget", 2), ("error", 0)])
def test_budget_or_outage_window_is_not_counted_unconfirmable(tmp_path, monkeypatch, mode, n_ok):
    # A window starved by the budget, or whose AI calls failed, says nothing about the field:
    # the per-video count stays as it was (here 1), so a later re-run with a bigger budget (or
    # after the outage) still asks.
    oracle = _MemoOracle(1, mode=mode, n_ok=n_ok)
    result = _memo_run(tmp_path, monkeypatch, oracle)
    mk = result.stats["field_markings"]
    assert all(h is None for h in result.calib.values())
    assert oracle.memo["markings_failed_windows"] == 1
    if mode == "budget" and n_ok == 0:
        assert oracle.asked == []
    else:
        assert mk["calls_short"] > 0 and mk.get("failed_windows") == 1


def test_confirmed_window_resets_the_unconfirmable_count(tmp_path, monkeypatch):
    from soccer_analyzer.calibration.pitch_model import PitchModel

    def joint(render_frames, motion, images, masks, jobs, fps, more=None):
        calib = {f: np.eye(3) for f in render_frames}
        source = {f: "vlm markings (joint)" for f in render_frames}
        return (calib, source, PitchModel()), dict(method="joint", kept=sorted(jobs))

    oracle = _MemoOracle(1)
    result = _memo_run(tmp_path, monkeypatch, oracle, joint=joint)
    assert result.stats["field_markings"]["failed_windows"] == 0
    assert oracle.memo["markings_failed_windows"] == 0


@pytest.mark.parametrize(
    "fails,confirmed,exhausted,n_marked,short,expect",
    [
        (0, False, True, 8, 0, 1),  # full budget, real answers, still refused: counts
        (1, False, True, 8, 0, 2),
        (1, False, True, 7, 0, 1),  # fewer answers than the full budget: not evidence
        (1, False, True, 8, 1, 1),  # a call skipped (budget) or failed: not evidence
        (1, False, False, 8, 0, 1),  # not exhausted
        (3, True, False, 2, 0, 0),  # confirmed: the camera setup calibrates -> reset
    ],
)
def test_markings_memo_rule(fails, confirmed, exhausted, n_marked, short, expect):
    from soccer_analyzer.fast_pipeline import _markings_memo

    assert _markings_memo(fails, confirmed, exhausted, n_marked, short, max_keys=8) == expect


def test_pipeline_preserves_interpolation_provenance(tmp_path, monkeypatch):
    import soccer_analyzer.fast_pipeline as fp
    from test_video import write_indexed_video

    path = write_indexed_video(tmp_path / "provenance.avi", n=40)
    pipe = fp.FastPipeline(
        None, _NoPeople(), config=FastConfig(multisport_fallback=False, cpu_workers=0)
    )
    monkeypatch.setattr(pipe, "_anchor", lambda *a: (None, None))
    monkeypatch.setattr(
        fp,
        "track_people",
        lambda frames, *a: (
            {
                f: [(7, (1, 1, 5, 8), "det" if i % 2 == 0 else "interp")]
                for i, f in enumerate(frames)
            },
            [],
        ),
    )
    result = pipe.run(str(path), max_frames=30)
    assert result.md.interpolated == {(f, 7) for f in result.render_frames[1::2]}
    assert all((f, 7) not in result.md.interpolated for f in result.render_frames[::2])
