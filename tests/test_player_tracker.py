import numpy as np
import pytest

from soccer_analyzer.player_tracker import TrackCfg, containment_nms, track_people
from soccer_analyzer.vision_oracle import OraclePerson

FPS = 30.0
RENDER = list(range(0, 120, 2))  # 15 Hz render frames over 4 s
DETECT = RENDER[::2]  # the detector runs on every 2nd render frame


def _box(x, y, h=60.0):
    """Person box with feet at (x, y)."""
    return (x - 0.2 * h, y - h, x + 0.2 * h, y)


def _shift(dx):
    return np.array([[1.0, 0, dx], [0, 1, 0], [0, 0, 1]])


def _ids_per_frame(drawn):
    return {f: sorted(t for t, _, _ in drawn[f]) for f in drawn}


def test_camera_pan_keeps_every_player_with_one_id():
    # Three standing players 120 px apart; the camera pans 50 px per render frame, i.e. 100 px
    # between detector runs: a camera-blind tracker swaps or drops them.
    pan = -50.0
    motion = {f: (_shift(pan) if i else None) for i, f in enumerate(RENDER)}
    xs = [700.0, 820.0, 940.0]

    def at(f, x):
        return x + pan * RENDER.index(f)

    dets = {
        f: (np.array([_box(at(f, x), 500.0) for x in xs]), np.full(3, 0.8))
        for f in DETECT
        if all(0 < at(f, x) < 1900 for x in xs)
    }
    drawn, tracks = track_people(RENDER, dets, motion, fps=FPS)
    assert len(tracks) == 3
    for t in tracks:  # each track stays on one player
        feet = [(f, (b[0] + b[2]) / 2) for f, b, _, _ in t.obs]
        x0 = feet[0][1] - pan * RENDER.index(feet[0][0])
        assert all(abs(x - at(f, x0)) < 1.0 for f, x in feet)
    first, last = min(dets), max(dets)
    for f in RENDER:
        if first <= f <= last:
            assert len(drawn[f]) == 3
            for _, b, _ in drawn[f]:  # filled boxes follow the pan
                assert min(abs((b[0] + b[2]) / 2 - at(f, x)) for x in xs) < 1.0


def test_occlusion_gap_is_bridged_with_the_same_id():
    # One player walking right at 4 px per render frame; hidden for 0.8 s in the middle.
    motion = {f: np.eye(3) for f in RENDER}

    def x(f):
        return 300.0 + 4.0 * RENDER.index(f)

    hidden = set(DETECT[10:16])
    dets = {f: (np.array([_box(x(f), 400.0)]), np.array([0.7])) for f in DETECT if f not in hidden}
    drawn, tracks = track_people(RENDER, dets, motion, fps=FPS)
    assert len(tracks) == 1
    gap = [f for f in RENDER if min(hidden) <= f <= max(hidden)]
    assert all(len(drawn[f]) == 1 for f in gap)
    assert {s for f in gap for _, _, s in drawn[f]} == {"interp"}
    for f in gap:
        b = drawn[f][0][1]
        assert (b[0] + b[2]) / 2 == pytest.approx(x(f), abs=1.0)


def test_partial_body_duplicates_do_not_make_a_second_track():
    motion = {f: np.eye(3) for f in RENDER}
    full = _box(500.0, 400.0)
    upper = (full[0] + 2, full[1], full[2] - 2, full[1] + 30)  # head and chest only
    dets = {f: (np.array([full, upper]), np.array([0.8, 0.5])) for f in DETECT}
    drawn, tracks = track_people(RENDER, dets, motion, fps=FPS)
    assert len(tracks) == 1
    assert all(len(drawn[f]) == 1 for f in DETECT)
    keep = containment_nms(np.array([full, upper]), np.array([0.8, 0.5]), 0.7, 0.5)
    assert keep.tolist() == [True, False]


def test_low_confidence_flicker_never_starts_a_track():
    motion = {f: np.eye(3) for f in RENDER}
    dets = {f: (np.zeros((0, 4)), np.zeros(0)) for f in DETECT}
    dets[DETECT[5]] = (np.array([_box(900.0, 300.0)]), np.array([0.2]))  # one-off junk box
    drawn, tracks = track_people(RENDER, dets, motion, fps=FPS)
    assert tracks == [] and not any(drawn.values())


def test_oracle_person_the_detector_misses_becomes_a_labelled_seed():
    motion = {f: np.eye(3) for f in RENDER}
    seen = _box(400.0, 400.0)
    missed = _box(1200.0, 300.0, h=30.0)  # small far player: no detector box
    dets = {f: (np.array([seen]), np.array([0.8])) for f in DETECT}
    kf = {
        DETECT[0]: [OraclePerson(seen, "player", "red"), OraclePerson(missed, "player", "blue")],
        DETECT[8]: [OraclePerson(seen, "player", "red"), OraclePerson(missed, "player", "blue")],
    }
    drawn, tracks = track_people(RENDER, dets, motion, keyframes=kf, fps=FPS)
    labels = sorted(t.label for t in tracks)
    assert labels == ["blue", "red"]
    seed = next(t for t in tracks if t.label == "blue")
    assert {s for _, _, _, s in seed.obs} == {"seed"}
    assert all(any(t == seed.tid for t, _, _ in drawn[f]) for f in RENDER if 0 <= f <= DETECT[8])


def test_oracle_keyframe_between_detector_runs_still_seeds():
    # The oracle may answer on a render frame where the people detector did not run.
    motion = {f: np.eye(3) for f in RENDER}
    seen, missed = _box(400.0, 400.0), _box(1200.0, 300.0, h=30.0)
    dets = {f: (np.array([seen]), np.array([0.8])) for f in DETECT}
    kf_frames = [RENDER[1], RENDER[17]]
    assert not set(kf_frames) & set(dets)
    kf = {f: [OraclePerson(missed, "goalkeeper", None)] for f in kf_frames}
    drawn, tracks = track_people(RENDER, dets, motion, keyframes=kf, fps=FPS)
    seed = [t for t in tracks if all(s == "seed" for _, _, _, s in t.obs)]
    assert len(seed) == 1 and [o[0] for o in seed[0].obs] == kf_frames
    assert all(len(drawn[f]) == 2 for f in RENDER if kf_frames[0] <= f <= kf_frames[1])


def test_config_validation():
    with pytest.raises(ValueError):
        TrackCfg(conf_low=0.5, conf_new=0.2)
    with pytest.raises(ValueError):
        TrackCfg(fill="lk")
    assert track_people([], {}, {}) == ({}, [])


def test_render_frames_after_the_last_detector_run_keep_their_boxes():
    render = list(range(0, 60, 2))  # the window ends on a render frame the detector skips
    dets = {
        f: (np.array([_box(500.0 - 5.0 * i, 400.0)]), np.array([0.9]))
        for i, f in enumerate(render)
        if i % 2 == 0
    }
    motion = {f: (_shift(-5.0) if i else None) for i, f in enumerate(render)}
    drawn, _ = track_people(render, dets, motion, fps=FPS)
    assert render[-1] not in dets and len(drawn[render[-1]]) == 1
    (_, b, _), (_, b0, _) = drawn[render[-1]][0], drawn[render[-2]][0]
    assert b[0] == pytest.approx(b0[0] - 5.0, abs=0.5)  # carried with the camera motion


def test_interpolation_does_not_cross_failed_camera_motion():
    from soccer_analyzer.player_tracker import _fill_gap

    frames = [0, 1, 2, 3, 4]
    pos = {f: i for i, f in enumerate(frames)}
    box = (10, 10, 20, 30)

    def chain(a, b):
        return np.eye(3) if (a < 2) == (b < 2) else None

    assert _fill_gap(frames, pos, chain, 0, box, 4, box) == {}
    assert set(_fill_gap(frames, pos, None, 0, box, 4, box)) == {1, 2, 3}
