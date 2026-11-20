from concurrent.futures import Future

import cv2
import numpy as np
import pytest

from soccer_analyzer.calibration import field_markings as fm
from soccer_analyzer.calibration import refcore as rc
from soccer_analyzer.fast_pipeline import FastPipeline
from soccer_analyzer.vision_oracle import OracleResult

W_IMG, H_IMG = 1280, 720


def _look_at(C, T, f):
    z = np.subtract(T, C).astype(float)
    z /= np.linalg.norm(z)
    y = np.array([0, 0, 1.0]) - z[2] * z  # world Z points into the ground = image down
    y /= np.linalg.norm(y)
    R = np.stack([np.cross(y, z), y, z])
    return dict(
        rvec=cv2.Rodrigues(R)[0].ravel(), C=np.array(C, float), f=f, cx=W_IMG / 2, cy=H_IMG / 2
    )


@pytest.fixture(scope="module")
def scene():
    """A broadcast-like camera beside the near touchline, its painted lines and its markings."""
    cam = _look_at((52.5, 98.0, -14.0), (52.5, 30.0, 0.0), 900.0)
    img = np.zeros((H_IMG, W_IMG, 3), np.uint8)
    for polys in rc.marking_polylines(cam, 105, 68, W_IMG, H_IMG, include_goals=False).values():
        for q in polys:
            cv2.polylines(img, [np.round(q).astype(np.int32)], False, (255, 255, 255), 3)
    P = fm.model_points(105, 68)
    names = sorted(n for n in P if not n.endswith("_top"))
    p, z = rc.project(cam, np.array([P[n] for n in names]))
    marks = [
        (n, float(x), float(y), 0.9)
        for n, (x, y), d in zip(names, p, z)
        if d > 0 and 0 <= x < W_IMG and 0 <= y < H_IMG
    ]
    return cam, img, marks


def _pitch_xy(H, px):
    q = np.c_[np.asarray(px, float), np.ones(len(px))] @ np.asarray(H, float).T
    return q[:, :2] / q[:, 2:3]


def test_parse_markings_scales_dedups_and_drops_unknown_names():
    resp = {
        "_scale": (2.0, 2.0),
        "markings": [
            {"name": "centre", "x": 10, "y": 20, "confidence": 0.6},
            {"name": "CENTRE", "x": 11, "y": 21, "confidence": 0.9},  # duplicate: keep higher conf
            {"name": "yard_line_30", "x": 1, "y": 1, "confidence": 1.0},  # not a soccer marking
            {"name": "l_spot", "x": "bad", "y": 1},
        ],
    }
    assert fm.parse_markings(resp) == [("centre", 22.0, 42.0, 0.9)]
    assert fm.parse_markings(None) == []


def test_fit_camera_recovers_the_pitch_from_named_markings(scene):
    cam, _, marks = scene
    fit = fm.fit_camera(marks, W_IMG, H_IMG)
    assert fit is not None
    np.testing.assert_allclose(fit["pitch_dims"], (105.0, 68.0), atol=0.5)
    px = [(200, 650), (640, 600), (1100, 500)]
    np.testing.assert_allclose(
        _pitch_xy(fit["H_img2pitch"], px), _pitch_xy(rc.H_img2pitch(cam), px), atol=0.3
    )


def test_window_needs_two_agreeing_keyframes_and_never_fills_failed_frames(scene):
    _, img, marks = scene
    frames = list(range(0, 24, 2))
    motion = {f: (None if i == 0 else np.eye(3)) for i, f in enumerate(frames)}
    masks = {f: fm.verify_mask(img) for f in frames}
    masks[22] = fm.verify_mask(np.zeros_like(img))  # e.g. a frame where the lines vanish
    jittered = [(n, x + 1.5, y - 1.0, c) for n, x, y, c in marks]
    per, info = fm.calibrate_window(
        frames,
        motion,
        {0: img, 12: img},
        masks,
        answers={0: (marks, "white"), 12: (jittered, "white")},
    )
    assert info["kept"] == [0, 12]
    assert all(per[f]["H_img2pitch"] is not None for f in frames[:-1])
    assert per[22]["H_img2pitch"] is None  # unverified: unavailable, not filled
    # a lone keyframe is never trusted, however good its fit looks
    per1, info1 = fm.calibrate_window(
        frames, motion, {0: img}, masks, answers={0: (marks, "white")}
    )
    assert info1["kept"] == [] and all(p is None for p in per1.values())


def _done(result):
    f = Future()
    f.set_result(result)
    return f


def test_pipeline_markings_path_maps_to_one_pitch_and_needs_two_answers(scene):
    _, img, marks = scene
    frames = list(range(0, 12, 2))
    motion = {f: (None if i == 0 else np.eye(3)) for i, f in enumerate(frames)}
    masks = {f: fm.verify_mask(img) for f in frames}

    def answer(k, m):
        return _done(OracleResult(k, "markings", markings=m, line_colour="white"))

    pipe = FastPipeline(None, None)
    out, info = pipe._markings_calibration(
        frames, motion, {0: img}, masks, {0: answer(0, marks)}, fps=30.0
    )
    assert out is None and info["answered"] == 1  # consensus needs two answers
    jobs = {0: answer(0, marks), 8: answer(8, [(n, x + 1, y, c) for n, x, y, c in marks])}
    out, info = pipe._markings_calibration(frames, motion, {0: img, 8: img}, masks, jobs, fps=30.0)
    calib, source, pitch = out
    assert info["kept"] == [0, 8] and info["calibrated_frames"] == len(frames)
    assert pitch.length == pytest.approx(105.0, abs=0.5) and pitch.width == pytest.approx(
        68, abs=0.5
    )
    assert all(s.startswith("vlm markings") for s in source.values())
    centre = _pitch_xy(calib[4], [rc.project(scene[0], np.array([[52.5, 34.0, 0.0]]))[0][0]])[0]
    np.testing.assert_allclose(centre, (pitch.length / 2, pitch.width / 2), atol=0.5)


def test_second_opinion_markings_win_only_when_they_name_more(scene):
    _, img, marks = scene
    frames = [0, 2, 4, 6, 8]
    motion = {f: (None if i == 0 else np.eye(3)) for i, f in enumerate(frames)}
    masks = {f: fm.verify_mask(img) for f in frames}

    def answer(k, m):
        return _done(OracleResult(k, "markings", markings=m, line_colour="white"))

    pipe = FastPipeline(None, None)
    # primary names only 2 markings (unfittable); the second opinion names them all -> used
    jobs = {k: (answer(k, marks[:2]), answer(k, marks)) for k in (0, 8)}
    out, info = pipe._markings_calibration(frames, motion, {0: img, 8: img}, masks, jobs, 30.0)
    assert out is not None and info["kept"] == [0, 8]
    # equal counts: the primary (stronger) model is kept, so a bad second opinion is ignored
    bad = [(n, x + 200, y, c) for n, x, y, c in marks]
    jobs = {k: (answer(k, marks), answer(k, bad)) for k in (0, 8)}
    out2, info2 = pipe._markings_calibration(frames, motion, {0: img, 8: img}, masks, jobs, 30.0)
    assert info2["kept"] == [0, 8]


def test_consensus_never_lets_a_fit_confirm_itself_through_shared_answers(scene):
    cam, _, _ = scene
    fit = dict(H_img2pitch=rc.H_img2pitch(cam), pitch_dims=(105.0, 68.0))
    tr = {(0, 30): (np.eye(3), "direct"), (30, 0): (np.eye(3), "direct")}
    keep, _ = fm.consensus({0: fit, 30: fit}, tr, (W_IMG, H_IMG), sources={0: {0, 30}, 30: {30}})
    assert keep == {}  # fit 0 was pooled with 30's answer: no independent confirmation
    keep, rep = fm.consensus({0: fit, 30: fit}, tr, (W_IMG, H_IMG))
    assert sorted(keep) == [0, 30] and rep[0][0][2] == "direct"


def test_pair_transfers_use_the_chain_only_across_short_spans(scene):
    _, img, _ = scene
    frames = list(range(0, 80, 2))
    chain = fm.MotionChain(
        frames, {f: (None if i == 0 else np.eye(3)) for i, f in enumerate(frames)}
    )
    blank = np.zeros_like(img)  # nothing to register directly
    near = fm.pair_transfers({0: blank, 8: blank.copy()}, chain)
    far = fm.pair_transfers({0: blank, 60: blank.copy()}, chain)
    assert near[(0, 8)][1] == "chain" and (0, 60) not in far


def test_detector_frames_must_agree_with_unconfirmed_markings(scene):
    cam, _, marks = scene
    frames = list(range(0, 200, 2))
    motion = {f: (None if i == 0 else np.eye(3)) for i, f in enumerate(frames)}
    H = rc.H_img2pitch(cam)
    shifted = np.array([[1.0, 0.0, 6.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ H  # 6 m wrong
    calib = {f: None for f in frames}
    calib.update({10: H, 12: shifted, 190: H})
    source = {f: "anchor" for f in frames}
    pipe = FastPipeline(None, None)
    pipe._mk_answers = {0: (marks, "white")}
    ev = pipe._marking_evidence()
    out, src, info = pipe._check_against_markings(calib, source, frames, motion, ev, fps=30.0)
    assert out[10] is not None  # agrees with the markings
    assert out[12] is None and src[12].startswith("none")  # the markings contradict it
    assert out[190] is None  # no marking keyframe within the window: cannot be checked
    assert info["kept"] == 1 and info["rejected"] == 2
