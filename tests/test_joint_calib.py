import json
from concurrent.futures import Future

import numpy as np
import pytest

from soccer_analyzer.calibration import joint_calib as jc
from soccer_analyzer.calibration import refcore as rc
from soccer_analyzer.fast_pipeline import FastConfig, FastPipeline
from soccer_analyzer.vision_oracle import OracleResult


def _err_m(per, cams, frames, w, h):
    """Median ground error (m) of each calibrated frame against the true camera."""
    xs, ys = np.meshgrid(np.linspace(0.1, 0.9, 6) * w, np.linspace(0.45, 0.95, 4) * h)
    g = np.c_[xs.ravel(), ys.ravel()]
    out = {}
    for f in frames:
        H = per[f]["H_img2pitch"]
        if H is None:
            continue
        _, W = per[f]["pitch_dims"]
        est = jc._apply(H, g) + np.array([0.0, (68.0 - W) / 2])
        true = jc._apply(jc.rc.H_img2pitch(cams[f]), g)
        out[f] = float(np.median(np.linalg.norm(est - true, axis=1)))
    return out


def test_shared_centre_recovers_a_turning_zooming_camera():
    # 60 frames, 40 deg pan, 1.3x zoom, landmarks (2 px noise) at only 3 keyframes.
    out = jc.selftest()
    assert out["coverage"] > 0.9 and out["median_err_m"] < 0.3
    assert np.linalg.norm(np.subtract(out["C"], (52.0, 90.0, -15.0))) < 1.0


def test_one_keyframe_cannot_be_confirmed_and_stays_unavailable():
    # The split-half confirmation needs two independent answers: one keyframe -> nothing shown.
    frames, G, answers, _, (w, h) = jc.synthetic_window(keys=(30,))
    per, info = jc.calibrate_window_joint(
        frames, G, {}, None, answers, frame_size=(w, h), use_direct=False, chamfer=False
    )
    assert info["coverage"] == 0.0
    assert all(p["H_img2pitch"] is None for p in per.values())


def test_keyframes_are_chosen_by_view_not_by_time():
    frames, G, _, cams, (w, h) = jc.synthetic_window()
    keys4 = jc.select_keyframes(frames, G, frames, (w, h), k=4, fps=30.0)
    keys2 = jc.select_keyframes(frames, G, frames, (w, h), k=2, fps=30.0)
    assert len(keys4) == 4 and keys4 == sorted(keys4)
    assert set(keys2) <= set(keys4)  # prefix-stable: asking more only adds keyframes
    # widest view first: the camera zooms in over time, so the first frame is the widest
    assert frames[0] in keys2
    # the two views chosen first are the two ends of the pan
    assert frames[-1] in keys2


def test_keyframe_choice_is_per_motion_segment():
    frames, G, _, _, (w, h) = jc.synthetic_window()
    G = dict(G)
    G[frames[30]] = None  # a cut: two segments
    keys = jc.select_keyframes(frames, G, frames, (w, h), k=2, fps=30.0)
    assert len([k for k in keys if k < frames[30]]) == 2
    assert len([k for k in keys if k >= frames[30]]) == 2


def test_static_camera_spreads_keyframes_in_time():
    frames = list(range(0, 120, 2))
    G = {f: (None if i == 0 else np.eye(3)) for i, f in enumerate(frames)}
    keys = jc.select_keyframes(frames, G, frames, (1920, 1080), k=3, fps=30.0, min_gap_s=1.0)
    assert len(keys) == 3
    assert min(np.diff(keys)) >= 30  # at least min_gap_s apart


def test_yard_numbers_parse_scale_and_trigger_football_mode():
    reply = {
        "numbers": [
            {"value": 40, "x": 100, "y": 50, "toward": "Right", "confidence": 0.9},
            {"value": "x", "x": 1, "y": 1},
            "junk",
            {"value": 30, "x": 10, "y": 5, "confidence": 0.4},
        ]
    }
    ns = jc.parse_numbers(reply, scale=(2.0, 2.0))
    assert [(n["value"], n["x"], n["y"], n["toward"]) for n in ns] == [
        (40, 200.0, 100.0, "right"),
        (30, 20.0, 10.0, ""),
    ]
    assert jc.football_numbers_found({5: ns})
    assert not jc.football_numbers_found({5: ns[1:]})  # only a low-confidence read
    assert jc.parse_numbers({"numbers": None}) == [] and jc.parse_numbers("x") == []


def test_paint_fraction_counts_all_colours():
    vm = {
        "white": np.packbits(np.array([1, 0, 0, 0, 0, 0, 0, 0] * 4, np.uint8)),
        "yellow": np.packbits(np.array([1, 1, 0, 0, 0, 0, 0, 0] * 4, np.uint8)),
    }
    assert jc.paint_fraction(vm) == pytest.approx(3 / 8)
    assert jc.paint_fraction(None) == 0.0


def _done(r):
    f = Future()
    f.set_result(r)
    return f


def test_pipeline_asks_more_keyframes_until_the_field_is_confirmed():
    # One answered keyframe cannot be confirmed; the pipeline asks the next coverage keyframes
    # (`more`) and calibrates once two independent answers exist. Wrong-free by construction:
    # frames are only shown after the split-half confirmation.
    frames, G, answers, cams, (w, h) = jc.synthetic_window()
    keys = sorted(answers)
    img = np.zeros((h, w, 3), np.uint8)

    def result(k):
        marks, colour = answers[k]
        return OracleResult(k, "markings", markings=list(marks), line_colour=colour)

    jobs = {keys[0]: (_done(result(keys[0])), None)}
    images = {keys[0]: img}
    asked = []

    def more(n):
        asked.append(n)
        for k in keys[:n]:
            jobs.setdefault(k, (_done(result(k)), None))
            images.setdefault(k, img)
        return len(jobs)

    cfg = FastConfig(markings_keys=2, markings_max_keys=12)
    pipe = FastPipeline(None, None, config=cfg)
    out, info = pipe._joint_calibration(frames, G, images, {}, jobs, 30.0, more=more)
    assert asked == [4]
    assert len(info["attempts"]) == 2 and info["attempts"][0]["answered"] == 1
    json.dumps(info)  # the run summary is printed / written as JSON
    assert out is not None
    calib, source, pitch = out
    shown = [f for f in frames if calib[f] is not None]
    assert len(shown) > 0.9 * len(frames)
    assert all(source[f] == "vlm markings (joint)" for f in shown)
    per = {f: dict(H_img2pitch=calib[f], pitch_dims=(pitch.length, pitch.width)) for f in frames}
    errs = _err_m(per, cams, frames, w, h)
    assert max(errs.values()) < 0.5


def test_pipeline_joint_keeps_the_field_unavailable_without_a_second_answer():
    frames, G, answers, _, (w, h) = jc.synthetic_window()
    k = sorted(answers)[1]
    marks, colour = answers[k]
    jobs = {k: (_done(OracleResult(k, "markings", markings=marks, line_colour=colour)), None)}
    images = {k: np.zeros((h, w, 3), np.uint8)}
    pipe = FastPipeline(None, None, config=FastConfig())
    out, info = pipe._joint_calibration(frames, G, images, {}, jobs, 30.0, more=None)
    assert out is None and info["answered"] == 1


def test_config_rejects_bad_marking_knobs():
    with pytest.raises(ValueError):
        FastConfig(markings_method="other")
    with pytest.raises(ValueError):
        FastConfig(markings_select="random")
    with pytest.raises(ValueError):
        FastConfig(markings_football="maybe")
    with pytest.raises(ValueError):
        FastConfig(markings_keys=1)
    with pytest.raises(ValueError):
        FastConfig(markings_keys=6, markings_max_keys=4)


def test_player_height_term_is_zero_at_the_true_camera_and_sees_a_wrong_height():
    # Synthetic players 1.75 m tall standing on the pitch, boxed through the true camera: the
    # bundle's player term has (near) zero cost there and a clear cost for a camera at twice the
    # height (same view direction), which painted landmarks near one goal cannot tell apart.
    frames, _, _, cams, (w, h) = jc.synthetic_window()
    f = frames[30]
    cam = cams[f]
    rng = np.random.default_rng(0)
    X = np.c_[rng.uniform(20, 85, 30), rng.uniform(5, 63, 30), np.zeros(30)]
    feet, _ = jc.rc.project(cam, X)
    head, _ = jc.rc.project(cam, X - np.array([0.0, 0.0, 1.75]))
    xyxy = np.c_[feet[:, 0] - 0.15 * (feet[:, 1] - head[:, 1]), head[:, 1]]
    xyxy = np.c_[xyxy, feet[:, 0] + 0.15 * (feet[:, 1] - head[:, 1]), feet[:, 1]]
    keep = (xyxy[:, 0] > 5) & (xyxy[:, 2] < w - 5) & (xyxy[:, 1] > 5) & (xyxy[:, 3] < h - 5)
    obs = jc.people_obs({f: (xyxy[keep], np.full(keep.sum(), 0.9))}, {f: cam}, (105.0, 68.0), w, h)
    assert len(obs) >= 10

    def cost(c):
        _, _, info = jc.bundle(
            [f],
            {f: c},
            (105.0, 68.0),
            w,
            h,
            [],
            [],
            fit_dims=(False, False),
            max_nfev=1,
            people=dict(obs=obs, n_eff=30, h=1.75),
        )
        return info["cost"]

    high = dict(cam, C=np.asarray(cam["C"], float) * np.array([1.0, 1.0, 2.0]))
    assert cost(cam) < 1e-6 < 0.5 < cost(high)


def test_a_shifted_keyframe_answer_is_withheld_or_stays_within_a_metre():
    # One keyframe's marks shifted by 60 px (a confidently wrong VLM answer): the split-half
    # check (median AND worst tenth of the view) may only show frames that remain right.
    frames, G, answers, cams, (w, h) = jc.synthetic_window(keys=(5, 30, 55))
    k = sorted(answers)[1]
    marks, col = answers[k]
    answers = dict(answers)
    answers[k] = ([(n, x + 60.0, y, c) for n, x, y, c in marks], col)
    per, info = jc.calibrate_window_joint(
        frames, G, {}, None, answers, frame_size=(w, h), use_direct=False, chamfer=False
    )
    errs = _err_m(per, cams, frames, w, h)
    assert all(e < 1.0 for e in errs.values()), max(errs.values())


def test_frames_far_outside_every_keyframe_view_stay_unavailable():
    # Keyframes only at the start of a long pan: the end of the pan shares no view with them,
    # so its field would rest on chained motion alone -> withheld ("outside keyframe views").
    frames, G, answers, _, (w, h) = jc.synthetic_window(pan_deg=(-40.0, 40.0), keys=(2, 8))
    per, info = jc.calibrate_window_joint(
        frames, G, {}, None, answers, frame_size=(w, h), use_direct=False, chamfer=False
    )
    late = [per[f] for f in frames[-10:]]
    assert all(p["H_img2pitch"] is None for p in late)
    assert {p["reason"] for p in late} == {"outside keyframe views"}
    assert all(p["seen"] < 0.5 for p in late)
    assert per[frames[5]]["H_img2pitch"] is not None  # between the keyframes: shown


def test_stability_repeat_runs_with_jittered_motion_and_keeps_a_stable_field():
    frames, G, answers, cams, (w, h) = jc.synthetic_window()
    Gj = jc.jitter_motion(G, w, 0.3, rng=1)
    assert Gj[frames[0]] is None
    d = [np.abs(Gj[f] - G[f])[:2, 2].max() for f in frames[1:]]
    assert 0 < max(d) < 3.0  # sub-pixel-scale translation noise only
    per, info = jc.calibrate_window_joint(
        frames,
        G,
        {},
        None,
        answers,
        frame_size=(w, h),
        use_direct=False,
        chamfer=False,
        stability=1,
    )
    assert len(info["stability"]) == 1 and info["coverage"] > 0.9
    errs = _err_m(per, cams, frames, w, h)
    assert max(errs.values()) < 0.5


def test_stability_withholds_frames_a_jittered_repeat_does_not_confirm(monkeypatch):
    # If the repeat with jittered motion lands elsewhere (knife-edge input), nothing is shown.
    frames, G, answers, _, (w, h) = jc.synthetic_window()
    calls = []
    orig = jc._joint_once

    def fake(render_frames, G_, key_images, masks, answers_, **kw):
        calls.append(1)
        per, info = orig(render_frames, G_, key_images, masks, answers_, **kw)
        if len(calls) > 1:  # the repeat: move every camera 5 m along the touchline
            for p in per.values():
                if p.get("camera") is not None:
                    p["camera"] = dict(p["camera"], C=np.asarray(p["camera"]["C"]) + [5.0, 0, 0])
        return per, info

    monkeypatch.setattr(jc, "_joint_once", fake)
    per, info = jc.calibrate_window_joint(
        frames,
        G,
        {},
        None,
        answers,
        frame_size=(w, h),
        use_direct=False,
        chamfer=False,
        stability=1,
    )
    assert len(calls) == 2
    assert info["coverage"] == 0.0
    assert {p["reason"] for p in per.values()} <= {
        "unstable",
        "no consensus",
        "outside keyframe views",
    }


def test_majority_colour_breaks_ties_toward_the_non_white_answer():
    assert jc.majority_colour(["white", "yellow"]) == "yellow"
    assert jc.majority_colour(["yellow", "white"]) == "yellow"
    assert jc.majority_colour(["white", "white", "yellow"]) == "white"
    assert jc.majority_colour([]) == "white"


def test_keyframe_choice_skips_low_paint_frames_hidden_by_an_overlay():
    frames, G, _, _, (w, h) = jc.synthetic_window()
    paint = {f: 0.065 for f in frames}
    paint[frames[0]] = 0.046  # the widest view, but a phone UI covers part of the field
    keys = jc.select_keyframes(frames, G, frames, (w, h), k=2, paint=paint, min_paint_ratio=0.8)
    assert frames[0] not in keys and len(keys) == 2
    keys0 = jc.select_keyframes(frames, G, frames, (w, h), k=2, paint=paint)
    assert frames[0] in keys0  # without the paint rule the overlay frame would be chosen


def _football_view(L, W=68.0, w=1920, h=1080):
    """A camera on the near side filming the right end of a soccer pitch laid on an American-
    football field: WHITE paint mask of the football grid (yard lines every 5 yd, end lines,
    sidelines), the four goal-post points a VLM names, two read yard numbers."""
    import cv2

    C = np.array([50.0, W + 30.0, -11.0])
    z = np.array([88.0, 30.0, 0.0]) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, -1.0])
    x /= np.linalg.norm(x)
    R = np.vstack([x, np.cross(z, x), z])
    cam = dict(rvec=cv2.Rodrigues(R)[0].ravel(), C=C, f=1.1 * w, cx=w / 2.0, cy=h / 2.0)
    s = 960.0 / w
    sh = (int(round(h * s)), 960)
    white = np.zeros(sh, np.uint8)
    p, zz = jc.rc.project(cam, jc.football_samples_param(L, W, step=0.25)(L, W))
    for (u, v), d in zip(p * s, zz):
        if d > 0 and 0 <= u < sh[1] and 0 <= v < sh[0]:
            cv2.circle(white, (int(u), int(v)), 1, 1, -1)
    blank = np.packbits(np.zeros(sh, np.uint8))
    vm = {"scale": s, "shape": sh, "white": np.packbits(white), "yellow": blank, "red": blank}
    P = jc.fm.model_points(L, W)
    marks = []
    for n in ("r_post_far_base", "r_post_near_base", "r_post_far_top", "r_post_near_top"):
        q, _ = jc.rc.project(cam, P[n][None])
        marks.append((n, float(q[0, 0]), float(q[0, 1]), 0.9))
    yard = []
    for v in (30, 20):
        xr = (50 - v) * jc.YD
        q, _ = jc.rc.project(cam, np.array([[L / 2 + xr, W / 2 + 12.0, 0.0]]))
        yard.append((xr, float(q[0, 0]), float(q[0, 1]), 0.9))
    return cam, vm, marks, yard


def test_football_mode_recovers_the_camera_from_posts_grid_and_yard_numbers():
    # Football mode (rpi4k-like): the goal posts alone are degenerate (collinear on the goal
    # line); the white football grid + two read yard numbers fix the camera. Goals on the end
    # lines (L = 120 yd): recovered within 0.3 m.
    L = 120 * jc.YD
    cam, vm, marks, yard = _football_view(L)
    est, info = jc.football_init(
        marks, jc.fm._mask_dt(vm, "white"), vm["scale"], 1920, 1080, yard=yard
    )
    assert est is not None
    assert jc.disagreement_m(cam, (L, 68.0), est, (info["L"], 68.0), 1920, 1080) < 0.3


def test_football_mode_goal_line_inside_the_end_line_is_a_known_bias():
    # Known limitation (round-4 verifier: rpi4k's drawn goal line sits 0.6-0.8 m behind the
    # posts): with the goal line ~0.7 m inside the end line, the end-line prior pulls the pitch
    # length to the end line and the view is ~0.6 m off -- still well inside the 1.5 m gate, so
    # this is an accuracy limit, not a wrong field. A looser prior was NOT better with posts
    # alone (lab probe, 2026-10-14): another along-field soccer landmark is needed to free it.
    L = 120 * jc.YD - 1.33
    cam, vm, marks, yard = _football_view(L)
    est, info = jc.football_init(
        marks, jc.fm._mask_dt(vm, "white"), vm["scale"], 1920, 1080, yard=yard
    )
    d = jc.disagreement_m(cam, (L, 68.0), est, (info["L"], 68.0), 1920, 1080)
    assert 0.3 < d < 1.0 and info["L"] > L + 0.5


def _scripted_attempts(pipe, monkeypatch, attempts):
    """Replace `_joint_attempt` with scripted per-frame results, one per call."""
    calls = iter(attempts)

    def attempt(render_frames, motion, images, masks, jobs, fps):
        per = next(calls)
        withheld = {}
        for p in per.values():
            if p.get("H_img2pitch") is None:
                withheld[p["reason"]] = withheld.get(p["reason"], 0) + 1
        res = (
            dict(pitch_dims=(105.0, 68.0))
            if any(p.get("H_img2pitch") is not None for p in per.values())
            else None
        )
        return per, res, dict(answered=len(jobs), kept=sorted(jobs), withheld=withheld)

    monkeypatch.setattr(pipe, "_joint_attempt", attempt)


def _per(frames, cams, shown, reason="no consensus"):
    out = {}
    for f in frames:
        H = rc.H_img2pitch(cams[f])
        ok = f in shown
        out[f] = dict(
            H_img2pitch=H if ok else None,
            camera=cams[f],
            pitch_dims=(105.0, 68.0),
            reason=None if ok else reason,
        )
    return out


@pytest.mark.parametrize("second", ["disagrees", "agrees"])
def test_more_evidence_never_falls_back_to_a_more_permissive_attempt(monkeypatch, second):
    # Attempt 1 (2 keyframes) confirms 80% of the frames but leaves 20% outside every keyframe's
    # view, so one more keyframe is asked. Attempt 2 (more evidence) confirms only the first half.
    # The product must show only what the LAST attempt confirms -- and of that, only frames
    # where both attempts land within joint_tol_m -- never attempt 1's larger, less-informed set.
    frames, G, answers, cams, (w, h) = jc.synthetic_window()
    moved = {
        f: dict(c, C=np.asarray(c["C"], float) + np.array([6.0, 0.0, 0.0])) for f, c in cams.items()
    }
    n = len(frames)
    first = _per(frames, cams, set(frames[: int(0.8 * n)]), reason="outside keyframe views")
    half = set(frames[: n // 2])
    second_per = _per(frames, moved if second == "disagrees" else cams, half)
    pipe = FastPipeline(None, None, config=FastConfig(markings_keys=2, markings_max_keys=3))
    _scripted_attempts(pipe, monkeypatch, [first, second_per])
    img = np.zeros((h, w, 3), np.uint8)
    jobs = {k: None for k in sorted(answers)[:2]}
    images = {k: img for k in jobs}

    def more(n_keys):
        jobs[sorted(answers)[2]] = None
        return len(jobs)

    out, info = pipe._joint_calibration(frames, G, images, {}, jobs, 30.0, more=more)
    assert len(info["attempts"]) == 2
    if second == "disagrees":
        assert out is None and info["attempts"][1]["attempts_disagree"] == len(half)
    else:
        calib = out[0]
        assert {f for f in frames if calib[f] is not None} == half


def test_withhold_disagreeing_compares_median_and_worst_tenth():
    frames, G, answers, cams, (w, h) = jc.synthetic_window()
    f = frames[10]
    per = _per([f], cams, {f})
    same = _per([f], cams, {f})
    assert jc.withhold_disagreeing(per, same, w, h, 1.5) == 0
    assert per[f]["H_img2pitch"] is not None
    far = {f: dict(cams[f], C=np.asarray(cams[f]["C"], float) + np.array([0.0, 0.0, 3.0]))}
    other = _per([f], far, {f})
    assert jc.withhold_disagreeing(per, other, w, h, 1.5, reason="unstable") == 1
    assert per[f]["H_img2pitch"] is None and per[f]["reason"] == "unstable"
    # a frame the other fit does not show: withheld only when it is required
    per2, none = _per([f], cams, {f}), _per([f], cams, set())
    assert jc.withhold_disagreeing(per2, none, w, h, 1.5, require=False) == 0
    assert jc.withhold_disagreeing(per2, none, w, h, 1.5, require=True) == 1


def test_dims_pinned_flags_only_fitted_dims_on_a_bound():
    # rpi4k: every wrong-basin fit had the soccer goal line pinned on the football end-line
    # bound (L = 110.33 m); correct fits sat inside it.
    fb = (jc.FB_L_BOUNDS, jc.DIMS_BOUNDS[1])
    assert jc.dims_pinned((jc.FB_L_BOUNDS[1], 68.0), (True, False), fb) == [0]
    assert jc.dims_pinned((109.85, 68.0), (True, False), fb) == []
    assert jc.dims_pinned((109.85, 55.0), (True, False), fb) == []  # width not fitted
    assert jc.dims_pinned((105.0, 55.0), (True, True), jc.DIMS_BOUNDS) == [1]


def test_stability_repeat_also_perturbs_the_keyframe_images(monkeypatch):
    # The repeat sees the keyframes resampled (as if stored at another resolution): rpi4k's fit
    # flipped between a right and a wrong basin with the keyframe resolution alone.
    frames, G, answers, _, (w, h) = jc.synthetic_window()
    rng = np.random.default_rng(0)
    imgs = {k: (rng.random((h // 8, w // 8, 3)) * 255).astype(np.uint8) for k in answers}
    seen = []
    orig = jc._joint_once

    def spy(render_frames, G_, key_images, masks, answers_, **kw):
        seen.append(dict(key_images))
        return orig(render_frames, G_, key_images, masks, answers_, **kw)

    monkeypatch.setattr(jc, "_joint_once", spy)
    per, info = jc.calibrate_window_joint(
        frames,
        G,
        imgs,
        None,
        answers,
        frame_size=(w, h),
        use_direct=False,
        chamfer=False,
        stability=1,
    )
    assert len(seen) == 2
    for k, im in imgs.items():
        assert seen[0][k] is im
        assert seen[1][k].shape == im.shape and not np.array_equal(seen[1][k], im)
    assert info["coverage"] > 0.9 and info["stability"][0]["lost"] == 0
