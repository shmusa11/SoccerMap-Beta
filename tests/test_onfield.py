"""On-field rule (`soccer_analyzer.onfield`): synthetic tracks on a scale homography
(pixel = 0.1 m), pitch 105 x 68 m, 10 fps; the far touchline is y = 0."""

import numpy as np

from soccer_analyzer.onfield import (
    OnFieldCfg,
    ai_candidates,
    ask_participants,
    onfield_visibility,
)

FPS, L, W = 10.0, 105.0, 68.0
H = np.diag([0.1, 0.1, 1.0])  # pixel -> metres (mpp 0.1 -> margin 0.5 + 6 * 0.1 = 1.1 m)
CFG = OnFieldCfg()


def box(x, y):  # feet at pitch (x, y) metres
    return (x * 10 - 5, y * 10 - 18, x * 10 + 5, y * 10)


def people_of(tracks):
    people = {}
    for tid, pts in tracks.items():
        for f, (x, y) in pts.items():
            people.setdefault(f, []).append((tid, box(x, y)))
    return people


def run(tracks, calib_ok=lambda f: True, cfg=CFG, **kw):
    people = people_of(tracks)
    calib = {f: (H if calib_ok(f) else None) for f in people}
    return onfield_visibility(people, calib, FPS, (L, W), cfg, **kw)


def shown(v, tid):
    return sorted(f for f, s in v.visible.items() if tid in s)


def test_throw_in_taker_with_identity_breaks_is_relinked_and_drawn_off_the_pitch():
    # A on the pitch (0-29), B = new id 1.5-2.5 m outside (33-72), C = new id back on (75-99)
    A = {f: (40, 5 - f * 0.15) for f in range(30)}
    B = {f: (40, -1.5 - (f % 10) * 0.1) for f in range(33, 73)}
    C = {f: (40, 1 + (f - 75) * 0.1) for f in range(75, 100)}
    v = run({1: A, 2: B, 3: C})
    assert v.status[2] == "relinked" and len(shown(v, 2)) == 40
    assert all(2 in v.pitch_xy[f] and v.pitch_xy[f][2][1] < 0 for f in B)
    v0 = run({1: A, 2: B, 3: C}, cfg=OnFieldCfg(relink=False))
    assert v0.status[2] == "bystander" and not shown(v0, 2)
    assert v.hidden == set() and v0.hidden == {2}


def test_fence_bystander_hidden_despite_a_glitch_and_assistant_referee_kept():
    by = {f: (60, -4.0) for f in range(100)}
    by[50] = (60, 0.5)  # one-frame box glitch onto the pitch (median filtered)
    ar = {f: (30 + f * 0.2, -2.5) for f in range(100)}
    v = run({4: by, 5: ar}, kind={5: "referee"})
    assert v.status[4] == "bystander" and not shown(v, 4)
    assert v.status[5] == "protected" and len(shown(v, 5)) == 100


def test_off_pitch_frames_only_shown_within_the_window():
    P = {f: (70, -3.0) for f in range(200)}  # 20 s off the pitch, then on
    P.update({f: (70, 2.0) for f in range(200, 230)})
    s = shown(run({6: P}), 6)
    assert s[0] == 200 - int(CFG.window_s * FPS) and s[-1] == 229


def test_uncalibrated_tracks_use_the_oracle_vote_then_the_ai():
    U = {f: (50, 30) for f in range(50)}
    v = run(
        {7: U, 8: U, 9: U},
        calib_ok=lambda f: False,
        track_prob={7: 0.05, 8: 0.05, 9: 0.6},
        kind={8: "player"},  # the AI never overrules an oracle label
    )
    assert v.status == {7: "ai-hide", 8: "uncalibrated", 9: "uncalibrated"}
    assert run({7: U}, calib_ok=lambda f: False, legacy_offfield={7}).status[7] == "bystander"


def test_calibration_gap_keeps_a_player_shown():
    G = {f: (50, 30) for f in range(100)}
    v = run({10: G}, calib_ok=lambda f: not 30 <= f < 70)
    assert len(shown(v, 10)) == 100


def test_ai_is_only_asked_about_tracks_its_answer_can_change():
    U = {f: (50, 30) for f in range(10)}
    people = people_of({1: U, 2: U, 3: U, 4: U})
    calib = {f: (H if f < 2 else None) for f in people}
    calib_none = {f: None for f in people}
    assert ai_candidates(people, calib, {}, set()) == set()  # all have calibrated frames
    assert ai_candidates(people, calib_none, {2: "player"}, {3}) == {1, 4}


def test_ask_participants_mosaics_and_averages_per_track():
    frames = {f: np.full((200, 300, 3), 90, np.uint8) for f in (0, 15, 30)}
    people = {
        f: [(1, (10, 20, 30, 80)), (2, (200, 20, 220, 80)), (3, (100, 20, 120, 80))] for f in frames
    }
    asked = []

    def decide(images, questions, text=None, max_width=1920):
        asked.append((images[0].shape, len(questions)))
        names = [q["name"] for q in questions if q["name"].startswith("participant_")]
        return {"answers": {n: {"probability": 0.1 * int(n.split("_")[1])} for n in names}}

    probs, info = ask_participants(decide, frames, people, {1, 2}, k=3)
    assert set(probs) == {1, 2} and all(0.0 < p <= 0.6 for p in probs.values())
    assert asked == [((768, 1024, 3), 12)] and info["tiles"] == 6 and info["calls"] == 1
    # budget exhausted / call failed: no answers, the tracks keep the default (shown)
    probs, info = ask_participants(lambda *a, **k: None, frames, people, {1}, k=3)
    assert probs == {} and info["failed"] == 1


def test_oracle_bystanders_standing_on_the_touchline_stay_hidden():
    # Bench row on the far touchline (0.2-1 m inside under an assumed pitch width), never
    # matched by the oracle: hidden. A player the oracle also missed but who runs into the
    # pitch, and a labelled player standing at the line, stay shown.
    bench = {f: (60 + (f % 5) * 0.1, 0.2 + (f % 9) * 0.1) for f in range(100)}
    runner = {f: (40, 0.5 + f * 0.1) for f in range(100)}
    taker = {f: (80, 0.3) for f in range(100)}
    off = {7, 8}
    v = run({7: bench, 8: runner, 9: taker}, legacy_offfield=off, kind={9: "player"})
    assert v.status[7] == "bystander" and not shown(v, 7)
    assert v.status[8] == "on-pitch" and len(shown(v, 8)) == 100
    assert v.status[9] == "on-pitch" and len(shown(v, 9)) == 100
    v0 = run({7: bench}, legacy_offfield=off, cfg=OnFieldCfg(edge_m=0.0))
    assert v0.status[7] == "on-pitch"
