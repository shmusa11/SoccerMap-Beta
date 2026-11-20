import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from soccer_analyzer.vision_oracle import (
    LANDMARKS,
    CachedVisionOracle,
    OpenAIVisionOracle,
    landmark_homography,
    parse_landmarks,
    parse_people,
)


def test_parse_people_scales_clips_and_drops_malformed():
    payload = {
        "people": [
            {"role": "player", "team": " Red ", "box": [10, 20, 30, 60]},
            {"role": "referee", "team": "yellow", "box": [50, 5, 40, 45]},  # swapped corners
            {"role": "coach", "team": "white", "box": [900, 0, 1100, 50]},  # clipped, role->player
            {"role": "player", "box": [1, 2, 3]},  # malformed
            {"role": "player", "box": ["a", 1, 2, 3]},  # malformed
            "nonsense",
        ],
        "ball": [100, 50],
    }
    people, ball = parse_people(payload, sx=2.0, sy=2.0, w=2000, h=1000)
    assert [p.box for p in people] == [
        (20.0, 40.0, 60.0, 120.0),
        (80.0, 10.0, 100.0, 90.0),
        (1800.0, 0.0, 2000.0, 100.0),
    ]
    assert [p.role for p in people] == ["player", "referee", "player"]
    assert people[0].team == "red" and people[1].team is None
    assert ball == (200.0, 100.0)
    assert parse_people({"ball": [5000, 1]}, 1, 1, 100, 100)[1] is None  # off-image ball


def test_parse_landmarks_keeps_known_ids_only():
    payload = {
        "landmarks": [
            {"id": "centre_spot", "xy": [10, 10]},
            {"id": "made_up", "xy": [1, 1]},
            {"id": "left_penalty_spot", "xy": [1e9, 1]},
        ]
    }
    marks, ball = parse_landmarks(payload, 1.0, 1.0, 100, 100)
    assert marks == {"centre_spot": (10.0, 10.0)} and ball is None


def test_landmark_homography_recovers_a_camera_and_rejects_collinear_points():
    # Ground truth: pitch metres -> pixels by a simple projective map.
    G = np.array([[12.0, 2.0, 100.0], [1.0, 8.0, 50.0], [0.0, 0.004, 1.0]])

    def to_px(x, y):
        q = G @ np.array([x, y, 1.0])
        return float(q[0] / q[2]), float(q[1] / q[2])

    ids = ["centre_spot", "circle_left", "circle_right", "halfway_far", "left_box_front_far"]
    marks = {k: to_px(*LANDMARKS[k][:2]) for k in ids}
    H = landmark_homography(marks)
    p = H @ np.array([*marks["left_box_front_far"], 1.0])
    np.testing.assert_allclose(p[:2] / p[2], LANDMARKS["left_box_front_far"][:2], atol=1e-3)
    on_halfway = ("halfway_far", "circle_far", "centre_spot", "circle_near")
    assert landmark_homography({k: to_px(*LANDMARKS[k][:2]) for k in on_halfway}) is None
    assert landmark_homography({k: marks[k] for k in ids[:3]}) is None  # too few points


class FakeClient:
    """Stands in for the OpenAI client: returns a fixed JSON answer and usage. Calls block until
    `gate` is set, so tests control how many requests are in flight."""

    def __init__(self, payload, tokens=(1000, 200)):
        self.payload, self.tokens, self.calls = payload, tokens, 0
        self.gate = threading.Event()
        self.gate.set()
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.gate.wait(5)
        self.calls += 1
        return SimpleNamespace(
            output_text=json.dumps(self.payload),
            usage=SimpleNamespace(input_tokens=self.tokens[0], output_tokens=self.tokens[1]),
        )


def test_oracle_costs_calls_and_enforces_budget():
    client = FakeClient({"people": [{"role": "player", "team": "red", "box": [0, 0, 10, 20]}]})
    frame = np.zeros((540, 960, 3), np.uint8)
    # gpt-6-luna: 1000 in + 200 out tokens = $0.0002; reservation per call is $0.00125
    oracle = OpenAIVisionOracle(people_model="gpt-6-luna", max_usd=0.003, client=client)
    client.gate.clear()  # hold both first calls in flight while the third is submitted
    jobs = [oracle.submit_people(i, frame) for i in range(3)]
    client.gate.set()
    results = [j.result() for j in jobs if j is not None]
    oracle.close()
    assert sum(j is None for j in jobs) == 1  # third call would exceed the cap while reserved
    assert client.calls == 2
    assert oracle.spent_usd == pytest.approx(2 * 0.0002)
    assert results[0].people[0].team == "red" and results[0].people[0].box == (0, 0, 10, 20)


def test_budget_reserves_a_realistic_amount_per_markings_call():
    # gpt-6-sol markings calls cost $0.012-0.021 (measured); each reserves $0.025 while in
    # flight. A $0.10 budget must allow 4 keyframes in flight (the old $0.048 reservation
    # allowed only 2, so joint calibration could not reach its 4-keyframe step), and refuse a
    # 5th until answers come back.
    client = FakeClient({"markings": [], "soccer_line_colour": "white"}, tokens=(2500, 1200))
    frame = np.zeros((540, 960, 3), np.uint8)
    oracle = OpenAIVisionOracle(markings_model="gpt-6-sol", max_usd=0.10, client=client)
    client.gate.clear()
    jobs = [oracle.submit_markings(i, frame) for i in range(5)]
    assert sum(j is not None for j in jobs) == 4 and jobs[4] is None
    client.gate.set()
    [j.result() for j in jobs if j is not None]
    # actual cost replaces the reservation: 4 x (2500 x 2 + 1200 x 10) / 1e6 = $0.068
    assert oracle.spent_usd == pytest.approx(4 * 0.017)
    assert oracle.submit_markings(9, frame) is not None  # $0.032 left >= one $0.025 reservation
    oracle.close()


def test_cached_oracle_replays_without_calling(tmp_path):
    client = FakeClient({"landmarks": [{"id": "centre_spot", "xy": [5, 6]}], "ball": [7, 8]})
    path = str(tmp_path / "cache.json")
    frame = np.zeros((100, 200, 3), np.uint8)
    first = CachedVisionOracle(OpenAIVisionOracle(max_usd=1.0, client=client), path, "clip")
    r1 = first.submit_landmarks(3, frame).result()
    first.close()
    second = CachedVisionOracle(OpenAIVisionOracle(max_usd=1.0, client=client), path, "clip")
    r2 = second.submit_landmarks(3, frame).result()
    second.close()
    assert client.calls == 1
    assert r2.landmarks == r1.landmarks == {"centre_spot": (5.0, 6.0)} and r2.ball == (7.0, 8.0)


def test_oracle_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIVisionOracle()


def test_markings_and_ball_requests_parse_scale_and_replay_from_cache(tmp_path):
    marks = {
        "soccer_line_colour": "Yellow",
        "markings": [
            {"name": "centre", "x": 480, "y": 270, "confidence": 0.9},
            {"name": "not_a_marking", "x": 1, "y": 1, "confidence": 1.0},
        ],
    }
    client = FakeClient(marks)
    frame = np.zeros((1080, 3840, 3), np.uint8)  # sent at 1920 px: coordinates scale by 2
    path = str(tmp_path / "cache.json")
    first = CachedVisionOracle(OpenAIVisionOracle(max_usd=1.0, client=client), path, "clip")
    r = first.submit_markings(7, frame).result()
    assert r.markings == [("centre", 960.0, 540.0, 0.9)] and r.line_colour == "yellow"
    first.close()
    second = CachedVisionOracle(OpenAIVisionOracle(max_usd=1.0, client=client), path, "clip")
    assert second.submit_markings(7, frame).result().markings == r.markings
    second.close()
    assert client.calls == 1  # replayed

    ball = FakeClient({"ball": [30, 40], "confidence": 0.8})
    crop = np.zeros((100, 200, 3), np.uint8)
    cached = CachedVisionOracle(OpenAIVisionOracle(max_usd=1.0, client=ball), path, "clip")
    a = cached.submit_ball(7, crop, tag="@0,0,200,100").result()
    b = cached.submit_ball(7, crop, tag="@50,0,250,100").result()  # another crop: another call
    c = cached.submit_ball(7, crop, tag="@0,0,200,100").result()
    cached.close()
    assert a.ball == b.ball == c.ball == (30.0, 40.0) and ball.calls == 2
    assert "clip|7|ball@0,0,200,100|gpt-6-luna" in json.load(open(path))


def test_decide_costs_calls_enforces_budget_and_caches(tmp_path):
    calls = []

    def transport(images, questions, text=None, max_width=1920):
        calls.append((len(images), len(questions), text, max_width))
        return {"answers": {q["name"]: {"probability": 0.7} for q in questions}, "_usd": 0.0002}

    q = [dict(type="predicate", name="participant_1", instructions="Tile 1: playing.")]
    img = np.zeros((64, 64, 3), np.uint8)
    # reservation for 1 image + 1 question = $0.000306: the budget allows exactly one call
    oracle = OpenAIVisionOracle(max_usd=0.0005, client=FakeClient({}), decide_fn=transport)
    path = str(tmp_path / "cache.json")
    cached = CachedVisionOracle(oracle, path, "clip")
    r = cached.decide([img], q, text="ctx", max_width=1024)
    assert r["answers"]["participant_1"]["probability"] == 0.7
    assert calls == [(1, 1, "ctx", 1024)] and oracle.spent_usd == pytest.approx(0.0002)
    assert cached.decide([img], q, text="ctx", max_width=1024) == r  # replayed, no new call
    assert oracle.decide([img + 1], q) is None  # over budget: skipped, never raises
    cached.close()
    again = CachedVisionOracle(
        OpenAIVisionOracle(max_usd=0.0005, client=FakeClient({}), decide_fn=transport), path, "clip"
    )
    assert again.decide([img], q, text="ctx", max_width=1024) == r and len(calls) == 1
    again.close()


def test_decide_failure_is_no_answer_and_logs_no_details():
    def broken(*a, **k):
        raise RuntimeError("HTTP 500 secret-ish body")

    oracle = OpenAIVisionOracle(max_usd=1.0, client=FakeClient({}), decide_fn=broken)
    assert oracle.decide([np.zeros((8, 8, 3), np.uint8)], []) is None
    assert oracle.spent_usd == 0.0 and "secret" not in json.dumps(oracle.calls)
    oracle.close()


def test_yard_number_requests_parse_scale_and_replay_from_cache(tmp_path):
    reply = {"numbers": [{"value": 30, "x": 100, "y": 40, "toward": "left", "confidence": 0.8}]}
    client = FakeClient(reply)
    frame = np.zeros((1080, 3840, 3), np.uint8)  # sent at 1920 px: coordinates scale by 2
    path = str(tmp_path / "cache.json")
    first = CachedVisionOracle(OpenAIVisionOracle(max_usd=1.0, client=client), path, "clip")
    r = first.submit_numbers(9, frame).result()
    assert [(n["value"], n["x"], n["y"], n["toward"]) for n in r.numbers] == [
        (30, 200.0, 80.0, "left")
    ]
    first.close()
    second = CachedVisionOracle(OpenAIVisionOracle(max_usd=1.0, client=client), path, "clip")
    assert second.submit_numbers(9, frame).result().numbers == r.numbers
    second.close()
    assert client.calls == 1  # replayed
