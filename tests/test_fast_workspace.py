"""The fast-engine adapter converts pipeline output to the workspace format (no models)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from soccer_analyzer.calibration.pitch_model import PitchModel
from soccer_analyzer.fast_workspace import frame_record
from soccer_analyzer.workspace_analysis import Settings, analyze_video


def fake_result():
    # Image pixels are 10x the pitch metres, so projections are easy to check.
    H = np.diag([0.1, 0.1, 1.0])
    md = SimpleNamespace(
        people={
            20: [
                (1, (100, 100, 120, 300), (110, 300)),  # player at (11, 30) m
                (2, (500, 100, 520, 200), (510, 200)),  # goalkeeper
                (3, (0, 0, 10, 10), (5, 10)),  # bystander, hidden by the on-field filter
                (4, (900, 200, 920, 400), (910, 400)),  # interpolated referee
            ],
            22: [(1, (100, 100, 120, 300), (110, 300))],
        },
        team={1: 0, 2: 1, 3: 1, 4: -1},
        kind={1: "player", 2: "goalkeeper", 3: "player", 4: "referee"},
        ball={20: (300.0, 340.0, 0.9), 22: (2000.0, 50.0, 0.01)},
        interpolated={(20, 4)},
    )
    return SimpleNamespace(
        md=md,
        calib={20: H, 22: None},
        visible={20: {1, 2, 4}, 22: {1}},
        offfield=set(),
        pitch_xy={20: {1: (11.0, 30.0)}},
        render_frames=[20, 22],
        fps=10.0,
    )


def test_frame_record_maps_tracks_teams_provenance_and_ball():
    res = fake_result()
    record = frame_record(res.md, res, 20, PitchModel(105, 68), ball_gate=0.05)
    assert record["valid"] and record["time"] == 0 and record["source_time"] == 2.0
    players = {p["id"]: p for p in record["players"]}
    assert set(players) == {1, 2, 4}  # the bystander is not reported
    assert (players[1]["x"], players[1]["y"]) == (11.0, 30.0)
    assert players[2]["team"] == -3 and players[2]["kind"] == "goalkeeper"
    assert players[4]["team"] == -2 and players[4]["measured"] is False
    assert players[1]["measured"] is True
    assert record["ball"] == pytest.approx([30.0, 34.0])
    assert record["ball_measured"] is True
    assert np.allclose(record["homography"], res.calib[20])


def test_unmapped_frames_keep_tracks_but_no_positions():
    res = fake_result()
    record = frame_record(res.md, res, 22, PitchModel(105, 68), ball_gate=0.05)
    assert not record["valid"] and record["homography"] is None
    assert record["time"] == pytest.approx(0.2)
    assert record["players"][0]["x"] is None
    assert record["ball"] is None and record["ball_measured"] is False


def test_fast_engine_is_the_default_and_classic_stays_available(monkeypatch, tmp_path):
    assert Settings().engine == "fast"
    with pytest.raises(ValueError, match="engine"):
        Settings.parse({"engine": "turbo"})
    calls = []
    monkeypatch.setattr(
        "soccer_analyzer.fast_workspace.analyze_video_fast",
        lambda *args: calls.append(args) or {"engine": "fast"},
    )
    assert analyze_video(Path("clip.mp4"), tmp_path, Settings(), lambda *a: None) == {
        "engine": "fast"
    }
    assert len(calls) == 1


def test_cloud_frames_require_explicit_consent_even_when_key_exists(monkeypatch):
    from soccer_analyzer.fast_workspace import make_oracle

    monkeypatch.setenv("OPENAI_API_KEY", "test-" + "x" * 24)
    calls = []
    monkeypatch.setattr(
        "soccer_analyzer.vision_oracle.OpenAIVisionOracle",
        lambda **kwargs: calls.append(kwargs) or object(),
    )
    assert make_oracle(Settings(), []) is None
    assert not calls
    assert make_oracle(Settings(ai_keyframes=True), []) is not None
    assert calls == [{"max_usd": 0.30}]
    with pytest.raises(ValueError, match="ai_keyframes"):
        Settings.parse({"ai_keyframes": "false"})


def test_requested_cloud_without_key_falls_back_to_local(monkeypatch):
    from soccer_analyzer.fast_workspace import make_oracle

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("soccer_analyzer.fast_workspace.load_dotenv", lambda *a: None)
    warnings = []
    assert make_oracle(Settings(ai_keyframes=True), warnings) is None
    assert warnings == ["AI keyframes requested but no API key is configured; ran locally."]


def test_camera_motion_break_scopes_the_same_tracker_id():
    res = fake_result()
    res.motion = {20: None, 22: None}
    a = frame_record(res.md, res, 20, PitchModel(105, 68), 0.05)
    b = frame_record(res.md, res, 22, PitchModel(105, 68), 0.05)
    assert a["segment_id"] == 0 and b["segment_id"] == 1
