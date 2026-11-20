"""Recovery checks use generated footage and injected observations, never neural weights."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from soccer_analyzer.checkpoints import (
    CheckpointStore,
    analysis_signature,
    pack_match,
    unpack_match,
)
from soccer_analyzer.match_processor import MatchData
from soccer_analyzer.workspace_analysis import Settings, analyze_video


def source_video(tmp_path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 100))
    for index in range(12):
        writer.write(np.full((100, 160, 3), index * 10, np.uint8))
    writer.release()
    return path


def calibration(*args, **kwargs):
    return {i: None if i == 6 else np.diag([105 / 160, 68 / 100, 1.0]) for i in range(12)}, {
        "frames": 12,
        "covered": 11,
        "inference_calls": 12,
    }


def test_checkpoint_rejects_stale_corrupt_and_partial_files(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path, "input-a")
    assert store.read("calibration") is None
    store.write("calibration", {"samples": [[3, None]]})
    assert store.read("calibration")["samples"] == [[3, None]]
    assert CheckpointStore(tmp_path, "input-b").read("calibration") is None
    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", Mock(side_effect=OSError("disk failure")))
        with pytest.raises(OSError):
            store.write("calibration", {"samples": []})
    assert store.read("calibration")["samples"] == [[3, None]]
    path = tmp_path / "calibration.checkpoint.json"
    document = json.loads(path.read_text())
    document["payload"]["samples"] = []
    path.write_text(json.dumps(document))
    assert store.read("calibration") is None  # Digest mismatch even though JSON parses.
    path.write_text('{"schema":')
    assert store.read("calibration") is None


def test_signature_changes_with_source_settings_camera_and_weights(tmp_path, monkeypatch):
    source = source_video(tmp_path)
    monkeypatch.setenv("WASB_ROOT", str(tmp_path))
    settings = Settings()
    reference = (np.zeros((2, 2, 3), np.uint8), np.eye(3))
    original = analysis_signature(source, settings, reference)
    assert original == analysis_signature(source, settings, reference)
    assert original != analysis_signature(source, replace(settings, stride=4), reference)
    moved = (reference[0].copy(), reference[1])
    moved[0][0, 0] = 1
    assert original != analysis_signature(source, settings, moved)
    (tmp_path / "wasb_soccer_best.pth.tar").write_bytes(b"weight fixture")
    assert original != analysis_signature(source, settings, reference)
    original = analysis_signature(source, settings, reference)
    source.write_bytes(source.read_bytes())
    assert original != analysis_signature(source, settings, reference)


def test_match_checkpoint_keeps_raw_indices_and_provenance(tmp_path):
    match = MatchData(
        frames=[12, 15],
        people={12: [(7, (np.float32(1), 2, 3, 4), (np.float32(2), 4))]},
        team={7: 0},
        kind={7: "player"},
        ball={15: (12.5, 20.5, False)},
        team_centers=np.ones((2, 3)),
        interpolated={(12, 7)},
        cuts=[15],
        stats={"processed": 2},
    )
    store = CheckpointStore(tmp_path, "fixture")
    store.write("tracking", {"match": pack_match(match)})
    restored = unpack_match(store.read("tracking")["match"])
    assert restored.frames == [12, 15]
    assert restored.people == match.people
    assert restored.ball == match.ball
    assert restored.interpolated == {(12, 7)} and restored.cuts == [15]
    np.testing.assert_array_equal(restored.team_centers, match.team_centers)


def test_render_retry_reuses_stages_without_loading_models(tmp_path, monkeypatch):
    source = source_video(tmp_path)
    calibrator = Mock(side_effect=calibration)
    loader = Mock(return_value=(None, None, None, []))
    with monkeypatch.context() as patch:
        patch.setattr(
            "soccer_analyzer.workspace_analysis.open_writer",
            Mock(side_effect=OSError("encoder failed")),
        )
        with pytest.raises(OSError, match="encoder"):
            analyze_video(
                source,
                tmp_path,
                Settings(players=False),
                lambda *a: None,
                model_loader=loader,
                calibrator=calibrator,
            )
    loader.reset_mock()
    result = analyze_video(
        source,
        tmp_path,
        Settings(players=False),
        lambda *a: None,
        model_loader=loader,
        calibrator=calibrator,
    )
    loader.assert_not_called()
    assert calibrator.call_count == 1
    assert result["reused_stages"] == ["calibration", "tracking"]
    assert result["calibration"]["inference_calls"] == 0
    assert result["calibration"]["cached_inference_calls"] == 12
    assert result["timings"]["models"] == result["timings"]["calibration"] == 0
    assert [f["index"] for f in result["frames"]] == [0, 3, 6, 9]
    assert result["frames"][2]["valid"] is False


def test_tracking_failure_reuses_only_finished_calibration(tmp_path, monkeypatch):
    source = source_video(tmp_path)
    calibrator = Mock(side_effect=calibration)
    loader = Mock(return_value=(None, None, None, []))
    with monkeypatch.context() as patch:
        patch.setattr(
            "soccer_analyzer.workspace_analysis._track_video", Mock(side_effect=MemoryError("stop"))
        )
        with pytest.raises(MemoryError):
            analyze_video(
                source,
                tmp_path,
                Settings(players=False),
                lambda *a: None,
                model_loader=loader,
                calibrator=calibrator,
            )
    assert not (tmp_path / "tracking.checkpoint.json").exists()
    result = analyze_video(
        source,
        tmp_path,
        Settings(players=False),
        lambda *a: None,
        model_loader=loader,
        calibrator=calibrator,
    )
    assert calibrator.call_count == 1
    assert result["reused_stages"] == ["calibration"]
    # A changed configuration must recompute both stages, preserving sample timing.
    result = analyze_video(
        source,
        tmp_path,
        Settings(players=False, stride=4),
        lambda *a: None,
        model_loader=loader,
        calibrator=calibrator,
    )
    assert calibrator.call_count == 2 and result["reused_stages"] == []
    assert [f["index"] for f in result["frames"]] == [0, 4, 8]
