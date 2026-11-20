import csv
import io
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from soccer_analyzer.workspace_analysis import Settings, analyze_video, positions_csv
from workspace import Handler, Workspace, write_json


@pytest.mark.parametrize(
    "values",
    [
        {"stride": 0},
        {"length": float("nan")},
        {"width": 68.1},
        {"max_frames": 1},
        {"players": "false"},
        {"stride": True},
        {"path": "/tmp"},
        {"start_frame": -1},
        {"start_frame": True},
    ],
)
def test_settings_reject_invalid_values(values):
    with pytest.raises(ValueError):
        Settings.parse(values)


def test_pipeline_exports_source_indices_and_missing_calibration(tmp_path):
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 60, (160, 100))
    assert writer.isOpened()
    for _ in range(12):
        writer.write(np.full((100, 160, 3), (30, 120, 30), np.uint8))
    writer.release()
    original = source.read_bytes()

    def calibrator(*args, **kwargs):
        return {i: None if i == 6 else np.diag([105 / 160, 68 / 100, 1.0]) for i in range(12)}, {
            "frames": 12,
            "covered": 11,
        }

    result = analyze_video(
        source,
        tmp_path,
        Settings(players=False),
        lambda *args: None,
        model_loader=lambda *args: (None, None, None, []),
        calibrator=calibrator,
    )
    assert [f["index"] for f in result["frames"]] == [0, 3, 6, 9]
    assert result["frames"][2]["valid"] is False
    assert result["coverage"] == 0.75
    assert result["duration"] == 0.2
    assert source.read_bytes() == original
    cap = cv2.VideoCapture(str(tmp_path / "result.mp4"))
    try:
        assert cap.read()[0]
        assert cap.get(cv2.CAP_PROP_FPS) == 20
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 4
    finally:
        cap.release()
    rows = list(csv.DictReader(io.StringIO(positions_csv(result))))
    assert rows[2]["source_frame"] == "6"
    assert rows[2]["calibration_valid"] == "False"
    assert rows[2]["x_m"] == rows[2]["y_m"] == ""


def test_releasing_models_preserves_ball_availability(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from soccer_analyzer.match_processor import MatchData

    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 100))
    for _ in range(6):
        writer.write(np.full((100, 160, 3), 100, np.uint8))
    writer.release()
    monkeypatch.setattr(
        "soccer_analyzer.workspace_analysis.load_models",
        lambda *args, detection_models=True, **kwargs: (
            None,
            object() if detection_models else None,
            object() if detection_models else None,
            [],
        ),
    )
    monkeypatch.setattr(
        "soccer_analyzer.match_processor.MatchProcessor",
        lambda *args, **kwargs: SimpleNamespace(
            sweep=lambda *a, **kw: MatchData(frames=[0, 3], ball={3: (80, 50, True)})
        ),
    )
    result = analyze_video(
        source,
        tmp_path,
        Settings(),
        lambda *args: None,
        calibrator=lambda *args, **kwargs: (
            {0: np.diag([105 / 160, 68 / 100, 1.0])},
            {"frames": 1, "covered": 1},
        ),
    )
    assert result["ball_available"] is True
    assert result["frames"][1]["ball_image"] == [80, 50]


def test_later_passage_preserves_source_indices_and_relative_playback_time(tmp_path):
    source = tmp_path / "passage.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 100))
    for index in range(12):
        writer.write(np.full((100, 160, 3), index * 10, np.uint8))
    writer.release()

    def calibrator(*args, **kwargs):
        assert kwargs["start_frame"] == 5 and kwargs["max_frames"] == 7
        return {i: None if i == 8 else np.diag([105 / 160, 68 / 100, 1.0]) for i in range(5, 12)}, {
            "frames": 7,
            "covered": 6,
        }

    result = analyze_video(
        source,
        tmp_path,
        Settings(players=False, start_frame=5),
        lambda *a: None,
        model_loader=lambda *a: (None, None, None, []),
        calibrator=calibrator,
    )
    assert [f["index"] for f in result["frames"]] == [5, 8, 11]
    assert [f["time"] for f in result["frames"]] == [0, 0.1, 0.2]
    assert result["frames"][1]["valid"] is False
    assert result["window"]["end_frame_exclusive"] == 12
    rows = list(csv.DictReader(io.StringIO(positions_csv(result))))
    assert float(rows[0]["timestamp_s"]) == pytest.approx(5 / 30, abs=1e-6)
    assert rows[0]["clip_time_s"] == "0.0"
    with pytest.raises(ValueError, match="past the end"):
        analyze_video(
            source,
            tmp_path,
            Settings(start_frame=12),
            lambda *a: None,
            model_loader=lambda *a: pytest.fail("must reject before loading models"),
        )


@pytest.fixture
def server(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.workspace = Workspace(tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.workspace.close()
    server.shutdown()
    server.server_close()
    thread.join()


def request(server, path, body=None, headers=None):
    url = f"http://127.0.0.1:{server.server_port}{path}"
    head = {"X-SoccerMap": "local", "Content-Type": "application/json", **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    return urllib.request.urlopen(urllib.request.Request(url, data=data, headers=head), timeout=20)


def test_local_boundary_blocks_cross_origin_and_bad_hosts(server):
    for headers in ({"Origin": "https://example.com"}, {"Host": "example.com"}):
        with pytest.raises(urllib.error.HTTPError) as error:
            request(server, "/api/environment", headers=headers)
        assert error.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server, "/api/runs", {"upload": "demo"}, {"X-SoccerMap": ""})
    assert error.value.code == 403


def test_demo_run_downloads_save_and_reopen(server):
    with request(
        server, "/api/runs", {"upload": "demo", "settings": {"max_frames": 60}}
    ) as response:
        run = json.load(response)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with request(server, f"/api/runs/{run['id']}") as response:
            status = json.load(response)
        if status["state"] != "running":
            break
        time.sleep(0.1)
    assert status["state"] == "complete", status
    with request(server, f"/api/runs/{run['id']}/result") as response:
        result = json.load(response)
    assert result["synthetic"] is True
    assert len(result["frames"]) == 20
    with request(
        server, f"/api/runs/{run['id']}/video", headers={"Range": "bytes=10-20"}
    ) as response:
        assert response.status == 206
        assert len(response.read()) == 11
    with request(server, f"/api/runs/{run['id']}/csv") as response:
        assert "synthetic" in response.read().decode().splitlines()[0]
    with request(server, f"/api/runs/{run['id']}/save", {}) as response:
        assert json.load(response)["saved"]
    restored = Workspace(server.workspace.directory)
    assert restored.status(run["id"])["saved"]


def test_cancel_stops_worker_and_does_not_publish_partial_result(server):
    run = server.workspace.start({"upload": "demo", "settings": {"max_frames": 1800}})
    with pytest.raises(ValueError, match="already running"):
        server.workspace.start({"upload": "demo"})
    cancelled = server.workspace.cancel(run["id"])
    assert cancelled["state"] == "cancelled"
    assert not server.workspace.process.is_alive()
    assert not (server.workspace.run_path(run["id"]) / "result.mp4").exists()
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server, f"/api/runs/{run['id']}/video")
    assert error.value.code == 400


def test_restart_marks_unfinished_run_interrupted(tmp_path):
    workspace = Workspace(tmp_path)
    ident = "a" * 32
    folder = workspace.runs / ident
    folder.mkdir()
    write_json(folder / "manifest.json", {"id": ident, "settings": asdict(Settings())})
    write_json(folder / "status.json", {"state": "running"})
    assert Workspace(tmp_path).status(ident)["state"] == "interrupted"


def test_cancelled_run_resumes_in_place_and_completed_runs_are_immutable(server):
    run = server.workspace.start({"upload": "demo", "settings": {"max_frames": 60}})
    server.workspace.cancel(run["id"])
    with request(server, f"/api/runs/{run['id']}/resume", {}) as response:
        assert response.status == 202
        resumed = json.load(response)
    assert resumed["id"] == run["id"] and resumed["attempts"] == 2
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = server.workspace.status(run["id"])
        if status["state"] != "running":
            break
        time.sleep(0.1)
    assert status["state"] == "complete", status
    assert status["resumable"] is False
    with pytest.raises(ValueError, match="stopped"):
        server.workspace.resume(run["id"])


def test_resume_respects_single_worker_limit(server):
    first = server.workspace.start({"upload": "demo", "settings": {"max_frames": 60}})
    server.workspace.cancel(first["id"])
    second = server.workspace.start({"upload": "demo", "settings": {"max_frames": 60}})
    with pytest.raises(ValueError, match="Another analysis"):
        server.workspace.resume(first["id"])
    server.workspace.cancel(second["id"])


def test_coach_tool_modules_are_served_locally(server):
    # The CSP allows self-hosted scripts only, so the 3D engine must come from this server.
    for path, marker in (
        ("/insights.js", "coverageAt"),
        ("/scene3d.js", "Pitch3D"),
        ("/vendor/OrbitControls.js", "./three.module.min.js"),
    ):
        with request(server, path) as response:
            assert marker in response.read().decode()
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server, "/vendor/THREE_LICENSE")
    assert error.value.code == 404


def test_no_arbitrary_file_serving(server):
    with request(server, "/workflow.js") as response:
        assert "passageWindow" in response.read().decode()
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server, "/.env")
    assert error.value.code == 404
    with pytest.raises(ValueError):
        server.workspace.run_path("../../.env")


def test_corrupt_video_upload_is_rejected_and_cleaned(server):
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/api/uploads",
        data=b"this is not a video",
        headers={
            "X-SoccerMap": "local",
            "Content-Type": "application/octet-stream",
            "X-File-Name": "broken.mp4",
        },
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(req)
    assert error.value.code == 400
    assert "readable video" in json.load(error.value)["error"]
    assert list(server.workspace.uploads.iterdir()) == []


def test_linked_video_stays_in_place_and_public_catalog_hides_paths(server, tmp_path):
    source = tmp_path / "match.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 100))
    for value in (40, 80, 120):
        writer.write(np.full((100, 160, 3), value, np.uint8))
    writer.release()
    original = source.read_bytes()
    linked = server.workspace.link_video(source)
    assert linked["storage"] == "linked"
    assert server.workspace.link_video(source)["id"] == linked["id"]
    folder = server.workspace.uploads / linked["id"]
    assert [file.name for file in folder.iterdir()] == ["metadata.json"]
    assert server.workspace.source_path(linked["id"]) == source
    assert server.workspace.source_frame(linked["id"], 2).mean() == pytest.approx(120, abs=2)
    with request(server, "/api/uploads") as response:
        catalog = json.load(response)
    assert len(catalog) == 1 and catalog[0]["available"] is True
    for metadata in (linked, catalog[0]):
        assert "external_path" not in metadata and "source_identity" not in metadata
        assert str(tmp_path) not in json.dumps(metadata)
    assert source.read_bytes() == original

    # Changed source files must never silently reuse an earlier setup/checkpoint.
    source.write_bytes(original + b"changed")
    assert server.workspace.clips()[0]["available"] is False
    with pytest.raises(ValueError, match="changed"):
        server.workspace.start({"upload": linked["id"]})
    source.unlink()
    with pytest.raises(ValueError, match="no longer"):
        server.workspace.source_frame(linked["id"], 0)


def test_browser_cannot_register_arbitrary_local_paths(server):
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server, "/api/uploads/link", {"path": "/etc/passwd"})
    assert error.value.code == 404
    with pytest.raises(ValueError, match="imported clip"):
        server.workspace.source_path("../../.env")


def test_camera_setup_preview_save_and_run_without_field_models(server):
    ident = "b" * 32
    folder = server.workspace.uploads / ident
    folder.mkdir()
    source = folder / "source.avi"
    rng = np.random.default_rng(4)
    frame = rng.integers(0, 255, (450, 640, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 450))
    for _ in range(6):
        writer.write(frame)
    writer.release()
    write_json(
        folder / "metadata.json",
        {
            "stored_name": "source.avi",
            "name": "fixture.avi",
            "frames": 6,
            "width": 640,
            "height": 450,
        },
    )
    body = {
        "upload": ident,
        "frame": 0,
        "name": "Test camera",
        "length": 105,
        "width": 68,
        "points": [
            {"landmark": key, "x": x, "y": y}
            for key, x, y in [
                ("left_top", 50, 50),
                ("right_top", 590, 50),
                ("right_bottom", 590, 400),
                ("left_bottom", 50, 400),
            ]
        ],
    }
    with request(server, f"/api/uploads/{ident}/frame?index=0") as response:
        preview = json.load(response)
        assert preview["width"] == 640 and preview["image"].startswith("data:image/jpeg;base64,")
    with request(server, "/api/cameras/preview", body) as response:
        assert json.load(response)["image"].startswith("data:image/jpeg;base64,")
    with pytest.raises(urllib.error.HTTPError):
        request(server, "/api/cameras", body)
    with request(server, "/api/cameras", {**body, "reviewed": True}) as response:
        camera = json.load(response)
    assert Workspace(server.workspace.directory).camera(camera["id"])["name"] == "Test camera"
    settings = {"players": False, "camera_setup": camera["id"], "max_frames": 30}
    with request(server, "/api/runs", {"upload": ident, "settings": settings}) as response:
        run = json.load(response)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        status = server.workspace.status(run["id"])
        if status["state"] != "running":
            break
        time.sleep(0.1)
    assert status["state"] == "complete", status
    with request(server, f"/api/runs/{run['id']}/result") as response:
        result = json.load(response)
    assert result["calibration"]["inference_calls"] == 0
    assert result["coverage"] == 1
    with pytest.raises(ValueError, match="dimensions"):
        server.workspace.start({"upload": ident, "settings": {**settings, "width": 60}})
    with pytest.raises(urllib.error.HTTPError):
        request(server, f"/api/uploads/{ident}/frame?index=6")


def test_completed_mapping_can_be_reviewed_and_saved_as_a_camera(server):
    ident, upload = "c" * 32, "d" * 32
    source = server.workspace.uploads / upload
    source.mkdir()
    writer = cv2.VideoWriter(
        str(source / "source.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 100)
    )
    writer.write(np.full((100, 160, 3), 100, np.uint8))
    writer.release()
    write_json(source / "metadata.json", {"stored_name": "source.avi", "frames": 1})
    folder = server.workspace.runs / ident
    folder.mkdir()
    write_json(folder / "manifest.json", {"synthetic": False, "upload": upload})
    write_json(folder / "status.json", {"state": "complete"})
    write_json(
        folder / "result.json",
        {
            "settings": {"length": 105, "width": 68},
            "frames": [
                {
                    "index": 0,
                    "valid": True,
                    "homography": np.diag([105 / 160, 68 / 100, 1]).tolist(),
                }
            ],
        },
    )
    with request(server, f"/api/runs/{ident}/camera/preview", {"frame": 0}) as response:
        assert json.load(response)["frame"] == 0
    with pytest.raises(urllib.error.HTTPError):
        request(server, f"/api/runs/{ident}/camera/save", {"frame": 0})
    with request(
        server,
        f"/api/runs/{ident}/camera/save",
        {"frame": 0, "name": "Automatic camera", "reviewed": True},
    ) as response:
        camera = json.load(response)
    assert camera["method"] == "reviewed_run"
    assert camera["width_px"] == 160
    with pytest.raises(urllib.error.HTTPError):
        request(server, f"/api/runs/{ident}/camera/preview", {"frame": 1})


def test_beta_retry_new_id_delete_confirmation_and_input_preservation(server):
    original = server.workspace.start({"upload": "demo", "settings": {"max_frames": 60}})
    server.workspace.cancel(original["id"])
    before = (server.workspace.run_path(original["id"]) / "manifest.json").read_bytes()
    with request(server, f"/api/runs/{original['id']}/retry", {}) as response:
        retry = json.load(response)
    assert retry["id"] != original["id"] and retry["retry_of"] == original["id"]
    assert server.workspace.status(original["id"])["state"] == "cancelled"
    assert (server.workspace.run_path(original["id"]) / "manifest.json").read_bytes() == before
    source_marker = server.workspace.uploads / "preserve.txt"
    source_marker.write_text("original upload")
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server, f"/api/runs/{original['id']}/delete", {"confirmed": False})
    assert error.value.code == 400
    assert server.workspace.run_path(original["id"]).exists()
    with request(server, f"/api/runs/{original['id']}/delete", {"confirmed": True}) as response:
        assert json.load(response)["source_preserved"]
    assert source_marker.read_text() == "original upload"
    assert server.workspace.run_path(retry["id"]).exists()
    server.workspace.cancel(retry["id"])


def test_beta_completed_manifest_media_and_csv_contract(server):
    import csv
    import io

    run = server.workspace.start({"upload": "demo", "settings": {"max_frames": 300, "stride": 3}})
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = server.workspace.status(run["id"])
        if status["state"] != "running":
            break
        time.sleep(0.05)
    assert status["state"] == "complete", status
    assert len(status["input_fingerprint"]["sha256"]) == 64
    assert len(status["code_version"]["value"]) == 64
    assert status["model_identifiers"] == []
    video = status["encoded_video"]
    assert video["width"] > 0 and video["height"] > 0 and video["codec"]
    assert abs(video["frames"] / video["fps"] - 10) <= 1 / video["fps"]
    with request(server, f"/api/runs/{run['id']}/csv") as response:
        rows = list(csv.DictReader(io.StringIO(response.read().decode("utf-8"))))
    expected = {
        "run_id",
        "segment_id",
        "frame_index",
        "timestamp_s",
        "track_id",
        "entity_class",
        "x_m",
        "y_m",
        "calibration_valid",
        "observation_status",
    }
    assert expected <= rows[0].keys()
    assert all(r["run_id"] == run["id"] for r in rows)
    assert all(r["x_m"] == r["y_m"] == "" for r in rows if r["calibration_valid"] == "False")


def wait_for(workspace, ident, timeout=30):
    """Poll one run until it leaves the running state and return its final status."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = workspace.status(ident)
        if status["state"] != "running":
            return status
        time.sleep(0.05)
    return status


def read_log(folder):
    return [json.loads(line) for line in (folder / "run.log").read_text().splitlines()]


def test_run_log_names_run_stage_time_and_error_type_without_secrets(server, monkeypatch):
    from workspace import worker

    marker = "FAKE-SECRET-MARKER-1234"
    monkeypatch.setenv("OPENAI_API_KEY", marker)
    run = server.workspace.start({"upload": "demo", "settings": {"max_frames": 60}})
    assert wait_for(server.workspace, run["id"])["state"] == "complete"
    folder = server.workspace.run_path(run["id"])
    entries = read_log(folder)
    assert entries[-1]["state"] == "complete" and entries[-1]["error_type"] is None
    assert {e["run_id"] for e in entries} == {run["id"]}
    assert all(e["stage"] and e["elapsed_s"] >= 0 for e in entries)

    # A failed run records the exception type only, never its message or the source path.
    failed = server.workspace.runs / ("c" * 32)
    failed.mkdir()
    write_json(failed / "manifest.json", {"id": failed.name, "created": time.time()})
    missing = server.workspace.directory / "private-clip-name.mp4"
    worker(str(failed), str(missing), asdict(Settings(engine="classic")))
    last = read_log(failed)[-1]
    assert last["state"] == "failed" and last["error_type"] == "RuntimeError"
    assert json.loads((failed / "status.json").read_text())["state"] == "failed"
    for path in (folder / "run.log", failed / "run.log"):
        text = path.read_text()
        assert marker not in text and "private-clip-name" not in text
    with request(server, f"/api/runs/{failed.name}/log") as response:
        assert failed.name[:8] in response.headers["Content-Disposition"]


def test_cancelled_and_interrupted_runs_are_logged_with_their_stage(server):
    run = server.workspace.start({"upload": "demo", "settings": {"max_frames": 1800}})
    server.workspace.cancel(run["id"])
    assert read_log(server.workspace.run_path(run["id"]))[-1]["state"] == "cancelled"
    ident = "d" * 32
    folder = server.workspace.runs / ident
    folder.mkdir()
    write_json(folder / "manifest.json", {"id": ident, "created": time.time()})
    write_json(folder / "status.json", {"state": "running", "stage": "Rendering"})
    restored = Workspace(server.workspace.directory)
    assert restored.status(ident)["state"] == "interrupted"
    assert "new run" in restored.status(ident)["stage"]
    assert read_log(folder)[-1]["state"] == "interrupted"


def test_missing_required_model_is_named_before_any_run_starts(server, tmp_path, monkeypatch):
    source = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 100))
    for _ in range(40):
        writer.write(np.full((100, 160, 3), 90, np.uint8))
    writer.release()
    linked = server.workspace.link_video(source)
    models = tmp_path / "models"
    models.mkdir()
    (models / "SV_kp").write_bytes(b"weights")
    monkeypatch.setenv("PNLCALIB_ROOT", str(models))
    with pytest.raises(ValueError, match="missing: Field lines"):
        server.workspace.start({"upload": linked["id"]})
    assert list(server.workspace.runs.iterdir()) == []
    with request(server, "/api/environment") as response:
        report = json.load(response)
    assert report["ready"] is False
    assert {"os", "cpu", "ram_bytes", "free_disk_bytes", "python", "qualified"} <= set(
        report["system"]
    )


def test_downloads_are_named_after_their_run(server):
    run = server.workspace.start({"upload": "demo", "settings": {"max_frames": 60}})
    assert wait_for(server.workspace, run["id"])["state"] == "complete"
    for action, name in (("video", "result.mp4"), ("csv", "positions.csv")):
        with request(server, f"/api/runs/{run['id']}/{action}?download=1") as response:
            assert (
                response.headers["Content-Disposition"]
                == f'attachment; filename="soccermap-{run["id"][:8]}-{name}"'
            )


def test_csv_keeps_five_classes_and_leaves_unresolved_observations_unknown():
    players = [
        {"id": 1, "team": 0, "kind": "player", "x": 0.0, "y": 0.0},
        {"id": 2, "team": 1, "kind": "player", "x": 20.0, "y": 30.0},
        {"id": 3, "team": -2, "kind": "referee", "x": 40.0, "y": 30.0},
        {"id": 4, "team": -3, "kind": "goalkeeper", "x": 5.0, "y": 34.0},
        {"id": 5, "team": -1, "kind": "player", "x": 60.0, "y": 10.0},
    ]
    frames = [
        {"index": 0, "time": 0.0, "valid": True, "players": players, "ball": [50.0, 34.0]},
        {"index": 3, "time": 0.1, "valid": False, "players": players, "ball": None},
    ]
    text = positions_csv({"frames": frames, "synthetic": True, "run_id": "e" * 32})
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [r["entity_class"] for r in rows[:6]] == [
        "team_a",
        "team_b",
        "referee",
        "goalkeeper",
        "unknown",
        "ball",
    ]
    # A real observation at the corner keeps its zero; an unmapped frame stays blank.
    assert (rows[0]["x_m"], rows[0]["y_m"]) == ("0.0", "0.0")
    assert all(r["x_m"] == r["y_m"] == "" for r in rows[6:])
    assert {r["observation_status"] for r in rows[6:]} == {"unavailable"}


def test_unreadable_field_weights_fail_with_a_next_step():
    from soccer_analyzer.workspace_analysis import load_field_models

    class Broken:
        def _ensure_loaded(self):
            raise EOFError("Ran out of input")

    with pytest.raises(ValueError, match="Open Environment to check the model files"):
        load_field_models(Broken())


def test_failure_while_publishing_never_marks_the_run_complete(server, monkeypatch):
    import workspace

    earlier = server.workspace.start({"upload": "demo", "settings": {"max_frames": 60}})
    assert wait_for(server.workspace, earlier["id"])["state"] == "complete"
    before = {p.name: p.read_bytes() for p in server.workspace.run_path(earlier["id"]).iterdir()}
    folder = server.workspace.runs / ("f" * 32)
    folder.mkdir()
    write_json(folder / "manifest.json", {"id": folder.name, "created": time.time()})

    def interrupted(result):
        raise OSError("disk write failed")

    # The video is already rendered when the final result files are written.
    monkeypatch.setattr(workspace, "positions_csv", interrupted)
    workspace.worker(str(folder), None, asdict(Settings(max_frames=60)))
    status = json.loads((folder / "status.json").read_text())
    assert status["state"] == "failed" and not (folder / "result.json").exists()
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server, f"/api/runs/{folder.name}/video")
    assert error.value.code == 400
    after = {p.name: p.read_bytes() for p in server.workspace.run_path(earlier["id"]).iterdir()}
    assert after == before


def test_missing_ball_model_keeps_players_and_states_the_reason(monkeypatch):
    from soccer_analyzer import ball_wasb, detection, resources
    from soccer_analyzer.workspace_analysis import load_models

    def missing(**kwargs):
        raise FileNotFoundError("wasb_soccer_best.pth.tar")

    monkeypatch.setattr(resources, "configure_inference", lambda low_memory: "cpu")
    monkeypatch.setattr(detection, "SoccerDetector", lambda **kwargs: "people-detector")
    monkeypatch.setattr(ball_wasb, "WASBBallDetector", missing)
    field, player, ball, warnings = load_models(Settings(), lambda *a: None, field_models=False)
    assert (field, player, ball) == (None, "people-detector", None)
    assert warnings == [
        "Ball model unavailable. This run contains player observations only; check Environment for setup."
    ]


def upload_clip(server, tmp_path):
    """Import a small generated clip as a stored copy and return its metadata."""
    source = tmp_path / "copy.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 100))
    for _ in range(40):
        writer.write(np.full((100, 160, 3), 90, np.uint8))
    writer.release()
    req = urllib.request.Request(
        f"http://127.0.0.1:{server.server_port}/api/uploads",
        data=source.read_bytes(),
        headers={
            "X-SoccerMap": "local",
            "Content-Type": "application/octet-stream",
            "X-File-Name": "copy.avi",
        },
    )
    with urllib.request.urlopen(req) as response:
        return json.load(response)


def test_start_and_retry_are_blocked_when_the_imported_clip_is_gone(server, tmp_path, monkeypatch):
    models = tmp_path / "models"
    models.mkdir()
    for name in ("SV_kp", "SV_lines"):
        (models / name).write_bytes(b"weights")
    monkeypatch.setenv("PNLCALIB_ROOT", str(models))
    clip = upload_clip(server, tmp_path)
    (server.workspace.uploads / clip["id"] / clip["stored_name"]).unlink()
    with pytest.raises(ValueError, match="source clip is missing. Import it again"):
        server.workspace.start({"upload": clip["id"]})
    assert list(server.workspace.runs.iterdir()) == []


def test_next_run_can_start_as_soon_as_the_previous_one_is_complete(server):
    for _ in range(3):
        run = server.workspace.start({"upload": "demo", "settings": {"max_frames": 30}})
        assert wait_for(server.workspace, run["id"])["state"] == "complete"
    assert len(list(server.workspace.runs.iterdir())) == 3
