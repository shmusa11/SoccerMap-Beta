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
