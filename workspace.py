"""Launch the local SoccerMap workspace: python workspace.py [--port 8765]."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import multiprocessing
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

from dotenv import load_dotenv

from soccer_analyzer.workspace_analysis import (
    Settings,
    analyze_video,
    environment,
    inspect_video,
    make_demo,
    positions_csv,
)
from soccer_analyzer.calibration.manual import landmarks, solve_manual
from soccer_analyzer.calibration.pitch_model import PitchModel
from soccer_analyzer.calibration.mapper import FrameCalibration
from soccer_analyzer.calibration.visualize import draw_calibration_overlay
from soccer_analyzer.video import open_capture
from soccer_analyzer.resources import check_memory

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD = 500 * 1024 * 1024
IDENTIFIER = re.compile(r"^[a-f0-9]{32}$")


def encode_image(frame: np.ndarray) -> str:
    """Return a browser preview, with native pixel dimensions preserved for point placement."""
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ValueError("The frame preview could not be created.")
    return "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")


def write_json(path: Path, data: dict) -> None:
    """Replace atomically so polling never sees half-written JSON."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, allow_nan=False))
    temporary.replace(path)


def worker(directory: str, source: str | None, config: dict) -> None:
    """One process per run isolates external imports and allows prompt cancellation."""
    folder = Path(directory)
    status_lock = threading.Lock()
    finished = threading.Event()

    def watch_memory():
        import psutil

        process = psutil.Process()
        while not finished.wait(1):
            try:
                check_memory(process.memory_info().rss, psutil.virtual_memory().available)
            except MemoryError as exc:
                with status_lock:
                    if finished.is_set():
                        return
                    write_json(
                        folder / "status.json",
                        {
                            "state": "failed",
                            "stage": "Memory limit reached",
                            "error": str(exc),
                            "progress": 0,
                        },
                    )
                # Exiting the isolated worker also releases native-model allocations.
                os._exit(1)

    def report(stage, progress):
        with status_lock:
            write_json(
                folder / "status.json",
                {"state": "running", "stage": stage, "progress": round(progress, 4)},
            )

    try:
        started = time.monotonic()
        settings = Settings.parse(config)
        if source:
            import psutil

            check_memory(psutil.Process().memory_info().rss, psutil.virtual_memory().available)
            threading.Thread(target=watch_memory, daemon=True).start()
        reference = None
        if settings.camera_setup is not None:
            setup = json.loads((folder / "camera.json").read_text())
            image = cv2.imread(str(folder / "camera.png"))
            if image is None:
                raise ValueError("The camera reference image is missing. Save the setup again.")
            reference = (image, np.array(setup["homography"], dtype=np.float64))
        result = (
            analyze_video(Path(source), folder, settings, report, camera_reference=reference)
            if source
            else make_demo(folder, settings, report)
        )
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (folder / "positions.csv").write_text(positions_csv(result))
        write_json(folder / "result.json", result)
        finished.set()
        with status_lock:
            write_json(
                folder / "status.json",
                {"state": "complete", "stage": "Ready to review", "progress": 1},
            )
    except Exception as exc:
        finished.set()
        with status_lock:
            write_json(
                folder / "status.json",
                {"state": "failed", "stage": "Analysis stopped", "error": str(exc), "progress": 0},
            )


class Workspace:
    """Manage local uploads and completed analyses. Only one model worker runs at once."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.uploads = directory / "uploads"
        self.runs = directory / "runs"
        self.cameras = directory / "cameras"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.runs.mkdir(parents=True, exist_ok=True)
        self.cameras.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.process = None
        self.active = None
        for folder in self.runs.iterdir():
            status_file = folder / "status.json"
            if status_file.exists() and json.loads(status_file.read_text())["state"] == "running":
                write_json(
                    status_file,
                    {
                        "state": "interrupted",
                        "stage": "Server was stopped. Resume to reuse completed stages.",
                        "progress": 0,
                    },
                )

    def run_path(self, ident: str) -> Path:
        if not IDENTIFIER.fullmatch(ident):
            raise ValueError("Invalid analysis identifier.")
        path = self.runs / ident
        if not (path / "manifest.json").is_file():
            raise FileNotFoundError("This analysis is no longer available.")
        return path

    def upload_metadata(self, ident: str) -> dict:
        """Resolve an opaque clip ID; browser requests cannot submit filesystem paths."""
        if not isinstance(ident, str) or not IDENTIFIER.fullmatch(ident):
            raise ValueError("Choose an imported clip.")
        return json.loads((self.uploads / ident / "metadata.json").read_text())

    def source_path(self, ident: str) -> Path:
        """Read an uploaded copy or a CLI-linked file, checking linked files for changes."""
        from soccer_analyzer.checkpoints import file_identity

        metadata = self.upload_metadata(ident)
        if "external_path" in metadata:
            path = Path(metadata["external_path"])
            if not path.is_file():
                raise ValueError(
                    "This linked video is no longer at its original location. Link it again."
                )
            if file_identity(path) != metadata.get("source_identity"):
                raise ValueError(
                    "This linked video changed since it was imported. Link it again before processing."
                )
            return path
        stored = metadata["stored_name"]
        if Path(stored).name != stored:
            raise ValueError("The imported clip metadata is invalid. Import the video again.")
        return self.uploads / ident / stored

    def clips(self) -> list[dict]:
        """List clip metadata without exposing linked filesystem paths or fingerprints."""
        clips = []
        for folder in self.uploads.iterdir():
            try:
                metadata = self.upload_metadata(folder.name)
                public = {
                    k: v
                    for k, v in metadata.items()
                    if k not in {"external_path", "source_identity", "stored_name"}
                }
                try:
                    public["available"] = self.source_path(folder.name).is_file()
                except (ValueError, FileNotFoundError):
                    public["available"] = False
                clips.append(public)
            except (ValueError, KeyError, FileNotFoundError):
                continue  # Interrupted uploads are not usable catalogue entries.
        return sorted(clips, key=lambda item: item.get("created", 0), reverse=True)

    def link_video(self, path: Path) -> dict:
        """CLI-only registration of an existing video: no copy, modification or model loading."""
        from soccer_analyzer.checkpoints import file_identity

        path = path.expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mov", ".avi", ".m4v"}:
            raise ValueError("Link an existing MP4, MOV, AVI or M4V file.")
        identity = file_identity(path)
        for folder in self.uploads.iterdir():
            try:
                metadata = self.upload_metadata(folder.name)
                if metadata.get("source_identity") == identity:
                    return {
                        k: v
                        for k, v in metadata.items()
                        if k not in {"external_path", "source_identity"}
                    }
            except (ValueError, FileNotFoundError):
                continue
        video_metadata = inspect_video(path)
        ident = uuid.uuid4().hex
        folder = self.uploads / ident
        folder.mkdir()
        metadata = {
            "id": ident,
            "name": path.name,
            "size": identity["size"],
            "created": time.time(),
            "storage": "linked",
            "external_path": str(path),
            "source_identity": identity,
            **video_metadata,
        }
        try:
            write_json(folder / "metadata.json", metadata)
        except Exception:
            shutil.rmtree(folder)
            raise
        return {k: v for k, v in metadata.items() if k not in {"external_path", "source_identity"}}

    def status(self, ident: str) -> dict:
        with self.lock:
            folder = self.run_path(ident)
            status = json.loads((folder / "status.json").read_text())
            if (
                ident == self.active
                and self.process
                and not self.process.is_alive()
                and status["state"] == "running"
            ):
                status = {
                    "state": "failed",
                    "stage": "Worker stopped unexpectedly. Try a shorter clip.",
                    "progress": 0,
                }
                write_json(folder / "status.json", status)
            return {
                **json.loads((folder / "manifest.json").read_text()),
                **status,
                "updated": (folder / "status.json").stat().st_mtime,
                "resumable": status["state"] in {"failed", "cancelled", "interrupted"},
            }

    def start(self, request: dict) -> dict:
        settings = Settings.parse(request.get("settings", {}))
        source = None
        name = "Riverside training · synthetic demo"
        if request.get("upload") != "demo":
            ident = request.get("upload", "")
            if not IDENTIFIER.fullmatch(str(ident)):
                raise ValueError("Choose a clip before starting.")
            metadata = self.upload_metadata(ident)
            source = str(self.source_path(ident))
            name = metadata["name"]
            from soccer_analyzer.video import frame_window

            frame_window(metadata.get("frames", 0), settings.max_frames, settings.start_frame)
            if settings.camera_setup is None and not environment()["ready"]:
                raise ValueError(
                    "Required field models are missing. Open Environment for the file locations, or use the synthetic demo."
                )
        elif settings.start_frame:
            raise ValueError(
                "The synthetic demo starts at frame 0. Select a later passage in real footage."
            )
        elif settings.camera_setup is not None:
            raise ValueError(
                "Saved camera setups apply to real footage. Choose automatic for the demo."
            )
        setup = None
        if settings.camera_setup is not None:
            setup = self.camera(settings.camera_setup)
            if (settings.length, settings.width) != (setup["length"], setup["width"]):
                raise ValueError("Field dimensions must match the saved camera setup.")
            if (metadata["width"], metadata["height"]) != (setup["width_px"], setup["height_px"]):
                raise ValueError(
                    "This clip has a different image size. Place four points for this resolution."
                )
        with self.lock:
            if self.process and self.process.is_alive():
                raise ValueError(
                    "An analysis is already running. Review its progress or cancel it first."
                )
            ident = uuid.uuid4().hex
            folder = self.runs / ident
            folder.mkdir()
            if setup is not None:
                write_json(folder / "camera.json", setup)
                shutil.copyfile(
                    self.cameras / settings.camera_setup / "reference.png", folder / "camera.png"
                )
            manifest = {
                "id": ident,
                "name": name,
                "synthetic": source is None,
                "created": time.time(),
                "saved": False,
                "settings": asdict(settings),
                "camera_name": setup["name"] if setup else None,
                "upload": request.get("upload"),
                "attempts": 1,
            }
            write_json(folder / "manifest.json", manifest)
            return self._launch(folder, source, settings)

    def _launch(self, folder: Path, source: str | None, settings: Settings) -> dict:
        """Launch under the workspace lock; a failed spawn must not leave a running status."""
        write_json(
            folder / "status.json",
            {
                "state": "running",
                "stage": "Starting local worker",
                "progress": 0,
            },
        )
        self.process = multiprocessing.get_context("spawn").Process(
            target=worker, args=(str(folder), source, asdict(settings)), daemon=True
        )
        self.active = folder.name
        try:
            self.process.start()
        except Exception:
            self.process = None
            write_json(
                folder / "status.json",
                {
                    "state": "failed",
                    "stage": "The local worker could not start. Try resuming.",
                    "progress": 0,
                },
            )
            raise
        return self.status(folder.name)

    def resume(self, ident: str) -> dict:
        """Retry one stopped run with its immutable settings and validated stage checkpoints."""
        with self.lock:
            status = self.status(ident)
            if not status["resumable"]:
                raise ValueError("Only a stopped analysis can be resumed.")
            if self.process and self.process.is_alive():
                raise ValueError(
                    "Another analysis is running. Finish or cancel it before resuming."
                )
            folder = self.run_path(ident)
            manifest = json.loads((folder / "manifest.json").read_text())
            settings = Settings.parse(manifest["settings"])
            source = None
            if not manifest["synthetic"]:
                upload = manifest.get("upload")
                if not isinstance(upload, str) or not IDENTIFIER.fullmatch(upload):
                    raise ValueError(
                        "This older run has no source reference. Import its clip again."
                    )
                path = self.source_path(upload)
                if not path.is_file():
                    raise ValueError("The source clip is missing. Import it again before retrying.")
                source = str(path)
            for name in ("result.mp4", "review.mp4", "result.json", "positions.csv"):
                (folder / name).unlink(missing_ok=True)
            manifest["attempts"] = manifest.get("attempts", 1) + 1
            manifest["attempt_started"] = time.time()
            write_json(folder / "manifest.json", manifest)
            return self._launch(folder, source, settings)

    def camera(self, ident: str) -> dict:
        """Return a saved setup by opaque ID; reference images never leave local storage."""
        if not isinstance(ident, str) or not IDENTIFIER.fullmatch(ident):
            raise ValueError("Choose a saved camera setup.")
        return json.loads((self.cameras / ident / "setup.json").read_text())

    def source_frame(self, ident: str, index: int) -> np.ndarray:
        """Decode an original video frame for manual setup, without loading any model."""
        if not isinstance(ident, str) or not IDENTIFIER.fullmatch(ident):
            raise ValueError("Choose a real clip before placing points.")
        metadata = self.upload_metadata(ident)
        if type(index) is not int or index < 0 or index >= max(1, metadata["frames"]):
            raise ValueError("Choose a frame inside this clip.")
        from soccer_analyzer.video import seek_capture

        with open_capture(self.source_path(ident)) as cap:
            seek_capture(cap, index)
            ok, image = cap.read()
        if not ok:
            raise ValueError("This frame could not be read. Try another frame.")
        return image

    def prepare_camera(self, request: dict) -> tuple[dict, np.ndarray, np.ndarray]:
        """Validate all correspondences before previewing or persisting a setup."""
        settings = Settings.parse({k: request[k] for k in ("length", "width") if k in request})
        image = self.source_frame(request.get("upload"), request.get("frame", 0))
        height, width = image.shape[:2]
        matrix = solve_manual(
            request.get("points"), PitchModel(settings.length, settings.width), (width, height)
        )
        name = request.get("name", "Camera setup")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError("Name this camera setup using 1 to 80 characters.")
        setup = {
            "name": name.strip(),
            "length": settings.length,
            "width": settings.width,
            "width_px": width,
            "height_px": height,
            "source_frame": request.get("frame", 0),
            "points": request["points"],
            "homography": matrix.tolist(),
            "created": time.time(),
            "method": "manual_four_points",
        }
        return setup, image, matrix

    def save_camera(self, request: dict) -> dict:
        setup, image, _ = self.prepare_camera(request)
        if request.get("reviewed") is not True:
            raise ValueError("Review the projected lines before saving this camera setup.")
        return self.store_camera(setup, image)

    def store_camera(self, setup: dict, image: np.ndarray) -> dict:
        """Persist an immutable, reviewed setup and its original reference image together."""
        ident = uuid.uuid4().hex
        folder = self.cameras / ident
        folder.mkdir()
        try:
            if not cv2.imwrite(str(folder / "reference.png"), image):
                raise ValueError("The reference frame could not be saved. Check disk space.")
            setup["id"] = ident
            write_json(folder / "setup.json", setup)
        except Exception:
            shutil.rmtree(folder)
            raise
        return setup

    def camera_from_run(self, ident: str, request: dict) -> tuple[dict, np.ndarray, np.ndarray]:
        """Retrieve a valid mapped source frame for review and reuse in another clip."""
        status = self.status(ident)
        if status["state"] != "complete" or status["synthetic"] or not status.get("upload"):
            raise ValueError("Choose a completed analysis of real footage to save its camera view.")
        result = json.loads((self.run_path(ident) / "result.json").read_text())
        frame = next((f for f in result["frames"] if f["index"] == request.get("frame")), None)
        if frame is None or not frame["valid"] or frame.get("homography") is None:
            raise ValueError(
                "This frame has no verified field mapping. Choose another view or place four points."
            )
        name = request.get("name", "Reviewed camera view")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError("Name this camera setup using 1 to 80 characters.")
        image = self.source_frame(status["upload"], frame["index"])
        setup = {
            "name": name.strip(),
            "length": result["settings"]["length"],
            "width": result["settings"]["width"],
            "width_px": image.shape[1],
            "height_px": image.shape[0],
            "source_frame": frame["index"],
            "points": [],
            "homography": frame["homography"],
            "created": time.time(),
            "method": "reviewed_run",
        }
        return setup, image, np.asarray(frame["homography"], np.float64)

    def cancel(self, ident: str) -> dict:
        with self.lock:
            status = self.status(ident)
            if status["state"] != "running":
                return status
            if ident != self.active or self.process is None:
                raise ValueError("This analysis is not running.")
            self.process.terminate()
            self.process.join(timeout=3)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=3)
            # A cancelled export is never presented as a completed result.
            folder = self.run_path(ident)
            for name in ("result.mp4", "review.mp4", "result.json", "positions.csv", "status.tmp"):
                (folder / name).unlink(missing_ok=True)
            write_json(
                folder / "status.json",
                {
                    "state": "cancelled",
                    "stage": "Cancelled. Your source clip is unchanged; you can retry.",
                    "progress": 0,
                },
            )
            return self.status(ident)

    def close(self):
        if self.active and self.process and self.process.is_alive():
            self.cancel(self.active)


class Handler(BaseHTTPRequestHandler):
    server_version = "SoccerMapLocal/1.0"

    @property
    def workspace(self) -> Workspace:
        return self.server.workspace

    def log_message(self, format, *args):
        # Polling is intentionally quiet; no source filenames or request bodies in logs.
        pass

    def local_request(self) -> bool:
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in allowed or (
            origin and origin not in {f"http://{host}" for host in allowed}
        ):
            self.respond({"error": "Use this workspace from its localhost address."}, 403)
            return False
        if self.command == "POST" and self.headers.get("X-SoccerMap") != "local":
            self.respond({"error": "Missing local request header."}, 403)
            return False
        return True

    def respond(self, value, status=200):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        size = int(self.headers.get("Content-Length", 0))
        if not 0 < size < 32768 or self.headers.get_content_type() != "application/json":
            raise ValueError("Expected a small JSON request.")
        body = json.loads(self.rfile.read(size))
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON object.")
        return body

    def do_GET(self):
        if not self.local_request():
            return
        path = urlparse(self.path).path
        try:
            if path == "/api/uploads":
                return self.respond(self.workspace.clips())
            if path == "/api/cameras":
                return self.respond(
                    [
                        self.workspace.camera(p.name)
                        for p in self.workspace.cameras.iterdir()
                        if (p / "setup.json").is_file()
                    ]
                )
            if path == "/api/landmarks":
                query = parse_qs(urlparse(self.path).query)
                settings = Settings.parse(
                    {k: float(query[k][0]) for k in ("length", "width") if k in query}
                )
                return self.respond(landmarks(PitchModel(settings.length, settings.width)))
            match = re.fullmatch(r"/api/uploads/([a-f0-9]{32})/frame", path)
            if match:
                index = int(parse_qs(urlparse(self.path).query).get("index", ["0"])[0])
                frame = self.workspace.source_frame(match[1], index)
                return self.respond(
                    {
                        "image": encode_image(frame),
                        "index": index,
                        "width": frame.shape[1],
                        "height": frame.shape[0],
                    }
                )
            if path == "/api/environment":
                return self.respond(environment())
            if path == "/api/runs":
                return self.respond(
                    sorted(
                        [
                            self.workspace.status(p.name)
                            for p in self.workspace.runs.iterdir()
                            if (p / "manifest.json").is_file()
                        ],
                        key=lambda r: r["created"],
                        reverse=True,
                    )
                )
            match = re.fullmatch(
                r"/api/runs/([a-f0-9]{32})(?:/(result|video|review|csv|settings))?", path
            )
            if match:
                ident, action = match.groups()
                status = self.workspace.status(ident)
                folder = self.workspace.run_path(ident)
                if action is None:
                    return self.respond(status)
                if status["state"] != "complete":
                    raise ValueError("This analysis has no completed export yet.")
                if action == "settings":
                    result = json.loads((folder / "result.json").read_text())
                    return self.respond(
                        {**status, **{k: v for k, v in result.items() if k != "frames"}}
                    )
                filename = {
                    "result": "result.json",
                    "video": "result.mp4",
                    "review": "review.mp4",
                    "csv": "positions.csv",
                }[action]
                return self.send_file(
                    folder / filename, download="download" in urlparse(self.path).query
                )
            assets = {
                "/": "index.html",
                "/styles.css": "styles.css",
                "/app.js": "app.js",
                "/analysis.js": "analysis.js",
                "/calibration.js": "calibration.js",
                "/workflow.js": "workflow.js",
                "/insights.js": "insights.js",
                "/overlay.js": "overlay.js",
                "/telestrator.js": "telestrator.js",
                "/scene3d.js": "scene3d.js",
                "/vendor/three.module.min.js": "vendor/three.module.min.js",
                "/vendor/OrbitControls.js": "vendor/OrbitControls.js",
            }
            if path in assets:
                return self.send_file(ROOT / "web" / assets[path])
            self.respond({"error": "Not found"}, 404)
        except FileNotFoundError:
            self.respond({"error": "That file is no longer available."}, 404)
        except (ValueError, KeyError) as exc:
            self.respond({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        if not self.local_request():
            return
        try:
            match = re.fullmatch(r"/api/runs/([a-f0-9]{32})/camera/(preview|save)", self.path)
            if match:
                request = self.body()
                setup, frame, matrix = self.workspace.camera_from_run(match[1], request)
                if match[2] == "save":
                    if request.get("reviewed") is not True:
                        raise ValueError(
                            "Review the projected lines before saving this camera view."
                        )
                    return self.respond(self.workspace.store_camera(setup, frame), 201)
                overlay = draw_calibration_overlay(
                    frame,
                    FrameCalibration(0, matrix, 1.0, "ok", 0, 0, None),
                    PitchModel(setup["length"], setup["width"]),
                    fade_by_confidence=False,
                    show_hud=False,
                )
                return self.respond(
                    {"image": encode_image(overlay), "frame": setup["source_frame"]}
                )
            if self.path == "/api/cameras/preview":
                setup, frame, matrix = self.workspace.prepare_camera(self.body())
                pitch = PitchModel(setup["length"], setup["width"])
                overlay = draw_calibration_overlay(
                    frame,
                    FrameCalibration(0, matrix, 1.0, "manual", 0, 0, None),
                    pitch,
                    fade_by_confidence=False,
                    show_hud=False,
                )
                return self.respond({"image": encode_image(overlay)})
            if self.path == "/api/cameras":
                return self.respond(self.workspace.save_camera(self.body()), 201)
            if self.path == "/api/uploads":
                return self.receive_upload()
            if self.path == "/api/runs":
                return self.respond(self.workspace.start(self.body()), 201)
            match = re.fullmatch(r"/api/runs/([a-f0-9]{32})/(cancel|save|resume)", self.path)
            if match:
                ident, action = match.groups()
                self.body()
                if action == "cancel":
                    return self.respond(self.workspace.cancel(ident))
                if action == "resume":
                    return self.respond(self.workspace.resume(ident), 202)
                with self.workspace.lock:
                    status = self.workspace.status(ident)
                    if status["state"] != "complete":
                        raise ValueError("Only a completed analysis can be saved.")
                    manifest_path = self.workspace.run_path(ident) / "manifest.json"
                    manifest = json.loads(manifest_path.read_text())
                    manifest["saved"] = True
                    write_json(manifest_path, manifest)
                return self.respond(self.workspace.status(ident))
            self.respond({"error": "Not found"}, 404)
        except FileNotFoundError:
            self.respond(
                {"error": "The selected clip is no longer available. Import it again."}, 404
            )
        except (ValueError, KeyError, TypeError) as exc:
            self.respond({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self.respond(
                {"error": "The local operation failed. Check disk space and try again."}, 500
            )

    def receive_upload(self):
        size = int(self.headers.get("Content-Length", 0))
        name = Path(unquote(self.headers.get("X-File-Name", "clip"))).name[:200]
        suffix = Path(name).suffix.lower()
        if suffix not in {".mp4", ".mov", ".avi", ".m4v"}:
            raise ValueError("Choose an MP4, MOV, AVI or M4V video.")
        if not 0 < size <= MAX_UPLOAD:
            raise ValueError("Choose a non-empty clip under 500 MB.")
        if shutil.disk_usage(self.workspace.uploads).free < size * 2 + 64 * 1024 * 1024:
            raise ValueError(
                "There is not enough free disk space for this upload and its results. Free some space or choose a smaller clip."
            )
        ident = uuid.uuid4().hex
        folder = self.workspace.uploads / ident
        folder.mkdir()
        try:
            destination = folder / ("source" + suffix)
            with destination.open("wb") as file:
                remaining = size
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Upload was interrupted. Choose the clip again.")
                    file.write(chunk)
                    remaining -= len(chunk)
            try:
                video_metadata = inspect_video(destination)
            except RuntimeError as exc:
                raise ValueError(
                    "This file is not a readable video. Try an MP4 exported with H.264."
                ) from exc
            metadata = {
                "id": ident,
                "name": name,
                "stored_name": destination.name,
                "size": size,
                "created": time.time(),
                "storage": "copied",
                **video_metadata,
            }
            write_json(folder / "metadata.json", metadata)
            self.respond(metadata, 201)
        except Exception:
            shutil.rmtree(folder)
            raise

    def send_file(self, path: Path, download=False):
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
            if not match:
                self.send_error(416)
                return
            start = int(match[1])
            end = min(int(match[2]) if match[2] else end, end)
            if start > end:
                self.send_error(416)
                return
        self.send_response(206 if range_header else 200)
        self.send_header(
            "Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'",
        )
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="soccermap-{path.name}"')
        self.end_headers()
        with path.open("rb") as file:
            file.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = file.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=ROOT / ".soccermap")
    parser.add_argument(
        "--import-video",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="Link an existing video without copying it (repeat for multiple videos)",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    workspace = Workspace(args.data_dir)
    for path in args.import_video:
        linked = workspace.link_video(path)
        print(f"Linked local video {linked['id']}; available in Imported clips.", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.workspace = workspace
    print(f"SoccerMap is ready at http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        workspace.close()
        server.server_close()


if __name__ == "__main__":
    main()
