"""Portable release provenance: hashes and versions, never credentials or local paths."""

from __future__ import annotations

import hashlib
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(source: Path | None, root: Path) -> dict:
    """Identify actual input, executable source and locally available model weights."""
    digest = hashlib.sha256()
    paths = list((root / "soccer_analyzer").rglob("*.py")) + list((root / "web").glob("*.js"))
    paths += [root / "workspace.py", root / "app.py"]
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    packages = {}
    for name in (
        "numpy",
        "opencv-python",
        "scipy",
        "scikit-learn",
        "torch",
        "ultralytics",
        "supervision",
        "streamlit",
        "python-dotenv",
        "psutil",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    models = []
    if source is not None:
        field = Path(os.environ.get("PNLCALIB_ROOT", root.parent / "PnLCalib"))
        ball = Path(os.environ.get("WASB_ROOT", root.parent / "WASB-SBDT"))
        for component, path in [
            ("field_keypoints", field / "SV_kp"),
            ("field_lines", field / "SV_lines"),
            ("ball", ball / "wasb_soccer_best.pth.tar"),
            ("people_yolo11s", root / "models/yolo11s.pt"),
            ("people_yolov8n", root / "models/yolov8n.pt"),
            ("people_legacy", root / "pythonProject1/yolov8n.pt"),
        ]:
            models.append(
                {
                    "component": component,
                    "file": path.name,
                    "available": path.is_file(),
                    "sha256": file_sha256(path) if path.is_file() else None,
                }
            )
    return {
        "input_fingerprint": {
            "kind": "file" if source else "synthetic_fixture",
            "sha256": file_sha256(source)
            if source
            else hashlib.sha256(b"SoccerMap DemoMatch v1").hexdigest(),
        },
        "code_version": {"algorithm": "sha256-source-tree", "value": digest.hexdigest()},
        "model_identifiers": models,
        "package_versions": packages,
        "model_note": "Available weight inventory; synthetic runs execute no models. See engine and cloud-use record for actual run mode.",
    }


def encoded_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        ok, _ = cap.read()
        if not ok:
            raise ValueError("The exported video is unreadable. Retry rendering before sharing.")
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        return {
            "codec": "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        cap.release()
