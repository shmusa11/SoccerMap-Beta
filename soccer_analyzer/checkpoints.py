"""Completed-stage checkpoints for interrupted local analyses.

Only JSON is read: no pickle or executable model state. A checkpoint belongs to one source,
configuration, camera reference, pipeline version and local model installation. Source and
weight fingerprints use filesystem metadata, avoiding another full read of a match video.
They detect ordinary replacement/editing, not deliberately forged timestamps and sizes.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from soccer_analyzer.match_processor import MatchData
    from soccer_analyzer.workspace_analysis import Settings

SCHEMA_VERSION = 1
STAGES = {"calibration", "tracking"}


def _encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_digest(value: object) -> str:
    """Bind dependent stages to the exact serialized observations they used."""
    return hashlib.sha256(_encoded(value)).hexdigest()


def file_identity(path: Path) -> dict:
    """Return a local file fingerprint; no file contents or credentials are exposed."""
    path = path.resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def analysis_signature(
    source: Path, settings: Settings, reference: tuple[np.ndarray, np.ndarray] | None
) -> str:
    """Fingerprint inputs without importing neural libraries or reading a whole video."""
    package = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    packages = {}
    for name in (
        "numpy",
        "opencv-python",
        "opencv-python-headless",
        "torch",
        "ultralytics",
        "supervision",
        "scikit-learn",
        "inference",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    field_root = Path(os.environ.get("PNLCALIB_ROOT", package.parent.parent / "PnLCalib"))
    ball_root = Path(os.environ.get("WASB_ROOT", package.parent.parent / "WASB-SBDT"))
    weights = [
        field_root / "SV_kp",
        field_root / "SV_lines",
        ball_root / "wasb_soccer_best.pth.tar",
        package.parent / "models" / "yolov8n.pt",
        package.parent / "pythonProject1" / "yolov8n.pt",
    ]
    camera = None
    if reference is not None:
        camera = hashlib.sha256()
        for array in reference:
            array = np.ascontiguousarray(array)
            camera.update(str((array.dtype.str, array.shape)).encode())
            camera.update(array.tobytes())
        camera = camera.hexdigest()
    return hashlib.sha256(
        _encoded(
            {
                "source": file_identity(source),
                "settings": asdict(settings),
                "camera": camera,
                "code": digest.hexdigest(),
                "packages": packages,
                "weights": [file_identity(p) if p.is_file() else str(p) for p in weights],
                "officials_configured": bool(os.environ.get("ROBOFLOW_API_KEY", "").strip()),
            }
        )
    ).hexdigest()


class CheckpointStore:
    """Atomically publish a stage only after it completes; reject stale/partial checkpoints."""

    def __init__(self, directory: Path, signature: str):
        self.directory, self.signature = directory, signature

    def _path(self, stage: str) -> Path:
        if stage not in STAGES:
            raise ValueError("Unknown analysis checkpoint stage")
        return self.directory / f"{stage}.checkpoint.json"

    def read(self, stage: str) -> dict | None:
        try:
            document = json.loads(self._path(stage).read_text())
            if (
                document["schema"] != SCHEMA_VERSION
                or document["signature"] != self.signature
                or hashlib.sha256(_encoded(document["payload"])).hexdigest() != document["digest"]
            ):
                return None
            return document["payload"]
        except (FileNotFoundError, ValueError, KeyError, TypeError):
            return None

    def write(self, stage: str, payload: dict) -> None:
        path = self._path(stage)
        temporary = path.with_suffix(".tmp")
        data = _encoded(
            {
                "schema": SCHEMA_VERSION,
                "signature": self.signature,
                "digest": hashlib.sha256(_encoded(payload)).hexdigest(),
                "payload": payload,
            }
        )
        try:
            with temporary.open("wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def pack_match(match: MatchData | None) -> dict | None:
    """Serialize source-indexed observations, preserving measured/interpolated provenance."""
    if match is None:
        return None
    return {
        "frames": match.frames,
        "people": {
            str(k): [
                [int(tid), list(map(float, box)), list(map(float, feet))] for tid, box, feet in rows
            ]
            for k, rows in match.people.items()
        },
        "team": {str(k): v for k, v in match.team.items()},
        "kind": {str(k): v for k, v in match.kind.items()},
        "ball": {str(k): [float(v[0]), float(v[1]), bool(v[2])] for k, v in match.ball.items()},
        "team_centers": match.team_centers.tolist() if match.team_centers is not None else None,
        "stats": match.stats,
        "interpolated": sorted(match.interpolated),
        "cuts": match.cuts,
    }


def unpack_match(payload: dict | None) -> MatchData | None:
    """Restore a checkpoint produced by ``pack_match``; source indices remain integers."""
    if payload is None:
        return None
    from soccer_analyzer.match_processor import MatchData

    return MatchData(
        frames=payload["frames"],
        people={
            int(k): [(int(tid), tuple(box), tuple(feet)) for tid, box, feet in rows]
            for k, rows in payload["people"].items()
        },
        team={int(k): v for k, v in payload["team"].items()},
        kind={int(k): v for k, v in payload["kind"].items()},
        ball={int(k): tuple(v) for k, v in payload["ball"].items()},
        team_centers=np.asarray(payload["team_centers"])
        if payload["team_centers"] is not None
        else None,
        stats=payload["stats"],
        interpolated={tuple(v) for v in payload["interpolated"]},
        cuts=payload["cuts"],
    )
