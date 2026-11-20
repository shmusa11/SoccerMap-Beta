"""Local workspace analysis and exports; all frame indices refer to the source video."""

from __future__ import annotations

import csv
import io
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from soccer_analyzer.calibration.clip_calibrator import calibrate_clip_auto
from soccer_analyzer.calibration.field_detector import PnLCalibFieldDetector
from soccer_analyzer.calibration.fusion import CoverageMap
from soccer_analyzer.calibration.lookup import CalibrationLookup
from soccer_analyzer.calibration.mapper import FrameCalibration
from soccer_analyzer.calibration.height_check import (
    camera_from_homography,
    estimate_height,
    summarize_heights,
)
from soccer_analyzer.calibration.pitch_model import PitchModel
from soccer_analyzer.calibration.visualize import compose, draw_calibration_overlay, draw_topdown
from soccer_analyzer.review_render import draw_track_box, review_caption
from soccer_analyzer.video import (
    frame_limit,
    frame_window,
    seek_capture,
    open_capture,
    open_writer,
    sampled_fps,
)

COLORS = {
    0: (58, 75, 213),
    1: (205, 114, 65),
    -1: (160, 160, 160),
    -2: (45, 215, 245),
    -3: (190, 65, 180),
}


@dataclass(frozen=True)
class Settings:
    """Immutable configuration. Dimensions are metres; start/count are original video frames."""

    length: float = 105
    width: float = 68
    max_frames: int = 300
    start_frame: int = 0
    stride: int = 3
    ball_every: int = 2
    players: bool = True
    ball: bool = True
    enhance: bool = False
    multisport: bool = True
    verify_lines: bool = False
    camera_setup: str | None = None
    low_memory: bool = True
    register_views: bool = False
    # "fast": fast keyframe pipeline (best field coverage); "classic": three-pass pipeline,
    # also used for saved four-point camera setups.
    engine: str = "fast"
    ai_keyframes: bool = False  # explicit consent before any frame leaves this computer

    @classmethod
    def parse(cls, data: dict) -> Settings:
        if not isinstance(data, dict) or set(data) - set(cls.__dataclass_fields__):
            raise ValueError("Unrecognized settings. Refresh the page and try again.")
        settings = cls(**data)
        if settings.camera_setup is not None:
            import re

            if not isinstance(settings.camera_setup, str) or not re.fullmatch(
                r"[a-f0-9]{32}", settings.camera_setup
            ):
                raise ValueError("Choose a saved camera setup or automatic calibration.")
        for name, lo, hi, step in (
            ("length", 90, 120, 0.5),
            ("width", 45, 90, 0.5),
            ("max_frames", 30, 7200, 1),
            ("start_frame", 0, 100_000_000, 1),
            ("stride", 1, 10, 1),
            ("ball_every", 1, 8, 1),
        ):
            value = getattr(settings, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not lo <= value <= hi
                or value % step
            ):
                raise ValueError(
                    f"{name.replace('_', ' ').capitalize()} must be {lo}–{hi}, in steps of {step}."
                )
        if settings.engine not in ("fast", "classic"):
            raise ValueError("Choose the fast or classic analysis engine.")
        for name in (
            "players",
            "ball",
            "enhance",
            "multisport",
            "verify_lines",
            "low_memory",
            "register_views",
            "ai_keyframes",
        ):
            if type(getattr(settings, name)) is not bool:
                raise ValueError(f"{name} must be on or off.")
        return cls(
            **{
                **asdict(settings),
                **{
                    k: int(getattr(settings, k))
                    for k in ("start_frame", "max_frames", "stride", "ball_every")
                },
            }
        )


def inspect_video(path: Path) -> dict:
    """Decode a first frame; container extensions alone are not validation."""
    with open_capture(path) as cap:
        ok, frame = cap.read()
        if not ok:
            raise ValueError("This video has no readable frames. Try an MP4 exported with H.264.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(
                "The video has no usable frame rate. Export a constant-frame-rate MP4 and try again."
            )
        count = frame_limit(cap.get(cv2.CAP_PROP_FRAME_COUNT), None)
        height, width = frame.shape[:2]
        return {
            "width": width,
            "height": height,
            "fps": fps,
            "frames": count,
            "duration": count / fps if count else None,
        }


def environment() -> dict:
    """Check local files without loading models, making network requests or exposing secrets."""
    root = Path(
        os.environ.get("PNLCALIB_ROOT", str(Path(__file__).resolve().parents[2] / "PnLCalib"))
    )
    wasb = Path(os.environ.get("WASB_ROOT", str(Path(__file__).resolve().parents[2] / "WASB-SBDT")))
    required = [("Field keypoints", root / "SV_kp"), ("Field lines", root / "SV_lines")]
    checks = [
        {"name": name, "ready": path.is_file(), "required": True, "path": str(path)}
        for name, path in required
    ]
    checks.append(
        {
            "name": "Ball detection",
            "ready": (wasb / "wasb_soccer_best.pth.tar").is_file(),
            "required": False,
            "path": str(wasb / "wasb_soccer_best.pth.tar"),
        }
    )
    return {
        "ready": all(c["ready"] for c in checks if c["required"]),
        "checks": checks,
        "note": "File checks only. Model loading is verified when a run starts. First-time player-model setup may download weights.",
    }


def load_models(
    settings: Settings,
    report: Callable,
    *,
    field_models: bool = True,
    detection_models: bool = True,
) -> tuple:
    from soccer_analyzer.resources import configure_inference

    needs_field = field_models and settings.camera_setup is None
    device = (
        configure_inference(settings.low_memory)
        if needs_field or (detection_models and settings.players)
        else None
    )
    root = os.environ.get("PNLCALIB_ROOT", str(Path(__file__).resolve().parents[2] / "PnLCalib"))
    field = None
    if needs_field:
        field = PnLCalibFieldDetector(
            pnlcalib_root=root,
            weights_kp=str(Path(root) / "SV_kp"),
            weights_line=str(Path(root) / "SV_lines"),
            pitch=PitchModel(settings.length, settings.width),
            enhance=settings.enhance,
            max_reproj_px=1000.0,
            min_line_support=0.0,
            device=device,
        )
        field._ensure_loaded()
        field.fusion = None
    player = ball = None
    warnings = []
    if detection_models and settings.players:
        from soccer_analyzer.detection import SoccerDetector

        player = SoccerDetector(
            enhance_dark=settings.enhance,
            device=device,
            people_imgsz=1280 if settings.low_memory else 1920,
        )
        if settings.ball:
            try:
                from soccer_analyzer.ball_wasb import WASBBallDetector

                ball = WASBBallDetector(device=device, max_batch=1 if settings.low_memory else 2)
            except Exception as exc:
                from soccer_analyzer.resources import is_memory_error

                if is_memory_error(exc):
                    raise
                warnings.append(
                    "Ball model unavailable. This run contains player observations only; check Environment for setup."
                )
    return field, player, ball, warnings


def positions_csv(result: dict) -> str:
    """Include unavailable frames explicitly, leaving invalid coordinates blank."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(
        [
            "source_frame",
            "timestamp_s",
            "calibration_valid",
            "track_id",
            "team",
            "kind",
            "x_m",
            "y_m",
            "synthetic",
            "measured",
            "clip_time_s",
            "run_id",
            "segment_id",
            "frame_index",
            "entity_class",
            "observation_status",
        ]
    )
    for frame in result["frames"]:
        for player in frame["players"] or [None]:
            writer.writerow(
                [
                    frame["index"],
                    frame.get("source_time", frame["time"]),
                    frame["valid"],
                    player["id"] if player else "",
                    player["team"] if player else "",
                    player["kind"] if player else "",
                    player["x"] if player and frame["valid"] else "",
                    player["y"] if player and frame["valid"] else "",
                    result["synthetic"],
                    player.get("measured", True) if player else "",
                    frame["time"],
                    result.get("run_id", ""),
                    frame.get("segment_id", 0),
                    frame["index"],
                    player["kind"] if player else "unknown",
                    (
                        "unavailable"
                        if not frame["valid"] or not player or player.get("x") is None
                        else "observed"
                        if player.get("measured", True)
                        else "interpolated"
                    ),
                ]
            )
        if frame.get("ball"):
            ball = frame["ball"]
            writer.writerow(
                [
                    frame["index"],
                    frame.get("source_time", frame["time"]),
                    frame["valid"],
                    "ball",
                    "",
                    "ball",
                    ball[0] if frame["valid"] else "",
                    ball[1] if frame["valid"] else "",
                    result["synthetic"],
                    frame.get("ball_measured", True),
                    frame["time"],
                    result.get("run_id", ""),
                    frame.get("segment_id", 0),
                    frame["index"],
                    "ball",
                    (
                        "unavailable"
                        if not frame["valid"]
                        else "observed"
                        if frame.get("ball_measured", True)
                        else "interpolated"
                    ),
                ]
            )
    return out.getvalue()


def _calibrate_video(
    path, settings, total, pitch, detector, report, camera_reference, calibrator, warnings
):
    """Run field inference or reference checks; preserve rejected source samples."""
    if settings.camera_setup is not None:
        if camera_reference is None:
            raise ValueError(
                "The saved camera setup is unavailable. Choose it again before running."
            )
        from soccer_analyzer.calibration.camera import calibrate_from_reference

        res, stats = calibrate_from_reference(
            str(path),
            *camera_reference,
            stride=settings.stride,
            max_frames=total,
            start_frame=settings.start_frame,
            progress=lambda p: report("Checking the saved camera view", 0.06 + p * 0.49),
        )
        if stats["covered"] < stats["frames"]:
            warnings.append(
                "Some views do not match the saved setup. Their maps are hidden; select new points after a pan, zoom or camera move."
            )
    else:
        res, stats = calibrator(
            detector,
            str(path),
            pitch=pitch,
            stride=4,
            max_frames=total,
            start_frame=settings.start_frame,
            register_views=settings.register_views,
            progress=lambda p: report("Calibrating the field", 0.06 + p * 0.23),
        )
    if settings.camera_setup is None and stats["covered"] / max(1, stats["frames"]) < 0.3:
        first_calls = stats.get("inference_calls")
        res, stats = calibrator(
            detector,
            str(path),
            pitch=pitch,
            stride=2,
            max_frames=total,
            start_frame=settings.start_frame,
            register_views=settings.register_views,
            progress=lambda p: report("Checking difficult views", 0.29 + p * 0.16),
        )
        if first_calls is not None and "inference_calls" in stats:
            stats["inference_calls"] += first_calls
        stats["retried"] = True
    if (
        settings.camera_setup is None
        and settings.multisport
        and stats["covered"] / max(1, stats["frames"]) < 0.1
    ):
        from soccer_analyzer.calibration.multisport import calibrate_multisport

        try:
            candidate, _ = calibrate_multisport(
                str(path),
                pitch,
                max_frames=total,
                start_frame=settings.start_frame,
                progress=lambda p: report("Checking multi-sport lines", 0.45 + p * 0.1),
            )
            if candidate:
                res = candidate
                stats.update(
                    mode="multisport",
                    frames=len(res),
                    covered=sum(h is not None for h in res.values()),
                )
        except Exception as exc:
            from soccer_analyzer.resources import is_memory_error

            if is_memory_error(exc):
                raise
            warnings.append(
                "Multi-sport fallback could not complete. Unverified frames remain unavailable."
            )
    return res, stats


def _track_video(path, settings, total, meta, lookup, player, ball, report):
    """Compute source-indexed observations without retaining model objects in the result."""
    match = None
    output_fps = sampled_fps(meta["fps"], settings.stride)
    if player is not None:
        from soccer_analyzer.match_processor import MatchProcessor

        processor = MatchProcessor(
            player,
            wasb=ball,
            fps=max(1, round(output_fps)),
            ball_every=settings.ball_every,
            yolo_batch=1 if settings.low_memory else 2,
            ball_thresh=0.28 if total // settings.stride <= 150 else 0.32,
        )

        def on_pitch(index, feet):
            matrix = lookup.at(index)
            if matrix is None:
                return False
            point = cv2.perspectiveTransform(np.array([[feet]], np.float32), matrix).reshape(-1, 2)[
                0
            ]
            return -2 <= point[0] <= settings.length + 2 and -2 <= point[1] <= settings.width + 2

        match = processor.sweep(
            str(path),
            stride=settings.stride,
            max_frames=total,
            start_frame=settings.start_frame,
            progress=lambda p: report("Tracking players and solving paths", 0.55 + p * 0.25),
            pitch_filter=on_pitch
            if lookup.coverage(total, settings.stride, settings.start_frame)[0]
            else None,
        )
    return match


def analyze_video(
    path: Path,
    directory: Path,
    settings: Settings,
    report: Callable,
    *,
    model_loader=None,
    calibrator=calibrate_clip_auto,
    camera_reference: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Analyse one passage in a separately cancellable worker process.

    The fast keyframe engine is the default. The classic three-pass pipeline handles saved
    camera setups and injected test models, and stays selectable with ``engine="classic"``.
    """
    injected = model_loader is not None or calibrator is not calibrate_clip_auto
    if settings.engine == "fast" and camera_reference is None and not injected:
        from soccer_analyzer.fast_workspace import analyze_video_fast

        return analyze_video_fast(path, directory, settings, report)
    from soccer_analyzer.checkpoints import (
        CheckpointStore,
        analysis_signature,
        content_digest,
        pack_match,
        unpack_match,
    )
    from soccer_analyzer.resources import release_model_memory

    meta = inspect_video(path)
    start_frame, stop_frame = frame_window(
        meta["frames"], settings.max_frames, settings.start_frame
    )
    total = stop_frame - start_frame
    pitch = PitchModel(settings.length, settings.width)
    store = CheckpointStore(directory, analysis_signature(path, settings, camera_reference))
    calibration_cache = store.read("calibration")
    # Tracking is coupled to the exact accepted calibration, so both stages must be present.
    tracking_cache = store.read("tracking") if calibration_cache is not None else None
    reused_stages = []
    warnings = []
    timings = {"models": 0.0, "calibration": 0.0, "detection_and_solve": 0.0}
    detector = player = ball = None
    injected_loaded = False
    if calibration_cache is not None:
        report("Reusing completed field calibration", 0.55)
        res = {
            i: np.asarray(matrix, dtype=np.float64) if matrix is not None else None
            for i, matrix in calibration_cache["samples"]
        }
        stats = dict(calibration_cache["stats"])
        stats["cached_inference_calls"] = stats.get("inference_calls", 0)
        stats["inference_calls"] = 0
        warnings.extend(calibration_cache["warnings"])
        reused_stages.append("calibration")
    else:
        report("Loading field models", 0.02)
        started = time.perf_counter()
        detector, player, ball, model_warnings = (
            load_models(settings, report, detection_models=False)
            if model_loader is None
            else model_loader(settings, report)
        )
        injected_loaded = model_loader is not None
        warnings.extend(model_warnings)
        timings["models"] += time.perf_counter() - started
        started = time.perf_counter()
        report("Calibrating the field", 0.06)
        res, stats = _calibrate_video(
            path,
            settings,
            total,
            pitch,
            detector,
            report,
            camera_reference,
            calibrator,
            warnings,
        )
        timings["calibration"] = time.perf_counter() - started
        calibration_cache = {
            "samples": [
                [i, matrix.tolist() if matrix is not None else None] for i, matrix in res.items()
            ],
            "stats": stats,
            "warnings": warnings,
        }
        store.write("calibration", calibration_cache)
    detector = None
    release_model_memory()
    lookup = CalibrationLookup(res)
    calibration_digest = content_digest(calibration_cache["samples"])
    if (
        tracking_cache is not None
        and tracking_cache.get("calibration_digest") != calibration_digest
    ):
        tracking_cache = None
    if tracking_cache is not None:
        report("Reusing completed player and ball tracking", 0.8)
        match = unpack_match(tracking_cache["match"])
        ball_available = tracking_cache["ball_available"]
        warnings.extend(tracking_cache["warnings"])
        reused_stages.append("tracking")
    else:
        report("Loading player and ball models", 0.55)
        started = time.perf_counter()
        detection_warnings = []
        if model_loader is None:
            _, player, ball, detection_warnings = load_models(settings, report, field_models=False)
        elif not injected_loaded:
            _, player, ball, detection_warnings = model_loader(settings, report)
        timings["models"] += time.perf_counter() - started
        warnings.extend(detection_warnings)
        started = time.perf_counter()
        match = _track_video(path, settings, total, meta, lookup, player, ball, report)
        timings["detection_and_solve"] = time.perf_counter() - started
        ball_available = ball is not None
        store.write(
            "tracking",
            {
                "match": pack_match(match),
                "calibration_digest": calibration_digest,
                "ball_available": ball_available,
                "warnings": detection_warnings,
            },
        )
    player = ball = None
    release_model_memory()
    output_fps = sampled_fps(meta["fps"], settings.stride)
    started = time.perf_counter()
    report("Rendering review video", 0.8)
    frames = []
    heights = []
    height_matrix = height_camera = None
    coverage = CoverageMap(pitch)
    writer = review_writer = None
    try:
        with open_capture(path) as cap:
            seek_capture(cap, start_frame)
            for index in range(start_frame, stop_frame):
                if not cap.grab():
                    break
                if (index - start_frame) % settings.stride:
                    continue
                ok, image = cap.retrieve()
                if not ok:
                    break
                matrix = lookup.at(index)
                valid = matrix is not None and bool(np.all(np.isfinite(matrix)))
                if valid and settings.verify_lines:
                    from soccer_analyzer.calibration.line_support import (
                        line_distance_field,
                        line_support,
                    )

                    distance = line_distance_field(image)
                    valid = (
                        distance is not None
                        and line_support(matrix, distance, pitch, radius=8) >= 0.28
                    )
                people = match.people.get(index, []) if match else []
                players = []
                projected = []
                calib = FrameCalibration(
                    index,
                    matrix if valid else np.eye(3),
                    1.0 if valid else 0.0,
                    "ok" if valid else "none",
                    0,
                    0,
                    None,
                )
                overlay = draw_calibration_overlay(
                    image, calib, pitch, fade_by_confidence=False, show_hud=False
                )
                if valid and matrix is not height_matrix:
                    height_matrix = matrix
                    height_camera = camera_from_homography(matrix, (meta["width"], meta["height"]))
                for tid, box, feet in people:
                    measured = (index, tid) not in match.interpolated
                    kind = match.kind.get(tid, "player")
                    team = (
                        -2
                        if kind == "referee"
                        else -3
                        if kind == "goalkeeper"
                        else match.team.get(tid, -1)
                    )
                    x = y = None
                    if valid:
                        point = cv2.perspectiveTransform(
                            np.array([[feet]], np.float32), matrix
                        ).reshape(-1, 2)[0]
                        if (
                            np.isfinite(point).all()
                            and 0 <= point[0] <= pitch.length
                            and 0 <= point[1] <= pitch.width
                        ):
                            x, y = map(float, point)
                            projected.append((x, y, team, measured))
                    players.append(
                        {
                            "id": int(tid),
                            "team": int(team),
                            "kind": kind,
                            "x": x,
                            "y": y,
                            "measured": (index, tid) not in match.interpolated,
                        }
                    )
                    x1, y1, x2, y2 = map(int, box)
                    if (
                        kind == "player"
                        and x is not None
                        and height_camera is not None
                        and (index, tid) not in match.interpolated
                        and len(heights) < 5000
                        and x1 > 3
                        and y1 > 3
                        and x2 < meta["width"] - 3
                        and y2 < meta["height"] - 3
                    ):
                        estimated = estimate_height(height_camera, (x, y), ((x1 + x2) / 2, y1))
                        if estimated is not None:
                            heights.append(estimated)
                    draw_track_box(overlay, box, COLORS.get(team, COLORS[-1]), measured)
                ball_xy = None
                observed_ball = match.ball.get(index) if match else None
                if observed_ball:
                    bx, by = map(int, observed_ball[:2])
                    cv2.circle(overlay, (bx, by), 8, (255, 255, 255), 2)
                    if observed_ball[2]:
                        cv2.circle(overlay, (bx, by), 3, (255, 255, 255), -1)
                    if valid:
                        b = cv2.perspectiveTransform(
                            np.array([[observed_ball[:2]]], np.float32), matrix
                        ).reshape(-1, 2)[0]
                        if (
                            np.isfinite(b).all()
                            and 0 <= b[0] <= pitch.length
                            and 0 <= b[1] <= pitch.width
                        ):
                            ball_xy = list(map(float, b))
                if valid:
                    coverage.add(matrix, (meta["width"], meta["height"]))
                overlay = review_caption(overlay, index, valid)
                topdown = draw_topdown(
                    pitch,
                    calib if valid else None,
                    projected,
                    ball_xy,
                    coverage.normalized(),
                    frame_size=(meta["width"], meta["height"]),
                    ball_measured=bool(observed_ball and observed_ball[2]),
                )
                canvas = compose(overlay, topdown, out_width=1280)
                if writer is None:
                    writer = open_writer(
                        str(directory / "result.mp4"),
                        output_fps,
                        (canvas.shape[1], canvas.shape[0]),
                    )
                writer.write(canvas)
                review_size = (960, max(2, round(overlay.shape[0] / overlay.shape[1] * 480) * 2))
                if review_writer is None:
                    review_writer = open_writer(
                        str(directory / "review.mp4"), output_fps, review_size
                    )
                review_writer.write(cv2.resize(overlay, review_size))
                frames.append(
                    {
                        "index": index,
                        "time": round((index - start_frame) / meta["fps"], 6),
                        "source_time": round(index / meta["fps"], 6),
                        "valid": valid,
                        "players": players,
                        "ball": ball_xy,
                        "ball_measured": bool(observed_ball and observed_ball[2]),
                        "ball_image": list(map(float, observed_ball[:2]))
                        if observed_ball
                        else None,
                        "homography": matrix.tolist() if valid else None,
                    }
                )
                if len(frames) % 10 == 0:
                    report("Rendering review video", 0.8 + 0.19 * (index - start_frame + 1) / total)
    finally:
        if writer is not None:
            writer.release()
        if review_writer is not None:
            review_writer.release()
    if not frames:
        raise ValueError("The video ended before any frames could be processed.")
    timings["render"] = round(time.perf_counter() - started, 3)
    height_summary = summarize_heights(heights)
    if height_summary["state"] == "review":
        warnings.append(
            "Player heights suggest the field scale may be off. Review the field overlay before using mapped distances. Cropping, crouching and partial detections can also trigger this check."
        )
    return {
        "synthetic": False,
        "settings": asdict(settings),
        "source": meta,
        "window": {
            "start_frame": start_frame,
            "end_frame_exclusive": stop_frame,
            "raw_frames": total,
            "source_start_seconds": start_frame / meta["fps"],
            "sample_alignment": "window_start",
        },
        "frames": frames,
        "warnings": warnings,
        "duration": len(frames) / output_fps,
        "coverage": sum(f["valid"] for f in frames) / len(frames),
        "ball_available": ball_available,
        "timings": {key: round(value, 3) for key, value in timings.items()},
        "reused_stages": reused_stages,
        "calibration": stats,
        "tracking": match.stats if match else {},
        "height_check": height_summary,
        "coordinate_system": "Corner origin, x along length and y along width, metres. Mapped positions are restricted to the pitch; measured flags distinguish detections from gap interpolation.",
        "limitations": "Estimated positions; coverage is not accuracy. Constant-frame-rate silent export. Nonstandard field geometry and full matches remain unvalidated.",
    }


class DemoMatch:
    """Deterministic, invented eleven-a-side play for the synthetic demo.

    The ball follows a scripted path (dribble, pass at 15 m/s, dribble), sometimes
    intercepted. Teams hold formations that slide with the ball and push up in possession;
    every position is continuous in time. It is illustrative, not a simulation.
    """

    # Formation slots as (fraction of length from own goal, fraction of width).
    SHAPES = (
        [
            (0.04, 0.5),
            (0.2, 0.15),
            (0.18, 0.38),
            (0.18, 0.62),
            (0.2, 0.85),
            (0.38, 0.2),
            (0.36, 0.4),
            (0.36, 0.6),
            (0.38, 0.8),
            (0.55, 0.4),
            (0.56, 0.62),
        ],
        [
            (0.04, 0.5),
            (0.2, 0.18),
            (0.18, 0.4),
            (0.18, 0.6),
            (0.2, 0.82),
            (0.36, 0.3),
            (0.34, 0.5),
            (0.36, 0.7),
            (0.54, 0.2),
            (0.57, 0.5),
            (0.54, 0.8),
        ],
    )
    OFFSET = 0.7  # m: the ball sits just ahead of its carrier

    def __init__(self, length: float, width: float, seconds: float, seed: int = 7):
        import random

        self.length, self.width = length, width
        rng = random.Random(seed)
        self.changes = [(-10.0, 0)]  # (time, team that gains possession)
        # Each leg: carrier holds from `t0` (ball at a) to `t1` (ball at r), then the ball
        # travels to the next carrier until `t2`.
        self.legs = []
        team, carrier, t = 0, 7, 0.0
        a = self._slot(team, carrier, (length / 2, width / 2), t)
        a = (a[0] + self._ahead(team), a[1])
        while t < seconds + 3:
            t1 = t + rng.uniform(0.8, 2.0)
            r = (
                min(length - 2, max(2, a[0] + self._ahead(team) * rng.uniform(2, 5))),
                min(width - 2, max(2, a[1] + rng.uniform(-1.5, 1.5))),
            )
            mates = [(i, self._slot(team, i, r, t1)) for i in range(1, 11) if i != carrier]
            forward = 1 if team == 0 else -1
            choices = [m for m in mates if 9 <= math.dist(m[1], r) <= 30] or mates
            weights = [1.8 if (xy[0] - r[0]) * forward > 0 else 1.0 for _, xy in choices]
            target, guess = rng.choices(choices, weights)[0]
            next_team = team
            if rng.random() < 0.22:
                mid = ((r[0] + guess[0]) / 2, (r[1] + guess[1]) / 2)
                next_team = 1 - team
                target = min(
                    range(1, 11),
                    key=lambda i: math.dist(self._slot(next_team, i, mid, t1), mid),
                )
            end = guess
            for _ in range(3):  # The receiver keeps moving; aim where they will be.
                t2 = t1 + max(0.5, math.dist(r, end) / 15)
                slot = self._slot(next_team, target, end, t2)
                end = (slot[0] + self._ahead(next_team), slot[1])
            t2 = t1 + max(0.5, math.dist(r, end) / 15)
            self.legs.append((t, t1, t2, team, carrier, a, r, end))
            if next_team != team:
                self.changes.append((t2, next_team))
            team, carrier, t, a = next_team, target, t2, end

    def _ahead(self, team):
        return self.OFFSET if team == 0 else -self.OFFSET

    def _share(self, team, time):
        """How much `team` is in possession (0..1), blended over 3 s after a change."""
        prev, current = 0.5, 0.5
        for t, owner in self.changes:
            if t > time:
                break
            prev = current
            current = 1.0 if owner == team else 0.0
            since = time - t
        if prev == current:
            return current
        return prev + (current - prev) * min(1.0, since / 3.0)

    def _slot(self, team, index, ball, time):
        """Formation position (team 0 attacks right) given the ball and possession."""
        fx, fy = self.SHAPES[team][index]
        bx, by = ball
        share = self._share(team, time) if self.changes else 0.5
        side = 1 if team == 0 else -1
        depth = fx + (0.05 * share - 0.02 if index else 0.01 * share)
        if index:
            depth += 0.22 * (bx / self.length - 0.5) * side
        x = (depth if team == 0 else 1 - depth) * self.length
        lateral = (0.15 if index else 0.05) * (by - self.width / 2) * (1 - abs(fy - 0.5))
        x += (1.2 if index else 0.4) * math.sin(time * 0.9 + index * 1.7 + team)
        y = fy * self.width + lateral + 0.8 * math.cos(time * 0.7 + index * 2.3)
        return (min(self.length - 0.5, max(0.5, x)), min(self.width - 0.5, max(0.5, y)))

    def _ball(self, time):
        for leg in self.legs:
            t0, t1, t2, _, _, a, r, end = leg
            if time <= t2:
                if time <= t1:
                    f = max(0.0, (time - t0) / (t1 - t0))
                    return (a[0] + (r[0] - a[0]) * f, a[1] + (r[1] - a[1]) * f)
                f = (time - t1) / (t2 - t1)
                return (r[0] + (end[0] - r[0]) * f, r[1] + (end[1] - r[1]) * f)
        return self.legs[-1][7]

    def _pull(self, team, index, time):
        """Weight and point drawing a player toward the ball as receiver or carrier."""
        for i, (t0, t1, t2, owner, carrier, a, r, end) in enumerate(self.legs):
            if owner != team or carrier != index:
                continue
            if t0 - 0.6 <= time <= t0:
                return (time - (t0 - 0.6)) / 0.6, a
            if t0 <= time <= t1:
                return 1.0, None
            if t1 <= time <= t1 + 0.8:
                return 1 - (time - t1) / 0.8, r
        return 0.0, None

    def state(self, time):
        """Players as (id, team, kind, x, y) and the ball [x, y] at a passage time."""
        ball = self._ball(time)
        out = []
        for team in (0, 1):
            for index in range(11):
                x, y = self._slot(team, index, ball, time)
                weight, anchor = self._pull(team, index, time)
                if weight > 0:
                    point = ball if anchor is None else anchor
                    cx, cy = point[0] - self._ahead(team), point[1]
                    x, y = x + (cx - x) * weight, y + (cy - y) * weight
                kind = "goalkeeper" if index == 0 else "player"
                out.append((team * 11 + index + 1, -3 if index == 0 else team, kind, x, y))
        # The referee trails play: a damped follow of the ball, never at pass speed.
        ref = (
            self.length / 2 + 0.45 * (ball[0] - self.length / 2) + 6,
            self.width / 2 + 0.35 * (ball[1] - self.width / 2) - 9,
        )
        out.append((23, -2, "referee", *ref))
        return out, [float(ball[0]), float(ball[1])]


def make_demo(directory: Path, settings: Settings, report: Callable) -> dict:
    """Generate owned synthetic footage and observations; never represent it as inference."""
    if settings.start_frame:
        raise ValueError(
            "The synthetic demo starts at frame 0. Use real footage to select a later passage."
        )
    fps, raw_total = 30, min(settings.max_frames, 1800)
    frames = []
    pitch = PitchModel(settings.length, settings.width)
    match = DemoMatch(settings.length, settings.width, raw_total / fps)
    writer = review_writer = None
    try:
        for index in range(0, raw_total, settings.stride):
            time = index / fps
            valid = not 18 <= time % 60 < 24
            players = []
            scripted, scripted_ball = match.state(time) if settings.players else ([], None)
            for tid, team, kind, x, y in scripted:
                players.append(
                    {
                        "id": tid,
                        "team": team,
                        "kind": kind,
                        "x": x if valid else None,
                        "y": y if valid else None,
                        "demo_x": x,
                        "demo_y": y,
                    }
                )
            ball = scripted_ball if settings.ball and settings.players and valid else None
            mapped = [(p["x"], p["y"], p["team"]) for p in players] if valid else []
            top = draw_topdown(
                pitch, None, mapped, ball, np.zeros((40, 64), np.float32), frame_size=(960, 540)
            )
            # Both panels are synthetic diagrams, with permanent in-frame provenance.
            camera = draw_topdown(
                pitch,
                None,
                [(p["demo_x"], p["demo_y"], p["team"]) for p in players],
                ball,
                np.zeros((40, 64), np.float32),
                frame_size=(960, 540),
            )
            if not valid:
                top[:] = (35, 56, 45)
                cv2.putText(
                    top,
                    "MAP UNAVAILABLE",
                    (25, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            canvas = compose(camera, top, out_width=1280)
            cv2.rectangle(canvas, (0, 0), (1280, 36), (28, 49, 38), -1)
            cv2.putText(
                canvas,
                f"SYNTHETIC DEMO - NOT MODEL OUTPUT    {time:05.1f}s",
                (18, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            if writer is None:
                writer = open_writer(
                    str(directory / "result.mp4"),
                    fps / settings.stride,
                    (canvas.shape[1], canvas.shape[0]),
                )
            writer.write(canvas)
            cv2.rectangle(camera, (0, 0), (camera.shape[1], 32), (28, 49, 38), -1)
            cv2.putText(
                camera,
                "SYNTHETIC DEMO - NOT MODEL OUTPUT",
                (15, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            if review_writer is None:
                review_writer = open_writer(
                    str(directory / "review.mp4"),
                    fps / settings.stride,
                    (camera.shape[1], camera.shape[0]),
                )
            review_writer.write(camera)
            frames.append(
                {
                    "index": index,
                    "time": round(time, 6),
                    "valid": valid,
                    "players": [
                        {k: v for k, v in p.items() if not k.startswith("demo_")} for p in players
                    ],
                    "ball": ball,
                }
            )
            if len(frames) % 10 == 0:
                report("Building the synthetic walkthrough", (index + 1) / raw_total * 0.95)
    finally:
        if writer is not None:
            writer.release()
        if review_writer is not None:
            review_writer.release()
    return {
        "synthetic": True,
        "settings": asdict(settings),
        "source": {"fps": fps, "frames": raw_total},
        "frames": frames,
        "warnings": [
            "Synthetic demonstration. No footage was analyzed; positions and the 18–24 s calibration gap are deliberately generated."
        ],
        "duration": len(frames) * settings.stride / fps,
        "coverage": sum(f["valid"] for f in frames) / len(frames),
        "ball_available": settings.ball and settings.players,
        "coordinate_system": "Corner origin, x along length and y along width, metres.",
        "limitations": "Illustrative data only. This demo demonstrates the interface, not model accuracy or processing speed.",
    }
