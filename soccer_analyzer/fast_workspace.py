"""Run the fast keyframe pipeline for the browser workspace and convert its output.

The fast pipeline (decode once, sparse verified calibration, camera-motion-compensated
player tracking, ball path tracker, on-field filter) gave the best field coverage on the
local clips. This adapter returns the workspace result format: for every rendered frame,
mapped tracks with measured/interpolated provenance, the ball and the image-to-pitch
homography, plus the camera-view review video and the side-by-side export.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from dotenv import load_dotenv

AI_KEYFRAME_CAP_USD = 0.30  # hard per-run ceiling on OpenAI spend


def frame_record(md, res, idx: int, pitch, ball_gate: float) -> dict:
    """One workspace frame from a FastResult render frame (positions in pitch metres)."""
    H = res.calib.get(idx)
    valid = H is not None
    shown = res.visible.get(idx, set()) if res.visible is not None else None
    fps = res.fps
    first = res.render_frames[0]
    players = []
    for tid, box, feet in md.people.get(idx, []):
        if (shown is not None and tid not in shown) or (shown is None and tid in res.offfield):
            continue
        kind = md.kind.get(tid, "player")
        team = -2 if kind == "referee" else -3 if kind == "goalkeeper" else md.team.get(tid, -1)
        x = y = None
        if valid:
            xy = res.pitch_xy.get(idx, {}).get(tid)
            if xy is None:
                point = cv2.perspectiveTransform(np.array([[feet]], np.float32), H)[0, 0]
                xy = (float(point[0]), float(point[1]))
            # The workspace keeps mapped positions on the pitch (CSV and analytics contract).
            if np.isfinite(xy).all() and 0 <= xy[0] <= pitch.length and 0 <= xy[1] <= pitch.width:
                x, y = float(xy[0]), float(xy[1])
        players.append(
            {
                "id": int(tid),
                "team": int(team),
                "kind": kind,
                "x": x,
                "y": y,
                "measured": (idx, tid) not in md.interpolated,
                "box": [float(v) for v in box],
            }
        )
    ball = md.ball.get(idx)
    ball_xy, ball_measured = None, False
    if ball is not None:
        ball_measured = len(ball) < 3 or ball[2] >= ball_gate
        if valid:
            point = cv2.perspectiveTransform(np.array([[ball[:2]]], np.float32), H)[0, 0]
            if np.isfinite(point).all() and 0 <= point[0] <= pitch.length:
                if 0 <= point[1] <= pitch.width:
                    ball_xy = [float(point[0]), float(point[1])]
    return {
        "index": int(idx),
        "time": round((idx - first) / fps, 6),
        "source_time": round(idx / fps, 6),
        "valid": valid,
        "players": players,
        "ball": ball_xy,
        "ball_measured": bool(ball_measured),
        "ball_image": [float(ball[0]), float(ball[1])] if ball is not None else None,
        "homography": H.tolist() if valid else None,
    }


def make_oracle(settings, warnings: list[str]):
    """Create a cloud client only after explicit per-run consent; otherwise stay local."""
    if not settings.ai_keyframes:
        return None
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        warnings.append("AI keyframes requested but no API key is configured; ran locally.")
        return None
    try:
        from soccer_analyzer.vision_oracle import OpenAIVisionOracle

        return OpenAIVisionOracle(max_usd=AI_KEYFRAME_CAP_USD)
    except Exception as exc:
        warnings.append(f"AI keyframes unavailable ({type(exc).__name__}); ran locally.")
        return None


def analyze_video_fast(path: Path, directory: Path, settings, report: Callable) -> dict:
    """Analyse one passage with the fast keyframe pipeline (offline: no AI keyframes)."""
    from soccer_analyzer.calibration.field_detector import PnLCalibFieldDetector
    from soccer_analyzer.calibration.mapper import FrameCalibration
    from soccer_analyzer.calibration.pitch_model import PitchModel
    from soccer_analyzer.calibration.visualize import draw_calibration_overlay
    from soccer_analyzer.detection import SoccerDetector
    from soccer_analyzer.fast_pipeline import FastConfig, FastPipeline
    from soccer_analyzer.resources import configure_inference, release_model_memory
    from soccer_analyzer.review_render import draw_track_box, review_caption
    from soccer_analyzer.video import frame_window, open_writer
    from soccer_analyzer.workspace_analysis import COLORS, inspect_video

    meta = inspect_video(path)
    start, stop = frame_window(meta["frames"], settings.max_frames, settings.start_frame)
    pitch = PitchModel(settings.length, settings.width)
    timings = {}
    warnings = []
    began = time.perf_counter()
    report("Loading field, player and ball models", 0.02)
    device = configure_inference(settings.low_memory)
    root = os.environ.get("PNLCALIB_ROOT", str(Path(__file__).resolve().parents[2] / "PnLCalib"))
    field = PnLCalibFieldDetector(
        pnlcalib_root=root,
        weights_kp=str(Path(root) / "SV_kp"),
        weights_line=str(Path(root) / "SV_lines"),
        pitch=pitch,
        max_reproj_px=1000.0,
        min_line_support=0.0,
        device=device,
    )
    field._ensure_loaded()
    people = SoccerDetector(
        roboflow_model_id=None,
        device=device,
        people_imgsz=1280 if settings.low_memory else 1920,
    )
    wasb = None
    if settings.ball:
        try:
            from soccer_analyzer.ball_wasb import WASBBallDetector

            wasb = WASBBallDetector(device=device, max_batch=2 if settings.low_memory else 4)
        except Exception as exc:
            from soccer_analyzer.resources import is_memory_error

            if is_memory_error(exc):
                raise
            warnings.append("Ball model unavailable. This run contains player observations only.")
    oracle = make_oracle(settings, warnings)
    if oracle is None:
        config = FastConfig(field_markings=False, ball_vlm=False, onfield_ai=False, cpu_workers=2)
    else:
        config = FastConfig(cpu_workers=2)
    pipe = FastPipeline(field, people, wasb=wasb, oracle=oracle, pitch=pitch, config=config)
    timings["models"] = time.perf_counter() - began

    began = time.perf_counter()
    try:
        res = pipe.run(
            str(path),
            max_frames=stop - start,
            start_frame=start,
            progress=lambda p: report(
                "Mapping the field, tracking players and the ball", 0.05 + 0.75 * p
            ),
        )
    finally:
        if oracle is not None:
            oracle.close()
    timings["analysis"] = time.perf_counter() - began
    del field, people, wasb
    pipe.field = pipe.people = pipe.wasb = None
    release_model_memory()
    if not res.render_frames:
        raise ValueError("The video ended before any frames could be processed.")

    began = time.perf_counter()
    report("Rendering the review video", 0.82)
    out_pitch = res.pitch or pitch
    frames_idx = res.render_frames
    step = frames_idx[1] - frames_idx[0] if len(frames_idx) > 1 else 1
    render_fps = res.fps / step
    fw, fh = res.frame_size
    size = (960, max(2, round(fh * 960 / fw / 2) * 2))
    scale = size[0] / fw
    S_inv = np.linalg.inv(np.diag([scale, scale, 1.0]))
    frames, writer = [], None
    try:
        with pipe._render_frames(str(path), res, size) as smalls:
            for idx, small in smalls:
                if small.shape[1] != size[0]:
                    small = cv2.resize(small, size, interpolation=cv2.INTER_AREA)
                record = frame_record(res.md, res, idx, out_pitch, config.ball_gate_offline)
                H = res.calib.get(idx)
                if H is not None:
                    calib = FrameCalibration(idx, H @ S_inv, 1.0, "fast", 0, 0)
                    overlay = draw_calibration_overlay(
                        small, calib, out_pitch, fade_by_confidence=False, show_hud=False
                    )
                else:
                    overlay = small.copy()
                for player in record["players"]:
                    box = [v * scale for v in player.pop("box")]
                    draw_track_box(
                        overlay, box, COLORS.get(player["team"], COLORS[-1]), player["measured"]
                    )
                if record["ball_image"] is not None:
                    centre = tuple(int(v * scale) for v in record["ball_image"])
                    cv2.circle(overlay, centre, 7, (255, 255, 255), 2)
                    if record["ball_measured"]:
                        cv2.circle(overlay, centre, 3, (255, 255, 255), -1)
                overlay = review_caption(overlay, idx, record["valid"])
                if writer is None:
                    writer = open_writer(
                        str(directory / "review.mp4"),
                        render_fps,
                        (overlay.shape[1], overlay.shape[0]),
                    )
                writer.write(overlay)
                frames.append(record)
                if len(frames) % 10 == 0:
                    report(
                        "Rendering the review video", 0.82 + 0.08 * len(frames) / len(frames_idx)
                    )
    finally:
        if writer is not None:
            writer.release()
    report("Rendering the side-by-side export", 0.9)
    pipe.render(str(path), res, str(directory / "result.mp4"))
    timings["render"] = time.perf_counter() - began

    stats = res.stats
    covered = sum(f["valid"] for f in frames)
    return {
        "synthetic": False,
        "engine": "fast",
        "settings": {
            **asdict(settings),
            "stride": step,
            "length": out_pitch.length,
            "width": out_pitch.width,
        },
        "source": meta,
        "window": {
            "start_frame": frames_idx[0],
            "end_frame_exclusive": stop,
            "raw_frames": stop - start,
            "source_start_seconds": frames_idx[0] / res.fps,
            "sample_alignment": "fast_render_grid",
        },
        "frames": frames,
        "warnings": warnings,
        "duration": len(frames) / render_fps,
        "coverage": covered / len(frames),
        "ball_available": settings.ball and not warnings,
        "timings": {key: round(value, 3) for key, value in timings.items()},
        "reused_stages": [],
        "ai_keyframes": {
            "enabled": oracle is not None,
            "usd": float(stats.get("oracle_usd") or 0.0),
            "cap_usd": AI_KEYFRAME_CAP_USD,
        },
        "calibration": {
            "frames": len(frames),
            "covered": covered,
            "mode": "fast_keyframe",
            "inference_calls": stats.get("detect_frames"),
            "reused": 0,
            "anchors": stats.get("anchors"),
        },
        "tracking": {
            key: stats.get(key)
            for key in ("tracks", "offfield_tracks", "ball_path_frames", "calibrated_pct")
        },
        "coordinate_system": "Corner origin, x along length and y along width, metres. Mapped positions are restricted to the pitch; measured flags distinguish detections from gap interpolation.",
        "limitations": "Estimated positions; coverage is not accuracy. AI keyframes, when enabled, send a few sampled frames to OpenAI. Constant-frame-rate silent export.",
    }
