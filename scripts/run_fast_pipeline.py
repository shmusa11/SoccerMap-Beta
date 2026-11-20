"""Run the fast keyframe pipeline on a clip and render the side-by-side result.

    export PNLCALIB_ROOT=/path/to/PnLCalib          # field detector
    export OPENAI_API_KEY=...                       # optional: enables the vision oracle
    python scripts/run_fast_pipeline.py test_clips/bardNight.MP4 --seconds 10 \
        --out output/fast_bardNight.mp4 --oracle openai --max-usd 0.30

With --oracle openai the pipeline also asks the stronger model (--markings-model) to name the
visible field markings at a few keyframes (VLM field-marking calibration; about $0.016 per
keyframe). By default the keyframes are chosen after the whole window has been read, by view
coverage (--markings-keys per camera-motion segment, default 2: ~$0.033 per window, doubled
up to --markings-max-keys while nothing can be confirmed; a video left unconfirmed in 2
windows is not asked again when --oracle-cache keeps the note), and one
shared camera centre is fitted to all of them (--markings-method joint); on fields with
American-football lines two yard-number reads are added (~$0.01 each). The cheap model
(--ball-model) checks the ball track about once a second (~$0.0003 per check, skipped at night).

Prints a per-stage timing table and writes <out>.json with timings and coverage statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=10.0, help="how much video to process")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument(
        "--start-frame", type=int, help="first ORIGINAL frame index (overrides --start)"
    )
    ap.add_argument("--frames", type=int, help="raw frames to process (overrides --seconds)")
    ap.add_argument("--out", default="output/fast_result.mp4")
    ap.add_argument("--oracle", choices=["none", "openai"], default="none")
    ap.add_argument("--max-usd", type=float, default=0.30, help="hard oracle budget for this run")
    ap.add_argument("--people-model", default="gpt-6-luna")
    ap.add_argument("--landmark-model", default="gpt-6-sol")
    ap.add_argument(
        "--markings-model", default="gpt-6-sol", help="names field markings (calibration)"
    )
    ap.add_argument("--ball-model", default="gpt-6-luna", help="checks the ball track")
    ap.add_argument(
        "--no-markings", action="store_true", help="skip the VLM field-marking calibration"
    )
    ap.add_argument(
        "--markings-every", type=float, default=2.0, help="seconds between marking keyframes"
    )
    ap.add_argument(
        "--markings-frames",
        help="comma-separated ORIGINAL frame indices to use as marking keyframes instead",
    )
    ap.add_argument(
        "--markings-method",
        choices=["joint", "window"],
        default="joint",
        help="joint = one shared camera centre for the window (default); window = round-2 fits",
    )
    ap.add_argument(
        "--markings-select",
        choices=["coverage", "time"],
        default="coverage",
        help="choose marking keyframes by view coverage (default) or every --markings-every s",
    )
    ap.add_argument(
        "--markings-keys", type=int, default=2, help="first coverage keyframes per motion segment"
    )
    ap.add_argument(
        "--markings-max-keys",
        type=int,
        default=8,
        help="when no frame is confirmed, double the keyframes per segment up to this many",
    )
    ap.add_argument(
        "--markings-candidates",
        help="comma-separated ORIGINAL frame indices the coverage choice may pick from",
    )
    ap.add_argument(
        "--football",
        choices=["auto", "on", "off"],
        default="auto",
        help="American-football yard-line grid as extra evidence (auto: non-white soccer lines)",
    )
    ap.add_argument("--no-ball-oracle", action="store_true", help="ball tracker without checks")
    ap.add_argument(
        "--no-onfield-ai",
        action="store_true",
        help="no Decisions API answers for tracks without calibration (on-field rule only)",
    )
    ap.add_argument("--no-ball", action="store_true")
    ap.add_argument("--yolo-batch", type=int, default=4, help="lower for 4K to save memory")
    ap.add_argument(
        "--cpu-workers",
        type=int,
        default=3,
        help="threads for CPU stages next to the GPU models (0 = inline; same results)",
    )
    ap.add_argument("--oracle-cache", help="JSON file reusing oracle answers across runs")
    return ap.parse_args(argv)


def main(argv=None, client=None) -> int:
    run(parse_args(argv), client=client)
    return 0


def run(args: argparse.Namespace, client=None, decide_fn=None):
    """Load the models, analyse, render. `client`: an OpenAI-compatible client for the oracle
    (default: one built from OPENAI_API_KEY); `decide_fn`: an injected Decisions API transport
    (default: HTTPS with OPENAI_API_KEY). Returns (pipeline, result, render info, summary)."""

    import cv2

    from soccer_analyzer.calibration.field_detector import PnLCalibFieldDetector
    from soccer_analyzer.calibration.pitch_model import STANDARD_PITCH
    from soccer_analyzer.detection import SoccerDetector
    from soccer_analyzer.fast_pipeline import FastConfig, FastPipeline, StageTimer

    t_start = time.perf_counter()
    setup = StageTimer()
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    max_frames = args.frames or max(1, int(round(args.seconds * fps)))
    start_frame = args.start_frame if args.start_frame is not None else int(round(args.start * fps))

    root = os.environ.get("PNLCALIB_ROOT")
    with setup("setup: load field CNN"):
        field = PnLCalibFieldDetector(
            pnlcalib_root=root, max_reproj_px=1000.0, min_line_support=0.0
        )
        field._ensure_loaded()
    with setup("setup: load YOLO"):
        people = SoccerDetector(roboflow_model_id=None)
    wasb = None
    if not args.no_ball:
        with setup("setup: load WASB"):
            from soccer_analyzer.ball_wasb import WASBBallDetector

            wasb = WASBBallDetector()
    oracle = None
    if args.oracle == "openai":
        from soccer_analyzer.vision_oracle import OpenAIVisionOracle

        oracle = OpenAIVisionOracle(
            people_model=args.people_model,
            landmark_model=args.landmark_model,
            markings_model=args.markings_model,
            ball_model=args.ball_model,
            max_usd=args.max_usd,
            client=client,
            decide_fn=decide_fn,
        )
        if args.oracle_cache:
            from soccer_analyzer.vision_oracle import CachedVisionOracle

            key = f"{Path(args.video).name}:{Path(args.video).stat().st_size}"
            oracle = CachedVisionOracle(oracle, args.oracle_cache, key)

    config = FastConfig(
        yolo_batch=args.yolo_batch,
        cpu_workers=args.cpu_workers,
        field_markings=not args.no_markings,
        markings_every_s=args.markings_every,
        markings_frames=(
            [int(v) for v in args.markings_frames.split(",")] if args.markings_frames else None
        ),
        markings_method=getattr(args, "markings_method", "joint"),
        markings_select=getattr(args, "markings_select", "coverage"),
        markings_keys=getattr(args, "markings_keys", 2),
        markings_max_keys=max(
            getattr(args, "markings_keys", 2), getattr(args, "markings_max_keys", 8)
        ),
        markings_candidates=(
            [int(v) for v in args.markings_candidates.split(",")]
            if getattr(args, "markings_candidates", None)
            else None
        ),
        markings_football=getattr(args, "football", "auto"),
        ball_vlm=not args.no_ball_oracle,
        onfield_ai=not getattr(args, "no_onfield_ai", False),
    )
    pipe = FastPipeline(
        field, people, wasb=wasb, oracle=oracle, pitch=STANDARD_PITCH, config=config
    )
    t_run = time.perf_counter()
    result = pipe.run(args.video, max_frames=max_frames, start_frame=start_frame)
    t_render = time.perf_counter()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rinfo = pipe.render(args.video, result, args.out)
    t_end = time.perf_counter()
    if oracle is not None:
        oracle.close()

    rows = setup.table() + pipe.timer.table()
    total = t_end - t_start
    print(f"\n{'stage':40s} {'seconds':>8s} {'calls':>6s} {'share':>6s}")
    for name, sec, n in rows:
        print(f"{name:40s} {sec:8.2f} {n:6d} {100 * sec / total:5.1f}%")
    print("(CPU stages run on worker threads next to the main thread: their shares overlap)")
    print(f"{'models loaded (setup)':40s} {t_run - t_start:8.2f}")
    print(f"{'analysis pass (decode..solves)':40s} {t_render - t_run:8.2f}")
    print(f"{'render pass':40s} {t_end - t_render:8.2f}")
    print(f"{'TOTAL wall time':40s} {total:8.2f}")
    print(json.dumps(result.stats, indent=1))
    summary = dict(
        video=args.video,
        start_frame=start_frame,
        raw_frames=max_frames,
        stats=result.stats,
        render=rinfo,
        team_names=result.team_names,
        wall_s=dict(
            setup=t_run - t_start, analysis=t_render - t_run, render=t_end - t_render, total=total
        ),
        stages=[dict(stage=n, seconds=round(s, 3), calls=c) for n, s, c in rows],
        oracle_calls=getattr(oracle, "calls", []),
    )
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1, default=str))
    return pipe, result, rinfo, summary


if __name__ == "__main__":
    raise SystemExit(main())
