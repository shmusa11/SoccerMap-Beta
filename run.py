"""
Entry point for soccer analysis.

Usage:
    python run.py <video> --calibration <calibration.json> [options]

Options:
    --calibration, -c   Path to calibration JSON (from calibrate.py)
    --max-frames, -m    Max frames to process (default: all)
    --team-a             Team A color name (e.g. "red", "maroon")
    --team-b             Team B color name (e.g. "white", "yellow")
    --output, -o        Output video path (default: output/analysis.mp4)
"""

import argparse
import os
import warnings

from soccer_analyzer.pipeline import FootballAnalyzer


def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Soccer video analyzer")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument(
        "-c",
        "--calibration",
        default=None,
        help="Optional calibration JSON bridge (from the old calibrate.py). "
        "Omit to use automatic field registration (requires PnLCalib).",
    )
    parser.add_argument("-m", "--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument(
        "--pnlcalib-root",
        default=None,
        help="Path to the PnLCalib repo (or set PNLCALIB_ROOT) for "
        "automatic field registration when no --calibration is given",
    )
    parser.add_argument("--weights-kp", default=None, help="PnLCalib keypoint weights path")
    parser.add_argument("--weights-line", default=None, help="PnLCalib line weights path")
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Phase B: flow-propagate + outlier-gate calibration between keyframes",
    )
    parser.add_argument("--team-a", default=None, help="Team A color (e.g. red, maroon, blue)")
    parser.add_argument("--team-b", default=None, help="Team B color (e.g. white, yellow, black)")
    parser.add_argument("-o", "--output", default="output/analysis.mp4", help="Output video path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("=" * 60)
    print("Soccer Analyzer")
    print(f"Video: {args.video}")
    print(f"Calibration: {args.calibration or 'automatic (PnLCalib)'}")
    if args.team_a and args.team_b:
        print(f"Teams: {args.team_a} vs {args.team_b}")
    else:
        print("Teams: auto-detect")
    print(f"Max frames: {args.max_frames or 'all'}")
    print("=" * 60)

    # Automatic field registration: if no calibration JSON bridge is given, build the
    # PnLCalib detector (needs --pnlcalib-root / PNLCALIB_ROOT + weights).
    detector = None
    if not args.calibration:
        root = args.pnlcalib_root or os.environ.get("PNLCALIB_ROOT")
        if root:
            from soccer_analyzer.calibration.field_detector import PnLCalibFieldDetector

            detector = PnLCalibFieldDetector(
                pnlcalib_root=root,
                weights_kp=args.weights_kp,
                weights_line=args.weights_line,
            )

    analyzer = FootballAnalyzer(
        calibration_path=args.calibration,
        detector=detector,
        team_a_color=args.team_a,
        team_b_color=args.team_b,
        smooth=args.smooth,
    )
    analytics = analyzer.analyze_video(
        args.video,
        max_frames=args.max_frames,
        output_path=args.output,
    )

    if analytics:
        print(f"Done. Output: {args.output}")
    else:
        print("Analysis failed.")


if __name__ == "__main__":
    main()
