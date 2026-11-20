"""
Self-supervised calibration validation — measure quality WITHOUT ground truth.

Since no amateur ground truth exists, we judge a calibration by internal consistency:
  * liveness   — fraction of frames with a valid (gated) calibration;
  * confidence — mean/over-time gate value;
  * reprojection error — for frames where the detector reports landmarks, project the
        pitch model into the image via inv(H) and measure pixel distance to the detected
        landmarks (the core accuracy proxy; ~0 for the cached bridge by construction, and
        meaningful once the PnLCalib detector is wired);
  * drift — stddev over time of a fixed pitch point projected into the image, on a static
        camera; the old optical-flow chaining drifts, the new system should stay flat.

Run:
    python -m soccer_analyzer.calibration.validate <video> [--calibration cal.json]
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH
from soccer_analyzer.calibration.mapper import PitchMapper


def reprojection_error_px(keypoints: dict, H: np.ndarray, pitch: PitchModel) -> np.ndarray:
    """Pixel error between detected landmark pixels and the pitch model reprojected via
    inv(H) (pitch->image). Returns an array of per-landmark errors (empty if none)."""
    if H is None or not keypoints:
        return np.array([])
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.array([])
    errs = []
    for name, (u, v) in keypoints.items():
        if name not in pitch.keypoints:
            continue
        x, y = pitch.keypoints[name]
        p = H_inv @ np.array([x, y, 1.0])
        if abs(p[2]) < 1e-9:
            continue
        errs.append(float(np.hypot(p[0] / p[2] - u, p[1] / p[2] - v)))
    return np.array(errs)


def run_report(
    video_path: str,
    mapper: Optional[PitchMapper] = None,
    calibration_path: Optional[str] = None,
    pitch: PitchModel = STANDARD_PITCH,
    max_frames: int = 300,
    out_dir: str = "output",
) -> dict:
    import cv2

    if mapper is None:
        mapper = PitchMapper(calibration_path=calibration_path, pitch=pitch)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    confidences, sources, n_landmarks, reproj_meds = [], [], [], []
    drift_track = []  # pitch centre projected into image, per frame
    centre = pitch.keypoints["centre"]

    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        calib = mapper.update(frame, idx)
        confidences.append(calib.confidence)
        sources.append(calib.source)
        n_landmarks.append(calib.n_landmarks)

        if calib.source != "none":
            # reprojection error: prefer the detector's own reported error (PnLCalib),
            # else compute it from named landmarks (cached bridge has none -> skipped).
            if calib.source == "detect" and mapper.last_observation is not None:
                obs = mapper.last_observation
                if obs.reproj_err is not None:
                    reproj_meds.append(float(obs.reproj_err))
                else:
                    e = reprojection_error_px(obs.keypoints, obs.homography, pitch)
                    if len(e):
                        reproj_meds.append(float(np.median(e)))
            # drift: centre pitch point -> image
            try:
                Hi = np.linalg.inv(calib.homography)
                p = Hi @ np.array([centre[0], centre[1], 1.0])
                if abs(p[2]) > 1e-9:
                    drift_track.append((p[0] / p[2], p[1] / p[2]))
            except np.linalg.LinAlgError:
                pass
        idx += 1
    cap.release()

    n = max(1, len(confidences))
    live = sum(1 for s in sources if s != "none") / n
    drift = np.array(drift_track)
    drift_px = float(np.hypot(*(drift.std(axis=0))) if len(drift) > 1 else 0.0)

    summary = {
        "frames": len(confidences),
        "liveness": round(live, 3),
        "mean_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 3),
        "median_reproj_px": round(float(np.median(reproj_meds)) if reproj_meds else -1, 2),
        "centre_drift_px": round(drift_px, 2),
        "detector": type(mapper.detector).__name__,
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "calibration_report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    _maybe_plot(confidences, sources, reproj_meds, out_dir)

    print("=== calibration report ===")
    for k, v in summary.items():
        print(f"  {k:18} {v}")
    print(f"  (saved {out_dir}/calibration_report.json)")
    return summary


def _maybe_plot(confidences, sources, reproj_meds, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(confidences, lw=1)
    axes[0].set_title("calibration confidence over frames")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_ylabel("confidence")
    if reproj_meds:
        axes[1].hist(reproj_meds, bins=30)
        axes[1].set_title("median reprojection error per frame (px)")
        axes[1].set_xlabel("pixels")
    else:
        axes[1].text(0.5, 0.5, "no reprojection samples", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "calibration_report.png"), dpi=90)
    plt.close(fig)


def main():
    import argparse

    p = argparse.ArgumentParser(description="Self-supervised calibration validation")
    p.add_argument("video")
    p.add_argument("-c", "--calibration", default=None)
    p.add_argument("-m", "--max-frames", type=int, default=300)
    args = p.parse_args()
    run_report(args.video, calibration_path=args.calibration, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
