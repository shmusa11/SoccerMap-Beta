"""Evaluate ball locations against explicit, source-indexed human annotations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def evaluate_ball(
    result: dict, annotations: list[dict], tolerance_px: float = 12, *, measured_only: bool = False
) -> dict:
    """Score labelled source frames, including explicit invisible-ball negatives.

    Annotation rows contain ``index`` (original frame index), ``visible`` (bool), and
    ``point`` ([x_px, y_px]) when visible. Unlabelled frames do not affect any metric.
    A wrong location counts as both a false positive and a missed visible ball. Empty
    denominators return None, not a fabricated perfect score. Synthetic runs are refused.
    """
    if result.get("synthetic"):
        raise ValueError("Synthetic demo positions cannot establish detector accuracy.")
    if not math.isfinite(tolerance_px) or tolerance_px <= 0:
        raise ValueError("The pixel tolerance must be positive and finite.")
    frames = {f["index"]: f for f in result["frames"]}
    if len(frames) != len(result["frames"]):
        raise ValueError("Predictions contain duplicate source frame indices.")
    seen, distances = set(), []
    tp = fp = fn = tn = unavailable = 0
    for row in annotations:
        index, visible = row.get("index"), row.get("visible")
        if type(index) is not int or index < 0 or index in seen or type(visible) is not bool:
            raise ValueError(
                "Annotations need unique nonnegative source indices and boolean visibility."
            )
        seen.add(index)
        target = row.get("point")
        if visible and (
            not isinstance(target, list)
            or len(target) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in target)
        ):
            raise ValueError(
                "Visible-ball annotations need two finite, nonnegative image coordinates."
            )
        frame = frames.get(index)
        if frame is None:
            unavailable += 1
        predicted = frame.get("ball_image") if frame else None
        if measured_only and frame and not frame.get("ball_measured", False):
            predicted = None
        if predicted is not None and (len(predicted) != 2 or not np.isfinite(predicted).all()):
            raise ValueError("Predicted ball image coordinates must be finite pairs.")
        if visible:
            distance = (
                float(np.linalg.norm(np.asarray(predicted) - target))
                if predicted is not None
                else None
            )
            if distance is not None:
                distances.append(distance)
            if distance is not None and distance <= tolerance_px:
                tp += 1
            else:
                fn += 1
                fp += int(predicted is not None)
        elif predicted is not None:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "labelled_frames": len(seen),
        "unprocessed_labelled_frames": unavailable,
        "tolerance_px": tolerance_px,
        "measured_only": measured_only,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "median_error_px": float(np.median(distances)) if distances else None,
        "p95_error_px": float(np.percentile(distances, 95)) if distances else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result", type=Path, help="Workspace result.json containing ball_image positions"
    )
    parser.add_argument(
        "annotations", type=Path, help="JSON array of labelled original video frames"
    )
    parser.add_argument("--tolerance-px", type=float, default=12)
    parser.add_argument("--measured-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_ball(
                json.loads(args.result.read_text()),
                json.loads(args.annotations.read_text()),
                args.tolerance_px,
                measured_only=args.measured_only,
            ),
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
