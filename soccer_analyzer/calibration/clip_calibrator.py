"""Field calibration with verified reference reuse on stationary views.

The automatic path checks camera agreement before reusing an anchor, runs field inference
when the view changes, and retains rejected samples as unavailable. Optional registration
proposes mappings for overlapping pans and verifies them against current field lines.
The older line-support and propagation routines remain available for research callers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH
from soccer_analyzer.video import frame_limit, open_capture


def _gray_small(frame, w=480):
    h0, w0 = frame.shape[:2]
    s = w / w0
    return cv2.cvtColor(cv2.resize(frame, (w, max(1, int(h0 * s)))), cv2.COLOR_BGR2GRAY)


def _propagate_static(entries, max_gap):
    """Fill every frame from its NEAREST anchor (entries with Hv) if the camera was static the
    whole way between them; else leave offline."""
    anchors = [i for i, e in enumerate(entries) if e["Hv"] is not None]
    result = {e["idx"]: None for e in entries}
    stats = {"anchors": len(anchors), "frames": len(entries), "covered": 0, "moving": 0}
    if not anchors:
        return result, stats
    aa = np.array(anchors)
    for i, e in enumerate(entries):
        a = int(aa[np.argmin(np.abs(aa - i))])
        if abs(a - i) > max_gap:
            continue
        lo, hi = (a, i) if a <= i else (i, a)
        if all(entries[k]["motion"] < 2.5 for k in range(lo + 1, hi + 1)):
            result[e["idx"]] = entries[a]["Hv"].copy()
            stats["covered"] += 1
        else:
            stats["moving"] += 1
    return result, stats


def calibrate_clip_ls(
    detector,
    video_path: str,
    pitch: PitchModel = STANDARD_PITCH,
    stride: int = 2,
    max_frames: Optional[int] = None,
    ls_thresh: float = 0.40,
    radius: float = 8.0,
    optimize_every: int = 6,
    warm_starts: int = 3,
    max_gap: int = 90,
    progress=None,
):
    """BREAKTHROUGH path: calibrate by line-support optimization (correspondence-free), made
    FAST by exploiting the static camera.

    1) COLD START: one robust multi-start solve finds the fixed camera's calibration.
    2) Optimize only every `optimize_every`-th frame (warm, seeded from the running solution),
       then COPY each verified solution across the static stretch around it. Drifted frames are
       caught by the app's re-verification, so we don't pay to optimize every frame."""
    from soccer_analyzer.calibration.line_support import line_distance_field
    from soccer_analyzer.calibration.ls_calibrator import optimize_calibration, cold_start

    fusion = getattr(detector, "fusion", None)
    if fusion is None or not getattr(fusion, "ready", False):
        return {}, {"frames": 0, "covered": 0, "cold_ls": 0.0, "error": "no fusion"}
    pos, pp = list(fusion.position), list(fusion.principal)

    cold_ls, cur_rvec, cur_f = cold_start(detector, video_path, pos, pp, pitch, radius=radius)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames:
        total = min(total, max_frames)
    if cold_ls < ls_thresh * 0.75:
        cap.release()
        return (
            {},
            {
                "frames": total // max(1, stride),
                "covered": 0,
                "cold_ls": round(cold_ls, 2),
                "error": "cold start failed",
            },
        )

    entries = []
    prev = None
    kf = idx = 0
    while True:
        if total and idx >= total:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        g = _gray_small(frame)
        motion = 0.0 if prev is None else float(np.mean(cv2.absdiff(g, prev)))
        prev = g
        Hv = None
        if kf % optimize_every == 0:
            df = line_distance_field(frame)
            if df is not None:
                H, ls, x = optimize_calibration(
                    df,
                    pos,
                    pp,
                    cur_rvec,
                    cur_f,
                    pitch,
                    n_starts=warm_starts,
                    radius=radius,
                    jitter=(0.04, 0.04, 0.03, 150.0),
                )
                if ls >= ls_thresh:
                    Hv = H
                    cur_rvec, cur_f = x[:3], x[3]
        entries.append({"idx": idx, "Hv": Hv, "motion": motion})
        kf += 1
        idx += 1
        if progress is not None and kf % 5 == 0 and total:
            progress(min(1.0, idx / total))
    cap.release()
    res, st = _propagate_static(entries, max_gap)
    st["cold_ls"] = round(cold_ls, 2)
    return res, st


def detector_candidates(
    observation, pitch: PitchModel = STANDARD_PITCH
) -> List[Tuple[str, np.ndarray]]:
    """Candidate image->pitch homographies from one field-detector observation: the detector's
    own camera fit ("cnn") and a RANSAC fit to its labelled keypoints ("keypoint")."""
    from soccer_analyzer.calibration.keypoint_calib import keypoint_homography

    cands: List[Tuple[str, np.ndarray]] = []
    H = getattr(observation, "homography", None)
    if H is not None and np.all(np.isfinite(H)):
        cands.append(("cnn", H))
    Hk, inliers = keypoint_homography(observation, pitch)
    if Hk is not None and inliers >= 4:
        cands.append(("keypoint", Hk))
    return cands


def select_calibration(
    candidates,
    dist_field: Optional[np.ndarray],
    frame_size: Tuple[int, int],
    pitch: PitchModel = STANDARD_PITCH,
    ls_floor: float = 0.38,
    spread_floor: float = 0.10,
    radius: float = 8.0,
) -> Optional[Tuple[np.ndarray, float, str]]:
    """Best verified candidate as (sign-normalised H, line_support, tag), or None.

    Each (tag, H) candidate must spread over the frame (not a degenerate smear), be a physically
    plausible camera, and put the pitch markings on real extracted line pixels. Among survivors,
    fits whose camera stands beside the field are preferred over fits placing a low camera on
    the pitch (line support cannot tell those apart; see `camera_inside_pitch`), then the
    highest line support wins. Nothing is returned when every candidate fails: an unverified
    frame stays unavailable rather than borrowing a guess."""
    from soccer_analyzer.calibration.homography import camera_inside_pitch, normalize_sign
    from soccer_analyzer.calibration.keypoint_calib import homography_plausible
    from soccer_analyzer.calibration.line_support import line_support
    from soccer_analyzer.calibration.ls_calibrator import spread_score

    if dist_field is None:
        return None
    best = None
    for tag, H in candidates:
        if H is None or not np.all(np.isfinite(H)):
            continue
        H = normalize_sign(H, frame_size)
        if spread_score(H, frame_size, pitch) < spread_floor:  # sliver / smear
            continue
        if not homography_plausible(H, pitch, frame_size=frame_size):  # shear / fold / mirror
            continue
        ls = line_support(H, dist_field, pitch, radius=radius)
        if ls < ls_floor:
            continue
        rank = (not camera_inside_pitch(H, frame_size, pitch.length, pitch.width), ls)
        if best is None or rank > best[0]:
            best = (rank, (H, ls, tag))
    return None if best is None else best[1]


def calibrate_clip_auto(
    detector,
    video_path: str,
    pitch: PitchModel = STANDARD_PITCH,
    stride: int = 2,
    max_frames: Optional[int] = None,
    ls_floor: float = 0.38,
    spread_floor: float = 0.10,
    radius: float = 8.0,
    max_gap: int = 90,
    progress=None,
    reuse_static: bool = True,
    register_views: bool = False,
    start_frame: int = 0,
):
    """Choose a line-supported, geometrically plausible CNN or keypoint mapping.

    Return source-indexed samples and counters. Static reuse compares with an independently
    verified image; registration, when enabled, also needs current line support. Neither
    method fills rejected frames. ``max_gap`` is retained for caller compatibility; the
    automatic path now uses direct reference checks instead of gap-based propagation.
    """
    from soccer_analyzer.calibration.homography import normalize_sign
    from soccer_analyzer.calibration.keypoint_calib import homography_plausible
    from soccer_analyzer.calibration.line_support import line_distance_field, line_support
    from soccer_analyzer.calibration.ls_calibrator import spread_score
    from soccer_analyzer.video import frame_window, seek_capture

    if stride <= 0:
        raise ValueError("stride must be positive")
    frame_limit(0, max_frames)
    with open_capture(video_path) as cap:
        start, stop = frame_window(cap.get(cv2.CAP_PROP_FRAME_COUNT), max_frames, start_frame)
        total = stop - start if stop is not None else 0
        seek_capture(cap, start)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        entries = []
        prev = None
        idx = start
        best_ls = 0.0
        src_count = {"cnn": 0, "keypoint": 0}
        from soccer_analyzer.calibration.camera import CameraReference

        references = []
        registered_anchors = []
        inference_calls = reused = registered = 0
        while True:
            if stop is not None and idx >= stop:
                break
            if not cap.grab():
                break
            if (idx - start) % stride != 0:
                idx += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            g = _gray_small(frame)
            motion = 0.0 if prev is None else float(np.mean(cv2.absdiff(g, prev)))
            prev = g

            Hv = None
            if reuse_static:
                for reference, matrix in reversed(references):
                    if reference.matches(frame):
                        Hv = matrix
                        reused += 1
                        break
            if Hv is None and register_views and registered_anchors:
                distance = line_distance_field(frame)
                if distance is not None:
                    for anchor in reversed(registered_anchors):
                        candidate = anchor.propose(frame)
                        if candidate is not None:
                            candidate = normalize_sign(candidate, (w, h))
                        if (
                            candidate is not None
                            and homography_plausible(candidate, pitch, frame_size=(w, h))
                            and spread_score(candidate, (w, h), pitch) >= spread_floor
                            and line_support(candidate, distance, pitch, radius=radius) >= ls_floor
                        ):
                            Hv = candidate
                            registered += 1
                            break
            try:
                o = None
                if Hv is None:
                    inference_calls += 1  # Failed attempts still consume inference work.
                    o = detector.detect(frame, idx)
            except Exception as exc:
                from soccer_analyzer.resources import is_memory_error

                if is_memory_error(exc):
                    raise
                o = None
            if o is not None:
                # Verified selection (fast-keyframe-pipeline): sign-normalised, camera-consistent,
                # line-supported, preferring a camera beside the field.
                pick = select_calibration(
                    detector_candidates(o, pitch),
                    line_distance_field(frame),
                    (w, h),
                    pitch,
                    ls_floor=ls_floor,
                    spread_floor=spread_floor,
                    radius=radius,
                )
                if pick is not None:
                    Hv, ls, tag = pick
                    src_count[tag] += 1
                    best_ls = max(best_ls, ls)
                    if reuse_static:
                        references.append((CameraReference(frame), Hv))
                        references = references[-4:]
                    if register_views:
                        from soccer_analyzer.calibration.registration import RegisteredAnchor

                        registered_anchors.append(RegisteredAnchor(frame, Hv))
                        registered_anchors = registered_anchors[-3:]
            entries.append({"idx": idx, "Hv": Hv, "motion": motion})
            idx += 1
            if progress is not None and ((idx - start) // stride) % 5 == 0 and total:
                progress(min(1.0, (idx - start) / total))

    # Each rejected sample remains unavailable. Reuse is based on direct reference evidence,
    # not interpolation through a failed calibration or an unchecked camera movement.
    res = {e["idx"]: e["Hv"] for e in entries}
    st = {"frames": len(entries), "covered": sum(h is not None for h in res.values())}
    st.update(
        inference_calls=inference_calls, reused=reused, registered=registered, mode="automatic"
    )
    st["best_ls"] = round(best_ls, 2)
    st["cnn_frames"] = src_count["cnn"]
    st["keypoint_frames"] = src_count["keypoint"]
    return res, st


def calibrate_clip(
    detector,
    video_path: str,
    pitch: PitchModel = STANDARD_PITCH,
    stride: int = 2,
    max_frames: Optional[int] = None,
    static_thresh: float = 2.5,
    max_gap: int = 90,
) -> Tuple[Dict[int, Optional[np.ndarray]], dict]:
    """Returns {frame_idx: homography or None} plus stats."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames:
        total = min(total, max_frames)

    entries: List[dict] = []
    prev = None
    idx = 0
    while True:
        if total and idx >= total:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride != 0:
            idx += 1
            continue
        g = _gray_small(frame)
        motion = 0.0 if prev is None else float(np.mean(cv2.absdiff(g, prev)))
        prev = g
        try:
            o = detector.detect(frame, idx)
        except Exception:
            o = None
        Hv = o.homography if (o is not None and o.is_valid) else None
        entries.append({"idx": idx, "Hv": Hv, "motion": motion})
        idx += 1
    cap.release()

    anchors = [i for i, e in enumerate(entries) if e["Hv"] is not None]
    result: Dict[int, Optional[np.ndarray]] = {e["idx"]: None for e in entries}
    stats = {"anchors": len(anchors), "frames": len(entries), "covered": 0, "moving": 0}
    if not anchors:
        return result, stats

    anchors_arr = np.array(anchors)
    for i, e in enumerate(entries):
        a = int(anchors_arr[np.argmin(np.abs(anchors_arr - i))])
        if abs(a - i) > max_gap:
            continue
        lo, hi = (a, i) if a <= i else (i, a)
        # static if no real motion anywhere between the anchor and this frame
        if all(entries[k]["motion"] < static_thresh for k in range(lo + 1, hi + 1)):
            result[e["idx"]] = entries[a]["Hv"].copy()
            stats["covered"] += 1
        else:
            stats["moving"] += 1
    return result, stats
