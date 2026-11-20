"""
Visual verification for the calibration subsystem.

The gold-standard human check: reproject the known pitch model (lines + circles) back
ONTO the broadcast frame via inv(H). If the drawn lines sit exactly on the real painted
field lines, the calibration is correct. Alongside it we render a top-down minimap showing
the camera's field-of-view footprint and the detected landmarks.

Two modes:
  * render(...)      -> write a side-by-side MP4 (broadcast+overlay | minimap). Headless-safe.
  * interactive(...) -> a cv2 window with a frame scrubber + overlay toggles (needs GUI cv2).

CLI:
    python -m soccer_analyzer.calibration.visualize <video> \
        --pnlcalib-root <PnLCalib> --weights-kp <..SV_kp> --weights-line <..SV_lines> \
        [--smooth] [-m 300] [-o output/calib_overlay.mp4] [--interactive]
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH
from soccer_analyzer.calibration.mapper import PitchMapper, FrameCalibration
from soccer_analyzer.calibration.homography import normalize_sign


GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
OFFPITCH_BAND_M = 5.0  # 2D map: how far outside the lines a player is still drawn


# ---- geometry helpers ----------------------------------------------------------------


def _apply_H(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to Nx2 points -> Nx3 homogeneous result (un-normalized)."""
    p = np.hstack([pts, np.ones((len(pts), 1))])
    return (H @ p.T).T


def _pitch_to_image_polyline(
    pts_m: np.ndarray, H_inv: np.ndarray, w: int, h: int, margin: int = 4000
) -> List[np.ndarray]:
    """Map a polyline of pitch points (meters) into image space via inv(H) = pitch->image.
    Returns a list of runs (each an Nx2 int array) of consecutive on-screen-ish points,
    splitting wherever a point falls behind the camera / past the horizon."""
    out = _apply_H(H_inv, pts_m)
    runs, cur = [], []
    for x, y, wgt in out:
        ok = wgt > 1e-6 and np.isfinite(x) and np.isfinite(y)
        if ok:
            u, v = x / wgt, y / wgt
            ok = (-margin <= u <= w + margin) and (-margin <= v <= h + margin)
        if ok:
            cur.append((u, v))
        elif cur:
            runs.append(np.array(cur, np.int32))
            cur = []
    if cur:
        runs.append(np.array(cur, np.int32))
    return runs


def _sample_segment(p0, p1, step=1.0) -> np.ndarray:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.linalg.norm(p1 - p0) / step))
    return np.array([p0 + t * (p1 - p0) for t in np.linspace(0, 1, n)])


def _sample_circle(cx, cy, r, clip=None, step=0.5) -> np.ndarray:
    n = max(16, int(2 * np.pi * r / step))
    pts = []
    for a in np.linspace(0, 2 * np.pi, n):
        x, y = cx + r * np.cos(a), cy + r * np.sin(a)
        if clip is not None:
            lo, hi = clip
            if lo is not None and x < lo:
                continue
            if hi is not None and x > hi:
                continue
        pts.append((x, y))
    return np.array(pts) if pts else np.empty((0, 2))


# ---- broadcast overlay ---------------------------------------------------------------


def _anchors_in_pitch(obs, H) -> Optional[np.ndarray]:
    """Detected keypoints mapped into pitch meters — the locations we actually measured."""
    if obs is None or not getattr(obs, "keypoints", None):
        return None
    pts = np.array(list(obs.keypoints.values()), dtype=np.float64)
    m = _apply_H(H, pts)
    ok = np.abs(m[:, 2]) > 1e-6
    return (m[ok, :2] / m[ok, 2:3]) if ok.any() else None


def _conf_style(d_m: float):
    """Map distance-to-nearest-landmark (meters) -> (color, thickness) or None to hide.
    Bright solid where we measured; faded where extrapolated; hidden where pure guess."""
    if d_m < 9:
        return (0, 255, 0), 3
    if d_m < 18:
        return (0, 205, 70), 2
    if d_m < 30:
        return (35, 130, 35), 1
    return None  # too far from any landmark -> don't assert it


def _draw_pitch_polyline_conf(out, pts_m, H_inv, anchors, w, h):
    """Draw a pitch polyline, colouring each segment by its distance from detected
    landmarks (when anchors is None, draw a flat medium green = no confidence info)."""
    img = _apply_H(H_inv, pts_m)
    # Samples are <= 1 m apart, so a segment spanning a large part of the frame lies at the
    # camera plane (e.g. a touchline just behind/below the camera): its projection is a streak
    # across the image, not a visible marking. Skip such segments.
    max_seg = 0.25 * (w + h)
    prev = None
    for i, (x, y, wgt) in enumerate(img):
        on = wgt > 1e-6 and np.isfinite(x) and np.isfinite(y)
        cur = (x / wgt, y / wgt) if on else None
        if cur is not None and not (-4000 <= cur[0] <= w + 4000 and -4000 <= cur[1] <= h + 4000):
            cur = None
        if (
            prev is not None
            and cur is not None
            and np.hypot(cur[0] - prev[0], cur[1] - prev[1]) > max_seg
        ):
            prev = cur
            continue
        if prev is not None and cur is not None:
            if anchors is None:
                style = ((0, 205, 70), 2)
            else:
                d = float(np.min(np.linalg.norm(anchors - pts_m[i], axis=1)))
                style = _conf_style(d)
            if style is not None:
                color, th = style
                cv2.line(
                    out,
                    (int(prev[0]), int(prev[1])),
                    (int(cur[0]), int(cur[1])),
                    color,
                    th,
                    cv2.LINE_AA,
                )
        prev = cur


def draw_calibration_overlay(
    frame: np.ndarray,
    calib: FrameCalibration,
    pitch: PitchModel = STANDARD_PITCH,
    obs=None,
    fade_by_confidence: bool = True,
    show_hud: bool = True,
) -> np.ndarray:
    """Reproject the pitch model onto the frame + draw detected keypoints + a HUD.
    Lines fade/disappear far from detected landmarks. App previews omit the diagnostic
    HUD because their review instructions and source frame are shown beside the image.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    valid = calib is not None and calib.source != "none" and np.all(np.isfinite(calib.homography))

    anchors = None
    if valid:
        try:
            # Either sign of H is the same mapping; drawing tests w > 0, so normalise first.
            H_inv = np.linalg.inv(normalize_sign(calib.homography, (w, h)))  # pitch(m) -> image(px)
        except np.linalg.LinAlgError:
            H_inv = None
        if H_inv is not None:
            anchors = _anchors_in_pitch(obs, calib.homography) if fade_by_confidence else None
            for p0, p1 in pitch.lines():
                _draw_pitch_polyline_conf(out, _sample_segment(p0, p1), H_inv, anchors, w, h)
            for cx, cy, r, clip in pitch.circles():
                samp = _sample_circle(cx, cy, r, clip)
                if len(samp):
                    _draw_pitch_polyline_conf(out, samp, H_inv, anchors, w, h)

    # detected landmarks (only fresh on keyframes)
    if obs is not None and getattr(obs, "keypoints", None):
        for u, v in obs.keypoints.values():
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(out, (int(u), int(v)), 6, YELLOW, -1)
                cv2.circle(out, (int(u), int(v)), 6, (0, 0, 0), 1)

    if show_hud:
        _draw_hud(out, calib, landmark_fading=anchors is not None)
    return out


def _draw_hud(img: np.ndarray, calib: FrameCalibration, landmark_fading: bool = False):
    h, w = img.shape[:2]
    bar = img.copy()
    cv2.rectangle(bar, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.45, img, 0.55, 0, img)

    def put_line(text, y, scale, color, thickness):
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        scale *= min(1.0, max(1, w - 30) / max(1, text_width))
        cv2.putText(
            img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
        )

    if calib is None or calib.source == "none":
        put_line("CALIBRATION OFFLINE", 45, 1.1, RED, 2)
        return
    rep = f"{calib_reproj(calib):.1f}px" if calib_reproj(calib) is not None else "n/a"
    col = GREEN if calib.confidence > 0.5 else YELLOW
    txt = (
        f"frame {calib.frame_idx}  |  {calib.source.upper()}  |  conf {calib.confidence:.2f}"
        f"  |  reproj {rep}  |  kp {calib.n_landmarks}  |  age {calib.frames_since_detection}"
    )
    put_line(txt, 30, 0.8, col, 2)
    legend = (
        "bright green = near landmarks | faded = extrapolated"
        if landmark_fading
        else "green = estimated pitch projection; not all markings were observed"
    )
    put_line(legend, 58, 0.6, WHITE, 1)


def calib_reproj(calib: FrameCalibration) -> Optional[float]:
    return getattr(calib, "reproj_err", None)


# ---- top-down minimap ----------------------------------------------------------------


def draw_fov_minimap(
    calib: FrameCalibration,
    pitch: PitchModel = STANDARD_PITCH,
    size: Tuple[int, int] = (640, 440),
    frame_size=None,
    obs=None,
) -> np.ndarray:
    mw, mh = size
    img = np.full((mh, mw, 3), (40, 90, 40), np.uint8)
    margin = 24
    sx = (mw - 2 * margin) / pitch.length
    sy = (mh - 2 * margin) / pitch.width

    def to_px(x, y):
        return int(margin + x * sx), int(margin + y * sy)

    for p0, p1 in pitch.lines():
        cv2.line(img, to_px(*p0), to_px(*p1), WHITE, 1, cv2.LINE_AA)
    for cx, cy, r, clip in pitch.circles():
        samp = _sample_circle(cx, cy, r, clip, step=0.5)
        for i in range(len(samp) - 1):
            cv2.line(img, to_px(*samp[i]), to_px(*samp[i + 1]), WHITE, 1, cv2.LINE_AA)

    valid = calib is not None and calib.source != "none" and np.all(np.isfinite(calib.homography))
    if valid and frame_size is not None:
        fw, fh = frame_size
        # sample a grid across the image, map -> pitch, keep on-pitch points -> FOV hull
        gx, gy = np.meshgrid(np.linspace(0, fw, 16), np.linspace(0, fh, 16))
        grid = np.stack([gx.ravel(), gy.ravel()], 1)
        mapped = _apply_H(calib.homography, grid)
        on = []
        for x, y, wgt in mapped:
            if wgt > 1e-6:
                px, py = x / wgt, y / wgt
                if -10 <= px <= pitch.length + 10 and -10 <= py <= pitch.width + 10:
                    on.append(to_px(np.clip(px, 0, pitch.length), np.clip(py, 0, pitch.width)))
        if len(on) >= 3:
            hull = cv2.convexHull(np.array(on, np.int32))
            overlay = img.copy()
            cv2.fillConvexPoly(overlay, hull, (0, 140, 255))
            cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
            cv2.polylines(img, [hull], True, (0, 165, 255), 2, cv2.LINE_AA)

    # detected landmarks mapped onto the pitch
    if valid and obs is not None and getattr(obs, "keypoints", None):
        for u, v in obs.keypoints.values():
            m = _apply_H(calib.homography, np.array([[u, v]]))[0]
            if m[2] > 1e-6:
                px, py = m[0] / m[2], m[1] / m[2]
                if 0 <= px <= pitch.length and 0 <= py <= pitch.width:
                    cv2.circle(img, to_px(px, py), 5, YELLOW, -1)

    cv2.putText(
        img,
        "TOP-DOWN  (orange = camera view)",
        (12, mh - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    return img


# ---- composition + render ------------------------------------------------------------


def draw_topdown(
    pitch: PitchModel,
    calib=None,
    players=None,
    ball=None,
    coverage=None,
    size=(900, 620),
    frame_size=None,
    team_colors=None,
    ball_measured: bool = True,
) -> np.ndarray:
    """The 2D top-down map: the FULL estimated field (always drawn) + players + ball +
    camera footprint + optional coverage shading (regions measured over the clip).

    Players are (x_m, y_m, team[, measured]) tuples. Missing measurement flags retain
    legacy filled dots; explicit False draws a hollow ring for interpolated positions.
    Players up to `OFFPITCH_BAND_M` outside the lines (a throw-in taker, a corner taker) are
    drawn as hollow rings in the margin beside the line; anything further out is not drawn.
    """
    mw, mh = size
    img = np.full((mh, mw, 3), (40, 90, 40), np.uint8)
    margin = 42  # ~5.4 m at the default size, so the off-pitch band fits beside the lines
    sx = (mw - 2 * margin) / pitch.length
    sy = (mh - 2 * margin) / pitch.width

    def to_px(x, y):
        return int(margin + x * sx), int(margin + y * sy)

    if coverage is not None and coverage.max() > 0:  # measured-over-time shading
        cov = cv2.resize((coverage * 120).astype(np.uint8), (mw - 2 * margin, mh - 2 * margin))
        roi = img[margin : mh - margin, margin : mw - margin]
        roi[..., 1] = np.clip(roi[..., 1].astype(int) + cov, 0, 255).astype(np.uint8)

    for p0, p1 in pitch.lines():
        cv2.line(img, to_px(*p0), to_px(*p1), WHITE, 1, cv2.LINE_AA)
    for cx, cy, r, clip in pitch.circles():
        s = _sample_circle(cx, cy, r, clip, 0.5)
        for i in range(len(s) - 1):
            cv2.line(img, to_px(*s[i]), to_px(*s[i + 1]), WHITE, 1, cv2.LINE_AA)

    if calib is not None and frame_size is not None and getattr(calib, "source", "none") != "none":
        fw, fh = frame_size
        # FIELD OF VIEW = which PITCH cells the camera actually captures. Reproject a dense PITCH
        # grid into the image (pitch->image via inv H) and keep the cells whose image point is
        # in-frame. (Sampling a uniform IMAGE grid instead under-samples the far field near the
        # horizon, so the far side was wrongly left unshaded.)
        try:
            H_inv = np.linalg.inv(normalize_sign(calib.homography, (fw, fh)))
        except np.linalg.LinAlgError:
            H_inv = None
        if H_inv is not None:
            gx, gy = np.meshgrid(np.linspace(0, pitch.length, 90), np.linspace(0, pitch.width, 60))
            q = H_inv @ np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], 0)
            ok = q[2] > 1e-6
            u = np.full(gx.size, -1.0)
            v = np.full(gx.size, -1.0)
            u[ok] = q[0, ok] / q[2, ok]
            v[ok] = q[1, ok] / q[2, ok]
            vis = ok & (u >= 0) & (u < fw) & (v >= 0) & (v < fh)
            pts = [to_px(x, y) for x, y in zip(gx.ravel()[vis], gy.ravel()[vis])]
            if len(pts) >= 3:
                hull = cv2.convexHull(np.array(pts, np.int32))  # camera view of a plane is convex
                ov = img.copy()
                cv2.fillConvexPoly(ov, hull, (0, 130, 235))
                cv2.addWeighted(ov, 0.18, img, 0.82, 0, img)
                cv2.polylines(img, [hull], True, (0, 165, 255), 1, cv2.LINE_AA)

    team_col = {
        0: (40, 40, 235),
        1: (235, 120, 40),
        -1: (170, 170, 170),
        -2: (0, 255, 255),
        -3: (255, 0, 255),
    }
    if team_colors:
        team_col[0], team_col[1] = tuple(team_colors[0]), tuple(team_colors[1])
    band = OFFPITCH_BAND_M
    for p in players or []:
        x, y, team = p[:3]
        measured = len(p) < 4 or p[3]
        if not (-band <= x <= pitch.length + band and -band <= y <= pitch.width + band):
            continue
        r = 6 if team in (-2, -3) else 7  # officials a touch smaller
        u, v = to_px(x, y)
        px = (min(max(u, r + 1), mw - r - 2), min(max(v, r + 1), mh - r - 2))  # keep on canvas
        col = team_col.get(team, (170, 170, 170))
        if 0 <= x <= pitch.length and 0 <= y <= pitch.width:
            # Measured: filled dot; interpolated (gap-filled): hollow ring.
            cv2.circle(img, px, r, col, -1 if measured else 2)
            if measured:
                cv2.circle(img, px, r, (0, 0, 0), 1)
        else:  # just off the pitch (throw-in / corner): hollow ring in the team colour
            cv2.circle(img, px, r, (0, 0, 0), 4)
            cv2.circle(img, px, r, col, 2)
    if ball is not None and 0 <= ball[0] <= pitch.length and 0 <= ball[1] <= pitch.width:
        cv2.circle(img, to_px(*ball), 5, (255, 255, 255), -1 if ball_measured else 2)
        if ball_measured:
            cv2.circle(img, to_px(*ball), 5, (0, 0, 0), 1)
    return img


def compose(overlay: np.ndarray, minimap: np.ndarray, out_width: int = 1600) -> np.ndarray:
    h, w = overlay.shape[:2]
    left_w = int(out_width * 0.66)
    left_h = int(h * (left_w / w))
    left = cv2.resize(overlay, (left_w, left_h))
    right_w = out_width - left_w
    mh = minimap.shape[0] * right_w // minimap.shape[1]
    right = cv2.resize(minimap, (right_w, mh))
    canvas = np.full((max(left_h, mh), out_width, 3), 20, np.uint8)
    canvas[:left_h, :left_w] = left
    canvas[:mh, left_w : left_w + right_w] = right
    return canvas


def render(
    video_path: str,
    mapper: PitchMapper,
    out_path: str = "output/calib_overlay.mp4",
    pitch: PitchModel = STANDARD_PITCH,
    max_frames: Optional[int] = None,
    out_width: int = 1600,
    fps: float = 25.0,
    undistorter=None,
) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    writer = None
    idx = 0
    while True:
        if max_frames is not None and idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if undistorter is not None:
            frame = undistorter.frame(frame)  # Stage 0: undistort up front (no-op if ~0)
        calib = mapper.update(frame, idx)
        obs = mapper.last_observation if calib.source == "detect" else None
        overlay = draw_calibration_overlay(frame, calib, pitch, obs=obs)
        minimap = draw_fov_minimap(calib, pitch, frame_size=(fw, fh), obs=obs)
        canvas = compose(overlay, minimap, out_width)
        if writer is None:
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                src_fps,
                (canvas.shape[1], canvas.shape[0]),
            )
        writer.write(canvas)
        idx += 1
    if writer is not None:
        writer.release()
    cap.release()
    print(f"wrote {idx} frames -> {out_path}")
    return out_path


def interactive(
    video_path: str,
    mapper: PitchMapper,
    pitch: PitchModel = STANDARD_PITCH,
    max_frames: int = 600,
    out_width: int = 1600,
    undistorter=None,
):
    """Scrubbable cv2 window. Needs a GUI build of opencv (not -headless). Pre-computes
    calibration for up to max_frames, then lets you scrub with a trackbar."""
    cap = cv2.VideoCapture(video_path)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames, calibs, observations = [], [], []
    idx = 0
    while idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if undistorter is not None:
            frame = undistorter.frame(frame)
        calib = mapper.update(frame, idx)
        frames.append(frame)
        calibs.append(calib)
        observations.append(mapper.last_observation if calib.source == "detect" else None)
        idx += 1
    cap.release()
    if not frames:
        print("no frames")
        return

    win = "calibration verifier  (q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("frame", win, 0, len(frames) - 1, lambda v: None)
    while True:
        i = cv2.getTrackbarPos("frame", win)
        overlay = draw_calibration_overlay(frames[i], calibs[i], pitch, obs=observations[i])
        minimap = draw_fov_minimap(calibs[i], pitch, frame_size=(fw, fh), obs=observations[i])
        cv2.imshow(win, compose(overlay, minimap, out_width))
        if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def main():
    import argparse
    from soccer_analyzer.calibration.field_detector import PnLCalibFieldDetector

    p = argparse.ArgumentParser(description="Calibration visual verifier")
    p.add_argument("video")
    p.add_argument("--pnlcalib-root", default=os.environ.get("PNLCALIB_ROOT"))
    p.add_argument("--weights-kp", default=None)
    p.add_argument("--weights-line", default=None)
    p.add_argument(
        "-c", "--calibration", default=None, help="use a cached homography bridge instead"
    )
    p.add_argument("--smooth", action="store_true", help="use the Phase B smoother")
    p.add_argument(
        "--undistort",
        action="store_true",
        help="auto-estimate + correct lens distortion (Stage 0; no-op if ~zero)",
    )
    p.add_argument("--keyframe-interval", type=int, default=10)
    p.add_argument("-m", "--max-frames", type=int, default=300)
    p.add_argument("-o", "--output", default="output/calib_overlay.mp4")
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()

    if args.calibration:
        mapper = PitchMapper(
            calibration_path=args.calibration, keyframe_interval=args.keyframe_interval
        )
    else:
        root = args.pnlcalib_root
        wk = args.weights_kp or (os.path.join(root, "SV_kp") if root else None)
        wl = args.weights_line or (os.path.join(root, "SV_lines") if root else None)
        det = PnLCalibFieldDetector(pnlcalib_root=root, weights_kp=wk, weights_line=wl)
        mapper = PitchMapper(
            detector=det, keyframe_interval=args.keyframe_interval, smooth=args.smooth
        )

    undistorter = None
    if args.undistort and not args.calibration:
        from soccer_analyzer.calibration.distortion import estimate_distortion, Undistorter

        print("Estimating lens distortion from the footage (plumb-line over many frames)...")
        model = estimate_distortion(det, args.video, n_frames=30)
        print(
            f"  k1={model.dist[0]:+.4f}  line_samples={model.n_samples}  "
            f"straightening={model.improvement * 100:.0f}%  -> correction {'ON' if model.enabled else 'OFF (no-op)'}"
        )
        undistorter = Undistorter(model)

    if args.interactive:
        interactive(args.video, mapper, max_frames=args.max_frames, undistorter=undistorter)
    else:
        render(
            args.video,
            mapper,
            out_path=args.output,
            max_frames=args.max_frames,
            undistorter=undistorter,
        )


if __name__ == "__main__":
    main()
