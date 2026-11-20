"""VLM field-marking calibration: keyframe camera fits from named soccer markings.

A vision model lists every visible SOCCER marking point (pitch corners, 18-/6-yard box corners,
penalty spots, arc points, goal posts, halfway ends, centre spot / circle points) with a
confidence (`PROMPT`). `fit_camera` fits a CONSTRAINED pinhole camera to them (principal point
at the image centre, no distortion; focal, rotation, position free; pitch length/width free but
bounded) with confidence-weighted RANSAC over the named correspondences, and
`refine_with_paint` snaps the model lines onto the painted lines.

`calibrate_window` is the whole-window entry point used by FastPipeline: each keyframe is fitted
from its OWN answer (a keyframe whose answer is too thin is fitted from the answers within
+-window_s, and remembers which answers it used); keyframes are related by DIRECT feature
registration of the two images (`register_pair`; the chained frame-to-frame motion only across
<= 8 render frames), and a keyframe fit is kept only when a fit built from DIFFERENT answers
agrees with it (consensus). Only then are confirmed fits pooled. Every render frame is carried
from its nearest kept keyframe and VERIFIED against its own paint mask. A frame that fails stays
unavailable; nothing is filled in.

Measured (lab 2026-10-13, held-out reference frames, cached gpt-6-sol/luna answers, CPU replay
`integrate/fix/consensus_exp.py`): night clip 95.6-99.4% of render frames at 0.50-0.70 m on the
test references under four perturbations (lab vs product camera motion x JPEG vs decoded frames).
Multi-sport clip: 0% under all four - only 2 of 6 keyframe answers hold >= 5 confident ground
markings, so no two fits from disjoint answers exist; it fails safe (unavailable). The previous
pooled-before-consensus rule confirmed it at 2.0 m on one motion estimate and 4.3 m / 0% on
another. Clips whose views show only goal posts or a goal mouth get no consensus either.

Pitch coordinates: corner origin, X along the length from the LEFT goal line (0..L), Y from the
FAR touchline (0..W), metres (the repo's PitchModel convention), with the FITTED dims.
"""

from __future__ import annotations

import itertools

import cv2
import numpy as np
from scipy.optimize import least_squares

from soccer_analyzer.calibration import refcore as rc

DIMS0 = (105.0, 68.0)
DIMS_BOUNDS = ((90.0, 120.0), (55.0, 80.0))

PROMPT = """You are calibrating a camera that films a soccer pitch. The image is {W}x{H} pixels (x to the right, y down).
List every SOCCER pitch marking point you can see and its pixel position.
Only soccer markings: ignore American-football yard lines, numbers and hash marks, lacrosse lines and other sports' lines (on multi-sport fields the soccer lines are often a different colour, e.g. yellow or red).
Naming: 'l'/'r' prefix = the LEFT or RIGHT END of the pitch as seen from this camera (which goal); 'near'/'far' = the touchline side closer to / farther from the camera.
Allowed names (s = l or r):
- s_corner_near, s_corner_far: pitch corner (goal line meets the touchline)
- s_pbox_near_gl, s_pbox_far_gl: where an 18-yard (penalty) box side line meets the goal line
- s_pbox_near, s_pbox_far: the two FRONT corners of the 18-yard box (on the field side)
- s_gbox_near_gl, s_gbox_far_gl, s_gbox_near, s_gbox_far: the same four corners of the 6-yard (goal) box
- s_spot: penalty spot
- s_arc_near, s_arc_far: where the penalty arc (the 'D') meets the 18-yard box front line
- s_arc_top: the point of the penalty arc farthest from the goal
- s_post_near_base, s_post_far_base: where each goal post stands on the ground
- s_post_near_top, s_post_far_top: top of each goal post (crossbar ends)
- half_near, half_far: where the halfway line meets the near / far touchline
- centre: centre spot
- circle_near, circle_far: where the centre circle crosses the halfway line (near / far)
Give a point only if it is visible, or can be placed precisely as the intersection of two visible lines (e.g. a corner hidden behind a player). confidence 0-1 = how sure you are of BOTH its identity and its position (within about 1% of the image width).
Reply with JSON only: {"soccer_line_colour": "white|yellow|red|blue|other", "markings": [{"name": "...", "x": 0, "y": 0, "confidence": 0.0}]}"""


# ------------------------------------------------------------------------------- pitch model
def model_points(L, W):
    P = rc.model_points(L, W)
    P["l_arc_top"] = np.array([rc.SPOT + rc.R_CIRC, W / 2, 0.0])
    P["r_arc_top"] = np.array([L - rc.SPOT - rc.R_CIRC, W / 2, 0.0])
    return P


NAMES = set(model_points(*DIMS0))


def _swap(name):
    if name[:2] == "l_":
        return "r_" + name[2:]
    if name[:2] == "r_":
        return "l_" + name[2:]
    return name


def parse_markings(resp, scale=(1.0, 1.0)):
    """VLM reply -> [(name, x, y, conf)] in ORIGINAL pixels (dedup by name: highest conf)."""
    out = {}
    if not isinstance(resp, dict):
        return []
    sx, sy = resp.get("_scale", scale)
    for m in resp.get("markings", []) or []:
        try:
            n = str(m["name"]).strip().lower()
            x, y, c = float(m["x"]) * sx, float(m["y"]) * sy, float(m.get("confidence", 0.5))
        except (KeyError, TypeError, ValueError):
            continue
        if n not in NAMES or not (np.isfinite(x) and np.isfinite(y)):
            continue
        if n not in out or c > out[n][3]:
            out[n] = (n, x, y, c)
    return list(out.values())


# ------------------------------------------------------------------------------------ fitting
def _tri_area(a, b, c):
    return 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _ok_quad(q, min_area):
    return min(_tri_area(*t) for t in itertools.combinations(q, 3)) > min_area


def _ransac_ground(names, img, conf, P, tau, max_sets=4000, rng=0):
    """Best inlier set of ground correspondences under one plane homography (MSAC, conf-weighted)."""
    n = len(names)
    W3 = np.array([P[k] for k in names])
    world = W3[:, :2]
    best = (None, -1.0)
    combos = list(itertools.combinations(range(n), 4))
    if len(combos) > max_sets:
        R = np.random.default_rng(rng)
        combos = [combos[i] for i in R.choice(len(combos), max_sets, replace=False)]
    for c in combos:
        c = list(c)
        if not _ok_quad(world[c], 4.0) or not _ok_quad(img[c], (0.01 * tau) ** 2 * 50):
            continue
        Hm = cv2.getPerspectiveTransform(world[c].astype(np.float32), img[c].astype(np.float32))
        q = np.c_[world, np.ones(n)] @ Hm.T
        if np.any(q[:, 2] * np.sign(q[c, 2].mean()) <= 0):
            e = np.full(n, np.inf)
        else:
            e = np.linalg.norm(q[:, :2] / q[:, 2:3] - img, axis=1)
        s = float(np.sum(conf * np.clip(1 - (e / tau) ** 2, 0, None)))
        if s > best[1]:
            best = (e < tau, s)
    return best


def _valid_cam(cam, dims, w):
    C = np.asarray(cam["C"])
    return (-80.0 < C[2] < -1.0) and C[1] > 0.5 * dims[1] and 0.25 * w < cam["f"] < 20 * w


def fit_camera(
    marks,
    w,
    h,
    conf_min=0.5,
    tau_frac=0.025,
    dims0=DIMS0,
    fit_dims=True,
    min_inliers=5,
    dims_sigma=8.0,
):
    """[(name,x,y,conf)] -> best constrained-pinhole fit dict or None.
    Tries the labels as given and with the l/r ends swapped (the model often confuses which
    END it sees); the mirrored hypothesis needs a camera under the ground and is rejected."""
    marks = [m for m in marks if m[3] >= conf_min]
    if len(marks) < 4:
        return None
    tau = tau_frac * w
    best = None
    for swap in (False, True):
        mk = [(_swap(n) if swap else n, x, y, c) for n, x, y, c in marks]
        P0 = model_points(*dims0)
        g = [m for m in mk if P0[m[0]][2] == 0.0]
        t = [m for m in mk if P0[m[0]][2] != 0.0]
        if len(g) < 4:
            continue
        names = [m[0] for m in g]
        img = np.array([[m[1], m[2]] for m in g])
        conf = np.array([m[3] for m in g])
        inl, score = _ransac_ground(names, img, conf, P0, tau)
        if inl is None or inl.sum() < 4:
            continue
        use = [g[i] for i in range(len(g)) if inl[i]] + t
        cam0, _ = rc.init_camera(
            np.array([P0[m[0]] for m in use]), np.array([[m[1], m[2]] for m in use]), w, h
        )
        if cam0 is None:
            continue
        res = _refine_points(cam0, use, w, h, dims0, fit_dims, dims_sigma, tau)
        if res is None:
            continue
        cam, dims, e = res
        # drop tops / points the full model rejects, refit once
        keep = [m for m, ei in zip(use, e) if ei < tau]
        if len(keep) < len(use) and len(keep) >= 4:
            res2 = _refine_points(cam, keep, w, h, dims, fit_dims, dims_sigma, tau)
            if res2 is not None:
                cam, dims, e = res2
                use = keep
        if not _valid_cam(cam, dims, w):
            continue
        ninl = int(np.sum(e < tau))
        sc = float(np.sum([m[3] * max(0.0, 1 - (ei / tau) ** 2) for m, ei in zip(use, e)]))
        cand = dict(
            camera=cam,
            pitch_dims=tuple(float(v) for v in dims),
            swapped=swap,
            inliers=[m[0] for m in use],
            n_inliers=ninl,
            n_marks=len(marks),
            rms_px=float(np.sqrt(np.mean(np.square(e)))),
            score=sc,
            markings=mk,
        )
        if best is None or sc > best["score"]:
            best = cand
    if best is None or best["n_inliers"] < min_inliers:
        return None
    best["H_img2pitch"] = rc.H_img2pitch(best["camera"])
    best["confidence"] = float(min(1.0, best["score"] / 8.0))
    return best


def _refine_points(cam0, use, w, h, dims0, fit_dims, dims_sigma, tau):
    names = [m[0] for m in use]
    q = np.array([[m[1], m[2]] for m in use])
    wt = np.sqrt(np.array([max(0.05, m[3]) for m in use]))
    # dims observable only if a point off the boxes' symmetric frame constrains them
    usesL = (
        any(n.startswith(("half", "centre", "circle")) for n in names)
        and any(n[:2] in ("l_", "r_") for n in names)
        or (any(n.startswith("l_") for n in names) and any(n.startswith("r_") for n in names))
    )
    usesW = any("corner" in n or n.startswith("half") for n in names)
    fd = (fit_dims and usesL, fit_dims and usesW)
    s_px = 0.3 * tau  # a dims_sigma deviation from the prior costs like this many pixels

    def unpack(x):
        cam = rc.cam_from_params(x[:7], w / 2, h / 2)
        L = x[7] if fd[0] else dims0[0]
        W = x[8] if fd[1] else dims0[1]
        return cam, (L, W)

    def resid(x):
        cam, (L, W) = unpack(x)
        P = model_points(L, W)
        p, z = rc.project(cam, np.array([P[n] for n in names]))
        r = ((p - q) * wt[:, None]).ravel()
        pri = [(L - DIMS0[0]) / dims_sigma * s_px, (W - DIMS0[1]) / dims_sigma * s_px]
        return np.r_[r, pri, np.where(z > 0, 0.0, 1e3)]

    x0 = np.r_[rc.params_from_cam(cam0), dims0]
    lo = np.r_[[-np.inf] * 7, DIMS_BOUNDS[0][0], DIMS_BOUNDS[1][0]]
    hi = np.r_[[np.inf] * 7, DIMS_BOUNDS[0][1], DIMS_BOUNDS[1][1]]
    x0[7] = np.clip(x0[7], lo[7] + 1e-6, hi[7] - 1e-6)
    x0[8] = np.clip(x0[8], lo[8] + 1e-6, hi[8] - 1e-6)
    try:
        r = least_squares(
            resid, x0, bounds=(lo, hi), loss="soft_l1", f_scale=0.3 * tau, max_nfev=300
        )
    except ValueError:
        return None
    cam, dims = unpack(r.x)
    P = model_points(*dims)
    p, z = rc.project(cam, np.array([P[n] for n in names]))
    if np.any(z <= 0):
        return None
    return cam, dims, np.linalg.norm(p - q, axis=1)


# ---------------------------------------------------------------------------- paint refinement
SNAP_LINES = tuple(n for n in rc.GROUND_LINES)
SNAP_CIRCLES = ("centre_circle", "l_arc", "r_arc")


def _visible_rough(cam, dims, w, h, margin=12):
    L, W = dims
    out = []
    for name, polys in rc.marking_polylines(cam, L, W, w, h, include_goals=False).items():
        kind = "circle" if name in SNAP_CIRCLES else "line"
        for qq in polys:
            m = (
                (qq[:, 0] > margin)
                & (qq[:, 0] < w - margin)
                & (qq[:, 1] > margin)
                & (qq[:, 1] < h - margin)
            )
            if m.sum() < 2:
                continue
            qq = qq[m]
            if np.linalg.norm(qq[-1] - qq[0]) < 0.04 * w:
                continue
            k = np.linspace(0, len(qq) - 1, 2 if kind == "line" else min(40, len(qq))).astype(int)
            out.append((name, kind, qq[k]))
    return out


def paint_response(img, colour="white"):
    mode = colour if colour in ("white", "yellow", "red") else "white"
    h, w = img.shape[:2]
    k = 15 if w <= 2000 else 25
    return rc.response_map(img, mode, k)


def refine_with_paint(img, fit, colour="white", rads=(24, 10, 6), min_lines=2, resp=None):
    """Snap the fitted camera's projected soccer lines onto the paint and refit (constrained
    pinhole, points kept as weak priors). Returns an updated fit dict, or the input fit when the
    paint does not support a refinement (then refined=False)."""
    h, w = img.shape[:2]
    resp = paint_response(img, colour) if resp is None else resp
    s = w / 1920.0
    cam, dims = fit["camera"], fit["pitch_dims"]
    pts = [(m[0], (m[1], m[2]), 0.05 * m[3]) for m in fit["markings"] if m[0] in fit["inliers"]]
    out = dict(fit, refined=False)
    for rad in rads:
        lines, circs = [], []
        for name, kind, rough in _visible_rough(cam, dims, w, h):
            if kind == "line":
                q, f, st = rc.snap_line(resp, rough, rad=int(rad * s), min_resp=6.0, tol=1.5 * s)
                if f is not None and len(q) >= 6 and st.get("frac", 0) > 0.3:
                    lines.append((name, q))
            else:
                q, st = rc.snap_curve(resp, rough, rad=int(rad * s), min_resp=6.0)
                if len(q) >= 8:
                    circs.append((name, q))
        if len(lines) + len(circs) < min_lines:
            return out
        fo = dict(cam0=cam, points=pts, lines=lines, circles=circs)
        try:
            cams, dd, stats, r = rc.fit_frames(
                [fo], w, h, list(dims), fit_dims=(False, False), loss="soft_l1", f_scale=2.0 * s
            )
        except Exception:
            return out
        c2 = cams[0]
        if not _valid_cam(c2, dd, w):
            return out
        cam = c2
        out.update(
            camera=cam,
            pitch_dims=tuple(dd),
            refined=True,
            n_lines=len(lines),
            n_circles=len(circs),
            paint_median_px=stats[0]["median"],
        )
    out["H_img2pitch"] = rc.H_img2pitch(cam)
    return out


# ------------------------------------------------------------------------------- public API
def ask_markings(frame_bgr, ask_fn, prompt=PROMPT):
    """-> ([(name, x, y, conf)] in original pixels, soccer line colour)."""
    resp = ask_fn(frame_bgr, prompt)
    return parse_markings(resp), str((resp or {}).get("soccer_line_colour", "white")).lower()


def keyframe_from_marks(
    frame_bgr, marks, colour="white", conf_min=0.5, refine=True, tau_frac=0.025
):
    h, w = frame_bgr.shape[:2]
    fit = fit_camera(marks, w, h, conf_min=conf_min, tau_frac=tau_frac)
    if fit is None:
        return None
    fit["colour"] = colour
    if refine:
        fit = refine_with_paint(frame_bgr, fit, colour)
    fit["ls"] = line_support(frame_bgr, fit["H_img2pitch"], fit["pitch_dims"], colour)
    return fit


class MotionChain:
    """Chained camera motion over the render frames. G[f] maps full-res pixels of the previous
    render frame to frame f (None = motion failed -> a new segment starts)."""

    def __init__(self, render_frames, G):
        self.frames = list(render_frames)
        self.pos = {f: i for i, f in enumerate(self.frames)}
        n = len(self.frames)
        self.seg, self.C = (
            [0] * n,
            [np.eye(3)] * n,
        )  # C[i]: pixel in frame i -> segment-origin pixel
        for i in range(1, n):
            g = G.get(self.frames[i])
            if g is None or not np.all(np.isfinite(g)):
                self.seg[i], self.C[i] = self.seg[i - 1] + 1, np.eye(3)
            else:
                self.seg[i] = self.seg[i - 1]
                self.C[i] = self.C[i - 1] @ np.linalg.inv(np.asarray(g, float))

    def transfer(self, a, b):
        """3x3 pixel map frame a -> frame b, or None (different segment / not a render frame)."""
        i, j = self.pos.get(a), self.pos.get(b)
        if i is None or j is None or self.seg[i] != self.seg[j]:
            return None
        T = np.linalg.inv(self.C[j]) @ self.C[i]
        return T / T[2, 2]


def pool_markings(chain, marks_by_frame, k, window_s=3.0, fps=30.0, transfers=None):
    """Markings of keyframes within +-window_s of k, mapped into frame k. `transfers` {j: 3x3
    pixel map j -> k} (e.g. direct `register_pair` maps) replaces the motion chain when given."""
    out = []
    for j, marks in marks_by_frame.items():
        if abs(j - k) > window_s * fps:
            continue
        if j == k:
            T = np.eye(3)
        elif transfers is not None:
            T = transfers.get(j)
        else:
            T = chain.transfer(j, k)
        if T is None:
            continue
        for n, x, y, c in marks:
            q = T @ np.array([x, y, 1.0])
            out.append((n, q[0] / q[2], q[1] / q[2], c))
    return out


# ------------------------------------------------------------------ direct keyframe registration
REG_WIDTH = 960


def _reg_features(img, width=REG_WIDTH):
    """SIFT keypoints/descriptors on the turf (HSV green, closed over the painted lines) of a
    `width`-px copy; the whole frame when too little turf is found. Static screen overlays and
    stands are mostly outside the turf, so they cannot pull the registration to identity."""
    h, w = img.shape[:2]
    s = min(1.0, width / float(w))
    small = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (25, 30, 20), (95, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.erode(m, np.ones((5, 5), np.uint8))
    if m.mean() / 255.0 < 0.15:
        m = None
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(
        cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    )  # CLAHE + a low contrast threshold: dark night turf has little texture otherwise
    kp, des = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.01).detectAndCompute(g, m)
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    return s, pts, des


def register_pair(img_a, img_b, min_inliers=40, min_ratio=0.25, feats=None):
    """Direct feature registration between two frames (SIFT + ratio test + RANSAC homography on
    the turf): 3x3 full-res pixel map a -> b, or None when it is not trustworthy. Independent of
    the chained frame-to-frame motion, so its error does not grow with the time between them.
    `feats` optionally caches `_reg_features` per image id."""
    fa = feats(img_a) if feats is not None else _reg_features(img_a)
    fb = feats(img_b) if feats is not None else _reg_features(img_b)
    (sa, pa, da), (sb, pb, db) = fa, fb
    if da is None or db is None or len(pa) < min_inliers or len(pb) < min_inliers:
        return None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < min_inliers:
        return None
    A = pa[[m.queryIdx for m in good]]
    B = pb[[m.trainIdx for m in good]]
    Hs, inl = cv2.findHomography(A, B, cv2.RANSAC, 3.0, maxIters=5000, confidence=0.999)
    if Hs is None or inl is None:
        return None
    n = int(inl.sum())
    if n < min_inliers or n < min_ratio * len(good):
        return None
    T = np.linalg.inv(np.diag([sb, sb, 1.0])) @ Hs @ np.diag([sa, sa, 1.0])
    if not np.all(np.isfinite(T)) or abs(T[2, 2]) < 1e-12:
        return None
    return T / T[2, 2]


def ground_evidence(marks, conf_min=0.5):
    """Confident GROUND markings of one answer (posts' tops excluded): [(name, x, y, conf)]."""
    P = model_points(*DIMS0)
    return [m for m in marks if m[3] >= conf_min and m[0] in P and P[m[0]][2] == 0.0]


def marking_error_m(H_img2pitch, dims, marks, T_k2f=None, conf_min=0.5):
    """How far a candidate calibration of frame f puts the keyframe's confident ground markings
    (carried into f by T_k2f) from their named pitch points: median metres, best of the two l/r
    labellings (the VLM confuses ends). inf with fewer than 4 usable markings."""
    ev = ground_evidence(marks, conf_min)
    if len(ev) < 4:
        return np.inf
    T = np.eye(3) if T_k2f is None else np.asarray(T_k2f, float)
    q = (
        np.c_[np.array([[m[1], m[2]] for m in ev]), np.ones(len(ev))]
        @ (np.asarray(H_img2pitch, float) @ T).T
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        P = q[:, :2] / q[:, 2:3]
    M = model_points(*dims)
    best = np.inf
    for swap in (False, True):
        ref = np.array([M[_swap(m[0]) if swap else m[0]][:2] for m in ev])
        e = np.linalg.norm(P - ref, axis=1)
        e = np.where(np.isfinite(e), e, 1e6)
        best = min(best, float(np.median(e)))
    return best


def line_support(img, H_img2pitch, dims, colour="white", resp=None, radius=None):
    """Fraction of the projected visible soccer markings that land on paint (VERIFIER only:
    it cannot rank geometries, but a correct fit keeps a stable fraction across frames)."""
    h, w = img.shape[:2]
    resp = paint_response(img, colour) if resp is None else resp
    thr = max(6.0, np.percentile(resp, 97))
    mask = (resp > thr).astype(np.uint8)
    dt = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
    radius = radius or max(3.0, 4.0 * w / 1920.0)
    try:
        G = np.linalg.inv(np.asarray(H_img2pitch, float))
    except np.linalg.LinAlgError:
        return 0.0
    L, W = dims
    tot = hit = 0
    for name, (A, B) in rc.model_lines(L, W).items():
        if name in rc.GOAL_LINES3D:
            continue
        P = A[:2] + np.linspace(0, 1, 200)[:, None] * (B - A)[:2]
        q = np.c_[P, np.ones(len(P))] @ G.T
        ok = q[:, 2] * np.sign(G[2, 2] if G[2, 2] != 0 else 1) > -np.inf
        p = q[:, :2] / q[:, 2:3]
        m = ok & (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        if m.sum() < 5:
            continue
        pi = p[m].astype(int)
        tot += len(pi)
        hit += int(np.sum(dt[pi[:, 1], pi[:, 0]] <= radius))
    return hit / tot if tot else 0.0


def fit_agreement_m(fit_a, fit_b, T_a2b, w, h):
    """Median metric disagreement of two keyframe fits over frame a's lower image (fit_b carried
    into frame a with T_a2b: pixel in a -> pixel in b). dy re-centres different pitch widths."""
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, 12) * w, np.linspace(0.35, 0.95, 8) * h)
    p = np.c_[xs.ravel(), ys.ravel(), np.ones(xs.size)]
    qa = p @ np.asarray(fit_a["H_img2pitch"], float).T
    qb = p @ (np.asarray(fit_b["H_img2pitch"], float) @ T_a2b).T
    Pa, Pb = qa[:, :2] / qa[:, 2:3], qb[:, :2] / qb[:, 2:3]
    Pb[:, 1] += (fit_a["pitch_dims"][1] - fit_b["pitch_dims"][1]) / 2.0
    m = np.all(np.isfinite(Pa), 1) & np.all(np.isfinite(Pb), 1) & (np.abs(Pa).max(1) < 300)
    if m.sum() < 10:
        return np.inf
    return float(np.median(np.linalg.norm(Pa[m] - Pb[m], axis=1)))


def pair_transfers(key_images, chain=None, max_chain_steps=8, frames=None):
    """Pixel maps between every pair of keyframes: {(a, b): 3x3 a -> b}. A DIRECT registration
    of the two images (`register_pair`) when it succeeds; otherwise the chained camera motion
    only across short spans (<= max_chain_steps render frames: the chain's error grows with
    every step, and long chains made the consensus flip on a different motion estimate)."""
    cache = {}

    def feats(img):
        if id(img) not in cache:
            cache[id(img)] = _reg_features(img)
        return cache[id(img)]

    ks = sorted(key_images)
    out = {}
    for i, a in enumerate(ks):
        for b in ks[i + 1 :]:
            T = register_pair(key_images[a], key_images[b], feats=feats)
            how = "direct"
            if T is None and chain is not None:
                ia, ib = chain.pos.get(a), chain.pos.get(b)
                if ia is not None and ib is not None and abs(ib - ia) <= max_chain_steps:
                    T, how = chain.transfer(a, b), "chain"
            if T is not None:
                out[(a, b)] = (T, how)
                out[(b, a)] = (np.linalg.inv(T) / np.linalg.inv(T)[2, 2], how)
    return out


def consensus(key_fits, transfers, frame_size, tol_m=2.5, min_sep=1, sources=None):
    """Keep only keyframe fits CONFIRMED by at least one other keyframe fit built from DIFFERENT
    VLM answers (`sources` {k: set of answering keyframes}; default {k}; pairs that share an
    answer are never compared, so a fit cannot confirm itself): after carrying one into the
    other's frame (`transfers` {(a, b): (T a -> b, how)} from `pair_transfers`) they must agree
    within tol_m (median over the visible field). A lone or contradicted keyframe is dropped ->
    its frames stay unavailable. Returns ({frame: fit}, {frame: [(other, metres, how)]})."""
    w, h = frame_size
    ks = sorted(k for k, v in key_fits.items() if v is not None)
    src = {k: set((sources or {}).get(k, {k})) for k in ks}
    rep = {k: [] for k in ks}
    for i, a in enumerate(ks):
        for b in ks[i + 1 :]:
            if abs(b - a) < min_sep or (a, b) not in transfers or src[a] & src[b]:
                continue
            T, how = transfers[(a, b)]
            d = fit_agreement_m(key_fits[a], key_fits[b], T, w, h)
            rep[a].append((b, round(d, 2), how))
            rep[b].append((a, round(d, 2), how))
    keep = {k: key_fits[k] for k in ks if any(d <= tol_m for _, d, _ in rep[k])}
    return keep, rep


# ------------------------------------------------------------- streaming API (for FastPipeline)
# FastPipeline decodes the window once (pass 1) and resolves calibration afterwards (pass 2), so
# it cannot keep every full frame. Pass 1 stores, per render frame, its camera motion G (already
# done) and a small packed paint mask (verify_mask); at keyframes it keeps the image and asks the
# VLM (async). Pass 2 calls calibrate_window.
MASK_WIDTH = 960
MASK_COLOURS = ("white", "yellow", "red")


def verify_mask(frame_bgr, colours=MASK_COLOURS, width=MASK_WIDTH):
    """Pass-1 helper: packed paint masks of one render frame at `width` px (~50-65 KB per colour at 960 px)."""
    h, w = frame_bgr.shape[:2]
    s = width / float(w)
    small = cv2.resize(frame_bgr, (width, int(round(h * s))), interpolation=cv2.INTER_AREA)
    out = {"scale": s, "shape": small.shape[:2]}
    for c in colours:
        resp = rc.response_map(small, c, 9)
        thr = max(6.0, float(np.percentile(resp, 97)))
        out[c] = np.packbits(resp > thr)
    return out


def _mask_dt(vm, colour):
    c = colour if colour in vm else "white"
    sh = vm["shape"]
    m = np.unpackbits(vm[c])[: sh[0] * sh[1]].reshape(sh)
    return cv2.distanceTransform((1 - m).astype(np.uint8), cv2.DIST_L2, 3)


def support_from_mask(vm, H_img2pitch, dims, colour="white", radius=3.0):
    """line_support computed on a verify_mask (same meaning: fraction of the projected visible
    soccer markings within `radius` mask pixels of paint). VERIFIER only, never a ranker."""
    dt = _mask_dt(vm, colour)
    hh, ww = dt.shape
    s = vm["scale"]
    try:
        Gi = np.linalg.inv(np.asarray(H_img2pitch, float))
    except np.linalg.LinAlgError:
        return 0.0
    L, W = dims
    tot = hit = 0
    for name, (A, B) in rc.model_lines(L, W).items():
        if name in rc.GOAL_LINES3D:
            continue
        P = A[:2] + np.linspace(0, 1, 200)[:, None] * (B - A)[:2]
        q = np.c_[P, np.ones(len(P))] @ Gi.T
        with np.errstate(divide="ignore", invalid="ignore"):
            p = q[:, :2] / q[:, 2:3] * s
        m = np.all(np.isfinite(p), 1)
        m[m] &= (p[m, 0] >= 0) & (p[m, 0] < ww) & (p[m, 1] >= 0) & (p[m, 1] < hh)
        if m.sum() < 5:
            continue
        pi = p[m].astype(int)
        tot += len(pi)
        hit += int(np.sum(dt[pi[:, 1], pi[:, 0]] <= radius))
    return hit / tot if tot else 0.0


def calibrate_window(
    render_frames,
    G,
    key_images,
    masks,
    ask_fn=None,
    answers=None,
    fps=30.0,
    window_s=3.0,
    conf_min=0.5,
    tol_m=2.5,
    ratio=0.8,
    min_ls=0.25,
    prompt=PROMPT,
    refine_keys=True,
    radius=3.0,
):
    """Whole-window calibration from VLM field markings. Never fills: a frame is calibrated only
    if (a) a keyframe fit in its motion segment is CONFIRMED by a keyframe fit built from
    different answers, related by direct registration (consensus, tol_m) and (b) the fit carried to the frame by the chained camera motion keeps >= ratio of the
    keyframe's own paint support (and >= min_ls). Everything else stays None = UNAVAILABLE.

    render_frames: TRUE frame indices in order; G: {frame: 3x3 prev render frame -> frame, None on
    failure}; key_images: {frame: BGR} (render frames, e.g. one per 2 s); masks: {frame:
    verify_mask(...)} for every render frame; answers: {frame: (marks, colour)} if the VLM was
    already asked (async in pass 1), else ask_fn(img, prompt) is called here.
    Returns (per_frame {frame: dict(H_img2pitch or None, pitch_dims, key, ls, ls_key, verified)},
    info dict(colour, keyfits, kept, consensus))."""
    chain = MotionChain(render_frames, G)
    if answers is None:
        answers = {k: ask_markings(img, ask_fn, prompt) for k, img in key_images.items()}
    cols = [c for _, c in answers.values()] or ["white"]
    colour = max(set(cols), key=cols.count)
    colour = colour if colour in MASK_COLOURS else "white"
    marks = {j: a[0] for j, a in answers.items()}
    # 1. one fit per keyframe from ITS OWN answer. A keyframe whose answer alone is too thin is
    # fitted from the answers within +-window_s mapped by DIRECT registration; `sources` records
    # which answers each fit used, and the consensus only compares fits with disjoint sources.
    any_img = next(iter(key_images.values()))
    size = (any_img.shape[1], any_img.shape[0])
    transfers = pair_transfers(key_images, chain)
    keyfits, sources = {}, {}
    for k, img in key_images.items():
        keyfits[k] = keyframe_from_marks(
            img, marks.get(k, []), colour, conf_min=conf_min, refine=refine_keys
        )
        sources[k] = {k}
        if keyfits[k] is None:
            tr = {
                j: transfers[(j, k)][0]
                for j in key_images
                if j != k and (j, k) in transfers and abs(j - k) <= window_s * fps
            }
            if tr:
                pm = pool_markings(
                    chain, {j: marks.get(j, []) for j in [k, *tr]}, k, window_s, fps, tr
                )
                keyfits[k] = keyframe_from_marks(
                    img, pm, colour, conf_min=conf_min, refine=refine_keys
                )
                sources[k] = {k, *tr}
    # 2. agreement through direct keyframe-to-keyframe registration (not the long motion chain).
    good, rep = consensus(keyfits, transfers, size, tol_m=tol_m, sources=sources)
    # 3. only now pool: a confirmed keyframe is refitted with the answers of the other confirmed
    # keyframes within +-window_s (mapped by direct registration); the pooled fit replaces the
    # single-answer fit only if the two agree within tol_m.
    pooled_keys = []
    for k in list(good):
        if len(sources[k]) > 1:
            continue  # already pooled
        tr = {
            j: transfers[(j, k)][0]
            for j in good
            if j != k and (j, k) in transfers and abs(j - k) <= window_s * fps
        }
        if not tr:
            continue
        pm = pool_markings(
            chain, {j: marks.get(j, []) for j in [k, *tr]}, k, window_s, fps, transfers=tr
        )
        pf = keyframe_from_marks(key_images[k], pm, colour, conf_min=conf_min, refine=refine_keys)
        if pf is not None and fit_agreement_m(good[k], pf, np.eye(3), *size) <= tol_m:
            good[k] = pf
            pooled_keys.append(k)
    ls_key = {
        k: support_from_mask(masks[k], v["H_img2pitch"], v["pitch_dims"], colour, radius)
        for k, v in good.items()
        if k in masks
    }
    per = {}
    for f in render_frames:
        best = None
        for k in ls_key:
            T = chain.transfer(f, k) if k != f else np.eye(3)
            if T is not None and (best is None or abs(k - f) < best[0]):
                best = (abs(k - f), k, T)
        if best is None or f not in masks:
            per[f] = None
            continue
        _, k, T = best
        Hp = np.asarray(good[k]["H_img2pitch"], float) @ T
        Hp = Hp / Hp[2, 2]
        dims = good[k]["pitch_dims"]
        ls = support_from_mask(masks[f], Hp, dims, colour, radius)
        ok = bool(ls >= max(min_ls, ratio * ls_key[k]))
        per[f] = dict(
            H_img2pitch=Hp if ok else None,
            H_prop=Hp,
            pitch_dims=dims,
            key=k,
            ls=ls,
            ls_key=ls_key[k],
            verified=ok,
        )
    dims_kept = [good[k]["pitch_dims"] for k in good]
    info = dict(
        colour=colour,
        kept=sorted(good),
        pooled=sorted(pooled_keys),
        consensus=rep,
        pitch_dims=tuple(np.median(np.array(dims_kept), 0).tolist()) if dims_kept else None,
        keyfits={
            k: None
            if v is None
            else dict(ls=v["ls"], n_inliers=v["n_inliers"], pitch_dims=v["pitch_dims"])
            for k, v in keyfits.items()
        },
    )
    return per, info
