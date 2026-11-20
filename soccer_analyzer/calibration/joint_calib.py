"""Whole-window SHARED-CAMERA field calibration (ported from the round-4 lab, 2026-10-14).

Idea. The camera stays in ONE place and only pans/tilts/zooms, so every render frame t is
    x_t ~ K(f_t) R_t (X - C)          with ONE camera centre C for the whole segment.
Frame-to-frame image registration (the camera motion G, plus direct SIFT registrations between
keyframes) fixes every frame's rotation and zoom RELATIVE to the others far more precisely than any
landmark, so landmarks seen in DIFFERENT frames (a goal post here, the D there, a corner flag in a
third view) all constrain the same (C, rotations, focals). "Too few non-collinear points per view"
becomes a well-posed problem, and the VLM is needed only a few times per camera setup.

Pipeline (calibrate_window_joint):
  1. registration: chain G per motion segment (a failed G starts a new segment = a cut / lost
     track), add direct SIFT registrations between keyframes (drift stoppers);
  2. pool every keyframe's landmark answers into ONE virtual reference view through those exact
     rotation homographies and fit a constrained pinhole there (RANSAC over named points, l/r end
     swap) -> C, R_ref, f_ref, pitch dims; decompose every frame's homography into (R_t, f_t);
  3. joint robust bundle over ALL render frames: unknowns per frame (rotation, log focal) + shared C
     + pitch dims; residuals = consecutive-G registration, direct keyframe registrations, named
     landmarks (confidence-weighted), optional paint chamfer on the per-frame paint masks of many
     frames at once, dims prior;
  4. per frame: H_img2pitch from (C, R_t, f_t); gate: registration consistent with a pure
     rotation+zoom AND paint support verified on the frame's own mask; else None = UNAVAILABLE.

Conventions are those of soccer_analyzer.calibration.field_markings.calibrate_window: H maps
full-resolution image pixels to pitch metres, corner origin, x along the length from the LEFT goal
line, y from the FAR touchline; fitted dims in info['pitch_dims'].
"""

from __future__ import annotations

import math
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from soccer_analyzer.calibration import field_markings as fm
from soccer_analyzer.calibration import refcore as rc

DIMS0 = (105.0, 68.0)
DIMS_BOUNDS = ((90.0, 120.0), (55.0, 80.0))


# ------------------------------------------------------------------------------- small helpers
def _K(f, w, h):
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def _nrm(H):
    return H / H[2, 2]


def _apply(H, p):
    q = np.c_[p, np.ones(len(p))] @ np.asarray(H, float).T
    return q[:, :2] / q[:, 2:3]


def _grid(w, h, n=4, m=0.08):
    xs, ys = np.meshgrid(np.linspace(m, 1 - m, n) * w, np.linspace(m, 1 - m, n) * h)
    return np.c_[xs.ravel(), ys.ravel()]


def _rotmats(rv):
    """(N,3) rotation vectors -> (N,3,3) rotation matrices (vectorised Rodrigues)."""
    rv = np.atleast_2d(rv)
    th = np.linalg.norm(rv, axis=1)
    k = rv / np.where(th < 1e-12, 1.0, th)[:, None]
    Kx = np.zeros((len(rv), 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * Kx + (1 - c) * (Kx @ Kx)


def decompose_rotation_homography(H_r2t, Kr, Rr, w, h):
    """For a camera that only rotates/zooms about a fixed centre, the pixel map ref -> t is
    K_t R_t R_r^T K_r^-1. Given the reference (K_r, R_r) recover (R_t, f_t) and a consistency
    residual (0 for an exact rotation+zoom; large when the registration is not one)."""
    B = np.asarray(H_r2t, float) @ Kr @ Rr  # ~ K_t R_t (up to scale)
    B = B / np.linalg.norm(B[2])
    cx, cy = w / 2.0, h / 2.0
    a1 = B[0] - cx * B[2]
    a2 = B[1] - cy * B[2]
    n1, n2 = np.linalg.norm(a1), np.linalg.norm(a2)
    f = 0.5 * (n1 + n2)
    M = np.vstack([a1 / f, a2 / f, B[2]])
    if np.linalg.det(M) < 0:
        M = -M
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = -R
    resid = float(abs(n1 - n2) / f + abs(a1 @ a2) / (n1 * n2) + np.abs(S - 1).max())
    return R, float(f), resid


# ------------------------------------------------------------------------------ 1. registration
class Chain:
    """Chained camera motion over the render frames. G[f]: 3x3 map of full-res pixels of the
    previous render frame to frame f (None/NaN = failed -> a new segment starts)."""

    def __init__(self, render_frames, G):
        self.frames = [int(f) for f in render_frames]
        self.pos = {f: i for i, f in enumerate(self.frames)}
        n = len(self.frames)
        self.seg = [0] * n
        self.A = [np.eye(3)] * n  # pixel in frame i -> pixel in the segment's first frame
        self.G = [None] * n
        for i in range(1, n):
            g = G.get(self.frames[i]) if isinstance(G, dict) else G[i]
            ok = (
                g is not None
                and np.all(np.isfinite(g))
                and abs(np.linalg.det(np.asarray(g, float))) > 1e-8
            )
            if not ok:
                self.seg[i], self.A[i] = self.seg[i - 1] + 1, np.eye(3)
            else:
                g = _nrm(np.asarray(g, float))
                self.G[i] = g
                self.seg[i] = self.seg[i - 1]
                self.A[i] = _nrm(self.A[i - 1] @ np.linalg.inv(g))

    def transfer(self, a, b):
        i, j = self.pos.get(a), self.pos.get(b)
        if i is None or j is None or self.seg[i] != self.seg[j]:
            return None
        return _nrm(np.linalg.inv(self.A[j]) @ self.A[i])

    def segments(self):
        out = {}
        for f, s in zip(self.frames, self.seg):
            out.setdefault(s, []).append(f)
        return out


def direct_pairs(key_images, chain, min_inliers=40, span=2):
    """Direct SIFT registrations between keyframes of the same motion segment: {(a, b): 3x3 a->b}.
    A pair is kept only if it also roughly agrees with the chained motion (guards against a
    repeated-texture mismatch)."""
    cache = {}

    def feats(img):
        if id(img) not in cache:
            cache[id(img)] = fm._reg_features(img)
        return cache[id(img)]

    ks = sorted(k for k in key_images if k in chain.pos)
    out = {}
    for i, a in enumerate(ks):
        for b in ks[i + 1 : i + 1 + span]:  # each keyframe with its next `span` keyframes
            Tc = chain.transfer(a, b)
            if Tc is None:
                continue
            T = fm.register_pair(key_images[a], key_images[b], min_inliers=min_inliers, feats=feats)
            if T is None:
                continue
            h, w = key_images[a].shape[:2]
            g = _grid(w, h, 3)
            d = np.median(np.linalg.norm(_apply(T, g) - _apply(Tc, g), axis=1))
            if d > 0.08 * w:  # disagrees with the chain by > 8% of the width: distrust it
                continue
            out[(a, b)] = T
    return out


# --------------------------------------------------------------------------- 2. pooled world fit
def _parse(ans):
    """answers value -> ([(name, x, y, conf)], colour). Accepts (marks, colour), a raw VLM reply
    dict (parsed with field_markings.parse_markings), or a list of marks."""
    if ans is None:
        return [], None
    if isinstance(ans, dict):
        return fm.parse_markings(ans), str(ans.get("soccer_line_colour", "white")).lower()
    if isinstance(ans, tuple) and len(ans) == 2 and isinstance(ans[1], (str, type(None))):
        return list(ans[0]), ans[1]
    return list(ans), None


def _swap_marks(marks):
    return [(fm._swap(n), x, y, c) for n, x, y, c in marks]


def plausible_C(C, dims=DIMS0):
    """A fixed sideline camera: 1.5-60 m above the grass, on the near side (or behind a goal),
    within ~100 m of the pitch."""
    L, W = dims
    C = np.asarray(C, float)
    return (-60.0 < C[2] < -1.5) and (-100 < C[0] < L + 100) and (W * 0.3 < C[1] < W + 120)


def init_plausible(obj, img, w, h, dims=DIMS0, fs=None):
    """Best solvePnP over a focal grid among PLAUSIBLE camera positions (a telephoto camera 400 m
    away fits points clustered at one goal almost as well as the right one; the physical
    constraint removes it). Returns (cam, rms) or (None, inf)."""
    obj = np.asarray(obj, np.float64)
    img = np.asarray(img, np.float64)
    best = (None, np.inf)
    for f in fs if fs is not None else w * np.geomspace(0.4, 12, 50):
        K = _K(f, w, h)
        for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_IPPE):
            try:
                ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=flag)
            except cv2.error:
                continue
            if not ok:
                continue
            R = rc.rot(rv)
            C = (-R.T @ tv).ravel()
            if not plausible_C(C, dims):
                continue
            cam = dict(rvec=rv.ravel(), C=C, f=float(f), cx=w / 2.0, cy=h / 2.0)
            p, z = rc.project(cam, obj)
            if np.any(z <= 0):
                continue
            e = float(np.sqrt(np.mean(np.sum((p - img) ** 2, 1))))
            if e < best[1]:
                best = (cam, e)
    return best


def camera_from_plane_h(Hw2i, w, h):
    """Constrained pinhole (principal point at the centre, square pixels) from a pitch-plane ->
    image homography, closed form. Returns cam dict or None."""
    A = np.array([[1, 0, -w / 2.0], [0, 1, -h / 2.0], [0, 0, 1.0]]) @ np.asarray(Hw2i, float)
    f2 = []
    d1 = A[2, 0] * A[2, 1]
    if abs(d1) > 1e-12:
        f2.append(-(A[0, 0] * A[0, 1] + A[1, 0] * A[1, 1]) / d1)
    d2 = A[2, 0] ** 2 - A[2, 1] ** 2
    if abs(d2) > 1e-12:
        f2.append(-(A[0, 0] ** 2 + A[1, 0] ** 2 - A[0, 1] ** 2 - A[1, 1] ** 2) / d2)
    f2 = [v for v in f2 if v > 0]
    if not f2:
        return None
    f = float(math.sqrt(np.median(f2)))
    B = np.diag([1 / f, 1 / f, 1.0]) @ A
    lam = 0.5 * (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    B = B / lam
    for sgn in (1.0, -1.0):
        r1, r2, t = sgn * B[:, 0], sgn * B[:, 1], sgn * B[:, 2]
        R = np.column_stack([r1, r2, np.cross(r1, r2)])
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            continue
        C = -R.T @ t
        if C[2] < 0:  # exactly one sign puts the camera above the ground
            return dict(rvec=cv2.Rodrigues(R)[0].ravel(), C=C, f=f, cx=w / 2.0, cy=h / 2.0)
    return None


def ransac_camera(marks, w, h, dims=DIMS0, tau=None, max_sets=3000, rng=0):
    """MSAC over named points with the CAMERA model (not a free homography): 4 ground points ->
    plane homography -> closed-form constrained pinhole -> must be a plausible position and
    focal -> scored on ALL points (goal-post tops included, i.e. 3-D). Returns (cam, inlier mask)."""
    import itertools

    tau = tau or 0.02 * w
    P0 = fm.model_points(*dims)
    names = [m[0] for m in marks]
    q = np.array([[m[1], m[2]] for m in marks])
    conf = np.array([m[3] for m in marks])
    P = np.array([P0[n] for n in names])
    gi = [i for i, n in enumerate(names) if P0[n][2] == 0.0]
    combos = list(itertools.combinations(gi, 4))
    if len(combos) > max_sets:
        R_ = np.random.default_rng(rng)
        combos = [combos[i] for i in R_.choice(len(combos), max_sets, replace=False)]
    best = (None, None, -1.0)
    for c in combos:
        c = list(c)
        if (
            len({names[i] for i in c}) < 4
            or not fm._ok_quad(P[c, :2], 4.0)
            or not fm._ok_quad(q[c], (0.01 * tau) ** 2 * 50)
        ):
            continue
        Hm = cv2.getPerspectiveTransform(P[c, :2].astype(np.float32), q[c].astype(np.float32))
        cam = camera_from_plane_h(Hm, w, h)
        if cam is None:
            continue
        # the closed form is only approximate on noisy points: re-solve the pose for a few focals
        for fk in (0.7, 0.85, 1.0, 1.2, 1.45):
            f = cam["f"] * fk
            if not (0.25 * w < f < 20 * w):
                continue
            try:
                ok, rv, tv = cv2.solvePnP(P[c], q[c], _K(f, w, h), None, flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error:
                continue
            if not ok:
                continue
            Rm = rc.rot(rv)
            cm = dict(rvec=rv.ravel(), C=(-Rm.T @ tv).ravel(), f=float(f), cx=w / 2.0, cy=h / 2.0)
            if not plausible_C(cm["C"], dims):
                continue
            p, z = rc.project(cm, P)
            if np.any(z[c] <= 0):
                continue
            e = np.where(z > 0, np.linalg.norm(p - q, axis=1), np.inf)
            s = float(np.sum(conf * np.clip(1 - (e / tau) ** 2, 0, None)))
            if s > best[2]:
                best = (cm, e < tau, s)
    return best[0], best[1]


def _cluster_marks(pooled, tau):
    """Merge repeated observations of one name that agree within tau (conf-weighted mean, conf
    raised); conflicting observations of a name stay separate (RANSAC picks)."""
    out = []
    by = {}
    for m in pooled:
        by.setdefault(m[0], []).append(m)
    for n, ms in by.items():
        ms = sorted(ms, key=lambda m: -m[3])
        used = [False] * len(ms)
        for i, m in enumerate(ms):
            if used[i]:
                continue
            grp = [
                j
                for j in range(len(ms))
                if not used[j] and math.hypot(ms[j][1] - m[1], ms[j][2] - m[2]) < tau
            ]
            for j in grp:
                used[j] = True
            wts = np.array([ms[j][3] for j in grp])
            xy = np.array([[ms[j][1], ms[j][2]] for j in grp])
            c = 1 - np.prod(1 - np.clip(wts, 0, 0.99))  # independent confirmations raise confidence
            out.append(
                (
                    n,
                    float(xy[:, 0] @ wts / wts.sum()),
                    float(xy[:, 1] @ wts / wts.sum()),
                    float(min(0.99, c)),
                    len(grp),
                )
            )
    return out


def pooled_fit(obs, maps_to_ref, w, h, conf_min=0.5, tau_frac=0.025, dims0=DIMS0):
    """obs {k: [(name,x,y,conf)]}; maps_to_ref {k: 3x3 pixel k -> ref}. Fit ONE constrained
    pinhole in the virtual reference view from all pooled answers: named ground-point RANSAC
    (plane homography, MSAC) -> plausible-position PnP over a focal grid -> robust refinement
    (field_markings._refine_points), for the labels as given and with the l/r ends swapped.
    Returns (fit or None, pooled list)."""
    pooled = []
    for k, marks in obs.items():
        T = maps_to_ref.get(k)
        if T is None:
            continue
        for n, x, y, c in marks:
            if c < conf_min:
                continue
            q = _apply(T, np.array([[x, y]]))[0]
            pooled.append((n, float(q[0]), float(q[1]), float(c)))
    tau = tau_frac * w
    merged = [m[:4] for m in _cluster_marks(pooled, 0.5 * tau)]
    if len(merged) < 4:
        return None, pooled
    best = None
    for swap in (False, True):
        mk = _swap_marks(merged) if swap else merged
        P0 = fm.model_points(*dims0)
        mk = [m for m in mk if m[0] in P0]
        g = [m for m in mk if P0[m[0]][2] == 0.0]
        if len(g) < 4:
            continue
        cam0, inl = ransac_camera(mk, w, h, dims0, tau=0.8 * tau)
        if cam0 is None or inl.sum() < 4:
            continue
        use = [mk[i] for i in range(len(mk)) if inl[i]]
        res = fm._refine_points(cam0, use, w, h, dims0, True, 8.0, tau)
        if res is None:
            continue
        cam, dims, e = res
        keep = [m for m, ei in zip(use, e) if ei < tau]
        if len(keep) < len(use) and len(keep) >= 4:
            res2 = fm._refine_points(cam, keep, w, h, dims, True, 8.0, tau)
            if res2 is not None:
                cam, dims, e = res2
                use = keep
        if not plausible_C(cam["C"], dims):
            continue
        sc = float(np.sum([m[3] * max(0.0, 1 - (ei / tau) ** 2) for m, ei in zip(use, e)]))
        cand = dict(
            camera=cam,
            pitch_dims=tuple(float(v) for v in dims),
            swapped=swap,
            inliers=[m[0] for m in use],
            n_inliers=int(np.sum(e < tau)),
            rms_px=float(np.sqrt(np.mean(np.square(e)))),
            score=sc,
        )
        if best is None or sc > best["score"]:
            best = cand
    return best, pooled


# ------------------------------------------------------------------------------ 3. joint bundle
def _ground_samples(L, W, step=1.0):
    """Pitch points (N,3) along every painted ground marking (lines + circle/arcs), at `step` m."""
    P = []
    for name, (A, B) in rc.model_lines(L, W).items():
        if name in rc.GOAL_LINES3D:
            continue
        n = max(2, int(np.linalg.norm(B - A) / step))
        P.append(A + np.linspace(0, 1, n)[:, None] * (B - A))
    for name, (cx, cy, r, keep) in rc.model_circles(L, W).items():
        a = np.linspace(0, 2 * np.pi, int(2 * np.pi * r / step), endpoint=False)
        X, Y = cx + r * np.cos(a), cy + r * np.sin(a)
        m = keep(X)
        P.append(np.stack([X[m], Y[m], np.zeros(m.sum())], 1))
    return np.vstack(P)


def _ground_samples_param(L, W, step=1.0):
    """Same samples, as a function of (L, W): returns a callable dims -> (N,3) with a FIXED N
    (each sample keeps its relative position along its marking)."""
    base = []
    for name, (A, B) in rc.model_lines(L, W).items():
        if name in rc.GOAL_LINES3D:
            continue
        n = max(2, int(np.linalg.norm(B - A) / step))
        base.append(("line", name, np.linspace(0, 1, n)))
    for name, (cx, cy, r, keep) in rc.model_circles(L, W).items():
        a = np.linspace(0, 2 * np.pi, int(2 * np.pi * r / step), endpoint=False)
        X = cx + r * np.cos(a)
        base.append(("circ", name, a[keep(X)]))

    def gen(Ld, Wd):
        Lm, Cm = rc.model_lines(Ld, Wd), rc.model_circles(Ld, Wd)
        out = []
        for kind, name, t in base:
            if kind == "line":
                A, B = Lm[name]
                out.append(A + t[:, None] * (B - A))
            else:
                cx, cy, r, _ = Cm[name]
                out.append(np.stack([cx + r * np.cos(t), cy + r * np.sin(t), np.zeros(len(t))], 1))
        return np.vstack(out)

    return gen


YD = 0.9144
# A soccer pitch laid on an American-football field normally ends at the football END LINES (goal
# posts on the end line): L ~ 120 yd. Without it the 5-yd periodic grid is ambiguous by whole periods.
FB_L_PRIOR = (120 * 0.9144, 1.0)
FB_L_BOUNDS = (120 * 0.9144 - 3.0, 120 * 0.9144 + 0.6)  # posts on/just inside the end line
FB_HALF_LEN, FB_HALF_WID = (
    60 * YD,
    80 * 0.3048 * 1.0,
)  # end line at 60 yd from midfield; sideline 80 ft from centre


def football_samples_param(L, W, step=1.0):
    """American-football grid (white) on a multi-sport field, ASSUMED centred on the soccer pitch and
    parallel to it: yard lines every 5 yd between the goal lines (+-50 yd), end lines (+-60 yd),
    sidelines (53 1/3 yd apart). Returns gen(L, W) -> (N,3) points with a fixed N."""
    ys = np.arange(-FB_HALF_WID, FB_HALF_WID + 1e-9, step)
    xs = np.arange(-FB_HALF_LEN, FB_HALF_LEN + 1e-9, step)
    rel = []
    for k in range(-10, 11):
        rel.append(np.c_[np.full(len(ys), 5 * YD * k), ys])
    for xe in (-FB_HALF_LEN, FB_HALF_LEN):
        rel.append(np.c_[np.full(len(ys), xe), ys])
    for ye in (-FB_HALF_WID, FB_HALF_WID):
        rel.append(np.c_[xs, np.full(len(xs), ye)])
    rel = np.vstack(rel)

    def gen(Ld, Wd):
        return np.c_[rel[:, 0] + Ld / 2.0, rel[:, 1] + Wd / 2.0, np.zeros(len(rel))]

    return gen


def football_support(vm, cam, dims, radius=3.0):
    """Fraction of the visible football-grid samples within `radius` mask px of WHITE paint."""
    dt = fm._mask_dt(vm, "white")
    sc = vm["scale"]
    S = football_samples_param(*dims, step=1.0)(*dims)
    ps, zs = rc.project(cam, S)
    u, v = ps[:, 0] * sc, ps[:, 1] * sc
    m = (zs > 0.5) & (u >= 0) & (u < dt.shape[1] - 1) & (v >= 0) & (v < dt.shape[0] - 1)
    if m.sum() < 20:
        return 0.0
    return float(np.mean(_bilinear(dt, u[m], v[m]) <= radius))


def kabsch_rotation(C, P, q, f, w, h):
    """Rotation R (world -> camera) that best maps the rays C->P onto the image rays of q (focal f)."""
    a = P - C
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.c_[(q[:, 0] - w / 2) / f, (q[:, 1] - h / 2) / f, np.ones(len(q))]
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    U, _, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def _trimmed_rms(d, keep=0.75):
    """RMS of the best `keep` fraction of 2-D residuals (one mislabelled landmark must not veto)."""
    e = np.sort(np.linalg.norm(np.asarray(d, float), axis=1))
    k = max(3, int(math.ceil(keep * len(e))))
    return float(np.sqrt(np.mean(e[:k] ** 2)))


def football_init(
    marks, dt, sc, w, h, dims=None, tau=None, top=48, yard=(), max_nfev=200, l_sigma=None
):
    """Camera search on a multi-sport (American-football-lined) field. Stage 1: grid over camera
    centres and focals; the rotation fitting the named landmarks is closed form (Kabsch on rays);
    candidates whose landmark rms < tau are scored by the white-mask chamfer of the football grid.
    Stage 2: the best `top` candidates are refined (rotation, centre, focal, pitch length) on
    landmarks + chamfer and ranked by the fraction of grid samples ON the paint (<= 1.5 mask px).
    Returns (cam, info dict) or (None, None)."""
    dims = dims or (FB_L_PRIOR[0], DIMS0[1])
    P0 = fm.model_points(*dims)
    mk = [m for m in marks if m[0] in P0]
    if len(mk) < 3:
        return None, None
    names = [m[0] for m in mk]
    q = np.array([[m[1], m[2]] for m in mk])
    tau = tau or 0.02 * w
    L, W = dims
    gen = football_samples_param(L, W, step=1.0)
    S2 = football_samples_param(L, W, step=2.5)(L, W)
    cands = []
    P = np.array([P0[n] for n in names])
    if yard:
        YA = np.array([[L / 2 + xr, W / 2 - 15.0, 0.0] for xr, _, _, _ in yard])
        YB = np.array([[L / 2 + xr, W / 2 + 15.0, 0.0] for xr, _, _, _ in yard])
        YQ = np.array([[px, py] for _, px, py, _ in yard])
    for cx in np.linspace(-20, L + 20, 16):
        for cy in np.linspace(W + 2, W + 90, 14):
            for cz in (-3.0, -5.0, -7.5, -10.5, -14.0, -19.0, -26.0, -35.0):
                C = np.array([cx, cy, cz])
                for f in w * np.geomspace(0.5, 6.0, 16):
                    R = kabsch_rotation(C, P, q, f, w, h)
                    cam = dict(
                        rvec=cv2.Rodrigues(R)[0].ravel(), C=C, f=float(f), cx=w / 2.0, cy=h / 2.0
                    )
                    p, z = rc.project(cam, P)
                    if (
                        len(P) >= 5
                    ):  # one trimmed re-fit: a mislabelled landmark must not tilt the rotation
                        e = np.linalg.norm(p - q, axis=1)
                        kk = np.argsort(e)[: max(4, int(math.ceil(0.75 * len(e))))]
                        R = kabsch_rotation(C, P[kk], q[kk], f, w, h)
                        cam = dict(
                            rvec=cv2.Rodrigues(R)[0].ravel(),
                            C=C,
                            f=float(f),
                            cx=w / 2.0,
                            cy=h / 2.0,
                        )
                        p, z = rc.project(cam, P)
                    if np.any(z <= 0) or _trimmed_rms(p - q) > tau:
                        continue
                    if (
                        yard
                    ):  # read yard numbers must sit on their yard lines (kills whole-period slips)
                        pa, _ = rc.project(cam, YA)
                        pb, _ = rc.project(cam, YB)
                        dd = pb - pa
                        nn = np.c_[-dd[:, 1], dd[:, 0]] / np.maximum(
                            np.linalg.norm(dd, axis=1, keepdims=True), 1e-9
                        )
                        ey = np.abs(np.sum((YQ - pa) * nn, 1))
                        if np.median(ey) > tau:
                            continue
                    ps, zs = rc.project(cam, S2)
                    u, v = ps[:, 0] * sc, ps[:, 1] * sc
                    m = (
                        (zs > 0.5)
                        & (u >= 0)
                        & (u < dt.shape[1] - 1)
                        & (v >= 0)
                        & (v < dt.shape[0] - 1)
                    )
                    if m.sum() < 40:
                        continue
                    d = _bilinear(dt, u[m], v[m])
                    cands.append((-float(np.mean(d <= 2.0)), cam))
    if not cands:
        return None, None
    cands.sort(key=lambda c: c[0])
    best = (None, -1e9, None)
    sols = []
    S = gen(L, W)

    def resid(x, Ld):
        cam = dict(rvec=x[:3], C=x[3:6], f=math.exp(x[6]), cx=w / 2.0, cy=h / 2.0)
        Pm = fm.model_points(Ld, W)
        p, z = rc.project(cam, np.array([Pm[n] for n in names]))
        r1 = ((p - q) / (0.006 * w)).ravel()
        ps, zs = rc.project(cam, gen(Ld, W)[sel])
        u, v = ps[:, 0] * sc, ps[:, 1] * sc
        d = _bilinear(dt, u, v)
        inside = (zs > 0) & (u >= 0) & (u < dt.shape[1]) & (v >= 0) & (v < dt.shape[0])
        d = np.where(inside, np.minimum(d, 8.0), 8.0) / 2.0 * math.sqrt(30.0 / max(1, len(sel)))
        ry = []
        for xr, px, py, c in yard:  # read yard numbers lie on their yard line
            pa, za = rc.project(
                cam, np.array([[Ld / 2 + xr, W / 2 - 15.0, 0.0], [Ld / 2 + xr, W / 2 + 15.0, 0.0]])
            )
            dd = pa[1] - pa[0]
            nn = np.array([-dd[1], dd[0]]) / max(np.linalg.norm(dd), 1e-9)
            ry.append(float((np.array([px, py]) - pa[0]) @ nn) * math.sqrt(c) / (0.006 * w))
        return np.r_[r1, d, ry, (Ld - FB_L_PRIOR[0]) / (l_sigma or FB_L_PRIOR[1])]

    for _, cam in cands[:top]:
        ps, zs = rc.project(cam, S)
        u, v = ps[:, 0] * sc, ps[:, 1] * sc
        sel = np.nonzero(
            (zs > 0.5) & (u >= 0) & (u < dt.shape[1] - 1) & (v >= 0) & (v < dt.shape[0] - 1)
        )[0]
        if len(sel) < 40:
            continue
        x0 = np.r_[np.asarray(cam["rvec"], float), cam["C"], math.log(cam["f"]), FB_L_PRIOR[0]]
        try:
            r = least_squares(
                lambda x: resid(x[:7], float(np.clip(x[7], *FB_L_BOUNDS))),
                x0,
                loss="soft_l1",
                f_scale=1.0,
                x_scale="jac",
                max_nfev=max_nfev,
            )
        except ValueError:
            continue
        cm = dict(rvec=r.x[:3], C=r.x[3:6], f=float(math.exp(r.x[6])), cx=w / 2.0, cy=h / 2.0)
        Ld = float(np.clip(r.x[7], *FB_L_BOUNDS))
        if not plausible_C(cm["C"], (Ld, W)):
            continue
        ps, zs = rc.project(cm, gen(Ld, W))
        u, v = ps[:, 0] * sc, ps[:, 1] * sc
        m = (zs > 0.5) & (u >= 0) & (u < dt.shape[1] - 1) & (v >= 0) & (v < dt.shape[0] - 1)
        if m.sum() < 40:
            continue
        on = float(np.mean(_bilinear(dt, u[m], v[m]) <= 1.5))
        Pm = fm.model_points(Ld, W)
        p, z = rc.project(cm, np.array([Pm[n] for n in names]))
        lm_rms = _trimmed_rms(p - q)
        if lm_rms > tau:
            continue
        # paint agreement, penalised by landmark misfit (a whole-period slip of the 5-yd grid can
        # match the paint about as well, but then the goal posts no longer fit)
        score = -float(r.cost)  # MAP: paint + landmarks (robust) + end-line prior, one objective
        sols.append(
            (score, cm, dict(on_paint=on, lm_rms=lm_rms, L=Ld, n_cands=len(cands), score=score))
        )
        if score > best[1]:
            best = (cm, score, sols[-1][2])
    if best[0] is None:
        return None, None
    # periodic-grid ambiguity: the best DISTINCT alternative (camera > 2 m away) and its cost gap
    alt = [s_ for s_ in sols if np.linalg.norm(np.asarray(s_[1]["C"]) - best[0]["C"]) > 2.0]
    if alt:
        a = max(alt, key=lambda s_: s_[0])
        best[2].update(
            alt_gap=best[1] - a[0], alt_C=np.asarray(a[1]["C"]).round(1).tolist(), alt_L=a[2]["L"]
        )
    else:
        best[2].update(alt_gap=float("inf"))
    return best[0], best[2]


def _bilinear(img, x, y):
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - x0, y - y0
    return (
        img[y0, x0] * (1 - dx) * (1 - dy)
        + img[y0, x0 + 1] * dx * (1 - dy)
        + img[y0 + 1, x0] * (1 - dx) * dy
        + img[y0 + 1, x0 + 1] * dx * dy
    )


def bundle(
    frames,
    cams0,
    dims0,
    w,
    h,
    reg_edges,
    landmarks,
    chamfer=None,
    fit_dims=(True, True),
    sig_reg=None,
    sig_lm=None,
    dims_sigma=(8.0, 5.0),
    max_nfev=60,
    verbose=0,
    uncertainty=False,
    dims_prior=DIMS0,
    dims_bounds=DIMS_BOUNDS,
    yard_obs=(),
    people=None,
):
    """Joint robust refinement of every frame's (rotation, log focal) + shared C + pitch dims.
    frames: list of frame ids (order = parameter blocks); cams0 {f: cam}; reg_edges: list of
    (a, b, H a->b, weight); landmarks: list of (f, name, x, y, conf); chamfer: None or dict(
    frames={f: (dt, scale)}, tau, sigma, n_eff, step); people: None or dict(obs=[(f, u, v_feet,
    v_head, box_h)], n_eff, h, sig_frac) -- standing players as vertical sticks of height h: the
    feet back-projected to the ground, lifted by h and projected must meet the head (sigma =
    sig_frac x box height). Returns (cams, dims, info)."""
    nF = len(frames)
    ix = {f: i for i, f in enumerate(frames)}
    sig_reg = sig_reg or 0.0015 * w
    sig_lm = sig_lm or 0.006 * w
    C0 = np.mean([np.asarray(cams0[f]["C"], float) for f in frames], 0)
    x0 = []
    for f in frames:
        c = cams0[f]
        x0 += list(np.asarray(c["rvec"], float).ravel()) + [math.log(c["f"])]
    x0 = np.r_[x0, C0, dims0[0], dims0[1]]
    nG = 4 * nF
    gp = _grid(w, h, 3)
    gph = np.c_[gp, np.ones(len(gp))]
    # landmark pack
    Pm0 = fm.model_points(*DIMS0)
    lm = [(ix[f], n, x, y, c) for f, n, x, y, c in landmarks if f in ix and n in Pm0]
    lm_i = np.array([a[0] for a in lm], int)
    lm_n = [a[1] for a in lm]
    lm_q = np.array([[a[2], a[3]] for a in lm], float).reshape(-1, 2)
    lm_w = np.sqrt(np.array([max(0.05, a[4]) for a in lm], float))
    # chamfer pack: fixed sample set per frame (those visible at the start); one or more channels
    # (soccer markings on the soccer-colour mask; optionally the football grid on the white mask)
    ch = []
    gens = []
    chans = [] if not chamfer else (chamfer if isinstance(chamfer, (list, tuple)) else [chamfer])
    R0 = _rotmats(x0[:nG].reshape(nF, 4)[:, :3])
    for ci, chn in enumerate(chans):
        mk = football_samples_param if chn.get("model") == "football" else _ground_samples_param
        gen = mk(*dims0, step=chn.get("step", 1.0))
        gens.append(gen)
        S0 = gen(*dims0)
        for f, (dt, sc) in chn["frames"].items():
            if f not in ix:
                continue
            i = ix[f]
            Kf = _K(math.exp(x0[4 * i + 3]), w, h)
            Xc = (S0 - C0) @ R0[i].T
            z = Xc[:, 2]
            p = Xc @ Kf.T
            p = p[:, :2] / np.where(np.abs(p[:, 2:3]) < 1e-9, 1e-9, p[:, 2:3])
            m = (z > 0.5) & (p[:, 0] > 2) & (p[:, 0] < w - 2) & (p[:, 1] > 2) & (p[:, 1] < h - 2)
            for x0_, y0_, x1_, y1_ in chn.get("exclude", []):
                m &= ~((p[:, 0] >= x0_) & (p[:, 0] < x1_) & (p[:, 1] >= y0_) & (p[:, 1] < y1_))
            sel = np.nonzero(m)[0]
            if len(sel) < 20:
                continue
            ch.append(
                (
                    i,
                    sel,
                    dt,
                    sc,
                    math.sqrt(chn.get("n_eff", 20) / len(sel)),
                    ci,
                    chn["tau"],
                    chn["sigma"],
                )
            )

    def unpack(x):
        P = x[:nG].reshape(nF, 4)
        R = _rotmats(P[:, :3])
        f = np.exp(P[:, 3])
        C = x[nG : nG + 3]
        L = float(np.clip(x[nG + 3], *dims_bounds[0])) if fit_dims[0] else dims0[0]
        W = float(np.clip(x[nG + 4], *dims_bounds[1])) if fit_dims[1] else dims0[1]
        return R, f, C, (L, W)

    edges = [
        (ix[a], ix[b], np.asarray(Hab, float), wt)
        for a, b, Hab, wt in reg_edges
        if a in ix and b in ix
    ]
    obs_e = [(_apply(Hab, gp)) for _, _, Hab, _ in edges]
    # registration edges, vectorised (one batched product instead of a Python loop per edge)
    e_a = np.array([a for a, _, _, _ in edges], int)
    e_b = np.array([b for _, b, _, _ in edges], int)
    e_w = np.array([wt for _, _, _, wt in edges], float) / sig_reg
    e_q = np.stack(obs_e) if edges else np.zeros((0, len(gp), 2))

    yo = [(ix[f_], xr, px, py, c) for f_, xr, px, py, c in yard_obs if f_ in ix]
    yo_i = np.array([a[0] for a in yo], int)
    yo_x = np.array([a[1] for a in yo], float)
    yo_q = np.array([[a[2], a[3]] for a in yo], float).reshape(-1, 2)
    yo_w = np.sqrt(np.array([max(0.05, a[4]) for a in yo], float))

    po = [] if not people else [o for o in people["obs"] if o[0] in ix]
    po_i = np.array([ix[o[0]] for o in po], int)
    po_u = np.array([o[1] for o in po], float)
    po_vf = np.array([o[2] for o in po], float)
    po_vh = np.array([o[3] for o in po], float)
    po_s = np.array([o[4] for o in po], float) * (people or {}).get("sig_frac", 0.07)
    po_wt = math.sqrt((people or {}).get("n_eff", 30) / max(1, len(po)))
    po_h = float((people or {}).get("h", 1.75))

    def resid(x):
        R, f, C, (L, W) = unpack(x)
        out = []
        if len(po):  # players: feet on the ground, head h above them
            Rp, fp = R[po_i], f[po_i]
            dcam = np.c_[(po_u - w / 2) / fp, (po_vf - h / 2) / fp, np.ones(len(po))]
            dw = np.einsum("nji,nj->ni", Rp, dcam)  # R^T d: the feet ray in the world
            dz = np.where(np.abs(dw[:, 2]) < 1e-9, 1e-9, dw[:, 2])
            t = -C[2] / dz  # ground plane Z = 0 (Z points into the ground)
            head = C + t[:, None] * dw - np.array([0.0, 0.0, po_h])
            Xc = np.einsum("nij,nj->ni", Rp, head - C)
            vh = fp * Xc[:, 1] / np.maximum(Xc[:, 2], 1e-3) + h / 2
            r = (vh - po_vh) / po_s * po_wt
            out.append(np.where((t > 0) & (Xc[:, 2] > 0), r, 5.0 * po_wt))
        if len(yo):  # a read yard number lies ON its yard line (1-D: no assumption on its y)
            A3 = np.c_[L / 2 + yo_x, np.full(len(yo), W / 2 - 15.0), np.zeros(len(yo))]
            B3 = np.c_[L / 2 + yo_x, np.full(len(yo), W / 2 + 15.0), np.zeros(len(yo))]
            pa = np.einsum("nij,nj->ni", R[yo_i], A3 - C)
            pb = np.einsum("nij,nj->ni", R[yo_i], B3 - C)
            ua = np.c_[
                f[yo_i] * pa[:, 0] / np.maximum(pa[:, 2], 1e-3) + w / 2,
                f[yo_i] * pa[:, 1] / np.maximum(pa[:, 2], 1e-3) + h / 2,
            ]
            ub = np.c_[
                f[yo_i] * pb[:, 0] / np.maximum(pb[:, 2], 1e-3) + w / 2,
                f[yo_i] * pb[:, 1] / np.maximum(pb[:, 2], 1e-3) + h / 2,
            ]
            d = ub - ua
            nrm = np.c_[-d[:, 1], d[:, 0]] / np.maximum(
                np.linalg.norm(d, axis=1, keepdims=True), 1e-9
            )
            out.append(np.sum((yo_q - ua) * nrm, 1) * yo_w / sig_lm)
        if len(e_a):
            fa, fb = f[e_a], f[e_b]
            Kb = np.zeros((len(e_a), 3, 3))
            Kb[:, 0, 0], Kb[:, 1, 1], Kb[:, 0, 2], Kb[:, 1, 2], Kb[:, 2, 2] = (
                fb,
                fb,
                w / 2,
                h / 2,
                1,
            )
            Kia = np.zeros((len(e_a), 3, 3))
            Kia[:, 0, 0], Kia[:, 1, 1] = 1 / fa, 1 / fa
            Kia[:, 0, 2], Kia[:, 1, 2], Kia[:, 2, 2] = -w / 2 / fa, -h / 2 / fa, 1
            M = Kb @ R[e_b] @ np.transpose(R[e_a], (0, 2, 1)) @ Kia
            p = np.einsum("nij,kj->nki", M, gph)
            p = p[..., :2] / p[..., 2:3]
            out.append(((p - e_q) * e_w[:, None, None]).reshape(-1))
        if len(lm):
            Pm = fm.model_points(L, W)
            P = np.array([Pm[n] for n in lm_n])
            Xc = np.einsum("nij,nj->ni", R[lm_i], P - C)
            z = Xc[:, 2]
            zs = np.where(z < 1e-3, 1e-3, z)
            p = np.c_[f[lm_i] * Xc[:, 0] / zs + w / 2, f[lm_i] * Xc[:, 1] / zs + h / 2]
            r = (p - lm_q) * (lm_w / sig_lm)[:, None]
            r = np.where((z > 0)[:, None], r, 50.0)
            out.append(r.ravel())
        if ch:
            Ss = [g(L, W) for g in gens]
            for i, sel, dt, sc, wt, ci, tau_c, sig_c in ch:
                Xc = (Ss[ci][sel] - C) @ R[i].T
                z = np.maximum(Xc[:, 2], 1e-3)
                u = (f[i] * Xc[:, 0] / z + w / 2) * sc
                v = (f[i] * Xc[:, 1] / z + h / 2) * sc
                d = _bilinear(dt, u, v)
                inside = (u >= 0) & (u < dt.shape[1]) & (v >= 0) & (v < dt.shape[0])
                d = np.where(inside, np.minimum(d, tau_c), tau_c)
                out.append(d / sig_c * wt)
        pri = []
        if fit_dims[0]:
            pri.append((x[nG + 3] - dims_prior[0]) / dims_sigma[0])
        if fit_dims[1]:
            pri.append((x[nG + 4] - dims_prior[1]) / dims_sigma[1])
        out.append(np.array(pri + [0.0]))
        return np.concatenate(out)

    # sparsity
    sizes = [2 * len(gp)] * len(edges)
    rows = (
        len(po)
        + len(yo)
        + sum(sizes)
        + 2 * len(lm)
        + sum(len(c[1]) for c in ch)
        + (sum(fit_dims) + 1)
    )
    S = lil_matrix((rows, len(x0)), dtype=np.int8)
    r0 = 0
    for k in range(len(po)):
        S[r0, 4 * po_i[k] : 4 * po_i[k] + 4] = 1
        S[r0, nG : nG + 3] = 1
        r0 += 1
    for k in range(len(yo)):
        S[r0, 4 * yo_i[k] : 4 * yo_i[k] + 4] = 1
        S[r0, nG:] = 1
        r0 += 1
    for a, b, _, _ in edges:
        S[r0 : r0 + 2 * len(gp), 4 * a : 4 * a + 4] = 1
        S[r0 : r0 + 2 * len(gp), 4 * b : 4 * b + 4] = 1
        r0 += 2 * len(gp)
    for k in range(len(lm)):
        i = lm_i[k]
        S[r0 : r0 + 2, 4 * i : 4 * i + 4] = 1
        S[r0 : r0 + 2, nG:] = 1
        r0 += 2
    for i, sel, *_ in ch:
        S[r0 : r0 + len(sel), 4 * i : 4 * i + 4] = 1
        S[r0 : r0 + len(sel), nG:] = 1
        r0 += len(sel)
    S[r0:, nG + 3 :] = 1
    t0 = time.time()
    r = least_squares(
        resid,
        x0,
        jac_sparsity=S.tocsr(),
        loss="soft_l1",
        f_scale=1.0,
        x_scale="jac",
        max_nfev=max_nfev,
        method="trf",
        tr_solver="lsmr",
        verbose=verbose,
    )
    R, f, C, dims = unpack(r.x)
    cams = {
        fr: dict(
            rvec=cv2.Rodrigues(R[i])[0].ravel(), C=C.copy(), f=float(f[i]), cx=w / 2.0, cy=h / 2.0
        )
        for fr, i in ix.items()
    }
    # per-edge registration residual (px) and per-landmark residual (px) at the solution
    rr = resid(r.x)
    k = len(yo)
    edge_px = []
    for _ in edges:
        e = rr[k : k + 2 * len(gp)].reshape(-1, 2) * sig_reg
        edge_px.append(float(np.median(np.linalg.norm(e, axis=1))))
        k += 2 * len(gp)
    lm_px = (
        np.linalg.norm(rr[k : k + 2 * len(lm)].reshape(-1, 2), axis=1)
        * sig_lm
        / np.maximum(lm_w, 1e-9)
        if len(lm)
        else np.zeros(0)
    )
    sig_m = frame_uncertainty(r, nG, ix, cams, dims, w, h, len(rr)) if uncertainty else None
    info = dict(
        cost=float(r.cost),
        nfev=int(r.nfev),
        secs=time.time() - t0,
        edge_px=edge_px,
        edges=[(frames[a], frames[b]) for a, b, _, _ in edges],
        lm_px=lm_px.tolist(),
        lm=[(frames[a[0]], a[1]) for a in lm],
        n_chamfer_frames=len(ch),
        sig_m=sig_m,
    )
    return cams, dims, info


def frame_uncertainty(r, nG, ix, cams, dims, w, h, n_res, n_draws=24, rng=0):
    """Metric 1-sigma uncertainty (m, median over the visible ground) of every frame's calibration,
    from the bundle's Gauss-Newton covariance (conditioned on the pitch dims, which the evaluation
    treats as a convention: x is anchored at the visible goal). A degenerate geometry (all
    landmarks collinear / clustered, dolly-zoom ambiguity of a narrow grazing view) shows up here
    as a large sigma even when every residual is small -> such frames must stay UNAVAILABLE."""
    from scipy.sparse import csc_matrix

    J = r.jac.tocsc() if hasattr(r.jac, "tocsc") else csc_matrix(r.jac)
    n = nG + 3
    Jk = J[:, :n]
    A = (Jk.T @ Jk).toarray() + 1e-9 * np.eye(n)
    s2 = max(
        1.0, 2.0 * float(r.cost) / max(1, n_res - n)
    )  # inflate when residuals exceed the assumed sigmas
    try:
        cov = np.linalg.inv(A) * s2
    except np.linalg.LinAlgError:
        return {fr: float("inf") for fr in ix}
    R_ = np.random.default_rng(rng)
    L, W = dims
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, 9) * w, np.linspace(0.3, 0.97, 6) * h)
    gpix = np.c_[xs.ravel(), ys.ravel()]
    out = {}
    for fr, i in ix.items():
        idx = list(range(4 * i, 4 * i + 4)) + [nG, nG + 1, nG + 2]
        Ci = cov[np.ix_(idx, idx)]
        Ci = 0.5 * (Ci + Ci.T)
        ev_, V = np.linalg.eigh(Ci)
        Lc = V @ np.diag(np.sqrt(np.clip(ev_, 0, None)))
        d = R_.standard_normal((n_draws, 7)) @ Lc.T
        cam = cams[fr]
        H0 = rc.H_img2pitch(cam)
        P0 = _apply(H0, gpix)
        q = np.c_[gpix, np.ones(len(gpix))] @ H0.T
        ok = (
            (q[:, 2] * np.sign(np.median(q[:, 2])) > 0)
            & (P0[:, 0] > -10)
            & (P0[:, 0] < L + 10)
            & (P0[:, 1] > -10)
            & (P0[:, 1] < W + 10)
        )
        if ok.sum() < 5:
            out[fr] = float("inf")
            continue
        rv0 = np.asarray(cam["rvec"], float).ravel()
        D = []
        for dk in d:
            ck = dict(
                rvec=rv0 + dk[:3],
                C=np.asarray(cam["C"], float) + dk[4:7],
                f=cam["f"] * math.exp(dk[3]),
                cx=cam["cx"],
                cy=cam["cy"],
            )
            D.append(np.linalg.norm(_apply(rc.H_img2pitch(ck), gpix[ok]) - P0[ok], axis=1))
        D = np.array(D)
        out[fr] = float(np.median(np.sqrt(np.mean(D**2, 0))))
    return out


# ------------------------------------------------------------------------------- 4. public API
def _maps_to_ref(chain, seg_frames, ref, pairs):
    """{f: 3x3 pixel f -> ref} for every frame of the segment: direct keyframe registration where
    available (drift-free), else the chain to the nearest directly-registered keyframe."""
    anchors = {ref: np.eye(3)}
    for (a, b), T in pairs.items():
        if b == ref and a in chain.pos:
            anchors[a] = T
        elif a == ref and b in chain.pos:
            anchors[b] = _nrm(np.linalg.inv(T))
    out = {}
    for f in seg_frames:
        k = min(anchors, key=lambda a: abs(chain.pos[a] - chain.pos[f]))
        T = chain.transfer(f, k)
        out[f] = _nrm(anchors[k] @ T) if T is not None else None
    return out


def landmark_stage(
    chain,
    seg_frames,
    keys,
    obs,
    pairs,
    w,
    h,
    conf_min=0.5,
    uncertainty=False,
    football_mask=None,
    yard=None,
    fb_l_sigma=None,
):
    """Pooled named-landmark fit of one motion segment from the answers at `keys` + joint bundle
    (registration + landmarks, no paint). Returns dict(cams, dims, lms, edges, fit_dims, swapped,
    ref, dec_res, binfo, pooled_rms) or dict(status=...) when it cannot be fitted."""
    if not keys:
        return dict(status="no landmarks")
    A = {f: chain.A[chain.pos[f]] for f in seg_frames}
    cen = {f: _apply(A[f], np.array([[w / 2, h / 2]]))[0] for f in keys}
    med = np.median(np.array(list(cen.values())), 0)
    ref = min(keys, key=lambda f: np.linalg.norm(cen[f] - med))  # keyframe nearest the median view
    maps = _maps_to_ref(chain, seg_frames, ref, pairs)
    fit, pooled = pooled_fit({k: obs[k] for k in keys}, maps, w, h, conf_min=conf_min)
    if football_mask is not None and ref in football_mask:
        # multi-sport field: the camera is found by a coarse search scored on the white football grid
        vm = football_mask[ref]
        merged = [
            m[:4]
            for m in _cluster_marks([p for p in pooled if p[3] >= max(conf_min, 0.7)], 0.0125 * w)
        ]
        yard_ref = []
        for k, ys in (yard or {}).items():
            if k in keys and maps.get(k) is not None:
                for xr, px, py, c in ys:
                    qq = _apply(maps[k], np.array([[px, py]]))[0]
                    yard_ref.append((xr, float(qq[0]), float(qq[1]), c))
        cam_fb, fbi = football_init(
            merged, fm._mask_dt(vm, "white"), vm["scale"], w, h, yard=yard_ref, l_sigma=fb_l_sigma
        )
        if cam_fb is not None:
            fit = dict(
                swapped=False,
                camera=cam_fb,
                pitch_dims=(fbi["L"], DIMS0[1]),
                rms_px=fbi["lm_rms"],
                football=fbi,
            )
    if fit is None:
        return dict(status="pooled fit failed", ref=ref, n_pooled=len(pooled))
    swapped, cam_r, dims = fit["swapped"], fit["camera"], fit["pitch_dims"]
    Kr, Rr = _K(cam_r["f"], w, h), rc.rot(cam_r["rvec"])
    cams0, dec_res = {}, {}
    for f in seg_frames:
        T = maps[f]
        if T is None:
            continue
        Rt, ft, res = decompose_rotation_homography(np.linalg.inv(T), Kr, Rr, w, h)
        cams0[f] = dict(
            rvec=cv2.Rodrigues(Rt)[0].ravel(),
            C=np.asarray(cam_r["C"], float),
            f=ft,
            cx=w / 2.0,
            cy=h / 2.0,
        )
        dec_res[f] = res
    tau = 0.025 * w
    # in football mode the grid carries the geometry: only confident landmarks, tight inlier gate
    cmin, gate = (max(conf_min, 0.7), tau) if football_mask is not None else (conf_min, 2 * tau)
    Pm = fm.model_points(*dims)
    lms = []
    for k in keys:
        mk = _swap_marks(obs[k]) if swapped else obs[k]
        for n, x, y, c in mk:
            if c < cmin or n not in Pm or k not in cams0:
                continue
            p, z = rc.project(cams0[k], Pm[n][None])
            if z[0] > 0 and np.hypot(p[0, 0] - x, p[0, 1] - y) < gate:
                lms.append((k, n, x, y, c))
    frames = [f for f in seg_frames if f in cams0]
    edges = []
    for i in range(1, len(frames)):
        a, b = frames[i - 1], frames[i]
        ia, ib = chain.pos[a], chain.pos[b]
        if ib == ia + 1 and chain.G[ib] is not None:
            edges.append((a, b, chain.G[ib], 1.0))
    for (a, b), T in pairs.items():
        if a in cams0 and b in cams0:
            edges.append((a, b, T, 1.0))
    names = [n for _, n, _, _, _ in lms]
    fitL = any(n.startswith(("half", "centre", "circle")) for n in names) and any(
        n[:2] in ("l_", "r_") for n in names
    )
    fitW = any(("corner" in n) or n.startswith("half") for n in names)
    if football_mask is not None:
        fitL = True  # the goal posts fix L against the football grid centred on the halfway line
    n_yard = (
        sum(len(ys) for k, ys in (yard or {}).items() if k in keys)
        if football_mask is not None
        else 0
    )
    if len(lms) + n_yard < 4 or len(lms) < (1 if n_yard >= 3 else 2):
        return dict(status="too few landmark inliers", ref=ref, n_pooled=len(pooled))
    chans = None
    if football_mask is not None:
        # posts-only landmarks are degenerate on their own: keep the football grid in the bundle
        chans = [
            dict(
                frames={
                    f: (fm._mask_dt(football_mask[f], "white"), football_mask[f]["scale"])
                    for j, f in enumerate(frames)
                    if j % 6 == 0 and f in football_mask
                },
                tau=6.0,
                sigma=2.0,
                n_eff=15,
                model="football",
            )
        ]
    fbkw = (
        dict(
            dims_prior=(FB_L_PRIOR[0], DIMS0[1]),
            dims_sigma=(fb_l_sigma or FB_L_PRIOR[1], 5.0),
            dims_bounds=(FB_L_BOUNDS, DIMS_BOUNDS[1]),
        )
        if football_mask is not None
        else {}
    )
    yobs = (
        [(k, xr, px, py, c) for k, ys in (yard or {}).items() if k in keys for xr, px, py, c in ys]
        if football_mask is not None
        else []
    )
    if yobs:
        fbkw["yard_obs"] = yobs
    cams, dims, binfo = bundle(
        frames,
        cams0,
        dims,
        w,
        h,
        edges,
        lms,
        chamfer=chans,
        fit_dims=(fitL, fitW),
        uncertainty=uncertainty,
        **fbkw,
    )
    return dict(
        cams=cams,
        dims=dims,
        lms=lms,
        edges=edges,
        fit_dims=(fitL, fitW),
        swapped=swapped,
        ref=ref,
        fbkw=fbkw,
        dec_res=dec_res,
        binfo=binfo,
        pooled_rms=fit["rms_px"],
        n_pooled=len(pooled),
        frames=frames,
    )


def _disagreement_d(cam_a, dims_a, cam_b, dims_b, w, h):
    """Metric distances between where two calibrations put the same image pixels (ground grid of
    the frame, within the pitch +-10 m under `a`), or None when too few grid points are usable."""
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, 10) * w, np.linspace(0.3, 0.97, 7) * h)
    g = np.c_[xs.ravel(), ys.ravel()]
    Ha, Hb = rc.H_img2pitch(cam_a), rc.H_img2pitch(cam_b)
    qa = np.c_[g, np.ones(len(g))] @ Ha.T
    Pa, Pb = _apply(Ha, g), _apply(Hb, g)
    La, Wa = dims_a
    Lb, Wb = dims_b
    ok = (
        (qa[:, 2] * np.sign(np.median(qa[:, 2])) > 0)
        & (Pa[:, 0] > -10)
        & (Pa[:, 0] < La + 10)
        & (Pa[:, 1] > -10)
        & (Pa[:, 1] < Wa + 10)
        & np.all(np.isfinite(Pb), 1)
    )
    if ok.sum() < 5:
        return None
    right = np.median(Pa[ok, 0]) > La / 2
    Pb = Pb + np.array([(La - Lb) if right else 0.0, (Wa - Wb) / 2.0])
    return np.linalg.norm(Pa[ok] - Pb[ok], axis=1)


def disagreement_m(cam_a, dims_a, cam_b, dims_b, w, h, q=50.0):
    """Distance (metres) between where two calibrations put the same image pixels: the `q`-th
    percentile over the frame's ground grid (q=50: median; q=90: the worst tenth of the view);
    b is re-anchored for different pitch dims at the goal end the view looks at (as the
    evaluation does). inf when the grid is unusable."""
    d = _disagreement_d(cam_a, dims_a, cam_b, dims_b, w, h)
    return float("inf") if d is None else float(np.percentile(d, q))


def view_seen(chain, f, keys, w, h):
    """Fraction of frame f's image (a 10 x 7 grid over its lower 70%, where the ground is) that at
    least one keyframe in `keys` also saw, through the chained camera motion. A field carried far
    past every keyframe's view rests on extrapolated motion alone."""
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, 10) * w, np.linspace(0.3, 0.97, 7) * h)
    g = np.c_[xs.ravel(), ys.ravel()]
    seen = np.zeros(len(g), bool)
    for k in keys:
        T = chain.transfer(f, k)
        if T is None:
            continue
        q = _apply(T, g)
        seen |= (
            np.all(np.isfinite(q), 1)
            & (q[:, 0] >= 0)
            & (q[:, 0] < w)
            & (q[:, 1] >= 0)
            & (q[:, 1] < h)
        )
    return float(seen.mean())


def dims_pinned(dims, fit_dims, bounds, eps=0.01):
    """Indices of FITTED pitch dimensions that ended on a bound of their allowed range (within eps
    metres). The bundle clips them there, so a pinned dimension means the evidence pulls the fit
    outside the modelled geometry: on rpi4k every fit in the wrong basin (camera 1.5 m too close,
    focal 5% short, 1.5-2.2 m wrong at the end of the pan) had the soccer goal line pinned on the
    football end-line bound (L = 110.33 m), while the correct fits sat inside (109.8-110.2 m)."""
    out = []
    for i, (v, fit, (lo, hi)) in enumerate(zip(dims, fit_dims, bounds)):
        if fit and (float(v) <= lo + eps or float(v) >= hi - eps):
            out.append(i)
    return out


def withhold_disagreeing(per, other, w, h, tol_m, tol_p90_m=None, reason="unstable", require=True):
    """Cross-check two joint fits of the same frames. A frame shown in `per` stays shown only if
    `other` lands within tol_m on the median of its view AND within tol_p90_m (default 2 x tol_m)
    on the worst tenth (the consensus gate's two tests); `require`: `other` must show the frame
    too (else a frame `other` does not show is not compared). The rest become unavailable with
    `reason` (in place). Returns the number of frames withheld."""
    t90 = 2.0 * tol_m if tol_p90_m is None else tol_p90_m
    lost = 0
    for f, p in per.items():
        if p.get("H_img2pitch") is None:
            continue
        q = (other or {}).get(f) or {}
        if q.get("H_img2pitch") is None:
            if not require:
                continue
            d = d90 = float("inf")
        else:
            dd = _disagreement_d(p["camera"], p["pitch_dims"], q["camera"], q["pitch_dims"], w, h)
            d, d90 = (
                (float("inf"), float("inf"))
                if dd is None
                else (float(np.median(dd)), float(np.percentile(dd, 90)))
            )
        if not (d <= tol_m and d90 <= t90):
            p["H_img2pitch"], p["verified"], p["reason"] = None, False, reason
            p["stab_m"], p["stab_p90_m"] = d, d90
            lost += 1
    return lost


def jitter_motion(G, w, sigma_px=0.3, rng=0):
    """The camera motion with a small random translation (sigma_px at 1920 px width, per render
    step) added to every frame-to-frame map: a stability probe for the joint fit."""
    R_ = np.random.default_rng(rng)
    s = sigma_px * w / 1920.0

    def jit(g):
        if g is None:
            return None
        g = np.asarray(g, float)
        if not np.all(np.isfinite(g)):
            return g
        T = np.eye(3)
        T[:2, 2] = R_.normal(0.0, s, 2)
        return T @ g

    if isinstance(G, dict):
        return {k: jit(v) for k, v in G.items()}
    return [jit(v) for v in G]


def people_obs(boxes, cams, dims, w, h, conf_min=0.5, margin_m=2.0, max_n=300):
    """Standing-player observations for the bundle's height term: [(frame, u, v_feet, v_head,
    box_h)]. boxes: {frame: (xyxy (n, 4), conf (n,))} (person detections). Kept: confident,
    upright (height 1.6-4.5x width), clear of the image border and of other boxes (IoU <= 0.3),
    and with the feet inside the pitch +- `margin_m` under the camera `cams[frame]` (spectators on
    stands and benches are not on the ground plane). Evenly subsampled to `max_n`."""
    L, W = dims
    out = []
    for f in sorted(boxes):
        if f not in cams:
            continue
        xyxy, conf = boxes[f]
        xyxy, conf = np.asarray(xyxy, float).reshape(-1, 4), np.asarray(conf, float).ravel()
        if not len(xyxy):
            continue
        bw, bh = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
        ok = (conf >= conf_min) & (bh >= 12) & (bh >= 1.6 * bw) & (bh <= 4.5 * bw)
        ok &= (xyxy[:, 0] > 4) & (xyxy[:, 1] > 4) & (xyxy[:, 2] < w - 4) & (xyxy[:, 3] < h - 4)
        if len(xyxy) > 1:
            x1 = np.maximum(xyxy[:, None, 0], xyxy[None, :, 0])
            y1 = np.maximum(xyxy[:, None, 1], xyxy[None, :, 1])
            x2 = np.minimum(xyxy[:, None, 2], xyxy[None, :, 2])
            y2 = np.minimum(xyxy[:, None, 3], xyxy[None, :, 3])
            inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
            a = bw * bh
            iou = inter / np.maximum(a[:, None] + a[None, :] - inter, 1e-9)
            np.fill_diagonal(iou, 0.0)
            ok &= iou.max(axis=1) <= 0.3
        if not np.any(ok):
            continue
        b = xyxy[ok]
        u = (b[:, 0] + b[:, 2]) / 2.0
        P = _apply(rc.H_img2pitch(cams[f]), np.c_[u, b[:, 3]])
        inside = (P[:, 0] > -margin_m) & (P[:, 0] < L + margin_m)
        inside &= (P[:, 1] > -margin_m) & (P[:, 1] < W + margin_m)
        for k in np.nonzero(inside)[0]:
            out.append((f, float(u[k]), float(b[k, 3]), float(b[k, 1]), float(b[k, 3] - b[k, 1])))
    if len(out) > max_n:
        out = [out[i] for i in np.linspace(0, len(out) - 1, max_n).round().astype(int)]
    return out


def majority_colour(cols):
    """The most common soccer-line colour among the keyframe answers; a tie goes to a NON-white
    colour (white is the model's default when unsure), then alphabetically -- deterministic, not
    set-order dependent."""
    cols = [str(c).lower() for c in cols if c]
    if not cols:
        return "white"
    return min(set(cols), key=lambda c: (-cols.count(c), c == "white", c))


def rescaled_images(key_images, scale):
    """Key images resampled to `scale` x their size and back (INTER_AREA down, INTER_CUBIC up):
    the same views with float-level different pixels, as a keyframe stored at another resolution
    gives (the direct registrations then differ slightly)."""
    out = {}
    for k, im in key_images.items():
        h, w = im.shape[:2]
        sm = cv2.resize(
            im,
            (max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        out[k] = cv2.resize(sm, (w, h), interpolation=cv2.INTER_CUBIC)
    return out


def calibrate_window_joint(
    render_frames,
    G,
    key_images,
    masks,
    answers,
    stability=0,
    jitter_px=0.3,
    stab_scale=0.67,
    **kw,
):
    """Whole-window shared-camera calibration (`_joint_once`), optionally STABILITY-checked.

    stability = n: the whole fit (full fit + both consensus halves + gates) is repeated n times
    with PERTURBED INPUTS: the camera motion jittered (`jitter_motion`, jitter_px per render step
    at 1920 px) AND the keyframe images resampled (`rescaled_images`, stab_scale; the direct
    keyframe registrations change slightly). A frame stays shown only if every repeat also
    confirms it and lands within tol_m on the median of its view and 2 x tol_m on its worst tenth
    (`withhold_disagreeing`). Union's football-grid fit flipped from refused to shown (1.5-2.6 m
    wrong) under 0.2 px of motion noise, and rpi4k's fit flipped between a right and a wrong basin
    with the keyframe resolution (1280 / 1920 / 3828 px): a field that tiny input changes can flip
    is not confirmed. The extra fits run only when the first fit shows something. Same returns as
    `_joint_once`; info gains `stability` (per repeat: frames shown, frames that lost
    confirmation)."""
    per, info = _joint_once(render_frames, G, key_images, masks, answers, **kw)
    if not stability or not any(p.get("H_img2pitch") is not None for p in per.values()):
        return per, info
    if kw.get("frame_size") is not None:
        w, h = kw["frame_size"]
    else:
        im = next(iter(key_images.values()))
        w, h = im.shape[1], im.shape[0]
    tol = kw.get("tol_m", 1.5)
    info["stability"] = []
    for i in range(int(stability)):
        Gj = jitter_motion(G, w, jitter_px, rng=1000 + i)
        kij = (
            rescaled_images(key_images, stab_scale)
            if stab_scale and stab_scale != 1
            else key_images
        )
        per_j, info_j = _joint_once(render_frames, Gj, kij, masks, answers, **kw)
        del kij
        lost = withhold_disagreeing(
            per, per_j, w, h, tol, kw.get("tol_p90_m"), reason="unstable", require=True
        )
        info["stability"].append(
            dict(
                shown=int(round(info_j["coverage"] * len(per_j))),
                lost=lost,
                secs=info_j["secs"],
                C=(info_j["segments"].get(0) or {}).get("C"),
                dims_pinned=(info_j["segments"].get(0) or {}).get("dims_pinned"),
            )
        )
        info["secs"] += info_j["secs"]
    frames = [f for f in per]
    info["coverage"] = float(np.mean([per[f]["H_img2pitch"] is not None for f in frames]))
    return per, info


def _joint_once(
    render_frames,
    G,
    key_images,
    masks,
    answers,
    fps=30.0,
    frame_size=None,
    conf_min=0.5,
    colour=None,
    use_direct=True,
    chamfer=True,
    chamfer_every=3,
    chamfer_tau=6.0,
    chamfer_sigma=2.0,
    chamfer_neff=15,
    exclude_px=(),
    min_ls=0.25,
    ls_ratio=0.5,
    max_edge_px=None,
    pair_span=2,
    max_sigma_m=None,
    consensus=True,
    tol_m=1.5,
    tol_p90_m=None,
    min_seen=0.5,
    football=False,
    football_neff=15,
    football_l_sigma=None,
    numbers=None,
    people=None,
    people_neff=30.0,
    h_person=1.75,
    verbose=False,
):
    """Whole-window shared-camera calibration.

    render_frames: TRUE frame indices (ordered); G: {frame: 3x3 prev render frame -> frame, None on
    failure} (or an array aligned with render_frames); key_images: {frame: full-res BGR} (frames
    with VLM answers; also used for direct registration); masks: {frame: field_markings.verify_mask}
    for the render frames (verification + chamfer); answers: {frame: (marks, colour) | raw VLM reply
    dict | marks list} in ORIGINAL pixels. frame_size = (w, h).
    A frame is calibrated (H_img2pitch not None) only if ALL hold:
      * its motion segment has a joint fit (>= 4 landmark inliers);
      * consensus: two fits from DISJOINT halves of the keyframe answers (alternate keyframes)
        each agree with the final calibration within tol_m on this frame (median over the
        view) AND within tol_p90_m (default 2 x tol_m) on its worst tenth (independent evidence;
        line support alone cannot rank geometries) -- needs >= 2 keyframes with answers;
      * at least `min_seen` of the frame's view was also seen by a keyframe of its segment (no
        field carried far past every keyframe's view on motion alone);
      * no fitted pitch dimension of its segment is pinned on its bound (`dims_pinned`);
      * its registration is consistent with a rotating/zooming camera (bundle edge residual);
      * paint support on its own mask >= max(min_ls, ls_ratio x the keyframes' median).
    Everything else stays None = UNAVAILABLE (never filled).
    people: optional {frame: (xyxy, conf)} person detections; standing players (`people_obs`,
    feet on the pitch under the landmark fit) then add a height term (h_person, weight
    people_neff) to the paint bundles of the full fit AND of both halves (shared evidence, like
    the paint): it fixes the camera height a few far-end landmarks leave free.
    Returns (per {frame: dict(H_img2pitch or None, H_model, camera, pitch_dims, ls, reg_px, cons_m,
    verified, segment, reason)}, info)."""
    t0 = time.time()
    if frame_size is None:
        im = next(iter(key_images.values()))
        frame_size = (im.shape[1], im.shape[0])
    w, h = frame_size
    chain = Chain(render_frames, G)
    obs, cols = {}, []
    for k, a in (answers or {}).items():
        mk, col = _parse(a)
        if int(k) in chain.pos and mk:
            obs[int(k)] = mk
            if col:
                cols.append(col)
    if colour is None:
        colour = majority_colour(cols)
    colour = colour if colour in fm.MASK_COLOURS else "white"
    pairs = direct_pairs(key_images, chain, span=pair_span) if use_direct else {}
    t_reg = time.time() - t0
    per = {
        f: dict(H_img2pitch=None, verified=False, reason="no calibrated segment")
        for f in chain.frames
    }
    info = dict(colour=colour, segments={}, pairs=sorted(pairs), t_register=t_reg)
    for s, seg_frames in chain.segments().items():
        keys = [k for k in seg_frames if k in obs]
        sinfo = dict(frames=len(seg_frames), keys=keys)
        info["segments"][s] = sinfo
        fbm = masks if football else None
        yard = {}
        for k, ns in (
            numbers or {}
        ).items():  # read yard numbers -> (x offset from midfield m, px, py, conf)
            k = int(k)
            for n in ns:
                try:
                    v, c = int(n["value"]), float(n.get("confidence", 0.5))
                except (KeyError, TypeError, ValueError):
                    continue
                if v not in (10, 20, 30, 40, 50) or c < 0.7:
                    continue
                sgn = 1.0 if str(n.get("toward", "right")).lower() == "right" else -1.0
                yard.setdefault(k, []).append(
                    (sgn * (50 - v) * YD, float(n["x"]), float(n["y"]), c)
                )
        if yard:
            for k in yard:  # number reads are keyframe evidence too (consensus halves split them)
                if k in chain.pos and k not in keys and k in seg_frames:
                    keys.append(k)
                    obs.setdefault(k, [])
            keys.sort()
        st = landmark_stage(
            chain,
            seg_frames,
            keys,
            obs,
            pairs,
            w,
            h,
            conf_min,
            football_mask=fbm,
            yard=yard,
            fb_l_sigma=football_l_sigma,
        )
        if "cams" not in st:
            sinfo.update(st)
            continue
        cams, dims, frames, binfo = st["cams"], st["dims"], st["frames"], st["binfo"]
        sinfo.update(
            ref=st["ref"],
            n_pooled=st["n_pooled"],
            swapped=st["swapped"],
            n_landmarks=len(st["lms"]),
            dims_points=dims,
            pooled_fit_rms=st["pooled_rms"],
            bundle1=dict(
                cost=binfo["cost"],
                secs=binfo["secs"],
                lm_px_median=float(np.median(binfo["lm_px"])) if binfo["lm_px"] else None,
            ),
        )
        ppl = None  # player-height term (only with the paint bundle)
        if chamfer and masks:
            chf = {}
            for j, f in enumerate(frames):
                if j % chamfer_every and f not in obs:
                    continue
                vm = masks.get(f)
                if vm is None:
                    continue
                chf[f] = (fm._mask_dt(vm, colour), vm["scale"])
            if chf:
                chans = [
                    dict(
                        frames=chf,
                        tau=chamfer_tau,
                        sigma=chamfer_sigma,
                        n_eff=chamfer_neff,
                        exclude=list(exclude_px),
                    )
                ]
                if football:
                    chw = {f: (fm._mask_dt(masks[f], "white"), masks[f]["scale"]) for f in chf}
                    chans.append(
                        dict(
                            frames=chw,
                            tau=chamfer_tau,
                            sigma=chamfer_sigma,
                            n_eff=football_neff,
                            exclude=list(exclude_px),
                            model="football",
                        )
                    )
                pobs = people_obs(people, cams, dims, w, h) if people else []
                ppl = dict(obs=pobs, n_eff=people_neff, h=h_person) if pobs else None
                sinfo["people_obs"] = len(pobs)
                cams_c, dims_c, binfo_c = bundle(
                    frames,
                    cams,
                    dims,
                    w,
                    h,
                    st["edges"],
                    st["lms"],
                    chamfer=chans,
                    fit_dims=st["fit_dims"],
                    uncertainty=max_sigma_m is not None,
                    people=ppl,
                    **st["fbkw"],
                )
                sinfo["bundle2"] = dict(
                    cost=binfo_c["cost"],
                    secs=binfo_c["secs"],
                    n_chamfer=binfo_c["n_chamfer_frames"],
                    lm_px_median=float(np.median(binfo_c["lm_px"])) if binfo_c["lm_px"] else None,
                )
                cams, dims, binfo = cams_c, dims_c, binfo_c
        sinfo["pitch_dims"] = tuple(float(v) for v in dims)
        pinned = dims_pinned(dims, st["fit_dims"], st["fbkw"].get("dims_bounds", DIMS_BOUNDS))
        sinfo["dims_pinned"] = pinned
        sinfo["C"] = np.asarray(cams[frames[0]]["C"]).round(2).tolist()
        sinfo["f_range"] = [
            round(min(c["f"] for c in cams.values())),
            round(max(c["f"] for c in cams.values())),
        ]
        # independent evidence: split-half consensus
        cons = {f: None for f in frames}
        cons90 = {f: None for f in frames}
        if consensus:
            ks = sorted(keys)
            halves = [ks[0::2], ks[1::2]]
            hst = [
                landmark_stage(
                    chain,
                    seg_frames,
                    hk,
                    obs,
                    pairs,
                    w,
                    h,
                    conf_min,
                    football_mask=fbm,
                    yard=yard,
                    fb_l_sigma=football_l_sigma,
                )
                if hk
                else dict(status="empty")
                for hk in halves
            ]
            if consensus == "paint" and chamfer and masks:
                # each half also aligned to the paint (the paint is shared evidence, the landmark
                # answers are disjoint): the halves then differ only through their VLM answers
                for hs in hst:
                    if "cams" not in hs:
                        continue
                    chf2 = {
                        f: (fm._mask_dt(masks[f], colour), masks[f]["scale"])
                        for j, f in enumerate(hs["frames"])
                        if (j % (2 * chamfer_every) == 0) and f in masks
                    }
                    chans2 = [
                        dict(
                            frames=chf2,
                            tau=chamfer_tau,
                            sigma=chamfer_sigma,
                            n_eff=chamfer_neff,
                            exclude=list(exclude_px),
                        )
                    ]
                    if football:
                        chans2.append(
                            dict(
                                frames={
                                    f: (fm._mask_dt(masks[f], "white"), masks[f]["scale"])
                                    for f in chf2
                                },
                                tau=chamfer_tau,
                                sigma=chamfer_sigma,
                                n_eff=football_neff,
                                exclude=list(exclude_px),
                                model="football",
                            )
                        )
                    hs["cams"], hs["dims"], _ = bundle(
                        hs["frames"],
                        hs["cams"],
                        hs["dims"],
                        w,
                        h,
                        hs["edges"],
                        hs["lms"],
                        chamfer=chans2,
                        fit_dims=hs["fit_dims"],
                        people=ppl,
                        **hs["fbkw"],
                    )
            sinfo["halves"] = [
                dict(
                    keys=hk,
                    status=hs.get("status", "ok"),
                    dims=hs.get("dims"),
                    C=None
                    if "cams" not in hs
                    else np.asarray(next(iter(hs["cams"].values()))["C"]).round(1).tolist(),
                )
                for hk, hs in zip(halves, hst)
            ]
            for f in frames:
                ds, d90 = [], []
                for hs in hst:
                    d = (
                        _disagreement_d(cams[f], dims, hs["cams"][f], hs["dims"], w, h)
                        if "cams" in hs and f in hs["cams"]
                        else None
                    )
                    ds.append(float("inf") if d is None else float(np.median(d)))
                    d90.append(float("inf") if d is None else float(np.percentile(d, 90)))
                cons[f], cons90[f] = max(ds), max(d90)
        reg_px = {f: 0.0 for f in frames}
        for (a, b), e in zip(binfo["edges"], binfo["edge_px"]):
            for f in (a, b):
                reg_px[f] = max(reg_px[f], e)
        mepx = max_edge_px if max_edge_px is not None else 0.01 * w
        ls = {}
        for f in frames:
            vm = masks.get(f) if masks else None
            if vm is None:
                ls[f] = None
            elif football:
                ls[f] = football_support(vm, cams[f], dims)
            else:
                ls[f] = fm.support_from_mask(vm, rc.H_img2pitch(cams[f]), dims, colour, 3.0)
        kls = [ls[k] for k in keys if ls.get(k) is not None]
        ref_ls = float(np.median(kls)) if kls else None
        sinfo["ls_keys_median"] = ref_ls
        sinfo["lm_rms_px"] = (
            float(np.sqrt(np.mean(np.square(binfo["lm_px"])))) if binfo["lm_px"] else None
        )
        t90 = 2.0 * tol_m if tol_p90_m is None else tol_p90_m
        seen = {f: view_seen(chain, f, keys, w, h) for f in frames}
        sinfo["seen_min"] = float(min(seen.values())) if seen else None
        for f in frames:
            H = rc.H_img2pitch(cams[f])
            ok_cons = cons[f] is None or (cons[f] <= tol_m and cons90[f] <= t90)
            ok_seen = seen[f] >= min_seen
            ok_reg = reg_px[f] <= mepx
            ok_ls = ls[f] is None or ref_ls is None or ls[f] >= max(min_ls, ls_ratio * ref_ls)
            sig = (binfo.get("sig_m") or {}).get(f)
            ok_sig = max_sigma_m is None or sig is None or sig <= max_sigma_m
            ok_dims = not pinned
            ok = bool(ok_cons and ok_dims and ok_seen and ok_reg and ok_ls and ok_sig)
            reason = (
                None
                if ok
                else (
                    "no consensus"
                    if not ok_cons
                    else "dims at bound"
                    if not ok_dims
                    else "outside keyframe views"
                    if not ok_seen
                    else "registration"
                    if not ok_reg
                    else "paint"
                    if not ok_ls
                    else "uncertain"
                )
            )
            per[f] = dict(
                H_img2pitch=H if ok else None,
                H_model=H,
                camera=cams[f],
                pitch_dims=tuple(dims),
                ls=ls[f],
                reg_px=reg_px[f],
                cons_m=cons[f],
                cons_p90_m=cons90[f],
                seen=seen[f],
                dec_res=st["dec_res"].get(f),
                sigma_m=sig,
                verified=ok,
                segment=s,
                reason=reason,
            )
    dl = [v["pitch_dims"] for v in info["segments"].values() if "pitch_dims" in v]
    info["pitch_dims"] = tuple(np.median(np.array(dl), 0).tolist()) if dl else None
    info["secs"] = time.time() - t0
    info["coverage"] = float(np.mean([per[f]["H_img2pitch"] is not None for f in chain.frames]))
    return per, info


# ---------------------------------------------------------------- keyframe choice + yard numbers
# The measured lab prompt, verbatim.
PROMPT_NUMBERS = """This frame shows a multi-sport field with American-football yard lines. The image is {W}x{H} pixels (x right, y down).
List every yard-line NUMBER painted on the grass that you can read (10, 20, 30, 40 or 50), and for each the pixel position of the CENTRE of the painted number (both digits together).
'side' = 'near' if the number is on the half of the field closer to the camera (bottom), 'far' if on the far half (top).
'toward' = which way the numbers DEcrease from this one along the field as seen in the image: 'right' or 'left' (e.g. a 40 with a 30 to its right -> 'right').
confidence 0-1 = how sure you are of the value AND the position.
Reply with JSON only: {"numbers": [{"value": 40, "x": 0, "y": 0, "side": "near", "toward": "right", "confidence": 0.0}]}"""


def parse_numbers(payload, scale=(1.0, 1.0)):
    """Yard-number reply -> [dict(value, x, y, toward, side, confidence)] in ORIGINAL pixels
    (`scale` = original / sent image size). Malformed entries are dropped."""
    out = []
    items = payload.get("numbers") if isinstance(payload, dict) else None
    for n in items if isinstance(items, list) else []:
        if not isinstance(n, dict):
            continue
        try:
            v, x, y = int(n["value"]), float(n["x"]), float(n["y"])
            c = float(n.get("confidence", 0.5))
        except (KeyError, TypeError, ValueError):
            continue
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)):
            continue
        out.append(
            dict(
                value=v,
                x=x * scale[0],
                y=y * scale[1],
                toward=str(n.get("toward", "")).lower(),
                side=str(n.get("side", "")).lower(),
                confidence=c,
            )
        )
    return out


def football_numbers_found(numbers, conf_min=0.7):
    """True when any keyframe read a valid yard number confidently (the football-mode trigger)."""
    return any(
        int(n.get("value", 0)) in (10, 20, 30, 40, 50) and float(n.get("confidence", 0)) >= conf_min
        for ns in (numbers or {}).values()
        for n in ns
    )


def paint_fraction(vm):
    """Fraction of a packed paint mask (`field_markings.verify_mask`) that is paint, all colours:
    a cheap 'how many markings are visible' score for choosing keyframes."""
    if not vm:
        return 0.0
    tot = 0.0
    for c in fm.MASK_COLOURS:
        bits = vm.get(c)
        if isinstance(bits, np.ndarray) and bits.size:
            tot += float(np.unpackbits(bits).mean())
    return tot


def view_features(chain, frames, frame_size):
    """Per frame: (view centre x, y in its segment's middle frame, as fractions of the width;
    0.5 log of the view's area relative to that frame) -- where the camera looks and how wide.
    Frames whose chained map is degenerate are left out."""
    w, h = frame_size
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    segs = chain.segments()
    mid = {s: fs[len(fs) // 2] for s, fs in segs.items()}
    out = {}
    for f in frames:
        i = chain.pos.get(f)
        if i is None:
            continue
        T = chain.transfer(f, mid[chain.seg[i]])
        if T is None:
            continue
        q = _apply(T, corners)
        x, y = q[:, 0], q[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        c = _apply(T, np.array([[w / 2.0, h / 2.0]]))[0]
        if not (np.all(np.isfinite(q)) and area > 1e-3 * w * h):
            continue
        out[f] = (c[0] / w, c[1] / w, 0.5 * math.log(area / (w * h)), chain.seg[i])
    return out


def select_keyframes(
    render_frames,
    G,
    candidates,
    frame_size,
    k=4,
    paint=None,
    fps=30.0,
    min_gap_s=1.0,
    min_paint_ratio=0.0,
):
    """Choose up to `k` marking keyframes per motion segment by VIEW COVERAGE, not by time.

    The camera only turns and zooms, so the chained motion G says where every frame looks
    (`view_features`). First the widest view (most of the field; ties by `paint`, e.g. the
    paint-mask fraction, i.e. most markings visible), then repeatedly the candidate whose view
    is farthest from every chosen one (pan, tilt and zoom), at least `min_gap_s` apart in time
    so the two consensus halves see different moments (closer ones only when nothing else is
    left). At least 2 per segment when possible
    (the split-half confirmation needs two independent answers). With `paint` and
    `min_paint_ratio` > 0, a candidate showing less than that fraction of its segment's most
    paint (a screen overlay hides paint) is picked only when no other candidate is left.
    Returns sorted frames."""
    chain = Chain(render_frames, G)
    cand = sorted({int(c) for c in candidates if int(c) in chain.pos})
    feats = view_features(chain, cand, frame_size)
    paint = paint or {}
    keys = []
    by_seg = {}
    for f, v in feats.items():
        by_seg.setdefault(v[3], []).append(f)
    for _, fs in sorted(by_seg.items()):
        X = np.array([feats[f][:3] for f in fs])
        pv = np.array([float(paint.get(f, 0.0)) for f in fs])
        low = (pv < min_paint_ratio * pv.max()) if pv.max() > 0 else np.zeros(len(fs), bool)
        wide = X[:, 2] + 0.05 * pv - 1e3 * low
        chosen = [int(np.argmax(wide))]
        while len(chosen) < min(k, len(fs)):
            d = np.min(np.linalg.norm(X[:, None, :] - X[chosen][None], axis=2), axis=1)
            tgap = np.min(np.abs(np.array(fs)[:, None] - np.array(fs)[chosen][None]), axis=1)
            # a near-static camera: all views alike -> spread in time instead
            d = d + 1e-3 * tgap / max(1.0, fps)
            d[chosen] = -np.inf
            # prefer views at least min_gap_s from every chosen one; closer ones only when no
            # other candidate is left
            far = tgap >= min_gap_s * fps
            d = np.where(far, d, np.where(np.isfinite(d), d - 1e6, d))
            d = np.where(low & np.isfinite(d), d - 1e3, d)
            j = int(np.argmax(d))
            if not np.isfinite(d[j]):
                break
            chosen.append(j)
        keys.extend(fs[j] for j in chosen)
    return sorted(keys)


# ------------------------------------------------------------------------------------ self-test
def synthetic_window(
    n=60,
    w=1920,
    h=1080,
    C=(52.0, 90.0, -15.0),
    f0=2000.0,
    pan_deg=(-25.0, 15.0),
    zoom=1.3,
    keys=(5, 30, 55),
    noise_px=2.0,
    rng=0,
):
    """A rotating + zooming camera at one centre filming a 105 x 68 pitch: exact G between render
    frames, VLM-like named landmarks (with pixel noise) at `keys`. Returns (frames, G, answers,
    truth cams, (w, h))."""
    R_ = np.random.default_rng(rng)
    C = np.asarray(C, float)
    frames = list(range(0, 2 * n, 2))
    cams = {}
    for i, fr in enumerate(frames):
        a = math.radians(pan_deg[0] + (pan_deg[1] - pan_deg[0]) * i / (n - 1))
        target = np.array([52.5 + math.tan(a) * (C[1] - 34.0), 34.0, 0.0])
        z = target - C
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, -1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.vstack([x, y, z])
        if np.linalg.det(R) < 0:
            R[0] = -R[0]
        f = f0 * (1 + (zoom - 1) * i / (n - 1))
        cams[fr] = dict(rvec=cv2.Rodrigues(R)[0].ravel(), C=C, f=f, cx=w / 2.0, cy=h / 2.0)
    G = {frames[0]: None}
    for a, b in zip(frames[:-1], frames[1:]):
        Ka, Kb = _K(cams[a]["f"], w, h), _K(cams[b]["f"], w, h)
        G[b] = _nrm(Kb @ rc.rot(cams[b]["rvec"]) @ rc.rot(cams[a]["rvec"]).T @ np.linalg.inv(Ka))
    P = fm.model_points(105.0, 68.0)
    answers = {}
    for k in keys:
        fr = frames[k]
        mk = []
        for name, X in P.items():
            p, z = rc.project(cams[fr], X[None])
            if z[0] > 0 and 0 <= p[0, 0] < w and 0 <= p[0, 1] < h:
                q = p[0] + R_.normal(0, noise_px, 2)
                mk.append((name, float(q[0]), float(q[1]), 0.9))
        answers[fr] = (mk, "white")
    return frames, G, answers, cams, (w, h)


def selftest():
    """Recover a synthetic rotating/zooming camera from landmarks at 3 keyframes + motion only.
    Returns dict(coverage, median_err_m, max_err_m); asserts sub-0.3 m accuracy."""
    frames, G, answers, cams, (w, h) = synthetic_window()
    per, info = calibrate_window_joint(
        frames,
        G,
        {},
        None,
        answers,
        frame_size=(w, h),
        use_direct=False,
        chamfer=False,
        consensus=True,
    )
    xs, ys = np.meshgrid(np.linspace(0.1, 0.9, 6) * w, np.linspace(0.45, 0.95, 4) * h)
    g = np.c_[xs.ravel(), ys.ravel()]
    errs = []
    for fr in frames:
        H = per[fr]["H_img2pitch"]
        if H is None:
            continue
        Ht = rc.H_img2pitch(cams[fr])
        L, W = per[fr]["pitch_dims"]
        Pe = _apply(H, g) + np.array([0.0, (68.0 - W) / 2])  # left-goal anchored; re-centre W
        Pt = _apply(Ht, g)
        m = (Pt[:, 0] > -5) & (Pt[:, 0] < 110) & (Pt[:, 1] > -5) & (Pt[:, 1] < 73)
        errs.append(float(np.median(np.linalg.norm(Pe[m] - Pt[m], axis=1))))
    out = dict(
        coverage=info["coverage"],
        median_err_m=float(np.median(errs)),
        max_err_m=float(np.max(errs)),
        C=info["segments"][0].get("C"),
    )
    assert out["coverage"] > 0.9 and out["median_err_m"] < 0.3, out
    return out


if __name__ == "__main__":
    print(selftest())
