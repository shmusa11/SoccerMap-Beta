"""Constrained pinhole camera + pitch geometry for the VLM field-marking calibration.

Ported from the 2026-10-12 calibration lab (it also builds the hand-made reference cameras).

Pitch geometry (named points / lines / circles), a constrained pinhole camera (focal, rotation,
position; principal point at the image centre; no distortion), paint-snapping of rough
hand-placed lines/arcs onto the painted markings, a robust joint least-squares fit, and drawing.

World frame = the repo's PitchModel convention: corner origin, X along the length 0..L (left goal
X=0 as seen from the main camera side), Y across 0..W (Y=0 far touchline, Y=W near touchline),
Z = X cross Y = INTO the ground (so a point h metres above the grass has Z = -h; a camera above
the pitch has negative Z). Right-handed, so a proper rotation maps it to OpenCV camera axes.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from scipy.optimize import least_squares

# Marking sizes (metres) — identical to soccer_analyzer.calibration.pitch_model.PitchModel.
# (NCAA sizes are the yard versions of the same markings; differences are <= 9 cm.)
PB_D, PB_W = 16.5, 40.32
GB_D, GB_W = 5.5, 18.32
SPOT, R_CIRC = 11.0, 9.15
GOAL_W, GOAL_H = 7.32, 2.44
ARC_HALF = math.sqrt(
    R_CIRC**2 - (PB_D - SPOT) ** 2
)  # half-chord where the penalty arc meets the box


# ----------------------------------------------------------------------------------- geometry
def _side(s, L):
    return (lambda d: d) if s == "l" else (lambda d: L - d)


def model_points(L, W):
    """name -> (X, Y, Z) world point (Z = -height)."""
    P = {
        "half_far": (L / 2, 0, 0),
        "half_near": (L / 2, W, 0),
        "centre": (L / 2, W / 2, 0),
        "circle_far": (L / 2, W / 2 - R_CIRC, 0),
        "circle_near": (L / 2, W / 2 + R_CIRC, 0),
        "circle_left": (L / 2 - R_CIRC, W / 2, 0),
        "circle_right": (L / 2 + R_CIRC, W / 2, 0),
    }
    for s in "lr":
        x = _side(s, L)
        P.update(
            {
                f"{s}_corner_far": (x(0), 0, 0),
                f"{s}_corner_near": (x(0), W, 0),
                f"{s}_pbox_far_gl": (x(0), W / 2 - PB_W / 2, 0),
                f"{s}_pbox_far": (x(PB_D), W / 2 - PB_W / 2, 0),
                f"{s}_pbox_near": (x(PB_D), W / 2 + PB_W / 2, 0),
                f"{s}_pbox_near_gl": (x(0), W / 2 + PB_W / 2, 0),
                f"{s}_gbox_far_gl": (x(0), W / 2 - GB_W / 2, 0),
                f"{s}_gbox_far": (x(GB_D), W / 2 - GB_W / 2, 0),
                f"{s}_gbox_near": (x(GB_D), W / 2 + GB_W / 2, 0),
                f"{s}_gbox_near_gl": (x(0), W / 2 + GB_W / 2, 0),
                f"{s}_spot": (x(SPOT), W / 2, 0),
                f"{s}_arc_far": (x(PB_D), W / 2 - ARC_HALF, 0),
                f"{s}_arc_near": (x(PB_D), W / 2 + ARC_HALF, 0),
                f"{s}_pbox_front_mid": (x(PB_D), W / 2, 0),
                f"{s}_post_far_base": (x(0), W / 2 - GOAL_W / 2, 0),
                f"{s}_post_far_top": (x(0), W / 2 - GOAL_W / 2, -GOAL_H),
                f"{s}_post_near_base": (x(0), W / 2 + GOAL_W / 2, 0),
                f"{s}_post_near_top": (x(0), W / 2 + GOAL_W / 2, -GOAL_H),
                f"{s}_goal_mid": (x(0), W / 2, 0),
            }
        )
    return {k: np.array(v, float) for k, v in P.items()}


def model_lines(L, W):
    """name -> (A, B) 3-D endpoints of each straight marking (painted segment extents)."""
    Lk = {
        "touch_far": ((0, 0, 0), (L, 0, 0)),
        "touch_near": ((0, W, 0), (L, W, 0)),
        "halfway": ((L / 2, 0, 0), (L / 2, W, 0)),
    }
    for s in "lr":
        x = _side(s, L)
        a, b = W / 2 - PB_W / 2, W / 2 + PB_W / 2
        c, d = W / 2 - GB_W / 2, W / 2 + GB_W / 2
        g0, g1 = W / 2 - GOAL_W / 2, W / 2 + GOAL_W / 2
        Lk.update(
            {
                f"{s}_goalline": ((x(0), 0, 0), (x(0), W, 0)),
                f"{s}_pbox_front": ((x(PB_D), a, 0), (x(PB_D), b, 0)),
                f"{s}_pbox_far": ((x(0), a, 0), (x(PB_D), a, 0)),
                f"{s}_pbox_near": ((x(0), b, 0), (x(PB_D), b, 0)),
                f"{s}_gbox_front": ((x(GB_D), c, 0), (x(GB_D), d, 0)),
                f"{s}_gbox_far": ((x(0), c, 0), (x(GB_D), c, 0)),
                f"{s}_gbox_near": ((x(0), d, 0), (x(GB_D), d, 0)),
                f"{s}_crossbar": ((x(0), g0, -GOAL_H), (x(0), g1, -GOAL_H)),
                f"{s}_post_far": ((x(0), g0, 0), (x(0), g0, -GOAL_H)),
                f"{s}_post_near": ((x(0), g1, 0), (x(0), g1, -GOAL_H)),
            }
        )
    return {k: (np.array(a, float), np.array(b, float)) for k, (a, b) in Lk.items()}


def model_circles(L, W):
    """name -> (cx, cy, r, keep) where keep(X) says which part of the circle is painted."""
    return {
        "centre_circle": (L / 2, W / 2, R_CIRC, lambda X: np.ones_like(X, bool)),
        "l_arc": (SPOT, W / 2, R_CIRC, lambda X: X >= PB_D),
        "r_arc": (L - SPOT, W / 2, R_CIRC, lambda X: X <= L - PB_D),
    }


GROUND_LINES = ("touch_far", "touch_near", "halfway") + tuple(
    f"{s}_{n}"
    for s in "lr"
    for n in (
        "goalline",
        "pbox_front",
        "pbox_far",
        "pbox_near",
        "gbox_front",
        "gbox_far",
        "gbox_near",
    )
)
GOAL_LINES3D = tuple(f"{s}_{n}" for s in "lr" for n in ("crossbar", "post_far", "post_near"))


# ------------------------------------------------------------------------------------ camera
def rot(rvec):
    return cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]


def project(cam, P):
    """cam = dict(rvec, C, f, cx, cy). P (N,3) -> (N,2) pixels, (N,) depth."""
    R = rot(cam["rvec"])
    Xc = (np.atleast_2d(P) - np.asarray(cam["C"], float)) @ R.T
    z = Xc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return np.stack(
        [cam["cx"] + cam["f"] * Xc[:, 0] / zs, cam["cy"] + cam["f"] * Xc[:, 1] / zs], 1
    ), z


def ground_homography(cam):
    """pitch (X, Y, 1) on Z=0 -> image pixels (3x3)."""
    K = np.array([[cam["f"], 0, cam["cx"]], [0, cam["f"], cam["cy"]], [0, 0, 1.0]])
    R = rot(cam["rvec"])
    t = -R @ np.asarray(cam["C"], float)
    G = K @ np.column_stack([R[:, 0], R[:, 1], t])
    return G / G[2, 2]


def H_img2pitch(cam):
    H = np.linalg.inv(ground_homography(cam))
    H = H / H[2, 2]
    # sign so that pixels the camera sees map with positive w (the repo convention, see homography.normalize_sign)
    p = np.array([cam["cx"], 2 * cam["cy"] - 1, 1.0])
    return -H if (H @ p)[2] < 0 else H


def cam_from_params(x, cx, cy):
    return dict(rvec=x[0:3], C=x[3:6], f=float(np.exp(x[6])), cx=cx, cy=cy)


def params_from_cam(cam):
    return np.r_[
        np.asarray(cam["rvec"], float).ravel(), np.asarray(cam["C"], float), math.log(cam["f"])
    ]


def init_camera(obj, img, w, h, fs=None, C_hint=None, hint_px_per_m=2.0):
    """Best solvePnP over a focal grid. obj (N,3) world, img (N,2). Returns cam, rms.
    With C_hint (a camera position known from another frame of the same fixed camera), candidates
    are ranked by rms + hint_px_per_m * |C - C_hint| to break line-only degeneracies."""
    obj = np.asarray(obj, np.float64)
    img = np.asarray(img, np.float64)
    best = (None, 1e18)
    for f in fs if fs is not None else w * np.geomspace(0.4, 12, 60):
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_IPPE):
            try:
                ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=flag)
            except cv2.error:
                continue
            if not ok:
                continue
            R = rot(rv)
            C = (-R.T @ tv).ravel()
            if C[2] > -0.5:  # camera must be above the ground (Z negative)
                continue
            cam = dict(rvec=rv.ravel(), C=C, f=f, cx=w / 2, cy=h / 2)
            p, z = project(cam, obj)
            if np.any(z <= 0):
                continue
            e = float(np.sqrt(np.mean(np.sum((p - img) ** 2, 1))))
            sc = e + (
                hint_px_per_m * float(np.linalg.norm(C - np.asarray(C_hint)))
                if C_hint is not None
                else 0.0
            )
            if sc < best[1]:
                best = (cam, sc)
    return best


# ------------------------------------------------------------------------------ paint snapping
def response_map(img, mode, ksize=15):
    """Float image where painted lines of the given colour are bright thin ridges."""
    b, g, r = [c.astype(np.float32) for c in cv2.split(img)]
    if mode == "white":
        base = np.minimum(np.minimum(b, g), r)
    elif mode == "gray":
        base = 0.299 * r + 0.587 * g + 0.114 * b
    elif (
        mode == "yellow"
    ):  # yellow/gold paint: R ~ G, both above B; suppresses white (R=G=B), red (G low), grass (G > R)
        base = np.minimum(r, g) - b - np.abs(r - g)
    elif mode == "orange":  # orange/yellow-orange paint: R high, B low
        base = r - b - 0.5 * np.abs(g - 0.6 * r)
    elif mode == "red":
        base = r - np.maximum(g, b)
    else:
        raise ValueError(mode)
    # ellipse kernel at 1080p; a (separable, much faster) square kernel for the larger 4K kernels
    shape = cv2.MORPH_ELLIPSE if ksize <= 15 else cv2.MORPH_RECT
    k = cv2.getStructuringElement(shape, (ksize, ksize))
    return cv2.morphologyEx(base, cv2.MORPH_TOPHAT, k)


def _profiles(resp, pts, normals, rad, step=0.5, smooth=1.0):
    t = np.arange(-rad, rad + 1e-9, step, dtype=np.float32)
    X = (pts[:, 0:1] + normals[:, 0:1] * t[None]).astype(np.float32)
    Y = (pts[:, 1:2] + normals[:, 1:2] * t[None]).astype(np.float32)
    prof = cv2.remap(resp, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # smooth across the line (sigma `smooth` px); a larger sigma merges double stripes into their centre
    prof = cv2.GaussianBlur(prof, (0, 0), sigmaX=smooth / step, sigmaY=0.01)
    return t, prof


def _peaks(resp, pts, normals, rad, min_resp, smooth=1.0):
    t, prof = _profiles(resp, pts, normals, rad, smooth=smooth)
    k = np.argmax(prof, 1)
    val = prof[np.arange(len(k)), k]
    ok = (k > 0) & (k < len(t) - 1) & (val >= min_resp)
    kk = np.clip(k, 1, len(t) - 2)
    y0, y1, y2 = (
        prof[np.arange(len(k)), kk - 1],
        prof[np.arange(len(k)), kk],
        prof[np.arange(len(k)), kk + 1],
    )
    den = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):  # flat profiles: offset 0 below
        off = np.where(np.abs(den) > 1e-6, 0.5 * (y0 - y2) / den, 0.0)
    off = np.clip(off, -1, 1)
    tt = t[kk] + off * (t[1] - t[0])
    out = pts + normals * tt[:, None]
    return out[ok], val[ok]


def _densify(poly, spacing):
    poly = np.asarray(poly, float)
    out, nrm = [], []
    for a, b in zip(poly[:-1], poly[1:]):
        d = b - a
        Ls = np.linalg.norm(d)
        n = max(2, int(Ls / spacing) + 1)
        tt = np.linspace(0, 1, n)[:-1] if len(out) or True else np.linspace(0, 1, n)
        seg = a + tt[:, None] * d
        out.append(seg)
        nrm.append(np.repeat(np.array([[-d[1], d[0]]]) / max(Ls, 1e-9), len(seg), 0))
    out.append(poly[-1:])
    nrm.append(nrm[-1][-1:])
    return np.vstack(out), np.vstack(nrm)


def snap_line(resp, rough, rad=8, spacing=4, min_resp=8.0, tol=1.2, iters=400, rng=0, smooth=1.0):
    """Snap a rough polyline (2+ points along ONE straight marking) onto the paint ridge.
    Returns (inlier ridge points (N,2), fitted (point, direction), stats)."""
    pts, nrm = _densify(rough, spacing)
    q, v = _peaks(resp, pts, nrm, rad, min_resp, smooth)
    if len(q) < 4:
        return np.zeros((0, 2)), None, dict(n=0)
    R = np.random.default_rng(rng)
    best = None
    for _ in range(iters):
        i, j = R.choice(len(q), 2, replace=False)
        d = q[j] - q[i]
        if np.linalg.norm(d) < 10:
            continue
        nn = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        inl = np.abs((q - q[i]) @ nn) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None:
        return np.zeros((0, 2)), None, dict(n=0)
    for _ in range(3):
        m = q[best].mean(0)
        U, S, Vt = np.linalg.svd(q[best] - m)
        dvec = Vt[0]
        nn = np.array([-dvec[1], dvec[0]])
        best = np.abs((q - m) @ nn) < tol
    res = (q[best] - m) @ nn
    return (
        q[best],
        (m, dvec),
        dict(
            n=int(best.sum()),
            n_cand=len(q),
            rms=float(np.sqrt(np.mean(res**2))),
            frac=float(best.mean()),
        ),
    )


def snap_curve(resp, rough, rad=8, spacing=4, min_resp=8.0, tol=1.5, win=7, smooth=1.0):
    """Snap a rough polyline along a CURVED marking (arc). Outliers (players, other paint) are
    rejected by comparing each ridge offset with the running median of its neighbours."""
    pts, nrm = _densify(rough, spacing)
    q, v = _peaks(resp, pts, nrm, rad, min_resp, smooth)
    if len(q) < 5:
        return np.zeros((0, 2)), dict(n=0)
    # offsets along the normal relative to the rough curve, via nearest rough sample
    # (peaks are in the same order as pts[ok]) -> smooth with a running median of positions
    keep = np.ones(len(q), bool)
    for it in range(2):
        idx = np.where(keep)[0]
        qq = q[idx]
        # distance to the local chord through neighbours: fit a local quadratic would be better;
        # use the median position of neighbours projected on the local normal
        good = np.ones(len(qq), bool)
        for k in range(len(qq)):
            lo, hi = max(0, k - win), min(len(qq), k + win + 1)
            nb = qq[lo:hi]
            if len(nb) < 5:
                continue
            # local 2nd-order fit in a rotated frame
            d = nb[-1] - nb[0]
            if np.linalg.norm(d) < 1e-6:
                continue
            u = d / np.linalg.norm(d)
            n2 = np.array([-u[1], u[0]])
            s = (nb - nb[0]) @ u
            o = (nb - nb[0]) @ n2
            c = np.polyfit(s, o, 2) if len(nb) >= 5 else np.polyfit(s, o, 1)
            ok = (qq[k] - nb[0]) @ u
            pred = np.polyval(c, ok)
            good[k] = abs((qq[k] - nb[0]) @ n2 - pred) < tol
        keep[idx[~good]] = False
    return q[keep], dict(n=int(keep.sum()), n_cand=len(q))


def snap_point(img, xy, mode, ksize=15, rad=6):
    """Centroid of the bright paint blob near xy (for penalty spots)."""
    resp = response_map(img, mode, ksize)
    x, y = int(round(xy[0])), int(round(xy[1]))
    patch = resp[y - rad : y + rad + 1, x - rad : x + rad + 1].astype(float)
    if patch.size == 0 or patch.max() <= 0:
        return np.array(xy, float)
    patch = np.clip(patch - 0.5 * patch.max(), 0, None)
    yy, xx = np.mgrid[-rad : rad + 1, -rad : rad + 1]
    s = patch.sum()
    return np.array([x + (patch * xx).sum() / s, y + (patch * yy).sum() / s])


# ------------------------------------------------------------------------------------ fitting
def _line_res(cam, A, B, q):
    p, z = project(cam, np.stack([A, B]))
    d = p[1] - p[0]
    n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
    return (q - p[0]) @ n


def _circle_poly(cam, c, n=2880, keep=None):
    cx, cy, r, kp = c
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X, Y = cx + r * np.cos(a), cy + r * np.sin(a)
    P = np.stack([X, Y, np.zeros_like(X)], 1)
    p, z = project(cam, P)
    return p, z


def backproject_ground(cam, q):
    """Pixels (N,2) -> ground-plane points (N,3) (Z=0) along the camera rays (nan if parallel/behind)."""
    R = rot(cam["rvec"])
    d = (
        np.stack(
            [(q[:, 0] - cam["cx"]) / cam["f"], (q[:, 1] - cam["cy"]) / cam["f"], np.ones(len(q))], 1
        )
        @ R
    )  # world dirs
    C = np.asarray(cam["C"], float)
    t = -C[2] / np.where(np.abs(d[:, 2]) < 1e-12, 1e-12, d[:, 2])
    P = C[None] + t[:, None] * d
    P[t <= 0] = np.nan
    return P


def _circle_res_fast(cam, circ, q):
    """Signed pixel distance from observed arc pixels to the projected circle (first-order: radial
    metric error at the back-projected ground point times the local pixels-per-metre)."""
    cx, cy, r, _ = circ
    Q = backproject_ground(cam, q)
    v = Q[:, :2] - np.array([cx, cy])
    rho = np.linalg.norm(v, axis=1)
    n = v / np.maximum(rho, 1e-9)[:, None]
    eps = 0.05
    Q2 = Q.copy()
    Q2[:, :2] = Q[:, :2] + eps * n
    p1, _ = project(cam, Q)
    p2, _ = project(cam, Q2)
    scale = np.linalg.norm(p2 - p1, axis=1) / eps
    out = (rho - r) * scale
    return np.where(np.isfinite(out), out, 50.0)


def _circle_res(cam, circ, q):
    p, z = _circle_poly(cam, circ)
    p = p[z > 0]
    if len(p) < 3:
        return np.full(len(q), 50.0)
    d2 = ((q[:, None, :] - p[None]) ** 2).sum(-1)
    k = np.argmin(d2, 1)
    best = np.sqrt(d2[np.arange(len(q)), k])
    for nb in ((k - 1) % len(p), (k + 1) % len(p)):
        a, b = p[k], p[nb]
        ab = b - a
        t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(q - (a + t[:, None] * ab), axis=1))
    return best


def fit_frames(
    frames_obs,
    w,
    h,
    dims,
    fit_dims=(False, False),
    share_C=False,
    loss="soft_l1",
    f_scale=1.0,
    point_w=1.0,
    max_line_samples=40,
    priors=None,
):
    """Joint robust fit. frames_obs: list of dict(cam0, points=[(name, xy, weight)],
    lines=[(name, (N,2))], circles=[(name, (N,2))]). dims = [L, W] (initial). fit_dims=(fit L, fit W).
    Returns (cams, dims, per-frame residual dict)."""
    nF = len(frames_obs)
    x0 = []
    for fo in frames_obs:
        x0.append(params_from_cam(fo["cam0"]))
    x0 = np.concatenate(x0)
    if share_C:
        Cm = np.mean([fo["cam0"]["C"] for fo in frames_obs], 0)
    extra = []
    if fit_dims[0]:
        extra.append(dims[0])
    if fit_dims[1]:
        extra.append(dims[1])
    x0 = np.r_[x0, extra, Cm if share_C else []]

    # subsample line observations
    obs = []
    for fo in frames_obs:
        lines = []
        for name, q in fo.get("lines", []):
            q = np.asarray(q, float)
            if len(q) > max_line_samples:
                q = q[np.linspace(0, len(q) - 1, max_line_samples).astype(int)]
            lines.append((name, q))
        circs = []
        for name, q in fo.get("circles", []):
            q = np.asarray(q, float)
            if len(q) > 2 * max_line_samples:
                q = q[np.linspace(0, len(q) - 1, 2 * max_line_samples).astype(int)]
            circs.append((name, q))
        obs.append((fo.get("points", []), lines, circs))

    def unpack(x):
        k = 7 * nF
        Ld, Wd = dims
        j = k
        if fit_dims[0]:
            Ld = x[j]
            j += 1
        if fit_dims[1]:
            Wd = x[j]
            j += 1
        cams = []
        for i in range(nF):
            xi = x[7 * i : 7 * i + 7].copy()
            if share_C:
                xi[3:6] = x[j : j + 3]
            cams.append(cam_from_params(xi, w / 2, h / 2))
        return cams, (Ld, Wd)

    # pre-pack observations per frame for vectorised residuals
    packs = []
    for pts, lines, circs in obs:
        pk = dict(
            pn=[n for n, _, _ in pts],
            pxy=np.array([xy for _, xy, _ in pts], float).reshape(-1, 2),
            pw=np.array([wt for _, _, wt in pts], float),
            ln=[n for n, _ in lines],
            lq=[np.asarray(q, float) for _, q in lines],
            cn=[n for n, _ in circs],
            cq=[np.asarray(q, float) for _, q in circs],
        )
        packs.append(pk)

    def resid(x):
        cams, (Ld, Wd) = unpack(x)
        Pm, Lm, Cm_ = model_points(Ld, Wd), model_lines(Ld, Wd), model_circles(Ld, Wd)
        out = []
        for cam, pk in zip(cams, packs):
            if pk["pn"]:
                P = np.array([Pm[n] for n in pk["pn"]])
                p, z = project(cam, P)
                out.append(((p - pk["pxy"]) * (pk["pw"] * point_w)[:, None]).ravel())
            if pk["ln"]:
                AB = np.array([np.r_[Lm[n][0], Lm[n][1]] for n in pk["ln"]]).reshape(-1, 3)
                pab, _ = project(cam, AB)
                for k, q in enumerate(pk["lq"]):
                    a, b = pab[2 * k], pab[2 * k + 1]
                    d = b - a
                    nrm = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
                    out.append((q - a) @ nrm)
            for n, q in zip(pk["cn"], pk["cq"]):
                out.append(_circle_res_fast(cam, Cm_[n], q))
        if priors:
            out.append(np.atleast_1d(priors(cams, (Ld, Wd))))
        return np.concatenate([np.ravel(o) for o in out])

    # sparse Jacobian: frame i's residuals depend only on its own 7 params (+ dims, + shared C)
    sizes = []
    for pts, lines, circs in obs:
        sizes.append(2 * len(pts) + sum(len(q) for _, q in lines) + sum(len(q) for _, q in circs))
    nres = sum(sizes) + (len(np.atleast_1d(priors(unpack(x0)[0], unpack(x0)[1]))) if priors else 0)
    from scipy.sparse import lil_matrix

    S = lil_matrix((nres, len(x0)), dtype=int)
    r0 = 0
    for i, n in enumerate(sizes):
        S[r0 : r0 + n, 7 * i : 7 * i + 7] = 1
        S[r0 : r0 + n, 7 * nF :] = 1
        r0 += n
    if priors:
        S[r0:, :] = 1
    kw = dict(jac_sparsity=S.tocsc()) if nF > 1 else {}
    r = least_squares(resid, x0, loss=loss, f_scale=f_scale, x_scale="jac", max_nfev=4000, **kw)
    cams, dd = unpack(r.x)
    # per-frame residual summaries
    Pm, Lm, Cm_ = model_points(*dd), model_lines(*dd), model_circles(*dd)
    stats = []
    for cam, (pts, lines, circs) in zip(cams, obs):
        st = dict(points={}, lines={}, circles={})
        for name, xy, wt in pts:
            p, z = project(cam, Pm[name][None])
            st["points"][name] = float(np.linalg.norm(p[0] - np.asarray(xy)))
        for name, q in lines:
            A, B = Lm[name]
            rr = np.abs(_line_res(cam, A, B, q))
            st["lines"][name] = (float(np.median(rr)), float(rr.max()), len(q))
        for name, q in circs:
            rr = _circle_res(cam, Cm_[name], q)
            st["circles"][name] = (float(np.median(rr)), float(rr.max()), len(q))
        allr = (
            list(st["points"].values())
            + [v for nm, q in lines for v in np.abs(_line_res(cam, *Lm[nm], q))]
            + [v for nm, q in circs for v in _circle_res(cam, Cm_[nm], q)]
        )
        st["median"] = float(np.median(allr)) if allr else float("nan")
        st["max"] = float(np.max(allr)) if allr else float("nan")
        st["p90"] = float(np.percentile(allr, 90)) if allr else float("nan")
        stats.append(st)
    return cams, dd, stats, r


# ------------------------------------------------------------------------------------ drawing
def marking_polylines(cam, L, W, w, h, step=0.25, include_goals=True, margin=0):
    """name -> list of in-front (N,2) pixel polylines of each marking (clipped to depth>0)."""
    out = {}
    Lm = model_lines(L, W)
    for name, (A, B) in Lm.items():
        if not include_goals and name in GOAL_LINES3D:
            continue
        n = max(2, int(np.linalg.norm(B - A) / step))
        P = A + np.linspace(0, 1, n)[:, None] * (B - A)
        out[name] = _split(cam, P, w, h, margin)
    for name, (cx, cy, r, keep) in model_circles(L, W).items():
        a = np.linspace(0, 2 * np.pi, 1441)
        X, Y = cx + r * np.cos(a), cy + r * np.sin(a)
        m = keep(X)
        P = np.stack([X, Y, np.zeros_like(X)], 1)
        segs = []
        cur = []
        for i in range(len(P)):
            if m[i]:
                cur.append(P[i])
            elif cur:
                segs.append(np.array(cur))
                cur = []
        if cur:
            segs.append(np.array(cur))
        polys = []
        for s in segs:
            polys += _split(cam, s, w, h, margin)
        out[name] = polys
    return out


def _split(cam, P, w, h, margin):
    p, z = project(cam, P)
    good = (
        (z > 0.1)
        & (p[:, 0] > -margin - 4 * w)
        & (p[:, 0] < w + margin + 4 * w)
        & (p[:, 1] > -margin - 4 * h)
        & (p[:, 1] < h + margin + 4 * h)
    )
    polys, cur = [], []
    for i in range(len(p)):
        if good[i]:
            cur.append(p[i])
        elif cur:
            polys.append(np.array(cur))
            cur = []
    if cur:
        polys.append(np.array(cur))
    return [q for q in polys if len(q) >= 2]


COLORS = {
    "line": (255, 0, 255),
    "goal": (0, 165, 255),
    "obs": (0, 255, 0),
    "pt": (0, 255, 255),
    "rough": (255, 255, 0),
}


def draw(img, cam, L, W, thick=2, obs=None, pts=None, draw_names=False):
    out = img.copy()
    h, w = img.shape[:2]
    for name, polys in marking_polylines(cam, L, W, w, h).items():
        col = COLORS["goal"] if name in GOAL_LINES3D else COLORS["line"]
        for q in polys:
            cv2.polylines(
                out, [np.round(q * 4).astype(np.int32)], False, col, thick, cv2.LINE_AA, shift=2
            )
    if obs is not None:
        for q in obs:
            for x, y in np.asarray(q):
                cv2.circle(
                    out,
                    (int(round(x * 4)), int(round(y * 4))),
                    4 * max(1, thick // 2),
                    COLORS["obs"],
                    -1,
                    cv2.LINE_AA,
                    shift=2,
                )
    if pts:
        Pm = model_points(L, W)
        for name, xy in pts.items():
            p, z = project(cam, Pm[name][None])
            cv2.drawMarker(
                out,
                (int(round(xy[0])), int(round(xy[1]))),
                COLORS["pt"],
                cv2.MARKER_TILTED_CROSS,
                6 * thick,
                max(1, thick // 2),
            )
            if z[0] > 0:
                cv2.circle(
                    out,
                    (int(round(p[0, 0])), int(round(p[0, 1]))),
                    3 * thick,
                    (0, 0, 255),
                    max(1, thick // 2),
                )
    return out
