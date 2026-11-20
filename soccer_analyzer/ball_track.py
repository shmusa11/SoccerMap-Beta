"""Ball track: candidates -> appearance probability -> global DP with camera-compensated motion,
an explicit "no ball" state, and optional sparse vision-model anchors.

All coordinates are full-resolution original pixels; frames are ORIGINAL frame indices on the
render grid (15 Hz).

Candidates per render frame f:
  * WASB peaks on the full frame (`WASBBallDetector.peaks`: top 20 local maxima, no 0.30
    threshold), CLAHE input on night footage (`is_night`).
  * Camera-compensated 3-frame difference (`motion_blobs` in pass 1, `motion_candidates_from_blobs`
    once person boxes are known): the previous / next DISTINCT raw frames warped onto f,
    min(|F-P|, |F-N|), robust threshold, blobs sized against the expected ball diameter
    (0.125 x median person-box height near that image row).
  * Person boxes (conf >= 0.3) for the jersey/head feature.

`merge_candidates` -> `score_candidates` (logistic model) -> `track_ball2` (second-order DP over
(candidate, predecessor) states, camera motion through the chained frame homographies, a
physical speed limit in m/s via px-per-metre from person heights, a capped velocity-change cost,
and a "no ball" state). A segment is reported only when its mean confidence clears
`min_mean_p`; otherwise its frames are None (ball not visible / not trusted). Inside a reported
segment, gaps of up to `max_gap` render frames between detections ARE bridged by interpolation,
with conf = interp_conf x the segment confidence; FastPipeline keeps only points whose conf
clears its gate, so bridged points of a weak segment are dropped. `track_with_anchors` adds 1 Hz vision-model crop checks around the DP's hypothesis
(skipped on night frames, where the measured hit rate was ~7%).

Measured (lab 2026-10-12, hand-verified labels, 4 clips): the previous WASB-threshold +
BallPathSolver path put the ball within 8 px (16 px at 4K) on 3% of held-out ball-present frames
and a wrong point on 55%; this tracker with Luna anchors at gate 0.10: 73% hit, 3% wrong
(dev split 38% / 8%). Recall is limited by candidates (night: the ball is among them on ~37%).
"""

import threading

import numpy as np

FEATS = [
    "ws",
    "hw",
    "wrank",
    "ms",
    "hm",
    "mrank",
    "m_lr",
    "m_lr2",
    "dark",
    "in_body",
    "logn",
    "both",
]
DEFAULT_PARAMS = dict(
    theta=-3.0,  # node gain = logit(p) - theta (p ~ 0.05 breaks even)
    c_start=2.0,  # cost to start a new segment
    lam_d=1.0,  # speed cost weight (per unit of vmax * dt)
    lam_g=0.25,  # cost per skipped render frame inside a segment
    max_gap=8,  # render frames a segment may bridge (0.53 s at 15 Hz)
    vmax_mps=35.0,  # physical ball speed limit
    slack_px=12.0,  # position slack on top of vmax * dt, HD px (scaled by radius_px/8 for 4K)
    min_nodes=3,  # segments with fewer detections are reported as "not visible"
    min_mean_p=0.0,  # segments whose mean p is lower are reported as "not visible"
    interp_conf=0.5,  # conf factor for frames bridged by interpolation (conf = segment mean p)
    anchor_p=0.97,  # probability given to a VLM-confirmed point
    neg_factor=0.25,  # p multiplier for candidates inside a crop the VLM saw no ball in
)


# Logistic appearance model fitted on the lab's dev split (4 clips, 95 positive candidates):
# p(ball | candidate features FEATS).
APPEARANCE_MODEL = dict(
    w=[
        0.34818513,
        -0.01080798,
        -0.15579844,
        0.07180175,
        0.05088114,
        -0.26642713,
        0.38820715,
        0.33903937,
        -0.53565881,
        -0.17071243,
        -0.07942056,
        0.16083775,
    ],
    b=-4.32284623,
    mu=[
        0.0959437,
        0.38695556,
        0.78091496,
        0.36698338,
        0.62770849,
        0.66967607,
        0.08492079,
        0.1468754,
        0.4035894,
        0.26986212,
        3.48973227,
        0.01466404,
    ],
    sd=[
        0.16059272,
        0.48704809,
        0.33073559,
        0.32658908,
        0.48342332,
        0.34218565,
        0.37371278,
        0.30001581,
        0.49062386,
        0.44388008,
        0.3185899,
        0.12020264,
    ],
)


# ---------------------------------------------------------------------------- candidates
def merge_candidates(frames, sources, boxes, radius_px):
    """sources: {"wasb": {f: (n,3) x,y,s}, "motion": {f: (n,6) x,y,s,area_ratio,in_body,dark}}
    boxes: {f: (m,4) person xyxy}. Points of different sources within radius_px merge into one
    candidate. Returns {f: (xy (n,2), X (n, len(FEATS)))}."""
    out = {}
    for f in frames:
        w = _clean(sources.get("wasb", {}).get(f), 3)
        m = _clean(sources.get("motion", {}).get(f), 6)
        pts, rows = [], []
        for r, (x, y, s) in enumerate(w[:, :3]):
            pts.append([x, y])
            rows.append(dict(ws=s, hw=1.0, wrank=r / 20.0))
        for r, c in enumerate(m):
            x, y = c[0], c[1]
            j = None
            if pts:
                d = np.hypot(np.array(pts)[:, 0] - x, np.array(pts)[:, 1] - y)
                k = int(np.argmin(d))
                if d[k] <= radius_px and "ms" not in rows[k]:
                    j = k
            mr = dict(
                ms=c[2], hm=1.0, mrank=r / 20.0, m_lr=float(np.log(max(c[3], 1e-3))), dark=c[5]
            )
            if j is None:
                pts.append([x, y])
                rows.append(mr)
            else:
                rows[j].update(mr)
        bx = boxes.get(f, np.zeros((0, 4)))
        X = np.zeros((len(rows), len(FEATS)), np.float32)
        for i, (p, r) in enumerate(zip(pts, rows)):
            r.setdefault("wrank", 1.0)
            r.setdefault("mrank", 1.0)
            r["m_lr2"] = r.get("m_lr", 0.0) ** 2
            r["in_body"] = float(
                any(
                    b[0] - 4 <= p[0] <= b[2] + 4 and b[1] - 4 <= p[1] <= b[1] + 0.78 * (b[3] - b[1])
                    for b in bx
                )
            )
            r["logn"] = float(np.log1p(len(rows)))
            r["both"] = float(r.get("hw", 0) * r.get("hm", 0))
            X[i] = [r.get(k, 0.0) for k in FEATS]
        out[f] = (np.array(pts, np.float64).reshape(-1, 2), X)
    return out


def _clean(a, k):
    if a is None:
        return np.zeros((0, k), np.float32)
    a = np.asarray(a, np.float32)
    return a[np.isfinite(a[:, 0])]


# ---------------------------------------------------------------------------- tracking
def _cum_h(G, T):
    C = [np.eye(3)]
    for t in range(1, T):
        g = None if G is None else G[t]
        g = np.eye(3) if g is None or not np.all(np.isfinite(g)) else np.asarray(g, float)
        C.append(g @ C[-1])
    return C


def _proj(H, xy):
    p = np.column_stack([xy, np.ones(len(xy))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def apply_anchors(cands, anchors, radius_px, params=DEFAULT_PARAMS):
    """anchors: {f: {"xy": (x, y) | None, "box": (x0, y0, x1, y1)}} from VLM crop queries.
    Confirmed point -> added as a high-p candidate (other candidates in the crop down-weighted);
    "no ball in this crop" -> every candidate inside the crop down-weighted."""
    out = {f: c.copy() for f, c in cands.items()}
    for f, a in (anchors or {}).items():
        if f not in out:
            continue
        c = out[f]
        x0, y0, x1, y1 = a["box"]
        inside = (
            (c[:, 0] >= x0) & (c[:, 0] < x1) & (c[:, 1] >= y0) & (c[:, 1] < y1)
            if len(c)
            else np.zeros(0, bool)
        )
        if a.get("xy") is None:
            c[inside, 2] *= params["neg_factor"]
        else:
            far = (
                np.hypot(c[:, 0] - a["xy"][0], c[:, 1] - a["xy"][1]) > 2 * radius_px
                if len(c)
                else inside
            )
            c[inside & far, 2] *= params["neg_factor"]
            c = np.vstack([c, [a["xy"][0], a["xy"][1], params["anchor_p"]]])
        out[f] = c
    return out


# ---------------------------------------------------------------------------- VLM anchors
BALL_PROMPT = (
    "This soccer image is {W}x{H} pixels. Find the match ball (it may be small, partly "
    'hidden, in the air or at a player\'s feet). Return JSON {"ball": [x, y] or null, '
    '"confidence": 0.0-1.0} with the ball centre in absolute pixel coordinates of this '
    "image. Return null if the ball is not visible."
)


def _crop_box(x, y, W, H):
    """Half-width x half-height window around (x, y), snapped to a W/8 x H/8 grid (so small
    hypothesis changes reuse cached answers) and clipped to the frame."""
    cw, ch = W // 2, H // 2
    sx, sy = W // 8, H // 8
    x0 = int(np.clip(round((x - cw / 2) / sx) * sx, 0, W - cw))
    y0 = int(np.clip(round((y - ch / 2) / sy) * sy, 0, H - ch))
    return x0, y0, x0 + cw, y0 + ch


def is_night(img, dark_frac=0.33):
    """Night / floodlit footage: a large share of near-black pixels (sky, unlit surroundings).
    Measured grey<30 fraction: bardNight 0.37-0.45, union (dull day) 0.20-0.25, oneonta 0.00,
    rpi4k 0.04-0.06. (Mean brightness does NOT separate them: union's mean is darker than bardNight's.)"""
    g = img.mean(2) if img.ndim == 3 else img
    return float((g[::4, ::4] < 30).mean()) > dark_frac


# ---------------------------------------------------------------------------- second-order DP
# tuned by random search on the DEV split (run_lab.rtune, objective hit - 0.5 wrong - 0.5 false-present);
# min_mean_p 0.10 = the dev-best "visible" gate (lower it to 0.05 for more coverage, more wrong points)
DEFAULT_PARAMS2 = dict(DEFAULT_PARAMS, lam_a=1.0, sig_a=0.25, acc_cap=3.0, top_m=12, beam=4)
DEFAULT_PARAMS2.update(
    dict(
        theta=-4.7773,
        c_start=1.2857,
        lam_d=0.3854,
        lam_g=0.0441,
        max_gap=4,
        min_nodes=4,
        vmax_mps=19.5959,
        lam_a=0.2228,
        sig_a=0.925,
        acc_cap=1.5949,
        top_m=16,
        slack_px=15.6481,
        min_mean_p=0.1,
    )
)


def track_ball2(
    frames, cands, G=None, anchors=None, fps=15.0, scale=None, params=None, radius_px=8.0
):
    """Global DP whose state is (candidate, predecessor), so each link also pays for the
    change of velocity (camera-compensated): cost lam_a * min(|x_i - x_pred| / (sig_a*scale*k), acc_cap)
    where x_pred extrapolates the previous link's velocity. Kicks/deflections pay at most acc_cap.
    Only the top_m candidates by p per frame enter the solve, and each candidate keeps its `beam`
    best predecessor states for the velocity term (beam search; exact when beam >= 1 + max_gap*top_m)."""
    P = dict(DEFAULT_PARAMS2, **(params or {}))
    frames = list(frames)
    T, K, M = len(frames), int(P["max_gap"]), int(P["top_m"])
    if anchors:
        cands = apply_anchors(cands, anchors, radius_px, P)
    C = _cum_h(G, T)
    Ci = [np.linalg.inv(c) for c in C]
    sc = np.broadcast_to(np.asarray(scale if scale is not None else 30.0, float), (T,))
    X, p = [], []
    for f in frames:
        c = np.asarray(cands.get(f, np.zeros((0, 3))), float)
        c = c[np.argsort(-c[:, 2])[:M]] if len(c) else c.reshape(0, 3)
        X.append(c[:, :2])
        p.append(np.clip(c[:, 2], 1e-4, 1 - 1e-4))
    g = [np.log(q / (1 - q)) - P["theta"] for q in p]
    S = 1 + K * M  # slot 0 = segment start; slot 1 + (k-1)*M + j = predecessor j at t-k
    V, B, VEL, TOP = [], [], [], []
    null = np.zeros(T)
    null_from = np.full(T, -1)
    for t in range(T):
        n = len(X[t])
        prev_null = null[t - 1] if t > 0 else 0.0
        v = np.full((n, S), -np.inf)
        b = np.full((n, S), -1, int)
        vel = np.full((n, S, 2), np.nan)
        v[:, 0] = prev_null - P["c_start"] + g[t]
        vpx = P["vmax_mps"] * sc[t] / fps
        sa = P["sig_a"] * sc[t]
        for k in range(1, K + 1):
            tt = t - k
            if tt < 0 or n == 0:
                break
            m = len(X[tt])
            if m == 0:
                continue
            Hk = C[t] @ Ci[tt]
            pj = _proj(Hk, X[tt])  # (m,2) j in frame t
            D = np.hypot(
                pj[:, None, 0] - X[t][None, :, 0], pj[:, None, 1] - X[t][None, :, 1]
            )  # (m,n)
            link = P["lam_d"] * D / (vpx * k) + P["lam_g"] * (k - 1)
            link[D > vpx * k + P["slack_px"] * radius_px / 8.0] = np.inf
            # predecessor states of j: extrapolate with their velocity (frame tt coords) k steps
            top = TOP[tt]  # (m,L) best predecessor slots of j (beam)
            L = top.shape[1]
            vj = np.take_along_axis(VEL[tt], top[..., None], 1)  # (m,L,2)
            ext = X[tt][:, None, :] + vj * k  # (m,L,2) in tt coords
            ok = np.isfinite(ext[..., 0])
            e2 = np.where(ok[..., None], ext, 0.0).reshape(-1, 2)
            pe = _proj(Hk, e2).reshape(m, L, 2)
            err = np.hypot(
                pe[None, :, :, 0] - X[t][:, None, None, 0],
                pe[None, :, :, 1] - X[t][:, None, None, 1],
            )  # (n,m,L)
            acc = P["lam_a"] * np.minimum(err / (sa * k), P["acc_cap"])
            acc = np.where(ok[None], acc, 0.0)  # start slots: no velocity yet
            tot = np.take_along_axis(V[tt], top, 1)[None] - acc  # (n,m,L)
            bl = np.argmax(tot, 2)  # (n,m)
            bv = np.take_along_axis(tot, bl[..., None], 2)[..., 0] - link.T  # (n,m)
            bi = top[np.arange(m)[None, :], bl]  # slot index at tt
            sl = 1 + (k - 1) * M + np.arange(m)
            v[:, sl] = bv + g[t][:, None]
            b[:, sl] = bi
            vel[:, sl, :] = (X[t][:, None, :] - pj[None, :, :]) / k
        V.append(v)
        B.append(b)
        VEL.append(vel)
        TOP.append(
            np.argsort(-v, 1)[:, : int(P["beam"])] if n else np.zeros((0, int(P["beam"])), int)
        )
        if t > 0:
            null[t] = prev_null
            if V[t - 1].size and np.max(V[t - 1]) > prev_null:
                null[t] = np.max(V[t - 1])
                null_from[t] = int(np.argmax(V[t - 1]))
    # backtrack
    path, seg = [], 0
    if V[T - 1].size and np.max(V[T - 1]) > null[T - 1]:
        i, s = np.unravel_index(int(np.argmax(V[T - 1])), V[T - 1].shape)
        state = (T - 1, int(i), int(s))
    else:
        state = ("null", T - 1)
    while True:
        if state[0] == "null":
            t = state[1]
            if t <= 0:
                break
            j = null_from[t]
            if j >= 0:
                i, s = np.unravel_index(int(j), V[t - 1].shape)
                state = (t - 1, int(i), int(s))
                seg += 1
            else:
                state = ("null", t - 1)
            continue
        t, i, s = state
        path.append((t, i, seg))
        if s == 0:
            seg += 1
            if t == 0:
                break
            state = ("null", t - 1)
        else:
            k = (s - 1) // M + 1
            j = (s - 1) % M
            state = (t - k, j, int(B[t][i, s]))
    path.reverse()
    out = {f: None for f in frames}
    segs = {}
    for t, i, s in path:
        segs.setdefault(s, []).append((t, i))
    for nodes in segs.values():
        ps = [p[t][i] for t, i in nodes]
        sp = float(np.mean(ps))  # segment confidence: mean appearance p of its detections
        if len(nodes) < P["min_nodes"] or sp < P["min_mean_p"]:
            continue
        for k, (t, i) in enumerate(nodes):
            out[frames[t]] = (float(X[t][i, 0]), float(X[t][i, 1]), sp)
            if k + 1 < len(nodes):
                t2, i2 = nodes[k + 1]
                for u in range(t + 1, t2):
                    a = _proj(C[u] @ Ci[t], X[t][i : i + 1])[0]
                    c = _proj(C[u] @ Ci[t2], X[t2][i2 : i2 + 1])[0]
                    w = (u - t) / (t2 - t)
                    q = (1 - w) * a + w * c
                    out[frames[u]] = (float(q[0]), float(q[1]), P["interp_conf"] * sp)
    return out


def score_candidates(feats, model=None):
    """Logistic appearance model -> {f: (n, 3) x, y, p}."""
    model = APPEARANCE_MODEL if model is None else model
    w = np.array(model["w"])
    b = model["b"]
    mu = np.array(model["mu"])
    sd = np.array(model["sd"])
    out = {}
    for f, (xy, X) in feats.items():
        if len(xy) == 0:
            out[f] = np.zeros((0, 3))
            continue
        z = ((X - mu) / sd) @ w + b
        out[f] = np.column_stack([xy, 1.0 / (1.0 + np.exp(-z))])
    return out


def vlm_anchors(frames, track, get_frame, ask, every=15, night_dark_frac=0.33, max_queries=20):
    """Sparse vision-model checks of a first-pass track. At every `every`-th render frame: if the
    track has a point, ask about the half-size crop around it ("confirm / deny"); otherwise about
    the full frame ("re-acquire"). Skipped on night frames (`is_night`): the measured point hit
    rate there was ~0.07, so its answers would hurt.

    get_frame(f) -> full-res BGR or None. ask(questions) gets every question at once, as a list
    of (f, image, box), so they can run in parallel, and returns one answer per question: a dict
    with "ball": [x, y] in that image's pixels or None (no ball seen), or None when the call
    failed / was not made. Returns ({f: {"xy", "box"}}, number of questions asked)."""
    questions = []
    for k in range(0, len(frames), every):
        if len(questions) >= max_queries:
            break
        f = frames[k]
        img = get_frame(f)
        if img is None or is_night(img, night_dark_frac):
            continue
        H, W = img.shape[:2]
        pt = track.get(f)
        box = _crop_box(pt[0], pt[1], W, H) if pt is not None else (0, 0, W, H)
        questions.append((f, np.ascontiguousarray(img[box[1] : box[3], box[0] : box[2]]), box))
    answers = ask(questions) if questions else []
    anchors = {}
    for (f, _, box), r in zip(questions, answers):
        if r is None:
            continue
        b = r.get("ball")
        xy = None
        if isinstance(b, (list, tuple)) and len(b) >= 2:
            try:
                xy = (float(b[0]) + box[0], float(b[1]) + box[1])
            except (TypeError, ValueError):
                xy = None
            if xy is not None and not (box[0] <= xy[0] < box[2] and box[1] <= xy[1] < box[3]):
                xy = None
        anchors[f] = dict(xy=xy, box=box)
    return anchors, len(questions)


def track_with_anchors(
    frames,
    cands,
    G,
    get_frame,
    ask,
    fps=15.0,
    scale=None,
    params=None,
    radius_px=8.0,
    every=15,
    night_dark_frac=0.33,
    max_queries=20,
):
    """Two passes: DP -> vision-model checks around the DP's hypothesis -> DP with anchors."""
    t1 = track_ball2(frames, cands, G, fps=fps, scale=scale, params=params, radius_px=radius_px)
    anchors, n = vlm_anchors(frames, t1, get_frame, ask, every, night_dark_frac, max_queries)
    t2 = track_ball2(
        frames, cands, G, anchors=anchors, fps=fps, scale=scale, params=params, radius_px=radius_px
    )
    return t2, anchors, n


# ---------------------------------------------------------------------------- motion candidates
_TLS = threading.local()  # per-thread CLAHE: an OpenCV CLAHE object is not thread-safe
GRAY_WIDTH = 480  # duplicate-frame / overlay tests run on 480-px greys
MAX_BLOBS = 400


def small_gray(frame):
    """480-px grey (uint8) for duplicate-frame and static-overlay tests."""
    import cv2

    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h = int(round(GRAY_WIDTH * g.shape[0] / g.shape[1]))
    return cv2.resize(g, (GRAY_WIDTH, h), interpolation=cv2.INTER_AREA)


def is_duplicate(g1, g0):
    """Repeated frame (screen recording / frame-rate conversion / paused player): almost no pixel
    of the 480-px grey changes by more than 6 levels (measured: true repeats < 0.05%, genuine
    frames of a static night camera 0.2-0.5%, panning day footage 4-15%)."""
    d = np.abs(g1.astype(np.int16) - g0.astype(np.int16))
    return float((d > 6).mean()) < 0.0005 and float(d.mean()) < 0.5


def static_overlay_mask(grays, size):
    """Screen-static GRAPHICS (scoreboard, player UI) as a bool mask of `size` (w, h): pixels
    whose raw value stays constant (90th percentile |diff| < 3 over consecutive greys taken with
    clear camera motion) AND that are textured (Sobel > 40 on the temporal median). Flat turf or
    sky also stays constant but cannot create false candidates. Empty without camera motion.
    grays: 480-px greys of every 2nd raw frame, in order."""
    import cv2

    keys = list(grays)
    diffs = []
    for g0, g1 in zip(keys[:-1], keys[1:]):
        d = np.abs(g1.astype(np.int16) - g0.astype(np.int16)).astype(np.float32)
        if np.median(d) > 2.0:
            diffs.append(d)
    if len(diffs) < 5:
        return np.zeros((size[1], size[0]), bool)
    p = np.percentile(np.stack(diffs), 90, axis=0)
    med = np.median(np.stack(keys[::2]), 0).astype(np.float32)
    gr = np.hypot(cv2.Sobel(med, cv2.CV_32F, 1, 0), cv2.Sobel(med, cv2.CV_32F, 0, 1))
    m = ((p < 3) & (gr > 40)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    for k in range(1, n):
        if st[k, 4] >= 60:
            keep[lbl == k] = 1
    keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))
    return cv2.resize(keep, size, interpolation=cv2.INTER_NEAREST) > 0


def motion_blobs(cur, prev, nxt, k_sigma=5.0, min_th=8.0, work_width=1920.0):
    """Pass-1 half of the motion candidates (needs no person boxes): connected components of the
    camera-compensated 3-frame difference. cur/prev/nxt: full-res BGR of the render frame and
    its previous / next DISTINCT raw frames (None when unknown). Returns (n, 9) float32 rows
    x, y (full-res weighted centroid), row (full-res bbox centre y), w, h, area (work px),
    peak / threshold, dark, work scale; at most MAX_BLOBS rows (strongest first)."""
    import cv2

    from soccer_analyzer.fast_pipeline import _gray_small, frame_motion

    cl = getattr(_TLS, "clahe", None)
    if cl is None:
        cl = _TLS.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    empty = np.zeros((0, 9), np.float32)
    H, W = cur.shape[:2]
    s = min(1.0, float(work_width) / W)
    ww, wh = int(round(W * s)), int(round(H * s))
    S = np.diag([s, s, 1.0])

    def work(f):
        return cv2.resize(f, (ww, wh), interpolation=cv2.INTER_AREA) if s < 1 else f

    wf = work(cur)
    gf, sc = _gray_small(cur, 640, cl)
    terms, valid = [], np.ones((wh, ww), bool)
    for J in (prev, nxt):
        if J is None:
            continue
        gj, _ = _gray_small(J, 640, cl)
        G = frame_motion(gj, gf, sc)
        if G is None:
            continue
        Gw = S @ G @ np.linalg.inv(S)
        warped = cv2.warpPerspective(work(J), Gw, (ww, wh), flags=cv2.INTER_LINEAR, borderValue=0)
        ones = np.full((wh, ww), 255, np.uint8)
        vm = cv2.warpPerspective(ones, Gw, (ww, wh), flags=cv2.INTER_NEAREST) > 0
        valid &= cv2.erode(vm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        terms.append(wf.astype(np.int16) - warped.astype(np.int16))
    if len(terms) < 2:
        return empty
    absd = np.minimum(np.abs(terms[0]).max(2), np.abs(terms[1]).max(2)).astype(np.float32)
    signed = np.minimum(terms[0].mean(2), terms[1].mean(2))
    absd[~valid] = 0
    absd = cv2.GaussianBlur(absd, (3, 3), 0)
    vals = absd[valid]
    sig = 1.4826 * float(np.median(np.abs(vals - np.median(vals)))) + 1.0 if vals.size else 3.0
    th = max(min_th, k_sigma * sig)
    n, lbl, st, _ = cv2.connectedComponentsWithStats((absd > th).astype(np.uint8), connectivity=8)
    if n <= 1:
        return empty
    idx = np.flatnonzero(lbl)
    lab = lbl.ravel()[idx]
    wv = absd.ravel()[idx].astype(np.float64)
    ys, xs = np.divmod(idx, ww)
    sw = np.bincount(lab, weights=wv, minlength=n)
    cx = np.bincount(lab, weights=wv * xs, minlength=n) / np.maximum(sw, 1e-9)
    cy = np.bincount(lab, weights=wv * ys, minlength=n) / np.maximum(sw, 1e-9)
    peak = np.zeros(n)
    np.maximum.at(peak, lab, wv)
    rows = []
    for k in range(1, n):
        x, y, w, h, area = st[k]
        dark = float(signed[int(round(cy[k])), int(round(cx[k]))] < 0)
        rows.append(
            (
                (cx[k] + 0.5) / s - 0.5,
                (cy[k] + 0.5) / s - 0.5,
                (y + h / 2) / s,
                w,
                h,
                area,
                peak[k] / th,
                dark,
                s,
            )
        )
    out = np.array(rows, np.float32).reshape(-1, 9)
    return out[np.argsort(-out[:, 6])[:MAX_BLOBS]]


def motion_candidates_from_blobs(blobs, boxes, overlay=None, K=20):
    """Second half of the motion candidates: size-filter the blobs against the expected ball
    diameter (0.125 x median height of the 5 person boxes whose feet are nearest in y) and score
    them. boxes: (m, 4) person xyxy (conf >= 0.3) of this frame; overlay: optional bool mask
    (full-res size) of screen-static graphics, whose blobs are dropped. Returns (n, 6) x, y,
    score, area_ratio, in_body, dark (top K by score)."""
    blobs = np.asarray(blobs, np.float32).reshape(-1, 9)
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    hh = boxes[:, 3] - boxes[:, 1]
    fallback = float(np.median(hh)) if len(hh) else 60.0

    def diam(y):
        if len(boxes) < 3:
            return 0.125 * fallback
        o = np.argsort(np.abs(boxes[:, 3] - y))[:5]
        return 0.125 * float(np.median(hh[o]))

    out = []
    for X, Y, row, w, h, area, pk, dark, s in blobs:
        if overlay is not None:
            oy = int(np.clip(round(Y), 0, overlay.shape[0] - 1))
            ox = int(np.clip(round(X), 0, overlay.shape[1] - 1))
            if overlay[oy, ox]:
                continue
        dexp = diam(row) * s
        ratio = area / max(2.0, np.pi * dexp * dexp / 4)
        if ratio < 0.2 or ratio > 8 or max(w, h) > max(4 * dexp, dexp + 30 * s):
            continue
        size_f = float(np.exp(-0.5 * (np.log(ratio) / 0.9) ** 2))
        contrast = 1.0 - np.exp(-(float(pk) - 1.0) / 1.5)
        in_body = any(
            b[0] - 2 <= X <= b[2] + 2 and b[1] <= Y <= b[1] + 0.75 * (b[3] - b[1]) for b in boxes
        )
        out.append((X, Y, max(0.01, contrast * size_f), ratio, float(in_body), float(dark)))
    out.sort(key=lambda c: -c[2])
    return np.array(out[:K], np.float32).reshape(-1, 6)


def motion_candidates(cur, prev, nxt, boxes, overlay=None, K=20):
    """Camera-compensated 3-frame-difference ball candidates of one frame (both halves)."""
    return motion_candidates_from_blobs(motion_blobs(cur, prev, nxt), boxes, overlay, K)
