"""Offline player tracker: every on-field player keeps a box on every rendered frame, with one
identity, through camera pans, detector flicker and short occlusions.

Why not ByteTrack: ByteTrack's Kalman filter is camera-blind. During a pan of more than ~20 px
per detection interval it matched 0-2 of 12-22 detections, so players blinked out frame after
frame although YOLO had found them (lab, 2026-10-12: 65-93% of the dropped boxes were people the
detector did see). This tracker warps every track by the measured camera motion first.

    drawn, tracks = track_people(render_frames, dets, motion, keyframes, app, fps)
    # drawn = {render frame: [(tid, (x1, y1, x2, y2), source)]}, source in
    #         "det" (detector box), "seed" (oracle box), "interp" (camera-compensated gap fill)

Inputs (frame indices are ORIGINAL video indices):
  render_frames  sorted rendered frames.
  dets           {frame: (xyxy (n, 4), conf (n,))} on the frames where the detector ran (any
                 subset of render_frames). Pass boxes down to conf ~0.05: low-confidence boxes
                 only continue existing tracks (two-stage association), they never start one.
  motion         {render frame: 3x3 homography previous render frame -> this one, None on a cut}.
  keyframes      optional {frame: [OraclePerson]}: oracle boxes no detection covers become
                 high-confidence "seed" detections; oracle team/role labels forbid joining two
                 tracklets of different teams.
  app            optional {frame: (n, 104) jersey-colour histograms aligned with dets[frame]}
                 (`appearance`), used as an association and stitching cost.

Algorithm (CPU, ~0.1-0.2 s per second of video):
  1. Online pass: each live track's box is warped by the camera motion (BoT-SORT-style GMC) plus
     a damped constant-velocity feet prediction. Cost = feet distance / gate (the gate grows with
     time lost) + expanded-IoU term + jersey-colour distance. Stage 1 gives confident boxes to
     all tracks; stage 2 gives low-confidence boxes to recently seen tracks only. Partial-body
     duplicates are suppressed before association and at track birth.
  2. Offline: tracklets are stitched (Hungarian over ends x starts, camera-compensated motion
     hypotheses, colour, size, oracle team), duplicate tracks riding on another player are
     dropped, and junk tracks (few hits, never confident, flickering) are pruned.
  3. Gaps between a track's observations are filled by camera-compensated interpolation, so
     filled boxes follow a pan instead of sliding across the image.

Measured on four 10 s lab clips (night, multi-sport, 4K broadcast, 4K screen recording; Luna
reference, detector every 2nd render frame): recall 0.83/0.40/0.72/0.65 -> 0.98/0.97/0.98/0.99
and drops per player-minute 8/22/22/18 -> 1.2/0/2.7/2.8 versus the ByteTrack path.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

_INF = 1e6
Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackCfg:
    # confidence stages
    conf_high: float = 0.30  # stage 1: may match any live track
    conf_low: float = 0.08  # stage 2 floor: only continues recently seen tracks
    conf_new: float = 0.15  # may start a tentative track
    # online association
    gmc: bool = True  # warp predictions by the camera motion
    motion: bool = True  # damped constant-velocity feet prediction
    gate0: float = 0.8  # feet-distance gate in box heights right after an observation
    gate_rate: float = 2.5  # + box heights per second lost
    gate_max: float = 4.0
    max_size_ratio: float = 1.8
    eiou: float = 0.3  # expanded-IoU margin (fraction of w/h on each side)
    w_iou: float = 0.5
    w_app: float = 1.0
    app_gate: float = 0.6  # Bhattacharyya distance above which a pair is never associated
    max_lost_s: float = 3.0  # track memory
    low_lost_s: float = 1.0  # stage 2 only for tracks seen this recently
    low_gate_frac: float = 0.6  # stage 2 uses a tighter distance gate
    # offline stitching
    stitch: bool = True
    stitch_gap_s: float = 4.0
    stitch_gate0: float = 0.8
    stitch_rate: float = 2.0
    stitch_gate_max: float = 5.0
    stitch_app_gate: float = 0.5
    stitch_max_cost: float = 1.6
    # pruning
    min_hits: int = 2
    min_maxconf: float = 0.30
    min_track_score: float = 0.15  # mean conf x fraction of detector runs that saw it; 0 = off
    prestitch_prune: bool = True  # junk tracklets never take part in stitching
    # gap filling
    fill: str = "comp"  # "comp" (camera-compensated) | "linear"
    fill_max_s: float = 5.0
    # oracle seeds
    keyframe_seed: bool = True
    seed_conf: float = 0.9
    # duplicates (YOLO often adds a partial-body box on the same player)
    pre_nms: bool = True  # drop a box mostly inside / overlapping a more confident box
    pre_nms_contain: float = 0.7
    pre_nms_iou: float = 0.5
    birth_suppress: bool = True  # no new track from a box overlapping a box already taken
    dup_iou: float = 0.35
    dup_contain: float = 0.6  # intersection / area of the smaller box
    dedup_tracks: bool = True  # drop a track that rides on another one most of its life
    dedup_frac: float = 0.6

    def __post_init__(self):
        if not 0 <= self.conf_low <= min(self.conf_new, self.conf_high):
            raise ValueError("confidence stages must satisfy conf_low <= conf_new, conf_high")
        if self.fill not in ("comp", "linear"):
            raise ValueError("fill must be 'comp' or 'linear'")


@dataclass
class PlayerTrack:
    """One player's observations. `det_idx[frame]` is the index into the caller's dets[frame]
    (-1 for an oracle seed box)."""

    tid: int
    obs: List[Tuple[int, Box, float, str]]  # (frame, box, conf, source)
    det_idx: Dict[int, int]
    label: Optional[str]  # oracle team name / "referee", or None
    hists: List[np.ndarray]


# ---- geometry ----------------------------------------------------------------------------------


def _warp_pts(M: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, float).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    w = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    return np.stack(
        [(M[0, 0] * x + M[0, 1] * y + M[0, 2]) / w, (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / w],
        axis=1,
    )


def warp_box(M: np.ndarray, b) -> np.ndarray:
    """Move the feet point by M and scale the box by M's local scale there (aspect kept)."""
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    fx, fy = (x1 + x2) / 2.0, y2
    p = _warp_pts(M, [[fx, fy], [fx, fy - h]])
    nh = float(np.hypot(*(p[0] - p[1])))
    s = nh / h if h > 0 else 1.0
    nw = w * s
    return np.array([p[0, 0] - nw / 2, p[0, 1] - nh, p[0, 0] + nw / 2, p[0, 1]])


def _feet(b) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, b[3]])


def _height(b) -> float:
    return max(1.0, b[3] - b[1])


def _eiou(a, b, e: float) -> float:
    """IoU of the two boxes each grown by `e` of its width/height on every side."""
    wa, ha, wb, hb = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    A = (a[0] - e * wa, a[1] - e * ha, a[2] + e * wa, a[3] + e * ha)
    B = (b[0] - e * wb, b[1] - e * hb, b[2] + e * wb, b[3] + e * hb)
    x1, y1, x2, y2 = max(A[0], B[0]), max(A[1], B[1]), min(A[2], B[2]), min(A[3], B[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (A[2] - A[0]) * (A[3] - A[1]) + (B[2] - B[0]) * (B[3] - B[1]) - inter
    return inter / u if u > 0 else 0.0


def _iou(a, b) -> float:
    return _eiou(a, b, 0.0)


def _dup(a, b, cfg: TrackCfg) -> bool:
    """Two boxes on the same person: high IoU or one mostly inside the other."""
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return False
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return (
        inter / (aa + ab - inter) >= cfg.dup_iou
        or inter / max(1e-6, min(aa, ab)) >= cfg.dup_contain
    )


def containment_nms(X: np.ndarray, C: np.ndarray, contain: float, iou_thr: float) -> np.ndarray:
    """Keep-mask dropping boxes that lie mostly inside (or overlap heavily) a more confident box
    of the same frame: YOLO's upper-body / legs duplicates that would start a second track."""
    keep = np.ones(len(X), bool)
    if len(X) < 2:
        return keep
    area = (X[:, 2] - X[:, 0]) * (X[:, 3] - X[:, 1])
    iw = np.clip(
        np.minimum(X[:, None, 2], X[None, :, 2]) - np.maximum(X[:, None, 0], X[None, :, 0]), 0, None
    )
    ih = np.clip(
        np.minimum(X[:, None, 3], X[None, :, 3]) - np.maximum(X[:, None, 1], X[None, :, 1]), 0, None
    )
    inter = iw * ih
    with np.errstate(divide="ignore", invalid="ignore"):
        small = np.maximum(1e-6, np.minimum(area[:, None], area[None]))
        sup = (inter > 0) & (
            (inter / small >= contain) | (inter / (area[:, None] + area[None] - inter) >= iou_thr)
        )
    kept: List[int] = []
    for j in np.argsort(-C):
        if kept and sup[j, kept].any():
            keep[j] = False
        else:
            kept.append(int(j))
    return keep


# ---- appearance ----------------------------------------------------------------------------------


def _part_hist(hsv: np.ndarray) -> np.ndarray:
    """16 hue x 3 saturation bins for chromatic pixels + 4 value bins for achromatic ones."""
    h, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    out = np.zeros(52, np.float32)
    if h.size == 0:
        return out
    chrom = (s >= 50) & (v >= 40)
    hb = np.minimum(h[chrom].astype(int) * 16 // 180, 15)
    sb = np.minimum((s[chrom].astype(int) - 50) * 3 // 206, 2)
    np.add.at(out, hb * 3 + sb, 1.0)
    vb = np.minimum(v[~chrom].astype(int) * 4 // 256, 3)
    np.add.at(out, 48 + vb, 1.0)
    return out / max(1.0, out.sum())


def appearance(frame: np.ndarray, box) -> np.ndarray:
    """(104,) jersey (upper body) + shorts (lower body) colour histograms of one person box."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = frame.shape[:2]
    halves = []
    for fy0, fy1 in ((0.15, 0.5), (0.5, 0.8)):
        a, b = int(max(0, y1 + fy0 * h)), int(min(H, y1 + fy1 * h))
        c, d = int(max(0, x1 + 0.2 * w)), int(min(W, x2 - 0.2 * w))
        if b - a < 2 or d - c < 2:
            halves.append(np.zeros(52, np.float32))
        else:
            halves.append(_part_hist(cv2.cvtColor(frame[a:b, c:d], cv2.COLOR_BGR2HSV)))
    return np.r_[halves[0], halves[1]]


def appearance_features(frame: np.ndarray, xyxy, conf, min_conf: float = 0.08) -> np.ndarray:
    """(n, 104) histograms for one frame's detections (zeros below `min_conf`)."""
    out = np.zeros((len(xyxy), 104), np.float32)
    for i, (b, c) in enumerate(zip(xyxy, conf)):
        if c >= min_conf:
            out[i] = appearance(frame, b)
    return out


def _bhatt(p, q) -> Optional[float]:
    """Mean Bhattacharyya distance over the two body halves; None if either is empty."""
    if p is None or q is None:
        return None
    P = np.asarray(p, float).reshape(-1, 52)
    Q = np.asarray(q, float).reshape(-1, 52)
    sp, sq = P.sum(1), Q.sum(1)
    ok = (sp > 0) & (sq > 0)
    if not ok.any():
        return None
    bc = np.sqrt(P[ok] * Q[ok]).sum(1) / np.sqrt(sp[ok] * sq[ok])
    return float(np.mean(np.sqrt(np.maximum(0.0, 1.0 - bc))))


# ---- camera motion -------------------------------------------------------------------------------


class CameraChain:
    """Cumulative camera motion over the render frames: chain(a, b) maps pixels of frame a to
    frame b. A cut (motion None) starts a new segment; chains across segments are None."""

    def __init__(
        self, render_frames: Sequence[int], motion: Optional[Dict[int, Optional[np.ndarray]]]
    ):
        self.cum: Dict[int, np.ndarray] = {}
        self.seg: Dict[int, int] = {}
        M, s = np.eye(3), 0
        for i, f in enumerate(render_frames):
            g = motion.get(f) if motion is not None else np.eye(3)
            if i > 0:
                if g is None:
                    M, s = np.eye(3), s + 1
                else:
                    M = g @ M
                    M = M / M[2, 2]
            self.cum[f], self.seg[f] = M.copy(), s
        self.inv = {f: np.linalg.inv(M) for f, M in self.cum.items()}

    def __call__(self, a: int, b: int) -> Optional[np.ndarray]:
        if self.seg[a] != self.seg[b]:
            return None
        M = self.cum[b] @ self.inv[a]
        return M / M[2, 2]


def _fill_gap(rf, pos, chain: Optional[CameraChain], f1, b1, f2, b2) -> Dict[int, np.ndarray]:
    """Boxes on the render frames strictly between f1 and f2. With a chain, both end boxes are
    warped into each frame and blended (boxes follow the pan); without, plain interpolation."""
    out = {}
    a0, b0 = np.asarray(b1, float), np.asarray(b2, float)
    for f in rf[pos[f1] + 1 : pos[f2]]:
        t = (f - f1) / float(f2 - f1)
        a, b = a0, b0
        if chain is not None:
            Ma, Mb = chain(f1, f), chain(f2, f)
            if Ma is None or Mb is None:
                continue  # A cut or failed camera-motion estimate is not evidence for a fill.
            a = warp_box(Ma, a0)
            b = warp_box(Mb, b0)
        out[f] = (1 - t) * a + t * b
    return out


# ---- tracker -------------------------------------------------------------------------------------


class _Live:
    """Online state of one track (`ref` = last observed box warped into the current frame)."""

    def __init__(self, track: PlayerTrack, box, hist):
        self.t = track
        self.ref = np.asarray(box, float)
        self.v = np.zeros(2)  # feet velocity, px per render step (current-frame coordinates)
        self.n = 0  # render steps since the last observation
        self.last = track.obs[0][0]
        self.hist = None if hist is None else np.asarray(hist, float).copy()

    def pred(self, motion: bool, max_steps: int) -> np.ndarray:
        b = self.ref.copy()
        if motion and self.n:
            d = self.v * min(self.n, max_steps)
            b[[0, 2]] += d[0]
            b[[1, 3]] += d[1]
        return b


def _team_vocab(keyframes) -> set:
    """The two most frequent oracle player-team names are the teams; other names are unknown
    (the oracle names one kit "black" / "dark" / "dark green" on different frames)."""
    c = Counter(p.team for ps in keyframes.values() for p in ps if p.role == "player" and p.team)
    return {n for n, _ in c.most_common(2)}


def _oracle_label(p, vocab) -> Optional[str]:
    if p.role == "referee":
        return "referee"
    if p.role == "player" and p.team in vocab:
        return p.team
    return None


def _assign(C: np.ndarray) -> List[Tuple[int, int]]:
    if C.size == 0:
        return []
    r, c = linear_sum_assignment(C)
    return [(i, j) for i, j in zip(r, c) if C[i, j] < _INF / 2]


def track_people(
    render_frames: Sequence[int],
    dets: Dict[int, Tuple[np.ndarray, np.ndarray]],
    motion: Optional[Dict[int, Optional[np.ndarray]]],
    keyframes: Optional[dict] = None,
    app: Optional[Dict[int, np.ndarray]] = None,
    fps: float = 30.0,
    cfg: Optional[TrackCfg] = None,
) -> Tuple[Dict[int, List[Tuple[int, Box, str]]], List[PlayerTrack]]:
    """Track people over a window; see the module docstring. Returns (drawn, final tracks)."""
    cfg = cfg or TrackCfg()
    rf = list(render_frames)
    if not rf:
        return {}, []
    rstep = (rf[1] - rf[0]) if len(rf) > 1 else 1
    steps_per_s = fps / rstep
    motion = motion if cfg.gmc else None
    chain = CameraChain(rf, motion)
    keyframes = keyframes or {}
    vocab = _team_vocab(keyframes)
    tracks: List[PlayerTrack] = []
    live: List[_Live] = []
    max_pred_steps = int(round(steps_per_s))

    for k, f in enumerate(rf):
        # 1) carry every live track through the camera motion
        if k > 0:
            g = motion.get(f) if motion is not None else np.eye(3)
            for t in live:
                t.n += 1
                if g is None:  # cut: keep boxes, the gate opens with time lost
                    continue
                fo = _feet(t.ref)
                t.ref = warp_box(g, t.ref)
                if np.any(t.v):
                    t.v = _warp_pts(g, [fo + t.v])[0] - _warp_pts(g, [fo])[0]
        if f not in dets and not (cfg.keyframe_seed and keyframes.get(f)):
            continue
        # 2) this frame's candidate boxes (an oracle keyframe without a detector run still
        # contributes its people as seeds)
        X, C = dets.get(f, (np.zeros((0, 4)), np.zeros(0)))
        X = np.asarray(X, float).reshape(-1, 4)
        C = np.asarray(C, float).ravel()
        A = app.get(f) if app is not None else None
        keep = C >= cfg.conf_low
        if cfg.pre_nms and keep.any():
            keep &= containment_nms(X, C, cfg.pre_nms_contain, cfg.pre_nms_iou)
        X, C = X[keep], C[keep].copy()
        A = A[keep] if A is not None else None
        didx = [int(i) for i in np.nonzero(keep)[0]]
        hists = [None if A is None else A[i] for i in range(len(X))]
        srcs = ["det"] * len(X)
        labels: List[Optional[str]] = [None] * len(X)
        if f in keyframes and cfg.keyframe_seed and len(keyframes[f]):
            # oracle keyframe: confirm matching boxes, add the people nothing covers as seeds
            P = keyframes[f]
            M = (
                np.array([[_iou(p.box, x) for x in X] for p in P])
                if len(X)
                else np.zeros((len(P), 0))
            )
            matched = set()
            for i, j in _assign(np.where(M >= 0.4, 1 - M, _INF)) if M.size else []:
                matched.add(i)
                C[j] = max(C[j], cfg.seed_conf)
                labels[j] = _oracle_label(P[i], vocab)
            extra = [i for i in range(len(P)) if i not in matched]
            if extra:
                E = np.array([P[i].box for i in extra], float).reshape(-1, 4)
                X = np.vstack([X, E])
                C = np.r_[C, np.full(len(extra), cfg.seed_conf)]
                hists += [None] * len(extra)
                srcs += ["seed"] * len(extra)
                didx += [-1] * len(extra)
                labels += [_oracle_label(P[i], vocab) for i in extra]
        ndet = len(X)
        used = np.zeros(ndet, bool)
        live[:] = [t for t in live if (f - t.last) / fps <= cfg.max_lost_s]

        def cost_matrix(trks: List[_Live], idx: List[int], gate_scale: float) -> np.ndarray:
            Cm = np.full((len(trks), len(idx)), _INF)
            if not trks or not idx:
                return Cm
            PB = np.array([t.pred(cfg.motion, max_pred_steps) for t in trks])
            XB = X[idx]
            PH = np.maximum(1.0, PB[:, 3] - PB[:, 1])
            XH = np.maximum(1.0, XB[:, 3] - XB[:, 1])
            PF = np.c_[(PB[:, 0] + PB[:, 2]) / 2.0, PB[:, 3]]
            XF = np.c_[(XB[:, 0] + XB[:, 2]) / 2.0, XB[:, 3]]
            DD = np.hypot(XF[None, :, 0] - PF[:, None, 0], XF[None, :, 1] - PF[:, None, 1])
            DD = DD / (0.5 * (PH[:, None] + XH[None]))
            lost = np.array([(f - t.last) / fps for t in trks])
            gate = np.minimum(cfg.gate_max, cfg.gate0 + cfg.gate_rate * lost) * gate_scale
            ok = (DD <= gate[:, None]) & (
                np.abs(np.log(XH[None] / PH[:, None])) <= math.log(cfg.max_size_ratio)
            )
            for a, b in zip(*np.nonzero(ok)):
                t, j = trks[a], idx[b]
                if labels[j] is not None and t.t.label is not None and labels[j] != t.t.label:
                    continue
                a_d = _bhatt(t.hist, hists[j])
                if a_d is not None and a_d > cfg.app_gate:
                    continue
                c = DD[a, b] / gate[a] + cfg.w_iou * (1 - _eiou(PB[a], X[j], cfg.eiou))
                Cm[a, b] = c + (cfg.w_app * a_d if a_d is not None else 0.0)
            return Cm

        def update(t: _Live, j: int):
            x = X[j]
            if t.n > 0:
                vm = (_feet(x) - _feet(t.ref)) / t.n
                hmax = 0.5 * _height(x)
                nv = np.linalg.norm(vm)
                if nv > hmax:
                    vm = vm * hmax / nv
                t.v = vm if len(t.t.obs) == 1 else 0.6 * t.v + 0.4 * vm
            t.ref = x.copy()
            t.n = 0
            t.last = f
            t.t.obs.append((f, tuple(float(v) for v in x), float(C[j]), srcs[j]))
            t.t.det_idx[f] = didx[j]
            if hists[j] is not None and C[j] >= 0.25:
                t.t.hists.append(hists[j])
                t.hist = hists[j].copy() if t.hist is None else 0.8 * t.hist + 0.2 * hists[j]
            if labels[j] is not None and t.t.label is None:
                t.t.label = labels[j]

        # 3) two-stage association, then births
        hi = [j for j in range(ndet) if C[j] >= cfg.conf_high]
        lo = [j for j in range(ndet) if cfg.conf_low <= C[j] < cfg.conf_high]
        trks = list(live)
        matched_t = set()
        for a, b in _assign(cost_matrix(trks, hi, 1.0)):
            update(trks[a], hi[b])
            used[hi[b]] = True
            matched_t.add(a)
        rest = [
            t
            for a, t in enumerate(trks)
            if a not in matched_t and (f - t.last) / fps <= cfg.low_lost_s
        ]
        if rest and lo:
            for a, b in _assign(cost_matrix(rest, lo, cfg.low_gate_frac)):
                update(rest[a], lo[b])
                used[lo[b]] = True
        taken = [j for j in range(ndet) if used[j]]
        for j in sorted(range(ndet), key=lambda j: -C[j]):
            if used[j] or C[j] < cfg.conf_new:
                continue
            if (
                cfg.birth_suppress
                and srcs[j] != "seed"
                and any(_dup(X[j], X[q], cfg) for q in taken)
            ):
                continue
            box = tuple(float(v) for v in X[j])
            tr = PlayerTrack(
                len(tracks), [(f, box, float(C[j]), srcs[j])], {f: didx[j]}, labels[j], []
            )
            if hists[j] is not None:
                tr.hists.append(np.asarray(hists[j], float))
            tracks.append(tr)
            live.append(_Live(tr, box, hists[j]))
            taken.append(j)

    # ---- offline: prune junk, stitch tracklets, drop duplicates, prune again
    tl = [t for t in tracks if t.obs]
    if cfg.prestitch_prune:
        tl = [
            t
            for t in tl
            if any(o[3] == "seed" for o in t.obs)
            or (
                max(o[2] for o in t.obs) >= cfg.min_maxconf
                and (len(t.obs) >= 2 or t.obs[0][2] >= 0.5)
            )
        ]
    if cfg.stitch and len(tl) > 1:
        tl = _stitch(tl, chain, cfg, fps, rstep)
    if cfg.dedup_tracks:
        tl = _dedup(tl, cfg)
    dpos = {f: i for i, f in enumerate(sorted(set(dets) | (set(keyframes) & set(rf))))}
    final = []
    for t in tl:
        confs = [o[2] for o in t.obs]
        seeded = any(o[3] == "seed" for o in t.obs)
        if len(t.obs) < cfg.min_hits and not seeded:
            continue
        if max(confs) < cfg.min_maxconf:
            continue
        fs = sorted({o[0] for o in t.obs})
        span = dpos[fs[-1]] - dpos[fs[0]] + 1
        # track quality = mean confidence x fraction of detector runs that saw it: junk on line
        # markings / shadows flickers at low confidence; real players (even far, at night) don't
        if not seeded and float(np.mean(confs)) * len(fs) / span < cfg.min_track_score:
            continue
        t.obs.sort(key=lambda o: o[0])
        final.append(t)

    # ---- draw, filling the gaps between observations
    pos = {f: i for i, f in enumerate(rf)}
    fill_chain = chain if cfg.fill == "comp" else None
    out: Dict[int, List[Tuple[int, Box, str]]] = {f: [] for f in rf}
    for t in final:
        for (f1, b1, _, _), (f2, b2, _, _) in zip(t.obs, t.obs[1:]):
            if (f2 - f1) / fps > cfg.fill_max_s or pos[f2] - pos[f1] <= 1:
                continue
            for fm, bm in _fill_gap(rf, pos, fill_chain, f1, b1, f2, b2).items():
                out[fm].append((t.tid, tuple(float(v) for v in bm), "interp"))
        for f1, b1, _, s1 in t.obs:
            out[f1].append((t.tid, b1, s1))
    # Window edges: render frames before the first / after the last detector run (the detector
    # runs on every Nth render frame) have no observation on one side to fill towards. Tracks
    # seen at that edge run are carried there through the camera motion, as inside a gap.
    runs = sorted(f for f in dpos if f in pos)
    if runs:
        for edge, extra in (
            (runs[0], rf[: pos[runs[0]]]),
            (runs[-1], rf[pos[runs[-1]] + 1 :]),
        ):
            if not extra or max(abs(extra[0] - edge), abs(extra[-1] - edge)) / fps > cfg.fill_max_s:
                continue
            for t in final:
                b = next((o[1] for o in t.obs if o[0] == edge), None)
                if b is None:
                    continue
                for fm in extra:
                    M = fill_chain(edge, fm) if fill_chain is not None else None
                    if fill_chain is not None and M is None:
                        continue
                    bm = warp_box(M, b) if M is not None else np.asarray(b, float)
                    out[fm].append((t.tid, tuple(float(v) for v in bm), "interp"))
    return out, final


def _dedup(tl: List[PlayerTrack], cfg: TrackCfg) -> List[PlayerTrack]:
    """Remove tracks sitting on another track's player for most of their life (partial-body
    duplicates). The weaker track (fewer observations, then lower mean confidence) goes."""
    boxes = [{o[0]: o[1] for o in t.obs} for t in tl]
    strength = [(len(t.obs), float(np.mean([o[2] for o in t.obs]))) for t in tl]
    kept: List[int] = []
    for i in sorted(range(len(tl)), key=lambda i: strength[i], reverse=True):
        bi = boxes[i]
        dup = False
        for k in kept:
            common = [f for f in bi if f in boxes[k]]
            if common and sum(
                _dup(bi[f], boxes[k][f], cfg) for f in common
            ) >= cfg.dedup_frac * len(bi):
                dup = True
                break
        if not dup:
            kept.append(i)
    return [tl[i] for i in sorted(kept)]


def _stitch(tl: List[PlayerTrack], chain: CameraChain, cfg: TrackCfg, fps: float, rstep: int):
    """Join tracklets: Hungarian over tracklet ends x tracklet starts (one round forms chains)."""

    def end_state(t: PlayerTrack, at_end: bool):
        obs = sorted(t.obs, key=lambda o: o[0])
        o = obs[-1] if at_end else obs[0]
        v = np.zeros(2)  # velocity from the two outermost observations, camera-compensated
        if len(obs) >= 2:
            o2 = obs[-2] if at_end else obs[1]
            M = chain(o2[0], o[0])
            if M is not None:
                v = (_feet(o[1]) - _feet(warp_box(M, o2[1]))) / ((o[0] - o2[0]) / rstep)
                if not at_end:
                    v = -v
        hist = np.mean(t.hists, axis=0) if t.hists else None
        return o[0], np.asarray(o[1], float), v, hist

    ends = [end_state(t, True) for t in tl]
    starts = [end_state(t, False) for t in tl]
    n = len(tl)
    Cm = np.full((n, n), _INF)
    for i in range(n):
        fe, be, ve, he = ends[i]
        for j in range(n):
            fs, bs, vs, hs = starts[j]
            if i == j or fs <= fe or (fs - fe) / fps > cfg.stitch_gap_s:
                continue
            li, lj = tl[i].label, tl[j].label
            if li is not None and lj is not None and li != lj:
                continue
            M = chain(fe, fs)
            if M is None:
                continue
            we = warp_box(M, be)
            if abs(math.log(_height(bs) / _height(we))) > math.log(cfg.max_size_ratio):
                continue
            steps = (fs - fe) / rstep
            dt = (fs - fe) / fps
            h = 0.5 * (_height(we) + _height(bs))
            fwd = _feet(we) + ve * min(steps, fps / rstep)
            bwd = _feet(bs) + vs * min(steps, fps / rstep)  # vs points backwards in time
            cands = [np.hypot(*(_feet(we) - _feet(bs))), np.hypot(*(fwd - _feet(bs)))]
            Minv = chain(fs, fe)
            if Minv is not None:
                back = np.hypot(*(_warp_pts(Minv, [bwd])[0] - _feet(be)))
                cands.append(back * _height(we) / _height(be))
            d = float(min(cands)) / h
            gate = min(cfg.stitch_gate_max, cfg.stitch_gate0 + cfg.stitch_rate * dt)
            if d > gate:
                continue
            a = _bhatt(he, hs)
            if a is not None and a > cfg.stitch_app_gate:
                continue
            c = d / gate + 0.2 * dt + (cfg.w_app * a if a is not None else 0.3)
            if c <= cfg.stitch_max_cost:
                Cm[i, j] = c
    nxt = dict(_assign(Cm))
    has_prev = set(nxt.values())
    merged = []
    for i in range(n):
        if i in has_prev:
            continue
        t, k, seen = tl[i], i, {i}
        while k in nxt and nxt[k] not in seen:
            k = nxt[k]
            seen.add(k)
            o = tl[k]
            t.obs += o.obs
            t.hists += o.hists
            t.det_idx.update(o.det_idx)
            if t.label is None:
                t.label = o.label
        merged.append(t)
    return merged
