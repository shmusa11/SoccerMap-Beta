"""Who is drawn: the on-field rule for tracked people.

A person is shown when their track steps onto the pitch at some moment within +-`window_s` of
the frame; a track that never steps on (spectators, fence-line bystanders, bench staff) is hidden.
So a throw-in or corner taker standing just outside the touchline stays visible, also on the 2D
map beside the line, while bystanders who never enter are deselected.

    vis = onfield_visibility(md.people, calib, fps, (L, W), OnFieldCfg(), kind=kind,
                             track_prob=ai_probs, legacy_offfield=oracle_offfield)
    vis.visible[f]  -> track ids drawn on frame f (video boxes and 2D map)
    vis.pitch_xy[f] -> {tid: (x, y)} pitch metres of the visible tracks on calibrated frames,
                       including off-pitch positions
    vis.status[tid] -> "on-pitch" | "relinked" | "protected" | "bystander" | "ai-keep" |
                       "ai-hide" | "uncalibrated"

Rule (frame indices are ORIGINAL video indices; only verified calibrations are used):
  1. Feet (box bottom-centre) -> pitch metres with the frame's homography; d_out = distance
     outside the pitch rectangle (0 inside).
  2. Inside when median_{+-smooth_s}(d_out) <= min(m0 + e_px * metres-per-pixel at the feet,
     m_max): far-touchline feet at a grazing angle get a wider margin than near-side feet, and a
     one-frame box glitch neither enters nor leaves the pitch.
  3. A calibrated frame of a track is shown iff the track is inside at some frame within
     +-window_s. Uncalibrated frames copy the track's nearest calibrated observation.
  4. Touchline identity breaks: a never-inside track is re-linked to an inside track that starts
     within relink_gap_s after it ends (or ends before it starts), close in pitch position, and
     only when the never-inside track ends/starts within relink_max_out_m of the lines.
  5. Never-inside tracks are hidden, except oracle roles in protect_kinds (assistant referees
     run outside the line all match) and tracks the AI calls a participant with p >= ai_keep.
  6. A track the oracle keyframe vote put off-field whose feet never get deeper than edge_m
     inside the lines (90th percentile) is hidden too (touchline bench rows / spectators).
  Tracks with no calibrated observation are hidden iff the oracle keyframe vote put them
  off-field, or the AI's mean p(participant) < ai_hide_thr and no oracle keyframe labelled them
  (the AI never overrules an oracle label).

Lab (round 3): night clip off-field phantoms 1.00 -> 0.05 per audited frame with
no recall loss and the throw-in taker kept on all 42 throw-in frames; on uncalibrated clips the Decisions
API answers (`ask_participants`) cut phantoms with no recall loss. `window_s` could not be tuned
from the lab clips (same results 0.5-10 s); 6 s is an assumption.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class OnFieldCfg:
    window_s: float = 6.0  # off-pitch positions shown within +-this of an on-pitch moment
    m0: float = 0.5  # base margin (m) beyond the outer lines
    e_px: float = 6.0  # feet / calibration uncertainty in image pixels -> extra margin
    m_max: float = 2.0  # margin cap (m); fence-line bystanders stood >= 2.6 m out in the lab
    smooth_s: float = 0.25  # median filter half-width on d_out
    relink: bool = True
    relink_gap_s: float = 2.0
    relink_r0: float = 2.5
    relink_speed: float = 7.0  # m/s
    relink_max_out_m: float = 3.0  # the never-inside track must end/start this close to the lines
    protect_kinds: Tuple[str, ...] = ("referee",)
    oracle_wins: bool = False  # True = any oracle label protects (the round-2 behaviour)
    ai_keep: float = 0.8  # never-inside calibrated track kept when AI p(participant) >= this
    ai_hide_thr: float = 0.15  # uncalibrated, unlabelled track hidden when AI p < this
    # A track the oracle keyframe vote put off-field that only ever stands within this many
    # metres inside a line (90th percentile) stays a bystander: bench rows and spectators on the
    # touchline of a football-lined venue, where the soccer width is an assumption (2026-10-14:
    # 12 extra phantoms per frame on the screen recording once it was calibrated).
    edge_m: float = 3.0


@dataclass
class OnFieldResult:
    visible: Dict[int, Set[int]]
    pitch_xy: Dict[int, Dict[int, Tuple[float, float]]]
    status: Dict[int, str]
    links: Dict[int, int] = field(default_factory=dict)  # never-inside tid -> lender tid

    @property
    def hidden(self) -> Set[int]:
        return {t for t, s in self.status.items() if s in ("bystander", "ai-hide")}


def _feet(b) -> Tuple[float, float]:
    return (float(b[0]) + float(b[2])) / 2.0, float(b[3])


def _d_out_and_mpp(H: np.ndarray, pts: np.ndarray, L: float, W: float):
    """Pitch position, distance outside the pitch rectangle (m) and metres per vertical image
    pixel at each feet point (the dominant direction of box-bottom and far-side error)."""
    a = cv2.perspectiveTransform(pts[None].astype(np.float32), H)[0]
    b = cv2.perspectiveTransform((pts + [0.0, 1.0])[None].astype(np.float32), H)[0]
    dx = np.maximum.reduce([-a[:, 0], a[:, 0] - L, np.zeros(len(a))])
    dy = np.maximum.reduce([-a[:, 1], a[:, 1] - W, np.zeros(len(a))])
    return a, np.hypot(dx, dy), np.linalg.norm(b - a, axis=1)


def onfield_visibility(
    people: Mapping[int, Iterable],
    calib: Mapping[int, Optional[np.ndarray]],
    fps: float,
    pitch_dims: Tuple[float, float],
    cfg: OnFieldCfg = OnFieldCfg(),
    kind: Optional[Mapping[int, str]] = None,
    track_prob: Optional[Mapping[int, float]] = None,
    legacy_offfield: Optional[Set[int]] = None,
) -> OnFieldResult:
    """people: {frame: [(tid, box, ...), ...]} (box = x1, y1, x2, y2 pixels; tid < 0 ignored);
    calib: {frame: image->pitch H or None} (verified only; unavailable = None);
    kind: oracle role per tid; track_prob: {tid: AI P(on-field participant)};
    legacy_offfield: tids the oracle keyframe vote put off-field (used for tracks without any
    calibrated frame)."""
    L, W = pitch_dims
    kind = kind or {}
    track_prob = track_prob or {}
    legacy_offfield = legacy_offfield or set()
    frames = sorted(people)
    # per-track observations: (frame, pitch xy or None, d_out or None, margin or None)
    obs: Dict[int, list] = {}
    for f in frames:
        ps = [p for p in people[f] if p[0] >= 0]
        if not ps:
            continue
        H = calib.get(f)
        if H is None:
            for p in ps:
                obs.setdefault(p[0], []).append((f, None, None, None))
            continue
        pts = np.array([_feet(p[1]) for p in ps], float)
        xy, d, mpp = _d_out_and_mpp(np.asarray(H, float), pts, L, W)
        marg = np.minimum(cfg.m0 + cfg.e_px * mpp, cfg.m_max)
        for p, q, dd, mm in zip(ps, xy, d, marg):
            obs.setdefault(p[0], []).append((f, (float(q[0]), float(q[1])), float(dd), float(mm)))
    half = cfg.smooth_s * fps
    inside: Dict[int, np.ndarray] = {}  # tid -> frames where the track is on the pitch
    cal_frames: Dict[int, np.ndarray] = {}
    depth: Dict[int, float] = {}  # tid -> 90th percentile of how far inside the lines (m)
    for t, ob in obs.items():
        q = np.array([x[1] for x in ob if x[1] is not None], float).reshape(-1, 2)
        if len(q):
            dep = np.minimum.reduce([q[:, 0], L - q[:, 0], q[:, 1], W - q[:, 1]])
            depth[t] = float(np.percentile(dep, 90))
    for t, ob in obs.items():
        c = [(f, d, m) for f, _, d, m in ob if d is not None]
        if not c:
            continue
        fr = np.array([x[0] for x in c], float)
        d = np.array([x[1] for x in c])
        m = np.array([x[2] for x in c])
        ds = np.array([np.median(d[np.abs(fr - f) <= half]) for f in fr])
        cal_frames[t] = fr
        inside[t] = fr[ds <= m]
    links = _relink(obs, inside, fps, cfg) if cfg.relink else {}
    # ---- per-track decision ---------------------------------------------------------------
    status: Dict[int, str] = {}
    win = cfg.window_s * fps
    show_cal: Dict[int, Dict[int, bool]] = {}
    for t, ob in obs.items():
        k = kind.get(t)
        protected = (k in cfg.protect_kinds) or (cfg.oracle_wins and k is not None)
        p_ai = track_prob.get(t)
        if t not in cal_frames:  # no calibrated observation: oracle vote, then the AI
            if t in legacy_offfield and not protected:
                status[t] = "bystander"
            elif p_ai is not None and k is None and p_ai < cfg.ai_hide_thr:
                status[t] = "ai-hide"
            else:
                status[t] = "uncalibrated"
            continue
        ref = inside[t]
        if t in legacy_offfield and not protected and depth.get(t, 0.0) < cfg.edge_m:
            ref = ref[:0]  # oracle: not a participant, and only at the edge of the pitch
        if not len(ref) and t in links:
            ref = inside[links[t]]
            status[t] = "relinked"
        elif len(ref):
            status[t] = "on-pitch"
        if not len(ref):
            if protected:
                status[t] = "protected"
            elif p_ai is not None and p_ai >= cfg.ai_keep:
                status[t] = "ai-keep"
            else:
                status[t] = "bystander"
            show_cal[t] = {int(f): status[t] != "bystander" for f in cal_frames[t]}
            continue
        ref = np.asarray(ref)
        show_cal[t] = {
            int(f): bool(protected or np.min(np.abs(ref - f)) <= win) for f in cal_frames[t]
        }
    # ---- per-frame output -----------------------------------------------------------------
    visible: Dict[int, Set[int]] = {f: set() for f in frames}
    pitch_xy: Dict[int, Dict[int, Tuple[float, float]]] = {f: {} for f in frames}
    for t, ob in obs.items():
        sc = show_cal.get(t)
        cf = np.array(sorted(sc)) if sc else None
        for f, q, _, _ in ob:
            if sc is None:
                vis = status[t] in ("ai-keep", "uncalibrated", "protected")
            elif f in sc:
                vis = sc[f]
            else:  # uncalibrated frame of a calibrated track: nearest calibrated observation
                vis = sc[int(cf[np.argmin(np.abs(cf - f))])]
            if vis:
                visible[f].add(t)
                if q is not None:
                    pitch_xy[f][t] = q
    return OnFieldResult(visible, pitch_xy, status, links)


def _relink(obs, inside, fps: float, cfg: OnFieldCfg) -> Dict[int, int]:
    """Never-inside track -> the adjacent inside track it most likely continues (a throw-in
    taker often gets a new id when stepping back in)."""
    span = {t: (ob[0][0], ob[-1][0]) for t, ob in obs.items()}

    def end_xy(t, at_end):
        for f, q, d, _ in reversed(obs[t]) if at_end else obs[t]:
            if q is not None:
                return np.array(q), d
        return None, None

    links: Dict[int, int] = {}
    lenders = [u for u in inside if len(inside[u])]
    for t in inside:
        if len(inside[t]):
            continue
        best = None
        for u in lenders:
            if u == t:
                continue
            for t_first in (False, True):  # t then u, or u then t
                a, b = (t, u) if t_first else (u, t)
                gap = (span[b][0] - span[a][1]) / fps
                if gap < -0.15 or gap > cfg.relink_gap_s:  # <= ~2 render frames overlap
                    continue
                pa, da = end_xy(a, True)
                pb, db = end_xy(b, False)
                if pa is None or pb is None:
                    continue
                if (da if t_first else db) > cfg.relink_max_out_m:  # t's end at the link
                    continue
                dist = float(np.linalg.norm(pa - pb))
                if dist <= cfg.relink_r0 + cfg.relink_speed * max(gap, 0.0):
                    if best is None or dist < best[0]:
                        best = (dist, u)
        if best is not None:
            links[t] = best[1]
    return links


# ---- AI evidence for tracks without calibration (OpenAI Decisions API) ----------------------

TILE, COLS, ROWS = 256, 4, 3
PARTICIPANT_TEXT = (
    "Frames from a soccer match video. Each numbered tile is a crop around ONE detected person, "
    "outlined by a YELLOW box. Judge only the person in the yellow box of that tile."
)


def person_tile(img: np.ndarray, box, n: int, min_side: float = 160.0) -> np.ndarray:
    """TILE x TILE context crop around a person (feet and surroundings), the person outlined in
    yellow and the tile numbered `n` in the top-left corner."""
    x1, y1, x2, y2 = box
    h = max(8.0, y2 - y1)
    S = max(5.0 * h, min_side)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + 0.5 * h
    a = np.array([cx - S / 2, cy - S / 2])
    k = TILE / S
    M = np.array([[k, 0, -a[0] * k], [0, k, -a[1] * k]])
    t = cv2.warpAffine(img, M, (TILE, TILE), flags=cv2.INTER_AREA, borderValue=(0, 0, 0))

    def p(x, y):
        return int(round((x - a[0]) * k)), int(round((y - a[1]) * k))

    cv2.rectangle(t, p(x1, y1), p(x2, y2), (0, 255, 255), 2)
    cv2.rectangle(t, (0, 0), (34, 24), (0, 0, 0), -1)
    cv2.putText(t, str(n), (3, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return t


def participant_questions(nums: Iterable[int]) -> List[dict]:
    """Two Decisions predicates per tile; only `participant_<i>` is used (the `inside_<i>`
    question is kept so the request matches the lab-measured one; football lines confuse it)."""
    q = []
    for i in nums:
        q.append(
            dict(
                type="predicate",
                name=f"participant_{i}",
                instructions=(
                    f"Tile {i}: the person in the yellow box is taking part in the match on the "
                    "field right now: an outfield player, a goalkeeper, a referee, or an assistant "
                    "referee with a flag along the touchline - including a player standing just "
                    "outside the line to take a throw-in or corner. False for substitutes, "
                    "coaches, team staff, ball persons, photographers, spectators, and anyone "
                    "behind a fence, on a bench, or standing well off the playing surface."
                ),
            )
        )
        q.append(
            dict(
                type="predicate",
                name=f"inside_{i}",
                instructions=(
                    f"Tile {i}: the feet of the person in the yellow box are on the playing "
                    "surface inside the outer boundary lines (touchlines and goal lines) of the "
                    "soccer field."
                ),
            )
        )
    return q


def ai_candidates(
    people: Mapping[int, Iterable],
    calib: Mapping[int, Optional[np.ndarray]],
    kind: Mapping[int, str],
    legacy_offfield: Set[int],
) -> Set[int]:
    """Tracks whose status the AI answer can change: no calibrated frame, no oracle label, not
    already off-field by the oracle vote."""
    seen: Set[int] = set()
    calibrated: Set[int] = set()
    for f, ps in people.items():
        ok = calib.get(f) is not None
        for p in ps:
            if p[0] >= 0:
                seen.add(p[0])
                if ok:
                    calibrated.add(p[0])
    return {t for t in seen - calibrated if t not in kind and t not in legacy_offfield}


def ask_participants(
    decide: Callable[..., Optional[dict]],
    frames: Mapping[int, np.ndarray],
    people: Mapping[int, Iterable],
    tids: Set[int],
    k: int = 3,
    seed: int = 0,
    box_scale: float = 1.0,
    workers: int = 4,
) -> Tuple[Dict[int, float], dict]:
    """Mean Decisions-API P(on-field participant) per track in `tids`.

    frames: {frame: BGR image}; boxes (full-resolution pixels) are scaled by `box_scale` to
    the image (e.g. render-size frames); up to `k`
    tiles per track at frames spread over its life, 12 numbered tiles per 1024x768 mosaic, the
    tiles shuffled so one track's answers come from different requests.
    decide(images, questions, text=, max_width=) -> {"answers": {name: {"probability": p}}} or
    None (budget exhausted / unavailable); up to `workers` requests run at once. The answers
    depend on the other tiles of a mosaic, so ask about every track of the window and read the
    ones that matter (lab: asking only the uncertain tracks hid real players). Returns
    ({tid: p}, info)."""
    obs: Dict[int, list] = {}
    for f in sorted(frames):
        for p in people.get(f, []):
            if p[0] in tids:
                obs.setdefault(p[0], []).append((f, p[1]))
    samples = []
    for t, ob in sorted(obs.items()):
        idx = sorted({int(round((j + 0.5) / k * len(ob) - 0.5)) for j in range(k)})
        samples += [(t, *ob[i]) for i in idx]
    random.Random(seed).shuffle(samples)
    per = COLS * ROWS
    chunks = [samples[c0 : c0 + per] for c0 in range(0, len(samples), per)]

    def mosaic_of(chunk):
        tiles = [
            person_tile(frames[f], [v * box_scale for v in b], i + 1, 160.0 * box_scale)
            for i, (_, f, b) in enumerate(chunk)
        ]
        tiles += [np.zeros((TILE, TILE, 3), np.uint8)] * (per - len(tiles))
        return np.vstack([np.hstack(tiles[r * COLS : (r + 1) * COLS]) for r in range(ROWS)])

    def ask(item):
        chunk, mosaic = item
        q = participant_questions(range(1, len(chunk) + 1))
        return decide([mosaic], q, text=PARTICIPANT_TEXT, max_width=1024)

    items = [(c, mosaic_of(c)) for c in chunks]
    if workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(min(workers, len(items))) as pool:
            answers = list(pool.map(ask, items))  # order kept: results do not depend on timing
    else:
        answers = [ask(it) for it in items]
    ps: Dict[int, list] = {}
    failed = 0
    for chunk, r in zip(chunks, answers):
        if not r:
            failed += 1
            continue
        ans = r.get("answers") or {}
        for i, (t, _, _) in enumerate(chunk, 1):
            p = (ans.get(f"participant_{i}") or {}).get("probability")
            if p is not None:
                ps.setdefault(int(t), []).append(float(p))
    probs = {t: float(np.mean(v)) for t, v in ps.items() if v}
    return probs, dict(asked=len(obs), tiles=len(samples), calls=len(chunks), failed=failed)
