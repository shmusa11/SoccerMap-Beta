"""Fast keyframe pipeline: expensive models on sparse keyframes, cheap propagation in between.

Profiling the three-pass app showed ~80% of the time is field calibration: the field CNN ran on
every 4th raw frame (24-45 calls per video-second), then the whole clip was re-run at stride 2
when coverage was low. This pipeline decodes once and spends compute only where it changes the
answer:

  Camera     LK optical-flow homography between rendered frames (640 px, ~2 ms per pair).
  Markings   With a vision oracle, every ~2 s the stronger tier lists the visible soccer markings
             with confidences; `field_markings.calibrate_window` fits constrained pinhole cameras
             to each keyframe's own answer, keeps a keyframe only when a fit from different
             answers agrees (keyframes related by direct image registration), and carries it to
             every render frame through the camera motion, verifying each frame against its own
             paint mask. When it keeps a keyframe it replaces the field-detector path below.
             When answers exist but none is confirmed, the multi-sport fallback is skipped and
             every field-detector frame must agree with the nearest keyframe's markings
             (`_check_against_markings`) or stays unavailable; while markings are asked the
             field CNN anchors only every `anchor_gap_with_markings_s`.
  Anchors    The field detector runs only at the start, after the camera has moved, or after a
             maximum interval. Candidates must be physically plausible cameras that put the pitch
             markings on real line pixels. When it fails and a vision oracle is configured, the
             oracle's named landmarks provide another candidate.
  Propagate  Between anchors the calibration is carried by the chained frame-to-frame motion,
             forwards and backwards, and every detection frame is re-verified against the real
             lines. Frames that fail stay unavailable; nothing is filled in to look complete.
  Players    YOLO (no CLAHE, boxes down to conf 0.08) at a reduced rate, then the offline
             camera-compensated tracker (`player_tracker`): every player keeps a box on every
             rendered frame through pans, flicker and short occlusions. Oracle keyframe boxes the
             detector missed seed tracks; oracle answers vote each track's role, team and
             on-field status; jersey colours learnt from the labelled tracks label the rest
             (without an oracle, a colour K=2 fit). Tracks whose feet the calibration mostly
             puts off the pitch (bench, spectators) are hidden.
  Ball       On every render frame: WASB heatmap peaks (full frame, no score cut) and
             camera-compensated motion blobs between the neighbouring distinct raw frames. A
             logistic appearance model scores them and one global DP (`ball_track`) picks the
             path or "no ball"; with an oracle, ~1 Hz crop checks anchor it (not at night). A
             frame without a trusted ball stays empty.

  Threads    Decoding runs one thread ahead; CPU stages (motion blobs, WASB tile preparation and
             night enhance, paint / line masks, render-size frame copies) run on `cpu_workers`
             threads while the main thread drives the GPU models, and the GPU cache is released
             only when another model takes over. Same outputs as inline (bit-identical on the lab
             windows), 2.0-2.6x faster per 10 s window (2026-10-14).
Frame indices are original video frame indices; pitch coordinates are corner-origin metres.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from soccer_analyzer import ball_track as bt
from soccer_analyzer.calibration import field_markings as fm
from soccer_analyzer.calibration import joint_calib as jc
from soccer_analyzer.calibration.clip_calibrator import detector_candidates, select_calibration
from soccer_analyzer.calibration.fusion import CoverageMap
from soccer_analyzer.calibration.homography import camera_inside_pitch
from soccer_analyzer.calibration.line_support import line_mask, line_support
from soccer_analyzer.calibration.mapper import FrameCalibration
from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH
from soccer_analyzer.calibration.visualize import compose, draw_calibration_overlay, draw_topdown
from soccer_analyzer.detection import TeamAssigner
from soccer_analyzer.match_processor import MatchData
from soccer_analyzer.onfield import OnFieldCfg, ai_candidates, ask_participants, onfield_visibility
from soccer_analyzer.player_tracker import TrackCfg, appearance_features, track_people
from soccer_analyzer.video import (
    frame_limit,
    open_capture,
    open_writer,
    prefetch_frames,
    skip_frames,
)

# Box / dot colours (BGR): team A red, team B blue, referee yellow, goalkeeper magenta, unknown grey.
TEAM_BGR = {
    0: (40, 40, 235),
    1: (235, 120, 40),
    -1: (170, 170, 170),
    -2: (0, 255, 255),
    -3: (255, 0, 255),
}
_CHROMA_WEIGHT = np.array([0.5, 1.6, 1.6])  # LAB weights: chroma survives shade, lightness doesn't


@dataclass
class FastConfig:
    render_hz: float = 15.0  # output video / camera-motion rate
    detect_every: int = 2  # calibration verification on every Nth rendered frame
    people_every: int = 2  # people detector on every Nth rendered frame (tracker fills the rest)
    people_conf: float = 0.08  # YOLO floor (= tracking.conf_low: lower boxes are never used)
    # Who is drawn (`onfield`): a track is shown within +-window_s of a moment it is on the
    # pitch (throw-in takers stay, beside the touchline on the 2D map); never-on tracks hidden.
    onfield: OnFieldCfg = field(default_factory=OnFieldCfg)
    # Tracks with no calibrated frame and no oracle label: ask the Decisions API (oracle
    # `decide`) "is this person taking part in the match" on up to `onfield_ai_k` crops each,
    # from the ~1 Hz oracle frames (~$0.0002 per 12 crops).
    onfield_ai: bool = True
    onfield_ai_k: int = 3
    tracking: TrackCfg = field(default_factory=TrackCfg)
    # Ball tracker (`ball_track`), on every render frame when a WASB detector is given.
    ball_vlm: bool = True  # ~1 Hz oracle crop checks anchor the track (needs an oracle)
    ball_vlm_every: int = 15  # render frames between crop checks
    ball_vlm_max: int = 20  # crop checks per run
    ball_gate: float = 0.10  # min mean confidence of a reported ball segment (with checks)
    ball_gate_offline: float = 0.05  # ... without checks (the DP alone needs a looser gate)
    keyframe_s: float = 2.0  # oracle people call cadence
    # VLM field-marking calibration (`calibration.field_markings`), when the oracle can name
    # markings. Keyframes every `markings_every_s`, or exactly `markings_frames` (original
    # indices, snapped to the render grid) when given.
    field_markings: bool = True
    markings_every_s: float = 2.0
    # Also ask the cheap tier at each marking keyframe and use its list when it names more
    # markings (+~$0.0007 per keyframe). The lab's held-out window results used this rule: on
    # the multi-sport clip, 3 of 6 keyframes took the cheap tier's longer list.
    markings_second_opinion: bool = True
    markings_frames: Optional[Sequence[int]] = None
    max_markings_calls: int = 40
    markings_window_s: float = 3.0  # keyframe markings pooled within +-this many seconds
    markings_tol_m: float = 2.5  # consensus: two keyframe fits must agree within this
    markings_min_ls: float = 0.25  # per-frame paint support floor (verify mask)
    markings_ratio: float = 0.8  # ...and this fraction of its keyframe's own support
    # The older landmark seeds (same model tier) are skipped while markings run: on the lab
    # clips they returned 0-1 landmarks where the detector failed and never set the coverage.
    landmarks_with_markings: bool = False
    # A keyframe-agreement check for every frame the field-detector path calibrates while
    # marking answers exist but no marking fit was confirmed: the confident markings of the
    # nearest keyframe (same motion segment, within markings_window_s) must land within this
    # many metres (median) of their pitch points, else the frame stays unavailable.
    markings_check_tol_m: float = 2.5
    # How the marking answers become a calibration. "joint" (`calibration.joint_calib`): the
    # camera only turns and zooms, so ONE camera centre is shared by every frame; landmarks
    # from all keyframes, the camera motion and the paint of many frames are fitted together
    # (forward and backward from the keyframes), and a frame is shown only when two fits from
    # disjoint halves of the keyframe answers agree with it within `joint_tol_m`. "window" =
    # the round-2 per-keyframe fits (`field_markings.calibrate_window`).
    markings_method: str = "joint"
    joint_tol_m: float = 1.5
    # How marking keyframes are chosen when `markings_frames` is not given. "coverage": after
    # the sweep, up to `markings_keys` per motion segment by VIEW COVERAGE (widest view first,
    # then the view farthest in pan / tilt / zoom from those chosen) among candidates every
    # `markings_candidate_s` (or exactly `markings_candidates`); the cost then follows the
    # camera's range of views, not the window length. "time": every `markings_every_s`.
    markings_select: str = "coverage"
    # 2 per segment first (night: 2 keyframes, 0.40-0.45 m on the dev references); when no
    # frame is confirmed the budget doubles (2 -> 4 -> 8 -> 12) up to `markings_max_keys`.
    markings_keys: int = 2
    markings_max_keys: int = 8
    markings_candidate_s: float = 0.5
    markings_candidates: Optional[Sequence[int]] = None
    # A candidate must show >= this fraction of the paint of its segment's best candidate:
    # a screen overlay (phone UI, play button) hides paint, so such a frame is not picked as
    # the "widest view" (night clip: frame 0 under the phone UI).
    markings_paint_ratio: float = 0.8
    # Candidates are kept as JPEGs at <= this width (the oracle sends <= 1920 px anyway) and
    # within this many MB; chosen keyframes are scaled back to the full frame size.
    markings_candidate_width: int = 1920
    markings_candidate_mb: float = 100.0
    # Stop asking when a confirmed field still leaves this share of frames outside every
    # keyframe's view; else one more keyframe per segment (the next-farthest view) is asked.
    markings_unseen_frac: float = 0.05
    # Joint-fit safety: a frame must have >= joint_min_seen of its view seen by a keyframe,
    # the halves must agree on the worst tenth of the view within 2 x joint_tol_m, and the
    # whole fit must survive `joint_stability_runs` repeats with jittered camera motion.
    joint_min_seen: float = 0.5
    joint_stability_runs: int = 1
    # Football mode: prior width (m) on the soccer goal line sitting on the football end line
    # (None: joint_calib.FB_L_PRIOR, 1 m).
    joint_football_l_sigma: Optional[float] = None
    # A video whose field stayed unconfirmed after the full keyframe budget in this many
    # windows is not asked again (needs an oracle with a memo, e.g. CachedVisionOracle):
    # the per-match cost of an uncalibratable camera setup is bounded. Only windows with >=
    # markings_max_keys answers naming markings and no AI call short count (`_markings_memo`);
    # a confirmed window resets the count.
    markings_fail_windows: int = 2
    # American-football-lined fields ("joint" only): "auto" reads the yard numbers at two
    # keyframes when the soccer lines are not white, and aligns the football grid too when a
    # number is read; "on" always asks, "off" never.
    markings_football: str = "auto"
    # While marking keyframes are being asked, the field CNN only anchors sparsely (every this
    # many seconds; no motion-triggered anchors): it is needed only if the markings fail.
    anchor_gap_with_markings_s: float = 6.0
    max_anchor_gap_s: float = 3.0  # re-anchor at least this often
    anchor_motion: float = 0.05  # ...or after this much camera motion (fraction of diagonal)
    retry_s: float = 0.5  # after a failed anchor attempt, wait before trying again
    max_landmark_calls: int = 8  # oracle landmark calls per run (only when the detector fails)
    multisport_fallback: bool = True  # colour-agnostic calibration when < 10% is calibrated
    # Anchor acceptance (line support at native resolution). Stricter than a single frame's
    # 0.38: an anchor is copied across many frames, and broadcast close-ups with few markings
    # produced wrong fits at 0.38-0.39 while correct night anchors scored 0.45-0.65.
    ls_floor: float = 0.42
    verify_ratio: float = 0.6  # propagated frame keeps >= this fraction of its anchor's support
    verify_min: float = 0.15
    work_width: int = 1280  # verification line masks are computed at <= this width
    lk_width: int = 640
    yolo_batch: int = 4
    out_width: int = 1600  # rendered side-by-side video width
    # Speed (results unchanged): CPU stages (motion blobs, paint / line masks, frame copies for
    # the renderer) run on this many worker threads while the main thread drives the GPU models;
    # 0 runs them inline. Decoding runs on its own thread `prefetch` frames ahead.
    cpu_workers: int = 3
    prefetch: int = 6
    # Keep a JPEG of every render frame at the render size from pass 1, so `render` does not
    # decode the video a second time (0.12-0.32 MB per frame at 1280 px). A window longer than
    # this budget is decoded again instead (0 = always decode again).
    render_cache_mb: float = 400.0

    def __post_init__(self):
        if min(self.render_hz, self.keyframe_s, self.max_anchor_gap_s, self.markings_every_s) <= 0:
            raise ValueError("rates and intervals must be positive")
        if min(self.detect_every, self.people_every, self.yolo_batch, self.ball_vlm_every) <= 0:
            raise ValueError("cadences and batch size must be positive")
        if self.cpu_workers < 0 or self.prefetch < 1 or self.render_cache_mb < 0:
            raise ValueError("cpu_workers and render_cache_mb must be >= 0, prefetch >= 1")
        if self.markings_method not in ("joint", "window"):
            raise ValueError("markings_method must be 'joint' or 'window'")
        if self.markings_select not in ("coverage", "time"):
            raise ValueError("markings_select must be 'coverage' or 'time'")
        if self.markings_football not in ("auto", "on", "off"):
            raise ValueError("markings_football must be 'auto', 'on' or 'off'")
        if self.markings_keys < 2 or self.markings_candidate_s <= 0 or self.joint_tol_m <= 0:
            raise ValueError("markings_keys >= 2; markings_candidate_s and joint_tol_m > 0")
        if self.markings_max_keys < self.markings_keys:
            raise ValueError("markings_max_keys must be >= markings_keys")
        if self.joint_stability_runs < 0 or not 0 <= self.joint_min_seen <= 1:
            raise ValueError("joint_stability_runs >= 0 and 0 <= joint_min_seen <= 1")


def release_gpu_cache() -> None:
    """Return cached GPU (MPS/CUDA) blocks to the system. Three models share one GPU here, and
    the caching allocator otherwise keeps every model's peak (8.4 GB on a 64-frame HD window with
    the per-frame ball detector, 2026-10-13); after each batch it stays bounded."""
    try:
        import torch
    except ImportError:
        return
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


class StageTimer:
    """Wall-clock seconds per named stage."""

    def __init__(self):
        self.seconds: Dict[str, float] = defaultdict(float)
        self.calls: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()  # stages also run on worker threads

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            with self._lock:
                self.seconds[name] += dt
                self.calls[name] += 1

    def table(self) -> List[Tuple[str, float, int]]:
        return sorted(((k, v, self.calls[k]) for k, v in self.seconds.items()), key=lambda r: -r[1])


@dataclass
class FastResult:
    """Everything the renderer needs, keyed by original frame index."""

    fps: float
    frame_size: Tuple[int, int]
    render_frames: List[int]
    detect_frames: List[int]
    calib: Dict[int, Optional[np.ndarray]]  # render frame -> image->pitch H or None
    calib_source: Dict[int, str]  # "anchor:<tag>" | "propagated" | "none"
    md: MatchData
    offfield: set  # hidden tracks (never on the pitch / oracle vote / AI), for stats
    team_names: Dict[int, str]
    keyframes: Dict[int, list] = field(default_factory=dict)  # frame -> [OraclePerson]
    stats: dict = field(default_factory=dict)
    pitch: Optional[PitchModel] = None  # the pitch `calib` maps to (fitted dims with markings)
    # render frame -> JPEG of the frame at the render size (`FastConfig.render_cache_mb`)
    render_cache: Dict[int, np.ndarray] = field(default_factory=dict)
    # render frame -> track ids drawn (`onfield`); None = draw every track not in `offfield`
    visible: Optional[Dict[int, Set[int]]] = None
    # render frame -> {tid: pitch (x, y) m} of the drawn tracks on calibrated frames, including
    # positions just off the pitch (throw-in takers)
    pitch_xy: Dict[int, Dict[int, Tuple[float, float]]] = field(default_factory=dict)
    onfield_status: Dict[int, str] = field(default_factory=dict)
    motion: Dict[int, Optional[np.ndarray]] = field(default_factory=dict)


class _LazyJpegs(Mapping):
    """{frame: BGR image} decoded on first access from {frame: JPEG bytes}."""

    def __init__(self, jpgs: Mapping[int, np.ndarray]):
        self._jpgs, self._img = jpgs, {}

    def __getitem__(self, f):
        if f not in self._img:
            self._img[f] = cv2.imdecode(self._jpgs[f], cv2.IMREAD_COLOR)
        return self._img[f]

    def __iter__(self):
        return iter(self._jpgs)

    def __len__(self):
        return len(self._jpgs)


class _Work:
    """CPU stages on worker threads, overlapping the GPU models on the main thread (OpenCV and
    large NumPy operations release the GIL). Each job is a pure function of its arguments, so the
    results are the same as running inline (`workers` = 0). At most `max_pending` jobs are queued
    (each holds full-resolution frames); beyond that the main thread waits for the oldest.
    Stage times of worker jobs are thread time and overlap the main-thread stages."""

    def __init__(self, workers: int, timer: StageTimer, max_pending: Optional[int] = None):
        self.T = timer
        self.pool = ThreadPoolExecutor(workers, "cpu-stage") if workers > 0 else None
        self.max_pending = max_pending or 2 * max(1, workers)
        self._pending: deque = deque()

    def submit(self, stage: str, fn, *args) -> Future:
        if self.pool is None:
            fut: Future = Future()
            with self.T(stage):
                fut.set_result(fn(*args))
            return fut

        def job():
            with self.T(stage):
                return fn(*args)

        fut = self.pool.submit(job)
        self._pending.append(fut)
        while self._pending and self._pending[0].done():
            self._pending.popleft()
        if len(self._pending) > self.max_pending:
            with self.T("workers: main thread waiting"):
                while len(self._pending) > self.max_pending:
                    self._pending.popleft().result()
        return fut

    def __enter__(self) -> "_Work":
        return self

    def __exit__(self, *exc) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


def _timed(items, timer: StageTimer, stage: str):
    """Iterate `items`, timing the waits for each next item as `stage`."""
    it = iter(items)
    while True:
        with timer(stage):
            item = next(it, _DONE)
        if item is _DONE:
            return
        yield item


_DONE = object()


def _resolve(jobs: Dict[int, Future], timer: StageTimer) -> Dict[int, object]:
    with timer("workers: main thread waiting"):
        return {k: f.result() for k, f in jobs.items()}


def _render_size(fw: int, fh: int) -> Tuple[float, int, int]:
    scale = min(1.0, 1280.0 / fw)
    return scale, int(round(fw * scale)), int(round(fh * scale))


def _render_jpeg(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return jpg


def _line_mask_packed(frame: np.ndarray, work_scale: float) -> Tuple[np.ndarray, tuple]:
    small = cv2.resize(frame, None, fx=work_scale, fy=work_scale)
    m = line_mask(small) > 0
    return np.packbits(m), m.shape


_TLS = threading.local()


def _clahe(clip: float) -> "cv2.CLAHE":
    """Per-thread CLAHE (8x8 tiles): an OpenCV CLAHE object keeps scratch buffers and is not
    thread-safe."""
    d = _TLS.__dict__.setdefault("clahe", {})
    if clip not in d:
        d[clip] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return d[clip]


def _gray_small(frame: np.ndarray, width: int, clahe) -> Tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    s = width / float(w)
    g = cv2.cvtColor(cv2.resize(frame, (width, max(1, int(round(h * s))))), cv2.COLOR_BGR2GRAY)
    return clahe.apply(g), s


def frame_motion(prev: np.ndarray, cur: np.ndarray, scale: float) -> Optional[np.ndarray]:
    """Homography mapping full-res pixels of `prev` to `cur` from LK flow on small grays, or None
    on a cut / unreliable estimate. Moving players are a minority and fall out as RANSAC outliers;
    for a panning (rotating) camera the whole view obeys one homography."""
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=500, qualityLevel=0.01, minDistance=7)
    if pts is None or len(pts) < 30:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 30:
        return None
    G, inl = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)
    if G is None or inl is None or inl.sum() < max(25, 0.4 * ok.sum()):
        return None
    S = np.diag([scale, scale, 1.0])
    return np.linalg.inv(S) @ G @ S


def mean_displacement(G: np.ndarray, size: Tuple[int, int]) -> float:
    """Mean pixel displacement of a 3x3 grid under G, as a fraction of the frame diagonal."""
    w, h = size
    gx, gy = np.meshgrid(np.linspace(0.1 * w, 0.9 * w, 3), np.linspace(0.1 * h, 0.9 * h, 3))
    p = np.stack([gx.ravel(), gy.ravel(), np.ones(9)])
    q = G @ p
    q = q[:2] / q[2:3]
    return float(np.mean(np.linalg.norm(q - p[:2], axis=0)) / np.hypot(w, h))


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(tracked, boxes, min_iou: float = 0.25) -> Dict[int, int]:
    """Greedy one-to-one match {track_id: oracle box index} by IoU (highest first)."""
    pairs = sorted(
        ((_iou(t[1], b), t[0], j) for t in tracked if t[0] >= 0 for j, b in enumerate(boxes)),
        reverse=True,
    )
    out, used = {}, set()
    for s, tid, j in pairs:
        if s < min_iou:
            break
        if tid not in out and j not in used:
            out[tid] = j
            used.add(j)
    return out


def vote_labels(keyframes, people_at) -> Tuple[Dict[int, str], Dict[int, int], set, Dict[int, str]]:
    """Per-track labels from oracle keyframes.

    keyframes: {frame_idx: [OraclePerson]}; people_at: {frame_idx: [(tid, bbox, feet)]}.
    Returns (kind per tid, team code per tid, off-field tids, {team code: colour name}). The two
    most frequent player jersey colours become teams 0 and 1. A track present at a keyframe but
    never matched to an on-field person at two or more keyframes is off-field (spectator,
    bench, coach)."""
    colour_counts = Counter(
        p.team for ps in keyframes.values() for p in ps if p.role == "player" and p.team
    )
    names = [c for c, _ in colour_counts.most_common(2)]
    # Boxes use a fixed palette (code 0 red, code 1 blue): put a red kit on code 0 and a blue
    # kit on code 1 so the box colour never contradicts the shirt.
    if len(names) == 2 and ("red" in names[1] or "blue" in names[0]):
        names.reverse()
    code_of = {c: i for i, c in enumerate(names)}
    roles: Dict[int, Counter] = defaultdict(Counter)
    teams: Dict[int, Counter] = defaultdict(Counter)
    seen: Counter = Counter()
    for fidx, persons in keyframes.items():
        tracked = people_at.get(fidx, [])
        centres = np.array(
            [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in (p.box for p in persons)]
        )
        for tid, (x1, y1, x2, y2), _ in tracked:
            if tid < 0:
                continue
            # A miss right next to oracle people is a player the oracle skipped in a cluster,
            # not evidence of a spectator; only isolated misses count towards off-field.
            c = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            if len(centres) and np.min(np.linalg.norm(centres - c, axis=1)) < 1.5 * (y2 - y1):
                continue
            seen[tid] += 1
        for tid, j in match_boxes(tracked, [p.box for p in persons]).items():
            roles[tid][persons[j].role] += 1
            if persons[j].role != "referee" and persons[j].team in code_of:
                teams[tid][code_of[persons[j].team]] += 1
    kind = {tid: c.most_common(1)[0][0] for tid, c in roles.items()}
    team = {tid: c.most_common(1)[0][0] for tid, c in teams.items() if kind.get(tid) != "referee"}
    # Off-field only when present at >= 2 keyframes and never matched: the oracle occasionally
    # misses one player inside a tight cluster, and that must not hide a real player.
    offfield = {tid for tid, n in seen.items() if n >= 2 and tid not in roles}
    return kind, team, offfield, {i: c for c, i in code_of.items()}


def two_means(X: np.ndarray, iters: int = 30) -> Optional[np.ndarray]:
    """Deterministic 2-means: start from the two points farthest apart, then Lloyd iterations."""
    if len(X) < 2:
        return None
    d = np.linalg.norm(X[:, None] - X[None], axis=2)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    c = X[[i, j]].astype(float)
    for _ in range(iters):
        lab = np.argmin(np.linalg.norm(X[:, None] - c[None], axis=2), axis=1)
        new = np.array([X[lab == k].mean(axis=0) if np.any(lab == k) else c[k] for k in (0, 1)])
        if np.allclose(new, c):
            break
        c = new
    return c


def _markings_memo(
    fails: int, confirmed: bool, exhausted: bool, n_marked: int, shortfall: int, max_keys: int
) -> int:
    """The per-video count of windows whose field was UNCONFIRMABLE, after this window.

    A confirmed window resets it to 0 (the camera setup calibrates; a later failure starts over).
    A window counts only when it is real evidence that the field cannot be confirmed: the
    keyframe budget ran out (`exhausted`) with at least `max_keys` answers that named markings,
    no AI call short (none skipped for budget, none failed), and still no frame confirmed. A
    window starved by the budget or by an outage leaves the count unchanged."""
    if confirmed:
        return 0
    if exhausted and n_marked >= max_keys and shortfall == 0:
        return fails + 1
    return fails


# Keyframe candidates kept in memory per run (full-res JPEG, ~0.5-3 MB each).
_MAX_CANDIDATES = 240


def _jpeg(frame: np.ndarray, width: Optional[int] = None) -> Optional[np.ndarray]:
    if width is not None and frame.shape[1] > width:
        s = width / float(frame.shape[1])
        frame = cv2.resize(
            frame,
            (int(width), int(round(frame.shape[0] * s))),
            interpolation=cv2.INTER_AREA,
        )
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return jpg if ok else None


class _BallSweep:
    """Pass-1 ball candidates on every render frame, from a sequential decode (no seeking).

    WASB sees the raw trio [f-1, f, f+1] (CLAHE on night footage, `ball_track.is_night`); the
    motion blobs compare f with its previous / next DISTINCT raw frames (screen recordings and
    frame-rate conversions repeat frames), so a render frame is processed once the next distinct
    frame has been decoded. Also keeps 480-px greys of every 2nd raw frame (static-overlay mask)
    and JPEGs of the render frames the oracle may be asked about."""

    def __init__(self, wasb, timer: StageTimer, start_frame: int, vlm_every: int, work: _Work):
        self.wasb, self.T, self.work = wasb, timer, work
        self.start = start_frame
        self.vlm_every = vlm_every
        self.wasb_c: Dict[int, np.ndarray] = {}  # render frame -> (n, 3) x, y, score
        self.blobs: Dict[int, np.ndarray] = {}  # render frame -> ball_track.motion_blobs rows
        self.grays: List[np.ndarray] = []
        self.jpgs: Dict[int, np.ndarray] = {}
        self._blob_jobs: Dict[int, Future] = {}
        self._jpg_jobs: Dict[int, Future] = {}
        # WASB inputs (night enhance + tile preparation) are made on the workers; the GPU pass
        # runs this many render frames behind, so its input is usually ready.
        self._wasb_jobs: deque = deque()
        self._wasb_lag = 2
        self.night_frames = 0
        self.duplicates = 0
        self._pending: List[dict] = []
        self._last = None  # (frame, grey) of the previous raw frame
        self._run_prev = None  # last raw frame before the current run of identical frames
        self._n_render = 0

    def push(self, idx: int, frame: np.ndarray, is_render: bool, g: Optional[np.ndarray] = None):
        """`g`: `ball_track.small_gray(frame)` when already computed (the decode thread does)."""
        with self.T("ball: duplicate test"):
            g = bt.small_gray(frame) if g is None else g
            dup = self._last is not None and bt.is_duplicate(g, self._last[1])
        if (idx - self.start) % 2 == 0:
            self.grays.append(g)
        if dup:
            self.duplicates += 1
        else:
            for p in self._pending:
                if p["nextd"] is None:
                    p["nextd"] = frame
            self._run_prev = None if self._last is None else self._last[0]
        for p in self._pending:
            if p["f"] == idx - 1:
                p["next1"] = frame
        done = [p["next1"] is not None and p["nextd"] is not None for p in self._pending]
        ready = [p for p, d in zip(self._pending, done) if d]
        self._pending = [p for p, d in zip(self._pending, done) if not d]
        for p in ready:
            self._process(p)
        if is_render:
            if self._n_render % self.vlm_every == 0:
                self._jpg_jobs[idx] = self.work.submit("ball: oracle frames", _jpeg, frame)
            self._n_render += 1
            prev1 = None if self._last is None else self._last[0]
            self._pending.append(
                dict(f=idx, cur=frame, prev1=prev1, prevd=self._run_prev, next1=None, nextd=None)
            )
        self._last = (frame, g)

    def finish(self) -> None:
        for p in self._pending:
            self._process(p)
        self._pending = []
        while self._wasb_jobs:
            self._wasb_oldest()
        self.blobs = _resolve(self._blob_jobs, self.T)
        self.jpgs = {f: j for f, j in _resolve(self._jpg_jobs, self.T).items() if j is not None}

    @staticmethod
    def _enhance(frame: np.ndarray) -> np.ndarray:
        l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
        return cv2.cvtColor(cv2.merge([_clahe(2.5).apply(l), a, b]), cv2.COLOR_LAB2BGR)

    def _wasb_input(self, trio: List[np.ndarray]) -> Tuple[bool, object]:
        """Worker job: (night?, WASB input) - the prepared tiles when the detector can split its
        CPU and GPU halves, else the (enhanced) trio."""
        night = bt.is_night(trio[1])
        if night:
            trio = [self._enhance(t) for t in trio]
        prepare = getattr(self.wasb, "prepare", None)
        return night, (trio if prepare is None else prepare(trio))

    def _wasb_oldest(self) -> None:
        f, job = self._wasb_jobs.popleft()
        with self.T("workers: main thread waiting"):
            night, x = job.result()
        self.night_frames += int(night)
        # No cache release per frame: consecutive WASB calls reuse the same blocks; the cache is
        # released before another model runs (YOLO, field CNN).
        with self.T("ball: WASB peaks"):
            if hasattr(self.wasb, "peaks_prepared"):
                peaks = self.wasb.peaks_prepared(x)
            else:
                peaks = self.wasb.peaks(x)
            self.wasb_c[f] = np.array(peaks, float).reshape(-1, 3)

    def _process(self, p: dict) -> None:
        cur = p["cur"]
        trio = [cur if q is None else q for q in (p["prev1"], cur, p["next1"])]
        self._blob_jobs[p["f"]] = self.work.submit(
            "ball: motion blobs", bt.motion_blobs, cur, p["prevd"], p["nextd"]
        )
        job = self.work.submit("ball: WASB inputs", self._wasb_input, trio)
        self._wasb_jobs.append((p["f"], job))
        while len(self._wasb_jobs) > self._wasb_lag:
            self._wasb_oldest()


class FastPipeline:
    """Decode once, anchor calibration sparsely, track cheaply, label with sparse oracle calls.

    Detectors are injected: `field_detector.detect(frame, idx)` -> FieldObservation,
    `people` a SoccerDetector (its `_yolo` finds people; `_enhance` prepares ball frames),
    `wasb` a WASBBallDetector or None, `oracle` a VisionOracle or None (offline: colour teams,
    no landmark fallback)."""

    def __init__(
        self,
        field_detector,
        people,
        wasb=None,
        oracle=None,
        pitch: PitchModel = STANDARD_PITCH,
        config: Optional[FastConfig] = None,
    ):
        self.field = field_detector
        self.people = people
        self.wasb = wasb
        self.oracle = oracle
        self.pitch = pitch
        self.cfg = config or FastConfig()
        self.timer = StageTimer()
        self._dims = (pitch.length, pitch.width)

    # ---- pass 1: sweep ------------------------------------------------------------------------

    def run(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        progress=None,
        start_frame: int = 0,
    ) -> FastResult:
        """Analyse `max_frames` raw frames starting at `start_frame` (original indices)."""
        cfg, T = self.cfg, self.timer
        frame_limit(0, max_frames)
        if start_frame < 0:
            raise ValueError("start_frame must be >= 0")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        with open_capture(video_path) as cap, _Work(cfg.cpu_workers, T) as work:
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if np.isfinite(fps) and fps > 0 else 30.0
            count = frame_limit(cap.get(cv2.CAP_PROP_FRAME_COUNT), None)
            total = start_frame + max_frames if max_frames else count
            if count:
                total = min(total, count)
            fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Sequential skip, never a seek: seeking lands on the wrong frame for some codecs.
            skip_frames(cap, start_frame)
            render_stride = max(1, int(round(fps / cfg.render_hz)))
            detect_stride = render_stride * cfg.detect_every
            people_stride = render_stride * cfg.people_every
            # Mean brightness cannot tell night from day (letterboxing and dark turf put a daytime
            # clip below a floodlit night one), so the field CNN's CLAHE mode is chosen by trial
            # at the first anchors (`_anchor`). Players/ball use the detector's per-frame rule.
            self._field_enh = None
            self.anchor_log: List[dict] = []
            self.landmark_log: List[dict] = []
            self._anchor_masks: Dict[int, tuple] = {}
            self._mk_answers: Dict[int, tuple] = {}
            self._numbers: Dict[int, list] = {}  # yard-number reads per keyframe (football)
            # AI calls this window could not make (budget) or that failed (error / timeout): such
            # a window says nothing about whether the field is confirmable (`_markings_memo`).
            self._mk_short = 0
            self._mk_failed: set = set()
            work_scale = min(1.0, cfg.work_width / float(fw))
            md = MatchData()
            render_frames: List[int] = []
            motion: Dict[int, Optional[np.ndarray]] = {}  # render frame -> G(prev render -> it)
            masks: Dict[int, Tuple[np.ndarray, tuple]] = {}  # detect frame -> packed line mask
            anchors: Dict[int, Tuple[np.ndarray, float, str]] = {}  # frame -> (H, ls, tag)
            pending_lm: Dict[int, Tuple[object, np.ndarray, tuple]] = {}  # failed anchor -> ...
            people_jobs: Dict[int, object] = {}
            raw_dets: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}  # frame -> (xyxy, conf)
            app_feats: Dict[int, np.ndarray] = {}  # frame -> (n, 104) jersey histograms
            jersey: Dict[int, Dict[int, np.ndarray]] = {}  # frame -> {det index: LAB colour}
            ball = (
                _BallSweep(self.wasb, T, start_frame, cfg.ball_vlm_every, work)
                if self.wasb is not None
                else None
            )
            render_size = _render_size(fw, fh)[1:]
            render_jobs: Dict[int, Future] = {}  # render frame -> JPEG at the render size
            # measured 0.13-0.35 bytes per pixel at JPEG quality 95
            cache_max = int(cfg.render_cache_mb * 1e6 / (0.35 * render_size[0] * render_size[1]))
            use_markings = (
                cfg.field_markings
                and self.oracle is not None
                and hasattr(self.oracle, "submit_markings")
            )
            field_markings_requested = use_markings
            mk_frames = (
                None
                if cfg.markings_frames is None
                else {int(round(f / render_stride)) * render_stride for f in cfg.markings_frames}
            )
            # Keyframes by view coverage: candidate frames are kept (JPEG) during the sweep and
            # the keyframes are chosen once the whole window's camera motion is known.
            by_coverage = use_markings and mk_frames is None and cfg.markings_select == "coverage"
            mk_cands = (
                None
                if cfg.markings_candidates is None
                else {
                    int(round(f / render_stride)) * render_stride for f in cfg.markings_candidates
                }
            )
            mk_store: Dict[int, Future] = {}  # candidate keyframe -> JPEG (<= candidate width)
            last_cand_t, cand_s = -1e9, cfg.markings_candidate_s
            cand_w = min(fw, cfg.markings_candidate_width)
            # bounded by bytes, not only by count (measured <= 0.35 bytes/pixel at quality 95)
            max_cands = max(
                8,
                min(
                    _MAX_CANDIDATES,
                    int(cfg.markings_candidate_mb * 1e6 / (0.35 * cand_w * cand_w * fh / fw)),
                ),
            )
            mk_memo = getattr(self.oracle, "memo_get", None)
            mk_fails = int((mk_memo("markings_failed_windows") or 0) if mk_memo else 0)
            if use_markings and mk_fails >= cfg.markings_fail_windows:
                use_markings = False  # this video's field was never confirmable: stop paying
            mk_masks: Dict[int, dict] = {}  # render frame -> packed paint masks (960 px)
            mk_images: Dict[int, np.ndarray] = {}  # marking keyframe -> full-res frame
            mk_jobs: Dict[int, object] = {}
            last_mk_t = -1e9
            batch: List[Tuple[int, np.ndarray]] = []
            prev_gray = None
            G_since = np.eye(3)
            last_anchor_t = last_attempt_t = -1e9
            last_attempt_ok = True
            cut_since_anchor = False
            last_kf_t = -1e9
            n_landmark_calls = 0

            def flush():
                if not batch:
                    return
                # Raw frames: CLAHE on dark frames lowered people recall (lab, night and union).
                release_gpu_cache()  # the ball model's cached blocks
                with T("players: yolo"):
                    results = self.people._yolo.predict(
                        [f for _, f in batch],
                        classes=[0],
                        conf=cfg.people_conf,
                        imgsz=self.people.people_imgsz,
                        iou=0.6,
                        device=self.people.device,
                        verbose=False,
                    )
                for (fidx, f), r in zip(batch, results):
                    xyxy = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
                    conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
                    usable = conf >= cfg.tracking.conf_low  # the tracker ignores lower boxes
                    xyxy, conf = xyxy[usable], conf[usable]
                    with T("players: colours"):
                        raw_dets[fidx] = (xyxy.astype(float), conf.astype(float))
                        app_feats[fidx] = appearance_features(
                            f, xyxy, conf, min_conf=cfg.tracking.conf_low
                        )
                        jersey[fidx] = {}
                        for j, (b, c) in enumerate(zip(xyxy, conf)):
                            if c >= cfg.tracking.conf_low and b[3] - b[1] >= 20:
                                lab = TeamAssigner._jersey_lab(f, b)
                                if lab is not None:
                                    jersey[fidx][j] = lab
                batch.clear()
                release_gpu_cache()

            def want(i: int) -> bool:
                return ball is not None or i % render_stride == 0  # the ball needs every raw frame

            def small_grey(f: np.ndarray) -> Optional[np.ndarray]:
                if ball is None:
                    return None
                with T("ball: duplicate test"):
                    return bt.small_gray(f)

            n_raw = 0
            with prefetch_frames(cap, start_frame, total, want, cfg.prefetch, small_grey) as dec:
                for idx, frame, grey in _timed(dec, T, "decode (waiting)"):
                    n_raw += 1
                    is_render = idx % render_stride == 0
                    if progress is not None and total and (idx + 1) % 30 == 0:
                        done = (idx + 1 - start_frame) / max(1, total - start_frame)
                        progress(min(1.0, done))
                    if ball is not None:
                        ball.push(idx, frame, is_render, grey)
                    if not is_render:
                        continue
                    t_now = idx / fps
                    with T("camera motion (LK)"):
                        gray, s = _gray_small(frame, cfg.lk_width, clahe)
                        G = None if prev_gray is None else frame_motion(prev_gray, gray, s)
                        prev_gray = gray
                    render_frames.append(idx)
                    motion[idx] = G
                    if len(render_jobs) < cache_max:
                        render_jobs[idx] = work.submit(
                            "render: frame cache", _render_jpeg, frame, render_size
                        )
                    elif render_jobs:  # over the budget: `render` decodes the video again
                        render_jobs.clear()
                        cache_max = 0
                    if use_markings:
                        mk_masks[idx] = work.submit(
                            "calibration: marking masks", fm.verify_mask, frame
                        )
                        if by_coverage:
                            cand = (
                                idx in mk_cands
                                if mk_cands is not None
                                else t_now - last_cand_t >= cand_s - 1e-6
                            )
                            if cand:
                                mk_store[idx] = work.submit(
                                    "calibration: keyframe candidates", _jpeg, frame, cand_w
                                )
                                last_cand_t = t_now
                                if len(mk_store) > max_cands:
                                    # bounded memory on long videos: keep every other
                                    # candidate and halve the candidate rate from here on
                                    for f in sorted(mk_store)[1::2]:
                                        del mk_store[f]
                                    cand_s *= 2.0
                        due = not by_coverage and (
                            idx in mk_frames
                            if mk_frames is not None
                            else t_now - last_mk_t >= cfg.markings_every_s - 1e-6
                        )
                        if due and len(mk_jobs) < cfg.max_markings_calls:
                            job = self.oracle.submit_markings(idx, frame)
                            if job is None:
                                self._mk_short += 1
                            else:
                                alt = (
                                    self.oracle.submit_markings(idx, frame, alt=True)
                                    if cfg.markings_second_opinion
                                    and cfg.markings_method == "window"
                                    else None
                                )
                                mk_jobs[idx] = (job, alt)
                                mk_images[idx] = frame.copy()
                                last_mk_t = t_now
                    if len(render_frames) > 1:
                        if G is None:
                            cut_since_anchor = True
                        else:
                            G_since = G @ G_since
                    if idx % detect_stride == 0:
                        masks[idx] = work.submit(
                            "verify: line masks", _line_mask_packed, frame, work_scale
                        )
                        moved = mean_displacement(G_since, (fw, fh)) >= cfg.anchor_motion
                        due = (
                            (
                                not anchors
                                or t_now - last_anchor_t >= cfg.max_anchor_gap_s
                                or moved
                                or cut_since_anchor
                            )
                            if not use_markings
                            else (
                                not anchors
                                or t_now - last_anchor_t >= cfg.anchor_gap_with_markings_s
                            )
                        )
                        if use_markings and not anchors:
                            # sparse: one attempt per gap, not a retry every retry_s
                            due = due and t_now - last_attempt_t >= cfg.anchor_gap_with_markings_s
                        if due and (last_attempt_ok or t_now - last_attempt_t >= cfg.retry_s):
                            last_attempt_t = t_now
                            pick, df = self._anchor(frame, idx)
                            last_attempt_ok = pick is not None
                            if pick is not None:
                                anchors[idx] = pick
                                last_anchor_t, G_since, cut_since_anchor = t_now, np.eye(3), False
                            elif (
                                self.oracle is not None
                                and (cfg.landmarks_with_markings or not use_markings)
                                and df is not None
                                and n_landmark_calls < cfg.max_landmark_calls
                            ):
                                job = self.oracle.submit_landmarks(idx, frame)
                                if job is not None:
                                    n_landmark_calls += 1
                                    lm = line_mask(frame) > 0
                                    pending_lm[idx] = (job, np.packbits(lm), lm.shape)
                        if self.oracle is not None and t_now - last_kf_t >= cfg.keyframe_s:
                            job = self.oracle.submit_people(idx, frame)
                            if job is not None:
                                people_jobs[idx] = job
                                last_kf_t = t_now
                    if idx % people_stride == 0:
                        batch.append((idx, frame))
                        if len(batch) >= cfg.yolo_batch:
                            flush()
            flush()
            if ball is not None:
                ball.finish()
            release_gpu_cache()
            masks = _resolve(masks, T)
            mk_masks = _resolve(mk_masks, T)
            render_cache = _resolve(render_jobs, T)
            ask_more = None
            if by_coverage:
                mk_store = _resolve(mk_store, T)
                cands = [f for f, jpg in mk_store.items() if jpg is not None]
                paint = {f: jc.paint_fraction(mk_masks.get(f)) for f in cands}

                def ask_more(n: int) -> int:
                    """Ask the first `n` coverage keyframes per motion segment (the greedy
                    choice is prefix-stable: asking more only adds frames). Returns how many
                    keyframes have been asked in total."""
                    with T("calibration: keyframe choice"):
                        keys = jc.select_keyframes(
                            render_frames,
                            motion,
                            cands,
                            (fw, fh),
                            k=n,
                            paint=paint,
                            fps=fps,
                            min_paint_ratio=cfg.markings_paint_ratio,
                        )[: cfg.max_markings_calls]
                    for k in keys:
                        if k in mk_jobs:
                            continue
                        img = cv2.imdecode(mk_store[k], cv2.IMREAD_COLOR)
                        if img is not None and img.shape[1] != fw:
                            # stored small: back to full-frame pixels (answers, registration)
                            img = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_CUBIC)
                        job = None if img is None else self.oracle.submit_markings(k, img)
                        if job is None:
                            self._mk_short += 1  # over budget (or no image): not asked
                        else:
                            mk_jobs[k] = (job, None)
                            mk_images[k] = img
                    return len(mk_jobs)

                ask_more(cfg.markings_keys)

        # ---- pass 2: resolve oracle answers, calibration, labels, ball ------------------------
        with T("oracle: waiting for answers"):
            keyframes = {}
            for fidx, job in people_jobs.items():
                r = job.result()
                if r is not None:
                    keyframes[fidx] = r.people
            seeds: Dict[int, np.ndarray] = {}
            for fidx, (job, packed, shape) in pending_lm.items():
                r = job.result()
                if r is None:
                    continue
                from soccer_analyzer.vision_oracle import landmark_homography

                H = landmark_homography(r.landmarks, self.pitch)
                if H is None:
                    self.landmark_log.append(
                        dict(frame=fidx, n_landmarks=len(r.landmarks), ls=None, tag="no homography")
                    )
                    continue
                seeds[fidx] = H
                self._anchor_masks[fidx] = (packed, shape)
        markings, mk_info = None, {"asked": len(mk_jobs), "kept": []}
        mk_skipped = (
            field_markings_requested and not use_markings and mk_fails >= cfg.markings_fail_windows
        )
        if mk_skipped:
            mk_info["skipped"] = f"unconfirmed in {mk_fails} earlier windows of this video"
        if mk_jobs:
            with T("calibration: field markings"):
                if cfg.markings_method == "joint":
                    markings, mk_info = self._joint_calibration(
                        render_frames, motion, mk_images, mk_masks, mk_jobs, fps, more=ask_more
                    )
                    n_fails = _markings_memo(
                        mk_fails,
                        confirmed=markings is not None,
                        exhausted=bool(mk_info.get("exhausted")),
                        n_marked=sum(1 for m, _ in self._mk_answers.values() if m),
                        shortfall=self._mk_short + len(self._mk_failed),
                        max_keys=cfg.markings_max_keys,
                    )
                    memo_set = getattr(self.oracle, "memo_set", None)
                    if memo_set is not None and n_fails != mk_fails:
                        memo_set("markings_failed_windows", n_fails)
                    mk_info["failed_windows"] = n_fails
                    mk_info["calls_short"] = self._mk_short + len(self._mk_failed)
                else:
                    markings, mk_info = self._markings_calibration(
                        render_frames, motion, mk_images, mk_masks, mk_jobs, fps
                    )
        mk_images.clear()
        mk_masks.clear()
        mk_store.clear()
        pitch = self.pitch
        multisport = verify = pooled = None
        lm_used = 0
        if markings is not None:
            # Confirmed marking fits replace the field-detector path and the multi-sport
            # fallback (on the multi-sport clip that fallback was 6-11 m wrong).
            calib, source, pitch = markings
        else:
            anchors, pooled = self._pooled_refine(anchors, seeds, (fw, fh))
            lm_used = sum(1 for (_, _, t) in anchors.values() if t.startswith("oracle"))
            with T("calibration: propagate + verify"):
                calib, source, verify = self._propagate(
                    render_frames, motion, anchors, masks, (fw, fh), work_scale, detect_stride
                )
            n_cal = sum(h is not None for h in calib.values())
            evidence = self._marking_evidence()
            if evidence:
                # Markings were answered but no keyframe fit was confirmed. The field-detector
                # path may only show frames the markings agree with, and the multi-sport
                # fallback (6-11 m wrong on the multi-sport clip) is not run.
                with T("calibration: marking agreement check"):
                    calib, source, mk_info["old_path_check"] = self._check_against_markings(
                        calib, source, render_frames, motion, evidence, fps
                    )
            elif (
                cfg.multisport_fallback
                and not mk_skipped
                and start_frame == 0
                and n_cal < 0.10 * max(1, len(calib))
            ):
                # Multi-sport venues (football/lacrosse lines over the soccer pitch) defeat the
                # field CNN; fall back to the colour-agnostic calibrator, as the classic app does.
                # Not when markings were skipped by the per-video memo: that video's markings
                # were answered and never confirmed, which counts as marking evidence.
                calib, source, multisport = self._multisport(video_path, total, calib, source)
        with T("players: tracking"):
            drawn, tracks = track_people(
                render_frames, raw_dets, motion, keyframes, app_feats, fps, cfg.tracking
            )
            md.people = {
                f: [(tid, b, ((b[0] + b[2]) / 2.0, b[3])) for tid, b, _ in drawn[f]]
                for f in render_frames
            }
            # Preserve detector versus estimated provenance at the workspace boundary.
            md.interpolated = {
                (f, tid) for f in render_frames for tid, _, source in drawn[f] if source == "interp"
            }
            md.frames = list(render_frames)
            box_sources = Counter(s for f in render_frames for _, _, s in drawn[f])
            colour: Dict[int, list] = defaultdict(list)
            for t in tracks:
                for f, j in t.det_idx.items():
                    if j >= 0 and j in jersey.get(f, {}):
                        colour[t.tid].append(jersey[f][j])
        with T("labels: teams / roles / on-field"):
            kind, team, offfield, names, vis, ai_info = self._labels(
                md,
                keyframes,
                colour,
                calib,
                pitch,
                fps=fps,
                # Tiles from any render frame (render-size JPEGs, as the lab measured), else
                # the ~1 Hz full-resolution oracle frames.
                **(
                    dict(oracle_frames=render_cache, frame_scale=_render_size(fw, fh)[0])
                    if render_cache and all(f in render_cache for f in render_frames)
                    else dict(oracle_frames=ball.jpgs if ball is not None else None)
                ),
            )
            md.kind, md.team = kind, team
        ball_info: dict = {}
        if ball is not None:
            with T("ball: tracker"):
                md.ball, ball_info = self._track_ball(
                    ball, render_frames, motion, raw_dets, fps / render_stride, (fw, fh)
                )

        n_render = len(render_frames)
        n_cal = sum(h is not None for h in calib.values())
        srcs = Counter(t for (_, _, t) in anchors.values())
        stats = dict(
            raw_frames=n_raw,
            seconds_of_video=round(n_raw / fps, 2),
            render_frames=n_render,
            detect_frames=len(raw_dets),
            people_box_sources=dict(box_sources),
            calibrated_render_frames=n_cal,
            calibrated_pct=round(100.0 * n_cal / max(1, n_render), 1),
            anchors=len(anchors),
            anchor_sources=dict(srcs),
            oracle_landmark_calls=len(pending_lm),
            oracle_landmark_anchors=lm_used,
            oracle_people_keyframes=len(keyframes),
            oracle_usd=round(self.oracle.spent_usd, 4) if self.oracle is not None else 0.0,
            tracks=len({p[0] for ps in md.people.values() for p in ps if p[0] >= 0}),
            offfield_tracks=len(offfield),
            onfield=dict(
                Counter(vis.status.values()),
                relinked=len(vis.links),
                ai=ai_info,
            ),
            referees=sum(k == "referee" for k in kind.values()),
            goalkeepers=sum(k == "goalkeeper" for k in kind.values()),
            ball_path_frames=len(md.ball),
            ball=ball_info,
            field_markings=mk_info,
            pitch_dims=[round(pitch.length, 2), round(pitch.width, 2)],
            verify=verify,
            multisport=multisport,
            field_clahe=self._field_enh,
            anchor_attempts=self.anchor_log,
            landmark_seeds=self.landmark_log,
            pooled_camera_position=None if pooled is None else [round(float(v), 2) for v in pooled],
        )
        return FastResult(
            fps=fps,
            frame_size=(fw, fh),
            render_frames=render_frames,
            detect_frames=sorted(raw_dets),
            calib=calib,
            calib_source=source,
            md=md,
            offfield=offfield,
            team_names=names,
            keyframes=keyframes,
            stats=stats,
            pitch=pitch,
            render_cache=render_cache,
            visible=vis.visible,
            pitch_xy=vis.pitch_xy,
            onfield_status=vis.status,
            motion=motion,
        )

    def _collect_answers(self, jobs, images):
        """Marking answers of the keyframes: {frame: (marks, soccer line colour)}."""
        answers = {}
        for k, pair in sorted(jobs.items()):
            job, alt = pair if isinstance(pair, tuple) else (pair, None)
            r, ra = job.result(), None if alt is None else alt.result()
            if r is None and ra is None:
                self.__dict__.setdefault("_mk_failed", set()).add(k)  # error / timeout
            if ra is not None and (r is None or len(ra.markings) > len(r.markings)):
                r = ra  # the second opinion names more markings
            if r is not None and k in images:
                answers[k] = (list(r.markings), r.line_colour or "white")
        self._mk_answers = answers
        return answers

    def _joint_calibration(self, render_frames, motion, images, masks, jobs, fps, more=None):
        """Shared-camera-centre calibration of the window (`joint_calib.calibrate_window_joint`).

        `more(n)` (coverage keyframes) asks the first n keyframes per segment: when no frame is
        confirmed, the number per segment doubles, up to `markings_max_keys` -- easy fields stop
        at the first few, fields with few soccer markings per view get more evidence; when a
        confirmed field leaves >= `markings_unseen_frac` of the frames outside every keyframe's
        view, one more keyframe per segment is asked. The LAST attempt (the most evidence) decides:
        an attempt with less evidence is never preferred, and a frame that an earlier confirmed
        attempt also showed stays shown only if both land within joint_tol_m (median of the view;
        2 x joint_tol_m on its worst tenth). info["exhausted"] says the whole budget ended
        unconfirmed. Returns
        ((calib, source, pitch) or None, info). None when fewer than two keyframes were answered
        (the split-half confirmation needs two) or no frame was confirmed; frames that fail
        confirmation, registration or paint verification stay None (unavailable)."""
        cfg = self.cfg
        info = dict(method="joint", attempts=[])
        n_keys = cfg.markings_keys
        prev = None  # per-frame results of the latest attempt that confirmed frames
        exhausted = False
        while True:
            per, res, step = self._joint_attempt(render_frames, motion, images, masks, jobs, fps)
            step["keys_per_segment"] = n_keys
            info["attempts"].append(step)
            ok = res is not None and any(p.get("H_img2pitch") is not None for p in per.values())
            if ok and prev is not None:
                # more evidence after a confirmed attempt: the frames both confirm must agree
                fw, fh = next(iter(images.values())).shape[1::-1]
                step["attempts_disagree"] = jc.withhold_disagreeing(
                    per, prev, fw, fh, cfg.joint_tol_m, reason="attempts disagree", require=False
                )
                step["withheld"] = dict(
                    Counter(p["reason"] for p in per.values() if p.get("H_img2pitch") is None)
                )
                ok = any(p.get("H_img2pitch") is not None for p in per.values())
                step["coverage"] = round(
                    float(np.mean([p.get("H_img2pitch") is not None for p in per.values()])), 3
                )
            if ok:
                prev = per
            unseen = step.get("withheld", {}).get("outside keyframe views", 0) / max(
                1, len(render_frames)
            )
            if (ok and unseen < cfg.markings_unseen_frac) or more is None:
                break
            if n_keys >= cfg.markings_max_keys:
                exhausted = not ok
                break
            before = len(jobs)
            # unconfirmed: double the evidence; confirmed but views left unseen: one more
            # keyframe per segment (the greedy choice adds the next-farthest view)
            n_keys = min(cfg.markings_max_keys, n_keys + 1 if ok else 2 * n_keys)
            if more(n_keys) <= before:
                exhausted = not ok
                break
        info.update({k: v for k, v in step.items() if k != "keys_per_segment"})
        info["asked"] = len(jobs)
        info["exhausted"] = bool(exhausted and not ok)
        if not ok:
            return None, info
        L, W = res["pitch_dims"]
        pitch = PitchModel(length=float(L), width=float(W))
        calib: Dict[int, Optional[np.ndarray]] = {}
        source: Dict[int, str] = {}
        for f in render_frames:
            p = per.get(f)
            if p is None or p.get("H_img2pitch") is None:
                calib[f], source[f] = None, "none"
                continue
            Lk, Wk = p["pitch_dims"]
            shift = np.array([[1.0, 0.0, (L - Lk) / 2], [0.0, 1.0, (W - Wk) / 2], [0.0, 0.0, 1.0]])
            calib[f] = shift @ np.asarray(p["H_img2pitch"], float)
            source[f] = "vlm markings (joint)"
        info["calibrated_frames"] = sum(h is not None for h in calib.values())
        return (calib, source, pitch), info

    def _joint_attempt(self, render_frames, motion, images, masks, jobs, fps):
        """One joint fit with the keyframes answered so far: (per, res or None, info)."""
        cfg = self.cfg
        answers = self._collect_answers(jobs, images)
        info: dict = dict(answered=len(answers), kept=sorted(answers))
        if len(answers) < 2:
            return {}, None, info
        cols = [str(c).lower() for _, c in answers.values() if c]
        colour = jc.majority_colour(cols)
        # ANY non-white answer makes football mode a candidate: the cheap yard-number read
        # decides (a 1-1 tie between white and yellow used to switch it off alphabetically)
        football = cfg.markings_football == "on" or (
            cfg.markings_football == "auto" and any(c != "white" for c in cols)
        )
        numbers = dict(getattr(self, "_numbers", None) or {})
        if football and hasattr(self.oracle, "submit_numbers"):
            # two reads in all (one per consensus half on the first attempt); a later attempt
            # with more keyframes does not ask again
            have = [k for k in numbers if k in answers]
            asked = {
                k: self.oracle.submit_numbers(k, images[k])
                for k in [k for k in sorted(answers) if k not in numbers][: max(0, 2 - len(have))]
            }
            for k, job in asked.items():
                r = None if job is None else job.result()
                if r is None:
                    # over budget or failed: football mode undecided
                    self._mk_short = getattr(self, "_mk_short", 0) + 1
                numbers[k] = [] if r is None else list(r.numbers)
            self._numbers = numbers
            if cfg.markings_football == "auto":
                football = jc.football_numbers_found(numbers)
        elif cfg.markings_football == "auto":
            football = False
        use_numbers = {k: v for k, v in numbers.items() if k in answers} if football else None
        if football and colour == "white":
            # football lines are the white ones: the soccer lines are the non-white answers
            colour = jc.majority_colour(
                [c for c in cols if c != "white" and c in fm.MASK_COLOURS] or ["yellow"]
            )
        fw, fh = next(iter(images.values())).shape[1::-1]
        per, res = jc.calibrate_window_joint(
            render_frames,
            motion,
            {k: images[k] for k in answers},
            masks,
            answers,
            fps=fps,
            frame_size=(fw, fh),
            consensus="paint",
            tol_m=cfg.joint_tol_m,
            min_seen=cfg.joint_min_seen,
            stability=cfg.joint_stability_runs,
            football_l_sigma=cfg.joint_football_l_sigma,
            colour=colour,
            football=football,
            numbers=use_numbers,
        )
        segs = {}
        for sid, si in res["segments"].items():
            segs[int(sid)] = dict(
                keys=[int(k) for k in si.get("keys", [])],
                status=si.get("status", "ok"),
                camera_centre=si.get("C"),
                focal_range=si.get("f_range"),
                halves=[
                    dict(keys=[int(k) for k in hv["keys"]], status=hv["status"], C=hv["C"])
                    for hv in si.get("halves", [])
                ],
            )
        info.update(
            colour=colour,
            football=bool(football),
            numbers={int(k): len(v) for k, v in (use_numbers or {}).items()},
            pitch_dims=None
            if res["pitch_dims"] is None
            else [round(float(v), 2) for v in res["pitch_dims"]],
            segments=segs,
            seconds=round(float(res["secs"]), 1),
            stability=res.get("stability"),
            coverage=round(float(res["coverage"]), 3),
            withheld=dict(
                Counter(p["reason"] for p in per.values() if p.get("H_img2pitch") is None)
            ),
        )
        if res["pitch_dims"] is None:
            return per, None, info
        return per, res, info

    def _markings_calibration(self, render_frames, motion, images, masks, jobs, fps):
        """VLM field-marking calibration of the window (`field_markings.calibrate_window`).

        Returns ((calib, source, pitch) or None, info). None when fewer than two keyframes were
        answered or no keyframe fit was confirmed by another one; frames that fail verification
        stay None. Each keyframe fit has its own fitted pitch dims: frames are re-expressed in
        the window's median dims about the centre spot (shift of half the dims difference)."""
        answers = self._collect_answers(jobs, images)
        info = dict(method="window", asked=len(jobs), answered=len(answers), kept=[])
        if len(answers) < 2:  # consensus needs two independent answers
            return None, info
        cfg = self.cfg
        per, res = fm.calibrate_window(
            render_frames,
            motion,
            {k: images[k] for k in answers},
            masks,
            answers=answers,
            fps=fps,
            window_s=cfg.markings_window_s,
            tol_m=cfg.markings_tol_m,
            ratio=cfg.markings_ratio,
            min_ls=cfg.markings_min_ls,
        )
        info.update(
            kept=list(res["kept"]),
            colour=res["colour"],
            pitch_dims=None
            if res["pitch_dims"] is None
            else [round(v, 2) for v in res["pitch_dims"]],
            consensus={int(k): v for k, v in res["consensus"].items()},
        )
        if not res["kept"]:
            return None, info
        L, W = res["pitch_dims"]
        pitch = PitchModel(length=float(L), width=float(W))
        calib: Dict[int, Optional[np.ndarray]] = {}
        source: Dict[int, str] = {}
        for f in render_frames:
            p = per.get(f)
            if p is None or p["H_img2pitch"] is None:
                calib[f], source[f] = None, "none"
                continue
            Lk, Wk = p["pitch_dims"]
            shift = np.array([[1.0, 0.0, (L - Lk) / 2], [0.0, 1.0, (W - Wk) / 2], [0.0, 0.0, 1.0]])
            calib[f] = shift @ np.asarray(p["H_img2pitch"], float)
            source[f] = f"vlm markings (key {p['key']})"
        info["calibrated_frames"] = sum(h is not None for h in calib.values())
        return (calib, source, pitch), info

    def _marking_evidence(self) -> Dict[int, list]:
        """Answered marking keyframes with >= 4 confident ground markings: {frame: marks}."""
        out = {}
        for k, (marks, _) in (getattr(self, "_mk_answers", None) or {}).items():
            ev = fm.ground_evidence(marks)
            if len(ev) >= 4:
                out[k] = ev
        return out

    def _check_against_markings(self, calib, source, render_frames, motion, evidence, fps):
        """Field-detector frames must agree with the marking evidence (`fm.marking_error_m`
        through the camera motion from the nearest evidence keyframe in the same segment within
        markings_window_s); frames without such a keyframe, or that disagree, become None."""
        cfg = self.cfg
        chain = fm.MotionChain(render_frames, motion)
        dims = (self.pitch.length, self.pitch.width)
        calib, source = dict(calib), dict(source)
        kept = rejected = 0
        errs = []
        for f, H in calib.items():
            if H is None:
                continue
            near = sorted(
                (abs(k - f), k)
                for k in evidence
                if abs(k - f) <= cfg.markings_window_s * fps and chain.transfer(k, f) is not None
            )
            err = (
                fm.marking_error_m(H, dims, evidence[near[0][1]], chain.transfer(near[0][1], f))
                if near
                else np.inf
            )
            if err <= cfg.markings_check_tol_m:
                kept += 1
                errs.append(err)
            else:
                calib[f], source[f] = None, "none (disagrees with the field markings)"
                rejected += 1
        info = dict(
            kept=kept,
            rejected=rejected,
            median_err_m=None if not errs else round(float(np.median(errs)), 2),
        )
        return calib, source, info

    def _track_ball(self, sweep: _BallSweep, render_frames, motion, raw_dets, render_fps, size):
        """Ball track over the window (`ball_track`): merge + score the pass-1 candidates, then
        the global DP, with oracle crop checks when available. -> ({frame: (x, y, conf)}, info);
        frames without a trusted ball are absent."""
        cfg = self.cfg
        fw, _ = size
        R = 8.0 * fw / 1920.0
        dframes = np.array(sorted(raw_dets))
        boxes: Dict[int, np.ndarray] = {}
        px_per_m = []
        for f in render_frames:
            b = np.zeros((0, 4))
            if len(dframes):
                xyxy, conf = raw_dets[int(dframes[np.argmin(np.abs(dframes - f))])]
                b = xyxy[conf >= 0.3]
            boxes[f] = b
            h = b[:, 3] - b[:, 1]
            px_per_m.append(float(np.median(h)) / 1.75 if len(h) >= 3 else np.nan)
        scale = np.array(px_per_m)
        ok = np.isfinite(scale)
        scale[~ok] = np.median(scale[ok]) if ok.any() else 30.0 * fw / 1920.0
        overlay = bt.static_overlay_mask(sweep.grays, size) if sweep.grays else None
        empty = np.zeros((0, 9), np.float32)
        motion_c = {
            f: bt.motion_candidates_from_blobs(sweep.blobs.get(f, empty), boxes[f], overlay)
            for f in render_frames
        }
        feats = bt.merge_candidates(
            render_frames, {"wasb": sweep.wasb_c, "motion": motion_c}, boxes, R
        )
        cands = bt.score_candidates(feats)
        G = [None] + [motion.get(f) for f in render_frames[1:]]
        use_vlm = cfg.ball_vlm and self.oracle is not None and hasattr(self.oracle, "submit_ball")
        anchors, n_q = {}, 0
        if use_vlm:

            def get_frame(f):
                jpg = sweep.jpgs.get(f)
                return None if jpg is None else cv2.imdecode(jpg, cv2.IMREAD_COLOR)

            def ask(questions):
                jobs = [
                    self.oracle.submit_ball(f, img, tag="@" + ",".join(map(str, box)))
                    for f, img, box in questions
                ]  # submitted together: the answers arrive in parallel
                out = []
                for job in jobs:
                    r = None if job is None else job.result()
                    out.append(None if r is None else {"ball": r.ball})
                return out

            track, anchors, n_q = bt.track_with_anchors(
                render_frames,
                cands,
                G,
                get_frame,
                ask,
                fps=render_fps,
                scale=scale,
                params={"min_mean_p": cfg.ball_gate},
                radius_px=R,
                every=cfg.ball_vlm_every,
                max_queries=cfg.ball_vlm_max,
            )
        else:
            track = bt.track_ball2(
                render_frames,
                cands,
                G,
                fps=render_fps,
                scale=scale,
                params={"min_mean_p": cfg.ball_gate_offline},
                radius_px=R,
            )
        gate = cfg.ball_gate if use_vlm else cfg.ball_gate_offline
        # Only points at or above the gate are reported. Frames a segment bridges by
        # interpolation carry conf x interp_conf and are dropped when that falls below the gate.
        out = {f: v for f, v in track.items() if v is not None and v[2] >= gate}
        info = dict(
            frames_with_ball=len(out),
            candidate_frames=sum(1 for c in cands.values() if len(c)),
            oracle_checks=n_q,
            oracle_confirmed=sum(1 for a in anchors.values() if a["xy"] is not None),
            night_frames=sweep.night_frames,
            duplicate_raw_frames=sweep.duplicates,
            overlay_px=0 if overlay is None else int(overlay.sum()),
            gate=gate,
        )
        return out, info

    def _anchor(self, frame: np.ndarray, idx: int):
        """Field-detector anchor attempt -> ((H, ls, tag) or None, native distance field).

        Until the first verified anchor, the CNN runs on both the raw and the CLAHE-enhanced
        frame (brightness cannot tell night from day: letterboxing and dark turf put a daytime
        clip below a floodlit night one); the input mode that produced that anchor is then kept
        for the rest of the clip."""
        T, cfg = self.timer, self.cfg
        h, w = frame.shape[:2]
        with T("calibration: line field + gates"):
            mask = line_mask(frame)
            if (mask > 0).sum() < 50:
                return None, None
            df = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        modes = [self._field_enh] if self._field_enh is not None else [False, True]
        best = None
        for mode in modes:
            if hasattr(self.field, "enhance"):
                self.field.enhance = mode
            release_gpu_cache()  # the ball model's cached blocks
            with T("calibration: field CNN"):
                try:
                    obs = self.field.detect(frame, idx)
                except Exception:  # the detector's solver can fail on degenerate keypoints
                    obs = None
                release_gpu_cache()
            pick = None
            if obs is not None:
                with T("calibration: line field + gates"):
                    pick = select_calibration(
                        detector_candidates(obs, self.pitch),
                        df,
                        (w, h),
                        self.pitch,
                        ls_floor=cfg.ls_floor,
                    )
            self.anchor_log.append(
                dict(
                    frame=idx,
                    clahe=mode,
                    ls=None if pick is None else round(pick[1], 3),
                    tag=None if pick is None else pick[2],
                )
            )
            if pick is not None:
                # same ranking as select_calibration: camera beside the field, then support
                rank = (not camera_inside_pitch(pick[0], (w, h), *self._dims), pick[1])
                if best is None or rank > best[0]:
                    best = (rank, pick, mode)
        if best is None:
            return None, df
        if self._field_enh is None:
            self._field_enh = best[2]
        self._anchor_masks[idx] = (np.packbits(mask > 0), mask.shape)
        return best[1], df

    def _multisport(self, video_path, total, calib, source):
        """Colour-agnostic multi-sport calibration over the analysed window; each render frame
        takes the nearest calibrated sample within 3 frames (never across a rejected sample)."""
        from soccer_analyzer.calibration.lookup import CalibrationLookup
        from soccer_analyzer.calibration.multisport import calibrate_multisport

        with self.timer("calibration: multi-sport fallback"):
            res, st = calibrate_multisport(video_path, self.pitch, max_frames=total or None)
        if not res:
            return calib, source, {"used": False, "reason": "not multi-sport"}
        look = CalibrationLookup(res)
        out, src = dict(calib), dict(source)
        for f in calib:
            H = look.at(f)
            if H is not None:
                out[f], src[f] = H, "multi-sport (" + str(st.get("winning_colour")) + " lines)"
        used = sum(1 for f in out if src[f].startswith("multi-sport"))
        return out, src, {"used": True, "colour": st.get("winning_colour"), "frames": used}

    def _pooled_refine(self, anchors, seeds, size):
        """Refine every anchor with one camera position pooled over the clip.

        A stand- or pole-mounted camera pans and zooms but does not move, while single-frame
        fits on night footage scatter its position by metres (and a wrong position tilts lines
        receding towards the camera). The median position of the field-detector anchors that place
        the camera beside the field is held fixed and each anchor's rotation + focal
        re-optimised. A pooled refit replaces an anchor only if it passes the same gates with at
        least 90% of the anchor's line support. An oracle seed (rough by nature) is accepted only
        as a refit against the pooled position, so seeds supplement detector anchors but never
        calibrate a clip on their own.

        anchors: {frame: (H, ls, tag)}; seeds: {frame: H}. Both need `self._anchor_masks`.
        Returns (anchors, pooled position or None)."""
        from soccer_analyzer.calibration.homography import decompose_camera
        from soccer_analyzer.calibration.ls_calibrator import refine_homography

        cfg, T = self.cfg, self.timer
        L, W = self.pitch.length, self.pitch.width

        def dist_field(f):
            packed, shape = self._anchor_masks[f]
            m = np.unpackbits(packed)[: shape[0] * shape[1]].reshape(shape)
            return cv2.distanceTransform((1 - m).astype(np.uint8) * 255, cv2.DIST_L2, 3)

        # Only field-detector anchors define the pooled position: seeds are too rough, and a
        # position pooled from seeds alone let a refit overfit extra lines on a multi-line field.
        positions = []
        for H, _, _ in anchors.values():
            cam = decompose_camera(H, size, L, W)
            if cam is not None and not camera_inside_pitch(H, size, L, W):
                positions.append(cam["position_meters"])
        pooled = np.median(np.array(positions), axis=0) if positions else None
        out = dict(anchors)
        with T("calibration: pooled-position refine"):
            for f, (H, ls, tag) in anchors.items():
                if pooled is None or f not in self._anchor_masks:
                    continue
                df = dist_field(f)
                Hr, _ = refine_homography(H, df, size, self.pitch, position=pooled)
                if Hr is None:
                    continue
                sel = select_calibration(
                    [(tag + "+pooled", Hr)], df, size, self.pitch, ls_floor=cfg.ls_floor
                )
                if sel is not None and sel[1] >= 0.9 * ls:
                    out[f] = sel
            for f, H in seeds.items():
                if pooled is None:
                    self.landmark_log.append(
                        dict(frame=f, ls=None, tag="no detector anchor to pool")
                    )
                    continue
                df = dist_field(f)
                cands = [("oracle", H)]
                if line_support(H, df, self.pitch, radius=8.0) < 0.25:  # hopeless: skip refits
                    self.landmark_log.append(dict(frame=f, ls=None, tag="seed off the lines"))
                    continue
                Hr, _ = refine_homography(H, df, size, self.pitch, position=pooled)
                if Hr is not None:
                    cands.append(("oracle+pooled", Hr))
                picks = [
                    select_calibration([c], df, size, self.pitch, ls_floor=cfg.ls_floor)
                    for c in cands
                ]
                # A seed's own camera position is rough: refitting around it tilted the halfway
                # line at night. A seed alone is not trusted either: on a field with extra
                # (lacrosse) lines a wrong seed scored 0.63 line support. So seeds only count as
                # refits against the position pooled from field-detector anchors.
                best = next((p for p in picks if p and p[2] == "oracle+pooled"), None)
                self.landmark_log.append(
                    dict(
                        frame=f,
                        ls=None if best is None else round(best[1], 3),
                        tag=None if best is None else best[2],
                    )
                )
                if best is not None:
                    out[f] = best
        return out, pooled

    def _propagate(self, frames, motion, anchors, masks, size, work_scale, detect_stride):
        """Carry anchors along the chained camera motion (both directions, within a segment) and
        verify detection frames against their own line masks. Returns (H per render frame,
        source label, verification stats). Unverified frames are None."""
        cfg = self.cfg
        n = len(frames)
        pos = {f: i for i, f in enumerate(frames)}
        # cumulative transform to the start of each segment: C[i] maps frame i -> segment origin
        seg = [0] * n
        C = [np.eye(3)] * n
        for i in range(1, n):
            G = motion.get(frames[i])
            if G is None:
                seg[i], C[i] = seg[i - 1] + 1, np.eye(3)
            else:
                seg[i] = seg[i - 1]
                C[i] = C[i - 1] @ np.linalg.inv(G)
        S = np.diag([work_scale, work_scale, 1.0])
        Sinv = np.linalg.inv(S)
        dfs: Dict[int, np.ndarray] = {}

        def work_ls(H, f):
            if f not in dfs:
                packed, shape = masks[f]
                m = np.unpackbits(packed)[: shape[0] * shape[1]].reshape(shape)
                dfs[f] = cv2.distanceTransform((1 - m).astype(np.uint8) * 255, cv2.DIST_L2, 3)
            radius = max(3.0, 8.0 * work_scale)
            return line_support(H @ Sinv, dfs[f], self.pitch, radius=radius)

        anchor_ref = {}
        for a, (H, _, _) in anchors.items():
            if a in pos and a in masks:
                anchor_ref[a] = max(1e-6, work_ls(H, a))
        apos = sorted(pos[a] for a in anchors if a in pos)
        out: Dict[int, Optional[np.ndarray]] = {}
        src: Dict[int, str] = {}
        ver = {"checked": 0, "passed": 0}
        max_gap = int(round(3 * cfg.max_anchor_gap_s * cfg.render_hz))
        for i, f in enumerate(frames):
            if f in anchors:
                out[f], src[f] = anchors[f][0], "anchor (" + anchors[f][2] + ")"
                continue
            cands = []
            for j in apos:
                if seg[j] == seg[i] and abs(j - i) <= max_gap:
                    a = frames[j]
                    # pixel in frame i -> segment origin -> pixel in anchor frame -> pitch
                    H = anchors[a][0] @ np.linalg.inv(C[j]) @ C[i]
                    cands.append((abs(j - i), a, H / np.linalg.norm(H)))
            cands.sort(key=lambda c: c[0])
            out[f], src[f] = None, "none"
            if not cands:
                continue
            if f in masks:
                # Verification frame: the NEAREST anchor whose propagation still verifies. Picking
                # the highest line support instead would let a farther, drifted fit win, because
                # line support rewards markings landing on any line, not the right one.
                ver["checked"] += 1
                for _, a, H in cands[:2]:
                    ls = work_ls(H, f)
                    if ls >= max(cfg.verify_min, cfg.verify_ratio * anchor_ref.get(a, ls)):
                        ver["passed"] += 1
                        out[f], src[f] = H, f"propagated from {a}"
                        break
            else:  # validated against the neighbouring verification frames below
                out[f], src[f] = cands[0][2], f"propagated from {cands[0][1]}"
        # a non-verification frame is shown only if its neighbouring verification frames passed
        for i, f in enumerate(frames):
            if f in masks or out[f] is None:
                continue
            nb = [
                g for g in frames[max(0, i - self.cfg.detect_every) : i + self.cfg.detect_every + 1]
            ]
            checks = [out[g] is not None for g in nb if g in masks]
            if not checks or not all(checks):
                out[f], src[f] = None, "none"
        return out, src, ver

    def _labels(
        self,
        md: MatchData,
        keyframes,
        colour,
        calib,
        pitch: Optional[PitchModel] = None,
        fps: float = 30.0,
        oracle_frames: Optional[Mapping[int, np.ndarray]] = None,
        frame_scale: float = 1.0,
    ):
        """Roles/teams/visibility per track: oracle votes first, then who is drawn (`onfield`:
        on the pitch at some moment within +-window_s on calibrated frames; for tracks without
        any calibrated frame, the oracle off-field vote and the Decisions API answer), then jersey
        colours learnt from the oracle-labelled tracks (or a colour K=2 fit without an oracle).
        Hidden tracks are left out of the colour fit and not drawn.
        Returns (kind, team, hidden tids, team names, OnFieldResult, AI info)."""
        kind, team, oracle_off, names = ({}, {}, set(), {})
        if keyframes:
            kind, team, oracle_off, names = vote_labels(keyframes, md.people)
        tids = {p[0] for ps in md.people.values() for p in ps if p[0] >= 0}
        pitch = pitch or self.pitch
        L, W = pitch.length, pitch.width
        ai_probs: Dict[int, float] = {}
        ai_info: dict = {}
        decide = getattr(self.oracle, "decide", None)
        if self.cfg.onfield_ai and decide is not None and oracle_frames:
            # Only when some track's status can change, but then about EVERY track: the answers
            # depend on the other tiles of a mosaic (lab, union: asking only the uncertain
            # tracks pushed real players' p from 0.15-0.83 down to 0.05-0.13 and hid them).
            if ai_candidates(md.people, calib, kind, oracle_off):
                with self.timer("labels: Decisions API (tracks)"):
                    ai_probs, ai_info = ask_participants(
                        decide,
                        _LazyJpegs(oracle_frames),
                        md.people,
                        tids,
                        k=self.cfg.onfield_ai_k,
                        box_scale=frame_scale,
                    )
        vis = onfield_visibility(
            md.people,
            calib,
            fps,
            (L, W),
            self.cfg.onfield,
            kind=kind,
            track_prob=ai_probs,
            legacy_offfield=oracle_off,
        )
        offfield = vis.hidden
        mean_lab = {
            t: np.median(np.array(c), axis=0) * _CHROMA_WEIGHT
            for t, c in colour.items()
            if c and t not in offfield
        }
        centres = None
        labelled = [
            t for t in mean_lab if team.get(t) in (0, 1) and kind.get(t, "player") == "player"
        ]
        if {team[t] for t in labelled} == {0, 1}:
            centres = np.array(
                [np.median([mean_lab[t] for t in labelled if team[t] == k], axis=0) for k in (0, 1)]
            )
        elif not keyframes and len(mean_lab) >= 4:
            centres = two_means(np.array(list(mean_lab.values())))
        if centres is not None and np.all(np.isfinite(centres)):
            sep = float(np.linalg.norm(centres[0] - centres[1]))
            for t, lab in mean_lab.items():
                if t in team or kind.get(t) in ("referee", "goalkeeper"):
                    continue
                d = np.linalg.norm(centres - lab, axis=1)
                if d.min() <= 0.75 * sep:
                    team[t] = int(np.argmin(d))
        for t in tids:
            team.setdefault(t, -1)
        return kind, team, offfield, names, vis, ai_info

    # ---- pass 3: render -----------------------------------------------------------------------

    @contextmanager
    def _render_frames(self, video_path: str, result: FastResult, size: Tuple[int, int]):
        """(index, frame at the render size) for every render frame: the pass-1 JPEGs when they
        cover the window, else a second sequential decode (seeking lands on wrong frames)."""
        T, frames, cache = self.timer, result.render_frames, result.render_cache
        if cache and all(f in cache for f in frames):

            def from_cache():
                for f in frames:
                    with T("render: decode"):
                        img = cv2.imdecode(cache[f], cv2.IMREAD_COLOR)
                    yield f, img

            yield from_cache()
            return
        with open_capture(video_path) as cap:

            def from_video():
                idx = skip_frames(cap, frames[0])
                last, fset = frames[-1], set(frames)
                while idx <= last:
                    with T("render: decode"):
                        if not cap.grab():
                            return
                        if idx not in fset:
                            idx += 1
                            continue
                        ok, frame = cap.retrieve()
                        if not ok:
                            return
                    yield idx, cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                    idx += 1

            yield from_video()

    def render(self, video_path: str, result: FastResult, out_path: str) -> dict:
        """Side-by-side video (broadcast overlay | 2D map) at the render rate.

        People drawn are the tracker's boxes on each rendered frame (detections, oracle seeds and
        camera-compensated gap fills; see `player_tracker`) that `onfield` shows on the frame: a
        throw-in taker just off the pitch is drawn on the 2D map beside the line (up to ~5 m)."""
        cfg, T, md = self.cfg, self.timer, result.md
        pitch = result.pitch or self.pitch
        fw, fh = result.frame_size
        frames = result.render_frames
        if not frames:
            raise ValueError("nothing to render")
        render_fps = result.fps / max(1, frames[1] - frames[0]) if len(frames) > 1 else result.fps

        def code(tid):
            k = md.kind.get(tid, "player")
            return -2 if k == "referee" else -3 if k == "goalkeeper" else md.team.get(tid, -1)

        legend = (
            "  ".join(f"team {'AB'[k]}: {n}" for k, n in sorted(result.team_names.items()))
            or "teams: colour clusters"
        )
        cov = CoverageMap(pitch)
        scale, sw, sh = _render_size(fw, fh)
        S = np.diag([scale, scale, 1.0])
        writer = None
        n_out = 0
        with self._render_frames(video_path, result, (sw, sh)) as smalls:
            for idx, small in smalls:
                if result.visible is not None:
                    shown = result.visible.get(idx, set())
                    tracked = [(tid, b) for tid, b, _ in md.people.get(idx, []) if tid in shown]
                else:
                    tracked = [
                        (tid, b)
                        for tid, b, _ in md.people.get(idx, [])
                        if tid not in result.offfield
                    ]
                with T("render: draw"):
                    H = result.calib.get(idx)
                    good = H is not None
                    Hs = H @ np.linalg.inv(S) if good else np.eye(3)
                    label = result.calib_source.get(idx, "none")
                    calib = FrameCalibration(
                        idx, Hs, 1.0 if good else 0.0, label if good else "none", 0, 0
                    )
                    overlay = draw_calibration_overlay(
                        small, calib, pitch, fade_by_confidence=False
                    )
                    players_pitch = []
                    people = [(b, code(tid)) for tid, b in tracked]
                    for (x1, y1, x2, y2), c in people:
                        cv2.rectangle(
                            overlay,
                            (int(x1 * scale), int(y1 * scale)),
                            (int(x2 * scale), int(y2 * scale)),
                            TEAM_BGR.get(c, TEAM_BGR[-1]),
                            2,
                        )
                        if good:
                            p = cv2.perspectiveTransform(
                                np.array([[[(x1 + x2) / 2, y2]]], np.float32), H
                            )[0, 0]
                            if -5 <= p[0] <= pitch.length + 5 and -5 <= p[1] <= pitch.width + 5:
                                players_pitch.append((float(p[0]), float(p[1]), c))
                    b = md.ball.get(idx)
                    ball_pitch = None
                    if b is not None:
                        c = (int(b[0] * scale), int(b[1] * scale))
                        # solid: a confident segment; hollow: a bridged or weak one
                        solid = len(b) < 3 or b[2] >= cfg.ball_gate
                        cv2.circle(overlay, c, 9, (255, 255, 255), -1 if solid else 2)
                        cv2.circle(overlay, c, 10 if solid else 12, (0, 0, 0), 2)
                        if good:
                            p = cv2.perspectiveTransform(np.array([[b[:2]]], np.float32), H)[0, 0]
                            if 0 <= p[0] <= pitch.length and 0 <= p[1] <= pitch.width:
                                ball_pitch = (float(p[0]), float(p[1]))
                    if good:
                        cov.add(H, (fw, fh))
                    cv2.putText(
                        overlay,
                        f"{legend}   calibration: {label}",
                        (12, sh - 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    top = draw_topdown(
                        pitch,
                        calib if good else None,
                        players_pitch,
                        ball_pitch,
                        cov.normalized(),
                        frame_size=(sw, sh),
                    )
                    canvas = compose(overlay, top, out_width=cfg.out_width)
                with T("render: encode"):
                    if writer is None:
                        writer = open_writer(
                            out_path, render_fps, (canvas.shape[1], canvas.shape[0])
                        )
                    writer.write(canvas)
                n_out += 1
        if writer is not None:
            writer.release()
        return {
            "rendered_frames": n_out,
            "render_fps": round(render_fps, 2),
            "out_path": out_path,
        }
