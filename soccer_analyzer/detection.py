"""Soccer-specific detection + tracking — players, goalkeepers, referees, and the ball.

Local, no per-frame API cost. Backend is Roboflow's `football-players-detection` model run
LOCALLY via the `inference` package (weights download once, then offline), which natively
distinguishes player / goalkeeper / referee / ball. Falls back to a COCO YOLO (person + ball)
if that model can't be loaded.

Design for SPEED on long, high-res clips:
  * People are detected on a DOWNSCALED frame (a 4K frame is overkill for a person box).
  * The BALL is ~10px at 4K and vanishes when the frame is downscaled, so it's found with
    native-resolution tiling — but only in a small ROI around the Kalman-predicted position
    once we have a track, so it costs one extra inference, not nine.
  * ByteTrack gives persistent IDs; teams are assigned per-track by majority vote so colours
    don't flicker frame to frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]  # x1,y1,x2,y2 in full-res image px
    feet: Tuple[float, float]  # ground-contact point (bottom-centre)
    kind: str  # 'player' | 'goalkeeper' | 'referee' | 'ball'
    conf: float = 0.0
    track_id: int = -1
    team: int = -1  # 0/1 team, -1 unknown/not-a-player


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# Roboflow class id/name -> our kind. The football-players-detection model uses these names.
_RF_NAMES = {"player": "player", "goalkeeper": "goalkeeper", "referee": "referee", "ball": "ball"}


def default_people_weights() -> str:
    """Local COCO person weights, best first. YOLO11s at 1920 px costs about what YOLOv8n did
    and found more people (audited lab recall 0.93 -> 0.98 at conf 0.25) with 2-5x fewer
    half-body duplicates. Weights live in models/ (git-ignored); place yolo11s.pt there to use
    it. Without it yolov8n.pt is used (downloaded to models/ on first use)."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    best = repo / "models" / "yolo11s.pt"
    if best.is_file():
        return str(best)
    target = repo / "models" / "yolov8n.pt"
    target.parent.mkdir(exist_ok=True)
    return str(target)


class SoccerDetector:
    """Hybrid backend tuned for SPEED on long clips:
      * players  -> a fast local COCO YOLO on a downscaled frame (every frame),
      * referees -> the soccer-specific model, run SPARSELY (they move slowly),
      * ball     -> the soccer-specific model at native resolution (tiny object), ROI-tracked.
    The soccer model (Roboflow football-players-detection, run locally via `inference`) is the
    only thing COCO can't do, so we use it as little as possible."""

    def __init__(
        self,
        roboflow_model_id: Optional[str] = "football-players-detection-3zvbc/9",
        coco_weights: Optional[str] = None,
        people_imgsz: int = 1920,
        conf: float = 0.13,
        device: Optional[str] = None,
        ref_imgsz: int = 1280,
        enhance_dark: bool = True,
    ):
        if coco_weights is None:
            coco_weights = default_people_weights()
        self.device = device or _pick_device()
        self.people_imgsz = people_imgsz
        self.ref_imgsz = ref_imgsz
        self.conf = conf
        self.enhance_dark = enhance_dark
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        from ultralytics import YOLO

        self._yolo = YOLO(coco_weights)  # fast people backbone (MPS)
        # soccer model: refs + ball. None skips it (e.g. when a vision oracle supplies roles).
        self._rf = self._load_soccer(roboflow_model_id) if roboflow_model_id else None
        self.has_soccer = self._rf is not None

    @staticmethod
    def _load_soccer(model_id: str):
        """Load Roboflow's soccer model locally. PnLCalib ships its own top-level `inference.py`
        and puts itself on sys.path, which SHADOWS the Roboflow `inference` PACKAGE — so we must
        strip those entries (and any wrong cached module) before importing get_model."""
        import sys
        import os

        saved_path = sys.path[:]
        wrong = sys.modules.get("inference")
        try:
            sys.path = [
                p for p in sys.path if "PnLCalib" not in p and p not in ("", ".", os.getcwd())
            ]
            if wrong is not None and "PnLCalib" in str(getattr(wrong, "__file__", "")):
                del sys.modules["inference"]
            from dotenv import load_dotenv

            from pathlib import Path

            load_dotenv(Path(__file__).resolve().parents[1] / ".env")
            api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
            if not api_key:
                return None
            from inference import get_model

            return get_model(model_id=model_id, api_key=api_key)
        except Exception as exc:
            from soccer_analyzer.resources import is_memory_error

            if is_memory_error(exc):
                raise
            return None
        finally:
            sys.path = saved_path

    # ---- players (fast COCO) --------------------------------------------------------------

    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE on the L channel for dark/night frames, for the ball detectors only. People are
        detected on raw frames: CLAHE lowered audited YOLOv8n people recall (lab 2026-10-12: night
        0.974 raw vs 0.948 with CLAHE; the daytime union clip, dark enough to trip the rule, 0.972
        vs 0.954). Never grayscale/invert (out of distribution)."""
        if not self.enhance_dark or frame.mean() >= 80:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        return cv2.cvtColor(cv2.merge([self._clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

    def detect_players(self, frame: np.ndarray) -> List[Detection]:
        # Native-ish resolution (1920): far players are tiny and vanish at 1280. Raw frames, no
        # CLAHE (see `_enhance`).
        r = self._yolo.predict(
            frame,
            classes=[0],
            conf=self.conf,
            imgsz=self.people_imgsz,
            iou=0.6,
            device=self.device,
            verbose=False,
        )[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            out.append(Detection((x1, y1, x2, y2), ((x1 + x2) / 2, y2), "player", float(b.conf)))
        return out

    # ---- referees (sparse soccer model) ---------------------------------------------------

    def detect_officials(self, frame: np.ndarray) -> List[Detection]:
        """Referees AND goalkeepers from the soccer model (downscaled). Keepers get their own
        class so they aren't mistaken for referees (both wear colours distinct from the teams)."""
        if not self.has_soccer:
            return []
        h, w = frame.shape[:2]
        scale = self.ref_imgsz / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
        inv = 1.0 / scale if scale < 1.0 else 1.0
        out = []
        res = self._rf.infer(small, confidence=0.35)[0]
        for p in res.predictions:
            if p.class_name not in ("referee", "goalkeeper"):
                continue
            x1 = (p.x - p.width / 2) * inv
            y1 = (p.y - p.height / 2) * inv
            x2 = (p.x + p.width / 2) * inv
            y2 = (p.y + p.height / 2) * inv
            out.append(
                Detection((x1, y1, x2, y2), ((x1 + x2) / 2, y2), p.class_name, float(p.confidence))
            )
        return out

    # ---- ball (tiny -> native-res tiling / ROI) -------------------------------------------

    def _ball_candidates(
        self, crop: np.ndarray, ox: float, oy: float, conf: float
    ) -> List[Detection]:
        """ALL ball detections in a native-res crop (offset into full-frame coords). Returning
        every candidate (not just the top one) lets the caller drop off-field false positives and
        keep the real on-pitch ball, which is often a lower-confidence detection."""
        if crop.size == 0 or not self.has_soccer:
            return []
        out = []
        for p in self._rf.infer(crop, confidence=conf)[0].predictions:
            if p.class_name != "ball":
                continue
            out.append(
                Detection(
                    (
                        p.x - p.width / 2 + ox,
                        p.y - p.height / 2 + oy,
                        p.x + p.width / 2 + ox,
                        p.y + p.height / 2 + oy,
                    ),
                    (p.x + ox, p.y + oy),
                    "ball",
                    float(p.confidence),
                )
            )
        return out

    @staticmethod
    def _pick_ball(cands: List[Detection], valid, near) -> Optional[Detection]:
        """Among ball candidates keep the valid (on-field) ones; prefer the nearest to `near`
        (the predicted position) else the highest confidence."""
        cands = [c for c in cands if valid is None or valid(c.feet)]
        if not cands:
            return None
        if near is not None:
            return min(cands, key=lambda c: (c.feet[0] - near[0]) ** 2 + (c.feet[1] - near[1]) ** 2)
        return max(cands, key=lambda c: c.conf)

    def detect_ball_roi(
        self,
        frame: np.ndarray,
        center: Tuple[float, float],
        size: int = 640,
        conf: float = 0.10,
        valid=None,
    ) -> Optional[Detection]:
        """Detect the ball in a native-res window around `center` (cheap: one inference)."""
        h, w = frame.shape[:2]
        cx, cy = center
        x0 = int(np.clip(cx - size / 2, 0, max(0, w - size)))
        y0 = int(np.clip(cy - size / 2, 0, max(0, h - size)))
        cands = self._ball_candidates(frame[y0 : y0 + size, x0 : x0 + size], x0, y0, conf)
        return self._pick_ball(cands, valid, center)

    def detect_ball_full(
        self, frame: np.ndarray, grid: int = 3, conf: float = 0.10, valid=None
    ) -> Optional[Detection]:
        """Full native-res tiled scan for the ball (used to (re)acquire the track)."""
        return self._pick_ball(self.all_ball_candidates(frame, conf), valid, None)

    def all_ball_candidates(
        self, frame: np.ndarray, conf: float = 0.15, grid: int = 3, valid=None
    ) -> List[Detection]:
        """ALL ball candidates (tiled native-res). The trajectory tracker needs every candidate,
        not just the top one, to find the moving ball among the many static false positives."""
        h, w = frame.shape[:2]
        cands = []
        for ry in range(grid):
            for rx in range(grid):
                y0, y1 = ry * h // grid, (ry + 1) * h // grid
                x0, x1 = rx * w // grid, (rx + 1) * w // grid
                cands += self._ball_candidates(frame[y0:y1, x0:x1], x0, y0, conf)
        if valid is not None:
            cands = [c for c in cands if valid(c.feet)]
        return cands


class BallTracker:
    """Constant-velocity Kalman tracker for the ball: predict every frame, correct on detection,
    coast (extrapolate) through gaps so the ball keeps moving when the detector misses it."""

    def __init__(self, max_coast: int = 25, gate_px: float = 250.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 4.0
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0
        self.initialised = False
        self.coast = 0  # frames since last real measurement
        self.max_coast = max_coast
        self.gate_px = gate_px
        self.pos: Optional[Tuple[float, float]] = None
        self.measured = False  # was the current pos a real detection?

    def _init(self, xy):
        self.kf.statePost = np.array([[xy[0]], [xy[1]], [0], [0]], np.float32)
        self.initialised = True

    def predict(self) -> Optional[Tuple[float, float]]:
        if not self.initialised:
            return None
        p = self.kf.predict()
        self.pos = (float(p[0]), float(p[1]))
        return self.pos

    def update(self, det: Optional[Detection]):
        """Feed a (possibly None) ball detection. Returns the current best ball position and
        whether it was measured (True) or predicted/coasted (False)."""
        if det is not None:
            xy = det.feet
            if not self.initialised:
                self._init(xy)
                self.pos = xy
                self.measured = True
                self.coast = 0
                return self.pos, True
            # gate against the prediction so a stray detection doesn't yank the track
            pred = self.pos or xy
            if np.hypot(xy[0] - pred[0], xy[1] - pred[1]) <= self.gate_px + self.coast * 30:
                self.kf.correct(np.array([[np.float32(xy[0])], [np.float32(xy[1])]]))
                self.pos = xy
                self.measured = True
                self.coast = 0
                return self.pos, True
        # no usable detection -> coast on the prediction
        self.coast += 1
        self.measured = False
        if self.coast > self.max_coast:
            self.initialised = False
            self.pos = None
        return self.pos, False


class BallTrajectoryTracker:
    """Trajectory-based ball tracking — the fix for 'the detector finds 82 white specks at equal
    confidence and the tracker locks onto a static line-mark, never the real ball'.

    Instead of picking the best candidate per frame, we LINK candidates across frames into motion-
    consistent tracks and output the one that PERSISTS and MOVES. Static false positives are
    rejected (no motion); transient false positives die (no persistence); only the real ball forms
    a long, smoothly-moving track. Online multi-hypothesis, ~pure numpy."""

    def __init__(self, gate: float = 150.0, max_age: int = 14, sticky: float = 1.4):
        self.tracks: List[dict] = []
        self.t = 0
        self.gate = gate
        self.max_age = max_age
        self.sticky = sticky  # bonus for the currently-selected ball track (continuity)
        self.ball_id = None
        self._next = 0

    @staticmethod
    def _predict(tr: dict, at: int) -> np.ndarray:
        ts = sorted(tr["pts"])[-3:]
        pts = [np.asarray(tr["pts"][k], float) for k in ts]
        last = ts[-1]
        if len(pts) >= 3:
            acc = (pts[-1] - pts[-2]) - (pts[-2] - pts[-3])
            vel = (pts[-1] - pts[-2]) + acc
        elif len(pts) == 2:
            vel = pts[-1] - pts[-2]
        else:
            vel = np.zeros(2)
        return pts[-1] + vel * (at - last)

    def _score(self, tr: dict) -> float:
        if tr["hits"] < 3:
            return -1.0
        ts = sorted(tr["pts"])
        pts = np.array([tr["pts"][k] for k in ts])
        path = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        recency = 1.0 / (1.0 + (self.t - tr["last"]))
        s = tr["hits"] * recency * (1.0 + min(path, 1500.0) / 300.0)
        if tr["id"] == self.ball_id:
            s *= self.sticky
        return s

    def update(self, candidates: List[Detection]):
        """candidates: list of ball Detections this frame (may be empty). Returns (xy, measured)."""
        cands = [(c.feet[0], c.feet[1]) for c in (candidates or [])]
        used = set()
        for tr in self.tracks:
            if self.t - tr["last"] > self.max_age:
                continue
            pred = self._predict(tr, self.t)
            lim = self.gate + (self.t - tr["last"]) * 45
            best, bd = None, lim
            for i, (x, y) in enumerate(cands):
                if i in used:
                    continue
                d = np.hypot(x - pred[0], y - pred[1])
                if d < bd:
                    bd, best = d, i
            if best is not None:
                tr["pts"][self.t] = cands[best]
                tr["last"] = self.t
                tr["hits"] += 1
                used.add(best)
        for i, (x, y) in enumerate(cands):
            if i not in used:
                self.tracks.append(
                    {"id": self._next, "pts": {self.t: (x, y)}, "last": self.t, "hits": 1}
                )
                self._next += 1
        self.tracks = [tr for tr in self.tracks if self.t - tr["last"] <= self.max_age]
        best = max(self.tracks, key=self._score, default=None)
        measured = False
        pos = None
        if best is not None and self._score(best) > 0:
            self.ball_id = best["id"]
            measured = self.t in best["pts"]
            pos = best["pts"][self.t] if measured else tuple(self._predict(best, self.t))
        self.t += 1
        return pos, measured


_KIND_ID = {"ball": 0, "goalkeeper": 1, "player": 2, "referee": 3}
_ID_KIND = {v: k for k, v in _KIND_ID.items()}


class TeamAssigner:
    """Two-team split from jersey colour — deliberately simple/programmatic (no per-frame ML):
    average each tracked player's chest colour over their WHOLE track (stable, denoised), then
    split those per-player averages into two clusters. One running colour per player, ~22 colours,
    clustered into 2 = the two teams. Robust because it clusters stable per-player averages, not
    noisy per-frame samples."""

    def __init__(self):
        from sklearn.cluster import KMeans

        self._KMeans = KMeans
        self.samples: List[np.ndarray] = []  # global per-detection chest colours
        self.centers: Optional[np.ndarray] = None  # 2x3 LAB team centroids (None until fit)
        self.votes: Dict[int, np.ndarray] = {}  # track_id -> [team0_votes, team1_votes]

    @staticmethod
    def _jersey_lab(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
        """Median LAB of the chest band, with green grass removed (white AND black jerseys kept)."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h = y2 - y1
        w = x2 - x1
        cy1, cy2 = y1 + int(0.18 * h), y1 + int(0.50 * h)  # chest band
        cx1, cx2 = x1 + int(0.20 * w), x2 - int(0.20 * w)  # central (drop arms/edges)
        crop = frame[max(0, cy1) : max(cy1 + 1, cy2), max(0, cx1) : max(cx1 + 1, cx2)]
        if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 2:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
        hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        grass = (hue >= 35) & (hue <= 90) & (sat > 35)
        m = (~grass) & (val > 12) & (val < 252)
        if m.sum() < 12:
            return None
        return np.median(lab[m], axis=0)

    def observe(self, frame, det: Detection):
        """Record this detection's chest colour as a global sample AND, once teams are fit, cast a
        per-track vote for its nearest team (per-detection clustering + per-track voting is what
        validated cleanly offline — silhouette 0.46-0.59 day/night)."""
        c = self._jersey_lab(frame, det.bbox)
        if c is None:
            return
        self.samples.append(c)
        if self.centers is not None and det.track_id >= 0:
            lab = int(np.argmin(np.linalg.norm(self.centers - c, axis=1)))
            v = self.votes.setdefault(det.track_id, np.zeros(2))
            v[lab] += 1

    def maybe_fit(self):
        """K=2 on the global per-detection chest colours -> two team centroids. (Plain K=2 was
        the offline winner; K=3-take-2 wrongly split one team into lit/shadow halves.)"""
        if len(self.samples) < 20:
            return
        X = np.array(self.samples[-3000:])  # recent window, bounded
        km = self._KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        c0, c1 = km.cluster_centers_[0], km.cluster_centers_[1]
        if c0[0] > c1[0]:  # order by lightness -> stable team 0/1
            c0, c1 = c1, c0
        self.centers = np.array([c0, c1])

    def team(self, track_id: int) -> int:
        """Majority-vote team for a track (0/1), or -1 until it has evidence."""
        v = self.votes.get(track_id)
        return int(np.argmax(v)) if (v is not None and v.sum()) else -1

    def is_outlier(self, frame, bbox, factor: float = 0.6) -> bool:
        """True if a crop's colour is far from BOTH team centroids (a real referee), relative to
        the team-to-team separation — used to tell a true ref from a dark-jersey player."""
        if self.centers is None:
            return True
        c = self._jersey_lab(frame, bbox)
        if c is None:
            return False
        sep = float(np.linalg.norm(self.centers[0] - self.centers[1]))
        return float(np.linalg.norm(self.centers - c, axis=1).min()) > max(35.0, factor * sep)


class MatchTracker:
    """Per-frame people tracking (ByteTrack) + stable teams + Kalman ball. Call step() on each
    frame you choose to process (stride for speed); it returns players/keepers/referees with
    persistent ids+teams and the ball (measured or coasted)."""

    def __init__(
        self,
        detector: SoccerDetector,
        fps: int = 30,
        fit_frames: int = 8,
        ref_every: int = 10,
        ball_every: int = 1,
        ball_rescan: int = 8,
    ):
        import supervision as sv

        self.det = detector
        self.tracker = sv.ByteTrack(frame_rate=max(2, fps), lost_track_buffer=60)
        self._sv = sv
        self.ball = BallTrajectoryTracker()
        self.teams = TeamAssigner()
        self.fit_frames = fit_frames
        self.ref_every = ref_every
        self.ball_every = max(1, ball_every)  # detect ball every Nth frame (Kalman coasts)
        self.ball_rescan = ball_rescan
        self._last_refs: List[Detection] = []
        self.n = 0

    @staticmethod
    def _near_ref(p: Detection, refs: List[Detection], tol: float = 60.0) -> bool:
        for rf in refs:
            if abs(p.feet[0] - rf.feet[0]) < tol and abs(p.feet[1] - rf.feet[1]) < tol:
                return True
        return False

    def step(self, frame: np.ndarray, ball_ok=None, player_ok=None) -> dict:
        """ball_ok / player_ok (image_xy) -> bool: optional validators ('maps onto the pitch').
        ball_ok rejects off-field false-positive balls; player_ok drops off-pitch crowd so the
        team-colour clustering isn't polluted by spectators (the cause of muddy team colours)."""
        # players: fast COCO every frame -> ByteTrack for stable ids
        players = self.det.detect_players(frame)
        tracked: List[Detection] = []
        if players:
            xyxy = np.array([p.bbox for p in players], dtype=float)
            conf = np.array([p.conf for p in players], dtype=float)
            cid = np.zeros(len(players), dtype=int)
            t = self.tracker.update_with_detections(
                self._sv.Detections(xyxy=xyxy, confidence=conf, class_id=cid)
            )
            for i in range(len(t)):
                x1, y1, x2, y2 = t.xyxy[i]
                tid = int(t.tracker_id[i]) if t.tracker_id is not None else -1
                tracked.append(
                    Detection(
                        (x1, y1, x2, y2),
                        ((x1 + x2) / 2, y2),
                        "player",
                        float(t.confidence[i]) if t.confidence is not None else 0.0,
                        track_id=tid,
                    )
                )

        # drop off-pitch crowd when the pitch is known (calibration); on uncalibrated clips the
        # team clustering tolerates the crowd minority instead.
        if player_ok is not None:
            tracked = [p for p in tracked if player_ok(p.feet)]

        # teams: accumulate each track's running-average jersey colour, (re)fit periodically, then
        # label every player by nearest team. COCO sees keepers/refs as 'person' too, so they're
        # in `tracked` and get re-labelled below.
        for d in tracked:
            self.teams.observe(frame, d)
        if self.n % 12 == 0:
            self.teams.maybe_fit()
        for d in tracked:
            d.team = self.teams.team(d.track_id)

        # officials: the soccer model gives keeper + referee. A keeper is its own class. A
        # 'referee' is only believed when that person's jersey is a colour OUTLIER vs both teams
        # (else the model just tagged a dark-jersey player as a ref).
        if self.det.has_soccer and self.n % self.ref_every == 0:
            self._last_refs = self.det.detect_officials(frame)
        for off in self._last_refs:
            if player_ok is not None and not player_ok(off.feet):
                continue
            near = min(
                tracked,
                key=lambda p: (p.feet[0] - off.feet[0]) ** 2 + (p.feet[1] - off.feet[1]) ** 2,
                default=None,
            )
            if (
                near is None
                or abs(near.feet[0] - off.feet[0]) > 70
                or abs(near.feet[1] - off.feet[1]) > 70
            ):
                continue
            if off.kind == "goalkeeper":
                near.kind = "goalkeeper"  # keep team (keeper belongs to a team)
            elif self.teams.is_outlier(frame, near.bbox):
                near.kind = "referee"
                near.team = -1

        # ball: Kalman predicts every frame; we only run the (soccer-model) detector every
        # ball_every frames (ROI when tracking, periodic full native-res rescan to (re)acquire)
        # so the ball stays present without dominating compute.
        # ball: gather ALL native-res candidates every ball_every frames and feed the trajectory
        # tracker (it links them over time and keeps the persistent, MOVING track = the real ball;
        # static/transient false positives are dropped). Predicts on the in-between frames.
        cands = []
        if self.n % self.ball_every == 0:
            cands = self.det.all_ball_candidates(frame, conf=0.18, valid=ball_ok)
        bpos, bmeas = self.ball.update(cands)

        self.n += 1
        return {"people": tracked, "ball": bpos, "ball_measured": bmeas}
