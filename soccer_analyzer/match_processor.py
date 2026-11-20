"""Offline player tracking, team assignment and ball-path selection.


Detection runs in batches; teams are assigned once per track from jersey color samples.
Ball paths use measured candidates from three consecutive raw frames. Short interpolated
positions are recorded separately so analytics can distinguish them from observations.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from soccer_analyzer.detection import SoccerDetector, TeamAssigner
from soccer_analyzer.video import frame_limit, frame_window, seek_capture, open_capture


@dataclass
class MatchData:
    """Everything Pass C needs, keyed by ORIGINAL frame index."""

    frames: List[int] = field(default_factory=list)  # processed frame indices
    people: Dict[int, list] = field(default_factory=dict)  # fidx -> [(tid, bbox, feet)]
    team: Dict[int, int] = field(default_factory=dict)  # tid -> 0/1/-1
    kind: Dict[int, str] = field(default_factory=dict)  # tid -> player/referee/goalkeeper
    ball: Dict[int, tuple] = field(default_factory=dict)  # fidx -> (x, y, measured)
    team_centers: Optional[np.ndarray] = None
    stats: dict = field(default_factory=dict)
    interpolated: set[tuple[int, int]] = field(default_factory=set)  # (source frame, track ID)
    cuts: list[int] = field(default_factory=list)  # first sampled frame after each scene cut


class BallPathSolver:
    """Score candidate paths within each uninterrupted stretch of evidence.

    Long gaps and supplied scene cuts split the graph. Each stretch needs enough measured
    nodes; short gaps may interpolate, with provenance retained. Persistent static distractors
    still need labelled evaluation; a stationary ball is not automatically rejected.
    """

    def __init__(
        self,
        vmax_px: float = 90.0,
        gap_frames: int = 75,
        alpha: float = 1.2,
        beta: float = 0.02,
        min_nodes: int = 4,
    ):
        if not np.isfinite(vmax_px) or vmax_px <= 0 or gap_frames < 1 or min_nodes < 1:
            raise ValueError("speed, gap and minimum observations must be positive")
        if not np.isfinite(alpha) or not np.isfinite(beta) or alpha < 0 or beta < 0:
            raise ValueError("path penalties must be finite and nonnegative")
        self.vmax, self.gap = vmax_px, gap_frames
        self.alpha, self.beta, self.min_nodes = alpha, beta, min_nodes

    def solve(
        self,
        cands: Dict[int, List[Tuple[float, float, float]]],
        interp_gap: int = 40,
        *,
        cuts: list[int] | None = None,
    ) -> Dict[int, tuple]:
        """Return source-frame positions (x_px, y_px, measured), never across a scene cut."""
        from bisect import bisect_right

        if interp_gap < 0:
            raise ValueError("interpolation gap must be nonnegative")
        barriers = sorted(cuts or [])
        groups, group = [], []
        last_frame, last_segment = None, None
        for frame in sorted(cands):
            candidates = [
                (x, y, score)
                for x, y, score in cands[frame]
                if np.isfinite([x, y, score]).all() and x >= 0 and y >= 0 and 0 < score <= 1
            ]
            if not candidates:
                continue
            segment = bisect_right(barriers, frame)
            if last_frame is not None and (
                frame - last_frame > self.gap or segment != last_segment
            ):
                groups.append(group)
                group = []
            group.extend((frame, x, y, score) for x, y, score in candidates)
            last_frame, last_segment = frame, segment
        if group:
            groups.append(group)
        out = {}
        for group in groups:
            out.update(self._solve_segment(np.asarray(group, dtype=np.float64), interp_gap))
        return out

    def _solve_segment(self, nodes: np.ndarray, interp_gap: int) -> dict:
        times, xy = nodes[:, 0], nodes[:, 1:3]
        scores = nodes[:, 3].copy()
        parents = np.full(len(nodes), -1, dtype=int)
        # Search only the bounded temporal neighborhood, and calculate link scores in NumPy.
        for i, (frame, x, y, score) in enumerate(nodes):
            lo = int(np.searchsorted(times, frame - self.gap))
            hi = int(np.searchsorted(times, frame))
            if hi <= lo:
                continue
            dt = frame - times[lo:hi]
            distance = np.linalg.norm(xy[lo:hi] - [x, y], axis=1)
            gain = (
                scores[lo:hi]
                + score
                - self.alpha * distance / (self.vmax * dt)
                - self.beta * (dt - 1)
            )
            gain[distance > self.vmax * dt + 60] = -np.inf
            best = int(np.argmax(gain))
            if gain[best] > scores[i]:
                scores[i], parents[i] = gain[best], lo + best
        end = int(np.argmax(scores))
        path = []
        while end >= 0:
            path.append(nodes[end])
            end = parents[end]
        path.reverse()
        if len(path) < self.min_nodes:
            return {}
        out = {}
        for k, (frame, x, y, _) in enumerate(path):
            frame = int(frame)
            out[frame] = (float(x), float(y), True)
            if k + 1 < len(path):
                next_frame, nx, ny, _ = path[k + 1]
                next_frame = int(next_frame)
                if 1 < next_frame - frame <= interp_gap:
                    for missing in range(frame + 1, next_frame):
                        t = (missing - frame) / (next_frame - frame)
                        out[missing] = (float(x + t * (nx - x)), float(y + t * (ny - y)), False)
        return out


class MatchProcessor:
    """Pass A sweep + Pass B global solves. Detectors are injected (shared/warm)."""

    def __init__(
        self,
        detector: SoccerDetector,
        wasb=None,
        fps: int = 30,
        officials_every: int = 8,
        ball_every: int = 1,
        yolo_batch: int = 4,
        ball_thresh: float = 0.32,
    ):
        if officials_every <= 0 or ball_every <= 0 or yolo_batch <= 0 or fps <= 0:
            raise ValueError("cadences, batch size, and fps must be positive")
        import supervision as sv

        self.det = detector
        self.wasb = wasb
        self.tracker = sv.ByteTrack(frame_rate=max(2, fps), lost_track_buffer=90)
        self._sv = sv
        self.officials_every = officials_every
        self.ball_every = max(1, ball_every)
        self.yolo_batch = yolo_batch
        self.ball_thresh = ball_thresh

    @staticmethod
    def _interpolate_tracks(md: "MatchData", max_gap_frames: int):
        """Fill each track's short disappearances: if a tid is seen at processed frames f1 and f2
        (gap <= max_gap_frames) but missing in between, linearly interpolate its box there."""
        by_tid: Dict[int, list] = {}
        for f in md.frames:
            for tid, bbox, feet in md.people.get(f, []):
                if tid >= 0:
                    by_tid.setdefault(tid, []).append((f, bbox))
        frames_sorted = sorted(md.frames)
        fpos = {f: i for i, f in enumerate(frames_sorted)}
        for tid, obs in by_tid.items():
            obs.sort()
            for (f1, b1), (f2, b2) in zip(obs, obs[1:]):
                if f2 - f1 <= 1 or f2 - f1 > max_gap_frames:
                    continue
                if any(f1 < cut <= f2 for cut in md.cuts):
                    continue
                # Do not draw a player across a large image-space jump during an occlusion.
                height = max(1.0, (b1[3] - b1[1] + b2[3] - b2[1]) / 2)
                if np.hypot((b2[0] + b2[2] - b1[0] - b1[2]) / 2, b2[3] - b1[3]) > 2 * height:
                    continue
                i1, i2 = fpos.get(f1), fpos.get(f2)
                if i1 is None or i2 is None or i2 - i1 <= 1:
                    continue
                for k in range(i1 + 1, i2):
                    fm = frames_sorted[k]
                    if any(p[0] == tid for p in md.people.get(fm, [])):
                        continue
                    t = (fm - f1) / (f2 - f1)
                    bb = tuple(a + t * (b - a) for a, b in zip(b1, b2))
                    md.people.setdefault(fm, []).append((tid, bb, ((bb[0] + bb[2]) / 2.0, bb[3])))
                    md.interpolated.add((fm, tid))

    # ---- Pass A ----------------------------------------------------------------------------

    def sweep(
        self,
        video_path: str,
        stride: int = 3,
        max_frames: Optional[int] = None,
        progress=None,
        pitch_filter=None,
        start_frame: int = 0,
    ) -> MatchData:
        """pitch_filter(fidx, feet_xy) -> bool: optional 'is this on the pitch' test built from
        the calibration. CRITICAL for team quality: without it, bench players / subs in bibs /
        spectators pollute the global two-team colour fit (K=2 centroids collapse onto crowd
        colours). Off-pitch tracks are excluded from the fit and left team=-1."""
        if stride <= 0:
            raise ValueError("stride must be positive")
        frame_limit(0, max_frames)  # Validate before opening a resource.
        self.tracker.reset()  # Each sweep starts a new clip.
        md = MatchData()

        colour_samples: List[np.ndarray] = []  # global jersey colours (on-pitch)
        track_samples: Dict[int, deque] = {}  # bounded colour history per on-pitch track
        track_onpitch: Dict[int, list] = {}  # tid -> [n_on, n_total]
        official_votes: Dict[int, dict] = {}  # tid -> {'referee': n, 'goalkeeper': n}
        n_official_scans = [0]
        ball_cands: Dict[int, list] = {}

        raw3 = deque(maxlen=3)  # rolling raw frames for WASB
        batch_frames: List[np.ndarray] = []
        batch_idx: List[int] = []
        nproc = 0
        advanced = retrieved = ball_scans = 0
        previous_view = None
        track_offset = highest_id = 0
        rng = np.random.default_rng(0)
        color_seen = 0

        def flush_batch():
            nonlocal nproc, previous_view, track_offset, highest_id, color_seen
            if not batch_frames:
                return
            # Raw frames: CLAHE lowered people recall (see SoccerDetector._enhance).
            results = self.det._yolo.predict(
                batch_frames,
                classes=[0],
                conf=self.det.conf,
                imgsz=self.det.people_imgsz,
                iou=0.6,
                device=self.det.device,
                verbose=False,
            )
            for f, fidx, r in zip(batch_frames, batch_idx, results):
                view = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36))
                if previous_view is not None:
                    difference = float(np.mean(cv2.absdiff(view, previous_view)))
                    histogram = cv2.calcHist([view], [0], None, [32], [0, 256])
                    previous_histogram = cv2.calcHist([previous_view], [0], None, [32], [0, 256])
                    if (
                        difference > 45
                        and cv2.compareHist(
                            histogram, previous_histogram, cv2.HISTCMP_BHATTACHARYYA
                        )
                        > 0.45
                    ):
                        md.cuts.append(fidx)
                        self.tracker.reset()
                        track_offset = highest_id
                previous_view = view
                xyxy = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
                conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros((0,))
                dets = self._sv.Detections(
                    xyxy=xyxy, confidence=conf, class_id=np.zeros(len(xyxy), dtype=int)
                )
                t = self.tracker.update_with_detections(dets)
                people = []
                for i in range(len(t)):
                    x1, y1, x2, y2 = t.xyxy[i]
                    tid = int(t.tracker_id[i]) if t.tracker_id is not None else -1
                    if tid >= 0:
                        tid += track_offset
                        highest_id = max(highest_id, tid)
                    feet = ((x1 + x2) / 2.0, float(y2))
                    people.append((tid, (float(x1), float(y1), float(x2), float(y2)), feet))
                    if tid < 0:
                        continue
                    on = pitch_filter is None or pitch_filter(fidx, feet)
                    st = track_onpitch.setdefault(tid, [0, 0])
                    st[0] += int(on)
                    st[1] += 1
                    if on:
                        c = TeamAssigner._jersey_lab(f, (x1, y1, x2, y2))
                        if c is not None:
                            if (y2 - y1) >= 35:  # centroid FIT from near/large boxes
                                # A deterministic reservoir bounds full-match color fitting.
                                color_seen += 1
                                if len(colour_samples) < 5000:
                                    colour_samples.append(c)
                                else:
                                    slot = int(rng.integers(color_seen))
                                    if slot < len(colour_samples):
                                        colour_samples[slot] = c
                            samples = track_samples.setdefault(tid, deque(maxlen=240))
                            samples.append(c)
                md.people[fidx] = people
                md.frames.append(fidx)
                # officials on schedule (uses the tracked people just computed)
                if nproc % self.officials_every == 0 and self.det.has_soccer:
                    n_official_scans[0] += 1
                    for off in self.det.detect_officials(f):
                        near = min(
                            people,
                            key=lambda p: (p[2][0] - off.feet[0]) ** 2
                            + (p[2][1] - off.feet[1]) ** 2,
                            default=None,
                        )
                        if (
                            near
                            and abs(near[2][0] - off.feet[0]) < 70
                            and abs(near[2][1] - off.feet[1]) < 70
                        ):
                            v = official_votes.setdefault(near[0], {})
                            v[off.kind] = v.get(off.kind, 0) + 1
                nproc += 1
                if progress and nproc % 10 == 0 and total:
                    progress(min(1.0, (fidx - start_frame) / total))
            batch_frames.clear()
            batch_idx.clear()

        with open_capture(video_path) as cap:
            start, stop = frame_window(cap.get(cv2.CAP_PROP_FRAME_COUNT), max_frames, start_frame)
            total = stop - start if stop is not None else 0
            seek_capture(cap, start)
            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            raw_fps = raw_fps if np.isfinite(raw_fps) and raw_fps > 0 else 30
            source_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920
            idx = start
            ball_period = stride * self.ball_every
            while True:
                if stop is not None and idx >= stop:
                    break
                if not cap.grab():
                    break
                advanced += 1
                sampled = (idx - start) % stride == 0
                # Sequential grabbing keeps codec dependencies intact. Retrieve BGR pixels
                # only for player samples or a scheduled WASB center and its two neighbors.
                ball_context = self.wasb is not None and any(
                    center > start
                    and (stop is None or center + 1 < stop)
                    and (center - start) % ball_period == 0
                    for center in range(idx - 1, idx + 2)
                )
                if not sampled and not ball_context:
                    idx += 1
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    break
                retrieved += 1
                if self.wasb is not None:
                    if raw3 and raw3[-1][0] != idx - 1:
                        raw3.clear()  # Never present skipped source frames as consecutive.
                    raw3.append((idx, frame))
                # WASB runs for the MIDDLE of 3 consecutive raw frames when that middle frame is due.
                # Night frames get the same CLAHE the player detector uses (WASB is contrast-hungry).
                if self.wasb is not None and len(raw3) == 3:
                    mid_idx = raw3[1][0]
                    if (mid_idx - start) % ball_period == 0:
                        trio = [self.det._enhance(f) for (_, f) in raw3]
                        ball_scans += 1
                        c = self.wasb.detect(trio, thresh=self.ball_thresh)
                        if c:
                            ball_cands[mid_idx] = c
                if sampled:
                    batch_frames.append(frame)
                    batch_idx.append(idx)
                    if len(batch_frames) >= self.yolo_batch:
                        flush_batch()
                idx += 1
            flush_batch()

        # Body overlaps are ambiguous: often jersey details, sometimes a real airborne ball.
        # Compare with boxes from the same source frame, and retain weaker evidence.
        if ball_cands and md.frames:

            def on_body(fidx, x, y):
                # WASB centers are already on the processed schedule; no whole-clip search.
                for _, (x1, y1, x2, y2), _f in md.people.get(fidx, []):
                    if x1 - 4 <= x <= x2 + 4 and y1 - 4 <= y <= y1 + 0.78 * (y2 - y1):
                        return True
                return False

            ball_cands = {
                # A real airborne ball can overlap a body. Downweight that ambiguous evidence.
                f: [(x, y, score * (0.55 if on_body(f, x, y) else 1)) for x, y, score in cl]
                for f, cl in ball_cands.items()
            }
            ball_cands = {f: cl for f, cl in ball_cands.items() if cl}

        # ---- Pass B: global solves ----------------------------------------------------------
        # Team fit in a CHROMA-WEIGHTED LAB space: shade drags lightness (far-side white looks
        # grey and jumps clusters) but chroma survives illumination — a red jersey keeps a>>128
        # in shade, white stays neutral. Weight (L,a,b) by (0.5, 1.6, 1.6) for fit + assignment.
        CW = np.array([0.5, 1.6, 1.6])
        teams = TeamAssigner()
        centers = None
        if len(colour_samples) >= 20:
            X = np.array(colour_samples) * CW
            km = teams._KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
            c0, c1 = km.cluster_centers_[0], km.cluster_centers_[1]
            if c0[0] > c1[0]:
                c0, c1 = c1, c0
            centers = np.array([c0, c1])  # in the WEIGHTED space
        md.team_centers = centers / CW if centers is not None else None
        for tid, samples in track_samples.items():
            if centers is None or not samples:
                md.team[tid] = -1
                continue
            labs = np.array(samples) * CW
            d = np.linalg.norm(labs[:, None, :] - centers[None], axis=2)  # n x 2
            votes = np.bincount(np.argmin(d, axis=1), minlength=2)
            md.team[tid] = int(np.argmax(votes))
        teams.centers = md.team_centers  # for the outlier rule below

        # officials: accumulated votes decide; referees must also be colour outliers vs teams
        sep = (
            float(np.linalg.norm(teams.centers[0] - teams.centers[1]))
            if teams.centers is not None
            else 60.0
        )
        need = 2 if n_official_scans[0] >= 5 else 1  # short clip -> fewer chances to vote
        for tid, votes in official_votes.items():
            gk = votes.get("goalkeeper", 0)
            rf = votes.get("referee", 0)
            if gk >= need and gk >= rf:
                md.kind[tid] = "goalkeeper"
            elif rf >= need:
                samples = track_samples.get(tid, [])
                if samples and teams.centers is not None:
                    avg = np.mean(samples, axis=0)
                    if float(np.linalg.norm(teams.centers - avg, axis=1).min()) > max(
                        35.0, 0.6 * sep
                    ):
                        md.kind[tid] = "referee"
                        md.team[tid] = -1
                else:
                    md.kind[tid] = "referee"
                    md.team[tid] = -1

        # SMOOTH TRACKS: fill short per-track gaps by interpolation (a recorded clip lets us —
        # the detector missing a player for a few frames shouldn't make his box flicker out).
        self._interpolate_tracks(md, max_gap_frames=min(stride * 8, max(1, round(raw_fps * 0.5))))

        # A cut is detected at sampled frames, so its raw location lies somewhere after the
        # preceding sample. Drop temporal inputs on both sides of that uncertain interval.
        ball_cands = {
            f: c
            for f, c in ball_cands.items()
            if not any(cut - stride <= f <= cut + 1 for cut in md.cuts)
        }
        md.ball = BallPathSolver(
            vmax_px=90 * (source_width / 1920) * (30 / raw_fps),
            gap_frames=max(1, round(raw_fps * 0.8)),
        ).solve(ball_cands, interp_gap=max(1, round(raw_fps * 0.3)), cuts=md.cuts)
        md.stats = {
            "processed": nproc,
            "source_frames_advanced": advanced,
            "images_retrieved": retrieved,
            "ball_scans": ball_scans,
            "tracks": len(track_samples),
            "ball_frames_with_cands": len(ball_cands),
            "ball_path_frames": len(md.ball),
            "ball_measured_frames": sum(p[2] for p in md.ball.values()),
            "interpolated_player_positions": len(md.interpolated),
            "scene_cuts": md.cuts,
            "color_fit_samples": len(colour_samples),
            "officials": {k: v for k, v in md.kind.items()},
        }
        return md
