"""
Colour-agnostic multi-sport field calibration (productized from the validated prototype in
scratch/session_2026-07-05/color_agnostic/).

The problem: on multi-sport venues (an American-football field with yellow soccer lines and
red lacrosse lines) the soccer paint is a SPARSE minority colour, so both the CNN detector
and the plain white-line-support optimizer lock onto the dominant football clutter. This
module never hardcodes a line colour. Instead:

  1. EXTRACT painted lines of ANY colour (turf = dominant chroma mode; per-LAB-channel
     top-hat with per-channel adaptive thresholds; width filter + turf-ring veto) and
     cluster them into paint-colour sets by hue-angle histogram peaks (imbalance-robust).
  2. TEMPLATE-BANK ANCHOR: fit the DENSEST colour set against {soccer, football} templates.
     On this venue the football template locks onto the white yard lines -> a rigid METRIC
     ground-plane anchor. (If the soccer template wins, the venue is a normal soccer field
     and the main calibrator should handle it -> we return not_multisport.)
  3. Warm-track the anchor across frames sampled over the WHOLE clip (camera position +
     focal locked; rotation only), ACCUMULATE every colour set in ground meters.
  4. DISCOVER the soccer colour: fit the soccer template (length pinned, width free) to each
     colour family's accumulated raster, scored only inside the observed coverage, and
     discriminate by the soccer-vs-football score margin.
  5. JOINT BUNDLE: refine shared camera position+focal and per-frame rotations against the
     discovered soccer paint (regularized by the football anchor), then propagate the
     verified per-frame homographies across the clip by optical-flow chaining (these
     cameras pan almost continuously; a static stretch is just the chain's identity case).

Honesty: when only one half of the pitch is ever seen, its length is unmeasurable — the
returned stats carry a 'partial_length' flag and the measured 'visible_x' interval, and
VisiblePitch lets callers draw only the measured part.

Pure CPU: numpy / OpenCV / scipy. No torch, no CNN.

Entry point:
    calibrate_multisport(video_path, pitch, max_frames=None, stride=3, progress=None,
                         effort=1) -> ({original_frame_idx: H_image_to_pitch}, stats)
H maps original-frame pixels -> pitch meters in corner-origin PitchModel coordinates
(same contract as clip_calibrator.calibrate_clip_auto).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import differential_evolution, minimize

from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH
from soccer_analyzer.calibration.ls_calibrator import homography_7dof, _feasibility_penalty
from soccer_analyzer.calibration.keypoint_calib import homography_plausible
from soccer_analyzer.video import skip_frames
from soccer_analyzer.calibration.clip_calibrator import _gray_small
from soccer_analyzer.video import open_capture, frame_window, seek_capture
from soccer_analyzer.resources import is_memory_error

YD = 0.9144
_PROC_MAX_W = 1920  # extraction was tuned at 1920px width; downscale bigger frames
_STATIC_THRESH = 2.5  # gray_small mean abs-diff below this = camera static


# =========================================================================================
# American-football template (duck-typed like PitchModel)
# =========================================================================================


class FootballModel:
    """NCAA football field: 120 x 53.33 yd incl. end zones, full-width yard lines every
    5 yd, hash-mark rows 20 yd from each sideline. Duck-typed for the solvers/overlays:
    exposes .length, .width, .lines(), .circles(), .sample_markings(step)."""

    def __init__(self, hash_rows: bool = True):
        self.length = 120 * YD  # 109.73 m
        self.width = 160 / 3 * YD  # 48.77 m
        self.hash_rows = hash_rows

    def lines(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        L, W = self.length, self.width
        segs = [
            ((0.0, 0.0), (L, 0.0)),
            ((0.0, W), (L, W)),
            ((0.0, 0.0), (0.0, W)),
            ((L, 0.0), (L, W)),
        ]
        for k in range(0, 21):  # goal line .. goal line
            x = (10 + 5 * k) * YD
            segs.append(((x, 0.0), (x, W)))
        return segs

    def circles(self):
        return []

    def sample_markings(self, step: float = 1.0) -> np.ndarray:
        pts: List[Tuple[float, float]] = []
        for p0, p1 in self.lines():
            p0 = np.asarray(p0, float)
            p1 = np.asarray(p1, float)
            n = max(2, int(np.linalg.norm(p1 - p0) / step))
            for t in np.linspace(0, 1, n):
                pts.append(tuple(p0 + t * (p1 - p0)))
        if self.hash_rows:
            for y in (20 * YD, self.width - 20 * YD):
                for xy in np.arange(10.0, 110.0, 1.0):
                    pts.append((xy * YD, y))
        return np.array(pts, dtype=np.float64)


class VisiblePitch:
    """A PitchModel restricted to the MEASURED x-interval — callers should draw only this
    part when stats['partial_length'] is set (the rest of the field was never observed)."""

    def __init__(self, base: PitchModel, xlo: float, xhi: float):
        self.base = base
        self.length = base.length
        self.width = base.width
        self.xlo, self.xhi = float(xlo), float(xhi)

    def _clip(self, p0, p1):
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        dx = p1[0] - p0[0]
        if abs(dx) < 1e-9:
            return (p0, p1) if self.xlo <= p0[0] <= self.xhi else None
        ta, tb = (self.xlo - p0[0]) / dx, (self.xhi - p0[0]) / dx
        t0, t1 = max(0.0, min(ta, tb)), min(1.0, max(ta, tb))
        if t0 >= t1:
            return None
        return p0 + t0 * (p1 - p0), p0 + t1 * (p1 - p0)

    def lines(self):
        out = []
        for p0, p1 in self.base.lines():
            seg = self._clip(p0, p1)
            if seg is not None:
                out.append((tuple(seg[0]), tuple(seg[1])))
        return out

    def circles(self):
        return [c for c in self.base.circles() if self.xlo <= c[0] <= self.xhi]

    def sample_markings(self, step: float = 1.0) -> np.ndarray:
        mk = self.base.sample_markings(step)
        return mk[(mk[:, 0] >= self.xlo) & (mk[:, 0] <= self.xhi)]


# =========================================================================================
# Colour-agnostic painted-line extraction + colour clustering (ported from the prototype)
# =========================================================================================


def turf_chroma_mask(bgr: np.ndarray) -> np.ndarray:
    """Per-pixel 'is turf-coloured': the turf is the dominant (a, b) chroma mode of the
    frame, whatever colour it is. Lightness ignored (shadows move L, chroma barely)."""
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    body = lab[lab.shape[0] // 5 :, :]  # skip top fifth (sky/stands/HUD)
    hist, ae, be = np.histogram2d(
        body[..., 1].ravel(), body[..., 2].ravel(), bins=32, range=[[0, 255], [0, 255]]
    )
    ai, bi = np.unravel_index(np.argmax(hist), hist.shape)
    a0 = (ae[ai] + ae[ai + 1]) / 2.0
    b0 = (be[bi] + be[bi + 1]) / 2.0
    d = np.sqrt((lab[..., 1] - a0) ** 2 + (lab[..., 2] - b0) ** 2)
    m = (d < 12).astype(np.uint8) * 255
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def _line_responses(bgr: np.ndarray, kernel: int = 21) -> List[np.ndarray]:
    """Per-LAB-channel thin-structure responses (top-hat; + black-hat on chroma channels so
    e.g. blue lines fire too). Kept separate so each channel gets its OWN threshold."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    out = []
    for i in range(3):
        ch = lab[..., i]
        th = cv2.morphologyEx(ch, cv2.MORPH_TOPHAT, k).astype(np.float32)
        if i > 0:
            th = np.maximum(th, cv2.morphologyEx(ch, cv2.MORPH_BLACKHAT, k).astype(np.float32))
        out.append(th)
    return out


_RING_K_OUT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
_RING_K_IN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


def _keep_line_shaped(
    raw: np.ndarray,
    turf: np.ndarray,
    min_area: int = 12,
    fat_radius: int = 5,
    ring_frac: float = 0.5,
) -> np.ndarray:
    """Width filter (painted lines are thin EVERYWHERE; per-component shape stats fail
    because the line network is one giant component) + turf-ring veto per component."""
    fat = cv2.morphologyEx(
        raw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * fat_radius + 1, 2 * fat_radius + 1)),
    )
    fat = cv2.dilate(fat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    thin = cv2.bitwise_and(raw, cv2.bitwise_not(fat))

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(thin, 8)
    out = np.zeros_like(raw)
    H, W = raw.shape
    turf_b = turf > 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        pad = 9
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + bw + pad), min(H, y + bh + pad)
        comp = (lbl[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = (cv2.dilate(comp, _RING_K_OUT) - cv2.dilate(comp, _RING_K_IN)) > 0
        if ring.sum() < 8:
            continue
        if turf_b[y0:y1, x0:x1][ring].mean() >= ring_frac:
            out[lbl == i] = 255
    return out


def extract_line_pixels(
    bgr: np.ndarray, turf_mask: Optional[np.ndarray] = None, kernel: int = 21, nsigma: float = 3.0
) -> np.ndarray:
    """Binary mask of painted-line pixels of ANY colour, verified to sit on turf."""
    if turf_mask is None:
        turf_mask = turf_chroma_mask(bgr)
    fm = turf_mask > 0
    if fm.sum() < 1000:
        fm = np.ones_like(fm)
    raw = np.zeros(bgr.shape[:2], np.uint8)
    for i, resp in enumerate(_line_responses(bgr, kernel)):
        mu, sd = float(resp[fm].mean()), float(resp[fm].std())
        t = max(6.0 if i else 10.0, mu + nsigma * sd)
        raw |= (resp > t).astype(np.uint8) * 255
    raw = cv2.morphologyEx(
        raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    return _keep_line_shaped(raw, turf_mask)


@dataclass
class ColorLineSet:
    center_lab: np.ndarray
    mask: np.ndarray
    n_px: int
    family: str = ""


def hue_family(center_lab) -> str:
    """Coarse hue family used for cross-frame accumulation. Derived from data, only used to
    POOL pixels — never to decide which colour is soccer (the geometry decides that)."""
    a, b = float(center_lab[1]) - 128, float(center_lab[2]) - 128
    if np.hypot(a, b) < 17:
        return "achromatic"
    h = np.degrees(np.arctan2(b, a)) % 360
    if 15 <= h < 75:
        return "red"
    if 75 <= h < 118:
        return "yellow"
    if 118 <= h < 150:
        return "turfjunk"  # greenish residue of the turf itself
    return "other"


def cluster_line_colors(
    bgr: np.ndarray,
    line_mask: np.ndarray,
    min_px: int = 600,
    chroma_thresh: float = 17.0,
    peak_sep_deg: float = 25.0,
    max_sets: int = 5,
) -> List[ColorLineSet]:
    """Split extracted line pixels into paint-colour sets, robust to extreme count imbalance
    (30k white football px vs 2k yellow soccer px): achromatic pixels form one set; chromatic
    pixels are clustered by hue-ANGLE histogram peaks (size-invariant)."""
    ys, xs = np.where(line_mask > 0)
    if len(ys) < min_px:
        return []
    lab_img = cv2.cvtColor(cv2.medianBlur(bgr, 3), cv2.COLOR_BGR2LAB).astype(np.float32)
    samples = lab_img[ys, xs]
    ca = samples[:, 1] - 128.0
    cb = samples[:, 2] - 128.0
    chroma = np.sqrt(ca * ca + cb * cb)
    hue = np.degrees(np.arctan2(cb, ca)) % 360.0

    def make_set(sel: np.ndarray) -> Optional[ColorLineSet]:
        if sel.sum() < min_px:
            return None
        m = np.zeros(line_mask.shape, np.uint8)
        m[ys[sel], xs[sel]] = 255
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for c in range(1, n):  # de-speckle
            if stats[c, cv2.CC_STAT_AREA] < 10:
                m[lbl == c] = 0
        if (m > 0).sum() < min_px:
            return None
        center = samples[sel].mean(0)
        return ColorLineSet(
            center_lab=center, mask=m, n_px=int((m > 0).sum()), family=hue_family(center)
        )

    out: List[ColorLineSet] = []
    achrom = chroma < chroma_thresh
    s = make_set(achrom)
    if s is not None:
        out.append(s)

    ch_sel = ~achrom
    if ch_sel.sum() >= min_px:
        h = hue[ch_sel]
        nbins = 72
        hist, _ = np.histogram(h, bins=nbins, range=(0, 360))
        kern = np.array([1, 4, 6, 4, 1], float)
        kern /= kern.sum()
        sm = np.convolve(np.r_[hist[-2:], hist, hist[:2]].astype(float), kern, "same")[2:-2]
        order = np.argsort(-sm)
        peaks: List[float] = []
        for bi in order:
            if sm[bi] * nbins < 0.5 * len(h):
                break
            c = (bi + 0.5) * (360.0 / nbins)
            if all(min(abs(c - p), 360 - abs(c - p)) >= peak_sep_deg for p in peaks):
                peaks.append(c)
            if len(peaks) >= max_sets - 1:
                break
        if peaks:
            d = np.stack([np.minimum(np.abs(h - p), 360 - np.abs(h - p)) for p in peaks])
            assign = np.argmin(d, 0)
            near = d[assign, np.arange(len(h))] < peak_sep_deg
            for pi in range(len(peaks)):
                sel = np.zeros(len(ys), bool)
                sel[np.where(ch_sel)[0][(assign == pi) & near]] = True
                s = make_set(sel)
                if s is not None:
                    out.append(s)

    out.sort(key=lambda s: -s.n_px)
    return out


# =========================================================================================
# Fast correspondence-free scoring (precomputed markings — the prototype re-sampled the
# template inside every optimizer eval, which was the whole runtime)
# =========================================================================================


class _TemplateScorer:
    """line_support (precision) + recall for one (distance field, marking point set).
    Point sets are precomputed ONCE; the prototype re-sampled the template in a Python
    loop inside every optimizer eval, which dominated its runtime. With the standard
    step-0.5 ls points and step-1.0 recall points the scores are numerically identical
    to line_support.line_support / the prototype's _recall_fn."""

    def __init__(
        self,
        dist_field: np.ndarray,
        pts_m: np.ndarray,
        recall_pts_m: Optional[np.ndarray] = None,
        down: int = 4,
    ):
        self.df = dist_field
        self.h, self.w = dist_field.shape
        self.P = np.hstack([pts_m, np.ones((len(pts_m), 1))])  # N x 3
        rp = pts_m if recall_pts_m is None else recall_pts_m
        self.PR = np.hstack([rp, np.ones((len(rp), 1))])
        self.down = down
        Hs, Ws = max(1, self.h // down), max(1, self.w // down)
        rys, rxs = np.where(dist_field <= 1.5)
        small = np.zeros((Hs, Ws), np.uint8)
        if len(rxs):
            small[np.clip(rys // down, 0, Hs - 1), np.clip(rxs // down, 0, Ws - 1)] = 1
        self._real = small > 0
        self._n_real = int(self._real.sum())
        self._Hs, self._Ws = Hs, Ws
        self._krn = np.ones((3, 3), np.uint8)

    @staticmethod
    def _project(P, H):
        if H is None or not np.all(np.isfinite(H)):
            return None, None
        try:
            Hi = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None, None
        q = P @ Hi.T
        ok = np.abs(q[:, 2]) > 1e-6
        if not ok.any():
            return None, None
        return q[ok, 0] / q[ok, 2], q[ok, 1] / q[ok, 2]

    def ls(self, H, radius: float = 8.0) -> float:
        u, v = self._project(self.P, H)
        if u is None:
            return 0.0
        inb = (u >= 0) & (u < self.w) & (v >= 0) & (v < self.h)
        if inb.sum() < 20:
            return 0.0
        d = self.df[v[inb].astype(int), u[inb].astype(int)]
        return float((d <= radius).mean())

    def recall(self, H) -> float:
        if self._n_real < 30:
            return 0.0
        u, v = self._project(self.PR, H)
        if u is None:
            return 0.0
        u = u / self.down
        v = v / self.down
        m = (u >= 0) & (u < self._Ws) & (v >= 0) & (v < self._Hs)
        pred = np.zeros((self._Hs, self._Ws), np.uint8)
        if m.any():
            pred[v[m].astype(int), u[m].astype(int)] = 1
            pred = cv2.dilate(pred, self._krn)
        return float((pred[self._real] > 0).mean())


def _distance_field(mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)


def _stands_seed_rvec(cam_pos=(0.0, 50.0, -20.0), target=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Generic 'elevated stands camera looking at field centre' rotation seed."""
    pos = np.asarray(cam_pos, float)
    f = np.asarray(target, float) - pos
    f /= np.linalg.norm(f)
    y0 = np.array([0.0, 0.0, 1.0])
    x = np.cross(y0, f)
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    R = np.stack([x, y, f])
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(-1)


def _stands_bounds(seed_rvec, rot_span: float = 0.9):
    """Stands-manifold camera bounds: behind one touchline, elevated (pitch-centred coords,
    z<0 = up). The pitch's 180deg symmetry makes the other side redundant."""
    return [
        (-65.0, 65.0),
        (18.0, 80.0),
        (-45.0, -5.0),
        (seed_rvec[0] - rot_span, seed_rvec[0] + rot_span),
        (seed_rvec[1] - rot_span, seed_rvec[1] + rot_span),
        (seed_rvec[2] - rot_span, seed_rvec[2] + rot_span),
        (np.log(800.0), np.log(6000.0)),
    ]


def image_on_pitch_fraction(H, frame_size, model, pad: float = 10.0, grid: int = 12) -> float:
    """Fraction of an image grid mapping INSIDE the (padded) field — anti-sliver gate."""
    w, h = frame_size
    gx, gy = np.meshgrid(np.linspace(0, w, grid), np.linspace(0, h, grid))
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    q = H @ pts
    ok = q[2] > 1e-6
    x = q[0, ok] / q[2, ok]
    y = q[1, ok] / q[2, ok]
    inside = (x >= -pad) & (x <= model.length + pad) & (y >= -pad) & (y <= model.width + pad)
    return float(inside.sum()) / float(gx.size)


def _hough_segments(mask: np.ndarray, min_len_frac: float = 0.10) -> np.ndarray:
    h = mask.shape[0]
    m = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    segs = cv2.HoughLinesP(
        m, 1, np.pi / 360, threshold=80, minLineLength=int(min_len_frac * h), maxLineGap=30
    )
    return np.zeros((0, 4)) if segs is None else segs[:, 0, :].astype(float)


def _ransac_vp(segs: np.ndarray, iters: int = 400, tol_deg: float = 1.5, seed: int = 0):
    """Robust vanishing point of one family of image segments (length-weighted RANSAC on
    pairwise line intersections). Returns (vp_homogeneous, inlier_mask) or (None, zeros)."""
    n = len(segs)
    if n < 4:
        return None, np.zeros(n, bool)
    rng = np.random.default_rng(seed)
    p0 = np.hstack([segs[:, :2], np.ones((n, 1))])
    p1 = np.hstack([segs[:, 2:], np.ones((n, 1))])
    lines = np.cross(p0, p1)
    mid = (p0 + p1) / 2
    d = segs[:, 2:] - segs[:, :2]
    lens = np.hypot(d[:, 0], d[:, 1])
    dirs = d / lens[:, None]
    cos_tol = np.cos(np.radians(tol_deg))
    best = (None, np.zeros(n, bool), -1.0)
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False)
        v = np.cross(lines[i], lines[j])
        if abs(v[2]) < 1e-9:  # VP at infinity: use pure direction
            vd = v[:2] / (np.hypot(v[0], v[1]) + 1e-12)
            tovp = np.tile(vd, (n, 1))
        else:
            vp = v[:2] / v[2]
            tovp = vp[None, :] - mid[:, :2]
            tovp /= np.hypot(tovp[:, 0], tovp[:, 1])[:, None] + 1e-12
        inl = np.abs((tovp * dirs).sum(1)) > cos_tol
        score = float(lens[inl].sum())
        if score > best[2]:
            best = (v / (abs(v[2]) if abs(v[2]) > 1e-9 else 1.0), inl, score)
    return best[0], best[1]


def _vp_pose_candidates(mask: np.ndarray, principal) -> List[Tuple[np.ndarray, float]]:
    """Closed-form (rotation, log-focal) candidates from two orthogonal vanishing points
    of the dense painted-line network (on a football field: the yard-line fan = world-Y
    family, sidelines + hash rows = world-X). f^2 = -(v1-c).(v2-c); rotation columns from
    K^-1 v. Deterministic — this sidesteps the tiny-basin problem that makes blind 7-DOF
    DE unreliable on narrow-FOV stands footage. Returns [] when the structure is absent."""
    segs = _hough_segments(mask)
    if len(segs) < 6:
        return []
    v1, in1 = _ransac_vp(segs, seed=0)
    rest = segs[~in1]
    if v1 is None or len(rest) < 3:
        return []
    v2, _ = _ransac_vp(rest, seed=1)
    if v2 is None:
        return []
    c = np.asarray(principal, float)
    out: List[Tuple[np.ndarray, float]] = []
    for va, vb in ((v1, v2), (v2, v1)):  # va -> world-Y, vb -> world-X
        if abs(va[2]) < 1e-9 or abs(vb[2]) < 1e-9:
            continue
        pa, pb = va[:2] / va[2], vb[:2] / vb[2]
        f2 = -float(np.dot(pa - c, pb - c))
        if f2 < 500.0**2 or f2 > 9000.0**2:
            continue
        f = float(np.sqrt(f2))
        Ki = np.linalg.inv(np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]]))
        dy = Ki @ np.array([pa[0], pa[1], 1.0])
        dy /= np.linalg.norm(dy)
        dx = Ki @ np.array([pb[0], pb[1], 1.0])
        dx /= np.linalg.norm(dx)
        dx = dx - dy * np.dot(dx, dy)
        nx = np.linalg.norm(dx)
        if nx < 1e-6:
            continue
        dx /= nx
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                X, Y = sx * dx, sy * dy
                M = np.stack([X, Y, np.cross(X, Y)], axis=1)  # cols = world axes in cam
                if M[1, 2] <= 0.02:  # image-down must map toward world-down (+z)
                    continue
                rvec, _ = cv2.Rodrigues(M)
                out.append((rvec.reshape(-1), np.log(f)))
    return out


def _vp_cold_solve(
    mask: np.ndarray,
    scorer: _TemplateScorer,
    principal,
    frame_size,
    model,
    *,
    x_step: float = 0.75,
    polish_starts: int = 5,
):
    """VP-seeded cold anchor: rotation+focal from vanishing points (closed form), then a
    dense scan of the multimodal x-direction (yard-line phase) over a coarse (y, z) grid,
    then jittered Nelder-Mead polish (radius 8 -> 6 anneal) of the best grid cells.
    Deterministic. Returns (H, ls@8, recall, params7) or (None, 0, 0, None)."""
    w, h = frame_size

    def neg(p, radius=8.0):
        H = homography_7dof(p, principal, model)
        if H is None or not np.all(np.isfinite(H)):
            return 5.0
        fp = _feasibility_penalty(H, w, h, model)
        if fp > 0:
            return 1.0 + 0.02 * fp
        return -(scorer.ls(H, radius) * (scorer.recall(H) + 0.05))

    cands = _vp_pose_candidates(mask, principal)
    if not cands:
        return None, 0.0, 0.0, None
    xs = np.arange(-60.0, 60.0 + 1e-9, x_step)
    yzs = [(y, z) for y in (25, 35, 45, 55, 65, 75) for z in (-14, -22, -30, -38, -46)]
    tops = []
    for rvec, logf in cands:
        for y, z in yzs:
            vals = np.array([neg(np.array([x, y, z, rvec[0], rvec[1], rvec[2], logf])) for x in xs])
            i = int(np.argmin(vals))
            if vals[i] < 0:
                tops.append((vals[i], np.array([xs[i], y, z, rvec[0], rvec[1], rvec[2], logf])))
    tops.sort(key=lambda t: t[0])
    best = (None, 0.0, 0.0, None, -1.0)
    rng = np.random.default_rng(2)
    jit = np.array([2.0, 2.0, 2.0, 0.02, 0.02, 0.015, 0.03])
    for _, p0 in tops[:6]:
        cur = p0
        for radius in (8.0, 6.0):
            bx, bv = cur, -neg(cur, radius)
            for t in range(polish_starts):
                s = cur + (rng.normal(0, jit) if t else 0.0)
                r = minimize(
                    lambda p: neg(p, radius),
                    s,
                    method="Nelder-Mead",
                    options={"maxiter": 900, "xatol": 1e-3, "fatol": 1e-3},
                )
                if -r.fun > bv:
                    bv, bx = -r.fun, r.x
            cur = bx
        H = homography_7dof(cur, principal, model)
        if H is None or _feasibility_penalty(H, w, h, model) > 0:
            continue
        if image_on_pitch_fraction(H, frame_size, model) < 0.45:  # sliver/degenerate
            continue
        ls = scorer.ls(H, 8.0)
        rec = scorer.recall(H)
        if ls * rec > best[4]:
            best = (H, ls, rec, cur, ls * rec)
        if best[1] >= 0.50:
            break
    return best[:4]


def _cold_solve(
    scorer: _TemplateScorer,
    principal,
    frame_size,
    model,
    seed_rvec,
    *,
    restarts: int,
    maxiter: int,
    popsize: int,
    polish_starts: int = 4,
    radius_wide: float = 12.0,
    radius_final: float = 8.0,
    early_obj: float = 0.12,
):
    """Radius-annealed global fit on the stands manifold: DE at wide tolerance, then
    Nelder-Mead 7-DOF polish at the final radius. The good basin is found by roughly one
    in three DE restarts, so we run up to `restarts` and STOP EARLY once a restart's
    objective clears `early_obj` (a clearly-locked anchor; garbage basins score ~0.09).
    Returns (H, ls, recall, params7)."""
    w, h = frame_size
    bounds = _stands_bounds(np.asarray(seed_rvec, float))

    def make_obj(radius):
        def obj(p):
            H = homography_7dof(p, principal, model)
            if H is None or not np.all(np.isfinite(H)):
                return 5.0
            fp = _feasibility_penalty(H, w, h, model)
            if fp > 0:
                return 1.0 + 0.02 * fp
            return -(scorer.ls(H, radius) * (scorer.recall(H) + 0.05))

        return obj

    obj_w, obj_f = make_obj(radius_wide), make_obj(radius_final)
    cands = []
    for k in range(restarts):
        r = differential_evolution(
            obj_w,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            mutation=(0.4, 1.2),
            recombination=0.8,
            seed=100 + k,
            polish=True,
            init="sobol",
        )
        cands.append(r.x)
        if -r.fun >= early_obj:
            break

    best = (None, 0.0, 0.0, None, -1.0)
    rng = np.random.default_rng(1)
    jit = np.array([2.0, 2.0, 2.0, 0.02, 0.02, 0.015, 0.03])
    for x0 in cands:
        for t in range(polish_starts):
            s = x0 + (rng.normal(0, jit) if t else 0.0)
            r = minimize(
                obj_f,
                s,
                method="Nelder-Mead",
                options={"maxiter": 700, "xatol": 1e-3, "fatol": 1e-3},
            )
            H = homography_7dof(r.x, principal, model)
            if H is None or _feasibility_penalty(H, w, h, model) > 0:
                continue
            if image_on_pitch_fraction(H, frame_size, model) < 0.45:
                continue
            ls = scorer.ls(H, radius_final)
            cv_ = scorer.recall(H)
            if ls * cv_ > best[4]:
                best = (H, ls, cv_, r.x, ls * cv_)
    return best[:4]


def _warm_solve(
    scorer: _TemplateScorer,
    principal,
    frame_size,
    model,
    position,
    seed_rvec,
    seed_logf,
    *,
    rot_span: float = 0.7,
    logf_span: float = 0.06,
    maxiter: int = 45,
    popsize: int = 12,
    restarts: int = 1,
    radius_wide: float = 10.0,
    radius_final: float = 8.0,
):
    """Position-locked re-acquisition: global DE over rotation only (plus a whisker of
    focal freedom) — 4 dims, fast. Keeping the focal essentially LOCKED is what keeps all
    frames metrically consistent in ground space; a free focal happily compensates a
    wrong yard-line alias and smears the multi-frame accumulation."""
    w, h = frame_size
    pos = np.asarray(position, float)

    def p7(x):
        return np.array([pos[0], pos[1], pos[2], x[0], x[1], x[2], x[3]])

    def make_obj(radius):
        def obj(x):
            H = homography_7dof(p7(x), principal, model)
            if H is None or not np.all(np.isfinite(H)):
                return 5.0
            fp = _feasibility_penalty(H, w, h, model)
            if fp > 0:
                return 1.0 + 0.02 * fp
            return -(scorer.ls(H, radius) * (scorer.recall(H) + 0.05))

        return obj

    bounds = [
        (seed_rvec[0] - rot_span, seed_rvec[0] + rot_span),
        (seed_rvec[1] - rot_span, seed_rvec[1] + rot_span),
        (seed_rvec[2] - rot_span, seed_rvec[2] + rot_span),
        (seed_logf - logf_span, seed_logf + logf_span),
    ]
    obj_w, obj_f = make_obj(radius_wide), make_obj(radius_final)
    cands = []
    for k in range(restarts):
        r = differential_evolution(
            obj_w,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-5,
            mutation=(0.4, 1.1),
            recombination=0.8,
            seed=k,
            polish=True,
            init="sobol",
        )
        cands.append(r.x)
    best = (None, 0.0, 0.0, None, -1.0)
    rng = np.random.default_rng(0)
    for x0 in cands:
        for t in range(3):
            s = x0 + (rng.normal(0, [0.01, 0.01, 0.008, 0.02]) if t else 0.0)
            r = minimize(
                obj_f,
                s,
                method="Nelder-Mead",
                options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3},
            )
            H = homography_7dof(p7(r.x), principal, model)
            if H is None or _feasibility_penalty(H, w, h, model) > 0:
                continue
            if image_on_pitch_fraction(H, frame_size, model) < 0.45:
                continue
            ls = scorer.ls(H, radius_final)
            cv_ = scorer.recall(H)
            if ls * cv_ > best[4]:
                best = (H, ls, cv_, p7(r.x), ls * cv_)
    return best[:4]


# =========================================================================================
# Ground-plane accumulation + soccer-template colour discovery
# =========================================================================================

_GX0, _GX1 = -30.0, 140.0  # ground raster extent (m), football-anchor coordinates
_GY0, _GY1 = -40.0, 90.0
_GRES = 0.5
_GW = int((_GX1 - _GX0) / _GRES)
_GH = int((_GY1 - _GY0) / _GRES)


def _ground_points(H, mask, max_pts: int = 40000):
    ys, xs = np.where(mask > 0)
    if len(ys) > max_pts:
        sel = np.random.default_rng(0).choice(len(ys), max_pts, replace=False)
        ys, xs = ys[sel], xs[sel]
    q = H @ np.stack([xs, ys, np.ones(len(xs))]).astype(float)
    ok = q[2] > 1e-6
    return q[0, ok] / q[2, ok], q[1, ok] / q[2, ok]


def _rasterize(gx, gy) -> np.ndarray:
    r = np.zeros((_GH, _GW), np.float32)
    u = ((gx - _GX0) / _GRES).astype(int)
    v = ((gy - _GY0) / _GRES).astype(int)
    m = (u >= 0) & (u < _GW) & (v >= 0) & (v < _GH)
    np.add.at(r, (v[m], u[m]), 1.0)
    return r


def _soccer_template_points(pitch_L: float, W: float, mode: str, step: float = 0.5) -> np.ndarray:
    """Observed-half soccer template markings (see _half_template_kernel)."""
    mk = PitchModel(pitch_L, float(W)).sample_markings(step)
    return mk[mk[:, 0] >= pitch_L - 50.0] if mode == "east" else mk[mk[:, 0] <= 50.0]


def _half_template_kernel(pitch_L: float, W: float, mode: str, radius_m: float = 1.2):
    """Rasterized point-count kernel of ONE HALF of the soccer template (for matchTemplate
    correlation) + its dilated binary (recall term). Only the OBSERVED half is scored —
    scoring the never-seen half lets in-coverage clutter pull the fit (the identifiability
    lesson the prototype learned). Returns (K, K_dil, x_local_offset): the FULL template's
    x0 = raster_x_of_kernel_origin - x_local_offset."""
    mk = PitchModel(pitch_L, float(W)).sample_markings(0.5)
    if mode == "east":
        half = mk[mk[:, 0] >= pitch_L - 50.0]
        xoff = pitch_L - 50.0
    else:
        half = mk[mk[:, 0] <= 50.0]
        xoff = 0.0
    loc = half - np.array([xoff, 0.0])
    kw = int(np.ceil(loc[:, 0].max() / _GRES)) + 3
    kh = int(np.ceil(W / _GRES)) + 3
    K = np.zeros((kh, kw), np.float32)
    u = np.clip((loc[:, 0] / _GRES).astype(int), 0, kw - 1)
    v = np.clip((loc[:, 1] / _GRES).astype(int), 0, kh - 1)
    np.add.at(K, (v, u), 1.0)
    r = int(round(radius_m / _GRES))
    K_dil = cv2.dilate(
        (K > 0).astype(np.float32),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)),
    )
    return K, K_dil, xoff


def _fit_family_ground(
    binary,
    coverage,
    pitch_L: float,
    radius_m: float = 1.2,
    w_grid=None,
    min_cov_pts: int = 300,
    y0_range=(-6.0, 5.0),
    fb_length: float = 120 * YD,
    fb_width: float = 160 / 3 * YD,
    x_margin: float = 4.0,
    y_far_margin: float = 12.0,
):
    """Fit the soccer template (L pinned, W on a grid, translation exhaustive via FFT
    correlation, east/west observed-half scoring) to one colour family's accumulated
    ground raster, scored ONLY inside the observed coverage. Returns best dict or None.

    Physical priors (the soccer paint must lie ON the turf): the far touchline can sit at
    most a few meters beyond the football sideline (y0_range — the turf strip before the
    fence is narrow), and the fitted goal line at most x_margin beyond the end line.
    Without these the weakly-observed far touchline drifts several meters into the fence
    while still scoring (parallel junk structures)."""
    if w_grid is None:
        w_grid = np.arange(52.0, 72.5, 0.5)
    near = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3) * _GRES <= radius_m
    covf = (coverage > 0).astype(np.float32)
    scov = (near & (coverage > 0)).astype(np.float32)
    colf = binary.astype(np.float32)
    n_col = float(colf.sum())
    if n_col < 100:
        return None
    best = None
    for W in w_grid:
        for mode in ("east", "west"):
            K, K_dil, xoff = _half_template_kernel(pitch_L, W, mode, radius_m)
            if K.shape[0] >= _GH or K.shape[1] >= _GW:
                continue
            n_cov = cv2.matchTemplate(covf, K, cv2.TM_CCORR)
            n_sup = cv2.matchTemplate(scov, K, cv2.TM_CCORR)
            n_rec = cv2.matchTemplate(colf, K_dil, cv2.TM_CCORR)
            support = np.where(n_cov >= min_cov_pts, n_sup / np.maximum(n_cov, 1.0), 0.0)
            recall = n_rec / n_col
            score = support * recall
            # physical bands: BOTH touchlines near/on the turf strips; goal line on field
            y0_hi = min(y0_range[1], fb_width + y_far_margin - W)  # near touchline bound
            if y0_hi < y0_range[0]:
                continue  # this W can't fit
            dy_lo = int(np.clip((y0_range[0] - _GY0) / _GRES, 0, score.shape[0] - 1))
            dy_hi = int(np.clip((y0_hi - _GY0) / _GRES, 0, score.shape[0] - 1))
            m = np.zeros_like(score, bool)
            m[dy_lo : dy_hi + 1, :] = True
            if mode == "east":  # x0 = _GX0 + dx*RES - xoff; goal at x0 + pitch_L
                dx_hi = (fb_length + x_margin - pitch_L + xoff - _GX0) / _GRES
                m[:, int(np.clip(dx_hi, 0, score.shape[1] - 1)) + 1 :] = False
            else:  # west goal line at x0 >= -x_margin
                dx_lo = (-x_margin + xoff - _GX0) / _GRES
                m[:, : int(np.clip(dx_lo, 0, score.shape[1] - 1))] = False
            score[~m] = 0.0
            j = int(np.argmax(score))
            dy, dx = np.unravel_index(j, score.shape)
            if best is None or score[dy, dx] > best["score"]:
                best = {
                    "score": float(score[dy, dx]),
                    "support": float(support[dy, dx]),
                    "recall": float(recall[dy, dx]),
                    "x0": _GX0 + dx * _GRES - xoff,
                    "y0": _GY0 + dy * _GRES,
                    "W": float(W),
                    "mode": mode,
                }
    return best


def _ls_points(H, dist_field, pts_m, radius: float = 10.0) -> float:
    """line_support of an arbitrary point set (meters) against a frame's distance field."""
    if H is None or dist_field is None or not np.all(np.isfinite(H)):
        return 0.0
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 0.0
    h, w = dist_field.shape
    q = np.hstack([pts_m, np.ones((len(pts_m), 1))]) @ Hi.T
    ok = np.abs(q[:, 2]) > 1e-6
    if not ok.any():
        return 0.0
    u = q[ok, 0] / q[ok, 2]
    v = q[ok, 1] / q[ok, 2]
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if inb.sum() < 20:
        return 0.0
    d = dist_field[v[inb].astype(int), u[inb].astype(int)]
    return float((d <= radius).mean())


def _football_score_of_family(binary, coverage, fm: FootballModel, radius_m: float = 1.2) -> float:
    """How well the family fits the FOOTBALL template (identity placement in anchor
    coordinates). Football-paint families fit this far better than soccer geometry."""
    near = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3) * _GRES <= radius_m
    mk = fm.sample_markings(0.5)
    u = ((mk[:, 0] - _GX0) / _GRES).astype(int)
    v = ((mk[:, 1] - _GY0) / _GRES).astype(int)
    inb = (u >= 0) & (u < _GW) & (v >= 0) & (v < _GH)
    if inb.sum() < 50:
        return 0.0
    cov = coverage[v[inb], u[inb]] > 0
    if cov.sum() < 50:
        return 0.0
    support = float(near[v[inb][cov], u[inb][cov]].mean())
    tr = np.zeros((_GH, _GW), np.uint8)
    tr[v[inb], u[inb]] = 1
    r = int(round(radius_m / _GRES))
    tr = cv2.dilate(tr, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return 0.0
    recall = float((tr[ys, xs] > 0).mean())
    return support * recall


# =========================================================================================
# Joint multi-frame bundle (shared camera position + focal, per-frame rotation)
# =========================================================================================


def _refine_template_shift(
    anchors: List[dict], t_range: float = 2.5, step: float = 0.5, radius: float = 12.0
) -> Tuple[float, float]:
    """Phase 1 of the bundle: refine the ground-fit TRANSLATION against the per-frame
    image-space soccer paint with all cameras FIXED. The accumulated 0.5 m raster is only
    good to ~1-2 m; the image evidence is ~10x crisper (at these focals 1 m is ~30 px).
    Exhaustive (dx, dy) grid — the projection of the shifted template is an incremental
    vector add, so the whole grid costs ~a second. t_range is deliberately SMALL (sub-
    raster refinement only): a wide search can hop the whole template onto a parallel
    yellow structure and destabilize the placement. Returns the best (dx, dy)."""
    per = []
    for af in anchors:
        sc = af["soc_scorer"]
        if sc is None:
            continue
        try:
            Hi = np.linalg.inv(af["H"])
        except np.linalg.LinAlgError:
            continue
        Q0 = sc.P @ Hi.T  # N x 3 projected template (un-normalized)
        per.append((sc, Q0, Hi.T[0], Hi.T[1]))
    if not per:
        return 0.0, 0.0
    deltas = np.arange(-t_range, t_range + 1e-9, step)
    best = (0.0, 0.0, -1.0)
    for dx in deltas:
        for dy in deltas:
            tot = 0.0
            for sc, Q0, hx, hy in per:
                q = Q0 + dx * hx + dy * hy
                ok = np.abs(q[:, 2]) > 1e-6
                if not ok.any():
                    continue
                u = q[ok, 0] / q[ok, 2]
                v = q[ok, 1] / q[ok, 2]
                inb = (u >= 0) & (u < sc.w) & (v >= 0) & (v < sc.h)
                if inb.sum() < 20:
                    continue
                d = sc.df[v[inb].astype(int), u[inb].astype(int)]
                tot += float((d <= radius).mean())
            if tot > best[2]:
                best = (float(dx), float(dy), tot)
    return best[0], best[1]


def _per_frame_shift_refine(anchors: List[dict], radius: float = 10.0, accept_gain: float = 0.04):
    """Per-frame residual drift correction against the DISCOVERED soccer paint: each
    frame searches a small fine-shift grid of template offsets maximizing its own
    winner-colour line support. Accepted only on a clear gain, so a frame with scant
    colour evidence keeps its flow-consensus alignment. Updates a['shift'] and the
    soc_scorer template points in place."""
    for af in anchors:
        sc = af.get("soc_scorer")
        if sc is None:
            continue
        try:
            Hi = np.linalg.inv(af["H"])
        except np.linalg.LinAlgError:
            continue
        Q0 = sc.P @ Hi.T
        hx, hy = Hi.T[0], Hi.T[1]

        def ls_at(dx, dy):
            q = Q0 + dx * hx + dy * hy
            ok = np.abs(q[:, 2]) > 1e-6
            if not ok.any():
                return 0.0
            u = q[ok, 0] / q[ok, 2]
            v = q[ok, 1] / q[ok, 2]
            inb = (u >= 0) & (u < sc.w) & (v >= 0) & (v < sc.h)
            if inb.sum() < 20:
                return 0.0
            d = sc.df[v[inb].astype(int), u[inb].astype(int)]
            return float((d <= radius).mean())

        base = ls_at(0.0, 0.0)
        best = (0.0, 0.0, base)
        # FINE shifts only: the yard-line alias is already pinned by the flow chains,
        # and letting a frame jump an alias here can snap it onto a parallel structure
        for dx in np.arange(-1.5, 1.5 + 1e-9, 0.5):
            for dy in np.arange(-3.0, 3.0 + 1e-9, 0.5):
                s = ls_at(dx, dy)
                if s >= base + accept_gain and s > best[2]:
                    best = (dx, dy, s)
        dx, dy, s = best
        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
            sc.P[:, :2] += np.array([dx, dy])
            af["shift"] = (af["shift"][0] - dx, af["shift"][1] - dy)


def _bundle_refine(
    anchors: List[dict],
    position,
    logf,
    principal,
    frame_size,
    fm: FootballModel,
    *,
    rounds: int = 2,
    fb_weight: float = 0.5,
    radius: float = 8.0,
    per_frame_starts: int = 3,
    fb_floor_frac: float = 0.75,
):
    """Phase 2 of the joint bundle: alternating refinement over the anchored frames —
    per-frame rotation (3 DOF NM) against the discovered soccer paint + football-anchor
    regularizer, then shared camera position + focal (4 DOF NM) on the summed objective.
    A football-support FLOOR rejects any per-frame update that erodes the frame's anchor
    registration (a phantom soccer basin can otherwise destroy a frame).
    anchors[i] needs: 'H', 'rvec', 'soc_scorer' (may be None), 'fb_scorer'."""
    w, h = frame_size
    pos = np.asarray(position, float).copy()
    lf = float(logf)
    for af in anchors:
        af["fb_base"] = af["fb_scorer"].ls(af["H"], radius)

    def parts(af, rvec, p, l, rad):
        H = homography_7dof(
            np.array([p[0], p[1], p[2], rvec[0], rvec[1], rvec[2], l]), principal, fm
        )
        if H is None or not np.all(np.isfinite(H)) or _feasibility_penalty(H, w, h, fm) > 0:
            return None, 0.0, 0.0
        fb = af["fb_scorer"].ls(H, rad)
        soc = af["soc_scorer"].ls(H, rad) if af["soc_scorer"] is not None else 0.0
        return H, soc, fb

    def J(af, rvec, p, l, rad):
        H, soc, fb = parts(af, rvec, p, l, rad)
        if H is None:
            return -1.0
        return soc + fb_weight * fb

    rng = np.random.default_rng(0)
    radii = [1.5 * radius] + [radius] * max(1, rounds - 1)
    for rad in radii:
        for af in anchors:  # per-frame rotation
            x0 = np.asarray(af["rvec"], float)
            best_x, best_v = x0, J(af, x0, pos, lf, rad)
            for t in range(per_frame_starts):
                s = x0 + (rng.normal(0, [0.004, 0.004, 0.003]) if t else 0.0)
                r = minimize(
                    lambda x: -J(af, x, pos, lf, rad),
                    s,
                    method="Nelder-Mead",
                    options={"maxiter": 250, "xatol": 5e-4, "fatol": 5e-4},
                )
                if -r.fun > best_v:
                    _, _, fb_new = parts(af, r.x, pos, lf, radius)
                    if fb_new >= fb_floor_frac * af["fb_base"]:  # keep anchor intact
                        best_v, best_x = -r.fun, r.x
            af["rvec"] = best_x

        def neg_total(q):  # shared position + focal
            return -sum(J(af, af["rvec"], q[:3], q[3], rad) for af in anchors)

        q0 = np.array([pos[0], pos[1], pos[2], lf])
        r = minimize(
            neg_total,
            q0,
            method="Nelder-Mead",
            options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-3},
        )
        if -r.fun > -neg_total(q0):
            ok = all(
                parts(af, af["rvec"], r.x[:3], r.x[3], radius)[2] >= fb_floor_frac * af["fb_base"]
                for af in anchors
            )
            if ok:
                pos, lf = r.x[:3].copy(), float(r.x[3])
    return pos, lf


def _flow_align_anchors(
    anchors: List[dict],
    Gs: List[Optional[np.ndarray]],
    frame_size,
    yd5: float = 5 * YD,
    dy_max: float = 8.0,
) -> Tuple[List[dict], int]:
    """Tie the anchors together through the VIDEO itself: chain the frame-to-frame flow
    homographies between consecutive anchors and measure each anchor's ground displacement
    against the flow prediction. The x displacement is snapped to the 5-yd yard-line alias
    (the only discrete freedom the position-locked warm solve has); y is taken as measured.
    This replaces the noisy per-frame chroma alignment — sparse colour evidence made those
    shift estimates scatter by many meters. Anchors unreachable by flow are dropped.
    Sets a['shift'] (frame coords + shift = reference coords); returns (kept, dropped)."""
    if not anchors:
        return [], 0
    order = sorted(anchors, key=lambda a: a["cand"]["entry_i"])
    ref = max(order, key=lambda a: a["ls"])
    ri = order.index(ref)
    w, h = frame_size
    us = np.array(
        [
            [w * 0.3, h * 0.45, 1.0],
            [w * 0.7, h * 0.45, 1.0],
            [w * 0.5, h * 0.35, 1.0],
            [w * 0.5, h * 0.65, 1.0],
        ]
    ).T

    def ground(H):
        q = H @ us
        return q[:2] / q[2]

    def chain(Hc, e_from, e_to):
        """Chain a pitch-mapping H from entry e_from to e_to through the flow (either
        direction). Returns None when the flow broke anywhere along the way."""
        if e_to > e_from:
            for i in range(e_from + 1, e_to + 1):
                if Gs[i] is None:
                    return None
                try:
                    Hc = Hc @ np.linalg.inv(Gs[i])
                except np.linalg.LinAlgError:
                    return None
        else:
            for i in range(e_from, e_to, -1):
                if Gs[i] is None:
                    return None
                Hc = Hc @ Gs[i]
        return Hc

    ref["shift"] = (0.0, 0.0)
    kept = [ref]
    dropped = 0
    for direction in (1, -1):
        prev = ref
        j = ri + direction
        while 0 <= j < len(order):
            cur = order[j]
            Hp = (
                np.array([[1.0, 0, prev["shift"][0]], [0, 1.0, prev["shift"][1]], [0, 0, 1.0]])
                @ prev["H"]
            )
            Hc = chain(Hp, prev["cand"]["entry_i"], cur["cand"]["entry_i"])
            if Hc is None:
                dropped += 1
                j += direction
                continue  # prev stays; try to bridge to the next one
            d = (ground(Hc) - ground(cur["H"])).mean(axis=1)
            sx = round(float(d[0]) / yd5) * yd5
            sy = float(np.clip(d[1], -dy_max, dy_max))
            cur["shift"] = (sx, sy)
            kept.append(cur)
            prev = cur
            j += direction
    return kept, dropped


def _flow_propagate(
    entries: List[dict],
    Gs: List[Optional[np.ndarray]],
    used_anchors: Dict[int, np.ndarray],
    pitch: PitchModel,
    max_gap: int = 60,
) -> Tuple[Dict[int, Optional[np.ndarray]], int]:
    """Propagate verified anchors across the clip by chaining Lucas-Kanade frame-to-frame
    homographies (H_i = H_{i-1} @ inv(G_i), the Phase-B recipe). This camera PANS almost
    continuously, so static copying alone covers almost nothing; flow chaining holds the
    calibration through the pans, capped at max_gap entries from the nearest anchor.
    used_anchors: {entry_index: H_processing_px->pitch}. Returns ({idx: H or None}, n_flow)."""
    n = len(entries)
    result: Dict[int, Optional[np.ndarray]] = {e["idx"]: None for e in entries}
    if not used_anchors:
        return result, 0
    aidx = np.array(sorted(used_anchors))
    nearest = {i: int(aidx[np.argmin(np.abs(aidx - i))]) for i in range(n)}
    n_flow = 0
    for a, Ha in used_anchors.items():
        result[entries[a]["idx"]] = Ha
        H = Ha.copy()
        for i in range(a + 1, min(a + max_gap, n - 1) + 1):  # forward chain
            if Gs[i] is None:
                break
            try:
                H = H @ np.linalg.inv(Gs[i])
            except np.linalg.LinAlgError:
                break
            if nearest[i] == a and i not in used_anchors:
                if homography_plausible(H, pitch):
                    result[entries[i]["idx"]] = H.copy()
                    n_flow += 1
        H = Ha.copy()
        for i in range(a, max(a - max_gap, 0), -1):  # backward chain
            if Gs[i] is None:
                break
            H = H @ Gs[i]
            j = i - 1
            if nearest[j] == a and j not in used_anchors:
                if homography_plausible(H, pitch):
                    result[entries[j]["idx"]] = H.copy()
                    n_flow += 1
    return result, n_flow


# =========================================================================================
# Entry point
# =========================================================================================

_EFFORT = {
    # candidates, cold(restarts, maxiter, popsize), warm(restarts, maxiter, popsize), rounds
    # cold restarts are a CAP: _cold_solve stops early once a restart locks the anchor
    1: dict(n_cand=6, n_probe=24, cold=(6, 110, 18), bank=(2, 60, 12), warm=(1, 45, 12), rounds=3),
    2: dict(n_cand=8, n_probe=32, cold=(8, 110, 18), bank=(3, 80, 14), warm=(2, 60, 14), rounds=3),
    3: dict(
        n_cand=10, n_probe=40, cold=(10, 130, 18), bank=(3, 100, 16), warm=(2, 80, 16), rounds=4
    ),
}


def calibrate_multisport(
    video_path: str,
    pitch: PitchModel = STANDARD_PITCH,
    max_frames: Optional[int] = None,
    stride: int = 3,
    progress: Optional[Callable[[float], None]] = None,
    effort: int = 1,
    start_frame: int = 0,
) -> Tuple[Dict[int, Optional[np.ndarray]], dict]:
    """Colour-agnostic multi-sport calibration of a whole clip. See module docstring.

    Returns ({original_frame_idx: H or None}, stats). H maps original-frame pixels ->
    pitch meters (corner-origin). On a non-multi-sport venue returns
    ({}, {'not_multisport': True, ...}). Memory exhaustion is propagated to the worker."""
    t_start = time.time()
    cfg = _EFFORT.get(int(np.clip(effort, 1, 3)), _EFFORT[1])
    stats: dict = {
        "multisport": False,
        "winning_colour": None,
        "n_anchored": 0,
        "coverage_pct": 0.0,
        "per_frame": {},
        "partial_length": False,
    }

    def _prog(f):
        if progress is not None:
            try:
                progress(float(np.clip(f, 0.0, 1.0)))
            except Exception:
                pass

    try:
        return _calibrate_impl(
            video_path, pitch, max_frames, stride, _prog, cfg, stats, t_start, start_frame
        )
    except Exception as e:
        if is_memory_error(e):
            raise
        stats["error"] = f"{type(e).__name__}: {e}"
        stats["runtime_s"] = round(time.time() - t_start, 1)
        return {}, stats


def _calibrate_impl(video_path, pitch, max_frames, stride, _prog, cfg, stats, t_start, start_frame):
    fm = FootballModel()

    # ---- Pass A: decode, measure motion, keep one low-motion candidate frame per bin ----
    with open_capture(video_path) as cap:
        start, stop = frame_window(cap.get(cv2.CAP_PROP_FRAME_COUNT), max_frames, start_frame)
        total = stop - start if stop is not None else 0
        seek_capture(cap, start)
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        n_bins = cfg["n_cand"]
        n_entries_est = max(1, (total or 10**6) // stride)

        scale = min(1.0, _PROC_MAX_W / float(ow))
        from soccer_analyzer.calibration.mapper import _FlowTracker

        ft = _FlowTracker(target_w=960)
        Gs: List[Optional[np.ndarray]] = []  # per entry: flow H from prev entry
        entries: List[dict] = []
        bin_best: Dict[int, Tuple[float, int, np.ndarray]] = {}  # bin -> (motion, entry_i, frame)
        prev = None
        idx = start
        while True:
            if stop is not None and idx >= stop:
                break
            if (idx - start) % stride != 0:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            g = _gray_small(frame)
            motion = 0.0 if prev is None else float(np.mean(cv2.absdiff(g, prev)))
            prev = g
            ei = len(entries)
            if ei == 0:
                ft.reset(frame)
                Gs.append(None)
            else:
                Gs.append(ft.step(frame))
            entries.append({"idx": idx, "Hv": None, "motion": motion})
            b = min(n_bins - 1, ei * n_bins // n_entries_est)
            if b not in bin_best or motion < bin_best[b][0]:
                bin_best[b] = (motion, ei, frame.copy())
            idx += 1
            if ei % 40 == 0 and total:
                _prog(0.22 * (idx - start) / total)
    if not entries:
        stats["error"] = "no frames decoded"
        return {}, stats
    stats["frames"] = len(entries)
    ph, pw = bin_best[min(bin_best)][2].shape[:2]
    principal = (pw / 2.0, ph / 2.0)
    frame_size = (pw, ph)
    _prog(0.22)

    # ---- extraction + multi-sport gate ---------------------------------------------------
    candidates = []  # {entry_i, idx, frame, sets}
    for b in sorted(bin_best):
        _, ei, frm = bin_best[b]
        sets = cluster_line_colors(frm, extract_line_pixels(frm, turf_chroma_mask(frm)))
        if len(sets) >= 2 and sets[0].n_px >= 6000:
            candidates.append(
                {"entry_i": ei, "idx": entries[ei]["idx"], "frame": frm, "sets": sets}
            )
    n_chrom = sum(
        1
        for c in candidates
        if any(s.family in ("red", "yellow", "other") and s.n_px >= 900 for s in c["sets"])
    )
    stats["candidates"] = len(candidates)
    if len(candidates) < 2 or n_chrom < max(2, (len(candidates) + 1) // 2):
        stats["not_multisport"] = True
        stats["reason"] = (
            "no consistent secondary paint colour discovered "
            f"({n_chrom}/{len(candidates)} candidate frames)"
        )
        stats["runtime_s"] = round(time.time() - t_start, 1)
        return {}, stats
    stats["multisport"] = True
    _prog(0.30)

    # ---- template-bank cold anchor on the densest colour of the richest frame -----------
    candidates.sort(key=lambda c: -c["sets"][0].n_px)
    seed = _stands_seed_rvec()
    fb_pts = fm.sample_markings(0.5)
    fb_rec = fm.sample_markings(1.0)
    soc_pts = pitch.sample_markings(0.5)
    soc_rec = pitch.sample_markings(1.0)
    r_fb, m_fb, p_fb = cfg["cold"]
    r_sc, m_sc, p_sc = cfg["bank"]
    # Cold anchor: the VP path (deterministic, seconds) first, across candidates; blind DE
    # (stochastic, tiny basin on narrow-FOV footage) only as fallback on the densest frame.
    cold = candidates[0]
    H_fb, ls_fb, rec_fb, params_fb = None, 0.0, 0.0, None
    cold_source = "vp"
    for cand in candidates[:4]:
        sc_c = _TemplateScorer(_distance_field(cand["sets"][0].mask), fb_pts, fb_rec)
        H_c, ls_c, rec_c, p_c = _vp_cold_solve(
            cand["sets"][0].mask, sc_c, principal, frame_size, fm
        )
        if H_c is not None and ls_c > ls_fb:
            cold, H_fb, ls_fb, rec_fb, params_fb = cand, H_c, ls_c, rec_c, p_c
        if ls_fb >= 0.45:
            break
    if ls_fb < 0.45:
        cold_source = "de"
        for cand in candidates[:2]:
            H_c, ls_c, rec_c, p_c = _cold_solve(
                _TemplateScorer(_distance_field(cand["sets"][0].mask), fb_pts, fb_rec),
                principal,
                frame_size,
                fm,
                seed,
                restarts=r_fb,
                maxiter=m_fb,
                popsize=p_fb,
            )
            if H_c is not None and ls_c > ls_fb:
                cold, H_fb, ls_fb, rec_fb, params_fb = cand, H_c, ls_c, rec_c, p_c
            if ls_fb >= 0.45:
                break
    dense_df = _distance_field(cold["sets"][0].mask)
    _prog(0.42)
    # template bank: the soccer template gets the same VP seeding + a small DE budget
    sc_soc = _TemplateScorer(dense_df, soc_pts, soc_rec)
    H_sc, ls_sc, rec_sc, _ = _vp_cold_solve(
        cold["sets"][0].mask, sc_soc, principal, frame_size, pitch
    )
    H_sc2, ls_sc2, rec_sc2, _ = _cold_solve(
        sc_soc, principal, frame_size, pitch, seed, restarts=r_sc, maxiter=m_sc, popsize=p_sc
    )
    if H_sc is None or ls_sc2 * rec_sc2 > ls_sc * rec_sc:
        H_sc, ls_sc, rec_sc = H_sc2, ls_sc2, rec_sc2
    score_fb = (
        ls_fb * rec_fb
        if (H_fb is not None and image_on_pitch_fraction(H_fb, frame_size, fm) >= 0.45)
        else 0.0
    )
    score_sc = (
        ls_sc * rec_sc
        if (H_sc is not None and image_on_pitch_fraction(H_sc, frame_size, pitch) >= 0.45)
        else 0.0
    )
    stats["template_bank"] = {
        "football": round(score_fb, 4),
        "soccer": round(score_sc, 4),
        "anchor_ls": round(ls_fb, 3),
        "anchor_source": cold_source,
        "anchor_frame": cold["idx"],
    }
    if score_fb <= 0.0 or ls_fb < 0.40:
        if score_sc > 0.0:
            stats["not_multisport"] = True
            stats["reason"] = "soccer template explains the densest paint (normal soccer venue)"
        else:
            stats["error"] = f"football anchor failed (ls={ls_fb:.2f})"
        stats["runtime_s"] = round(time.time() - t_start, 1)
        return {}, stats
    if score_sc >= score_fb:
        stats["not_multisport"] = True
        stats["reason"] = "soccer template explains the densest paint (normal soccer venue)"
        stats["runtime_s"] = round(time.time() - t_start, 1)
        return {}, stats
    _prog(0.5)

    # ---- warm-track the anchor across candidates + probes spread over the clip ----------
    # More anchored viewpoints = richer ground accumulation (different pans reveal
    # different parts of the soccer paint — the single-frame fit is ambiguous) AND more
    # propagation anchors. Probes are extra frames evenly spaced across the clip.
    position = params_fb[:3]
    logf = float(params_fb[6])
    anchors = [{"cand": cold, "H": H_fb, "p7": np.asarray(params_fb, float), "ls": ls_fb}]
    r_w, m_w, p_w = cfg["warm"]
    warm_jobs = [c for c in candidates if c is not cold]
    have_ei = {c["entry_i"] for c in candidates}
    probe_eis = [
        int(e) for e in np.linspace(0, len(entries) - 1, cfg["n_probe"]) if int(e) not in have_ei
    ]
    cap = cv2.VideoCapture(video_path)
    pos = 0  # sequential reads: a seek lands on the wrong frame for some codecs
    for ei in sorted(probe_eis, key=lambda e: entries[e]["idx"]):
        target = entries[ei]["idx"]
        if target < pos:
            continue
        pos += skip_frames(cap, target - pos)
        ok, frm = cap.read() if pos == target else (False, None)
        if not ok:
            break
        pos += 1
        if scale < 1.0:
            frm = cv2.resize(frm, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sets = cluster_line_colors(frm, extract_line_pixels(frm, turf_chroma_mask(frm)))
        if len(sets) >= 1 and sets[0].n_px >= 6000:
            warm_jobs.append({"entry_i": ei, "idx": entries[ei]["idx"], "frame": frm, "sets": sets})
    cap.release()
    for i, cand in enumerate(warm_jobs):
        df = _distance_field(cand["sets"][0].mask)
        H, ls, rec, p7 = _warm_solve(
            _TemplateScorer(df, fb_pts, fb_rec),
            principal,
            frame_size,
            fm,
            position,
            params_fb[3:6],
            logf,
            restarts=r_w,
            maxiter=m_w,
            popsize=p_w,
        )
        if H is not None and ls >= 0.42 and image_on_pitch_fraction(H, frame_size, fm) >= 0.45:
            anchors.append({"cand": cand, "H": H, "p7": np.asarray(p7, float), "ls": ls})
        _prog(0.5 + 0.2 * (i + 1) / max(1, len(warm_jobs)))
    stats["anchor_ls"] = {a["cand"]["idx"]: round(a["ls"], 3) for a in anchors}

    # ---- inter-frame alias alignment + accumulation in ground meters --------------------
    # warm anchors can disagree by the 5-yd yard-line alias (the white fan is periodic);
    # the flow chains through the video tie them together deterministically
    kept, dropped = _flow_align_anchors(anchors, Gs, frame_size)
    stats["alias_dropped"] = dropped
    stats["anchor_shifts"] = {
        a["cand"]["idx"]: (round(a["shift"][0], 2), round(a["shift"][1], 2)) for a in kept
    }
    if not kept:
        stats["error"] = "no anchor frame carries consistent secondary paint"
        stats["runtime_s"] = round(time.time() - t_start, 1)
        return {}, stats
    acc: Dict[str, np.ndarray] = {}
    for a in kept:
        sx, sy = a["shift"]
        for s in a["cand"]["sets"]:
            gx, gy = _ground_points(a["H"], s.mask)
            acc.setdefault(s.family, np.zeros((_GH, _GW), np.float32))
            acc[s.family] += _rasterize(gx + sx, gy + sy)
    _prog(0.75)

    # ---- discover the soccer colour in ground space --------------------------------------
    dense_fam = cold["sets"][0].family
    cov_src = (acc.get(dense_fam, np.zeros((_GH, _GW), np.float32)) > 0.5).astype(np.uint8)
    coverage = cv2.dilate(cov_src, np.ones((int(25 / _GRES),) * 2, np.uint8))
    table = {}
    for fam, raster in acc.items():
        if fam == "turfjunk":
            continue
        binary = (raster > 0.5).astype(np.uint8)
        fit = _fit_family_ground(binary, coverage, pitch.length)
        if fit is None:
            continue
        fit["football"] = _football_score_of_family(binary, coverage, fm)
        fit["margin"] = fit["score"] - fit["football"]
        table[fam] = fit
    stats["ground_table"] = {
        f: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}
        for f, t in table.items()
    }
    winners = {f: t for f, t in table.items() if t["margin"] > 0 and t["score"] >= 0.03}
    if not winners:
        stats["error"] = "no colour family fits soccer geometry better than football"
        stats["runtime_s"] = round(time.time() - t_start, 1)
        return {}, stats
    win = max(winners, key=lambda f: winners[f]["score"])
    stats["winning_colour"] = win

    # ---- placement selection by CROSS-FRAME validation -----------------------------------
    # The raster fit can self-score on parallel junk. Generate candidate placements (the
    # accumulated fit + per-frame fits on the chroma-richest strong anchors) and pick the
    # one whose template lands on the winner-colour paint across the MOST frames — the
    # exact quantity the final per-frame gate measures.
    for a in kept:
        win_sets = [s for s in a["cand"]["sets"] if s.family == win]
        a["win_df"] = _distance_field(win_sets[0].mask) if win_sets else None
        a["win_px"] = win_sets[0].n_px if win_sets else 0
    cand_fits = [table[win]]
    rich = sorted(
        [a for a in kept if a["win_df"] is not None and a["ls"] >= 0.50], key=lambda a: -a["win_px"]
    )
    for a in rich[:3]:
        sx, sy = a["shift"]
        ws = [s for s in a["cand"]["sets"] if s.family == win][0]
        gx, gy = _ground_points(a["H"], ws.mask)
        binary = (_rasterize(gx + sx, gy + sy) > 0.5).astype(np.uint8)
        gxd, gyd = _ground_points(a["H"], a["cand"]["sets"][0].mask)
        covf = cv2.dilate(
            (_rasterize(gxd + sx, gyd + sy) > 0.5).astype(np.uint8),
            np.ones((int(25 / _GRES),) * 2, np.uint8),
        )
        f = _fit_family_ground(binary, covf, pitch.length)
        if f is not None:
            cand_fits.append(f)

    def _placement_score(x0_, y0_, W_, mode_, radius=10.0):
        mk = _soccer_template_points(pitch.length, W_, mode_)
        tot = 0.0
        for a in kept:
            if a["win_df"] is None:
                continue
            sx, sy = a["shift"]
            tot += _ls_points(
                a["H"], a["win_df"], mk + np.array([x0_ - sx, y0_ - sy]), radius=radius
            )
        return tot

    # refine each candidate CONTINUOUSLY on the cross-frame objective (radius-annealed),
    # then keep the best — the 0.5 m raster grid alone leaves a couple meters of slack
    refined = []
    for f in cand_fits:
        cur = np.array([f["x0"], f["y0"], f["W"]])
        for rad in (20.0, 10.0):
            r = minimize(
                lambda p: -_placement_score(
                    p[0], p[1], float(np.clip(p[2], 50.0, 75.0)), f["mode"], rad
                ),
                cur,
                method="Nelder-Mead",
                options={"maxiter": 250, "xatol": 5e-3, "fatol": 5e-3},
            )
            cur = r.x
        W_ = float(np.clip(cur[2], 50.0, 75.0))
        sc = _placement_score(cur[0], cur[1], W_, f["mode"])
        # box-edge/goal-line disambiguation: winner-colour mass OUTSIDE the candidate
        # rectangle is evidence AGAINST it (paint never lies beyond the goal line)
        win_bin = acc[win] > 0.5
        gy_, gx_ = np.where(win_bin)
        wx = _GX0 + gx_ * _GRES
        wy = _GY0 + gy_ * _GRES
        pad = 2.0
        inside = (
            (wx >= cur[0] - pad)
            & (wx <= cur[0] + pitch.length + pad)
            & (wy >= cur[1] - pad)
            & (wy <= cur[1] + W_ + pad)
        )
        out_frac = 1.0 - float(inside.mean()) if len(wx) else 1.0
        refined.append((sc * (1.0 - out_frac), float(cur[0]), float(cur[1]), W_, f["mode"]))
    refined.sort(key=lambda t: -t[0])
    _, x0, y0, W, win_mode = refined[0]
    soc_pitch = PitchModel(pitch.length, float(W))
    stats["observed_half"] = win_mode
    stats["placements_tried"] = [
        (round(x, 1), round(y, 1), round(w_, 1), round(s, 3)) for s, x, y, w_, _m in refined
    ]
    stats["width_m"] = round(W, 1)
    stats["ground_offset"] = (round(x0, 1), round(y0, 1))

    # measured x-interval of the soccer template (pitch coords) -> honest partial coverage
    mk = soc_pitch.sample_markings(0.5)
    u = ((mk[:, 0] + x0 - _GX0) / _GRES).astype(int)
    v = ((mk[:, 1] + y0 - _GY0) / _GRES).astype(int)
    inb = (u >= 0) & (u < _GW) & (v >= 0) & (v < _GH)
    covd = np.zeros(len(mk), bool)
    covd[inb] = coverage[v[inb], u[inb]] > 0
    if covd.any():
        vis_lo, vis_hi = float(mk[covd, 0].min()), float(mk[covd, 0].max())
    else:
        vis_lo, vis_hi = 0.0, pitch.length
    stats["visible_x"] = (round(vis_lo, 1), round(vis_hi, 1))
    stats["partial_length"] = (vis_hi - vis_lo) < 0.75 * pitch.length
    _prog(0.8)

    # ---- joint bundle: template-shift refinement, then shared camera + per-frame rot ----
    vis = VisiblePitch(soc_pitch, max(0.0, vis_lo - 2.0), min(pitch.length, vis_hi + 2.0))
    vis_pts = vis.sample_markings(0.5)
    for a in kept:
        a["rvec"] = a["p7"][3:6].copy()
        a["fb_scorer"] = _TemplateScorer(_distance_field(a["cand"]["sets"][0].mask), fb_pts, fb_rec)
        win_sets = [s for s in a["cand"]["sets"] if s.family == win]
        # template points in THIS frame's anchor coords (undo its alias shift)
        sx, sy = a["shift"]
        a["soc_scorer"] = (
            _TemplateScorer(
                _distance_field(win_sets[0].mask), vis_pts + np.array([x0 - sx, y0 - sy])
            )
            if win_sets
            else None
        )
    dx, dy = _refine_template_shift(kept)
    x0 += dx
    y0 += dy
    stats["ground_offset"] = (round(x0, 2), round(y0, 2))
    stats["template_shift"] = (round(dx, 2), round(dy, 2))
    for a in kept:  # shift the soccer scorers' template points too
        if a["soc_scorer"] is not None:
            a["soc_scorer"].P[:, :2] += np.array([dx, dy])
    _per_frame_shift_refine(kept)
    stats["anchor_shifts"] = {
        a["cand"]["idx"]: (round(a["shift"][0], 2), round(a["shift"][1], 2)) for a in kept
    }
    position, logf = _bundle_refine(
        kept, position, logf, principal, frame_size, fm, rounds=cfg["rounds"]
    )
    _prog(0.9)

    # ---- final per-frame homographies + quality gate -------------------------------------
    used_anchors: Dict[int, np.ndarray] = {}  # entry_i -> H (processing px -> m)
    for a in kept:
        p7 = np.array(
            [position[0], position[1], position[2], a["rvec"][0], a["rvec"][1], a["rvec"][2], logf]
        )
        Ha = homography_7dof(p7, principal, fm)
        if Ha is None or not np.all(np.isfinite(Ha)):
            continue
        sx, sy = a["shift"]  # this frame's template origin in ITS anchor coords
        T = np.array([[1.0, 0.0, -(x0 - sx)], [0.0, 1.0, -(y0 - sy)], [0.0, 0.0, 1.0]])
        H_pitch = T @ Ha
        if not homography_plausible(H_pitch, soc_pitch):
            continue
        q_soc = a["soc_scorer"].ls(Ha, 8.0) if a["soc_scorer"] is not None else None
        q_fb = a["fb_scorer"].ls(Ha, 8.0)
        # verified = the frame's own winner-colour paint confirms the placement (0.09
        # empirically separates visually-correct anchors from meters-off ones)
        ok = q_soc is not None and q_soc >= 0.09
        stats["per_frame"][a["cand"]["idx"]] = {
            "ls_soccer": None if q_soc is None else round(q_soc, 3),
            "ls_football": round(q_fb, 3),
            "used": bool(ok),
        }
        if ok:
            used_anchors[a["cand"]["entry_i"]] = H_pitch
    stats["n_anchored"] = len(used_anchors)
    if not used_anchors:
        stats["error"] = "all anchors failed the final soccer-paint quality gate"
        stats["runtime_s"] = round(time.time() - t_start, 1)
        return {}, stats

    # ---- propagation across the clip (flow chaining; the camera pans continuously) ------
    result, n_flow = _flow_propagate(entries, Gs, used_anchors, soc_pitch)
    S = np.diag([scale, scale, 1.0])  # original px -> processing px
    result = {i: (None if H is None else H @ S) for i, H in result.items()}
    covered = sum(1 for H in result.values() if H is not None)
    stats["flow_frames"] = n_flow
    stats["covered"] = covered
    stats["coverage_pct"] = round(100.0 * covered / max(1, len(entries)), 1)
    stats["runtime_s"] = round(time.time() - t_start, 1)
    _prog(1.0)
    return result, stats
