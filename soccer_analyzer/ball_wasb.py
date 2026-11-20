"""WASB soccer-ball heatmaps from three consecutive raw BGR frames.

The optional adapter loads the soccer checkpoint from the external WASB-SBDT repository
(BMVC 2023, https://github.com/nttcom/WASB-SBDT, MIT). It constructs HRNet from the model
configuration, extracts the center-frame heatmap and returns weighted blob centroids.
High-resolution inputs use overlapping tiles in bounded batches. No accuracy or runtime
is implied by selecting this backend; evaluate it on labelled footage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# the exact model config from WASB-SBDT/src/configs/model/wasb.yaml
_WASB_MODEL_CFG = {
    "name": "hrnet",
    "frames_in": 3,
    "frames_out": 3,
    "inp_height": 288,
    "inp_width": 512,
    "out_height": 288,
    "out_width": 512,
    "rgb_diff": False,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "PRETRAINED_LAYERS": ["*"],
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 1,
                "BLOCK": "BOTTLENECK",
                "NUM_BLOCKS": [1],
                "NUM_CHANNELS": [32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2],
                "NUM_CHANNELS": [16, 32],
                "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64],
                "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [2, 2, 2, 2],
                "NUM_CHANNELS": [16, 32, 64, 128],
                "FUSE_METHOD": "SUM",
            },
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": [], "NUM_BASIC_BLOCKS": 2},
        },
        "INIT_WEIGHTS": True,
    },
}

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _local_peaks(hm: np.ndarray, tile, thresh: float) -> List[Tuple[float, float, float]]:
    """Local maxima of one tile heatmap above `thresh` -> [(x, y, score)] in full-res pixels."""
    if hm.max() <= thresh:
        return []
    x0, y0, w, h = tile
    sx, sy = w / hm.shape[1], h / hm.shape[0]
    dil = cv2.dilate(hm, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero((hm >= dil) & (hm > thresh))
    out = []
    Hh, Wh = hm.shape
    for y, x in zip(ys, xs):
        a, b, c, d = max(0, y - 2), min(Hh, y + 3), max(0, x - 2), min(Wh, x + 3)
        win = hm[a:b, c:d]
        wv = np.where(win > thresh, win - thresh, 0.0)
        if wv.sum() <= 0:
            continue
        gy, gx = np.mgrid[a:b, c:d]
        cx, cy = (wv * gx).sum() / wv.sum(), (wv * gy).sum() / wv.sum()
        out.append(((cx + 0.5) * sx - 0.5 + x0, (cy + 0.5) * sy - 0.5 + y0, float(hm[y, x])))
    return out


class _attrdict(dict):
    """dict with attribute access (WASB's HRNet expects cfg.MODEL.EXTRA-style access), applied
    recursively. Both cfg['model']-style and cfg.MODEL-style work."""

    def __init__(self, d):
        super().__init__({k: _attrdict(v) if isinstance(v, dict) else v for k, v in d.items()})

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)


class WASBBallDetector:
    """detect(frames3) -> list of (x, y, score) ball candidates in FULL-res coords of the CENTER
    frame. frames3 = [prev, cur, next] BGR frames (any equal size)."""

    def __init__(
        self,
        root: str = None,
        weights: str = None,
        device: Optional[str] = None,
        score_threshold: float = 0.5,
        max_batch: int = 4,
    ):
        import torch

        root = root or os.environ.get(
            "WASB_ROOT", str(Path(__file__).resolve().parents[2] / "WASB-SBDT")
        )
        weights = weights or os.path.join(root, "wasb_soccer_best.pth.tar")
        self.torch = torch
        self.thresh = float(score_threshold)
        from soccer_analyzer.detection import _pick_device

        self.device = device or _pick_device()
        if type(max_batch) is not int or max_batch < 1:
            raise ValueError("max_batch must be a positive integer")
        # Tiles per forward pass, split evenly: a 4K frame's 6 tiles in one pass pushed a full
        # pipeline run over 7.9 GB (HRNet keeps high-resolution branches), so 4K runs 3 + 3; an
        # HD frame's 4 tiles go in one pass (143 vs 167 ms per frame as 3 + 1 on MPS, same
        # peaks; 2026-10-14). Results are identical per tile.
        self.max_batch = max_batch
        # Import WASB's HRNet with its src TEMPORARILY on sys.path, then remove BOTH the path and
        # the generic-named modules it registered ('models', 'utils', ...). Leaving them shadows
        # other repos' imports — PnLCalib lazily imports its own 'models'/'utils' later and
        # silently breaks (calibration -> 0%). Same class of bug PnLCalib itself once caused.
        src = os.path.join(root, "src")
        added = src not in sys.path
        if added:
            sys.path.insert(0, src)
        pre = set(sys.modules)
        try:
            from models.hrnet import HRNet  # WASB's module
        finally:
            if added and src in sys.path:
                sys.path.remove(src)
            for name in list(sys.modules):
                if name not in pre and (name == "models" or name.startswith("models.")):
                    m = sys.modules[name]
                    if src in (getattr(m, "__file__", "") or "") + str(getattr(m, "__path__", "")):
                        del sys.modules[name]
        self.model = HRNet(_attrdict(_WASB_MODEL_CFG))
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.to(self.device).eval()
        self.in_w, self.in_h = 512, 288

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(frame, (self.in_w, self.in_h)), cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        return ((x - _MEAN) / _STD).transpose(2, 0, 1)  # 3xHxW

    def _infer_hm_batch(self, stacks: List[List[np.ndarray]]) -> np.ndarray:
        """Batched center-frame ball heatmaps (sigmoid), Nx288x512. Each element of `stacks` is
        a 3-frame [prev, cur, next] tile; up to `max_batch` tiles go through the HRNet in one
        forward pass (much faster on MPS than tile-by-tile, bounded memory)."""
        return self._infer_inputs(
            np.stack([np.concatenate([self._prep(f) for f in s], axis=0) for s in stacks])
        )

    def _infer_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Heatmaps of prepared tiles (N x 9 x 288 x 512), at most `max_batch` tiles per forward
        pass, in equal-sized passes."""
        torch = self.torch
        out_maps = []
        n = len(inputs)
        step = -(-n // -(-n // self.max_batch)) if n else 1  # ceil(n / number of passes)
        for i in range(0, n, step):
            x = inputs[i : i + step]
            with torch.inference_mode():
                out = self.model(torch.from_numpy(x).to(self.device))
            hm = out[0] if isinstance(out, (list, tuple)) else out
            if isinstance(hm, dict):  # {scale: tensor}
                hm = hm[0]
            hm = torch.sigmoid(hm)  # model emits raw logits
            out_maps.append(hm[:, 1].float().cpu().numpy())
            del x, out, hm
        return np.concatenate(out_maps, axis=0)

    def _infer_hm(self, frames3: List[np.ndarray]) -> np.ndarray:
        return self._infer_hm_batch([frames3])[0]

    def _blobs(
        self, hm: np.ndarray, sx: float, sy: float, ox: float, oy: float, thresh: float
    ) -> List[Tuple[float, float, float]]:
        out = []
        if hm.max() > thresh:
            _, mth = cv2.threshold(hm, thresh, 1, cv2.THRESH_BINARY)
            n, lbl = cv2.connectedComponents(mth.astype(np.uint8))
            ys, xs = np.nonzero(lbl)
            labels, weights = lbl[ys, xs], hm[ys, xs]
            mass = np.bincount(labels, weights=weights, minlength=n)
            sum_x = np.bincount(labels, weights=weights * xs, minlength=n)
            sum_y = np.bincount(labels, weights=weights * ys, minlength=n)
            peaks = np.zeros(n, np.float32)
            np.maximum.at(peaks, labels, weights)
            for m in range(1, n):
                if mass[m] > 0:
                    out.append(
                        (
                            float(sum_x[m] / mass[m]) * sx + ox,
                            float(sum_y[m] / mass[m]) * sy + oy,
                            float(peaks[m]),
                        )
                    )
        return out

    def detect(
        self, frames3: List[np.ndarray], max_cands: int = 10, thresh: Optional[float] = None
    ) -> List[Tuple[float, float, float]]:
        """Return full-resolution center-frame candidates from overlapping image tiles.

        Use one tile up to 1280 pixels wide, four for HD and six above 2200 pixels. Limit
        concurrent inference with ``max_batch`` and merge duplicate seam detections.
        """
        thresh = self.thresh if thresh is None else thresh
        if len(frames3) != 3 or any(f.shape != frames3[0].shape for f in frames3):
            raise ValueError("WASB needs three consecutive BGR frames of the same size")
        if (
            not np.isfinite(thresh)
            or not 0 < thresh < 1
            or type(max_cands) is not int
            or max_cands < 1
        ):
            raise ValueError("threshold must be between 0 and 1, and max_cands positive")
        H, W = frames3[1].shape[:2]
        gx, gy = self._grid(W)
        stacks, geoms = [], []
        for ty in range(gy):
            for tx in range(gx):
                y0, y1 = ty * H // gy, (ty + 1) * H // gy
                x0, x1 = tx * W // gx, (tx + 1) * W // gx
                # Overlap gives a ball on a tile seam enough visual and motion context.
                y0, y1 = max(0, y0 - 32), min(H, y1 + 32)
                x0, x1 = max(0, x0 - 32), min(W, x1 + 32)
                stacks.append([f[y0:y1, x0:x1] for f in frames3])
                geoms.append(((x1 - x0) / self.in_w, (y1 - y0) / self.in_h, x0, y0))
        cands: List[Tuple[float, float, float]] = []
        for start in range(0, len(stacks), self.max_batch):
            hms = self._infer_hm_batch(stacks[start : start + self.max_batch])
            for hm, (sx, sy, ox, oy) in zip(hms, geoms[start : start + self.max_batch]):
                cands += self._blobs(hm, sx, sy, ox, oy, thresh)
        cands.sort(key=lambda c: -c[2])
        selected = []
        for candidate in cands:
            if all(np.hypot(candidate[0] - c[0], candidate[1] - c[1]) > 8 for c in selected):
                selected.append(candidate)
            if len(selected) == max_cands:
                break
        return selected

    def peaks(
        self, frames3: List[np.ndarray], thresh: float = 0.05, k: int = 20, merge_px: float = 6.0
    ) -> List[Tuple[float, float, float]]:
        """Every local maximum of the centre-frame heatmap above `thresh` (5x5 non-maximum
        suppression in heatmap pixels, sub-pixel centroid in a 5x5 window) on the full-frame
        tiles of `detect`, mapped to full-res pixels, merged across tiles within `merge_px`, top
        `k` by score. No 0.30 cut: the ball tracker's appearance model scores the peaks, and the
        thresholded blobs missed most balls (lab 2026-10-12)."""
        return self.peaks_prepared(self.prepare(frames3), thresh, k, merge_px)

    def prepare(self, frames3: List[np.ndarray]) -> Tuple[np.ndarray, list]:
        """CPU half of `peaks` (thread-safe, no GPU): the network inputs of the full-frame tiles
        (N x 9 x 288 x 512 float32) and the tile boxes."""
        H, W = frames3[1].shape[:2]
        gx, gy = self._grid(W)
        inputs, tiles = [], []
        for ty in range(gy):
            for tx in range(gx):
                y0, y1 = ty * H // gy, (ty + 1) * H // gy
                x0, x1 = tx * W // gx, (tx + 1) * W // gx
                inputs.append(np.concatenate([self._prep(f[y0:y1, x0:x1]) for f in frames3], 0))
                tiles.append((x0, y0, x1 - x0, y1 - y0))
        return np.stack(inputs), tiles

    def peaks_prepared(
        self, prepared: Tuple[np.ndarray, list], thresh: float = 0.05, k: int = 20, merge_px=6.0
    ) -> List[Tuple[float, float, float]]:
        """GPU half of `peaks`, on the output of `prepare`."""
        inputs, tiles = prepared
        hms = self._infer_inputs(inputs)
        cands = [p for hm, tile in zip(hms, tiles) for p in _local_peaks(hm, tile, thresh)]
        keep: List[Tuple[float, float, float]] = []
        for c in sorted(cands, key=lambda c: -c[2]):
            if all((c[0] - q[0]) ** 2 + (c[1] - q[1]) ** 2 > merge_px**2 for q in keep):
                keep.append(c)
            if len(keep) >= k:
                break
        return keep

    @staticmethod
    def _grid(width: int) -> Tuple[int, int]:
        """Tile grid that keeps the ball above ~1 heatmap pixel: 1 tile <=1280w, 2x2 HD, 3x2 4K."""
        if width <= 1280:
            return 1, 1
        if width <= 2200:
            return 2, 2
        return 3, 2

    def detect_roi(
        self,
        frames3: List[np.ndarray],
        center: Tuple[float, float],
        max_cands: int = 5,
        thresh: Optional[float] = None,
    ) -> List[Tuple[float, float, float]]:
        """Ball candidates in ONE tile centred on `center` (full-res px), at the same scale as a
        `detect` tile. Costs one tile instead of up to six, so a tracker can search around the
        predicted ball position and fall back to `detect` only when the ball is lost."""
        thresh = self.thresh if thresh is None else thresh
        H, W = frames3[1].shape[:2]
        gx, gy = self._grid(W)
        tw, th = W // gx, H // gy
        x0 = int(np.clip(center[0] - tw / 2, 0, W - tw))
        y0 = int(np.clip(center[1] - th / 2, 0, H - th))
        stack = [f[y0 : y0 + th, x0 : x0 + tw] for f in frames3]
        hm = self._infer_hm_batch([stack])[0]
        cands = self._blobs(hm, tw / self.in_w, th / self.in_h, x0, y0, thresh)
        cands.sort(key=lambda c: -c[2])
        return cands[:max_cands]
