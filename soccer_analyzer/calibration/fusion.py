"""
Temporal fusion for the fixed (stands) camera.

A camera fixed in the stands only ROTATES (and maybe zooms); its position and — absent
zoom — its focal length are CONSTANT for the whole clip. PnLCalib estimates them per frame
with noise, and frames that don't see the near side extrapolate it badly. GlobalCameraFusion
pools the camera position (and focal, when no zoom is detected) across many frames into ONE
globally-consistent camera, so the frames that DID observe the near side improve the
calibration everywhere and per-frame jitter is removed. This is the "measure the near side
over time" mechanism.

CoverageMap records which pitch regions have actually been observed across the clip — the
field literally built up over time — so we can show measured vs never-seen regions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _robust_center(x, k=2.5, axis=0):
    """Median with MAD-based outlier rejection."""
    x = np.asarray(x, float)
    med = np.median(x, axis=axis)
    mad = np.median(np.abs(x - med), axis=axis) + 1e-9
    keep = (
        np.all(np.abs(x - med) <= k * mad, axis=tuple(range(1, x.ndim)))
        if x.ndim > 1
        else np.abs(x - med) <= k * mad
    )
    return np.median(x[keep], axis=axis) if np.any(keep) else med


class GlobalCameraFusion:
    def __init__(self):
        self.ready = False
        self.focal: Optional[float] = None
        self.fuse_focal = False
        self.principal = None
        self.position = None
        self.n = 0
        self.zoom_detected = False

    def calibrate(self, detector, video_path: str, n_frames: int = 30, min_frames: int = 5):
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_frames * 20
        idxs = np.linspace(0, max(0, total - 1), n_frames).astype(int)
        focals, positions, principals = [], [], []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            try:
                obs = detector.detect(frame, int(fi))
            except Exception:
                obs = None
            if obs is None or obs.cam_params is None:
                continue
            cp = obs.cam_params
            focals.append(0.5 * (cp["x_focal_length"] + cp["y_focal_length"]))
            positions.append(np.asarray(cp["position_meters"], float))
            principals.append(np.asarray(cp["principal_point"], float))
        cap.release()

        if len(focals) < min_frames:
            self.ready = False
            return self

        focals = np.array(focals)
        self.position = _robust_center(np.stack(positions))
        self.principal = _robust_center(np.stack(principals))
        self.focal = float(_robust_center(focals))
        # zoom detection: if focal varies a lot, the operator is zooming -> keep per-frame focal
        spread = float(np.median(np.abs(focals - np.median(focals)))) / (
            abs(np.median(focals)) + 1e-9
        )
        self.zoom_detected = spread > 0.06
        self.fuse_focal = not self.zoom_detected
        self.n = len(focals)
        self.ready = True
        return self

    def fuse_params(self, cp: dict) -> dict:
        """Replace this frame's position (+ focal, if no zoom) with the locked global values,
        keeping the frame's own rotation."""
        out = dict(cp)
        out["position_meters"] = list(self.position)
        out["principal_point"] = list(self.principal)
        if self.fuse_focal:
            out["x_focal_length"] = self.focal
            out["y_focal_length"] = self.focal
        return out

    def summary(self) -> dict:
        if not self.ready:
            return {"ready": False}
        return {
            "ready": True,
            "frames_used": self.n,
            "focal": round(self.focal, 1),
            "position_m": [round(float(v), 1) for v in self.position],
            "zoom_detected": self.zoom_detected,
            "fused_focal": self.fuse_focal,
        }


class CoverageMap:
    """Accumulates which pitch cells have been observed (in camera view) over the clip."""

    def __init__(self, pitch, res: float = 1.5):
        self.pitch = pitch
        self.res = res
        self.W = int(np.ceil(pitch.length / res))
        self.H = int(np.ceil(pitch.width / res))
        self.grid = np.zeros((self.H, self.W), np.float32)

    def add(self, homography: np.ndarray, frame_size, step: int = 24):
        if homography is None or not np.all(np.isfinite(homography)):
            return
        w, h = frame_size
        gx, gy = np.meshgrid(np.linspace(0, w, step), np.linspace(0, h, step))
        pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], 0)
        m = homography @ pts
        ok = np.abs(m[2]) > 1e-6
        xs = m[0, ok] / m[2, ok]
        ys = m[1, ok] / m[2, ok]
        for x, y in zip(xs, ys):
            if 0 <= x < self.pitch.length and 0 <= y < self.pitch.width:
                self.grid[int(y / self.res), int(x / self.res)] += 1.0

    def fraction_seen(self) -> float:
        return float((self.grid > 0).mean())

    def normalized(self) -> np.ndarray:
        g = self.grid.copy()
        if g.max() > 0:
            g /= g.max()
        return g
