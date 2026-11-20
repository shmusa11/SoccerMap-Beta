"""
Stage 0 — lock the camera intrinsics + lens distortion ONCE per clip.

The phone lens is fixed for a whole clip and a stands camera doesn't translate, so we
estimate focal length + distortion once (over the best-framed wide frames) and freeze
them. That collapses every per-frame solve from 6-DOF to rotation-only (3-DOF), which is
what lets later stages register the field from as few as 2-4 visible landmarks.

Primary estimator: GeoCalib (Apache code / CC-BY weights) -> focal + distortion + gravity.
Fallback (if geocalib isn't installed): focal = max(W, H), zero distortion, centred
principal point -- the same rough heuristic the old pipeline used, good enough to run.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Intrinsics:
    K: np.ndarray  # 3x3 camera matrix
    dist: np.ndarray  # (k1, k2, p1, p2) distortion, OpenCV order
    image_size: tuple  # (w, h)
    gravity: Optional[np.ndarray] = None  # up-vector in camera frame, if available
    source: str = "fallback"  # "geocalib" | "fallback"

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(np.abs(self.dist) > 1e-9))

    @classmethod
    def fallback(cls, width: int, height: int) -> "Intrinsics":
        f = float(max(width, height))
        K = np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]], dtype=np.float64)
        return cls(K=K, dist=np.zeros(4), image_size=(width, height), source="fallback")


def _pick_device():
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def estimate_intrinsics(
    frames: List[np.ndarray],
    camera_model: str = "radial",
    device: Optional[str] = None,
) -> Intrinsics:
    """Estimate locked intrinsics from a handful of representative frames.

    Args:
        frames: list of BGR frames (use the best-framed / widest ones).
        camera_model: GeoCalib model -- "pinhole", "simple_radial", "radial",
            or "simple_divisional" (fisheye).
        device: torch device override; auto-detected (cuda/mps/cpu) if None.

    Returns:
        Intrinsics with K + distortion locked. Falls back to a focal=max(W,H) guess
        if GeoCalib is unavailable so the rest of the pipeline still runs.
    """
    if not frames:
        raise ValueError("estimate_intrinsics needs at least one frame")
    h, w = frames[0].shape[:2]

    try:
        import torch  # noqa: F401
        from geocalib import GeoCalib
    except Exception:
        # GeoCalib not installed -> rough but usable intrinsics.
        return Intrinsics.fallback(w, h)

    device = device or _pick_device()
    model = GeoCalib().to(device)

    focals, dists, gravities = [], [], []
    for frame in frames:
        try:
            import torch

            # BGR (HxWx3, uint8) -> RGB CHW float tensor in [0, 1]
            rgb = frame[:, :, ::-1].copy()
            img = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.0
            result = model.calibrate(img, camera_model=camera_model)
            cam = result["camera"]
            focals.append(float(cam.f.detach().cpu().numpy().reshape(-1)[0]))
            try:
                k = cam.dist.detach().cpu().numpy().reshape(-1)
                dists.append(k)
            except Exception:
                pass
            try:
                gravities.append(result["gravity"].vec3d.detach().cpu().numpy().reshape(-1))
            except Exception:
                pass
        except Exception:
            continue

    if not focals:
        return Intrinsics.fallback(w, h)

    f = float(np.median(focals))
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]], dtype=np.float64)

    if dists:
        m = min(len(d) for d in dists)
        dist = np.median(np.stack([d[:m] for d in dists], axis=0), axis=0)
        dist = np.pad(dist, (0, max(0, 4 - len(dist))))[:4]
    else:
        dist = np.zeros(4)

    gravity = np.median(np.stack(gravities), axis=0) if gravities else None
    return Intrinsics(
        K=K, dist=dist.astype(np.float64), image_size=(w, h), gravity=gravity, source="geocalib"
    )
