"""
Self-hosted player + ball detection (no API cost).

Uses a local YOLO (COCO) model — person (0) and sports ball (32) — so the UI can place
players on the 2D pitch without the paid Roboflow API. Feet = bottom-centre of the person
box (the ground-contact point the calibration maps to pitch coordinates). A cheap jersey
hue split assigns a provisional team colour for the demo (replace with an embedding-based
classifier later).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Detected:
    bbox: Tuple[float, float, float, float]
    feet: Tuple[float, float]
    team: int = 0  # 0 / 1 provisional team, -1 unknown
    conf: float = 0.0


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


class PlayerDetector:
    def __init__(
        self,
        weights: str = "pythonProject1/yolov8n.pt",
        conf: float = 0.30,
        device: Optional[str] = None,
        imgsz: int = 1280,
    ):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or _pick_device()

    def detect(self, frame_bgr) -> Tuple[List[Detected], Optional[Detected]]:
        r = self.model.predict(
            frame_bgr,
            classes=[0, 32],
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        players: List[Detected] = []
        ball: Optional[Detected] = None
        for b in r.boxes:
            cls = int(b.cls)
            xyxy = b.xyxy[0].cpu().numpy()
            cf = float(b.conf)
            x1, y1, x2, y2 = xyxy
            if cls == 0:
                feet = ((x1 + x2) / 2.0, y2)
                team = self._team(frame_bgr, xyxy)
                players.append(Detected(tuple(xyxy), feet, team, cf))
            elif cls == 32:
                c = ((x1 + x2) / 2.0, y2)
                if ball is None or cf > ball.conf:
                    ball = Detected(tuple(xyxy), c, -1, cf)
        return players, ball

    @staticmethod
    def _team(frame_bgr, bbox) -> int:
        """Cheap jersey split: reddish torso -> team 0, light/other -> team 1."""
        import cv2

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h = y2 - y1
        ty1, ty2 = y1 + int(0.2 * h), y1 + int(0.5 * h)  # torso band
        crop = frame_bgr[max(0, ty1) : max(ty1 + 1, ty2), max(0, x1) : max(x1 + 1, x2)]
        if crop.size == 0:
            return -1
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        m = (sat > 60) & (val > 50)
        if m.sum() < 10:
            return 1 if val.mean() > 150 else 0
        h2 = hue[m]
        redish = np.mean((h2 < 12) | (h2 > 168))
        return 0 if redish > 0.35 else 1


def map_to_pitch(points, mapper) -> List[Tuple[float, float]]:
    """Map image points (feet) to pitch meters via the current calibration."""
    if not points:
        return []
    out = mapper.transform(points)
    return [tuple(p) for p in out] if len(out) else []
