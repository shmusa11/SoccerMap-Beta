"""
The pitch as known geometry: named landmarks, lines, and circles in meters.

Coordinate convention (matches the old calibrate.py):
    x in [0, length]  -> goal-to-goal (left goal x=0, right goal x=length)
    y in [0, width]   -> far sideline y=0 (top of screen), near sideline y=width
    z = 0             -> everything lies on the ground plane

Because the pitch is known geometry on a plane, it is the strongest prior we have:
every calibration solve registers the image to these coordinates. This module is the
single source of truth for that geometry and provides the sampled markings used by the
validation harness to measure reprojection error.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PitchModel:
    length: float = 105.0  # x, goal to goal
    width: float = 68.0  # y, sideline to sideline
    # Standard FIFA markings (meters)
    centre_circle_r: float = 9.15
    penalty_box_depth: float = 16.5
    penalty_box_width: float = 40.32
    goal_box_depth: float = 5.5
    goal_box_width: float = 18.32
    penalty_spot_dist: float = 11.0

    keypoints: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.keypoints:
            self.keypoints = self._build_keypoints()

    # ---- geometry construction ----------------------------------------------------

    def _build_keypoints(self) -> Dict[str, Tuple[float, float]]:
        L, W = self.length, self.width
        pb_y0 = (W - self.penalty_box_width) / 2.0  # 13.84
        pb_y1 = (W + self.penalty_box_width) / 2.0  # 54.16
        gb_y0 = (W - self.goal_box_width) / 2.0  # 24.84
        gb_y1 = (W + self.goal_box_width) / 2.0  # 43.16
        pbd = self.penalty_box_depth  # 16.5
        gbd = self.goal_box_depth  # 5.5
        ps = self.penalty_spot_dist  # 11
        cx, cy = L / 2.0, W / 2.0
        r = self.centre_circle_r

        return {
            # outer corners
            "corner_tl": (0.0, 0.0),
            "corner_tr": (L, 0.0),
            "corner_bl": (0.0, W),
            "corner_br": (L, W),
            # halfway line
            "halfway_top": (cx, 0.0),
            "halfway_bottom": (cx, W),
            "centre": (cx, cy),
            # centre circle cardinal points
            "circle_left": (cx - r, cy),
            "circle_right": (cx + r, cy),
            "circle_top": (cx, cy - r),
            "circle_bottom": (cx, cy + r),
            # left penalty box
            "l_pbox_top_goalline": (0.0, pb_y0),
            "l_pbox_bottom_goalline": (0.0, pb_y1),
            "l_pbox_top": (pbd, pb_y0),
            "l_pbox_bottom": (pbd, pb_y1),
            # left goal area
            "l_gbox_top_goalline": (0.0, gb_y0),
            "l_gbox_bottom_goalline": (0.0, gb_y1),
            "l_gbox_top": (gbd, gb_y0),
            "l_gbox_bottom": (gbd, gb_y1),
            "l_penalty_spot": (ps, cy),
            # right penalty box
            "r_pbox_top_goalline": (L, pb_y0),
            "r_pbox_bottom_goalline": (L, pb_y1),
            "r_pbox_top": (L - pbd, pb_y0),
            "r_pbox_bottom": (L - pbd, pb_y1),
            # right goal area
            "r_gbox_top_goalline": (L, gb_y0),
            "r_gbox_bottom_goalline": (L, gb_y1),
            "r_gbox_top": (L - gbd, gb_y0),
            "r_gbox_bottom": (L - gbd, gb_y1),
            "r_penalty_spot": (L - ps, cy),
        }

    def lines(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Straight-line markings as (p0, p1) segments in meters."""
        k = self.keypoints
        return [
            # touchlines + goal lines (the outer rectangle)
            (k["corner_tl"], k["corner_tr"]),
            (k["corner_bl"], k["corner_br"]),
            (k["corner_tl"], k["corner_bl"]),
            (k["corner_tr"], k["corner_br"]),
            # halfway line
            (k["halfway_top"], k["halfway_bottom"]),
            # left penalty box
            (k["l_pbox_top_goalline"], k["l_pbox_top"]),
            (k["l_pbox_top"], k["l_pbox_bottom"]),
            (k["l_pbox_bottom"], k["l_pbox_bottom_goalline"]),
            # left goal area
            (k["l_gbox_top_goalline"], k["l_gbox_top"]),
            (k["l_gbox_top"], k["l_gbox_bottom"]),
            (k["l_gbox_bottom"], k["l_gbox_bottom_goalline"]),
            # right penalty box
            (k["r_pbox_top_goalline"], k["r_pbox_top"]),
            (k["r_pbox_top"], k["r_pbox_bottom"]),
            (k["r_pbox_bottom"], k["r_pbox_bottom_goalline"]),
            # right goal area
            (k["r_gbox_top_goalline"], k["r_gbox_top"]),
            (k["r_gbox_top"], k["r_gbox_bottom"]),
            (k["r_gbox_bottom"], k["r_gbox_bottom_goalline"]),
        ]

    def circles(self) -> List[Tuple[float, float, float, Tuple[float, float]]]:
        """Circle markings as (cx, cy, r, x_clip) where x_clip filters arc points
        to keep only the visible portion of penalty arcs (None = full circle)."""
        cx, cy = self.length / 2.0, self.width / 2.0
        ps = self.penalty_spot_dist
        r = self.centre_circle_r
        pbd = self.penalty_box_depth
        return [
            (cx, cy, r, None),  # centre circle
            (ps, cy, r, (pbd, None)),  # left penalty arc (x >= box edge)
            (self.length - ps, cy, r, (None, self.length - pbd)),  # right penalty arc
        ]

    # ---- sampling helpers ---------------------------------------------------------

    def model_points(self, names: List[str]) -> np.ndarray:
        """Nx3 (x, y, 0) world points for the given keypoint names."""
        pts = [(self.keypoints[n][0], self.keypoints[n][1], 0.0) for n in names]
        return np.array(pts, dtype=np.float64)

    def sample_markings(self, step: float = 1.0) -> np.ndarray:
        """Dense Nx2 sample of all line + circle markings (meters), for reprojection
        error: project these into the image and compare against detected field pixels."""
        pts: List[Tuple[float, float]] = []
        for p0, p1 in self.lines():
            p0 = np.asarray(p0, float)
            p1 = np.asarray(p1, float)
            n = max(2, int(np.linalg.norm(p1 - p0) / step))
            for t in np.linspace(0, 1, n):
                pts.append(tuple(p0 + t * (p1 - p0)))
        for ccx, ccy, r, clip in self.circles():
            n = max(8, int(2 * np.pi * r / step))
            for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
                x, y = ccx + r * np.cos(a), ccy + r * np.sin(a)
                if clip is not None:
                    lo, hi = clip
                    if lo is not None and x < lo:
                        continue
                    if hi is not None and x > hi:
                        continue
                pts.append((x, y))
        return np.array(pts, dtype=np.float64)


# A module-level default so callers can `from ... import STANDARD_PITCH`
STANDARD_PITCH = PitchModel()
