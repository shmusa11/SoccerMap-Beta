"""Look up sampled calibration without overriding explicitly rejected frames."""

from bisect import bisect_left

import numpy as np


class CalibrationLookup:
    def __init__(self, calibrations):
        self.calibrations = calibrations
        self.frames = sorted(calibrations)

    def at(self, frame_idx: int, max_distance: int = 3):
        if not self.frames:
            return None
        pos = bisect_left(self.frames, frame_idx)
        candidates = self.frames[max(0, pos - 1) : pos + 1]
        nearest = min(candidates, key=lambda f: abs(f - frame_idx))
        if abs(nearest - frame_idx) > max_distance:
            return None
        # Include rejected samples in the search: do not jump across them to a good anchor.
        homography = self.calibrations[nearest]
        if homography is None or not np.all(np.isfinite(homography)):
            return None
        return homography

    def coverage(self, total: int, stride: int, start_frame: int = 0) -> tuple[int, int]:
        if stride <= 0:
            raise ValueError("stride must be positive")
        frames = range(start_frame, start_frame + total, stride)
        return sum(self.at(f) is not None for f in frames), len(frames)
