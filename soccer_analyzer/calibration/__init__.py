"""
Automatic camera-calibration subsystem (Stage 0-1 of the architecture).

Replaces the manual 4-point calibration + optical-flow homography chaining with
automatic field registration that maps any image point to pitch coordinates (meters)
with a per-frame confidence, from any input video and zero manual setup.

Phase A (this milestone): automatic per-keyframe field registration -> homography,
held between keyframes, wired into the pipeline. The drift-free factor-graph smoother
(Phase B) and mosaic relocalization (Phase C) build on these interfaces.

See docs/calibration_prototype_plan.md.
"""

from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH
from soccer_analyzer.calibration.intrinsics import Intrinsics, estimate_intrinsics
from soccer_analyzer.calibration.field_detector import (
    FieldObservation,
    FieldDetector,
    CachedHomographyDetector,
    PnLCalibFieldDetector,
)
from soccer_analyzer.calibration.mapper import PitchMapper, FrameCalibration

__all__ = [
    "PitchModel",
    "STANDARD_PITCH",
    "Intrinsics",
    "estimate_intrinsics",
    "FieldObservation",
    "FieldDetector",
    "CachedHomographyDetector",
    "PnLCalibFieldDetector",
    "PitchMapper",
    "FrameCalibration",
]
