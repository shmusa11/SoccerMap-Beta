"""
Stage 1 — per-keyframe field registration.

A FieldDetector looks at one frame and returns the pitch landmarks it can see plus a
frame->pitch homography. The rest of the calibration subsystem is written against this
interface, so the detector backend is swappable:

  * CachedHomographyDetector  — loads a saved homography (from the old calibrate.py JSON).
        BRIDGE ONLY: lets us validate the new architecture end-to-end today without the
        heavy/GPL field model. Not the product path.
  * PnLCalibFieldDetector     — the real automatic path: PnLCalib keypoint+line models
        detect pitch landmarks every keyframe. GPL-2.0, prototype use only; the adapter
        isolates it so a clean-room (Apache) detector can replace it without touching
        anything else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Tuple

import numpy as np

from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH


@dataclass
class FieldObservation:
    frame_idx: int
    homography: Optional[np.ndarray]  # 3x3 image(px) -> pitch(m)
    keypoints: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # name -> (u,v) px
    score: float = 1.0
    n_landmarks: int = 0
    reproj_err: Optional[float] = None  # detector-reported reprojection error (px)
    cam_params: Optional[dict] = None  # focal/principal/position/rotation (for fusion)

    @property
    def is_valid(self) -> bool:
        return (
            self.homography is not None
            and np.shape(self.homography) == (3, 3)
            and np.all(np.isfinite(self.homography))
            and np.linalg.matrix_rank(self.homography) == 3
        )


class FieldDetector(Protocol):
    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> Optional[FieldObservation]: ...


# ---------------------------------------------------------------------------------------
# Bridge detector: replays a saved homography (validates plumbing without the GPL model)
# ---------------------------------------------------------------------------------------


class CachedHomographyDetector:
    """Returns a fixed, previously-saved frame->pitch homography for every frame.

    This is a temporary bridge so the new PitchMapper/pipeline can run end-to-end before
    the automatic detector is wired. It does NOT track the camera as it pans (the saved
    homography is for one viewpoint), so on panning footage it will be wrong off the
    calibration frame -- which is exactly the limitation Phase B (the smoother) removes.
    """

    def __init__(self, calibration_path: str, pitch: PitchModel = STANDARD_PITCH):
        with open(calibration_path) as f:
            cal = json.load(f)
        if "homography" not in cal:
            raise ValueError(f"{calibration_path} has no 'homography' field")
        self.H = np.array(cal["homography"], dtype=np.float64)  # frame -> pitch
        if not FieldObservation(0, self.H).is_valid:
            raise ValueError("Calibration homography must be a finite, invertible 3x3 matrix")
        self.pitch = pitch
        self.frame_size = (cal.get("frame_width"), cal.get("frame_height"))
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError:
            self.H_inv = None

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> Optional[FieldObservation]:
        h, w = frame_bgr.shape[:2]
        kps = self._project_keypoints_into_frame(w, h)
        return FieldObservation(
            frame_idx=frame_idx,
            homography=self.H.copy(),
            keypoints=kps,
            score=1.0,
            n_landmarks=len(kps),
        )

    def _project_keypoints_into_frame(self, w: int, h: int) -> Dict[str, Tuple[float, float]]:
        """pitch keypoints -> image px (via inv(H)), keeping those inside the frame."""
        if self.H_inv is None:
            return {}
        out: Dict[str, Tuple[float, float]] = {}
        for name, (x, y) in self.pitch.keypoints.items():
            p = self.H_inv @ np.array([x, y, 1.0])
            if abs(p[2]) < 1e-9:
                continue
            u, v = p[0] / p[2], p[1] / p[2]
            if -50 <= u <= w + 50 and -50 <= v <= h + 50:
                out[name] = (float(u), float(v))
        return out


# ---------------------------------------------------------------------------------------
# Real automatic detector: PnLCalib (GPL-2.0, prototype only) behind a swappable adapter
# ---------------------------------------------------------------------------------------


def projection_from_cam_params(final_params_dict) -> np.ndarray:
    """3x4 world->image projection P = Q @ (R | -position). Replicated verbatim from
    PnLCalib's inference.py (GPL-2.0) so we don't depend on importing the top-level script."""
    cam = final_params_dict["cam_params"]
    fx = cam["x_focal_length"]
    fy = cam["y_focal_length"]
    pp = np.array(cam["principal_point"])
    pos = np.array(cam["position_meters"])
    R = np.array(cam["rotation_matrix"])
    It = np.eye(4)[:-1]
    It[:, -1] = -pos
    Q = np.array([[fx, 0, pp[0]], [0, fy, pp[1]], [0, 0, 1]])
    return Q @ (R @ It)


class PnLCalibFieldDetector:
    """Automatic per-keyframe field registration via PnLCalib's keypoint + line models.

    The automatic path runs PnLCalib's
    HRNet keypoint + line networks, then its FramebyFrameCalib heuristic voting (with
    optional PnL line refinement and lens-distortion modelling), and converts the result
    into a frame(px) -> pitch(m) homography in OUR pitch coordinate frame.

    PnLCalib is GPL-2.0 -> prototype / evaluation only. The whole backend is isolated in
    this class so a clean-room (Apache) reimplementation can replace it later.

    Setup:
        git clone https://github.com/mguti97/PnLCalib
        # download v1.0.0 weights (e.g. SV_kp.pt, SV_lines.pt)
        export PNLCALIB_ROOT=/path/to/PnLCalib
    Then construct with pnlcalib_root + the two weight file paths. See docs/pnlcalib_setup.md.
    """

    # PnLCalib world frame is the 105x68 pitch centred at the midpoint ([x-52.5, y-34]).
    # Our pitch frame has the corner at the origin, so mine = world + [52.5, 34].
    def _pnl_to_pitch_affine(self) -> np.ndarray:
        return np.array(
            [
                [1.0, 0.0, self.pitch.length / 2.0],
                [0.0, 1.0, self.pitch.width / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )

    def __init__(
        self,
        pnlcalib_root: Optional[str] = None,
        weights_kp: Optional[str] = None,
        weights_line: Optional[str] = None,
        config_kp: str = "config/hrnetv2_w48.yaml",
        config_line: str = "config/hrnetv2_w48_l.yaml",
        pitch: PitchModel = STANDARD_PITCH,
        device: Optional[str] = None,
        pnl_refine: bool = True,
        kp_threshold: float = 0.3434,
        line_threshold: float = 0.7867,
        min_landmarks: int = 4,
        max_reproj_px: float = 12.0,
        min_line_support: float = 0.0,
        enhance: bool = False,
        input_size=(540, 960),
    ):
        import os

        self.root = pnlcalib_root or os.environ.get("PNLCALIB_ROOT")
        self.weights_kp = weights_kp or (os.path.join(self.root, "SV_kp") if self.root else None)
        self.weights_line = weights_line or (
            os.path.join(self.root, "SV_lines") if self.root else None
        )
        self.config_kp = config_kp
        self.config_line = config_line
        self.pitch = pitch
        self.device = device
        self.pnl_refine = pnl_refine
        self.kp_threshold = kp_threshold
        self.line_threshold = line_threshold
        self.min_landmarks = min_landmarks
        self.max_reproj_px = max_reproj_px
        self.min_line_support = min_line_support
        self.enhance = enhance
        self.input_size = input_size
        self._loaded = None  # cached models + fns
        self.fusion = None  # optional GlobalCameraFusion (locked fixed-camera intrinsics+position)

    # ---- lazy backend loading ---------------------------------------------------------

    def _ensure_loaded(self):
        if self._loaded is not None:
            return self._loaded
        import os
        import sys
        import yaml
        import torch
        import torchvision.transforms as T
        from torchvision.transforms import functional as F  # noqa: F401

        if not self.root or not os.path.isdir(self.root):
            raise RuntimeError(
                "PnLCalibFieldDetector needs the PnLCalib repo. Set PNLCALIB_ROOT or pass "
                "pnlcalib_root=..., and weights_kp/weights_line. See docs/pnlcalib_setup.md."
            )
        if not (self.weights_kp and self.weights_line):
            raise RuntimeError("PnLCalibFieldDetector needs weights_kp and weights_line paths.")
        if self.root not in sys.path:
            sys.path.insert(0, self.root)

        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l
        from utils.utils_calib import FramebyFrameCalib
        from utils.utils_heatmap import (
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
            complete_keypoints,
            coords_to_dict,
        )

        device = self.device or _pick_torch_device()

        cfg = yaml.safe_load(open(os.path.join(self.root, self.config_kp)))
        cfg_l = yaml.safe_load(open(os.path.join(self.root, self.config_line)))

        model = get_cls_net(cfg)
        model.load_state_dict(torch.load(self.weights_kp, map_location="cpu"))
        model.to(device).eval()

        model_l = get_cls_net_l(cfg_l)
        model_l.load_state_dict(torch.load(self.weights_line, map_location="cpu"))
        model_l.to(device).eval()

        self._loaded = dict(
            torch=torch,
            T=T,
            device=device,
            model=model,
            model_l=model_l,
            FramebyFrameCalib=FramebyFrameCalib,
            kp_from_hm=get_keypoints_from_heatmap_batch_maxpool,
            kp_from_hm_l=get_keypoints_from_heatmap_batch_maxpool_l,
            complete_keypoints=complete_keypoints,
            coords_to_dict=coords_to_dict,
            resize=T.Resize(self.input_size),
        )
        return self._loaded

    # ---- detection --------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray, frame_idx: int) -> Optional[FieldObservation]:
        import cv2

        L = self._ensure_loaded()
        torch = L["torch"]
        h, w = frame_bgr.shape[:2]
        orig_frame = frame_bgr  # pre-enhance, for consistent line-support

        if self.enhance:  # brighten/contrast for dark clips
            lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.createCLAHE(2.5, (8, 8)).apply(l)
            frame_bgr = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        if tuple(t.shape[-2:]) != tuple(self.input_size):
            t = L["resize"](t)
        t = t.to(L["device"])

        with torch.no_grad():
            heatmaps = L["model"](t)
            heatmaps_l = L["model_l"](t)

        kp_coords = L["kp_from_hm"](heatmaps[:, :-1, :, :])
        line_coords = L["kp_from_hm_l"](heatmaps_l[:, :-1, :, :])
        # get_keypoints returns coords in the MODEL-INPUT space (e.g. 960x540), so we must
        # normalize against the model input size, NOT the full frame. (PnLCalib's own code
        # gets away with passing w,h because SoccerNet frames are already 960 wide.)
        in_h, in_w = self.input_size
        # ADAPTIVE THRESHOLD: amateur/night/wide footage gives weak heatmaps -> too few
        # keypoints at the strict default. Re-threshold the SAME heatmaps progressively lower
        # (nearly free) until we have enough landmarks. Clean footage keeps the strict
        # threshold; hard footage recovers automatically.
        kp_dict, lines_dict = {}, {}
        for kp_th, ln_th in [(self.kp_threshold, self.line_threshold), (0.15, 0.50), (0.08, 0.40)]:
            kd = L["coords_to_dict"](kp_coords, threshold=kp_th)
            ld = L["coords_to_dict"](line_coords, threshold=ln_th)
            kd, ld = L["complete_keypoints"](kd[0], ld[0], w=in_w, h=in_h, normalize=True)
            kp_dict, lines_dict = kd, ld
            if len(kd) >= self.min_landmarks:
                break

        keypoints = self._pixel_keypoints(kp_dict, w, h)
        n_landmarks = len(kp_dict)
        if n_landmarks < self.min_landmarks:
            return FieldObservation(
                frame_idx, None, keypoints=keypoints, score=0.0, n_landmarks=n_landmarks
            )

        # Snapshot detected keypoints as full-res pixels BEFORE update() — FramebyFrameCalib
        # with denormalize=True mutates kp_dict in place (normalized -> pixel coords).
        keypoints = self._pixel_keypoints(kp_dict, w, h)

        cam = L["FramebyFrameCalib"](iwidth=w, iheight=h, denormalize=True)
        cam.update(kp_dict, lines_dict)
        try:
            params = cam.heuristic_voting(refine_lines=self.pnl_refine)
        except TypeError:
            params = cam.heuristic_voting()

        if not params:
            return FieldObservation(
                frame_idx, None, keypoints=keypoints, score=0.0, n_landmarks=n_landmarks
            )

        H = self._homography_from_params(params)
        if H is None:
            return FieldObservation(
                frame_idx, None, keypoints=keypoints, score=0.0, n_landmarks=n_landmarks
            )

        rep_err = float(params.get("rep_err")) if params.get("rep_err") is not None else None
        # QUALITY GATE: a high reprojection error means the calibration is unreliable
        # (out-of-distribution footage: night, wide-angle, oblique). Reject it rather than
        # present a confidently-wrong field + player positions.
        if rep_err is not None and rep_err > self.max_reproj_px:
            return FieldObservation(
                frame_idx,
                None,
                keypoints=keypoints,
                score=0.0,
                n_landmarks=n_landmarks,
                reproj_err=rep_err,
            )
        score = min(1.0, n_landmarks / 8.0)
        cp = params.get("cam_params")

        # Temporal fusion: re-derive the homography using the globally-locked fixed-camera
        # focal + position (only this frame's rotation is kept). Frames that observed the
        # near side contribute to the locked camera, improving near-side mapping everywhere.
        if self.fusion is not None and getattr(self.fusion, "ready", False) and cp is not None:
            fused_cp = self.fusion.fuse_params(cp)
            Hf = self._homography_from_params({"cam_params": fused_cp})
            if Hf is not None:
                H = Hf

        # PROGRAMMATIC VERIFICATION (hybrid): does the reprojected pitch actually land on the
        # real painted lines? Catches confidently-wrong calibrations that reprojection error
        # (measured on the CNN's own keypoints) cannot.
        if self.min_line_support > 0.0:
            from soccer_analyzer.calibration.line_support import line_distance_field, line_support

            df = line_distance_field(orig_frame)
            if df is not None and line_support(H, df, self.pitch) < self.min_line_support:
                return FieldObservation(
                    frame_idx,
                    None,
                    keypoints=keypoints,
                    score=0.0,
                    n_landmarks=n_landmarks,
                    reproj_err=rep_err,
                )

        return FieldObservation(
            frame_idx,
            H,
            keypoints=keypoints,
            score=score,
            n_landmarks=n_landmarks,
            reproj_err=rep_err,
            cam_params=cp,
        )

    def detect_line_points(
        self, frame_bgr: np.ndarray, thresh: float = 0.3, min_pts: int = 20
    ) -> list:
        """Dense points along each detected pitch line, in full-res image pixels.

        Reads the line model's heatmaps and takes the high-activation ridge of each line
        channel. Robust line evidence (trained detector) for distortion estimation — straight
        pitch lines bow under lens distortion. Returns a list of Nx2 arrays, one per line.
        """
        import cv2

        L = self._ensure_loaded()
        torch = L["torch"]
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        if tuple(t.shape[-2:]) != tuple(self.input_size):
            t = L["resize"](t)
        t = t.to(L["device"])
        with torch.no_grad():
            hml = L["model_l"](t)
        hm = hml[0].detach().cpu().numpy()  # (C, Hh, Ww)
        Hh, Ww = hm.shape[1], hm.shape[2]
        sx, sy = w / Ww, h / Hh
        lines = []
        for c in range(hm.shape[0] - 1):  # last channel is background
            ys, xs = np.where(hm[c] > thresh)
            if len(xs) < min_pts:
                continue
            lines.append(np.stack([xs * sx, ys * sy], axis=1).astype(np.float64))
        return lines

    def _homography_from_params(self, params) -> Optional[np.ndarray]:
        """Ground-plane (z=0) homography image(px) -> pitch(m, our frame)."""
        P = projection_from_cam_params(params)  # world(m, centred) -> image
        H_world2img = P[:, [0, 1, 3]]  # drop Z column for the ground plane
        try:
            H_img2world = np.linalg.inv(H_world2img)
        except np.linalg.LinAlgError:
            return None
        H = self._pnl_to_pitch_affine() @ H_img2world  # -> our corner-origin pitch frame
        if abs(H[2, 2]) < 1e-12 or not np.all(np.isfinite(H)):
            return None
        return H / H[2, 2]

    @staticmethod
    def _pixel_keypoints(kp_dict, w, h) -> Dict[str, Tuple[float, float]]:
        """Detected keypoints as pixel coords keyed by index (for debug/overlay)."""
        out: Dict[str, Tuple[float, float]] = {}
        for idx, v in kp_dict.items():
            try:
                x, y = v["x"], v["y"]
            except (TypeError, KeyError):
                continue
            # complete_keypoints(normalize=True) -> coords in [0, 1]
            out[f"pnl_{idx}"] = (float(x) * w, float(y) * h)
        return out


def _pick_torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
