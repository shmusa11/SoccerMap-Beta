"""
Correspondence-free calibration by line-support optimization (the breakthrough).

Instead of trusting a per-frame CNN that mis-matches keypoints, we OPTIMIZE the camera to
make the reprojected pitch lines sit on the REAL extracted line pixels. The objective is the
geometric truth itself (line_support), so it can't be fooled by wrong correspondences. The
CNN (or a coarse search) only provides a rough seed; the optimizer finds the right answer.

Camera = locked position (from fusion) + free rotation (Rodrigues 3-vec) + focal. 4 DOF,
derivative-free multi-start Nelder-Mead maximizing line_support. On a fixed camera this needs
to run only occasionally (seed the next frame from the last solution).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import minimize

from soccer_analyzer.calibration.line_support import line_support
from soccer_analyzer.calibration.pose_graph import _homography_from
from soccer_analyzer.calibration.pitch_model import PitchModel, STANDARD_PITCH


def homography_from_rvec(rvec, focal, position, principal, pitch: PitchModel):
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    cp = {
        "x_focal_length": float(focal),
        "y_focal_length": float(focal),
        "principal_point": list(principal),
        "position_meters": list(position),
        "rotation_matrix": R,
    }
    return _homography_from(cp, pitch)


def optimize_calibration(
    dist_field,
    position,
    principal,
    seed_rvec,
    seed_focal,
    pitch: PitchModel = STANDARD_PITCH,
    n_starts: int = 8,
    radius: float = 8.0,
    jitter=(0.05, 0.05, 0.03, 200.0),
    rng_seed: int = 0,
) -> Tuple[Optional[np.ndarray], float, np.ndarray]:
    """Maximize line_support over (rvec[3], focal) starting near the seed. Returns
    (homography, best_line_support, params)."""
    fh, fw = dist_field.shape

    def neg_ls(x):
        H = homography_from_rvec(x[:3], x[3], position, principal, pitch)
        if H is None:
            return 0.0
        if spread_score(H, (fw, fh), pitch) < 0.05:  # reject degenerate "smear" optima
            return 0.0
        return -line_support(H, dist_field, pitch, radius=radius)

    x0 = np.array([seed_rvec[0], seed_rvec[1], seed_rvec[2], seed_focal], float)
    best_x, best_v = x0, -neg_ls(x0)
    rng = np.random.default_rng(rng_seed)
    for t in range(n_starts):
        s = x0 + (rng.normal(0, jitter, 4) if t > 0 else 0.0)
        r = minimize(
            neg_ls, s, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 2e-3}
        )
        if -r.fun > best_v:
            best_v, best_x = -r.fun, r.x
    H = homography_from_rvec(best_x[:3], best_x[3], position, principal, pitch)
    return H, float(best_v), best_x


def refine_homography(
    H: np.ndarray,
    dist_field: np.ndarray,
    frame_size,
    pitch: PitchModel = STANDARD_PITCH,
    n_starts: int = 3,
    radius: float = 8.0,
    position=None,
) -> Tuple[Optional[np.ndarray], float]:
    """Snap a rough calibration onto the real painted lines with a constrained camera model.

    H (image px -> pitch m) is decomposed into a pinhole camera; the camera position is held
    fixed (its own, or `position` in PnLCalib's centred metres, e.g. pooled over a clip) and the
    rotation + focal length are optimised for line support. 4 DOF is the configuration that
    aligned night footage, unlike a free 7-DOF or 8-DOF solve that can smear onto a few lines.
    Returns (refined H, its line support) or (None, 0.0) if H is not a plausible camera."""
    from soccer_analyzer.calibration.homography import decompose_camera

    cam = decompose_camera(H, frame_size, pitch.length, pitch.width)
    if cam is None:
        return None, 0.0
    f = cam["x_focal_length"]
    jitter = (0.02, 0.02, 0.015, 0.03 * f)
    if position is not None:  # a different position needs a wider search around the rotation
        n_starts, jitter = max(n_starts, 4), (0.04, 0.04, 0.03, 0.08 * f)
    Hr, ls, _ = optimize_calibration(
        dist_field,
        list(position) if position is not None else cam["position_meters"],
        cam["principal_point"],
        rvec_of(cam["rotation_matrix"]),
        f,
        pitch,
        n_starts=n_starts,
        radius=radius,
        jitter=jitter,
    )
    return Hr, ls


def rvec_of(rotation_matrix) -> np.ndarray:
    r, _ = cv2.Rodrigues(np.asarray(rotation_matrix, float))
    return r.reshape(-1)


def spread_score(H, frame_size, pitch, grid: int = 10) -> float:
    """Fraction of an image grid occupied by reprojected in-frame pitch markings. A correct
    calibration spreads across the visible field; a degenerate 'smear' clusters tightly."""
    w, h = frame_size
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 0.0
    mk = pitch.sample_markings(1.0)
    q = (Hi @ np.hstack([mk, np.ones((len(mk), 1))]).T).T
    ok = np.abs(q[:, 2]) > 1e-6
    u = q[ok, 0] / q[ok, 2]
    v = q[ok, 1] / q[ok, 2]
    m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if m.sum() < 5:
        return 0.0
    cells = {(int(u[i] / w * grid), int(v[i] / h * grid)) for i in np.where(m)[0]}
    return len(cells) / float(grid * grid)


def refine_7dof(
    dist_field,
    frame_size,
    position,
    principal,
    rvec,
    focal,
    pitch: PitchModel = STANDARD_PITCH,
    radius: float = 8.0,
    n_starts: int = 5,
):
    """Locally refine ALL 7 DOF (incl. camera position) from a good 4-DOF seed, to escape the
    fusion position-noise that caps the 4-DOF solve. Local polish only -> won't jump to a smear;
    a spread guard further blocks degeneracy. Returns (H, line_support, params)."""
    from scipy.optimize import minimize

    x0 = np.array(
        [position[0], position[1], position[2], rvec[0], rvec[1], rvec[2], np.log(focal)], float
    )

    def neg(x):
        H = homography_7dof(x, principal, pitch)
        if H is None or not np.all(np.isfinite(H)):
            return 0.0
        sp = spread_score(H, frame_size, pitch)
        if sp < 0.05:  # degeneracy guard
            return 0.0
        return -line_support(H, dist_field, pitch, radius=radius)

    best_x, best_v = x0, -neg(x0)
    rng = np.random.default_rng(0)
    jit = np.array([3.0, 3.0, 3.0, 0.03, 0.03, 0.02, 0.05])
    for t in range(n_starts):
        s = x0 + (rng.normal(0, jit) if t > 0 else 0.0)
        r = minimize(
            neg, s, method="Nelder-Mead", options={"maxiter": 600, "xatol": 1e-3, "fatol": 2e-3}
        )
        if -r.fun > best_v:
            best_v, best_x = -r.fun, r.x
    return homography_7dof(best_x, principal, pitch), float(best_v), best_x


def homography_7dof(params, principal, pitch):
    """7-DOF camera -> image->pitch homography. params = [px,py,pz, rx,ry,rz, log_focal]."""
    pos = params[:3]
    R, _ = cv2.Rodrigues(np.asarray(params[3:6], float))
    cp = {
        "x_focal_length": float(np.exp(params[6])),
        "y_focal_length": float(np.exp(params[6])),
        "principal_point": list(principal),
        "position_meters": list(pos),
        "rotation_matrix": R,
    }
    return _homography_from(cp, pitch)


def _feasibility_penalty(H, w, h, pitch, padx=70.0, pady=70.0):
    """0 if the image corners+center reproject onto a padded pitch; else a SOFT (gradient-
    bearing) penalty = total distance outside the bounds + behind-camera penalty. This is the
    barrier that rejects degenerate/near-affine optima that smear the template across the lines
    at huge metric coordinates (the 678m garbage that line_support alone cannot catch)."""
    pen = 0.0
    pts = [(0, 0), (w, 0), (w, h), (0, h), (w / 2, h / 2)]
    for u, v in pts:
        q = H @ np.array([u, v, 1.0])
        if q[2] <= 1e-6:  # behind camera / at infinity
            pen += 200.0
            continue
        x, y = q[0] / q[2], q[1] / q[2]
        pen += max(0.0, -padx - x, x - (pitch.length + padx))
        pen += max(0.0, -pady - y, y - (pitch.width + pady))
    return pen


def coverage_score(H, dist_field, pitch, radius=8.0):
    """Symmetric check: fraction of EXTRACTED line pixels that have a reprojected marking
    nearby (catches 'smear over a few lines'). Cheap proxy: render the reprojected markings to
    a mask and measure overlap of the real line pixels with it."""
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return 0.0
    h, w = dist_field.shape
    pm = np.zeros((h, w), np.uint8)
    pts = pitch.sample_markings(0.5)
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    q = (H_inv @ ph.T).T
    ok = np.abs(q[:, 2]) > 1e-6
    u = q[ok, 0] / q[ok, 2]
    v = q[ok, 1] / q[ok, 2]
    m = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    pm[v[m].astype(int), u[m].astype(int)] = 1
    pm = cv2.dilate(pm, np.ones((int(2 * radius), int(2 * radius)), np.uint8))
    real = dist_field == 0
    if real.sum() < 50:
        return 0.0
    return float((pm[real] > 0).mean())


def global_solve(
    dist_field,
    principal,
    frame_size,
    pitch: PitchModel = STANDARD_PITCH,
    seed_rvec=None,
    seed_focal=2000.0,
    radius: float = 8.0,
    restarts: int = 3,
    maxiter: int = 70,
    popsize: int = 16,
):
    """Correspondence-free GLOBAL calibration: differential evolution over the 7-DOF camera,
    constrained to the physically-feasible stands manifold, maximizing a SYMMETRIC objective
    (precision = markings on real lines AND recall = real lines covered). The recall term kills
    the high-precision "smear" optima that line_support alone cannot. No seed/training needed.
    Returns (H, line_support, coverage, params) or (None, 0, 0, None)."""
    from scipy.optimize import differential_evolution

    w, h = frame_size

    # --- precompute for a fast downsampled coverage (recall) computed every eval ---
    down = 4
    Hs, Ws = max(1, h // down), max(1, w // down)
    rys, rxs = np.where(dist_field <= 1.5)
    real_small = np.zeros((Hs, Ws), np.uint8)
    if len(rxs):
        real_small[np.clip(rys // down, 0, Hs - 1), np.clip(rxs // down, 0, Ws - 1)] = 1
    real_idx = real_small > 0
    n_real = int(real_idx.sum())
    mk = pitch.sample_markings(1.0)
    mk_h = np.hstack([mk, np.ones((len(mk), 1))])
    krn = np.ones((3, 3), np.uint8)

    def recall(H):
        if n_real < 30:
            return 0.0
        try:
            Hi = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return 0.0
        q = (Hi @ mk_h.T).T
        ok = np.abs(q[:, 2]) > 1e-6
        u = (q[ok, 0] / q[ok, 2]) / down
        v = (q[ok, 1] / q[ok, 2]) / down
        m = (u >= 0) & (u < Ws) & (v >= 0) & (v < Hs)
        pred = np.zeros((Hs, Ws), np.uint8)
        if m.any():
            pred[v[m].astype(int), u[m].astype(int)] = 1
            pred = cv2.dilate(pred, krn)
        return float((pred[real_idx] > 0).mean())

    rseed = np.asarray(seed_rvec, float) if seed_rvec is not None else np.zeros(3)
    bounds = [
        (-95, 95),
        (-95, 95),
        (-48, -3),  # camera position (m), z<0 = up
        (rseed[0] - 1.2, rseed[0] + 1.2),
        (rseed[1] - 1.2, rseed[1] + 1.2),
        (rseed[2] - 1.2, rseed[2] + 1.2),  # rotation (rad) around seed
        (np.log(700), np.log(6500)),
    ]  # log focal

    def obj(p):
        H = homography_7dof(p, principal, pitch)
        if H is None or not np.all(np.isfinite(H)):
            return 5.0
        fp = _feasibility_penalty(H, w, h, pitch)
        if fp > 0:
            return 1.0 + 0.02 * fp  # soft barrier (gradient -> feasible)
        ls = line_support(H, dist_field, pitch, radius=radius)  # precision
        cov = recall(H)  # recall (fast, every eval)
        return -(ls * (cov + 0.05))  # continuous; recall kills the smear

    best = (None, 0.0, 0.0, None, -1.0)
    for k in range(restarts):
        r = differential_evolution(
            obj,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-4,
            mutation=(0.4, 1.2),
            recombination=0.8,
            seed=k,
            polish=True,
            init="sobol",
        )
        H = homography_7dof(r.x, principal, pitch)
        if H is None or _feasibility_penalty(H, w, h, pitch) > 0:
            continue
        ls = line_support(H, dist_field, pitch, radius=radius)
        cov = recall(H)
        if ls * cov > best[4]:
            best = (H, ls, cov, r.x, ls * cov)
    return best[:4]


def cold_start(
    detector,
    video_path,
    position,
    principal,
    pitch: PitchModel = STANDARD_PITCH,
    sample_frames: int = 3,
    n_starts: int = 25,
    radius: float = 8.0,
):
    """Find ONE good calibration for the fixed camera: wide multi-start optimization on a few
    frames (seeded by whatever the CNN gives, even if wrong), keep the global best. Returns
    (best_line_support, rvec, focal)."""
    from soccer_analyzer.calibration.line_support import line_distance_field

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    best = (0.0, np.zeros(3), float(getattr(getattr(detector, "fusion", None), "focal", 2000.0)))
    for fi in np.linspace(total * 0.3, total * 0.6, sample_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f = cap.read()
        if not ok:
            continue
        try:
            o = detector.detect(f, 0)
        except Exception:
            o = None
        seed = rvec_of(o.cam_params["rotation_matrix"]) if (o and o.cam_params) else best[1]
        sf = o.cam_params["x_focal_length"] if (o and o.cam_params) else best[2]
        df = line_distance_field(f)
        if df is None:
            continue
        H, ls, x = optimize_calibration(
            df,
            position,
            principal,
            seed,
            sf,
            pitch,
            n_starts=n_starts,
            radius=radius,
            jitter=(0.15, 0.15, 0.10, 400.0),
        )
        if ls > best[0]:
            best = (ls, x[:3], x[3])
    cap.release()
    return best
