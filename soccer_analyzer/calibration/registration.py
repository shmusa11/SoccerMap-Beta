"""Experimental panorama-style view registration to a verified field anchor.

A tripod pan can share a projective transform with an earlier view. ORB correspondences
propose that transform; callers must still verify the transferred field against current
painted lines. This is a registration experiment, not a stitched panorama or a trusted
replacement for field landmarks. Repetitive markings and parallax can still be ambiguous.
"""

from __future__ import annotations

import cv2
import numpy as np

from soccer_analyzer.calibration.camera import camera_gray


class RegisteredAnchor:
    """A bounded set of descriptors from one independently verified image-to-pitch mapping."""

    def __init__(self, frame: np.ndarray, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, np.float64)
        self.size = frame.shape[:2]
        self.gray = camera_gray(frame, 640)
        self.orb = cv2.ORB_create(nfeatures=1200, edgeThreshold=20, fastThreshold=12)
        self.keypoints, self.descriptors = self.orb.detectAndCompute(self.gray, None)

    def propose(self, frame: np.ndarray) -> np.ndarray | None:
        """Return a candidate image-to-pitch transform supported by distributed matches."""
        if self.descriptors is None or len(self.keypoints) < 30 or frame.shape[:2] != self.size:
            return None
        gray = camera_gray(frame, 640)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 30:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(descriptors, self.descriptors, k=2)
        matches = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
        ]
        if len(matches) < 30:
            return None
        # A repeated texture must not cast many votes for the same anchor feature.
        unique = {}
        for match in sorted(matches, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        if len(matches) < 30:
            return None
        source = np.float32([keypoints[m.queryIdx].pt for m in matches])
        target = np.float32([self.keypoints[m.trainIdx].pt for m in matches])
        transform, mask = cv2.findHomography(
            source, target, cv2.RANSAC, 2.0, maxIters=1000, confidence=0.995
        )
        if transform is None or mask is None or not np.isfinite(transform).all():
            return None
        inliers = mask.ravel().astype(bool)
        if inliers.sum() < 30 or np.mean(inliers) < 0.7:
            return None
        height, width = gray.shape
        for points in (source[inliers], target[inliers]):
            hull = cv2.convexHull(points)
            if cv2.contourArea(hull) < width * height * 0.12:
                return None
        projected = cv2.perspectiveTransform(source[inliers][None], transform)[0]
        if np.median(np.linalg.norm(projected - target[inliers], axis=1)) > 1.2:
            return None
        scale = np.diag([width / frame.shape[1], height / frame.shape[0], 1.0])
        full_transform = np.linalg.inv(scale) @ transform @ scale
        # Restrict proposals to modest view changes with overlap. Large scene jumps need an
        # independent calibration, even if descriptor matching happens to find a consensus.
        corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        mapped = cv2.perspectiveTransform(corners[None], transform)[0]
        area = abs(cv2.contourArea(mapped)) / (width * height)
        if (
            not 0.65 <= area <= 1.6
            or np.max(np.linalg.norm(mapped - corners, axis=1)) > width * 0.4
        ):
            return None
        candidate = self.matrix @ full_transform
        return candidate if np.isfinite(candidate).all() else None
