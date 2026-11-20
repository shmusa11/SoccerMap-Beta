"""Synthetic view-registration tests; these do not establish real-venue accuracy."""

import cv2
import numpy as np

from soccer_analyzer.calibration.registration import RegisteredAnchor


def texture():
    rng = np.random.default_rng(6)
    image = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
    return cv2.GaussianBlur(image, (3, 3), 0)


def test_registered_pan_preserves_original_pitch_coordinates():
    frame = texture()
    anchor = RegisteredAnchor(frame, np.diag([0.1, 0.1, 1.0]))
    pan = np.float32([[1, 0, 35], [0, 1, 7]])
    current = cv2.warpAffine(frame, pan, (640, 360))
    matrix = anchor.propose(current)
    assert matrix is not None
    actual = cv2.perspectiveTransform(np.float32([[[335, 187]]]), matrix)[0, 0]
    np.testing.assert_allclose(actual, [30, 18], atol=0.15)


def test_registration_rejects_blank_and_unrelated_views():
    frame = texture()
    anchor = RegisteredAnchor(frame, np.eye(3))
    assert anchor.propose(np.zeros_like(frame)) is None
    unrelated = np.random.default_rng(100).integers(0, 255, frame.shape, dtype=np.uint8)
    assert anchor.propose(unrelated) is None
    assert RegisteredAnchor(np.zeros_like(frame), np.eye(3)).propose(frame) is None
