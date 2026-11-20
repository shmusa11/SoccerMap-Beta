"""WASB post-processing tests run without torch, weights or external model repositories."""

import numpy as np
import pytest

from soccer_analyzer.ball_wasb import WASBBallDetector


def detector():
    model = WASBBallDetector.__new__(WASBBallDetector)
    model.thresh = 0.3
    model.in_w, model.in_h = 512, 288
    model.max_batch = 2
    return model


def test_weighted_blob_centers_and_scores():
    model = detector()
    hm = np.zeros((10, 10), np.float32)
    hm[2, 3], hm[2, 4], hm[8, 8] = 0.8, 0.4, 0.9
    out = model._blobs(hm, 2, 3, 10, 20, 0.3)
    assert out[0] == pytest.approx((16 + 2 / 3, 26, 0.8))
    assert out[1] == pytest.approx((26, 44, 0.9))


def test_overlapping_tiles_are_bounded_and_deduplicated():
    model = detector()
    batches = []

    def infer(stacks):
        batches.append([stack[0].shape for stack in stacks])
        return np.zeros((len(stacks), 288, 512), np.float32)

    model._infer_hm_batch = infer
    model._blobs = lambda *args: [(960, 540, 0.8)]
    frames = [np.zeros((1080, 1920, 3), np.uint8)] * 3
    assert model.detect(frames) == [(960, 540, 0.8)]
    assert len(batches) == 2 and all(len(b) == 2 for b in batches)
    assert all(shape[:2] == (572, 992) for batch in batches for shape in batch)


def test_invalid_temporal_input_rejected():
    model = detector()
    with pytest.raises(ValueError, match="three consecutive"):
        model.detect([np.zeros((20, 20, 3), np.uint8)] * 2)


def _detector(max_batch):
    # Only these tests need torch (a fake network); the ones above run without it.
    torch = pytest.importorskip("torch")

    w = object.__new__(WASBBallDetector)  # no weights needed: a fake network below
    w.torch, w.device, w.max_batch, w.thresh = torch, "cpu", max_batch, 0.5
    w.in_w, w.in_h = 512, 288
    w.passes = []

    def model(x):
        w.passes.append(len(x))
        # channel 1 = the centre frame's map: a logit that encodes the tile's own input
        return [x[:, 3:6].mean(1, keepdim=True).repeat(1, 3, 1, 1)]

    w.model = model
    return w


@pytest.mark.parametrize("width,passes", [(1920, [4]), (3840, [3, 3]), (1280, [1])])
def test_tiles_go_in_equal_passes_and_keep_their_order(width, passes):
    rng = np.random.default_rng(0)
    h = width * 9 // 16
    frames = [(rng.random((h, width, 3)) * 255).astype(np.uint8) for _ in range(3)]
    w = _detector(max_batch=4)
    inputs, tiles = w.prepare(frames)
    hms = w._infer_inputs(inputs)
    assert w.passes == passes and len(hms) == len(tiles) == sum(passes)
    expect = 1.0 / (1.0 + np.exp(-inputs[:, 3:6].mean(1)))
    assert np.allclose(hms, expect, atol=1e-5)
    # the split API gives exactly what `peaks` gives
    assert w.peaks_prepared((inputs, tiles)) == w.peaks(frames)
