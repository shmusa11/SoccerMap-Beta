import pytest

from soccer_analyzer.evaluation import evaluate_ball


def test_ball_metrics_separate_wrong_locations_absence_and_unlabelled_frames():
    result = {
        "synthetic": False,
        "frames": [
            {"index": 0, "ball_image": [10, 10], "ball_measured": True},
            {"index": 3, "ball_image": [100, 100], "ball_measured": False},
            {"index": 6, "ball_image": [10, 10], "ball_measured": True},
            {"index": 9, "ball_image": [10, 10], "ball_measured": True},
        ],
    }
    annotations = [
        {"index": 0, "visible": True, "point": [10, 11]},
        {"index": 3, "visible": True, "point": [20, 20]},
        {"index": 6, "visible": False},
        {"index": 12, "visible": True, "point": [1, 2]},
    ]
    metrics = evaluate_ball(result, annotations)
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 2
    assert metrics["false_negatives"] == 2
    assert metrics["recall"] == pytest.approx(1 / 3)
    assert metrics["unprocessed_labelled_frames"] == 1
    assert evaluate_ball(result, annotations, measured_only=True)["false_positives"] == 1


def test_empty_metrics_stay_unavailable_and_bad_labels_are_rejected():
    result = {"synthetic": False, "frames": []}
    assert evaluate_ball(result, [])["f1"] is None
    with pytest.raises(ValueError, match="Synthetic"):
        evaluate_ball({"synthetic": True}, [])
    with pytest.raises(ValueError, match="unique"):
        evaluate_ball(result, [{"index": 1, "visible": False}] * 2)
    with pytest.raises(ValueError, match="coordinates"):
        evaluate_ball(result, [{"index": 1, "visible": True, "point": [float("nan"), 3]}])
