from unittest.mock import Mock

import pytest

from soccer_analyzer.calibration.multisport import calibrate_multisport


def test_multisport_memory_exhaustion_is_not_swallowed(monkeypatch):
    implementation = Mock(side_effect=MemoryError("stop inference"))
    monkeypatch.setattr("soccer_analyzer.calibration.multisport._calibrate_impl", implementation)
    with pytest.raises(MemoryError):
        calibrate_multisport("unused", start_frame=500, max_frames=100)
    assert implementation.call_args.args[-1] == 500


def test_multisport_ordinary_failure_stays_unavailable(monkeypatch):
    monkeypatch.setattr(
        "soccer_analyzer.calibration.multisport._calibrate_impl",
        Mock(side_effect=ValueError("no field")),
    )
    result, stats = calibrate_multisport("unused")
    assert result == {} and "no field" in stats["error"]
