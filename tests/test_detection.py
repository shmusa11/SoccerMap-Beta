import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from soccer_analyzer.detection import SoccerDetector


@pytest.mark.parametrize(
    "error", [MemoryError("allocation failed"), RuntimeError("MPS out of memory")]
)
def test_optional_soccer_model_propagates_memory_failure(monkeypatch, error):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-only-placeholder")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
    monkeypatch.setitem(
        sys.modules, "inference", SimpleNamespace(get_model=Mock(side_effect=error))
    )
    original_path = sys.path[:]
    with pytest.raises(type(error), match=str(error)):
        SoccerDetector._load_soccer("test-model")
    assert sys.path == original_path


def test_optional_soccer_model_missing_dependency_still_falls_back(monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "test-only-placeholder")
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: None)
    monkeypatch.setitem(
        sys.modules,
        "inference",
        SimpleNamespace(get_model=Mock(side_effect=ImportError("optional package"))),
    )
    assert SoccerDetector._load_soccer("test-model") is None
