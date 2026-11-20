"""Resource budgets are exercised without allocating model-sized tensors."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from soccer_analyzer.resources import MIB, check_memory, configure_inference, is_memory_error


def test_memory_budget_and_system_reserve():
    check_memory(1000 * MIB, 3000 * MIB)
    with pytest.raises(MemoryError, match="4 GB"):
        check_memory(4097 * MIB, 3000 * MIB)
    with pytest.raises(MemoryError, match="1 GB"):
        check_memory(1000 * MIB, 1000 * MIB)


def test_low_memory_mode_uses_cpu_and_limits_threads(monkeypatch):
    torch = SimpleNamespace(set_num_threads=Mock())
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    assert configure_inference(True) == "cpu"
    torch.set_num_threads.assert_called_once_with(2)


def test_accelerated_mode_sets_allocation_limit(monkeypatch):
    torch = SimpleNamespace(
        set_num_threads=Mock(),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(set_per_process_memory_fraction=Mock()),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    assert configure_inference(False) is None
    torch.mps.set_per_process_memory_fraction.assert_called_once_with(0.25)


def test_model_oom_is_not_treated_as_a_bad_field_observation():
    assert is_memory_error(MemoryError())
    assert is_memory_error(RuntimeError("MPS backend out of memory"))
    assert not is_memory_error(ValueError("No field points"))
