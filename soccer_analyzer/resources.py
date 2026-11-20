"""Per-run resource controls for local inference; never change system-wide settings."""

from __future__ import annotations

import gc
import os
import platform
import shutil
import sys
from pathlib import Path

MIB = 1024 * 1024
WORKER_MEMORY_MB = 4096
MIN_AVAILABLE_MB = 1024
MIN_RAM_BYTES = 16 * 1024**3  # tested reference minimum
MIN_FREE_DISK_BYTES = 5 * 1024**3


def system_report(directory: Path) -> dict:
    """Describe this computer before a run: OS, CPU, RAM, free disk and Python version.

    Free disk is measured where `directory` lives. `qualified` is False below the tested
    minimum of 16 GB RAM and 5 GB free disk. No paths or user names are included.
    """
    import psutil

    ram = psutil.virtual_memory().total
    free = shutil.disk_usage(directory).free
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_bytes": ram,
        "free_disk_bytes": free,
        "python": platform.python_version(),
        "qualified": ram >= MIN_RAM_BYTES and free >= MIN_FREE_DISK_BYTES,
    }


def check_memory(used: int, available: int, limit_mb: int = WORKER_MEMORY_MB) -> None:
    """Raise before a worker exhausts its RAM budget or the computer's remaining memory."""
    if used > limit_mb * MIB:
        raise MemoryError(
            "Analysis exceeded its 4 GB memory budget. Try a shorter passage, or turn off Low memory mode if this computer has a GPU."
        )
    if available < MIN_AVAILABLE_MB * MIB:
        raise MemoryError(
            "Analysis stopped because this computer has less than 1 GB of available memory. Close other applications before retrying in Low memory mode."
        )


def configure_inference(low_memory: bool) -> str | None:
    """Bound CPU threads and accelerator allocations in the isolated analysis worker.

    Low memory mode uses CPU inference. Accelerated mode caps MPS allocations at 25% of
    Metal's recommended working set and CUDA allocations at 40% of visible device memory.
    These allocator limits complement the worker's RSS/system-memory monitor.
    """
    import torch

    torch.set_num_threads(2 if low_memory else 4)
    if low_memory:
        return "cpu"
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.25)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.4)
    return None


def release_model_memory() -> None:
    """Release unreferenced models and unused accelerator cache between pipeline stages."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_memory_error(error: Exception) -> bool:
    """Recognize allocator failures without importing optional inference libraries."""
    return isinstance(error, MemoryError) or "out of memory" in str(error).lower()
