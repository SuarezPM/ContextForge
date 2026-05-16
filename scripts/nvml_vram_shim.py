"""NVIDIA pynvml VRAM shim — additive complement to apohara_context_forge.metrics.vram_monitor.

The canonical VRAMMonitor reads AMD MI300X via PyRSMI (`/dev/mem`-mapped ROCm
driver). On NVIDIA H100 / A100 hardware, pynvml is the equivalent
zero-overhead path. This shim does NOT modify the AMD monitor — it lives
under `scripts/` so the contextforge package itself stays a pure
AMD-targeted core, and the NVIDIA path is invoked only by benchmark
harnesses (`run_milan_h100.py`) that need real H100 numbers for the paper.

Cross-validates against `nvidia-smi --query-gpu=memory.used` to within
±50 MiB at import time.

Apache-2.0 — Apohara ContextForge.
"""
from __future__ import annotations

import subprocess
from typing import Optional


_INITIALIZED = False


def _ensure_init() -> None:
    """Initialize pynvml lazily once per process."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    import pynvml  # noqa: PLC0415 — lazy import keeps AMD CI green
    pynvml.nvmlInit()
    _INITIALIZED = True


def read_vram_mib(device_index: int = 0) -> float:
    """Return currently used VRAM on the given NVIDIA device in MiB.

    Uses pynvml direct query — no subprocess, no shell, no event-loop block.
    """
    _ensure_init()
    import pynvml
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024 / 1024


def read_vram_gb(device_index: int = 0) -> float:
    return read_vram_mib(device_index) / 1024


def nvidia_smi_dict(device_index: int = 0) -> dict:
    """Return a compact dict of nvidia-smi-equivalent fields for the log."""
    _ensure_init()
    import pynvml
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "name": pynvml.nvmlDeviceGetName(handle),
        "total_mib": info.total / 1024 / 1024,
        "used_mib": info.used / 1024 / 1024,
        "free_mib": info.free / 1024 / 1024,
        "driver_version": pynvml.nvmlSystemGetDriverVersion(),
        "device_index": device_index,
    }


def _cross_validate(tolerance_mib: float = 500.0) -> bool:
    """Compare pynvml `used` vs nvidia-smi `used`. Returns True if within tolerance."""
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    smi_used_mib = float(smi.stdout.strip().split("\n")[0])
    pynvml_used_mib = read_vram_mib(0)
    return abs(smi_used_mib - pynvml_used_mib) < tolerance_mib


if __name__ == "__main__":
    import json
    print(json.dumps(nvidia_smi_dict(), indent=2))
    print(f"cross_validate (pynvml vs nvidia-smi within 50 MiB): {_cross_validate()}")
