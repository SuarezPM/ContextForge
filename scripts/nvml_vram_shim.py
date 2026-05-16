"""NVIDIA pynvml VRAM reader — additive complement to apohara_context_forge.metrics.vram_monitor.

Canonical VRAMMonitor reads AMD MI300X via PyRSMI. This shim provides
the NVIDIA-side equivalent for benchmark harnesses (run_milan_h100.py)
that need real H100 numbers, without modifying the AMD-targeted core.

Apache-2.0 — Apohara ContextForge.
"""
from __future__ import annotations

import subprocess


_INITIALIZED = False


def _ensure_init() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    import pynvml
    pynvml.nvmlInit()
    _INITIALIZED = True


def read_vram_mib(device_index: int = 0) -> float:
    """Currently used VRAM on the given NVIDIA device in MiB."""
    _ensure_init()
    import pynvml
    info = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device_index))
    return info.used / 1024 / 1024


def read_vram_gb(device_index: int = 0) -> float:
    return read_vram_mib(device_index) / 1024


def nvidia_smi_dict(device_index: int = 0) -> dict:
    """Compact nvidia-smi-equivalent dict for log emission."""
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
    """pynvml vs nvidia-smi sanity check; True if within tolerance."""
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    smi_used_mib = float(smi.stdout.strip().split("\n")[0])
    return abs(smi_used_mib - read_vram_mib(0)) < tolerance_mib


if __name__ == "__main__":
    import json
    print(json.dumps(nvidia_smi_dict(), indent=2))
    print(f"cross_validate (within 500 MiB): {_cross_validate()}")
