"""Tests for the NVIDIA/CUDA degradation path of VRAMMonitor.

These exercise the CUDA read path with KNOWN byte values (mocking the
nvml_vram_shim, nvidia-smi, and torch backends) and assert the cardinal
honesty rules of FASE 2:

* ``vram_source`` reports a ``cuda_*`` label when the CUDA path is used.
* The AMD 192 GB MI300X default is NEVER returned on an NVIDIA context.
* When no CUDA reader works the monitor reports 0 bytes with
  ``vram_source == "cuda_unavailable"`` instead of fabricating a total.

No GPU is touched — every backend is mocked. No vLLM/lmcache needed.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import apohara_context_forge.metrics.vram_monitor as vm


def _cuda_monitor() -> vm.VRAMMonitor:
    """Build a VRAMMonitor wired for the CUDA path without running _init().

    Bypasses _init() (which would probe the real host) and arms the CUDA
    backend directly, as if PyRSMI was absent and NVIDIA was detected.
    """
    m = vm.VRAMMonitor.__new__(vm.VRAMMonitor)
    m._device_id = 0
    m._initialized = False
    m._pyrsml = None
    m._current_pressure = 0.0
    m._vram_source = vm.SOURCE_UNKNOWN
    m._is_cuda = True
    m._monitor_task = None
    return m


def _fake_shim(used_mib: float, total_mib: float) -> types.ModuleType:
    """A stand-in scripts.nvml_vram_shim module returning KNOWN MiB values."""
    mod = types.ModuleType("scripts.nvml_vram_shim")
    mod.read_vram_mib = lambda device_index=0: used_mib
    mod.nvidia_smi_dict = lambda device_index=0: {"total_mib": total_mib}
    return mod


def test_cuda_nvml_known_bytes():
    """nvml shim path returns exact known bytes and labels source cuda_nvml."""
    m = _cuda_monitor()
    # 4096 MiB used, 8192 MiB total — known, exact.
    shim = _fake_shim(4096.0, 8192.0)
    with patch.dict(sys.modules, {"scripts.nvml_vram_shim": shim}):
        assert m.get_used_bytes() == 4096 * (1024 ** 2)
        assert m.get_total_bytes() == 8192 * (1024 ** 2)
    assert m.get_vram_source() == vm.SOURCE_CUDA_NVML
    # Honesty: a 24/40/80 GB NVIDIA card, never the 192 GB MI300X default.
    assert m.get_total_gb() == pytest.approx(8.0)
    assert m.get_total_bytes() != 192 * (1024 ** 3)


def test_cuda_nvml_pressure_and_eviction():
    """Pressure/eviction derive from the mocked CUDA bytes, not AMD defaults."""
    m = _cuda_monitor()
    # ~93.75% utilisation: 7680 MiB of 8192 MiB -> "critical" band (0.92-0.96).
    shim = _fake_shim(7680.0, 8192.0)
    with patch.dict(sys.modules, {"scripts.nvml_vram_shim": shim}):
        p = m.get_pressure()
        mode = m.get_eviction_mode()
    assert p == pytest.approx(7680.0 / 8192.0, rel=1e-6)
    assert mode == "critical"


def test_cuda_nvidia_smi_fallback():
    """nvml absent -> nvidia-smi subprocess produces known bytes (cuda_nvidia_smi)."""
    m = _cuda_monitor()
    fake_proc = MagicMock()
    fake_proc.stdout = "2048, 24576\n"  # used MiB, total MiB
    # nvml shim import fails (None in sys.modules), torch path not reached.
    with patch.dict(sys.modules, {"scripts.nvml_vram_shim": None}), \
         patch.object(vm.shutil, "which", return_value="/usr/bin/nvidia-smi"), \
         patch.object(vm.subprocess, "run", return_value=fake_proc):
        used = m.get_used_bytes()
        total = m.get_total_bytes()
    assert used == 2048 * (1024 ** 2)
    assert total == 24576 * (1024 ** 2)  # 24 GB — a real RTX-class card
    assert m.get_vram_source() == vm.SOURCE_CUDA_SMI
    assert total != 192 * (1024 ** 3)


def test_cuda_torch_fallback():
    """nvml + nvidia-smi absent -> torch reports process-reserved (cuda_torch)."""
    m = _cuda_monitor()
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.memory_reserved.return_value = 1234 * (1024 ** 2)
    fake_props = MagicMock()
    fake_props.total_memory = 6144 * (1024 ** 2)
    fake_torch.cuda.get_device_properties.return_value = fake_props
    with patch.dict(sys.modules, {"scripts.nvml_vram_shim": None, "torch": fake_torch}), \
         patch.object(vm.shutil, "which", return_value=None):
        used = m.get_used_bytes()
        total = m.get_total_bytes()
    assert used == 1234 * (1024 ** 2)
    assert total == 6144 * (1024 ** 2)
    assert m.get_vram_source() == vm.SOURCE_CUDA_TORCH
    assert total != 192 * (1024 ** 3)


def test_cuda_unavailable_reports_zero_not_192gb():
    """No CUDA reader works -> 0 bytes + honest cuda_unavailable, NEVER 192 GB."""
    m = _cuda_monitor()
    # torch present but reporting no CUDA so even that path declines.
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"scripts.nvml_vram_shim": None, "torch": fake_torch}), \
         patch.object(vm.shutil, "which", return_value=None):
        used = m.get_used_bytes()
        total = m.get_total_bytes()
    assert used == 0
    assert total == 0
    assert m.get_vram_source() == vm.SOURCE_CUDA_UNAVAILABLE
    # Cardinal rule: the AMD default is unreachable on the CUDA path.
    assert total != 192 * (1024 ** 3)


def test_cuda_path_never_calls_amd_fallback():
    """The CUDA branch must not reach the AMD sysfs/192 GB fallback at all."""
    m = _cuda_monitor()
    shim = _fake_shim(100.0, 200.0)
    with patch.object(vm.VRAMMonitor, "_fallback_total_bytes") as amd_total, \
         patch.object(vm.VRAMMonitor, "_fallback_used_bytes") as amd_used, \
         patch.dict(sys.modules, {"scripts.nvml_vram_shim": shim}):
        m.get_used_bytes()
        m.get_total_bytes()
    amd_total.assert_not_called()
    amd_used.assert_not_called()


def test_amd_default_is_labelled_honestly():
    """AMD path: the 192 GB guess is labelled amd_default_192gb, not silent."""
    m = vm.VRAMMonitor.__new__(vm.VRAMMonitor)
    m._device_id = 0
    m._initialized = False
    m._pyrsml = None
    m._current_pressure = 0.0
    m._vram_source = vm.SOURCE_UNKNOWN
    m._is_cuda = False  # AMD context, no PyRSMI
    m._monitor_task = None
    # Force the sysfs read to fail so it hits the 192 GB default.
    with patch("builtins.open", side_effect=OSError("no drm sysfs")):
        total = m.get_total_bytes()
    assert total == 192 * (1024 ** 3)
    assert m.get_vram_source() == vm.SOURCE_AMD_DEFAULT


def test_detect_cuda_false_when_nothing_present():
    """_detect_cuda is False when pynvml/nvidia-smi/torch-cuda are all absent."""
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"pynvml": None, "torch": fake_torch}), \
         patch.object(vm.shutil, "which", return_value=None):
        assert vm.VRAMMonitor._detect_cuda() is False


def test_detect_cuda_true_via_nvidia_smi():
    """_detect_cuda is True when nvidia-smi is on PATH (pynvml absent)."""
    with patch.dict(sys.modules, {"pynvml": None}), \
         patch.object(vm.shutil, "which", return_value="/usr/bin/nvidia-smi"):
        assert vm.VRAMMonitor._detect_cuda() is True
