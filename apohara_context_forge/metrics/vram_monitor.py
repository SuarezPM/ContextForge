"""Zero-overhead GPU memory monitor — fixes BUG-003 / IMPROVEMENT-004.

Backend-neutral: two equal-level read paths, each naming HONESTLY which
backend produced the numbers via a ``vram_source`` field.

AMD / ROCm path
---------------
Native PyRSMI C bindings (no subprocess, no shell, no event loop blocking,
<1ms overhead) replacing the blocking subprocess.run(["rocm-smi"]); falls
back to /sys/class/drm. Install: ``pip install pyrsmi``
(docs: https://github.com/ROCm/pyrsmi).

NVIDIA / CUDA path
------------------
PyRSMI only reads AMD GPUs. On an NVIDIA box the AMD path returns 0 used
bytes and (historically) the 192 GB MI300X default for total — a value
that is simply a lie on a 24/40/80 GB NVIDIA card. The CUDA path (via
scripts/nvml_vram_shim -> pynvml, nvidia-smi, or torch.cuda.memory_reserved)
supersedes it: when CUDA is active the AMD 192 GB default is NEVER used; if
no CUDA source can be read the monitor reports 0 bytes with
``vram_source == "cuda_unavailable"`` rather than fabricating a total.

All GPU dependencies are import-guarded so this module imports on a box
with neither ROCm nor CUDA installed.
"""
import asyncio
import logging
import shutil
import subprocess
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# vram_source labels — each names HONESTLY which backend produced the numbers.
SOURCE_PYRSMI = "pyrsmi"                  # AMD ROCm native C bindings
SOURCE_DRM_SYSFS = "drm_sysfs"            # AMD /sys/class/drm fallback
SOURCE_AMD_DEFAULT = "amd_default_192gb"  # AMD last-resort 192 GB MI300X guess
SOURCE_CUDA_NVML = "cuda_nvml"            # NVIDIA pynvml (via nvml_vram_shim)
SOURCE_CUDA_SMI = "cuda_nvidia_smi"       # NVIDIA nvidia-smi subprocess
SOURCE_CUDA_TORCH = "cuda_torch"          # torch.cuda.* (reserved, not device-wide)
SOURCE_CUDA_UNAVAILABLE = "cuda_unavailable"  # NVIDIA detected, no reader worked
SOURCE_UNKNOWN = "unknown"


class VRAMMonitor:
    """
    Zero-overhead AMD GPU memory monitor using PyRSMI native C bindings.
    
    MI300X specs:
    - 192GB HBM3 total
    - PyRSMI reads via ROCm SMI kernel driver (/dev/mem mapped)
    - Native bindings return bytes directly, no shell parsing
    
    Usage:
        monitor = VRAMMonitor()
        monitor.start()  # Start background monitoring
        pressure = monitor.get_pressure()  # 0.0-1.0
        mode = monitor.get_eviction_mode()  # "relaxed", "normal", "pressure", "critical", "emergency"
        used_gb = monitor.get_used_gb()
        available_gb = monitor.get_available_gb()
        monitor.stop()
    """
    
    VRAM_CHECK_INTERVAL = 2.0  # seconds between checks
    
    def __init__(self, device_id: int = 0):
        self._device_id = device_id
        self._initialized = False
        self._pyrsml = None
        self._current_pressure = 0.0
        # vram_source is set by the read path that actually produces the bytes.
        # Until a read happens it reflects which backend init selected.
        self._vram_source = SOURCE_UNKNOWN
        # Backend the monitor will read from: "amd" (PyRSMI/DRM) or "cuda".
        # When "cuda" is selected the AMD 192 GB default is NEVER returned.
        self._is_cuda = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._init()

    def _init(self) -> None:
        """Initialize PyRSMI; if absent, detect NVIDIA and arm the CUDA path.

        Fails gracefully if neither backend is available — the module still
        imports and reads (reporting 0 bytes with an honest ``vram_source``).
        """
        try:
            from pyrsmi import rocml
            rocml.smi_initialize()
            self._pyrsml = rocml
            self._initialized = True
            self._vram_source = SOURCE_PYRSMI
            logger.info(f"PyRSMI initialized for device {self._device_id}")
            return
        except ImportError:
            logger.warning(
                "pyrsmi not available. Install with: pip install pyrsmi. "
                "Falling back to /sys/class/drm (read-only, ~5ms overhead)."
            )
        except Exception as e:
            logger.error(f"PyRSMI initialization failed: {e}")

        # No working AMD backend. Probe for NVIDIA so we never hand back the
        # AMD 192 GB default on a CUDA box.
        if self._detect_cuda():
            self._is_cuda = True
            self._vram_source = SOURCE_CUDA_UNAVAILABLE
            logger.info(
                "NVIDIA/CUDA detected; VRAMMonitor will read via the CUDA path "
                "(nvml/nvidia-smi/torch). AMD 192 GB default is disabled."
            )

    @staticmethod
    def _detect_cuda() -> bool:
        """True if this host looks like NVIDIA/CUDA (not AMD ROCm).

        Cheap, import-guarded checks in order of reliability: pynvml init,
        nvidia-smi on PATH, torch.cuda. Any positive signal arms the CUDA path.
        """
        try:
            import pynvml  # noqa: F401
            pynvml.nvmlInit()
            try:
                count = pynvml.nvmlDeviceGetCount()
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            if count > 0:
                return True
        except Exception:
            pass
        if shutil.which("nvidia-smi") is not None:
            return True
        try:
            import torch
            if torch.cuda.is_available() and torch.version.hip is None:
                return True
        except Exception:
            pass
        return False
    
    async def start(self) -> None:
        """Start background VRAM monitoring loop."""
        if self._monitor_task is not None:
            return
        self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop(self) -> None:
        """Stop background monitoring."""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
    
    async def _monitor_loop(self) -> None:
        """Background loop: updates pressure every VRAM_CHECK_INTERVAL."""
        while True:
            try:
                self._current_pressure = self.get_pressure()
                await asyncio.sleep(self.VRAM_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"VRAM monitor loop error: {e}")
    
    def get_used_bytes(self) -> int:
        """Get used VRAM in bytes."""
        if self._initialized and self._pyrsml:
            try:
                used = self._pyrsml.smi_get_device_memory_used(self._device_id)
                self._vram_source = SOURCE_PYRSMI
                return used
            except Exception as e:
                logger.warning(f"PyRSMI get_used_bytes failed: {e}")
        if self._is_cuda:
            return self._cuda_used_bytes()
        self._vram_source = SOURCE_DRM_SYSFS
        return self._fallback_used_bytes()

    def get_total_bytes(self) -> int:
        """Get total VRAM in bytes."""
        if self._initialized and self._pyrsml:
            try:
                total = self._pyrsml.smi_get_device_memory_total(self._device_id)
                self._vram_source = SOURCE_PYRSMI
                return total
            except Exception as e:
                logger.warning(f"PyRSMI get_total_bytes failed: {e}")
        if self._is_cuda:
            return self._cuda_total_bytes()
        return self._fallback_total_bytes()  # sets _vram_source honestly
    
    def get_available_bytes(self) -> int:
        """Get available VRAM in bytes."""
        return self.get_total_bytes() - self.get_used_bytes()
    
    def get_used_gb(self) -> float:
        """Get used VRAM in gigabytes."""
        return self.get_used_bytes() / (1024 ** 3)
    
    def get_total_gb(self) -> float:
        """Get total VRAM in gigabytes."""
        return self.get_total_bytes() / (1024 ** 3)
    
    def get_available_gb(self) -> float:
        """Get available VRAM in gigabytes."""
        return self.get_available_bytes() / (1024 ** 3)
    
    def get_pressure(self) -> float:
        """
        Returns VRAM utilization 0.0–1.0. <1ms overhead.
        
        Returns:
            Pressure ratio (0.0 = free, 1.0 = saturated)
        """
        total = self.get_total_bytes()
        if total == 0:
            return 0.0
        return self.get_used_bytes() / total
    
    def get_eviction_mode(self) -> str:
        """
        Returns eviction mode based on VRAM pressure.

        Returns:
            One of: "relaxed", "normal", "pressure", "critical", "emergency"
        """
        p = self.get_pressure()
        if p < 0.70:   return "relaxed"
        if p < 0.85:   return "normal"
        if p < 0.92:   return "pressure"
        if p < 0.96:   return "critical"
        return "emergency"

    def get_vram_source(self) -> str:
        """Honest label of the backend that produced the last byte reading.

        One of the module-level ``SOURCE_*`` constants. On NVIDIA this is a
        ``cuda_*`` value and is NEVER ``amd_default_192gb``.
        """
        return self._vram_source

    def _cuda_used_bytes(self) -> int:
        """Used VRAM on NVIDIA via nvml shim -> nvidia-smi -> torch.

        Sets ``_vram_source`` to the backend that succeeded, or
        ``cuda_unavailable`` (and returns 0) if none worked. The AMD 192 GB
        default is intentionally unreachable from this path.
        """
        # 1) pynvml via the existing shim (device-wide used bytes, ground truth)
        try:
            from scripts.nvml_vram_shim import read_vram_mib
            used = int(round(read_vram_mib(self._device_id) * (1024 ** 2)))
            self._vram_source = SOURCE_CUDA_NVML
            return used
        except Exception as e:
            logger.debug(f"CUDA nvml used read failed: {e}")
        # 2) nvidia-smi subprocess (device-wide used bytes)
        smi = self._nvidia_smi_used_total_bytes()
        if smi is not None:
            self._vram_source = SOURCE_CUDA_SMI
            return smi[0]
        # 3) torch — process-reserved only, NOT device-wide; honestly labelled
        try:
            import torch
            if torch.cuda.is_available():
                self._vram_source = SOURCE_CUDA_TORCH
                return int(torch.cuda.memory_reserved(self._device_id))
        except Exception as e:
            logger.debug(f"CUDA torch used read failed: {e}")
        self._vram_source = SOURCE_CUDA_UNAVAILABLE
        return 0

    def _cuda_total_bytes(self) -> int:
        """Total VRAM on NVIDIA via nvml shim -> nvidia-smi -> torch.

        Returns 0 with ``_vram_source == cuda_unavailable`` if no reader works,
        rather than fabricating the 192 GB MI300X total.
        """
        # 1) pynvml via the shim
        try:
            from scripts.nvml_vram_shim import nvidia_smi_dict
            total_mib = nvidia_smi_dict(self._device_id)["total_mib"]
            self._vram_source = SOURCE_CUDA_NVML
            return int(round(total_mib * (1024 ** 2)))
        except Exception as e:
            logger.debug(f"CUDA nvml total read failed: {e}")
        # 2) nvidia-smi subprocess
        smi = self._nvidia_smi_used_total_bytes()
        if smi is not None:
            self._vram_source = SOURCE_CUDA_SMI
            return smi[1]
        # 3) torch device total properties
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(self._device_id)
                self._vram_source = SOURCE_CUDA_TORCH
                return int(props.total_memory)
        except Exception as e:
            logger.debug(f"CUDA torch total read failed: {e}")
        self._vram_source = SOURCE_CUDA_UNAVAILABLE
        return 0

    def _nvidia_smi_used_total_bytes(self) -> Optional[tuple[int, int]]:
        """(used_bytes, total_bytes) from nvidia-smi, or None on any failure.

        Parses ``--query-gpu=memory.used,memory.total --format=csv,noheader,
        nounits`` (values are MiB).
        """
        if shutil.which("nvidia-smi") is None:
            return None
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self._device_id}",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5.0,
            )
            line = proc.stdout.strip().split("\n")[0]
            used_mib, total_mib = (float(x.strip()) for x in line.split(","))
            return (
                int(round(used_mib * (1024 ** 2))),
                int(round(total_mib * (1024 ** 2))),
            )
        except Exception as e:
            logger.debug(f"nvidia-smi read failed: {e}")
            return None

    @staticmethod
    def _fallback_used_bytes() -> int:
        """
        Fallback: read from Linux DRM sysfs (read-only, ~5ms overhead).
        Works on any Linux system with AMD GPU.
        """
        try:
            with open("/sys/class/drm/card0/device/mem_info_vram_used", "r") as f:
                return int(f.read().strip())
        except Exception:
            return 0
    
    def _fallback_total_bytes(self) -> int:
        """
        Fallback: read from Linux DRM sysfs.
        Default to 192GB MI300X if unable to read.

        Sets ``_vram_source`` to ``drm_sysfs`` when the real sysfs value is read
        and to ``amd_default_192gb`` only when it actually falls back to the
        hard-coded MI300X total — so the 192 GB guess is always labelled as a
        guess and is unreachable on the CUDA path (which never calls this).
        """
        try:
            with open("/sys/class/drm/card0/device/mem_info_vram_total", "r") as f:
                self._vram_source = SOURCE_DRM_SYSFS
                return int(f.read().strip())
        except Exception:
            # MI300X has 192GB HBM3
            self._vram_source = SOURCE_AMD_DEFAULT
            return 192 * (1024 ** 3)
    
    def __del__(self):
        """Cleanup PyRSMI on destruction."""
        if self._initialized and self._pyrsml:
            try:
                self._pyrsml.smi_shutdown()
            except Exception:
                pass


# Module-level singleton
_monitor: Optional[VRAMMonitor] = None
_monitor_lock = threading.Lock()


def get_monitor() -> VRAMMonitor:
    """Get or create module-level VRAMMonitor singleton."""
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = VRAMMonitor()
    return _monitor


def get_vram_pressure() -> float:
    """Quick VRAM pressure check."""
    return get_monitor().get_pressure()


def get_vram_used_gb() -> float:
    """Quick VRAM used GB."""
    return get_monitor().get_used_gb()


def get_vram_available_gb() -> float:
    """Quick VRAM available GB."""
    return get_monitor().get_available_gb()


def get_eviction_mode() -> str:
    """Quick eviction mode check."""
    return get_monitor().get_eviction_mode()


def get_vram_source() -> str:
    """Quick honest backend label for the last VRAM reading."""
    return get_monitor().get_vram_source()
