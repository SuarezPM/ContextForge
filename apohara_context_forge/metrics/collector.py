"""Metrics collector — VRAM, TTFT, token stats.

V6.1 truth-up: removed the corrupted `--showgpu占用率` rocm-smi flag that
silently fell back to hardcoded `(45.0, 192.0)` GB on every call. VRAM
readings now flow through the existing `VRAMMonitor` (pyrsmi →
/sys/class/drm → 192 GB total default), and the snapshot's
`vram_source` field reports the actual backend so the dashboard cannot
falsely claim a measurement provenance it does not have.
"""
import asyncio
import logging
import subprocess
from datetime import datetime
from typing import Iterable, Optional, Tuple

from apohara_context_forge.models import (
    CompressionDecision,
    Degradation,
    MetricsSnapshot,
)
from apohara_context_forge.metrics.vram_monitor import VRAMMonitor

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects real GPU metrics via `VRAMMonitor` (pyrsmi /
    /sys/class/drm) with a documented synthetic fallback for non-ROCm
    development hosts.

    The collector does NOT call `rocm-smi` directly anymore; the previous
    implementation hardcoded a malformed flag and silently masked failure
    as a fake (45.0, 192.0) GB tuple. Use `VRAMMonitor` instead — it
    exposes a real numeric path on MI300X hosts and an explicit
    `_fallback_used_bytes()` path everywhere else.
    """

    # Tag returned by `vram_source` when no real GPU telemetry is
    # available. Anything but this value means a real backend served the
    # number.
    SYNTHETIC_SOURCE = "synthetic-dev"

    def __init__(self):
        self._tokens_processed = 0
        self._tokens_saved = 0
        self._ttft_records: list[float] = []
        self._active_agents = 0
        # Surface counters for the MCP server endpoints. record_register
        # fires once per /tools/register_context call (with
        # matched=False since the simple endpoint doesn't try
        # cross-agent dedup); record_decision fires once per successful
        # /tools/get_optimized_context call.
        self._register_calls: list[bool] = []
        self._decision_calls: list[CompressionDecision] = []
        # VRAMMonitor handles its own backend negotiation (pyrsmi →
        # /sys/class/drm → synthetic). We delegate.
        try:
            self._vram_monitor = VRAMMonitor()
            self._vram_monitor._init()
            self._vram_source = self._resolve_vram_source()
        except Exception as exc:
            logger.warning("VRAMMonitor unavailable (%s); using synthetic source.", exc)
            self._vram_monitor = None
            self._vram_source = self.SYNTHETIC_SOURCE

    # ------------------------------------------------------------------ #
    # Backend detection                                                    #
    # ------------------------------------------------------------------ #

    def _resolve_vram_source(self) -> str:
        """Identify which backend `VRAMMonitor` actually negotiated.

        We can't read this directly from `VRAMMonitor` (it lazy-initialises),
        so we probe: if `pyrsmi` is importable AND `/opt/rocm` exists, the
        monitor uses pyrsmi; if only `/sys/class/drm/card0/device/mem_info_vram_used`
        exists, it falls back there; otherwise `synthetic-dev`.
        """
        try:
            import pyrsmi  # noqa: F401
            import os
            if os.path.isdir("/opt/rocm"):
                return "pyrsmi"
        except ImportError:
            pass
        import os
        if os.path.exists("/sys/class/drm/card0/device/mem_info_vram_used"):
            return "sysfs-drm"
        return self.SYNTHETIC_SOURCE

    def _check_rocm(self) -> bool:
        """Best-effort probe for the rocm-smi binary. Kept for backwards
        compatibility with callers that expect this method to exist."""
        try:
            result = subprocess.run(
                ["/opt/rocm/bin/rocm-smi", "--showid"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    # ------------------------------------------------------------------ #
    # VRAM read path                                                      #
    # ------------------------------------------------------------------ #

    async def get_vram_usage(self) -> Tuple[float, float]:
        """Return `(used_gb, total_gb)` from the configured backend.

        On real hardware this reflects pyrsmi or `/sys/class/drm` readings.
        On development hosts without ROCm, returns the synthetic
        fallback documented on `VRAMMonitor._fallback_used_bytes()` —
        explicit, not pretending.
        """
        if self._vram_monitor is None:
            return 0.0, 192.0
        # VRAMMonitor reads are synchronous file IO; wrap to keep the
        # surface async-friendly without blocking the loop in the rare
        # case the backend is slow.
        loop = asyncio.get_event_loop()
        used = await loop.run_in_executor(None, self._vram_monitor.get_used_gb)
        total = await loop.run_in_executor(None, self._vram_monitor.get_total_gb)
        return used, total

    # ------------------------------------------------------------------ #
    # Recording API                                                       #
    # ------------------------------------------------------------------ #

    async def record_ttft(self, ttft_ms: float) -> None:
        """Record time-to-first-token in milliseconds."""
        self._ttft_records.append(ttft_ms)
        if len(self._ttft_records) > 1000:
            self._ttft_records = self._ttft_records[-1000:]

    async def record_tokens(self, original: int, final: int) -> None:
        """Record token counts for compression tracking."""
        self._tokens_processed += original
        self._tokens_saved += max(0, original - final)

    async def set_active_agents(self, count: int) -> None:
        """Set number of active agents."""
        self._active_agents = count

    def record_register(self, matched: bool) -> None:
        """Record a /tools/register_context call. `matched` is True when
        LSH cross-agent dedup found a reusable block; False otherwise."""
        self._register_calls.append(matched)

    def record_decision(self, decision: CompressionDecision) -> None:
        """Record a successful /tools/get_optimized_context decision."""
        self._decision_calls.append(decision)

    def _resolve_gpu_label(self) -> str:
        """Return a short label identifying the active GPU backend.

        ROCm hosts: "rocm". Anything else: "cpu". The /health endpoint
        passes whatever this returns straight through to clients, so any
        exception raised here is caught upstream and reported as the
        degraded path.
        """
        return "rocm" if self._vram_source in ("pyrsmi", "sysfs-drm") else "cpu"

    # ------------------------------------------------------------------ #
    # Snapshot                                                            #
    # ------------------------------------------------------------------ #

    async def snapshot(
        self,
        *,
        current_compressor_model: Optional[str] = None,
        compressor_degradations: Optional[Iterable[Degradation]] = None,
    ) -> MetricsSnapshot:
        """Capture current metrics snapshot.

        Optional kwargs let the MCP server inject compressor identity and
        degradation events captured during this snapshot window — neither
        is known to the collector itself, so we accept them at the
        boundary.

        `vram_source` reflects the *actual* backend that produced the
        VRAM numbers in this snapshot — `pyrsmi`, `sysfs-drm`, or
        `synthetic-dev`. The previous implementation lied and always
        reported `rocm-smi`.
        """
        vram_used, vram_total = await self.get_vram_usage()
        avg_ttft = sum(self._ttft_records) / len(self._ttft_records) if self._ttft_records else 0.0
        dedup_rate = (self._tokens_saved / self._tokens_processed * 100) if self._tokens_processed > 0 else 0.0
        compression_ratio = (self._tokens_processed / (self._tokens_processed - self._tokens_saved)) if self._tokens_saved > 0 else 1.0

        return MetricsSnapshot(
            timestamp=datetime.now(),
            vram_source=self._vram_source,
            compressor_model=current_compressor_model or "xlm-roberta-large",
            vram_used_gb=vram_used,
            vram_total_gb=vram_total,
            ttft_ms=avg_ttft,
            tokens_processed=self._tokens_processed,
            tokens_saved=self._tokens_saved,
            dedup_rate=dedup_rate,
            compression_ratio=compression_ratio,
            active_agents=self._active_agents,
            degradations=list(compressor_degradations) if compressor_degradations else [],
        )

    # ------------------------------------------------------------------ #
    # Backwards-compat: callers that toggled on this property             #
    # ------------------------------------------------------------------ #

    @property
    def _use_rocm(self) -> bool:
        """True iff the VRAM backend is a real ROCm path. Kept for
        backwards-compatibility with callers (and tests) that read this
        attribute. Prefer `vram_source` for new code."""
        return self._vram_source in ("pyrsmi", "sysfs-drm")
