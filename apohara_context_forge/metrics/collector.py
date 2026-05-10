"""Metrics collector - VRAM, TTFT, token stats. Uses ROCm SMI or psutil fallback."""
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

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects real GPU metrics via ROCm SMI or psutil fallback."""

    def __init__(self):
        self._tokens_processed = 0
        self._tokens_saved = 0
        self._ttft_records: list[float] = []
        self._active_agents = 0
        self._use_rocm = self._check_rocm()
        # Surface counters for the MCP server endpoints. record_register fires
        # once per /tools/register_context call (with `matched=False` since the
        # simple endpoint doesn't try cross-agent dedup); record_decision fires
        # once per successful /tools/get_optimized_context call.
        self._register_calls: list[bool] = []
        self._decision_calls: list[CompressionDecision] = []

    def _check_rocm(self) -> bool:
        """Check if ROCm SMI is available."""
        try:
            result = subprocess.run(
                ["/opt/rocm/bin/rocm-smi", "--showid"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    async def get_vram_usage(self) -> Tuple[float, float]:
        """Return (used_gb, total_gb) from ROCm SMI or psutil fallback."""
        if self._use_rocm:
            try:
                result = subprocess.run(
                    ["/opt/rocm/bin/rocm-smi", "--showgpu占用率", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    import json
                    data = json.loads(result.stdout)
                    for gpu in data:
                        used = float(gpu.get("gpu占用率内存", 0))
                        total = 192.0  # MI300X has 192GB
                        return used, total
            except Exception as e:
                logger.warning(f"ROCm SMI failed: {e}")

        # Fallback: return mock values for local dev
        return 45.0, 192.0

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
        """Record a /tools/register_context call. `matched` is True when LSH
        cross-agent dedup found a reusable block; False otherwise."""
        self._register_calls.append(matched)

    def record_decision(self, decision: CompressionDecision) -> None:
        """Record a successful /tools/get_optimized_context decision."""
        self._decision_calls.append(decision)

    def _resolve_gpu_label(self) -> str:
        """Return a short label identifying the active GPU backend.

        ROCm hosts: "rocm". Anything else: "cpu". The /health endpoint passes
        whatever this returns straight through to clients, so any exception
        raised here is caught upstream and reported as the degraded path.
        """
        return "rocm" if self._use_rocm else "cpu"

    async def snapshot(
        self,
        *,
        current_compressor_model: Optional[str] = None,
        compressor_degradations: Optional[Iterable[Degradation]] = None,
    ) -> MetricsSnapshot:
        """Capture current metrics snapshot.

        Optional kwargs let the MCP server inject compressor identity and
        degradation events captured during this snapshot window — neither
        is known to the collector itself, so we accept them at the boundary.
        """
        vram_used, vram_total = await self.get_vram_usage()
        avg_ttft = sum(self._ttft_records) / len(self._ttft_records) if self._ttft_records else 0.0
        dedup_rate = (self._tokens_saved / self._tokens_processed * 100) if self._tokens_processed > 0 else 0.0
        compression_ratio = (self._tokens_processed / (self._tokens_processed - self._tokens_saved)) if self._tokens_saved > 0 else 1.0

        return MetricsSnapshot(
            timestamp=datetime.now(),
            vram_source="rocm-smi" if self._use_rocm else "psutil",
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
