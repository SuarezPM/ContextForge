"""Prometheus metrics observability stack - Section 5 implementation.

Exposes cache metrics, VRAM telemetry, compression stats, dedup performance,
and pipeline TTFT via Prometheus client.

Metrics categories:
- Cache: hits, misses, registry size, evictions
- VRAM: pressure ratio, eviction mode, tokens evicted
- Compression: ratio histogram, latency histogram
- Dedup: LSH match confidence, dedup latency
- Pipeline: per-agent TTFT, token savings
"""
import logging
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram, Summary

logger = logging.getLogger(__name__)

# ============================================================
# CACHE METRICS
# ============================================================

cache_hits = Counter(
    "contextforge_cache_hits_total",
    "Number of KV cache block reuse hits found",
    ["agent_id", "segment_type"]
)

cache_misses = Counter(
    "contextforge_cache_misses_total",
    "Cache misses requiring full prefill",
    ["agent_id"]
)

cache_registry_size = Gauge(
    "contextforge_registry_entries",
    "Active entries in context registry",
    ["cache_type"]  # "ttl" or "vram_aware"
)

cache_evictions_total = Counter(
    "contextforge_evictions_total",
    "Total entries evicted from cache",
    ["reason"]  # "ttl_expired", "pressure", "critical", "emergency"
)

tokens_evicted = Counter(
    "contextforge_tokens_evicted_total",
    "Total tokens removed from registry by eviction",
    ["eviction_mode"]  # "normal", "pressure", "critical", "emergency"
)

# ============================================================
# VRAM METRICS
# ============================================================

vram_pressure_ratio = Gauge(
    "contextforge_vram_pressure_ratio",
    "Current VRAM utilization (0.0-1.0) from PyRSMI"
)

vram_used_gb = Gauge(
    "contextforge_vram_used_gb",
    "Current VRAM used in gigabytes"
)

vram_available_gb = Gauge(
    "contextforge_vram_available_gb",
    "Current VRAM available in gigabytes"
)

eviction_mode = Gauge(
    "contextforge_eviction_mode_code",
    "Current eviction mode as numeric code (0=relaxed, 1=normal, 2=pressure, 3=critical, 4=emergency)"
)

# ============================================================
# COMPRESSION METRICS
# ============================================================

compression_ratio_histogram = Histogram(
    "contextforge_compression_ratio",
    "Achieved compression ratios per segment type",
    ["segment_type"],
    buckets=[1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 14.0, 20.0]
)

compression_latency_ms = Histogram(
    "contextforge_compression_latency_ms",
    "LLMLingua-2 compression latency in milliseconds",
    buckets=[5, 10, 25, 50, 100, 250, 500, 1000, 2000]
)

compression_requests_total = Counter(
    "contextforge_compression_requests_total",
    "Total compression requests",
    ["segment_type", "decision"]  # decision: "compressed", "skipped_short", "skipped_protected"
)

# ============================================================
# DEDUP METRICS
# ============================================================

lsh_match_confidence = Histogram(
    "contextforge_lsh_match_confidence",
    "LSH block match confidence scores (0.0-1.0)",
    buckets=[0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.99, 1.0]
)

lsh_blocks_indexed = Counter(
    "contextforge_lsh_blocks_indexed_total",
    "Total LSH blocks indexed",
    ["agent_id"]
)

lsh_blocks_reused = Counter(
    "contextforge_lsh_blocks_reused_total",
    "Total LSH blocks reused across agents",
    ["agent_id", "source_agent"]
)

dedup_latency_ms = Histogram(
    "contextforge_dedup_latency_ms",
    "Total deduplication pipeline latency in milliseconds (critical path)",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0]
)

faiss_search_latency_ms = Histogram(
    "contextforge_faiss_search_latency_ms",
    "FAISS ANN search latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
)

# ============================================================
# PIPELINE METRICS
# ============================================================

agent_ttft_ms = Histogram(
    "contextforge_agent_ttft_ms",
    "Time-to-first-token per agent in milliseconds",
    ["agent_id", "thinking_mode"],  # thinking_mode: "cot" or "non_thinking"
    buckets=[20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
)

agent_tokens_before = Histogram(
    "contextforge_agent_tokens_before",
    "Token count before optimization per agent",
    ["agent_id"],
    buckets=[100, 250, 500, 1000, 2000, 4000, 8000, 16000]
)

agent_tokens_after = Histogram(
    "contextforge_agent_tokens_after",
    "Token count after optimization per agent",
    ["agent_id"],
    buckets=[100, 250, 500, 1000, 2000, 4000, 8000, 16000]
)

token_savings_pct = Histogram(
    "contextforge_token_savings_pct",
    "Percentage of tokens saved per pipeline run",
    buckets=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
)

pipeline_duration_ms = Histogram(
    "contextforge_pipeline_duration_ms",
    "Total pipeline duration in milliseconds",
    ["agent_count"],
    buckets=[100, 250, 500, 1000, 2000, 5000, 10000, 30000]
)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def record_cache_hit(agent_id: str, segment_type: str) -> None:
    """Record a cache hit."""
    cache_hits.labels(agent_id=agent_id, segment_type=segment_type).inc()


def record_cache_miss(agent_id: str) -> None:
    """Record a cache miss."""
    cache_misses.labels(agent_id=agent_id).inc()


def record_vram_metrics(pressure: float, used_gb: float, available_gb: float, mode: str) -> None:
    """Update all VRAM gauges."""
    vram_pressure_ratio.set(pressure)
    vram_used_gb.set(used_gb)
    vram_available_gb.set(available_gb)
    mode_code = {"relaxed": 0, "normal": 1, "pressure": 2, "critical": 3, "emergency": 4}.get(mode, 0)
    eviction_mode.set(mode_code)


def record_compression(segment_type: str, ratio: float, latency_ms: float, decision: str) -> None:
    """Record compression metrics."""
    compression_ratio_histogram.labels(segment_type=segment_type).observe(ratio)
    compression_latency_ms.observe(latency_ms)
    compression_requests_total.labels(segment_type=segment_type, decision=decision).inc()


def record_lsh_match(confidence: float) -> None:
    """Record LSH match confidence."""
    lsh_match_confidence.observe(confidence)


def record_agent_ttft(agent_id: str, thinking_mode: str, ttft_ms: float) -> None:
    """Record agent TTFT."""
    agent_ttft_ms.labels(agent_id=agent_id, thinking_mode=thinking_mode).observe(ttft_ms)


def record_token_savings(before: int, after: int) -> None:
    """Record token savings for pipeline."""
    if before > 0:
        savings_pct = ((before - after) / before) * 100
        token_savings_pct.observe(savings_pct)
    agent_tokens_before.labels(agent_id="pipeline").observe(before)
    agent_tokens_after.labels(agent_id="pipeline").observe(after)
