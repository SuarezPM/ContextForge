"""ContextForge - Shared context compiler for multi-agent LLM systems on AMD MI300X."""
__version__ = "3.0.0"

from contextforge.registry.context_registry import ContextRegistry, SharedContextResult, RegisteredAgent
from contextforge.pipeline_config import PipelineConfig
from contextforge.token_counter import TokenCounter, count_tokens, encode_tokens, compute_kv_gb
from contextforge.metrics.vram_monitor import VRAMMonitor, get_monitor, get_vram_pressure
from contextforge.dedup.lsh_engine import LSHTokenMatcher, TokenBlockMatch
from contextforge.dedup.faiss_index import FAISSContextIndex, FAISSMatch
from contextforge.registry.vram_aware_cache import VRAMAwareCache, EvictionMode

__all__ = [
    # Core registry
    "ContextRegistry",
    "SharedContextResult",
    "RegisteredAgent",
    # Pipeline
    "PipelineConfig",
    # Token counting
    "TokenCounter",
    "count_tokens",
    "encode_tokens",
    "compute_kv_gb",
    # VRAM monitoring
    "VRAMMonitor",
    "get_monitor",
    "get_vram_pressure",
    # LSH deduplication
    "LSHTokenMatcher",
    "TokenBlockMatch",
    # FAISS ANN search
    "FAISSContextIndex",
    "FAISSMatch",
    # VRAM-aware cache
    "VRAMAwareCache",
    "EvictionMode",
]