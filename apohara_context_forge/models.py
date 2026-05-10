"""Pydantic data models - typed contracts for ContextForge."""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal, Optional


class ContextEntry(BaseModel):
    """A registered agent context with compression support."""
    agent_id: str
    context: str
    compressed_context: str | None = None
    embedding: list[float] | None = None
    token_count: int
    compressed_token_count: int | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    ttl_seconds: int = 300

    def model_post_init(self, __context) -> None:
        if self.embedding is None:
            self.embedding = []


class ContextMatch(BaseModel):
    """A semantic match between contexts."""
    agent_id: str
    similarity: float
    shared_prefix: str
    tokens_saved: int


class CompressionDecision(BaseModel):
    """Decision made by the compression coordinator."""
    strategy: Literal["apc_reuse", "compress", "compress_and_reuse", "passthrough"]
    shared_prefix: str | None = None
    compressed_context: str | None = None
    original_tokens: int
    final_tokens: int
    savings_pct: float


class MetricsSnapshot(BaseModel):
    """Real-time system metrics."""
    timestamp: datetime = Field(default_factory=datetime.now)
    vram_used_gb: float
    vram_total_gb: float
    ttft_ms: float
    tokens_processed: int
    tokens_saved: int
    dedup_rate: float
    compression_ratio: float
    active_agents: int


class ContextRegistration(BaseModel):
    """Request to register a new context."""
    agent_id: str
    context: str


class OptimizedContextRequest(BaseModel):
    """Request for optimized context."""
    agent_id: str
    context: str


class Degradation(BaseModel):
    """A degradation event (component falling back to a lower-fidelity path).

    Used by the metrics snapshot and the /health endpoint so the dashboard
    can show *why* a component is operating below its primary configuration —
    e.g. compressor falling back to CPU because the GPU model failed to load,
    or coordinator falling back to passthrough on OOM.
    """
    component: str                  # e.g. "compressor", "coordinator", "embedding_engine"
    reason: str                     # short human-readable cause, e.g. "OOM", "model unavailable"
    fallback: Optional[str] = None  # what was used instead, e.g. "cpu", "passthrough"
    severity: float = 0.5           # 0.0 = informational, 1.0 = critical
    timestamp: datetime = Field(default_factory=datetime.now)
