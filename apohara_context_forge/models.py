"""Pydantic data models - typed contracts for ContextForge."""
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ContextEntry(BaseModel):
    """A registered agent context with compression support."""
    agent_id: str
    context: str
    compressed_context: str | None = None
    embedding: list[float] | None = None
    token_count: int
    compressed_token_count: int | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    ttl_seconds: int = 300

    def model_post_init(self, __context) -> None:
        if self.embedding is None:
            self.embedding = []


class ContextMatch(BaseModel):
    """A semantic match between contexts.

    `shared_prefix_tokens` is the token length of the longest shared prefix
    between the incoming context and this match. The CompressionCoordinator
    uses it to choose apc_reuse vs compress.
    """
    agent_id: str
    similarity: float
    shared_prefix: str
    shared_prefix_tokens: int


class CompressionDecision(BaseModel):
    """Decision made by the compression coordinator.

    `compressed_context` and `final_context` carry the same payload; the latter
    is the canonical name used by the MCP API and tests. We keep both so older
    callers in the pipeline continue to work without churn.
    """
    strategy: Literal["apc_reuse", "compress", "compress_and_reuse", "passthrough"]
    shared_prefix: str | None = None
    compressed_context: str | None = None
    final_context: str = ""
    original_tokens: int = 0
    final_tokens: int = 0
    tokens_saved: int = 0
    rationale: str = ""
    savings_pct: float = 0.0


class Degradation(BaseModel):
    """A degradation event (component falling back to a lower-fidelity path).

    Used by the metrics snapshot and the /health endpoint so the dashboard
    can show *why* a component is operating below its primary configuration —
    e.g. compressor falling back to CPU because the GPU model failed to load,
    or coordinator falling back to passthrough on OOM.
    """
    component: str                  # e.g. "compressor", "coordinator", "embedding_engine"
    reason: str                     # short human-readable cause
    fallback: Optional[str] = None  # what was used instead, e.g. "cpu", "passthrough"
    severity: float = 0.5           # 0.0 = informational, 1.0 = critical
    timestamp: datetime = Field(default_factory=datetime.now)


class MetricsSnapshot(BaseModel):
    """Real-time system metrics."""
    timestamp: datetime = Field(default_factory=datetime.now)
    vram_source: str = "unknown"
    compressor_model: str = "xlm-roberta-large"
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    ttft_ms: float = 0.0
    tokens_processed: int = 0
    tokens_saved: int = 0
    dedup_rate: float = 0.0
    compression_ratio: float = 0.0
    active_agents: int = 0
    degradations: list[Degradation] = Field(default_factory=list)


class ContextRegistration(BaseModel):
    """Request to register a new context. Strict — extra fields are rejected."""
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1)
    context: str


class OptimizedContextRequest(BaseModel):
    """Request for optimized context. Strict — extra fields are rejected."""
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1)
    context: str
