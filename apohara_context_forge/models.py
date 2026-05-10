"""Pydantic data models - typed contracts for ContextForge."""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal


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
