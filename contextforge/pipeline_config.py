"""Pipeline configuration dataclass for ContextForge v3.0."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineConfig:
    """
    Configuration for ContextForge pipeline.

    All values have sane defaults; only model_id is required.

    Usage:
        config = PipelineConfig(
            model_id="Qwen/Qwen3-235B-A22B",
            vram_budget_tokens=50_000_000,
        )
        pipeline = Pipeline(config=config)
    """
    # Model configuration
    model_id: str = "Qwen/Qwen3-235B-A22B"

    # LSHTokenMatcher configuration
    block_size: int = 16  # vLLM PagedAttention block size
    hamming_threshold: int = 8  # <8 bits different = high confidence

    # VRAMAwareCache configuration
    vram_budget_tokens: int = 50_000_000  # ~3GB for 64-layer model

    # FAISS configuration
    faiss_dim: int = 384  # all-MiniLM-L6-v2 embedding dimension
    faiss_nlist: int = 100  # IVF cluster count (sqrt of expected entries)

    # Compression configuration
    compression_min_tokens: int = 512
    compression_emergency_threshold: float = 0.85  # VRAM pressure threshold

    # VRAM monitoring
    vram_check_interval: float = 2.0  # seconds between VRAM pressure checks

    # Anchor pool (KV offset alignment)
    anchor_pool_max_size: int = 20  # max anchors before LFU pruning

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.hamming_threshold < 1:
            raise ValueError(f"hamming_threshold must be >= 1, got {self.hamming_threshold}")
        if self.vram_budget_tokens < 1000:
            raise ValueError(f"vram_budget_tokens must be >= 1000, got {self.vram_budget_tokens}")
        if self.faiss_dim < 1:
            raise ValueError(f"faiss_dim must be >= 1, got {self.faiss_dim}")