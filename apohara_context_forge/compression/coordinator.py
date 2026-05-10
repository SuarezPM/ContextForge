"""Compression coordinator - decision engine for ContextForge."""
import asyncio
import logging
from typing import Literal

from apohara_context_forge.config import settings
from apohara_context_forge.dedup.dedup_engine import SemanticDedupEngine
from apohara_context_forge.models import CompressionDecision

logger = logging.getLogger(__name__)


class CompressionCoordinator:
    """
    Decision engine - the brain of ContextForge.
    
    Logic:
      IF similarity >= 0.85 AND shared_prefix > 200 tokens → "apc_reuse"
      IF similarity < 0.85 AND context > 500 tokens → "compress"
      IF similarity >= 0.85 AND context > 500 tokens → "compress_and_reuse"
      ELSE → "passthrough"
    """

    def __init__(self):
        self._dedup = SemanticDedupEngine()
        self._min_tokens = settings.contextforge_min_tokens_to_compress

    async def decide(self, agent_id: str, context: str) -> CompressionDecision:
        """Make compression decision for an agent's context."""
        from apohara_context_forge.registry.context_registry import ContextRegistry
        
        registry = ContextRegistry()
        original_tokens = len(context.split())
        
        # Find similar contexts
        matches = await registry.find_similar(context)
        
        if not matches:
            return CompressionDecision(
                strategy="passthrough",
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                savings_pct=0.0,
            )

        best_match = matches[0]
        similarity = best_match.similarity
        shared_prefix = best_match.shared_prefix
        shared_tokens = len(shared_prefix.split()) if shared_prefix else 0

        # Decision logic
        if similarity >= 0.85 and shared_tokens > 200:
            # APC reuse - share the prefix directly
            return CompressionDecision(
                strategy="apc_reuse",
                shared_prefix=shared_prefix,
                original_tokens=original_tokens,
                final_tokens=shared_tokens,
                savings_pct=((original_tokens - shared_tokens) / original_tokens * 100) if original_tokens > 0 else 0.0,
            )
        elif similarity < 0.85 and original_tokens > 500:
            # Compress only
            from apohara_context_forge.compression.compressor import ContextCompressor
            compressor = ContextCompressor()
            compressed, ratio = await compressor.compress(context, settings.contextforge_compression_rate)
            final_tokens = len(compressed.split())
            return CompressionDecision(
                strategy="compress",
                compressed_context=compressed,
                original_tokens=original_tokens,
                final_tokens=final_tokens,
                savings_pct=((original_tokens - final_tokens) / original_tokens * 100) if original_tokens > 0 else 0.0,
            )
        elif similarity >= 0.85 and original_tokens > 500:
            # Both reuse and compress
            from apohara_context_forge.compression.compressor import ContextCompressor
            compressor = ContextCompressor()
            compressed, ratio = await compressor.compress(context, settings.contextforge_compression_rate)
            final_tokens = len(compressed.split())
            return CompressionDecision(
                strategy="compress_and_reuse",
                shared_prefix=shared_prefix,
                compressed_context=compressed,
                original_tokens=original_tokens,
                final_tokens=final_tokens,
                savings_pct=((original_tokens - final_tokens) / original_tokens * 100) if original_tokens > 0 else 0.0,
            )
        else:
            return CompressionDecision(
                strategy="passthrough",
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                savings_pct=0.0,
            )
