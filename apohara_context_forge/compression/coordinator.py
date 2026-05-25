"""Compression coordinator - the strategy decision engine for ContextForge."""
import logging
from typing import Any, Optional

from apohara_context_forge.config import settings
from apohara_context_forge.models import CompressionDecision

logger = logging.getLogger(__name__)


class CompressionCoordinator:
    """Decides one of four strategies for an agent's raw incoming context:

        apc_reuse          long shared prefix + short context -> reuse prefix, no compression
        compress_and_reuse long shared prefix + long context  -> reuse prefix, compress unique tail
        compress           no usable prefix   + long context  -> compress the full context
        passthrough        otherwise                          -> do nothing

    Boundaries are strict ``>``: a context is "long" when its token count
    exceeds COMPRESS_MIN_CONTEXT_TOKENS; a prefix is "long" when it exceeds
    APC_REUSE_MIN_SHARED_PREFIX_TOKENS.

    Dependencies are injected. ``dedup`` defaults to ``registry.dedup`` so we
    never spin up a second tokenizer/embedder.
    """

    def __init__(
        self,
        registry: Optional[Any] = None,
        compressor: Optional[Any] = None,
        dedup: Optional[Any] = None,
    ):
        self.registry = registry
        self.compressor = compressor
        if dedup is not None:
            self.dedup = dedup
        elif registry is not None:
            self.dedup = registry.dedup
        else:
            self.dedup = None

    async def decide(self, agent_id: str, context: str) -> CompressionDecision:
        """Make a compression-strategy decision for an agent's raw context."""
        ctx_tokens = self.dedup.count_prefix_tokens(context)

        matches = await self.registry.find_similar(context)
        best = matches[0] if matches else None

        long_enough = ctx_tokens > settings.COMPRESS_MIN_CONTEXT_TOKENS
        has_long_prefix = (
            best is not None
            and best.shared_prefix_tokens > settings.APC_REUSE_MIN_SHARED_PREFIX_TOKENS
        )

        # apc_reuse: strong shared prefix on a short context -> reuse, no compression.
        if has_long_prefix and not long_enough:
            return CompressionDecision(
                strategy="apc_reuse",
                shared_prefix=best.shared_prefix,
                final_context=context,
                original_tokens=ctx_tokens,
                final_tokens=ctx_tokens,
                tokens_saved=0,
                savings_pct=0.0,
            )

        # compress_and_reuse: strong prefix + long context -> reuse prefix, compress the tail only.
        if has_long_prefix and long_enough:
            tail = context[len(best.shared_prefix):]
            compressed_tail, _ratio = await self.compressor.compress(
                tail, settings.CONTEXTFORGE_COMPRESSION_RATE
            )
            final_context = best.shared_prefix + compressed_tail
            final_tokens = best.shared_prefix_tokens + self.dedup.count_prefix_tokens(
                compressed_tail
            )
            tokens_saved = ctx_tokens - final_tokens
            return CompressionDecision(
                strategy="compress_and_reuse",
                shared_prefix=best.shared_prefix,
                compressed_context=compressed_tail,
                final_context=final_context,
                original_tokens=ctx_tokens,
                final_tokens=final_tokens,
                tokens_saved=tokens_saved,
                savings_pct=(tokens_saved / ctx_tokens * 100) if ctx_tokens > 0 else 0.0,
            )

        # compress: no usable prefix but a long context -> compress everything.
        if long_enough:
            compressed, _ratio = await self.compressor.compress(
                context, settings.CONTEXTFORGE_COMPRESSION_RATE
            )
            final_tokens = self.dedup.count_prefix_tokens(compressed)
            tokens_saved = ctx_tokens - final_tokens
            return CompressionDecision(
                strategy="compress",
                shared_prefix="",
                compressed_context=compressed,
                final_context=compressed,
                original_tokens=ctx_tokens,
                final_tokens=final_tokens,
                tokens_saved=tokens_saved,
                savings_pct=(tokens_saved / ctx_tokens * 100) if ctx_tokens > 0 else 0.0,
            )

        # passthrough: surface a weak prefix for observability, but act on nothing.
        return CompressionDecision(
            strategy="passthrough",
            shared_prefix=best.shared_prefix if best is not None else "",
            final_context=context,
            original_tokens=ctx_tokens,
            final_tokens=ctx_tokens,
            tokens_saved=0,
            savings_pct=0.0,
        )
