"""Core context registry with semantic search."""
import asyncio
import hashlib
import logging
from datetime import datetime
from typing import Any

from contextforge.models import ContextEntry, ContextMatch, CompressionDecision
from contextforge.registry.ttl_cache import TTLCache
from contextforge.config import settings

logger = logging.getLogger(__name__)


class ContextRegistry:
    """Stores/retrieves agent contexts with TTL eviction and semantic search."""

    def __init__(self, default_ttl: int | None = None):
        self._cache = TTLCache(default_ttl or settings.contextforge_ttl_seconds)
        self._embeddings: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def register(self, agent_id: str, context: str) -> ContextEntry:
        """Register a new context entry."""
        token_count = self._estimate_tokens(context)
        entry = ContextEntry(
            agent_id=agent_id,
            context=context,
            token_count=token_count,
            ttl_seconds=settings.contextforge_ttl_seconds,
        )
        cache_key = f"context:{agent_id}"
        await self._cache.set(cache_key, entry)
        logger.debug(f"Registered context for agent {agent_id}, tokens={token_count}")
        return entry

    async def get(self, agent_id: str) -> ContextEntry | None:
        """Retrieve context for an agent."""
        cache_key = f"context:{agent_id}"
        return await self._cache.get(cache_key)

    async def find_similar(
        self, context: str, threshold: float | None = None
    ) -> list[ContextMatch]:
        """Find contexts with similarity above threshold."""
        from contextforge.dedup.dedup_engine import SemanticDedupEngine

        threshold = threshold or settings.contextforge_dedup_threshold
        dedup = SemanticDedupEngine()
        input_embedding = await dedup.embed(context)

        matches = []
        async with self._lock:
            keys = await self._cache.keys()

        for key in keys:
            if not key.startswith("context:"):
                continue
            entry: ContextEntry | None = await self._cache.get(key)
            if entry is None or entry.agent_id == "":
                continue
            if entry.embedding:
                similarity = await dedup.similarity(input_embedding, entry.embedding)
                if similarity >= threshold:
                    shared = await dedup.find_shared_prefix(context, entry.context)
                    tokens_saved = entry.token_count - len(shared.split())
                    matches.append(ContextMatch(
                        agent_id=entry.agent_id,
                        similarity=similarity,
                        shared_prefix=shared[:200] if len(shared) > 200 else shared,
                        tokens_saved=max(0, tokens_saved),
                    ))

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

    async def get_all_active(self) -> list[ContextEntry]:
        """Get all non-expired context entries."""
        entries = []
        async with self._lock:
            keys = await self._cache.keys()
        for key in keys:
            if key.startswith("context:"):
                entry = await self._cache.get(key)
                if entry is not None:
                    entries.append(entry)
        return entries

    async def evict_expired(self) -> int:
        """Evict all expired contexts, returns count."""
        return await self._cache.evict_expired()

    async def clear(self) -> None:
        """Clear all contexts."""
        await self._cache.clear()
        async with self._lock:
            self._embeddings.clear()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using simple heuristic."""
        return len(text.split()) // 4 * 3  # ~0.75 tokens per word
