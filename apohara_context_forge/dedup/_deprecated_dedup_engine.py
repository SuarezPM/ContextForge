"""Semantic deduplication using SBERT embeddings.

.. deprecated:: v3.0
    Use :class:`contextforge.dedup.lsh_engine.LSHTokenMatcher` + 
    :class:`contextforge.dedup.faiss_index.FAISSContextIndex` instead.
    This module has O(n) Python loop scan and word-level prefix detection
    which is incompatible with vLLM PagedAttention block alignment.
"""
import asyncio
import warnings
warnings.warn(
    "This module is deprecated as of v3.0. Use LSHTokenMatcher + FAISSContextIndex.",
    DeprecationWarning,
    stacklevel=2
)
import logging

from apohara_context_forge.dedup.embedder import Embedder

logger = logging.getLogger(__name__)


class SemanticDedupEngine:
    """Semantic similarity + cosine deduplication using SBERT."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._embedder = Embedder(model_name)
        self._lock = asyncio.Lock()

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        return await self._embedder.encode(text)

    async def similarity(self, emb1: list[float], emb2: list[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        dot = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    async def find_shared_prefix(self, context_a: str, context_b: str) -> str:
        """Find overlapping text between two contexts."""
        words_a = context_a.split()
        words_b = context_b.split()
        shared = []
        min_len = min(len(words_a), len(words_b))
        for i in range(min_len):
            if words_a[i] == words_b[i]:
                shared.append(words_a[i])
            else:
                break
        return " ".join(shared)

    async def batch_deduplicate(
        self, contexts: list[str]
    ) -> dict[str, list[dict]]:
        """Deduplicate a batch of contexts."""
        if not contexts:
            return {}

        embeddings = await self._embedder.encode_batch(contexts)
        results: dict[str, list[dict]] = {}

        for i, (ctx, emb) in enumerate(zip(contexts, embeddings)):
            matches = []
            for j, (other_ctx, other_emb) in enumerate(zip(contexts, embeddings)):
                if i == j:
                    continue
                sim = await self.similarity(emb, other_emb)
                if sim >= 0.85:
                    shared = await self.find_shared_prefix(ctx, other_ctx)
                    matches.append({
                        "index": j,
                        "similarity": sim,
                        "shared_prefix": shared,
                    })
            results[f"context_{i}"] = matches

        return results
