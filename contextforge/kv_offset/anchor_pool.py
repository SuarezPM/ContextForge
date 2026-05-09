"""Anchor-based KV cache offset alignment - KVCOMM-inspired (arXiv:2510.12872).

Addresses the offset-variance problem: identical token sequences produce different
KV cache values when preceded by different agent-specific prefixes due to RoPE
position encoding.

Key insight from KVCOMM: KV offset variance across different prefix contexts is
predictable via token embedding proximity. RoPE de-rotation is mandatory before
measuring key similarity.

Usage:
    pool = AnchorPool(max_size=20)
    await pool.update_pool(token_ids, agent_id, real_kv_offset)
    shareable = await pool.predict_shareable(token_ids, target_agent_id)
    offset_hint = await pool.approximate_offset(token_ids, target_agent_id)
"""
import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Length compatibility tolerance (10%)
LENGTH_TOLERANCE = 0.10

# Maximum anchor pool size before LFU pruning
DEFAULT_MAX_SIZE = 20

# Embedding dimension for Qwen3 token embeddings
EMBEDDING_DIM = 128


@dataclass
class Anchor:
    """A stored anchor for KV offset estimation."""
    base_kv_hash: int
    agent_offsets: dict[str, np.ndarray]
    embedding: np.ndarray  # shape (EMBEDDING_DIM,)
    token_length: int
    access_count: int = 0
    created_at: float = field(default_factory=time.monotonic)

    def __lt__(self, other: "Anchor") -> bool:
        if self.access_count == other.access_count:
            return self.created_at < other.created_at
        return self.access_count < other.access_count


class AnchorPool:
    """
    Anchor-based KV offset estimator for cross-context KV cache reuse.

    Implements KVCOMM's key insight: shared token sequences produce predictable
    KV offsets when preceded by different prefixes, provided we account for
    RoPE position encoding.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        length_tolerance: float = LENGTH_TOLERANCE,
    ):
        self._max_size = max_size
        self._length_tolerance = length_tolerance
        self._anchors: dict[int, Anchor] = {}
        self._agent_anchors: dict[str, set[int]] = {}
        self._lock = asyncio.Lock()

    async def update_pool(
        self,
        token_ids: list[int],
        agent_id: str,
        real_kv_offset: np.ndarray,
    ) -> None:
        """Add a new anchor to the pool (or update existing)."""
        loop = asyncio.get_event_loop()

        block_hash = await loop.run_in_executor(
            None, self._simhash_token_ids, tuple(token_ids)
        )

        embedding = await loop.run_in_executor(
            None, self._token_ids_to_embedding, token_ids
        )

        async with self._lock:
            if block_hash in self._anchors:
                anchor = self._anchors[block_hash]
                anchor.agent_offsets[agent_id] = real_kv_offset
                anchor.access_count += 1
            else:
                anchor = Anchor(
                    base_kv_hash=block_hash,
                    agent_offsets={agent_id: real_kv_offset},
                    embedding=embedding,
                    token_length=len(token_ids),
                    access_count=1,
                )
                self._anchors[block_hash] = anchor

                if agent_id not in self._agent_anchors:
                    self._agent_anchors[agent_id] = set()
                self._agent_anchors[agent_id].add(block_hash)

            if len(self._anchors) > self._max_size:
                await self._prune_anchors()

    async def predict_shareable(
        self,
        token_ids: list[int],
        target_agent_id: str,
    ) -> bool:
        """
        Predict whether token_ids are shareable with target_agent_id.

        Uses entropy-based criterion: P_anchor = max_A { L(φ) * H_A * log(A) }
        """
        loop = asyncio.get_event_loop()
        target_length = len(token_ids)

        candidates = []
        async with self._lock:
            for block_hash, anchor in self._anchors.items():
                if target_agent_id in anchor.agent_offsets:
                    continue

                length_diff = abs(anchor.token_length - target_length) / target_length
                if length_diff <= self._length_tolerance:
                    candidates.append(anchor)

        if not candidates:
            return False

        def length_compatibility(ref_len: int) -> float:
            diff = abs(ref_len - target_length) / target_length
            return 1.0 - (diff / self._length_tolerance)

        target_embedding = await loop.run_in_executor(
            None, self._token_ids_to_embedding, token_ids
        )

        best_score = 0.0
        for anchor in candidates:
            L_phi = length_compatibility(anchor.token_length)

            distances = []
            for other_anchor in candidates:
                dist = np.linalg.norm(anchor.embedding - other_anchor.embedding)
                distances.append(dist)

            if distances:
                neg_dist = [-d for d in distances]
                exp_weights = np.exp(neg_dist - np.max(neg_dist))
                softmax_weights = exp_weights / exp_weights.sum()
                H_A = -np.sum(softmax_weights * np.log(softmax_weights + 1e-10))
            else:
                H_A = 0.0

            A = len(candidates)
            score = L_phi * H_A * np.log(A + 1)

            if score > best_score:
                best_score = score

        return best_score > 0.3

    async def approximate_offset(
        self,
        token_ids: list[int],
        target_agent_id: str,
    ) -> Optional[np.ndarray]:
        """Approximate KV offset for token_ids when used by target_agent_id."""
        loop = asyncio.get_event_loop()

        target_embedding = await loop.run_in_executor(
            None, self._token_ids_to_embedding, token_ids
        )

        async with self._lock:
            candidates = [
                (anchor, anchor.agent_offsets.get(target_agent_id))
                for anchor in self._anchors.values()
                if target_agent_id in anchor.agent_offsets
            ]

        if not candidates:
            return None

        distances = []
        offsets = []
        for anchor, offset in candidates:
            dist = np.linalg.norm(anchor.embedding - target_embedding)
            distances.append(dist)
            offsets.append(offset)

        neg_dist = [-d for d in distances]
        exp_weights = np.exp(neg_dist - np.max(neg_dist))
        softmax_weights = exp_weights / exp_weights.sum()

        result = np.zeros_like(offsets[0])
        for w, offset in zip(softmax_weights, offsets):
            result += w * offset

        return result

    async def apply_rope_derotation(
        self,
        kv_keys: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Apply RoPE de-rotation to KV keys before similarity comparison.

        Args:
            kv_keys: Key vectors of shape (seq_len, head_dim)
            positions: Position indices of shape (seq_len,)

        Returns:
            De-rotated keys of same shape
        """
        seq_len, head_dim = kv_keys.shape
        d = head_dim // 2

        base = 10000.0
        theta = np.zeros(d)
        for i in range(d):
            theta[i] = base ** (-2.0 * i / d)

        cos_vals = np.cos(positions[:, None] * theta[None, :])
        sin_vals = np.sin(positions[:, None] * theta[None, :])

        derotated = np.zeros_like(kv_keys)
        derotated[:, :d] = (
            kv_keys[:, :d] * cos_vals + kv_keys[:, d:] * sin_vals
        )
        derotated[:, d:] = (
            -kv_keys[:, :d] * sin_vals + kv_keys[:, d:] * cos_vals
        )

        return derotated

    async def _prune_anchors(self) -> None:
        """Prune least-frequently-used anchors when pool exceeds max_size."""
        if len(self._anchors) <= self._max_size:
            return

        anchor_heap = [
            (a.access_count, a.created_at, hash)
            for hash, a in self._anchors.items()
        ]
        heapq.heapify(anchor_heap)

        evict_count = max(1, int(len(self._anchors) * 0.25))
        for _ in range(evict_count):
            if not anchor_heap:
                break
            _, _, hash_to_evict = heapq.heappop(anchor_heap)
            if hash_to_evict in self._anchors:
                anchor = self._anchors[hash_to_evict]
                for aid in anchor.agent_offsets:
                    if aid in self._agent_anchors:
                        self._agent_anchors[aid].discard(hash_to_evict)
                del self._anchors[hash_to_evict]

        logger.debug(f"Pruned {evict_count} anchors, pool size: {len(self._anchors)}")

    def _simhash_token_ids(self, token_ids: tuple[int, ...]) -> int:
        """Compute 64-bit SimHash for a token sequence."""
        v = np.zeros(64, dtype=np.float32)

        for tid in token_ids:
            h = int(tid)
            for _ in range(4):
                h ^= h << 13
                h ^= h >> 7
                h ^= h << 17
                h = h & 0xFFFFFFFF

            for bit in range(64):
                if (h >> (bit % 32)) & 1:
                    v[bit] += 1
                else:
                    v[bit] -= 1

        bits = (v > 0).astype(np.uint8)
        result = 0
        for i, b in enumerate(bits):
            result |= (int(b) << i)

        return result

    def _token_ids_to_embedding(self, token_ids: list[int]) -> np.ndarray:
        """Convert token IDs to fixed-dim embedding via pseudo-random projection."""
        embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        for i, tid in enumerate(token_ids[:1024]):
            h = int(tid)
            for _ in range(4):
                h ^= h << 13
                h ^= h >> 7
                h ^= h << 17
                h = h & 0xFFFFFFFF

            for dim in range(EMBEDDING_DIM):
                if (h >> (dim % 32)) & 1:
                    embedding[dim] += 1.0

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    async def get_stats(self) -> dict:
        """Return anchor pool statistics."""
        async with self._lock:
            total_offsets = sum(len(a.agent_offsets) for a in self._anchors.values())
            return {
                "total_anchors": len(self._anchors),
                "total_agent_offsets": total_offsets,
                "agents_tracked": len(self._agent_anchors),
                "max_size": self._max_size,
            }