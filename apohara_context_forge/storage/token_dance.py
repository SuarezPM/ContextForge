"""TokenDance — Master-Mirror Storage for collective KV cache sharing.

Based on TokenDance (arXiv:2604.03143, Apr 2026): "Collective KV Cache
Sharing for Multi-Agent Inference."

Idea: instead of storing N independent KV caches for N agents, store one
"master" KV cache and (N-1) sparse diffs ("mirrors"). When agents share a
common prefix and diverge only on a small subset of blocks, the diff is
mostly zero — block-sparse storage compresses it 11–17x.

Storage layout:
    master_cache[m_id]                     full KV blocks for master agent
    mirrors[a_id] = SparseKVDiff(          sparse delta vs master:
        block_indices: indices of blocks that differ
        diff_values:   the per-block deltas at those indices
    )

Reconstruction:
    full_kv[a_id] = master_cache[m_id].copy()
    full_kv[a_id][block_indices] += diff_values

Diff threshold (default 1e-4) controls sparsity: blocks with L2 norm of
delta below threshold are dropped (reconstruction within tolerance).

Collective reuse step (All-Gather pattern): given a new round's shared
context, push the update once to the master and re-derive all mirror
diffs. Cost is O(blocks) regardless of agent count.

Pure numpy. No GPU dependency. Graceful degradation principle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SparseKVDiff:
    """Sparse delta of an agent's KV blocks vs the master agent's blocks.

    Only blocks whose L2 norm of the delta exceeds the diff threshold are
    stored. Reconstruction adds these deltas back to the corresponding
    master blocks; all other blocks are byte-identical to the master.
    """

    block_indices: np.ndarray  # shape (n_diff_blocks,) int
    diff_values: np.ndarray    # shape (n_diff_blocks, *block_shape) float
    total_blocks: int          # original number of blocks (for reconstruction)
    threshold: float = 1e-4

    @property
    def n_diff_blocks(self) -> int:
        return int(self.block_indices.shape[0])

    @property
    def sparsity(self) -> float:
        if self.total_blocks == 0:
            return 0.0
        return 1.0 - self.n_diff_blocks / self.total_blocks


class TokenDanceStorage:
    """Master-Mirror diff storage for multi-agent KV cache.

    Stores 1 full Master KV cache + (N-1) block-sparse diffs.
    Achieves 11-17x compression vs storing N full KV caches when agents
    share large prefixes (typical in 5-agent RAG/Critic pipelines).

    Based on: TokenDance (arXiv:2604.03143, Apr 2026).
    """

    def __init__(self, diff_threshold: float = 1e-4):
        self.diff_threshold: float = diff_threshold
        self.master_id: str | None = None
        self.master_cache: dict[str, np.ndarray] = {}
        self.mirrors: dict[str, SparseKVDiff] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def register_master(self, agent_id: str, kv_blocks: np.ndarray) -> None:
        """Register the master agent. The first call sets the reference KV.

        Calling this again with a different agent_id replaces the master
        and clears mirror state — all mirrors must be re-registered.
        """
        if kv_blocks.ndim < 2:
            raise ValueError(
                f"kv_blocks must be at least 2D (n_blocks, ...); got shape {kv_blocks.shape}"
            )
        if self.master_id is not None and self.master_id != agent_id:
            self.mirrors.clear()
            self.master_cache.clear()
        self.master_id = agent_id
        self.master_cache[agent_id] = kv_blocks.copy()

    def register_mirror(self, agent_id: str, kv_blocks: np.ndarray) -> SparseKVDiff:
        """Compute and store a sparse diff vs the master.

        Only blocks whose per-block L2 norm of the delta exceeds
        self.diff_threshold are kept; the rest are treated as identical.
        """
        if self.master_id is None:
            raise RuntimeError("register_master() must be called before register_mirror()")
        master = self.master_cache[self.master_id]
        if kv_blocks.shape != master.shape:
            raise ValueError(
                f"kv_blocks shape {kv_blocks.shape} must match master shape {master.shape}"
            )

        delta = kv_blocks - master
        # Per-block L2 norm collapses all non-block dims into a single scalar.
        flat = delta.reshape(delta.shape[0], -1)
        per_block_norm = np.linalg.norm(flat, axis=1)
        diff_mask = per_block_norm > self.diff_threshold
        diff_indices = np.flatnonzero(diff_mask)

        diff = SparseKVDiff(
            block_indices=diff_indices.astype(np.int64),
            diff_values=delta[diff_indices].copy() if diff_indices.size else np.empty(
                (0,) + master.shape[1:], dtype=delta.dtype
            ),
            total_blocks=master.shape[0],
            threshold=self.diff_threshold,
        )
        self.mirrors[agent_id] = diff
        return diff

    def reconstruct(self, agent_id: str) -> np.ndarray:
        """Reconstruct the full KV cache for an agent."""
        if self.master_id is None:
            raise RuntimeError("No master registered")
        if agent_id == self.master_id:
            return self.master_cache[self.master_id].copy()
        if agent_id not in self.mirrors:
            raise KeyError(f"Unknown agent_id: {agent_id}")

        diff = self.mirrors[agent_id]
        out = self.master_cache[self.master_id].copy()
        if diff.n_diff_blocks > 0:
            out[diff.block_indices] = out[diff.block_indices] + diff.diff_values
        return out

    def compression_ratio(self) -> float:
        """Returns (sum of full per-agent block counts) / (master + diffs)."""
        if self.master_id is None or not self.master_cache:
            return 1.0
        master_blocks = self.master_cache[self.master_id].shape[0]
        n_agents = 1 + len(self.mirrors)
        full_blocks = n_agents * master_blocks
        stored_blocks = master_blocks + sum(d.n_diff_blocks for d in self.mirrors.values())
        if stored_blocks == 0:
            return float(n_agents)
        return full_blocks / stored_blocks

    def collective_reuse_step(
        self,
        agent_ids: list[str],
        shared_blocks: np.ndarray,
    ) -> dict[str, int]:
        """All-Gather pattern: apply a shared-context update across agents.

        Given a batch of new shared blocks (e.g. a freshly retrieved
        context), append them to the master once and re-derive each
        mirror's sparsity against the extended master.

        The cost is O(master_blocks + total_diff_blocks) — paid once
        regardless of agent count. The return value is per-agent diff
        counts after the update for telemetry.
        """
        if self.master_id is None:
            raise RuntimeError("No master registered")
        if shared_blocks.ndim < 2:
            raise ValueError("shared_blocks must be at least 2D")

        master = self.master_cache[self.master_id]
        extended_master = np.concatenate([master, shared_blocks], axis=0)
        self.master_cache[self.master_id] = extended_master

        # Mirrors need to be extended to match the new master length.
        # We assume agents adopt the shared blocks exactly (i.e. shared
        # blocks are zero-diff for the mirrors). New mirror blocks are
        # therefore identical to the appended master tail.
        diff_counts: dict[str, int] = {self.master_id: 0}
        for aid in agent_ids:
            if aid == self.master_id:
                continue
            existing = self.mirrors.get(aid)
            if existing is None:
                # New mirror: identical to extended master so far.
                self.mirrors[aid] = SparseKVDiff(
                    block_indices=np.empty((0,), dtype=np.int64),
                    diff_values=np.empty(
                        (0,) + extended_master.shape[1:], dtype=extended_master.dtype
                    ),
                    total_blocks=extended_master.shape[0],
                    threshold=self.diff_threshold,
                )
            else:
                # Pre-existing diffs unchanged; total_blocks bumps to new length.
                self.mirrors[aid] = SparseKVDiff(
                    block_indices=existing.block_indices,
                    diff_values=existing.diff_values,
                    total_blocks=extended_master.shape[0],
                    threshold=existing.threshold,
                )
            diff_counts[aid] = self.mirrors[aid].n_diff_blocks
        return diff_counts

    # ------------------------------------------------------------------ #
    # Introspection                                                       #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, float | int]:
        master_blocks = (
            self.master_cache[self.master_id].shape[0]
            if self.master_id is not None
            else 0
        )
        diff_blocks_total = sum(d.n_diff_blocks for d in self.mirrors.values())
        return {
            "master_id": self.master_id or "",
            "master_blocks": master_blocks,
            "n_mirrors": len(self.mirrors),
            "diff_blocks_total": diff_blocks_total,
            "compression_ratio": self.compression_ratio(),
            "diff_threshold": self.diff_threshold,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        s = self.stats()
        return (
            f"TokenDanceStorage(master={s['master_id']!r}, "
            f"master_blocks={s['master_blocks']}, mirrors={s['n_mirrors']}, "
            f"diff_blocks={s['diff_blocks_total']}, "
            f"compression={s['compression_ratio']:.2f}x, "
            f"threshold={s['diff_threshold']:.0e})"
        )
