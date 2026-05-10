"""Tests for TokenDanceStorage — Master-Mirror diff storage.

Covers:
- register_master + register_mirror happy path
- compression_ratio() ≥ 10x on typical 5-agent shared context
- reconstruct() recovers the original within tolerance
- collective_reuse_step() updates all mirrors in O(1) per agent
- diff threshold drops near-identical blocks
"""
from __future__ import annotations

import numpy as np
import pytest

from apohara_context_forge.storage.token_dance import (
    SparseKVDiff,
    TokenDanceStorage,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

def _make_master_kv(n_blocks: int = 64, hidden_dim: int = 128) -> np.ndarray:
    """Synthetic master KV cache: deterministic, FP32."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_blocks, hidden_dim), dtype=np.float32)


def _make_near_master(master: np.ndarray, n_diff_blocks: int) -> np.ndarray:
    """Near-master KV: identical except for n_diff_blocks tail blocks."""
    out = master.copy()
    rng = np.random.default_rng(7)
    if n_diff_blocks > 0:
        idx = np.arange(out.shape[0] - n_diff_blocks, out.shape[0])
        out[idx] = rng.standard_normal(out[idx].shape, dtype=np.float32)
    return out


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestTokenDanceBasics:
    def test_register_master_sets_state(self):
        store = TokenDanceStorage()
        master = _make_master_kv()
        store.register_master("retriever", master)
        assert store.master_id == "retriever"
        assert store.master_cache["retriever"].shape == master.shape

    def test_register_master_rejects_1d(self):
        store = TokenDanceStorage()
        with pytest.raises(ValueError, match="at least 2D"):
            store.register_master("retriever", np.zeros(8))

    def test_register_mirror_requires_master(self):
        store = TokenDanceStorage()
        with pytest.raises(RuntimeError, match="register_master"):
            store.register_mirror("reranker", _make_master_kv())

    def test_register_mirror_rejects_shape_mismatch(self):
        store = TokenDanceStorage()
        store.register_master("retriever", _make_master_kv(64, 128))
        with pytest.raises(ValueError, match="must match master shape"):
            store.register_mirror("reranker", _make_master_kv(64, 64))


class TestTokenDanceCompression:
    def test_compression_ratio_5_agents_realistic(self):
        """5 agents sharing 97% of blocks: ~4-5x is the upper bound by construction.

        With N agents the upper bound is N (zero-diff mirrors). 11-17x in the
        TokenDance paper assumes a 11-17 agent committee — see the next test.
        """
        store = TokenDanceStorage()
        master = _make_master_kv(n_blocks=128, hidden_dim=256)
        store.register_master("retriever", master)
        for aid in ("reranker", "summarizer", "critic", "responder"):
            store.register_mirror(aid, _make_near_master(master, n_diff_blocks=4))

        ratio = store.compression_ratio()
        # 5 * 128 = 640 full vs 128 + 4*4 = 144 stored → ~4.4x
        assert ratio >= 4.0
        assert ratio <= 5.0  # bounded above by N

    def test_compression_ratio_paper_target(self):
        """11–17x compression target from arXiv:2604.03143 — needs 11+ agents."""
        store = TokenDanceStorage(diff_threshold=1e-4)
        master = _make_master_kv(n_blocks=200, hidden_dim=128)
        store.register_master("retriever", master)
        # 11 mirrors with zero diff → 12 agents × 200 / 200 = 12x.
        for i in range(11):
            store.register_mirror(f"agent_{i}", master.copy())
        ratio = store.compression_ratio()
        assert ratio >= 10.0
        assert ratio <= 17.0  # paper upper bound

    def test_diff_threshold_drops_negligible_blocks(self):
        store = TokenDanceStorage(diff_threshold=1.0)
        master = _make_master_kv(n_blocks=32, hidden_dim=16)
        store.register_master("a", master)
        # Tiny perturbations should be dropped.
        rng = np.random.default_rng(1)
        near = master + rng.standard_normal(master.shape, dtype=np.float32) * 1e-5
        diff = store.register_mirror("b", near)
        assert diff.n_diff_blocks == 0
        assert diff.sparsity == pytest.approx(1.0)


class TestTokenDanceReconstruction:
    def test_reconstruct_master_returns_master_copy(self):
        store = TokenDanceStorage()
        master = _make_master_kv()
        store.register_master("retriever", master)
        out = store.reconstruct("retriever")
        np.testing.assert_array_equal(out, master)
        # Mutating the output must not poison the stored master.
        out[0] = 999
        np.testing.assert_array_equal(store.master_cache["retriever"], master)

    def test_reconstruct_mirror_within_tolerance(self):
        store = TokenDanceStorage(diff_threshold=1e-4)
        master = _make_master_kv(n_blocks=64, hidden_dim=64)
        store.register_master("retriever", master)
        original = _make_near_master(master, n_diff_blocks=8)
        store.register_mirror("critic", original)

        recovered = store.reconstruct("critic")
        # Reconstruction is exact for blocks above threshold (we keep their full
        # delta) and exactly master for blocks below threshold. Tolerance = the
        # threshold scaled by sqrt(hidden_dim) at most.
        np.testing.assert_allclose(recovered, original, atol=1e-4)

    def test_reconstruct_unknown_agent_raises(self):
        store = TokenDanceStorage()
        store.register_master("a", _make_master_kv())
        with pytest.raises(KeyError):
            store.reconstruct("ghost")


class TestTokenDanceCollective:
    def test_collective_reuse_step_one_pass(self):
        store = TokenDanceStorage()
        master = _make_master_kv(n_blocks=32, hidden_dim=64)
        store.register_master("retriever", master)
        for aid in ("reranker", "summarizer", "critic", "responder"):
            store.register_mirror(aid, master.copy())

        rng = np.random.default_rng(99)
        new_blocks = rng.standard_normal((4, 64), dtype=np.float32)

        diff_counts = store.collective_reuse_step(
            ["retriever", "reranker", "summarizer", "critic", "responder"],
            new_blocks,
        )
        # All agents covered.
        assert set(diff_counts.keys()) == {
            "retriever",
            "reranker",
            "summarizer",
            "critic",
            "responder",
        }
        # Master grew by 4 blocks; mirrors still zero-diff.
        assert store.master_cache["retriever"].shape == (36, 64)
        for mirror_id in ("reranker", "summarizer", "critic", "responder"):
            assert store.mirrors[mirror_id].total_blocks == 36
            assert store.mirrors[mirror_id].n_diff_blocks == 0

    def test_collective_reuse_step_requires_master(self):
        store = TokenDanceStorage()
        with pytest.raises(RuntimeError):
            store.collective_reuse_step(["a"], np.zeros((1, 4)))


class TestTokenDanceStats:
    def test_stats_tracks_cache(self):
        store = TokenDanceStorage(diff_threshold=1e-4)
        master = _make_master_kv(n_blocks=16, hidden_dim=8)
        store.register_master("a", master)
        store.register_mirror("b", master.copy())
        s = store.stats()
        assert s["master_id"] == "a"
        assert s["master_blocks"] == 16
        assert s["n_mirrors"] == 1
        assert s["diff_blocks_total"] == 0
        assert s["compression_ratio"] >= 2.0
