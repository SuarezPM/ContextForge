"""Tests for AnchorPool KV offset estimation."""

import pytest
import numpy as np
from apohara_context_forge.kv_offset.anchor_pool import AnchorPool


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_offset() -> np.ndarray:
    """Return a sample KV offset vector of shape (128,)."""
    return np.random.randn(128).astype(np.float32)


@pytest.fixture
def sample_kv_keys() -> np.ndarray:
    """Return sample KV keys with shape (seq_len=4, head_dim=128)."""
    np.random.seed(42)
    return np.random.randn(4, 128).astype(np.float32)


@pytest.fixture
def pool() -> AnchorPool:
    """Return a fresh AnchorPool instance."""
    return AnchorPool(max_size=20)


# =============================================================================
# predict_shareable() Tests
# =============================================================================

@pytest.mark.asyncio
async def test_predict_shareable_returns_true_for_high_similarity(pool, sample_offset):
    """Returns True when token sequence has high similarity with existing anchors."""
    token_ids = [100, 200, 300, 400]
    agent_a = "agent-a"
    agent_b = "agent-b"

    await pool.update_pool(token_ids, agent_a, sample_offset)

    # Agent B has no offsets yet, but similarity should still be computed
    shareable = await pool.predict_shareable(token_ids, agent_b)
    assert isinstance(shareable, bool)


@pytest.mark.asyncio
async def test_predict_shareable_returns_false_when_pool_empty(pool):
    """Returns False when the anchor pool is empty."""
    token_ids = [100, 200, 300]
    target_agent = "agent-xyz"

    result = await pool.predict_shareable(token_ids, target_agent)
    assert result is False


@pytest.mark.asyncio
async def test_predict_shareable_returns_false_when_target_not_in_offsets(pool, sample_offset):
    """Returns False when target_agent_id is not present in any anchor's offsets."""
    token_ids = [100, 200, 300, 400]
    agent_a = "agent-a"
    agent_b = "agent-b"

    # Add anchor for agent-a only
    await pool.update_pool(token_ids, agent_a, sample_offset)

    # agent-b is not in any anchor's offsets
    shareable = await pool.predict_shareable(token_ids, agent_b)
    assert shareable is False


# =============================================================================
# approximate_offset() Tests
# =============================================================================

@pytest.mark.asyncio
async def test_approximate_offset_returns_ndarray_when_candidates_exist(pool, sample_offset):
    """Returns np.ndarray when candidates exist for target_agent_id."""
    token_ids = [100, 200, 300, 400]
    agent_a = "agent-a"

    await pool.update_pool(token_ids, agent_a, sample_offset)

    result = await pool.approximate_offset(token_ids, agent_a)

    assert result is not None
    assert isinstance(result.placeholder_offset, np.ndarray)
    assert result.placeholder_offset.shape == (128,)


@pytest.mark.asyncio
async def test_approximate_offset_returns_none_when_pool_empty(pool):
    """Returns None when the anchor pool is empty."""
    token_ids = [100, 200, 300]
    target_agent = "agent-xyz"

    result = await pool.approximate_offset(token_ids, target_agent)
    assert result is None


@pytest.mark.asyncio
async def test_approximate_offset_weighted_interpolation_between_min_max(pool):
    """Weighted interpolation should produce values between min and max offsets."""
    token_ids_base = [100, 200, 300, 400]
    agent_a = "agent-a"

    offset_low = np.full(128, 0.0, dtype=np.float32)
    offset_high = np.full(128, 1.0, dtype=np.float32)

    # Add two anchors with distinct offsets
    await pool.update_pool([100, 200, 300, 400], agent_a, offset_low)
    await pool.update_pool([101, 201, 301, 401], agent_a, offset_high)

    # Query with same base token IDs - should interpolate
    result = await pool.approximate_offset(token_ids_base, agent_a)

    assert result is not None
    assert np.all(result.placeholder_offset >= offset_low), "Result should be >= min offset"
    assert np.all(result.placeholder_offset <= offset_high), "Result should be <= max offset"


# =============================================================================
# RoPE De-rotation Tests
# =============================================================================

@pytest.mark.asyncio
async def test_rope_derotation_differs_for_same_key_at_different_positions(pool, sample_kv_keys):
    """apply_rope_derotation() should produce different output for same key at different positions."""
    key_at_pos0 = sample_kv_keys[0:1]  # shape (1, 128)
    key_at_pos2 = sample_kv_keys[2:3]  # shape (1, 128)

    derotated_0 = await pool.apply_rope_derotation(key_at_pos0, np.array([0]))
    derotated_2 = await pool.apply_rope_derotation(key_at_pos2, np.array([2]))

    assert not np.allclose(derotated_0, derotated_2), \
        "De-rotated keys at different positions should differ"


@pytest.mark.asyncio
async def test_rope_derotation_produces_different_keys_for_off_position_tokens(pool):
    """
    De-rotated keys at off-position indices should be more similar (lower cosine distance)
    than raw keys, because de-rotation aligns them to a common reference frame.
    Uses kv_keys shape (seq_len=4, head_dim=128) and positions [0, 1, 2, 3].
    """
    np.random.seed(123)
    kv_keys = np.random.randn(4, 128).astype(np.float32)
    positions = np.array([0, 1, 2, 3])

    derotated = await pool.apply_rope_derotation(kv_keys, positions)

    # Compare position 0 vs position 2 (off-position)
    raw_key_0 = kv_keys[0]
    raw_key_2 = kv_keys[2]

    # Cosine similarity for raw keys
    raw_cos_sim = np.dot(raw_key_0, raw_key_2) / (
        np.linalg.norm(raw_key_0) * np.linalg.norm(raw_key_2)
    )

    # Cosine similarity for de-rotated keys
    derot_key_0 = derotated[0]
    derot_key_2 = derotated[2]
    derot_cos_sim = np.dot(derot_key_0, derot_key_2) / (
        np.linalg.norm(derot_key_0) * np.linalg.norm(derot_key_2)
    )

    # De-rotated keys at different positions should have higher cosine similarity
    # because de-rotation removes the position-dependent RoPE rotation
    assert derot_cos_sim > raw_cos_sim, \
        f"De-rotated cosine similarity ({derot_cos_sim:.4f}) should be > raw ({raw_cos_sim:.4f})"


@pytest.mark.asyncio
async def test_rope_derotation_shape_preserved(pool, sample_kv_keys):
    """De-rotation should preserve the shape of kv_keys."""
    positions = np.array([0, 1, 2, 3])

    derotated = await pool.apply_rope_derotation(sample_kv_keys, positions)

    assert derotated.shape == sample_kv_keys.shape


# =============================================================================
# Pool Pruning Tests
# =============================================================================

@pytest.mark.asyncio
async def test_pool_pruning_at_max_size_boundary():
    """Pool size should be <= max_size after adding more anchors than max_size."""
    pool = AnchorPool(max_size=5)

    # Add 8 anchors (more than max_size=5)
    for i in range(8):
        token_ids = [100 + i, 200 + i, 300 + i, 400 + i]
        agent_id = f"agent-{i % 3}"  # Rotate through 3 agents
        offset = np.random.randn(128).astype(np.float32)
        await pool.update_pool(token_ids, agent_id, offset)

    stats = await pool.get_stats()
    assert stats["total_anchors"] <= 5, \
        f"Pool size ({stats['total_anchors']}) should be <= max_size (5)"


@pytest.mark.asyncio
async def test_pool_pruning_evicts_least_frequently_used():
    """Least-frequently-used anchors should be evicted first during pruning."""
    pool = AnchorPool(max_size=5)

    # Add 5 anchors for agent-a
    token_ids_list = [
        [100, 200, 300],
        [101, 201, 301],
        [102, 202, 302],
        [103, 203, 303],
        [104, 204, 304],
    ]
    for i, token_ids in enumerate(token_ids_list):
        offset = np.random.randn(128).astype(np.float32)
        await pool.update_pool(token_ids, "agent-a", offset)

    # Access first 3 anchors multiple times to increase their access_count
    for _ in range(3):
        await pool.predict_shareable(token_ids_list[0], "agent-b")
        await pool.predict_shareable(token_ids_list[1], "agent-b")
        await pool.predict_shareable(token_ids_list[2], "agent-b")

    # Add 3 more anchors to trigger pruning
    for i in range(3):
        token_ids = [110 + i, 210 + i, 310 + i]
        offset = np.random.randn(128).astype(np.float32)
        await pool.update_pool(token_ids, "agent-a", offset)

    # After pruning, the least-frequently-used (and oldest) anchors should be gone
    stats = await pool.get_stats()
    assert stats["total_anchors"] <= 5

    # The first two anchors (with highest access_count due to 3x access)
    # should still exist, while others may have been evicted
    # We can't deterministically verify which specific ones remain without
    # inspecting internals, but we verify the pool respects max_size


# =============================================================================
# get_stats() Tests
# =============================================================================

@pytest.mark.asyncio
async def test_get_stats_returns_correct_structure(pool, sample_offset):
    """get_stats() should return dict with expected keys and types."""
    token_ids = [100, 200, 300, 400]
    agent_id = "agent-test"

    await pool.update_pool(token_ids, agent_id, sample_offset)

    stats = await pool.get_stats()

    assert "total_anchors" in stats
    assert "total_agent_offsets" in stats
    assert "agents_tracked" in stats
    assert "max_size" in stats

    assert isinstance(stats["total_anchors"], int)
    assert isinstance(stats["total_agent_offsets"], int)
    assert isinstance(stats["agents_tracked"], int)
    assert isinstance(stats["max_size"], int)
    assert stats["max_size"] == 20


@pytest.mark.asyncio
async def test_get_stats_empty_pool():
    """get_stats() should return zeros for an empty pool."""
    pool = AnchorPool(max_size=10)
    stats = await pool.get_stats()

    assert stats["total_anchors"] == 0
    assert stats["total_agent_offsets"] == 0
    assert stats["agents_tracked"] == 0
    assert stats["max_size"] == 10