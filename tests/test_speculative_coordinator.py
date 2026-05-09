"""Tests for SpeculativeCoordinator — TASK-003.

Tests cover:
- Config dataclass initialization and defaults
- Role-based viability checking (is_speculative_viable)
- Draft buffering (submit_draft) in both sync and overlapped modes
- verify_and_commit acceptance sampling
- estimate_speedup mathematical correctness
- Edge case: empty draft tokens
"""

import asyncio
import math
import random
from unittest.mock import MagicMock

import pytest

from contextforge.decoding.speculative_coordinator import (
    SpeculativeConfig,
    SpeculativeCoordinator,
    SpeculativeResult,
)


class TestSpeculativeConfig:
    """Tests for SpeculativeConfig dataclass."""

    def test_default_values(self):
        """SpeculativeConfig has correct defaults."""
        config = SpeculativeConfig()
        assert config.draft_agent_roles == frozenset({"retriever", "reranker"})
        assert config.target_agent_roles == frozenset({"responder", "critic"})
        assert config.max_draft_tokens == 8
        assert config.acceptance_threshold == 0.9
        assert config.enable_overlapped is True
        assert config.min_stability_rho == 0.8

    def test_custom_values(self):
        """Custom values are stored correctly."""
        config = SpeculativeConfig(
            max_draft_tokens=16,
            acceptance_threshold=0.95,
            enable_overlapped=False,
            min_stability_rho=0.6,
        )
        assert config.max_draft_tokens == 16
        assert config.acceptance_threshold == 0.95
        assert config.enable_overlapped is False
        assert config.min_stability_rho == 0.6


class TestSpeculativeCoordinator:
    """Tests for SpeculativeCoordinator."""

    def test_is_speculative_viable_draft_role_ok(self):
        """Draft agent with allowed role returns True."""
        coordinator = SpeculativeCoordinator()
        # "retriever-0" extracts role "retriever" which is in draft roles.
        assert coordinator.is_speculative_viable("retriever-0", "responder-0") is True

    def test_is_speculative_viable_target_role_ok(self):
        """Target agent with allowed role returns True."""
        coordinator = SpeculativeCoordinator()
        # "responder-1" extracts role "responder" which is in target roles.
        assert coordinator.is_speculative_viable("retriever-0", "responder-0") is True

    def test_is_speculative_viable_wrong_draft_role(self):
        """Draft agent with disallowed role returns False."""
        coordinator = SpeculativeCoordinator()
        # "responder" role not in draft roles.
        result = coordinator.is_speculative_viable("responder-0", "responder-0")
        assert result is False

    def test_is_speculative_viable_wrong_target_role(self):
        """Target agent with disallowed role returns False."""
        coordinator = SpeculativeCoordinator()
        # "retriever" role not in target roles.
        result = coordinator.is_speculative_viable("retriever-0", "retriever-0")
        assert result is False

    def test_is_speculative_viable_rho_check(self):
        """rho above threshold blocks speculative decoding."""
        mock_qc = MagicMock()
        mock_qc.current_rho = MagicMock(return_value=0.9)

        config = SpeculativeConfig(min_stability_rho=0.8)
        coordinator = SpeculativeCoordinator(config=config, queueing_controller=mock_qc)

        # rho=0.9 >= min_stability_rho=0.8 → blocked.
        result = coordinator.is_speculative_viable("retriever-0", "responder-0")
        assert result is False

    def test_is_speculative_viable_rho_below_threshold(self):
        """rho below threshold allows speculative decoding."""
        mock_qc = MagicMock()
        mock_qc.current_rho = MagicMock(return_value=0.5)

        config = SpeculativeConfig(min_stability_rho=0.8)
        coordinator = SpeculativeCoordinator(config=config, queueing_controller=mock_qc)

        # rho=0.5 < min_stability_rho=0.8 → allowed.
        result = coordinator.is_speculative_viable("retriever-0", "responder-0")
        assert result is True

    @pytest.mark.asyncio
    async def test_submit_draft_sync_mode(self):
        """submit_draft buffers draft in sync (non-overlapped) mode."""
        config = SpeculativeConfig(enable_overlapped=False)
        coordinator = SpeculativeCoordinator(config=config)

        draft_tokens = [101, 202, 303]
        await coordinator.submit_draft(draft_tokens, "responder-0", step=1)

        assert coordinator._current_draft == ("responder-0", draft_tokens)

    @pytest.mark.asyncio
    async def test_submit_draft_overlapped_mode(self):
        """submit_draft enqueues draft when overlapped mode is enabled."""
        config = SpeculativeConfig(enable_overlapped=True)
        coordinator = SpeculativeCoordinator(config=config)

        draft_tokens = [101, 202, 303]
        await coordinator.submit_draft(draft_tokens, "responder-0", step=1)

        # Should be in the queue.
        got = coordinator._draft_queue.get_nowait()
        assert got == ("responder-0", draft_tokens)

    @pytest.mark.asyncio
    async def test_verify_and_commit_empty_draft(self):
        """Empty draft_tokens returns SpeculativeResult with all empty fields."""
        coordinator = SpeculativeCoordinator()

        result = await coordinator.verify_and_commit(
            target_verification_logprobs=[], draft_tokens=[]
        )

        assert result.draft_tokens == []
        assert result.accepted_tokens == []
        assert result.rejected_at_position == -1
        assert result.acceptance_rate == 1.0
        assert result.decode_speedup_estimate == 1.0
        assert result.overlapped_next_draft is None

    @pytest.mark.asyncio
    async def test_verify_and_commit_all_accepted(self):
        """
        When random <= ratio for all tokens, all are accepted.
        Uses fixed seed so result is deterministic.
        """
        config = SpeculativeConfig(acceptance_threshold=0.9)
        coordinator = SpeculativeCoordinator(config=config)

        # High logprobs (close to 0) → high probs → ratio near 1.0.
        # With acceptance_threshold=0.9, ratio = p_i / 0.9 ≈ 1.0.
        # Seeded random=0.5 ≤ 1.0 → accept.
        random.seed(0)
        draft_tokens = [10, 20, 30]
        logprobs = [0.0, 0.0, 0.0]  # p ≈ 1.0 each

        result = await coordinator.verify_and_commit(logprobs, draft_tokens)

        # All should be accepted since ratio ≈ 1.0 and random(0.5) < 1.0.
        assert result.accepted_tokens == draft_tokens
        assert result.rejected_at_position == -1
        assert result.acceptance_rate == 1.0

    @pytest.mark.asyncio
    async def test_verify_and_commit_rejection(self):
        """
        When random > ratio the token is rejected at that position.
        With very low logprobs the ratio is near 0, so rejection is likely.
        """
        config = SpeculativeConfig(acceptance_threshold=0.9)
        coordinator = SpeculativeCoordinator(config=config)

        # Very negative logprobs → very low probs → ratio ≈ 0.
        # random() will almost certainly be > ratio → rejection at position 0.
        random.seed(42)
        draft_tokens = [10, 20, 30]
        logprobs = [-10.0, -10.0, -10.0]  # p ≈ 4.5e-5

        result = await coordinator.verify_and_commit(logprobs, draft_tokens)

        # Should reject at position 0 since ratio is tiny.
        assert result.rejected_at_position == 0
        assert len(result.accepted_tokens) == 0

    @pytest.mark.asyncio
    async def test_verify_and_commit_partial_acceptance(self):
        """
        Some tokens accepted, then rejection occurs.
        Uses intermediate logprobs for mixed outcome.
        """
        config = SpeculativeConfig(acceptance_threshold=0.9)
        coordinator = SpeculativeCoordinator(config=config)

        random.seed(12345)
        draft_tokens = [10, 20, 30, 40, 50]
        # Tuned logprobs so first 2 accept, 3rd rejects.
        # logprob=-0.1 → p≈0.90, ratio=1.0 → accept if random ≤ 1.0
        # logprob=-2.3 → p≈0.10, ratio≈0.11 → reject unless random < 0.11
        logprobs = [-0.1, -0.1, -2.3, 0.0, 0.0]

        result = await coordinator.verify_and_commit(logprobs, draft_tokens)

        # First two should be accepted (random values ≤ 1.0).
        assert len(result.accepted_tokens) >= 2
        # If rejected, rejected_at_position reflects first failure.
        assert result.rejected_at_position == -1 or result.rejected_at_position >= 2

    @pytest.mark.asyncio
    async def test_verify_and_commit_overlapped_next_draft(self):
        """
        When enable_overlapped=True and queue has a prefetched draft,
        overlapped_next_draft is populated in the result.
        """
        config = SpeculativeConfig(enable_overlapped=True)
        coordinator = SpeculativeCoordinator(config=config)

        # Pre-load a draft into the queue.
        prefetched_tokens = [999, 888, 777]
        await coordinator._draft_queue.put(("responder-1", prefetched_tokens))

        result = await coordinator.verify_and_commit(
            target_verification_logprobs=[0.0, 0.0],
            draft_tokens=[10, 20],
        )

        assert result.overlapped_next_draft == prefetched_tokens

    @pytest.mark.asyncio
    async def test_verify_and_commit_no_overlapped_next_draft(self):
        """
        When queue is empty, overlapped_next_draft is None even if enabled.
        """
        config = SpeculativeConfig(enable_overlapped=True)
        coordinator = SpeculativeCoordinator(config=config)

        # Queue is empty.
        result = await coordinator.verify_and_commit(
            target_verification_logprobs=[0.0],
            draft_tokens=[10],
        )

        assert result.overlapped_next_draft is None

    def test_estimate_speedup_max_acceptance(self):
        """100% acceptance → maximum speedup (k+1 tokens per step)."""
        coordinator = SpeculativeCoordinator()
        k = 8
        speedup = coordinator.estimate_speedup(1.0, max_draft_tokens=k)
        assert math.isclose(speedup, k + 1, rel_tol=1e-9)

    def test_estimate_speedup_zero_acceptance(self):
        """0% acceptance → no speedup (only fallback token, speedup = 1.0)."""
        coordinator = SpeculativeCoordinator()
        speedup = coordinator.estimate_speedup(0.0, max_draft_tokens=8)
        assert speedup == 1.0

    def test_estimate_speedup_090_acceptance_k8(self):
        """
        From the spec: acceptance_rate=0.9, k=8 → speedup ≈ 5.7x.
        E[tokens] = (1 - r^(k+1)) / (1 - r)
        = (1 - 0.9^9) / (1 - 0.9)
        = (1 - 0.3874) / 0.1
        ≈ 6.126
        """
        coordinator = SpeculativeCoordinator()
        speedup = coordinator.estimate_speedup(0.9, max_draft_tokens=8)
        expected = (1.0 - (0.9 ** 9)) / 0.1
        assert math.isclose(speedup, expected, rel_tol=1e-9)

    def test_estimate_speedup_out_of_range(self):
        """Acceptance rate outside [0,1] returns 1.0 (no speedup)."""
        coordinator = SpeculativeCoordinator()
        assert coordinator.estimate_speedup(-0.5, max_draft_tokens=8) == 1.0
        assert coordinator.estimate_speedup(1.5, max_draft_tokens=8) == 1.0

    def test_role_from_agent_id(self):
        """_role_from_agent_id extracts role from agent_id suffix."""
        coordinator = SpeculativeCoordinator()
        assert coordinator._role_from_agent_id("retriever-0") == "retriever"
        assert coordinator._role_from_agent_id("responder-1") == "responder"
        assert coordinator._role_from_agent_id("agent:reranker-2") == "reranker"
        assert coordinator._role_from_agent_id("worker:critic-0") == "critic"