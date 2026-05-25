"""Tests for JCRSafetyGate.

Covers:
- Risk score computation across the role / candidate / shuffle / reuse axes
- INV-15: Critic with risk > threshold ALWAYS uses dense prefill
- Non-judge roles never trigger dense fallback
- gate_decision logging + summary stats
- Edge case: invalid args
"""
from __future__ import annotations

import pytest

from apohara_context_forge.safety.jcr_gate import (
    JCRDecision,
    JCRSafetyGate,
)


class TestJCRSafetyGateDefaults:
    def test_default_threshold(self):
        gate = JCRSafetyGate()
        assert gate.jcr_threshold == 0.7

    def test_invalid_threshold_rejected(self):
        with pytest.raises(ValueError, match="must be in"):
            JCRSafetyGate(jcr_threshold=1.5)
        with pytest.raises(ValueError, match="must be in"):
            JCRSafetyGate(jcr_threshold=-0.1)


class TestJCRRiskComputation:
    def test_critic_base_risk(self):
        gate = JCRSafetyGate()
        risk = gate.compute_jcr_risk(
            agent_role="critic",
            candidate_count=2,
            reuse_rate=0.5,
            layout_shuffled=False,
        )
        assert risk == pytest.approx(0.6)

    def test_non_critic_base_risk(self):
        gate = JCRSafetyGate()
        risk = gate.compute_jcr_risk(
            agent_role="retriever",
            candidate_count=2,
            reuse_rate=0.5,
            layout_shuffled=False,
        )
        assert risk == pytest.approx(0.1)

    def test_extra_candidates_increase_risk(self):
        gate = JCRSafetyGate()
        baseline = gate.compute_jcr_risk("critic", 2, 0.0, False)
        five = gate.compute_jcr_risk("critic", 5, 0.0, False)
        assert five == pytest.approx(baseline + 0.3)

    def test_layout_shuffled_increases_risk(self):
        gate = JCRSafetyGate()
        plain = gate.compute_jcr_risk("critic", 2, 0.0, False)
        shuffled = gate.compute_jcr_risk("critic", 2, 0.0, True)
        assert shuffled == pytest.approx(plain + 0.2)

    def test_high_reuse_rate_increases_risk(self):
        gate = JCRSafetyGate()
        low = gate.compute_jcr_risk("critic", 2, 0.5, False)
        high = gate.compute_jcr_risk("critic", 2, 0.95, False)
        assert high == pytest.approx(low + 0.15)

    def test_risk_clamped_to_one(self):
        gate = JCRSafetyGate()
        risk = gate.compute_jcr_risk(
            agent_role="critic",
            candidate_count=20,
            reuse_rate=1.0,
            layout_shuffled=True,
        )
        assert 0.0 <= risk <= 1.0
        assert risk == pytest.approx(1.0)

    def test_invalid_candidate_count_rejected(self):
        gate = JCRSafetyGate()
        with pytest.raises(ValueError, match="non-negative"):
            gate.compute_jcr_risk("critic", -1, 0.5, False)

    def test_invalid_reuse_rate_rejected(self):
        gate = JCRSafetyGate()
        with pytest.raises(ValueError, match="reuse_rate must be"):
            gate.compute_jcr_risk("critic", 2, 1.5, False)


class TestINV15CriticAlwaysDense:
    """INV-15: Critic with risk > threshold ALWAYS returns use_dense=True."""

    def test_critic_5_candidates_shuffle_uses_dense(self):
        gate = JCRSafetyGate()
        # Risk = 0.6 + 0.3 + 0.2 = 1.1 → clamped to 1.0 → > 0.7
        assert gate.should_use_dense_prefill(
            agent_role="critic",
            candidate_count=5,
            reuse_rate=0.5,
            layout_shuffled=True,
        ) is True

    def test_retriever_2_candidates_no_dense(self):
        gate = JCRSafetyGate()
        assert gate.should_use_dense_prefill(
            agent_role="retriever",
            candidate_count=2,
            reuse_rate=0.5,
            layout_shuffled=False,
        ) is False

    def test_non_critic_never_uses_dense_even_with_high_risk(self):
        """Non-judge roles aren't protected by INV-15."""
        gate = JCRSafetyGate()
        # Even with all risk knobs cranked up, a retriever passes through.
        assert gate.should_use_dense_prefill(
            agent_role="retriever",
            candidate_count=10,
            reuse_rate=1.0,
            layout_shuffled=True,
        ) is False

    @pytest.mark.parametrize("candidates,shuffle,reuse", [
        (5, True, 0.9),
        (4, True, 0.85),
        (8, False, 0.85),
        (10, True, 0.5),
    ])
    def test_critic_above_threshold_always_dense(self, candidates, shuffle, reuse):
        """Comprehensive sweep: Critic above threshold always dense (INV-15)."""
        gate = JCRSafetyGate()
        decision = gate.gate_decision(
            agent_role="critic",
            candidate_count=candidates,
            reuse_rate=reuse,
            layout_shuffled=shuffle,
        )
        if decision.risk_score > gate.jcr_threshold:
            assert decision.use_dense is True, (
                f"INV-15 violated: critic with risk {decision.risk_score} "
                f"> threshold {gate.jcr_threshold} did not get dense prefill"
            )

    def test_critic_exactly_at_threshold_uses_reuse(self):
        """Threshold is strict: > threshold triggers dense, not >=."""
        gate = JCRSafetyGate(jcr_threshold=0.6)
        # Critic, 2 candidates, no shuffle, low reuse → exactly 0.6
        decision = gate.gate_decision(
            agent_role="critic",
            candidate_count=2,
            reuse_rate=0.5,
            layout_shuffled=False,
        )
        assert decision.risk_score == pytest.approx(0.6)
        assert decision.use_dense is False


class TestGateDecisionLogging:
    def test_gate_decision_returns_structured_record(self):
        gate = JCRSafetyGate()
        decision = gate.gate_decision("critic", 5, 0.9, True)
        assert isinstance(decision, JCRDecision)
        assert decision.agent_role == "critic"
        assert decision.use_dense is True
        assert "INV-15" in decision.reason
        assert decision.timestamp > 0

    def test_log_accumulates(self):
        gate = JCRSafetyGate()
        for _ in range(3):
            gate.gate_decision("critic", 5, 0.9, True)
        gate.gate_decision("retriever", 2, 0.1, False)
        assert len(gate.gate_log) == 4

    def test_summary_aggregates(self):
        gate = JCRSafetyGate()
        gate.gate_decision("critic", 5, 0.9, True)   # dense
        gate.gate_decision("critic", 2, 0.1, False)  # reuse
        gate.gate_decision("retriever", 2, 0.1, False)  # reuse
        s = gate.summary()
        assert s["total_decisions"] == 3
        assert s["dense_fallback_count"] == 1
        # 2 critic decisions, 1 dense → 0.5
        assert s["critic_dense_rate"] == pytest.approx(0.5)
        assert 0.0 <= s["avg_risk_score"] <= 1.0

    def test_summary_empty_safe(self):
        gate = JCRSafetyGate()
        s = gate.summary()
        assert s["total_decisions"] == 0
        assert s["dense_fallback_count"] == 0
        assert s["avg_risk_score"] == 0.0
        assert s["critic_dense_rate"] == 0.0

    def test_role_case_insensitive(self):
        gate = JCRSafetyGate()
        # Upper-case role still resolves to "critic".
        decision = gate.gate_decision("CRITIC", 5, 0.9, True)
        assert decision.agent_role == "critic"
        assert decision.use_dense is True


class TestINV15JudgeRole:
    """INV-15 now protects the generic 'judge' role too (J = {critic, judge})."""

    def test_judge_base_risk(self):
        gate = JCRSafetyGate()
        assert gate.compute_jcr_risk("judge", 2, 0.5, False) == pytest.approx(0.6)

    def test_judge_above_threshold_uses_dense(self):
        gate = JCRSafetyGate()
        # judge, 5 candidates, shuffled, high reuse → risk clamps to 1.0 > 0.7
        assert gate.should_use_dense_prefill("judge", 5, 0.95, True) is True

    def test_judge_case_insensitive(self):
        gate = JCRSafetyGate()
        decision = gate.gate_decision("JUDGE", 5, 0.9, True)
        assert decision.agent_role == "judge"
        assert decision.use_dense is True
