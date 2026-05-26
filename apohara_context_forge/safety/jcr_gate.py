"""JCR Safety Gate — protects judge-type agents from KV-reuse drift.

Based on arXiv:2601.08343 (Jan 2026): "When KV Cache Reuse Fails in
Multi-Agent Systems."

The paper shows that aggressive KV-cache reuse can silently degrade the
Judge Consistency Rate (JCR) of judge-type agents (Critic, evaluator)
even when raw accuracy looks unchanged. The Critic in our 5-agent
pipeline is especially vulnerable because it jointly compares multiple
candidates: shuffling the candidate order or reusing KV blocks across
candidates can flip the verdict.

INV-15
======
Any judge-class agent (role in {"critic", "judge"}) MUST use dense prefill —
bypassing the shared KV cache — whenever the JCR risk score exceeds the
threshold (default 0.7). This is enforced unconditionally inside
should_use_dense_prefill().
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

# Roles considered "judge-type" — these are the protected callers (INV-15).
# Per the paper §4.1, J = {critic, judge}: both get dense prefill when risky.
JUDGE_ROLES = frozenset({"critic", "judge"})

# Default risk threshold above which dense prefill is mandated.
DEFAULT_JCR_THRESHOLD = 0.7

# Risk-model constants (from arXiv:2601.08343 Sec. 4 table 2).
_BASE_RISK_JUDGE = 0.6
_BASE_RISK_OTHER = 0.1
_RISK_PER_EXTRA_CANDIDATE = 0.10  # +0.1 per candidate beyond 2
_RISK_LAYOUT_SHUFFLED = 0.20      # +0.2 if order changed since last round
_RISK_HIGH_REUSE = 0.15           # +0.15 if reuse_rate > 0.8
_HIGH_REUSE_THRESHOLD = 0.8


@dataclass
class JCRDecision:
    """A single gate decision, captured for telemetry / dashboard."""

    agent_role: str
    risk_score: float
    use_dense: bool
    reason: str
    timestamp: float = field(default_factory=time.time)


class JCRSafetyGate:
    """Safety gate that detects when KV reuse is risky for judge-type agents.

    Falls back to dense prefill for the Critic agent when JCR risk is
    high. INV-15 is enforced inside should_use_dense_prefill() and
    gate_decision() — Critic above the threshold ALWAYS gets dense.
    """

    def __init__(self, jcr_threshold: float = DEFAULT_JCR_THRESHOLD):
        if not 0.0 <= jcr_threshold <= 1.0:
            raise ValueError(
                f"jcr_threshold must be in [0, 1]; got {jcr_threshold}"
            )
        self.jcr_threshold: float = jcr_threshold
        self.gate_log: list[JCRDecision] = []

    # ------------------------------------------------------------------ #
    # Risk scoring                                                        #
    # ------------------------------------------------------------------ #

    def compute_jcr_risk(
        self,
        agent_role: str,
        candidate_count: int,
        reuse_rate: float,
        layout_shuffled: bool,
    ) -> float:
        """Compute the JCR risk score for an upcoming agent step.

        Returns a value in [0.0, 1.0]. Higher means KV reuse is more
        likely to corrupt the judge's verdict.
        """
        if candidate_count < 0:
            raise ValueError("candidate_count must be non-negative")
        if not 0.0 <= reuse_rate <= 1.0:
            raise ValueError("reuse_rate must be in [0, 1]")

        role = (agent_role or "").lower()
        risk = _BASE_RISK_JUDGE if role in JUDGE_ROLES else _BASE_RISK_OTHER
        if candidate_count > 2:
            risk += _RISK_PER_EXTRA_CANDIDATE * (candidate_count - 2)
        if layout_shuffled:
            risk += _RISK_LAYOUT_SHUFFLED
        if reuse_rate > _HIGH_REUSE_THRESHOLD:
            risk += _RISK_HIGH_REUSE

        return max(0.0, min(1.0, risk))

    # ------------------------------------------------------------------ #
    # Gate decision (INV-15 enforcement)                                  #
    # ------------------------------------------------------------------ #

    def should_use_dense_prefill(
        self,
        agent_role: str,
        candidate_count: int,
        reuse_rate: float,
        layout_shuffled: bool,
    ) -> bool:
        """INV-15: returns True iff judge-role risk exceeds the threshold.

        Non-judge roles always pass through (use_dense=False) — the
        threshold is only meaningful for the Critic and other judge-type
        agents because non-judges aren't protected by this invariant.
        """
        risk = self.compute_jcr_risk(
            agent_role, candidate_count, reuse_rate, layout_shuffled
        )
        role = (agent_role or "").lower()
        if role in JUDGE_ROLES and risk > self.jcr_threshold:
            return True
        return False

    def gate_decision(
        self,
        agent_role: str,
        candidate_count: int,
        reuse_rate: float,
        layout_shuffled: bool,
    ) -> JCRDecision:
        """Make a gate decision and append it to the audit log."""
        risk = self.compute_jcr_risk(
            agent_role, candidate_count, reuse_rate, layout_shuffled
        )
        role = (agent_role or "").lower()
        is_judge = role in JUDGE_ROLES
        use_dense = is_judge and risk > self.jcr_threshold

        if not is_judge:
            reason = f"role={role!r} not judge-type → reuse OK"
        elif use_dense:
            reason = (
                f"INV-15: judge role={role!r} risk={risk:.2f} > "
                f"threshold={self.jcr_threshold:.2f} → dense prefill mandated"
            )
        else:
            reason = (
                f"judge role={role!r} risk={risk:.2f} ≤ "
                f"threshold={self.jcr_threshold:.2f} → reuse permitted"
            )

        decision = JCRDecision(
            agent_role=role,
            risk_score=risk,
            use_dense=use_dense,
            reason=reason,
        )
        self.gate_log.append(decision)

        # Late import of recorders avoids a load-time cycle; best-effort telemetry.
        try:
            from apohara_context_forge.observability import recorders
            gate_action = "block" if use_dense else "allow"
            if os.environ.get("APOHARA_FORGE_LEDGER"):
                recorders.record_certified_inv15_decision(
                    agent_id=role, anchor_hash="", risk_score=risk,
                    gate_action=gate_action, predicted_jcr_delta=0.0,
                    candidate_count=candidate_count, reuse_rate=reuse_rate,
                    layout_shuffled=layout_shuffled, use_dense=use_dense,
                )
            else:
                recorders.record_inv15_decision(
                    agent_id=role, anchor_hash="", risk_score=risk,
                    gate_action=gate_action, predicted_jcr_delta=0.0,
                )
        except Exception:
            pass

        return decision

    # ------------------------------------------------------------------ #
    # Telemetry                                                           #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict[str, float | int]:
        """Aggregate stats over all decisions logged so far."""
        total = len(self.gate_log)
        if total == 0:
            return {
                "total_decisions": 0,
                "dense_fallback_count": 0,
                "avg_risk_score": 0.0,
                "critic_dense_rate": 0.0,
            }

        dense_count = sum(1 for d in self.gate_log if d.use_dense)
        avg_risk = sum(d.risk_score for d in self.gate_log) / total
        critic_decisions = [d for d in self.gate_log if d.agent_role == "critic"]
        critic_dense = sum(1 for d in critic_decisions if d.use_dense)
        critic_rate = (
            critic_dense / len(critic_decisions) if critic_decisions else 0.0
        )

        return {
            "total_decisions": total,
            "dense_fallback_count": dense_count,
            "avg_risk_score": avg_risk,
            "critic_dense_rate": critic_rate,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        s = self.summary()
        return (
            f"JCRSafetyGate(threshold={self.jcr_threshold:.2f}, "
            f"decisions={s['total_decisions']}, "
            f"dense={s['dense_fallback_count']}, "
            f"avg_risk={s['avg_risk_score']:.2f}, "
            f"critic_dense_rate={s['critic_dense_rate']:.2f})"
        )
