"""Tests for the Z3 SMT formal proof of INV-15."""
from __future__ import annotations

import pytest

z3 = pytest.importorskip("z3")

from apohara_context_forge.safety.z3_inv15_proof import (  # noqa: E402
    build_inv15_constraints,
    prove_inv15,
)


def test_inv15_proved_under_canonical_antecedent():
    """The canonical antecedent (critic + n>=9 + reuse=0 + shuffled) MUST prove."""
    result = prove_inv15()
    assert result["status"] == "PROVED", f"INV-15 not proved: {result}"
    assert result["theorem"] == "INV-15-DENSE-PREFILL"
    assert result["model"] is None
    assert "z3_version" in result


def test_inv15_proof_completes_quickly():
    """Linear-arithmetic SMT should finish in <1000 ms."""
    result = prove_inv15()
    assert result["elapsed_ms"] < 1000, (
        f"Proof too slow: {result['elapsed_ms']}ms"
    )


def test_inv15_counterexample_when_assumptions_relaxed():
    """When the antecedent is relaxed to a benign config, dense_prefill can be FALSE.

    This exercises the model: drop the 'critic' role, low candidate count,
    no shuffle, low reuse. risk_score should fall below 0.7 → dense_prefill
    must be FALSE (and SAT, since a concrete assignment exists).
    """
    solver = z3.Solver()
    (
        agent_role_critic,
        candidate_count,
        reuse_rate,
        layout_shuffled,
        _risk_score,
        use_dense_prefill,
    ) = build_inv15_constraints(solver)

    solver.add(agent_role_critic == False)  # noqa: E712
    solver.add(candidate_count == 1)
    solver.add(reuse_rate == 1.0)
    solver.add(layout_shuffled == False)  # noqa: E712

    assert solver.check() == z3.sat
    model = solver.model()
    # Non-critic role => dense_prefill must be False regardless of risk.
    assert bool(model[use_dense_prefill]) is False
