"""Formal Z3 SMT proof of INV-15 (multi-agent KV-cache isolation invariant).

Complementary to the empirical sweep over JCRSafetyGate.gate_decision().
The empirical sweep samples the input space; Z3 reasons over the ENTIRE
modeled domain — proving the invariant holds for all inputs satisfying
the antecedent, not just sampled ones.

Theorem (INV-15-DENSE-PREFILL)
==============================
    For all inputs (agent_role, candidate_count, reuse_rate, layout_shuffled),
        agent_role is judge-class (critic or judge)
        AND candidate_count >= 9
        AND reuse_rate = 0
        AND layout_shuffled = TRUE
    ==> use_dense_prefill = TRUE

Proof strategy: assert the NEGATION of the conclusion under the
antecedent and ask Z3 if any assignment satisfies it. If Z3 returns
UNSAT, no counterexample exists and the invariant is formally valid
over the modeled domain.

The Z3 risk model mirrors the constants in jcr_gate.py:
    base_risk    = 0.6 if role in {critic, judge} else 0.1
    +0.10 * max(candidate_count - 2, 0)
    +0.20 if layout_shuffled
    +0.15 if reuse_rate > 0.8
    risk = clamp(sum, 0.0, 1.0)
    use_dense = (role in {critic, judge}) AND (risk > threshold)  # strict >

Use:
    python -m apohara_context_forge.safety.z3_inv15_proof
"""
from __future__ import annotations

import sys


try:
    import z3
except ImportError:
    print(
        "ERROR: z3-solver not installed. Run: pip install z3-solver",
        file=sys.stderr,
    )
    sys.exit(1)


# Risk-model constants mirror jcr_gate.py (kept in lockstep).
_BASE_RISK_JUDGE = "0.6"
_BASE_RISK_OTHER = "0.1"
_RISK_PER_EXTRA_CANDIDATE = "0.10"
_RISK_LAYOUT_SHUFFLED = "0.20"
_RISK_HIGH_REUSE = "0.15"
_HIGH_REUSE_THRESHOLD = "0.8"
_DEFAULT_JCR_THRESHOLD = "0.7"


def build_inv15_constraints(solver: "z3.Solver") -> tuple:
    """Encode the JCRSafetyGate decision logic as Z3 constraints.

    Returns the tuple of Z3 variables involved so callers can add
    antecedents and refute the invariant.
    """
    # Decision variables
    agent_role_judge = z3.Bool("agent_role_judge")
    candidate_count = z3.Int("candidate_count")
    reuse_rate = z3.Real("reuse_rate")
    layout_shuffled = z3.Bool("layout_shuffled")
    risk_threshold = z3.Real("risk_threshold")
    risk_raw = z3.Real("risk_raw")
    risk_score = z3.Real("risk_score")
    use_dense_prefill = z3.Bool("use_dense_prefill")

    # Domain constraints (input validation mirrors gate's ValueError checks)
    solver.add(candidate_count >= 0)
    solver.add(reuse_rate >= 0.0, reuse_rate <= 1.0)
    solver.add(risk_threshold == z3.RealVal(_DEFAULT_JCR_THRESHOLD))

    # Risk components (mirror jcr_gate.compute_jcr_risk).
    base_w = z3.If(
        agent_role_judge,
        z3.RealVal(_BASE_RISK_JUDGE),
        z3.RealVal(_BASE_RISK_OTHER),
    )
    # +0.10 per extra candidate above 2
    extra_candidates = z3.If(
        candidate_count > 2,
        z3.ToReal(candidate_count - 2) * z3.RealVal(_RISK_PER_EXTRA_CANDIDATE),
        z3.RealVal("0.0"),
    )
    shuffled_w = z3.If(
        layout_shuffled,
        z3.RealVal(_RISK_LAYOUT_SHUFFLED),
        z3.RealVal("0.0"),
    )
    reuse_w = z3.If(
        reuse_rate > z3.RealVal(_HIGH_REUSE_THRESHOLD),
        z3.RealVal(_RISK_HIGH_REUSE),
        z3.RealVal("0.0"),
    )

    solver.add(risk_raw == base_w + extra_candidates + shuffled_w + reuse_w)

    # Clamp to [0, 1] like the gate's max(0.0, min(1.0, risk)).
    solver.add(
        risk_score
        == z3.If(
            risk_raw > z3.RealVal("1.0"),
            z3.RealVal("1.0"),
            z3.If(risk_raw < z3.RealVal("0.0"), z3.RealVal("0.0"), risk_raw),
        )
    )

    # INV-15: dense prefill iff (judge-class role AND risk > threshold). Strict >.
    solver.add(
        use_dense_prefill == z3.And(agent_role_judge, risk_score > risk_threshold)
    )

    return (
        agent_role_judge,
        candidate_count,
        reuse_rate,
        layout_shuffled,
        risk_score,
        use_dense_prefill,
    )


def prove_inv15() -> dict:
    """Prove INV-15 under the canonical antecedent.

    Antecedent: agent_role is judge-class AND candidate_count >= 9
                AND reuse_rate == 0 AND layout_shuffled == TRUE.
    Conclusion: use_dense_prefill == TRUE.

    Returns
    -------
    dict with keys: status ("PROVED" | "COUNTEREXAMPLE" | "UNKNOWN"),
    theorem, model (str or None), elapsed_ms, z3_version.
    """
    import time

    t0 = time.perf_counter()

    solver = z3.Solver()
    (
        agent_role_judge,
        candidate_count,
        reuse_rate,
        layout_shuffled,
        _risk_score,
        use_dense_prefill,
    ) = build_inv15_constraints(solver)

    # Antecedent.
    solver.add(agent_role_judge == True)  # noqa: E712 (Z3 needs ==)
    solver.add(candidate_count >= 9)
    solver.add(reuse_rate == 0.0)
    solver.add(layout_shuffled == True)  # noqa: E712

    # Refute the conclusion: assert dense_prefill = FALSE (a violation).
    solver.add(use_dense_prefill == False)  # noqa: E712

    result = solver.check()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if result == z3.unsat:
        return {
            "status": "PROVED",
            "theorem": "INV-15-DENSE-PREFILL",
            "model": None,
            "elapsed_ms": round(elapsed_ms, 2),
            "z3_version": z3.get_version_string(),
        }
    if result == z3.sat:
        return {
            "status": "COUNTEREXAMPLE",
            "theorem": "INV-15-DENSE-PREFILL",
            "model": str(solver.model()),
            "elapsed_ms": round(elapsed_ms, 2),
            "z3_version": z3.get_version_string(),
        }
    return {
        "status": "UNKNOWN",
        "theorem": "INV-15-DENSE-PREFILL",
        "model": None,
        "elapsed_ms": round(elapsed_ms, 2),
        "z3_version": z3.get_version_string(),
    }


def main() -> None:
    import json

    result = prove_inv15()
    print(json.dumps(result, indent=2))
    if result["status"] == "PROVED":
        print("\nINV-15 is formally valid (Z3 UNSAT on negation)")
        sys.exit(0)
    if result["status"] == "COUNTEREXAMPLE":
        print("\nINV-15 has a counterexample:", file=sys.stderr)
        print(result["model"], file=sys.stderr)
        sys.exit(2)
    print("\nZ3 returned UNKNOWN", file=sys.stderr)
    sys.exit(3)


if __name__ == "__main__":
    main()
