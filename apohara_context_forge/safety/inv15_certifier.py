"""Per-decision INV-15 certificate (Z3). Complements z3_inv15_proof.prove_inv15()
(which proves the general theorem) by certifying that ONE observed gate decision
matches what INV-15 mandates for its exact input point. Used by FORGE-LEDGER."""
from __future__ import annotations
import time
from apohara_context_forge.safety.z3_inv15_proof import build_inv15_constraints

def certify_decision(*, agent_role: str, candidate_count: int, reuse_rate: float,
                     layout_shuffled: bool, use_dense: bool) -> dict:
    import z3
    solver = z3.Solver()
    (agent_role_judge, candidate_count_v, reuse_rate_v, layout_shuffled_v,
     _risk, use_dense_v) = build_inv15_constraints(solver)
    role = (agent_role or "").lower()
    solver.add(agent_role_judge == (role in ("critic", "judge")))
    solver.add(candidate_count_v == int(candidate_count))
    solver.add(reuse_rate_v == z3.RealVal(str(reuse_rate)))
    solver.add(layout_shuffled_v == bool(layout_shuffled))
    t0 = time.perf_counter()
    solver.add(use_dense_v != bool(use_dense))   # refute: any model where invariant != observed?
    status = solver.check()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "satisfies_inv15": status == z3.unsat,   # unsat => observed == mandated
        "agent_role": role, "candidate_count": int(candidate_count),
        "reuse_rate": float(reuse_rate), "layout_shuffled": bool(layout_shuffled),
        "observed_use_dense": bool(use_dense),
        "z3_status": str(status), "elapsed_ms": round(elapsed_ms, 3),
        "z3_version": z3.get_version_string(),
    }
