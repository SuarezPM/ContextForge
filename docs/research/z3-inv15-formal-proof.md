# Z3 SMT Formal Proof of INV-15

## Statement

**INV-15-DENSE-PREFILL**: For all inputs `(agent_role, candidate_count,
reuse_rate, layout_shuffled)`, if
`agent_role == "critic" AND candidate_count >= 9 AND reuse_rate == 0
AND layout_shuffled == TRUE`,
then the JCR safety gate mandates `use_dense_prefill == TRUE`.

This is the formal complement to the empirical sweep over
`JCRSafetyGate.gate_decision()`. The empirical sweep samples discrete
inputs and reports counterexample counts; the SMT proof reasons over
the ENTIRE modeled domain (`candidate_count` ranging over all
non-negative integers; `reuse_rate` over all reals in `[0, 1]`).

## Z3 Model

Implementation: [`apohara_context_forge/safety/z3_inv15_proof.py`](../../apohara_context_forge/safety/z3_inv15_proof.py).

The risk model mirrors `jcr_gate.compute_jcr_risk()` constant-for-constant:

| Term                       | Z3 encoding                                            |
| -------------------------- | ------------------------------------------------------ |
| `base_risk`                | `If(role_critic, 0.6, 0.1)`                            |
| `+0.10 * (n - 2)` if `n>2` | `If(n>2, ToReal(n-2) * 0.10, 0.0)`                     |
| `+0.20` if shuffled        | `If(layout_shuffled, 0.20, 0.0)`                       |
| `+0.15` if reuse > 0.8     | `If(reuse_rate > 0.8, 0.15, 0.0)`                      |
| Clamp to `[0, 1]`          | nested `If` matching `max(0.0, min(1.0, risk))`        |
| INV-15 dense decision      | `And(role_critic, risk_score > threshold)` (strict `>`)|

Threshold defaults to `0.7` (matches `DEFAULT_JCR_THRESHOLD`).

## Theorem and Proof

We prove by refutation: assert the antecedent AND the negation of the
conclusion (`use_dense_prefill == FALSE`), then ask Z3 for SAT. Z3
returns **UNSAT**, meaning no assignment satisfies the violation. By
the law of the excluded middle on Z3's modeled domain, INV-15 holds.

Under the canonical antecedent the risk-score lower bound is
`0.6 (judge) + 0.10*(9-2) (n>=9) + 0.20 (shuffled) = 1.5`, clamped to
`1.0`, which exceeds `0.7`. The reuse-rate term contributes `0` because
`reuse_rate == 0 <= 0.8`. So `use_dense_prefill = (role_critic AND
risk_score > 0.7) = TRUE`. Z3 confirms no integer/real assignment can
make `use_dense_prefill = FALSE` while the antecedent holds.

## Complementarity with the Empirical Sweep

| Aspect              | Empirical sweep                  | Z3 SMT proof                  |
| ------------------- | -------------------------------- | ----------------------------- |
| Input coverage      | Discrete samples                 | Entire modeled domain         |
| Guarantee           | Statistical (0/N counterexamples)| Logical (UNSAT on negation)   |
| Cost                | O(N) gate calls                  | Single SMT check (<1 s)       |
| Domain assumptions  | Same as runtime                  | Linear-arithmetic abstraction |

Both layers are needed: the empirical layer catches modeling drift (if
`compute_jcr_risk` ever stops matching the SMT abstraction), and the
SMT layer rules out missed corners that random sampling would not hit.

## Limitations

- The Z3 model is **linear arithmetic over reals and integers**. The
  Python implementation uses IEEE-754 floats; rounding at the
  `> 0.8` and `> 0.7` boundaries is not modeled. For the canonical
  antecedent the lower bound is far from any boundary, so this does
  not affect the proof.
- Only the four-input axis `(role, n, reuse_rate, layout_shuffled)`
  is modeled. Future extensions (e.g. judge-role hierarchy, anchor
  hash, per-agent thresholds) require expanding `build_inv15_constraints`.
- The proof targets the dense-prefill decision only. Downstream
  effects (telemetry, observability hooks) are not modeled.

## Reproduce

```bash
pip install z3-solver
PYTHONPATH=. python -m apohara_context_forge.safety.z3_inv15_proof
PYTHONPATH=. python -m pytest tests/test_z3_inv15.py -v
```

Expected: JSON output with `"status": "PROVED"` and exit code `0`;
pytest reports `3 passed`.
