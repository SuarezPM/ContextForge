# Paper Changelog — INV-15: A Formal Safety Invariant for KV-Cache Reuse

Source: `paper/inv15_paper.tex`. PDF artifact: `paper/inv15_paper.pdf`.
DOI v2 predecessor: [10.5281/zenodo.20114594](https://doi.org/10.5281/zenodo.20114594).

---

## v3.0 — 2026-05-18

### Added

- **Section: Formal Verification via Z3 SMT** (new
  `\section{Formal Verification via Z3 SMT}`, label `sec:z3-proof`).
  Encodes `JCRSafetyGate.gate_decision` as an SMT problem and proves
  INV-15 over the entire modeled domain.
  - Theorem `INV-15-DENSE-PREFILL`: critic + n>=9 + reuse=0 +
    shuffled ==> use_dense_prefill = true.
  - Proof by refutation: Z3 returns `unsat` in $10.08 \pm 0.5$ ms on
    z3-solver `4.16.0`.
  - Complementarity table contrasting the empirical sweep (1,210
    samples) with the SMT proof (entire domain).
  - Limitations subsection: linear-arithmetic-only model, FWHT
    lookup approximated, IEEE-754 vs reals.
- **Citation** `demoura2008z3` for the Z3 paper (TACAS 2008) in
  `paper/references.bib`.
- **Abstract sentence** noting v3.0 extends the empirical sweep with
  the Z3 SMT formal proof.
- **Introduction roadmap** mentions the new
  Section~\ref{sec:z3-proof}.

### Changed

- Title-page date: `May 13, 2026 -- V2.0.1 ...` ->
  `May 18, 2026 -- v3.0 with Z3 SMT formal proof (extends V2.0.1
  MI300X measurements)`.

### Reproduce

```bash
# Z3 proof (PROVED, 10.08 ms):
PYTHONPATH=. python3 -m apohara_context_forge.safety.z3_inv15_proof

# Pytest regression (3 tests):
PYTHONPATH=. python3 -m pytest tests/test_z3_inv15.py -v

# Rebuild the PDF (tectonic 0.15+):
cd paper && tectonic inv15_paper.tex
```

### Zenodo

Metadata draft: `paper/zenodo-v3-metadata.json`. New version of
[10.5281/zenodo.20114594](https://doi.org/10.5281/zenodo.20114594);
manual deposit via Zenodo web UI is Pablo's step.

---

## v2.0.1 — 2026-05-13 (predecessor; DOI 10.5281/zenodo.20114594)

- MI300X-measured numbers replace literature values (3.97x ->
  measured 3.55x reduction; 3.73 TB/s HBM3 measured bandwidth).
- 12 references with real authors from arXiv abstract pages.
- 1,210-point Cartesian sweep (5 roles x 11 cand x 11 reuse x 2
  shuffle) reports 0 violations.

## v2.0 — 2026-05-12

- First MI300X-grounded release. Replaces V1 vLLM-V0 baseline with
  vLLM-V1 ATOM plugin numbers.

## v1.0 — 2026-05-04

- Initial public deposit. Closed-form risk model + JCRSafetyGate
  description + simulator-only results (no MI300X measurements).
