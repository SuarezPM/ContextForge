# INV-15 Paper V2.0 — Preprint Draft

> **Status disclaimer.** This is a **preprint draft** committed during
> the 2026-05-16 Apohara Inti sprint (US-013). **Real arXiv submission
> requires the endorsement chain (2--3 days minimum) and is scheduled
> post-hackathon.** The version of record for citation today remains the
> Zenodo deposit:
> [DOI 10.5281/zenodo.20114594](https://doi.org/10.5281/zenodo.20114594).

---

## What this directory contains

| File | Purpose |
|------|---------|
| `inv15_v2.tex` | Main LaTeX source (1280+ lines, V2.0). Self-contained. |
| `inv15_v2.pdf` | Pre-built PDF, 13 pages, ~416 KiB. **Open this to read.** |
| `references.bib` | 23 BibTeX entries (17 carried over from V2.0.1, 6 new for V2.0). |
| `figures/` | Four PNG figures carried over from V2.0.1. |
| `README.md` | This file. |

---

## What V2.0 adds over V2.0.1 (`paper/inv15_paper.pdf`)

1. **Adjacent attack surfaces** (Section 2.4 *Adjacent Attack Surfaces*):
   NDSS 2025 KV-cache timing side-channel, KV-Cloak rotation defense,
   Adversa AI red-team toolchain, AMD vLLM-ATOM official launch
   (May 2026). Frames INV-15 against the broader cache-layer / content-layer
   threat surface.
2. **Sister-stack judge-defense validation** (new Section *Sister-Stack
   Judge-Defense Validation*, post-Conclusion): JailbreakBench
   `93.75%, Wilson 95% CI [86.2%, 97.3%], n=80` and HarmBench
   `77.50%, Wilson 95% CI [62.5%, 87.7%], n=40` from the Apohara Aegis
   sister repository. Grounds INV-15's conservatism-favors-safety
   philosophy against community-standard benchmarks.
3. **FallbackVendorAdapter sketch** (new Section *Vendor-Fallback
   Architecture*): decouples gate logic from a single LLM vendor;
   sketches a three-tier defense (INV-15 cache invariant +
   KV-Cloak side-channel + vendor fallback).
4. **Appendix A**: Reference-implementation pointer to
   `apohara_context_forge/safety/jcr_gate.py` with the coefficient
   mapping between Eq. 1 of the paper and the runtime Python constants.
5. **6 new BibTeX entries**: NDSS 2025 KV-cache leak, KV-Cloak, Adversa
   AI, AMD vLLM-ATOM launch, JailbreakBench (NeurIPS 2024 D&B),
   HarmBench (NeurIPS 2024 D&B).

---

## Build

### Tectonic (single-pass, recommended)

```bash
cd papers
tectonic inv15_v2.tex
```

Tested against tectonic 0.15+ on Linux. Output: `inv15_v2.pdf` (13 pages).

### pdfLaTeX + bibtex (alternative)

```bash
cd papers
pdflatex inv15_v2.tex
bibtex   inv15_v2
pdflatex inv15_v2.tex
pdflatex inv15_v2.tex
```

---

## Reproducibility

Every measurement claim in this draft is reproducible from committed
artifacts:

| Claim                              | Source                                     |
|------------------------------------|--------------------------------------------|
| 0 INV-15 violations / 1,210 sweep  | `demo/benchmark_v5.py` scenario S-15       |
| Critic dense prefill rate 0.851    | Same scenario, `jcr_critic_dense_rate`     |
| 10.81x TokenDance compression      | `demo/benchmark_v5.py` scenario S-14       |
| 3.55x INT4 reduction factor        | `scripts/mi300x_vram_measurement.py`       |
| 3.73 TB/s HBM3 triad bandwidth     | MI300X benchmark log in `logs/`            |
| 310/310 unit tests                 | `PYTHONPATH=. pytest tests/ -q`            |
| JBB 93.75% (n=80)                  | `apohara-aegis` repo, day-5 fallback log   |
| HarmBench 77.50% (n=40)            | `apohara-aegis` repo, day-6 subset log     |

Hardware: AMD Instinct MI300X (192 GB HBM3, ROCm 7.2.0) on AMD DevCloud
ATL1. Hardware label is `rocm-hip:6.2.41133:AMD Instinct MI300X VF`
(see CLAUDE.md §6 honesty discipline; **not** `cuda`).

---

## Plan for real arXiv submission (post-hackathon)

1. Secure endorsement for category `cs.LG` (or `cs.PF`) via an existing
   arXiv author in the network (2-3 days).
2. Final pre-submission pass: tighten captions, expand related-work
   subsection on AMD vLLM-ATOM once the official launch RFC is final,
   re-run honesty CI guard against the manuscript.
3. Bundle `inv15_v2.tex`, `references.bib`, and the generated
   `inv15_v2.bbl` into a single `.tar.gz`.
4. Submit at <https://arxiv.org/submit/>. License: CC-BY 4.0
   (Apache-2.0 compatible).
5. After acceptance, mirror the arXiv ID into:
   - `paper/README.md` (V2.0.1 README) — add the arXiv badge
   - root `README.md` paper-badge row
   - this file — update the disclaimer banner above

---

## Contact

Pablo M. Suarez --- `suarezpm@csnat.unt.edu.ar` (academic) ---
[@SuarezPM](https://github.com/SuarezPM) ---
[DOI 10.5281/zenodo.20114594](https://doi.org/10.5281/zenodo.20114594)
