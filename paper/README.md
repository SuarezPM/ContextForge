# INV-15 paper — submission package

Camera-ready LaTeX source for:

> **INV-15: A Formal Safety Invariant for KV-Cache Reuse in
> Multi-Agent Judge Pipelines**
> Pablo M. Suarez · APOHARA · ContextForge · May 2026

Target venues:
- arXiv (`cs.LG`, `cs.DC`) — primary, submit this week.
- MLSys 2027 — full conference submission.
- NeurIPS 2026 Workshop on Efficient LLM Inference — workshop track.

---

## Files

| File | Purpose |
|------|---------|
| `inv15_paper.tex`   | Main LaTeX source. Two-column, 10pt, A4. Self-contained. |
| `references.bib`    | All 10 citations in BibTeX (IEEEtran style). |
| `README.md`         | This file. |

---

## Build

### Overleaf (recommended)

1. Create a new project at <https://www.overleaf.com/>.
2. Upload `inv15_paper.tex` and `references.bib`.
3. Set the main document to `inv15_paper.tex`.
4. Set the compiler to **pdfLaTeX**.
5. Click *Recompile*.

### Local (TeX Live ≥ 2023)

```bash
cd paper
pdflatex inv15_paper.tex
bibtex   inv15_paper
pdflatex inv15_paper.tex
pdflatex inv15_paper.tex
```

Output: `inv15_paper.pdf` (≈ 8–10 pages, two-column).

### Local (Tectonic, single-pass)

```bash
cd paper
tectonic inv15_paper.tex
```

---

## arXiv submission checklist

- [ ] Compile with pdfLaTeX (not LuaLaTeX) — arXiv prefers it.
- [ ] Bundle `inv15_paper.tex`, `references.bib`, and the generated
      `inv15_paper.bbl` in a single `.tar.gz`.
- [ ] Primary category: `cs.LG`; cross-list: `cs.DC`.
- [ ] License: arXiv perpetual non-exclusive (CC-BY 4.0 acceptable).
- [ ] DOI: leave blank — arXiv assigns one on accept.

---

## Reproducibility pointers

Every numeric claim in the paper is reproducible from this repository:

| Claim in paper                              | Source                                   |
|---------------------------------------------|------------------------------------------|
| 0 INV-15 violations / 9 decisions           | `demo/benchmark_v5.py` · scenario S-15   |
| Critic dense rate = 1.000                   | Same · `jcr_critic_dense_rate`           |
| Gate throughput = 1,800 decisions/s         | Same · `throughput_tps`                  |
| 10.81× TokenDance compression               | `demo/benchmark_v5.py` · scenario S-14   |
| Reconstruction error ≤ 1e-4                 | Same · `reconstruction_max_err`          |
| 15 / 15 benchmark scenarios                 | `logs/benchmark_v6_final.txt`            |
| 310 / 310 unit tests                        | `pytest tests/ -q`                       |
| 79.85 % live token savings (5-agent demo)   | `demo/app.py` · `/run_with_contextforge` |

Hardware: AMD Instinct MI300X · 192 GB HBM3 · ROCm 7.x · AMD DevCloud
ATL1 · 2026-05-10.

---

## Contact

Pablo M. Suarez — `p.ms.08@hotmail.com` — [@SuarezPM](https://github.com/SuarezPM)
