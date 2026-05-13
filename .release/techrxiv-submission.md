# techRxiv submission metadata — V7.0.0-rc.2 paper v2.0.1

**Status:** Ready to submit. Paper PDF at `paper/inv15_paper.pdf` (409 KB).

## Submission form fields

### Title
INV-15: A Formal Safety Invariant for KV-Cache Reuse in Multi-Agent Judge Pipelines

### Author(s)
Pablo M. Suarez (corresponding author)
Affiliation: Universidad Nacional de Tucumán, Facultad de Ciencias Naturales
Email: suarezpm@csnat.unt.edu.ar

### Abstract (~200 words)
```
Multi-agent LLM pipelines routinely reuse KV caches across agents to amortize
the dominant prefill cost. Recent work shows this reuse silently corrupts
judge agents, dropping Judge Consistency Rate (JCR) by 8-23% with no other
visible regression. We introduce INV-15, a formal safety invariant that
gates KV-cache reuse based on per-agent risk scoring, and ship it as a
production-ready KV-coordination layer (Apohara ContextForge) on AMD
Instinct MI300X (192 GB HBM3, ROCm 7.2.0) through the vLLM V1 ATOM plugin
interface. Across an exhaustive Cartesian sweep of input space, we observe
zero INV-15 violations and full compatibility with TokenDance master-mirror
compression. On real MI300X hardware we measure 3.55x VRAM reduction
constant across context lengths 4K-262K (a 64x scale span), correcting
the 3.97x literature claim of RotateKV for our per-byte joint-quantization
codec. Measured HBM3 effective bandwidth: 3.73 TB/s (70.5% of advertised
5.3 TB/s peak under SR-IOV). We additionally characterize FWHT runtime,
the FP16 vs FP32 upcast trade-off, and quantization quality degradation
under naive FWHT integration. All measurements, scripts, logs, and the
honesty discipline that maintains them are open at MIT license.
```

### Subject categories
- **Primary:** Computer Science → Performance (cs.PF)
- **Secondary 1:** Computer Science → Multiagent Systems (cs.MA)
- **Secondary 2:** Computer Science → Distributed Computing (cs.DC)

### Keywords (5-10)
- KV cache reuse
- multi-agent LLM
- AMD MI300X
- ROCm
- INT4 quantization
- Hadamard transform
- judge consistency
- safety invariants
- vLLM
- LMCache

### License
CC BY 4.0 (Creative Commons Attribution 4.0 International)

### Comments / cover letter
```
Paper v2.0.1 of the V7.0.0-rc.2 Apohara ContextForge release.

Code, measurements, scripts, and the AUDIT.md honesty log are
open-source MIT at https://github.com/SuarezPM/Apohara_Context_Forge
(tag v7.0.0-rc.1).

Replication artifacts:
- Real MI300X measurement logs: logs/mi300x_*.json (13 JSON files)
- Paper figures: paper/figures/fig{5,7,8,9}_*.png (140 dpi)
- Hardware: AMD Instinct MI300X VF (192 GB HBM3, gfx942, ROCm 7.2.0,
  torch 2.5.1+rocm6.2) via AMD AI Dev Cloud ATL1
- Cost: ~$2.05 of $30 AMD AI Dev Cloud credits for the full evidence stack

Cross-references:
- Zenodo concept DOI: 10.5281/zenodo.20114594
- GitHub release tag: v7.0.0-rc.1 (commit d81fe00)
- Live demo: https://huggingface.co/spaces/SuarezPM/apohara-contextforge

Submitter is a solo independent researcher; this is the first submission
to techRxiv. No conflicts of interest. All code and data are openly
available for replication.
```

### Acknowledgments (in paper)
Add to paper if not present:
"This work was supported by AMD AI Dev Cloud credits; the author thanks
the AMD Developer Hackathon program for hardware access. We acknowledge
the open-source LMCache, vLLM, and ROCm communities."

## Submission checklist

- [ ] Account created at https://www.techrxiv.org/
- [ ] PDF uploaded: `paper/inv15_paper.pdf`
- [ ] Title + abstract pasted from above
- [ ] Authors entered (single-author: Pablo M. Suarez)
- [ ] Affiliation: UNT FCN
- [ ] Subject categories: cs.PF primary, cs.MA + cs.DC secondary
- [ ] Keywords: from list above
- [ ] License: CC BY 4.0
- [ ] Comments / cover letter pasted
- [ ] Submit → wait 3-5 business days for moderation
- [ ] Once DOI assigned: update README badge + Zenodo deposit cross-link

## After acceptance

1. Update README.md badge with techRxiv DOI
2. Refresh Zenodo deposit with techRxiv DOI cross-link
3. Update `paper/inv15_paper.tex` arXiv-instructions block with the
   actual techRxiv DOI
4. Post on Hacker News / r/LocalLLaMA / r/MachineLearning with the
   techRxiv link (Phase B.1 of roadmap)
5. Email AMD Developer Relations with the published paper
   (Phase B.2 of roadmap)
