# Zenodo deposit refresh — V7.0.0-rc.2

**Status:** Ready to refresh. Existing concept DOI: 10.5281/zenodo.20114594

This is a **new version** under the same concept DOI (existing V6.1 deposit
remains accessible but is now superseded). Zenodo will assign a fresh
version-specific DOI for V7.0.0-rc.2.

---

## Step-by-step refresh procedure

### 1. Login to Zenodo
- Go to https://zenodo.org/login
- Use your existing account (the one that owns DOI 10.5281/zenodo.20114594)

### 2. Find the existing deposit
- Navigate to https://zenodo.org/uploads (or "My uploads")
- Search for "Apohara" or "ContextForge"
- Open the existing record (DOI 10.5281/zenodo.20114594)

### 3. Create a new version
- Click **"New version"** button (top-right of the record page)
- Zenodo will create a draft pre-populated with V6.1 / V7.0.0-rc.1 metadata
- You'll get a new draft DOI — keep it open

### 4. Update metadata fields
Replace the existing content with the V7.0.0-rc.2 values from this repo:

**Title:** (keep existing)
```
Apohara ContextForge: KV-cache coordination for multi-agent LLM pipelines on AMD MI300X
```

**Version:** Update to
```
7.0.0-rc.2
```

**Description:** Paste from `.zenodo.json` `description` field:
```
Apohara ContextForge V7.0.0-rc.2: paper v2.0.1 with 7 new citations +
compiled PDF, on top of V7.0.0-rc.1 (Sprint 1-4 + Wave B MI300X
validation). Real KV-cache reuse end-to-end. Hardware-validated on AMD
Instinct MI300X (192 GB HBM3, ROCm 7.2.0, torch 2.5.1+rocm6.2). Measured
3.55x VRAM reduction constant across context lengths 4K-262K (64x scale
span), correcting the literature 3.97x claim for our per-byte
joint-quantization codec. Measured HBM3 effective bandwidth 3.73 TB/s
(70.5% of advertised 5.3 TB/s peak under SR-IOV). Paper v2.0.1 includes
12 references with full author info (5 fixed + 7 new). See AUDIT.md for
honesty discipline log (10/10 audit items closed) and
paper/inv15_paper.pdf (409 KB).
```

**Keywords:** Replace with the expanded set (15 keywords):
```
kv-cache, llm-inference, multi-agent-llm, amd-mi300x, rocm,
int4-quantization, rotatekv, fwht, fast-walsh-hadamard-transform,
hbm3-bandwidth, jcr, judge-consistency, safety-invariants, vllm, lmcache
```

**License:** MIT (unchanged)

**Upload type:** Software (unchanged)

### 5. Add related identifiers
Click "Add another related identifier" twice:

**Identifier 1:**
- Relation: "is supplement to"
- Identifier: `https://github.com/SuarezPM/Apohara_Context_Forge/releases/tag/v7.0.0-rc.2`
- Scheme: URL

**Identifier 2 (when techRxiv DOI arrives — placeholder for now):**
- Relation: "is identical to" or "is supplemented by"
- Identifier: `10.36227/techrxiv.XXXXXXXX` (replace XXXXXXXX with the
  techRxiv-assigned DOI; this can be left empty for now and added
  in a future version revision)
- Scheme: DOI

### 6. Upload files
Replace the existing files (or keep + add):

**MUST include:**
- `paper/inv15_paper.pdf` (409 KB) — the v2.0.1 paper
- `paper/references.bib` — clean BibTeX with 12 entries
- `AUDIT.md` — V6.1 honesty discipline log (10 items closed)
- `CHANGELOG.md` — V7.0.0-rc.1 through V7.0.0-rc.2 entries

**SHOULD include (replication evidence):**
- `logs/mi300x_*.json` (13 measurement logs from Wave B)
- `paper/figures/fig5_reduction_factor_vs_seq.png`
- `paper/figures/fig7_pure_torch_fwht.png`
- `paper/figures/fig8_quant_quality.png`
- `paper/figures/fig9_hbm3_bandwidth.png`

**MAY include:**
- `.release/v7.0.0-rc.1.txt` + `.release/v7.0.0-rc.2.txt` (tag annotations)
- `scripts/mi300x_*.py` + `scripts/mi300x_*.sh` (10 measurement scripts)

For convenience, you can either:
- Upload individually (drag-and-drop in Zenodo)
- Or upload a single tar.gz of all artifacts:
  ```bash
  cd /home/linconx/Documentos/Apohara_Context_Forge
  tar -czf /tmp/apohara-v7.0.0-rc.2-artifacts.tar.gz \
    paper/inv15_paper.pdf paper/references.bib \
    AUDIT.md CHANGELOG.md \
    logs/mi300x_*.json paper/figures/*.png \
    .release/*.md .release/*.txt \
    scripts/mi300x_*.py scripts/mi300x_*.sh
  ```
  Then upload that single file.

### 7. Publish
- Click **"Publish"** button
- Zenodo assigns the new version DOI (e.g., 10.5281/zenodo.XXXXXXXX)
- The concept DOI (10.5281/zenodo.20114594) automatically points to the
  latest version

### 8. After publication
- Copy the new version DOI
- Update README.md badge:
  ```markdown
  [![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.XXXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXXX)
  ```
- Commit + push the badge update
- Once techRxiv DOI arrives: create ANOTHER Zenodo version with both
  cross-references

---

## Estimated time

- Login + navigate: 2 min
- Update metadata: 5 min
- Upload artifacts (single tar.gz): 2 min
- Publish + DOI assignment: 1 min
- **Total: ~10 min**

## When NOT to do this

Wait for the techRxiv DOI ONLY if you prefer a single Zenodo version with
both cross-references. The drawback: 3-5 day delay before refresh.

Recommended: do BOTH refreshes (one now with V7.0.0-rc.2, one later with
techRxiv DOI cross-link). Each gets its own version DOI, the concept DOI
always points to the most recent. This documents the timeline of citations
explicitly, which is good academic hygiene.
