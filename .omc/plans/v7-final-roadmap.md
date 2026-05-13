# Apohara ContextForge — Post V7.0.0-rc.1 Formal Roadmap

**Status:** `active` (created 2026-05-13 after V7.0.0-rc.1 tag + main merge)
**Author:** Pablo M. Suarez (solo dev) + Claude Opus 4.7 (AI co-pilot)
**Horizon:** 12 months from V7.0.0-rc.1 (May 2026 → May 2027)
**Predecessor:** [`v7-roadmap.md`](v7-roadmap.md) — the original V7 strategic plan (now executed in 3 days)
**Canonical repo:** [github.com/SuarezPM/Apohara_Context_Forge](https://github.com/SuarezPM/Apohara_Context_Forge)
**Current tag:** `v7.0.0-rc.1` ([`/.release/v7.0.0-rc.1.txt`](../../.release/v7.0.0-rc.1.txt))

---

## TL;DR — Where we are, where we go

V6.0 (May 10, 2026 hackathon submission) → V7.0.0-rc.1 (May 13, 2026) in 3 days:
- 7 alpha releases + 1 rc + main merge + tag pushed
- All 10 AUDIT items 🟢 closed
- Real MI300X validation (3.55× reduction across 4K-262K context)
- Paper v2.0 with MI300X-measured numbers + 4 figures
- ~$2.05 of $30 AMD AI Dev Cloud budget consumed (saved ~$28 for V8+)

**Next 12 months:** convert V7.0.0-rc.1 into V7.0.0 final via publication + community + production deployment.

---

## Phase A — Immediate (Week 1: May 13-20, 2026)

### A.1 Local unification (USER action, 30 min)
- [ ] Salvage check: verify A has B's HF Spaces YAML fixes (`grep "sdk: gradio" hf_spaces/README.md`)
- [ ] Copy C's hackathon docs to `docs/legacy/hackathon/` (slides, video_script, M001 milestones)
- [ ] tar.gz B + C to `~/archived-apohara-clones/` (safety backups, exclude .venv)
- [ ] `rm -rf` B and C — free 17 GB local disk
- [ ] Cleanup pycache pollution in A (`git rm -rf --cached **/__pycache__/`)
- **Owner:** User
- **Acceptance:** `du -sh /home/linconx/{Apohara-ContextForge,CONTEXTFORGE}` returns 2× "no such directory"

### A.2 Paper v2.0 compile + verify (USER action, 30 min)
- [ ] Install tectonic: `cargo install tectonic` (or via apt)
- [ ] Compile: `tectonic paper/inv15_paper.tex` → produce `paper/inv15_paper.pdf` v2.0
- [ ] Read it through end-to-end (especially new §3, §6, §7)
- [ ] If any LaTeX errors: open issue, fix in V7.0.0-rc.2 patch
- **Owner:** User
- **Acceptance:** PDF compiles, all 4 figures render, table formatting OK

### A.3 techRxiv submission (USER action, 1-2 hours)
- [ ] Account creation at techrxiv.org (free, 5 min)
- [ ] Upload `paper/inv15_paper.pdf`
- [ ] Category: Computer Science → Systems & Performance
- [ ] Comments field: cross-reference Zenodo DOI + GitHub repo + V7.0.0-rc.1 tag
- [ ] License: CC BY 4.0
- [ ] Submit → wait ~3-5 days for moderation
- **Owner:** User
- **Acceptance:** techRxiv DOI assigned, paper publicly accessible

### A.4 ResearchGate parallel upload (USER action, 30 min)
- [ ] Login to existing account
- [ ] Add publication → upload `paper/inv15_paper.pdf`
- [ ] Cross-reference Zenodo DOI + techRxiv DOI + GitHub repo
- [ ] Request DOI on ResearchGate
- **Owner:** User
- **Acceptance:** Paper publicly listed on Pablo's RG profile

### A.5 Zenodo deposit refresh (USER action, 30 min)
- [ ] Login to Zenodo (concept DOI 10.5281/zenodo.20114594)
- [ ] Create new version → upload V7.0.0-rc.1 + paper v2.0 PDF
- [ ] Update metadata: "V7.0.0-rc.1 with MI300X-measured paper v2.0"
- [ ] Add `.zenodo.json` artifacts at repo root for auto-deposit on future tags
- [ ] Cross-reference techRxiv DOI + GitHub tag URL
- **Owner:** User
- **Acceptance:** Zenodo version 2 published, badge in README updates

---

## Phase B — Discovery / Outreach (Week 2-3: May 20 - June 3, 2026)

### B.1 Hacker News + Reddit post (USER action, 1 day prep + post)
- [ ] Draft a HN post: "Show HN: Apohara ContextForge — KV-cache reuse for multi-agent LLM pipelines, measured 3.55× reduction on real AMD MI300X"
- [ ] Link to repo + paper PDF + key findings (use_fwht=False discovery, HBM3 measurement)
- [ ] Reddit posts: r/LocalLLaMA + r/MachineLearning + r/AMD
- [ ] Twitter/X thread with 4-5 screenshots of figures
- **Owner:** User
- **Acceptance:** HN front page (top 30) OR >50 upvotes on Reddit OR >50 retweets

### B.2 AMD Developer Relations outreach (USER action, 1 day)
- [ ] Email amd-devrel + AMD AI Cloud team
- [ ] Subject: "Apohara ContextForge — $2 of MI300X credits → real paper v2.0 data + 4 figures"
- [ ] Body: link to repo, summary of measurements (HBM3 3.73 TB/s, 262K context validity), ask for case study / featured project
- **Owner:** User (Pablo, suarezpm@csnat.unt.edu.ar)
- **Acceptance:** AMD response within 2 weeks (any form — interest, polite no, partnership offer)

### B.3 Academic citation seeding (USER action, ongoing)
- [ ] Identify 5-10 related papers (RotateKV, TokenDance, KVCOMM, LMCache, vLLM) — DM authors with "we measured your literature claim on real hardware, here's the data"
- [ ] Post on relevant mailing lists / Discord channels (vLLM-discuss, ROCm-dev)
- [ ] Submit to MLSys 2027 newsletter / workshop CFP
- **Owner:** User
- **Acceptance:** ≥3 academic conversations initiated within 4 weeks

### B.4 HuggingFace Space refresh (USER + AI action, 4 hours)
- [ ] Update demo with V7.0.0-rc.1 metrics
- [ ] Add live "reduction_factor" plot using sweep data
- [ ] Add "MI300X-measured" badge prominently
- **Owner:** User triggers, AI assists
- **Acceptance:** HF Space at https://huggingface.co/spaces/SuarezPM/apohara-contextforge reflects V7.0.0-rc.1

---

## Phase C — V7.0.0 final + Sprint 5 (Month 2-3: June - July 2026)

### C.1 V7.0.0 final tag (after publications confirmed)
- [ ] Once techRxiv DOI assigned: update README badge with arXiv-equivalent badge
- [ ] Update paper PDF with techRxiv DOI in metadata
- [ ] Tag `v7.0.0` (final, drop `-rc.1`)
- [ ] GitHub Release with full CHANGELOG.md
- [ ] Zenodo deposit refresh with V7.0.0 final
- **Owner:** AI executes after user approval
- **Acceptance:** `git tag v7.0.0` exists, Release on GitHub, Zenodo version 3

### C.2 Sprint 5: LMCache v1 engine wiring (AI execution, ~$1-2 MI300X)
- [ ] Wire `LMCacheEngineBuilder.get_or_create` extra args (metadata, gpu_connector, broadcast_fn)
- [ ] May require importing from vLLM or implementing thin shims
- [ ] Test on AMD droplet (~$1-2 of remaining $27.95)
- [ ] V7.1.0 release on success
- **Owner:** AI executes via /autopilot, user MI300X budget
- **Acceptance:** `LMCacheConnectorV2.is_active()` returns True on AMD droplet

### C.3 Sprint 5: vLLM end-to-end smoke (AI execution, ~$3-5 MI300X)
- [ ] Install vLLM 0.17.1 on droplet
- [ ] Load TinyLlama-1.1B or similar small model
- [ ] Run 5-agent ContextForge pipeline with REAL inference
- [ ] Measure JCR (Judge Consistency Rate) with use_fwht=False INT4 codec
- [ ] Capture timings: prefill, decode, KV-cache fetch latency from LMCache
- **Owner:** AI executes via /autopilot
- **Acceptance:** `logs/mi300x_vllm_e2e_*.json` with real LLM inference metrics

### C.4 Vectorize remaining hot paths (AI execution, no MI300X needed)
- [ ] Profile current rotate_kv with cProfile, find next-biggest Python loop
- [ ] Vectorize using same pattern as Sprint 4 (`_quantize_block` → numpy broadcast)
- [ ] Bit-identical guarantee against current implementation
- [ ] V7.1.1 patch release
- **Owner:** AI via /team
- **Acceptance:** Profile shows ≥2× speedup on hot path

---

## Phase D — V8 codec rewrite (Month 4-6: August - October 2026)

### D.1 Per-nibble independent scales codec (large work)
- [ ] Design: each nibble in a packed byte gets its own (scale, zero_point)
- [ ] Trade-off explicit: doubles metadata storage but reclaims FWHT benefit
- [ ] Implementation: new `RotateKVQuantizerV8` class, side-by-side with current
- [ ] Hypothesis to test: with per-nibble scales, use_fwht=True should reclaim quality + literature 3.97× target
- **Owner:** AI via /autopilot
- **Acceptance:** MI300X measurement shows ≥3.7× reduction with use_fwht=True under V8 codec

### D.2 Real adversarial benchmark on GPU (depends on D.1)
- [ ] Port V6.2 adversarial bench from CPU-only to torch-CUDA
- [ ] Run M/G/1 queueing on real MI300X with real GPU-resident KV cache
- [ ] Use Apohara V8 codec to validate end-to-end
- [ ] Update paper to V2.1 with GPU-adversarial numbers
- **Owner:** AI execution, user budget
- **Acceptance:** `logs/mi300x_v62_adversarial_gpu_*.json` with real GPU timings

### D.3 Plugin marketplace SDK (CONDITIONAL — only if plugin count >5)
- [ ] Trigger: 5+ third-party plugins exist
- [ ] SDK design: scaffolding CLI (`apohara new-plugin`), ABI spec, signing
- [ ] **Status: DEFERRED** until plugin ecosystem grows
- **Owner:** Conditional / TBD

### D.4 AMD AI Cloud reference deployment (CONDITIONAL — depends on B.2)
- [ ] Trigger: AMD partnership confirmed via outreach
- [ ] Deploy K8s operator to AMD AI Cloud, full multi-node demo
- [ ] Case study / blog post co-authored with AMD DevRel
- **Owner:** Conditional / TBD

---

## Phase E — Long-term (Month 7-12: November 2026 - May 2027)

### E.1 Conference paper submission
- [ ] Target venues (in priority order):
  - **MLSys 2027** (abstract Dec 2026, paper Feb 2027)
  - **OSDI 2027** (April 2027 deadline)
  - **ASPLOS 2027** (October 2026 deadline — TIGHT)
- [ ] Reuse paper v2.0 substrate, expand evaluation with V8 codec + vLLM e2e + AMD partnership numbers
- **Owner:** User + AI editorial assist

### E.2 arXiv submission (after endorsements accumulate)
- [ ] By month 6-9 we should have 3-5 citations in techRxiv / ResearchGate / Hacker News
- [ ] Reach out to one cited paper's author for arXiv endorsement
- [ ] Submit to cs.PF + cs.LG with V8 codec data
- **Owner:** User
- **Acceptance:** arXiv ID assigned

### E.3 Community + ecosystem
- [ ] Sprint 4 PR: contributor flow validated (CONTRIBUTING + DCO + PR template were shipped in V7.0.0-alpha.1)
- [ ] Aim: 3-5 external contributors by V8 release
- [ ] Discord / GitHub Discussions enablement
- **Owner:** User governance, AI helps with issue triage

### E.4 Talks + outreach
- [ ] PyTorch ROCm meetup talk submission (Q4 2026)
- [ ] MLOps community talk (KubeCon AI track 2027)
- [ ] AMD Developer Conference 2027 if AMD partnership materializes
- **Owner:** User

---

## Decision points (USER must answer before AI executes Phase C+)

1. **Local unification approach** — Option α (clean delete after backup), Option β (cherry-pick B's 8 commits first), or Option γ (just tar.gz both, keep on disk forever)?
2. **Publication venue priority** — techRxiv first, ResearchGate first, or both in parallel?
3. **AMD outreach tone** — solo-dev independent researcher angle, or partnership-seeking angle?
4. **V8 codec timing** — start D.1 immediately after V7.0.0 final (aggressive), or wait for vLLM e2e results from C.3 first (cautious)?
5. **Sprint 5 budget** — burn the remaining $27.95 AMD credits across C.2 + C.3 + D.2 (high evidence), or hold reserve for D.2 only (conservative)?

---

## Risk register

| Risk | Mitigation |
|------|-----------|
| techRxiv moderation rejects (low probability) | Resubmit with cover letter, or fall back to ResearchGate-only |
| No HN traction | Pivot to Lobsters / r/MachineLearning / Twitter thread |
| AMD partnership outreach goes silent | Continue without partnership; V7.0.0 still ships |
| LMCache v1 wire-up requires vLLM context we don't have standalone | Document as known limitation, mark V6.x #3 LMCacheConnectorV2 as "vLLM-integrated only" |
| V8 codec rewrite breaks compatibility | Side-by-side: keep V7 codec as default, V8 opt-in via config |
| User burnout on solo dev | Pause Phase D/E if needed; V7.0.0 is already a complete release |

---

## Cost projection

| Phase | AMD credit cost | User time |
|-------|----------------|-----------|
| Phase A | $0 | ~3-4 hours |
| Phase B | $0 | ~1-2 days |
| Phase C | ~$5-7 | ~1 month |
| Phase D | ~$10-15 | ~3 months |
| Phase E | ~$5-10 | ~6 months |
| **Total** | **~$20-32 / $27.95 budget** | **~10 months** |

If AMD partnership confirmed in Phase B, additional credits likely available — Phase D + E unconstrained.

---

## Living document discipline

This roadmap follows V6.1 honesty: only items with concrete acceptance criteria are tracked. Speculative ideas live in `.release/sprint4-followups.md` (V8 nice-to-haves) until they crystallize into concrete acceptance criteria.

Update this file:
- When a phase completes (mark ✅ + commit-hash)
- When a decision point answer arrives from user
- When a risk materializes (move to "Realized" subsection)
- Monthly review: Aug 2026, Nov 2026, Feb 2027, May 2027

Maintained by: Pablo M. Suarez + Claude Opus 4.7 (AI co-pilot).
