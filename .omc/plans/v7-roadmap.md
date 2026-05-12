# Apohara ContextForge — V7 Roadmap

**Status:** `pending execution approval` (decisions locked, plan updated)
**Author:** Claude (under user direction — solo-dev Pablo M. Suarez)
**Date:** 2026-05-12
**Horizon:** 12 months (May 2026 → May 2027)
**Predecessor:** V6.2 (Poisson + M/G/1 adversarial benchmark, 6/6 PASS) + V6.x #3 (LMCacheConnectorV2)

---

## 0.0. Decisions resolved (2026-05-12 interview)

| # | Decision | User choice | Plan impact |
|---|----------|-------------|-------------|
| **D1** | arXiv submission timing | **WAIT for V7.0 paper v2.0** | B2 moves from Sprint 1 → V7.0 release milestone (Nov 2026). Paper updated with B1 truth-up data + adversarial benches B3. |
| **D2** | AMD partnership | **TBD / exploring** — but **$100 USD in AMD AI Dev Cloud credits available** | A4 stays conditional. **NEW Section 4.5 below** plans strategic burn of the $100 credit budget across sprints. |
| **D4** | Contribution policy | **PR-friendly: CONTRIBUTING + DCO + CoC** | Added to Sprint 1 deliverables (low effort, high optionality value). |
| **D5** | V7 thesis | **4 tracks parallel** (user override of substrate-first recommendation) | Sprint 1 redrawn as quad-track kickoff (B1a + A3 + A1 + CONTRIBUTING). **Triage protocol added in §4.4** — explicit cut order if capacity short. |

**Note on D5 override:** I recommended substrate-first. User chose 4 parallel. Accepted — but plan now includes triage protocol so the override is honest about its risk surface, not just permissive about it.

---

## 0. TL;DR — strategic recommendation

V6 was the **honest substrate** release. V7 must be the **real KV-cache reuse end-to-end** release.

Three parallel tracks, sequenced over the year:

| Track | Months | Theme | Key deliverables |
|-------|--------|-------|------------------|
| **A — Substrate truth-up** | 1-2 | Close remaining 🟠 audit items | Real FWHT, real encoder, real RoPE-derot, measured CLA |
| **B — Audit-grade telemetry** | 1-2 | "Anyone can verify V7 claims" | Prometheus + OTLP + JSONL audit log + Grafana JSON |
| **C — End-to-end deployment** | 3-6 | LMCache + K8s + MI300X cluster proof | Operator + helm + cross-worker hit demo |
| **D — Distribution** | 7-12 | Growth, partnerships, conf talks | Marketplace SDK *iff* plugin count > 5, AMD Cloud *iff* partnership |

**Recommendation:** Marketplace SDK and AMD AI Cloud reference deployment should **defer to V7.1+**. They amplify credibility but don't create it. Tracks A + B + C create it.

---

## 1. Per-candidate breakdown

Effort estimates assume **30-40 productive hrs/week** (solo dev, deep-work blocks of ~3 hrs each).

Leverage scale: HIGH (directly serves V7 thesis), MEDIUM (extends V7 reach), LOW (premature without prior traction).

### 1.A — Original 4 roadmap candidates

#### A1. K8s operator (`ApoharaContextForgeCluster` CRD)
- **Effort:** 4-6 weeks
- **Leverage:** MEDIUM → HIGH if multi-node is V7 thesis
- **Dependencies:** V6.x #3 LMCache connector ✓ done. operator-sdk OR kubebuilder learning curve.
- **Risks:** 
  - CRD versioning hell (v1alpha1 → v1beta1 churn)
  - kind-based test cluster GPU passthrough (probably won't work — need stub mode)
  - Helm chart vs raw manifests vs kustomize debate
- **DoD:**
  - `kubectl apply -f` creates an N-worker ContextForge cluster
  - operator reconciles missing pods / failed restarts
  - LMCache Redis sidecar provisioned
  - integration test on kind cluster (CPU stub mode)
  - helm chart published to `chartmuseum.apohara.dev` (or oci://ghcr.io/...)

#### A2. Plugin marketplace SDK
- **Effort:** 6-8 weeks
- **Leverage:** LOW (current plugin count: 1 — apohara-vllm-plugin itself)
- **Dependencies:** Need ≥5 third-party plugins to justify SDK abstraction
- **Risks:** Premature abstraction. SDK design depends on actual plugin variety, which doesn't exist yet.
- **Recommendation:** **DEFER to V8** unless a partner explicitly asks for it.
- **DoD (when triggered):** ABI spec, scaffolding CLI (`apohara new-plugin foo`), curated registry, plugin signing

#### A3. Audit-grade INV-15 telemetry export
- **Effort:** 1.5-2 weeks
- **Leverage:** **HIGH** — directly serves "anyone can verify V7 claims" thesis
- **Dependencies:** None (greenfield)
- **Risks:** OpenTelemetry semantic-conventions naming churn (low risk)
- **DoD:**
  - `prometheus_client` exporter on `:9090/metrics` exposing: `apohara_jcr_gate_decisions_total{action,agent}`, `apohara_inv15_risk_score`, `apohara_anchor_match_total`, `apohara_lmcache_hit_total`
  - OTLP gRPC exporter (optional, for Tempo/Jaeger trace correlation)
  - JSONL audit log: every gate decision → `{ts, agent_id, anchor_hash, risk_score, gate_action, predicted_jcr_delta, lmcache_consulted, lmcache_hit}`
  - Sample Grafana dashboard JSON committed under `dashboards/`
  - 8-12 unit tests + 1 integration test (scrape /metrics → parse → assert keys)

#### A4. AMD AI Cloud reference deployment
- **Effort:** 3-4 weeks
- **Leverage:** HIGH **iff** AMD partnership confirmed; ZERO otherwise
- **Dependencies:** A1 (K8s operator), AMD AI Cloud credits/access
- **Risks:** Partnership status is unknown to me — needs decision-point input from user
- **Recommendation:** Block until decision point 2 below is resolved

### 1.B — Items NOT on the original list but SHOULD be in V7

#### B1. Honest-stub upgrades (close remaining 🟠 audit items)
The V6.1 AUDIT.md flagged 4 modules where code exists but is stubbed. V7 closes them:

| Item | What's stubbed today | Effort | Leverage |
|------|---------------------|--------|----------|
| **B1a — RotateKV FWHT** | placeholder rotation (no real Fast Walsh-Hadamard Transform) | 2 wk | HIGH |
| **B1b — S-12 real encoder** | hashed/random embeddings, not CLIP/SigLIP | 1 wk | HIGH |
| **B1c — `anchor_pool` RoPE derotation** | math is implemented but never runs on real KV tensors in production path | 1.5 wk | HIGH |
| **B1d — `cla_metadata` measured CLA** | reports estimated, not measured VRAM reduction | 1 wk | HIGH |
| **Combined** | | **~5.5 wk** | **HIGH** |

These close the V6 promise. After B1, AUDIT.md table is fully 🟢.

#### B2. arXiv submission of paper v1.1
- **Effort:** 3 days (PDF resubmission + endorsement dance)
- **Leverage:** HIGH (academic indexing, citable, opens conference paths)
- **Dependencies:** Zenodo DOI ✓ deposited. Needs arXiv user ID + endorsement.
- **Risks:** arXiv moderation 1-2 weeks; ML systems category may need endorsement letter
- **DoD:** arXiv ID assigned, README badge updated, abstract page live

#### B3. Real adversarial benchmarks (TokenDance / JCR / RotateKV)
- **Effort:** 2-3 weeks
- **Leverage:** MEDIUM — extends V6.2 Poisson+M/G/1 discipline to other modules
- **Dependencies:** B1a, B1b, B1c (must be real before they can be adversarially tested)

#### B4. Real RAG / judge eval pipeline
- **Effort:** 2 weeks
- **Leverage:** MEDIUM — gives V7 a "real workload" story for the paper update
- **Dependencies:** External LLM API key (claude/gpt) for the judge passes
- **DoD:** `eval/rag_judge.py` runs 100-doc HotpotQA subset, reports JCR consistency with/without ContextForge, with confidence intervals

#### B5. CI hardening
- **Effort:** 1 week
- **Leverage:** MEDIUM — prevents truth-up regression
- **Tasks:** Python 3.11/3.12/3.13 matrix, GPU lane on self-hosted runner, plugin-install smoke test, honesty.yml extended with new patterns

#### B6. Documentation site (Astro / Starlight)
- **Effort:** 1 week
- **Leverage:** MEDIUM — discoverability
- **Dependencies:** None
- **DoD:** `apohara.dev` (or `contextforge.apohara.dev`) live, auto-builds from `docs/`, dark mode, search

---

## 2. 12-month sequencing (post-decisions, 4-tracks-parallel)

```
May       Jun       Jul       Aug       Sep       Oct       Nov       Dec       Jan       Feb       Mar       Apr
2026      2026      2026      2026      2026      2026      2026      2026      2027      2027      2027      2027

Track A (substrate truth-up):
├──── B1a FWHT ──────┤
       ├── B1b encoder ──┤
              ├── B1c RoPE ──┤
                    ├── B1d CLA ──┤

Track B (telemetry):
├── A3 audit telemetry ───┤
                   ├── A3.2 OTLP+Grafana full ─┤

Track C (operator + deployment):
├── A1 skeleton ───┤
        ├── A1 reconciler ───────────┤
                       ├── A1 helm + integration ──┤
                                            ├── A4 AMD Cloud (if partnership clarified) ──┤

Track D (workload + evidence):
                          ├── B3 adversarial benches ───┤
                                   ├── B4 RAG eval ───┤
                                                ├── B2 paper v2.0 + arXiv submit ─┤  ← V7.0 RELEASE
                                                                          ├── peer review window ───────┤

Track E (community + ops):
├── D4 CONTRIBUTING ┤
    ├── B5 CI hardening ────┤
                  ├── B6 docs site ──┤

Track F (growth, deferred):
                                                                                          ├── A2 marketplace (if ≥5 plugins) ──┤
```

**V7.0 release target:** end of October / early November 2026 (Sprint ~12-14). Drives the paper v2.0 + arXiv submission timing.

**Critical paths:**
- B1a→B1b→B1c→B1d is sequential within Track A (each modular but truth-up discipline says: don't claim B1b done until B1a is fully tested)
- B3 adversarial benches require B1a-d done → blocks B2 paper v2.0 → blocks arXiv submission
- A4 AMD Cloud depends on A1 helm + integration done

**Parallelizable from day 1:**
- Track A (substrate) ⊥ Track B (telemetry) ⊥ Track C (operator) ⊥ Track E (community/CI/docs)
- Within Sprint 1: B1a + A3 + A1-skeleton + D4 are entirely independent

---

## 3. Decisions needed BEFORE coding

These are blockers — answer them and the plan locks. Defer them and Track D / Track H stay theoretical.

| # | Decision | Default | Why it matters |
|---|---------|---------|----------------|
| **D1** | arXiv submission: NOW (paper v1.1 with V6.2 data) or WAIT (until V7.0 with B1 data)? | **NOW** — V6.2 is publishable; V7.0 → paper v2.0 later | Faster academic indexing, more citations possible |
| **D2** | AMD partnership status: yes / no / TBD? | TBD | Blocks A4 entirely. If "no", A4 is deleted. |
| **D3** | LLC registration timing: now or after first paid POC? | After paid POC | MIT license is fine for now; LLC for liability shielding when revenue starts |
| **D4** | Contribution policy: PR-friendly (CONTRIBUTING + DCO + CoC) or closed (issues only)? | **PR-friendly** — academic adjacent projects benefit from contributors | Different repo configuration |
| **D5** | V7 thesis: "real KV-cache reuse end-to-end" or "multi-node operator-driven"? | **Real reuse end-to-end** — substrate completion is more credible than infra growth | Determines whether A1 K8s operator is V7 core or V7.1 |

---

## 4. Recommended next sprint (May 12 → May 26, 2026)

**Sprint name:** "Quad-track kickoff"

Four parallel tracks. Goal of sprint 1 is to **land first commit on each track**, not to finish them. Each track gets a definition of "sprint-1 done" that's intentionally narrow.

### Track 1: Substrate (B1a — RotateKV FWHT)
- Replace placeholder rotation with real Fast Walsh-Hadamard Transform
- Math: y = H_n · x where H_n is the 2^n × 2^n Hadamard matrix, applied row-wise to KV head-dim
- Algorithm: butterfly recursion, O(d log d) per token, vectorized via numpy/torch
- Tests: round-trip identity (rotate → derotate ≈ original within fp16 epsilon), INT4 quantization MSE vs. baseline, INV-10 preservation
- **Sprint-1 done:** new module `rotate_kv/fwht.py` lands with round-trip + INV-10 tests passing. Full integration with `RotateKVQuantizer.quantize_pre_rope()` can spill to Sprint 2.
- **Acceptance:** `tests/test_rotate_kv.py` 18/18 still pass (regression check), new `tests/test_rotate_kv_fwht.py` ≥ 8 tests pass

### Track 2: Telemetry (A3 — audit-grade INV-15 export)
- `apohara_context_forge/observability/prometheus_exporter.py` (new)
- `apohara_context_forge/observability/audit_log.py` (new, JSONL writer)
- Wire JCRGate.decide() and AnchorPool.match() to emit metrics + audit records
- Sample Grafana dashboard committed to `dashboards/inv15.json`
- **Sprint-1 done:** Prometheus exporter ships with 4-6 core metrics. OTLP gRPC and full Grafana dashboard spill to Sprint 2.
- **Acceptance:** `tests/test_observability.py` ≥ 6 tests, 1 integration test (start exporter → curl /metrics → assert).

### Track 3: Operator (A1 — K8s operator skeleton)
- `operator/` directory at repo root, scaffolded via `operator-sdk init --domain apohara.dev --repo github.com/SuarezPM/Apohara_Context_Forge/operator`
- CRD definition for `ApohraContextForgeCluster` (apiVersion: contextforge.apohara.dev/v1alpha1)
- Reconciler skeleton (does nothing yet, just logs reconciliation events)
- Helm chart skeleton at `charts/apohara-contextforge/` with values.yaml stub
- **Sprint-1 done:** `operator-sdk run local` boots without errors, `kubectl apply -f sample.yaml` triggers a logged reconcile event. Actual reconciliation logic (LMCache sidecar, vLLM worker pods) spills to Sprint 2+3.
- **Acceptance:** kind-based smoke test in CI (deploy CRD → apply sample CR → operator logs "reconciled")

### Track 4: Contribution policy (D4 implementation)
- `CONTRIBUTING.md` (DCO sign-off requirement, PR template link, dev setup)
- `.github/PULL_REQUEST_TEMPLATE.md` (DCO checkbox, "ran honesty.yml locally" checkbox, "AUDIT.md updated if applicable" checkbox)
- `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1)
- `.github/workflows/dco.yml` (DCO bot or homemade grep guard for `Signed-off-by:` trailer)
- **Sprint-1 done:** all 4 files committed, DCO workflow green on a test PR
- **Effort:** ~6-8 hours total. Lowest-leverage track but highest async-value (unlocks external PRs from day 1 of V7).

### Sprint exit criteria
- 4 PR-worthy commits/branches (one per track)
- Zero AUDIT.md table cells regressed from 🟢 to 🟠
- Sprint summary appended to CHANGELOG.md as "V6.3.0-alpha1 (quad-track kickoff)"
- `mem_session_summary` saved
- AMD credits budget: ≤ $20 used (1 MI300X smoke test of B1a FWHT — see §4.5)

### Sprint capacity reality-check
- Track 1 (FWHT): ~12-16 hrs (math + tests, narrow scope helps)
- Track 2 (telemetry): ~10-12 hrs (greenfield, but well-defined surface)
- Track 3 (operator): ~14-18 hrs (operator-sdk learning curve is the risk)
- Track 4 (CONTRIBUTING): ~6-8 hrs (mostly templates)
- **Total: ~42-54 hrs** vs. 2-week budget of 60-80 hrs. Feasible but tight. Triage protocol below.

---

## 4.4. Triage protocol (if Sprint 1 capacity short)

If, by end of week 1 (May 19), any track is < 30% complete, apply this cut order:

1. **First to cut: Track 3 (K8s operator)** — operator-sdk learning curve is the most variable. Skeleton can move to Sprint 2.
2. **Second to cut: Track 4 (CONTRIBUTING)** — pure templating, low cost to defer 2 weeks.
3. **Never cut: Track 1 (FWHT)** — closes a 🟠 AUDIT item; substrate truth-up is the V7 thesis.
4. **Never cut: Track 2 (telemetry)** — creates externalized verifiability moat; competitors can catch up on operator/CONTRIBUTING easily.

If two tracks need cutting, defer to Sprint 2 and update CHANGELOG to say "V6.3.0-alpha1 (dual-track kickoff)".

If the user wants to revisit the 4-track choice mid-sprint, recommend dropping to 2-track (substrate + telemetry) and announce a longer 12-month timeline.

---

## 4.5. AMD AI Dev Cloud credit plan ($100 USD budget)

Strategic burn across the year. Goal: maximize **verifiable evidence** generated per dollar.

**Pricing assumption:** MI300X instances at ~$3-5/hr (will validate before first burn). $100 → ~25-30 hrs of cluster time.

| Sprint / month | Allocation | Use case | Output |
|----------------|-----------|----------|--------|
| Sprint 1 (May) | $15-20 (3-5 hrs) | Smoke-test B1a FWHT on real MI300X HBM3 — sanity check the rotation math against torch's `torch.linalg.qr` on a real KV tensor shape | Evidence appendix in B1a tests + screenshot for paper v2.0 |
| Sprint 2-3 (Jun) | $10-15 (2-3 hrs) | Smoke-test B1c (anchor_pool RoPE derotation on real KV) | Same |
| Sprint 4-6 (Jul-Aug) | $25-35 (5-8 hrs) | A1 K8s operator multi-node smoke test — 2-worker cluster, validate LMCacheConnectorV2 cross-worker hit | Operator integration test screencap + LMCache cross-worker hit confirmed in get_stats() |
| Sprint 8-10 (Oct-Nov) | $25-30 (5-7 hrs) | Paper v2.0 evidence: 1,210-point Cartesian sweep on real MI300X (current sweep is on CPU stub) | Promote paper v2.0's S-15 numbers from "simulated" to "MI300X-measured" |
| **Reserve** | $5-15 | Unplanned debug session if a sprint surfaces an MI300X-specific bug | Buffer |

**Audit trail:** every cloud session logs an entry in `dev-cloud-log.md` with: date, hours used, $ spent, what was tested, result.

**Credit caducation risk:** if AMD credits have an expiry, frontload usage in months 3-6 (Aug-Oct) before any risk window. User to confirm credit T&Cs.

---

## 5. Risks & mitigations (project-level)

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| arXiv moderation delays | MEDIUM | LOW (paper already has DOI) | Submit early, work in parallel |
| AMD partnership uncertainty | HIGH | HIGH for A4 only | D2 decision now; carve A4 out cleanly if "no" |
| FWHT correctness regression breaks INV-10 | LOW | HIGH | Test-first: round-trip identity test before implementation |
| Telemetry scope creep (full OTel ecosystem) | MEDIUM | MEDIUM | Hard-cap A3 at 2 weeks; OTLP gRPC is optional, not required |
| K8s operator-sdk learning slows A1 | MEDIUM | MEDIUM | Use Helm chart first (faster ROI), operator only if Helm proves insufficient |
| Truth-up reveals deeper bugs (B1 cascade) | LOW | HIGH | Sprint-by-sprint validation; if B1a uncovers an INV-10 break, AUDIT.md update before moving to B1b |
| Solo-dev burnout | MEDIUM | HIGH | 12-month plan, not 6-month; protect 1 day/week off |

---

## 6. ADR — Architecture Decision Record

**Decision:** V7.0 ships substrate truth-up + telemetry + K8s operator skeleton + community policy IN PARALLEL, on a ~6-month timeline. Paper v2.0 + arXiv submission gate the V7.0 release.

**Drivers:**
1. V6.1 set a "honest substrate" precedent (AUDIT.md, CI honesty guard) — V7 must complete it
2. Audit-grade telemetry creates verifiability moat — competitors can't easily catch up on "anyone can verify"
3. User override of substrate-first → 4-tracks-parallel (D5) accepts higher capacity risk in exchange for broader V7.0 surface area
4. arXiv timing shifted to V7.0 (D1) — paper v2.0 with V7.0 truth-up data is more defensible than paper v1.1 with V6.2 data alone
5. AMD $100 credit budget enables real MI300X evidence in the paper (was CPU-stub only in v1.1)

**Alternatives considered:**
- **Alt-α: Substrate-first sequential** (my original recommendation) — rejected by user; deemed too slow
- **Alt-β: K8s operator first** — rejected because it would amplify an incomplete substrate
- **Alt-γ: Marketplace SDK first** — rejected because plugin count is 1 (premature)
- **Alt-δ: All 4 parallel (chosen)** — accepted with explicit triage protocol (§4.4) to manage capacity risk

**Why chosen:** User's call. V7.0 with broader surface area + paper v2.0 backed by MI300X data + community-ready repo is a stronger release than V7.0 substrate-only. Capacity risk is mitigated by triage protocol that explicitly names which track gets cut first if reality bites.

**Consequences:**
- V7.0 release timeline: ~6 months (Jun-Nov 2026), assuming triage doesn't fire
- If triage fires (capacity short), V7.0 slips to ~7-8 months but substrate + telemetry land first
- AMD $100 credits create concrete MI300X evidence for paper v2.0 — this changes the paper's defensibility tier from "simulation-validated" to "MI300X-measured"
- Marketplace SDK explicitly deferred to V8; if no third-party plugins appear, deleted from roadmap entirely

**Follow-ups:**
- Bi-weekly sprint retro (end of each 2-week sprint) — check if triage needs to fire
- Quarterly roadmap review (Aug 2026, Nov 2026, Feb 2027)
- Re-visit D2 (AMD partnership) in Aug review — by then we'll have spent ~$30-40 on credits and have real MI300X data to point at
- If plugin count crosses 3 before V8 start, accelerate A2 to V7.1
- D3 (LLC registration) re-visited after first paid POC inquiry

---

## 7. RALPLAN-DR summary

**Principles (5):**
1. Honest semantics: every claim must trace to what code actually does (carried over from V6.1)
2. Audit-grade verifiability: external parties must be able to reproduce or refute every claim
3. Substrate before infrastructure: a complete kernel beats a half-kernel running on K8s
4. Solo-dev sustainability: 12-month horizon, not 6-month sprint
5. Strategic deferral: explicitly mark "not-yet" items rather than half-doing them

**Decision drivers (top 3):**
- D-A: Maintain V6.1's honesty discipline at V7 scale
- D-B: Create externalized verification (telemetry) as a defensibility moat
- D-C: Solo-dev capacity ceiling (~30-40 hrs/wk)

**Viable options (≥2):**
- **Option α — Substrate-first** (my original recommendation): Track A + B sequential, then C. Lower capacity risk, narrower V7.0 surface.
- **Option β — Infrastructure-first**: A1 K8s operator + A4 AMD Cloud, defer B1/A3 to V7.1. Rejected (amplifies incomplete substrate).
- **Option γ — Distribution-first**: A2 marketplace SDK + B6 docs site + outreach. Rejected (premature, plugin count = 1).
- **Option δ — 4 tracks parallel** (user chosen): Tracks A + B + C + E in Sprint 1. Higher capacity risk, broader V7.0 surface. Triage protocol §4.4 manages risk.

**Why δ over α (user override rationale):**
- α's substrate-first ordering optimizes for capacity safety but underuses Q1-Q2 calendar time for community/operator surface
- δ accepts the capacity risk in exchange for V7.0 having ALL 4 surfaces ready, not just substrate + telemetry
- Triage protocol makes δ recoverable to α-like (substrate + telemetry first) without re-planning the year

**Invalidation of β and γ (carried over):**
- β rejected because K8s operator on incomplete substrate (4 🟠 modules) amplifies untruth, not truth
- γ rejected because distribution without substrate completion violates V6.1's "state, not intent" discipline

---

## 8. Changelog (this plan's lineage)

- 2026-05-12: Initial draft (this version)

(Plan revisions will be appended here as decisions land.)
