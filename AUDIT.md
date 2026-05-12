# ContextForge — V6.0 Honest Audit

> **Status:** Living document. Maintained alongside the codebase.
> Every overclaim shipped in V6.0 is listed here with file:line evidence
> and a tracked fix in V6.1 ("Truth-Up Release"). New mechanisms must
> declare which of the four states below they live in *before* they
> show up in a benchmark.

Every research / systems project ships with a gap between *claims in the
README* and *what the code actually computes*. ContextForge is no exception.
The hackathon submission and the published paper (DOI
[10.5281/zenodo.20114594](https://doi.org/10.5281/zenodo.20114594))
captured the V6.0 state. This document is the public accountability layer:
it lists, with file:line evidence, the things that **look measured** but
are actually **synthesized**, and tracks each one through to a fix.

The document also lists the parts that are **production-grade**, so the
reader knows where the codebase carries its own weight.

---

## The four states

| State          | Meaning |
|----------------|---------|
| 🟢 PRODUCTION   | Real implementation. Computes its claimed value from real inputs. Tests cover real behavior. |
| 🟡 HONEST STUB  | Clearly marked as stub / fallback in docstring or runtime warning. Returns plausible defaults without claiming they are measured. |
| 🟠 PARTIAL      | Real algorithm but with synthetic inputs or hardcoded constants where the claim implies measurement. |
| 🔴 OPTIMISTIC   | The README / paper / benchmark implies "live" or "measured" but the code is actually mocked / hardcoded. |

---

## V6.0 confirmed overclaims (sorted by severity)

### 1. 🔴 Speculative coordinator: fabricated draft probability

- **Claim** *(README §benchmark, paper §1)*: "Speculative acceptance rate ≥ 0.875"; INV-12 (target output distribution preserved by speculation).
- **Reality** *(`apohara_context_forge/decoding/speculative_coordinator.py:261`)*:
  ```python
  draft_prob_estimate = max(0.4, 1.0 - 0.4 * self.config.acceptance_threshold)
  ratio = min(1.0, p_i / draft_prob_estimate)
  ```
  The draft probability `q_i` is **not from the draft model** — it is
  fabricated from a config knob. With `acceptance_threshold=0.9` the
  estimate is 0.64; any target probability above 0.64 gives `ratio=1.0`
  (deterministic accept). INV-12 (distribution-preservation guarantee
  from Leviathan et al. 2023) is **mathematically broken** under this
  formula.
- **Severity:** High. Reviewers reading the paper section on speculative
  decoding will spot this in five minutes.
- **V6.1 fix:** Either expose real draft logprobs across the agent
  boundary and use the real `min(1, p/q)` (preferred), or rename
  `verify_and_commit` to `verify_and_commit_stub`, document it as a
  placeholder, and drop the INV-12 claim from the README and paper §3.

### 2. 🔴 VRAM telemetry: corrupted rocm-smi flag, hardcoded fallback

- **Claim** *(README, paper §4.4)*: live MI300X VRAM monitoring via rocm-smi.
- **Reality** *(`apohara_context_forge/metrics/collector.py:50`)*:
  ```python
  result = subprocess.run(
      ["/opt/rocm/bin/rocm-smi", "--showgpu占用率", "--json"],
      ...
  )
  ```
  The flag contains Chinese characters ("占用率" = "usage rate") — almost
  certainly an LLM-generated mistranslation that stitched English and
  Chinese tokens. **This subprocess call fails on every ROCm install in
  existence.** The function then falls through to line 66:
  ```python
  return 45.0, 192.0
  ```
  Every VRAM number that flows through `MetricsCollector.snapshot()` is
  the hardcoded pair `(45.0 GB, 192.0 GB)`. The dashboard, `/health`,
  and `MetricsSnapshot.vram_source="rocm-smi"` all report fake values.
- **Severity:** High. The dashboard is the single most-visible artifact;
  it's also the one that ships fake numbers most frequently.
- **V6.1 fix:** Replace the flag with `--showuse --showmemuse --json` (or
  whichever valid combination), parse the real JSON keys, and delete the
  hardcoded fallback in favor of `apohara_context_forge/metrics/vram_monitor.py`
  (which already implements the honest pyrsmi → /sys/class/drm path).

### 3. 🔴 S-11 queueing controller: 299% real deviation reported as 0%

- **Claim** *(paper Table 2, S-11 benchmark)*: "QueueingController λ_critical deviation **0.00%**, target < 10%, PASS".
- **Reality** *(`demo/benchmark_v5.py:567-575`)*:
  ```python
  if not is_stable:
      ...
  else:
      # No failure observed — use highest rate as proxy
      observed_lambda_critical = arrival_rates[-1]
      predicted_lambda_critical = controller.compute_stability_state(...).lambda_critical
      deviation_pct = 0.0
  ```
  When the system never goes unstable (which the seeded toy load
  guarantees), the code **sets deviation_pct to 0 unconditionally**.
  The actual values in the published JSON (`demo/benchmark_v5_results.json`):
  ```
  lambda_critical_observed:  2.5
  lambda_critical_predicted: 9.99
  reported deviation_pct:    0.0
  real deviation_pct:        299.76%
  ```
  The controller's math is sound; the benchmark logic launders a 299%
  prediction error into a 0% PASS.
- **Severity:** High. This is the headline metric of S-11.
- **V6.1 fix:** When no instability is observed, report
  `|predicted - max(arrival_rates)| / max(arrival_rates) * 100`. Expect a
  large number under the current toy load — that is *honest signal* that
  we need an adversarial scenario (higher rates, smaller blocks) to stress
  the model, *not* a worse implementation of the model.

### 4. 🔴 Benchmark scenarios S-11..S-15: hardcoded duration_ms

- **Claim** *(paper Table 1)*: per-scenario latency and throughput.
- **Reality** *(`demo/benchmark_v5.py:580, 656, 730, 794, 855`)*:
  ```python
  duration_ms=250.0  # S-11
  duration_ms=150.0  # S-12
  duration_ms=100.0  # S-13
  duration_ms=120.0  # S-14
  duration_ms=  5.0  # S-15
  ```
  The reported `throughput_tps` is then `tokens_processed / (duration_ms
  / 1000)` — pure arithmetic, no actual timing. The work inside each
  scenario completes in microseconds; the "real MI300X durations" in
  paper Table 1 are constants.
- **Severity:** Medium-High. The PASS badges are tautologies, but any
  reviewer running `git grep "duration_ms\s*=\s*[0-9]"` finds it.
- **V6.1 fix:** Wrap each scenario body in `time.perf_counter()` and use
  the measured duration. Same change for `throughput_tps`.

### 5. 🟠 S-12 visual encoder: no encoder is ever called

- **Claim** *(README, paper)*: "5× encoder call reduction" via
  cross-agent VisualKVCache sharing.
- **Reality** *(`demo/benchmark_v5.py:644, 681`)*:
  ```python
  encoder_calls_baseline = 5      # hardcoded
  encoder_calls_actual   = 1      # hardcoded
  reduction              = 5 / 1  # = 5×
  ```
  No vision model is invoked anywhere. The scenario is `store()` once
  plus `lookup()` four times on a numpy random tensor. The cache, hash,
  and store mechanics are real; the "5×" is arithmetic.
- **Severity:** Medium. The VisualKVCache module is real; the headline
  is staged.
- **V6.1 fix:** Either integrate a small CLIP / SigLIP encoder (real
  call, measured wall time), or replace the headline with the legitimate
  one: "cache lookup latency vs. encoder-call latency = O(µs) vs O(ms)
  on the same hardware". Drop the "5×" claim unless we measure it.

### 6. 🟠→🟡 RotateKV: FWHT rotation now exists as a standalone module (not yet integrated)

- **Claim** *(README, paper §2 mechanism #5)*: "Pre-RoPE INT4 grouped-head
  rotation, 3.97× VRAM reduction".
- **Original V6.0 reality** *(`apohara_context_forge/quantization/rotate_kv.py:215-247`)*:
  The `use_fwht` flag is read in `__init__` but the Fast Walsh-Hadamard
  Transform step **never executes** — only channel reordering and
  asymmetric block-wise quantization are present.
- **V7.0.0-alpha.1 update** *(`apohara_context_forge/quantization/fwht.py`, 112 LOC)*:
  Real orthonormal FWHT now exists as a standalone module — in-place
  butterfly recursion in O(d log d), 8/8 tests passing (round-trip
  identity, Hadamard orthogonality, batched inputs, dtype preservation,
  zero-padding for non-power-of-two dims). The module itself is **🟢
  PRODUCTION**. The remaining 🟡 status is because
  `RotateKVQuantizer.quantize_pre_rope()` still does NOT call this
  module — that integration is Sprint 2 (V7.0.0-alpha.2). After
  Sprint 2 lands the wire-up, this item will be 🟢.
- **Severity:** Low (now). The pre-rotation quantization itself is
  unchanged; this delta is purely about closing the substrate honestly.

### 7. 🟠 S-15 JCR gate: cherry-picked sweep cases

- **Claim** *(paper §5.2, abstract)*: "0 INV-15 violations across the
  full sweep".
- **Reality** *(`demo/benchmark_v5.py:826-872`)*: the "sweep" is **5
  hand-picked Critic cases plus 4 non-judge cases**, all chosen so the
  invariant holds by construction. The gate module itself
  (`apohara_context_forge/safety/jcr_gate.py`) is honest and well-tested;
  it's the *framing* of S-15 as "empirical evidence" that overreaches.
- **Severity:** Low-Medium. The mechanism is novel and real; the result
  is closer to a unit test than an empirical sweep.
- **V6.1 fix:** Generate the sweep procedurally over the full Cartesian
  product of `(role ∈ {critic, judge, retriever, …}) × (candidates ∈ [1..10])
  × (reuse ∈ [0.1..1.0]) × (shuffle ∈ {0,1})`. Report both fire-rate and
  the *closed-form check* that the gate matches the spec on all points.
  Frame as "exhaustive contract check" rather than "empirical violation rate".

### 8. 🟠 `tests/test_pipeline.py` — pre-existing failures on branch HEAD

- **Discovered:** 2026-05-12 during V7.0.0-alpha.1 Sprint 1 verification
- **Symptom:** `TestDemoAgents::test_pipeline_run` and
  `TestPipeline::test_pipeline_metrics_tracking` both fail with
  `assert pipeline.metrics["total_tokens_before"] > 0` returning 0.
- **Provenance check:** Sprint 1 introduced exactly ONE prod-code
  change (a 15-line late-import wire-up in `safety/jcr_gate.py:159`).
  Verified by `git stash`-ing that file and re-running the test —
  failure persists with our change reverted. Therefore the regression
  is **pre-existing on branch HEAD**, not introduced by Sprint 1.
- **Severity:** Medium. The pipeline metrics tracking is what the
  Gradio demo uses to display "tokens saved" — if it reads 0, the demo
  shows 0 savings even when the registry is working.
- **Tracked for:** Sprint 2 (V7.0.0-alpha.2). Likely cause: the demo's
  `Pipeline` class stopped wiring `TokenCounter` into the per-agent
  metrics tally somewhere between V6.1 and V6.x #3. Will reproduce on
  V6.1.0 tag to isolate the regressing commit.

### V7.0.0-alpha.1 — Sprint 1 deltas added (2026-05-12)

Three new modules entered the audit, all marked at their honest status:

| Module | State | Why |
|--------|-------|-----|
| `apohara_context_forge/quantization/fwht.py` | 🟢 PRODUCTION | Real butterfly recursion, 8/8 tests, orthonormal, fp16 upcast. Standalone — not yet called by `RotateKVQuantizer` (closing #6 from 🟠 to 🟡 above). |
| `apohara_context_forge/observability/{prometheus_exporter,audit_log,recorders}.py` | 🟢 PRODUCTION | Real `prometheus_client` Counter/Gauge + real JSONL audit log. Honest-fallback when `prometheus_client` not installed. Smoke wire-up at `safety/jcr_gate.py:159` (late import, best-effort). 6/6 tests. |
| `operator/` + `charts/apohara-contextforge/` | 🟡 HONEST STUB | CRD + helm chart YAML validate (`bash operator/validate.sh` exits 0). Reconciler logs "reconciled" only — real reconciliation is Sprint 2. README declares this status. |

The community-policy track (CONTRIBUTING + DCO + CoC + PR template) is
governance, not a code module, so it does not enter the state table.

---

## What is actually real (don't apologize)

These modules are production-grade and back the substrate of the system:

| Module | What it does, honestly |
|--------|------------------------|
| `safety/jcr_gate.py` | Risk function + threshold + audit log. Deterministic. The INV-15 concept is the most original IP in the repo. |
| `storage/token_dance.py` | Real master-mirror sparse-diff numpy. Reconstructs byte-correct to ~1e-7 (float roundoff). |
| `registry/context_registry.py` + `registry/vram_aware_cache.py` | Real DI, real LSH+FAISS+VRAM-pressure eviction across five modes. |
| `dedup/lsh_engine.py` + `dedup/faiss_index.py` | Real 64-bit SimHash with Hamming distance + real FAISS IndexFlatIP with IVF upgrade path. |
| `scheduling/step_graph.py` + `scheduling/pbkv_predictor.py` | Real DAG with topological compute + real 2nd-order Markov with Laplace smoothing and JSONL persistence. |
| `compression/{coordinator,compressor,budget_manager}.py` | Real LLMLingua-2 wrapper + sensible per-segment compression policies. |
| `agents/*.py` + `mcp/server.py` | Real 5-agent pipeline, real FastAPI lifespan-managed MCP server with Depends-based DI. |
| `metrics/vram_monitor.py` | The *correct* VRAM path (pyrsmi → /sys/class/drm → 192GB default). Just needs to be wired into `MetricsCollector`. |

The substrate of the system — registries, indexes, schedulers, agents,
compressors, server — earns its keep. The lies are concentrated in
**(a) metrics/collector.py**, **(b) demo/benchmark_v5.py V5/V6
scenarios**, and **(c) speculative_coordinator.py:261**.

---

## V6.1 — "Truth-Up Release" (2 weeks, before any new feature)

Ordered by leverage; each item links to its fix above.

| # | Fix | Effort | Risk if skipped |
|---|-----|--------|-----------------|
| 1 | metrics/collector.py rocm-smi flag → real numbers via VRAMMonitor | 1 h | Anyone running on real MI300X sees the lie immediately. |
| 2 | benchmark_v5.py S-11 deviation logic + 5 hardcoded `duration_ms` → real timing | 4 h | Paper Table 1 cannot survive `git grep`. |
| 3 | speculative_coordinator.py:261 — either real `q_i` or downgrade to stub | 1 d | Reputationally the worst because the paper makes a formal-correctness claim about it. |
| 4 | S-15 procedural Cartesian sweep | 4 h | Reframes "0 violations" as "exhaustive contract check" — stronger, not weaker. |
| 5 | S-12 real encoder OR honest reframing | 4 h | The 5× claim is the easiest to disprove. |
| 6 | RotateKV: implement FWHT OR relabel as "follows IJCAI 2025; FWHT pending" | 1 d | Low urgency; can stay 🟠 if labeled. |
| 7 | `AUDIT.md` (this file) committed at root | — | Done. |
| 8 | README hero stat strip cross-references AUDIT.md for the figures | 30 min | Public accountability multiplies the credibility of the rest. |

Total V6.1 effort: **~3.5 dev-days**. Ship as **V6.1 with full
changelog**, including a Zenodo replacement deposit so the DOI tracks
the corrected numbers.

---

## Maintenance discipline (from V6.1 onward)

1. **No new mechanism enters the README mechanism table without an entry in this file** declaring its state (🟢/🟡/🟠/🔴).
2. **No benchmark scenario merges without** (a) real `time.perf_counter()` measurement and (b) a procedurally-generated input set, *not* a hand-curated one.
3. **Every paper-claimed invariant must have a test** that exhaustively verifies it on at least 100 procedurally-generated points, not 5 hand-picked ones.
4. **Every external paper we cite as "implemented"** must have one of: (a) faithful implementation with a passing test against the paper's reference output, OR (b) a "follows X, with delta Y" disclaimer that lists what we actually do differently.
5. **The CI runs `git grep -E "duration_ms\s*=\s*[0-9]"` on `demo/`** and fails if any match — same for `vram_peak_gb\s*=\s*[0-9]`. Hardcoded perf numbers are a build failure.

---

## Open questions deferred to V6.x scoping

These are the questions where the answer determines what we build next.
See the V6.x roadmap discussion for the current direction.

- Is the **speculative coordinator** worth implementing properly, or is
  the right move to remove it entirely (it isn't load-bearing for any
  other mechanism)?
- Is **RotateKV FWHT** worth implementing in Apohara given that the
  paper's authors have released CUDA reference code that we'd be
  duplicating, or do we cite-and-skip?
- Does the **vLLM ATOM plugin (V6.x item #1)** justify a true V1 plugin
  PR upstream to vLLM, or do we publish the standalone Apohara plugin
  on PyPI and let users wire it themselves?

---

*Last updated: 2026-05-10 · maintained by the same person who wrote the lies.*
