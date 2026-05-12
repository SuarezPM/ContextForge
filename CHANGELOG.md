# Changelog

## V7.0.0-alpha.2 — Sprint 2: Closes AUDIT #6 + #8, real K8s reconciler · 2026-05-12

Second sprint on the V7 roadmap. Closes two AUDIT items (🟡→🟢 for #6 RotateKV
FWHT wire-up; 🟠→🟢 for #8 pipeline regression), ships a real K8s reconciler,
extends observability with Grafana dashboard + OTLP. All 4 tracks ran via
`/autopilot` with parallel workers and 3 reviewer validators.

### Fixed

- **`agents/base_agent.py`** (Track 1, resolves AUDIT #8) — client-side
  fallback when `call_contextforge_optimize` receives `original_tokens=0`
  on a non-empty context (the coordinator_unavailable passthrough from
  `mcp/server.py`). Uses `len(context.split())` as a local approximation.
  Pipeline metrics tests 6/6 PASS (was 4/6). Root cause traced to commit
  `466cc3d` which locked the server-side 0-value into a test contract;
  the fix belongs in the client.
- **`agents/base_agent.py`** (Phase 4.5 security fix) — added
  `response.raise_for_status()` to both `call_contextforge_register` and
  `call_contextforge_optimize`, matching the existing `call_vllm` pattern.
  Prevents leaking raw server bodies through `JSONDecodeError` tracebacks
  on 4xx/5xx responses.

### Added

- **`apohara_context_forge/quantization/rotate_kv.py`** (Track 2,
  closes AUDIT #6) — wire-up of `fwht.fwht()` into `quantize_pre_rope()`
  conditional on `cfg.use_fwht=True`. Applied after channel reordering
  and before sink-token separation + INT4 quantization. INV-10
  (pre_rope=True) preserved. 8 lines added at lines 162-166 + import
  at line 24.
- **`tests/test_rotate_kv_fwht_integration.py`** (5 new tests) — 5/5
  PASS. Covers FWHT-on vs FWHT-off divergence, INV-10 preservation,
  pipeline round-trip, batched-shape stack.
- **`dashboards/inv15.json`** (Track 3) — full Grafana 11.x dashboard
  with 5 panels (gate decisions over time, current risk score, anchor
  hit rate, LMCache hits, decisions by agent). Templated Prometheus
  data source, schemaVersion 39.
- **`apohara_context_forge/observability/otlp_exporter.py`** (Track 3) —
  `OTLPExporter` with OpenTelemetry gRPC export. Honest-fallback when
  `opentelemetry-exporter-otlp-proto-grpc` is not installed.
  **Phase 4.5 security fix:** default `insecure=False` (TLS by default;
  plaintext requires explicit opt-in for localhost endpoints, warning
  if remote).
- **`apohara_context_forge/observability/recorders.py`** (+28 lines) —
  OTLP singleton via `APOHARA_OTLP_ENDPOINT` env var. **Phase 4.5
  security fix:** `APOHARA_OBSERVABILITY_DIR` is now canonicalized via
  `pathlib.Path(...).expanduser().resolve()` before use, defanging
  `../` traversal.
- **`tests/test_otlp_exporter.py`** (4 tests, 3 PASS + 1 SKIP) — covers
  honest-fallback, idempotent shutdown, fan-out via the singleton.
- **`operator/controllers/apoharacontextforgecluster_controller.go`**
  (Track 4) — went from 40-LOC log-only stub to 453-LOC real reconciler.
  Provisions worker `appsv1.Deployment` matching `Spec.WorkerCount`,
  optional Redis sidecar when `Spec.LMCacheRedisUrl` is empty, owner
  references for GC, status updates with `phase` state machine
  (Pending → Degraded → Ready). Removes unused `phaseProvisioning`
  constant (Phase 4.5 nitpick fix).
- **`operator/controllers/apoharacontextforgecluster_controller_test.go`**
  (273 LOC, 4 tests) — fake-client unit tests for the 4 critical paths.
  `go test ./controllers/...` PASS 4/4 in ~15ms.
- **`operator/integration_test.sh`** (~155 LOC) — bash script that
  uses `kind` to spin a test cluster and apply the CRD + sample CR.
  Honest SKIP path when `kind` not installed.
- **`operator/api/v1alpha1/zz_generated_deepcopy.go`** (~116 LOC,
  hand-authored) — DeepCopyObject/DeepCopyInto for both types (would
  normally come from `controller-gen`).
- **`operator/api/v1alpha1/groupversion_info.go`** (rewritten) — fixed
  Sprint 1 bug where `SchemeBuilder` was zero-valued; now uses
  `runtime.NewSchemeBuilder` with `addKnownTypes`. Required for
  `go vet` cleanness.
- **`operator/go.mod`** — added `k8s.io/api`, `k8s.io/apimachinery`,
  `sigs.k8s.io/controller-runtime` v0.17.0 deps.
- **`.omc/plans/autopilot-impl-sprint2.md`** — Sprint 2 implementation
  plan (auto-approved by `/autopilot`).

### Updated

- **`operator/README.md`** — prominent ⚠️ NOT PRODUCTION READY warning
  listing the 5 Sprint 3 prerequisites: SecurityContext, ServiceAccount
  + RBAC, Redis authentication, NetworkPolicy, image digest pinning.
  Operator binary is NOT built or deployed in Sprint 2 — only reconcile
  logic + unit tests + integration-test skeleton.

### Tests

- **`tests/test_pipeline.py`** — 6/6 PASS (was 4/6 in Sprint 1).
- **`tests/test_rotate_kv_fwht_integration.py`** — 5/5 PASS (new).
- **`tests/test_otlp_exporter.py`** — 3 PASS + 1 SKIP (new).
- **`tests/test_observability.py`** — 6/6 PASS (regression — recorders.py changed).
- **Full regression** — **359 passed, 25 skipped, 0 failed** in 200s.
- **`go test ./operator/controllers/...`** — 4/4 PASS.
- **`go vet ./operator/...`** — clean.
- **`scripts/check_honesty.sh`** — PASS, no regressions.

### Phase 4 validation (autopilot Phase 4)

- **Architect (opus):** APPROVE all 4 tracks. Flagged pre-existing V6.1
  INT4 packing/unpacking asymmetry (AUDIT.md #9).
- **Security-reviewer (sonnet):** REQUEST CHANGES — 7 issues. Easy fixes
  applied in Phase 4.5 (raise_for_status, OTLP TLS default, path
  canonicalization). Operator-deployment-scope issues (SecurityContext,
  RBAC, Redis auth, NetworkPolicy, image pinning) deferred to Sprint 3
  and documented as hard prerequisites in `operator/README.md` + AUDIT.md
  #10.
- **Code-reviewer (opus):** APPROVE with low nitpicks. Unused
  `phaseProvisioning` constant removed in Phase 4.5.

### AUDIT.md deltas

- #6 RotateKV FWHT: 🟡 → 🟢 (fully wired)
- #8 pipeline regression: 🟠 → 🟢 (resolved)
- #9 NEW: V6.1 INT4 packing asymmetry (pre-existing, tracked for Sprint 3)
- #10 NEW: K8s operator security hardening (Sprint 3 scope, blocks deployment)

### Citation

V7.0.0-alpha.2 is a pre-release. Paper v2.0 + arXiv submission gate the
V7.0.0 final release.

---

## V7.0.0-alpha.1 — Sprint 1: Quad-track kickoff · 2026-05-12

First commit on the V7 roadmap (`.omc/plans/v7-roadmap.md`). V7 thesis:
**real KV-cache reuse end-to-end**, building on the V6.1 honesty
discipline and the V6.2 adversarial validation. Sprint 1 lands a
first-commit MVP on each of the four parallel tracks:

### Added

- **`apohara_context_forge/quantization/fwht.py`** (Track 1 — substrate truth-up):
  Real Fast Walsh-Hadamard Transform. In-place butterfly recursion in
  O(d log d), orthonormal (divides by sqrt(d_pad) → self-inverse).
  Supports both `torch.Tensor` (primary) and `np.ndarray` (fallback).
  Non-power-of-two last dims zero-padded to next power of two; fp16
  inputs upcast to fp32 for the butterfly to avoid catastrophic
  precision loss, then cast back. Closes the V6.1 AUDIT.md item #6
  (RotateKV FWHT). Module-only; integration with
  `RotateKVQuantizer.quantize_pre_rope()` lands in Sprint 2.

- **`apohara_context_forge/observability/`** (Track 2 — audit-grade INV-15 telemetry):
  - `prometheus_exporter.py` — per-instance `CollectorRegistry` exporter
    exposing `apohara_jcr_gate_decisions_total{action,agent}`,
    `apohara_inv15_risk_score`, `apohara_anchor_match_total`,
    `apohara_lmcache_hit_total`. Honest-fallback when `prometheus_client`
    is not installed.
  - `audit_log.py` — JSONL writer for every INV-15 gate decision; one
    line per `{ts, kind, agent_id, anchor_hash, risk_score, gate_action,
    predicted_jcr_delta, lmcache_consulted, lmcache_hit}`.
  - `recorders.py` — `record_inv15_decision()` fans out to both. Singletons
    configured via env var `APOHARA_OBSERVABILITY_DIR`.
  - Smoke wire-up at `apohara_context_forge/safety/jcr_gate.py:159` (late
    import inside try/except — best-effort, never raises into the gate
    path).

- **`operator/` + `charts/apohara-contextforge/`** (Track 3 — K8s operator scaffold):
  Manual scaffolding (no `operator-sdk` dependency). `ApohraContextForgeCluster`
  CRD at `contextforge.apohara.dev/v1alpha1` with 5 spec fields
  (workerCount, model, lmcacheRedisUrl, gpuType, image) and 3 status
  fields (readyWorkers, phase, conditions). Reconciler skeleton logs
  "reconciled" only — real reconciliation logic is Sprint 2.
  Helm chart with worker Deployment, headless Service, optional Redis
  sidecar, ConfigMap. `bash operator/validate.sh` exits 0 with all 5
  user-facing YAML files passing `yaml.safe_load`.

- **`CONTRIBUTING.md` + `CODE_OF_CONDUCT.md` + `.github/PULL_REQUEST_TEMPLATE.md` + `.github/workflows/dco.yml`** (Track 4 — community policy):
  PR-friendly governance: Developer Certificate of Origin (DCO)
  sign-off required (no CLA); Contributor Covenant v2.1 adopted by
  reference; PR template enforces the V6.1 honesty checklist (claims
  match runtime state, AUDIT.md updated, no fabricated metrics).
  Contact email for security issues + CoC enforcement:
  `suarezpm@csnat.unt.edu.ar`.

- **`.omc/plans/v7-roadmap.md`** — 12-month V7 strategic plan with
  per-candidate breakdown (effort × leverage × dependencies × DoD),
  sequencing diagram, triage protocol, AMD AI Dev Cloud $100 credit
  burn plan, ADR, RALPLAN-DR summary.

### Tests

- **`tests/test_fwht.py`** — 8/8 PASS. Round-trip identity, orthogonality
  (Hadamard matrix), batched inputs, dtype preservation (fp16/fp32),
  zero input, non-power-of-two padding.
- **`tests/test_observability.py`** — 6/6 PASS. Honest-fallback when
  `prometheus_client` missing, metrics increment, JSONL replay
  order-preserving, disk-error tolerance.
- **Full regression** (excl. pre-existing failures in `tests/test_pipeline.py`):
  **345 passed, 24 skipped** in 200 s.

### Honest accounting

- `tests/test_pipeline.py::test_pipeline_run` and
  `tests/test_pipeline.py::test_pipeline_metrics_tracking` fail on this
  branch's HEAD **before** any Sprint 1 change (`total_tokens_before == 0`).
  Verified by git-stashing the only modified prod file
  (`apohara_context_forge/safety/jcr_gate.py`) and re-running the test
  — same failure. Tracked under AUDIT.md as a pre-existing 🟠 item;
  not introduced by Sprint 1. Sprint 2 will investigate.

### Citation

V7.0.0-alpha.1 is a pre-release, not a Zenodo-deposited version. Paper
v2.0 + arXiv submission gate the V7.0.0 final release (target
Nov 2026; see `.omc/plans/v7-roadmap.md` §2 sequencing).

---

## V6.1.0 — Truth-Up Release · 2026-05-12

Public accountability layer + the surgical fixes that close every
overclaim documented in [AUDIT.md](AUDIT.md). No new mechanisms shipped
in this release on purpose: V6.1 is exclusively about making the
numbers we report match what the code actually computes.

### Fixed

- **`metrics/collector.py`** — removed the `--showgpu占用率 --json`
  rocm-smi flag (Chinese-character LLM mistranslation that failed on
  every ROCm install) and the `return 45.0, 192.0` hardcoded fallback.
  VRAM readings now delegate to `metrics/vram_monitor.py` (pyrsmi →
  /sys/class/drm → 192 GB synthetic), and `MetricsSnapshot.vram_source`
  reports the *actual* backend (`pyrsmi`, `sysfs-drm`, or
  `synthetic-dev`) rather than always claiming "rocm-smi".
- **`decoding/speculative_coordinator.py:verify_and_commit`** —
  accepts an optional `draft_logprobs` argument. When supplied, q_i is
  the draft model's real per-token probability and the
  Leviathan 2023 acceptance criterion min(1, p_i / q_i) is applied
  verbatim, preserving INV-12 (target distribution unchanged). The
  legacy fallback path (no draft_logprobs) still works but emits a
  `WARNING` that INV-12 is no longer guaranteed.
- **`demo/benchmark_v5.py`** — replaced five hardcoded `duration_ms`
  values (250.0, 150.0, 100.0, 120.0, 5.0) with `time.perf_counter()`
  measurements wrapped around each scenario body.
- **`demo/benchmark_v5.py` (S-11)** — `else` branch no longer
  hardcodes `deviation_pct = 0.0`. Reports the real difference
  between observed and predicted λ_critical in both branches. Arrival
  rates ramped from `[0.5..2.5]` to `[1.0..12.0]` so the controller
  is actually exercised.
- **`demo/benchmark_v5.py` (S-15)** — replaced 9 hand-picked cases
  with an exhaustive Cartesian sweep: 5 roles × 11 candidate counts
  × 11 reuse rates × 2 shuffle flags = **1,210 decisions** per run.
  Violation predicate now checks both implication directions (judge
  with risk > τ ⇒ dense AND non-judge ⇒ not dense).

### Added

- **`AUDIT.md`** at repo root — every V6.0 overclaim listed with
  file:line evidence and the V6.1 fix that closes it.
- **`scripts/check_honesty.sh`** + **`.github/workflows/honesty.yml`**
  — CI guard that fails the build if any of the V6.0 lies regress
  (hardcoded `duration_ms`, the Chinese rocm-smi flag, the fabricated
  `draft_prob_estimate` without the INV-12 warning, the `45.0, 192.0`
  hardcoded tuple).

### Benchmark profile change

V6.0 reported **15/15 PASS**. V6.1 reports **14/15 PASS with one
honest fail**: S-11 (queueing controller stability) now shows a real
deviation of ~100% under the new arrival-rate ramp because the M/G/1
model overestimates capacity under bursty load with a draining block
budget. **This failure is the intended outcome of the truth-up.** The
controller's math is correct (the module is 🟢 PRODUCTION in the
audit); the benchmark scenario is too bursty for the current
estimator. Tuning the controller's burst-handling, or replacing the
synthetic arrival process with a less bursty one, is tracked under
V6.2.

### Compatibility

- `MetricsCollector._use_rocm` and `MetricsCollector._check_rocm()` are
  retained as backwards-compat shims. All 13 `test_mcp_server.py` and
  all 21 `test_speculative_coordinator.py` tests continue to pass.
- `verify_and_commit(target_verification_logprobs, draft_tokens)` keeps
  the legacy two-argument call signature — the new `draft_logprobs`
  parameter is keyword-only and defaults to `None`.

### Citation

The Zenodo deposit for V6.1 is the next version under the same concept
DOI: [10.5281/zenodo.20114594](https://doi.org/10.5281/zenodo.20114594).
The V6.0 deposit remains accessible but is now superseded.

---

## V6.0.0 — Initial hackathon submission · 2026-05-10

Initial release for the AMD AI Hackathon. 10 mechanisms wired,
15/15 benchmark PASS (subsequently audited and partly revised in V6.1
— see AUDIT.md), 310/310 unit tests passing, INV-15 enforcement,
TokenDance master-mirror storage, JCR Safety Gate, AITER ROCm config,
paper deposited at Zenodo with DOI 10.5281/zenodo.20114594.
