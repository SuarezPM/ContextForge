# Autopilot Sprint 2 Implementation Plan

**Status:** `auto-approved by /autopilot`
**Date:** 2026-05-12
**Predecessor:** V7.0.0-alpha.1 (Sprint 1 closed, commit `53309b0`)
**Spec:** [`.omc/plans/v7-roadmap.md`](v7-roadmap.md) §4.4 + Sprint 1 final summary backlog

---

## 1. Scope: 4 parallel tracks

| Track | Priority | Track focus | Worker type |
|-------|---------|-------------|-------------|
| 1 | HIGH | Investigate + fix `test_pipeline.py` pre-existing regression (AUDIT #8) | debugger (sonnet) |
| 2 | HIGH | Wire `fwht.py` into `RotateKVQuantizer.quantize_pre_rope()` (closes AUDIT #6 to 🟢) | executor (opus) |
| 3 | MEDIUM | Full Grafana dashboard + OTLP gRPC exporter | executor (sonnet) |
| 4 | MEDIUM | K8s operator real reconciler logic + kind-based integration test | executor (sonnet) |

Sprint 1 lessons applied:
- No haiku for multi-file work (Track 4 was abandoned by haiku in Sprint 1)
- Pre-existing regression-aware QA pass: `git stash` + retest before declaring our own break
- Honest accounting: every claim in commit msg + AUDIT/CHANGELOG must match runtime state

## 2. Per-track scope

### Track 1 — Pipeline regression debug (debugger)
- **Symptom:** `TestDemoAgents::test_pipeline_run` + `TestPipeline::test_pipeline_metrics_tracking` fail with `total_tokens_before == 0`
- **Hypothesis:** `TokenCounter` was disconnected from `Pipeline.metrics` between V6.1.0 and HEAD
- **Approach:** `git log -p tests/test_pipeline.py demo/pipeline.py agents/pipeline.py` to find regression point; identify the line where `total_tokens_before` is supposed to be incremented; fix
- **DoD:** Both pipeline tests pass; no new tests broken; AUDIT #8 status moves to ✅ resolved

### Track 2 — Wire FWHT into RotateKV (executor opus)
- **Current state:** `apohara_context_forge/quantization/fwht.py` exists with 8/8 tests; `RotateKVQuantizer.quantize_pre_rope()` does NOT call it (use_fwht flag is read but never applied)
- **Approach:**
  1. Read `apohara_context_forge/quantization/rotate_kv.py` to find `quantize_pre_rope()`
  2. Locate the rotation step (currently placeholder)
  3. Add: `if self.use_fwht: tensor = fwht(tensor)` at the correct stage
  4. Add integration test `tests/test_rotate_kv_fwht_integration.py` (round-trip test on a synthetic KV tensor)
- **DoD:** When `use_fwht=True`, FWHT really executes (verifiable by mocking + side-effect check); INV-10 (pre_rope=True) preserved; existing `tests/test_rotate_kv.py` 18/18 still pass; new integration test passes; AUDIT #6 moves from 🟡 to 🟢

### Track 3 — Grafana + OTLP (executor sonnet)
- **Current state:** Prometheus exporter + JSONL audit log shipped in Sprint 1; no Grafana dashboard yet; no OTLP support
- **Approach:**
  1. Create `dashboards/inv15.json` — full Grafana dashboard JSON (Prometheus data source, panels for: gate decisions over time, risk score histogram, anchor hit rate, LMCache hit rate, decisions by agent)
  2. Add `OTLPExporter` class in `prometheus_exporter.py` (gRPC, late import of `opentelemetry-exporter-otlp`, honest-fallback)
  3. Add 3-5 tests for the OTLP path
- **DoD:** Dashboard JSON parses as valid Grafana 11.x; OTLP exporter can be instantiated + can `record_*()`; tests pass

### Track 4 — K8s reconciler + integration test (executor sonnet)
- **Current state:** Sprint 1 scaffold has `Reconcile()` that logs "reconciled" only
- **Approach:**
  1. Read Sprint 1 scaffold at `operator/controllers/apoharacontextforgecluster_controller.go`
  2. Implement Reconcile() logic:
     - Read `ApohraContextForgeClusterSpec`
     - If `lmcacheRedisUrl == ""`: ensure a Redis Deployment+Service exists in same namespace
     - Ensure N worker Pods exist (matching `workerCount`) — Pod spec from Sprint 1 chart
     - Update status: readyWorkers count, phase (Pending→Provisioning→Ready)
  3. Add `operator/controllers/apoharacontextforgecluster_controller_test.go` — fake client based test
  4. Add `operator/integration_test.sh` — script that uses `kind` to spin a cluster and apply the sample CR (skip in CI if no kind installed)
- **DoD:** Controller test passes via `go test ./...` if Go toolchain available, else clearly logged as skipped; integration script lints; CRD reconcile flow is no longer just-a-log

## 3. Acceptance criteria

- Tracks 1+2 close two AUDIT items (#6 → 🟢, #8 → resolved)
- Track 3+4 extend Sprint 1 deliverables (no AUDIT regression)
- Full regression: ≥ 350 passed (was 345 in Sprint 1; +5 from new tests)
- `scripts/check_honesty.sh` PASS
- All 3 Phase 4 validators approve (architect + security-reviewer + code-reviewer)
- Commit as `V7.0.0-alpha.2 (Sprint 2)` with DCO sign-off + Co-Authored Claude

## 4. Risk register

| Risk | Mitigation |
|------|-----------|
| Track 1 regression discovery yields deep bug requiring multi-commit | Single-commit hard cap; document residual in AUDIT #8 if scope blow-up |
| Track 2 INV-10 invariant breaks under FWHT integration | Test-first: write `test_fwht_preserves_inv10` BEFORE wiring |
| Track 4 Go toolchain not installed → tests can't run | Skip gracefully + document in operator/README.md |
| Anthropic content filter trips on Track 3 OTLP code (unlikely but happened in Sprint 1 with CoC) | If 400 error: shorter, less verbose code |
| Combined token spend on 4 parallel Opus/Sonnet agents | Use sonnet for 3/4, opus only for Track 2 |

## 5. Validation strategy (Phase 4)

3 parallel reviewers, all must approve:

- **architect** (opus): functional completeness — does each track close its DoD?
- **security-reviewer** (sonnet): K8s reconciler RBAC + ServiceAccount review (Track 4 is the only security-relevant track)
- **code-reviewer** (opus): overall PR quality, style consistency, no fabrication

Rejection → fix and re-validate up to 3 rounds. If still rejected, stop and report.

## 6. Phase mapping (autopilot skill)

- **Phase 0 (Expansion):** ✅ SKIP — `.omc/plans/v7-roadmap.md` is the spec
- **Phase 1 (Planning):** ✅ THIS DOCUMENT
- **Phase 2 (Execution):** parallel Agent spawn for 4 tracks
- **Phase 3 (QA):** full regression + honesty check + per-track verification
- **Phase 4 (Validation):** 3 parallel reviewers
- **Phase 5 (Cleanup):** batch commit + push + summary + state cleanup
