# Autopilot Sprint 3 Implementation Plan

**Status:** `auto-approved by /autopilot`
**Date:** 2026-05-12
**Predecessor:** V7.0.0-alpha.2 (Sprint 2 closed, commit `cd86fed`)
**Spec:** [`.omc/plans/v7-roadmap.md`](v7-roadmap.md) + Sprint 2 AUDIT.md #9 + #10
**Budget constraint:** $30 USD remaining on AMD AI Dev Cloud @ $1.99/hr ≈ 15 hrs of MI300X time

---

## Strategic split: Wave A (CPU) + Wave B (MI300X)

Wave A is everything that does NOT require the MI300X. Wave B is everything that does.
Wave A runs autonomously now; Wave B requires user to power the droplet + provide SSH IP.

### Wave A scope (this autopilot run)

| Track | AUDIT closes | Effort | Worker |
|-------|--------------|--------|--------|
| 1 | #9 V6.1 INT4 packing/unpacking asymmetry | 3-4 hrs | executor (opus) |
| 2 | #10 SecurityContext + image digest pinning | 2-3 hrs | executor (sonnet) |
| 3 | #10 ServiceAccount + namespaced RBAC | 2-3 hrs | executor (sonnet) |
| 4 | #10 Redis auth + NetworkPolicy + MI300X scripts for Wave B | 3-4 hrs | executor (sonnet) |

### Wave B scope (separate autopilot run after user kicks off)

| Task | Cost (MI300X hrs × $1.99) | Output |
|------|---------------------------|--------|
| Smoke-test FWHT integration on real KV tensors | ~$2-4 (1-2 hrs) | Sprint 2 substrate validation |
| Measure 3.97× VRAM reduction on real MI300X | ~$6-8 (3-4 hrs) | Paper v2.0 evidence (promote claim to "MI300X-measured") |
| Run V6.2 adversarial bench on real MI300X | ~$6-8 (3-4 hrs) | Paper v2.0 evidence (promote from "simulation-validated") |
| Reserve | ~$10-15 (5-7 hrs) | Sprint 4+ |

---

## Wave A: per-track scope

### Track 1 — AUDIT #9 V6.1 INT4 packing/unpacking asymmetry
- **Files:** `apohara_context_forge/quantization/rotate_kv.py` (`_quantize_block` lines 222-229 + `_dequantize_block` lines 287-294)
- **Bug:** Write side packs two nibbles using SAME `i` index → both nibbles go to same byte at different positions (lower nibble unchanged from previous block, upper via `|= (val << 4)`). Read side correctly unpacks both nibbles from SAME byte but expecting the symmetric pack which doesn't exist.
- **Fix:** Pack nibble pairs into single bytes correctly: `keys_int4[blk, i//2, h, d] = (lower_nibble & 0xF) | ((upper_nibble & 0xF) << 4)` iterating `i` over `range(0, group_size, 2)` and reading both `quantized[i]` (lower) and `quantized[i+1]` (upper). The read side already handles this correctly — only the write side is buggy.
- **Verification:**
  - Add `tests/test_rotate_kv_int4_codec.py` with round-trip identity test on a random KV tensor (assert max abs error ≤ 1/16 = 0.0625, the INT4 step size after scale-shift)
  - Existing `tests/test_rotate_kv.py` (5 tests) must still pass
  - The existing `test_fwht_roundtrip_through_pipeline` (3× slack tolerance) will now succeed with tighter bound — update its tolerance to 1.5× INT4 envelope (the FWHT introduces some additional rounding via fp16 upcast/downcast, hence the 1.5×)
- **DoD:** AUDIT #9 moves to 🟢 RESOLVED; round-trip error drops from ~6.3 to ≤ 0.07 on standardized test tensor

### Track 2 — AUDIT #10 K8s SecurityContext + image digest pinning
- **Files:**
  - `operator/controllers/apoharacontextforgecluster_controller.go` (lines 137-150 Redis, lines 200-235 worker Deployment)
  - `operator/api/v1alpha1/apoharacontextforgecluster_types.go` (image default)
  - `operator/config/samples/contextforge_v1alpha1_apoharacontextforgecluster.yaml` (sample image)
- **Tasks:**
  1. Add `PodSecurityContext` with `runAsNonRoot=true`, `runAsUser=65534` (nobody), `seccompProfile.type=RuntimeDefault` on both Redis and worker Deployments
  2. Add `SecurityContext` per container with `allowPrivilegeEscalation=false`, `readOnlyRootFilesystem=true`, `capabilities.drop=["ALL"]`
  3. For `readOnlyRootFilesystem`, ensure Redis pod has emptyDir volume mounted at `/data` (Redis writes to /data); worker pod has emptyDir at `/tmp` if needed
  4. Change default image from `:latest` to `:v7.0.0-alpha.3` (alpha tag pending real digest pin in v7.0.0 final). Add `imagePullPolicy: IfNotPresent` explicitly.
  5. Update controller_test.go to assert SecurityContext is set
- **Verification:**
  - `go test ./operator/controllers/...` — existing 4 tests + new SecurityContext assertions PASS
  - `go vet ./operator/...` — clean
  - `bash operator/validate.sh` — exit 0
- **DoD:** 2 of 5 AUDIT #10 items closed

### Track 3 — AUDIT #10 K8s ServiceAccount + namespaced RBAC
- **Files (new):**
  - `operator/config/rbac/service_account.yaml` — `ServiceAccount: apohara-contextforge-controller` in `system` (or configurable) namespace
  - `operator/config/rbac/role.yaml` — namespaced `Role` granting only the verbs the controller needs (apoharacontextforgeclusters get/list/watch/update/patch; deployments + services + secrets get/list/watch/create/update/patch; pods get/list/watch; events create/patch)
  - `operator/config/rbac/role_binding.yaml` — `RoleBinding` linking the SA to the Role
  - `operator/config/rbac/leader_election_role.yaml` + `leader_election_rolebinding.yaml` — for leader election (configmaps + leases get/list/watch/create/update/patch in controller namespace)
- **Files modified:**
  - `operator/config/manager/kustomization.yaml` — reference the RBAC directory
  - `operator/README.md` — update Sprint 3 status (remove the ⚠️ once all 5 items are closed)
- **Verification:**
  - All new YAML files parse via `yaml.safe_load`
  - `bash operator/validate.sh` extended to lint the new RBAC files
- **DoD:** 1 of 5 AUDIT #10 items closed (RBAC)

### Track 4 — AUDIT #10 Redis auth + NetworkPolicy + MI300X scripts for Wave B
- **Files modified (Redis auth):**
  - `operator/controllers/apoharacontextforgecluster_controller.go` — `reconcileRedisSidecar`: generate a random 32-char password at first reconcile, store as Secret `<name>-redis-auth` with key `password`, pass to Redis via `--requirepass $(REDIS_PASSWORD)` env var, pass to workers via `secretKeyRef`
  - `operator/api/v1alpha1/apoharacontextforgecluster_types.go` — add `Status.RedisSecretName` field for visibility
  - `operator/controllers/apoharacontextforgecluster_controller_test.go` — add Test 5: Reconcile creates Redis Secret when auto-provisioning
- **Files new (NetworkPolicy):**
  - `operator/config/networkpolicy/worker_to_redis.yaml` — allow worker pods (label `apohara.dev/role=worker`) to egress to Redis (label `apohara.dev/role=redis`) on port 6379; deny all other egress
  - `operator/config/networkpolicy/worker_ingress.yaml` — allow ingress to workers only from named gateway labels OR same-namespace
  - Controller does NOT manage NetworkPolicies (out of scope — admin-provisioned per cluster)
- **Files new (Wave B prep):**
  - `scripts/mi300x_smoke_fwht.sh` — bash script that runs on droplet: pip install torch + pytest, runs `PYTHONPATH=. pytest tests/test_rotate_kv_fwht_integration.py -v` against real MI300X, writes results to `logs/mi300x_fwht_$(date +%s).json` with `nvidia-smi` / `rocm-smi` output
  - `scripts/mi300x_vram_measurement.py` — Python script that builds a synthetic 32K-token KV cache, applies RotateKVQuantizer with use_fwht=True, measures peak VRAM via `torch.cuda.max_memory_allocated()`, dumps `logs/mi300x_vram_$(date +%s).json` with the actual reduction ratio
  - `scripts/mi300x_v62_adversarial.sh` — bash script that runs `python demo/benchmark_v62_adversarial.py` on the droplet, captures real M/G/1 timings + arrival distributions, writes `logs/mi300x_v62_$(date +%s).json`
  - `scripts/mi300x_runbook.md` — instructions for the user on how to ssh into the droplet + which script to run when (this is the document I read when user gives me the SSH IP)
- **Verification:**
  - `go test ./operator/controllers/...` — 5/5 PASS (including new Test 5 for Secret creation)
  - YAML files lint
  - MI300X scripts: bash syntax check + Python `-c "import ast; ast.parse(open(...).read())"` — must parse
- **DoD:** 2 of 5 AUDIT #10 items closed (Redis auth + NetworkPolicy) + Wave B scripts ready

---

## Acceptance criteria (Wave A)

- AUDIT #9 → 🟢 (INT4 packing fixed)
- AUDIT #10 → 🟢 (all 5 items closed: SecurityContext, image pin, RBAC, Redis auth, NetworkPolicy)
- operator/README.md ⚠️ warning REMOVED (operator now deployable per security-reviewer's original concerns)
- Full regression: 365+ passed (was 359; +5-10 from new tests)
- Wave B scripts ready to run when user powers droplet
- Commit as `V7.0.0-alpha.3 (Sprint 3 Wave A)` with DCO sign-off

## Phase 4 validation (Wave A)

3 parallel reviewers, all must approve:
- **architect** (opus): closes #9 + all 5 of #10? Wave B scripts are real (not stubs)?
- **security-reviewer** (sonnet): does the new RBAC use least privilege? Is the Redis password actually random (not hardcoded)? Are SecurityContexts complete (no missing fields)?
- **code-reviewer** (opus): Go style, YAML quality, no fabrication

## Wave B execution (separate autopilot session)

When user runs `/autopilot Wave B with IP <ip>`, I will:
1. SSH into the droplet, pip install requirements, clone repo + checkout V7.0.0-alpha.3
2. Run the 3 MI300X scripts in order, capturing outputs to `logs/`
3. Commit the logs to repo as `V7.0.0-alpha.4 (Sprint 3 Wave B — MI300X evidence)`
4. Update paper draft (paper/inv15_paper.tex) with measured numbers
5. Update AUDIT.md to mark 3.97× VRAM claim as "🟢 MEASURED on MI300X"
