# Apohara ContextForge — K8s Operator

> **⚠️ NOT PRODUCTION READY.** Sprint 2 shipped real Reconcile() logic + 4 controller tests, but the operator is missing all of the following before any shared-cluster deployment:
> - **SecurityContext** on worker + Redis pods (runAsNonRoot, readOnlyRootFilesystem, drop ALL capabilities)
> - **ServiceAccount + namespaced RBAC** (Role + RoleBinding under `operator/config/rbac/`)
> - **Redis authentication** (auto-provisioned Redis currently runs unauth on cluster ClusterIP)
> - **NetworkPolicy** (no traffic isolation between worker pods or to Redis)
> - **Image digest pinning** (`:latest` tag is mutable, supply-chain risk)
>
> These are explicit Sprint 3 work items. See `AUDIT.md` for the audit log and `.omc/plans/v7-roadmap.md` §1.B for the V7 roadmap context. Sprint 2 verified the reconciliation logic on a fake-client unit test (`go test ./controllers/...` 4/4 PASS) and a kind-skipped integration script — that is the scope of "shipped" right now.

## What this is

This directory is the V7 Sprint 2 Kubernetes operator for Apohara ContextForge. The operator lets users deploy an N-worker ContextForge cluster with a shared LMCache Redis backend via a single `kubectl apply`.

The Custom Resource Definition (CRD) is `ApohraContextForgeCluster` in group `contextforge.apohara.dev/v1alpha1`.

## Current status: reconciler logic written + unit-tested; not deployed

The `Reconcile()` controller logic now actively creates worker Deployments (matching `Spec.WorkerCount`), provisions an optional Redis sidecar when `Spec.LMCacheRedisUrl` is empty, and updates `Status.{ReadyWorkers, Phase}` from real pod readiness. 4 controller-runtime unit tests cover the happy path + the 4 main branches. `go vet ./...` passes.

The operator **binary is not built and not deployed** in Sprint 2. The integration test (`integration_test.sh`) only verifies that `kubectl apply` of the CRD + sample CR succeeds; it does NOT run the controller against the resource. Real end-to-end deployment is Sprint 3, gated on the security hardening listed at the top of this README.

## Validate Sprint 1 deliverables

```bash
bash operator/validate.sh
```

This script parses every YAML file under `operator/` and `charts/` with `python3 yaml.safe_load`. It exits 0 if all are valid, 1 otherwise.

To manually inspect the CRD:

```bash
python3 -c "
import yaml, json
doc = yaml.safe_load(open('operator/config/crd/bases/contextforge.apohara.dev_apoharacontextforgeclusters.yaml'))
print(doc['spec']['versions'][0]['name'])       # v1alpha1
print(list(doc['spec']['versions'][0]['schema']['openAPIV3Schema']['properties']['spec']['properties'].keys()))
"
```

To apply the CRD to a live cluster (requires `kubectl` access):

```bash
kubectl apply -f operator/config/crd/bases/contextforge.apohara.dev_apoharacontextforgeclusters.yaml
kubectl apply -f operator/config/samples/contextforge_v1alpha1_apoharacontextforgecluster.yaml
kubectl get apoharacontextforgeclusters
```

## Helm chart

The companion Helm chart lives at `charts/apohara-contextforge/`. See its [README](../charts/apohara-contextforge/README.md) for install instructions.

## Sprint 2 status

### What changed

| Deliverable | File | Notes |
|-------------|------|-------|
| Real Reconcile() logic | `controllers/apoharacontextforgecluster_controller.go` | Replaces the Sprint 1 "log and return" stub with full logic: Redis sidecar provisioning, worker Deployment management, and status updates |
| Unit test scaffold | `controllers/apoharacontextforgecluster_controller_test.go` | 4 table-style tests using `fake.NewClientBuilder()` covering worker creation, Redis auto-provision, Redis skip, and status update |
| Integration test script | `integration_test.sh` | kind-based bash script; skips gracefully with `exit 0` when kind is not installed |

### Reconcile() logic summary

The `Reconcile()` method now:

1. **Fetches the CR** — returns cleanly on `NotFound` (CR deleted between queue and execution).
2. **Provisions Redis sidecar** (when `spec.lmcacheRedisUrl` is empty) — creates a `Deployment` and `ClusterIP` `Service` named `<cluster>-redis` with owner references for automatic garbage collection.
3. **Provisions worker Deployment** — creates or updates `<cluster>-workers` with exactly `spec.workerCount` replicas, the configured image, and env vars `LMCACHE_REDIS_URL` + `MODEL`. Includes an HTTP readiness probe on `/health:8000`.
4. **Updates status** — counts ready pods via label selector, computes `Pending/Degraded/Ready` phase, and patches `status.readyWorkers`, `status.phase`, and the `Available` condition.
5. **Re-queues after 30 s** to catch external drift (manual pod deletes, image updates).

### Build status

The Reconcile() logic is **authored but not compiled** in Sprint 2.

- Go toolchain IS present in the dev environment (`go 1.26.0`).
- `go.mod` now declares the required `k8s.io/api`, `k8s.io/apimachinery`, and `sigs.k8s.io/controller-runtime` dependencies.
- `go get ./...` (or `go mod tidy`) must be run once to populate `go.sum` before `go build` will succeed.
- Running `cd operator && go mod tidy && go vet ./...` will produce the first compilation validation.
- Building and deploying the operator binary is a **Sprint 3** deliverable.

### Integration test

`operator/integration_test.sh` validates the CRD + sample CR flow against a live Kubernetes API server using kind.

```bash
bash operator/integration_test.sh
```

- **kind not installed**: prints `SKIP: kind not installed` and exits `0`.
- **kind installed**: creates a `apohara-contextforge-test` cluster, applies the CRD and sample CR, asserts the CR exists and `spec.workerCount=4`, then deletes the cluster on exit.

The controller binary is **not** deployed during the integration test (Sprint 3+). The test validates schema acceptance and kubectl round-trips only — not reconciliation behaviour.

## Roadmap

| Sprint | Target | Content |
|--------|--------|---------|
| Sprint 1 | Scaffold | CRD + types + controller skeleton + Helm chart |
| Sprint 2 (current) | Real reconciler | Worker Deployment + Redis sidecar + status updates (authored, not compiled) |
| Sprint 3 | Build + deploy | Compile operator binary, push image, deploy to kind with full reconciliation test |
| Sprint 4 | AMD AI Cloud | Reference deployment on MI300X cluster with $100 credit budget |

Full V7 plan: [`.omc/plans/v7-roadmap.md`](../.omc/plans/v7-roadmap.md) §4 Track 3.
