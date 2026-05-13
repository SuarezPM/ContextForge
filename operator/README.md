# Apohara ContextForge — K8s Operator

> **Note:** The operator is not yet deployed to a live cluster. Sprint 3 Wave A shipped the remaining security hardening items below. The one open item before shared-cluster deployment is **image digest pinning** (`:latest` tag is mutable; replace with `@sha256:...` before production use).
>
> Sprint 3 Wave A deliverables (this PR): SecurityContext (Track 2), RBAC (Track 3), Redis auth + NetworkPolicy (Track 4). See `AUDIT.md` for the full audit log.

## What this is

This directory is the V7 Sprint 3 Kubernetes operator for Apohara ContextForge. The operator lets users deploy an N-worker ContextForge cluster with a shared LMCache Redis backend via a single `kubectl apply`.

The Custom Resource Definition (CRD) is `ApohraContextForgeCluster` in group `contextforge.apohara.dev/v1alpha1`.

## Current status: Sprint 3 security hardening complete; binary not yet deployed

The `Reconcile()` controller logic actively creates worker Deployments, provisions an authenticated Redis sidecar, and updates `Status.{ReadyWorkers, Phase, RedisSecretName}`. 6 controller-runtime unit tests pass. `go vet ./...` passes.

The operator **binary is not built and not deployed** yet. Real end-to-end deployment is the next milestone, gated on image digest pinning.

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

## Sprint 3 deliverables

| Track | Deliverable | Files | Status |
|-------|-------------|-------|--------|
| Track 2 | SecurityContext (runAsNonRoot, readOnlyRootFilesystem, drop ALL) | `controller.go` | SHIPPED |
| Track 3 | ServiceAccount + namespaced RBAC (Role + RoleBinding) | `config/rbac/` | SHIPPED |
| Track 4 | Redis authentication (auto-provisioned Secret, crypto/rand password) | `controller.go`, `types.go` | SHIPPED |
| Track 4 | NetworkPolicy manifests for admin apply | `config/networkpolicy/` | SHIPPED |
| Track 4 | MI300X smoke-test scripts for Wave B | `scripts/` | SHIPPED |

### NetworkPolicy note

The manifests under `config/networkpolicy/` are **NOT auto-managed by the operator**. They are provisioned once by a cluster-admin:

```bash
kubectl apply -n <namespace> -f operator/config/networkpolicy/
```

The policies use `apohara.dev/role: worker` and `apohara.dev/role: redis` labels that the operator automatically stamps on its managed pods.

### Redis authentication

When `spec.lmcacheRedisUrl` is empty (auto-provisioned Redis), the operator:

1. Creates a Secret named `<cluster>-redis-auth` with a 32-character alphanumeric password generated via `crypto/rand`.
2. Injects `REDIS_PASSWORD` into both the Redis Deployment (via `--requirepass $(REDIS_PASSWORD)`) and worker pods (via `SecretKeyRef`).
3. Sets `status.redisSecretName` for operator visibility.
4. Never rotates the password on subsequent reconciles (stable across restarts).

Worker pods read `REDIS_PASSWORD` via `os.environ.get` in the LMCacheConnector.

## Roadmap

| Sprint | Target | Content |
|--------|--------|---------|
| Sprint 1 | Scaffold | CRD + types + controller skeleton + Helm chart |
| Sprint 2 | Real reconciler | Worker Deployment + Redis sidecar + status updates |
| Sprint 3 (current) | Security hardening | SecurityContext + RBAC + Redis auth + NetworkPolicy + MI300X Wave B scripts |
| Sprint 4 | Build + deploy | Compile operator binary, push image, deploy to kind with full reconciliation test |
| Sprint 5 | AMD AI Cloud | Reference deployment on MI300X cluster |

Full V7 plan: [`.omc/plans/v7-roadmap.md`](../.omc/plans/v7-roadmap.md) §4 Track 3.
