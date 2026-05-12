# Apohara ContextForge — K8s Operator

## What this is

This directory is the **Sprint 1 scaffold** for the Apohara ContextForge Kubernetes operator. The operator will eventually let users deploy an N-worker ContextForge cluster with a shared LMCache Redis backend via a single `kubectl apply`.

The Custom Resource Definition (CRD) is `ApohraContextForgeCluster` in group `contextforge.apohara.dev/v1alpha1`.

## Current status: scaffold only

No reconciliation logic is active yet. The controller skeleton (`controllers/apoharacontextforgecluster_controller.go`) logs "reconciled" and returns. Real provisioning of worker pods, the headless service, and the optional Redis sidecar is targeted at Sprint 2.

The Go source files (`api/`, `controllers/`) are **authored but not compiled** in Sprint 1. They follow real kubebuilder patterns so Sprint 2 compilation starts clean.

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

## Roadmap

| Sprint | Target | Content |
|--------|--------|---------|
| Sprint 1 (current) | Scaffold | CRD + types + controller skeleton + Helm chart |
| Sprint 2 | Real reconciler | Worker Deployment + headless Service provisioning |
| Sprint 3 | LMCache sidecar | Auto-provision Redis pod when `lmcacheRedisUrl` is omitted |
| Sprint 4 | AMD AI Cloud | Reference deployment on MI300X cluster with $100 credit budget |

Full V7 plan: [`.omc/plans/v7-roadmap.md`](../.omc/plans/v7-roadmap.md) §4 Track 3.
