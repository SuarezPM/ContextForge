# apohara-contextforge Helm Chart

Helm chart for deploying an N-worker [Apohara ContextForge](https://github.com/SuarezPM/Apohara_Context_Forge) cluster with a shared LMCache Redis backend on Kubernetes.

## Prerequisites

- Kubernetes 1.24+
- Helm 3.10+
- Nodes labelled `apohara.dev/gpu-type: mi300x` (or h100/a100 per `values.yaml`)

## Installation

```bash
# Install with default values (4 workers, auto-provisioned Redis, mi300x GPU)
helm install my-cluster charts/apohara-contextforge

# Install with a custom model and existing Redis
helm install my-cluster charts/apohara-contextforge \
  --set model=meta-llama/Llama-3-70b \
  --set lmcacheRedisUrl=redis://my-redis:6379 \
  --set workerCount=8

# Dry-run to preview manifests
helm install my-cluster charts/apohara-contextforge --dry-run --debug
```

## Key values

| Key | Default | Description |
|-----|---------|-------------|
| `workerCount` | `4` | Number of vLLM+plugin worker pods |
| `model` | `meta-llama/Llama-3-8b` | HuggingFace model identifier |
| `lmcacheRedisUrl` | `""` | Redis URL (empty = auto-provision sidecar) |
| `gpuType` | `mi300x` | GPU node selector (`mi300x`, `h100`, `a100`) |
| `image.repository` | `ghcr.io/suarezpm/apohara-contextforge` | Worker image repository |
| `redis.enabled` | `true` | Provision an in-cluster Redis pod for LMCache |

## Architecture

Each worker pod runs `apohara-vllm-plugin` and connects to the shared LMCache Redis backend. The headless service enables direct pod-to-pod discovery. See [LMCACHE.md](../../LMCACHE.md) for the full multi-node architecture.

## Status

Sprint 1 scaffold. The Helm chart generates valid Kubernetes manifests but the operator reconciliation loop that enforces desired state is in Sprint 2+. See [`.omc/plans/v7-roadmap.md`](../../.omc/plans/v7-roadmap.md) for the full roadmap.
