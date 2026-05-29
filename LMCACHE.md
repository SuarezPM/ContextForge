# Multi-node KV-cache sharing with LMCache + ContextForge

V6.x #3 of the ContextForge roadmap: when you have more than one
inference worker, KV-cache reuse can't be local anymore. LMCache
(<https://github.com/LMCache/LMCache>) is the distributed KV store
ContextForge talks to for cross-worker / cross-node block sharing.
The integration lives at
[`apohara_context_forge/serving/lmcache_connector.py`](apohara_context_forge/serving/lmcache_connector.py)
(`LMCacheConnectorV2`); the V4-era stub at `lmcache_bridge.py` is kept
for backwards compatibility but new code should target V2.

## What V6.x #3 changes vs. V6.0 / V4.0

| Before (`LMCacheConnectorV1`)                               | After (`LMCacheConnectorV2`)                                       |
|-------------------------------------------------------------|--------------------------------------------------------------------|
| `on_save_kv_layer` and `on_load_kv_layer` logged but never called any LMCache API. | `store()` / `retrieve()` / `lookup()` invoke the real `LMCacheEngine` methods. |
| `is_active()` returned False whenever `lmcache_client=None`, but no path to make it True. | Connector builds the engine itself from `LMCacheConnectorConfig`, OR accepts a pre-built `engine=…`. |
| No prefetch logic.                                          | `prefetch()` returns per-block `cached_tokens` + `retrieved` so the ATOM pre-attention hook can decide whether to fetch or materialise. |
| Silent on errors.                                           | Every API method logs a single WARNING on failure and returns the documented null value — no exceptions ever propagate to the caller. |

## Architecture (multi-node story)

```
                                  ┌──────────────────────────┐
                                  │  LMCache Redis backend   │
                                  │   (one per cluster)      │
                                  └──────────┬───────────────┘
                                             │
                                             │  remote_url=redis://...
                                             │
        ┌────────────────────────┬───────────┴───────────┬────────────────────┐
        │                        │                       │                    │
        ▼                        ▼                       ▼                    ▼
   ┌─────────┐              ┌─────────┐             ┌─────────┐         ┌─────────┐
   │ vLLM W0 │              │ vLLM W1 │             │ vLLM W2 │         │ vLLM W3 │
   │ MI300X  │              │ MI300X  │             │ MI300X  │         │ MI300X  │
   │         │              │         │             │         │         │         │
   │ Apohara │              │ Apohara │             │ Apohara │         │ Apohara │
   │ ATOM    │              │ ATOM    │             │ ATOM    │         │ ATOM    │
   │ plugin  │              │ plugin  │             │ plugin  │         │ plugin  │
   │  └─ LMCacheConnectorV2 wires each plugin to the shared engine ──────────────┘
   └─────────┘              └─────────┘             └─────────┘         └─────────┘
```

Each worker runs the standard
[`apohara-vllm-plugin`](pypi/apohara-vllm-plugin/) (V6.x #1). The
plugin's `pre_attention_hook` consults the registry; on miss it
queries `LMCacheConnectorV2.lookup(tokens)` and, if hit, calls
`retrieve()` to fetch the KV cache from Redis instead of
materialising it locally. On store, the post-hook pushes the freshly-
materialised KV through `connector.store(...)` so the next worker
that sees the same tokens can read instead of recompute.

The end-to-end win: a 5-agent pipeline whose system prompt is shared
across all 5 agents pays the MI300X attention cost for that prefix
**once across the entire cluster**, not once per worker.

## One-time setup

### 1. Run an LMCache Redis backend

```bash
docker run -d --name lmcache-redis -p 6379:6379 redis:7-alpine
```

For production: use a managed Redis (AWS ElastiCache, GCP Memorystore,
etc.) or a dedicated Redis cluster — LMCache transfers KV chunks, not
small values, so memory bandwidth matters.

### 2. Install LMCache on every worker

```bash
pip install lmcache
```

The connector imports `lmcache` lazily, so if a worker is missing it
the connector enters honest-fallback mode and the worker just
materialises locally without crashing.

### 3. Wire ContextForge to LMCache

In your worker's startup code (e.g. inside the vLLM plugin's
`register()` callable, or wherever `vLLMAtomPlugin(...)` is
constructed):

```python
from apohara_vllm_plugin import vLLMAtomPlugin, ATOMConfig
from apohara_context_forge.serving.lmcache_connector import (
    LMCacheConnectorConfig,
    LMCacheConnectorV2,
)

# Build the LMCache connector once per worker.
lmcache = LMCacheConnectorV2(config=LMCacheConnectorConfig(
    instance_id=f"apohara-{worker_id}",
    chunk_size=256,
    local_device="cuda:0",
    remote_url="redis://lmcache-redis:6379",  # ← shared across workers
))

# Pass it to the plugin via the lsh_matcher slot or a custom
# adapter. The plugin's anchor-routing path then consults LMCache
# before falling back to local materialisation.
plugin = vLLMAtomPlugin(
    ATOMConfig(),
    lsh_matcher=lmcache,   # honest no-op without LMCache; real with
    jcr_gate=...,
    metrics=...,
)
plugin.initialize(worker_id, vllm_config={})
```

### 4. Verify cross-worker hits

After both workers have processed at least one request, the second
worker's `pre_attention_hook` should report:

```python
plugin.pre_attention_hook(...).get("anchor_match")
# {"block_ids": [...], "lookup_status": "pending_async", ...}
```

…and the connector's `get_stats()`:

```python
lmcache.get_stats()
# ILLUSTRATIVE example output — not a captured measurement; the
# connector is not yet on the hot path.
# {"active": True, "instance_id": "apohara-w1", "remote_url":
#  "redis://lmcache-redis:6379", "stores": 12, "retrieves_hit": 8,
#  "retrieves_miss": 4, "lookups": 12, ...}
```

A non-zero `retrieves_hit` is the signal that cross-worker reuse is
actually happening.

## Configuration knobs

| `LMCacheConnectorConfig` field | Default                        | Meaning |
|-------------------------------|--------------------------------|---------|
| `instance_id`                 | `"apohara-contextforge"`       | LMCache groups multiple engines by instance_id so workers can share. Use a per-worker suffix if you want per-worker accounting. |
| `chunk_size`                  | `256`                          | LMCache page size (tokens). Match vLLM's PagedAttention block size for best alignment; smaller chunks raise hit rate at the cost of metadata overhead. |
| `local_device`                | `"cpu"`                        | Where the engine stages tensors. `"cuda:0"` for on-GPU staging; `"cpu"` for RAM staging then DMA. |
| `remote_url`                  | `None`                         | Optional Redis URL. If set, all chunks are also persisted to Redis for cross-node access. If `None`, the engine is local-only (worker dies → cache lost). |
| `blocking_store`              | `False`                        | Async store by default — the worker doesn't block on the network. Set `True` for tests / strict consistency. |
| `blocking_retrieve`           | `True`                         | Block on retrieve (hot-path; you need the KV before attention runs). |

## Honest fallback

If `lmcache` is not importable (Python 3.14, ARM CPU without a wheel,
HF Spaces free tier, etc.) the connector logs a single WARNING and
runs in no-op mode: every API call returns the documented null
value, no exceptions are raised. Verify with:

```python
conn = LMCacheConnectorV2()
conn.is_active()  # False — no engine
conn.lookup(tokens=[1,2,3])  # 0 — honest miss
conn.store(tokens=[1,2,3], kv_tensors=...)  # None — honest no-op
```

This matches the V6.1 discipline (see [AUDIT.md](AUDIT.md)): the
state of the system matches what's reported. There is no path
through this code that claims "LMCache is active" while silently
doing nothing.

## Testing

The connector ships with 17 unit tests under
[`tests/test_lmcache_connector.py`](tests/test_lmcache_connector.py).
16 cover the in-process FakeEngine path (no lmcache install needed);
the 17th is gated by `@pytest.mark.skipif(not lmcache_installed)`
and actually builds a real `LMCacheEngine` on CI hosts that have the
package.

```bash
PYTHONPATH=. pytest tests/test_lmcache_connector.py -v
# → 16 passed, 1 skipped on Python 3.14
# → 17 passed       on Python 3.11-3.13 with `pip install lmcache`
```

## What's NOT V6.x #3 (yet)

The connector talks to LMCache; it does **not** yet:

* propagate `anchor_hash` / `cla_group` metadata into LMCache's
  chunk metadata field. The V4 `LMCacheConnectorV1` had that
  intention; V2 ships the connectivity first, the metadata
  marshalling is V6.x #3.1.
* RoPE-derotate retrieved KV blocks. The V2 returns whatever
  LMCache hands back; the caller is responsible for applying the
  AnchorPool offset hint. This is intentional — we want one place
  in the pipeline to own that math, and that's the ATOM plugin's
  pre-attention hook.
* register itself as the vLLM-side `KVConnector`. That requires
  vLLM ≥ 0.10 and the upstream connector ABI to stabilise; it lands
  in V6.x #3.2.

The V2 connector is the substrate. Each of those follow-ups is one
PR away from being honest about the existing data flow.

## The real KV-interception path is config-driven (2026-05-28)

KV interception in vLLM is **config-driven** via `--kv-transfer-config`
(LMCache), **NOT** attention hooks. vLLM never exposed a pre/post
attention-hook registry — so the ATOM plugin's `register()` no longer
probes for one (see [AUDIT.md](AUDIT.md) item 18). `register()` now just
constructs and initialises the plugin; the `PreAttentionHook` /
`PostAttentionHook` classes are unit-tested utilities, not runtime-cabled.

Wiring LMCache into a real worker therefore happens through vLLM's
`--kv-transfer-config` (Fase 1+), not by hand inside `register()`. The
setup snippet above shows the connector object; landing it on the real
config-driven path is the next phase of this track.
