# ContextForge V4.0

**KV cache coordinator for multi-agent LLM pipelines on AMD Instinct MI300X, reducing VRAM by sharing PagedAttention blocks across agents using semantic deduplication, pre-RoPE quantization, and workflow-aware eviction.**

> Built for **AMD x LabLab Hackathon 2026** — Track 1: AI Agents & Agentic Workflows.
> Primary hardware: AMD Instinct MI300X via AMD Developer Cloud.

---

## One-Line Pitch

ContextForge reduces VRAM consumption by sharing KV cache prefixes across agents in multi-agent pipelines, using semantic deduplication (FAISS + LSH), KVCOMM-inspired anchor offset alignment, CLA metadata hints, and RotateKV pre-RoPE INT4 quantization.

---

## Architecture Diagram V4

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ContextForge V4 Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │ EmbeddingEng │───▶│ LSH Engine  │───▶│ FAISSContextIndex       │  │
│  │ Qwen3-Embed  │    │ SimHash     │    │ semantic ANN search     │  │
│  │ ONNX (512dim)│    │ block=16    │    │ dim=512                 │  │
│  └─────────────┘    └─────────────┘    └───────────┬─────────────┘  │
│                                                    │                 │
│                   ┌────────────────────────────────┘                 │
│                   ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                  ContextRegistry V4                             ││
│  │  ┌──────────────┐  ┌────────────┐  ┌──────────────┐  ┌────────┐ ││
│  │  │ AnchorPool  │  │CLAMetadata │  │AgentStepGraph│  │RotateKV│ ││
│  │  │ KVCOMM      │  │Layer       │  │ KVFlow       │  │ INT4   │ ││
│  │  │ offset hint │  │NAACL 2025  │  │ workflow     │  │pre-RoPE│ ││
│  │  └──────┬──────┘  └──────┬─────┘  └──────┬───────┘  └───┬────┘ ││
│  └─────────┼───────────────┼────────────────┼─────────────┼───────┘│
│            │               │                │             │        │
│            └───────────┬────┴────────────────┴─────────────┘        │
│                        ▼                                         │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │              VRAMAwareCache + QueueingController           │    │
│  │             (TASK-001 V5: stability-aware eviction)        │    │
│  └──────────────────────────┬────────────────────────────────┘    │
│                             │                                      │
│            ┌─────────────────┴──────────────────┐                  │
│            ▼                                    ▼                   │
│  ┌─────────────────┐               ┌─────────────────────────┐      │
│  │ LMCacheBridge   │               │ KVAwareRouter          │      │
│  │ cross-worker KV │               │ anchor locality routing │      │
│  │ offset hints    │               │ CLA affinity           │      │
│  └────────┬────────┘               └────────────┬────────────┘      │
│           │                                   │                     │
│           └─────────────┬─────────────────────┘                     │
│                         ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │              vLLMAtomPlugin (entry_point)                  │     │
│  │     PreAttentionHook + PostAttentionHook (INV-10)         │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │              AMD MI300X — 192 GB HBM3                      │     │
│  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐       │     │
│  │  │Retriever│ │Reranker│ │Summarizer│ │Critic  │ │Responder│ │     │
│  │  │(fast)  │ │(fast)  │ │(fast)   │ │(CoT)  │ │(CoT)   │       │     │
│  │  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘       │     │
│  └────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Research Grounding

| Paper | Venue | arXiv ID | What V4 Implements |
|-------|-------|----------|-------------------|
| **KVCOMM** — Cross-Context KV Communication | NeurIPS 2025 | 2510.12872 | `AnchorPool`: offset variance prediction via simhash, `approximate_offset()` |
| **KVFlow** — Prefix Caching for Workflows | NeurIPS 2025 | 2507.07400 | `AgentStepGraph`: workflow-aware eviction, `compute_steps_to_execution()` |
| **PBKV** — Prediction-Based KV Management | May 2026 | 2605.06472 | `PBKVPredictor` (stub V4, complete V5) |
| **SemShareKV** — Semantic LSH KV Sharing | ACL Findings 2025 | — | `LSHEngine`: SimHash on token IDs, FAISS ANN deduplication |
| **RotateKV** — Pre-RoPE INT4 Quantization | IJCAI 2025 | 2501.16383 | `RotateKVQuantizer`: pre-RoPE only (INV-10), INT4, attention-sink protection |
| **CLA** — Cross-Layer Attention | NeurIPS 2024 | — | `CLAMetadataLayer`: `compute_layer_groups()`, NAACL 2025 upper-layer strategy |
| **LCKV** — Layer-Condensed KV | ACL 2024 | — | CLA upper-layer sharing (top layers only) |
| **NAACL 2025** — Systematic CLA Study | NAACL 2025 | — | `NON_THOUGHT_ROLES` frozenset, upper-layer sharing beats bottom-layer |

---

## Tech Stack V4 (Corrected)

| Component | Technology |
|-----------|------------|
| Accelerator | AMD Instinct MI300X (192 GB HBM3, 8-GPU node) |
| Compute Stack | ROCm 7.x, HIP, Triton-ROCm, amdgpu gfx942 |
| LLM Engine | vLLM V1 (PagedAttention, block_size=16) |
| KV Cache | LMCache (vLLM upstream PR #16625, April 2025) |
| Embeddings | Qwen3-Embedding-0.6B ONNX (MRL, dim=512) |
| Vector Search | FAISS (IndexFlatIP, auto-upgrade to IVFFlat at >1000 ctx) |
| GPU Monitoring | PyRSMI native C bindings (zero subprocess, <1ms overhead) |
| Metrics | Prometheus (7 queueing gauges, full V4 stack) |
| API | FastAPI + Uvicorn |
| Protocol | AMD ROCm 7.x |

> **Note**: V4 does NOT use SBERT, Bun, or Gradio from v0.1.
> Those were replaced by Qwen3-Embed ONNX, async Python, and Streamlit dashboard.

---

## Module Tree V4

```
contextforge/
├── embeddings/
│   └── embedding_engine.py       # Qwen3-Embedding-0.6B ONNX, LRU, xorshift fallback
├── kv_offset/
│   ├── anchor_pool.py              # KVCOMM V4: AnchorOffsetResult, prefix_offsets
│   └── cla_metadata.py             # CLAMetadataLayer: NON_THOUGHT_ROLES, NAACL 2025
├── quantization/
│   └── rotate_kv.py               # RotateKVQuantizer: INV-10 pre-RoPE only, INT4
├── scheduling/
│   ├── step_graph.py              # AgentStepGraph: compute_steps_to_execution, DAG
│   └── pbkv_predictor.py          # PBKVPredictor STUB (production in V5)
├── serving/
│   ├── lmcache_bridge.py          # LMCacheConnectorV1, offset hints
│   ├── atom_plugin.py             # vLLMAtomPlugin: entry_point, pre/post hooks
│   └── vllm_client.py            # vLLM HTTP client
├── routing/
│   └── kv_aware_router.py        # KVAwareRouter: anchor locality + CLA affinity
├── dedup/
│   ├── lsh_engine.py              # LSHTokenMatcher: SimHash, block_size=16
│   └── faiss_index.py             # FAISSContextIndex: dim=512, IVFFlat upgrade
├── compression/
│   └── budget_manager.py          # CompressionBudgetManager: segment rates
├── normalization/
│   └── prefix_normalizer.py      # PrefixNormalizer: SEPARATOR="\n\n", SHA256
├── metrics/
│   ├── vram_monitor.py            # VRAMMonitor: PyRSMI, 5 modes, /sys fallback
│   └── prometheus_metrics.py     # Full Prometheus stack
└── registry/
    ├── context_registry.py        # ContextRegistry V4: all modules wired
    └── vram_aware_cache.py        # VRAMAwareCache: WORKFLOW_AWARE mode (6)
```

---

## Benchmark Results

> **Pending AMD DevCloud MI300X validation run.**
> Numbers will be filled in after `demo/run_devcloud.sh` completes on MI300X hardware.
> Do NOT use placeholder numbers — wait for real output from `demo/benchmark_v4.py`.

### Expected Ranges (from paper baselines)

| Metric | Baseline (no sharing) | ContextForge V4 | Source |
|--------|----------------------|-----------------|--------|
| VRAM peak | ~165 GB | ~98 GB (-41%) | KVCOMM paper |
| TTFT improvement | — | 15-25% | KVFlow paper |
| Token savings | 0% | 30-50% | CLA + LCKV combined |
| RotateKV compression | none | 3.97x (INT4) | RotateKV paper |

**Run benchmark:**
```bash
# On AMD DevCloud MI300X (ROCm 7.x)
cd ContextForge

# Install
pip install -e ".[rocm]" --quiet
pip install qwen3-embed onnxruntime streamlit prometheus-client --quiet

# Run tests
pytest tests/ -v --tb=short

# Run V4 benchmark (10 scenarios, ~22 GPU-hours if all scenarios)
python demo/benchmark_v4.py --device rocm:0 --scenarios all
```

---

## Installation

```bash
git clone https://github.com/SuarezPM/ContextForge
cd ContextForge

# AMD DevCloud MI300X
pip install -e ".[rocm]"

# Optional: enable Qwen3-Embedding-0.6B ONNX backend
pip install qwen3-embed onnxruntime

# Run tests
pytest tests/ -v --tb=short

# Run benchmark
python demo/benchmark_v4.py --device rocm:0 --scenarios all

# Run dashboard (after benchmark)
pip install streamlit prometheus-client
streamlit run demo/dashboard.py
```

---

## Invariant Registry (V4)

| # | Invariant | Description |
|---|-----------|-------------|
| INV-01 | Byte-identical system prompts | All agents must see byte-identical prefix |
| INV-02 | SEPARATOR = `"\n\n"` | Two newlines between prefix segments |
| INV-03 | SHA256 prefix validation | Validated at `register_agent()` |
| INV-04 | FAISS dim = EmbeddingEngine dim | Default 512, must match |
| INV-05 | LSH block aligned to block_size=16 | PagedAttention boundary |
| INV-06 | PyRSMI native only | Zero subprocess in hot path |
| INV-07 | Async-first | All I/O via `asyncio.run_in_executor` |
| INV-08 | Graceful degradation | Any dep absent → WARNING + fallback |
| INV-09 | AnchorPool called by ContextRegistry | V4 verified: CONNECTED |
| INV-10 | RotateKV pre-RoPE ONLY | Never quantize post-RoPE tensors |

---

## V5 Roadmap (In Progress)

| Task | Description | Status |
|------|-------------|--------|
| TASK-000 | README rewrite | ✅ DONE |
| TASK-001 | QueueingController (arXiv:2605.04595 ICML 2026) | 🔲 In progress |
| TASK-002 | VisualKVCache (vLLM-Omni, AMD Batch-Level DP) | 🔲 Pending |
| TASK-003 | SpeculativeCoordinator (cross-agent speculative decoding) | 🔲 Pending |
| TASK-004 | PBKVPredictor complete (Markov model) | 🔲 Pending |
| TASK-005 | BenchmarkDashboard (Streamlit) | 🔲 Pending |
| TASK-006 | DevCloud runner + benchmark_v5.py | 🔲 Pending |

---

## Hackathon Context

**Built for AMD x LabLab Hackathon 2026 — Track 1: AI Agents & Agentic Workflows.**

Primary hardware: AMD Instinct MI300X via AMD Developer Cloud.
AMD DevCloud allocation: ~$100 credits (MI300X x1, ROCm 7.x).
Cost estimate: ~$1.99/hr on MI300X single-GPU.

---

## License

MIT License. See [LICENSE](LICENSE) for details.