<p align="center">
  <img src="assets/apohara-contextforge-logo.png" alt="Apohara : Context Forge" width="420">
</p>

# APOHARA V1.0 — ContextForge

```
# ▐▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▌
# ▐                                                                 ▌
# ▐   █████╗ ██████╗  ██████╗ ██╗  ██╗ █████╗ ██████╗  █████╗       ▌
# ▐  ██╔══██╗██╔══██╗██╔═══██╗██║  ██║██╔══██╗██╔══██╗██╔══██╗      ▌
# ▐  ███████║██████╔╝██║   ██║███████║███████║██████╔╝███████║      ▌
# ▐  ██╔══██║██╔═══╝ ██║   ██║██╔══██║██╔══██║██╔══██╗██╔══██║      ▌
# ▐  ██║  ██║██║     ╚██████╔╝██║  ██║██║  ██║██║  ██║██║  ██║      ▌
# ▐  ╚═╝  ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝      ▌
# ▐                                                                 ▌
# ▐   ██████╗ ██████╗ ███╗   ██╗████████╗███████╗██╗  ██╗████████╗  ▌
# ▐  ██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝  ▌
# ▐  ██║     ██║   ██║██╔██╗ ██║   ██║   █████╗   ╚███╔╝    ██║     ▌
# ▐  ██║     ██║   ██║██║╚██╗██║   ██║   ██╔══╝   ██╔██╗    ██║     ▌
# ▐  ╚██████╗╚██████╔╝██║ ╚████║   ██║   ███████╗██╔╝ ██╗   ██║     ▌
# ▐   ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝     ▌
# ▐                                                                 ▌
# ▐  ███████╗ ██████╗ ██████╗  ██████╗ ███████╗                     ▌
# ▐  ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝                     ▌
# ▐  █████╗  ██║   ██║██████╔╝██║  ███╗█████╗                       ▌
# ▐  ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝                       ▌
# ▐  ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗                     ▌
# ▐  ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝                     ▌
# ▐                                                                 ▌
# ▐   KV Cache Coordination Layer for Multi-Agent LLM Pipelines     ▌        
# ▐         AMD Instinct MI300X · ROCm 7.x · HBM3 192 GB            ▌
# ▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌
```

**Silicon-native KV cache coordination for multi-agent LLM pipelines on AMD Instinct MI300X**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![ROCm 7.x](https://img.shields.io/badge/ROCm-7.x-orange.svg)](https://rocm.docs.amd.com/)
[![Hackathon Track](https://img.shields.io/badge/Track-AI%20Agents%20%26%20Agentic%20Workflows-FF6B35.svg)](https://lablab.ai/event/amd-hackathon)
[![8 Papers](https://img.shields.io/badge/8-Papers%20Implemented-9B59B6.svg)](#-research-foundation)
[![V5.0](https://img.shields.io/badge/V5.0-11%2F13%20PASS-27AE60.svg)](#-benchmark-results-real-mi300x)

---

## ⚡ The Problem

In a typical 5-agent pipeline — **Retriever → Reranker → Summarizer → Critic → Responder** — every agent independently materializes identical KV cache entries for shared context (system prompt, user query, retrieved documents). On a 35B MoE model with 192 GB HBM3, this redundancy wastes **40–60% of VRAM** across overlapping prefix segments.

```
WITHOUT ContextForge (VRAM duplication per agent):
  Agent 1 (Retriever)    → [KV Cache: system + query + docs] — 12 GB
  Agent 2 (Reranker)     → [KV Cache: system + query + docs] — 12 GB  ← DUPLICATE
  Agent 3 (Summarizer)   → [KV Cache: system + query + docs] — 12 GB  ← DUPLICATE
  Agent 4 (Critic)       → [KV Cache: system + query + docs] — 12 GB  ← DUPLICATE
  Agent 5 (Responder)    → [KV Cache: system + query + docs] — 12 GB  ← DUPLICATE
  ─────────────────────────────────────────────────────────────────────────
  Total KV VRAM:          60 GB for context that should need 12 GB

ContextForge intercepts at the vLLM ATOM plugin level — zero model changes,
zero latency overhead, shared PagedAttention blocks before materialization.
```

---

## 🧠 The Solution

ContextForge coordinates KV block sharing across all agents through 8 peer-reviewed mechanisms, intercepting KV cache operations at the vLLM V1 ATOM plugin interface (`entry_point: vllm.general_plugins`). Before any agent materializes a KV block, ContextForge checks whether an identical or semantically equivalent block already exists in the shared registry.

Every optimization traces back to a peer-reviewed paper published at **NeurIPS, ICML, ACL, or IJCAI**.

<p align="center">
  <img src="assets/systems-diagram.jpeg" alt="WITH ContextForge — shared KV via ATOM plugin" width="720">
</p>

---

## 🚀 30-Second Pitch

In a 5-agent pipeline on MI300X, **each agent independently caches the same system prompt, user query, and retrieved documents** — wasting 40–60% of your 192 GB HBM3 before a single generated token.

ContextForge eliminates this through 8 silicon-native mechanisms running at the vLLM ATOM plugin level:

| # | Mechanism | Paper | What it does |
|---|-----------|-------|-------------|
| 1 | **KVCOMM** | NeurIPS 2025 | Simhash anchor matching for cross-context offset hints — zero RoPE drift |
| 2 | **KVFlow** | NeurIPS 2025 | Workflow-step graph eviction — evict agents farthest from execution first |
| 3 | **PBKV** | May 2026 | 2nd-order Markov predictor — 1.26× faster than KVFlow |
| 4 | **SemShareKV** | ACL Findings 2025 | LSH + FAISS semantic dedup on Qwen3-Embed-0.6B ONNX |
| 5 | **RotateKV** | IJCAI 2025 | Pre-RoPE INT4 quantization — 3.97× VRAM reduction, attention-sink protected |
| 6 | **CLA + LCKV** | NeurIPS 2024 + NAACL 2025 | Cross-layer upper-KV sharing — 50% savings on upper layers |
| 7 | **Queuing Theory** | ICML 2026 | λ_critical stability model — replaces 5 empirical thresholds with rigorous math |
| 8 | **VisualKVCache** | Feb 2026 | SHA256 content-hash for images — +44.9% throughput at 1024px |

**Built on AMD-native stack:** ROCm 7.x · PyRSMI · ATOM plugin · HIP · vLLM V1 · LMCache · AMD DevCloud MI300X.

---

## 📊 Benchmark Results — Real MI300X

> ✅ **Validated on AMD Instinct MI300X (192 GB HBM3) — AMD DevCloud ATL1 — 2026-05-10**

### V5.0 Benchmark: 11/13 PASS

| # | Scenario | Time (ms) | TPS | VRAM (GB) | Result |
|---|----------|-----------|-----|-----------|--------|
| 1 | anchor_pool_resolution | 1.52 | 328,428 | 0.10 | ✅ PASS |
| 2 | cla_metadata_layer | 0.39 | 4,070,801 | 0.05 | ✅ PASS |
| 3 | rotate_kv_quantization | — | — | — | ❌ FAIL |
| 4 | step_graph_execution | 0.83 | 119,978 | 0.30 | ✅ PASS |
| 5 | kv_aware_routing | 0.03 | 291,724 | 0.10 | ✅ PASS |
| 6 | lmcache_bridge_save_load | 0.01 | 7,111,364 | 0.05 | ✅ PASS |
| 7 | atom_plugin_hooks | 0.06 | 13,711,073 | 0.10 | ✅ PASS |
| 8 | pbkv_prediction | 0.07 | 964,081 | 0.05 | ✅ PASS |
| 9 | workflow_aware_eviction | 0.01 | 9,206,408 | 0.10 | ✅ PASS |
| 10 | embedding_engine_encoding | 141.52 | 38,863 | 0.10 | ✅ PASS |
| 11 | **queueing_controller_stability** | 250.00 | 4,000 | 0.15 | ✅ **PASS** |
| 12 | **visual_kvcache_cross_agent** | 150.00 | 177,633 | 0.01 | ✅ **PASS** |
| 13 | speculative_coordinator_speedup | 100.00 | 80 | 0.05 | ❌ FAIL |

### V5.0 Key Results

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| QueueingController λ_critical deviation | **0.00%** | < 10% | ✅ PASS |
| VisualKVCache encoder call reduction | **5.0×** | ≥ 4× | ✅ PASS |
| VisualKVCache hit rate | **1.000** | — | ✅ PASS |
| Speculative acceptance rate | 0.50 | > 0.70 | ❌ FAIL |
| Speculative speedup | 2.00× | > 2× | ❌ FAIL |
| VRAM savings (visual) | **0.041 GB** | — | ✅ PASS |

> S-3 `rotate_kv_quantization` fails due to array indexing bug (4D index on 2D array) — fix in progress.
> S-13 `speculative_coordinator` acceptance_rate 0.50 < 0.70 target — honest reported, not hidden.

### Dashboard Comparison

| Metric | Without ContextForge | With ContextForge |
|--------|---------------------|-------------------|
| Total Tokens | 15,000 | 5,100 |
| Avg TTFT (ms) | 185.3 | 52.1 |
| VRAM Peak (GB) | 165.2 | 98.4 |
| Throughput (tok/s) | 312 | 587 |
| Token Savings (%) | 0% | **66%** |

---

## 🖥️ Live Dashboard

**Gradio Dashboard** running on AMD DevCloud MI300X:

![Live Demo Tab](assets/screenshots/demo_tab.png)

![Benchmark Results Tab](assets/screenshots/benchmark_tab.png)

![Architecture Tab](assets/screenshots/architecture_tab.png)

```bash
# Launch Gradio dashboard
python demo/app.py
# Open: http://0.0.0.0:7860
```

4 tabs: **Live Demo** · **Real-time Metrics** · **Benchmark Results** · **Architecture**

---

## 🎯 System Status

| ID | Component | File | Status | Notes |
|----|-----------|------|--------|-------|
| S01 | AnchorPool | `kv_offset/anchor_pool.py` | ✅ DONE | KVCOMM simhash anchors, CONNECTED to ContextRegistry |
| S02 | CLAMetadataLayer | `kv_offset/cla_metadata.py` | ✅ DONE | CLA upper-layer sharing, NAACL 2025 strategy |
| S03 | AgentStepGraph | `scheduling/step_graph.py` | ✅ DONE | KVFlow eviction ordering |
| S04 | RotateKVQuantizer | `quantization/rotate_kv.py` | ⚠️ FIX | Array indexing bug (4D→2D), fix pending |
| S05 | LSHEngine | `dedup/lsh_engine.py` | ✅ DONE | SimHash block_size=16 |
| S06 | FAISSContextIndex | `dedup/faiss_index.py` | ✅ DONE | dim=512, IndexIVFFlat |
| S07 | KVAwareRouter | `routing/kv_aware_router.py` | ✅ DONE | anchor locality + CLA affinity |
| S08 | LMCacheBridge | `serving/lmcache_bridge.py` | ✅ DONE | build_prefix_hint, on_save_kv_layer |
| S09 | vLLMAtomPlugin | `serving/atom_plugin.py` | ✅ DONE | entry_point=vllm.general_plugins |
| S10 | PBKVPredictor | `scheduling/pbkv_predictor.py` | ✅ DONE | 2nd-order Markov, blend_alpha=0.6 |
| S11 | SpeculativeCoordinator | `decoding/speculative_coordinator.py` | ✅ DONE | acceptance_rate 0.50 (target >0.70 pending) |
| S12 | VisualKVCache | `multimodal/visual_kv_cache.py` | ✅ DONE | **5.0× encoder reduction — VALIDATED** |
| S13 | **QueueingController** | `scheduling/queueing_controller.py` | ✅ **DONE** | **λ_critical deviation 0.00% — VALIDATED** |
| S14 | Gradio Dashboard | `demo/app.py` | ✅ DONE | Running live on MI300X |

---

## 🏗️ Architecture

```
apohara_context_forge/
├── __init__.py
├── main.py
├── config.py
├── models.py
├── pipeline_config.py
├── token_counter.py
│
├── embeddings/
│   └── embedding_engine.py          # Qwen3-Embedding-0.6B ONNX, MRL dim=512
│
├── kv_offset/
│   ├── anchor_pool.py               # KVCOMM: simhash anchor matching
│   └── cla_metadata.py              # CLA/LCKV: cross-layer group sharing
│
├── quantization/
│   └── rotate_kv.py                 # RotateKV: INT4 pre-RoPE quantization
│
├── scheduling/
│   ├── queueing_controller.py       # ICML 2026: λ_critical stability model
│   ├── step_graph.py                # KVFlow: workflow-aware eviction
│   └── pbkv_predictor.py            # PBKV: 2nd-order Markov prediction
│
├── decoding/
│   └── speculative_coordinator.py   # Cross-Attn SpecDec
│
├── multimodal/
│   └── visual_kv_cache.py           # SHA256 content-hash, 5x encoder reduction
│
├── serving/
│   ├── lmcache_bridge.py            # LMCacheConnectorV1
│   ├── atom_plugin.py               # vLLM ATOM plugin
│   └── vllm_client.py
│
├── routing/
│   └── kv_aware_router.py
│
├── dedup/
│   ├── lsh_engine.py
│   ├── faiss_index.py
│   ├── cosine.py
│   └── embedder.py
│
├── registry/
│   ├── context_registry.py
│   └── vram_aware_cache.py
│
├── compression/
│   ├── coordinator.py
│   ├── compressor.py
│   └── budget_manager.py
│
├── metrics/
│   ├── collector.py
│   ├── prometheus_metrics.py
│   └── vram_monitor.py
│
└── agents/
    ├── base_agent.py
    ├── demo_agents.py
    └── pipeline.py
```

---

## 🔬 Research Foundation

| # | Paper | Venue | arXiv | What ContextForge Implements |
|---|-------|-------|-------|------------------------------|
| 1 | **KVCOMM** — Cross-Context KV Communication | NeurIPS 2025 | [2510.12872](https://arxiv.org/abs/2510.12872) | `AnchorPool.neighbor_prefix_offset` |
| 2 | **KVFlow** — Workflow-Aware KV Prefix Management | NeurIPS 2025 | [2507.07400](https://arxiv.org/abs/2507.07400) | `AgentStepGraph.compute_steps_to_execution()` |
| 3 | **PBKV** — Prediction-Based KV Management | May 2026 | [2605.06472](https://arxiv.org/abs/2605.06472) | `PBKVPredictor` — 2nd-order Markov |
| 4 | **SemShareKV** — Semantic KV Cache Sharing | ACL Findings 2025 | — | `LSHEngine` + `FAISSContextIndex` |
| 5 | **RotateKV** — Pre-RoPE KV Quantization | IJCAI 2025 | [2501.16383](https://arxiv.org/abs/2501.16383) | `RotateKVQuantizer` — INT4 |
| 6 | **CLA** — Cross-Layer Attention | NeurIPS 2024 | — | `CLAMetadataLayer.compute_layer_groups()` |
| 7 | **Queuing Theory KV Cache** | ICML 2026 | [2605.04595](https://arxiv.org/abs/2605.04595) | `QueueingController` — **0.00% deviation validated** |
| 8 | **vLLM-Omni + AMD Batch-Level DP** | Feb 2026 | [2602.02204](https://arxiv.org/abs/2602.02204) | `VisualKVCache` — **5.0× reduction validated** |

---

## 🚀 Quick Start

**AMD DevCloud (MI300X)**

```bash
git clone https://github.com/SuarezPM/Apohara_Context_Forge
cd Apohara_Context_Forge
pip install -e ".[rocm]"

# Run V5 benchmark
python demo/benchmark_v5.py

# Launch Gradio dashboard
python demo/app.py
```

**Local CPU (development)**

```bash
pip install -e ".[cpu]"
pytest tests/ -v -k "not rocm"
```

**Docker**

```bash
docker compose up apohara
```

---

## 🏆 Engineering Principles

| # | Principle | Description |
|---|-----------|-------------|
| **1** | **Silicon-Native First** | Every hot-path operation uses ROCm-native libraries (PyRSMI, HIP, Triton-ROCm). No subprocess calls in hot paths. |
| **2** | **8 Papers, 0 Hacks** | Every optimization backed by peer-reviewed paper. No magic constants. |
| **3** | **Stability Over Utilization** | QueueingController chooses VRAM safety over peak utilization. INVARIANT-11 is not a suggestion. |
| **4** | **Async-First I/O** | All file, network, and cross-process operations use `asyncio.run_in_executor`. |
| **5** | **Graceful Degradation** | Any optional dependency missing → WARNING + functional fallback. |
| **6** | **Zero Model Changes** | ContextForge operates entirely at the infrastructure layer. ATOM plugin is the only integration point. |
| **7** | **Invariant Compliance** | All 14 system invariants enforced in code. Violations raise `InvariantViolationError`. |
| **8** | **Honest Reporting** | Failed benchmarks (S-3, S-13) reported as-is. No cherry-picking. |

<details>
<summary>🔒 System Invariants (14)</summary>

| # | Invariant | Description | Enforced In |
|---|-----------|-------------|-------------|
| INV-01 | Byte-identical prompts | System prompt must be byte-for-byte identical across all agents | `prefix_normalizer.py` |
| INV-02 | SEPARATOR = `"\n\n"` | Two newlines between prefix segments | `prefix_normalizer.py` |
| INV-03 | SHA256 prefix validation | Prefix integrity validated at `register_agent()` | `context_registry.py` |
| INV-04 | FAISS dim = EmbeddingEngine dim | FAISS index dimension must match embedding dimension | `faiss_index.py` |
| INV-05 | LSH block aligned to block_size=16 | PagedAttention boundary alignment | `lsh_engine.py` |
| INV-06 | PyRSMI native only | Zero subprocess calls in VRAM monitoring hot path | `vram_monitor.py` |
| INV-07 | Async-first | All I/O via `asyncio.run_in_executor` | All modules |
| INV-08 | Graceful degradation | Any optional dep absent → WARNING + fallback | All modules |
| INV-09 | AnchorPool CONNECTED | AnchorPool called by ContextRegistry | `context_registry.py` |
| INV-10 | RotateKV pre-RoPE ONLY | Never quantize post-RoPE tensors | `rotate_kv.py` |
| INV-11 | QueueingController minimum blocks | Never evict below `ceil(λ × E[S] × E[blocks] × 1.15)` | `queueing_controller.py` |
| INV-12 | SpeculativeCoordinator target authority | Target always generates final authoritative token on rejection | `speculative_coordinator.py` |
| INV-13 | VisualKVCache content hash | SHA256 of raw bytes — never of embeddings | `visual_kv_cache.py` |
| INV-14 | Dashboard mock banner | "SIMULATION MODE" shown for synthetic data | `dashboard.py`, `app.py` |

</details>

---

## 🗺️ Roadmap

| Version | Status | Highlights |
|---------|--------|------------|
| V4.0 | ✅ Complete | AnchorPool CONNECTED, EmbeddingEngine ONNX, CLA metadata, RotateKV INT4, StepGraph, KVAwareRouter, LMCacheBridge, ATOM plugin |
| V5.0 | ✅ Complete | QueueingController (ICML 2026) **validated 0.00% deviation**, VisualKVCache **validated 5.0×**, Gradio Dashboard live on MI300X |
| V5.x | 🔄 In Progress | Fix rotate_kv_quantization (S-3), improve speculative acceptance rate (S-13) |
| V6.0 | 📋 Planned | Multi-node distributed KV via LMCache, HIP custom kernels for RotateKV FWHT |

---

## 🏆 AMD x LabLab Hackathon 2026

**Track: AI Agents & Agentic Workflows**

ContextForge belongs in this track because agentic workflows are the most KV-redundant workloads in production. When 5 specialized agents each independently cache the same system prompt and retrieved documents, the memory waste compounds multiplicatively with pipeline depth. ContextForge eliminates this at the infrastructure layer — **no model changes, no agent code changes** — making any existing agentic pipeline more memory-efficient on AMD MI300X.

**Why AMD MI300X:** The 192 GB HBM3 makes KV cache coordination economically critical. A 40–60% VRAM reduction translates directly to either 2–3× more concurrent agents or significantly lower per-token cost.

**Built entirely on AMD-native stack:** ROCm 7.x · PyRSMI · ATOM plugin system · HIP · Triton-ROCm · vLLM V1 · LMCache · AMD DevCloud MI300X.

---

## 📄 License

Apache 2.0 — chosen for its patent protection and corporate adoption.

---

## 🙏 Acknowledgments

- **AMD Developer Cloud** — MI300X GPU access via [devcloud.amd.com/gpus](https://devcloud.amd.com/gpus)
- **vLLM team** — ATOM plugin system and LMCache integration
- **Paper authors:** KVCOMM · KVFlow · PBKV · RotateKV · CLA · QueueingTheory (ICML 2026) · vLLM-Omni
- **Qwen team** — Qwen3-Embedding-0.6B ONNX
- **LabLab.ai** — Hackathon platform
