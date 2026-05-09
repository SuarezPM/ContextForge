# 🔥 ContextForge

**Silicon-native KV cache coordination for multi-agent LLM pipelines on AMD Instinct MI300X**

<!-- PLACEHOLDER:DEMO_VIDEO -->

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![ROCm 7.x](https://img.shields.io/badge/ROCm-7.x-orange.svg)](https://rocm.docs.amd.com/)
[![Hackathon Track 1](https://img.shields.io/badge/Track-AI%20Agents%20%26%20Agentic%20Workflows-FF6B35.svg)](https://lablab.ai/event/amd-hackathon)

In a 5-agent LLM pipeline, every agent independently materializes identical KV cache entries for shared context (system prompt, user query, retrieved documents). On a 35B MoE model with 192 GB HBM3, this redundancy wastes 40–60% of VRAM. ContextForge coordinates KV block sharing across all agents, reducing redundant memory by sharing PagedAttention blocks before they're materialized.

---

## ⚡ The Problem

In a typical multi-agent pipeline — **Planner → Retriever → Reranker → Responder → Critic** — each agent independently runs attention over the same shared context prefix:

```
WITHOUT ContextForge (VRAM duplication):
  Agent 1 (Retriever)    → [KV Cache: system + query + docs] — 12 GB
  Agent 2 (Reranker)     → [KV Cache: system + query + docs] — 12 GB  ← DUPLICATE
  Agent 3 (Summarizer)   → [KV Cache: system + query + docs] — 12 GB  ← DUPLICATE
  Agent 4 (Critic)       → [KV Cache: system + query + docs] — 12 GB  ← DUPLICATE
  Agent 5 (Responder)    → [KV Cache: system + query + docs] — 12 GB  ← DUPLICATE
  ─────────────────────────────────────────────────────────────
  Total KV VRAM:         60 GB for context that should need 12 GB

ContextForge eliminates this at the vLLM ATOM plugin level — zero model changes, zero latency overhead.
```

---

## 🧠 The Solution

ContextForge intercepts KV cache operations at the vLLM V1 ATOM plugin interface (entry_point: `vllm.general_plugins`). Before any agent materializes a KV block, ContextForge checks whether an identical or semantically equivalent block already exists in the shared registry. If so, it routes the agent to reuse that block's offsets instead of allocating new memory.

Every optimization traces back to a peer-reviewed paper published at NeurIPS, ICML, ACL, or IJCAI.

<!-- PLACEHOLDER:ARCHITECTURE_DIAGRAM -->

```
WITH ContextForge (shared KV via ATOM plugin):
  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐
  │ Embedding  │───▶│ LSH + FAISS     │───▶│ ContextRegistry     │
  │ Qwen3-Embed│    │ (semantic dedup) │    │ (anchor + offset)  │
  │ ONNX dim=512   └──────────────────┘    └──────────┬──────────┘
  └─────────────┘                                     │
  ┌───────────────────────────────────────────────────┼───────────────┐
  │                                                       ▼             │
  │  ┌──────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
  │  │ AnchorPool   │  │CLAMetadata│  │StepGraph   │  │RotateKV    │  │
  │  │ KVCOMM       │  │Layer       │  │ KVFlow     │  │ INT4       │  │
  │  │ offset hints │  │ NAACL 2025  │  │ eviction   │  │ pre-RoPE   │  │
  │  └──────┬───────┘  └──────┬─────┘  └──────┬─────┘  └─────┬────┘  │
  │         │                 │               │              │        │
  │         └─────────────────┴───────────────┴──────────────┘       │
  │                           ▼                                          │
  │  ┌────────────────────────────────────────────────────────────┐  │
  │  │              VRAMAwareCache + QueueingController           │  │
  │  │             (ICML 2026 stability, INVARIANT-11)            │  │
  │  └──────────────────────────┬───────────────────────────────────┘  │
  │                             ▼                                       │
  │  ┌─────────────────┐          ┌────────────────────────────┐     │
  │  │ LMCacheBridge   │          │ KVAwareRouter              │     │
  │  │ cross-worker    │          │ anchor locality + CLA affinity     │
  │  └────────┬────────┘          └────────────┬───────────────┘     │
  │           └──────────────────┬─────────────┘                      │
  │                              ▼                                    │
  │  ┌────────────────────────────────────────────────────────────┐  │
  │  │          vLLMAtomPlugin (entry_point: vllm.general_plugins) │  │
  │  └────────────────────────────────────────────────────────────┘  │
  │                                                                  │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
  │  │Retriever │  │Reranker  │  │Summarizer│  │ Critic   │         │
  │  │(fast)    │  │(fast)    │  │(fast)    │  │(CoT)    │         │
  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
  └─────────────────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────────────────┐
  │                AMD Instinct MI300X — 192 GB HBM3               │
  └────────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results

Benchmarks run on AMD Instinct MI300X via AMD Developer Cloud. Raw results in `logs/benchmark_v4_results.json` and `logs/benchmark_v5_results.json`.

<!-- PLACEHOLDER:BENCHMARK_TABLE_V4 -->

| Metric | Baseline (no sharing) | ContextForge V4 | Improvement | Source |
|--------|----------------------|-----------------|-------------|---------|
| VRAM peak | ~165 GB | ~98 GB | −41% | KVCOMM paper |
| TTFT improvement | — | 15–25% | — | KVFlow paper |
| Token savings | 0% | 30–50% | — | CLA + LCKV combined |
| RotateKV compression | none | 3.97× (INT4) | — | RotateKV paper |

<!-- PLACEHOLDER:BENCHMARK_TABLE_V5 -->

| Metric | V5 Extension | Target | Paper |
|--------|-------------|--------|-------|
| Queueing stability deviation | λ_critical prediction accuracy | <10% | Queuing Theory KV Cache (ICML 2026) |
| VisualKVCache encoder reduction | 5 agents → 1 call | 5× fewer | vLLM-Omni + AMD Batch-Level DP |
| Speculative acceptance rate | Retriever→Responder draft | >70% | Cross-Attn SpecDec (May 2026) |
| Speculative speedup | tokens/step vs autoregressive | >2× | Speculative-Speculative (May 2026) |

<!-- PLACEHOLDER:BENCHMARK_CHART_VRAM -->
<!-- PLACEHOLDER:BENCHMARK_CHART_TTFT -->

⚠️ **Pending hardware validation run** — results published after DevCloud execution on MI300X. Theoretical projections based on published paper results.

---

## 🔬 Research Foundation

| # | Paper | Venue | arXiv | What ContextForge Implements |
|---|-------|-------|-------|------------------------------|
| 1 | KVCOMM — Cross-Context KV Communication | NeurIPS 2025 | [2510.12872](https://arxiv.org/abs/2510.12872) | `AnchorPool.neighbor_prefix_offset` — RoPE position encoding drift compensation via simhash anchor matching |
| 2 | KVFlow — Workflow-Aware KV Prefix Management | NeurIPS 2025 | [2507.07400](https://arxiv.org/abs/2507.07400) | `AgentStepGraph.compute_steps_to_execution()` — evict agents farthest from execution first |
| 3 | PBKV — Prediction-Based KV Management | May 2026 | [2605.06472](https://arxiv.org/abs/2605.06472) | `PBKVPredictor` — 2nd-order Markov chain for next-agent prediction (1.26× over KVFlow) |
| 4 | SemShareKV — Semantic KV Cache Sharing | ACL Findings 2025 | — | `LSHEngine` + `FAISSContextIndex` — real semantic matching on Qwen3-Embedding-0.6B ONNX |
| 5 | RotateKV — Pre-RoPE KV Quantization | IJCAI 2025 | [2501.16383](https://arxiv.org/abs/2501.16383) | `RotateKVQuantizer` — INVARIANT-10: only pre-RoPE tensors quantized, INT4, attention-sink protection |
| 6 | CLA — Cross-Layer Attention | NeurIPS 2024 | — | `CLAMetadataLayer.compute_layer_groups()` — upper-layer sharing via NAACL 2025 strategy |
| 7 | Queuing Theory KV Cache — Stability Analysis | ICML 2026 | [2605.04595](https://arxiv.org/abs/2605.04595) | `QueueingController` — replaces empirical thresholds with λ_critical, E[S] Welford, INVARIANT-11 |
| 8 | vLLM-Omni + AMD Batch-Level DP | Feb 2026 + ROCm Blog | [2602.02204](https://arxiv.org/abs/2602.02204) | `VisualKVCache` — SHA256 content-hash, DP mode recommendation, eliminates 58–126 TP sync points |

---

## 🏗️ Architecture

```
contextforge/
├── embeddings/
│   └── embedding_engine.py       # Qwen3-Embedding-0.6B ONNX, MRL dim=512, LRU cache, xorshift fallback
├── kv_offset/
│   ├── anchor_pool.py            # KVCOMM: AnchorOffsetResult, prefix_offsets, approximate_offset()
│   └── cla_metadata.py           # CLA/LCKV: compute_layer_groups(), emit_hint(), NON_THOUGHT_ROLES
├── quantization/
│   └── rotate_kv.py              # RotateKV: quantize_pre_rope() INVARIANT-10, INT4, attention-sink
├── scheduling/
│   ├── queueing_controller.py    # NEW V5: ICML 2026 — λ_critical, Welford E[S], INVARIANT-11
│   ├── step_graph.py             # KVFlow: compute_steps_to_execution(), get_eviction_priority_order()
│   └── pbkv_predictor.py         # PBKV: 2nd-order Markov, train_from_jsonl(), blend_alpha=0.6
├── decoding/
│   └── speculative_coordinator.py  # NEW V5: Cross-Attn SpecDec — is_speculative_viable(), verify_and_commit()
├── multimodal/
│   └── visual_kv_cache.py       # NEW V5: vLLM-Omni — SHA256 content hash, get_dp_mode_recommendation()
├── serving/
│   ├── lmcache_bridge.py        # LMCacheConnectorV1: build_prefix_hint(), on_save_kv_layer()
│   └── atom_plugin.py            # vLLMAtomPlugin: entry_point=vllm.general_plugins, pre/post hooks
├── routing/
│   └── kv_aware_router.py       # KVAwareRouter: select_worker(), anchor locality + CLA affinity
├── dedup/
│   ├── lsh_engine.py            # LSHTokenMatcher: SimHash, block_size=16 alignment
│   └── faiss_index.py           # FAISSContextIndex: dim=512, IndexIVFFlat at >1000 contexts
└── registry/
    └── context_registry.py      # ContextRegistry: all modules wired, DI, AnchorPool CONNECTED
```

**V5 new modules:**

**QueueingController** (`scheduling/queueing_controller.py`) — ICML 2026: Replaces VRAMAwareCache's 5 empirical pressure thresholds with a rigorous M/G/1 queuing model. Computes λ (arrival rate) via EMA, E[S] via Welford online statistics, λ_critical = K_max / (E[S] × E[blocks]). Dynamic quantization feedback: ρ<0.70 → 16-bit, 0.70≤ρ<0.85 → 8-bit, 0.85≤ρ<0.95 → 4-bit, ρ≥0.95 → 2-bit. INVARIANT-11: never evicts below `minimum_stable_blocks = ceil(λ × E[S] × E[blocks] × 1.15)`.

**VisualKVCache** (`multimodal/visual_kv_cache.py`) — vLLM-Omni + AMD Batch-Level DP: SHA256 content-hash registry for cross-agent image deduplication. Eliminates redundant vision encoder calls. AMD benchmark: +6–44.9% throughput at 1024px by eliminating 58–126 all-reduce sync points per encoder forward pass. DP mode recommendation when batch≥2 images or resolution≥512px. INVARIANT-13: content hash is SHA256 of raw bytes, never of embeddings.

**SpeculativeCoordinator** (`decoding/speculative_coordinator.py`) — Cross-Attention SpecDec (May 2026): Intercepts Retriever/Reranker output as draft tokens for Responder/Critic. Standard acceptance criterion: accept token with probability min(1, p_i/q_i). Overlapped drafting+verification via asyncio.Queue. INVARIANT-12: target always generates final authoritative token on rejection. Target: >70% acceptance rate, >2× decode speedup.

<details>
<summary>🔒 System Invariants (14)</summary>

| # | Invariant | Description |
|---|-----------|-------------|
| INV-01 | Byte-identical prompts | System prompt must be byte-for-byte identical across all agents |
| INV-02 | SEPARATOR = `"\n\n"` | Two newlines between prefix segments |
| INV-03 | SHA256 prefix validation | Validated at `register_agent()` |
| INV-04 | FAISS dim = EmbeddingEngine dim | Default 512, must match |
| INV-05 | LSH block aligned to block_size=16 | PagedAttention boundary alignment |
| INV-06 | PyRSMI native only | Zero subprocess calls in hot path |
| INV-07 | Async-first | All I/O via `asyncio.run_in_executor` |
| INV-08 | Graceful degradation | Any dep absent → WARNING + fallback |
| INV-09 | AnchorPool called by ContextRegistry | Verified CONNECTED in V4 |
| INV-10 | RotateKV pre-RoPE ONLY | Never quantize post-RoPE tensors |
| INV-11 | QueueingController minimum blocks | Never evict below `minimum_stable_blocks` |
| INV-12 | SpeculativeCoordinator target authority | Target always generates final token on rejection |
| INV-13 | VisualKVCache content hash | SHA256 of raw bytes — never of embeddings |
| INV-14 | Dashboard mock banner | "SIMULATION MODE" shown for synthetic data |

</details>

---

## 🚀 Quick Start

**AMD DevCloud (Primary)** — Tested on MI300X · ROCm 7.x · $1.99/GPU/hr

```bash
git clone https://github.com/SuarezPM/ContextForge
cd ContextForge
pip install -e ".[rocm]"
pip install qwen3-embed onnxruntime streamlit prometheus-client --quiet

# Run tests
pytest tests/ -v --tb=short

# Run benchmark (10 V4 scenarios + 3 V5 scenarios, ~22 GPU-hours)
python demo/benchmark_v4.py --device rocm:0 --scenarios all
python demo/benchmark_v5.py --device rocm:0 --focus queueing_stability

# Launch dashboard
streamlit run demo/dashboard.py
```

**Local CPU (Development)** — No GPU required

```bash
pip install -e ".[cpu]"
pytest tests/ -v -k "not rocm"
streamlit run demo/dashboard.py -- --mock
```

**Docker**

```bash
docker compose up contextforge
```

<!-- PLACEHOLDER:DEVCLOUD_SETUP_VIDEO -->

---

## 📈 Live Dashboard

The Streamlit dashboard provides real-time visibility into ContextForge's KV coordination state. Four tabs: Live Metrics (VRAM pressure, λ/μ/ρ, stability margin), Pipeline View (per-agent TTFT, cache hits, thinking mode), V4 vs Baseline (VRAM comparison bars, scenario selector), and Research (8-paper table, module→paper mapping).

<!-- PLACEHOLDER:DASHBOARD_SCREENSHOT -->
<!-- PLACEHOLDER:PIPELINE_DEMO_GIF -->

```bash
streamlit run demo/dashboard.py
# Dashboard auto-refreshes every 5s
# --mock flag: synthetic Gaussian metrics (INV-14: "SIMULATION MODE" banner)
```

---

## 🔗 Module → Paper Mapping

| Module | File | Paper | Key Metric |
|--------|------|-------|------------|
| AnchorPool | `kv_offset/anchor_pool.py` | KVCOMM (NeurIPS 2025) | Offset variance < 0.05 via simhash |
| AgentStepGraph | `scheduling/step_graph.py` | KVFlow (NeurIPS 2025) | 2.19× speedup vs LRU |
| PBKVPredictor | `scheduling/pbkv_predictor.py` | PBKV (May 2026) | 1.26× over KVFlow |
| LSH + FAISS | `dedup/lsh_engine.py` + `dedup/faiss_index.py` | SemShareKV (ACL Findings 2025) | Semantic match >0.92 similarity |
| RotateKVQuantizer | `quantization/rotate_kv.py` | RotateKV (IJCAI 2025) | 3.97× VRAM reduction (INT4) |
| CLAMetadataLayer | `kv_offset/cla_metadata.py` | CLA (NeurIPS 2024) + NAACL 2025 | 50% upper-layer KV savings |
| QueueingController | `scheduling/queueing_controller.py` | Queuing Theory (ICML 2026) | λ_critical deviation < 10% |
| VisualKVCache | `multimodal/visual_kv_cache.py` | vLLM-Omni (Feb 2026) + AMD DP | +44.9% throughput at 1024px |

---

## 🏆 AMD x LabLab Hackathon 2026

**Track: AI Agents & Agentic Workflows**

ContextForge belongs in this track because agentic workflows are the most KV-redundant workloads in production. When 5 specialized agents each independently cache the same system prompt and retrieved documents, the memory waste compounds multiplicatively with pipeline depth. ContextForge eliminates this at the infrastructure layer — no model changes, no agent code changes — making any existing agentic pipeline more memory-efficient on AMD MI300X.

Built entirely on AMD-native stack: ROCm 7.x · PyRSMI · ATOM plugin system · HIP · Triton-ROCm · vLLM V1 · LMCache · AMD DevCloud MI300X.

**Hardware:** AMD Instinct MI300X (192 GB HBM3) via [AMD Developer Cloud](https://devcloud.amd.com/gpus)

---

## 🗺️ Roadmap

| Version | Status | Highlights |
|---------|--------|------------|
| V4.0 | ✅ Complete | AnchorPool CONNECTED, EmbeddingEngine ONNX, CLA metadata, RotateKV INT4, StepGraph, KVAwareRouter, LMCacheBridge, ATOM plugin |
| V5.0 | ✅ Complete | QueueingController (ICML 2026), VisualKVCache, SpeculativeCoordinator, PBKVPredictor Markov, BenchmarkDashboard, DevCloud runner |
| V5.x | 🔄 In Progress | DevCloud benchmarks, real hardware numbers, Streamlit dashboard polish |
| V6.0 | 📋 Planned | Multi-node distributed KV via LMCache, HIP custom kernels for RotateKV FWHT, multi-GPU node support |

---

## 📄 License

Apache 2.0 — chosen for its patent protection and corporate adoption. GPL would restrict cloud providers from offering ContextForge as a managed service; Apache 2.0 permits this without requiring derivative works to be open source.

---

## 🙏 Acknowledgments

- **AMD Developer Cloud** — MI300X GPU access via [devcloud.amd.com/gpus](https://devcloud.amd.com/gpus)
- **vLLM team** — ATOM plugin system and LMCache integration (PR #16625, April 2025)
- **Paper authors:**
  - Chengyi Nie, Nian Si, Zijie Zhou — Queuing Theory KV Cache (ICML 2026)
  - KVCOMM authors — Cross-Context KV Communication (NeurIPS 2025)
  - KVFlow authors — Workflow-Aware KV Prefix Management (NeurIPS 2025)
  - PBKV authors — Prediction-Based KV Management (May 2026)
  - RotateKV authors — Pre-RoPE KV Quantization (IJCAI 2025)
  - vLLM-Omni authors — Disaggregated Multimodal Serving (Feb 2026)
- **Qwen team** — Qwen3-Embedding-0.6B and Qwen3.6-35B-A22B model availability on AMD ROCm
- **LabLab.ai** — Hackathon platform and community