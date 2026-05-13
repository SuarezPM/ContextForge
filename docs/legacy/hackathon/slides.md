---
marp: true
theme: default
paginate: true
---

# Problem

**Multi-agent LLM redundancy is a hidden GPU tax.**

A modern pipeline runs five agents on the same context:

- **retriever** · **reranker** · **summarizer** · **critic** · **responder**
- Same documents. Same instructions. Same passed-through chunks.
- Each agent re-sends that context to the inference server.

The bill arrives three ways:

- **VRAM** — every duplicate prefix held resident.
- **TTFT** — every agent pays the full prefill cost.
- **Tokens billed** — linear in agent count, not in unique content.

Naive multi-agent is a hidden GPU tax — and almost no one prices it in.

---

# Solution

**Drop a layer between your agents and vLLM. Don't change agent code.**

Three primitives, one decision boundary:

- **Semantic overlap detection** — sentence-transformers (MiniLM), cosine ≥ 0.85, finds shared prefixes even when agents drift textually.
- **KV-cache reuse** — vLLM `--enable-prefix-caching` skips prefill on byte-exact prefixes recovered by the dedup engine.
- **Tail compression** — LLMLingua-2 at rate ≈ 0.5 on the unique remainder, with an OOM-safe ROCm fallback.

Routed per agent through one of four strategies, picked at request time:

| Strategy              | When                                       |
| --------------------- | ------------------------------------------ |
| `apc_reuse`           | byte-exact prefix already cached           |
| `compress`            | unique tail, no overlap                    |
| `compress_and_reuse`  | partial overlap + compressible tail        |
| `passthrough`         | nothing to gain — never make it worse      |

MCP surface keeps your agent stack unchanged. Add one client. Get the savings.

---

# Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Your Multi-Agent Pipeline                        │
│                                                                          │
│   ┌──────────┐ ┌──────────┐ ┌────────────┐ ┌────────┐ ┌───────────┐     │
│   │retriever │ │reranker  │ │summarizer  │ │critic  │ │responder  │     │
│   └────┬─────┘ └────┬─────┘ └─────┬──────┘ └───┬────┘ └─────┬─────┘     │
│        │            │             │            │            │            │
│        │   register_context()  +  get_optimized_context()                │
│        └────────────┴─────────────┴────────────┴────────────┘            │
│                                  │                                       │
└──────────────────────────────────┼───────────────────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                        ContextForge   (FastAPI :8001)                     │
│                                                                          │
│   ┌─────────────────────┐   ┌──────────────────────┐                    │
│   │ ContextRegistry     │   │ SemanticDedupEngine  │                    │
│   │ (TTL, lock-free)    │──▶│ MiniLM embeddings    │                    │
│   └─────────────────────┘   │ cosine ≥ 0.85        │                    │
│                             └──────────┬───────────┘                    │
│                                        ▼                                │
│                       ┌──────────────────────────────┐                  │
│                       │   CompressionCoordinator     │                  │
│                       │                              │                  │
│                       │   apc_reuse                  │                  │
│                       │   compress                   │                  │
│                       │   compress_and_reuse         │                  │
│                       │   passthrough                │                  │
│                       └──────────────┬───────────────┘                  │
│                                      │                                  │
│                       ┌──────────────▼───────────────┐                  │
│                       │   ContextCompressor          │                  │
│                       │   LLMLingua-2 (rate ≈ 0.5)   │                  │
│                       │   OOM-safe ROCm fallback     │                  │
│                       └──────────────┬───────────────┘                  │
└──────────────────────────────────────┼──────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      vLLM   (--enable-prefix-caching)                     │
│                       AMD Instinct MI300X · ROCm 6.1                      │
│                                                                          │
│         Prefix KV-cache reuse  ──┐    Tail recompute (compressed)        │
│                                  └──▶ generation                         │
└──────────────────────────────────────────────────────────────────────────┘
```

**ContextForge** is a FastAPI MCP server on `:8001` exposing `/tools/*`, `/health`, `/metrics/snapshot`. **vLLM** lives behind `--enable-prefix-caching`; ContextForge owns the dedup + compression + routing decision. **Same diagram across the README, this deck, and Tab 4 of the dashboard.**

---

# Demo

**Gradio 5 dashboard. Four tabs. One pane of glass.**

- **Tab 1 — Live Demo** — type a query, press two buttons, see the same pipeline run **with** and **without** ContextForge side-by-side. Five per-agent metrics: prompt tokens, TTFT, strategy, VRAM, dedup hits.
- **Tab 2 — Real-time Metrics** — Plotly charts driven by a 2-second `gr.Timer`: VRAM bar, TTFT-over-time line, dedup-rate gauge, degradations list (operators see _why_, not just a 500).
- **Tab 3 — Benchmark Results** — the persisted `benchmark_results.json` rendered as a table, downloadable artifact.
- **Tab 4 — Architecture** — the diagram from slide 3 plus repo + paper links.

**Hosted demo:** [`huggingface.co/spaces/<TBD-after-deploy>`](https://huggingface.co/spaces/)

Local fallback if the Space is cold:

```bash
python demo/app.py        # → http://0.0.0.0:7860
```

---

# Numbers

**The honest pitch is the curve, not a single number.**

We publish three runs:

- **Cold run** — first invocation, prefix cache empty.
- **Warm runs** — 1 warmup discarded, next 2 averaged.
- **Off runs** — 3 runs of the identical pipeline with ContextForge bypassed.

| Metric                        | Without | With   | Delta            |
| ----------------------------- | ------- | ------ | ---------------- |
| Prompt tokens (5-agent total) | _TBD_   | _TBD_  | **≥ 50 %** ↓     |
| TTFT (mean per agent)         | _TBD_   | _TBD_  | _TBD_ ↓          |
| VRAM peak                     | _TBD_   | _TBD_  | _TBD_ ↓          |

_Cells fill from the live MI300X `benchmark_results.json` row; the README is the canonical source._

**Why both curves?** Prefix caching is a warmup phenomenon. Reporting only the warm number is the industry's polite lie. We show the cold cost, then the warm payoff. The pitch is the curve, not a single number.

```bash
python demo/benchmark.py --warmup 1
cat benchmark_results.json | jq .totals
```
