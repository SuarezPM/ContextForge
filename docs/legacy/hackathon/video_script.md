# ContextForge — 3-Minute Walkthrough

_Spoken script. ~150 words per minute. Five beats, ~3 minutes total._

## Problem

Modern multi-agent LLM pipelines waste GPU. Picture five agents — a retriever, a reranker, a summarizer, a critic, and a responder. They all see the same documents. They all carry the same instructions. They all pass the same chunks through to the model. And each one re-sends that context to the inference server. You pay for it in VRAM. You pay for it in time-to-first-token. You pay for it in billed tokens. Naive multi-agent is a hidden GPU tax — and almost no one prices it in.

## Flow

ContextForge sits between your agents and vLLM as a drop-in MCP layer. Your agent code does not change. Each agent calls `register_context`, and the SemanticDedupEngine — sentence-transformers with cosine similarity at or above 0.85 — finds the overlapping prefix. The CompressionCoordinator then picks one of four strategies: `apc_reuse` when prefixes are byte-exact, `compress` for unique tails, `compress_and_reuse` for the common mixed case, or `passthrough` when neither helps. vLLM with `--enable-prefix-caching` reuses the shared prefix's KV-cache. LLMLingua-2 compresses the unique tail at roughly half its original length. One decision boundary, two complementary primitives.

## Dashboard

The dashboard is Gradio with four tabs. Tab one — Live Demo — you type a query, press two buttons, and see the same pipeline run with and without ContextForge side by side, with five per-agent metrics: prompt tokens, TTFT, strategy, VRAM, and dedup hits. Tab two — Real-time Metrics — a two-second `gr.Timer` drives a VRAM bar, a TTFT timeseries, and a dedup-rate gauge. Tab three — Benchmark Results — renders the persisted `benchmark_results.json` with the cold and warm runs cleanly separated. Tab four — Architecture — shows the diagram and links the repo and the paper.

## Benchmark

The honest pitch is the curve, not a single number. We publish three: a cold run with the prefix cache empty, warm runs with one warmup discarded and two measurements averaged, and off runs with ContextForge bypassed. Reporting only the warm number is the industry's polite lie. We target at least a fifty-percent prompt-token reduction on warm runs, because prefix caching is a warmup phenomenon — and we show you both sides of it. The cold cost is real. The warm payoff is too.

## Why MI300X

This stack lives on the AMD Instinct MI300X for one reason: real estate. One hundred ninety-two gigabytes of HBM3 is enough to hold reusable KV-cache prefixes for many agents in parallel — without evicting them between calls. The ROCm-native vLLM wheels turn on prefix caching cleanly. The whole point of these savings is room to keep the cache hot. This hardware has it.
