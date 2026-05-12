---
title: APOHARA ContextForge
emoji: 🧠
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
pinned: true
license: apache-2.0
short_description: KV-cache coordination for multi-agent LLM pipelines on AMD MI300X
tags:
  - llm
  - inference
  - kv-cache
  - multi-agent
  - vllm
  - amd
  - rocm
  - mi300x
---

# APOHARA · ContextForge — public benchmark sandbox

This Space is the **public, hosted version** of the ContextForge
benchmark dashboard. Run a 5-agent pipeline live in your browser
against the same Python implementation that backs the
[paper](https://doi.org/10.5281/zenodo.20114594) at the bottom of
this page, and see the live token-savings number that the JCR Safety
Gate (INV-15) and the LSH dedup registry produce together.

**Source:** [github.com/SuarezPM/Apohara_Context_Forge](https://github.com/SuarezPM/Apohara_Context_Forge)
· Apache-2.0 · DOI [10.5281/zenodo.20114594](https://doi.org/10.5281/zenodo.20114594)

## What this Space runs

* Real `ContextRegistry` (LSH + FAISS + VRAM-aware cache).
* Real `JCRSafetyGate` (INV-15 enforcement).
* Real `TokenDanceStorage` (Master-Mirror compression).
* Real Qwen3 tokenizer via `TokenCounter`.

vLLM is **not** running on this Space (HF free tier has no GPU); the
demo measures token deduplication + registry routing + INV-15 firing,
which are all CPU-only paths. The MI300X performance numbers in the
paper come from AMD DevCloud ATL1 runs (see
[`logs/benchmark_v6_final.txt`](https://github.com/SuarezPM/Apohara_Context_Forge/blob/main/logs/benchmark_v6_final.txt)
in the source repo).

## How to use

Type a multi-agent query, press **Run with ContextForge** and **Run
without ContextForge**, and compare:

* `tokens_before` / `tokens_after` (live token savings %)
* `avg_ttft_ms` (registration latency, real per-agent timing)
* `dedup_rate_pct`
* `[JCR Safety Gate / INV-15]` block — shows when the gate fires

Honest disclosure of what's real and what's synthesised on this
Space: see [AUDIT.md](https://github.com/SuarezPM/Apohara_Context_Forge/blob/main/AUDIT.md).
