# MI300X frontier-MoE + ContextForge end-to-end evidence (2026-05-26)

> All numbers measured at runtime on a Hot Aisle **AMD Instinct MI300X** (192 GB HBM3,
> ROCm 7.2.0 host). vLLM in Docker. Coordinator (ContextForge) host-side. Raw artifacts
> in this directory. No fabrication. Cumulative MI300X spend this session ≈ $3.6 of $17.34.

## 1. Frontier MoE serving on a SINGLE MI300X (vLLM)

| Model | Params | Quant | Image / vLLM | Serves 1-card? | HBM used | NIAH | Throughput |
|---|---|---|---|---|---|---|---|
| **Qwen3-30B-A3B-Instruct-2507** | 30B/3B MoE | FP8 | rocm7.0 / 0.11.2 | ✅ | ~186 GB | **12/12 → 174K tok** | **2667 tok/s** out |
| **Qwen3-Coder-Next** (hybrid Gated-DeltaNet) | 80B/3B MoE | FP8 | rocm7.13 / **0.19.1** | ✅ | ~175 GiB | **12/12 → 174K tok** | **2149 tok/s** out |
| **Qwen3-235B-A22B (GPTQ)** | 235B/22B MoE | INT4 W4A16 | rocm7.0 / 0.11.2 | ✅ | ~181 GiB | (40K ctx model) | slow INT4 decode |
| Qwen3-235B-A22B (FP8) | 235B/22B MoE | FP8 | — | ❌ | 221 GB weights > 192 GB | — | — |

Notes (our measurements):
- **Coder-Next-FP8 needs vLLM ≥ 0.17**: on vLLM 0.11.2 (rocm7.0 image) it crashed during CUDA-graph capture with a Triton kernel error (`arange's range must be a power of 2`) on its hybrid linear-attention layout. On the **vLLM 0.19.1 gfx94X image (rocm7.13)** it serves cleanly on the ROCm 7.2 host. Crash log: `vllm_coder_crash.log`.
- **235B-A22B-FP8 does not fit** one 192 GB card (221 GB FP8 weights); this is arithmetic, not a finding. INT4 (GPTQ W4A16) is the single-card path and serves (~181 GiB incl. FP8 KV). The original GPTQ checkpoint tops out at 40K context. Load-time crash log: `vllm_235b_crash.log`.
- INT4 GPTQ decode on ROCm is slow (no Marlin kernels); footprint/serve confirmed, throughput not optimized.

## 2. ContextForge end-to-end over LIVE MoE (the product test)

Ran the real coordinator (ContextRegistry dedup → CompressionCoordinator → LLMLingua-2 →
JCRSafetyGate INV-15 → FORGE-LEDGER) over a 5-agent shared-context workload, sending both
baseline and optimized contexts to the live vLLM endpoint. `scripts/mi300x_contextforge_e2e.py`.

| Model | Strategy | Baseline tok | ContextForge tok | **Token savings** | INV-15 fires | Ledger |
|---|---|---|---|---|---|---|
| Coder-Next-FP8 | compress ×5 | 5265 | 2926 | **44.4 %** | 1 (critic) | 5 certs, verify exit 0 |
| 30B-A3B-FP8 | compress ×5 | 5265 | 2926 | **44.4 %** | 1 (critic) | 5 certs, verify exit 0 |
| 235B-GPTQ-Int4 | compress_and_reuse ×5 | 5265 | 5205 | 1.1 % | 1 (critic) | 5 certs, verify exit 0 |

**What works (measured):**
- **Compression (LLMLingua-2): ~44 % prompt-token reduction** on real frontier-MoE inference (per-agent ~1034 → ~566 tokens, 2.23× on a probe). Confirmed on Coder-Next and 30B.
- **INV-15 JCR gate fires on the critic** (`use_dense=True`) on every model — over real inference.
- **FORGE-LEDGER**: every E2E produced 5 Z3-certified, SHA-256 hash-chained ledger entries; `ledger_cli verify` exit 0 on all three. Sample: `ledger_e2e_coder.jsonl`.

**Honest limitations (measured):**
- **Cross-agent dedup is degraded**: the EmbeddingEngine falls back to xorshift *pseudo-embeddings* because the real semantic embedder (`qwen3-embed`) is not installed. Strategy selection then becomes erratic: 30B/Coder picked `compress` (44 % saved); the 235B-GPTQ run spuriously picked `compress_and_reuse` (reused most of the context, compressed only a tiny tail → 1.1 %). With the real embedder the strategy choice would be reliable and the shared-prefix KV reuse would add prefill savings on top of compression.
- The compression % is prompt-token based; the dedup/`apc_reuse` benefit is a server-side KV/prefill saving not captured by prompt-token count.

## 3. Bugs found and fixed in ContextForge (while running the real stack)

1. **The compressor never worked.** `ContextCompressor` loaded the LLMLingua-2 token-classifier (`llmlingua-2-xlm-roberta-large-meetingbank`) but did **not** pass `use_llmlingua2=True`, so `PromptCompressor` ran the LLMLingua-1 perplexity path → `AttributeError: 'TokenClassifierOutput' has no attribute past_key_values` on every compress. Fix: `use_llmlingua2=True` (commit 476df4b). After the fix: 2.23× compression, 44 % E2E savings.
2. **Hardcoded CUDA device.** LLMLingua defaulted to CUDA and crashed (`Found no NVIDIA driver`) on a GPU-less / AMD coordinator host. Fix: CPU-default, env-overridable via `CONTEXTFORGE_COMPRESSOR_DEVICE` (commit 5d1e7d9).
3. **No chunking for the 512-token model limit.** Long contexts raised an index error. Fix: 160-word chunks (commit 95e1756).

## 4. Also validated this session (committed evidence)
- Full pytest suite **441 passed / 25 skipped** on MI300X/ROCm.
- FORGE-LEDGER **1,210-point Z3 sweep** verified + tamper-detected on-hardware (`logs_mi300x_p2/`).
- INT4 RotateKV codec **3.55×** reduction + FWHT-harmful reproduced; HBM3 triad **3.79 TB/s**.

## 5. Bottom line
ContextForge's **compression delivers ~44 % real token reduction on live frontier-MoE inference** (after fixing a critical bug that had left it non-functional), and its **INV-15 safety gate + tamper-evident certified ledger work end-to-end on three frontier MoE models** that each run on a single MI300X. The cross-agent dedup needs the real `qwen3-embed` embedder to be reliable; that is the clear next step.
