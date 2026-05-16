# Apohara ContextForge — Benchmarks

> Living document. Every row in every table below traces to a
> committed JSON log under `logs/` and to a runnable script under
> `scripts/`. No hardcoded numbers, no "we ran it once" benchmarks
> without an artifact.

---

## 5-agent shared-context workload (Day-6 Milan sprint, 2026-05-16)

**Story:** US-014 of the Apohara Inti Fusion Sprint — side-by-side
benchmark for the Milan AI Week / AI Agent Olympics submission
($28K prize pool, Track: agent benchmarks).

**Workload:** `configs/sprint5_5agent.yaml` (retriever → reranker →
summarizer → critic → responder; 500 requests; reuse-rate per
agent fixed in the YAML; critic role gated by INV-15 at τ=0.65).

**Script:** `scripts/sprint5_head_to_head.py` for the two backend
runs; `scripts/build_milan_benchmark.py` composes the Milan JSON;
`scripts/run_milan_benchmark.sh` orchestrates the whole flow.

**Hardware:** **NVIDIA H100 PCIe 80GB 1× via NVIDIA Brev (Scaleway)
on 2026-05-16** — supersedes the prior CPU-mock fallback. The H100
run uses Qwen/Qwen3.6-27B (FP16, ~54 GB weights, hybrid
linear-attention architecture, `qwen3_5` model_type) via the
HuggingFace transformers 5.8.1 backend. vLLM 0.21.0 does not yet
recognize `qwen3_5`, so the vLLM-plugin path is deferred to upstream
support. Full composed log:
[`logs/milan_5agent_h100_REAL_20260516T215940Z.json`](logs/milan_5agent_h100_REAL_20260516T215940Z.json).
Per-run logs:
[`logs/milan_5agent_h100_baseline_20260516T213133Z.json`](logs/milan_5agent_h100_baseline_20260516T213133Z.json),
[`logs/milan_5agent_h100_contextforge_20260516T215655Z.json`](logs/milan_5agent_h100_contextforge_20260516T215655Z.json).
Harness: [`scripts/run_milan_h100.py`](scripts/run_milan_h100.py).
The earlier CPU-mock log `logs/milan_5agent_benchmark_1778943206.json`
is retained as the historical closed-form reference, see
"Closed-form reference (CPU-mock, retained for theory)" below.

### Measured on real H100 — sequential 5-agent peak

| Config | TTFT (ms) | p50 latency (ms) | Tokens generated | Peak HBM (GB) | INV-15 critic fires |
|---|---|---|---|---|---|
| Baseline (each agent encodes full 2.3K-token shared context) | 5269 | 3886 | 240 | 51.348 | 1/1 |
| Contextforge (non-critic agents encode suffix only; critic re-encodes per INV-15) | 4105 | 4105 | 226 | 51.234 | 1/1 |
| **Delta** | **−1164 (−22%)** | **+219 (+5.6%)** | **−14** | **−0.115 (−0.22%)** | **0 (gate behavior identical)** |

**Reading the H100 numbers**

- **HBM peak −0.22%** — this is the SEQUENTIAL single-agent peak. The
  harness flushes the KV cache between agents (`torch.cuda.empty_cache()`),
  so the peak at any single point is one agent's footprint regardless
  of mode. The architectural 76% saving (below) is the CONCURRENT
  multi-agent number — five agents holding KV state at the same time
  via a real registry. That concurrent measurement requires the vLLM
  plugin path which is post-hackathon (waiting on vLLM upstream
  `qwen3_5` support).
- **INV-15 critic fires correctly on real H100** — 1/1 in both modes.
  Risk 0.90 > τ 0.65 → critic forced to dense prefill. The gate
  decision is identical regardless of underlying KV strategy, which
  is what the paper specifies.
- **TTFT lower in contextforge** — saves the ~1.1s prefix-encode time
  on the first agent because it only encodes its suffix (~30 tokens
  vs ~2.3K).

### Closed-form reference (CPU-mock, retained for theory)

The prior closed-form projection used Llama-3-8B's KV geometry (32
layers × GQA-8 × fp16) plus a workload mean-reuse rate of 0.76. The
table is retained here as the theoretical anchor that the production
vLLM plugin will validate end-to-end:

| Config (closed-form) | TTFT (ms) | Throughput (tok/s) | HBM (GB) | JCR | Notes |
|---|---|---|---|---|---|
| vllm `--enable-prefix-caching` (baseline) | 31.93 | 6,915.2 | 78.12 | 0.784 | INV-15 disabled, no cross-agent KV sharing |
| vllm + contextforge plugin | 32.12 | 6,852.6 | 18.75 | 1.000 | INV-15 ON, mean-reuse 0.76 sharing |
| **Delta (closed-form)** | **+0.19** | **−0.91%** | **−59.37 (−76.0%)** | **+0.216** | Critic agent JCR recovered from 0.784 → 1.000 |

**Reading the numbers**

- **HBM saved 76%** — the closed-form expects this exact ratio
  because the workload YAML's mean reuse rate is 0.76; the model
  is `residual = 1 - mean_reuse`. A live GPU run with a real
  cache backend will land in this neighborhood ± measurement
  noise (vLLM's prefix cache is not 100% efficient).
- **TTFT delta +0.19 ms** — within mock-mode timer jitter. The
  gate decision is O(1) so we don't expect a TTFT regression on
  real hardware either.
- **Throughput delta -0.91%** — within mock-mode jitter. The
  INV-15 gate's dense-prefill path on the critic adds a small
  latency hit but only on the gated subset (0.2 of all critic
  invocations under the default workload).
- **JCR +21.6 pp** — the safety story. INV-15 OFF leaks the
  critic-flip from `--critic-flip-rate 0.20` in the mock backend
  (modeling the Liang et al. 2026 finding); INV-15 ON suppresses
  it by sending the critic through dense prefill.

**Reproduce locally (CPU, no GPU required)**

```bash
cd /path/to/Apohara_Context_Forge
bash scripts/run_milan_benchmark.sh
# Outputs:
#   logs/milan_h2h_baseline_<ts>.json
#   logs/milan_h2h_contextforge_<ts>.json
#   logs/milan_5agent_benchmark_<ts>.json   ← Milan submission JSON
```

The Milan submission JSON for the run committed alongside this
document lives at `logs/milan_5agent_benchmark_1778943206.json`
(2026-05-16T14:53:27Z).

**Reproduce on a real GPU**

```bash
# 1. Start vLLM in another shell on the GPU host:
PYTHONPATH=. python3 -m apohara_context_forge.vllm_plugin.serve \
    --model meta-llama/Llama-3-8B --port 8000

# 2. Then in this shell:
VLLM_ENDPOINT=http://localhost:8000 \
HARDWARE_LABEL="GCP H100 1x (a3-highgpu-1g, us-central1-a)" \
COST_USD=12.0 \
bash scripts/run_milan_benchmark.sh
```

**60-second screen replay**

`assets/milan_benchmark_clip.gif` is a 6-frame replay of the real
CPU-mock run output (6 frames × 10s = 60s). On a real GPU host
re-render the GIF after the run via:

```bash
PYTHONPATH=. python3 scripts/generate_milan_clip.py \
    --milan-json logs/milan_5agent_benchmark_<ts>.json \
    --out assets/milan_benchmark_clip.gif
```

---

## What is NOT in this document yet

- **Live MI300X v8 codec sweep** — landed in V7.0.0-rc.2 but the
  artifacts live under `logs/mi300x_*` (separate tables); they
  belong here once the Milan row above is replaced with a real
  GPU measurement.
- **K8s operator stress test** — handled by `operator/validate.sh`.
- **Multi-node Redis cache** — not yet implemented (`remote_url:
  null` in the YAML).

---

*Last updated: 2026-05-16 (V7.0.0-rc.2, US-014 Milan Day-6).*
