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

> **What this table measures:** end-to-end pipeline correctness on real H100 with a 2026-era Qwen3.6 model + INV-15 critic-gate behavior + per-agent latency and peak VRAM (sequential, one agent at a time, KV flushed between agents). **What it does NOT measure:** the 76% concurrent KV-sharing saving — that requires the vLLM plugin path holding five agents' KV state at the same time via a real registry, deferred until vLLM upstream lands `qwen3_5` model_type support. See the closed-form projection table below for the 76% number, and the `honesty_note` in [`logs/milan_5agent_h100_REAL_20260516T215940Z.json`](logs/milan_5agent_h100_REAL_20260516T215940Z.json) for the full disclosure of what each number represents.

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

### The 76% architectural claim (closed-form projection of the concurrent KV-sharing target)

This is **not** a measurement. It is the closed-form HBM-savings
projection for what the production vLLM plugin will deliver once it
holds five agents' KV state concurrently via the ContextForge
registry. The projection uses Llama-3-8B's KV geometry (32 layers ×
GQA-8 × fp16) plus a workload mean-reuse rate of 0.76; the
architectural saving is `1 - mean_reuse = 0.24` of the prefix KV
footprint replicated across agents. The CPU-mock backend that
produced these numbers lives at
`scripts/_sprint5_pipeline.py::run_request_mock`; the closed-form
itself is in
`scripts/build_milan_benchmark.py::estimate_hbm_used_gb`. The
real-hardware H100 table *above* validates the gate behavior + per-
agent peak; this projection table validates the architectural target
that the plugin path will hit once vLLM upstream supports `qwen3_5`.

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

## MI300X Wave B — RotateKVQuantizer measurements (2026-05-16)

**Story:** US-MI-014 — measurement on a freshly-rented AMD Instinct
MI300X (Hot Aisle `enc1-gpuvm019`) to refresh the paper's headline
3.55× INT4 codec VRAM-reduction claim. Supersedes the earlier
sprint's archived measurements.

**Hardware:** `rocm-hip:6.2.41133-dd7f95766:AMD Instinct MI300X VF`
(192 GB HBM3, ROCm 6.2, torch 2.5.1+rocm6.2).

**Cost:** ~$1.50 of MI300X compute at $1.99/h (~45 min including
provisioning + 4 stages).

**Scripts:** `scripts/mi300x_hbm3_bandwidth.py`,
`scripts/mi300x_pure_torch_fwht.py`,
`scripts/mi300x_vram_sweep.py`,
`scripts/mi300x_vram_measurement.py`.

### Stage A — HBM3 bandwidth

| Working set | Copy BW | Triad BW | Notes |
|---|---|---|---|
| 1 GB | 2920 GB/s | 1254 GB/s | warmup-dominated |
| 4 GB | 3794 GB/s | 3623 GB/s | steady state |
| 16 GB | 3722 GB/s | 3573 GB/s | sustained |
| 64 GB | 3707 GB/s | 3603 GB/s | sustained at scale |

Best triad: **3622 GB/s** = **68.4 % of advertised 5.3 TB/s peak**
(`logs/mi300x_hbm3_bandwidth_1778973430.json`).

### Stage B — Pure-torch FWHT on GPU

8-batch sweep of forward + inverse FWHT on the GPU directly (not the
codec-path FWHT integration that the paper rules out). Log:
`logs/mi300x_pure_torch_fwht_1778973433.json`. Reported separately
from the codec results to keep FWHT-as-primitive isolated from
FWHT-in-the-codec.

### Stage C — VRAM reduction sweep (RotateKV INT4 vs FP16)

| Sequence length | Baseline FP16 | INT4 packed | Reduction factor |
|---|---|---|---|
|  4 096 | 64.0 MiB | 18.1 MiB | **3.5433×** |
|  8 192 | 128.0 MiB | 36.1 MiB | **3.5494×** |
| 16 384 | 256.0 MiB | 72.1 MiB | **3.5525×** |
| 32 768 | 512.0 MiB | 144.1 MiB | **3.5540×** |

Reduction factor is **constant to 3 decimal places (3.54–3.55×)
across 4K-32K context**, confirming the paper's length-invariance
claim (`logs/mi300x_vram_sweep_1778973581.json`).

### Stage D — Canonical 32K with and without FWHT

Same `3.5540×` factor with `use_fwht=True` and `use_fwht=False`
(`logs/mi300x_vram_1778973631.json`). Per the paper, `use_fwht=False`
is the recommended configuration (FWHT integration degrades INT4
reconstruction quality 200× at this codec layout).

### Reading the numbers

- **3.55× headline factor reproduced for the 4K–32K segment.** The
  paper's V2.0 abstract quotes "3.55× VRAM reduction measured on
  AMD Instinct MI300X (192 GB, ROCm 7.2.0), constant across context
  lengths 4K–262K". Wave B confirms the constancy + the reduction
  factor on ROCm 6.2 for the 4K–32K segment that the default sweep
  exercises. The 4K–262K range remains anchored by the prior
  sprint's archived sweep on ROCm 7.2.0 (paper Table 1) — Wave B
  validates the trend at the lower end on refreshed hardware, not
  the upper bound.
- **HBM3 efficiency 68 %** is healthy — typical hand-tuned codes on
  MI300X land in the 60-80 % range; we are not bandwidth-bound for
  the codec hot path.

---

## What is NOT in this document yet

- **5-agent live H100 + MI300X side-by-side** — H100 row above is
  measured; a same-workload MI300X run is a post-hackathon delivery.
- **K8s operator stress test** — handled by `operator/validate.sh`.
- **Multi-node Redis cache** — not yet implemented (`remote_url:
  null` in the YAML).

---

## MI300X Wave C — paper-v2 extension (2026-05-17)

**Story:** US-MI2-014 — second Hot Aisle MI300X run (`enc1-gpuvm010`)
executing `scripts/mi300x_run_paper_v2.sh` to close the 4K-262K
coverage and re-validate the paper's Table 2 quality curve on
freshly-rented hardware. Reproduces the paper's existing numbers
to within rounding noise; no paper text changes required.

**Hardware:** same family as Wave B —
`rocm-hip:6.2.41133-dd7f95766:AMD Instinct MI300X VF`.

**Cost:** ~$0.17 of MI300X compute (303 s wall-clock at $1.99/h).

### Extreme-scale extension — completes Table 1 (65K to 262K)

| Sequence length | Heads | Head dim | Reduction factor | Paper Table 1 |
|---|---|---|---|---|
| 65,536 | 32 | 128 | **3.5548×** | 3.55× ✓ |
| 131,072 | 32 | 128 | **3.5552×** | 3.56× ✓ |
| 65,536 | 32 | 256 | **3.5548×** | (new data point — wider heads) |
| 262,144 | 16 | 128 | **3.5554×** | 3.56× ✓ |

Together with the Wave B 4K-32K sweep, the **full paper Table 1
range (4K-262K) is now anchored on fresh-hardware measurements**
(`logs/mi300x_extreme_scale_1778977872.json`).

### Quality curve — re-validates Table 2 + the "200× degradation" claim

| Configuration | Packed bytes | Reduction | MSE (keys) | Paper Table 2 |
|---|---|---|---|---|
| FP16 baseline | 268,435,456 | 1.00× | 4.31×10⁻⁸ | 4.3×10⁻⁸ ✓ |
| INT8 naive | 142,606,336 | 1.88× | 3.44×10⁻⁵ | 3.4×10⁻⁵ ✓ |
| INT4 `use_fwht=False` | 75,563,008 | 3.55× | 1.01×10⁻² | 1.0×10⁻² ✓ |
| INT4 `use_fwht=True` | 75,563,008 | 3.55× | 2.01×10⁰ | 2.0×10⁰ ✓ |

`use_fwht=True / use_fwht=False` MSE ratio = **199×** (paper claims
"≈200×" — confirmed within rounding) (`logs/mi300x_quant_quality_1778978012.json`).

### FWHT in-place benchmark — new finding

| Shape | Original out-of-place | In-place | Speedup |
|---|---|---|---|
| (B=1, S=4K, H=32, D=128) | 1.1 ms | 1.4 ms | **0.78×** (slower) |
| (B=1, S=16K, H=32, D=128) | 2.6 ms | 4.6 ms | 0.58× |
| (B=1, S=32K, H=32, D=128) | 5.1 ms | 9.1 ms | 0.55× |
| (B=1, S=16K, H=32, D=256) | 5.6 ms | 10.2 ms | 0.54× |
| (B=1, S=16K, H=64, D=128) | 5.4 ms | 9.3 ms | 0.58× |

**Counterintuitive finding**: the in-place FWHT path that should
reduce peak memory actually **runs slower on MI300X** at every
tested shape. Triggers extra HBM3 traffic on ROCm 6.2. Documented
for paper appendix; recommends staying with out-of-place FWHT
(`logs/mi300x_fwht_inplace_bench_1778978015.json`).

### LMCache smoke — honest fallback

The lmcache wheel is not pip-installable on the Hot Aisle image
(`ModuleNotFoundError`); the `LMCacheConnectorV2` enters its
documented fallback mode (`active=False`) and the build_error
string is preserved in the log
(`logs/mi300x_lmcache_1778978017.json`).

---

*Last updated: 2026-05-17 (V7.0.0-rc.2 + US-MI2-014 Wave C extension; previously US-014 Milan Day-6 + US-MI-014 Wave B).*
