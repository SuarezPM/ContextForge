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

**Hardware:** **CPU-mock fallback (GCP H100 deferred).** The
GCP service account
`apohara-aegis-judge@gen-lang-client-0658922897.iam.gserviceaccount.com`
configured for this workstation lacks Compute Engine API access
(SERVICE_DISABLED) and cannot self-elevate to enable it. Until a
human owner enables the API for the project, the side-by-side runs
use the mock backend in `scripts/_sprint5_pipeline.py::run_request_mock`.
HBM numbers come from the closed-form in
`scripts/build_milan_benchmark.py::estimate_hbm_used_gb`
(Llama-3-8B; 32 layers × GQA-8 KV heads × fp16). The live GPU run
is deferred to Pablo's manual execution — see AUDIT.md #11.

| Config | TTFT (ms) | Throughput (tok/s) | HBM (GB) | JCR | Notes |
|---|---|---|---|---|---|
| vllm `--enable-prefix-caching` (baseline) | 31.93 | 6,915.2 | 78.12 | 0.784 | INV-15 disabled, no cross-agent KV sharing |
| vllm + contextforge plugin | 32.12 | 6,852.6 | 18.75 | 1.000 | INV-15 ON, mean-reuse 0.76 sharing |
| **Delta** | **+0.19** | **-0.91%** | **-59.37 (-76.0%)** | **+0.216** | Critic agent JCR recovered from 0.784 → 1.000 |

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
