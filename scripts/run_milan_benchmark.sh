#!/usr/bin/env bash
# US-014 — Milan AI Week 5-agent benchmark orchestrator.
#
# Runs the head-to-head workload twice (baseline / contextforge) and
# composes the Milan submission JSON at
# logs/milan_5agent_benchmark_<ts>.json.
#
# Default (no args): mock mode on CPU. This is the fallback path
# from US-014 §7 — the live GPU run is gated on AMD MI300X credit
# or GCP project-owner action to enable Compute Engine API for
# the apohara-aegis-judge service account.
#
# To run on a real vLLM endpoint (GPU host):
#   VLLM_ENDPOINT=http://localhost:8000 ./scripts/run_milan_benchmark.sh
#
# Override defaults via env vars:
#   N_REQUESTS    — # workload requests (default 500)
#   HARDWARE_LABEL — string to record in the JSON (auto-detected if unset)
#   COST_USD       — total cloud spend so far (default 0.0)

set -euo pipefail

cd "$(dirname "$0")/.."

N_REQUESTS="${N_REQUESTS:-500}"
COST_USD="${COST_USD:-0.0}"
TS="$(date +%s)"
HARDWARE_LABEL="${HARDWARE_LABEL:-}"

OUT_DIR="logs"
mkdir -p "${OUT_DIR}"

if [[ -n "${VLLM_ENDPOINT:-}" ]]; then
    BACKEND_FLAG=""
    BACKEND_LABEL="vllm @ ${VLLM_ENDPOINT}"
    ENDPOINT_FLAG="--vllm-endpoint ${VLLM_ENDPOINT}"
    if [[ -z "${HARDWARE_LABEL}" ]]; then
        HARDWARE_LABEL="vLLM-backed run (set HARDWARE_LABEL to override)"
    fi
else
    BACKEND_FLAG="--mock"
    BACKEND_LABEL="CPU-mock"
    ENDPOINT_FLAG=""
    if [[ -z "${HARDWARE_LABEL}" ]]; then
        HARDWARE_LABEL="CPU-mock fallback (GCP H100 deferred)"
    fi
fi

BASELINE_OUT="${OUT_DIR}/milan_h2h_baseline_${TS}.json"
CTX_OUT="${OUT_DIR}/milan_h2h_contextforge_${TS}.json"
COMBINED_OUT="${OUT_DIR}/milan_5agent_benchmark_${TS}.json"

echo "=== US-014 Milan 5-agent benchmark — ${BACKEND_LABEL} ==="
echo "Hardware label: ${HARDWARE_LABEL}"
echo "N requests:     ${N_REQUESTS}"
echo ""

echo "--- Run A: baseline (vllm prefix-cache only, no contextforge plugin) ---"
PYTHONPATH=. python3 scripts/sprint5_head_to_head.py \
    --mode apohara_off \
    --n-requests "${N_REQUESTS}" \
    ${BACKEND_FLAG} ${ENDPOINT_FLAG} \
    --out "${BASELINE_OUT}"

echo ""
echo "--- Run B: contextforge (INV-15 ON + cross-agent KV sharing) ---"
PYTHONPATH=. python3 scripts/sprint5_head_to_head.py \
    --mode apohara_on \
    --n-requests "${N_REQUESTS}" \
    ${BACKEND_FLAG} ${ENDPOINT_FLAG} \
    --out "${CTX_OUT}"

echo ""
echo "--- Composing Milan JSON ---"
PYTHONPATH=. python3 scripts/build_milan_benchmark.py \
    --baseline       "${BASELINE_OUT}" \
    --contextforge   "${CTX_OUT}" \
    --hardware       "${HARDWARE_LABEL}" \
    --cost-est-usd   "${COST_USD}" \
    --out            "${COMBINED_OUT}"

echo ""
echo "Milan benchmark JSON: ${COMBINED_OUT}"
echo "Run inputs:           ${BASELINE_OUT}"
echo "                      ${CTX_OUT}"
