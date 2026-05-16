#!/usr/bin/env bash
# US-MI-014b — extended MI300X measurements for paper v2.0 final
#
# Fixes the mi300x_run_all.sh wrapper (which hardcoded /root/...) and
# extends the Wave B coverage to the 4 measurements the paper v2.0 still
# needs:
#
#   Stage 1  extreme_scale     — 65K + 131K + 262K reduction factors
#                                 (closes paper Table 1 row coverage; Wave B
#                                  already had 4K, 8K, 16K, 32K)
#   Stage 2  quant_quality     — FP16 vs INT8-sim vs INT4 vs INT4+FWHT MSE
#                                 quality curve (paper v2.0 Table 2 anchor)
#   Stage 3  fwht_inplace      — FWHT in-place vs out-of-place memory
#                                 footprint (paper FWHT section figure)
#   Stage 4  lmcache_smoke     — vLLM LMCache integration smoke
#                                 ("codec works in serving path" claim)
#
# Usage on the rented Hot Aisle box (or any MI300X box):
#
#   cd ~/Apohara_Context_Forge
#   source .venv/bin/activate
#   export PYTHONPATH=.
#   bash scripts/mi300x_run_paper_v2.sh
#
# Each stage writes its own JSON log under logs/. Total wall-clock
# ~45 min, total MI300X spend ~$1.50 at $1.99/h.
#
# Apache-2.0 — Apohara ContextForge.

set -euo pipefail

# Run from the repo root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p logs

PY="${PY:-python3}"
[ -x .venv/bin/python ] && PY=".venv/bin/python"

TS_START=$(date -u +%s)
SUMMARY_LOG=logs/mi300x_run_paper_v2_$(date -u +%Y%m%dT%H%M%SZ).txt
echo "[$(date -Is)] mi300x_run_paper_v2 start" | tee "$SUMMARY_LOG"

run_stage () {
  local name=$1; shift
  echo "" | tee -a "$SUMMARY_LOG"
  echo "=== Stage: $name ===" | tee -a "$SUMMARY_LOG"
  echo "Command: $*" | tee -a "$SUMMARY_LOG"
  local t0=$(date +%s)
  local rc=0
  if "$@" 2>&1 | tee -a "$SUMMARY_LOG"; then
    rc=0
  else
    rc=$?
  fi
  local t1=$(date +%s)
  echo "[$(date -Is)] $name elapsed=$((t1 - t0))s exit=$rc" | tee -a "$SUMMARY_LOG"
  return $rc
}

# Stage 1 — extreme scale (paper Table 1 rows 5-7: 65K, 131K, 262K)
run_stage extreme_scale     $PY scripts/mi300x_extreme_scale.py || true

# Stage 2 — quantization quality curve (paper Table 2 anchor)
run_stage quant_quality     $PY scripts/mi300x_quant_quality.py || true

# Stage 3 — FWHT in-place memory footprint (FWHT section figure)
run_stage fwht_inplace      $PY scripts/mi300x_fwht_inplace_bench.py || true

# Stage 4 — vLLM LMCache integration smoke test
run_stage lmcache_smoke     $PY scripts/mi300x_lmcache_smoke.py || true

TS_END=$(date -u +%s)
echo "" | tee -a "$SUMMARY_LOG"
echo "=== Wall-clock: $((TS_END - TS_START))s ===" | tee -a "$SUMMARY_LOG"
echo "=== New MI300X logs from this run: ===" | tee -a "$SUMMARY_LOG"
find logs/ -name "mi300x_*.json" -newer "$SUMMARY_LOG.tmp" 2>/dev/null \
  -o -name "mi300x_*.json" -newermt "@$TS_START" \
  | tee -a "$SUMMARY_LOG"
echo "[$(date -Is)] mi300x_run_paper_v2 done" | tee -a "$SUMMARY_LOG"
