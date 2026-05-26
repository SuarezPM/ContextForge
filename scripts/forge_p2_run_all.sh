#!/usr/bin/env bash
# Phase 2 MI300X revalidation runner (FORGE-LEDGER branch). Runs ENTIRELY VM-side
# inside tmux so a flaky SSH link can't kill it. Steps: ensure suite deps + torch
# -> S2 full pytest suite -> S3 FORGE-LEDGER hardware proof -> S4 codec/bandwidth/
# extreme. All logs -> <repo>/logs/. Final: logs/_status.txt + ALL_DONE_SENTINEL.
# Excludes mi300x_needle_int4.py (host-side full-KV codec swap-killed the VM before).
set -uo pipefail
REPO=~/Apohara_Context_Forge
cd "$REPO" || { echo "NO_REPO"; exit 1; }
mkdir -p logs
export PYTHONPATH=.
PY=python3
S() { date -u +%FT%TZ; }
L=logs/_run_all.log
: > "$L"
log() { echo "$@" | tee -a "$L"; }

log "=== RUN_ALL START $(S) on $(hostname) ==="

# --- ordered, backtrack-free install. --no-cache-dir avoids the pip-22
#     "Memoryview is too large" crash on >2GB wheels. Installing the full
#     requirements.txt at once makes pip's resolver backtrack for minutes
#     (llmlingua + sentence-transformers + torch pin), so we install in
#     groups: torch-ROCm first (pinned by index), then core, then heavy. ---
log "[torch] ROCm build $(S)"
$PY -m pip install --user --no-cache-dir torch --index-url https://download.pytorch.org/whl/rocm6.3 \
  >>logs/torch_install.log 2>&1 && log "[torch] ROCm OK" || log "[torch] ROCm FAILED"

log "[deps] core (no heavy ML; resolves instantly) $(S)"
$PY -m pip install --user --no-cache-dir \
  numpy "pydantic>=2.9,<3" pydantic-settings httpx aiofiles rich psutil \
  faiss-cpu prometheus-client fastapi "uvicorn[standard]" \
  pytest pytest-asyncio pytest-json-report z3-solver \
  >>"$L" 2>&1 && log "[deps] core OK" || log "[deps] core WARN"

# Heavy ML deps isolated + time-bounded; the suite proceeds without them if
# they stall (the relevant tests will then error, reported honestly).
log "[deps] heavy ML (llmlingua, sentence-transformers) $(S)"
timeout 600 $PY -m pip install --user --no-cache-dir llmlingua sentence-transformers \
  >>logs/heavy_deps.log 2>&1 && log "[deps] heavy OK" || log "[deps] heavy SKIPPED/FAILED"
$PY -c "import torch; print('torch', torch.__version__, 'hip', getattr(torch.version,'hip',None), 'devs', torch.cuda.device_count())" \
  2>&1 | tee -a "$L" || log "[torch] still unavailable — GPU steps will skip"

GPU=0
$PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && GPU=1
log "[env] GPU_available=$GPU"

# --- S2: FULL pytest suite on real MI300X ---
log "=== S2 pytest full suite $(S) ==="
$PY -m pytest tests/ -q --json-report --json-report-file=logs/mi300x_p2_pytest.json \
  >logs/mi300x_p2_pytest.txt 2>&1
log "[S2] pytest exit=$? ; tail:"
tail -5 logs/mi300x_p2_pytest.txt | tee -a "$L"

# --- S3: FORGE-LEDGER hardware proof (z3 only, 1210-cert sweep + tamper) ---
log "=== S3 FORGE-LEDGER proof $(S) ==="
$PY scripts/mi300x_forge_ledger_proof.py >logs/mi300x_p2_forge_ledger.txt 2>&1
log "[S3] proof exit=$? ; tail:"
tail -6 logs/mi300x_p2_forge_ledger.txt | tee -a "$L"

# --- S4: codec quality + HBM3 bandwidth + extreme scale (GPU, bounded) ---
if [ "$GPU" = 1 ]; then
  for s in quant_quality hbm3_bandwidth extreme_scale; do
    log "=== S4 mi300x_${s} $(S) ==="
    timeout 900 $PY scripts/mi300x_${s}.py >logs/mi300x_p2_${s}.txt 2>&1
    log "[S4:${s}] exit=$? ; tail:"
    tail -6 logs/mi300x_p2_${s}.txt | tee -a "$L"
  done
else
  log "[S4] skipped (no GPU torch)"
fi

# --- status summary (one cheap read tells me everything) ---
{
  echo "STATUS $(S) host=$(hostname)"
  echo "GPU_available=$GPU"
  echo -n "S2_pytest: "; grep -E "passed|failed|error" logs/mi300x_p2_pytest.txt | tail -1
  echo -n "S3_proof: "; grep -E "PROOF_OK" logs/mi300x_p2_forge_ledger.txt | tail -1 || echo "no PROOF_OK line"
  echo "S4_logs:"; ls -1 logs/mi300x_quant_quality_*.json logs/mi300x_hbm3_bandwidth_*.json logs/mi300x_extreme*_*.json 2>/dev/null || echo "  (none)"
} > logs/_status.txt
cat logs/_status.txt | tee -a "$L"

log "=== RUN_ALL DONE $(S) ==="
log "ALL_DONE_SENTINEL"
