#!/usr/bin/env bash
# MI300X "squeeze" runner — measure ContextForge across 3 models in tokens + VRAM
# + concurrency + throughput + footprint, plus NIAH quality and (stretch) the
# cross-worker LMCache path. Runs ON the MI300X VM (one vLLM per model, sequential
# on the single card). Pushes logs to GitHub incrementally so results survive a
# dropped connection; deletes nothing (teardown is the operator's call).
#
# Usage (on the VM, inside tmux):
#   cd ~/Apohara_Context_Forge && bash scripts/mi300x_squeeze_all.sh 2>&1 | tee logs/mi300x_squeeze/_run.log
#
# Prereqs the runner checks (and FAILS LOUD if missing, rather than burning GPU):
#   - vllm importable / `vllm` on PATH (ROCm build), rocm-smi present
#   - repo deps installed (the measure/e2e scripts import apohara_context_forge)
#   - optional: redis reachable + lmcache installed for the cross-worker stretch
#
# Honesty: every stage writes its own JSON; a stage that fails is recorded with
# its error and the run CONTINUES (no fabricated numbers). RTX-2060-validated
# probe (mi300x_measure.py) is reused verbatim; the AMD path of VRAMMonitor
# (PyRSMI) provides device-wide HBM, cross-checked by rocm-smi.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
LOGDIR="logs/mi300x_squeeze"
mkdir -p "$LOGDIR"
PY="${PY:-python3}"; [ -x .venv/bin/python ] && PY=".venv/bin/python"
export PYTHONPATH="$REPO"
# AMD MI300X serving knobs (from the prior phase-2 MoE plan): AITER kernels on.
export VLLM_USE_AITER="${VLLM_USE_AITER:-1}"
export PYTHONHASHSEED=0

GITHUB_TOKEN="${GITHUB_TOKEN:-}"   # if set, logs are pushed to a results branch
RESULTS_BRANCH="mi300x-evidence-$(cat /proc/sys/kernel/random/uuid 2>/dev/null | cut -c1-8 || echo run)"

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOGDIR/_run.log"; }

push_logs(){  # best-effort incremental push; never fails the run
  [ -z "$GITHUB_TOKEN" ] && return 0
  git add "$LOGDIR" 2>/dev/null
  git -c user.email=ci@apohara -c user.name=mi300x-runner commit -q -m "mi300x evidence: $1" 2>/dev/null
  git push -q "https://x-access-token:${GITHUB_TOKEN}@github.com/SuarezPM/Apohara_Context_Forge.git" "HEAD:${RESULTS_BRANCH}" 2>/dev/null \
    && log "pushed logs -> ${RESULTS_BRANCH} ($1)" || log "push failed (non-fatal): $1"
}

preflight(){
  log "=== PREFLIGHT ==="
  command -v rocm-smi >/dev/null && rocm-smi --showproductname 2>/dev/null | grep -iE "card|mi300" | head -1 | tee -a "$LOGDIR/_run.log" || log "WARN: no rocm-smi"
  $PY -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -1 | tee -a "$LOGDIR/_run.log" \
    || { command -v vllm >/dev/null && log "vllm CLI on PATH" || { log "FATAL: no vllm — aborting before burning GPU"; exit 1; }; }
  $PY -c "import apohara_context_forge, numpy; print('repo deps OK')" 2>&1 | tail -1 | tee -a "$LOGDIR/_run.log" \
    || { log "FATAL: repo deps missing (apohara_context_forge import failed)"; exit 1; }
}

# Launch one vLLM, wait for health, run the full measurement set, tear it down.
# args: HF_MODEL  SERVED_NAME  KV_CACHE_DTYPE  MAX_MODEL_LEN  PORT  EXTRA_FLAGS...
run_model(){
  local model="$1" served="$2" kv="$3" maxlen="$4" port="$5"; shift 5
  local extra=("$@")
  local safe; safe="$(echo "$served" | tr '/:.' '___')"
  log "=== MODEL $served (kv=$kv maxlen=$maxlen) ==="

  vllm serve "$model" --served-model-name "$served" --port "$port" \
    --enable-prefix-caching --kv-cache-dtype "$kv" --max-model-len "$maxlen" \
    --gpu-memory-utilization 0.90 --trust-remote-code "${extra[@]}" \
    > "$LOGDIR/${safe}_server.log" 2>&1 &
  local pid=$!

  local ready=0 i
  for i in $(seq 1 240); do
    kill -0 "$pid" 2>/dev/null || { log "server EXITED early — see ${safe}_server.log"; break; }
    curl -sf -o /dev/null "http://127.0.0.1:$port/health" 2>/dev/null && { ready=1; break; }
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "$served: NOT ready (timeout/crash) — recording blocker, skipping"
    tail -25 "$LOGDIR/${safe}_server.log" > "$LOGDIR/${safe}_BLOCKER.txt" 2>/dev/null
    kill "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null
    push_logs "$served BLOCKED at startup"; return 1
  fi
  log "$served READY"
  rocm-smi --json > "$LOGDIR/${safe}_rocmsmi_loaded.json" 2>/dev/null || true

  # 1) VRAM / concurrency / throughput / footprint (validated probe)
  $PY scripts/mi300x_measure.py --endpoint "http://127.0.0.1:$port" --model "$served" \
    --out "$LOGDIR/${safe}_measure.json" 2>&1 | tail -8 | tee -a "$LOGDIR/_run.log" || log "measure FAILED for $served (continuing)"
  push_logs "$served measure"
  # 2) Tokens (compression + dedup, the 44.4% line) — needs LLMLingua in the env
  $PY scripts/mi300x_contextforge_e2e.py --endpoint "http://127.0.0.1:$port" --model "$served" \
    --output "$LOGDIR/${safe}_tokens.json" --max-tokens 64 2>&1 | tail -8 | tee -a "$LOGDIR/_run.log" || log "tokens FAILED for $served (continuing)"
  push_logs "$served tokens"
  # 3) NIAH quality (ahorro sin perder recall)
  $PY scripts/mi300x_niah.py --endpoint "http://127.0.0.1:$port" --model "$served" \
    --output "$LOGDIR/${safe}_niah.json" 2>&1 | tail -8 | tee -a "$LOGDIR/_run.log" || log "niah FAILED for $served (continuing)"
  push_logs "$served niah"

  kill "$pid" 2>/dev/null; sleep 5; kill -9 "$pid" 2>/dev/null
  log "$served DONE; server down"
}

# --------------------------------------------------------------------------- #
preflight
log "results branch: ${RESULTS_BRANCH} (push=${GITHUB_TOKEN:+on}${GITHUB_TOKEN:-off})"

# The 3 models. FP8 KV where the weights are FP8 / the model is large.
# served-name kept short & stable so the salt/prefix logic is model-agnostic.
run_model "Qwen/Qwen3-Coder-Next-FP8"      "coder-next"  "fp8"  16384 8001 || true
run_model "meta-llama/Llama-3.3-70B-Instruct" "llama70b"  "auto" 16384 8002 || true
run_model "Qwen/Qwen3-235B-A22B-Instruct-2507" "qwen235b" "fp8"  16384 8003 --tensor-parallel-size 1 || true

# Once, hardware-wide:
log "=== suite (ROCm cross-check) ==="
$PY -m pytest -q > "$LOGDIR/_pytest_rocm.txt" 2>&1 ; tail -3 "$LOGDIR/_pytest_rocm.txt" | tee -a "$LOGDIR/_run.log" || true
push_logs "pytest rocm"
log "=== HBM3 bandwidth ==="
$PY scripts/mi300x_hbm3_bandwidth.py > "$LOGDIR/_hbm3.json" 2>&1 || true
push_logs "hbm3"

log "=== ALL_DONE_SENTINEL ==="
push_logs "ALL DONE"
