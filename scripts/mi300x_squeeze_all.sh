#!/usr/bin/env bash
# MI300X "squeeze" runner — measure ContextForge across 3 models in tokens + VRAM
# + concurrency + throughput + footprint, plus NIAH quality. Runs ON the MI300X VM.
#
# Environment reality (enc1-gpuvm007): vLLM is NOT on the host; it runs inside the
# AMD `rocm/vllm` container (ROCm+AITER prebuilt). The measurement scripts run on
# the HOST python3 (they only do HTTP + rocm-smi/pyrsmi), reaching the container
# over `--network host`. Sequential: one container per model on the single card.
#
# Usage (on the VM, inside tmux):
#   cd ~/Apohara_Context_Forge && bash scripts/mi300x_squeeze_all.sh 2>&1 | tee logs/mi300x_squeeze/_run.log
#
# Honesty: each stage writes its own JSON; a failed stage is recorded with its
# error and the run CONTINUES (no fabricated numbers). The validated probe
# (mi300x_measure.py) is reused verbatim; VRAMMonitor's PyRSMI path gives
# device-wide HBM, cross-checked by rocm-smi.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
LOGDIR="logs/mi300x_squeeze"; mkdir -p "$LOGDIR"
export PYTHONPATH="$REPO"; export PYTHONHASHSEED=0
IMG="${VLLM_IMG:-rocm/vllm:latest}"
HF_CACHE="$HOME/.cache/huggingface"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
RESULTS_BRANCH="${RESULTS_BRANCH:-mi300x-evidence}"

log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOGDIR/_run.log"; }

push_logs(){  # best-effort incremental push; never fails the run
  [ -z "$GITHUB_TOKEN" ] && return 0
  git add "$LOGDIR" >/dev/null 2>&1
  git -c user.email=ci@apohara -c user.name=mi300x-runner commit -q -m "mi300x evidence: $1" >/dev/null 2>&1
  git push -q "https://x-access-token:${GITHUB_TOKEN}@github.com/SuarezPM/Apohara_Context_Forge.git" "HEAD:${RESULTS_BRANCH}" >/dev/null 2>&1 \
    && log "pushed logs -> ${RESULTS_BRANCH} ($1)" || log "push failed (non-fatal): $1"
}

preflight(){
  log "=== PREFLIGHT ==="
  rocm-smi --showproductname 2>/dev/null | grep -iE "card0|mi300" | head -1 | tee -a "$LOGDIR/_run.log" || log "WARN no rocm-smi"
  docker image inspect "$IMG" >/dev/null 2>&1 && log "image $IMG present" || { log "FATAL: image $IMG missing"; exit 1; }
  python3 -c "import apohara_context_forge, numpy, httpx; print('host repo deps OK')" 2>&1 | tail -1 | tee -a "$LOGDIR/_run.log" \
    || { log "FATAL: host deps missing (pip3 install --user numpy httpx pyrsmi ...)"; exit 1; }
  python3 -c "import pyrsmi" 2>/dev/null && log "pyrsmi OK (device-wide HBM)" || log "WARN no pyrsmi (rocm-smi second-source still works)"
}

# Launch one vLLM container, wait health, run the measurement set, tear it down.
# args: HF_MODEL  SERVED  KV_DTYPE  MAXLEN  PORT  [EXTRA vllm flags...]
run_model(){
  local model="$1" served="$2" kv="$3" maxlen="$4" port="$5"; shift 5
  local safe; safe="$(echo "$served" | tr '/:.' '___')"
  if [ -f "$LOGDIR/${safe}_measure.json" ]; then log "$served already measured — skipping (resume)"; return 0; fi
  log "=== MODEL $served (img=$IMG kv=$kv maxlen=$maxlen) ==="
  docker rm -f vllm_run >/dev/null 2>&1

  docker run -d --name vllm_run --network host \
    --device /dev/kfd --device /dev/dri --group-add video --ipc host --shm-size 32g \
    -v "$HF_CACHE":/root/.cache/huggingface -e VLLM_USE_AITER=1 -e PYTHONHASHSEED=0 -e HF_HUB_ENABLE_HF_TRANSFER=0 \
    "$IMG" vllm serve "$model" --served-model-name "$served" --port "$port" \
    --enable-prefix-caching --kv-cache-dtype "$kv" --max-model-len "$maxlen" \
    --gpu-memory-utilization 0.90 --trust-remote-code "$@" >/dev/null 2>&1 || { log "docker run failed for $served"; return 1; }

  local ready=0 i
  for i in $(seq 1 480); do          # up to 40min: 235B FP8 download+load is large
    docker ps -q -f name=vllm_run | grep -q . || { log "container EXITED — logs:"; docker logs --tail 30 vllm_run 2>&1 | tee -a "$LOGDIR/${safe}_server.log"; break; }
    curl -sf -o /dev/null "http://127.0.0.1:$port/health" 2>/dev/null && { ready=1; break; }
    sleep 5
  done
  if [ "$ready" != 1 ]; then
    log "$served NOT ready — recording blocker, skipping"
    docker logs --tail 40 vllm_run > "$LOGDIR/${safe}_BLOCKER.txt" 2>&1
    docker rm -f vllm_run >/dev/null 2>&1; push_logs "$served BLOCKED"; return 1
  fi
  log "$served READY"
  rocm-smi --json > "$LOGDIR/${safe}_rocmsmi_loaded.json" 2>/dev/null || true

  python3 scripts/mi300x_measure.py --endpoint "http://127.0.0.1:$port" --model "$served" \
    --out "$LOGDIR/${safe}_measure.json" 2>&1 | tail -8 | tee -a "$LOGDIR/_run.log" || log "measure FAILED $served"
  push_logs "$served measure"
  python3 scripts/mi300x_contextforge_e2e.py --endpoint "http://127.0.0.1:$port" --model "$served" \
    --output "$LOGDIR/${safe}_tokens.json" --max-tokens 64 2>&1 | tail -8 | tee -a "$LOGDIR/_run.log" || log "tokens FAILED $served"
  push_logs "$served tokens"
  python3 scripts/mi300x_niah.py --endpoint "http://127.0.0.1:$port" --model "$served" \
    --output "$LOGDIR/${safe}_niah.json" 2>&1 | tail -8 | tee -a "$LOGDIR/_run.log" || log "niah FAILED $served"
  push_logs "$served niah"

  docker rm -f vllm_run >/dev/null 2>&1
  log "$served DONE; container removed"
}

preflight
log "results branch: ${RESULTS_BRANCH} (push=${GITHUB_TOKEN:+on}${GITHUB_TOKEN:-off})"

# 3 PUBLIC models (no HF gating). Coder-Next = hybrid baseline; 72B = dense
# full-attention; 235B = frontier MoE full-attention. fp8 KV on the FP8 weights.
# Qwen3-235B-A22B (full-attention GQA, frontier 2025) at INT4 (~118GB): the
# heaviest full-attention frontier model that fits ONE MI300X (192GB), ~70GB for
# KV. INT4 on ROCm is the risk — GPTQ first, AWQ as automatic fallback; whichever
# loads, measures. (qwen3-32b already proved the mechanism on MI300X: 84.7% vs 0%.)
run_model "Qwen/Qwen3-235B-A22B-GPTQ-Int4"              "qwen235b-gptq" "auto" 16384 8001 || true
run_model "QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ" "qwen235b-awq"  "auto" 16384 8002 || true

log "=== suite (ROCm cross-check) ==="
python3 -m pytest -q > "$LOGDIR/_pytest_rocm.txt" 2>&1; tail -3 "$LOGDIR/_pytest_rocm.txt" | tee -a "$LOGDIR/_run.log" || true
push_logs "pytest rocm"
log "=== HBM3 bandwidth ==="
python3 scripts/mi300x_hbm3_bandwidth.py > "$LOGDIR/_hbm3.json" 2>&1 || true
push_logs "hbm3"
log "=== ALL_DONE_SENTINEL ==="
push_logs "ALL DONE"
