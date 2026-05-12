#!/usr/bin/env bash
# MI300X smoke test for FWHT integration (Sprint 3 Wave B).
# Run on droplet after: git clone + cd Apohara_Context_Forge + pip install -e .
#
# Drops the `|| true` on the pytest call: real test failures must surface as
# the script's exit code (caller scripts and CI will see the non-zero) instead
# of being silently swallowed.
set -euo pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS=$(date +%s)
OUT="$LOG_DIR/mi300x_fwht_${TS}.json"

if [[ ! -d tests ]]; then
  echo "FATAL: tests/ directory not found. Run from repo root." >&2
  exit 1
fi

echo "Recording rocm-smi snapshot..."
rocm-smi --json > "$LOG_DIR/rocm_smi_pre_${TS}.json" 2>/dev/null \
  || echo "rocm-smi not available; continuing without GPU telemetry"

echo "Running FWHT integration + INT4 codec tests..."
PYTHONPATH=. python3 -m pytest \
  tests/test_rotate_kv_fwht_integration.py \
  tests/test_rotate_kv_int4_codec.py \
  -v \
  --json-report --json-report-file="$OUT"

echo "Recording post-test rocm-smi snapshot..."
rocm-smi --json > "$LOG_DIR/rocm_smi_post_${TS}.json" 2>/dev/null || true

echo "Done. Result: $OUT"
