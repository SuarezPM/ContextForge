#!/usr/bin/env bash
# MI300X smoke test for FWHT integration (Sprint 3 Wave B).
# Run on droplet after: git clone + cd Apohara_Context_Forge + pip install -e .
set -euo pipefail
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS=$(date +%s)
OUT="$LOG_DIR/mi300x_fwht_${TS}.json"

echo "Recording rocm-smi snapshot..."
rocm-smi --json > "$LOG_DIR/rocm_smi_pre_${TS}.json" || echo "rocm-smi not available"

echo "Running FWHT integration tests..."
PYTHONPATH=. python3 -m pytest tests/test_rotate_kv_fwht_integration.py tests/test_rotate_kv_int4_codec.py -v --json-report --json-report-file="$OUT" || true

echo "Recording post-test rocm-smi snapshot..."
rocm-smi --json > "$LOG_DIR/rocm_smi_post_${TS}.json" || true

echo "Result: $OUT"
