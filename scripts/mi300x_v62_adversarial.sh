#!/usr/bin/env bash
# Run V6.2 adversarial benchmark on real MI300X (Sprint 3 Wave B).
#
# Fails loudly: drops the previous `|| true` that swallowed all errors,
# so a missing demo/ entry point or import failure surfaces immediately
# instead of silently producing no result.
set -euo pipefail

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS=$(date +%s)
OUT="$LOG_DIR/mi300x_v62_${TS}.json"

if [[ ! -f demo/benchmark_v62_adversarial.py ]]; then
  echo "FATAL: demo/benchmark_v62_adversarial.py not found. Run from repo root." >&2
  exit 1
fi

echo "Running V6.2 adversarial benchmark on real MI300X..."
echo "Output JSON: $OUT"
PYTHONPATH=. python3 demo/benchmark_v62_adversarial.py --out "$OUT"

echo "Done. Result: $OUT"
