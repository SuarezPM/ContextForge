#!/usr/bin/env bash
# Run V6.2 adversarial benchmark on real MI300X (Sprint 3 Wave B).
set -euo pipefail
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TS=$(date +%s)
OUT="$LOG_DIR/mi300x_v62_${TS}.json"

echo "Running V6.2 adversarial benchmark on real MI300X..."
PYTHONPATH=. python3 demo/benchmark_v62_adversarial.py --json-out "$OUT" || true

echo "Result: $OUT"
