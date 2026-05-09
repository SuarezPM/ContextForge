#!/bin/bash
# ContextForge benchmark runner for AMD DevCloud MI300X
# Prerequisites: ROCm 7.x, Python 3.11+, $100 AMD GPU credits
# Cost estimate: ~$1.99/hr on MI300X x1

set -euo pipefail

# GPU verification
rocm-smi --showproductname
python -c "import torch; print(torch.cuda.get_device_name())"

# Install
pip install -e ".[rocm]" --quiet
pip install qwen3-embed onnxruntime streamlit prometheus-client --quiet

# Smoke tests first (cheap, ~5 min, ~$0.17)
pytest tests/ -v --tb=short -x 2>&1 | tee logs/smoke_test.log

# V4 benchmarks (22 hr estimate if all scenarios, ~$44)
python demo/benchmark_v4.py \
  --device rocm:0 \
  --scenarios all \
  --output logs/benchmark_v4_results.json \
  --prometheus-port 9090 \
  2>&1 | tee logs/benchmark_v4.log

# V5 stability benchmark (QueueingController)
python demo/benchmark_v5.py \
  --device rocm:0 \
  --focus queueing_stability \
  --output logs/benchmark_v5_results.json \
  2>&1 | tee logs/benchmark_v5.log

echo "Benchmark complete. Total GPU time: $(cat logs/benchmark_v4.log | grep 'total_time_hrs')"