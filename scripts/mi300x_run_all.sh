#!/usr/bin/env bash
# Run all extended Wave B measurements on MI300X.
set -euo pipefail

mkdir -p logs
cd /root/Apohara_Context_Forge

echo "=========================================="
echo "=== Stage A: HBM3 bandwidth probe ==="
echo "=========================================="
PYTHONPATH=. python3 scripts/mi300x_hbm3_bandwidth.py

echo ""
echo "=========================================="
echo "=== Stage B: Pure torch FWHT on GPU ==="
echo "=========================================="
PYTHONPATH=. python3 scripts/mi300x_pure_torch_fwht.py

echo ""
echo "=========================================="
echo "=== Stage C: Re-run VRAM sweep with honest backend label ==="
echo "=========================================="
PYTHONPATH=. python3 scripts/mi300x_vram_sweep.py

echo ""
echo "=========================================="
echo "=== Stage D: Single VRAM (canonical 32K) with honest backend ==="
echo "=========================================="
PYTHONPATH=. python3 scripts/mi300x_vram_measurement.py

echo ""
echo "=========================================="
echo "=== Stage E: Quantization quality comparison ==="
echo "=========================================="
PYTHONPATH=. python3 scripts/mi300x_quant_quality.py

echo ""
echo "=========================================="
echo "=== Stage F: Full pytest regression on MI300X ==="
echo "=========================================="
PYTHONPATH=. python3 -m pytest tests/ --tb=short -q --json-report --json-report-file=logs/mi300x_full_pytest_$(date +%s).json 2>&1 | tail -20

echo ""
echo "=========================================="
echo "All stages done."
echo "Logs in: logs/"
ls -la logs/mi300x_*.json | tail -15
