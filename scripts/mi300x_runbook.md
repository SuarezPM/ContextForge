# MI300X Wave B Runbook — Sprint 3

One-page guide for running the Wave B smoke tests on the AMD AI Dev Cloud droplet.

## Prerequisites

- AMD AI Dev Cloud droplet with MI300X GPU ($30 remaining @ $1.99/hr)
- SSH access configured: `ssh user@<droplet-ip>`
- Git, Python 3.10+, pip, and ROCm 6.x pre-installed on the droplet

## Step 1 — Connect to the droplet

```bash
ssh user@<droplet-ip>
```

## Step 2 — Clone the repo and install

```bash
git clone https://github.com/SuarezPM/Apohara_Context_Forge.git
cd Apohara_Context_Forge
pip install -e .
pip install torch pytest pytest-json-report
```

> Use `pip install --extra-index-url https://download.pytorch.org/whl/rocm6.0 torch` if the default torch does not include ROCm support.

## Step 3 — Verify GPU visibility

```bash
rocm-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True  AMD Instinct MI300X`

## Step 4 — Run scripts in order

### 4a. FWHT integration smoke test

```bash
bash scripts/mi300x_smoke_fwht.sh
```

Output lands in `logs/mi300x_fwht_<timestamp>.json` and `logs/rocm_smi_pre_*.json` / `logs/rocm_smi_post_*.json`.

### 4b. VRAM measurement

```bash
PYTHONPATH=. python3 scripts/mi300x_vram_measurement.py
```

Output: `logs/mi300x_vram_<timestamp>.json`

Key field: `with_fwht.reduction_factor` — should be > 1.0 (FWHT reduces peak VRAM).

### 4c. V6.2 adversarial benchmark

```bash
bash scripts/mi300x_v62_adversarial.sh
```

Output: `logs/mi300x_v62_<timestamp>.json`

## Step 5 — Copy results back

```bash
# From your laptop:
scp -r user@<droplet-ip>:~/Apohara_Context_Forge/logs ./mi300x_wave_b_logs/
```

## Step 6 — Stop the droplet

After results are copied, stop the droplet to avoid burning the remaining credit:

```bash
# In the AMD AI Dev Cloud console — power off or delete the instance.
```

## Expected outputs

| File | What to look for |
|------|-----------------|
| `logs/mi300x_fwht_*.json` | All pytest tests PASSED |
| `logs/rocm_smi_pre_*.json` | GPU visible, VRAM free > 100 GB |
| `logs/mi300x_vram_*.json` | `with_fwht.reduction_factor` > 1.0 |
| `logs/mi300x_v62_*.json` | No Python exceptions in benchmark |

## Budget

$30 remaining @ $1.99/hr ≈ 15 hrs. All four scripts above finish in < 30 min total.
