# Sprint 5 MI300X Runbook — V8 codec + vLLM e2e + head-to-head

> **Purpose:** turn a fresh AMD MI300X droplet into Sprint 5 results
> in under 14 hours of compute. Single SSH session, copy-paste lines
> from this file top to bottom.
>
> **Budget envelope:** 14 hours @ $1.99/hr = $27.86. Plan: tear down
> the droplet the moment the last `scp` of `logs/` lands locally — no
> idle hours.
>
> **Prereq:** credit grant from AMD Developer Cloud (see
> `outreach/drafts/amd-free-credit-application.md`) OR personal
> budget approved.

---

## Step 0 — Local side: bookmark these paths

```bash
# On the laptop, before SSH:
export DROPLET_USER=user                  # adjust if AMD provisions different
export DROPLET_IP=<fill-in-from-AMD>      # e.g. 129.212.188.18 in Sprint 3
export REMOTE_REPO=/home/user/Apohara_Context_Forge
export LOCAL_LOGS=$HOME/Documentos/Apohara_Context_Forge/logs

# SSH options (Sprint 3 lesson learned — DO droplets need this):
export SSH_OPTS="-4 -o IPQoS=throughput -o ServerAliveInterval=60"
```

## Step 1 — Provision (15 minutes)

```bash
# Connect:
ssh $SSH_OPTS $DROPLET_USER@$DROPLET_IP

# Inside droplet:
git clone https://github.com/SuarezPM/Apohara_Context_Forge.git
cd Apohara_Context_Forge
pip install -e .
pip install torch pytest pytest-json-report --extra-index-url https://download.pytorch.org/whl/rocm6.2

# Smoke-check GPU:
rocm-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  AMD Instinct MI300X
```

**Cost gate:** if `torch.cuda.is_available()` is False, abort and
file a ticket with AMD support before burning more credit. The 192 GB
HBM3 only counts as "available" through `torch.cuda.*` once
`HIP_VISIBLE_DEVICES` is set correctly.

```bash
# If torch can't see the GPU, also try:
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Step 2 — Item 1: V8 codec validation (~3 hours)

```bash
# Capture KV snapshots first (10 minutes):
PYTHONPATH=. python3 scripts/capture_kv_snapshots.py \
    --model llama-3-8b --n-snapshots 5 --out logs/kv_snapshots/

# Run V7 vs V8 comparison:
PYTHONPATH=. python3 scripts/sprint5_v8_codec.py \
    --kv-dir logs/kv_snapshots/ \
    --out logs/mi300x_codec_v8_$(date +%s).json

# Eyeball the reduction factor:
jq '.results[].reduction' logs/mi300x_codec_v8_*.json | tail -5
# Acceptance: >= 3.80x. If V8 is worse than V7, document why in AUDIT.md.
```

**Cost gate:** budget 3 hours including re-runs. If at hour 3 the V8
metric is still < 3.80×, log the result honestly and move on. The
paper v2.1 can ship the negative result with the same discipline as
the V7.0.0-alpha.5 FWHT-degradation finding.

## Step 3 — Item 2: vLLM end-to-end demo (~5 hours)

```bash
# Install vLLM ROCm:
pip install vllm

# Pull Llama-3-8B (do it once, cache survives across runs):
huggingface-cli download meta-llama/Llama-3-8B \
    --local-dir /home/user/models/llama-3-8b

# Start vLLM server with Apohara plugin:
PYTHONPATH=. python3 -m apohara_context_forge.vllm_plugin.serve \
    --model /home/user/models/llama-3-8b \
    --apohara-config configs/sprint5_5agent.yaml \
    --port 8000 &

VLLM_PID=$!
sleep 60

# In another shell on the droplet:
PYTHONPATH=. python3 scripts/sprint5_5agent_workload.py \
    --vllm-endpoint http://localhost:8000 \
    --n-requests 200 \
    --out logs/mi300x_vllm_e2e_$(date +%s).json

# Inspect INV-15 firings:
grep '"inv15_fired": true' logs/mi300x_vllm_e2e_*.json | wc -l
# Expected: > 0 for the judge agent role with reuse_rate >= 0.7

kill $VLLM_PID
```

**Cost gate:** budget 5 hours. If vLLM cold-start exceeds 30 minutes
twice, escalate: it's a ROCm/vLLM mismatch, not a real workload
problem.

## Step 4 — Item 3: head-to-head benchmark (~4 hours)

```bash
# Same 5-agent workload, Apohara plugin ON vs OFF:
for mode in apohara_on apohara_off; do
    PYTHONPATH=. python3 scripts/sprint5_head_to_head.py \
        --mode $mode \
        --n-requests 500 \
        --out logs/mi300x_h2h_${mode}_$(date +%s).json
done

# Compare JCR (Judge Consistency Rate):
jq '.summary.jcr' logs/mi300x_h2h_apohara_off_*.json
jq '.summary.jcr' logs/mi300x_h2h_apohara_on_*.json
# Expected: apohara_off JCR is 0.77-0.92 (Liang et al. range);
#           apohara_on JCR is 0.99+ (INV-15 prevents the silent drop)
```

**Cost gate:** budget 4 hours including 2 re-runs. The H2H result is
the **single most important data point** for paper v2.1 §6 — do not
skip even if Step 2-3 took longer than estimated.

## Step 5 — Data hygiene + extraction (~30 minutes)

```bash
# Add raw rocm-smi snapshots to every JSON for honesty audit:
for log in logs/mi300x_*_$(date +%Y%m)*.json; do
    bash scripts/check_honesty.sh "$log"
done

# Pull everything to laptop:
exit  # disconnect from droplet
scp $SSH_OPTS -r $DROPLET_USER@$DROPLET_IP:$REMOTE_REPO/logs/mi300x_codec_v8_* \
    $LOCAL_LOGS/
scp $SSH_OPTS -r $DROPLET_USER@$DROPLET_IP:$REMOTE_REPO/logs/mi300x_vllm_e2e_* \
    $LOCAL_LOGS/
scp $SSH_OPTS -r $DROPLET_USER@$DROPLET_IP:$REMOTE_REPO/logs/mi300x_h2h_* \
    $LOCAL_LOGS/

# Acceptance: each ~/logs/mi300x_*.json has nonzero size and the
# rocm-smi snapshots are present.

ls -la $LOCAL_LOGS/mi300x_*$(date +%Y%m)*.json
```

## Step 6 — Tear down (1 minute)

**The most important step.** Idle droplets bleed credit.

1. AMD Developer Cloud dashboard → Droplets
2. Find the Sprint 5 droplet by IP
3. "Destroy droplet" — confirm
4. Credit usage report: download and add to `release/sprint5_cost.json`

```bash
# Confirm destroy from CLI (optional, if `doctl` is installed):
doctl compute droplet list | grep $DROPLET_IP
# Empty output = destroyed. Move on.
```

## Step 7 — Commit + release (1 hour, local)

```bash
cd $HOME/Documentos/Apohara_Context_Forge
git add logs/mi300x_codec_v8_*.json logs/mi300x_vllm_e2e_*.json logs/mi300x_h2h_*.json

# AUDIT.md update: add Sprint 5 line items
$EDITOR AUDIT.md

# Paper v2.1 update: add measured V8 row to Table 3,
# add JCR head-to-head row to §6
$EDITOR paper/inv15_paper.tex
tectonic paper/inv15_paper.tex

# Commit:
git commit -s -m "feat(sprint5): V8 codec + vLLM e2e + head-to-head on MI300X" \
              -m "" \
              -m "Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"

git push origin HEAD:main
```

## Cost ledger template

```
Item                    Plan h  Actual h  Cost@$1.99  Status
─────────────────────── ────── ───────── ─────────── ──────
0. Provision               0.25         ?    $    ?    ⬜
1. V8 codec validation     3.0          ?    $    ?    ⬜
2. vLLM e2e demo           5.0          ?    $    ?    ⬜
3. Head-to-head            4.0          ?    $    ?    ⬜
4. Data hygiene            0.5          ?    $    ?    ⬜
5. Tear down               0.0          ?    $    ?    ⬜
                                              ─────────
                          TOTAL              $    ?
```

Update this ledger as you go. Hard stop at hour 14 regardless of
completion — partial Sprint 5 results are still paper-shippable.

---

## Fallback budget plan (no credit grant, $15 personal)

If AMD free credits don't come through and the user only authorizes
~$15 personal spend, run **only Step 4 (Item 3, head-to-head)**:

```
Provision           0.25 h
Step 4 only         4.0 h
Data hygiene        0.5 h
Tear down           0.0 h
                  ─────
TOTAL               4.75 h  →  $9.45
```

The H2H is the single most paper-critical datum. V8 codec can be
deferred to a later mini-sprint funded by either AMD DevRel response
to publication outreach, or a second free-credit grant cycle.

---

## What this runbook does NOT cover

- **Item 4 (K8s operator on GCP GKE)**: separate runbook
  `scripts/sprint5_gcp_gke_runbook.md` (to be authored) — uses the
  $300 GCP credits, runs in parallel to the MI300X work
- **vLLM ROCm version pinning**: if vLLM upstream main breaks ROCm
  build, fall back to the last known-good tag (record it here when
  Sprint 5 actually runs)
- **HuggingFace Space refresh**: separate task, no MI300X compute
  needed once the H2H result is in hand

---

## Honesty discipline reminders

- Every measurement script writes to `logs/mi300x_*.json` with a
  timestamp. **Never reuse old timestamps**, never edit JSON post-hoc.
- If a number doesn't make it into AUDIT.md before commit, it
  doesn't go in the paper.
- If a step fails, log the failure in `logs/mi300x_sprint5_failures.md`
  with: command, expected output, actual output, root cause. Don't
  silently retry — the failure log is paper-shippable methodology.

See AUDIT.md and CLAUDE.md §3 for the full verification protocol.
