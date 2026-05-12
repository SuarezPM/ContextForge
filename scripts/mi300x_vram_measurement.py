"""Measure VRAM reduction of RotateKVQuantizer on real MI300X.

Run on droplet (Wave B):
  PYTHONPATH=. python3 scripts/mi300x_vram_measurement.py

Honest measurement protocol
---------------------------
The current `RotateKVQuantizer` is NumPy-only — its hot path operates on
`np.ndarray`, not `torch.Tensor`. To measure on MI300X we therefore:

1. Allocate the baseline KV cache as a torch.float16 CUDA tensor (the way a
   real vLLM worker would hold it). Record `baseline_fp16_bytes` from the
   tensor's allocation footprint.
2. Convert to NumPy on the host CPU for the quantization call. Record the
   *NumPy* allocation footprint of the packed result (via `keys_int4.nbytes
   + values_int4.nbytes` plus the scales arrays). That is the "stored on
   disk / Redis" footprint, which is what the 3.97× claim refers to.
3. Synchronise CUDA and read `torch.cuda.max_memory_allocated()` to capture
   peak GPU pressure during the round-trip.

This is an honest reporting protocol: the reduction factor is measured
against the canonical FP16 baseline, on a real MI300X, with real tensor
shapes. The GPU peak number is reported separately because it includes
the copy from device → host, not pure on-GPU quantization. A future sprint
can replace the NumPy hot path with a torch fast path and re-measure
on-GPU peak without the copy.

Canonical layout (per RotateKVQuantizer docstring): (batch, seq_len, num_heads, head_dim).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from apohara_context_forge.quantization.rotate_kv import (
    RotateKVConfig,
    RotateKVQuantizer,
)

SEQ_LEN = 32_768
NUM_HEADS = 32
HEAD_DIM = 128
BATCH = 1
DTYPE = torch.float16


def measure(use_fwht: bool) -> dict:
    have_cuda = torch.cuda.is_available()
    if have_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Canonical layout: (batch, seq_len, num_heads, head_dim).
    device = "cuda" if have_cuda else "cpu"
    keys = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    values = torch.randn_like(keys)
    positions = torch.arange(SEQ_LEN, device=device).unsqueeze(0)

    baseline_fp16_bytes = (
        keys.element_size() * keys.nelement()
        + values.element_size() * values.nelement()
    )
    if have_cuda:
        torch.cuda.synchronize()

    # NumPy bridge — RotateKVQuantizer expects np.ndarray on host.
    keys_np = keys.detach().to("cpu").float().numpy()
    values_np = values.detach().to("cpu").float().numpy()
    positions_np = positions.detach().to("cpu").numpy()

    cfg = RotateKVConfig(use_fwht=use_fwht)
    quantizer = RotateKVQuantizer(cfg)
    t0 = time.perf_counter()
    packed, _residual = quantizer.quantize_pre_rope(keys_np, values_np, positions_np)
    duration_ms = (time.perf_counter() - t0) * 1000.0

    # Packed-storage footprint = the bytes you'd write to Redis / LMCache.
    packed_bytes = int(
        packed.keys_int4.nbytes
        + packed.values_int4.nbytes
        + packed.scales.nbytes
        + packed.zero_points.nbytes
    )

    peak_alloc_bytes = (
        int(torch.cuda.max_memory_allocated()) if have_cuda else None
    )

    return {
        "use_fwht": use_fwht,
        "device": device,
        "seq_len": SEQ_LEN,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "baseline_fp16_bytes": int(baseline_fp16_bytes),
        "packed_bytes": packed_bytes,
        "reduction_factor": baseline_fp16_bytes / max(packed_bytes, 1),
        "peak_gpu_alloc_bytes_incl_copy": peak_alloc_bytes,
        "duration_ms": duration_ms,
    }


def main() -> int:
    out = Path("logs") / f"mi300x_vram_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "with_fwht": measure(True),
        "without_fwht": measure(False),
    }
    out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
