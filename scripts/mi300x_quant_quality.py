"""Quantization quality comparison on MI300X.

Compares reconstruction quality (MSE + max abs err) of the same KV tensor
under:
  - FP16 baseline (reference)
  - "INT8 simulated" (FP16 → INT8 quantize → dequantize): textbook ~3% MSE
  - INT4 with Apohara's codec (use_fwht=False)
  - INT4 + FWHT rotation
For each: stores reconstruction MSE, max abs err, packed_bytes,
reduction_factor. This is the quality-vs-compression plot for paper v2.0.

Honesty note: 'INT8 simulated' here is naive uniform quantization, not
a full INT8 codec with per-block scales like Apohara's INT4 path. The
goal is to bound the curve, not ship a real INT8 implementation.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch

from apohara_context_forge.quantization.rotate_kv import (
    RotateKVConfig,
    RotateKVQuantizer,
)


def detect_backend() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    is_rocm = bool(getattr(torch.version, "hip", None))
    name = torch.cuda.get_device_name(0)
    return f"rocm-hip:{torch.version.hip}:{name}" if is_rocm else f"cuda:{torch.version.cuda}:{name}"


SEQ_LEN = 16384
NUM_HEADS = 32
HEAD_DIM = 128
BATCH = 1
SEED = 42


def quantize_int8_naive(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel min-max INT8 quantization (along last axis)."""
    arr_min = arr.min(axis=-1, keepdims=True)
    arr_max = arr.max(axis=-1, keepdims=True)
    scale = (arr_max - arr_min) / 255.0
    scale = np.where(scale == 0, 1.0, scale)
    zero = arr_min
    q = np.round((arr - zero) / scale).clip(0, 255).astype(np.uint8)
    return q, scale.astype(np.float32), zero.astype(np.float32)


def dequantize_int8_naive(q: np.ndarray, scale: np.ndarray, zero: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale + zero


def main() -> int:
    print(f"Hardware: {detect_backend()}")
    rng = np.random.default_rng(SEED)
    keys = rng.standard_normal((BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM)).astype(np.float32)
    values = rng.standard_normal(keys.shape).astype(np.float32)
    positions = np.arange(SEQ_LEN, dtype=np.int64)[None, :]

    baseline_fp16_bytes = int(keys.size * 2 + values.size * 2)

    results = {"hardware": detect_backend(), "shape": list(keys.shape), "configs": []}

    # FP16 baseline (lossy cast)
    k_fp16 = keys.astype(np.float16).astype(np.float32)
    v_fp16 = values.astype(np.float16).astype(np.float32)
    mse_fp16_k = float(((k_fp16 - keys) ** 2).mean())
    mse_fp16_v = float(((v_fp16 - values) ** 2).mean())
    max_fp16_k = float(np.abs(k_fp16 - keys).max())
    fp16_bytes = baseline_fp16_bytes
    results["configs"].append({
        "name": "fp16 (reference)",
        "packed_bytes": fp16_bytes,
        "reduction_factor": 1.0,
        "mse_keys": mse_fp16_k,
        "mse_values": mse_fp16_v,
        "max_abs_err_keys": max_fp16_k,
    })
    print(f"fp16 baseline: mse_k={mse_fp16_k:.2e} max_err={max_fp16_k:.2e} bytes={fp16_bytes}")

    # INT8 naive
    t0 = time.perf_counter()
    qk, sk, zk = quantize_int8_naive(keys)
    qv, sv, zv = quantize_int8_naive(values)
    dur_int8 = (time.perf_counter() - t0) * 1000.0
    dk = dequantize_int8_naive(qk, sk, zk)
    dv = dequantize_int8_naive(qv, sv, zv)
    int8_bytes = int(qk.nbytes + qv.nbytes + sk.nbytes + sv.nbytes + zk.nbytes + zv.nbytes)
    mse_int8_k = float(((dk - keys) ** 2).mean())
    max_int8_k = float(np.abs(dk - keys).max())
    results["configs"].append({
        "name": "int8 naive (per-channel min-max)",
        "packed_bytes": int8_bytes,
        "reduction_factor": baseline_fp16_bytes / max(int8_bytes, 1),
        "mse_keys": mse_int8_k,
        "mse_values": float(((dv - values) ** 2).mean()),
        "max_abs_err_keys": max_int8_k,
        "duration_ms": dur_int8,
    })
    print(f"int8 naive:    mse_k={mse_int8_k:.2e} max_err={max_int8_k:.2e} bytes={int8_bytes} red={baseline_fp16_bytes/int8_bytes:.2f}x")

    # INT4 Apohara, use_fwht=False
    for use_fwht in (False, True):
        t0 = time.perf_counter()
        q = RotateKVQuantizer(RotateKVConfig(use_fwht=use_fwht))
        packed, _ = q.quantize_pre_rope(keys, values, positions)
        dk, dv = q.dequantize(packed)
        dur = (time.perf_counter() - t0) * 1000.0
        # Dequant returns seq_len padded up to next block boundary; slice to
        # original input length so MSE compares apples to apples.
        n = keys.shape[1]
        dk = dk[:, :n]
        dv = dv[:, :n]
        mse_k = float(((dk - keys) ** 2).mean())
        max_k = float(np.abs(dk - keys).max())
        int4_bytes = int(
            packed.keys_int4.nbytes + packed.values_int4.nbytes
            + packed.scales_k.nbytes + packed.zero_points_k.nbytes
            + packed.scales_v.nbytes + packed.zero_points_v.nbytes
            + packed.keys_sink_fp16.nbytes + packed.values_sink_fp16.nbytes
        )
        results["configs"].append({
            "name": f"int4 Apohara use_fwht={use_fwht}",
            "use_fwht": use_fwht,
            "packed_bytes": int4_bytes,
            "reduction_factor": baseline_fp16_bytes / max(int4_bytes, 1),
            "mse_keys": mse_k,
            "mse_values": float(((dv - values) ** 2).mean()),
            "max_abs_err_keys": max_k,
            "duration_ms": dur,
        })
        print(f"int4 use_fwht={use_fwht}: mse_k={mse_k:.2e} max_err={max_k:.2e} bytes={int4_bytes} red={baseline_fp16_bytes/int4_bytes:.2f}x dur={dur:.0f}ms")

    ts = int(time.time())
    out = Path("logs") / f"mi300x_quant_quality_{ts}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
