"""Can FWHT run in fp16 directly (no fp32 upcast)?

The current fwht.py upcasts fp16 input to fp32 before the butterfly to
avoid catastrophic precision loss. This script tests:
  - fp16-only FWHT (no upcast): peak alloc drops dramatically
  - precision: round-trip error vs the fp32-upcast reference

If the error is acceptable (< 1e-2 max), fp16-only is viable for KV
cache use (where the quantization noise is anyway > 1e-2).
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path

import torch


def detect_backend() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    is_rocm = bool(getattr(torch.version, "hip", None))
    name = torch.cuda.get_device_name(0)
    return f"rocm-hip:{torch.version.hip}:{name}" if is_rocm else f"cuda:{torch.version.cuda}:{name}"


def fwht_fp16_inplace(x: torch.Tensor) -> torch.Tensor:
    """FP16 in-place FWHT. No upcast. Self-inverse with orthonormal scaling."""
    d = x.shape[-1]
    if d & (d - 1) != 0:
        raise ValueError(f"power-of-two only, got d={d}")
    h = 1
    while h < d:
        view = x.view(*x.shape[:-1], d // (2 * h), 2, h)
        a = view[..., 0, :]
        b = view[..., 1, :]
        t = b.clone()
        b.copy_(a)
        b.sub_(t)
        a.add_(t)
        h *= 2
    x.mul_(1.0 / math.sqrt(d))
    return x


def fwht_fp32_reference(x: torch.Tensor) -> torch.Tensor:
    """fp32-upcast FWHT (current production behaviour)."""
    orig_dtype = x.dtype
    work = x.to(torch.float32)
    d = work.shape[-1]
    h = 1
    while h < d:
        view = work.view(*work.shape[:-1], d // (2 * h), 2, h)
        a = view[..., 0, :].clone()
        b = view[..., 1, :].clone()
        view[..., 0, :] = a + b
        view[..., 1, :] = a - b
        h *= 2
    work = work / math.sqrt(d)
    return work.to(orig_dtype)


def main() -> int:
    print(f"Hardware: {detect_backend()}")
    print()
    results = []
    for s in (4096, 16384, 32768):
        for h in (32, 64):
            for d in (64, 128, 256):
                x_orig = torch.randn(1, s, h, d, dtype=torch.float16, device="cuda")

                # fp32 reference path (current production)
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                t0 = time.perf_counter()
                y_ref = fwht_fp32_reference(x_orig.clone())
                torch.cuda.synchronize()
                dur_ref = (time.perf_counter() - t0) * 1000.0
                peak_ref = int(torch.cuda.max_memory_allocated())

                # fp16-only path
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
                x16 = x_orig.clone()
                t0 = time.perf_counter()
                y_fp16 = fwht_fp16_inplace(x16)
                torch.cuda.synchronize()
                dur_fp16 = (time.perf_counter() - t0) * 1000.0
                peak_fp16 = int(torch.cuda.max_memory_allocated())

                # Compare
                max_diff = float((y_ref.float() - y_fp16.float()).abs().max().item())
                mse = float(((y_ref.float() - y_fp16.float()) ** 2).mean().item())

                baseline = x_orig.nbytes
                r = {
                    "config": f"s{s}_h{h}_d{d}",
                    "fp32_ref": {"duration_ms": dur_ref, "peak_alloc_bytes": peak_ref,
                                  "overhead_pct": (peak_ref/baseline - 1)*100},
                    "fp16_inplace": {"duration_ms": dur_fp16, "peak_alloc_bytes": peak_fp16,
                                     "overhead_pct": (peak_fp16/baseline - 1)*100},
                    "speedup_x": dur_ref / max(dur_fp16, 1e-6),
                    "precision_max_diff": max_diff,
                    "precision_mse": mse,
                }
                results.append(r)
                print(
                    f"s={s:>5} h={h:>2} d={d:>3}: "
                    f"fp32-ref +{r['fp32_ref']['overhead_pct']:>5.0f}% {dur_ref:>6.1f}ms | "
                    f"fp16     +{r['fp16_inplace']['overhead_pct']:>5.0f}% {dur_fp16:>6.1f}ms | "
                    f"speedup {r['speedup_x']:>4.2f}x | max_diff {max_diff:.2e}"
                )

    ts = int(time.time())
    out = Path("logs") / f"mi300x_fwht_fp16_test_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"hardware": detect_backend(), "results": results}, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
