"""Benchmark in-place FWHT vs current cloning FWHT on MI300X.

Sprint 4 candidate validation: does the in-place rewrite actually drop
the +700% peak alloc overhead measured in V7.0.0-alpha.5?
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import torch

from apohara_context_forge.quantization.fwht import fwht as fwht_original
from apohara_context_forge.quantization.fwht_inplace import fwht_inplace


def detect_backend() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    is_rocm = bool(getattr(torch.version, "hip", None))
    name = torch.cuda.get_device_name(0)
    return f"rocm-hip:{torch.version.hip}:{name}" if is_rocm else f"cuda:{torch.version.cuda}:{name}"


CONFIGS = [
    # (batch, seq_len, num_heads, head_dim)
    (1,  4096, 32, 128),
    (1, 16384, 32, 128),
    (1, 32768, 32, 128),
    (1, 16384, 32, 256),
    (1, 16384, 64, 128),
]


def measure(impl, x, label):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    # Warm-up
    impl(x.clone() if label == "inplace" else x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    # Timed
    x_copy = x.clone() if label == "inplace" else x  # inplace mutates input
    t0 = time.perf_counter()
    out = impl(x_copy)
    torch.cuda.synchronize()
    dur = (time.perf_counter() - t0) * 1000.0
    peak = int(torch.cuda.max_memory_allocated())
    return out, peak, dur


def main() -> int:
    print(f"Hardware: {detect_backend()}")
    print()
    results = []
    for b, s, h, d in CONFIGS:
        baseline = b * s * h * d * 2  # fp16 bytes
        x = torch.randn(b, s, h, d, dtype=torch.float16, device="cuda")

        # Current (cloning) impl
        torch.cuda.empty_cache()
        out_orig, peak_orig, dur_orig = measure(fwht_original, x, "orig")
        # In-place impl
        torch.cuda.empty_cache()
        out_inplace, peak_inplace, dur_inplace = measure(fwht_inplace, x, "inplace")

        # Equivalence check
        max_diff = float((out_orig - out_inplace).abs().max().item())

        ovh_orig = (peak_orig / baseline - 1) * 100
        ovh_inplace = (peak_inplace / baseline - 1) * 100
        speedup = dur_orig / max(dur_inplace, 1e-6)

        r = {
            "config": f"b{b}_s{s}_h{h}_d{d}",
            "baseline_bytes": baseline,
            "original": {
                "peak_alloc_bytes": peak_orig,
                "overhead_pct": ovh_orig,
                "duration_ms": dur_orig,
            },
            "inplace": {
                "peak_alloc_bytes": peak_inplace,
                "overhead_pct": ovh_inplace,
                "duration_ms": dur_inplace,
            },
            "speedup_x": speedup,
            "equivalence_max_diff": max_diff,
        }
        results.append(r)
        print(
            f"s={s:>5} h={h:>2} d={d:>3}: "
            f"orig ovh +{ovh_orig:>5.1f}% {dur_orig:>6.1f}ms | "
            f"inplace ovh +{ovh_inplace:>5.1f}% {dur_inplace:>6.1f}ms | "
            f"speedup {speedup:>4.2f}x | equiv max_diff {max_diff:.2e}"
        )

    ts = int(time.time())
    out = Path("logs") / f"mi300x_fwht_inplace_bench_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "hardware": detect_backend(),
        "results": results,
    }, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
