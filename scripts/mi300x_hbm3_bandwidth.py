"""HBM3 bandwidth probe on MI300X.

MI300X advertises 5.3 TB/s HBM3 peak. This script measures effective
read+write bandwidth via large torch tensor copies, plus a sustained
read-modify-write loop. Numbers go into the paper v2.0 §3 (hardware
context) as "achieved bandwidth, not advertised peak".
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import torch


def detect_backend() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    is_rocm = bool(getattr(torch.version, "hip", None))
    name = torch.cuda.get_device_name(0)
    return f"rocm-hip:{torch.version.hip}:{name}" if is_rocm else f"cuda:{torch.version.cuda}:{name}"


def measure_one(size_gb: float, iters: int = 5) -> dict:
    """Allocate `size_gb` of fp16 + measure copy/triadic bandwidth."""
    if not torch.cuda.is_available():
        return {"error": "no GPU"}
    nelem = int(size_gb * 1e9 / 2)  # fp16 = 2 bytes
    a = torch.empty(nelem, dtype=torch.float16, device="cuda")
    b = torch.empty_like(a)
    a.uniform_(-1.0, 1.0)

    # Warm
    b.copy_(a)
    torch.cuda.synchronize()

    # Copy bandwidth (read + write)
    t0 = time.perf_counter()
    for _ in range(iters):
        b.copy_(a)
    torch.cuda.synchronize()
    copy_dur = (time.perf_counter() - t0) / iters
    copy_bw_gbps = (2 * a.nbytes / copy_dur) / 1e9  # both read+write

    # STREAM triad: b = a * alpha + b  (read a, read b, write b)
    alpha = torch.tensor(1.5, device="cuda", dtype=torch.float16)
    t0 = time.perf_counter()
    for _ in range(iters):
        b.add_(a, alpha=1.5)
    torch.cuda.synchronize()
    triad_dur = (time.perf_counter() - t0) / iters
    triad_bw_gbps = (3 * a.nbytes / triad_dur) / 1e9

    del a, b
    torch.cuda.empty_cache()

    return {
        "size_gb": size_gb,
        "iters": iters,
        "copy_duration_ms": copy_dur * 1000.0,
        "copy_bw_gbps": copy_bw_gbps,
        "triad_duration_ms": triad_dur * 1000.0,
        "triad_bw_gbps": triad_bw_gbps,
    }


def main() -> int:
    print(f"Hardware: {detect_backend()}")
    print()
    results = {"hardware": detect_backend(), "measurements": []}

    for size_gb in [1.0, 4.0, 16.0, 64.0]:
        r = measure_one(size_gb)
        results["measurements"].append(r)
        print(f"size={size_gb:>5.1f} GB: copy={r['copy_bw_gbps']:>7.1f} GB/s triad={r['triad_bw_gbps']:>7.1f} GB/s")

    # Theoretical 5.3 TB/s
    best = max(m["triad_bw_gbps"] for m in results["measurements"])
    print(f"\nBest triad: {best:.1f} GB/s = {best/1000:.2f} TB/s")
    print(f"Vs advertised 5.3 TB/s peak: {best/5300*100:.1f}% efficiency")
    results["peak_advertised_tbps"] = 5.3
    results["best_measured_triad_gbps"] = best
    results["efficiency_pct"] = best / 5300 * 100

    ts = int(time.time())
    out = Path("logs") / f"mi300x_hbm3_bandwidth_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
