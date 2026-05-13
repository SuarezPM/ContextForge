"""Pure torch-on-GPU FWHT measurement on MI300X.

Avoids the CPU NumPy bridge entirely: allocates KV cache as torch.float16
on the MI300X, calls `fwht()` directly on GPU tensors (the FWHT module
supports torch.Tensor natively), measures pure on-GPU peak alloc and
duration. This answers: 'is the 3.97x literature target achievable
end-to-end on GPU, given Apohara's INT4 codec?'

The answer is partial: this script measures FWHT alone (the rotation
step), not the full quantize_pre_rope which still needs NumPy. But it
shows the per-step GPU envelope and time, which is the most expensive
part of RotateKV at large seq.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import torch

from apohara_context_forge.quantization.fwht import fwht as fwht_fn


def detect_backend() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    is_rocm = bool(getattr(torch.version, "hip", None))
    name = torch.cuda.get_device_name(0)
    if is_rocm:
        return f"rocm-hip:{torch.version.hip}:{name}"
    return f"cuda:{torch.version.cuda}:{name}"


CONFIGS = [
    # (batch, seq_len, num_heads, head_dim)
    (1,  4096, 32, 128),
    (1,  8192, 32, 128),
    (1, 16384, 32, 128),
    (1, 32768, 32, 128),
    (1, 16384, 32,  64),
    (1, 16384, 32, 256),
    (1, 16384, 16, 128),
    (1, 16384, 64, 128),
]


def measure(b: int, s: int, h: int, d: int) -> dict:
    if not torch.cuda.is_available():
        return {"error": "no GPU available"}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    keys = torch.randn(b, s, h, d, dtype=torch.float16, device="cuda")
    baseline_bytes = keys.element_size() * keys.nelement()
    pre_peak = int(torch.cuda.max_memory_allocated())

    # Warm-up (compile / kernel cache)
    _ = fwht_fn(keys)
    torch.cuda.synchronize()

    # Timed run
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    rotated = fwht_fn(keys)
    torch.cuda.synchronize()
    duration_ms = (time.perf_counter() - t0) * 1000.0

    fwht_peak = int(torch.cuda.max_memory_allocated())

    # Round-trip identity check on a slice (FWHT is self-inverse under our
    # orthonormal normalization, so ifwht == fwht)
    derot = fwht_fn(rotated)
    torch.cuda.synchronize()
    max_err = float((derot - keys).abs().max().item())

    return {
        "hardware": detect_backend(),
        "batch": b, "seq_len": s, "num_heads": h, "head_dim": d,
        "baseline_fp16_bytes": int(baseline_bytes),
        "pre_fwht_peak_alloc_bytes": pre_peak,
        "during_fwht_peak_alloc_bytes": fwht_peak,
        "fwht_overhead_bytes": int(fwht_peak - baseline_bytes),
        "fwht_overhead_factor": (fwht_peak / max(baseline_bytes, 1)),
        "fwht_duration_ms": duration_ms,
        "fwht_throughput_gbps": (baseline_bytes / 1e9) / (duration_ms / 1000.0),
        "roundtrip_max_abs_err": max_err,
    }


def main() -> int:
    results = []
    print(f"Hardware: {detect_backend()}")
    print()
    for b, s, h, d in CONFIGS:
        r = measure(b, s, h, d)
        results.append(r)
        gb = r["baseline_fp16_bytes"] / 1e9
        ovh_pct = (r["fwht_overhead_factor"] - 1) * 100
        print(
            f"seq={s:>5} h={h:>2} d={d:>3}: baseline={gb:>5.3f}GB "
            f"fwht_overhead={ovh_pct:>+5.1f}% time={r['fwht_duration_ms']:>7.1f}ms "
            f"thrpt={r['fwht_throughput_gbps']:>5.2f}GB/s err={r['roundtrip_max_abs_err']:.2e}"
        )
    ts = int(time.time())
    out = Path("logs") / f"mi300x_pure_torch_fwht_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
