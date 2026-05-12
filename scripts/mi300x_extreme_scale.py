"""Extreme-scale validation on MI300X: seq up to 262K.

Paper v2.0 claim: reduction_factor of 3.55x holds at extreme context
lengths (well beyond the 32K used in prior measurements). MI300X 192 GB
VRAM accommodates these workloads where consumer GPUs cannot.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import scripts.mi300x_vram_measurement as m


CONFIGS = [
    (65536,  32, 128),    # 1 GB baseline
    (131072, 32, 128),    # 2 GB baseline
    (65536,  32, 256),    # 2 GB baseline (wider head_dim)
    (262144, 16, 128),    # 2 GB baseline (extreme context)
]


def main() -> int:
    results = []
    for seq, nh, hd in CONFIGS:
        m.SEQ_LEN, m.NUM_HEADS, m.HEAD_DIM = seq, nh, hd
        r = m.measure(False)
        r["config"] = f"seq{seq}_nh{nh}_hd{hd}"
        results.append(r)
        gb_base = r["baseline_fp16_bytes"] / 1e9
        gb_pkd = r["packed_bytes"] / 1e9
        rf = r["reduction_factor"]
        secs = r["duration_ms"] / 1000.0
        print(
            f"seq={seq:>6} nh={nh:>2} hd={hd:>3} fwht=False "
            f"baseline={gb_base:>5.2f}GB packed={gb_pkd:>5.2f}GB "
            f"reduction={rf:>5.2f}x dur={secs:>6.1f}s"
        )
    ts = int(time.time())
    out = Path("logs") / f"mi300x_extreme_scale_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
