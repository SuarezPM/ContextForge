"""VRAM scaling sweep on MI300X — Sprint 3 Wave B repurposed Stage 3.

Replaces the V6.2 adversarial bench (pure CPU sim, no GPU value) with a
shape-scaling sweep that gives paper v2.0 the reduction_factor curve as
context length and head_dim vary. Run on MI300X droplet.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import scripts.mi300x_vram_measurement as m

CONFIGS = [
    # seq_len, num_heads, head_dim
    (4096,  32, 128),
    (8192,  32, 128),
    (16384, 32, 128),
    (32768, 32, 128),
    (16384, 32, 64),
    (16384, 32, 256),
    (16384, 16, 128),
    (16384, 64, 128),
]


def main() -> int:
    results = []
    for seq, nh, hd in CONFIGS:
        m.SEQ_LEN, m.NUM_HEADS, m.HEAD_DIM = seq, nh, hd
        for use_fwht in (False, True):
            r = m.measure(use_fwht)
            r["config"] = f"seq{seq}_nh{nh}_hd{hd}"
            results.append(r)
            base_mb = r["baseline_fp16_bytes"] / 1e6
            pkd_mb = r["packed_bytes"] / 1e6
            fac = r["reduction_factor"]
            secs = r["duration_ms"] / 1000.0
            print(
                f"seq={seq:>5} nh={nh:>2} hd={hd:>3} fwht={str(use_fwht):>5} "
                f"baseline={base_mb:>7.1f}MB packed={pkd_mb:>7.1f}MB "
                f"reduction={fac:>5.2f}x dur={secs:>6.1f}s"
            )

    ts = int(time.time())
    out = Path("logs") / f"mi300x_vram_sweep_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
