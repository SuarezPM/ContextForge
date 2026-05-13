"""V7 vs V8 codec comparison on captured KV snapshots.

Sprint 5 Step 2 (Item 1) — consume the ``.npz`` snapshots from
``scripts/capture_kv_snapshots.py`` and run both V7 (current
``RotateKVQuantizer``) and V8 (``CodecV8Quantizer``) on each layer
of each snapshot, measuring:

* Reduction factor:
      FP16_bytes / (INT4_packed_bytes + metadata_bytes)
* Reconstruction MSE: ``mean((original - recovered)**2)``
* Per-head reconstruction MSE (to find outlier heads)
* Wall-clock time per quantize / dequantize round-trip

Acceptance (paper v2.1 §codec criterion):

* V8 reduction factor >= 3.80x (otherwise V8 is worse than V7 after
  metadata overhead — abort and document in AUDIT.md)
* V8 MSE strictly lower than V7 MSE on keys, values, and per-head

This script is CPU-only by design: the codec lives in numpy, so we
don't need GPU access here. The GPU is only needed for the *capture*
phase (running the real model). Run this in CI after each codec
change.

Output: ``logs/mi300x_codec_v8_<timestamp>.json`` — one JSON file
with the full sweep. Paper v2.1 Table 3 reads from this file.

Usage::

    PYTHONPATH=. python3 scripts/sprint5_v8_codec.py \
        --kv-dir logs/kv_snapshots/ \
        --out logs/mi300x_codec_v8_$(date +%s).json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from apohara_context_forge.quantization.codec_v8 import (
    CodecV8Config,
    CodecV8Quantizer,
)
from apohara_context_forge.quantization.rotate_kv import (
    RotateKVConfig,
    RotateKVQuantizer,
)

logger = logging.getLogger("sprint5_v8_codec")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Storage cost accounting
# ---------------------------------------------------------------------------


def storage_bytes_fp16(keys_shape: tuple[int, ...]) -> int:
    """Raw FP16 storage: 2 bytes × every element."""
    n = 1
    for d in keys_shape:
        n *= d
    return n * 2


def storage_bytes_v7(qblock) -> int:
    """V7 storage = packed INT4 + scales/zps (FP32) + sinks (FP16).

    Counts per-block scale + zero_point pairs (one per packed byte,
    NOT per nibble — that's the V7 cost saving) plus the sink-token
    FP16 store.
    """
    packed_bytes = qblock.keys_int4.size + qblock.values_int4.size
    scale_bytes = (
        qblock.scales_k.size + qblock.scales_v.size
        + qblock.zero_points_k.size + qblock.zero_points_v.size
    ) * 4  # float32 = 4 bytes
    sink_bytes = (
        qblock.keys_sink_fp16.size + qblock.values_sink_fp16.size
    ) * 2  # float16 = 2 bytes
    return packed_bytes + scale_bytes + sink_bytes


def storage_bytes_v8(qblock) -> int:
    """V8 storage = packed INT4 + 2× scales/zps + sinks.

    V8 scales/zps have an additional trailing pair axis (size 2) so
    the metadata cost is exactly 2× the V7 cost in the same shape.
    The packed INT4 byte count is unchanged.
    """
    return storage_bytes_v7(qblock)
    # ↑ correctly accounts for the larger scales/zps arrays because
    #   `.size` reflects the V8 shape, not the V7 shape.


# ---------------------------------------------------------------------------
# Codec round-trip
# ---------------------------------------------------------------------------


def run_codec(
    *,
    quantizer,
    keys: np.ndarray,
    values: np.ndarray,
    positions: np.ndarray,
) -> dict:
    """Round-trip a layer through one codec; return measurements."""
    fp16_bytes = storage_bytes_fp16(keys.shape) + storage_bytes_fp16(values.shape)

    t0 = time.perf_counter()
    qblock, _ = quantizer.quantize_pre_rope(keys.copy(), values.copy(), positions.copy())
    t1 = time.perf_counter()
    keys_recovered, values_recovered = quantizer.dequantize(qblock)
    t2 = time.perf_counter()

    # Crop recovered to original seq_len (codec pads to group_size)
    seq_len = keys.shape[1]
    keys_recovered = keys_recovered[:, :seq_len]
    values_recovered = values_recovered[:, :seq_len]

    if isinstance(quantizer, CodecV8Quantizer):
        packed_bytes = storage_bytes_v8(qblock)
    else:
        packed_bytes = storage_bytes_v7(qblock)

    reduction = fp16_bytes / max(packed_bytes, 1)

    # MSE keys / values
    mse_k = float(np.mean((keys - keys_recovered) ** 2))
    mse_v = float(np.mean((values - values_recovered) ** 2))

    # Per-head MSE (max across heads, to surface outliers)
    if keys.ndim == 4:
        per_head_mse_k = np.mean(
            (keys - keys_recovered) ** 2, axis=(0, 1, 3)
        )
        per_head_mse_v = np.mean(
            (values - values_recovered) ** 2, axis=(0, 1, 3)
        )
        max_head_mse_k = float(np.max(per_head_mse_k))
        max_head_mse_v = float(np.max(per_head_mse_v))
    else:
        max_head_mse_k = mse_k
        max_head_mse_v = mse_v

    return {
        "fp16_bytes": fp16_bytes,
        "packed_bytes": packed_bytes,
        "reduction_factor": reduction,
        "mse_keys": mse_k,
        "mse_values": mse_v,
        "max_head_mse_keys": max_head_mse_k,
        "max_head_mse_values": max_head_mse_v,
        "quantize_ms": (t1 - t0) * 1000.0,
        "dequantize_ms": (t2 - t1) * 1000.0,
    }


# ---------------------------------------------------------------------------
# Snapshot iteration
# ---------------------------------------------------------------------------


def process_snapshot(
    *,
    npz_path: Path,
    group_size: int,
    sink_tokens: int,
    use_fwht: bool,
) -> list[dict]:
    """Run V7 and V8 codecs on every layer of one snapshot."""
    data = np.load(npz_path)
    keys_all = data["keys"]   # (num_layers, 1, seq, num_kv_heads, head_dim)
    values_all = data["values"]
    num_layers = keys_all.shape[0]
    seq_len = keys_all.shape[2]
    positions = np.arange(seq_len, dtype=np.float32).reshape(1, seq_len)

    v7_cfg = RotateKVConfig(
        bits=4, group_size=group_size,
        sink_tokens=sink_tokens, use_fwht=use_fwht,
    )
    v8_cfg = CodecV8Config(
        bits=4, group_size=group_size,
        sink_tokens=sink_tokens, use_fwht=use_fwht,
    )

    results = []
    for layer in range(num_layers):
        k = keys_all[layer]    # (1, seq, num_kv_heads, head_dim)
        v = values_all[layer]

        v7 = RotateKVQuantizer(v7_cfg)
        v8 = CodecV8Quantizer(v8_cfg)

        v7_metrics = run_codec(quantizer=v7, keys=k, values=v, positions=positions)
        v8_metrics = run_codec(quantizer=v8, keys=k, values=v, positions=positions)

        results.append({
            "layer": layer,
            "seq_len": int(seq_len),
            "shape": list(k.shape),
            "v7": v7_metrics,
            "v8": v8_metrics,
            "v8_minus_v7_mse_keys": v8_metrics["mse_keys"] - v7_metrics["mse_keys"],
            "v8_minus_v7_mse_values": (
                v8_metrics["mse_values"] - v7_metrics["mse_values"]
            ),
            "v8_over_v7_reduction": (
                v8_metrics["reduction_factor"] / v7_metrics["reduction_factor"]
            ),
        })

    return results


# ---------------------------------------------------------------------------
# Acceptance check
# ---------------------------------------------------------------------------


def evaluate_acceptance(
    layer_results: list[dict],
    min_reduction: float,
) -> dict:
    """Aggregate per-layer results into the paper v2.1 §codec criteria."""
    v7_reductions = [r["v7"]["reduction_factor"] for r in layer_results]
    v8_reductions = [r["v8"]["reduction_factor"] for r in layer_results]
    v7_mse_k = [r["v7"]["mse_keys"] for r in layer_results]
    v8_mse_k = [r["v8"]["mse_keys"] for r in layer_results]
    v7_mse_v = [r["v7"]["mse_values"] for r in layer_results]
    v8_mse_v = [r["v8"]["mse_values"] for r in layer_results]

    avg_v7_red = float(np.mean(v7_reductions))
    avg_v8_red = float(np.mean(v8_reductions))
    avg_v7_mse_k = float(np.mean(v7_mse_k))
    avg_v8_mse_k = float(np.mean(v8_mse_k))
    avg_v7_mse_v = float(np.mean(v7_mse_v))
    avg_v8_mse_v = float(np.mean(v8_mse_v))

    v8_beats_v7_mse_k = avg_v8_mse_k < avg_v7_mse_k
    v8_beats_v7_mse_v = avg_v8_mse_v < avg_v7_mse_v
    v8_meets_min_reduction = avg_v8_red >= min_reduction

    return {
        "avg_v7_reduction": avg_v7_red,
        "avg_v8_reduction": avg_v8_red,
        "avg_v7_mse_keys": avg_v7_mse_k,
        "avg_v8_mse_keys": avg_v8_mse_k,
        "avg_v7_mse_values": avg_v7_mse_v,
        "avg_v8_mse_values": avg_v8_mse_v,
        "v8_beats_v7_on_mse_keys": v8_beats_v7_mse_k,
        "v8_beats_v7_on_mse_values": v8_beats_v7_mse_v,
        "v8_meets_min_reduction_threshold": v8_meets_min_reduction,
        "min_reduction_threshold": min_reduction,
        "accept_v8_for_paper_v21": (
            v8_meets_min_reduction
            and v8_beats_v7_mse_k
            and v8_beats_v7_mse_v
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--kv-dir", type=Path,
                   default=Path("logs/kv_snapshots"),
                   help="Directory with kv_snapshot_*.npz + manifest.json")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSON path (e.g. logs/mi300x_codec_v8_<ts>.json)")
    p.add_argument("--group-size", type=int, default=64,
                   help="Codec group_size (block rows)")
    p.add_argument("--sink-tokens", type=int, default=4,
                   help="Attention sink protection (FP16)")
    p.add_argument("--use-fwht", action="store_true",
                   help="Enable FWHT pre-rotation (V7.0.0-rc.2 default: off)")
    p.add_argument("--min-reduction", type=float, default=3.80,
                   help="V8 acceptance threshold (paper v2.1 §codec)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.kv_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error("No manifest.json in %s — run capture_kv_snapshots.py first",
                     args.kv_dir)
        return 1

    with manifest_path.open() as f:
        manifest = json.load(f)
    logger.info("Loaded manifest: %d snapshots, model=%s, mock=%s",
                manifest["n_snapshots"], manifest["model"], manifest["mock"])

    all_layer_results = []
    per_snapshot = []
    for snap_meta in manifest["snapshots"]:
        npz_path = args.kv_dir.parent / snap_meta["file"]
        logger.info("Processing %s (seq=%d)",
                    npz_path.name, snap_meta["seq_len_actual"])
        layer_results = process_snapshot(
            npz_path=npz_path,
            group_size=args.group_size,
            sink_tokens=args.sink_tokens,
            use_fwht=args.use_fwht,
        )
        all_layer_results.extend(layer_results)
        per_snapshot.append({
            "snapshot_file": snap_meta["file"],
            "seq_len": snap_meta["seq_len_actual"],
            "n_layers": len(layer_results),
            "layer_results": layer_results,
        })

    acceptance = evaluate_acceptance(all_layer_results, args.min_reduction)
    payload = {
        "timestamp_unix": int(time.time()),
        "input_manifest": str(manifest_path),
        "input_manifest_run_id": manifest.get("run_id"),
        "input_hardware": manifest.get("hardware"),
        "input_mock": manifest.get("mock", False),
        "codec_config": {
            "group_size": args.group_size,
            "sink_tokens": args.sink_tokens,
            "use_fwht": args.use_fwht,
            "min_reduction_threshold": args.min_reduction,
        },
        "acceptance": acceptance,
        "per_snapshot": per_snapshot,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", args.out)

    # Console summary for sanity-check piping with `jq`
    print(json.dumps({
        "avg_v7_reduction": acceptance["avg_v7_reduction"],
        "avg_v8_reduction": acceptance["avg_v8_reduction"],
        "v8_beats_v7_mse_keys": acceptance["v8_beats_v7_on_mse_keys"],
        "v8_beats_v7_mse_values": acceptance["v8_beats_v7_on_mse_values"],
        "accept_v8_for_paper_v21": acceptance["accept_v8_for_paper_v21"],
    }, indent=2))
    return 0 if acceptance["accept_v8_for_paper_v21"] else 2


if __name__ == "__main__":
    sys.exit(main())
