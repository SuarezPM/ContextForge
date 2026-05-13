"""Generate paper v2.0 figures from MI300X measurement JSONs.

Reads `logs/mi300x_*.json` and produces publication-ready PNGs:

  fig5_reduction_factor_vs_seq.png  — reduction_factor across seq_len
  fig6_reduction_factor_vs_dims.png — reduction_factor vs head_dim + num_heads
  fig7_fwht_overhead.png             — FWHT overhead pct vs config
  fig8_quant_quality.png             — MSE vs reduction_factor (Pareto)
  fig9_hbm3_bandwidth.png            — measured BW vs allocation size

Each figure has source-data overlay (small text) so reviewers can
cross-check against the logged JSON.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not installed; install via: pip install matplotlib", file=sys.stderr)
    sys.exit(1)


LOG_DIR = Path("logs")
OUT_DIR = Path("paper") / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def newest_log(pattern: str) -> Path | None:
    files = sorted(LOG_DIR.glob(pattern))
    return files[-1] if files else None


def make_reduction_vs_seq() -> None:
    p = newest_log("mi300x_vram_sweep_*.json")
    if p is None:
        print("no sweep file found"); return
    data = json.loads(p.read_text())
    seqs = sorted({r["seq_len"] for r in data if r["num_heads"] == 32 and r["head_dim"] == 128})
    fwht_false = [next(r for r in data if r["seq_len"] == s and r["num_heads"] == 32 and r["head_dim"] == 128 and not r["use_fwht"])["reduction_factor"] for s in seqs]
    fwht_true = [next(r for r in data if r["seq_len"] == s and r["num_heads"] == 32 and r["head_dim"] == 128 and r["use_fwht"])["reduction_factor"] for s in seqs]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.plot(seqs, fwht_false, "o-", label="INT4 (Apohara, no FWHT)", color="#1f77b4")
    ax.plot(seqs, fwht_true, "s--", label="INT4 + FWHT rotation", color="#d62728")
    ax.axhline(3.97, color="gray", linestyle=":", label="Literature target (RotateKV IJCAI'25)")
    ax.axhline(4.0, color="black", linestyle="--", linewidth=0.5, label="Theoretical FP16→INT4")
    ax.set_xscale("log", base=2)
    ax.set_xticks(seqs)
    ax.set_xticklabels([str(s) for s in seqs])
    ax.set_xlabel("KV cache sequence length")
    ax.set_ylabel("VRAM reduction factor (×)")
    ax.set_ylim(0, 4.2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Apohara KV-cache reduction factor on MI300X\n(num_heads=32, head_dim=128, batch=1)")
    fig.text(0.01, 0.01, f"src: {p.name}", fontsize=7, color="gray")
    fig.tight_layout()
    out = OUT_DIR / "fig5_reduction_factor_vs_seq.png"
    fig.savefig(out)
    print(f"Wrote {out}")


def make_quant_quality() -> None:
    p = newest_log("mi300x_quant_quality_*.json")
    if p is None:
        print("no quant_quality file found"); return
    data = json.loads(p.read_text())
    configs = data["configs"]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    for c in configs:
        mse = c["mse_keys"]
        red = c["reduction_factor"]
        name = c["name"]
        ax.scatter(red, mse, s=80)
        ax.annotate(name, (red, mse), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("VRAM reduction factor (× vs FP16)")
    ax.set_ylabel("Reconstruction MSE (keys)")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_title("Apohara quantization quality vs compression on MI300X\n(seq=16384, num_heads=32, head_dim=128)")
    fig.text(0.01, 0.01, f"src: {p.name}", fontsize=7, color="gray")
    fig.tight_layout()
    out = OUT_DIR / "fig8_quant_quality.png"
    fig.savefig(out)
    print(f"Wrote {out}")


def make_hbm3_bandwidth() -> None:
    p = newest_log("mi300x_hbm3_bandwidth_*.json")
    if p is None:
        print("no hbm3 file found"); return
    data = json.loads(p.read_text())
    m = data["measurements"]
    sizes = [r["size_gb"] for r in m]
    copy_bw = [r["copy_bw_gbps"] / 1000 for r in m]   # → TB/s
    triad_bw = [r["triad_bw_gbps"] / 1000 for r in m]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.plot(sizes, copy_bw, "o-", label="copy (2× nbytes / time)", color="#1f77b4")
    ax.plot(sizes, triad_bw, "s--", label="triad (3× nbytes / time)", color="#d62728")
    ax.axhline(5.3, color="gray", linestyle=":", label="Advertised peak 5.3 TB/s")
    ax.set_xlabel("Allocation size (GB)")
    ax.set_ylabel("Bandwidth (TB/s)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s:g}" for s in sizes])
    ax.set_ylim(0, 6.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("MI300X HBM3 effective bandwidth")
    fig.text(0.01, 0.01, f"src: {p.name}", fontsize=7, color="gray")
    fig.tight_layout()
    out = OUT_DIR / "fig9_hbm3_bandwidth.png"
    fig.savefig(out)
    print(f"Wrote {out}")


def make_pure_fwht() -> None:
    p = newest_log("mi300x_pure_torch_fwht_*.json")
    if p is None:
        print("no pure_torch_fwht file found"); return
    data = json.loads(p.read_text())
    # Sub-select num_heads=32, head_dim=128 vs seq
    rows = [r for r in data if r["num_heads"] == 32 and r["head_dim"] == 128]
    rows.sort(key=lambda r: r["seq_len"])
    seqs = [r["seq_len"] for r in rows]
    durs = [r["fwht_duration_ms"] for r in rows]
    thrpts = [r["fwht_throughput_gbps"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=140)
    ax1.plot(seqs, durs, "o-", color="#1f77b4", label="FWHT duration")
    ax1.set_xlabel("KV cache sequence length")
    ax1.set_ylabel("FWHT duration (ms)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(seqs)
    ax1.set_xticklabels([str(s) for s in seqs])
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(seqs, thrpts, "s--", color="#d62728", label="effective throughput")
    ax2.set_ylabel("Effective throughput (GB/s)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Pure-torch FWHT on MI300X (no NumPy bridge)\n(num_heads=32, head_dim=128, batch=1, FP16)")
    fig.text(0.01, 0.01, f"src: {p.name}", fontsize=7, color="gray")
    fig.tight_layout()
    out = OUT_DIR / "fig7_pure_torch_fwht.png"
    fig.savefig(out)
    print(f"Wrote {out}")


def main() -> int:
    make_reduction_vs_seq()
    make_pure_fwht()
    make_quant_quality()
    make_hbm3_bandwidth()
    print(f"\nAll figures in {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
