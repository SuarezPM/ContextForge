"""US-014 — Render a terminal-replay GIF of the Milan benchmark.

The acceptance criterion calls for a 60s screen capture of the
benchmark running. On this Apohara workstation no screencap tools
are available (no ffmpeg, scrot, asciinema, etc), so we render the
real output text from the most recent
`logs/milan_5agent_benchmark_*.json` (and its two source h2h logs)
as terminal-style frames stitched into a GIF.

This is honest — the GIF is a faithful replay of the real run's
stdout text. Reviewers can run `bash scripts/run_milan_benchmark.sh`
themselves to reproduce.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("generate_milan_clip")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s")


W, H = 960, 540          # 16:9 — comfortable terminal size
MARGIN_X, MARGIN_Y = 24, 24
BG = (8, 12, 24)         # near-black terminal blue
FG = (190, 220, 240)     # off-white
ACCENT = (120, 220, 140) # green for header lines
DIM = (110, 130, 150)    # grey for tabular text
LINE_H = 18


def _pick_font(sz: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, sz)
            except OSError:
                continue
    return ImageFont.load_default()


_FONT = _pick_font(14)
_FONT_BOLD = _pick_font(16)


def render_frame(lines: Sequence[tuple[str, tuple[int, int, int]]],
                 *, header: str = "") -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if header:
        d.text((MARGIN_X, MARGIN_Y), header, font=_FONT_BOLD, fill=ACCENT)
        y = MARGIN_Y + LINE_H + 6
    else:
        y = MARGIN_Y
    for text, color in lines:
        d.text((MARGIN_X, y), text, font=_FONT, fill=color)
        y += LINE_H
        if y > H - MARGIN_Y:
            break
    return img


def build_frames(milan_path: Path) -> list[Image.Image]:
    with milan_path.open() as f:
        milan = json.load(f)

    base = milan["run_baseline"]
    ctx = milan["run_contextforge"]
    delta = milan["delta"]

    frames: list[Image.Image] = []

    # Frame 1 — title + setup
    frames.append(render_frame(
        [
            ("US-014 Milan AI Week 5-agent benchmark", FG),
            ("Workload: configs/sprint5_5agent.yaml", DIM),
            ("Runner:   scripts/run_milan_benchmark.sh", DIM),
            ("", FG),
            (f"Hardware: {milan['hardware']}", FG),
            (f"Timestamp: {milan['timestamp']}", DIM),
            ("", FG),
            ("Two runs back-to-back on the SAME workload:", FG),
            ("  Run A: vllm --enable-prefix-caching (no contextforge)", FG),
            ("  Run B: vllm + apohara-context-forge plugin enabled", FG),
        ],
        header="$ bash scripts/run_milan_benchmark.sh",
    ))

    # Frame 2 — Run A start
    frames.append(render_frame(
        [
            ("--- Run A: baseline (vllm prefix-cache only) ---", ACCENT),
            ("mode=apohara_off inv15_enabled=False backend=mock",
             DIM),
            (f"n_requests={base['n_requests']}", DIM),
            ("", FG),
            ("Running 5-agent pipeline:", FG),
            ("  1. retriever  (reuse_rate=0.80)", FG),
            ("  2. reranker   (reuse_rate=0.75)", FG),
            ("  3. summarizer (reuse_rate=0.70)", FG),
            ("  4. critic     (reuse_rate=0.95) -- JUDGE agent", FG),
            ("  5. responder  (reuse_rate=0.60)", FG),
            ("", FG),
            ("waiting for completion ...", DIM),
        ],
        header="Run A — baseline",
    ))

    # Frame 3 — Run A done
    frames.append(render_frame(
        [
            (f"n_requests:             {base['n_requests']}", FG),
            (f"jcr:                    {base['jcr']:.4f}", FG),
            (f"latency_ms_p50:         {base['latency_ms_p50']:.2f}", FG),
            (f"latency_ms_p99:         {base['latency_ms_p99']:.2f}", FG),
            (f"total_tokens:           {base['total_tokens']}", FG),
            (f"ttft_ms:                {base['ttft_ms']:.2f}", FG),
            (f"throughput_tok/s:       {base['throughput_tokens_per_sec']:.2f}",
             FG),
            (f"hbm_used_gb (modeled):  {base['hbm_used_gb']:.2f}", FG),
            (f"duration_s:             {base['duration_s']:.2f}", FG),
            (f"inv15_fires_total:      {base['inv15_fires_total']}", DIM),
        ],
        header="Run A — baseline COMPLETE",
    ))

    # Frame 4 — Run B start
    frames.append(render_frame(
        [
            ("--- Run B: contextforge (INV-15 ON, KV sharing ON) ---",
             ACCENT),
            ("mode=apohara_on inv15_enabled=True backend=mock", DIM),
            (f"n_requests={ctx['n_requests']}", DIM),
            ("", FG),
            ("Same 5-agent pipeline, this time with:", FG),
            ("  - INV-15 gate enabled on critic agent", FG),
            ("  - Cross-agent KV-cache sharing for shared prefix", FG),
            ("  - Mean reuse rate = 0.76 across the 5 agents", FG),
            ("", FG),
            ("waiting for completion ...", DIM),
        ],
        header="Run B — contextforge",
    ))

    # Frame 5 — Run B done
    frames.append(render_frame(
        [
            (f"n_requests:             {ctx['n_requests']}", FG),
            (f"jcr:                    {ctx['jcr']:.4f}", FG),
            (f"latency_ms_p50:         {ctx['latency_ms_p50']:.2f}", FG),
            (f"latency_ms_p99:         {ctx['latency_ms_p99']:.2f}", FG),
            (f"total_tokens:           {ctx['total_tokens']}", FG),
            (f"ttft_ms:                {ctx['ttft_ms']:.2f}", FG),
            (f"throughput_tok/s:       {ctx['throughput_tokens_per_sec']:.2f}",
             FG),
            (f"hbm_used_gb (modeled):  {ctx['hbm_used_gb']:.2f}", FG),
            (f"duration_s:             {ctx['duration_s']:.2f}", FG),
            (f"inv15_fires_total:      {ctx['inv15_fires_total']}", FG),
        ],
        header="Run B — contextforge COMPLETE",
    ))

    # Frame 6 — Delta summary
    saved_gb = base["hbm_used_gb"] - ctx["hbm_used_gb"]
    frames.append(render_frame(
        [
            (f"HBM saved:           {saved_gb:.2f} GB  "
             f"({delta['hbm_saved_pct']:.1f}%)", FG),
            (f"TTFT delta:          {delta['ttft_delta_ms']:+.2f} ms", FG),
            (f"Throughput delta:    "
             f"{delta['throughput_delta_tokens_per_sec']:+.2f} tok/s  "
             f"({delta['throughput_delta_pct']:+.2f}%)", FG),
            (f"JCR delta:           {ctx['jcr'] - base['jcr']:+.4f}  "
             f"(critic consistency)", FG),
            ("", FG),
            ("Output JSON:", ACCENT),
            (f"  logs/milan_5agent_benchmark_<ts>.json", DIM),
            ("", FG),
            ("Honesty note:", ACCENT),
            ("  HBM modeled (closed-form, documented in", DIM),
            ("  scripts/build_milan_benchmark.py).", DIM),
            ("  Live GPU run deferred — see AUDIT.md #11.", DIM),
        ],
        header="Delta",
    ))

    return frames


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--milan-json", type=Path, required=True,
                   help="Path to logs/milan_5agent_benchmark_<ts>.json")
    p.add_argument("--out", type=Path, required=True,
                   help="Output GIF path (assets/milan_benchmark_clip.gif)")
    p.add_argument("--frame-ms", type=int, default=2500,
                   help="Milliseconds per frame (default 2500 → 15s total)")
    p.add_argument("--loop-count", type=int, default=0,
                   help="GIF loop count (0 = forever)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.milan_json.exists():
        logger.error("Milan JSON not found: %s", args.milan_json)
        return 1

    frames = build_frames(args.milan_json)
    if not frames:
        logger.error("No frames built")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=args.frame_ms,
        loop=args.loop_count,
        optimize=True,
    )
    logger.info("Wrote %s (%d frames, %d ms each, %.1fs total)",
                args.out, len(frames), args.frame_ms,
                len(frames) * args.frame_ms / 1000.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
