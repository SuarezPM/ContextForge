"""Generate the TechEx 2026 submission cover image.

1280×640 PNG, professional minimalist look with headline metrics.
Reproducible: re-run any time to regenerate `assets/techex-cover.png`.

Usage:
    PYTHONPATH=. python3 scripts/generate_techex_cover.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Layout constants (1280×640)
# ---------------------------------------------------------------------------
WIDTH = 1280
HEIGHT = 640

# Brand palette (matches paper figures + pitch deck)
COLOR_BG = (15, 23, 42)            # slate-900
COLOR_TEXT = (248, 250, 252)       # slate-50
COLOR_ACCENT = (239, 68, 68)       # red-500 (highlight numbers)
COLOR_MUTED = (148, 163, 184)      # slate-400 (caption text)
COLOR_GREEN = (34, 197, 94)        # green-500 (positive indicator)

# Font discovery (DejaVu is guaranteed on Ubuntu 26.04)
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONT_BOLD = f"{FONT_DIR}/DejaVuSans-Bold.ttf"
FONT_REGULAR = f"{FONT_DIR}/DejaVuSans.ttf"
FONT_MONO = f"{FONT_DIR}/DejaVuSansMono.ttf"


def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError as exc:
        print(f"WARN: font {path} not found ({exc}); falling back to default")
        return ImageFont.load_default()


def render() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), COLOR_BG)
    draw = ImageDraw.Draw(img)

    # Top accent bar (subtle, matches paper figure border)
    draw.rectangle([(0, 0), (WIDTH, 6)], fill=COLOR_ACCENT)

    # Project name — top-left
    title_font = load_font(FONT_BOLD, 64)
    draw.text((48, 56), "Apohara ContextForge", font=title_font, fill=COLOR_TEXT)

    # Subtitle — below title
    subtitle_font = load_font(FONT_REGULAR, 28)
    draw.text(
        (50, 138),
        "Defense-in-Depth Trust Layer for Multi-Agent LLMs",
        font=subtitle_font,
        fill=COLOR_MUTED,
    )

    # Three-pillar callout — middle band
    band_y = 230
    band_h = 220
    pillar_font_label = load_font(FONT_REGULAR, 20)
    pillar_font_metric = load_font(FONT_BOLD, 52)
    pillar_font_caption = load_font(FONT_REGULAR, 16)

    pillars = [
        ("3.55×", "INT4 KV reduction", "constant 4K → 262K context\n(AMD MI300X measured)"),
        ("0 / 1,210", "INV-15 violations", "exhaustive sweep,\nformal invariant"),
        ("3.73 TB/s", "HBM3 bandwidth", "70.5% of advertised peak\n(SR-IOV slice)"),
    ]
    col_width = WIDTH // 3
    for i, (metric, label, caption) in enumerate(pillars):
        cx = col_width * i + col_width // 2

        # Label (top)
        bbox = draw.textbbox((0, 0), label, font=pillar_font_label)
        label_w = bbox[2] - bbox[0]
        draw.text(
            (cx - label_w // 2, band_y),
            label,
            font=pillar_font_label,
            fill=COLOR_MUTED,
        )

        # Metric (big, accent)
        bbox = draw.textbbox((0, 0), metric, font=pillar_font_metric)
        metric_w = bbox[2] - bbox[0]
        draw.text(
            (cx - metric_w // 2, band_y + 30),
            metric,
            font=pillar_font_metric,
            fill=COLOR_ACCENT,
        )

        # Caption (muted, 2 lines)
        for j, line in enumerate(caption.split("\n")):
            bbox = draw.textbbox((0, 0), line, font=pillar_font_caption)
            cap_w = bbox[2] - bbox[0]
            draw.text(
                (cx - cap_w // 2, band_y + 100 + j * 22),
                line,
                font=pillar_font_caption,
                fill=COLOR_MUTED,
            )

    # Bottom band: layer narrative + credentials
    bottom_y = HEIGHT - 130
    narrative_font = load_font(FONT_REGULAR, 22)
    narrative_font_bold = load_font(FONT_BOLD, 22)
    cred_font = load_font(FONT_MONO, 18)

    # "Layer 1 · Lobster Trap perimeter  +  Layer 2 · INV-15 behavioral"
    layer_text_left = "Layer 1 · "
    layer_emphasis_1 = "Lobster Trap"
    layer_text_middle = "  (perimeter)  +  Layer 2 · "
    layer_emphasis_2 = "INV-15"
    layer_text_right = "  (behavioral integrity)"
    line_y = bottom_y
    x = 50
    draw.text((x, line_y), layer_text_left, font=narrative_font, fill=COLOR_TEXT)
    x += draw.textlength(layer_text_left, font=narrative_font)
    draw.text((x, line_y), layer_emphasis_1, font=narrative_font_bold, fill=COLOR_GREEN)
    x += draw.textlength(layer_emphasis_1, font=narrative_font_bold)
    draw.text((x, line_y), layer_text_middle, font=narrative_font, fill=COLOR_TEXT)
    x += draw.textlength(layer_text_middle, font=narrative_font)
    draw.text((x, line_y), layer_emphasis_2, font=narrative_font_bold, fill=COLOR_GREEN)
    x += draw.textlength(layer_emphasis_2, font=narrative_font_bold)
    draw.text((x, line_y), layer_text_right, font=narrative_font, fill=COLOR_TEXT)

    # Credentials line (DOI + license + repo)
    draw.text(
        (50, bottom_y + 50),
        "Zenodo DOI 10.5281/zenodo.20114594  ·  Apache-2.0  ·  github.com/SuarezPM/Apohara_Context_Forge",
        font=cred_font,
        fill=COLOR_MUTED,
    )

    # Hackathon tag (small, top-right)
    tag_font = load_font(FONT_BOLD, 14)
    tag = "TechEx 2026 — Track 1 · Agent Security & AI Governance"
    bbox = draw.textbbox((0, 0), tag, font=tag_font)
    draw.text(
        (WIDTH - (bbox[2] - bbox[0]) - 48, 70),
        tag,
        font=tag_font,
        fill=COLOR_MUTED,
    )

    return img


def main() -> int:
    out_dir = Path("assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "techex-cover.png"
    img = render()
    img.save(out_path, optimize=True)
    print(f"✅ Cover written to {out_path} ({out_path.stat().st_size // 1024} KB, "
          f"{img.size[0]}×{img.size[1]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
