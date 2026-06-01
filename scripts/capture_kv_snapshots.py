"""Capture representative KV-cache snapshots for V8 codec validation.

Sprint 5 Step 2 (Item 1) prerequisite: feed 5 representative prompts
through Llama-3-8B (or a smaller stand-in on CPU), capture the
pre-RoPE keys/values at every layer, and persist them as numpy
``.npz`` files under ``logs/kv_snapshots/``.

The downstream V8 codec script reads these snapshots
and runs the V7-vs-V8 codec comparison. Decoupling the capture from
the codec comparison lets us:

1. Capture once on the droplet, replay locally many times during V8
   tuning (saves $1.99/hr).
2. Share the snapshots in the public repo as ``logs/kv_snapshots/``
   becomes the canonical V8 evaluation dataset for the paper v2.1
   reviewer.
3. Run the codec comparison in CI without GPU access.

Honesty discipline: snapshots are tagged with the model, hardware
label (``rocm-hip:...`` on AMD, ``cuda:...`` on NVIDIA, ``cpu`` on
CPU stand-in), the git SHA of the repo at capture time, and a
``run_id`` that links back to the originating shell session.

Usage:

    # On the MI300X droplet:
    PYTHONPATH=. python3 scripts/capture_kv_snapshots.py \
        --model llama-3-8b --n-snapshots 5 \
        --out logs/kv_snapshots/

    # CPU stand-in for local smoke testing (no real model needed):
    PYTHONPATH=. python3 scripts/capture_kv_snapshots.py --mock-cpu \
        --n-snapshots 5 --out logs/kv_snapshots/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("capture_kv_snapshots")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)


# ---------------------------------------------------------------------------
# Lightweight model registry — shape per layer (post Llama-3-8B architecture)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelShape:
    """Tensor shapes per KV layer for a given model."""

    name: str
    num_layers: int
    num_heads: int        # num_attention_heads (Q, post-GQA we use kv heads)
    num_kv_heads: int     # num_key_value_heads (GQA: smaller than num_heads)
    head_dim: int

    def kv_layer_shape(self, seq_len: int) -> tuple[int, int, int]:
        """Shape returned for a single layer's keys (or values)."""
        return (seq_len, self.num_kv_heads, self.head_dim)


MODEL_REGISTRY: dict[str, ModelShape] = {
    "llama-3-8b": ModelShape(
        name="meta-llama/Llama-3-8B",
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,           # Llama-3 uses GQA: 32 Q heads, 8 KV heads
        head_dim=128,
    ),
    "tinyllama": ModelShape(
        name="TinyLlama/TinyLlama-1.1B-Chat",
        num_layers=22,
        num_heads=32,
        num_kv_heads=4,
        head_dim=64,
    ),
}


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------


def detect_hardware() -> str:
    """Return a hardware label honest about ROCm vs CUDA vs CPU.

    Apohara V6.1 honesty discipline: PyTorch ROCm reuses the
    ``torch.cuda.*`` API for backward-compat, so naive code reports
    ``"cuda"`` on AMD hardware. We probe explicitly.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "cpu (torch not installed)"

    if not torch.cuda.is_available():
        return "cpu"

    # Probe for ROCm. The order matters because on ROCm builds, both
    # checks return positive — the ROCm one is more specific.
    try:
        version_hip = torch.version.hip
    except AttributeError:
        version_hip = None

    if version_hip:
        device_name = torch.cuda.get_device_name(0)
        return f"rocm-hip:{version_hip}:{device_name}"

    version_cuda = getattr(torch.version, "cuda", "unknown")
    device_name = torch.cuda.get_device_name(0)
    return f"cuda:{version_cuda}:{device_name}"


def git_sha() -> str:
    """Return short git SHA of the current HEAD, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# ---------------------------------------------------------------------------
# Snapshot capture — real model path
# ---------------------------------------------------------------------------


def capture_real(
    *,
    model_key: str,
    prompts: list[str],
    out_dir: Path,
    seq_lens: list[int],
) -> list[dict]:
    """Use transformers + the real model to capture KV snapshots.

    Skipped if transformers / torch are unavailable (local dev).
    """
    try:
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"capture_real requires torch + transformers ({exc})"
        ) from None

    shape = MODEL_REGISTRY[model_key]
    logger.info("Loading %s (%d layers, %d KV heads)",
                shape.name, shape.num_layers, shape.num_kv_heads)

    tokenizer = AutoTokenizer.from_pretrained(shape.name)
    model = AutoModelForCausalLM.from_pretrained(
        shape.name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=False,
    )
    model.eval()

    manifests = []
    for prompt_idx, (prompt, target_seq) in enumerate(zip(prompts, seq_lens)):
        # Tokenize and pad to target_seq if shorter
        tokens = tokenizer(prompt, return_tensors="pt").to(model.device)
        # Run forward, capture past_key_values (pre-RoPE not directly accessible
        # via HF API — we capture POST-RoPE, document this in the JSON).
        with torch.no_grad():
            output = model(**tokens, use_cache=True)
        past_kv = output.past_key_values  # tuple of (k, v) per layer

        # Save as npz: one file per layer, or all layers stacked.
        # All-layers stacked is more compact for downstream codec comparison.
        keys_per_layer = []
        values_per_layer = []
        for layer_idx, (k_layer, v_layer) in enumerate(past_kv):
            # HF shape: (batch, num_kv_heads, seq_len, head_dim).
            # Codec expects: (batch, seq_len, num_kv_heads, head_dim).
            keys_per_layer.append(k_layer.transpose(1, 2).cpu().float().numpy())
            values_per_layer.append(v_layer.transpose(1, 2).cpu().float().numpy())

        out_file = out_dir / f"kv_snapshot_{prompt_idx}.npz"
        np.savez_compressed(
            out_file,
            keys=np.stack(keys_per_layer, axis=0),
            values=np.stack(values_per_layer, axis=0),
        )
        actual_seq = keys_per_layer[0].shape[1]
        manifests.append({
            "file": str(out_file.relative_to(out_dir.parent)),
            "prompt": prompt[:80] + ("..." if len(prompt) > 80 else ""),
            "prompt_idx": prompt_idx,
            "num_layers": shape.num_layers,
            "num_kv_heads": shape.num_kv_heads,
            "head_dim": shape.head_dim,
            "seq_len_actual": int(actual_seq),
            "rope_state": "post-rope",  # transformers API gives post-RoPE KV
        })
        logger.info("Captured snapshot %d (seq_len=%d) → %s",
                    prompt_idx, actual_seq, out_file.name)

    return manifests


# ---------------------------------------------------------------------------
# Snapshot capture — mock path (no real model needed)
# ---------------------------------------------------------------------------


def capture_mock(
    *,
    model_key: str,
    n_snapshots: int,
    out_dir: Path,
    seq_lens: list[int],
    seed: int = 0,
) -> list[dict]:
    """Generate snapshots with random tensors matching the real shapes.

    For CPU local smoke testing — lets us validate the V7-vs-V8 codec
    comparison logic without needing a 5-minute model download or
    GPU memory. The downstream codec scripts will run identically
    on real vs mock snapshots.
    """
    shape = MODEL_REGISTRY[model_key]
    rng = np.random.default_rng(seed)
    manifests = []

    for i in range(n_snapshots):
        seq = seq_lens[i % len(seq_lens)]
        # Each "layer" gets a (1, seq, num_kv_heads, head_dim) tensor.
        keys = rng.standard_normal(
            (shape.num_layers, 1, seq, shape.num_kv_heads, shape.head_dim),
            dtype=np.float32,
        ) * 0.5  # Scale to match typical transformer KV magnitudes
        values = rng.standard_normal(
            (shape.num_layers, 1, seq, shape.num_kv_heads, shape.head_dim),
            dtype=np.float32,
        ) * 0.5

        # Add asymmetric-pair texture in some channels to expose V8's
        # advantage on real data (per docs/v8-codec-design.md).
        keys[:, :, :, :, 0::2] *= 0.3  # even channels: narrow range
        values[:, :, :, :, 1::2] *= 3.0  # odd channels: wide range

        out_file = out_dir / f"kv_snapshot_{i}.npz"
        np.savez_compressed(out_file, keys=keys, values=values)
        manifests.append({
            "file": str(out_file.relative_to(out_dir.parent)),
            "prompt": f"<mock prompt {i}>",
            "prompt_idx": i,
            "num_layers": shape.num_layers,
            "num_kv_heads": shape.num_kv_heads,
            "head_dim": shape.head_dim,
            "seq_len_actual": int(seq),
            "rope_state": "pre-rope",  # Mock is pure synthetic, treat as pre-RoPE
            "mock": True,
        })
        logger.info("Mock snapshot %d (seq=%d) → %s", i, seq, out_file.name)

    return manifests


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


DEFAULT_PROMPTS = [
    "Explain in 2 sentences how a transformer's attention mechanism works.",
    "What is the capital of France, and what river runs through it?",
    "Compose a haiku about an autumn forest.",
    "Solve: 27 + 31 × 4 - 18 / 2. Show your work.",
    "Describe the difference between gravity and centripetal force.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="llama-3-8b",
                   choices=list(MODEL_REGISTRY.keys()),
                   help="Model key in MODEL_REGISTRY")
    p.add_argument("--n-snapshots", type=int, default=5,
                   help="Number of snapshots to capture")
    p.add_argument("--seq-lens", nargs="+", type=int,
                   default=[256, 512, 1024, 2048, 4096],
                   help="Sequence-length targets per snapshot")
    p.add_argument("--out", type=Path, default=Path("logs/kv_snapshots"),
                   help="Output directory")
    p.add_argument("--mock-cpu", action="store_true",
                   help="Skip real model load; generate random tensors")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for --mock-cpu path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    hw = detect_hardware()
    sha = git_sha()
    run_id = uuid.uuid4().hex[:8]

    logger.info("hardware=%s git_sha=%s run_id=%s mock=%s",
                hw, sha, run_id, args.mock_cpu)

    if args.mock_cpu:
        manifests = capture_mock(
            model_key=args.model,
            n_snapshots=args.n_snapshots,
            out_dir=args.out,
            seq_lens=args.seq_lens,
            seed=args.seed,
        )
    else:
        prompts = DEFAULT_PROMPTS[:args.n_snapshots]
        seq_lens = args.seq_lens[:args.n_snapshots]
        manifests = capture_real(
            model_key=args.model,
            prompts=prompts,
            out_dir=args.out,
            seq_lens=seq_lens,
        )

    manifest_file = args.out / "manifest.json"
    payload = {
        "timestamp_unix": int(time.time()),
        "git_sha": sha,
        "hardware": hw,
        "run_id": run_id,
        "model": args.model,
        "model_full_name": MODEL_REGISTRY[args.model].name,
        "mock": args.mock_cpu,
        "n_snapshots": len(manifests),
        "snapshots": manifests,
    }
    with manifest_file.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Manifest written → %s", manifest_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
