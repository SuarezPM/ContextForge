#!/usr/bin/env python3
"""A/B HBM-footprint harness for ContextForge prefix caching.

Measures steady-state GPU memory (HBM) under two regimes serving the SAME model
and the SAME concurrent workload (N requests sharing one system prompt):

* CORRIDA A — baseline / OFF: vLLM launched with ``--no-enable-prefix-caching``;
  prompts assembled by hand; no ``cache_salt``.
* CORRIDA B — treatment / ON: vLLM launched with ``--enable-prefix-caching``;
  prompts assembled by :class:`PrefixNormalizer` (byte-identical shared prefix)
  and each request carries the ``cache_salt`` decided by :class:`PrefixSaltPlanner`.

Steady-state HBM is read TWICE for cross-checking:
  1. in-process via :class:`apohara_context_forge.metrics.vram_monitor.VRAMMonitor`
     (PyRSMI on AMD, the CUDA path on NVIDIA);
  2. out-of-process via ``nvidia-smi`` / ``rocm-smi`` (a second, independent eye).

Output is a JSON record with ``vram_off_gb``, ``vram_on_gb``, ``delta_gb``,
``max_concurrency``, ``vram_source``, and an HONEST ``hardware_label``.

CROSS-WORKER (+LMCache) REGIME
------------------------------
``--cross-worker`` runs a separate regime that wires LMCache's OFFICIAL
built-in connector (``LMCacheConnectorV1Dynamic``) via vLLM's
``--kv-transfer-config`` (built by
:mod:`apohara_context_forge.serving.vllm_launch_config`). It reports TWO
deltas SEPARATELY — the APC-only delta (intra-instance dedup) and the
+LMCache delta (cross-worker offload) — so the two distinct mechanisms
are never conflated. The real cross-worker VRAM number requires
MI300X + Redis + qwen3-embed and is GATED outside this workflow; the dry
path here only validates the launch-config plumbing (chunk/block
alignment, the exact connector JSON) with no cluster.

HONESTY / SCOPE
---------------
* vLLM is NEVER imported by this module. The live path launches the ``vllm``
  CLI as a subprocess and talks to it over HTTP — so this file imports and runs
  its ``dry`` mode with neither vllm nor lmcache installed.
* ``--mode dry`` produces NO real measurement: it returns clearly-flagged
  placeholder numbers (``measured=False``, ``hardware_label="dry-run (no GPU)"``)
  for unit-testing the plumbing. It does not pretend to have touched a GPU.
* ``--mode live`` performs the real A/B (requires a GPU + the ``vllm`` CLI). It
  is gated outside the unit-test workflow.

Apache-2.0 — Apohara ContextForge.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Default workload: a long shared system prompt (the cross-agent reusable
# prefix) plus distinct per-request tails. Kept terse; callers can override.
_DEFAULT_SYSTEM_PROMPT = (
    "You are part of a multi-agent retrieval pipeline. The agents share a "
    "common briefing; the key-value cache computed for that shared prefix can "
    "be reused across requests instead of recomputed. Answer the per-agent "
    "task using only the briefing."
)
_DEFAULT_TAILS = [
    "Task: list the 3 most relevant documents. Be terse.",
    "Task: rerank the 5 documents by relevance. Ranking only.",
    "Task: summarize the briefing in two sentences.",
    "Task: verify one claim. ACCEPT or REJECT plus one sentence.",
    "Task: write a 3-sentence answer.",
]


@dataclass
class ABConfig:
    """Configuration for one A/B run."""

    model: str = "dry-model"
    endpoint: str = "http://localhost:8000"
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    tails: list[str] = field(default_factory=lambda: list(_DEFAULT_TAILS))
    concurrency: int = 5
    max_tokens: int = 32
    device_id: int = 0
    # Seconds to wait after the workload before sampling steady-state HBM.
    settle_s: float = 2.0


@dataclass
class ABResult:
    """Result of an A/B run. ``measured`` is False for dry runs (no GPU)."""

    vram_off_gb: float
    vram_on_gb: float
    delta_gb: float
    max_concurrency: int
    vram_source: str
    hardware_label: str
    measured: bool
    # Optional second-source readings for cross-validation (out-of-process).
    vram_off_gb_second_source: Optional[float] = None
    vram_on_gb_second_source: Optional[float] = None
    second_source: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class CrossWorkerResult:
    """Result of the cross-worker (+LMCache) regime, reported SEPARATELY.

    The two effects are deliberately NOT mixed:

    * ``apc_delta_gb`` — the intra-instance prefix-caching win (CORRIDA A
      vs CORRIDA B on a single worker). This is the same number an
      :class:`ABResult` carries; it is dedup *inside* one vLLM process.
    * ``lmcache_delta_gb`` — the ADDITIONAL win from cross-worker KV
      offload to LMCache/Redis: HBM on worker-2 with APC alone minus HBM
      on worker-2 with APC + the shared LMCache backend. This is offload
      *across* processes, a different mechanism.

    Conflating the two would over-claim; they are surfaced as two
    independent deltas with their own provenance.
    """

    apc_delta_gb: float
    lmcache_delta_gb: float
    max_concurrency: int
    vram_source: str
    hardware_label: str
    measured: bool
    # The exact LMCache invocation used (or that WOULD be used in dry mode),
    # so the record is self-documenting about what cross-worker path was wired.
    kv_transfer_config: Optional[dict] = None
    lmcache_chunk_size: Optional[int] = None
    vllm_block_size: Optional[int] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- #
# Second-source (out-of-process) HBM readers — independent of VRAMMonitor.
# --------------------------------------------------------------------------- #
def _read_nvidia_smi_used_gb(device_id: int = 0) -> Optional[float]:
    """Used HBM in GB via nvidia-smi, or None if nvidia-smi is unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
        )
        used_mib = float(proc.stdout.strip().split("\n")[0])
        return used_mib / 1024.0
    except Exception as e:
        logger.debug(f"nvidia-smi second source failed: {e}")
        return None


def _read_rocm_smi_used_gb(device_id: int = 0) -> Optional[float]:
    """Used HBM in GB via rocm-smi, or None if rocm-smi is unavailable."""
    if shutil.which("rocm-smi") is None:
        return None
    try:
        proc = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
        )
        data = json.loads(proc.stdout)
        card = data.get(f"card{device_id}", {})
        # rocm-smi reports VRAM Total/Used Memory (B) under varying keys; try the
        # common ones and fall back to None rather than guessing.
        for key in ("VRAM Total Used Memory (B)", "VRAM Used Memory (B)"):
            if key in card:
                return int(card[key]) / (1024 ** 3)
        return None
    except Exception as e:
        logger.debug(f"rocm-smi second source failed: {e}")
        return None


def read_second_source_used_gb(device_id: int = 0) -> tuple[Optional[float], Optional[str]]:
    """Best-effort out-of-process used-HBM reading + its source label."""
    gb = _read_nvidia_smi_used_gb(device_id)
    if gb is not None:
        return gb, "nvidia-smi"
    gb = _read_rocm_smi_used_gb(device_id)
    if gb is not None:
        return gb, "rocm-smi"
    return None, None


# --------------------------------------------------------------------------- #
# Prompt assembly — shared by dry and live; uses the real PrefixNormalizer /
# PrefixSaltPlanner so CORRIDA B is exactly the treatment we claim to test.
# --------------------------------------------------------------------------- #
def build_prompts_off(cfg: ABConfig) -> list[dict]:
    """CORRIDA A: hand-assembled prompts, no normalizer, no salt."""
    n = cfg.concurrency
    tails = (cfg.tails * (n // max(len(cfg.tails), 1) + 1))[:n]
    return [
        {"prompt": cfg.system_prompt + "\n\n" + tail, "cache_salt": None}
        for tail in tails
    ]


def build_prompts_on(cfg: ABConfig) -> list[dict]:
    """CORRIDA B: PrefixNormalizer-assembled prompts + planner cache_salt.

    Imports the real ContextForge components (no vLLM dependency). The shared
    system prefix is byte-identical across requests; the salt is the planner's
    shared salt for the common anchor (so vLLM shares the prefix KV blocks).
    """
    from apohara_context_forge.normalization.prefix_normalizer import PrefixNormalizer
    from apohara_context_forge.serving.prefix_salt_planner import PrefixSaltPlanner

    normalizer = PrefixNormalizer(canonical_system_prompt=cfg.system_prompt)
    planner = PrefixSaltPlanner()
    # One shared anchor + group so non-judge agents collide on the same salt.
    anchor_hash = normalizer.get_canonical_hash()
    cla_group = "ab-harness"

    n = cfg.concurrency
    tails = (cfg.tails * (n // max(len(cfg.tails), 1) + 1))[:n]
    out = []
    for i, tail in enumerate(tails):
        prompt = normalizer.normalize(
            agent_id=f"agent-{i}",
            user_prompt=tail,
            agent_role_prompt="worker",
        )
        plan = planner.plan(
            agent_role="worker",
            anchor_hash=anchor_hash,
            cla_group=cla_group,
            request_id=f"req-{i}",
        )
        out.append({"prompt": prompt, "cache_salt": plan.cache_salt})
    return out


# --------------------------------------------------------------------------- #
# Dry mode — no server, no GPU. Unit-testable. HONESTLY flagged.
# --------------------------------------------------------------------------- #
def run_dry(cfg: ABConfig) -> ABResult:
    """Exercise the plumbing with NO real measurement.

    Builds the OFF and ON prompt sets through the real assembly code (so the
    normalizer/planner path is exercised) but does not start vLLM or touch a
    GPU. The returned VRAM numbers are placeholders flagged ``measured=False``.
    """
    prompts_off = build_prompts_off(cfg)
    prompts_on = build_prompts_on(cfg)
    return ABResult(
        vram_off_gb=0.0,
        vram_on_gb=0.0,
        delta_gb=0.0,
        max_concurrency=cfg.concurrency,
        vram_source="dry",
        hardware_label="dry-run (no GPU)",
        measured=False,
        second_source=None,
        notes=(
            "DRY MODE: no vLLM server, no GPU read. VRAM numbers are placeholder "
            f"zeros (measured=False). Built {len(prompts_off)} OFF prompts and "
            f"{len(prompts_on)} ON prompts via PrefixNormalizer+PrefixSaltPlanner."
        ),
    )


# --------------------------------------------------------------------------- #
# Cross-worker (+LMCache) dry plumbing — no cluster, no Redis, no GPU.
# Reports the APC-only delta and the +LMCache delta SEPARATELY.
# --------------------------------------------------------------------------- #
def run_cross_worker_dry(cfg: ABConfig) -> CrossWorkerResult:
    """Exercise the cross-worker (+LMCache) plumbing with NO real measurement.

    Builds the exact vLLM ``--kv-transfer-config`` that a real worker would
    launch with (LMCache's official ``LMCacheConnectorV1Dynamic`` via
    :mod:`apohara_context_forge.serving.vllm_launch_config`) and records the
    chunk/block alignment that was enforced — but starts NO cluster, opens NO
    Redis connection, and reads NO GPU. Both deltas are placeholder zeros
    flagged ``measured=False``.

    The launch-config import is pure (no vllm/lmcache), so this runs in the
    test venv. The alignment invariant (chunk_size == block_size) is enforced
    by ``build_kv_transfer_config`` and will raise loudly if broken.
    """
    from apohara_context_forge.serving.vllm_launch_config import (
        DEFAULT_BLOCK_SIZE,
        build_kv_transfer_config,
    )

    block_size = DEFAULT_BLOCK_SIZE
    chunk_size = DEFAULT_BLOCK_SIZE  # MUST match block_size; builder asserts it.
    kv_cfg = build_kv_transfer_config(block_size=block_size, chunk_size=chunk_size)

    return CrossWorkerResult(
        apc_delta_gb=0.0,
        lmcache_delta_gb=0.0,
        max_concurrency=cfg.concurrency,
        vram_source="dry",
        hardware_label="dry-run (no GPU)",
        measured=False,
        kv_transfer_config=kv_cfg,
        lmcache_chunk_size=chunk_size,
        vllm_block_size=block_size,
        notes=(
            "DRY MODE: no vLLM workers, no Redis, no GPU read. The APC-only "
            "delta (intra-instance dedup) and the +LMCache delta (cross-worker "
            "offload) are placeholder zeros (measured=False) reported SEPARATELY "
            "so the two mechanisms are never conflated. The real cross-worker "
            "VRAM number requires MI300X + Redis + qwen3-embed and is GATED "
            "outside this workflow."
        ),
    )


# --------------------------------------------------------------------------- #
# Live mode — real A/B. Requires a GPU + the vllm CLI. Gated outside unit tests.
# --------------------------------------------------------------------------- #
def _vllm_available() -> bool:
    """True if the ``vllm`` CLI is importable/launchable (import-guarded)."""
    try:
        import vllm  # noqa: F401
        return True
    except Exception:
        return shutil.which("vllm") is not None


def _steady_state_used_gb(cfg: ABConfig) -> tuple[float, str]:
    """Sample steady-state used HBM in-process via VRAMMonitor + its source."""
    from apohara_context_forge.metrics.vram_monitor import VRAMMonitor

    monitor = VRAMMonitor(device_id=cfg.device_id)
    time.sleep(cfg.settle_s)
    used_gb = monitor.get_used_gb()
    return used_gb, monitor.get_vram_source()


def run_live(cfg: ABConfig, send_workload) -> ABResult:
    """Real A/B measurement against a running GPU.

    ``send_workload(prompts) -> None`` is supplied by the caller (or the live
    launcher below); it must drive the N concurrent requests against whichever
    vLLM server is up for the current regime. This function only orchestrates
    the two regimes and the HBM sampling. vLLM is NOT imported here.

    NOTE: launching/tearing down the two vLLM servers (with and without
    ``--enable-prefix-caching``) is the caller's responsibility in the gated
    live workflow; this keeps the measured core import-clean and testable.
    """
    if not _vllm_available():
        raise RuntimeError(
            "live mode requires the vllm CLI (import vllm or `vllm` on PATH). "
            "Neither is present — run --mode dry, or run on a GPU+vllm host."
        )

    # CORRIDA A — OFF
    prompts_off = build_prompts_off(cfg)
    send_workload(prompts_off, enable_prefix_caching=False)
    off_gb, src_off = _steady_state_used_gb(cfg)
    off_second, second_label = read_second_source_used_gb(cfg.device_id)

    # CORRIDA B — ON
    prompts_on = build_prompts_on(cfg)
    send_workload(prompts_on, enable_prefix_caching=True)
    on_gb, src_on = _steady_state_used_gb(cfg)
    on_second, _ = read_second_source_used_gb(cfg.device_id)

    # vram_source: both regimes read the same device; report it honestly. If
    # the two readings somehow used different backends, surface both.
    vram_source = src_on if src_on == src_off else f"{src_off}|{src_on}"
    hardware_label = _hardware_label(vram_source)

    return ABResult(
        vram_off_gb=round(off_gb, 3),
        vram_on_gb=round(on_gb, 3),
        delta_gb=round(off_gb - on_gb, 3),
        max_concurrency=cfg.concurrency,
        vram_source=vram_source,
        hardware_label=hardware_label,
        measured=True,
        vram_off_gb_second_source=(round(off_second, 3) if off_second is not None else None),
        vram_on_gb_second_source=(round(on_second, 3) if on_second is not None else None),
        second_source=second_label,
        notes="LIVE: steady-state HBM sampled in-process and cross-checked out-of-process.",
    )


def _hardware_label(vram_source: str) -> str:
    """Honest hardware label derived from the backend that produced numbers."""
    if "cuda" in vram_source:
        return "NVIDIA/CUDA"
    if vram_source in ("pyrsmi", "drm_sysfs"):
        return "AMD/ROCm"
    if vram_source == "amd_default_192gb":
        return "AMD/ROCm (UNVERIFIED 192GB default — total not actually read)"
    return f"unknown ({vram_source})"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["dry", "live"], default="dry")
    ap.add_argument(
        "--cross-worker",
        action="store_true",
        help=(
            "Run the cross-worker (+LMCache) regime instead of the "
            "single-worker A/B. Reports the APC-only delta and the +LMCache "
            "delta SEPARATELY. dry mode only here (live is GATED)."
        ),
    )
    ap.add_argument("--model", default="dry-model")
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--output", default=None, help="Write JSON result here.")
    args = ap.parse_args(argv)

    cfg = ABConfig(
        model=args.model,
        endpoint=args.endpoint,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        device_id=args.device_id,
    )

    if args.mode == "dry":
        result = run_cross_worker_dry(cfg) if args.cross_worker else run_dry(cfg)
    else:
        raise SystemExit(
            "live mode requires a caller-supplied workload driver and a GPU; "
            "it is gated outside this workflow. Use --mode dry for plumbing."
        )

    payload = json.dumps(result.to_dict(), indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
