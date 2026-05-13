"""Sprint 5 Step 3: vLLM end-to-end demo with the 5-agent pipeline.

Hits a running vLLM server with the 5-agent workload described in
``configs/sprint5_5agent.yaml`` and records the trace. INV-15 fires
live for the critic agent (when ``--inv15`` is set).

Acceptance (paper v2.1 §6):

* Total tokens consumed reflects expected ~256 tokens × 5 agents ×
  N requests
* INV-15 fires > 0 for the critic agent on a workload where
  ``reuse_rate=0.85, candidate_count=5, layout_shuffled=false``
  (risk ≈ 0.575, just above τ=0.65 default — adjust via the YAML
  if your tuning differs)
* p99 latency stays sub-second per request

This script DOES NOT compute JCR — for the consistency metric, run
``sprint5_head_to_head.py``, which measures both modes.

Usage on the droplet::

    # Start vLLM in another shell:
    PYTHONPATH=. python3 -m apohara_context_forge.vllm_plugin.serve \\
        --model meta-llama/Llama-3-8B --port 8000

    # Then in this shell:
    PYTHONPATH=. python3 scripts/sprint5_5agent_workload.py \\
        --vllm-endpoint http://localhost:8000 \\
        --n-requests 200 \\
        --out logs/mi300x_vllm_e2e_$(date +%s).json

Local mock smoke (no vLLM needed)::

    PYTHONPATH=. python3 scripts/sprint5_5agent_workload.py --mock \\
        --n-requests 50 --out /tmp/vllm_e2e_smoke.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow running as a script without ``-m``
sys.path.insert(0, str(Path(__file__).parent))
from _sprint5_pipeline import load_pipeline_config, run_workload  # noqa: E402

logger = logging.getLogger("sprint5_5agent_workload")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path,
                   default=Path("configs/sprint5_5agent.yaml"),
                   help="Pipeline config YAML")
    p.add_argument("--n-requests", type=int, default=200,
                   help="Number of pipeline requests")
    p.add_argument("--vllm-endpoint", default="http://localhost:8000",
                   help="vLLM HTTP endpoint (ignored if --mock)")
    p.add_argument("--mock", action="store_true",
                   help="Skip vLLM, generate synthetic responses")
    p.add_argument("--lobstertrap-endpoint", default=None,
                   help="If set (e.g. http://localhost:8080), route all 5-agent "
                        "requests through Lobster Trap proxy. Each request includes "
                        "_lobstertrap declared-intent metadata for mismatch detection.")
    p.add_argument("--critic-provider", default=None,
                   help="Override the critic agent's model (e.g. 'gemini-3-pro'). "
                        "Useful for the TechEx Gemini Award angle: demonstrate "
                        "INV-15 protects cross-vendor judge agents.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSON (logs/mi300x_vllm_e2e_<ts>.json)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return 1

    cfg = load_pipeline_config(args.config)
    cfg.n_requests = args.n_requests
    mode = "mock" if args.mock else "vllm"

    logger.info("Mode=%s n_requests=%d inv15_enabled=%s",
                mode, args.n_requests, cfg.inv15_enabled)

    payload = run_workload(
        config=cfg,
        n_requests=args.n_requests,
        mode=mode,
        vllm_endpoint=args.vllm_endpoint,
        seed=args.seed,
        lobstertrap_endpoint=args.lobstertrap_endpoint,
        critic_provider_override=args.critic_provider,
    )
    payload["timestamp_unix"] = int(time.time())
    payload["script"] = "sprint5_5agent_workload"
    payload["sprint"] = 5

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", args.out)

    summary = payload["summary"]
    print(json.dumps({
        "n_requests": summary["n_requests"],
        "mode": summary["mode"],
        "latency_ms_p50": summary["latency_ms_p50"],
        "latency_ms_p99": summary["latency_ms_p99"],
        "tokens_per_request_mean": summary["tokens_per_request_mean"],
        "inv15_fires_total": summary["inv15_fires_total"],
        "inv15_fire_rate": summary["inv15_fire_rate"],
        "jcr": summary["jcr"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
