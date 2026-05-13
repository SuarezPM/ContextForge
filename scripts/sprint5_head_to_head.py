"""Sprint 5 Step 4: Apohara plugin ON vs OFF head-to-head benchmark.

Single most important script in Sprint 5. Generates the paper v2.1
§6 headline number: **how much JCR (Judge Consistency Rate) drops
when KV-cache reuse is enabled without INV-15 protection**.

Methodology:

1. Run the 5-agent workload N times in ``apohara_on`` mode (INV-15
   gate enabled — judge agent gets dense prefill on risky reuses).
2. Run the same workload N times in ``apohara_off`` mode (gate
   disabled — all agents share cache freely).
3. Compute JCR for each mode. The Δ is the paper's headline.

Expected outcomes:

* ``apohara_on`` JCR: 0.99+ (INV-15 prevents the silent drop)
* ``apohara_off`` JCR: 0.77-0.92 (Liang et al. 2026 measured range)
* Δ JCR: 0.07-0.23 absolute drop

Acceptance (paper v2.1 §6):

* JCR(on) > JCR(off) — strict inequality (otherwise the entire
  thesis fails)
* JCR(on) > 0.95 — the gate works
* JCR delta > 0.05 — the gate is meaningfully useful

Usage::

    # On the droplet, after vLLM is up:
    for mode in apohara_on apohara_off; do
      PYTHONPATH=. python3 scripts/sprint5_head_to_head.py \\
          --mode $mode --n-requests 500 \\
          --vllm-endpoint http://localhost:8000 \\
          --out logs/mi300x_h2h_${mode}_$(date +%s).json
    done

    # Diff:
    jq '.summary.jcr' logs/mi300x_h2h_apohara_off_*.json
    jq '.summary.jcr' logs/mi300x_h2h_apohara_on_*.json

Local mock smoke::

    PYTHONPATH=. python3 scripts/sprint5_head_to_head.py --mock \\
        --mode apohara_on  --n-requests 100 --out /tmp/h2h_on.json
    PYTHONPATH=. python3 scripts/sprint5_head_to_head.py --mock \\
        --mode apohara_off --n-requests 100 --out /tmp/h2h_off.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _sprint5_pipeline import load_pipeline_config, run_workload  # noqa: E402

logger = logging.getLogger("sprint5_head_to_head")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)


VALID_MODES = ("apohara_on", "apohara_off")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path,
                   default=Path("configs/sprint5_5agent.yaml"),
                   help="Pipeline config YAML")
    p.add_argument("--mode", choices=VALID_MODES, required=True,
                   help="apohara_on (INV-15 gate enabled) or apohara_off")
    p.add_argument("--n-requests", type=int, default=500,
                   help="Number of pipeline requests")
    p.add_argument("--vllm-endpoint", default="http://localhost:8000")
    p.add_argument("--mock", action="store_true",
                   help="Skip vLLM, generate synthetic responses")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSON (logs/mi300x_h2h_<mode>_<ts>.json)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.exists():
        logger.error("Config not found: %s", args.config)
        return 1

    cfg = load_pipeline_config(args.config)
    cfg.n_requests = args.n_requests
    # Mode override: apohara_on forces INV-15 enabled, off disables it
    cfg.inv15_enabled = (args.mode == "apohara_on")
    backend_mode = "mock" if args.mock else "vllm"

    logger.info("mode=%s inv15_enabled=%s backend=%s n_requests=%d",
                args.mode, cfg.inv15_enabled, backend_mode, args.n_requests)

    payload = run_workload(
        config=cfg,
        n_requests=args.n_requests,
        mode=backend_mode,
        vllm_endpoint=args.vllm_endpoint,
        seed=args.seed,
    )
    payload["timestamp_unix"] = int(time.time())
    payload["script"] = "sprint5_head_to_head"
    payload["sprint"] = 5
    payload["apohara_mode"] = args.mode

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", args.out)

    summary = payload["summary"]
    print(json.dumps({
        "apohara_mode": args.mode,
        "n_requests": summary["n_requests"],
        "jcr": summary["jcr"],
        "accept_rate": summary["accept_rate"],
        "latency_ms_p50": summary["latency_ms_p50"],
        "latency_ms_p99": summary["latency_ms_p99"],
        "total_tokens": summary["total_tokens"],
        "inv15_fires_total": summary["inv15_fires_total"],
        "inv15_fire_rate": summary["inv15_fire_rate"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
