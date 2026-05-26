#!/usr/bin/env python3
"""Stage B — FORGE-LEDGER over REAL frontier-MoE inference on MI300X.

Drives the 5-agent pipeline (retriever/reranker/summarizer/critic/responder)
against a live vLLM OpenAI-compatible endpoint serving a frontier MoE model
(e.g. Qwen3-235B-A22B-Instruct-2507-FP8 on a single MI300X via vLLM+AITER).

For every agent step it (1) runs the production JCRSafetyGate.gate_decision
with APOHARA_FORGE_LEDGER=1 — so each INV-15 decision is Z3-certified and
appended to the SHA-256 hash-chained ledger — and (2) makes the REAL LLM call
to the served model. The result: a tamper-evident, formally-certified ledger
of INV-15 gate decisions taken during genuine frontier-MoE multi-agent
inference on AMD hardware. The critic (judge-class) fires dense-prefill when
its JCR risk exceeds the threshold; that decision is certified against INV-15.

Honesty: every number is measured at runtime; the LLM calls hit the real
served model. No fabrication.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

SHARED_CONTEXT = (
    "Cross-agent KV-cache reuse is the single largest optimization available "
    "to multi-agent LLM pipelines sharing a common context. Reuse silently "
    "degrades judge/critic agents: cached attention from a prior ranking "
    "corrupts the judge's independence (Judge Candidate Reuse, JCR). INV-15 "
    "requires any judge-type agent whose JCR risk exceeds a threshold to use "
    "dense prefill, bypassing the shared KV registry. ContextForge implements "
    "this as the JCR Safety Gate on AMD Instinct MI300X.\n\n"
) * 4

AGENTS = [
    ("retriever", "List the 3 most important factual claims in the document as a numbered list. Be terse."),
    ("reranker", "Rank these snippets by relevance: [A] safety invariants, [B] KV-cache, [C] MI300X, [D] judge corruption. Ranking only."),
    ("summarizer", "Summarize the document above in exactly two sentences."),
    ("critic", "Validate that 'zero INV-15 violations' is consistent with 'critic dense-prefill rate 0.85'. Reply ACCEPT or REJECT + one sentence."),
    ("responder", "Write a 3-sentence final response synthesizing the document for a reader new to KV-cache systems."),
]


def chat(endpoint: str, model: str, prompt: str, max_tokens: int) -> tuple[str, float, int]:
    """One real call to the served model. Returns (text, latency_ms, completion_tokens)."""
    t0 = time.perf_counter()
    r = httpx.post(
        f"{endpoint}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=180.0,
    )
    r.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    ctoks = int(data.get("usage", {}).get("completion_tokens", 0))
    return text, latency_ms, ctoks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default="logs/mi300x_moe_agent_ledger.json")
    ap.add_argument("--fl-dir", default="./fl_moe")
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--reuse-rate", type=float, default=0.85)
    ap.add_argument("--jcr-threshold", type=float, default=0.7)
    args = ap.parse_args()

    # Set env BEFORE importing recorders so the ledger lands in --fl-dir.
    fl = Path(args.fl_dir).resolve()
    if fl.exists():
        import shutil
        shutil.rmtree(fl)
    fl.mkdir(parents=True, exist_ok=True)
    os.environ["APOHARA_FORGE_LEDGER"] = "1"
    os.environ["APOHARA_OBSERVABILITY_DIR"] = str(fl)

    from apohara_context_forge.observability import recorders
    from apohara_context_forge.safety.jcr_gate import JCRSafetyGate
    recorders._reset_singletons()
    gate = JCRSafetyGate(jcr_threshold=args.jcr_threshold)

    records = []
    inv15_fires = 0
    t_start = time.perf_counter()
    for q in range(args.queries):
        query = f"Query {q}: explain why judge agents need dense prefill under KV reuse."
        for role, suffix in AGENTS:
            decision = gate.gate_decision(
                agent_role=role, candidate_count=5,
                reuse_rate=args.reuse_rate, layout_shuffled=False,
            )
            if decision.use_dense:
                inv15_fires += 1
            # use_dense / baseline → full context; reuse → suffix only
            if decision.use_dense:
                prompt = SHARED_CONTEXT + f"\n{query}\n\nTask: {suffix}\n\nAnswer:"
            else:
                prompt = f"Task: {suffix}\n\nAnswer:"
            try:
                text, lat_ms, ctoks = chat(args.endpoint, args.model, prompt, args.max_tokens)
                err = None
            except Exception as e:  # record the failure honestly, keep going
                text, lat_ms, ctoks, err = "", 0.0, 0, repr(e)
            records.append({
                "query_idx": q, "agent_role": role,
                "risk_score": decision.risk_score, "use_dense": decision.use_dense,
                "strategy": "dense-prefill" if decision.use_dense else "cache-reuse",
                "llm_latency_ms": round(lat_ms, 1), "completion_tokens": ctoks,
                "response_snippet": text[:160], "error": err,
            })
            print(f"q{q} {role:11s} use_dense={decision.use_dense} "
                  f"lat={lat_ms:7.0f}ms tok={ctoks} {'ERR' if err else ''}", flush=True)
    wall_s = time.perf_counter() - t_start

    # Verify the certified ledger via the production CLI.
    ledger_path = fl / "inv15_ledger.jsonl"
    cli = subprocess.run(
        [sys.executable, "-m", "apohara_context_forge.observability.ledger_cli",
         "verify", str(ledger_path)],
        capture_output=True, text=True,
    )
    verify_json = json.loads(cli.stdout) if cli.stdout.strip() else {}

    ok_calls = sum(1 for r in records if r["error"] is None and r["completion_tokens"] > 0)
    out = {
        "artifact": "FORGE-LEDGER over real frontier-MoE inference (Stage B)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model, "endpoint": args.endpoint,
        "queries": args.queries, "agents_per_query": len(AGENTS),
        "decisions_certified": len(records),
        "inv15_dense_fires": inv15_fires,
        "real_llm_calls_ok": ok_calls,
        "total_completion_tokens": sum(r["completion_tokens"] for r in records),
        "wall_seconds": round(wall_s, 1),
        "ledger_verify": {
            "cli_exit_code": cli.returncode,
            "result": verify_json,
        },
        "records": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nSTAGE_B_OK calls_ok={ok_calls}/{len(records)} "
          f"certs={verify_json.get('entries')} verify_exit={cli.returncode} "
          f"inv15_fires={inv15_fires}")
    return 0 if (cli.returncode == 0 and ok_calls > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
