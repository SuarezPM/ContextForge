#!/usr/bin/env python3
"""End-to-end ContextForge test against a LIVE MoE on MI300X.

This is the real product test: it runs ContextForge's actual coordination
stack (ContextRegistry dedup + CompressionCoordinator + LLMLingua compressor
+ JCRSafetyGate INV-15 + FORGE-LEDGER) over a multi-agent shared-context
workload, and sends BOTH the baseline (full) and the ContextForge-optimized
context to the live vLLM endpoint serving a frontier MoE.

Measures the actual value-prop: tokens before/after (compression), shared
prefix reused across agents (cross-agent KV dedup), INV-15 gate fires +
certified ledger, and that the optimized context still yields valid
generation on the real model. Honesty: every number is measured at runtime.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

# A long shared context (system prompt + shared briefing) — byte-identical
# across all agents, so it is the cross-agent reusable prefix. Long enough to
# exceed the compression threshold.
SHARED = (
    "SYSTEM: You are part of a multi-agent retrieval-augmented pipeline running "
    "on an AMD Instinct MI300X accelerator with 192 GB of HBM3 memory. The "
    "pipeline shares a common context across five agents (retriever, reranker, "
    "summarizer, critic, responder). Cross-agent KV-cache reuse is the single "
    "largest optimization available to such pipelines: when the agents share a "
    "common prefix, the key-value cache computed for that prefix can be reused "
    "instead of recomputed, and the unique per-agent tail can be compressed. "
    "However, aggressive KV reuse silently degrades judge and critic agents, "
    "because attention patterns cached from a prior ranking corrupt the judge's "
    "independence when it compares multiple candidates. This failure mode is "
    "called Judge Candidate Reuse (JCR). INV-15 is a formal safety invariant: "
    "any judge-type agent whose JCR risk score exceeds a fixed threshold must "
    "use dense prefill, bypassing the shared KV registry. The risk score is a "
    "closed-form function of agent role, candidate count, reuse rate, and "
    "candidate-layout shuffling. ContextForge implements INV-15 as the JCR "
    "Safety Gate, and additionally deduplicates shared prefixes across agents "
    "and compresses unique tails with LLMLingua, all coordinated by a single "
    "shared-context compiler. KNOWLEDGE BASE: Document 1 covers attention and "
    "KV caches. Document 2 covers Mixture-of-Experts routing and FP8 weights. "
    "Document 3 covers long-context needle-in-a-haystack evaluation. Document 4 "
    "covers formal verification of inference invariants with the Z3 SMT solver. "
    "Document 5 covers tamper-evident hash-chained audit ledgers for AI safety. "
) * 3  # ~ a few hundred tokens of shared prefix

AGENTS = [
    ("retriever", "List the 3 most relevant documents for a query about KV-cache reuse. Be terse."),
    ("reranker", "Rerank the 5 documents by relevance to judge-agent safety. Ranking only."),
    ("summarizer", "Summarize the shared briefing in exactly two sentences."),
    ("critic", "Verify whether 'INV-15 mandates dense prefill for risky judge agents' is consistent with the briefing. ACCEPT or REJECT + one sentence."),
    ("responder", "Write a 3-sentence answer explaining why judge agents need dense prefill under KV reuse."),
]
QUESTION = "\n\nAnswer the task above based on the briefing."


def chat(endpoint, model, prompt, max_tokens=64):
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{endpoint}/v1/chat/completions",
                       json={"model": model, "messages": [{"role": "user", "content": prompt}],
                             "max_tokens": max_tokens, "temperature": 0.0}, timeout=180.0)
        r.raise_for_status()
        d = r.json()
        return d["choices"][0]["message"]["content"], (time.perf_counter() - t0) * 1000, \
            int(d.get("usage", {}).get("prompt_tokens", 0)), None
    except Exception as e:
        return "", (time.perf_counter() - t0) * 1000, 0, repr(e)


async def run(args):
    fl = Path(args.fl_dir).resolve()
    if fl.exists():
        import shutil; shutil.rmtree(fl)
    fl.mkdir(parents=True, exist_ok=True)
    os.environ["APOHARA_FORGE_LEDGER"] = "1"
    os.environ["APOHARA_OBSERVABILITY_DIR"] = str(fl)

    from apohara_context_forge.registry.context_registry import ContextRegistry
    from apohara_context_forge.compression.compressor import ContextCompressor
    from apohara_context_forge.compression.coordinator import CompressionCoordinator
    from apohara_context_forge.safety.jcr_gate import JCRSafetyGate
    from apohara_context_forge.normalization.prefix_normalizer import PrefixNormalizer
    from apohara_context_forge.metrics.vram_monitor import VRAMMonitor
    from apohara_context_forge.observability import recorders
    recorders._reset_singletons()

    # PrefixNormalizer enforces a byte-identical shared system prefix across all
    # agents (the cross-agent reusable prefix); SHARED is the canonical prompt.
    normalizer = PrefixNormalizer(canonical_system_prompt=SHARED)
    # HBM instrumentation: sample steady-state used VRAM before/during/after.
    vram = VRAMMonitor()
    hbm_start_gb = vram.get_used_gb()
    vram_source = vram.get_vram_source()

    registry = ContextRegistry()
    await registry.start()
    compressor = ContextCompressor()
    coord = CompressionCoordinator(registry=registry, compressor=compressor)
    gate = JCRSafetyGate()

    # Register all agents so cross-agent dedup (FAISS+LSH) has a corpus.
    reg_ok = 0
    for aid, role in AGENTS:
        try:
            await registry.register_agent(aid, SHARED, role)
            reg_ok += 1
        except Exception as e:
            print(f"[register_agent {aid}] FAILED: {type(e).__name__}: {e}", flush=True)

    records = []
    base_tok_total = cf_tok_total = 0
    prefix_reused_total = 0
    inv15_fires = 0
    hbm_mid_gb = None
    for idx, (aid, role) in enumerate(AGENTS):
        # PrefixNormalizer assembles the prompt with a byte-identical SHARED
        # system prefix across agents (the cross-agent reusable prefix), instead
        # of the prior hand-assembly. The per-agent task is the user segment.
        full = normalizer.normalize(agent_id=aid, user_prompt=role, agent_role_prompt="Task:")
        if idx == 0:
            # "during" sample: HBM after the first agent's context is in flight.
            hbm_mid_gb = vram.get_used_gb()
        # ContextForge coordinator decision (dedup + compression)
        strat, otok, ftok, saved, pct, prefix_tok, final_ctx, derr = \
            "error", 0, 0, 0, 0.0, 0, full, None
        try:
            dec = await coord.decide(aid, full)
            strat = dec.strategy
            otok, ftok, saved = dec.original_tokens, dec.final_tokens, dec.tokens_saved
            pct = dec.savings_pct
            prefix_tok = getattr(dec, "shared_prefix", "") and \
                len((getattr(dec, "shared_prefix", "") or "").split()) or 0
            final_ctx = dec.final_context
        except Exception as e:
            derr = f"{type(e).__name__}: {e}"
            # Fallback: direct compression so we still measure the codec value.
            try:
                comp, _ratio = await compressor.compress(full, 0.5)
                strat, final_ctx = "compress(fallback)", comp
                otok, ftok = len(full.split()), len(comp.split())
                saved, pct = otok - ftok, (otok - ftok) / max(otok, 1) * 100
            except Exception as e2:
                derr += f" | compress fallback: {type(e2).__name__}: {e2}"

        # INV-15 gate (judge roles fire) -> Z3-certified, hash-chained ledger
        gdec = gate.gate_decision(agent_role=aid, candidate_count=5,
                                  reuse_rate=0.85, layout_shuffled=False)
        if gdec.use_dense:
            inv15_fires += 1

        # Real MoE calls: baseline (full) vs ContextForge-optimized (final_ctx)
        base_resp, base_lat, base_ptok, berr = chat(args.endpoint, args.model, full + QUESTION, args.max_tokens)
        cf_resp, cf_lat, cf_ptok, cferr = chat(args.endpoint, args.model, final_ctx + QUESTION, args.max_tokens)

        base_tok_total += (base_ptok or otok)
        cf_tok_total += (cf_ptok or ftok)
        prefix_reused_total += prefix_tok
        records.append({
            "agent": aid, "strategy": strat,
            "coord_original_tokens": otok, "coord_final_tokens": ftok,
            "coord_tokens_saved": saved, "coord_savings_pct": round(pct, 1),
            "shared_prefix_words": prefix_tok,
            "server_prompt_tokens_baseline": base_ptok,
            "server_prompt_tokens_contextforge": cf_ptok,
            "use_dense_inv15": gdec.use_dense, "risk": round(gdec.risk_score, 2),
            "base_latency_ms": round(base_lat, 0), "cf_latency_ms": round(cf_lat, 0),
            "base_ok": berr is None and bool(base_resp),
            "cf_ok": cferr is None and bool(cf_resp),
            "coord_error": derr,
        })
        print(f"{aid:11s} strat={strat:20s} tok {otok}->{ftok} ({pct:.0f}% saved) "
              f"srv_tok {base_ptok}->{cf_ptok} use_dense={gdec.use_dense} "
              f"base={base_lat:.0f}ms cf={cf_lat:.0f}ms", flush=True)

    # Verify the FORGE-LEDGER chain
    ledger = fl / "inv15_ledger.jsonl"
    cli = subprocess.run([sys.executable, "-m", "apohara_context_forge.observability.ledger_cli",
                          "verify", str(ledger)], capture_output=True, text=True)
    verify = json.loads(cli.stdout) if cli.stdout.strip() else {}

    hbm_end_gb = vram.get_used_gb()
    server_savings = (base_tok_total - cf_tok_total) / max(base_tok_total, 1) * 100
    out = {
        "artifact": "ContextForge end-to-end over live MoE (registry+dedup+compression+gate+ledger)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model, "endpoint": args.endpoint,
        "agents_registered": reg_ok, "agents": len(AGENTS),
        "server_prompt_tokens_baseline_total": base_tok_total,
        "server_prompt_tokens_contextforge_total": cf_tok_total,
        "server_token_savings_pct": round(server_savings, 1),
        "shared_prefix_words_reused_total": prefix_reused_total,
        "inv15_dense_fires": inv15_fires,
        # HBM used (GB) sampled in-process before / during / after the run, plus
        # the honest backend label. NOTE: VRAMMonitor reads this PROCESS's host
        # GPU, not the remote vLLM server's GPU; treat as local instrumentation.
        "hbm_used_gb_start": round(hbm_start_gb, 3),
        "hbm_used_gb_during": (round(hbm_mid_gb, 3) if hbm_mid_gb is not None else None),
        "hbm_used_gb_end": round(hbm_end_gb, 3),
        "vram_source": vram_source,
        "ledger_verify": {"exit": cli.returncode, "result": verify},
        "strategies": {s: sum(1 for r in records if r["strategy"] == s)
                       for s in {r["strategy"] for r in records}},
        "records": records,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nE2E_DONE model={args.model} server_savings={server_savings:.1f}% "
          f"baseline_tok={base_tok_total} cf_tok={cf_tok_total} "
          f"inv15_fires={inv15_fires} ledger_verify_exit={cli.returncode} "
          f"certs={verify.get('entries')}")
    try:
        await registry.stop()
    except Exception:
        pass
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fl-dir", default="./fl_e2e")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
