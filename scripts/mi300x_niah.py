#!/usr/bin/env python3
"""Needle-in-a-haystack (NIAH) long-context correctness probe — server-side.

Sends long prompts (filler + an embedded needle fact) to a live vLLM
OpenAI-compatible endpoint and checks whether the model retrieves the needle.
The long context lives entirely inside vLLM (paged attention + FP8 KV +
chunked prefill) — NOTHING is round-tripped through host memory, so this is
safe (unlike the host-side KV codec that swap-killed a VM previously).

Reports a pass/fail grid over (context length × needle depth) plus the
actual prompt-token counts the server reports. Honesty: pass = the exact
secret code string appears in the model's answer.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

NEEDLE_CODE = "QX-7731-ZULU"
NEEDLE = f" The secret authorization code for this session is {NEEDLE_CODE}. "
FILLER = (
    "The history of distributed systems is a long chronicle of trade-offs "
    "between consistency, availability, and partition tolerance, studied at "
    "length across decades of research and production practice. "
)
QUESTION = "What is the secret authorization code for this session? Answer with the code only."


def build_prompt(approx_tokens: int, depth: float) -> str:
    """Build a ~approx_tokens prompt with the needle at the given depth (0..1)."""
    # ~0.75 words/token for English prose → words ≈ tokens * 0.75.
    target_words = int(approx_tokens * 0.75)
    filler_words = FILLER.split()
    reps = max(1, target_words // len(filler_words))
    body_words = (filler_words * reps)
    insert_at = int(len(body_words) * depth)
    body = " ".join(body_words[:insert_at]) + NEEDLE + " ".join(body_words[insert_at:])
    return f"{body}\n\n{QUESTION}"


def probe(endpoint: str, model: str, approx_tokens: int, depth: float) -> dict:
    prompt = build_prompt(approx_tokens, depth)
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            f"{endpoint}/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 32, "temperature": 0.0},
            timeout=600.0,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        ptoks = int(data.get("usage", {}).get("prompt_tokens", 0))
        err = None
    except Exception as e:
        text, ptoks, err = "", 0, repr(e)
    dt = time.perf_counter() - t0
    found = NEEDLE_CODE in text
    return {"approx_tokens": approx_tokens, "depth": depth, "prompt_tokens": ptoks,
            "found": found, "ttft_plus_gen_s": round(dt, 2),
            "answer": text[:80], "error": err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", default="logs/mi300x_niah.json")
    ap.add_argument("--lengths", default="8000,32000,128000,190000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    depths = [float(x) for x in args.depths.split(",")]
    results = []
    for L in lengths:
        for d in depths:
            res = probe(args.endpoint, args.model, L, d)
            results.append(res)
            print(f"len~{L:>7} depth={d} prompt_tok={res['prompt_tokens']:>7} "
                  f"found={res['found']} t={res['ttft_plus_gen_s']}s "
                  f"{'ERR' if res['error'] else ''}", flush=True)

    passed = sum(1 for r in results if r["found"])
    out = {
        "artifact": "NIAH long-context probe (server-side vLLM)",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model, "endpoint": args.endpoint,
        "needle_code": NEEDLE_CODE,
        "probes": len(results), "passed": passed,
        "max_prompt_tokens": max((r["prompt_tokens"] for r in results), default=0),
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nNIAH_DONE passed={passed}/{len(results)} "
          f"max_prompt_tokens={out['max_prompt_tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
