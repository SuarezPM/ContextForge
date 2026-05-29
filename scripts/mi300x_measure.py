#!/usr/bin/env python3
"""ContextForge VRAM / concurrency / throughput measurements against a LIVE vLLM.

Backend-agnostic (AMD ROCm via PyRSMI, NVIDIA via the CUDA path) — it talks to an
already-running ``vllm serve`` over HTTP and reads HBM via the shared
:class:`apohara_context_forge.metrics.vram_monitor.VRAMMonitor` plus an
out-of-process second source (``rocm-smi``/``nvidia-smi``). The companion
orchestrator (``mi300x_squeeze_all.sh``) launches one vLLM per model and runs
this + the existing measurement scripts (``mi300x_contextforge_e2e.py`` for the
token side, ``mi300x_niah.py``, ``mi300x_lmcache_smoke.py``, etc.) against it.

This file measures the four things the existing scripts did NOT:

  footprint    — HBM used once the model is loaded (the "X GB on one MI300X" datum)
  vram_prefix  — KV-block sharing A/B: SHARED cache_salt (collaborators) vs
                 ISOLATED salt (judges) → vLLM prefix-cache hit-rate + HBM delta
  concurrency  — how many concurrent shared-prefix requests sustain before the
                 server saturates, with vs without shared prefix (operator metric)
  throughput   — TTFT + decode tok/s for the multi-agent pipeline, shared vs not

All numbers are honestly sourced (``vram_source`` names the backend). Nothing is
fabricated: if a reader is unavailable the field is null with a reason. NOT a
benchmark harness — a measurement probe; the orchestrator owns model lifecycle.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Long shared prefix (the cross-agent reusable briefing). Long enough to span
# many PagedAttention blocks so KV sharing is measurable on a real model.
SHARED_PREFIX = (
    "You are one agent in a multi-agent retrieval pipeline operated by Apohara "
    "ContextForge. Every agent is given this identical briefing verbatim; only "
    "the final task line differs. The shared system context is intentionally long "
    "so the key-value cache it produces spans many attention blocks, and reusing "
    "those blocks across agents instead of recomputing them per agent is the "
    "point of prefix caching. Ground all answers in the briefing; never invent "
    "sources; prefer measured evidence; keep answers terse and verifiable.\n\n"
) * 4
TAILS = [
    "Task: list the 3 most relevant documents. Terse.",
    "Task: rerank the documents by relevance. Ranking only.",
    "Task: summarize the briefing in two sentences.",
    "Task: verify one claim. ACCEPT or REJECT plus one sentence.",
    "Task: write a 3-sentence grounded answer.",
]


def _post(endpoint: str, model: str, prompt: str, *, salt=None, max_tokens=16, stream=False):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    if salt is not None:
        body["cache_salt"] = salt
    if stream:
        body["stream"] = True
    req = urllib.request.Request(
        f"{endpoint}/v1/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=180)


def fetch_prefix_metrics(endpoint: str) -> dict:
    """Sum vLLM prefix-cache + external-KV counters from /metrics (name-robust)."""
    import re
    try:
        with urllib.request.urlopen(f"{endpoint}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception as e:
        return {"error": repr(e)}
    out = {"queries": 0.0, "hits": 0.0, "external_queries": 0.0, "external_hits": 0.0,
           "external_kv_tokens": 0.0}
    for line in text.splitlines():
        if line.startswith("#") or "prefix_cache" not in line and "kv_transfer" not in line \
           and "external_kv" not in line:
            continue
        m = re.search(r"\s([0-9.eE+-]+)\s*$", line)
        if not m:
            continue
        v = float(m.group(1))
        if "external_prefix_cache_queries" in line: out["external_queries"] += v
        elif "external_prefix_cache_hits" in line: out["external_hits"] += v
        elif "prefix_cache_queries" in line: out["queries"] += v
        elif "prefix_cache_hits" in line: out["hits"] += v
        elif "external_kv_transfer" in line: out["external_kv_tokens"] += v
    return out


def read_hbm(device_id: int) -> dict:
    out = {}
    try:
        from apohara_context_forge.metrics.vram_monitor import VRAMMonitor
        m = VRAMMonitor(device_id=device_id)
        out["used_gb"] = round(m.get_used_gb(), 3)      # read used first
        out["total_gb"] = round(m.get_total_gb(), 3)
        out["vram_source"] = m.get_vram_source()
    except Exception as e:
        out["vram_monitor_error"] = repr(e)
    try:
        from scripts.vram_ab_harness import read_second_source_used_gb
        gb, src = read_second_source_used_gb(device_id)
        out["second_source_used_gb"] = round(gb, 3) if gb is not None else None
        out["second_source"] = src
    except Exception as e:
        out["second_source_error"] = repr(e)
    return out


# --------------------------------------------------------------------------- #
def stage_footprint(endpoint, model, device_id) -> dict:
    """HBM occupied with the model loaded + a warmup request done."""
    try:
        _post(endpoint, model, SHARED_PREFIX + TAILS[0], max_tokens=4).read()
    except Exception as e:
        return {"error": f"warmup failed: {e!r}"}
    time.sleep(2.0)
    return {"hbm": read_hbm(device_id)}


def stage_vram_prefix(endpoint, model, device_id, n=8) -> dict:
    """KV-block sharing A/B: SHARED salt vs ISOLATED salt → hit-rate + HBM delta."""
    import hashlib
    from apohara_context_forge.serving.prefix_salt_planner import PrefixSaltPlanner
    planner = PrefixSaltPlanner()
    anchor = hashlib.sha256(SHARED_PREFIX.encode()).hexdigest()[:16]

    def run(salt_fn, label):
        before = fetch_prefix_metrics(endpoint)
        for i in range(n):
            try:
                _post(endpoint, model, SHARED_PREFIX + TAILS[i % len(TAILS)],
                      salt=salt_fn(i), max_tokens=8).read()
            except Exception:
                pass
            time.sleep(0.2)
        time.sleep(1.0)
        after = fetch_prefix_metrics(endpoint)
        dq = after.get("queries", 0) - before.get("queries", 0)
        dh = after.get("hits", 0) - before.get("hits", 0)
        return {"queries_delta": dq, "hits_delta": dh,
                "hit_rate": round(dh / dq, 4) if dq > 0 else 0.0,
                "hbm": read_hbm(device_id)}

    shared = run(lambda i: planner.shared_salt(anchor, "mi300x"), "SHARED")
    isolated = run(lambda i: planner.isolated_salt(anchor, f"iso-{i}"), "ISOLATED")
    return {"shared": shared, "isolated": isolated,
            "mechanism_proven": bool(shared["hit_rate"] > 0 and shared["hit_rate"] > isolated["hit_rate"] + 0.05)}


def stage_concurrency(endpoint, model, device_id, levels=(1, 2, 4, 8, 16, 32, 64)) -> dict:
    """Ramp concurrent shared-prefix requests; record sustained level + peak HBM."""
    results = []
    for c in levels:
        ok = 0
        t0 = time.monotonic()
        with cf.ThreadPoolExecutor(max_workers=c) as ex:
            futs = [ex.submit(lambda i=i: _post(
                endpoint, model, SHARED_PREFIX + TAILS[i % len(TAILS)],
                salt="shared:mi300x", max_tokens=16).read()) for i in range(c)]
            for f in futs:
                try:
                    f.result(timeout=240); ok += 1
                except Exception:
                    pass
        dt = time.monotonic() - t0
        hbm = read_hbm(device_id)
        results.append({"concurrency": c, "completed": ok, "wall_s": round(dt, 2),
                        "hbm_used_gb": hbm.get("used_gb"), "vram_source": hbm.get("vram_source")})
        if ok < c:  # saturated
            break
    sustained = max((r["concurrency"] for r in results if r["completed"] == r["concurrency"]), default=0)
    return {"levels": results, "max_sustained_concurrency": sustained}


def stage_throughput(endpoint, model, n=5) -> dict:
    """TTFT + decode tok/s for the 5-agent pipeline, shared-prefix vs distinct."""
    def measure(shared: bool):
        ttfts, ntok, t_total = [], 0, 0.0
        for i in range(n):
            prefix = SHARED_PREFIX if shared else (SHARED_PREFIX + f" variant-{i} ")
            t0 = time.monotonic()
            try:
                resp = _post(endpoint, model, prefix + TAILS[i % len(TAILS)],
                             salt=("shared:tp" if shared else f"iso:tp-{i}"),
                             max_tokens=64, stream=True)
                first = None
                for raw in resp:
                    if first is None:
                        first = time.monotonic(); ttfts.append(first - t0)
                    ntok += 1
                t_total += time.monotonic() - t0
            except Exception:
                pass
        return {"mean_ttft_s": round(sum(ttfts) / len(ttfts), 4) if ttfts else None,
                "approx_decode_tok_s": round(ntok / t_total, 1) if t_total > 0 else None,
                "n": n}
    return {"shared_prefix": measure(True), "distinct_prefix": measure(False)}


STAGES = {"footprint": stage_footprint, "vram_prefix": stage_vram_prefix,
          "concurrency": stage_concurrency, "throughput": stage_throughput}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--stages", default="footprint,vram_prefix,concurrency,throughput")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    result = {"model": args.model, "endpoint": args.endpoint}
    for name in args.stages.split(","):
        name = name.strip()
        fn = STAGES.get(name)
        if not fn:
            result[name] = {"error": "unknown stage"}; continue
        print(f"[measure] stage {name} …", flush=True)
        try:
            if name == "throughput":
                result[name] = fn(args.endpoint, args.model)
            else:
                result[name] = fn(args.endpoint, args.model, args.device_id)
        except Exception as e:
            result[name] = {"error": repr(e)}
        print(f"[measure] {name}: {json.dumps(result[name])[:200]}", flush=True)

    payload = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
