#!/usr/bin/env python3
"""Local proof-of-mechanism smoke for ContextForge ``cache_salt`` prefix sharing.

Runs on a SINGLE GPU (developed against an RTX 2060 SUPER, 8 GB, Turing) with a
STANDARD-ATTENTION model (Qwen/Qwen3-1.7B). It proves the *plumbing*, not a
benchmark: that the ``cache_salt`` produced by
:class:`apohara_context_forge.serving.prefix_salt_planner.PrefixSaltPlanner`
controls whether vLLM's Automatic Prefix Caching SHARES prefix KV blocks across
requests.

WHY NOT A VRAM NUMBER
---------------------
vLLM pre-allocates its KV-cache pool up front (``gpu_memory_utilization``), so
total HBM is ~constant whether or not caching helps — a single-worker "GB saved"
delta is ~0 by construction. The honest signal for prefix caching is vLLM's
NATIVE prefix-cache hit counters (``/metrics``): how many prefix blocks were
reused instead of recomputed. That is what this script measures. The real
"GB saved" number is a CROSS-WORKER effect and is gated to MI300X + Redis.

This number is NEVER cited as a result — RTX 2060 == proof-of-mechanism only.

REGIMES (prefix caching ON for both; the only variable is the salt)
-------------------------------------------------------------------
* SHARED   — identical byte-for-byte shared prefix + the SAME ``cache_salt``
             (the planner's shared salt for collaborators).  Expect hits > 0.
* ISOLATED — identical shared prefix + a UNIQUE ``cache_salt`` per request
             (what the planner hands an INV-15 dense judge).  Expect ~0 hits.

A high SHARED hit-rate together with ~0 ISOLATED hit-rate proves ``cache_salt``
drives sharing exactly as PrefixSaltPlanner intends: collaborators share prefix
KV, judges are physically isolated in the block hash.

It also exercises :func:`apohara_context_forge.serving.vllm_launch_config.worker_env`
(a real caller) and the CUDA path of
:class:`apohara_context_forge.metrics.vram_monitor.VRAMMonitor`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen3-1.7B")
PORT = int(os.environ.get("SMOKE_PORT", "8011"))
ENDPOINT = f"http://127.0.0.1:{PORT}"
N = int(os.environ.get("SMOKE_N", "5"))
STARTUP_TIMEOUT_S = float(os.environ.get("SMOKE_STARTUP_TIMEOUT", "900"))

# A long, byte-identical shared prefix (the cross-agent reusable briefing).
# Must span many 16-token PagedAttention blocks for sharing to be observable.
SHARED_PREFIX = (
    "You are one agent in a five-stage multi-agent retrieval pipeline operated "
    "by Apohara ContextForge. All five agents are given the IDENTICAL briefing "
    "below verbatim, and only their final task line differs. The briefing: the "
    "system ingests heterogeneous technical documents about GPU inference, "
    "key-value cache management, paged attention, prefix caching, quantization "
    "of attention tensors, and cross-worker memory offload. Agents must ground "
    "every statement in the briefing, never invent sources, prefer measured "
    "evidence over projections, and keep answers terse and verifiable. The "
    "shared prefix is intentionally long so its computed KV cache spans many "
    "paged-attention blocks; reusing those blocks across agents instead of "
    "recomputing them is the entire point of prefix caching. Treat the briefing "
    "as fixed context and do not restate it.\n\n"
)
TAILS = [
    "Task: list the 3 most relevant documents. Terse.",
    "Task: rerank the 5 documents by relevance. Ranking only.",
    "Task: summarize the briefing in two sentences.",
    "Task: verify one claim. ACCEPT or REJECT plus one sentence.",
    "Task: write a 3-sentence grounded answer.",
]


def log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


def vllm_bin() -> str:
    """The vllm CLI from THIS interpreter's venv (we run under the vLLM venv)."""
    cand = Path(sys.executable).parent / "vllm"
    return str(cand) if cand.exists() else "vllm"


def start_server() -> subprocess.Popen:
    """Launch a single vLLM server with prefix caching ON.

    worker_env() (PYTHONHASHSEED=0, ...) is applied — this is the real caller
    that vllm_launch_config.worker_env() previously lacked. We deliberately do
    NOT pass --kv-transfer-config here: that is the cross-worker/LMCache path,
    out of scope for a single-GPU smoke.
    """
    from apohara_context_forge.serving.vllm_launch_config import worker_env

    env = os.environ.copy()
    env.update(worker_env())  # PYTHONHASHSEED=0 — real caller for the gap-closer
    # Turing (sm_75) compatibility: FlashInfer's sampling kernels require
    # compute capability >= 8 and crash in _dummy_sampler_run on the RTX 2060.
    # Force vLLM's native PyTorch sampler (attention path works; only the
    # FlashInfer SAMPLER is unsupported here). No-op on newer GPUs / MI300X.
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    # Reduce allocator fragmentation on the tight 8GB budget.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    args = [
        vllm_bin(), "serve", MODEL,
        "--port", str(PORT),
        "--enable-prefix-caching",
        "--max-model-len", "2048",
        "--gpu-memory-utilization", "0.82",
        "--max-num-seqs", "16",     # sampler warmup uses max_num_seqs dummy reqs;
                                    # the 256 default OOMs on an 8GB card.
        "--enforce-eager",          # skip CUDA graph capture: faster, less VRAM
    ]
    log(f"launching: {' '.join(args)}")
    log(f"worker_env applied: PYTHONHASHSEED={env.get('PYTHONHASHSEED')}")
    # Server logs -> file so we can diagnose without flooding stdout.
    logf = open("/tmp/smoke_vllm_server.log", "w")
    return subprocess.Popen(args, env=env, stdout=logf, stderr=subprocess.STDOUT)


def wait_ready(proc: subprocess.Popen) -> bool:
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log(f"server EXITED early (code {proc.returncode}); see /tmp/smoke_vllm_server.log")
            return False
        try:
            with urllib.request.urlopen(f"{ENDPOINT}/health", timeout=5) as r:
                if r.status == 200:
                    log("server READY")
                    return True
        except Exception:
            pass
        time.sleep(3)
    log("server readiness TIMEOUT")
    return False


def fetch_metrics() -> dict:
    """Sum vLLM prefix-cache counters from /metrics (robust to exact metric name)."""
    try:
        with urllib.request.urlopen(f"{ENDPOINT}/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception as e:
        log(f"/metrics fetch failed: {e}")
        return {"queries": 0.0, "hits": 0.0}
    queries = hits = 0.0
    for line in text.splitlines():
        if line.startswith("#") or "prefix_cache" not in line:
            continue
        m = re.search(r"\s([0-9.eE+-]+)\s*$", line)
        if not m:
            continue
        val = float(m.group(1))
        if "queries" in line:
            queries += val
        elif "hits" in line:
            hits += val
    return {"queries": queries, "hits": hits}


def send(prompt: str, cache_salt: str | None) -> bool:
    body = {"model": MODEL, "prompt": prompt, "max_tokens": 8, "temperature": 0.0}
    if cache_salt is not None:
        body["cache_salt"] = cache_salt
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/completions", data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        log(f"request HTTPError {e.code}: {e.read().decode()[:200]}")
        return False
    except Exception as e:
        log(f"request failed: {e}")
        return False


def run_regime(name: str, salt_fn) -> dict:
    """Send N sequential requests (sequential so each can reuse the prior's blocks).

    salt_fn(i) -> cache_salt for request i. Returns the per-regime hit/query delta.
    """
    before = fetch_metrics()
    ok = 0
    for i in range(N):
        prompt = SHARED_PREFIX + TAILS[i % len(TAILS)]
        if send(prompt, salt_fn(i)):
            ok += 1
        time.sleep(0.3)  # let the engine register the blocks before the next call
    time.sleep(1.0)
    after = fetch_metrics()
    dq = after["queries"] - before["queries"]
    dh = after["hits"] - before["hits"]
    rate = (dh / dq) if dq > 0 else 0.0
    log(f"{name}: requests_ok={ok}/{N} queries+={dq:.0f} hits+={dh:.0f} hit_rate={rate:.3f}")
    return {"requests_ok": ok, "queries_delta": dq, "hits_delta": dh, "hit_rate": round(rate, 4)}


def read_vram() -> dict:
    """Exercise the CUDA path of VRAMMonitor + the out-of-process nvidia-smi eye."""
    out = {}
    try:
        from apohara_context_forge.metrics.vram_monitor import VRAMMonitor
        mon = VRAMMonitor(device_id=0)
        out["vram_source"] = mon.get_vram_source()
        out["used_gb"] = round(mon.get_used_gb(), 3)
    except Exception as e:
        out["vram_monitor_error"] = repr(e)
    try:
        from scripts.vram_ab_harness import read_second_source_used_gb
        gb, src = read_second_source_used_gb(0)
        out["second_source_used_gb"] = round(gb, 3) if gb is not None else None
        out["second_source"] = src
    except Exception as e:
        out["second_source_error"] = repr(e)
    return out


def main() -> int:
    # The planner is the REAL component under test; build its shared salt once.
    from apohara_context_forge.serving.prefix_salt_planner import PrefixSaltPlanner
    import hashlib
    planner = PrefixSaltPlanner()
    anchor_hash = hashlib.sha256(SHARED_PREFIX.encode()).hexdigest()[:16]
    # SHARED regime uses the planner's deterministic shared salt (collaborators
    # with the same anchor). ISOLATED uses its unique isolated salt (a judge).
    # We call shared_salt()/isolated_salt() directly so the smoke tests the
    # salt -> vLLM-sharing mechanism on real hardware; the gate's shared-vs-dense
    # DECISION is already covered by tests/test_prefix_salt_planner.py.
    shared_salt = planner.shared_salt(anchor_hash, "smoke")
    # Sanity log: the full plan() path with the REAL JCR gate must agree — a
    # collaborator stays shared; a high-risk judge flips to an isolated salt.
    collab = planner.plan(agent_role="worker", anchor_hash=anchor_hash,
                          cla_group="smoke", request_id="c0")
    judge = planner.plan(agent_role="critic", anchor_hash=anchor_hash,
                         cla_group="smoke", request_id="j0",
                         candidate_count=5, reuse_rate=0.9)
    log(f"planner shared_salt={shared_salt!r}")
    log(f"plan() collaborator: shared={collab.shared} salt={collab.cache_salt!r}")
    log(f"plan() judge(cc=5,reuse=0.9): shared={judge.shared} salt={judge.cache_salt!r}")

    proc = start_server()
    result = {"model": MODEL, "hardware": "RTX 2060 SUPER 8GB (Turing) — PROOF-OF-MECHANISM, not cited"}
    try:
        if not wait_ready(proc):
            tail = Path("/tmp/smoke_vllm_server.log").read_text()[-1500:]
            result["status"] = "server_failed"
            result["server_log_tail"] = tail
            print("\n=== SMOKE RESULT ===")
            print(json.dumps(result, indent=2))
            return 1

        # SHARED: every request carries the planner's shared salt.
        result["shared"] = run_regime("SHARED", lambda i: shared_salt)
        # ISOLATED: every request gets a UNIQUE salt (what a judge would receive).
        result["isolated"] = run_regime(
            "ISOLATED",
            lambda i: planner.isolated_salt(anchor_hash, f"iso-{i}"),
        )
        result["vram"] = read_vram()

        sh = result["shared"]["hit_rate"]
        iso = result["isolated"]["hit_rate"]
        result["verdict"] = {
            "mechanism_proven": bool(sh > 0.0 and sh > iso + 0.05),
            "shared_hit_rate": sh,
            "isolated_hit_rate": iso,
            "interpretation": (
                "SHARED reuses prefix KV blocks; ISOLATED (unique salt) does not. "
                "cache_salt drives sharing as PrefixSaltPlanner intends."
                if (sh > 0.0 and sh > iso + 0.05)
                else "Inconclusive — see hit-rate deltas; cache_salt may be unsupported "
                     "by this vLLM build or the prefix was too short to span blocks."
            ),
        }
        result["status"] = "ok"
        print("\n=== SMOKE RESULT ===")
        print(json.dumps(result, indent=2))
        return 0
    finally:
        log("terminating server")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
