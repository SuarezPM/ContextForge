---
phase: M001
phase_name: ContextForge — Working core, benchmark, and ship
project: ContextForge
generated: 2026-05-10T01:30:00Z
counts:
  decisions: 10
  lessons: 9
  patterns: 13
  surprises: 5
missing_artifacts: []
---

# M001 — Structured Learnings

This is the audit trail of what was learned, decided, and patterned during M001
(S01 → S09). Cross-session memories (Patterns, Lessons, Decisions) are also
persisted to the GSD memory store via `capture_thought`. Surprises stay only
here per the extraction protocol.

## ### Decisions

- **D001 — Inference runtime: vLLM ROCm fork with `--enable-prefix-caching`.** Locked early; validated end-to-end through S09 docker-compose with `--enable-prefix-caching` in the vllm service `command:` list. Non-revisable.
  Source: DECISIONS.md/D001
- **D002 — Primary inference model: `Qwen/Qwen2.5-7B-Instruct`.** Stayed stable through every slice; sized the docker-compose vllm healthcheck `start_period: 120s × 12 retries` for the cold-load characteristic of this exact model.
  Source: DECISIONS.md/D002
- **D003 — Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.** Validated by S03 slice smoke (similarity 0.9884, 209-token shared prefix on two ~1.5 KB contexts). 22 MB and CPU-runnable kept the dev loop fast.
  Source: DECISIONS.md/D003
- **D004 — Prompt compressor: `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` with bert-base-multilingual fallback on OOM.** Original D004 row pointed at base `bert-base-multilingual-cased`; corrected mid-flight in S04/T01 to the fine-tuned `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` because base BERT lacks the LLMLingua-2 token-classification head. Three sites updated: `contextforge/config.py`, `.env.example`, `tests/test_config.py`.
  Source: DECISIONS.md/D004 + S04-SUMMARY.md/Key Decisions
- **D005 — MCP server framework: FastAPI + asyncio.** Validated. The vanilla-FastAPI fallback never had to fire; the official MCP SDK was not adopted for this milestone.
  Source: DECISIONS.md/D005
- **D006 — 4-strategy compression thresholds (sim ≥ 0.85; shared_prefix > 200; ctx > 500).** Validated by 11 hermetic coordinator tests + strict-`>` boundary tests. Thresholds env-configurable; revisable based on benchmark results.
  Source: DECISIONS.md/D006 + S04-SUMMARY.md/Key Decisions
- **D007 — Benchmark protocol: 1 warmup discarded, 2 measurement runs averaged, cold reported separately.** Validated hermetically by S07's 17 tests (run-count parity 6, warmup-discard via `--warmup 1`, no-partial-write on mid-run exception). Live MI300X numbers remain deferred.
  Source: DECISIONS.md/D007 + S07-SUMMARY.md (hermetic) + M001-VALIDATION.md/MV01
- **D008 — Local-first GPU-agnostic development through S04; ROCm only from S05+.** Validated. R013 grep gate confirms `rocm-smi`/`MI300X`/`HIP_VISIBLE_DEVICES` are confined to `contextforge/metrics/collector.py` and never appear in `agents/*.py`, `demo/benchmark.py`, or `demo/app.py`.
  Source: DECISIONS.md/D008 + M001-VALIDATION.md/MV04 R013 row
- **D009 — vLLM connection fail-fast: byte-identical preflight string in `main.py`, `pipeline.py`, `demo/benchmark.py`.** Extended in S05/S06/S07 to a single shared module constant `_VLLM_PREFLIGHT_MSG_TEMPLATE` exported from `agents/pipeline.py`; new entrypoints IMPORT it instead of redeclaring (MEM060). The container ENTRYPOINT preserves the contract at the docker boundary (S09).
  Source: DECISIONS.md/D009 + PROJECT.md/Architecture Patterns (MEM053)
- **D010 — Greenfield project; discard Apohara/CLAUDE.md prior context.** Validated. No Apohara coupling exists in any committed file; project shipped MIT-standalone.
  Source: DECISIONS.md/D010

## ### Lessons

- **`pydantic-settings` `extra="forbid"` is a footgun at module import.** It raises `ValidationError` on unrelated host env vars (SHELL, PATH, LANG, PYTEST_*). Use `extra="ignore"`; keep `case_sensitive=True`. Captured as MEM018.
  Source: PROJECT.md/Gotchas (MEM018) + S02-SUMMARY.md/Key Decisions
- **Base `bert-base-multilingual-cased` is the wrong LLMLingua-2 fallback.** It lacks the token-classification head LLMLingua-2 needs. Use the fine-tuned `meetingbank` variant. D004 was wrong until S04/T01 corrected it across config/env/tests.
  Source: PROJECT.md/Gotchas + S04-SUMMARY.md/Key Decisions
- **ROCm OOM appears as a bare `RuntimeError("HIP out of memory")` on PyTorch <2.4.** Modern builds raise `torch.OutOfMemoryError`. OOM-recovery code must check both classes — see `_is_oom` helper in `contextforge/compression/compressor.py`.
  Source: PROJECT.md/Gotchas (MEM035) + S04-SUMMARY.md/Key Decisions
- **`gr.Timer.tick` can race `demo.load(fn=_startup)`.** The Timer fires before the startup coroutine finishes constructing module-level clients. Add a defensive `if _app_mcp_client is None: return ('Initializing…', placeholder_figs...)` early-return to keep the output-tuple shape valid (MEM065).
  Source: PROJECT.md/Gotchas (MEM065) + S08 work captured in PROJECT.md
- **`Path(value).is_file()` overflows OS path limits on multi-thousand-char inline strings.** When a CLI arg can be either an inline string or a path, wrap `Path(value).is_file()` in `try/except OSError`. The default ≥600-token `SHARED_BASE` in `demo/benchmark.py` is ~3700 chars and trips this (MEM059).
  Source: PROJECT.md/Gotchas (MEM059)
- **Slice closers MUST re-run task-level grep gates from scratch.** Task-summary verification tables can silently omit one of multiple paths in a single grep. T03 listed an `asyncio.Lock` gate over BOTH `context_registry.py` AND `ttl_cache.py` but the summary only re-ran it against `context_registry.py`; a docstring mention in `ttl_cache.py:24` slipped through.
  Source: PROJECT.md/Gotchas (MEM025) + S03-SUMMARY.md/Key Decisions
- **httpx default connection pool (`max_keepalive_connections=20`, `max_connections=100`) breaks around ~50 concurrent agents driving the same host.** Does NOT apply at the 5-agent demo scale, but explicit pool sizing is required if scaling up (MEM052).
  Source: PROJECT.md/Architecture Patterns (MEM052)
- **`Embedder` import wall-clock is ~3.7s, dominated by `import torch` + `from sentence_transformers import SentenceTransformer`.** The lazy-load invariant holds (no model load at import) but the 0.5s import-time target floated in S03/T01 was unattainable given the dependency stack.
  Source: S03-SUMMARY.md/Known Limitations
- **`PASS-SUBSTITUTE` is the right call when a CI tool is genuinely absent.** When docker CLI was absent on the planning host, S09 substituted `python3 -c 'import yaml; yaml.safe_load(...)'` plus 9 grep contracts for `docker compose config -q`. Documented honestly as PASS-SUBSTITUTE; the live `docker compose up --build` step is enumerated in `space/README.md` §"Deferred verification".
  Source: M001-VALIDATION.md/MV01 + PROJECT.md/Architecture Patterns (MEM068)

## ### Patterns

- **Hermetic dependency-injection for ML wrappers.** Production class accepts the heavy collaborator (Embedder, dedup engine, registry, compressor, vram_source_func, `httpx.MockTransport`) as a constructor param defaulting to None; tests inject Fake* doubles. Saves real model-load time and keeps unit suites under 10s.
  Source: PROJECT.md/Architecture Patterns (MEM026) + S03-SUMMARY.md/Patterns established
- **Strict-`>` threshold semantics across modules.** Uniform across S03 (`TTLCache` expiry boundary) and S04 (coordinator thresholds at 200/500). Boundary tests pin the convention so future refactors cannot silently flip the comparison (MEM038).
  Source: S03-SUMMARY.md + S04-SUMMARY.md/Patterns established
- **Symmetric two-cache layout.** `TTLCache` is the single source of truth for liveness; auxiliary caches (embedding dict) are pure derived state and evict via key-diff against the primary, never via parallel TTL bookkeeping (MEM023).
  Source: PROJECT.md/Architecture Patterns (MEM023) + S03-SUMMARY.md/Patterns established
- **FastAPI lifespan-shared composition.** Single `@asynccontextmanager` constructs one each of registry/compressor/coordinator/metrics/vllm, attaches to `app.state`, exposes via `Depends()` wrappers; teardown awaits `vllm.aclose()` + `registry.clear()` in `finally` (MEM042).
  Source: PROJECT.md/Architecture Patterns (MEM042)
- **Constructor-injected agent clients.** `BaseAgent` takes `(name, system_prompt, mcp_client, vllm_client)` keyword-only — no module-level singletons, no parallel `ContextRegistry`/`Coordinator` inside agent code (MEM049).
  Source: PROJECT.md/Architecture Patterns (MEM049)
- **Caller-vs-pipeline ownership tracking.** `owns_X` booleans set BEFORE construction, with `try/finally` closing only what the function created; caller-injected clients survive the call (MEM048).
  Source: PROJECT.md/Architecture Patterns (MEM048)
- **`httpx.ASGITransport` for live FastAPI tests.** `httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")` instead of `TestClient` — no TCP, real request cycle. Fakes injected via `app.dependency_overrides` keyed by function identity, never strings (MEM041 + MEM047).
  Source: PROJECT.md/Architecture Patterns (MEM047)
- **Two-seam OOM fallback for ML wrappers.** Catch OOM at the load seam AND independently at the per-call seam; helper centralizes dual-class detection (`torch.cuda.OutOfMemoryError` + bare `RuntimeError("out of memory")`); always-CPU fallback (never retry on the same device that just OOMed); cache the swapped fallback so subsequent compresses skip reload.
  Source: S04-SUMMARY.md/Patterns established
- **D009 verbatim fail-fast across CLI surfaces.** A single shared module constant `_VLLM_PREFLIGHT_MSG_TEMPLATE` exported from `agents/pipeline.py`; `main.py`, `pipeline.py:cli()`, and `demo/benchmark.py:cli()` IMPORT it. `logger.warning(..., extra={component, base_url})` fires BEFORE the verbatim stderr print so log scrapers still see failures with `stderr=/dev/null`. Container ENTRYPOINT preserves the contract at the docker boundary.
  Source: PROJECT.md/Architecture Patterns (MEM053 + MEM060)
- **Single-asyncio.run CLI flow.** When a CLI does preflight then a main async phase that must reuse the same `httpx.AsyncClient`, use one `asyncio.run(_cli_async())` wrapping both, with `try/finally` aclose at the end. Two separate `asyncio.run` calls trip "event loop is closed" because the client binds to its construction loop (MEM058).
  Source: PROJECT.md/Architecture Patterns (MEM058)
- **Gradio single-loop discipline extends to ALL async clients.** `httpx.AsyncClient` (MCP) AND `VLLMClient` are module-level None at import; both constructed inside `demo.load(fn=_startup)` so they bind to Gradio's event loop. Import-time invariant `assert _app_mcp_client is None and _app_vllm_client is None` is a load-bearing test gate (MEM062).
  Source: PROJECT.md/Architecture Patterns (MEM062)
- **UI degrade-and-surface vs CLI fail-fast.** UI handlers catch `httpx.RequestError` / `httpx.HTTPStatusError` / `pydantic.ValidationError` and render an inline `gr.Markdown` banner with `_mcp_connect_url()` + exception class name. Stack traces NEVER reach the browser. Fail-fast (D009/MEM046) only applies to CLI startup paths (MEM064).
  Source: PROJECT.md/Architecture Patterns (MEM064)
- **Slice-close belt-and-suspenders.** Pair every bash grep gate (R013, MEM037 redaction sweep) with a pytest-form guard that reads the file as text and asserts presence/absence of the same substrings. CI-config drift can drop bash gates; pytest gates ride along with every test invocation (MEM066).
  Source: PROJECT.md/Architecture Patterns (MEM066)

## ### Surprises

- **The pi auto-mode artifact verifier and the gate's "do NOT call `gsd_complete_milestone`" instruction were structurally contradictory.** The verifier's "M001-SUMMARY.md must exist" check could not coexist with the gate's failure-path block while SC1/SC2/SC3 remained credentialed/hardware-bound. Resolution required a user override to choose Path A (placeholders) or Path B (re-scope). Eight retries did not break this loop.
  Source: M001-VERIFICATION-FAILURE.md (entire history)
- **D004 — the documented compressor decision — had the wrong fallback model.** Base `bert-base-multilingual-cased` (110M params; cited in the original decision row) lacks the token-classification head LLMLingua-2 needs to produce compression scores. Caught only at S04/T01 when the test stack tried to actually load it. Three sites had to be patched.
  Source: S04-SUMMARY.md/Key Decisions + PROJECT.md/Gotchas
- **The slice-close grep-gate gap (MEM025) only surfaced because the closer re-ran the literal task-plan verify command.** Trusting the task summary's verification table would have shipped a docstring `asyncio.Lock` mention in `ttl_cache.py:24` past the no-Lock invariant. Standing rule for all subsequent slice closes.
  Source: S03-SUMMARY.md/Patterns established
- **`Embedder` import takes ~3.7 wall-clock seconds.** The lazy-load contract (no model weights at import) holds, but the 0.5s import-time floated in S03/T01 was naive — `import torch` + `from sentence_transformers import SentenceTransformer` is the cost floor, and it dominates the budget.
  Source: S03-SUMMARY.md/Known Limitations
- **The hermetic-verification surface ended up nearly complete by S09 close.** 185 tests / 3 deselected smoke; 8/8 boundary contracts honored; 14/14 active+validated requirements covered. The only remaining gap is credentialed/hardware-bound (live MI300X allocation, HF auth token, Docker on a target host) — which a planning agent cannot autonomously produce. The substantive work shipped; the verification surface is honest about what it could and could not prove.
  Source: M001-VALIDATION.md (MV01-MV04 + Verification Class Compliance)
