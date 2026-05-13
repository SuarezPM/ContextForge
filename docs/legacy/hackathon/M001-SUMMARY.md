---
id: M001
title: "ContextForge — Working core, benchmark, and ship"
status: complete
completed_at: 2026-05-10T01:31:16.135Z
key_decisions:
  - D001 — Inference runtime: vLLM ROCm fork with --enable-prefix-caching (validated end-to-end through S09 docker-compose).
  - D002 — Primary inference model: Qwen/Qwen2.5-7B-Instruct (sized the docker-compose vllm healthcheck start_period: 120s × 12 retries for this exact model's cold-load characteristic).
  - D003 — Embedding model: sentence-transformers/all-MiniLM-L6-v2 (validated by S03 slice smoke similarity=0.9884).
  - D004 — Prompt compressor: microsoft/llmlingua-2-xlm-roberta-large-meetingbank with FINE-TUNED bert-base-multilingual-cased-meetingbank fallback (corrected mid-flight at S04/T01: base BERT lacks the LLMLingua-2 token-classification head).
  - D005 — MCP server framework: FastAPI + asyncio (vanilla-FastAPI fallback never had to fire).
  - D006 — 4-strategy compression coordinator: apc_reuse if sim≥0.85 AND shared_prefix>200; compress if sim<0.85 AND ctx>500; compress_and_reuse if sim≥0.85 AND ctx>500; passthrough otherwise. Strict-`>` thresholds.
  - D007 — Benchmark protocol: 1 warmup discarded, 2 measurement runs averaged, cold reported separately as cold_cache_baseline (validated hermetically; live MI300X delta_pct magnitude is the placeholder).
  - D008 — Local-first GPU-agnostic development through S04; ROCm only from S05+ when VLLM_BASE_URL points to AMD Cloud (validated by R013 grep gate confining rocm-smi/MI300X/HIP_VISIBLE_DEVICES to contextforge/metrics/collector.py).
  - D009 — vLLM connection fail-fast: byte-identical preflight string in main.py + pipeline.py + demo/benchmark.py via shared module constant _VLLM_PREFLIGHT_MSG_TEMPLATE imported (never redeclared).
  - D010 — Greenfield project; discard Apohara/CLAUDE.md prior context (validated; no Apohara coupling in any committed file).
key_files:
  - pyproject.toml
  - .env.example
  - Dockerfile
  - .dockerignore
  - docker-compose.yml
  - README.md
  - contextforge/__init__.py
  - contextforge/main.py
  - contextforge/config.py
  - contextforge/models.py
  - contextforge/registry/context_registry.py
  - contextforge/registry/ttl_cache.py
  - contextforge/dedup/dedup_engine.py
  - contextforge/dedup/embedder.py
  - contextforge/compression/compressor.py
  - contextforge/compression/coordinator.py
  - contextforge/serving/vllm_client.py
  - contextforge/mcp/server.py
  - contextforge/metrics/collector.py
  - agents/base_agent.py
  - agents/demo_agents.py
  - agents/pipeline.py
  - demo/benchmark.py
  - demo/app.py
  - space/README.md
  - docs/video_script.md
  - docs/slides.md
  - tests/test_models.py
  - tests/test_config.py
  - tests/test_registry.py
  - tests/test_dedup.py
  - tests/test_compressor.py
  - tests/test_coordinator.py
  - tests/test_vllm_client.py
  - tests/test_mcp_server.py
  - tests/test_metrics.py
  - tests/test_main.py
  - tests/test_base_agent.py
  - tests/test_demo_agents.py
  - tests/test_pipeline.py
  - tests/test_benchmark.py
  - tests/test_app.py
lessons_learned:
  - pydantic-settings extra='forbid' is a footgun at module import — it raises ValidationError on unrelated host env vars (SHELL, PATH, LANG, PYTEST_*). Use extra='ignore'; keep case_sensitive=True (MEM018).
  - Base bert-base-multilingual-cased lacks the LLMLingua-2 token-classification head. The fine-tuned meetingbank variant is required. D004 was wrong until S04/T01 corrected it across config.py, .env.example, and tests/test_config.py.
  - ROCm OOM appears as a bare RuntimeError('HIP out of memory') on PyTorch <2.4 builds; modern builds raise torch.OutOfMemoryError. OOM-recovery code must check both classes via a centralized _is_oom helper.
  - gr.Timer.tick can race demo.load(fn=_startup) — the Timer fires before the startup coroutine finishes constructing module-level clients. Add a defensive 'if _app_mcp_client is None: return placeholder_tuple' early-return in tick handlers (MEM065).
  - Path(value).is_file() raises OSError 'File name too long' when value is a multi-thousand-char inline string — wrap in try/except OSError when a CLI arg can be either an inline string or a path (MEM059).
  - Slice closers MUST re-run task-level grep gates from scratch — task-summary verification tables can silently omit one of multiple paths in a single grep (MEM025).
  - httpx default connection pool (max_keepalive_connections=20, max_connections=100) breaks around ~50 concurrent agents driving the same host — does NOT apply at the 5-agent demo scale, but explicit pool sizing is required for scale (MEM052).
  - Embedder import wall-clock is ~3.7s, dominated by import torch + from sentence_transformers import SentenceTransformer; the 0.5s import-time floated in S03/T01 was unattainable given the dependency stack.
  - PASS-SUBSTITUTE is the right call when a CI tool is genuinely absent: substitute structural validation (yaml.safe_load + grep contracts) for the missing tool, document honestly in verification evidence, and enumerate the live runtime step in the deploy artifact's 'Deferred verification' section (MEM068).
  - When a verifier's existence-check and a gate's failure-path block become structurally contradictory while gating items are credentialed/hardware-bound, no number of auto-fix retries will resolve it — the loop is structural, not transient. Resolution requires a human policy decision (Path A: execute live; Path B: re-scope; or Path A-with-placeholders as taken here).
---

# M001: ContextForge — Working core, benchmark, and ship

**Shipped the full M001 hermetic surface (S01–S09: scaffold → models → registry/dedup → compressor/coordinator → MCP server → 5-agent pipeline → cold/warm benchmark → 4-tab Gradio dashboard → docker-compose + HF Space artifact + README/video/slides) with 185 tests green; three success criteria (live MI300X benchmark numbers, HF Space URL, live `docker compose up --build`) ship as explicit `<TBD-after-live-deploy>` placeholders gated on AMD Cloud allocation, HF auth token, and a Docker-equipped MI300X target.**

## What Happened

M001 closed with placeholder-honest verification per user direction (Path A with placeholders).

**What shipped, end-to-end (hermetic).** Nine slices delivered the full code/artifact surface: S01 PEP 621 `pyproject.toml` with 13 pinned PART 5 deps + 19 zero-byte module placeholders + 16-key `.env.example` + SKELETON `Dockerfile`/`docker-compose.yml`; S02 five strict Pydantic v2 models (`ConfigDict(strict=True, extra="forbid")` with narrow Literal unions) + `Settings` instance; S03 `ContextRegistry`/`TTLCache`/`SemanticDedupEngine`/`Embedder` with the lock-free single-writer registry composing a `TTLCache[str, ContextEntry]` and a derived embedding dict; S04 `ContextCompressor` (lazy-loaded LLMLingua-2 with two-seam OOM fallback, dual-class `_is_oom`) + `CompressionCoordinator` (4-strategy router: `apc_reuse`/`compress`/`compress_and_reuse`/`passthrough`); S05 FastAPI MCP server with `@asynccontextmanager` lifespan-shared composition (one each of registry/compressor/coordinator/metrics/vllm) + 4 endpoints + `MetricsCollector` with rocm-smi → psutil tier with idempotent Degradation rows + D009 verbatim fail-fast in `main.py`; S06 `BaseAgent` with constructor-injected mcp/vllm clients + `AGENT_CONFIGS` (5 frozen entries with R008 overlap rates) + `run_pipeline` async API + D009 byte-identity in `pipeline.py:cli()`; S07 `demo/benchmark.py` CLI with cold/warm protocol (1 warmup discarded, 2 measurement runs averaged) + `benchmark_results.json` schema (cold/warm/off/cold_cache_baseline/totals/hardware/config) + dual D009/MCP fail-fast; S08 `demo/app.py` 4-tab Gradio Blocks (Live Demo / Real-time Metrics / Benchmark / Architecture) with single-event-loop client startup + Timer-tick race guard + UI-degrade-and-surface error rendering; S09 production `Dockerfile` (`rocm/dev-ubuntu-22.04:6.1-complete` base + ROCm 6.1 torch wheel + two-stage `pip install -e .` workaround) + `.dockerignore` + 3-service `docker-compose.yml` (vllm with `--enable-prefix-caching` + AMD device passthrough + healthcheck `start_period: 120s × 12 retries`, contextforge `depends_on: vllm: service_healthy`, gradio `depends_on: contextforge: service_healthy` with `CONTEXTFORGE_HOST: contextforge` for service-name DNS) + `space/README.md` HF Space artifact (frontmatter `sdk=docker app_port=7860 title=ContextForge` + Strategy B/A deploy procedures + `## Deferred verification (post-plan)` section) + `docs/video_script.md` (451 words, 5 H2 beats) + `docs/slides.md` (5 H1 marp-compatible slides) + surgical `README.md` delta.

**Verification posture.** Hermetic suite: 185 tests pass / 3 deselected smoke in 8.66s at HEAD `dfd24b2`. All 8 cross-slice boundary contracts in MV03 honored end-to-end (S05 FastAPI → S06 ASGI roundtrip → S07 benchmark CLI → S08 Tab 3 loads JSON; S04 coordinator → S05 `/tools/get_optimized_context` → S06 pipeline → S05 `/metrics/snapshot` → S08 Tab 2). All 14 active+validated requirements (R001–R014) covered with evidence in MV04. R013 GPU-agnostic constraint observable at the dependency-spec layer (`torch>=2.4,<2.6` generic; `VLLM_BASE_URL=http://localhost:8000/v1` default) and confirmed via grep gate confining `rocm-smi`/`MI300X`/`HIP_VISIBLE_DEVICES` to `contextforge/metrics/collector.py` only. R014 strict typing preserved across every public model added by every slice; zero TODOs in production code paths; absolute imports only.

**Placeholders for live deploy (the M001 closeout choice — Path A).** Three success criteria require credentialed/hardware-bound execution that the planning agent cannot autonomously produce: (SC1) live MI300X run producing real `benchmark_results.json` with measurable warm-run delta — README ships `_TBD_` cells under R012 footnote; (SC2) HF Space URL loading in a browser — `space/README.md` is committed but `git push` to `lablab-ai-amd-developer-hackathon` requires HF auth token; (SC3) `docker compose up --build` reaching all-services-healthy — compose validated structurally via `yaml.safe_load` + 9 grep contracts (PASS-SUBSTITUTE per MEM068) but the live up-and-healthy step requires a Docker-equipped MI300X target. Per user direction, M001 closes with these as explicit `<TBD-after-live-deploy>` placeholders rather than re-scoped success criteria. The placeholder slots and exact backfill procedure are enumerated in `space/README.md` §"Deferred verification (post-plan)" and in the README R012 footnote — they are pure data-injection points (no code changes), so the live deploy path is a single `git push` + `python demo/benchmark.py` + README backfill loop.

**Structural contradiction surfaced and resolved.** The pi auto-mode artifact verifier ("M001-SUMMARY.md must exist after the unit") and the gate's failure path ("Do NOT call `gsd_complete_milestone` while verification fails") were structurally contradictory while SC1/SC2/SC3 remained credentialed/hardware-bound. Eight retries reproduced the same deadlock without breaking it. User override (Path A with placeholders) is the resolution: the hermetic verification surface is treated as passed; the live items are committed-as-placeholder with clear backfill semantics. `M001-VERIFICATION-FAILURE.md` is preserved as the audit record of the eight-retry chain that surfaced this contradiction — it is not deleted.

**What unblocks the live close-out.** A single human-driven session: bring up the MI300X stack with `docker compose up --build`, confirm `docker compose ps` healthy, run `python demo/benchmark.py` against the live stack, commit the produced `benchmark_results.json`, backfill the README benchmark table from real numbers (no code change — schema validated end-to-end), `git push` to HF Spaces, capture the URL into README. No further code work is required; M001 is artifact-complete.

## Success Criteria Results

### SC1 — Live MI300X run produces `benchmark_results.json` with measurable warm-run delta (target ≥50% token reduction)

**Status:** PLACEHOLDER (hermetic surface complete; live numbers deferred)

- Hermetic: S07's 17 tests in `tests/test_benchmark.py` prove (a) full schema (cold/warm/off/totals/cold_cache_baseline/hardware/config), (b) run-count parity 6 (3 ON + 3 OFF), (c) cold/warm split via `--warmup 1` (cold len 1, warm len 2), (d) prompt-tokens delta sign (`tokens_with < tokens_without` on warm runs), (e) `delta_pct null-on-zero-without` sentinel, (f) no-partial-write on mid-run exception. Two `-m smoke` tests prove D009/MCP verbatim fail-fast.
- Placeholder: `find . -name benchmark_results.json -not -path './.venv/*' -not -path './node_modules/*'` returns empty. README ships `_TBD_` cells under the R012 footnote.
- Backfill: run `python demo/benchmark.py` against the live MI300X stack; commit the produced JSON; replace `_TBD_` cells from the JSON totals (`tokens_with`/`tokens_without`/`delta_pct`/`vram_with`/`vram_without`/`ttft_with`/`ttft_without`).

### SC2 — HF Space URL loads in a browser with all 4 tabs (Live Demo, Real-time Metrics, Benchmark, Architecture) rendering

**Status:** PLACEHOLDER (artifact + hermetic surface complete; live URL deferred)

- Hermetic: S08's 16 tests prove Blocks construction at import + all 4 tabs render + Plotly figures wire + UI banner-on-failure path. S09 produced `space/README.md` with mandatory frontmatter (`title=ContextForge sdk=docker app_port=7860`).
- Placeholder: no record of `git push` to the lablab-ai-amd-developer-hackathon HF org. README HF Space link cell shows `<TBD-after-S09-deploy>`.
- Backfill: `git push` the repo to HF Spaces under `lablab-ai-amd-developer-hackathon` with HF auth token; confirm Spaces UI reaches "Running"; replace the README `<TBD-after-S09-deploy>` cell with the captured URL.

### SC3 — `docker compose up --build` from a clean checkout brings vLLM + ContextForge + Gradio all healthy

**Status:** PLACEHOLDER (PASS-SUBSTITUTE on planning host; live up-and-healthy deferred)

- Hermetic: `docker-compose.yml` validated via `yaml.safe_load` + 9 grep contracts (`--enable-prefix-caching`, `/dev/kfd`, `/dev/dri`, `service_healthy`, `start_period 120s`, `healthcheck`, `env_file`, etc.). Documented as PASS-SUBSTITUTE per MEM068.
- Placeholder: docker CLI absent on planning host; live `docker compose up --build` reaching all-services-healthy with `curl :8001/health` returning `{"status":"ok","gpu":"MI300X"}` is the gating step.
- Backfill: on the MI300X target, run `docker compose up --build`; capture `docker compose ps` showing all three services healthy; record the `curl :8001/health` response.

### SC4 — README, 3-min video script, and 5-slide deck committed in the repo

**Status:** MET

- `README.md` shipped with architecture diagram + Quick Start (`docker compose up --build`) + benchmark table (with R012 _TBD_ footnote) + Video/Slides links + 185 hermetic tests badge.
- `docs/video_script.md` shipped: 451 words at ~150 wpm = 3 min spoken; exactly 5 H2 beats (Problem / Flow / Dashboard / Benchmark / Why MI300X).
- `docs/slides.md` shipped: marp-compatible 5-slide deck with exactly 5 H1 headings (Problem / Solution / Architecture / Demo / Numbers).

### SC5 — All hackathon submission requirements (PART 11) verified before 2026-05-10 16:00 Uruguay

**Status:** PARTIALLY MET (artifact-side complete; live items placeholdered)

- Artifact-side: hermetic structural validation complete — 7 slice-close gates pass at S09 (R013 grep, MEM037 redaction sweep, no SKELETON banners, compose YAML structural validity PASS-SUBSTITUTE, HF Space frontmatter, README delta, 185 hermetic tests).
- Placeholder: recorded video upload + URL injection into README, live MI300X benchmark numbers, and HF Space `git push` are deferred; SC5 closes only after SC1/SC2/SC3 backfill above.

## Definition of Done Results

- **All slices marked `[x]`:** YES — S01–S09 all `[x]` in `M001-ROADMAP.md`.
- **All slice SUMMARY.md files exist:** YES — every slice S01–S09 has `S0X-SUMMARY.md`. S09 records `verification_result: passed` in its frontmatter.
- **Cross-slice integration works:** YES — MV03 in `M001-VALIDATION.md` traces all 8 boundary-map contracts as PASS with concrete consumer evidence (S05 FastAPI → S06 ASGI roundtrip → S07 benchmark JSON → S08 Tab 3 load; S04 coordinator → S05 `/tools/get_optimized_context` → S06 pipeline → S05 `/metrics/snapshot` → S08 Tab 2).
- **Hermetic test suite green:** YES — 185 passed / 3 deselected smoke in 8.66s at HEAD `dfd24b2` (S09 close).
- **Code change verification:** PASS — 32 commits on main since milestone start; 23 contextforge module files, 4 agents files, 3 demo files, 16 test files, plus `Dockerfile`/`docker-compose.yml`/`README.md`/`docs/`/`space/`. Branch diff vs `main` merge-base lists implementation files (non-`.gsd/`).
- **No SKELETON banners remain:** YES — `Dockerfile` and `docker-compose.yml` no longer carry the SKELETON banner; `head -1 Dockerfile | grep -q SKELETON` returns no match.
- **Redaction discipline preserved:** YES — MEM037 grep sweep across `demo/`/`contextforge/`/`agents/` shows no PII in logger calls; `coordinator.py` has zero log statements; `compressor.py` log extras restricted to model paths and phase enums.
- **R013 grep gate green:** YES — `rocm-smi`/`MI300X`/`HIP_VISIBLE_DEVICES` confined to `contextforge/metrics/collector.py`; absent from `agents/*.py`, `demo/benchmark.py`, `demo/app.py`.
- **HF Space frontmatter parses:** YES — `space/README.md` frontmatter has the mandatory keys (`title`, `sdk: docker`, `app_port: 7860`).
- **All deferred items explicitly enumerated:** YES — `space/README.md` §"Deferred verification (post-plan)" lists the credentialed/hardware-bound steps; README R012 footnote points to the same backfill cells.

## Requirement Outcomes

**Summary of transitions persisted via `gsd_requirement_update` after this milestone:**

| ID | Class | Before | After | Evidence |
|---|---|---|---|---|
| R001 | core-capability | active | validated | S05 FastAPI MCP server with 4 endpoints + 13 TestClient tests; S06 lifted to live ASGI roundtrip via `httpx.ASGITransport` (MV04). |
| R002 | core-capability | active | validated | S03 slice smoke similarity=0.9884 + shared_prefix_len=209 on two ~1.5KB contexts; 17 dedup + 20 registry tests; KV-cache reuse decision wired in S04 coordinator (MV04). |
| R003 | core-capability | validated | (unchanged) | Already validated at S03 close. |
| R004 | core-capability | validated | (unchanged) | Already validated at S04/T01 close. |
| R005 | core-capability | validated | (unchanged) | Already validated at S04/T02 close. |
| R006 | integration | active | validated | S05 main.py preflight; S06 pipeline:cli() byte-identical mirror; S07 extracted shared `_VLLM_PREFLIGHT_MSG_TEMPLATE` + `_MCP_PREFLIGHT_MSG_TEMPLATE`. Subprocess smoke proved rc=1 + exact byte match (MV04). |
| R007 | failure-visibility | active | validated | S05: MetricsCollector with `vram_source` Literal pinning + idempotent Degradation row; 14 hermetic tests. Real rocm-smi invocation deferred to live MI300X (placeholder in MV01). |
| R008 | core-capability | active | validated | S06: AgentConfig (frozen) + AGENT_CONFIGS pinning retriever 0.6 / reranker 0.7 / summarizer 0.6 / critic 0.5 / responder 0.4 in canonical order (MV04). |
| R009 | core-capability | active | validated (placeholder) | S07: 17 hermetic tests prove schema, 6-run parity, `--warmup 1` discard, prompt-tokens delta sign, null-on-zero sentinel. Live MI300X delta_pct magnitude is the README backfill placeholder (SC1 above). |
| R010 | launchability | active | validated (placeholder) | S08: 4 tabs with real wiring; 16 hermetic tests; Blocks construction smoke. S09 `space/README.md` artifact (sdk=docker, app_port=7860). Actual `git push` is the SC2 placeholder. |
| R011 | launchability | active | validated | S09: Dockerfile + docker-compose.yml (3 services, --enable-prefix-caching, AMD device passthrough, healthchecks, depends_on:service_healthy). PASS-SUBSTITUTE yaml.safe_load + 9 grep contracts; live up --build is the SC3 placeholder. |
| R012 | launchability | active | validated | S09: README delta + `docs/video_script.md` + `docs/slides.md`. README `_TBD_` benchmark cells are the SC1/SC5 backfill placeholders, not gaps in R012 itself. |
| R013 | constraint | active | validated | S01 dependency-spec proof + S05+ grep gate confines `rocm-smi`/`MI300X`/`HIP_VISIBLE_DEVICES` to `contextforge/metrics/collector.py` only (MV04). |
| R014 | quality-attribute | active | validated | S02 5 public models with `ConfigDict(strict=True, extra="forbid")`; preserved across S03–S09. 185-test hermetic suite green. Zero TODOs in production code paths; absolute imports only (MV04). |

R015/R016/R017 remain **out-of-scope** (anti-features) — unchanged.

Coverage outcome: 11 active requirements transition to **validated** at M001 close (R009/R010 are validated-with-explicit-placeholder for live numbers/URL backfill). 3 already-validated rows (R003/R004/R005) unchanged. Out-of-scope set unchanged.

## Deviations

"Closeout was done with explicit `<TBD-after-live-deploy>` placeholders for SC1/SC2/SC3 rather than either (a) executing the live deploy steps inside this session or (b) re-scoping the success criteria. This deviates from the auto-mode verification gate's strict reading ('Do NOT call gsd_complete_milestone while verification fails') under user direction (Path A with placeholders). Rationale: the hermetic verification surface is complete and honest; the live items are credentialed/hardware-bound and a planning agent cannot autonomously produce them; the placeholder cells in README and `space/README.md` §'Deferred verification (post-plan)' are pure data-injection points so the backfill path is friction-free. `M001-VERIFICATION-FAILURE.md` is preserved as the audit record of the eight-retry chain that surfaced the structural contradiction between the artifact verifier and the gate's failure path."

## Follow-ups

"### Live deploy backfill (post-M001, human-driven)

Execute in order on a Docker-equipped MI300X target with HF auth token in env:

1. `docker compose up --build` — confirm `docker compose ps` shows vllm/contextforge/gradio all healthy; capture `curl :8001/health` output (closes SC3 placeholder).
2. `python demo/benchmark.py` — produces real `benchmark_results.json` with measurable warm-run delta. Commit the JSON (closes SC1 placeholder).
3. Backfill README benchmark table from the JSON `totals` (replace each `_TBD_` cell with `tokens_with`/`tokens_without`/`delta_pct`/`vram_with`/`vram_without`/`ttft_with`/`ttft_without` — pure data injection, no code change).
4. `git push` the repo to HF Spaces under `lablab-ai-amd-developer-hackathon`; confirm the Spaces UI reaches 'Running'; replace the README `<TBD-after-S09-deploy>` cell with the captured URL (closes SC2 placeholder).
5. Record the 3-min video using `docs/video_script.md`; upload; inject the URL into README (closes SC5 video item).
6. Re-run M001 verification — at this point the milestone closes truly end-to-end and `M001-VERIFICATION-FAILURE.md` can be archived as resolved.

### Open items unrelated to live deploy

- M002 candidate scope (post-hackathon): explicit httpx pool sizing if the demo is taken to >50 concurrent agents (MEM052). Not in M001 scope.
- M002 candidate scope: real MCP SDK adoption if the official SDK matures and offers concrete value over vanilla FastAPI (D005 fallback was never needed; reconsider when SDK ergonomics improve)."
