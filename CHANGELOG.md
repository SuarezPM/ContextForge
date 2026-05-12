# Changelog

## V6.1.0 — Truth-Up Release · 2026-05-12

Public accountability layer + the surgical fixes that close every
overclaim documented in [AUDIT.md](AUDIT.md). No new mechanisms shipped
in this release on purpose: V6.1 is exclusively about making the
numbers we report match what the code actually computes.

### Fixed

- **`metrics/collector.py`** — removed the `--showgpu占用率 --json`
  rocm-smi flag (Chinese-character LLM mistranslation that failed on
  every ROCm install) and the `return 45.0, 192.0` hardcoded fallback.
  VRAM readings now delegate to `metrics/vram_monitor.py` (pyrsmi →
  /sys/class/drm → 192 GB synthetic), and `MetricsSnapshot.vram_source`
  reports the *actual* backend (`pyrsmi`, `sysfs-drm`, or
  `synthetic-dev`) rather than always claiming "rocm-smi".
- **`decoding/speculative_coordinator.py:verify_and_commit`** —
  accepts an optional `draft_logprobs` argument. When supplied, q_i is
  the draft model's real per-token probability and the
  Leviathan 2023 acceptance criterion min(1, p_i / q_i) is applied
  verbatim, preserving INV-12 (target distribution unchanged). The
  legacy fallback path (no draft_logprobs) still works but emits a
  `WARNING` that INV-12 is no longer guaranteed.
- **`demo/benchmark_v5.py`** — replaced five hardcoded `duration_ms`
  values (250.0, 150.0, 100.0, 120.0, 5.0) with `time.perf_counter()`
  measurements wrapped around each scenario body.
- **`demo/benchmark_v5.py` (S-11)** — `else` branch no longer
  hardcodes `deviation_pct = 0.0`. Reports the real difference
  between observed and predicted λ_critical in both branches. Arrival
  rates ramped from `[0.5..2.5]` to `[1.0..12.0]` so the controller
  is actually exercised.
- **`demo/benchmark_v5.py` (S-15)** — replaced 9 hand-picked cases
  with an exhaustive Cartesian sweep: 5 roles × 11 candidate counts
  × 11 reuse rates × 2 shuffle flags = **1,210 decisions** per run.
  Violation predicate now checks both implication directions (judge
  with risk > τ ⇒ dense AND non-judge ⇒ not dense).

### Added

- **`AUDIT.md`** at repo root — every V6.0 overclaim listed with
  file:line evidence and the V6.1 fix that closes it.
- **`scripts/check_honesty.sh`** + **`.github/workflows/honesty.yml`**
  — CI guard that fails the build if any of the V6.0 lies regress
  (hardcoded `duration_ms`, the Chinese rocm-smi flag, the fabricated
  `draft_prob_estimate` without the INV-12 warning, the `45.0, 192.0`
  hardcoded tuple).

### Benchmark profile change

V6.0 reported **15/15 PASS**. V6.1 reports **14/15 PASS with one
honest fail**: S-11 (queueing controller stability) now shows a real
deviation of ~100% under the new arrival-rate ramp because the M/G/1
model overestimates capacity under bursty load with a draining block
budget. **This failure is the intended outcome of the truth-up.** The
controller's math is correct (the module is 🟢 PRODUCTION in the
audit); the benchmark scenario is too bursty for the current
estimator. Tuning the controller's burst-handling, or replacing the
synthetic arrival process with a less bursty one, is tracked under
V6.2.

### Compatibility

- `MetricsCollector._use_rocm` and `MetricsCollector._check_rocm()` are
  retained as backwards-compat shims. All 13 `test_mcp_server.py` and
  all 21 `test_speculative_coordinator.py` tests continue to pass.
- `verify_and_commit(target_verification_logprobs, draft_tokens)` keeps
  the legacy two-argument call signature — the new `draft_logprobs`
  parameter is keyword-only and defaults to `None`.

### Citation

The Zenodo deposit for V6.1 is the next version under the same concept
DOI: [10.5281/zenodo.20114594](https://doi.org/10.5281/zenodo.20114594).
The V6.0 deposit remains accessible but is now superseded.

---

## V6.0.0 — Initial hackathon submission · 2026-05-10

Initial release for the AMD AI Hackathon. 10 mechanisms wired,
15/15 benchmark PASS (subsequently audited and partly revised in V6.1
— see AUDIT.md), 310/310 unit tests passing, INV-15 enforcement,
TokenDance master-mirror storage, JCR Safety Gate, AITER ROCm config,
paper deposited at Zenodo with DOI 10.5281/zenodo.20114594.
