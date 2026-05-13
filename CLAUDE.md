# CLAUDE.md — Apohara ContextForge

> Project-level engineering contract. Extends `~/.claude/CLAUDE.md`.
> Living document. Co-evolves with use. Edit in place. Never rewrite.

---

## 0. First Action on Every Session

Before any other work, call:

```
mem_context(project='Apohara_Context_Forge')
```

This recovers the session history persisted via engram and prevents
cold-start blindness. If you cannot find `mem_context` in your tools,
load it via `ToolSearch(query="select:mcp__plugin_engram_engram__mem_context")`.

If the user asks you to "act as if continuing the prior session", this
call is mandatory, not optional.

---

## 1. Project Overview

**Apohara ContextForge** is an open-source KV-cache coordination layer
for multi-agent LLM pipelines, with a formal safety invariant
(`INV-15`) for judge agents. Hardware-validated on AMD Instinct MI300X
(192 GB HBM3, ROCm 7.2.0). Solo developer (Pablo M. Suarez, UNT
Argentina). License: **Apache-2.0**. V7.0.0-rc.2 shipped 2026-05-13.

Repo: <https://github.com/SuarezPM/Apohara_Context_Forge>
Paper: `paper/inv15_paper.pdf` (v2.0.1, 12 references, MI300X-grounded)
Zenodo DOI: `10.5281/zenodo.20114594`

---

## 2. Memory Hierarchy

Precedence from most general to most specific:

1. `~/.claude/CLAUDE.md` — Global defaults. Behavioral floor.
2. **This file** — Apohara-specific rules. Extends global.
3. `CLAUDE.local.md` — Maintainer private notes. Gitignored.
4. `<subdir>/CLAUDE.md` — Subsystem-scoped rules (none today).

On conflict, more specific wins. This file **overrides** any Bun /
TypeScript / Node guidance in the global file — Apohara is a Python
project. See §4 for the real stack.

---

## 3. Behavioral Guardrails (Karpathy 4 — non-negotiable)

### 3.1 Think Before Coding
State assumptions explicitly. If multiple interpretations exist,
present all. If a simpler approach exists, say so first. If
unclear, STOP and name what is confusing. Do not pick silently.

### 3.2 Simplicity First
No features beyond what was asked. No abstractions for single-use
code. No flexibility that was not requested. If 200 lines could be
50, rewrite it.

### 3.3 Surgical Changes
Touch only what the task requires. Do not improve adjacent code.
Match existing style even if you disagree. Notice dead code, mention
it, do not delete it. Every changed line must trace to the request.

### 3.4 Goal-Driven Execution
Transform tasks into verifiable goals. "Add validation" → "Write
failing tests, then pass". "Fix bug" → "Reproduce in test, then
fix". For multi-step tasks, state plan as `1. Step → verify: check`.

---

## 4. Tech Stack & Commands

### Stack
- **Python 3.11+** (3.14 is dev default; 3.12-3.14 tested)
- PyTorch 2.5.1 + ROCm 6.2 (NOT CUDA paths; AMD MI300X target)
- vLLM 0.17.1 via `vllm.general_plugins` entry-point
- LMCache with `non_cuda_equivalents` fallback for AMD
- NumPy 2.x, pytest 9.0.3, pytest-asyncio, pytest-json-report
- Go 1.22+ for the K8s operator (`operator/`)
- LaTeX via tectonic 0.15+ for the paper
- Helm 3.x for cluster deployment

### Commands
```bash
# Full regression (~200s, 350+ tests):
PYTHONPATH=. python3 -m pytest tests/ -q

# Quick subset:
PYTHONPATH=. python3 -m pytest tests/test_fwht.py tests/test_rotate_kv.py -v

# Honesty CI guard (must pass before any commit to main):
bash scripts/check_honesty.sh

# Operator manifest validation:
bash operator/validate.sh

# Compile paper to PDF:
tectonic paper/inv15_paper.tex

# Push directly to main (current workflow, post-detached-HEAD):
git push origin HEAD:main

# MI300X measurement (requires AMD AI Dev Cloud droplet):
PYTHONPATH=. python3 scripts/mi300x_vram_measurement.py
```

---

## 5. Architecture Map

```
apohara_context_forge/    # Core library
  quantization/           # FWHT + RotateKV INT4 codec
  safety/                 # JCRSafetyGate (INV-15 implementation)
  serving/                # vLLM ATOM plugin + LMCacheConnectorV2
  observability/          # Prometheus + JSONL audit log + OTLP
  registry/               # ContextRegistry (LSH + FAISS + VRAMCache)
  decoding/               # Speculative coordinator
  compression/            # LLMLingua wrapper
  scheduling/             # M/G/1 queueing controller
  metrics/                # VRAM monitor (pyrsmi → /sys/class/drm)
  mcp/                    # FastAPI MCP server
  dedup/                  # LSH token matching + FAISS context index

agents/                   # 5-agent demo pipeline
tests/                    # 350+ tests, pytest
demo/                     # Gradio dashboard + benchmark scenarios
paper/                    # LaTeX source + figures + references.bib + PDF
operator/                 # Go K8s operator + CRD
charts/apohara-contextforge/  # Helm chart
hf_spaces/                # HuggingFace Space shim
pypi/apohara-vllm-plugin/ # PyPI plugin package
scripts/                  # MI300X measurement + honesty CI guard
logs/                     # Committed JSON logs (evidence layer)
dashboards/inv15.json     # Grafana dashboard
docs/legacy/hackathon/    # Pre-rename ForgeContextCC artifacts
```

---

## 6. Honesty Discipline (V6.1+, Apohara-specific)

State matches reported. Every claim in code, docstring, README,
AUDIT.md must trace to what code actually does at runtime.

- **`AUDIT.md`** is the public accountability layer. Every V6.0
  overclaim listed with file:line evidence + tracked through fix.
  10 items closed as of V7.0.0-rc.2.
- **`scripts/check_honesty.sh`** runs in CI. Catches hardcoded
  `duration_ms = N`, the rocm-smi Chinese-character flag,
  fabricated `draft_prob_estimate`, hardcoded VRAM tuples, etc.
- **`"rocm-hip:..."`** is the backend label on AMD, NOT `"cuda"`.
  PyTorch ROCm reuses the `torch.cuda.*` API for backward-compat;
  this is misleading and must be corrected in any new logging.
- **Measured numbers come from `logs/*.json`** committed to repo.
  No hardcoded benchmark values in production code paths.

---

## 7. Safety Rules (hard limits, non-negotiable)

- Never commit secrets (`.env`, API keys, tokens, credentials).
- Never push to main with hardcoded benchmark numbers — the honesty
  guard catches it but spend the cycle locally first.
- Never claim a measurement without a log file in `logs/`.
- Never bypass `PYTHONPATH=. pytest tests/` before declaring done.
- Never fabricate citations. All BibTeX entries must trace to an
  arXiv abstract page or a published DOI.
- License is **Apache-2.0** (LICENSE file). Never reference MIT
  anywhere in docs, metadata, or comments.
- Never edit between `<!-- gitnexus:* -->` markers manually
  (gitnexus tooling owns those).
- Never edit between `<!-- lean-ctx -->` markers manually
  (lean-ctx tooling owns those).
- Hardware label is `"rocm-hip:6.2.41133:AMD Instinct MI300X VF"`
  on AMD, not `"cuda"`.

---

## 8. Verification Protocol

Before declaring any task done, run this checklist:

- [ ] Success criteria from §3.4 met and observable
- [ ] `PYTHONPATH=. python3 -m pytest tests/ -q` passes
- [ ] `bash scripts/check_honesty.sh` passes
- [ ] `bash operator/validate.sh` if operator/ touched
- [ ] Diff matches scope. No drive-by changes.
- [ ] AUDIT.md updated if a module status changes (🟢/🟡/🟠/🔴)
- [ ] CHANGELOG.md entry for any release-relevant change
- [ ] No secrets, debug prints, or commented dead code added
- [ ] Sources cited if external claims made
- [ ] Commit signed-off (`git commit -s`) and pushed via
      `git push origin HEAD:main` (current detached-HEAD workflow)

---

## 9. Spec-Driven Workflow (V7+ pattern)

Non-trivial changes follow this loop:

1. **Plan** saved to `~/Documentos/Apohara_PRIVATE/plans/` (private,
   not in repo). The `.omc/plans/` path is gitignored; do NOT commit
   plans to public repo.
2. **Implement** in surgical commits with `git commit -s`.
3. **AUDIT.md update** if a module status changes.
4. **CHANGELOG.md entry** for any release-relevant change.
5. **Tests added/updated** for new code paths.
6. **Honesty check passes** before push.
7. **Push directly to main** via `git push origin HEAD:main` (no
   PRs needed; merged-banner workflow was retired May 13, 2026).

---

## 10. Co-evolution Rules

This file is living configuration.

- If an instruction repeats more than twice in chat, promote it here.
- Edit in place. Never rewrite from scratch.
- Auto-managed blocks have markers; do not edit between them
  manually. Currently no auto-managed blocks in this file (they
  live in the global `~/.claude/CLAUDE.md`).
- Review monthly. Prune dead rules.
- When a rule misfires, refine it. Do not delete blindly.

---

## 11. Open Items

- **Global `~/.claude/CLAUDE.md` has Bun-first stack content** that
  contradicts §4 of this file. The global may also affect other
  user projects; cleanup of the global is a manual decision (see
  the May 13 conversation context in engram).
- **MI300X access is gated on AMD credits.** Sprint 5+ GPU work
  blocked unless fresh credits arrive via DevRel outreach or
  out-of-pocket spend ($1.99/GPU/hr on AMD AI Dev Cloud).
- **`mem_context` auto-load** is currently a manual directive (§0).
  A proper `SessionStart` hook in `~/.claude/settings.json` would
  be more robust but is not yet configured.
