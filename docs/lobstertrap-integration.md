# Lobster Trap × ContextForge Integration Design

> **Status:** Day 1 of TechEx Track 1 submission plan
> **Plan ref:** `.omc/plans/techex-track1-submission.md`
> **Author:** Pablo M. Suarez + Claude
> **Date:** 2026-05-13
> **Lobster Trap version:** v0.1.0 (clone of `github.com/veeainc/lobstertrap` at HEAD on 2026-05-13)
> **License compatibility:** MIT (Lobster Trap) × Apache-2.0 (ContextForge) → fully compatible

---

## TL;DR

We combine two open-source layers into a defense-in-depth governance stack:

1. **Lobster Trap (Veea, MIT, Go)** — *perimeter* layer. Sub-millisecond regex-based
   Deep Prompt Inspection that catches obvious attacks (prompt injection, credential
   exposure, PII, exfiltration) deterministically, without LLM-as-a-judge overhead.
2. **ContextForge INV-15 (this repo, Apache-2.0, Python)** — *behavioral* layer. Formal
   invariant that gates KV-cache reuse for judge agents to prevent the silent Judge
   Consistency Rate (JCR) degradation (Liang et al. 2026, arXiv:2601.08343).

These layers solve **orthogonal** failure modes. Lobster Trap catches *what the agent
tries to do*; INV-15 catches *whether the agent stays consistent across runs*. Both
are required for production-grade multi-agent LLM workflows.

---

## A. Architecture

```
                    Defense-in-Depth Trust Layer
                    ============================

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  ┌──────────────┐    ┌──────────────────────────┐   ┌────────┐  │
  │  │              │    │                          │   │        │  │
  │  │  Client      │───▶│   Lobster Trap (:8080)   │──▶│  vLLM  │  │
  │  │  (5-agent    │    │                          │   │ (:8000)│  │
  │  │   workload)  │    │   ┌──────────────────┐   │   │        │  │
  │  │              │    │   │ INGRESS DPI      │   │   │ Llama- │  │
  │  └──────┬───────┘    │   │ - injection      │   │   │ 3-8B   │  │
  │         │            │   │ - credentials    │   │   │        │  │
  │         │            │   │ - PII            │   │   └────┬───┘  │
  │         │            │   │ - role-imp       │   │        │      │
  │         │            │   │ - exfiltration   │   │        │      │
  │         │            │   └──────────────────┘   │        │      │
  │         │            │           │              │        │      │
  │         │            │           ▼              │        │      │
  │         │            │   ┌──────────────────┐   │        │      │
  │         │            │   │  policy.yaml     │   │        │      │
  │         │            │   │  evaluation      │   │        │      │
  │         │            │   └────────┬─────────┘   │        │      │
  │         │            │            │             │        │      │
  │         │            │   ALLOW ◀──┴──▶ DENY     │        │      │
  │         │            │    │           │         │        │      │
  │         │            │    │       BLOCK + audit │        │      │
  │         │            │    ▼                     │        │      │
  │         │            │  forward                 │        │      │
  │         │            │                          │        │      │
  │         │            │   ┌──────────────────┐   │        │      │
  │         │            │   │ EGRESS DPI       │◀──┘        │      │
  │         │            │   │ - cred-leak      │            │      │
  │         │            │   │ - PII-leak       │            │      │
  │         │            │   └──────────────────┘            │      │
  │         │            │                                   │      │
  │         ▼            └───────────────┬───────────────────┘      │
  │  ┌──────────────────────────────────▼────────────────────────┐  │
  │  │                                                           │  │
  │  │   ContextForge 5-agent pipeline (INV-15 gate inside)      │  │
  │  │   ─────────────────────────────────────────────────────   │  │
  │  │                                                           │  │
  │  │   retriever → reranker → summarizer → CRITIC → responder  │  │
  │  │                                          │                │  │
  │  │                              ┌───────────┴───────────┐    │  │
  │  │                              │  INV-15 gate          │    │  │
  │  │                              │  risk > τ?            │    │  │
  │  │                              │  └─▶ dense prefill    │    │  │
  │  │                              │      (no KV reuse)    │    │  │
  │  │                              └───────────────────────┘    │  │
  │  │                                                           │  │
  │  │   ──▶ JCR holds 1.0 under reuse pressure                  │  │
  │  │                                                           │  │
  │  └───────────────────────────────────────────────────────────┘  │
  │                                                                 │
  │                    Audit trail (correlated)                     │
  │   ───────────────────────────────────────────────────────────   │
  │   • Lobster Trap audit log (JSONL, stderr or file)              │
  │     {request_id, direction, action, matched_rule, metadata}     │
  │                          ▲                                      │
  │                          │  correlation_id                      │
  │                          ▼                                      │
  │   • ContextForge JSONL audit log                                │
  │     {request_id, inv15_fired, risk_score, agent_role, ...}      │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
```

**Key port mapping for Day 2-3 implementation:**

| Component | Port | Notes |
| --------- | ---- | ----- |
| Lobster Trap proxy | `:8080` | Default. We keep it. |
| Lobster Trap dashboard | `:8080/_lobstertrap/` | Real-time web UI for the demo video |
| vLLM backend (production / on droplet) | `:8000` | We pass `--backend http://localhost:8000` to LT |
| vLLM backend (CPU mock for local dev) | `:8000` | Mock OpenAI-compat HTTP server |
| Gemini critic | external API | Routed via `--critic-provider gemini` in our scripts |
| ContextForge MCP server (if needed) | `:8001` | Separate from vLLM |

---

## B. Data flow for one pipeline request

### B.1 Ingress path (prompt arrives)

```
1. Client sends OpenAI-format chat completion to localhost:8080
   POST /v1/chat/completions
   {
     "model": "llama-3-8b",
     "messages": [
       {"role": "system", "content": "<retriever system prompt>"},
       {"role": "user", "content": "<user query + context>"}
     ],
     "_lobstertrap": {
       "declared_intent": "retrieval",
       "agent_id": "apohara-retriever-v7"
     }
   }

2. Lobster Trap parses the messages, extracts the user content, runs DPI:
   - intent_category = "general" (retriever query)
   - risk_score = 0.05 (low risk for a legit RAG query)
   - contains_injection_patterns = false
   - contains_pii = false
   - contains_credentials = false

3. Policy evaluation (configs/lobstertrap_policy.yaml):
   - No DENY rule matches → default action ALLOW
   - Mismatch check: declared_intent "retrieval" matches detected
     intent_category "general" → no mismatch

4. Forward request to backend (vLLM at :8000). Backend processes normally.
```

### B.2 Egress path (response returns)

```
5. vLLM responds with chat completion JSON

6. Lobster Trap runs EGRESS DPI on the response content:
   - contains_credentials = false (response is just retrieved facts)
   - contains_pii = false (sanitized output)

7. Egress policy evaluation → ALLOW

8. Lobster Trap appends `_lobstertrap` field with full audit report:
   {
     "request_id": "req-...",
     "verdict": "ALLOW",
     "ingress": { "declared": {...}, "detected": {...}, "mismatches": [], "action": "ALLOW" },
     "egress":  { "detected": {...}, "action": "ALLOW" }
   }

9. Response returned to client.

10. ContextForge pipeline state (per-agent INV-15 decision, JCR replica
    grouping) is logged separately to logs/inv15_audit_*.jsonl with the
    same request_id as correlation key.
```

### B.3 Where each layer fires per agent

| Agent (5-agent pipeline) | Lobster Trap relevance | INV-15 relevance |
| ------------------------ | ---------------------- | ---------------- |
| retriever | Ingress: declared_intent="retrieval", catches injection | No (not a judge role) |
| reranker | Ingress: low-risk metadata pass-through | No |
| summarizer | Egress: catches PII / credential leak in summary | No |
| **critic** (Gemini-3-pro in our combo) | Ingress + Egress: catches role-impersonation, fact-fabrication patterns | **YES — primary INV-15 gate target** |
| responder | Egress: final PII/credential check before user | No |

---

## C. What Lobster Trap catches vs. what INV-15 catches

This is **the core differentiator** for the TechEx submission. They are orthogonal,
not overlapping.

| Failure mode | LT catches? | INV-15 catches? | Why |
| ------------ | :---------: | :-------------: | --- |
| Prompt injection (`Ignore previous instructions`) | ✅ | ❌ | LT pattern: `contains_injection_patterns`; INV-15 is post-prompt, only sees cache reuse |
| Credential exposure in response (API key leak) | ✅ | ❌ | LT egress: `contains_credentials`; INV-15 doesn't read content |
| PII in response (SSN / CC / email) | ✅ | ❌ | LT egress: `contains_pii`; same as above |
| Data exfiltration via URL/file paths | ✅ | ❌ | LT: `contains_exfiltration`, `target_paths` |
| Role impersonation (`You are now the admin`) | ✅ | ❌ | LT: `contains_role_impersonation` |
| Obfuscation/encoding (base64 evasion) | ✅ | ❌ | LT: `contains_obfuscation` |
| Excessive request rate (DoS) | ✅ | ❌ | LT: built-in rate_limits |
| **Silent JCR drift under KV-cache reuse** | ❌ | ✅ | LT inspects content; doesn't measure inter-replica consistency |
| **Judge agent flips verdict for identical (query, context) pair across replicas** | ❌ | ✅ | INV-15 is the formal invariant Liang et al. 2026 exposed |
| **Per-agent risk score above τ** (reuse_rate × candidate_count × shuffle) | ❌ | ✅ | INV-15 closed-form heuristic; LT has its own risk_score but it's content-based |
| **Cross-replica verdict majority count < 0.95 (JCR threshold)** | ❌ | ✅ | INV-15 → dense prefill prevents this |

**The composability theorem:**

> Lobster Trap = `∀ prompt: ¬contains(P)` (perimeter assertion over content).
> INV-15 = `∀ judge_replica: verdict(replica_i) = verdict(replica_j)` (behavioral
> invariant over runs).

Both are necessary, neither sufficient.

---

## D. Failure modes

### D.1 What happens if Lobster Trap crashes?

- **TCP refuses on :8080** → clients see connection refused.
- **Mitigation in our scripts**: `scripts/sprint5_5agent_workload.py` falls back to
  direct vLLM at :8000 when `--lobstertrap-endpoint` returns connection error within
  a 3-second timeout. Logged to stderr as a warning. Fail-open by design — the
  reasoning is that an unavailable perimeter should not block ContextForge's INV-15
  layer from doing its job. Production deployments may prefer fail-closed; we
  document this as a deployment-time choice.

### D.2 What happens if vLLM backend (port :8000) times out?

- **Lobster Trap returns 502 Bad Gateway** to the client (standard reverse-proxy
  behavior).
- **Mitigation**: `scripts/_sprint5_pipeline.py` catches 502 and retries up to 2
  times with exponential backoff. After 2 failures, the request is recorded as
  `latency_ms=null, critic_verdict="UNKNOWN"` and excluded from JCR computation.

### D.3 What happens if policy YAML is malformed?

- **Lobster Trap `serve` fails to start** with `loading policy: yaml: ...` error.
- **Mitigation**: Day-3 integration tests validate the policy YAML before the
  demo. CI step: `python3 -c "import yaml; yaml.safe_load(open('configs/lobstertrap_policy.yaml'))"`.

### D.4 What happens if Gemini API rate-limits during the demo?

- **Critic agent returns HTTP 429 from Gemini AI Studio.**
- **Mitigation**: `scripts/_sprint5_pipeline.py` with `--critic-provider gemini`
  falls back to the local vLLM/mock critic after 1 retry. Demo voiceover stays
  truthful: "if Gemini rate-limits us live, the system still works on local
  models. INV-15 is provider-agnostic."

### D.5 What happens if Lobster Trap policy file disagrees with what we say in the demo?

- **Risk**: We claim "blocks PII" in the demo but the policy doesn't actually fire.
- **Mitigation**: `tests/test_lobstertrap_integration.py` (Day 3) runs the exact
  prompts we use in the demo video against the live policy and asserts the
  expected actions. Demo video script in §3.5 of the plan references specific
  prompts and expected outcomes.

---

## E. Assumptions / uncertainties flagged for Day 2 verification

These are claims we made in §A-D based on the public README + the cloned
configs/default_policy.yaml. Day 2 will verify each one with a running binary:

| Assumption | Source | Verify in Day 2 |
| ---------- | ------ | --------------- |
| Lobster Trap binary listens on configurable `--listen` (default `:8080`) | `cmd/serve.go:37` (read on 2026-05-13) | Build binary with `make build`; run `./lobstertrap serve --listen :8080 --backend http://localhost:8000`; verify with curl |
| Policy YAML accepts ingress_rules + egress_rules with the schema in our policy file | `configs/default_policy.yaml` | Day 2 step 2a smoke test will load our policy and either succeed or report a schema error |
| Default policy already catches injection, PII, credentials, exfiltration | `configs/default_policy.yaml` lines 6-146 | Run `./lobstertrap test` (built-in adversarial suite) to confirm pass-rate |
| `_lobstertrap` metadata header is honored bidirectionally by the proxy | README lines 246-291 | Send a curl with `_lobstertrap.declared_intent` and verify the response has matching detected metadata |
| Audit log JSONL is correlatable with our request_id | README line 304 + `cmd/serve.go:62-69` | Read audit log file after a curl-driven smoke and verify request_id matches |
| `HUMAN_REVIEW` action blocks until manual approval, not just returns | README line 156 ("Block until a human approves") | Day 2: trigger a HUMAN_REVIEW rule and observe behavior (might be a queue endpoint or just a configurable timeout) |
| Lobster Trap can route to vLLM at port 8000 (not just Ollama's 11434) | README "Supported backends" line 298 | Just point `--backend http://localhost:8000` at a vLLM dev server (or mock) and curl |

---

## F. References

- **Lobster Trap repo (cloned)**: `~/Documentos/external/lobstertrap/` (HEAD as of 2026-05-13)
- **Lobster Trap upstream**: https://github.com/veeainc/lobstertrap (note: README references `github.com/coal/lobstertrap` — the package was renamed/transferred but code is identical)
- **TechEx Track 1**: https://lablab.ai/ai-hackathons/techex-intelligent-enterprise-solutions-hackathon
- **ContextForge paper v2.0.1 (INV-15 source of truth)**: Zenodo DOI [10.5281/zenodo.20114594](https://doi.org/10.5281/zenodo.20114594)
- **Liang et al. 2026** (JCR drop under naive KV reuse): arXiv:2601.08343
- **Our 5-agent pipeline config**: `configs/sprint5_5agent.yaml`
- **Submission plan**: `.omc/plans/techex-track1-submission.md`
