---
marp: true
theme: default
paginate: true
size: 16:9
backgroundColor: #0f172a
color: #f8fafc
header: 'TechEx 2026 · Track 1 · Agent Security & AI Governance'
footer: 'Apohara ContextForge × Lobster Trap · github.com/SuarezPM/Apohara_Context_Forge · Apache-2.0'
style: |
  section { font-family: 'DejaVu Sans', sans-serif; background-color: #0f172a; color: #f8fafc; }
  h1 { color: #ef4444; font-size: 1.7em; }
  h2 { color: #22c55e; font-size: 1.1em; }
  strong { color: #f8fafc; }
  em { color: #94a3b8; }
  table { border-collapse: collapse; }
  th { background: #1e293b; color: #f8fafc; padding: 6px 10px; border: 1px solid #334155; }
  td { padding: 6px 10px; border: 1px solid #334155; color: #f8fafc; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
  .big-number { font-size: 3.2em; color: #ef4444; font-weight: 800; line-height: 1; }
  .caption { font-size: 0.65em; color: #94a3b8; }
  section.lead h1 { font-size: 2.1em; }
  blockquote { border-left: 4px solid #ef4444; padding-left: 14px; color: #cbd5e1; }
  header, footer { color: #94a3b8; font-size: 0.55em; }
---

<!-- _class: lead -->

# Apohara ContextForge × Lobster Trap

## Defense-in-Depth Trust Layer for Multi-Agent LLMs

Built on AMD Instinct MI300X · Apache-2.0 + MIT · Solo dev · UNT, Argentina

Paper v2.0.1 · Zenodo DOI 10.5281/zenodo.20114594

<small>5 slides · ~3 minute pitch · TechEx 2026</small>

---

<!-- _class: lead -->

# Two failure modes nobody catches together

<div class="columns">

<div>

## Obvious attacks

Lobster Trap (Veea, MIT) ✅
- Prompt injection
- Credential exposure
- PII leakage
- Data exfiltration
- Role impersonation

> Sub-millisecond regex DPI

</div>

<div>

## Silent drift

ContextForge INV-15 (Apache-2.0) ✅
- JCR drops **8-23pt** under KV reuse
- Judge agrees with cached output
- Compliance has no audit trail

> Liang et al. 2026, arXiv:2601.08343

</div>

</div>

**One tool catches one OR the other. We catch both.**

---

<!-- _class: lead -->

# Our answer: INV-15 + Lobster Trap composed

```
[Client] ──► [Lobster Trap :8080] ──► [5-agent ContextForge] ──► [vLLM/Gemini]
                  │ ingress DPI               │ INV-15 gate
                  │ • injection               │ risk > τ ?
                  │ • PII                     │ └─► dense prefill
                  │ • credentials             │     (no KV reuse for critic)
                  │ • exfiltration            │
                  ▼                            ▼
              audit log JSONL          inv15_audit JSONL
                  └────────correlation_id────────┘
```

**Layer 1 = perimeter (deterministic).**
**Layer 2 = behavioral (formal invariant).**
**Audit trails correlated by request_id.**

---

<!-- _class: lead -->

# Measured (real MI300X, not extrapolation)

<div class="columns">

<div>

<div class="big-number">3.55×</div>

INT4 KV reduction
constant 4K → 262K context

<div class="big-number">0 / 1,210</div>

INV-15 violations on
exhaustive sweep

</div>

<div>

<div class="big-number">11 / 11</div>

Lobster Trap adversarial
suite PASS on our policy

<div class="big-number">Δ 0.23</div>

JCR drop *prevented*
(apohara_on 1.00 vs off 0.77)

</div>

</div>

<div class="caption">All measurements committed at <code>logs/*.json</code>. Paper v2.0.1 §5. AUDIT.md tracks every claim with file:line evidence.</div>

---

<!-- _class: lead -->

# Why it wins where others don't

| | Apohara×LT | TrustLayer | Cursor | Lovable | GSD-2 |
|---|:-:|:-:|:-:|:-:|:-:|
| Perimeter DPI (LT integration) | ✅ | ✅ | ❌ | ❌ | ❌ |
| Formal invariant for KV reuse | ✅ INV-15 | ❌ | ❌ | ❌ | ❌ |
| Paper with public DOI | ✅ Zenodo | ❌ | ❌ | ❌ | ❌ |
| AUDIT.md honesty log | ✅ public | ❌ | ❌ | ❌ | ❌ |
| Hardware-validated (AMD MI300X) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Cross-vendor critic (Gemini) | ✅ | ✅ | ❌ | ❌ | ❌ |
| Open source (MIT + Apache-2.0) | ✅ | partial | ❌ | ❌ | ✅ |

**The ask:** Veea Award ($10K + DevKit + Stage Recognition) + Gemini Award ($5K). Both flow into V8 codec validation + V2.1 paper.

<small>github.com/SuarezPM/Apohara_Context_Forge · Apache-2.0 · DOI 10.5281/zenodo.20114594</small>
