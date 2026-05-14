"""Generate a static INV-15 Governance Dashboard HTML page.

Reads a JSONL audit log (from Lobster Trap + ContextForge runs) and
renders a single self-contained HTML file with:

- Hero KPIs (requests processed, INV-15 fires, LT blocks, JCR
  maintained, zero violations on sweep)
- Last-N events table with action / rule / agent / risk score
- Per-rule activation count bar chart (pure CSS, no external deps)
- Compliance mapping footer (NIST AI RMF, EU AI Act, ISO/IEC 42001)

The page is **self-contained** — no JS frameworks, no external CSS, no
remote fonts. Single HTML file you can `scp` to any web server or
attach to a submission.

Usage:

    PYTHONPATH=. python3 scripts/generate_governance_dashboard.py \\
        --audit-log /path/to/audit.jsonl \\
        --inv15-log /path/to/inv15.jsonl \\
        --out assets/inv15-governance-dashboard.html

If no audit logs are provided, the script synthesizes a demo dataset
(200 requests, mix of allows + blocks + INV-15 fires) so the dashboard
can be rendered in any state — useful for the README link before live
data is collected.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Demo data synthesis (when no real audit log is provided)
# ---------------------------------------------------------------------------


def synthesize_demo_events(n: int = 200, seed: int = 0) -> list[dict]:
    """Generate a plausible JSONL audit-event stream for demo rendering.

    Distribution matches what we'd see in a real 5-agent workload with
    Lobster Trap policy from `configs/lobstertrap_policy.yaml`:

    - ~75% benign queries → allow_apohara_5agent_pipeline (LT ALLOW)
    - ~5% prompt injection → block_prompt_injection (LT DENY)
    - ~3% credential exposure → block_credential_in_prompt (LT DENY)
    - ~3% PII request → block_pii_request (LT DENY)
    - ~2% exfiltration → block_data_exfiltration (LT DENY)
    - ~2% role impersonation → review_role_impersonation (LT HUMAN_REVIEW)
    - ~10% reach the critic with high reuse → INV-15 FIRE → dense prefill
    """
    rng = random.Random(seed)
    events: list[dict] = []
    base_ts = int(time.time()) - n * 30  # spread over ~100 min

    for i in range(n):
        ts = base_ts + i * 30
        r = rng.random()
        agent_role = rng.choice(["retriever", "reranker", "summarizer", "critic", "responder"])

        if r < 0.05:
            ev = {
                "timestamp_unix": ts,
                "source": "lobster_trap",
                "request_id": f"req-{i:04d}",
                "direction": "ingress",
                "action": "DENY",
                "rule_name": "block_prompt_injection",
                "agent_role": agent_role,
                "risk_score": rng.uniform(0.7, 0.9),
                "category": "prompt_injection",
            }
        elif r < 0.08:
            ev = {
                "timestamp_unix": ts,
                "source": "lobster_trap",
                "request_id": f"req-{i:04d}",
                "direction": "ingress",
                "action": "DENY",
                "rule_name": "block_credential_in_prompt",
                "agent_role": agent_role,
                "risk_score": rng.uniform(0.6, 0.85),
                "category": "credential_exposure",
            }
        elif r < 0.11:
            ev = {
                "timestamp_unix": ts,
                "source": "lobster_trap",
                "request_id": f"req-{i:04d}",
                "direction": "ingress",
                "action": "DENY",
                "rule_name": "block_pii_request",
                "agent_role": agent_role,
                "risk_score": rng.uniform(0.5, 0.8),
                "category": "pii",
            }
        elif r < 0.13:
            ev = {
                "timestamp_unix": ts,
                "source": "lobster_trap",
                "request_id": f"req-{i:04d}",
                "direction": "ingress",
                "action": "DENY",
                "rule_name": "block_data_exfiltration",
                "agent_role": agent_role,
                "risk_score": rng.uniform(0.6, 0.85),
                "category": "exfiltration",
            }
        elif r < 0.15:
            ev = {
                "timestamp_unix": ts,
                "source": "lobster_trap",
                "request_id": f"req-{i:04d}",
                "direction": "ingress",
                "action": "HUMAN_REVIEW",
                "rule_name": "review_role_impersonation",
                "agent_role": agent_role,
                "risk_score": rng.uniform(0.5, 0.7),
                "category": "role_impersonation",
            }
        elif r < 0.25 and agent_role == "critic":
            ev = {
                "timestamp_unix": ts,
                "source": "contextforge_inv15",
                "request_id": f"req-{i:04d}",
                "direction": "behavioral_gate",
                "action": "DENSE_PREFILL",
                "rule_name": "inv15_judge_protection",
                "agent_role": "critic",
                "risk_score": rng.uniform(0.65, 0.85),
                "reuse_rate": rng.uniform(0.85, 0.98),
                "category": "inv15_fire",
                "inv15_fired": True,
            }
        else:
            ev = {
                "timestamp_unix": ts,
                "source": "lobster_trap",
                "request_id": f"req-{i:04d}",
                "direction": "ingress",
                "action": "ALLOW",
                "rule_name": "allow_apohara_5agent_pipeline",
                "agent_role": agent_role,
                "risk_score": rng.uniform(0.0, 0.3),
                "category": "benign",
            }
        events.append(ev)

    return events


# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    events = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


# ---------------------------------------------------------------------------
# KPI aggregation
# ---------------------------------------------------------------------------


def aggregate_kpis(events: list[dict]) -> dict:
    total = len(events)
    by_action = Counter(e.get("action", "?") for e in events)
    by_rule = Counter(e.get("rule_name", "?") for e in events)
    by_category = Counter(e.get("category", "?") for e in events)

    lt_blocks = sum(
        1 for e in events
        if e.get("source") == "lobster_trap" and e.get("action") in ("DENY", "HUMAN_REVIEW")
    )
    inv15_fires = sum(1 for e in events if e.get("inv15_fired") is True)
    allows = by_action.get("ALLOW", 0)

    return {
        "total_requests": total,
        "lt_blocks_total": lt_blocks,
        "inv15_fires_total": inv15_fires,
        "allows_total": allows,
        "block_rate_pct": (100.0 * lt_blocks / total) if total else 0.0,
        "inv15_fire_rate_pct": (100.0 * inv15_fires / total) if total else 0.0,
        "by_action": dict(by_action),
        "by_rule": dict(by_rule),
        "by_category": dict(by_category),
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Apohara ContextForge — INV-15 Governance Dashboard</title>
<style>
  :root {{
    --bg:        #0f172a;
    --surface:   #1e293b;
    --border:    #334155;
    --text:      #f8fafc;
    --muted:     #94a3b8;
    --accent:    #ef4444;
    --green:     #22c55e;
    --yellow:    #fbbf24;
    --blue:      #3b82f6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "DejaVu Sans", sans-serif;
    line-height: 1.5; padding: 32px; min-height: 100vh;
  }}
  header {{ border-bottom: 1px solid var(--border); padding-bottom: 24px; margin-bottom: 32px; }}
  h1 {{ font-size: 1.7em; color: var(--text); margin-bottom: 6px; }}
  h2 {{ font-size: 1.15em; color: var(--accent); margin: 28px 0 12px; }}
  .subtitle {{ color: var(--muted); font-size: 0.95em; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 24px 0; }}
  .kpi {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }}
  .kpi-label {{ font-size: 0.8em; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-value {{ font-size: 2.2em; font-weight: 800; color: var(--accent); line-height: 1; margin: 8px 0 4px; }}
  .kpi-caption {{ font-size: 0.75em; color: var(--muted); }}
  .kpi.green .kpi-value {{ color: var(--green); }}
  .kpi.yellow .kpi-value {{ color: var(--yellow); }}
  .kpi.blue .kpi-value {{ color: var(--blue); }}
  table {{ width: 100%; border-collapse: collapse; background: var(--surface); border-radius: 8px; overflow: hidden; }}
  th {{ background: #0f172a; color: var(--muted); padding: 10px 14px; text-align: left; font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid var(--border); }}
  td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); font-size: 0.92em; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 600; }}
  .badge.allow {{ background: rgba(34, 197, 94, 0.15); color: var(--green); }}
  .badge.deny {{ background: rgba(239, 68, 68, 0.15); color: var(--accent); }}
  .badge.review {{ background: rgba(251, 191, 36, 0.15); color: var(--yellow); }}
  .badge.gate {{ background: rgba(59, 130, 246, 0.15); color: var(--blue); }}
  .bar-row {{ display: flex; align-items: center; gap: 12px; padding: 6px 0; }}
  .bar-label {{ width: 250px; color: var(--muted); font-size: 0.85em; }}
  .bar-track {{ flex: 1; height: 10px; background: var(--surface); border-radius: 5px; overflow: hidden; }}
  .bar-fill {{ height: 100%; background: var(--accent); }}
  .bar-fill.green {{ background: var(--green); }}
  .bar-fill.yellow {{ background: var(--yellow); }}
  .bar-fill.blue {{ background: var(--blue); }}
  .bar-value {{ width: 60px; text-align: right; font-variant-numeric: tabular-nums; font-size: 0.85em; color: var(--text); }}
  .compliance {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 24px; margin-top: 32px; }}
  .compliance ul {{ list-style: none; }}
  .compliance li {{ padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 0.92em; }}
  .compliance li:last-child {{ border-bottom: none; }}
  .compliance strong {{ color: var(--blue); }}
  footer {{ margin-top: 32px; padding-top: 24px; border-top: 1px solid var(--border); font-size: 0.82em; color: var(--muted); text-align: center; }}
  code {{ background: var(--surface); padding: 2px 6px; border-radius: 4px; font-size: 0.85em; color: var(--green); font-family: "DejaVu Sans Mono", monospace; }}
  .demo-banner {{ background: rgba(251, 191, 36, 0.12); border: 1px solid var(--yellow); border-radius: 8px; padding: 14px 20px; margin-bottom: 24px; color: var(--yellow); font-size: 0.9em; }}
</style>
</head>
<body>

<header>
  <h1>Apohara ContextForge — INV-15 Governance Dashboard</h1>
  <p class="subtitle">
    Defense-in-depth governance stack · Lobster Trap (perimeter) + ContextForge INV-15 (behavioral) ·
    Generated <code>{generated_at}</code> · Source: <code>{source_path}</code>
  </p>
</header>

{demo_banner}

<h2>Aggregate KPIs</h2>
<div class="kpi-grid">
  <div class="kpi green">
    <div class="kpi-label">Requests processed</div>
    <div class="kpi-value">{total_requests:,}</div>
    <div class="kpi-caption">across {n_rules} distinct policy rules</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Lobster Trap blocks</div>
    <div class="kpi-value">{lt_blocks_total:,}</div>
    <div class="kpi-caption">{block_rate_pct:.1f}% of all traffic</div>
  </div>
  <div class="kpi blue">
    <div class="kpi-label">INV-15 fires</div>
    <div class="kpi-value">{inv15_fires_total:,}</div>
    <div class="kpi-caption">judge-agent gate, {inv15_fire_rate_pct:.1f}% activation</div>
  </div>
  <div class="kpi green">
    <div class="kpi-label">INV-15 violations</div>
    <div class="kpi-value">0 / 1,210</div>
    <div class="kpi-caption">paper sweep (independent of this run)</div>
  </div>
</div>

<h2>Policy actions distribution</h2>
<div>
{actions_bars}
</div>

<h2>Top rules activated</h2>
<div>
{rules_bars}
</div>

<h2>Recent events (last {n_recent})</h2>
<table>
  <thead>
    <tr>
      <th>Timestamp</th>
      <th>Source</th>
      <th>Action</th>
      <th>Rule</th>
      <th>Agent</th>
      <th>Risk</th>
      <th>Request ID</th>
    </tr>
  </thead>
  <tbody>
{events_rows}
  </tbody>
</table>

<div class="compliance">
  <h2 style="margin-top:0;">Compliance mapping</h2>
  <ul>
    <li><strong>NIST AI RMF</strong> · GOVERN: policy file <code>configs/lobstertrap_policy.yaml</code> · MAP: <code>docs/threat-model.md</code> · MEASURE: INV-15 closed-form risk score + JCR delta · MANAGE: this dashboard + JSONL audit log</li>
    <li><strong>EU AI Act</strong> · Article 9 (risk management): threat model · Article 12 (record-keeping): JSONL audit log · Article 14 (human oversight): <code>HUMAN_REVIEW</code> policy action (role_impersonation rule) · Article 15 (cybersecurity): 11/11 Lobster Trap adversarial PASS · enforcement deadline 2-Aug-2026</li>
    <li><strong>ISO/IEC 42001</strong> · A.7 Planning: threat model · A.8 Support: integration docs · A.9 Operation: live integration tests (4/4 PASS) · A.10 Performance evaluation: JCR delta measurements · A.11 Improvement: AUDIT.md (11/11 items)</li>
    <li><strong>Honesty discipline</strong> · <code>AUDIT.md</code> entry #11 documents an external Perplexity Pro audit catch + fix (2026-05-13) — proof we accept findings and fix them rather than hide them</li>
  </ul>
</div>

<footer>
  Apohara ContextForge V7.0.0-rc.2+ · Apache-2.0 · Paper <a href="https://doi.org/10.5281/zenodo.20114594" style="color: var(--blue);">Zenodo DOI 10.5281/zenodo.20114594</a> ·
  Repo <a href="https://github.com/SuarezPM/Apohara_Context_Forge" style="color: var(--blue);">github.com/SuarezPM/Apohara_Context_Forge</a> ·
  Generated by <code>scripts/generate_governance_dashboard.py</code>
</footer>

</body>
</html>
"""


def _action_class(action: str) -> str:
    return {
        "ALLOW": "allow",
        "DENY": "deny",
        "HUMAN_REVIEW": "review",
        "DENSE_PREFILL": "gate",
    }.get(action, "")


def _bar_class(name: str) -> str:
    if "ALLOW" in name or "allow" in name or "benign" in name:
        return "green"
    if "DENY" in name or "block" in name or "DENSE" in name:
        return "" # accent default
    if "REVIEW" in name or "review" in name or "role" in name:
        return "yellow"
    return "blue"


def render_bars(counter: dict, max_rows: int = 8) -> str:
    if not counter:
        return "<p style='color: var(--muted);'>No data.</p>"
    total = max(counter.values()) or 1
    rows = sorted(counter.items(), key=lambda x: -x[1])[:max_rows]
    html = []
    for label, value in rows:
        pct = 100 * value / total
        cls = _bar_class(label)
        html.append(f"""
        <div class="bar-row">
          <div class="bar-label">{label}</div>
          <div class="bar-track"><div class="bar-fill {cls}" style="width: {pct:.1f}%;"></div></div>
          <div class="bar-value">{value:,}</div>
        </div>""")
    return "".join(html)


def render_events_rows(events: list[dict], n: int = 20) -> str:
    rows = []
    for e in events[-n:]:
        ts = e.get("timestamp_unix", 0)
        ts_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S UTC") if ts else "—"
        source = e.get("source", "?")
        action = e.get("action", "?")
        rule = e.get("rule_name", "?")
        agent = e.get("agent_role", "—")
        risk = e.get("risk_score")
        risk_str = f"{risk:.2f}" if isinstance(risk, (int, float)) else "—"
        rid = e.get("request_id", "—")
        action_cls = _action_class(action)
        rows.append(f"""
    <tr>
      <td><code style="font-size:0.85em;">{ts_str}</code></td>
      <td>{source}</td>
      <td><span class="badge {action_cls}">{action}</span></td>
      <td><code style="font-size:0.85em;">{rule}</code></td>
      <td>{agent}</td>
      <td style="font-variant-numeric: tabular-nums;">{risk_str}</td>
      <td><code style="font-size:0.8em; color: var(--muted);">{rid}</code></td>
    </tr>""")
    return "".join(rows)


def render(events: list[dict], source_path: str, is_demo: bool) -> str:
    kpis = aggregate_kpis(events)
    demo_banner = ""
    if is_demo:
        demo_banner = """
<div class="demo-banner">
  <strong>Note:</strong> This dashboard is rendered from synthesized demo data
  for illustration. Production deployments point at a real Lobster Trap audit
  log + INV-15 JSONL stream via the <code>--audit-log</code> /
  <code>--inv15-log</code> flags of <code>scripts/generate_governance_dashboard.py</code>.
</div>"""

    return HTML_TEMPLATE.format(
        generated_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        source_path=source_path,
        demo_banner=demo_banner,
        total_requests=kpis["total_requests"],
        n_rules=len(kpis["by_rule"]),
        lt_blocks_total=kpis["lt_blocks_total"],
        block_rate_pct=kpis["block_rate_pct"],
        inv15_fires_total=kpis["inv15_fires_total"],
        inv15_fire_rate_pct=kpis["inv15_fire_rate_pct"],
        actions_bars=render_bars(kpis["by_action"]),
        rules_bars=render_bars(kpis["by_rule"], max_rows=10),
        n_recent=min(20, len(events)),
        events_rows=render_events_rows(events, n=20),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--audit-log", type=Path, default=None,
                   help="Lobster Trap JSONL audit log")
    p.add_argument("--inv15-log", type=Path, default=None,
                   help="ContextForge INV-15 JSONL audit log")
    p.add_argument("--out", type=Path, default=Path("assets/inv15-governance-dashboard.html"),
                   help="Output HTML file path")
    p.add_argument("--demo", action="store_true",
                   help="Force demo data even if logs are provided")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    events: list[dict] = []
    sources: list[str] = []
    is_demo = args.demo

    if not args.demo:
        if args.audit_log and args.audit_log.exists():
            events.extend(load_jsonl(args.audit_log))
            sources.append(str(args.audit_log))
        if args.inv15_log and args.inv15_log.exists():
            events.extend(load_jsonl(args.inv15_log))
            sources.append(str(args.inv15_log))

    if not events:
        events = synthesize_demo_events(seed=args.seed)
        sources = ["synthesized demo data"]
        is_demo = True

    events.sort(key=lambda e: e.get("timestamp_unix", 0))
    html = render(events, source_path=" + ".join(sources), is_demo=is_demo)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"✅ {len(events)} events → {args.out} ({args.out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
