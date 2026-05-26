"""Subprocess tests for the FORGE-LEDGER verify CLI: exercises real exit codes
(0 = chain intact, 2 = tampered, 64 = usage error)."""
import json
import os
import subprocess
import sys
from pathlib import Path

from apohara_context_forge.observability.ledger import Ledger

REPO = Path(__file__).resolve().parents[1]


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "apohara_context_forge.observability.ledger_cli", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPO), "PATH": os.environ.get("PATH", "")},
    )


def test_cli_exit0_on_valid(tmp_path):
    p = tmp_path / "led.jsonl"
    led = Ledger(p)
    led.append({"a": 1})
    led.append({"a": 2})
    r = _run("verify", str(p))
    assert r.returncode == 0
    assert json.loads(r.stdout)["valid"] is True


def test_cli_exit2_on_tampered(tmp_path):
    p = tmp_path / "led.jsonl"
    led = Ledger(p)
    led.append({"a": 1})
    led.append({"a": 2})
    lines = p.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["payload"]["a"] = 999
    lines[0] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n")
    r = _run("verify", str(p))
    assert r.returncode == 2
    assert json.loads(r.stdout)["valid"] is False


def test_cli_usage_error():
    r = _run()  # no args -> usage error
    assert r.returncode == 64
