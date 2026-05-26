import json
from apohara_context_forge.observability.ledger import Ledger, GENESIS

def test_append_and_verify_chain(tmp_path):
    led = Ledger(tmp_path / "led.jsonl")
    e1 = led.append({"a": 1}); e2 = led.append({"a": 2})
    assert e1["prev_hash"] == GENESIS
    assert e2["prev_hash"] == e1["entry_hash"]
    v = led.verify()
    assert v["valid"] is True and v["entries"] == 2

def test_tamper_is_detected(tmp_path):
    p = tmp_path / "led.jsonl"; led = Ledger(p)
    led.append({"a": 1}); led.append({"a": 2})
    lines = p.read_text().splitlines()
    rec = json.loads(lines[0]); rec["payload"]["a"] = 999
    lines[0] = json.dumps(rec); p.write_text("\n".join(lines) + "\n")
    v = Ledger(p).verify()
    assert v["valid"] is False and v["broken_at"] == 0

def test_corrupt_line_is_a_break_not_a_crash(tmp_path):
    p = tmp_path / "led.jsonl"; led = Ledger(p)
    led.append({"a": 1}); led.append({"a": 2})
    with p.open("a", encoding="utf-8") as fh:
        fh.write("}{ not valid json\n")
    v = Ledger(p).verify()                 # must NOT raise
    assert v["valid"] is False and v["broken_at"] == 2

def test_broken_prev_link_detected(tmp_path):
    p = tmp_path / "led.jsonl"; led = Ledger(p)
    led.append({"a": 1}); led.append({"a": 2})
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1]); rec["prev_hash"] = "f" * 64
    lines[1] = json.dumps(rec); p.write_text("\n".join(lines) + "\n")
    assert Ledger(p).verify()["broken_at"] == 1
