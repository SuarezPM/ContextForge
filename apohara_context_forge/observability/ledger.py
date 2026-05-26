"""SHA-256 hash-chained, tamper-evident ledger (FORGE-LEDGER). Each line:
{prev_hash, entry_hash=sha256(prev_hash+canonical(payload)), payload}. Append-only."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Iterator

GENESIS = "0" * 64

def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))

def _entry_hash(prev_hash: str, payload: dict) -> str:
    return hashlib.sha256((prev_hash + _canonical(payload)).encode("utf-8")).hexdigest()

class Ledger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        last = GENESIS
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        last = json.loads(line)["entry_hash"]
        return last

    def append(self, payload: dict) -> dict:
        prev = self._last_hash()
        entry = {"prev_hash": prev, "entry_hash": _entry_hash(prev, payload), "payload": payload}
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n"); fh.flush()
        return entry

    def verify(self) -> dict:
        prev, n, broken = GENESIS, 0, None
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    if e.get("prev_hash") != prev or e.get("entry_hash") != _entry_hash(prev, e["payload"]):
                        broken = i; break
                    prev, n = e["entry_hash"], n + 1
        return {"valid": broken is None, "entries": n, "broken_at": broken, "head": prev}

    def __iter__(self) -> Iterator[dict]:
        if self._path.exists():
            with self._path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)
