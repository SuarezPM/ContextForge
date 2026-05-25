from __future__ import annotations

import json
import aiofiles
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator, Iterator

logger = logging.getLogger(__name__)


class AuditLog:
    """JSONL audit log for INV-15 gate decisions."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._write_warned = False
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, record: dict) -> None:
        """Append a single JSONL record with an ISO-8601 timestamp."""
        entry = {"ts": datetime.now(tz=timezone.utc).isoformat(), **record}
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
                fh.flush()
        except OSError as exc:
            if not self._write_warned:
                logger.warning("AuditLog write failed (further errors suppressed): %s", exc)
                self._write_warned = True

    async def replay(self) -> AsyncIterator[dict]:
        """Yield all records in write order asynchronously."""
        try:
            async with aiofiles.open(self._path, "r", encoding="utf-8") as fh:
                async for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        except FileNotFoundError:
            return
