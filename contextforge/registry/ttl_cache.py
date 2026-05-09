"""TTL-based eviction cache for stale contexts."""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class TTLCache:
    """Thread-safe TTL cache with automatic eviction."""

    def __init__(self, default_ttl_seconds: int = 300):
        self._store: dict[str, tuple[Any, datetime]] = {}
        self._lock = asyncio.Lock()
        self._default_ttl = default_ttl_seconds

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store a value with optional custom TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        expiry = datetime.now() + timedelta(seconds=ttl)
        async with self._lock:
            self._store[key] = (value, expiry)

    async def get(self, key: str) -> Any | None:
        """Retrieve a value if it exists and is not expired."""
        async with self._lock:
            if key not in self._store:
                return None
            value, expiry = self._store[key]
            if datetime.now() > expiry:
                del self._store[key]
                return None
            return value

    async def delete(self, key: str) -> bool:
        """Delete a key, returns True if it existed."""
        async with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def evict_expired(self) -> int:
        """Remove all expired entries, returns count evicted."""
        count = 0
        now = datetime.now()
        async with self._lock:
            expired = [k for k, (_, exp) in self._store.items() if now > exp]
            for k in expired:
                del self._store[k]
                count += 1
        if count > 0:
            logger.info(f"Evicted {count} expired entries from TTL cache")
        return count

    async def clear(self) -> None:
        """Clear all entries."""
        async with self._lock:
            self._store.clear()

    async def size(self) -> int:
        """Return current entry count."""
        async with self._lock:
            return len(self._store)

    async def keys(self) -> list[str]:
        """Return all current keys."""
        async with self._lock:
            return list(self._store.keys())
