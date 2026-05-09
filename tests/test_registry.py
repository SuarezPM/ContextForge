"""Tests for ContextRegistry and TTLCache."""
import asyncio
import pytest

from contextforge.registry.ttl_cache import TTLCache
from contextforge.registry.context_registry import ContextRegistry


@pytest.fixture
def ttl_cache():
    return TTLCache(default_ttl_seconds=5)


@pytest.fixture
def registry():
    return ContextRegistry(default_ttl=10)


class TestTTLCache:
    """Tests for TTLCache."""

    async def test_set_and_get(self, ttl_cache):
        await ttl_cache.set("key1", "value1")
        result = await ttl_cache.get("key1")
        assert result == "value1"

    async def test_get_nonexistent(self, ttl_cache):
        result = await ttl_cache.get("nonexistent")
        assert result is None

    async def test_expiry(self, ttl_cache):
        await ttl_cache.set("key1", "value1", ttl_seconds=1)
        await asyncio.sleep(1.1)
        result = await ttl_cache.get("key1")
        assert result is None

    async def test_delete(self, ttl_cache):
        await ttl_cache.set("key1", "value1")
        deleted = await ttl_cache.delete("key1")
        assert deleted is True
        result = await ttl_cache.get("key1")
        assert result is None

    async def test_evict_expired(self, ttl_cache):
        await ttl_cache.set("key1", "value1", ttl_seconds=1)
        await asyncio.sleep(1.1)
        count = await ttl_cache.evict_expired()
        assert count == 1
        assert await ttl_cache.size() == 0

    async def test_clear(self, ttl_cache):
        await ttl_cache.set("key1", "value1")
        await ttl_cache.set("key2", "value2")
        await ttl_cache.clear()
        assert await ttl_cache.size() == 0


class TestContextRegistry:
    """Tests for ContextRegistry."""

    async def test_register_and_get(self, registry):
        entry = await registry.register("agent1", "This is a test context")
        assert entry.agent_id == "agent1"
        assert entry.context == "This is a test context"
        assert entry.token_count > 0

    async def test_get_nonexistent(self, registry):
        result = await registry.get("nonexistent")
        assert result is None

    async def test_register_updates_existing(self, registry):
        await registry.register("agent1", "First context")
        entry = await registry.register("agent1", "Second context")
        assert entry.context == "Second context"

    async def test_evict_expired(self, registry):
        await registry.register("agent1", "Test context")
        count = await registry.evict_expired()
        assert count >= 0

    async def test_clear(self, registry):
        await registry.register("agent1", "Context 1")
        await registry.register("agent2", "Context 2")
        await registry.clear()
        entries = await registry.get_all_active()
        assert len(entries) == 0