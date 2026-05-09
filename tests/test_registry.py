"""Tests for ContextRegistry, TTLCache, and VRAMAwareCache."""
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from contextforge.registry.ttl_cache import TTLCache
from contextforge.registry.context_registry import ContextRegistry
from contextforge.registry.vram_aware_cache import VRAMAwareCache, EvictionMode


@pytest.fixture
def ttl_cache():
    return TTLCache(default_ttl_seconds=5)


@pytest.fixture
def registry():
    return ContextRegistry(default_ttl=10)


@pytest.fixture
async def vram_cache():
    cache = VRAMAwareCache(max_token_budget=50_000_000)
    await cache.start()
    yield cache
    await cache.stop()


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


class TestVRAMAwareCache:
    """Tests for VRAMAwareCache."""

    async def test_set_and_get(self, vram_cache):
        await vram_cache.set("key1", "value1", token_count=100)
        result = await vram_cache.get("key1")
        assert result == "value1"

    async def test_get_nonexistent(self, vram_cache):
        result = await vram_cache.get("nonexistent")
        assert result is None

    async def test_delete(self, vram_cache):
        await vram_cache.set("key1", "value1", token_count=100)
        deleted = await vram_cache.delete("key1")
        assert deleted is True
        result = await vram_cache.get("key1")
        assert result is None

    async def test_delete_nonexistent(self, vram_cache):
        deleted = await vram_cache.delete("nonexistent")
        assert deleted is False

    async def test_size(self, vram_cache):
        assert vram_cache.size == 0
        await vram_cache.set("key1", "value1", token_count=100)
        assert vram_cache.size == 1
        await vram_cache.set("key2", "value2", token_count=200)
        assert vram_cache.size == 2

    async def test_token_tracking(self, vram_cache):
        assert vram_cache.total_tokens == 0
        await vram_cache.set("key1", "value1", token_count=500)
        assert vram_cache.total_tokens == 500
        await vram_cache.set("key2", "value2", token_count=300)
        assert vram_cache.total_tokens == 800
        await vram_cache.delete("key1")
        assert vram_cache.total_tokens == 300

    async def test_clear(self, vram_cache):
        await vram_cache.set("key1", "value1", token_count=100)
        await vram_cache.set("key2", "value2", token_count=200)
        assert vram_cache.size == 2
        await vram_cache.clear()
        assert vram_cache.size == 0
        assert vram_cache.total_tokens == 0

    async def test_update_existing_key(self, vram_cache):
        await vram_cache.set("key1", "value1", token_count=100)
        await vram_cache.set("key1", "value2", token_count=200)
        result = await vram_cache.get("key1")
        assert result == "value2"
        assert vram_cache.total_tokens == 200

    async def test_mode_initial_relaxed(self, vram_cache):
        """Cache starts in RELAXED mode by default."""
        assert vram_cache.mode == EvictionMode.RELAXED
        assert vram_cache.is_blocked is False

    async def test_eviction_modes(self, vram_cache):
        """Test that modes transition correctly based on pressure."""
        # Patch get_pressure to return specific values
        with patch.object(vram_cache._vram, 'get_pressure', return_value=0.0):
            await vram_cache._apply_eviction_policy()
            assert vram_cache.mode == EvictionMode.RELAXED

        with patch.object(vram_cache._vram, 'get_pressure', return_value=0.75):
            await vram_cache._apply_eviction_policy()
            assert vram_cache.mode == EvictionMode.NORMAL

        with patch.object(vram_cache._vram, 'get_pressure', return_value=0.88):
            await vram_cache._apply_eviction_policy()
            assert vram_cache.mode == EvictionMode.PRESSURE

        with patch.object(vram_cache._vram, 'get_pressure', return_value=0.94):
            await vram_cache._apply_eviction_policy()
            assert vram_cache.mode == EvictionMode.CRITICAL

        with patch.object(vram_cache._vram, 'get_pressure', return_value=0.97):
            await vram_cache._apply_eviction_policy()
            assert vram_cache.mode == EvictionMode.EMERGENCY
            assert vram_cache.is_blocked is True

    async def test_blocked_mode(self, vram_cache):
        """In EMERGENCY mode, set() should return False."""
        # Force EMERGENCY mode
        with patch.object(vram_cache._vram, 'get_pressure', return_value=0.97):
            await vram_cache._apply_eviction_policy()
            assert vram_cache.is_blocked is True

        # set() should be blocked
        result = await vram_cache.set("key1", "value1", token_count=100)
        assert result is False

        # After pressure drops, should unblock
        with patch.object(vram_cache._vram, 'get_pressure', return_value=0.50):
            await vram_cache._apply_eviction_policy()
            assert vram_cache.is_blocked is False

        # set() should work again
        result = await vram_cache.set("key2", "value2", token_count=100)
        assert result is True

    async def test_pressure_to_mode_boundaries(self):
        """Test exact boundary values for _pressure_to_mode."""
        assert VRAMAwareCache._pressure_to_mode(0.69) == EvictionMode.RELAXED
        assert VRAMAwareCache._pressure_to_mode(0.70) == EvictionMode.NORMAL
        assert VRAMAwareCache._pressure_to_mode(0.84) == EvictionMode.NORMAL
        assert VRAMAwareCache._pressure_to_mode(0.85) == EvictionMode.PRESSURE
        assert VRAMAwareCache._pressure_to_mode(0.91) == EvictionMode.PRESSURE
        assert VRAMAwareCache._pressure_to_mode(0.92) == EvictionMode.CRITICAL
        assert VRAMAwareCache._pressure_to_mode(0.95) == EvictionMode.CRITICAL
        assert VRAMAwareCache._pressure_to_mode(0.96) == EvictionMode.EMERGENCY
        assert VRAMAwareCache._pressure_to_mode(1.0) == EvictionMode.EMERGENCY

    async def test_emergency_unblocks_on_lower_pressure(self, vram_cache):
        """Verify is_blocked clears when pressure drops from EMERGENCY."""
        # Enter EMERGENCY
        with patch.object(vram_cache._vram, 'get_pressure', return_value=0.97):
            await vram_cache._apply_eviction_policy()
        assert vram_cache.is_blocked is True
        assert vram_cache.mode == EvictionMode.EMERGENCY

        # Drop to RELAXED
        with patch.object(vram_cache._vram, 'get_pressure', return_value=0.50):
            await vram_cache._apply_eviction_policy()
        assert vram_cache.is_blocked is False
        assert vram_cache.mode == EvictionMode.RELAXED