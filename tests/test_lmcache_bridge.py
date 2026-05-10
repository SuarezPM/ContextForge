"""Tests for LMCacheConnectorV1 — TASK-007."""
import pytest
from apohara_context_forge.serving.lmcache_bridge import LMCacheConnectorV1, LMCacheMeta


class TestLMCacheConnectorV1:
    """Tests for LMCache bridge."""

    def test_lmcache_meta_defaults(self):
        """LMCacheMeta has pre_rope=True by default (INVARIANT 10)."""
        meta = LMCacheMeta()
        assert meta.pre_rope == True

    def test_is_active_without_client(self):
        """is_active() returns False when no LMCache client."""
        bridge = LMCacheConnectorV1(lmcache_client=None)
        assert bridge.is_active() == False

    def test_is_active_with_client(self):
        """is_active() returns True when LMCache client is provided."""
        bridge = LMCacheConnectorV1(lmcache_client=object())
        assert bridge.is_active() == True

    def test_build_prefix_hint(self):
        """build_prefix_hint returns correct metadata dict."""
        bridge = LMCacheConnectorV1()
        hint = bridge.build_prefix_hint(
            token_ids=[101, 2003, 1996],
            agent_id="agent_1",
            anchor_hash="anchor_abc",
        )
        assert hint["anchor_hash"] == "anchor_abc"
        assert hint["agent_id"] == "agent_1"
        assert hint["token_length"] == 3
        assert hint["pre_rope"] == True  # INVARIANT 10

    @pytest.mark.asyncio
    async def test_on_save_kv_layer_noop_when_inactive(self):
        """on_save_kv_layer does nothing when bridge is inactive."""
        bridge = LMCacheConnectorV1(lmcache_client=None)
        await bridge.on_save_kv_layer("block_0", None, {"anchor_hash": "test"})
        # No error means graceful handling

    @pytest.mark.asyncio
    async def test_on_load_kv_layer_returns_none_when_inactive(self):
        """on_load_kv_layer returns None when bridge is inactive."""
        bridge = LMCacheConnectorV1(lmcache_client=None)
        result = await bridge.on_load_kv_layer("block_0", {"offset_hint": [1.0, 2.0]})
        assert result is None

    def test_get_stats_returns_dict(self):
        """get_stats returns bridge statistics."""
        bridge = LMCacheConnectorV1(enable_offset_hints=True, enable_cla_metadata=False)
        stats = bridge.get_stats()
        assert isinstance(stats, dict)
        assert stats["active"] == False
        assert stats["offset_hints_enabled"] == True
        assert stats["cla_metadata_enabled"] == False