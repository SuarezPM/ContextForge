"""Tests for vLLMAtomPlugin — TASK-008."""
import pytest
from apohara_context_forge.serving.atom_plugin import vLLMAtomPlugin, ATOMConfig, PreAttentionHook, PostAttentionHook


class TestATOMConfig:
    """Tests for ATOMConfig."""

    def test_atom_config_defaults(self):
        """ATOMConfig has sensible defaults."""
        config = ATOMConfig()
        assert config.enable_quantization == True
        assert config.enable_anchor_routing == True
        assert config.enable_cla_injection == True
        assert config.quantization_mode == "rotate_kv"


class TestvLLMAtomPlugin:
    """Tests for vLLMAtomPlugin."""

    def test_plugin_initialization(self):
        """Plugin initializes with ATOMConfig."""
        config = ATOMConfig()
        plugin = vLLMAtomPlugin(config)
        assert plugin._config is config
        assert plugin.is_initialized() == False

    def test_initialize_sets_worker_id(self):
        """initialize() sets worker_id and marks initialized."""
        config = ATOMConfig()
        plugin = vLLMAtomPlugin(config)
        plugin.initialize("worker_0", {})
        assert plugin.is_initialized() == True
        stats = plugin.get_stats()
        assert stats["worker_id"] == "worker_0"
        assert stats["initialized"] == True

    def test_pre_attention_hook_returns_dict(self):
        """pre_attention_hook returns metadata dict."""
        config = ATOMConfig(enable_quantization=True)
        hook = PreAttentionHook(config)
        result = hook(["b0", "b1"], [101, 2003], layer_idx=0)
        assert isinstance(result, dict)
        assert result["quantized"] == True
        assert result["pre_rope"] == True  # INVARIANT 10
        assert result["layer_idx"] == 0

    def test_post_attention_hook_returns_dict(self):
        """post_attention_hook returns stats dict."""
        config = ATOMConfig()
        hook = PostAttentionHook(config)
        result = hook(["b0", "b1"], [], layer_idx=0)
        assert isinstance(result, dict)
        assert result["processed_blocks"] == 2
        assert result["layer_idx"] == 0

    def test_plugin_pre_attention_hook_property(self):
        """Plugin exposes pre_attention_hook as property."""
        config = ATOMConfig()
        plugin = vLLMAtomPlugin(config)
        assert hasattr(plugin, "pre_attention_hook")
        assert callable(plugin.pre_attention_hook)

    def test_plugin_post_attention_hook_property(self):
        """Plugin exposes post_attention_hook as property."""
        config = ATOMConfig()
        plugin = vLLMAtomPlugin(config)
        assert hasattr(plugin, "post_attention_hook")
        assert callable(plugin.post_attention_hook)

    def test_get_stats_returns_config_and_state(self):
        """get_stats returns configuration and state."""
        config = ATOMConfig(
            enable_quantization=True,
            enable_anchor_routing=False,
            enable_cla_injection=True,
            quantization_mode="rotate_kv",
        )
        plugin = vLLMAtomPlugin(config)
        plugin.initialize("worker_test", {})

        stats = plugin.get_stats()
        assert stats["initialized"] == True
        assert stats["worker_id"] == "worker_test"
        assert stats["config"]["enable_quantization"] == True
        assert stats["config"]["quantization_mode"] == "rotate_kv"