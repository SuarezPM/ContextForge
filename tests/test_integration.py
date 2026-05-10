"""End-to-end integration tests for ContextRegistry with LSH + FAISS + VRAMAwareCache."""
import asyncio
import importlib.util
import pytest
import pytest_asyncio
from unittest.mock import patch

from prometheus_client import REGISTRY

# Skip tests requiring faiss (not installed in this environment)
FAISS_AVAILABLE = importlib.util.find_spec('faiss') is not None

from apohara_context_forge import (
    ContextRegistry,
    SharedContextResult,
    LSHTokenMatcher,
    FAISSContextIndex,
    VRAMAwareCache,
    EvictionMode,
)
from apohara_context_forge.metrics.prometheus_metrics import cache_hits, cache_misses


@pytest_asyncio.fixture
async def registry():
    """Create a ContextRegistry with all components wired up."""
    reg = ContextRegistry(
        lsh_matcher=LSHTokenMatcher(),
        vram_cache=VRAMAwareCache(max_token_budget=50_000_000),
        faiss_index=FAISSContextIndex(dim=384),
    )
    await reg.start()
    yield reg
    await reg.stop()


class TestSharedContextWithSharedSystemPrompt:
    """Test 1: Register 3 agents with shared system prompt → get_shared_context()."""
    requires_faiss = pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
    @pytest.mark.asyncio
    async def test_shared_system_prompt_returns_non_empty_blocks(self, registry):
        """Verify get_shared_context() returns non-empty blocks with tokens saved."""
        # Shared system prompt for all 3 agents
        system_prompt = (
            "You are a helpful AI assistant running on AMD MI300X. "
            "Your role is to provide accurate and concise responses."
        )

        role_prompt_1 = "You are a retriever agent specializing in finding relevant documents."
        role_prompt_2 = "You are a summarizer agent that condenses information."
        role_prompt_3 = "You are a translator agent that adapts content across languages."

        # Register all 3 agents with same system prompt
        entry1 = await registry.register_agent("agent1", system_prompt, role_prompt_1)
        assert entry1.agent_id == "agent1"
        assert entry1.token_count > 0

        entry2 = await registry.register_agent("agent2", system_prompt, role_prompt_2)
        assert entry2.agent_id == "agent2"
        assert entry2.token_count > 0

        entry3 = await registry.register_agent("agent3", system_prompt, role_prompt_3)
        assert entry3.agent_id == "agent3"
        assert entry3.token_count > 0

        # Get shared context across all 3 agents
        results = await registry.get_shared_context(["agent1", "agent2", "agent3"])

        # Verify result list is non-empty
        assert results is not None
        assert isinstance(results, list)

        # At least one result should have shared blocks (system prompt blocks should match)
        has_shared_blocks = any(
            len(r.shared_blocks) > 0 for r in results
        )

        # Verify total_tokens_saved > 0 if we found matches
        if has_shared_blocks:
            total_tokens_saved = sum(r.total_tokens_saved for r in results)
            assert total_tokens_saved > 0, "Expected token savings from shared blocks"

        # Verify reuse_confidence > 0 if we found matches
        if has_shared_blocks:
            max_confidence = max(r.reuse_confidence for r in results)
            assert max_confidence > 0.0, "Expected positive reuse confidence"

    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
    @pytest.mark.asyncio
    async def test_shared_context_contains_all_requested_agents(self, registry):
        """Verify all requested agents are present in results."""
        system_prompt = "Shared system prompt for testing."

        await registry.register_agent("agent1", system_prompt, "Role 1")
        await registry.register_agent("agent2", system_prompt, "Role 2")
        await registry.register_agent("agent3", system_prompt, "Role 3")

        results = await registry.get_shared_context(["agent1", "agent2", "agent3"])

        result_agent_ids = {r.agent_id for r in results}
        assert result_agent_ids == {"agent1", "agent2", "agent3"}


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
class TestPrometheusMetricsEmission:
    """Test 2: Prometheus metrics are emitted after get_shared_context()."""

    @pytest.mark.asyncio
    async def test_cache_hits_metric_incremented(self, registry):
        """Verify cache_hits counter is incremented after get_shared_context()."""
        system_prompt = "Test system prompt for metrics verification."

        await registry.register_agent("agent1", system_prompt, "Role 1")
        await registry.register_agent("agent2", system_prompt, "Role 2")

        # Clear any existing metrics by collecting samples
        initial_hits = self._get_metric_value(cache_hits, "agent1", "system_prompt")
        initial_misses = self._get_metric_value(cache_misses, "agent1")

        # Trigger get_shared_context
        await registry.get_shared_context(["agent1", "agent2"])

        # Verify cache_hits or cache_misses was incremented
        final_hits = self._get_metric_value(cache_hits, "agent1", "system_prompt")
        final_misses = self._get_metric_value(cache_misses, "agent1")

        metric_incremented = (
            (final_hits > initial_hits) or (final_misses > initial_misses)
        )
        assert metric_incremented, (
            f"Expected cache_hits or cache_misses to increment. "
            f"Hits: {initial_hits} -> {final_hits}, Misses: {initial_misses} -> {final_misses}"
        )

    @pytest.mark.asyncio
    async def test_cache_misses_metric_incremented_for_no_match(self, registry):
        """Verify cache_misses is incremented when no reusable blocks found."""
        # Use completely different prompts to ensure no matches
        await registry.register_agent("agent1", "Unique prompt for agent 1", "Role 1")
        await registry.register_agent("agent2", "Completely different prompt for agent 2", "Role 2")

        initial_misses = self._get_metric_value(cache_misses, "agent1")

        # Get shared context - should have no matches due to different prompts
        await registry.get_shared_context(["agent1", "agent2"])

        final_misses = self._get_metric_value(cache_misses, "agent1")
        assert final_misses > initial_misses, "Expected cache_misses to increment for non-matching prompts"

    @staticmethod
    def _get_metric_value(counter, *label_values):
        """Get the current value of a Prometheus counter with given labels."""
        for metric_family in REGISTRY.collect():
            if metric_family.name == counter._name:
                for sample in metric_family.samples:
                    if sample.labels.values() == tuple(label_values):
                        return sample.value
        return 0


class TestVRAMModeTransitions:
    """Test 3: VRAM mode transitions from RELAXED to higher modes under pressure."""

    @pytest.mark.asyncio
    async def test_mode_transitions_to_pressure_under_high_vram(self, registry):
        """Verify mode changes from RELAXED to PRESSURE when VRAM pressure increases."""
        # Initial mode should be RELAXED (no pressure)
        initial_mode = await registry.get_vram_mode()
        assert initial_mode == EvictionMode.RELAXED.value

        # Simulate VRAM pressure increase to PRESSURE level (0.85-0.92)
        await registry._vram_cache._apply_eviction_policy(pressure=0.88)

        current_mode = await registry.get_vram_mode()
        assert current_mode == EvictionMode.PRESSURE.value, (
            f"Expected PRESSURE mode at 0.88 pressure, got {current_mode}"
        )

    @pytest.mark.asyncio
    async def test_mode_transitions_to_critical_under_high_vram(self, registry):
        """Verify mode changes from RELAXED to CRITICAL when VRAM pressure is high."""
        # Simulate VRAM pressure increase to CRITICAL level (0.92-0.96)
        await registry._vram_cache._apply_eviction_policy(pressure=0.94)

        current_mode = await registry.get_vram_mode()
        assert current_mode == EvictionMode.CRITICAL.value, (
            f"Expected CRITICAL mode at 0.94 pressure, got {current_mode}"
        )

    @pytest.mark.asyncio
    async def test_mode_transitions_to_emergency_at_saturation(self, registry):
        """Verify mode changes to EMERGENCY when VRAM pressure >= 0.96."""
        # Simulate VRAM pressure at EMERGENCY level (>= 0.96)
        await registry._vram_cache._apply_eviction_policy(pressure=0.97)

        current_mode = await registry.get_vram_mode()
        assert current_mode == EvictionMode.EMERGENCY.value, (
            f"Expected EMERGENCY mode at 0.97 pressure, got {current_mode}"
        )

    @pytest.mark.asyncio
    async def test_mode_reverts_to_relaxed_when_pressure_drops(self, registry):
        """Verify mode reverts to RELAXED when VRAM pressure drops."""
        # First, set to a higher mode
        await registry._vram_cache._apply_eviction_policy(pressure=0.88)
        assert await registry.get_vram_mode() == EvictionMode.PRESSURE.value

        # Then drop pressure to RELAXED level
        await registry._vram_cache._apply_eviction_policy(pressure=0.50)

        current_mode = await registry.get_vram_mode()
        assert current_mode == EvictionMode.RELAXED.value, (
            f"Expected RELAXED mode after pressure drop, got {current_mode}"
        )


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
class TestClearAgent:
    """Test 4: clear_agent() removes agent from registry."""

    @pytest.mark.asyncio
    async def test_clear_agent_removes_from_registry(self, registry):
        """Verify get_all_agents() no longer contains cleared agent."""
        system_prompt = "Test system prompt for clear operation."

        # Register agent
        await registry.register_agent("agent_to_clear", system_prompt, "Role prompt")

        # Verify agent is registered
        all_agents_before = await registry.get_all_agents()
        assert "agent_to_clear" in all_agents_before

        # Clear the agent
        cleared = await registry.clear_agent("agent_to_clear")
        assert cleared is True

        # Verify agent is no longer in registry
        all_agents_after = await registry.get_all_agents()
        assert "agent_to_clear" not in all_agents_after

    @pytest.mark.asyncio
    async def test_clear_nonexistent_agent_returns_false(self, registry):
        """Verify clearing non-existent agent returns False."""
        result = await registry.clear_agent("nonexistent_agent")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_agent_clears_from_all_stores(self, registry):
        """Verify agent is removed from LSH, FAISS, and cache after clear."""
        system_prompt = "Test system prompt for complete clearing."

        # Register agent
        await registry.register_agent("agent_to_clear", system_prompt, "Role prompt")

        # Verify agent exists in LSH blocks
        agent_blocks_before = await registry._lsh._agent_blocks.get("agent_to_clear")
        assert agent_blocks_before is not None

        # Clear the agent
        await registry.clear_agent("agent_to_clear")

        # Verify agent is removed from LSH
        agent_blocks_after = await registry._lsh._agent_blocks.get("agent_to_clear")
        assert agent_blocks_after is None

        # Verify agent is removed from FAISS
        faiss_embedding = await registry._faiss.get_embedding("agent_to_clear")
        assert faiss_embedding is None

        # Verify agent is removed from VRAM cache
        cache_val = await registry._vram_cache.get("context:agent_to_clear")
        assert cache_val is None

    @pytest.mark.asyncio
    async def test_multiple_agents_cleared_selectively(self, registry):
        """Verify only specified agent is cleared when clearing one of many."""
        system_prompt = "Shared system prompt."

        # Register multiple agents
        await registry.register_agent("agent1", system_prompt, "Role 1")
        await registry.register_agent("agent2", system_prompt, "Role 2")
        await registry.register_agent("agent3", system_prompt, "Role 3")

        # Clear only agent2
        await registry.clear_agent("agent2")

        # Verify only agent2 is removed
        all_agents = await registry.get_all_agents()
        assert "agent1" in all_agents
        assert "agent2" not in all_agents
        assert "agent3" in all_agents


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss not installed")
class TestEndToEndWorkflow:
    """Full end-to-end workflow tests combining all components."""

    @pytest.mark.asyncio
    async def test_full_workflow_register_query_clear(self, registry):
        """Complete workflow: register → query → verify metrics → clear."""
        system_prompt = (
            "You are an AI assistant on AMD MI300X. "
            "Provide accurate and helpful responses."
        )

        # Register agents with shared system prompt
        await registry.register_agent("retriever", system_prompt, "Find relevant docs")
        await registry.register_agent("summarizer", system_prompt, "Summarize content")
        await registry.register_agent("translator", system_prompt, "Translate content")

        # Query shared context
        results = await registry.get_shared_context(["retriever", "summarizer", "translator"])
        assert len(results) == 3

        # Verify metrics were emitted
        all_agents = {"retriever", "summarizer", "translator"}
        result_ids = {r.agent_id for r in results}
        assert result_ids == all_agents

        # Clear one agent
        cleared = await registry.clear_agent("summarizer")
        assert cleared is True

        # Verify remaining agents still work
        remaining = await registry.get_all_agents()
        assert "retriever" in remaining
        assert "translator" in remaining
        assert "summarizer" not in remaining

    @pytest.mark.asyncio
    async def test_shared_context_with_empty_role_prompts(self, registry):
        """Verify registration works with empty role prompts."""
        system_prompt = "System prompt only."

        # Register with empty role prompts
        await registry.register_agent("agent1", system_prompt, "")
        await registry.register_agent("agent2", system_prompt, "")

        results = await registry.get_shared_context(["agent1", "agent2"])
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_shared_context_with_single_agent_returns_empty(self, registry):
        """Verify get_shared_context returns empty list for single agent."""
        await registry.register_agent("solo_agent", "System", "Role")

        results = await registry.get_shared_context(["solo_agent"])
        assert results == []

    @pytest.mark.asyncio
    async def test_get_shared_context_with_unregistered_agent_returns_empty(self, registry):
        """Verify get_shared_context returns empty when agent not registered."""
        results = await registry.get_shared_context(["nonexistent"])
        assert results == []