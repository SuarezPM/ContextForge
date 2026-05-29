# MERGED: OpenCode (deep KV physics) + CC (surface coverage)
# All tests hermetic: no GPU, no TCP, no downloaded weights required
"""Tests for agent pipeline."""
import pytest

from agents.demo_agents import create_agents, AGENT_CONFIGS
from agents.pipeline import Pipeline


class TestDemoAgents:
    """Tests for demo agents."""

    def test_create_agents_count(self):
        agents = create_agents()
        assert len(agents) == 5

    def test_agent_configs(self):
        assert len(AGENT_CONFIGS) == 5
        assert AGENT_CONFIGS[0]["id"] == "retriever"
        assert AGENT_CONFIGS[4]["id"] == "responder"

    @pytest.mark.asyncio
    async def test_retriever_agent_process(self):
        from agents.demo_agents import RetrieverAgent
        
        agent = RetrieverAgent()
        result = await agent.process({"query": "What is AI?"})
        
        assert result["agent_id"] == "retriever"
        assert "result" in result
        assert "tokens_before" in result
        assert "tokens_after" in result

    @pytest.mark.asyncio
    async def test_pipeline_run(self):
        pipeline = Pipeline(enable_contextforge=False)
        result = await pipeline.run("What is machine learning?")
        
        assert "query" in result
        assert "final_output" in result
        assert "summary" in result
        assert result["summary"]["total_tokens_before"] > 0


class TestPipeline:
    """Tests for Pipeline orchestrator."""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        pipeline = Pipeline()
        assert pipeline.enable_contextforge is True
        assert len(pipeline.agents) == 5

    @pytest.mark.asyncio
    async def test_pipeline_metrics_tracking(self):
        pipeline = Pipeline(enable_contextforge=False)
        await pipeline.run("Test query")

        assert pipeline.metrics["total_tokens_before"] > 0
        assert isinstance(pipeline.metrics["strategies_used"], dict)


class TestPrefixCacheSalt:
    """ATOM Fase 1: cross-agent prefix sharing via cache_salt."""

    @pytest.mark.asyncio
    async def test_non_judge_agents_share_one_salt(self):
        """The 4 non-judge agents collide on a single shared salt → they
        actually share prefix KV blocks intra-instance."""
        pipeline = Pipeline(enable_contextforge=False)
        result = await pipeline.run("What is machine learning?")
        am = result["agent_metrics"]

        shared = {
            aid: am[f"{aid}_metrics"]["cache_salt"]
            for aid in ("retriever", "reranker", "summarizer", "responder")
        }
        assert all(am[f"{aid}_metrics"]["salt_shared"] for aid in shared)
        assert len(set(shared.values())) == 1

    @pytest.mark.asyncio
    async def test_critic_gets_isolated_salt_inv15(self):
        """The critic (judge role) trips INV-15 via the real JCR gate and gets
        a UNIQUE salt, distinct from the shared group's salt."""
        pipeline = Pipeline(enable_contextforge=False)
        result = await pipeline.run("What is machine learning?")
        am = result["agent_metrics"]

        critic_salt = am["critic_metrics"]["cache_salt"]
        assert am["critic_metrics"]["salt_shared"] is False
        shared_salt = am["retriever_metrics"]["cache_salt"]
        assert critic_salt != shared_salt
