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
        
        agent = RetrieverAgent("retriever", "retrieve relevant documents")
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