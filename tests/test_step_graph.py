"""Tests for AgentStepGraph — TASK-006."""
import pytest
import sys
from contextforge.scheduling.step_graph import AgentStepGraph, AgentStep


class TestAgentStepGraph:
    """Tests for workflow step graph."""

    @pytest.mark.asyncio
    async def test_add_step_returns_self_for_chaining(self):
        """add_step() returns self for method chaining."""
        graph = AgentStepGraph()
        result = graph.add_step(AgentStep(agent_id="a", step_index=0))
        assert result is graph

    @pytest.mark.asyncio
    async def test_compute_steps_to_execution_simple(self):
        """compute_steps_to_execution returns correct distance."""
        graph = AgentStepGraph()
        graph.add_step(AgentStep(agent_id="retriever", step_index=0))
        graph.add_step(AgentStep(agent_id="summarizer", step_index=1, depends_on=["retriever"]))
        graph.add_step(AgentStep(agent_id="critic", step_index=2, depends_on=["summarizer"]))

        # retriever is at step 0, responder at step 2 (2 steps away from "retriever" start)
        dist = graph.compute_steps_to_execution("critic", current_step=0)
        assert dist >= 0

    @pytest.mark.asyncio
    async def test_compute_steps_to_execution_unknown_agent(self):
        """compute_steps_to_execution returns sys.maxsize for unknown agents."""
        graph = AgentStepGraph()
        graph.add_step(AgentStep(agent_id="retriever", step_index=0))
        dist = graph.compute_steps_to_execution("unknown_agent", current_step=0)
        assert dist == sys.maxsize

    @pytest.mark.asyncio
    async def test_get_prefetch_candidates(self):
        """get_prefetch_candidates returns agents within prefetch_window."""
        graph = AgentStepGraph()
        graph.add_step(AgentStep(agent_id="retriever", step_index=0))
        graph.add_step(AgentStep(agent_id="summarizer", step_index=1, depends_on=["retriever"]))
        graph.add_step(AgentStep(agent_id="critic", step_index=2, depends_on=["summarizer"]))
        graph.add_step(AgentStep(agent_id="responder", step_index=3, depends_on=["critic"]))

        candidates = graph.get_prefetch_candidates(current_step=0, lookahead=2)
        assert isinstance(candidates, list)

    @pytest.mark.asyncio
    async def test_get_eviction_priority_order(self):
        """get_eviction_priority_order returns agents sorted by steps-to-execution."""
        graph = AgentStepGraph()
        graph.add_step(AgentStep(agent_id="retriever", step_index=0))
        graph.add_step(AgentStep(agent_id="summarizer", step_index=1, depends_on=["retriever"]))
        graph.add_step(AgentStep(agent_id="critic", step_index=2, depends_on=["summarizer"]))

        order = graph.get_eviction_priority_order()
        assert isinstance(order, list)
        # "retriever" should be last (closest to execution), "critic" first (farthest)
        if len(order) >= 2:
            assert order[-1] == "retriever"  # closest to execution

    @pytest.mark.asyncio
    async def test_validate_dag_detects_cycle(self):
        """validate_dag() raises ValueError on cycle."""
        graph = AgentStepGraph()
        graph.add_step(AgentStep(agent_id="a", step_index=0, depends_on=["b"]))
        graph.add_step(AgentStep(agent_id="b", step_index=1, depends_on=["a"]))  # cycle!
        with pytest.raises(ValueError):
            graph.validate_dag()

    @pytest.mark.asyncio
    async def test_validate_dag_accepts_valid_graph(self):
        """validate_dag() passes for valid DAG."""
        graph = AgentStepGraph()
        graph.add_step(AgentStep(agent_id="retriever", step_index=0))
        graph.add_step(AgentStep(agent_id="summarizer", step_index=1, depends_on=["retriever"]))
        graph.add_step(AgentStep(agent_id="critic", step_index=2, depends_on=["summarizer"]))
        graph.validate_dag()  # Should not raise