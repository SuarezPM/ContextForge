"""Tests for PBKVPredictor — TASK-013."""
import pytest
import json
import tempfile
from pathlib import Path
from contextforge.scheduling.pbkv_predictor import PBKVPredictor, WorkflowStepRecord, PredictionResult


class TestPBKVPredictor:
    """Tests for PBKV predictor stub."""

    @pytest.mark.asyncio
    async def test_log_workflow_step(self):
        """log_workflow_step() records steps in history and JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir, max_history_steps=10)

            await predictor.log_workflow_step(
                step_idx=0,
                agent_id="agent_retriever",
                anchor_hash="anchor_0",
                token_length=100,
                cla_group=1,
            )

            assert len(predictor._history) == 1
            assert predictor._history[0].agent_id == "agent_retriever"

    @pytest.mark.asyncio
    async def test_predict_next_agents_returns_prediction_result(self):
        """predict_next_agents() returns PredictionResult."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir, max_history_steps=10)

            # Log some steps first
            for i in range(5):
                await predictor.log_workflow_step(
                    step_idx=i,
                    agent_id=f"agent_{i % 2}",
                    anchor_hash=f"anchor_{i}",
                    token_length=100,
                    cla_group=i % 2,
                )

            result = await predictor.predict_next_agents("agent_0", current_step=3, num_predictions=2)

            assert isinstance(result, PredictionResult)
            assert isinstance(result.predicted_agents, list)
            assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_predict_next_agents_empty_history(self):
        """predict_next_agents() returns default when no history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir, max_history_steps=10)

            result = await predictor.predict_next_agents("agent_0", current_step=0, num_predictions=3)

            assert isinstance(result, PredictionResult)
            # Empty history → confidence 0, returns current agent as fallback
            assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_get_prefetch_candidates(self):
        """get_prefetch_candidates() returns list of block IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir, max_history_steps=10)

            for i in range(5):
                await predictor.log_workflow_step(
                    step_idx=i,
                    agent_id=f"agent_{i % 2}",
                    anchor_hash=f"anchor_{i}",
                    token_length=100,
                    cla_group=i % 2,
                )

            candidates = await predictor.get_prefetch_candidates("agent_0", step=3)

            assert isinstance(candidates, list)

    def test_workflow_step_record(self):
        """WorkflowStepRecord dataclass works."""
        record = WorkflowStepRecord(
            step_idx=0,
            agent_id="test_agent",
            anchor_hash="anchor_x",
            token_length=100,
            cla_group=2,
        )
        assert record.step_idx == 0
        assert record.agent_id == "test_agent"
        assert record.cla_group == 2

    def test_prediction_result_defaults(self):
        """PredictionResult has correct defaults."""
        result = PredictionResult(
            predicted_agents=["a1"],
            predicted_anchor_hashes=["h1"],
            confidence=0.5,
        )
        assert result.prefetch_block_ids == []
        assert result.confidence == 0.5

    def test_get_stats(self):
        """get_stats() returns predictor statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir, max_history_steps=50)

            stats = predictor.get_stats()
            assert stats["history_size"] == 0
            assert stats["max_history_steps"] == 50
            assert "_pbkv_logs" in stats["log_file"]