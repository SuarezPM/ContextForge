"""Tests for PBKVPredictor — Markov chain implementation."""
import json
import pytest
import tempfile
from pathlib import Path

from contextforge.scheduling.pbkv_predictor import (
    PBKVPredictor,
    WorkflowStepRecord,
    PredictionResult,
)


class TestPBKVPredictor:
    """Tests for PBKV predictor Markov chain implementation."""

    # ===== Existing stub tests (backward compatibility) =====

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
        """predict_next_agents() returns PredictionResult via async path."""
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

            result = await predictor._predict_next_agents_async(
                "agent_0", current_step=3, num_predictions=2
            )

            assert isinstance(result, PredictionResult)
            assert isinstance(result.predicted_agents, list)
            assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_predict_next_agents_empty_history(self):
        """predict_next_agents() returns default when no history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir, max_history_steps=10)

            result = await predictor._predict_next_agents_async(
                "agent_0", current_step=0, num_predictions=3
            )

            assert isinstance(result, PredictionResult)
            # Empty history → confidence 0, returns current agent as fallback
            assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_get_prefetch_candidates(self):
        """get_prefetch_candidates() returns list of agent IDs."""
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
            assert "workflow_steps.jsonl" in stats["log_file"]
            assert stats["trained"] is False

    # ===== Markov chain training tests =====

    def test_train_from_jsonl(self):
        """train_from_jsonl() builds transition table correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "workflow_steps.jsonl"

            # Write JSONL with known sequence: A → B → C → A → B
            records = [
                {"step_idx": 0, "agent_id": "A", "anchor_hash": "h0", "token_length": 10, "cla_group": 1},
                {"step_idx": 1, "agent_id": "B", "anchor_hash": "h1", "token_length": 10, "cla_group": 1},
                {"step_idx": 2, "agent_id": "C", "anchor_hash": "h2", "token_length": 10, "cla_group": 1},
                {"step_idx": 3, "agent_id": "A", "anchor_hash": "h3", "token_length": 10, "cla_group": 1},
                {"step_idx": 4, "agent_id": "B", "anchor_hash": "h4", "token_length": 10, "cla_group": 1},
            ]
            with open(log_file, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            predictor = PBKVPredictor(log_dir=tmpdir)
            predictor.train_from_jsonl(tmpdir)

            assert predictor._trained is True
            assert predictor._all_agents == {"A", "B", "C"}
            # Check 2nd-order transitions exist
            assert ("A", "B") in predictor._transition_table
            assert ("B", "C") in predictor._transition_table
            assert ("C", "A") in predictor._transition_table
            assert ("A", "B") in predictor._transition_table

    def test_train_from_jsonl_with_multiple_sequences(self):
        """train_from_jsonl() handles multiple sequences (empty lines)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "workflow_steps.jsonl"

            # Two sequences: A→B and C→D
            records = [
                {"step_idx": 0, "agent_id": "A", "anchor_hash": "h0", "token_length": 10, "cla_group": 1},
                {"step_idx": 1, "agent_id": "B", "anchor_hash": "h1", "token_length": 10, "cla_group": 1},
                {},
                {"step_idx": 0, "agent_id": "C", "anchor_hash": "h2", "token_length": 10, "cla_group": 1},
                {"step_idx": 1, "agent_id": "D", "anchor_hash": "h3", "token_length": 10, "cla_group": 1},
            ]
            with open(log_file, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            predictor = PBKVPredictor(log_dir=tmpdir)
            predictor.train_from_jsonl(tmpdir)

            assert predictor._trained is True
            assert predictor._all_agents == {"A", "B", "C", "D"}

    def test_train_from_jsonl_missing_file(self):
        """train_from_jsonl() handles missing file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir)
            predictor.train_from_jsonl(str(Path(tmpdir) / "nonexistent.jsonl"))
            assert predictor._trained is False

    # ===== Prediction correctness tests =====

    def test_predict_next_agents_sync(self):
        """Synchronous predict_next_agents() returns list of agent IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir)

            # Train with known pattern: A → B → C
            log_file = Path(tmpdir) / "workflow_steps.jsonl"
            records = [
                {"step_idx": 0, "agent_id": "A", "anchor_hash": "h0", "token_length": 10, "cla_group": 1},
                {"step_idx": 1, "agent_id": "B", "anchor_hash": "h1", "token_length": 10, "cla_group": 1},
                {"step_idx": 2, "agent_id": "C", "anchor_hash": "h2", "token_length": 10, "cla_group": 1},
            ]
            with open(log_file, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            predictor.train_from_jsonl(tmpdir)
            predictions = predictor.predict_next_agents("B", top_k=2)

            assert isinstance(predictions, list)
            assert "C" in predictions  # B → C is the trained transition
            assert len(predictions) <= 2

    def test_predict_next_agents_fallback_on_empty_history(self):
        """predict_next_agents() falls back when no training data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir)

            # No training, no history
            predictions = predictor.predict_next_agents("X", top_k=3)

            assert predictions == ["X"]

    def test_predict_next_agents_fallback_1st_order(self):
        """predict_next_agents() uses 1st-order when 2nd-order state unseen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir)

            # Train: A → B → C (only 2nd-order state (A,B)→C)
            log_file = Path(tmpdir) / "workflow_steps.jsonl"
            records = [
                {"step_idx": 0, "agent_id": "A", "anchor_hash": "h0", "token_length": 10, "cla_group": 1},
                {"step_idx": 1, "agent_id": "B", "anchor_hash": "h1", "token_length": 10, "cla_group": 1},
                {"step_idx": 2, "agent_id": "C", "anchor_hash": "h2", "token_length": 10, "cla_group": 1},
            ]
            with open(log_file, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            predictor.train_from_jsonl(tmpdir)

            # Query for unseen state: should fall back to 1st-order
            predictions = predictor.predict_next_agents("B", top_k=1)
            assert "C" in predictions

    def test_predict_next_agents_top_k(self):
        """predict_next_agents() respects top_k parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir)

            log_file = Path(tmpdir) / "workflow_steps.jsonl"
            records = [
                {"step_idx": 0, "agent_id": "A", "anchor_hash": "h0", "token_length": 10, "cla_group": 1},
                {"step_idx": 1, "agent_id": "B", "anchor_hash": "h1", "token_length": 10, "cla_group": 1},
                {"step_idx": 2, "agent_id": "A", "anchor_hash": "h2", "token_length": 10, "cla_group": 1},
            ]
            with open(log_file, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            predictor.train_from_jsonl(tmpdir)
            predictions = predictor.predict_next_agents("B", top_k=1)

            assert len(predictions) == 1

    # ===== blend_alpha tests =====

    def test_blend_alpha_parameter(self):
        """blend_alpha is stored correctly in __init__."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir, blend_alpha=0.7)
            assert predictor._blend_alpha == 0.7

    def test_blend_alpha_default(self):
        """blend_alpha defaults to 0.6."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir)
            assert predictor._blend_alpha == 0.6

    @pytest.mark.asyncio
    async def test_get_eviction_priority_without_step_graph(self):
        """get_eviction_priority() works without AgentStepGraph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir)

            log_file = Path(tmpdir) / "workflow_steps.jsonl"
            records = [
                {"step_idx": 0, "agent_id": "A", "anchor_hash": "h0", "token_length": 10, "cla_group": 1},
                {"step_idx": 1, "agent_id": "B", "anchor_hash": "h1", "token_length": 10, "cla_group": 1},
                {"step_idx": 2, "agent_id": "C", "anchor_hash": "h2", "token_length": 10, "cla_group": 1},
            ]
            with open(log_file, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            predictor.train_from_jsonl(tmpdir)

            priority = await predictor.get_eviction_priority(["A", "B", "C"])
            assert isinstance(priority, list)
            assert len(priority) == 3

    @pytest.mark.asyncio
    async def test_get_eviction_priority_with_step_graph(self):
        """get_eviction_priority() blends with AgentStepGraph."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir, blend_alpha=0.6)

            # Train with pattern
            log_file = Path(tmpdir) / "workflow_steps.jsonl"
            records = [
                {"step_idx": 0, "agent_id": "retriever", "anchor_hash": "h0", "token_length": 10, "cla_group": 1},
                {"step_idx": 1, "agent_id": "summarizer", "anchor_hash": "h1", "token_length": 10, "cla_group": 1},
            ]
            with open(log_file, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            predictor.train_from_jsonl(tmpdir)

            # Create a simple step graph
            from contextforge.scheduling.step_graph import AgentStepGraph, AgentStep

            graph = AgentStepGraph()
            graph.add_step(AgentStep(agent_id="retriever", depends_on=[], step_index=0))
            graph.add_step(AgentStep(agent_id="summarizer", depends_on=["retriever"], step_index=1))

            priority = await predictor.get_eviction_priority(
                ["retriever", "summarizer"], step_graph=graph
            )

            assert isinstance(priority, list)
            assert len(priority) == 2

    # ===== Stats tests =====

    def test_get_stats_after_training(self):
        """get_stats() reflects training state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = PBKVPredictor(log_dir=tmpdir)

            log_file = Path(tmpdir) / "workflow_steps.jsonl"
            records = [
                {"step_idx": 0, "agent_id": "A", "anchor_hash": "h0", "token_length": 10, "cla_group": 1},
                {"step_idx": 1, "agent_id": "B", "anchor_hash": "h1", "token_length": 10, "cla_group": 1},
            ]
            with open(log_file, "w") as f:
                for rec in records:
                    f.write(json.dumps(rec) + "\n")

            predictor.train_from_jsonl(tmpdir)

            stats = predictor.get_stats()
            assert stats["trained"] is True
            assert stats["transition_table_size"] > 0
            assert stats["unique_agents"] == 2
