"""Unit tests for the training run analysis data model.

Tests the dataclass-based data model that provides a unified representation
of training run data across W&B and GCS sources.
"""

from __future__ import annotations

from alpha_zero.analysis.data_model import (
    IterationData,
    ModelVersionData,
    RunData,
    RunIdentifier,
)


class TestRunIdentifier:
    """Tests for RunIdentifier dataclass."""

    def test_run_identifier_defaults(self) -> None:
        """Test that RunIdentifier has sensible defaults."""
        rid = RunIdentifier()
        assert rid.wandb_run_id is None
        assert rid.wandb_group is None
        assert rid.wandb_project == "civ6-alphazero"
        assert rid.wandb_entity is None
        assert rid.gcs_bucket is None
        assert rid.gcs_prefix is None

    def test_run_identifier_to_dict(self) -> None:
        """Test RunIdentifier serialization."""
        rid = RunIdentifier(wandb_run_id="abc123", gcs_bucket="my-bucket")
        d = rid.to_dict()
        assert d["wandb_run_id"] == "abc123"
        assert d["gcs_bucket"] == "my-bucket"
        assert d["wandb_project"] == "civ6-alphazero"


class TestIterationData:
    """Tests for IterationData dataclass."""

    def test_iteration_data_defaults_to_none(self) -> None:
        """Test that all optional fields default to None or False."""
        it = IterationData(iteration=0)
        assert it.iteration == 0
        assert it.policy_loss is None
        assert it.value_loss is None
        assert it.total_loss is None
        assert it.learning_rate is None
        assert it.eval_win_rate is None
        assert it.eval_draw_rate is None
        assert it.eval_avg_turns is None
        assert it.value_correlation is None
        assert it.games_completed is None
        assert it.trajectory_count is None
        assert it.buffer_size is None
        assert it.model_version is None
        assert it.promoted is False
        assert it.duration_minutes is None
        assert it.selfplay_cost is None
        assert it.training_cost is None
        assert it.eval_cost is None
        assert it.cumulative_cost is None

    def test_iteration_data_to_dict(self) -> None:
        """Test IterationData serialization includes all fields."""
        it = IterationData(
            iteration=5,
            policy_loss=0.85,
            value_loss=0.42,
            promoted=True,
            model_version=10,
        )
        d = it.to_dict()
        assert d["iteration"] == 5
        assert d["policy_loss"] == 0.85
        assert d["value_loss"] == 0.42
        assert d["promoted"] is True
        assert d["model_version"] == 10
        assert d["total_loss"] is None

    def test_iteration_data_with_costs(self) -> None:
        """Test IterationData with cost tracking fields."""
        it = IterationData(
            iteration=3,
            selfplay_cost=0.50,
            training_cost=0.30,
            eval_cost=0.10,
            cumulative_cost=2.50,
        )
        assert it.selfplay_cost == 0.50
        assert it.cumulative_cost == 2.50


class TestModelVersionData:
    """Tests for ModelVersionData dataclass."""

    def test_model_version_data_optional_elo(self) -> None:
        """Test that elo defaults to None."""
        mv = ModelVersionData(version=5, iteration=2)
        assert mv.version == 5
        assert mv.iteration == 2
        assert mv.elo is None
        assert mv.timestamp is None
        assert mv.games_played == 0
        assert mv.model_config is None

    def test_model_version_data_to_dict(self) -> None:
        """Test ModelVersionData serialization."""
        mv = ModelVersionData(
            version=10,
            iteration=5,
            elo=1520.5,
            games_played=100,
        )
        d = mv.to_dict()
        assert d["version"] == 10
        assert d["elo"] == 1520.5
        assert d["games_played"] == 100


class TestRunData:
    """Tests for RunData dataclass."""

    def test_run_data_computes_total_iterations(self) -> None:
        """Test total_iterations property returns len of iterations list."""
        rd = RunData(
            identifier=RunIdentifier(),
            iterations=[IterationData(iteration=i) for i in range(10)],
            models=[],
            config={},
            summary={},
        )
        assert rd.total_iterations == 10

    def test_run_data_promotion_count(self) -> None:
        """Test promotions property counts promoted iterations."""
        iterations = [
            IterationData(iteration=0, promoted=False),
            IterationData(iteration=1, promoted=True),
            IterationData(iteration=2, promoted=False),
            IterationData(iteration=3, promoted=True),
            IterationData(iteration=4, promoted=True),
        ]
        rd = RunData(
            identifier=RunIdentifier(),
            iterations=iterations,
            models=[],
            config={},
            summary={},
        )
        assert rd.promotions == 3

    def test_run_data_to_dict_includes_computed_properties(self) -> None:
        """Test that to_dict includes total_iterations and promotions."""
        iterations = [
            IterationData(iteration=0, promoted=True),
            IterationData(iteration=1, promoted=False),
        ]
        models = [ModelVersionData(version=0, iteration=0)]
        rd = RunData(
            identifier=RunIdentifier(wandb_run_id="test"),
            iterations=iterations,
            models=models,
            config={"lr": 0.001},
            summary={"status": "complete"},
            start_model_version=0,
            end_model_version=5,
        )
        d = rd.to_dict()
        assert d["total_iterations"] == 2
        assert d["promotions"] == 1
        assert d["start_model_version"] == 0
        assert d["end_model_version"] == 5
        assert d["config"] == {"lr": 0.001}
        assert d["summary"] == {"status": "complete"}
        assert len(d["iterations"]) == 2
        assert len(d["models"]) == 1
        assert d["identifier"]["wandb_run_id"] == "test"

    def test_run_data_empty(self) -> None:
        """Test RunData with no iterations or models."""
        rd = RunData(
            identifier=RunIdentifier(),
            iterations=[],
            models=[],
            config={},
            summary={},
        )
        assert rd.total_iterations == 0
        assert rd.promotions == 0
