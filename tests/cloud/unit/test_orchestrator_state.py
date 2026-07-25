"""Unit tests for OrchestratorState and run_iterations().

Tests state persistence, iteration tracking, evaluation gate, circuit breaker,
and the dry-run integration path through run_iterations().
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.gcp.orchestrator_state import IterationResult, OrchestratorState

# =============================================================================
# OrchestratorState — Unit Tests
# =============================================================================


class TestOrchestratorState:
    """Tests for OrchestratorState dataclass logic."""

    def test_initial_state(self) -> None:
        """State initializes with correct defaults."""
        state = OrchestratorState(
            start_iteration=50,
            end_iteration=100,
            current_iteration=50,
            current_model_version=219,
        )
        assert state.remaining_iterations() == range(50, 100)
        assert state.total_iterations == 50
        assert not state.is_complete
        assert state.successful_iterations == 0
        assert state.failed_iterations == 0
        assert state.total_cost_usd == 0.0

    def test_is_complete(self) -> None:
        """is_complete returns True when all iterations done."""
        state = OrchestratorState(
            start_iteration=0,
            end_iteration=3,
            current_iteration=3,
            current_model_version=3,
        )
        assert state.is_complete
        assert len(state.remaining_iterations()) == 0

    def test_record_success_advances_state(self) -> None:
        """record_success increments counters and updates model version."""
        state = OrchestratorState(
            start_iteration=0,
            end_iteration=5,
            current_iteration=0,
            current_model_version=0,
        )
        state.record_success(
            iteration=0,
            model_version=1,
            win_rate=0.6,
            cost_usd=1.50,
            duration_seconds=300,
        )
        assert state.current_iteration == 1
        assert state.current_model_version == 1
        assert state.successful_iterations == 1
        assert state.consecutive_failures == 0
        assert state.total_cost_usd == pytest.approx(1.50)
        assert state.win_rate_history == [0.6]
        assert len(state.results) == 1

    def test_record_success_resets_failure_streak(self) -> None:
        """Consecutive failure counter resets on success."""
        state = OrchestratorState(
            start_iteration=0,
            end_iteration=5,
            current_iteration=0,
            current_model_version=0,
            consecutive_failures=2,
        )
        state.record_success(iteration=0, model_version=1)
        assert state.consecutive_failures == 0

    def test_record_success_rejected_model_keeps_old_version(self) -> None:
        """A rejected model does not update current_model_version."""
        state = OrchestratorState(
            start_iteration=0,
            end_iteration=5,
            current_iteration=0,
            current_model_version=10,
        )
        state.record_success(
            iteration=0,
            model_version=11,
            rejected=True,
        )
        assert state.current_model_version == 10  # Not updated
        assert state.current_iteration == 1  # But iteration advanced

    def test_record_failure_increments_counters(self) -> None:
        """record_failure increments failure and consecutive counters."""
        state = OrchestratorState(
            start_iteration=0,
            end_iteration=5,
            current_iteration=0,
            current_model_version=0,
        )
        state.record_failure(iteration=0, cost_usd=0.50)
        assert state.failed_iterations == 1
        assert state.consecutive_failures == 1
        assert state.total_cost_usd == pytest.approx(0.50)
        assert state.current_iteration == 1

    def test_remaining_iterations_after_partial_run(self) -> None:
        """remaining_iterations reflects partial progress correctly."""
        state = OrchestratorState(
            start_iteration=50,
            end_iteration=60,
            current_iteration=55,
            current_model_version=224,
        )
        remaining = list(state.remaining_iterations())
        assert remaining == [55, 56, 57, 58, 59]

    def test_to_summary(self) -> None:
        """to_summary returns a human-readable dict."""
        state = OrchestratorState(
            start_iteration=0,
            end_iteration=10,
            current_iteration=5,
            current_model_version=5,
            successful_iterations=4,
            failed_iterations=1,
            total_cost_usd=12.34,
            started_at=100.0,
            updated_at=700.0,
        )
        summary = state.to_summary()
        assert summary["iterations_completed"] == "5/10"
        assert summary["successful"] == 4
        assert summary["failed"] == 1
        assert summary["model_version"] == 5
        assert summary["total_cost_usd"] == "$12.34"

    def test_serialization_roundtrip(self) -> None:
        """State can be serialized to dict and back."""
        state = OrchestratorState(
            start_iteration=0,
            end_iteration=10,
            current_iteration=3,
            current_model_version=3,
            successful_iterations=3,
            total_cost_usd=5.0,
            win_rate_history=[0.55, 0.60, 0.58],
        )
        data = asdict(state)
        json_str = json.dumps(data)
        loaded = json.loads(json_str)

        # Roundtrip through JSON
        import dataclasses

        known_fields = {f.name for f in dataclasses.fields(OrchestratorState)}
        filtered = {k: v for k, v in loaded.items() if k in known_fields}
        restored = OrchestratorState(**filtered)
        assert restored.current_iteration == 3
        assert restored.win_rate_history == [0.55, 0.60, 0.58]

    def test_backwards_compatible_deserialization(self) -> None:
        """Loading state ignores unknown fields for schema evolution."""
        import dataclasses

        state_dict = {
            "start_iteration": 0,
            "end_iteration": 5,
            "current_iteration": 2,
            "current_model_version": 2,
            "unknown_future_field": "should be ignored",
        }
        known_fields = {f.name for f in dataclasses.fields(OrchestratorState)}
        filtered = {k: v for k, v in state_dict.items() if k in known_fields}
        state = OrchestratorState(**filtered)
        assert state.current_iteration == 2

    def test_blob_path_without_agent_or_run(self) -> None:
        """Blob path defaults to orchestrator/state.json."""
        state = OrchestratorState(
            start_iteration=0,
            end_iteration=5,
            current_iteration=0,
            current_model_version=0,
        )
        assert state._blob_path() == "orchestrator/state.json"

    def test_blob_path_with_agent_and_run(self) -> None:
        """Blob path includes agent_id and run_id when set."""
        state = OrchestratorState(
            start_iteration=0,
            end_iteration=5,
            current_iteration=0,
            current_model_version=0,
            agent_id="agent-a",
            run_id="run-001",
        )
        assert state._blob_path() == "orchestrator/agent-a/run-001/state.json"


class TestIterationResult:
    """Tests for the IterationResult dataclass."""

    def test_defaults(self) -> None:
        """IterationResult has sensible defaults."""
        result = IterationResult(iteration=5, model_version=10)
        assert result.win_rate == 0.0
        assert result.cost_usd == 0.0
        assert not result.rejected


# =============================================================================
# GCS Save/Load Tests (mocked)
# =============================================================================


class TestOrchestratorStatePersistence:
    """Tests for GCS-based state persistence."""

    def test_save_calls_gcs_backend(self) -> None:
        """save() uploads JSON to GCS."""
        state = OrchestratorState(
            start_iteration=0,
            end_iteration=5,
            current_iteration=2,
            current_model_version=2,
        )
        mock_config = MagicMock()
        mock_config.project_id = "test-project"
        mock_config.bucket = "test-bucket"

        with patch("alpha_zero.cloud.gcp.storage.GCSBackend") as mock_backend_cls:
            mock_backend = MagicMock()
            mock_backend_cls.return_value = mock_backend
            uri = state.save(mock_config)

        assert uri == "gs://test-bucket/orchestrator/state.json"
        mock_backend.upload_blob.assert_called_once()
        call_args = mock_backend.upload_blob.call_args
        assert call_args[0][0] == "test-bucket"
        assert call_args[0][1] == "orchestrator/state.json"
        # Verify JSON is valid
        data = json.loads(call_args[0][2].decode("utf-8"))
        assert data["current_iteration"] == 2

    def test_load_returns_state(self) -> None:
        """load() restores state from GCS."""
        state_dict = asdict(
            OrchestratorState(
                start_iteration=0,
                end_iteration=10,
                current_iteration=5,
                current_model_version=5,
            )
        )
        mock_config = MagicMock()
        mock_config.project_id = "test-project"
        mock_config.bucket = "test-bucket"

        with patch("alpha_zero.cloud.gcp.storage.GCSBackend") as mock_backend_cls:
            mock_backend = MagicMock()
            mock_backend.download_blob.return_value = json.dumps(state_dict).encode()
            mock_backend_cls.return_value = mock_backend

            loaded = OrchestratorState.load(mock_config)

        assert loaded is not None
        assert loaded.current_iteration == 5
        assert loaded.current_model_version == 5

    def test_load_returns_none_on_missing(self) -> None:
        """load() returns None when no state file exists."""
        mock_config = MagicMock()
        mock_config.project_id = "test-project"
        mock_config.bucket = "test-bucket"

        with patch("alpha_zero.cloud.gcp.storage.GCSBackend") as mock_backend_cls:
            mock_backend = MagicMock()
            mock_backend.download_blob.side_effect = Exception("Not found")
            mock_backend_cls.return_value = mock_backend

            loaded = OrchestratorState.load(mock_config)

        assert loaded is None

    def test_save_returns_none_on_failure(self) -> None:
        """save() returns None on upload failure without raising."""
        state = OrchestratorState(
            start_iteration=0,
            end_iteration=5,
            current_iteration=0,
            current_model_version=0,
        )
        mock_config = MagicMock()
        mock_config.project_id = "test-project"
        mock_config.bucket = "test-bucket"

        with patch("alpha_zero.cloud.gcp.storage.GCSBackend") as mock_backend_cls:
            mock_backend = MagicMock()
            mock_backend.upload_blob.side_effect = Exception("Upload failed")
            mock_backend_cls.return_value = mock_backend

            uri = state.save(mock_config)

        assert uri is None


# =============================================================================
# run_iterations() Dry-Run Integration Tests
# =============================================================================


@pytest.fixture()
def dry_run_args() -> argparse.Namespace:
    """Create minimal args for dry-run run_iterations()."""
    return argparse.Namespace(
        workers=2,
        games_per_worker=10,
        max_retries=1,
        trajectory_window=3,
        evaluation_games=0,
        mcts_type="gumbel",
        mcts_simulations=16,
        board_size=8,
        action_space_size=376,
        epochs=2,
        batch_size=32,
        learning_rate=0.001,
        max_turns=30,
        num_res_blocks=6,
        num_filters=64,
        reward_shaping=False,
        gumbel_num_actions=16,
        evaluation_win_threshold=0.55,
        agent_id=None,
        run_id="test-run",
    )


@pytest.fixture()
def mock_gcp_config() -> MagicMock:
    """Mock GCPConfig for dry-run tests."""
    config = MagicMock()
    config.project_id = "test-project"
    config.region = "us-central1"
    config.bucket = "test-bucket"
    config.service_account = None
    config.self_play.machine_type = "n1-standard-4"
    config.self_play.accelerator_type = "NVIDIA_TESLA_T4"
    config.self_play.accelerator_count = 1
    config.self_play.spot = True
    config.training.machine_type = "n1-standard-8"
    config.training.accelerator_type = "NVIDIA_TESLA_T4"
    config.training.accelerator_count = 1
    config.training.spot = False
    return config


class TestRunIterationsDryRun:
    """Tests for run_iterations() in dry-run mode."""

    @patch("alpha_zero.cloud.gcp.vertex_orchestrator.run_selfplay_phase", return_value=True)
    @patch("alpha_zero.cloud.gcp.vertex_orchestrator.run_training_phase", return_value=True)
    @patch("alpha_zero.cloud.gcp.cost.CostEstimator")
    def test_three_iteration_dry_run(
        self,
        mock_cost_cls: MagicMock,
        mock_train: MagicMock,
        mock_selfplay: MagicMock,
        dry_run_args: argparse.Namespace,
        mock_gcp_config: MagicMock,
    ) -> None:
        """Run 3 iterations in dry-run, verify state progression."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_iterations

        mock_cost = MagicMock()
        mock_cost.estimate_job_cost.return_value = 0.10
        mock_cost_cls.return_value = mock_cost

        client = MagicMock()
        state = run_iterations(
            config=mock_gcp_config,
            client=client,
            args=dry_run_args,
            start_iteration=0,
            end_iteration=3,
            initial_model_version=0,
            dry_run=True,
            resume=False,
        )

        assert state.current_iteration == 3
        assert state.is_complete
        assert state.successful_iterations == 3
        assert state.failed_iterations == 0
        assert state.current_model_version == 3
        assert len(state.results) == 3
        assert mock_selfplay.call_count == 3
        assert mock_train.call_count == 3

    @patch("alpha_zero.cloud.gcp.vertex_orchestrator.run_selfplay_phase", return_value=False)
    @patch("alpha_zero.cloud.gcp.cost.CostEstimator")
    def test_circuit_breaker_stops_after_failures(
        self,
        mock_cost_cls: MagicMock,
        mock_selfplay: MagicMock,
        dry_run_args: argparse.Namespace,
        mock_gcp_config: MagicMock,
    ) -> None:
        """Circuit breaker halts after max consecutive failures."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_iterations

        mock_cost = MagicMock()
        mock_cost.estimate_job_cost.return_value = 0.10
        mock_cost_cls.return_value = mock_cost

        client = MagicMock()
        state = run_iterations(
            config=mock_gcp_config,
            client=client,
            args=dry_run_args,
            start_iteration=0,
            end_iteration=10,
            initial_model_version=0,
            max_consecutive_failures=3,
            dry_run=True,
            resume=False,
        )

        assert state.successful_iterations == 0
        assert state.consecutive_failures == 3
        # Should stop at 3 failures, not process all 10
        assert mock_selfplay.call_count == 3

    @patch("alpha_zero.cloud.gcp.vertex_orchestrator.run_selfplay_phase", return_value=True)
    @patch("alpha_zero.cloud.gcp.vertex_orchestrator.run_training_phase", return_value=True)
    @patch("alpha_zero.cloud.gcp.vertex_orchestrator.run_evaluation_phase")
    @patch("alpha_zero.cloud.gcp.cost.CostEstimator")
    def test_evaluation_gate_rejects_model(
        self,
        mock_cost_cls: MagicMock,
        mock_eval: MagicMock,
        mock_train: MagicMock,
        mock_selfplay: MagicMock,
        dry_run_args: argparse.Namespace,
        mock_gcp_config: MagicMock,
    ) -> None:
        """Model is rejected when win rate is below threshold."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import run_iterations

        mock_cost = MagicMock()
        mock_cost.estimate_job_cost.return_value = 0.10
        mock_cost_cls.return_value = mock_cost

        # Win rate below threshold
        mock_eval.return_value = (True, 0.40)

        # Enable evaluation (need model_version > 0)
        dry_run_args.evaluation_games = 30

        client = MagicMock()
        state = run_iterations(
            config=mock_gcp_config,
            client=client,
            args=dry_run_args,
            start_iteration=1,
            end_iteration=2,
            initial_model_version=5,
            eval_win_threshold=0.55,
            dry_run=True,
            resume=False,
        )

        assert state.successful_iterations == 1
        # Model was rejected, version should stay at 5
        assert state.current_model_version == 5
        assert len(state.results) == 1
        assert state.results[0]["rejected"] is True
