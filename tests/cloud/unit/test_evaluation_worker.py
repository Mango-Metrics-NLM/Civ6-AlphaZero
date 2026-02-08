"""Unit tests for the evaluation worker module."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import torch

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.evaluation_worker import (
    load_model,
    parse_args,
    run_arena,
    save_results,
)
from alpha_zero.core.civ6_network import MicroCivNetwork

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gcp_config() -> GCPConfig:
    """Create test GCP config."""
    return GCPConfig(
        project_id="test-project",
        region="us-central1",
        bucket="test-bucket",
        container_image_uri="gcr.io/test-project/civ6az:latest",
    )


@pytest.fixture
def small_model() -> MicroCivNetwork:
    """Create a small model for testing."""
    return MicroCivNetwork(
        board_size=8,
        action_space_size=376,
        num_res_blocks=2,
        num_filters=16,
    )


# =============================================================================
# parse_args Tests
# =============================================================================


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_required_args(self) -> None:
        """Parses required arguments correctly."""
        with patch(
            "sys.argv",
            [
                "eval",
                "--model_version_new=2",
                "--model_version_old=1",
                "--iteration=5",
            ],
        ):
            args = parse_args()
            assert args.model_version_new == 2
            assert args.model_version_old == 1
            assert args.iteration == 5

    def test_default_values(self) -> None:
        """Default values are set correctly."""
        with patch(
            "sys.argv",
            [
                "eval",
                "--model_version_new=2",
                "--model_version_old=1",
                "--iteration=5",
            ],
        ):
            args = parse_args()
            assert args.games == 20
            assert args.mcts_simulations == 25
            assert args.mcts_type == "civ6"
            assert args.board_size == 8
            assert args.max_turns == 30
            assert args.dry_run is False

    def test_custom_values(self) -> None:
        """Custom argument values are parsed correctly."""
        with patch(
            "sys.argv",
            [
                "eval",
                "--model_version_new=3",
                "--model_version_old=2",
                "--iteration=10",
                "--games=50",
                "--mcts_type=gumbel",
                "--board_size=16",
                "--dry_run",
            ],
        ):
            args = parse_args()
            assert args.games == 50
            assert args.mcts_type == "gumbel"
            assert args.board_size == 16
            assert args.dry_run is True


# =============================================================================
# load_model Tests
# =============================================================================


class TestLoadModel:
    """Tests for model loading."""

    def test_dry_run_returns_fresh_model(self, gcp_config: GCPConfig) -> None:
        """Dry run creates a fresh model without GCS access."""
        model = load_model(
            gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=2,
            num_filters=16,
            dry_run=True,
        )

        assert isinstance(model, MicroCivNetwork)

    def test_dry_run_model_is_functional(self, gcp_config: GCPConfig) -> None:
        """Dry-run model can run forward pass."""
        model = load_model(
            gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=2,
            num_filters=16,
            dry_run=True,
        )

        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)
        action_mask = torch.ones(1, 376)

        policy, value = model(spatial, vector, action_mask)
        assert policy.shape == (1, 376)
        assert value.shape == (1,)


# =============================================================================
# run_arena Tests
# =============================================================================


class TestRunArena:
    """Tests for the arena evaluation logic."""

    @pytest.mark.slow
    def test_returns_valid_results(self, small_model: MicroCivNetwork) -> None:
        """Arena returns results with correct structure."""
        results = run_arena(
            model_new=small_model,
            model_old=small_model,
            num_games=2,
            mcts_type="gumbel",
            mcts_sims=4,
            gumbel_num_actions=8,
            board_size=8,
            max_turns=5,
            device=torch.device("cpu"),
        )

        assert "win_rate" in results
        assert "draw_rate" in results
        assert "loss_rate" in results
        assert "wins" in results
        assert "draws" in results
        assert "losses" in results
        assert "total_games" in results

    @pytest.mark.slow
    def test_rates_sum_to_one(self, small_model: MicroCivNetwork) -> None:
        """Win, draw, and loss rates sum to 1.0."""
        results = run_arena(
            model_new=small_model,
            model_old=small_model,
            num_games=2,
            mcts_type="gumbel",
            mcts_sims=4,
            gumbel_num_actions=8,
            board_size=8,
            max_turns=5,
            device=torch.device("cpu"),
        )

        total_rate = results["win_rate"] + results["draw_rate"] + results["loss_rate"]
        assert abs(total_rate - 1.0) < 1e-6

    @pytest.mark.slow
    def test_counts_match_total(self, small_model: MicroCivNetwork) -> None:
        """Win + draw + loss counts match total_games."""
        num_games = 4
        results = run_arena(
            model_new=small_model,
            model_old=small_model,
            num_games=num_games,
            mcts_type="gumbel",
            mcts_sims=4,
            gumbel_num_actions=8,
            board_size=8,
            max_turns=5,
            device=torch.device("cpu"),
        )

        assert results["wins"] + results["draws"] + results["losses"] == num_games
        assert results["total_games"] == num_games

    @pytest.mark.slow
    def test_same_model_gets_reasonable_results(self, small_model: MicroCivNetwork) -> None:
        """Same model vs itself should give roughly balanced results."""
        results = run_arena(
            model_new=small_model,
            model_old=small_model,
            num_games=4,
            mcts_type="gumbel",
            mcts_sims=4,
            gumbel_num_actions=8,
            board_size=8,
            max_turns=5,
            device=torch.device("cpu"),
        )

        # Not strictly deterministic, but rates should be valid numbers
        assert 0.0 <= results["win_rate"] <= 1.0
        assert 0.0 <= results["draw_rate"] <= 1.0
        assert 0.0 <= results["loss_rate"] <= 1.0


# =============================================================================
# save_results Tests
# =============================================================================


class TestSaveResults:
    """Tests for saving evaluation results."""

    def test_dry_run_returns_none(self, gcp_config: GCPConfig) -> None:
        """Dry run doesn't save and returns None."""
        results = {
            "win_rate": 0.6,
            "draw_rate": 0.1,
            "loss_rate": 0.3,
        }

        uri = save_results(
            config=gcp_config,
            results=results,
            iteration=5,
            model_version_new=2,
            model_version_old=1,
            dry_run=True,
        )

        assert uri is None

    def test_saves_to_gcs(self, gcp_config: GCPConfig) -> None:
        """Non-dry-run saves results to GCS."""
        results = {
            "win_rate": 0.6,
            "draw_rate": 0.1,
            "loss_rate": 0.3,
        }

        mock_backend = MagicMock()
        mock_backend.upload_blob.return_value = "gs://test-bucket/evaluations/iter_005/results.json"

        with patch("alpha_zero.cloud.gcp.storage.GCSBackend", return_value=mock_backend):
            uri = save_results(
                config=gcp_config,
                results=results,
                iteration=5,
                model_version_new=2,
                model_version_old=1,
                dry_run=False,
            )

        assert uri is not None
        mock_backend.upload_blob.assert_called_once()

        # Verify the data includes metadata
        call_args = mock_backend.upload_blob.call_args
        saved_data = json.loads(call_args[0][2].decode("utf-8"))
        assert saved_data["win_rate"] == 0.6
        assert saved_data["iteration"] == 5
        assert saved_data["model_version_new"] == 2
        assert saved_data["model_version_old"] == 1


# =============================================================================
# VertexEvaluationJob Tests
# =============================================================================


class TestVertexEvaluationJob:
    """Tests for the evaluation job Vertex AI wrapper."""

    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock Vertex AI client."""
        from alpha_zero.cloud.gcp.vertex_jobs import JobResult, JobState

        client = MagicMock()
        client.submit_custom_job.return_value = "projects/test/locations/us/customJobs/123"
        client.wait_for_completion.return_value = JobResult(
            job_id="projects/test/locations/us/customJobs/123",
            job_name="civ6az-eval",
            state=JobState.SUCCEEDED.value,
            success=True,
        )
        return client

    def test_launch_creates_job(self, gcp_config: GCPConfig, mock_client: MagicMock) -> None:
        """Launch creates evaluation job with correct args."""
        from alpha_zero.cloud.gcp.vertex_jobs import VertexEvaluationJob

        job = VertexEvaluationJob(gcp_config, client=mock_client)
        job_name = job.launch(
            model_version_new=2,
            model_version_old=1,
            iteration=5,
            num_games=20,
        )

        assert job_name == "projects/test/locations/us/customJobs/123"
        mock_client.submit_custom_job.assert_called_once()

        # Verify the spec args contain evaluation mode
        spec = mock_client.submit_custom_job.call_args[0][0]
        assert "--mode=evaluation" in spec.args
        assert "--model_version_new=2" in spec.args
        assert "--model_version_old=1" in spec.args
        assert "--games=20" in spec.args

    def test_launch_uses_cpu_only(self, gcp_config: GCPConfig, mock_client: MagicMock) -> None:
        """Evaluation jobs use CPU-only instances."""
        from alpha_zero.cloud.gcp.vertex_jobs import VertexEvaluationJob

        job = VertexEvaluationJob(gcp_config, client=mock_client)
        job.launch(
            model_version_new=2,
            model_version_old=1,
            iteration=5,
        )

        spec = mock_client.submit_custom_job.call_args[0][0]
        assert spec.accelerator_count == 0
        assert spec.machine_type == "n1-standard-4"

    def test_launch_passes_extra_args(self, gcp_config: GCPConfig, mock_client: MagicMock) -> None:
        """Extra args are appended to container args."""
        from alpha_zero.cloud.gcp.vertex_jobs import VertexEvaluationJob

        job = VertexEvaluationJob(gcp_config, client=mock_client)
        job.launch(
            model_version_new=2,
            model_version_old=1,
            iteration=5,
            extra_args=["--mcts_type=gumbel", "--board_size=16"],
        )

        spec = mock_client.submit_custom_job.call_args[0][0]
        assert "--mcts_type=gumbel" in spec.args
        assert "--board_size=16" in spec.args

    def test_launch_no_container_raises(self, mock_client: MagicMock) -> None:
        """Missing container image URI raises VertexJobError."""
        from alpha_zero.cloud.gcp.exceptions import VertexJobError
        from alpha_zero.cloud.gcp.vertex_jobs import VertexEvaluationJob

        config = GCPConfig(
            project_id="test",
            bucket="test-bucket",
            container_image_uri=None,
        )
        job = VertexEvaluationJob(config, client=mock_client)

        with pytest.raises(VertexJobError, match="container_image_uri"):
            job.launch(model_version_new=2, model_version_old=1, iteration=5)

    def test_wait_for_completion(self, gcp_config: GCPConfig, mock_client: MagicMock) -> None:
        """wait_for_completion returns success result."""
        from alpha_zero.cloud.gcp.vertex_jobs import VertexEvaluationJob

        job = VertexEvaluationJob(gcp_config, client=mock_client)
        job_name = job.launch(model_version_new=2, model_version_old=1, iteration=5)
        result = job.wait_for_completion(job_name)

        assert result.success
