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
    save_results,
)
from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

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
def small_model() -> Civ6TransformerNetwork:
    """Create a small model for testing."""
    config = TransformerConfig(
        spatial_size=8,
        spatial_channels=13,
        vector_dim=64,
        action_space_size=376,
        num_layers=2,
        embed_dim=64,
    )
    return Civ6TransformerNetwork(config=config)


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
            assert args.mcts_type == "gumbel"
            assert args.board_size == 8
            assert args.max_turns == 30
            assert args.action_space_size == 376
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

    def test_dry_run_returns_transformer_model(self, gcp_config: GCPConfig) -> None:
        """Dry run creates a fresh transformer model without GCS access."""
        model = load_model(
            gcp_config,
            version=1,
            dry_run=True,
        )

        assert isinstance(model, Civ6TransformerNetwork)

    def test_dry_run_model_forward_pass(self, gcp_config: GCPConfig) -> None:
        """Dry-run model can run forward pass with correct shapes."""
        model = load_model(
            gcp_config,
            version=1,
            dry_run=True,
        )

        # Use TransformerConfig defaults: spatial_size=16, action_space_size=381
        default_cfg = TransformerConfig()
        spatial = torch.randn(
            1, default_cfg.spatial_channels, default_cfg.spatial_size, default_cfg.spatial_size
        )
        vector = torch.randn(1, default_cfg.vector_dim)
        action_mask = torch.ones(1, default_cfg.action_space_size)

        policy, value = model(spatial, vector, action_mask)
        assert policy.shape == (1, default_cfg.action_space_size)
        assert value.shape == (1,)

    def test_load_model_with_config(self, gcp_config: GCPConfig) -> None:
        """Load model with model_config dict from GCS."""
        mock_storage = MagicMock()

        # Create state dict with explicit config
        config = TransformerConfig(
            embed_dim=64, num_layers=2, spatial_size=8, action_space_size=376
        )
        model = Civ6TransformerNetwork(config=config)
        state_dict = model.state_dict()

        # Model config dict matching the saved model
        model_config_dict = {
            "spatial_size": 8,
            "spatial_channels": 13,
            "vector_dim": 64,
            "action_space_size": 376,
            "embed_dim": 64,
            "num_layers": 2,
        }

        mock_storage.load_model.return_value = (state_dict, {"iteration": 5}, model_config_dict)

        with patch(
            "alpha_zero.cloud.gcp.evaluation_worker.GCSModelStorage", return_value=mock_storage
        ):
            loaded_model = load_model(gcp_config, version=5, dry_run=False)

        assert isinstance(loaded_model, Civ6TransformerNetwork)
        assert loaded_model.config.embed_dim == 64
        assert loaded_model.config.action_space_size == 376
        mock_storage.load_model.assert_called_once_with(5)

    def test_load_model_legacy_no_config(self, gcp_config: GCPConfig) -> None:
        """Load legacy model without model_config (falls back to pipeline defaults)."""
        mock_storage = MagicMock()

        # Create state dict matching pipeline architecture (NetworkArchitectureConfig defaults)
        pipeline_config = TransformerConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            num_layers=6,
            embed_dim=64,
        )
        model = Civ6TransformerNetwork(config=pipeline_config)
        state_dict = model.state_dict()

        # Legacy models return None for model_config
        mock_storage.load_model.return_value = (state_dict, {"iteration": 1}, None)

        with patch(
            "alpha_zero.cloud.gcp.evaluation_worker.GCSModelStorage", return_value=mock_storage
        ):
            loaded_model = load_model(gcp_config, version=1, dry_run=False)

        assert isinstance(loaded_model, Civ6TransformerNetwork)
        # Should use pipeline architecture defaults (from NetworkArchitectureConfig)
        assert loaded_model.config.embed_dim == 64
        assert loaded_model.config.action_space_size == 376

    def test_load_model_handles_enum_strings(self, gcp_config: GCPConfig) -> None:
        """Load model with enum values as strings (converts correctly)."""
        from alpha_zero.core.transformer_network import AttentionType, NormType

        mock_storage = MagicMock()

        config = TransformerConfig()
        model = Civ6TransformerNetwork(config=config)
        state_dict = model.state_dict()

        # Model config with enum values as strings (matching default TransformerConfig)
        model_config_dict = {
            "spatial_channels": config.spatial_channels,
            "spatial_size": config.spatial_size,
            "vector_dim": config.vector_dim,
            "action_space_size": config.action_space_size,
            "embed_dim": config.embed_dim,
            "num_layers": config.num_layers,
            "norm_type": "layer_norm",  # String instead of NormType enum
            "attention_type": "standard",  # String instead of AttentionType enum
        }

        mock_storage.load_model.return_value = (state_dict, {}, model_config_dict)

        with patch(
            "alpha_zero.cloud.gcp.evaluation_worker.GCSModelStorage", return_value=mock_storage
        ):
            loaded_model = load_model(gcp_config, version=1, dry_run=False)

        assert isinstance(loaded_model, Civ6TransformerNetwork)
        # Verify enums were converted
        assert isinstance(loaded_model.config.norm_type, NormType)
        assert isinstance(loaded_model.config.attention_type, AttentionType)

    def test_load_model_removes_computed_fields(self, gcp_config: GCPConfig) -> None:
        """Load model with computed fields in config (removes them)."""
        mock_storage = MagicMock()

        config = TransformerConfig()
        model = Civ6TransformerNetwork(config=config)
        state_dict = model.state_dict()

        # Model config with computed fields that should be removed
        model_config_dict = {
            "spatial_channels": config.spatial_channels,
            "spatial_size": config.spatial_size,
            "vector_dim": config.vector_dim,
            "action_space_size": config.action_space_size,
            "embed_dim": config.embed_dim,
            "num_layers": config.num_layers,
            "_num_patches": 64,  # Computed field — should be removed
            "_head_dim": 16,  # Computed field — should be removed
        }

        mock_storage.load_model.return_value = (state_dict, {}, model_config_dict)

        with patch(
            "alpha_zero.cloud.gcp.evaluation_worker.GCSModelStorage", return_value=mock_storage
        ):
            loaded_model = load_model(gcp_config, version=1, dry_run=False)

        assert isinstance(loaded_model, Civ6TransformerNetwork)
        # Should succeed despite computed fields being present

    def test_load_model_bad_config_falls_back(self, gcp_config: GCPConfig) -> None:
        """Load model with invalid config keys (falls back to pipeline defaults)."""
        mock_storage = MagicMock()

        # Create state dict matching pipeline architecture (what fallback will use)
        pipeline_config = TransformerConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            action_space_size=376,
            num_layers=6,
            embed_dim=64,
        )
        model = Civ6TransformerNetwork(config=pipeline_config)
        state_dict = model.state_dict()

        # Model config with invalid keys — will trigger TypeError fallback
        model_config_dict = {
            "board_size": 8,
            "invalid_key": "invalid_value",
            "another_bad_key": 42,
        }

        mock_storage.load_model.return_value = (state_dict, {}, model_config_dict)

        with patch(
            "alpha_zero.cloud.gcp.evaluation_worker.GCSModelStorage", return_value=mock_storage
        ):
            loaded_model = load_model(gcp_config, version=1, dry_run=False)

        assert isinstance(loaded_model, Civ6TransformerNetwork)
        # Should fall back to pipeline defaults due to TypeError
        assert loaded_model.config.embed_dim == 64
        assert loaded_model.config.action_space_size == 376


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
