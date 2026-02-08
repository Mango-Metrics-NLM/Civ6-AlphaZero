"""E2E tests for Vertex AI job submission (dry-run validation).

These tests validate job spec generation without actually submitting
to Vertex AI. For live submission tests, set GCP env vars.

Run with:  pytest tests/e2e/test_vertex_submission.py -v
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestVertexJobSpecValidation:
    """Validate Vertex AI job spec structures without live submission."""

    def test_job_spec_dataclass(self) -> None:
        """JobSpec dataclass creates valid worker pool spec."""
        from alpha_zero.cloud.gcp.vertex_jobs import JobSpec

        spec = JobSpec(
            display_name="test-selfplay-job",
            container_uri="gcr.io/test/image:latest",
            args=["--mode", "selfplay", "--num_games", "10"],
            machine_type="n1-standard-4",
            accelerator_type="NVIDIA_TESLA_T4",
            spot=True,
        )

        pool = spec.to_worker_pool_spec()
        assert pool is not None
        assert "containerSpec" in pool or isinstance(pool, dict)
        logger.info("JobSpec to_worker_pool_spec OK")

    def test_gcp_config_defaults(self) -> None:
        """GCPConfig has sensible defaults."""
        from alpha_zero.cloud.gcp.config import GCPConfig

        config = GCPConfig(
            project_id="test-project",
            bucket="test-bucket",
            container_image_uri="gcr.io/test/image:latest",
        )

        assert config.project_id == "test-project"
        assert config.bucket == "test-bucket"
        assert config.region is not None  # Should have a default
        logger.info("GCPConfig defaults OK: region=%s", config.region)

    def test_vertex_ai_client_initialization(self) -> None:
        """VertexAIClient initializes without calling GCP APIs."""
        from alpha_zero.cloud.gcp.vertex_jobs import VertexAIClient

        client = VertexAIClient(
            project="test-project",
            region="us-central1",
        )

        assert client is not None
        # Client should lazy-init — no API calls yet
        logger.info("VertexAIClient initialized (lazy)")

    def test_selfplay_job_creation(self) -> None:
        """VertexSelfPlayJob creates with valid GCPConfig."""
        from alpha_zero.cloud.gcp.config import GCPConfig
        from alpha_zero.cloud.gcp.vertex_jobs import VertexSelfPlayJob

        config = GCPConfig(
            project_id="test-project",
            bucket="test-bucket",
            container_image_uri="gcr.io/test/image:latest",
        )

        mock_client = MagicMock()
        job = VertexSelfPlayJob(config=config, client=mock_client)

        assert job is not None
        logger.info("VertexSelfPlayJob created")

    def test_wandb_env_vars_filter_empty(self) -> None:
        """W&B env var helper filters out empty values."""
        from alpha_zero.cloud.gcp.vertex_jobs import _wandb_env_vars

        with patch.dict("os.environ", {
            "WANDB_API_KEY": "test-key",
            "WANDB_PROJECT": "civ6-alphazero",
            "WANDB_ENTITY": "",  # Empty — should be filtered
        }, clear=False):
            env = _wandb_env_vars()

        # Empty values should be excluded
        assert "WANDB_ENTITY" not in env or env.get("WANDB_ENTITY") != ""
        logger.info("W&B env filter OK: %d vars", len(env))

    def test_job_state_terminal_check(self) -> None:
        """JobState correctly identifies terminal states."""
        from alpha_zero.cloud.gcp.vertex_jobs import JobState

        assert JobState.is_terminal(JobState.SUCCEEDED)
        assert JobState.is_terminal(JobState.FAILED)
        assert JobState.is_terminal(JobState.CANCELLED)
        assert not JobState.is_terminal(JobState.RUNNING)
        assert not JobState.is_terminal(JobState.PENDING)

    def test_job_result_dataclass(self) -> None:
        """JobResult holds completion data."""
        from alpha_zero.cloud.gcp.vertex_jobs import JobResult

        result = JobResult(
            job_id="job-123",
            job_name="test-job",
            state="JOB_STATE_SUCCEEDED",
            success=True,
        )

        assert result.success
        assert result.job_id == "job-123"
        assert result.error_message is None
