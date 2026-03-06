"""Integration smoke test for batch self-play enablement (Sprint 6 Phase 1).

Tests that the orchestrator correctly propagates batch self-play flags
(--num_parallel_games, --max_batch_size) from config defaults through
to Vertex AI job submissions.

This validates Phase 16 batch infrastructure + Sprint 6 default enablement.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.gcp.config import (
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_NUM_PARALLEL_GAMES,
    GCPConfig,
)
from alpha_zero.cloud.gcp.vertex_jobs import JobSpec, JobState, VertexAIClient
from alpha_zero.cloud.gcp.vertex_orchestrator import build_selfplay_extra_args, parse_args

# =============================================================================
# Mock Vertex AI Client (simplified for smoke test)
# =============================================================================


class MockVertexAIClient(VertexAIClient):
    """Mock Vertex AI client that captures job submission args."""

    def __init__(self) -> None:
        self._submitted_jobs: list[dict[str, Any]] = []
        self._job_counter = 0

    def _ensure_initialized(self) -> None:
        """Override to do nothing."""
        pass

    def submit_custom_job(
        self,
        spec: JobSpec,
        staging_bucket: str | None = None,
    ) -> str:
        """Capture job submission for inspection."""
        self._job_counter += 1
        job_name = f"projects/test/locations/us-central1/customJobs/{self._job_counter}"

        # Store the job spec for assertion
        self._submitted_jobs.append(
            {
                "job_name": job_name,
                "display_name": spec.display_name,
                "args": spec.args or [],
                "env_vars": spec.env_vars or {},
            }
        )

        return job_name

    def get_last_submitted_job(self) -> dict[str, Any] | None:
        """Get the most recently submitted job."""
        return self._submitted_jobs[-1] if self._submitted_jobs else None

    def get_job_state(self, job_name: str) -> tuple[str, dict[str, Any]]:
        """Mock implementation."""
        return JobState.SUCCEEDED.value, {}

    def cancel_job(self, job_name: str) -> None:
        """Mock implementation."""
        pass


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
        container_image_uri="us-central1-docker.pkg.dev/test-project/civ6az:v1.5.0",
    )


@pytest.fixture
def mock_vertex_client() -> MockVertexAIClient:
    """Create mock Vertex AI client."""
    return MockVertexAIClient()


# =============================================================================
# Smoke Tests
# =============================================================================


class TestBatchSelfPlayDefaults:
    """Test that batch self-play defaults are enabled and propagate correctly."""

    def test_config_defaults_are_batch_enabled(self) -> None:
        """Verify config.py constants enable batch self-play (Sprint 6 Phase 1)."""
        # These defaults were changed from 1 → 4 and 64 → 32 in Sprint 6
        assert DEFAULT_NUM_PARALLEL_GAMES == 4, (
            "Expected batch self-play enabled by default (num_parallel_games=4). "
            "If this fails, check alpha_zero/cloud/gcp/config.py"
        )
        assert DEFAULT_MAX_BATCH_SIZE == 32, (
            "Expected max_batch_size=32 default. "
            "If this fails, check alpha_zero/cloud/gcp/config.py"
        )

    def test_orchestrator_argparse_uses_batch_defaults(self) -> None:
        """Verify orchestrator parse_args() picks up batch defaults."""
        with patch("sys.argv", ["orchestrator"]):
            args = parse_args()

        assert args.num_parallel_games == 4, (
            "Orchestrator should default to num_parallel_games=4. "
            "Check vertex_orchestrator.py argparse defaults."
        )
        assert args.max_batch_size == 32, (
            "Orchestrator should default to max_batch_size=32. "
            "Check vertex_orchestrator.py argparse defaults."
        )


class TestBatchArgsPropagation:
    """Test that batch args propagate through build_selfplay_extra_args()."""

    def test_batch_args_included_when_enabled(self) -> None:
        """When num_parallel_games>1, both batch flags should be in extra args."""
        # Simulate orchestrator args with batch enabled
        mock_args = MagicMock()
        mock_args.num_parallel_games = 4
        mock_args.max_batch_size = 32
        mock_args.board_size = 8
        mock_args.action_space_size = 376
        mock_args.num_res_blocks = 6
        mock_args.num_filters = 64
        mock_args.num_heads = 8
        mock_args.policy_hidden_dim = 128
        mock_args.value_head_dim = 64
        mock_args.mcts_type = "gumbel"
        mock_args.mcts_simulations = 64
        mock_args.gumbel_num_actions = 16
        mock_args.max_turns = 50
        mock_args.curriculum = False
        mock_args.reward_shaping = False

        extra_args = build_selfplay_extra_args(mock_args)

        # Assert both batch flags are present
        assert "--num_parallel_games=4" in extra_args, (
            f"Expected --num_parallel_games=4 in extra_args, got: {extra_args}"
        )
        assert "--max_batch_size=32" in extra_args, (
            f"Expected --max_batch_size=32 in extra_args, got: {extra_args}"
        )

    def test_batch_args_omitted_when_sequential(self) -> None:
        """When num_parallel_games=1, batch args should be omitted (backward compat)."""
        mock_args = MagicMock()
        mock_args.num_parallel_games = 1  # Sequential mode
        mock_args.max_batch_size = 64
        mock_args.board_size = 8
        mock_args.action_space_size = 376
        mock_args.num_res_blocks = 6
        mock_args.num_filters = 64
        mock_args.num_heads = 8
        mock_args.policy_hidden_dim = 128
        mock_args.value_head_dim = 64
        mock_args.mcts_type = "gumbel"
        mock_args.mcts_simulations = 64
        mock_args.gumbel_num_actions = 16
        mock_args.max_turns = 50
        mock_args.curriculum = False
        mock_args.reward_shaping = False

        extra_args = build_selfplay_extra_args(mock_args)

        # Assert batch flags are NOT present (sequential mode)
        args_str = " ".join(extra_args)
        assert "--num_parallel_games=" not in args_str, (
            f"Sequential mode should not emit --num_parallel_games, got: {extra_args}"
        )
        assert "--max_batch_size=" not in args_str, (
            f"Sequential mode should not emit --max_batch_size, got: {extra_args}"
        )


class TestEndToEndJobSubmission:
    """Integration test: orchestrator → job spec → Vertex AI submission."""

    def test_batch_args_reach_vertex_job_spec(
        self, mock_vertex_client: MockVertexAIClient, gcp_config: GCPConfig
    ) -> None:
        """Verify batch flags end up in actual Vertex AI job submission."""
        # Create a VertexSelfPlayJob
        from alpha_zero.cloud.gcp.vertex_jobs import VertexSelfPlayJob

        job = VertexSelfPlayJob(
            config=gcp_config,
            client=mock_vertex_client,
        )

        # Build extra args with batch enabled (using defaults from Sprint 6)
        mock_args = MagicMock()
        mock_args.num_parallel_games = 4
        mock_args.max_batch_size = 32
        mock_args.board_size = 8
        mock_args.action_space_size = 376
        mock_args.num_res_blocks = 6
        mock_args.num_filters = 64
        mock_args.num_heads = 8
        mock_args.policy_hidden_dim = 128
        mock_args.value_head_dim = 64
        mock_args.mcts_type = "gumbel"
        mock_args.mcts_simulations = 64
        mock_args.gumbel_num_actions = 16
        mock_args.max_turns = 50
        mock_args.curriculum = False
        mock_args.reward_shaping = False

        extra_args = build_selfplay_extra_args(mock_args)

        # Launch the job (mocked) - VertexSelfPlayJob.launch() creates N workers
        job_names = job.launch(
            num_workers=1,
            model_version=218,
            iteration=0,
            extra_args=extra_args,
        )

        # Verify job was launched
        assert len(job_names) == 1, "Expected 1 job to be launched"

        # Retrieve the submitted job spec from mock
        submitted_job = mock_vertex_client.get_last_submitted_job()
        assert submitted_job is not None, "No job was submitted to mock client"

        # Assert batch flags are in the job args
        job_args = submitted_job["args"]
        assert "--num_parallel_games=4" in job_args, (
            f"Expected --num_parallel_games=4 in Vertex job args, got: {job_args}"
        )
        assert "--max_batch_size=32" in job_args, (
            f"Expected --max_batch_size=32 in Vertex job args, got: {job_args}"
        )

        # Additional sanity checks
        assert submitted_job["display_name"] == "civ6az-selfplay-iter000-worker00"
        assert "args" in submitted_job
        assert isinstance(job_args, list)

        # Verify standard self-play args are also present
        assert any("--worker_id=" in arg for arg in job_args)
        assert any("--model_version=" in arg for arg in job_args)
        assert any("--iteration=" in arg for arg in job_args)
