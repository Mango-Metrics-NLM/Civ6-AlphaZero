"""Unit tests for the vertex orchestrator pipeline logic.

Tests the orchestrator's build_*_extra_args helpers, run phases, and
iteration flow using mocked Vertex AI clients.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.exceptions import GCSStorageError
from alpha_zero.cloud.gcp.vertex_orchestrator import (
    build_evaluation_extra_args,
    build_selfplay_extra_args,
    build_training_extra_args,
    get_initial_model_version,
    run_evaluation_phase,
    run_selfplay_phase,
    run_training_phase,
)

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
        container_image_uri="gcr.io/test/civ6az:latest",
    )


@pytest.fixture
def base_args() -> argparse.Namespace:
    """Create base orchestrator args with all expected fields."""
    return argparse.Namespace(
        iterations=3,
        start_iteration=0,
        workers=2,
        games_per_worker=10,
        epochs=5,
        trajectory_window=3,
        max_retries=1,
        board_size=8,
        action_space_size=376,
        num_res_blocks=6,
        num_filters=64,
        mcts_type="gumbel",
        mcts_simulations=16,
        gumbel_num_actions=16,
        max_turns=30,
        evaluation_games=10,
        evaluation_win_threshold=0.55,
        curriculum=False,
        curriculum_stages=3,
        reward_shaping=False,
        reward_scale=1.0,
        # Training hyperparameters
        lr_schedule_type="constant",
        lr_initial=0.001,
        lr_final=0.0001,
        lr_warmup_steps=1000,
        lr_decay_steps=100000,
        weight_decay=0.01,
        value_lr_multiplier=1.0,
        policy_lr_multiplier=1.0,
        dry_run=True,
    )


# =============================================================================
# build_*_extra_args Tests
# =============================================================================


class TestBuildExtraArgs:
    """Tests for the build_*_extra_args helper functions."""

    def test_selfplay_extra_args_contains_mcts_type(self, base_args: argparse.Namespace) -> None:
        """Selfplay args include MCTS configuration."""
        extra = build_selfplay_extra_args(base_args)

        assert any("--mcts_type=gumbel" in a for a in extra)
        assert any("--mcts_simulations=16" in a for a in extra)
        assert any("--board_size=8" in a for a in extra)

    def test_selfplay_extra_args_reward_shaping(self, base_args: argparse.Namespace) -> None:
        """Reward shaping flag is included when enabled."""
        base_args.reward_shaping = True
        base_args.reward_scale = 0.5
        extra = build_selfplay_extra_args(base_args)

        assert "--reward_shaping" in extra
        assert any("--reward_scale=0.5" in a for a in extra)

    def test_selfplay_extra_args_no_reward_shaping(self, base_args: argparse.Namespace) -> None:
        """Reward shaping is not included when disabled."""
        base_args.reward_shaping = False
        extra = build_selfplay_extra_args(base_args)

        assert "--reward_shaping" not in extra

    def test_training_extra_args(self, base_args: argparse.Namespace) -> None:
        """Training args include board_size and action_space_size."""
        extra = build_training_extra_args(base_args)

        assert any("--board_size=8" in a for a in extra)
        assert any("--action_space_size=376" in a for a in extra)

    def test_training_extra_args_never_emits_model_version(
        self, base_args: argparse.Namespace
    ) -> None:
        """Orchestrator no longer emits --model_version.

        The training worker dropped the flag in favour of checkpoint resumption
        via --resume_from (see vertex_orchestrator._resolve_resume_token and
        the gemini-code-assist review on PR #81). The kwarg is preserved on the
        function signature for backward compatibility but the value is never
        forwarded; both with-value and without-value calls must omit it.
        """
        without = build_training_extra_args(base_args)
        with_value = build_training_extra_args(base_args, model_version=42)
        assert not any("--model_version" in a for a in without)
        assert not any("--model_version" in a for a in with_value)

    def test_evaluation_extra_args(self, base_args: argparse.Namespace) -> None:
        """Evaluation args include MCTS and board config."""
        extra = build_evaluation_extra_args(base_args)

        assert any("--mcts_type=gumbel" in a for a in extra)
        assert any("--board_size=8" in a for a in extra)
        assert any("--max_turns=30" in a for a in extra)


# =============================================================================
# get_initial_model_version Tests
# =============================================================================


class TestGetInitialModelVersion:
    """Tests for model version detection."""

    def test_dry_run_returns_zero(self, gcp_config: GCPConfig) -> None:
        """Dry run always returns version 0."""
        version = get_initial_model_version(gcp_config, dry_run=True)
        assert version == 0

    def test_gcs_returns_latest(self, gcp_config: GCPConfig) -> None:
        """Returns the latest version from GCS."""
        mock_storage = MagicMock()
        mock_storage.get_latest_version.return_value = 5

        with patch(
            "alpha_zero.cloud.gcp.vertex_orchestrator.GCSModelStorage",
            return_value=mock_storage,
        ):
            version = get_initial_model_version(gcp_config, dry_run=False)

        assert version == 5

    def test_gcs_error_returns_zero(self, gcp_config: GCPConfig) -> None:
        """Returns 0 if GCS has no models."""
        mock_storage = MagicMock()
        mock_storage.get_latest_version.side_effect = GCSStorageError("No models found")

        with patch(
            "alpha_zero.cloud.gcp.vertex_orchestrator.GCSModelStorage",
            return_value=mock_storage,
        ):
            version = get_initial_model_version(gcp_config, dry_run=False)

        assert version == 0


# =============================================================================
# run_*_phase Dry Run Tests
# =============================================================================


class TestDryRunPhases:
    """Tests that dry-run phases complete without cloud calls."""

    def test_selfplay_dry_run(
        self,
        gcp_config: GCPConfig,
        base_args: argparse.Namespace,
    ) -> None:
        """Selfplay dry run returns True without launching jobs."""
        result = run_selfplay_phase(
            config=gcp_config,
            client=None,
            iteration=0,
            model_version=0,
            num_workers=2,
            games_per_worker=10,
            args=base_args,
            max_retries=1,
            dry_run=True,
        )
        assert result is True

    def test_training_dry_run(
        self,
        gcp_config: GCPConfig,
        base_args: argparse.Namespace,
    ) -> None:
        """Training dry run returns True without launching jobs."""
        result = run_training_phase(
            config=gcp_config,
            client=None,
            iteration=0,
            trajectory_iterations=[0],
            args=base_args,
            max_retries=1,
            dry_run=True,
        )
        assert result is True

    def test_training_dry_run_with_model_version(
        self,
        gcp_config: GCPConfig,
        base_args: argparse.Namespace,
    ) -> None:
        """Training dry run accepts model_version parameter."""
        result = run_training_phase(
            config=gcp_config,
            client=None,
            iteration=0,
            trajectory_iterations=[0],
            args=base_args,
            max_retries=1,
            dry_run=True,
            model_version=218,
        )
        assert result is True

    def test_evaluation_dry_run(
        self,
        gcp_config: GCPConfig,
        base_args: argparse.Namespace,
    ) -> None:
        """Evaluation dry run returns (True, 0.55) simulated win rate."""
        ok, win_rate = run_evaluation_phase(
            config=gcp_config,
            client=None,
            iteration=1,
            model_version_new=1,
            model_version_old=0,
            args=base_args,
            dry_run=True,
        )
        assert ok is True
        assert win_rate == 0.55

    def test_evaluation_skipped_when_zero_games(
        self,
        gcp_config: GCPConfig,
        base_args: argparse.Namespace,
    ) -> None:
        """Evaluation is skipped when evaluation_games=0."""
        base_args.evaluation_games = 0
        ok, win_rate = run_evaluation_phase(
            config=gcp_config,
            client=None,
            iteration=1,
            model_version_new=1,
            model_version_old=0,
            args=base_args,
            dry_run=False,
        )
        assert ok is True
        assert win_rate == 0.0


# =============================================================================
# GPU Config Tests
# =============================================================================


class TestGPUConfigCentralization:
    """Tests for centralized GPU pricing in config.py."""

    def test_default_gpu_costs_exists(self) -> None:
        """DEFAULT_GPU_COSTS dict exists in config module."""
        from alpha_zero.cloud.gcp.config import DEFAULT_GPU_COSTS

        assert isinstance(DEFAULT_GPU_COSTS, dict)
        assert "NVIDIA_TESLA_T4" in DEFAULT_GPU_COSTS

    def test_get_gpu_costs_returns_defaults(self) -> None:
        """get_gpu_costs returns defaults when no env vars set."""
        from alpha_zero.cloud.gcp.config import DEFAULT_GPU_COSTS, get_gpu_costs

        costs = get_gpu_costs()
        assert costs["NVIDIA_TESLA_T4"] == DEFAULT_GPU_COSTS["NVIDIA_TESLA_T4"]

    def test_get_gpu_costs_env_override(self) -> None:
        """get_gpu_costs respects environment variable overrides."""
        from alpha_zero.cloud.gcp.config import get_gpu_costs

        with patch.dict("os.environ", {"CIV6AZ_GPU_COST_T4": "0.99"}):
            costs = get_gpu_costs()

        assert costs["NVIDIA_TESLA_T4"] == 0.99
