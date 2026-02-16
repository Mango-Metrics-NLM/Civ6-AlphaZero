"""
Unit tests for MuZero self-play worker paths.

Tests the MuZero-specific functions in selfplay_worker.py:
- parse_args with mcts_type=muzero
- load_muzero_model
- run_muzero_selfplay
- main() routing to MuZero path
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
import torch

from alpha_zero.cloud.gcp.config import GCPConfig, GCSPathConfig

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
        paths=GCSPathConfig(),
    )


# =============================================================================
# parse_args — MuZero flag Tests
# =============================================================================


class TestParseArgsMuZero:
    """Test that --mcts_type=muzero is accepted by arg parser."""

    def test_muzero_mcts_type_accepted(self) -> None:
        """Test parsing with mcts_type=muzero."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        test_args = [
            "selfplay_worker",
            "--worker_id=1",
            "--model_version=1",
            "--iteration=1",
            "--mcts_type=muzero",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.mcts_type == "muzero"

    def test_muzero_mcts_type_with_other_args(self) -> None:
        """Test muzero type alongside other MCTS arguments."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        test_args = [
            "selfplay_worker",
            "--worker_id=1",
            "--model_version=5",
            "--iteration=10",
            "--mcts_type=muzero",
            "--mcts_simulations=200",
            "--max_turns=50",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.mcts_type == "muzero"
        assert args.mcts_simulations == 200
        assert args.max_turns == 50

    def test_default_mcts_type_is_not_muzero(self) -> None:
        """Test that default mcts_type is NOT muzero (backwards compat)."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        test_args = [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.mcts_type != "muzero"


# =============================================================================
# load_muzero_model Tests
# =============================================================================


class TestLoadMuZeroModel:
    """Tests for MuZero model loading."""

    def test_dry_run_creates_muzero_model(self, gcp_config: GCPConfig) -> None:
        """Test that dry run creates a functional MuZero model."""
        from alpha_zero.cloud.gcp.selfplay_worker import load_muzero_model

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        assert model is not None
        # Verify it's a MuZero network
        from alpha_zero.core.muzero_network import MuZeroNetwork

        assert isinstance(model, MuZeroNetwork)

    def test_dry_run_model_can_forward(self, gcp_config: GCPConfig) -> None:
        """Test that dry-run MuZero model can perform forward pass."""
        from alpha_zero.cloud.gcp.selfplay_worker import load_muzero_model

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        model.eval()
        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)

        with torch.no_grad():
            policy, value = model(spatial, vector)

        assert policy.shape[0] == 1
        assert value.shape == (1,)

    def test_dry_run_model_initial_inference(self, gcp_config: GCPConfig) -> None:
        """Test that dry-run MuZero model can do initial inference."""
        from alpha_zero.cloud.gcp.selfplay_worker import load_muzero_model

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        model.eval()
        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)

        with torch.no_grad():
            output = model.initial_inference(spatial, vector)

        assert output.hidden_state is not None
        assert output.policy_logits.shape[1] == 376
        assert output.value.shape == (1,)

    def test_dry_run_model_recurrent_inference(self, gcp_config: GCPConfig) -> None:
        """Test that dry-run MuZero model can do recurrent inference."""
        from alpha_zero.cloud.gcp.selfplay_worker import load_muzero_model

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        model.eval()
        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)

        with torch.no_grad():
            output = model.initial_inference(spatial, vector)
            action = torch.tensor([0])
            next_hidden, reward, policy, value = model.recurrent_inference(
                output.hidden_state, action
            )

        assert next_hidden.shape == output.hidden_state.shape
        assert policy.shape[1] == 376

    def test_custom_architecture_params(self, gcp_config: GCPConfig) -> None:
        """Test dry run with custom architecture parameters."""
        from alpha_zero.cloud.gcp.selfplay_worker import load_muzero_model

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=16,
            action_space_size=500,
            num_res_blocks=4,
            num_filters=32,
            dry_run=True,
        )

        assert model.config.spatial_size == 16
        assert model.config.action_space_size == 500
        assert model.config.num_res_blocks == 4
        assert model.config.hidden_state_channels == 32


# =============================================================================
# run_muzero_selfplay Tests
# =============================================================================


class TestRunMuZeroSelfplay:
    """Tests for MuZero self-play execution."""

    def test_dry_run_returns_trajectories(self, gcp_config: GCPConfig) -> None:
        """Test that MuZero dry run returns non-empty trajectories."""
        from alpha_zero.cloud.gcp.selfplay_worker import (
            load_muzero_model,
            run_muzero_selfplay,
        )

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        trajectories = run_muzero_selfplay(
            model=model,
            num_games=3,
            mcts_sims=10,
            board_size=8,
            max_turns=30,
            dry_run=True,
        )

        assert len(trajectories) == 3

    def test_dry_run_trajectory_format(self, gcp_config: GCPConfig) -> None:
        """Test that MuZero dry run trajectory has expected keys."""
        from alpha_zero.cloud.gcp.selfplay_worker import (
            load_muzero_model,
            run_muzero_selfplay,
        )

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        trajectories = run_muzero_selfplay(
            model=model,
            num_games=1,
            mcts_sims=10,
            board_size=8,
            max_turns=30,
            dry_run=True,
        )

        traj = trajectories[0]
        # Standard keys
        assert "spatial" in traj
        assert "vector" in traj
        assert "policies" in traj
        assert "actions" in traj
        assert "rewards" in traj
        # MuZero-specific keys
        assert "reward_targets" in traj
        assert "value_targets" in traj

    def test_dry_run_trajectory_lengths_consistent(self, gcp_config: GCPConfig) -> None:
        """Test that all trajectory arrays have consistent lengths."""
        from alpha_zero.cloud.gcp.selfplay_worker import (
            load_muzero_model,
            run_muzero_selfplay,
        )

        model = load_muzero_model(
            config=gcp_config,
            version=1,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            dry_run=True,
        )

        trajectories = run_muzero_selfplay(
            model=model,
            num_games=2,
            mcts_sims=10,
            board_size=8,
            max_turns=30,
            dry_run=True,
        )

        for traj in trajectories:
            n = len(traj["spatial"])
            assert len(traj["vector"]) == n
            assert len(traj["policies"]) == n
            assert len(traj["rewards"]) == n


# =============================================================================
# main() MuZero Routing Tests
# =============================================================================


class TestMainMuZeroRouting:
    """Tests for the main() function MuZero routing."""

    def test_main_dry_run_muzero_success(self) -> None:
        """Test that main() with --mcts_type=muzero --dry_run completes."""
        from alpha_zero.cloud.gcp.selfplay_worker import main

        test_args = [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
            "--games=2",
            "--mcts_simulations=10",
            "--mcts_type=muzero",
            "--dry_run",
        ]

        with patch.object(sys, "argv", test_args):
            exit_code = main()

        assert exit_code == 0

    def test_main_muzero_error_returns_1(self) -> None:
        """Test that MuZero errors return code 1."""
        from alpha_zero.cloud.gcp.selfplay_worker import main

        test_args = [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
            "--mcts_type=muzero",
            "--dry_run",
        ]

        with patch.object(sys, "argv", test_args):
            with patch(
                "alpha_zero.cloud.gcp.selfplay_worker.load_muzero_model",
                side_effect=RuntimeError("MuZero model load failed"),
            ):
                exit_code = main()

        assert exit_code == 1

    def test_main_muzero_keyboard_interrupt(self) -> None:
        """Test that KeyboardInterrupt in MuZero path returns 130."""
        from alpha_zero.cloud.gcp.selfplay_worker import main

        test_args = [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
            "--mcts_type=muzero",
            "--dry_run",
        ]

        with patch.object(sys, "argv", test_args):
            with patch(
                "alpha_zero.cloud.gcp.selfplay_worker.load_muzero_model",
                side_effect=KeyboardInterrupt(),
            ):
                exit_code = main()

        assert exit_code == 130
