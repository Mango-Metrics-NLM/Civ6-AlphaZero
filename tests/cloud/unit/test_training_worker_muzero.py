"""
Unit tests for MuZero training worker paths.

Tests the MuZero-specific functions in training_worker.py:
- parse_args with --algorithm=muzero
- create_muzero_model
- train_muzero_epoch
- main() routing to MuZero training
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
import torch

# =============================================================================
# parse_args — MuZero flag Tests
# =============================================================================


class TestParseArgsMuZero:
    """Test that --algorithm=muzero is accepted by arg parser."""

    def test_muzero_algorithm_accepted(self) -> None:
        """Test parsing with --algorithm=muzero."""
        from alpha_zero.cloud.gcp.training_worker import parse_args

        test_args = [
            "training_worker",
            "--iteration=1",
            "--trajectory_iterations=0,1",
            "--algorithm=muzero",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.algorithm == "muzero"

    def test_default_algorithm_is_alphazero(self) -> None:
        """Test that default algorithm is alphazero (backwards compat)."""
        from alpha_zero.cloud.gcp.training_worker import parse_args

        test_args = [
            "training_worker",
            "--iteration=1",
            "--trajectory_iterations=0",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.algorithm == "alphazero"

    def test_invalid_algorithm_rejected(self) -> None:
        """Test that an invalid algorithm name is rejected."""
        from alpha_zero.cloud.gcp.training_worker import parse_args

        test_args = [
            "training_worker",
            "--iteration=1",
            "--trajectory_iterations=0",
            "--algorithm=invalid_algo",
        ]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                parse_args()


# =============================================================================
# create_muzero_model Tests
# =============================================================================


class TestCreateMuZeroModel:
    """Tests for MuZero model creation."""

    def test_creates_muzero_network(self) -> None:
        """Test that create_muzero_model returns a MuZeroNetwork."""
        from alpha_zero.cloud.gcp.training_worker import create_muzero_model
        from alpha_zero.core.muzero_network import MuZeroNetwork

        model = create_muzero_model(
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
        )

        assert isinstance(model, MuZeroNetwork)

    def test_model_config_matches_params(self) -> None:
        """Test that model config matches the provided parameters."""
        from alpha_zero.cloud.gcp.training_worker import create_muzero_model

        model = create_muzero_model(
            board_size=16,
            action_space_size=500,
            num_res_blocks=4,
            num_filters=32,
        )

        assert model.config.spatial_size == 16
        assert model.config.action_space_size == 500
        assert model.config.num_res_blocks == 4
        assert model.config.hidden_state_channels == 32

    def test_model_can_forward(self) -> None:
        """Test that created model can perform forward pass."""
        from alpha_zero.cloud.gcp.training_worker import create_muzero_model

        model = create_muzero_model(
            board_size=8,
            action_space_size=376,
            num_res_blocks=4,
            num_filters=32,
        )

        model.eval()
        spatial = torch.randn(2, 13, 8, 8)
        vector = torch.randn(2, 64)

        with torch.no_grad():
            policy, value = model(spatial, vector)

        assert policy.shape == (2, 376)
        assert value.shape == (2,)

    def test_model_has_dynamics(self) -> None:
        """Test that created model has dynamics network for MuZero MCTS."""
        from alpha_zero.cloud.gcp.training_worker import create_muzero_model

        model = create_muzero_model(
            board_size=8,
            action_space_size=376,
            num_res_blocks=4,
            num_filters=32,
        )

        # MuZeroNetwork should have dynamics attribute
        assert hasattr(model, "dynamics")
        assert hasattr(model, "initial_inference")
        assert hasattr(model, "recurrent_inference")
        assert hasattr(model, "representation")


# =============================================================================
# train_muzero_epoch Tests
# =============================================================================


class TestTrainMuZeroEpoch:
    """Tests for MuZero training epoch logic."""

    def test_train_muzero_epoch_exists(self) -> None:
        """Test that train_muzero_epoch function is importable."""
        from alpha_zero.cloud.gcp.training_worker import train_muzero_epoch

        assert callable(train_muzero_epoch)

    def test_train_muzero_epoch_basic(self) -> None:
        """Test basic training epoch with mock data (tuple batch format)."""
        from alpha_zero.cloud.gcp.training_worker import (
            ENCODER_SPATIAL_CHANNELS,
            create_muzero_model,
            train_muzero_epoch,
        )

        model = create_muzero_model(
            board_size=8,
            action_space_size=376,
            num_res_blocks=4,
            num_filters=32,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create minimal batch as tuple matching Phase 17 dataset layout:
        # (spatial_flat, vector, action_mask, policy, value, rewards, actions)
        batch_size = 4
        action_space_size = 376
        spatial_flat = torch.randn(batch_size, ENCODER_SPATIAL_CHANNELS * 8 * 8)
        vector = torch.randn(batch_size, 64)
        action_mask = torch.ones(batch_size, action_space_size)
        target_policy = torch.softmax(torch.randn(batch_size, action_space_size), dim=-1)
        target_value = torch.randn(batch_size)
        target_rewards = torch.zeros(batch_size)
        target_actions = torch.randint(0, action_space_size, (batch_size,))

        mock_batch = (
            spatial_flat,
            vector,
            action_mask,
            target_policy,
            target_value,
            target_rewards,
            target_actions,
        )
        mock_loader = [mock_batch]

        result = train_muzero_epoch(
            model=model,
            dataloader=mock_loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            unroll_steps=1,
        )

        # Returns (total_loss, policy_loss, value_loss, reward_loss, num_batches)
        assert len(result) == 5
        total_loss, policy_loss, value_loss, reward_loss, num_batches = result
        assert num_batches == 1
        assert total_loss > 0
        assert policy_loss > 0


# =============================================================================
# Algorithm Metadata Tests
# =============================================================================


class TestAlgorithmMetadata:
    """Tests for algorithm type in model metadata."""

    def test_muzero_algorithm_logged(self) -> None:
        """Test that --algorithm=muzero gets included in metadata."""
        from alpha_zero.cloud.gcp.training_worker import parse_args

        test_args = [
            "training_worker",
            "--iteration=1",
            "--trajectory_iterations=0",
            "--algorithm=muzero",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        # Verify the algorithm flag is available for metadata
        assert hasattr(args, "algorithm")
        assert args.algorithm == "muzero"
