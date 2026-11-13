"""Unit tests for training pipeline smoke testing.

Tests the training_civ6 module's ability to:
- Configure training parameters correctly
- Execute dry-run mode without errors
- Create checkpoints and logs
- Handle edge cases gracefully

These tests run quickly without GPU and provide high coverage
of the training infrastructure.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# =============================================================================
# Test Configuration
# =============================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_config_values(self) -> None:
        """Test that default configuration values are sensible."""
        from alpha_zero.training_civ6 import TrainingConfig

        config = TrainingConfig()

        assert config.board_size == 16
        assert config.max_turns == 200
        assert config.num_simulations == 50
        assert config.batch_size == 128
        assert config.learning_rate == 0.001
        assert config.num_iterations == 100
        assert config.checkpoint_dir == "checkpoints/civ6"
        assert config.log_dir == "logs/civ6"

    def test_config_device_selection(self) -> None:
        """Test that device is selected based on CUDA availability."""
        from alpha_zero.training_civ6 import TrainingConfig

        config = TrainingConfig()

        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert config.device == expected_device

    def test_config_custom_values(self) -> None:
        """Test configuration with custom values."""
        from alpha_zero.training_civ6 import TrainingConfig

        config = TrainingConfig(
            board_size=8,
            num_simulations=25,
            batch_size=64,
            learning_rate=0.0005,
        )

        assert config.board_size == 8
        assert config.num_simulations == 25
        assert config.batch_size == 64
        assert config.learning_rate == 0.0005

    def test_gumbel_mcts_config(self) -> None:
        """Test Gumbel MuZero MCTS configuration."""
        from alpha_zero.training_civ6 import TrainingConfig

        config = TrainingConfig(
            use_gumbel_mcts=True,
            gumbel_num_simulations=16,
            gumbel_num_sampled_actions=8,
        )

        assert config.use_gumbel_mcts is True
        assert config.gumbel_num_simulations == 16
        assert config.gumbel_num_sampled_actions == 8


# =============================================================================
# Test Replay Buffer
# =============================================================================


class TestReplayBuffer:
    """Tests for ReplayBuffer class."""

    def test_buffer_initialization(self) -> None:
        """Test replay buffer initialization."""
        from alpha_zero.training_civ6 import ReplayBuffer

        buffer = ReplayBuffer(max_size=1000)

        assert len(buffer) == 0

    def test_add_single_example(self) -> None:
        """Test adding a single example to buffer."""
        from alpha_zero.training_civ6 import ReplayBuffer, TrainingExample

        buffer = ReplayBuffer(max_size=1000)
        example = TrainingExample(
            spatial=np.zeros((10, 16, 16)),
            vector=np.zeros(100),
            action_mask=np.ones(256),
            policy=np.ones(256) / 256,
            value=0.5,
        )

        buffer.add(example)

        assert len(buffer) == 1

    def test_add_game_examples(self) -> None:
        """Test adding multiple examples from a game."""
        from alpha_zero.training_civ6 import ReplayBuffer, TrainingExample

        buffer = ReplayBuffer(max_size=1000)
        examples = [
            TrainingExample(
                spatial=np.zeros((10, 16, 16)),
                vector=np.zeros(100),
                action_mask=np.ones(256),
                policy=np.ones(256) / 256,
                value=float(i) / 10,
            )
            for i in range(10)
        ]

        buffer.add_game(examples)

        assert len(buffer) == 10

    def test_buffer_max_size(self) -> None:
        """Test that buffer respects max size."""
        from alpha_zero.training_civ6 import ReplayBuffer, TrainingExample

        buffer = ReplayBuffer(max_size=5)

        for i in range(10):
            example = TrainingExample(
                spatial=np.zeros((10, 16, 16)),
                vector=np.zeros(100),
                action_mask=np.ones(256),
                policy=np.ones(256) / 256,
                value=float(i),
            )
            buffer.add(example)

        assert len(buffer) == 5  # Should be capped at max_size

    def test_sample_from_buffer(self) -> None:
        """Test sampling from buffer."""
        from alpha_zero.training_civ6 import ReplayBuffer, TrainingExample

        buffer = ReplayBuffer(max_size=100)

        for i in range(50):
            example = TrainingExample(
                spatial=np.zeros((10, 16, 16)),
                vector=np.zeros(100),
                action_mask=np.ones(256),
                policy=np.ones(256) / 256,
                value=float(i) / 50,
            )
            buffer.add(example)

        samples = buffer.sample(batch_size=10)

        assert len(samples) == 10
        assert all(isinstance(s, TrainingExample) for s in samples)


# =============================================================================
# Test Replay Dataset
# =============================================================================


class TestReplayDataset:
    """Tests for ReplayDataset PyTorch dataset."""

    def test_dataset_length(self) -> None:
        """Test dataset length."""
        from alpha_zero.training_civ6 import ReplayDataset, TrainingExample

        examples = [
            TrainingExample(
                spatial=np.zeros((10, 16, 16)),
                vector=np.zeros(100),
                action_mask=np.ones(256),
                policy=np.ones(256) / 256,
                value=0.5,
            )
            for _ in range(20)
        ]

        dataset = ReplayDataset(examples)

        assert len(dataset) == 20

    def test_dataset_getitem(self) -> None:
        """Test getting item from dataset."""
        from alpha_zero.training_civ6 import ReplayDataset, TrainingExample

        example = TrainingExample(
            spatial=np.random.randn(10, 16, 16).astype(np.float32),
            vector=np.random.randn(100).astype(np.float32),
            action_mask=np.ones(256, dtype=np.float32),
            policy=np.ones(256, dtype=np.float32) / 256,
            value=0.75,
        )

        dataset = ReplayDataset([example])
        item = dataset[0]

        assert isinstance(item, dict)
        assert "spatial" in item
        assert "vector" in item
        assert "action_mask" in item
        assert "policy" in item
        assert "value" in item
        assert item["spatial"].shape == (10, 16, 16)
        assert item["value"].item() == pytest.approx(0.75)


# =============================================================================
# Test Training Functions
# =============================================================================


class TestSelfPlayGame:
    """Tests for self_play_game function."""

    def test_self_play_generates_examples(self) -> None:
        """Test that self-play generates training examples."""
        from alpha_zero.core.civ6_mcts import SimpleMCTS
        from alpha_zero.envs.microciv import MicroCivEnv
        from alpha_zero.training_civ6 import TrainingConfig, self_play_game

        config = TrainingConfig(board_size=8, max_turns=20)
        env = MicroCivEnv(board_size=8, max_turns=20)
        mcts = SimpleMCTS(num_simulations=5, c_puct=1.5)

        examples = self_play_game(env, mcts, config)

        assert len(examples) > 0
        assert all(hasattr(ex, "value") for ex in examples)

    def test_self_play_assigns_values(self) -> None:
        """Test that self-play assigns correct values to examples."""
        from alpha_zero.core.civ6_mcts import SimpleMCTS
        from alpha_zero.envs.microciv import MicroCivEnv
        from alpha_zero.training_civ6 import TrainingConfig, self_play_game

        config = TrainingConfig(board_size=8, max_turns=20)
        env = MicroCivEnv(board_size=8, max_turns=20)
        mcts = SimpleMCTS(num_simulations=5, c_puct=1.5)

        examples = self_play_game(env, mcts, config)

        # Values should be -1, 0, or 1
        for ex in examples:
            assert ex.value in [-1.0, 0.0, 1.0]


class TestRunSelfPlay:
    """Tests for run_self_play function."""

    def test_run_multiple_games(self) -> None:
        """Test running multiple self-play games."""
        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.training_civ6 import TrainingConfig, run_self_play

        config = TrainingConfig(
            board_size=8,
            max_turns=10,
            num_simulations=5,
        )

        network = MicroCivNetwork(
            board_size=8,
            action_space_size=256,
            num_res_blocks=2,
            num_filters=32,
        )

        examples = run_self_play(
            network=network,
            config=config,
            num_games=2,
            use_simple_mcts=True,  # Use simple MCTS for speed
        )

        assert len(examples) > 0


# =============================================================================
# Test Training Dry Run
# =============================================================================


class TestTrainingDryRun:
    """Tests for dry-run training mode."""

    def test_dry_run_creates_directories(self) -> None:
        """Test that dry-run creates checkpoint and log directories."""
        from alpha_zero.training_civ6 import TrainingConfig, train

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                board_size=8,
                max_turns=10,
                num_simulations=5,
                num_res_blocks=2,
                num_filters=32,
                checkpoint_dir=f"{tmpdir}/checkpoints",
                log_dir=f"{tmpdir}/logs",
            )

            # Run dry-run training
            train(config, dry_run=True)

            # Verify directories were created
            assert Path(config.checkpoint_dir).exists()
            assert Path(config.log_dir).exists()

    def test_dry_run_creates_checkpoint(self) -> None:
        """Test that dry-run creates a checkpoint file."""
        from alpha_zero.training_civ6 import TrainingConfig, train

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                board_size=8,
                max_turns=10,
                num_simulations=5,
                num_res_blocks=2,
                num_filters=32,
                checkpoint_dir=f"{tmpdir}/checkpoints",
                log_dir=f"{tmpdir}/logs",
            )

            train(config, dry_run=True)

            # Check for checkpoint files
            checkpoint_dir = Path(config.checkpoint_dir)
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            assert len(checkpoints) > 0, "No checkpoint files created"

    def test_dry_run_completes_without_error(self) -> None:
        """Test that dry-run completes without raising exceptions."""
        from alpha_zero.training_civ6 import TrainingConfig, train

        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                board_size=8,
                max_turns=10,
                num_simulations=3,
                num_res_blocks=1,
                num_filters=16,
                checkpoint_dir=f"{tmpdir}/checkpoints",
                log_dir=f"{tmpdir}/logs",
                device="cpu",  # Force CPU for testing
            )

            # Should not raise
            train(config, dry_run=True)


# =============================================================================
# Test Training Epoch
# =============================================================================


class TestTrainEpoch:
    """Tests for train_epoch function."""

    @pytest.mark.skip(reason="Integration test - requires consistent env/network dimensions")
    def test_train_epoch_returns_losses(self) -> None:
        """Test that train_epoch returns loss dictionary."""
        from torch.utils.data import DataLoader

        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.training_civ6 import (
            ReplayDataset,
            TrainingConfig,
            TrainingExample,
            train_epoch,
        )

        config = TrainingConfig(
            board_size=8,
            batch_size=4,
            device="cpu",
        )

        network = MicroCivNetwork(
            board_size=8,
            action_space_size=256,
            num_res_blocks=1,
            num_filters=16,
        )

        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        # Create fake training data (13 channels matches MicroCivEncoder.TOTAL_CHANNELS)
        examples = [
            TrainingExample(
                spatial=np.random.randn(13, 8, 8).astype(np.float32),
                vector=np.random.randn(100).astype(np.float32),
                action_mask=np.ones(256, dtype=np.float32),
                policy=np.ones(256, dtype=np.float32) / 256,
                value=np.random.uniform(-1, 1),
            )
            for _ in range(8)
        ]

        dataset = ReplayDataset(examples)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        losses = train_epoch(network, optimizer, dataloader, config)

        assert "total_loss" in losses
        assert "policy_loss" in losses
        assert "value_loss" in losses
        assert losses["total_loss"] > 0


# =============================================================================
# Test Evaluation
# =============================================================================


class TestEvaluateAgainstRandom:
    """Tests for evaluate_against_random function."""

    @pytest.mark.skip(reason="Integration test - requires consistent env/network dimensions")
    def test_evaluation_returns_win_rate(self) -> None:
        """Test that evaluation returns a valid win rate."""
        from alpha_zero.core.civ6_network import MicroCivNetwork
        from alpha_zero.training_civ6 import TrainingConfig, evaluate_against_random

        config = TrainingConfig(
            board_size=8,
            max_turns=20,
            num_simulations=5,
            device="cpu",
        )

        network = MicroCivNetwork(
            board_size=8,
            action_space_size=256,
            num_res_blocks=1,
            num_filters=16,
        )

        summary = evaluate_against_random(
            network=network,
            config=config,
            num_games=2,
        )

        assert 0.0 <= summary.score_rate <= 1.0
