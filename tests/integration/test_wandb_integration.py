"""Integration tests for W&B logging with training pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.utils.wandb_logger import (
    WandbConfig,
    WandbLogger,
)


class TestWandbTrainingIntegration:
    """Test W&B integration with training components."""

    @pytest.fixture
    def mock_wandb(self):
        """Fixture to mock wandb module."""
        with patch("alpha_zero.utils.wandb_logger.WANDB_AVAILABLE", True):
            with patch("alpha_zero.utils.wandb_logger.wandb") as mock:
                mock_run = MagicMock()
                mock_run.name = "test-run"
                mock_run.id = "abc123"
                mock_run.summary = {}
                mock.init.return_value = mock_run
                mock.run = mock_run
                yield mock

    def test_training_loop_logging(self, mock_wandb):
        """Test logging during a simulated training loop."""
        # Setup
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._initialized = True
        logger._run = mock_wandb.init.return_value

        # Simulate training loop
        for step in range(5):
            loss = 0.5 - step * 0.05
            policy_loss = loss * 0.6
            value_loss = loss * 0.4

            logger.log_training_step(
                step=step,
                loss=loss,
                policy_loss=policy_loss,
                value_loss=value_loss,
                learning_rate=0.001,
            )

        # Verify logs were called
        assert mock_wandb.log.call_count == 5

    def test_game_logging(self, mock_wandb):
        """Test logging game results."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._initialized = True
        logger._run = mock_wandb.init.return_value

        # Log multiple games
        for i in range(3):
            logger.log_game(
                game_id=f"game_{i}",
                winner=i % 2,
                total_turns=100 + i * 10,
                reward=1.0 if i % 2 == 0 else -1.0,
            )

        assert mock_wandb.log.call_count == 3

    def test_evaluation_logging(self, mock_wandb):
        """Test logging evaluation metrics."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._initialized = True
        logger._run = mock_wandb.init.return_value

        logger.log_evaluation(
            win_rate=0.65,
            games_played=20,
            average_reward=0.3,
            average_turns=85,
        )

        mock_wandb.log.assert_called_once()
        data = mock_wandb.log.call_args[0][0]
        assert data["eval/win_rate"] == 0.65
        assert data["eval/games_played"] == 20

    def test_full_training_cycle(self, mock_wandb):
        """Test complete training cycle with all logging types."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._initialized = True
        logger._run = mock_wandb.init.return_value

        # Self-play phase
        logger.log_selfplay(
            games_completed=10,
            buffer_size=1000,
            samples_generated=5000,
        )

        # Training phase
        for step in range(3):
            logger.log_training_step(
                step=step,
                loss=0.5,
                policy_loss=0.3,
                value_loss=0.2,
            )

        # MCTS stats
        logger.log_mcts(
            average_depth=12.5,
            average_visits=150.0,
        )

        # Evaluation phase
        logger.log_evaluation(
            win_rate=0.55,
            games_played=10,
            average_reward=0.1,
            average_turns=90,
        )

        # All logging calls should have been made
        assert mock_wandb.log.call_count == 6


class TestWandbWithRealComponents:
    """Test W&B with real game components (mocked W&B calls)."""

    @pytest.fixture
    def mock_wandb(self):
        """Fixture to mock wandb module."""
        with patch("alpha_zero.utils.wandb_logger.WANDB_AVAILABLE", True):
            with patch("alpha_zero.utils.wandb_logger.wandb") as mock:
                mock_run = MagicMock()
                mock_run.name = "test-run"
                mock_run.id = "abc123"
                mock_run.summary = {}
                mock.init.return_value = mock_run
                mock.run = mock_run
                yield mock

    def test_game_with_logging(self, mock_wandb):
        """Test playing a game with W&B logging."""
        # Setup logger
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._initialized = True
        logger._run = mock_wandb.init.return_value

        # Setup game
        env = MicroCivEnv(board_size=8, seed=42)
        obs, _ = env.reset()

        # Play game with logging
        total_reward = 0
        turn = 0
        done = False

        while not done and turn < 50:
            valid_actions = obs["action_mask"]
            valid_indices = np.where(valid_actions > 0)[0]

            if len(valid_indices) == 0:
                break

            action = np.random.choice(valid_indices)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            turn += 1

        # Log game result
        logger.log_game(
            game_id="test_game",
            winner=0 if total_reward > 0 else 1,
            total_turns=turn,
            reward=total_reward,
        )

        # Verify log was called
        mock_wandb.log.assert_called()

    def test_network_training_with_logging(self, mock_wandb):
        """Test network training with W&B logging."""
        # Setup logger
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._initialized = True
        logger._run = mock_wandb.init.return_value

        # Setup network and optimizer
        network = MicroCivNetwork(board_size=8, action_space_size=356)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        # Create dummy batch
        batch_size = 4
        spatial = torch.randn(batch_size, 13, 8, 8)
        vector = torch.randn(batch_size, 64)
        target_policy = torch.softmax(torch.randn(batch_size, 356), dim=1)
        target_value = torch.randn(batch_size)

        # Training step
        network.train()
        optimizer.zero_grad()

        policy, value = network(spatial, vector)
        value = value.squeeze()

        # Compute losses
        policy_loss = -torch.mean(torch.sum(target_policy * torch.log(policy + 1e-8), dim=1))
        value_loss = torch.mean((value - target_value) ** 2)
        total_loss = policy_loss + value_loss

        total_loss.backward()
        optimizer.step()

        # Log training step
        logger.log_training_step(
            step=0,
            loss=total_loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            learning_rate=0.001,
        )

        # Verify logging
        mock_wandb.log.assert_called()
        data = mock_wandb.log.call_args[0][0]
        assert "train/loss" in data
        assert "train/policy_loss" in data
        assert "train/value_loss" in data


@pytest.mark.skipif(not os.getenv("WANDB_API_KEY"), reason="WANDB_API_KEY not set for live test")
class TestWandbLiveIntegration:
    """Live W&B integration tests (require API key)."""

    def test_live_init_and_log(self):
        """Test live W&B initialization and logging."""
        config = WandbConfig(
            enabled=True,
            project="civ6-alphazero-test",
            mode="online",
            tags=["test", "integration"],
        )
        logger = WandbLogger(config)

        try:
            run = logger.init(config_dict={"test": True, "lr": 0.001})
            assert run is not None

            # Log some test data
            logger.log_training_step(
                step=0,
                loss=0.5,
                policy_loss=0.3,
                value_loss=0.2,
            )

            logger.log_game(
                game_id="test_game",
                winner=0,
                total_turns=50,
                reward=1.0,
            )

        finally:
            logger.finish()

    def test_live_full_training_simulation(self):
        """Test full training simulation with live W&B."""
        config = WandbConfig(
            enabled=True,
            project="civ6-alphazero-test",
            mode="online",
            tags=["test", "training-sim"],
        )
        logger = WandbLogger(config)

        try:
            _run = logger.init(
                config_dict={
                    "board_size": 8,
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "num_simulations": 50,
                }
            )

            # Simulate training
            for step in range(10):
                loss = 0.5 * (0.9**step)
                logger.log_training_step(
                    step=step,
                    loss=loss,
                    policy_loss=loss * 0.6,
                    value_loss=loss * 0.4,
                    learning_rate=0.001,
                )

            # Log evaluation
            logger.log_evaluation(
                win_rate=0.55,
                games_played=10,
                average_reward=0.1,
                average_turns=85,
            )

            # Update summary
            logger.summary(
                {
                    "final_win_rate": 0.55,
                    "total_games": 10,
                }
            )

        finally:
            logger.finish()


class TestWandbConfigurationIntegration:
    """Test W&B configuration integration."""

    def test_config_from_yaml(self):
        """Test W&B configuration from project config."""
        from alpha_zero.utils.config import load_config

        project_root = Path(__file__).parent.parent.parent
        config_file = project_root / "configs" / "civ6_config.yaml"

        if config_file.exists():
            _civ6_config = load_config(config_file)
            # W&B config should be loadable alongside main config
            wandb_config = WandbConfig.from_env()
            assert isinstance(wandb_config, WandbConfig)

    def test_config_environment_override(self):
        """Test environment variable override for W&B config."""
        env_vars = {
            "WANDB_ENABLED": "true",
            "WANDB_PROJECT": "override-project",
            "WANDB_MODE": "offline",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = WandbConfig.from_env()
            assert config.enabled is True
            assert config.project == "override-project"
            assert config.mode == "offline"
