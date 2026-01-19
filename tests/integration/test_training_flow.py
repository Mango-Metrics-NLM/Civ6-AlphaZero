"""Integration tests for the full training flow.

These tests verify that all training components work together correctly
using real MicroCivEnv and MicroCivNetwork instances.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.utils.wandb_logger import WandbLogger, WandbConfig

from run_training_wandb import (
    TrainingConfig,
    GameResult,
    TrainingResult,
    EvaluationResult,
    select_action,
    run_self_play,
    train_network,
    evaluate_network,
    main,
)


class TestFullTrainingCycle:
    """Test complete training cycle with real components."""

    @pytest.fixture
    def tiny_config(self):
        """Create a minimal configuration for fast testing."""
        return TrainingConfig(
            board_size=8,
            num_games=2,
            num_epochs=2,
            batch_size=4,
            learning_rate=0.001,
            max_turns_per_game=20,
            eval_games=2,
        )

    @pytest.fixture
    def env(self, tiny_config):
        """Create a real MicroCiv environment."""
        return MicroCivEnv(board_size=tiny_config.board_size, seed=42)

    @pytest.fixture
    def network(self, tiny_config, env):
        """Create a real MicroCiv network."""
        return MicroCivNetwork(
            board_size=tiny_config.board_size,
            action_space_size=env.action_space.n,
        )

    def test_self_play_with_real_components(self, env, network, tiny_config):
        """Test self-play generates valid training data."""
        replay_buffer, game_results = run_self_play(
            env=env,
            network=network,
            config=tiny_config,
            wandb_logger=None,
            verbose=False,
        )

        # Verify replay buffer structure
        assert len(replay_buffer) > 0, "Should generate experiences"

        for exp in replay_buffer:
            # Check spatial observation
            assert exp["spatial"].shape == (13, tiny_config.board_size, tiny_config.board_size)
            assert exp["spatial"].dtype == np.float32

            # Check vector observation
            assert exp["vector"].shape == (64,)
            assert exp["vector"].dtype == np.float32

            # Check policy is valid probability distribution
            assert exp["policy"].shape == (env.action_space.n,)
            assert np.isclose(exp["policy"].sum(), 1.0, atol=1e-5)
            assert np.all(exp["policy"] >= 0)

            # Check action is valid
            assert 0 <= exp["action"] < env.action_space.n

            # Check value is a number
            assert isinstance(exp["value"], (int, float))

        # Verify game results
        assert len(game_results) == tiny_config.num_games

        for result in game_results:
            assert isinstance(result, GameResult)
            assert result.turns > 0
            assert result.game_time > 0

    def test_training_with_real_components(self, env, network, tiny_config):
        """Test training on real replay buffer."""
        # First generate replay buffer
        replay_buffer, _ = run_self_play(
            env=env,
            network=network,
            config=tiny_config,
            wandb_logger=None,
            verbose=False,
        )

        # Create optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=tiny_config.learning_rate)

        # Get initial loss
        initial_params = {
            name: param.clone().detach()
            for name, param in network.named_parameters()
        }

        # Train
        result = train_network(
            network=network,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            config=tiny_config,
            wandb_logger=None,
            verbose=False,
        )

        # Verify result structure
        assert isinstance(result, TrainingResult)
        assert result.total_steps >= 0

        # If there was training, verify some weights changed
        if result.total_steps > 0:
            weights_changed = False
            for name, param in network.named_parameters():
                if not torch.equal(initial_params[name], param):
                    weights_changed = True
                    break
            assert weights_changed, "Weights should change after training"

    def test_evaluation_with_real_components(self, env, network, tiny_config):
        """Test evaluation with real environment and network."""
        result = evaluate_network(
            env=env,
            network=network,
            config=tiny_config,
            wandb_logger=None,
            verbose=False,
        )

        # Verify result structure
        assert isinstance(result, EvaluationResult)
        assert result.games_played == tiny_config.eval_games
        assert 0.0 <= result.win_rate <= 1.0
        assert result.total_wins <= result.games_played
        assert result.average_turns > 0
        assert -1.0 <= result.average_reward <= 1.0

    def test_complete_pipeline_integration(self, env, network, tiny_config):
        """Test the complete self-play -> train -> evaluate pipeline."""
        optimizer = torch.optim.Adam(network.parameters(), lr=tiny_config.learning_rate)

        # Self-play
        replay_buffer, game_results = run_self_play(
            env=env,
            network=network,
            config=tiny_config,
            wandb_logger=None,
            verbose=False,
        )

        assert len(replay_buffer) > 0, "Self-play should generate data"
        assert len(game_results) == tiny_config.num_games

        # Training
        training_result = train_network(
            network=network,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            config=tiny_config,
            wandb_logger=None,
            verbose=False,
        )

        assert isinstance(training_result, TrainingResult)

        # Evaluation
        eval_result = evaluate_network(
            env=env,
            network=network,
            config=tiny_config,
            wandb_logger=None,
            verbose=False,
        )

        assert isinstance(eval_result, EvaluationResult)

    def test_pipeline_with_wandb_logging(self, env, network, tiny_config):
        """Test pipeline with mocked W&B logging."""
        optimizer = torch.optim.Adam(network.parameters(), lr=tiny_config.learning_rate)

        # Create mock W&B logger
        mock_logger = MagicMock(spec=WandbLogger)
        mock_logger.log_game = MagicMock()
        mock_logger.log_selfplay = MagicMock()
        mock_logger.log_training_step = MagicMock()
        mock_logger.log_evaluation = MagicMock()

        # Self-play with logging
        replay_buffer, _ = run_self_play(
            env=env,
            network=network,
            config=tiny_config,
            wandb_logger=mock_logger,
            verbose=False,
        )

        assert mock_logger.log_game.call_count == tiny_config.num_games
        mock_logger.log_selfplay.assert_called_once()

        # Training with logging
        train_network(
            network=network,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            config=tiny_config,
            wandb_logger=mock_logger,
            verbose=False,
        )

        # Should have logged training steps
        assert mock_logger.log_training_step.call_count > 0

        # Evaluation with logging
        evaluate_network(
            env=env,
            network=network,
            config=tiny_config,
            wandb_logger=mock_logger,
            verbose=False,
        )

        mock_logger.log_evaluation.assert_called_once()


class TestDataFlowIntegrity:
    """Test that data flows correctly between components."""

    @pytest.fixture
    def tiny_config(self):
        """Create a minimal configuration."""
        return TrainingConfig(
            board_size=8,
            num_games=3,
            num_epochs=1,
            batch_size=8,
            max_turns_per_game=30,
            eval_games=2,
        )

    def test_observation_dimensions_consistency(self, tiny_config):
        """Test that observation dimensions are consistent across components."""
        env = MicroCivEnv(board_size=tiny_config.board_size, seed=42)
        network = MicroCivNetwork(
            board_size=tiny_config.board_size,
            action_space_size=env.action_space.n,
        )

        # Get observation from environment
        obs, _ = env.reset()

        # Verify dimensions match network expectations
        spatial_shape = obs["spatial"].shape
        vector_shape = obs["vector"].shape

        # Create tensors as network expects
        spatial_tensor = torch.tensor(obs["spatial"]).unsqueeze(0).float()
        vector_tensor = torch.tensor(obs["vector"]).unsqueeze(0).float()

        # Network should process without error
        with torch.no_grad():
            policy, value = network(spatial_tensor, vector_tensor)

        assert policy.shape == (1, env.action_space.n)
        assert value.numel() == 1

    def test_action_mask_enforced_during_play(self, tiny_config):
        """Test that action mask is properly enforced."""
        env = MicroCivEnv(board_size=tiny_config.board_size, seed=42)
        network = MicroCivNetwork(
            board_size=tiny_config.board_size,
            action_space_size=env.action_space.n,
        )

        obs, _ = env.reset()

        for _ in range(10):  # Play 10 turns
            spatial = torch.tensor(obs["spatial"]).unsqueeze(0).float()
            vector = torch.tensor(obs["vector"]).unsqueeze(0).float()

            with torch.no_grad():
                policy, _ = network(spatial, vector)

            policy_np = policy.squeeze().numpy()
            mask = obs["action_mask"]

            action = select_action(policy_np, mask)

            if action is None:
                break

            # Verify selected action is valid according to mask
            assert mask[action] > 0, f"Selected invalid action {action}"

            obs, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

    def test_value_targets_assigned_correctly(self, tiny_config):
        """Test that value targets are assigned to all experiences in a game."""
        env = MicroCivEnv(board_size=tiny_config.board_size, seed=42)
        network = MicroCivNetwork(
            board_size=tiny_config.board_size,
            action_space_size=env.action_space.n,
        )

        replay_buffer, game_results = run_self_play(
            env=env,
            network=network,
            config=tiny_config,
            wandb_logger=None,
            verbose=False,
        )

        # Track experiences per game
        exp_idx = 0
        for game_result in game_results:
            game_value = game_result.total_reward
            num_experiences = len(game_result.experiences)

            # All experiences from this game should have same value target
            for _ in range(num_experiences):
                if exp_idx < len(replay_buffer):
                    assert replay_buffer[exp_idx]["value"] == game_value
                    exp_idx += 1


class TestMainFunction:
    """Test the main() orchestration function."""

    @pytest.fixture
    def mock_wandb(self):
        """Mock W&B to prevent actual API calls."""
        with patch("run_training_wandb.WandbLogger") as mock:
            mock_instance = MagicMock()
            mock_instance.init.return_value = None
            mock_instance.log_game = MagicMock()
            mock_instance.log_selfplay = MagicMock()
            mock_instance.log_training_step = MagicMock()
            mock_instance.log_evaluation = MagicMock()
            mock_instance.summary = MagicMock()
            mock_instance.finish = MagicMock()
            mock.return_value = mock_instance
            yield mock

    def test_main_returns_results_dict(self, mock_wandb):
        """Test that main() returns a proper results dictionary."""
        config = TrainingConfig(
            board_size=8,
            num_games=1,
            num_epochs=1,
            batch_size=4,
            max_turns_per_game=10,
            eval_games=1,
        )

        results = main(
            config=config,
            wandb_enabled=False,
            verbose=False,
        )

        assert isinstance(results, dict)
        assert "config" in results
        assert "replay_buffer_size" in results
        assert "game_results" in results
        assert "training_result" in results
        assert "eval_result" in results

        assert results["config"]["board_size"] == 8
        assert results["replay_buffer_size"] >= 0
        assert isinstance(results["training_result"], TrainingResult)
        assert isinstance(results["eval_result"], EvaluationResult)

    def test_main_calls_wandb_lifecycle(self, mock_wandb):
        """Test that main() properly manages W&B lifecycle."""
        config = TrainingConfig(
            board_size=8,
            num_games=1,
            num_epochs=1,
            batch_size=4,
            max_turns_per_game=10,
            eval_games=1,
        )

        main(
            config=config,
            wandb_enabled=True,
            verbose=False,
        )

        # Verify W&B logger was created and methods called
        mock_wandb.assert_called_once()
        mock_instance = mock_wandb.return_value
        mock_instance.init.assert_called_once()
        mock_instance.summary.assert_called_once()
        mock_instance.finish.assert_called_once()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_action_mask_handling(self):
        """Test handling of empty action mask."""
        policy = np.array([0.25, 0.25, 0.25, 0.25])
        mask = np.array([0, 0, 0, 0])

        action = select_action(policy, mask)
        assert action is None

    def test_single_valid_action(self):
        """Test selection with only one valid action."""
        np.random.seed(42)
        policy = np.array([0.1, 0.7, 0.1, 0.1])
        mask = np.array([0, 1, 0, 0])  # Only action 1 valid

        for _ in range(10):
            action = select_action(policy, mask)
            assert action == 1

    def test_very_short_game(self):
        """Test handling of very short games."""
        config = TrainingConfig(
            board_size=8,
            num_games=1,
            num_epochs=1,
            batch_size=2,
            max_turns_per_game=1,  # Only 1 turn allowed
            eval_games=1,
        )

        env = MicroCivEnv(board_size=8, seed=42)
        network = MicroCivNetwork(board_size=8, action_space_size=env.action_space.n)

        replay_buffer, game_results = run_self_play(
            env=env,
            network=network,
            config=config,
            wandb_logger=None,
            verbose=False,
        )

        # Should still produce valid results
        assert len(game_results) == 1
        assert game_results[0].turns <= 1

    def test_training_with_small_buffer(self):
        """Test training with replay buffer smaller than batch size."""
        config = TrainingConfig(
            board_size=8,
            num_epochs=1,
            batch_size=100,  # Larger than buffer
        )

        network = MicroCivNetwork(board_size=8, action_space_size=356)
        optimizer = torch.optim.Adam(network.parameters())

        # Small buffer
        buffer = [{
            "spatial": np.random.randn(13, 8, 8).astype(np.float32),
            "vector": np.random.randn(64).astype(np.float32),
            "policy": np.ones(356, dtype=np.float32) / 356,
            "action": 0,
            "value": 1.0,
        } for _ in range(10)]  # Only 10 samples

        result = train_network(
            network=network,
            optimizer=optimizer,
            replay_buffer=buffer,
            config=config,
            wandb_logger=None,
            verbose=False,
        )

        # Should complete without error, but no steps taken
        assert result.total_steps == 0


class TestReproducibility:
    """Test reproducibility of training components."""

    def test_self_play_reproducibility(self):
        """Test that self-play is reproducible with same seed."""
        config = TrainingConfig(
            board_size=8,
            num_games=2,
            max_turns_per_game=20,
        )

        # First run
        np.random.seed(42)
        torch.manual_seed(42)
        env1 = MicroCivEnv(board_size=8, seed=42)
        network1 = MicroCivNetwork(board_size=8, action_space_size=env1.action_space.n)

        # Save network state
        state_dict = network1.state_dict()

        buffer1, results1 = run_self_play(
            env=env1,
            network=network1,
            config=config,
            wandb_logger=None,
            verbose=False,
        )

        # Second run with same seeds
        np.random.seed(42)
        torch.manual_seed(42)
        env2 = MicroCivEnv(board_size=8, seed=42)
        network2 = MicroCivNetwork(board_size=8, action_space_size=env2.action_space.n)
        network2.load_state_dict(state_dict)

        buffer2, results2 = run_self_play(
            env=env2,
            network=network2,
            config=config,
            wandb_logger=None,
            verbose=False,
        )

        # Results should be identical
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.turns == r2.turns
            assert r1.total_reward == r2.total_reward
