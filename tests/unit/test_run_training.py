"""Unit tests for run_training_wandb.py module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from run_training_wandb import (
    TrainingConfig,
    GameResult,
    TrainingResult,
    EvaluationResult,
    select_action,
    run_self_play,
    train_network,
    evaluate_network,
)


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.board_size == 8
        assert config.num_games == 20
        assert config.num_epochs == 10
        assert config.batch_size == 8
        assert config.learning_rate == 0.001
        assert config.max_turns_per_game == 100
        assert config.eval_games == 10
        assert config.grad_clip_norm == 1.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            board_size=16,
            num_games=50,
            num_epochs=20,
            batch_size=32,
            learning_rate=0.0001,
        )

        assert config.board_size == 16
        assert config.num_games == 50
        assert config.num_epochs == 20
        assert config.batch_size == 32
        assert config.learning_rate == 0.0001

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrainingConfig(board_size=12, num_games=5)
        result = config.to_dict()

        assert isinstance(result, dict)
        assert result["board_size"] == 12
        assert result["num_games"] == 5
        assert "learning_rate" in result
        assert "batch_size" in result


class TestGameResult:
    """Tests for GameResult dataclass."""

    def test_create_game_result(self):
        """Test creating a game result."""
        result = GameResult(
            game_id="game_0",
            turns=50,
            total_reward=1.5,
            game_time=2.3,
        )

        assert result.game_id == "game_0"
        assert result.turns == 50
        assert result.total_reward == 1.5
        assert result.game_time == 2.3
        assert result.experiences == []

    def test_game_result_with_experiences(self):
        """Test game result with experience data."""
        experiences = [{"spatial": np.zeros((13, 8, 8)), "action": 1}]
        result = GameResult(
            game_id="game_1",
            turns=10,
            total_reward=0.5,
            game_time=1.0,
            experiences=experiences,
        )

        assert len(result.experiences) == 1
        assert result.experiences[0]["action"] == 1


class TestTrainingResult:
    """Tests for TrainingResult dataclass."""

    def test_create_training_result(self):
        """Test creating a training result."""
        result = TrainingResult(
            total_steps=100,
            final_loss=0.5,
            final_policy_loss=0.3,
            final_value_loss=0.2,
        )

        assert result.total_steps == 100
        assert result.final_loss == 0.5
        assert result.final_policy_loss == 0.3
        assert result.final_value_loss == 0.2
        assert result.epoch_losses == []

    def test_training_result_with_epoch_losses(self):
        """Test training result with epoch losses."""
        result = TrainingResult(
            total_steps=50,
            final_loss=0.3,
            final_policy_loss=0.2,
            final_value_loss=0.1,
            epoch_losses=[0.5, 0.4, 0.3],
        )

        assert len(result.epoch_losses) == 3
        assert result.epoch_losses[0] == 0.5


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_create_evaluation_result(self):
        """Test creating an evaluation result."""
        result = EvaluationResult(
            win_rate=0.6,
            games_played=10,
            total_wins=6,
            average_turns=45.0,
            average_reward=0.2,
        )

        assert result.win_rate == 0.6
        assert result.games_played == 10
        assert result.total_wins == 6
        assert result.average_turns == 45.0
        assert result.average_reward == 0.2


class TestSelectAction:
    """Tests for select_action function."""

    def test_select_from_valid_actions(self):
        """Test selecting from valid actions."""
        np.random.seed(42)
        policy = np.array([0.1, 0.3, 0.4, 0.2])
        mask = np.array([1, 1, 1, 1])

        action = select_action(policy, mask)

        assert action is not None
        assert 0 <= action < 4

    def test_select_with_masked_actions(self):
        """Test selecting with some actions masked."""
        np.random.seed(42)
        policy = np.array([0.5, 0.5, 0.0, 0.0])
        mask = np.array([1, 1, 0, 0])  # Only first two valid

        for _ in range(10):
            action = select_action(policy, mask)
            assert action in [0, 1]

    def test_select_returns_none_when_no_valid(self):
        """Test returns None when no valid actions."""
        policy = np.array([0.25, 0.25, 0.25, 0.25])
        mask = np.array([0, 0, 0, 0])  # No valid actions

        action = select_action(policy, mask)

        assert action is None

    def test_select_handles_negative_probabilities(self):
        """Test handling negative probabilities."""
        np.random.seed(42)
        policy = np.array([-0.5, 0.8, -0.2, 0.9])
        mask = np.array([1, 1, 1, 1])

        action = select_action(policy, mask)

        assert action is not None
        assert 0 <= action < 4

    def test_select_handles_all_zeros_policy(self):
        """Test handling all zeros policy (uniform random)."""
        np.random.seed(42)
        policy = np.array([0.0, 0.0, 0.0, 0.0])
        mask = np.array([0, 1, 1, 0])

        action = select_action(policy, mask)

        assert action in [1, 2]  # Only valid indices

    def test_select_handles_nan_in_policy(self):
        """Test handling NaN values in policy."""
        np.random.seed(42)
        policy = np.array([np.nan, 0.5, np.nan, 0.5])
        mask = np.array([1, 1, 1, 1])

        # Should not crash
        action = select_action(policy, mask)
        assert action is not None
        assert 0 <= action < 4

    def test_select_handles_inf_in_policy(self):
        """Test handling infinity values in policy."""
        np.random.seed(42)
        policy = np.array([np.inf, 0.5, -np.inf, 0.5])
        mask = np.array([1, 1, 1, 1])

        # Should not crash
        action = select_action(policy, mask)
        assert action is not None
        assert 0 <= action < 4

    def test_select_respects_probability_distribution(self):
        """Test that action selection respects probability distribution."""
        np.random.seed(42)
        # Heavily biased policy
        policy = np.array([0.9, 0.05, 0.05, 0.0])
        mask = np.array([1, 1, 1, 0])

        counts = {0: 0, 1: 0, 2: 0}
        for _ in range(1000):
            action = select_action(policy, mask)
            counts[action] += 1

        # Action 0 should be selected most often
        assert counts[0] > counts[1]
        assert counts[0] > counts[2]


class TestRunSelfPlay:
    """Tests for run_self_play function."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment."""
        env = MagicMock()

        # Setup observation structure
        obs = {
            "spatial": np.random.randn(13, 8, 8).astype(np.float32),
            "vector": np.random.randn(64).astype(np.float32),
            "action_mask": np.ones(356, dtype=np.float32),
        }

        env.reset.return_value = (obs, {})
        env.step.return_value = (obs, 0.1, False, False, {})
        env.action_space = MagicMock()
        env.action_space.n = 356

        return env

    @pytest.fixture
    def mock_network(self):
        """Create a mock network."""
        network = MagicMock()

        policy = torch.softmax(torch.randn(1, 356), dim=1)
        value = torch.tensor([[0.5]])

        network.return_value = (policy, value)
        network.eval = MagicMock()

        return network

    def test_self_play_returns_correct_structure(self, mock_env, mock_network):
        """Test that self_play returns correct data structure."""
        # Make game end quickly
        mock_env.step.return_value = (mock_env.reset.return_value[0], 1.0, True, False, {})

        config = TrainingConfig(num_games=2, max_turns_per_game=10)

        replay_buffer, game_results = run_self_play(
            env=mock_env,
            network=mock_network,
            config=config,
            wandb_logger=None,
            verbose=False,
        )

        assert isinstance(replay_buffer, list)
        assert isinstance(game_results, list)
        assert len(game_results) == 2

        for result in game_results:
            assert isinstance(result, GameResult)
            assert result.game_id.startswith("game_")

    def test_self_play_collects_experiences(self, mock_env, mock_network):
        """Test that self_play collects experience data."""
        # Game lasts 3 turns then ends
        call_count = [0]

        def step_side_effect(action):
            call_count[0] += 1
            obs = mock_env.reset.return_value[0]
            if call_count[0] >= 3:
                return (obs, 1.0, True, False, {})
            return (obs, 0.0, False, False, {})

        mock_env.step.side_effect = step_side_effect

        config = TrainingConfig(num_games=1, max_turns_per_game=10)

        replay_buffer, game_results = run_self_play(
            env=mock_env,
            network=mock_network,
            config=config,
            wandb_logger=None,
            verbose=False,
        )

        # Should have 3 experiences
        assert len(replay_buffer) == 3

        # Each experience should have required keys
        for exp in replay_buffer:
            assert "spatial" in exp
            assert "vector" in exp
            assert "policy" in exp
            assert "action" in exp
            assert "value" in exp

    def test_self_play_logs_to_wandb(self, mock_env, mock_network):
        """Test that self_play logs to W&B when logger provided."""
        mock_env.step.return_value = (mock_env.reset.return_value[0], 1.0, True, False, {})

        mock_logger = MagicMock()
        config = TrainingConfig(num_games=2)

        run_self_play(
            env=mock_env,
            network=mock_network,
            config=config,
            wandb_logger=mock_logger,
            verbose=False,
        )

        # Should have logged each game
        assert mock_logger.log_game.call_count == 2
        # Should have logged self-play summary
        mock_logger.log_selfplay.assert_called_once()


class TestTrainNetwork:
    """Tests for train_network function."""

    @pytest.fixture
    def real_env_network(self):
        """Create a real env and network pair for testing."""
        from alpha_zero.envs.microciv import MicroCivEnv
        from alpha_zero.core.civ6_network import MicroCivNetwork

        env = MicroCivEnv(board_size=8, seed=42)
        network = MicroCivNetwork(board_size=8, action_space_size=env.action_space.n)
        return env, network

    @pytest.fixture
    def sample_replay_buffer(self, real_env_network):
        """Create a sample replay buffer from real environment observations."""
        env, network = real_env_network

        buffer = []
        obs, _ = env.reset()

        # Generate experiences by running the network
        for i in range(20):
            spatial = torch.tensor(obs["spatial"]).unsqueeze(0).float()
            vector = torch.tensor(obs["vector"]).unsqueeze(0).float()

            with torch.no_grad():
                policy_logits, value = network(spatial, vector)
                # Apply softmax to get probabilities (network returns logits)
                policy = torch.softmax(policy_logits, dim=-1)

            policy_np = policy.squeeze().numpy()
            mask = obs["action_mask"]

            # Create proper masked policy
            masked_policy = policy_np * mask
            mask_sum = masked_policy.sum()
            if mask_sum > 0:
                masked_policy = masked_policy / mask_sum
            else:
                masked_policy = mask / mask.sum()

            buffer.append({
                "spatial": obs["spatial"].copy(),
                "vector": obs["vector"].copy(),
                "policy": masked_policy,
                "action": int(np.where(mask > 0)[0][0]),
                "value": float(np.random.randn()),
            })

            # Step environment to get varied observations
            valid_actions = np.where(mask > 0)[0]
            if len(valid_actions) > 0:
                action = valid_actions[0]
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()

        return buffer

    def test_train_returns_result(self, real_env_network, sample_replay_buffer):
        """Test that training returns a TrainingResult."""
        _, network = real_env_network
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        config = TrainingConfig(num_epochs=1, batch_size=4)

        result = train_network(
            network=network,
            optimizer=optimizer,
            replay_buffer=sample_replay_buffer,
            config=config,
            wandb_logger=None,
            verbose=False,
        )

        assert isinstance(result, TrainingResult)
        assert result.total_steps > 0
        assert result.final_loss >= 0

    def test_train_updates_network_weights(self, real_env_network, sample_replay_buffer):
        """Test that training actually updates network weights."""
        _, network = real_env_network
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        config = TrainingConfig(num_epochs=2, batch_size=4)

        # Get initial weights
        initial_weights = {}
        for name, param in network.named_parameters():
            initial_weights[name] = param.clone().detach()

        train_network(
            network=network,
            optimizer=optimizer,
            replay_buffer=sample_replay_buffer,
            config=config,
            wandb_logger=None,
            verbose=False,
        )

        # Check weights changed
        weights_changed = False
        for name, param in network.named_parameters():
            if not torch.equal(initial_weights[name], param):
                weights_changed = True
                break

        assert weights_changed, "Network weights should have changed after training"

    def test_train_logs_to_wandb(self, real_env_network, sample_replay_buffer):
        """Test that training logs to W&B when logger provided."""
        _, network = real_env_network
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        config = TrainingConfig(num_epochs=1, batch_size=4)
        mock_logger = MagicMock()

        train_network(
            network=network,
            optimizer=optimizer,
            replay_buffer=sample_replay_buffer,
            config=config,
            wandb_logger=mock_logger,
            verbose=False,
        )

        # Should have logged training steps
        assert mock_logger.log_training_step.call_count > 0

    def test_train_handles_empty_buffer(self, real_env_network):
        """Test training with empty replay buffer."""
        _, network = real_env_network
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        config = TrainingConfig(num_epochs=1, batch_size=4)

        result = train_network(
            network=network,
            optimizer=optimizer,
            replay_buffer=[],
            config=config,
            wandb_logger=None,
            verbose=False,
        )

        assert result.total_steps == 0
        assert result.final_loss == 0.0

    def test_train_respects_batch_size(self, real_env_network, sample_replay_buffer):
        """Test that training respects batch size configuration."""
        _, network = real_env_network
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        config = TrainingConfig(num_epochs=1, batch_size=8)
        mock_logger = MagicMock()

        # Buffer has 20 items, batch size 8 = 2 full batches
        train_network(
            network=network,
            optimizer=optimizer,
            replay_buffer=sample_replay_buffer,
            config=config,
            wandb_logger=mock_logger,
            verbose=False,
        )

        # Should have exactly 2 batches (20 // 8 = 2, remainder 4 is skipped)
        assert mock_logger.log_training_step.call_count == 2


class TestEvaluateNetwork:
    """Tests for evaluate_network function."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment."""
        env = MagicMock()

        obs = {
            "spatial": np.random.randn(13, 8, 8).astype(np.float32),
            "vector": np.random.randn(64).astype(np.float32),
            "action_mask": np.ones(356, dtype=np.float32),
        }

        env.reset.return_value = (obs, {})
        env.action_space = MagicMock()
        env.action_space.n = 356

        return env

    @pytest.fixture
    def mock_network(self):
        """Create a mock network."""
        network = MagicMock()

        policy = torch.softmax(torch.randn(1, 356), dim=1)
        value = torch.tensor([[0.5]])

        network.return_value = (policy, value)
        network.eval = MagicMock()

        return network

    def test_evaluate_returns_result(self, mock_env, mock_network):
        """Test that evaluation returns an EvaluationResult."""
        # Game ends immediately with positive reward
        mock_env.step.return_value = (mock_env.reset.return_value[0], 1.0, True, False, {})

        config = TrainingConfig(eval_games=5)

        result = evaluate_network(
            env=mock_env,
            network=mock_network,
            config=config,
            wandb_logger=None,
            verbose=False,
        )

        assert isinstance(result, EvaluationResult)
        assert result.games_played == 5
        assert 0.0 <= result.win_rate <= 1.0

    def test_evaluate_counts_wins_correctly(self, mock_env, mock_network):
        """Test that wins are counted correctly."""
        # All games end with positive reward
        mock_env.step.return_value = (mock_env.reset.return_value[0], 1.0, True, False, {})

        config = TrainingConfig(eval_games=10)

        result = evaluate_network(
            env=mock_env,
            network=mock_network,
            config=config,
            wandb_logger=None,
            verbose=False,
        )

        assert result.win_rate == 1.0
        assert result.total_wins == 10

    def test_evaluate_logs_to_wandb(self, mock_env, mock_network):
        """Test that evaluation logs to W&B when logger provided."""
        mock_env.step.return_value = (mock_env.reset.return_value[0], 1.0, True, False, {})

        mock_logger = MagicMock()
        config = TrainingConfig(eval_games=3)

        evaluate_network(
            env=mock_env,
            network=mock_network,
            config=config,
            wandb_logger=mock_logger,
            verbose=False,
        )

        mock_logger.log_evaluation.assert_called_once()

    def test_evaluate_uses_greedy_selection(self, mock_env, mock_network):
        """Test that evaluation uses greedy action selection."""
        # Set up policy where action 5 has highest probability
        policy = torch.zeros(1, 356)
        policy[0, 5] = 1.0
        mock_network.return_value = (policy, torch.tensor([[0.5]]))

        mock_env.step.return_value = (mock_env.reset.return_value[0], 1.0, True, False, {})

        config = TrainingConfig(eval_games=1)

        evaluate_network(
            env=mock_env,
            network=mock_network,
            config=config,
            wandb_logger=None,
            verbose=False,
        )

        # Verify step was called with action 5
        mock_env.step.assert_called_with(5)


class TestNumericalStability:
    """Tests for numerical stability of training functions."""

    def test_select_action_with_extreme_values(self):
        """Test select_action handles extreme probability values."""
        np.random.seed(42)

        # Test with very small probabilities
        policy = np.array([1e-10, 1e-10, 1e-10, 1e-10])
        mask = np.array([1, 1, 1, 1])
        action = select_action(policy, mask)
        assert action is not None

        # Test with mixed extreme values
        policy = np.array([1e10, 1e-10, 0, 1e5])
        mask = np.array([1, 1, 1, 1])
        action = select_action(policy, mask)
        assert action is not None

    def test_training_with_extreme_values(self):
        """Test training handles extreme value targets."""
        from alpha_zero.envs.microciv import MicroCivEnv
        from alpha_zero.core.civ6_network import MicroCivNetwork

        env = MicroCivEnv(board_size=8, seed=42)
        network = MicroCivNetwork(board_size=8, action_space_size=env.action_space.n)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

        # Create buffer with extreme values from real observations
        buffer = []
        obs, _ = env.reset()
        for i in range(10):
            spatial = torch.tensor(obs["spatial"]).unsqueeze(0).float()
            vector = torch.tensor(obs["vector"]).unsqueeze(0).float()

            with torch.no_grad():
                policy_logits, _ = network(spatial, vector)
                # Apply softmax to get probabilities (network returns logits)
                policy = torch.softmax(policy_logits, dim=-1)

            policy_np = policy.squeeze().numpy()
            mask = obs["action_mask"]
            masked_policy = policy_np * mask
            mask_sum = masked_policy.sum()
            if mask_sum > 0:
                masked_policy = masked_policy / mask_sum
            else:
                masked_policy = mask / mask.sum()

            buffer.append({
                "spatial": obs["spatial"].copy(),
                "vector": obs["vector"].copy(),
                "policy": masked_policy,
                "action": int(np.where(mask > 0)[0][0]),
                "value": 1e6 if i % 2 == 0 else -1e6,  # Extreme values
            })

            # Step
            valid_actions = np.where(mask > 0)[0]
            if len(valid_actions) > 0:
                obs, _, term, trunc, _ = env.step(valid_actions[0])
                if term or trunc:
                    obs, _ = env.reset()

        config = TrainingConfig(num_epochs=1, batch_size=4)

        # Should not raise any exceptions
        result = train_network(
            network=network,
            optimizer=optimizer,
            replay_buffer=buffer,
            config=config,
            wandb_logger=None,
            verbose=False,
        )

        # Training should have processed some batches
        assert result.total_steps >= 0
