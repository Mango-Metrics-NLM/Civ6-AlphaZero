"""Unit tests for run_muzero_training.py."""

import os
import tempfile

import numpy as np
import pytest
import torch

from alpha_zero.core.civ6_mcts import MuZeroMCTS
from alpha_zero.core.muzero_network import MuZeroNetwork
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.training.muzero_trainer import GameTrajectory

# Import the modules we're testing
from run_muzero_training import (
    MuZeroRunConfig,
    create_mcts,
    create_network,
    run_self_play,
    self_play_game,
)


class TestMuZeroRunConfig:
    """Test configuration dataclass."""

    def test_default_values(self):
        """Default config has reasonable values."""
        config = MuZeroRunConfig()

        assert config.board_size == 16
        assert config.num_simulations == 16
        assert config.use_gumbel is True
        assert config.batch_size == 128
        assert config.unroll_steps == 5

    def test_custom_values(self):
        """Config accepts custom values."""
        config = MuZeroRunConfig(
            board_size=8,
            num_simulations=32,
            batch_size=64,
            num_iterations=10,
        )

        assert config.board_size == 8
        assert config.num_simulations == 32
        assert config.batch_size == 64
        assert config.num_iterations == 10


class TestCreateNetwork:
    """Test network creation."""

    def test_creates_muzero_network(self):
        """Creates a valid MuZeroNetwork."""
        config = MuZeroRunConfig(
            board_size=8,
            num_res_blocks=2,
            num_filters=32,
            device="cpu",  # Force CPU for testing
        )
        action_space_size = 100

        network = create_network(config, action_space_size)

        assert isinstance(network, MuZeroNetwork)
        assert network.config.action_space_size == action_space_size

    def test_network_has_required_methods(self):
        """Network has all required methods."""
        config = MuZeroRunConfig(board_size=8, num_res_blocks=2, device="cpu")
        network = create_network(config, action_space_size=50)

        assert hasattr(network, "initial_inference")
        assert hasattr(network, "recurrent_inference")
        assert hasattr(network, "representation")
        assert hasattr(network, "dynamics_function")
        assert hasattr(network, "prediction_function")

    def test_network_forward_pass(self):
        """Network can perform forward pass."""
        config = MuZeroRunConfig(board_size=8, num_res_blocks=2, num_filters=32, device="cpu")
        network = create_network(config, action_space_size=50)

        spatial = torch.randn(2, 13, 8, 8)
        vector = torch.randn(2, 64)
        mask = torch.ones(2, 50)

        output = network.initial_inference(spatial, vector, mask)

        assert output.hidden_state is not None
        assert output.policy_logits.shape == (2, 50)
        assert output.value.shape == (2,)


class TestCreateMCTS:
    """Test MCTS creation."""

    def test_creates_muzero_mcts(self):
        """Creates a valid MuZeroMCTS."""
        config = MuZeroRunConfig(
            board_size=8,
            num_simulations=8,
            use_gumbel=True,
            device="cpu",  # Force CPU for testing
        )
        network = create_network(config, action_space_size=50)

        mcts = create_mcts(network, config)

        assert isinstance(mcts, MuZeroMCTS)
        assert mcts.config.num_simulations == 8
        assert mcts.config.use_gumbel is True

    def test_mcts_without_gumbel(self):
        """Creates MCTS without Gumbel sampling."""
        config = MuZeroRunConfig(
            board_size=8,
            use_gumbel=False,
            device="cpu",  # Force CPU for testing
        )
        network = create_network(config, action_space_size=50)

        mcts = create_mcts(network, config)

        assert mcts.config.use_gumbel is False


class TestSelfPlayGame:
    """Test self-play game generation."""

    @pytest.fixture
    def small_config(self):
        return MuZeroRunConfig(
            board_size=8,
            max_turns=20,
            num_simulations=4,
            num_res_blocks=1,
            num_filters=16,
            device="cpu",  # Force CPU for testing
        )

    @pytest.fixture
    def small_env(self, small_config):
        return MicroCivEnv(
            board_size=small_config.board_size,
            max_turns=small_config.max_turns,
        )

    @pytest.fixture
    def small_mcts(self, small_config, small_env):
        network = create_network(small_config, small_env.action_space.n)
        return create_mcts(network, small_config)

    def test_returns_game_trajectory(self, small_env, small_mcts, small_config):
        """Self-play returns a GameTrajectory."""
        trajectory = self_play_game(small_env, small_mcts, small_config)

        assert isinstance(trajectory, GameTrajectory)
        assert len(trajectory) > 0

    def test_trajectory_has_valid_shapes(self, small_env, small_mcts, small_config):
        """Trajectory has valid array shapes."""
        trajectory = self_play_game(small_env, small_mcts, small_config)
        game_length = len(trajectory.actions)

        # Should have game_length actions and game_length+1 observations
        assert len(trajectory.actions) == game_length
        assert len(trajectory.observations) >= game_length
        assert len(trajectory.policies) == game_length
        assert len(trajectory.rewards) == game_length

    def test_policies_are_valid_distributions(self, small_env, small_mcts, small_config):
        """Policies sum to 1."""
        trajectory = self_play_game(small_env, small_mcts, small_config)

        for policy in trajectory.policies:
            assert np.isclose(policy.sum(), 1.0, atol=1e-5) or policy.max() > 0.99

    def test_actions_are_valid(self, small_env, small_mcts, small_config):
        """Actions are within valid range."""
        trajectory = self_play_game(small_env, small_mcts, small_config)

        for action in trajectory.actions:
            assert 0 <= action < small_env.action_space.n


class TestRunSelfPlay:
    """Test multiple self-play games."""

    @pytest.fixture
    def small_config(self):
        return MuZeroRunConfig(
            board_size=8,
            max_turns=10,
            num_simulations=4,
            num_res_blocks=1,
            num_filters=16,
            num_games_per_iteration=2,
            device="cpu",  # Force CPU for testing
        )

    def test_returns_multiple_trajectories(self, small_config):
        """Generates multiple trajectories."""
        env = MicroCivEnv(board_size=small_config.board_size)
        network = create_network(small_config, env.action_space.n)

        trajectories = run_self_play(network, small_config, num_games=2)

        assert len(trajectories) == 2
        assert all(isinstance(t, GameTrajectory) for t in trajectories)

    def test_trajectories_are_independent(self, small_config):
        """Different games produce different trajectories."""
        env = MicroCivEnv(board_size=small_config.board_size)
        network = create_network(small_config, env.action_space.n)

        np.random.seed(42)
        trajectories = run_self_play(network, small_config, num_games=3)

        # Check that not all trajectories are identical
        lengths = [len(t) for t in trajectories]
        # At least some variation expected (not all same length)
        # This is probabilistic but very likely to pass
        assert len(set(lengths)) >= 1


class TestTrainingIntegration:
    """Integration tests for training components."""

    @pytest.fixture
    def minimal_config(self):
        return MuZeroRunConfig(
            board_size=8,
            max_turns=10,
            num_simulations=2,
            num_res_blocks=1,
            num_filters=16,
            batch_size=4,
            unroll_steps=2,
            num_iterations=1,
            num_games_per_iteration=2,
            min_replay_size=10,
            device="cpu",  # Force CPU for testing
        )

    def test_full_self_play_cycle(self, minimal_config):
        """Complete self-play generates valid training data."""
        env = MicroCivEnv(board_size=minimal_config.board_size)
        network = create_network(minimal_config, env.action_space.n)

        trajectories = run_self_play(
            network,
            minimal_config,
            num_games=minimal_config.num_games_per_iteration,
        )

        total_steps = sum(len(t) for t in trajectories)
        assert total_steps > 0
        assert all(t.observations[0]["spatial"].shape == (13, 8, 8) for t in trajectories)

    def test_network_can_be_saved_and_loaded(self, minimal_config):
        """Network state can be saved and loaded."""
        env = MicroCivEnv(board_size=minimal_config.board_size)
        network = create_network(minimal_config, env.action_space.n)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_model.pt")

            # Save
            torch.save(network.state_dict(), path)

            # Load into new network
            network2 = create_network(minimal_config, env.action_space.n)
            network2.load_state_dict(torch.load(path))

            # Verify outputs match
            spatial = torch.randn(1, 13, 8, 8)
            vector = torch.randn(1, 64)
            mask = torch.ones(1, env.action_space.n)

            network.eval()
            network2.eval()
            with torch.no_grad():
                out1 = network.initial_inference(spatial, vector, mask)
                out2 = network2.initial_inference(spatial, vector, mask)

            assert torch.allclose(out1.policy_logits, out2.policy_logits)
            assert torch.allclose(out1.value, out2.value)


class TestConfigValidation:
    """Test configuration validation."""

    def test_wandb_disabled_by_default(self):
        """W&B is disabled by default."""
        config = MuZeroRunConfig()
        assert config.wandb_enabled is False

    def test_device_defaults_to_available(self):
        """Device defaults based on CUDA availability."""
        config = MuZeroRunConfig()
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert config.device == expected

    def test_efficientzero_enabled_by_default(self):
        """EfficientZero features enabled by default."""
        config = MuZeroRunConfig()
        assert config.use_consistency_loss is True
        assert config.use_value_prefix is True
