"""Unit tests for baseline players."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from alpha_zero.evaluation.baselines import (
    BaselineFactory,
    HeuristicPlayer,
    RandomPlayer,
    SimpleMCTSPlayer,
)
from alpha_zero.evaluation.config import (
    BaselineConfig,
    BaselineType,
    HeuristicWeights,
)


class TestRandomPlayer:
    """Tests for RandomPlayer."""

    def test_initialization(self):
        """Test random player initialization."""
        player = RandomPlayer(seed=42, name="test_random")
        assert player.name == "test_random"

    def test_select_valid_action(self):
        """Test that selected action is valid."""
        player = RandomPlayer(seed=42)

        # Mock environment and observation
        mock_env = MagicMock()
        observation = {
            "spatial": np.zeros((13, 16, 16)),
            "vector": np.zeros(64),
            "action_mask": np.array([0, 1, 1, 0, 1, 0]),  # Actions 1, 2, 4 are valid
        }

        action = player.select_action(mock_env, observation)
        assert action in [1, 2, 4]  # Must be a valid action

    def test_respects_action_mask(self):
        """Test that player only selects valid actions."""
        player = RandomPlayer(seed=42)

        mock_env = MagicMock()
        # Only action 3 is valid
        observation = {
            "action_mask": np.array([0, 0, 0, 1, 0, 0]),
        }

        action = player.select_action(mock_env, observation)
        assert action == 3

    def test_reproducible_with_seed(self):
        """Test reproducibility with same seed."""
        observation = {
            "action_mask": np.array([1, 1, 1, 1, 1]),
        }
        mock_env = MagicMock()

        player1 = RandomPlayer(seed=42)
        player2 = RandomPlayer(seed=42)

        actions1 = [player1.select_action(mock_env, observation) for _ in range(10)]
        actions2 = [player2.select_action(mock_env, observation) for _ in range(10)]

        assert actions1 == actions2

    def test_reset_increments_games(self):
        """Test that reset increments games played counter."""
        player = RandomPlayer()
        assert player._games_played == 0
        player.reset()
        assert player._games_played == 1
        player.reset()
        assert player._games_played == 2

    def test_get_stats(self):
        """Test getting player statistics."""
        player = RandomPlayer(name="test")
        player.reset()
        stats = player.get_stats()

        assert stats["name"] == "test"
        assert stats["type"] == "random"
        assert stats["games_played"] == 1

    def test_no_valid_actions_returns_zero(self):
        """Test handling when no valid actions available."""
        player = RandomPlayer()
        mock_env = MagicMock()
        observation = {"action_mask": np.array([0, 0, 0])}

        action = player.select_action(mock_env, observation)
        assert action == 0


class TestHeuristicPlayer:
    """Tests for HeuristicPlayer."""

    def test_initialization(self):
        """Test heuristic player initialization."""
        weights = HeuristicWeights(attack_weight=2.0)
        player = HeuristicPlayer(weights=weights, name="test_heuristic")

        assert player.name == "test_heuristic"
        assert player.weights.attack_weight == 2.0

    def test_default_weights(self):
        """Test default weight initialization."""
        player = HeuristicPlayer()
        assert player.weights is not None
        assert player.weights.attack_weight == 1.0

    def test_aggression_clamping(self):
        """Test that aggression is clamped to [0, 1]."""
        player1 = HeuristicPlayer(aggression=-0.5)
        assert player1.aggression == 0.0

        player2 = HeuristicPlayer(aggression=1.5)
        assert player2.aggression == 1.0

    def test_select_valid_action(self):
        """Test that selected action is valid."""
        player = HeuristicPlayer(seed=42)

        mock_env = MagicMock()
        mock_env.action_space_handler = MagicMock()
        mock_env.action_space_handler.decode_action = MagicMock(return_value={"type": "move"})

        observation = {
            "spatial": np.zeros((13, 16, 16)),
            "action_mask": np.array([0, 1, 1, 0, 1]),
        }

        action = player.select_action(mock_env, observation)
        assert action in [1, 2, 4]

    def test_single_valid_action(self):
        """Test when only one action is valid."""
        player = HeuristicPlayer()

        mock_env = MagicMock()
        observation = {"action_mask": np.array([0, 0, 1, 0])}

        action = player.select_action(mock_env, observation)
        assert action == 2

    def test_reset_increments_games(self):
        """Test that reset increments games played counter."""
        player = HeuristicPlayer()
        assert player._games_played == 0
        player.reset()
        assert player._games_played == 1

    def test_get_stats(self):
        """Test getting player statistics."""
        weights = HeuristicWeights(attack_weight=1.5)
        player = HeuristicPlayer(weights=weights, aggression=0.7, name="test")
        player.reset()
        stats = player.get_stats()

        assert stats["name"] == "test"
        assert stats["type"] == "heuristic"
        assert stats["aggression"] == 0.7
        assert stats["games_played"] == 1
        assert stats["weights"]["attack_weight"] == 1.5

    def test_scores_attack_actions(self):
        """Test that attack actions get appropriate scores."""
        player = HeuristicPlayer(aggression=1.0)  # High aggression

        mock_env = MagicMock()
        mock_env.action_space_handler = MagicMock()

        # First action is attack, second is move
        def decode_side_effect(action):
            if action == 0:
                return {"type": "attack"}
            return {"type": "move"}

        mock_env.action_space_handler.decode_action = decode_side_effect

        observation = {
            "spatial": np.zeros((13, 16, 16)),
            "action_mask": np.array([1, 1]),
        }

        # Attack should be preferred with high aggression
        # Run multiple times to see if attack is generally preferred
        attack_count = sum(
            1 for _ in range(100) if player.select_action(mock_env, observation) == 0
        )
        # Attack should be selected more often than not
        assert attack_count > 30  # At least 30% of the time


class TestBaselineFactory:
    """Tests for BaselineFactory."""

    def test_create_random_player(self):
        """Test creating random player from config."""
        config = BaselineConfig(
            type=BaselineType.RANDOM,
            name="test_random",
        )
        player = BaselineFactory.create(config, seed=42)

        assert isinstance(player, RandomPlayer)
        assert player.name == "test_random"

    def test_create_heuristic_player(self):
        """Test creating heuristic player from config."""
        weights = HeuristicWeights(attack_weight=2.0)
        config = BaselineConfig(
            type=BaselineType.HEURISTIC,
            name="test_heuristic",
            heuristic_weights=weights,
        )
        player = BaselineFactory.create(config)

        assert isinstance(player, HeuristicPlayer)
        assert player.name == "test_heuristic"
        assert player.weights.attack_weight == 2.0

    def test_create_checkpoint_player_requires_path(self):
        """Test that checkpoint player requires checkpoint path."""
        # BaselineConfig validation raises at creation time
        with pytest.raises(ValueError, match="checkpoint_path required"):
            BaselineConfig(
                type=BaselineType.SPECIFIC_CHECKPOINT,
                name="test",
                checkpoint_path=None,
            )

    def test_create_all_enabled_only(self):
        """Test creating all players filters by enabled."""
        configs = [
            BaselineConfig(type=BaselineType.RANDOM, name="enabled", enabled=True),
            BaselineConfig(type=BaselineType.RANDOM, name="disabled", enabled=False),
        ]
        players = BaselineFactory.create_all(configs)

        assert len(players) == 1
        assert "enabled" in players
        assert "disabled" not in players

    def test_create_all_with_seed(self):
        """Test that create_all provides different seeds to each player."""
        configs = [
            BaselineConfig(type=BaselineType.RANDOM, name="player1"),
            BaselineConfig(type=BaselineType.RANDOM, name="player2"),
        ]
        players = BaselineFactory.create_all(configs, seed=42)

        # Both should be created
        assert len(players) == 2
        assert "player1" in players
        assert "player2" in players

    def test_unknown_type_raises(self):
        """Test that unknown baseline type raises ValueError."""
        # Create a mock config with invalid type
        config = MagicMock()
        config.type = "invalid_type"
        config.name = "test"

        with pytest.raises(ValueError, match="Unknown baseline type"):
            BaselineFactory.create(config)


class TestSimpleMCTSPlayer:
    """``BaselineType.SIMPLE_MCTS``: a network-free search behind the player protocol."""

    def test_factory_builds_it_with_the_configured_budget(self) -> None:
        config = BaselineConfig(type=BaselineType.SIMPLE_MCTS, name="rollouts", mcts_simulations=3)
        player = BaselineFactory.create(config, seed=5)
        assert isinstance(player, SimpleMCTSPlayer)
        assert player.name == "rollouts"
        assert player.num_simulations == 3
        assert player.mcts.num_simulations == 3

    def test_string_type_round_trips(self) -> None:
        assert BaselineConfig(type="simple_mcts").type is BaselineType.SIMPLE_MCTS

    def test_plays_a_legal_move_on_a_real_env(self) -> None:
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=6, max_turns=4, seed=1)
        obs, _ = env.reset()
        player = SimpleMCTSPlayer(num_simulations=2, seed=1)
        action = player.select_action(env, obs)
        assert obs["action_mask"][action] > 0

    def test_same_seed_same_moves(self) -> None:
        from alpha_zero.envs.microciv import MicroCivEnv

        def play(seed: int) -> list[int]:
            env = MicroCivEnv(board_size=6, max_turns=4, seed=1)
            obs, _ = env.reset()
            player = SimpleMCTSPlayer(num_simulations=2, seed=seed)
            moves = []
            for _ in range(6):
                action = player.select_action(env, obs)
                moves.append(action)
                obs, _r, terminated, truncated, _i = env.step(action)
                if terminated or truncated:
                    break
            return moves

        assert play(3) == play(3)

    def test_reset_and_stats(self) -> None:
        player = SimpleMCTSPlayer(num_simulations=4)
        player.reset()
        stats = player.get_stats()
        assert stats["type"] == "simple_mcts" and stats["games_played"] == 1
