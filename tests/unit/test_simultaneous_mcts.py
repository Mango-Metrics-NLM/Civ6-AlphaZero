"""Tests for Decoupled UCT (simultaneous-move MCTS).

Tests SimultaneousNode, SimultaneousMCTSConfig, and DecoupledUCT
for correct behavior in simultaneous-move multiplayer games.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import torch

from alpha_zero.multiplayer.simultaneous_mcts import (
    ChildStats,
    DecoupledUCT,
    SimultaneousMCTSConfig,
    SimultaneousNode,
)

# =============================================================================
# Helper fixtures
# =============================================================================


def _make_mock_env(
    num_players: int = 2,
    action_space_size: int = 10,
    current_player: int = 0,
) -> MagicMock:
    """Create a mock multiplayer environment.

    Each call to env.clone() returns a fresh mock with the same
    configuration, preventing mutations from leaking across simulations.
    """

    def _build_env() -> MagicMock:
        env = MagicMock()
        env.num_players = num_players
        env.current_player = current_player

        action_space = MagicMock()
        action_space.n = action_space_size
        env.action_space = action_space

        spatial = np.random.randn(13, 8, 8).astype(np.float32)
        vector = np.random.randn(64).astype(np.float32)
        action_mask = np.ones(action_space_size, dtype=np.float32)

        env.get_observation.return_value = {
            "spatial": spatial,
            "vector": vector,
            "action_mask": action_mask,
        }
        env.get_valid_actions.return_value = action_mask

        # step returns non-terminal by default
        env.step.return_value = (
            {"spatial": spatial, "vector": vector, "action_mask": action_mask},
            0.0,
            False,
            False,
            {},
        )

        # clone() returns a fresh mock each time
        env.clone.side_effect = lambda: _build_env()
        return env

    return _build_env()


def _make_mock_network(
    num_players: int = 2,
    action_space_size: int = 10,
) -> MagicMock:
    """Create a mock multiplayer network."""
    network = MagicMock()
    type(network).num_players = PropertyMock(return_value=num_players)

    def mock_predict(spatial, vector, action_mask):
        batch = spatial.shape[0]
        policy = torch.ones(batch, action_space_size) / action_space_size
        values = torch.zeros(batch, num_players)
        for i in range(num_players):
            values[0, i] = 0.1 * (i + 1)
        return policy, values

    network.predict = mock_predict
    return network


# =============================================================================
# SimultaneousNode tests
# =============================================================================


class TestSimultaneousNode:
    """Tests for SimultaneousNode dataclass."""

    def test_node_creation_defaults(self) -> None:
        """Default node has expected initial values."""
        node = SimultaneousNode()
        assert node.num_players == 2
        assert node.visit_count == 0
        assert node.value_sums == {}
        assert node.per_player_children == {}
        assert node.joint_children == {}

    def test_per_player_stats_independence(self) -> None:
        """Per-player action stats are independent."""
        node = SimultaneousNode(num_players=2)
        policy = np.array([0.3, 0.7])
        mask = np.array([1, 1])

        node.expand_simultaneous(
            policies={0: policy, 1: policy.copy()},
            action_masks={0: mask, 1: mask.copy()},
        )

        # Each player has independent stats
        assert 0 in node.per_player_children
        assert 1 in node.per_player_children

        # Modify player 0's stats - should not affect player 1
        node.per_player_children[0][0].visit_count = 10
        assert node.per_player_children[1][0].visit_count == 0

    def test_expand_simultaneous_creates_stats(self) -> None:
        """expand_simultaneous creates ChildStats for valid actions."""
        node = SimultaneousNode(num_players=2)
        policy = np.array([0.2, 0.3, 0.0, 0.5])
        mask = np.array([1, 1, 0, 1])  # Action 2 is invalid

        node.expand_simultaneous(
            policies={0: policy, 1: policy.copy()},
            action_masks={0: mask, 1: mask.copy()},
        )

        # Only valid actions should have stats
        for pid in [0, 1]:
            assert 0 in node.per_player_children[pid]
            assert 1 in node.per_player_children[pid]
            assert 2 not in node.per_player_children[pid]
            assert 3 in node.per_player_children[pid]

        # Priors should be normalized over valid actions
        p0_priors = [node.per_player_children[0][a].prior for a in [0, 1, 3]]
        assert abs(sum(p0_priors) - 1.0) < 1e-6

    def test_ucb_selection_exploration(self) -> None:
        """UCB selects unvisited actions when exploration is high."""
        node = SimultaneousNode(num_players=1)
        node.visit_count = 100

        # Create stats with one visited and one unvisited action
        node.per_player_children[0] = {
            0: ChildStats(visit_count=50, value_sum=25.0, prior=0.5),
            1: ChildStats(visit_count=0, value_sum=0.0, prior=0.5),
        }

        # With high c_puct, should select unvisited action
        selected = node.select_action_for_player(0, c_puct=10.0)
        assert selected == 1  # Unvisited action has infinite UCB bonus

    def test_ucb_selection_exploitation(self) -> None:
        """UCB selects high-value actions when exploration is low."""
        node = SimultaneousNode(num_players=1)
        node.visit_count = 1000

        # Action 0 has high value, action 1 has low value
        node.per_player_children[0] = {
            0: ChildStats(visit_count=500, value_sum=450.0, prior=0.5),
            1: ChildStats(visit_count=500, value_sum=50.0, prior=0.5),
        }

        # With low c_puct, should select high-value action
        selected = node.select_action_for_player(0, c_puct=0.01)
        assert selected == 0

    def test_visit_count_tracking(self) -> None:
        """update_stats correctly increments visit counts."""
        node = SimultaneousNode(num_players=2)
        policy = np.array([0.5, 0.5])
        mask = np.array([1, 1])

        node.expand_simultaneous(
            policies={0: policy, 1: policy.copy()},
            action_masks={0: mask, 1: mask.copy()},
        )

        # Update with a joint action
        joint_action = {0: 0, 1: 1}
        values = {0: 0.5, 1: -0.5}
        node.update_stats(joint_action, values)

        assert node.visit_count == 1
        assert node.per_player_children[0][0].visit_count == 1
        assert node.per_player_children[0][1].visit_count == 0
        assert node.per_player_children[1][0].visit_count == 0
        assert node.per_player_children[1][1].visit_count == 1

    def test_action_probabilities(self) -> None:
        """get_action_probabilities_for_player returns valid distribution."""
        node = SimultaneousNode(num_players=2)
        node.per_player_children[0] = {
            0: ChildStats(visit_count=30, value_sum=15.0, prior=0.5),
            1: ChildStats(visit_count=70, value_sum=35.0, prior=0.5),
        }

        probs = node.get_action_probabilities_for_player(
            player_id=0, action_space_size=5, temperature=1.0
        )

        assert probs.shape == (5,)
        assert abs(probs.sum() - 1.0) < 1e-6
        assert probs[1] > probs[0]  # Action 1 has more visits

    def test_action_probabilities_greedy(self) -> None:
        """Temperature=0 gives greedy (argmax) probabilities."""
        node = SimultaneousNode(num_players=2)
        node.per_player_children[0] = {
            0: ChildStats(visit_count=30, value_sum=15.0, prior=0.5),
            1: ChildStats(visit_count=70, value_sum=35.0, prior=0.5),
        }

        probs = node.get_action_probabilities_for_player(
            player_id=0, action_space_size=5, temperature=0.0
        )

        assert probs[1] == 1.0  # Most visited action gets all probability
        assert probs[0] == 0.0

    def test_joint_children_lazy_creation(self) -> None:
        """Joint children are created lazily (only when visited)."""
        node = SimultaneousNode(num_players=2)
        policy = np.array([0.5, 0.5])
        mask = np.array([1, 1])

        node.expand_simultaneous(
            policies={0: policy, 1: policy.copy()},
            action_masks={0: mask, 1: mask.copy()},
        )

        # Initially no joint children
        assert len(node.joint_children) == 0

        # Create a joint child
        joint_key = (0, 1)  # Player 0 -> action 0, Player 1 -> action 1
        child = SimultaneousNode(num_players=2)
        node.joint_children[joint_key] = child

        assert len(node.joint_children) == 1
        assert node.joint_children[joint_key] is child

    def test_is_expanded(self) -> None:
        """is_expanded returns True only after expansion."""
        node = SimultaneousNode(num_players=2)
        assert not node.is_expanded()

        policy = np.array([0.5, 0.5])
        mask = np.array([1, 1])
        node.expand_simultaneous(
            policies={0: policy},
            action_masks={0: mask},
        )
        assert node.is_expanded()

    def test_update_stats_accumulates_values(self) -> None:
        """update_stats accumulates values across multiple updates."""
        node = SimultaneousNode(num_players=2)
        policy = np.array([0.5, 0.5])
        mask = np.array([1, 1])
        node.expand_simultaneous(
            policies={0: policy, 1: policy.copy()},
            action_masks={0: mask, 1: mask.copy()},
        )

        # First update
        node.update_stats({0: 0, 1: 0}, {0: 1.0, 1: -1.0})
        # Second update
        node.update_stats({0: 0, 1: 1}, {0: 0.5, 1: 0.5})

        assert node.visit_count == 2
        assert node.value_sums[0] == 1.5
        assert node.value_sums[1] == -0.5
        assert node.per_player_children[0][0].visit_count == 2

    def test_empty_player_stats(self) -> None:
        """get_action_probabilities returns zeros for unknown player."""
        node = SimultaneousNode(num_players=2)
        probs = node.get_action_probabilities_for_player(player_id=0, action_space_size=5)
        assert probs.shape == (5,)
        assert probs.sum() == 0.0


# =============================================================================
# SimultaneousMCTSConfig tests
# =============================================================================


class TestSimultaneousMCTSConfig:
    """Tests for SimultaneousMCTSConfig dataclass."""

    def test_defaults(self) -> None:
        """Default config has expected values."""
        config = SimultaneousMCTSConfig()
        assert config.num_simulations == 200
        assert config.c_puct == 1.5
        assert config.dirichlet_alpha == 0.3
        assert config.dirichlet_epsilon == 0.25
        assert config.temperature == 1.0
        assert config.temperature_threshold == 30
        assert config.action_resolution == "independent"

    def test_config_validation(self) -> None:
        """Invalid config values raise ValueError."""
        with pytest.raises(ValueError, match="num_simulations must be >= 1"):
            SimultaneousMCTSConfig(num_simulations=0)

        with pytest.raises(ValueError, match="c_puct must be > 0"):
            SimultaneousMCTSConfig(c_puct=-1.0)

        with pytest.raises(ValueError, match="dirichlet_epsilon must be in"):
            SimultaneousMCTSConfig(dirichlet_epsilon=1.5)

        with pytest.raises(ValueError, match="action_resolution must be one of"):
            SimultaneousMCTSConfig(action_resolution="invalid")

    def test_frozen(self) -> None:
        """Config is frozen."""
        config = SimultaneousMCTSConfig()
        with pytest.raises(AttributeError):
            config.num_simulations = 100  # type: ignore[misc]


# =============================================================================
# DecoupledUCT tests
# =============================================================================


class TestDecoupledUCT:
    """Tests for DecoupledUCT class."""

    def test_init_defaults(self) -> None:
        """Constructor stores config and player info."""
        config = SimultaneousMCTSConfig(num_simulations=50)
        duct = DecoupledUCT(config, num_players=2, action_space_size=10)

        assert duct.config is config
        assert duct.num_players == 2
        assert duct.action_space_size == 10

    def test_config_validation(self) -> None:
        """DecoupledUCT rejects invalid config."""
        with pytest.raises(ValueError):
            DecoupledUCT(
                SimultaneousMCTSConfig(num_simulations=0),
                num_players=2,
                action_space_size=10,
            )

    def test_search_returns_per_player_probs(self) -> None:
        """search() returns dict of per-player probability arrays."""
        config = SimultaneousMCTSConfig(num_simulations=10)
        duct = DecoupledUCT(config, num_players=2, action_space_size=10)

        env = _make_mock_env(num_players=2, action_space_size=10)
        network = _make_mock_network(num_players=2, action_space_size=10)

        result = duct.search(env, network, active_players=[0, 1])

        assert isinstance(result, dict)
        assert 0 in result
        assert 1 in result
        assert result[0].shape == (10,)
        assert result[1].shape == (10,)

    def test_search_probs_sum_to_one(self) -> None:
        """Search probabilities sum to 1.0 for each player."""
        config = SimultaneousMCTSConfig(num_simulations=20)
        duct = DecoupledUCT(config, num_players=2, action_space_size=10)

        env = _make_mock_env(num_players=2, action_space_size=10)
        network = _make_mock_network(num_players=2, action_space_size=10)

        result = duct.search(env, network, active_players=[0, 1])

        assert abs(result[0].sum() - 1.0) < 1e-5
        assert abs(result[1].sum() - 1.0) < 1e-5

    def test_2player_search(self) -> None:
        """Search works correctly for 2-player games."""
        config = SimultaneousMCTSConfig(num_simulations=15)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_mock_env(num_players=2, action_space_size=5)
        network = _make_mock_network(num_players=2, action_space_size=5)

        result = duct.search(env, network, active_players=[0, 1])

        assert len(result) == 2
        for pid in [0, 1]:
            assert result[pid].shape == (5,)
            assert all(result[pid] >= 0)

    def test_4player_search(self) -> None:
        """Search works correctly for 4-player games."""
        config = SimultaneousMCTSConfig(num_simulations=10)
        duct = DecoupledUCT(config, num_players=4, action_space_size=8)

        env = _make_mock_env(num_players=4, action_space_size=8)
        network = _make_mock_network(num_players=4, action_space_size=8)

        result = duct.search(env, network, active_players=[0, 1, 2, 3])

        assert len(result) == 4
        for pid in range(4):
            assert result[pid].shape == (8,)
            assert abs(result[pid].sum() - 1.0) < 1e-5

    def test_dirichlet_noise_applied(self) -> None:
        """Dirichlet noise modifies root priors."""
        config = SimultaneousMCTSConfig(
            num_simulations=1,
            dirichlet_epsilon=0.5,
        )
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        # Create and expand root manually
        root = SimultaneousNode(num_players=2)
        policy = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        mask = np.ones(5)
        root.expand_simultaneous(
            policies={0: policy, 1: policy.copy()},
            action_masks={0: mask, 1: mask.copy()},
        )

        # Store original priors
        original_priors_p0 = [root.per_player_children[0][a].prior for a in range(5)]

        # Apply noise
        np.random.seed(42)
        duct._add_dirichlet_noise(root, active_players=[0, 1])

        # Priors should be modified
        new_priors_p0 = [root.per_player_children[0][a].prior for a in range(5)]
        assert original_priors_p0 != new_priors_p0
        # Should still be valid distribution (approximately)
        assert abs(sum(new_priors_p0) - 1.0) < 1e-5

    def test_backprop_updates_all_players(self) -> None:
        """Backpropagation updates statistics for all players."""
        config = SimultaneousMCTSConfig(num_simulations=5)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        # Create root and expand
        root = SimultaneousNode(num_players=2)
        policy = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        mask = np.ones(5)
        root.expand_simultaneous(
            policies={0: policy, 1: policy.copy()},
            action_masks={0: mask, 1: mask.copy()},
        )

        # Manually backpropagate
        joint_action = {0: 2, 1: 3}
        values = {0: 0.8, 1: -0.3}
        duct._backpropagate([root], [joint_action], values)

        assert root.visit_count == 1
        assert root.value_sums[0] == 0.8
        assert root.value_sums[1] == -0.3
        assert root.per_player_children[0][2].visit_count == 1
        assert root.per_player_children[1][3].visit_count == 1

    def test_backward_compat_2player(self) -> None:
        """DecoupledUCT works as a drop-in for 2-player simultaneous games."""
        config = SimultaneousMCTSConfig(num_simulations=5)
        duct = DecoupledUCT(config, num_players=2, action_space_size=10)

        env = _make_mock_env(num_players=2, action_space_size=10)
        network = _make_mock_network(num_players=2, action_space_size=10)

        result = duct.search(env, network, active_players=[0, 1])

        # Should return exactly 2 player probability distributions
        assert len(result) == 2
        assert 0 in result
        assert 1 in result

    def test_terminal_state_handling(self) -> None:
        """Search handles terminal states correctly."""
        config = SimultaneousMCTSConfig(num_simulations=5)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_mock_env(num_players=2, action_space_size=5)
        # Make step return terminal state
        env.step.return_value = (
            env.get_observation(),
            1.0,
            True,  # terminated
            False,
            {"winner": 0, "player_scores": {0: 1.0, 1: -1.0}},
        )

        network = _make_mock_network(num_players=2, action_space_size=5)

        result = duct.search(env, network, active_players=[0, 1])
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_search_count_increments(self) -> None:
        """Search count increments with each search call."""
        config = SimultaneousMCTSConfig(num_simulations=3)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_mock_env(num_players=2, action_space_size=5)
        network = _make_mock_network(num_players=2, action_space_size=5)

        assert duct._search_count == 0
        duct.search(env, network, active_players=[0, 1])
        assert duct._search_count == 1
        duct.search(env, network, active_players=[0, 1])
        assert duct._search_count == 2

    def test_subset_active_players(self) -> None:
        """Search works when only a subset of players are active."""
        config = SimultaneousMCTSConfig(num_simulations=5)
        duct = DecoupledUCT(config, num_players=4, action_space_size=5)

        env = _make_mock_env(num_players=4, action_space_size=5)
        network = _make_mock_network(num_players=4, action_space_size=5)

        # Only players 0 and 2 act simultaneously
        result = duct.search(env, network, active_players=[0, 2])

        assert len(result) == 2
        assert 0 in result
        assert 2 in result
        assert 1 not in result
        assert 3 not in result


# =============================================================================
# ChildStats tests
# =============================================================================


class TestChildStats:
    """Tests for ChildStats dataclass."""

    def test_defaults(self) -> None:
        """Default ChildStats has zero values."""
        stats = ChildStats()
        assert stats.visit_count == 0
        assert stats.value_sum == 0.0
        assert stats.prior == 0.0

    def test_custom_values(self) -> None:
        """ChildStats accepts custom values."""
        stats = ChildStats(visit_count=10, value_sum=5.0, prior=0.3)
        assert stats.visit_count == 10
        assert stats.value_sum == 5.0
        assert stats.prior == 0.3


# =============================================================================
# Additional coverage tests
# =============================================================================


class TestDecoupledUCTExtended:
    """Extended tests for DecoupledUCT edge cases."""

    def test_temperature_threshold_switches_to_greedy(self) -> None:
        """After temperature_threshold, search uses greedy (temperature=0)."""
        config = SimultaneousMCTSConfig(
            num_simulations=10,
            temperature=1.0,
            temperature_threshold=5,
        )
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_mock_env(num_players=2, action_space_size=5)
        network = _make_mock_network(num_players=2, action_space_size=5)

        # Before threshold: proportional probabilities (warm up the search)
        duct.search(env, network, active_players=[0, 1], turn_number=3)
        # After threshold: greedy probabilities
        result_after = duct.search(env, network, active_players=[0, 1], turn_number=10)

        # Greedy should have exactly one action with probability 1.0
        for pid in [0, 1]:
            assert result_after[pid].max() == 1.0, (
                "After threshold, max prob should be 1.0 (greedy)"
            )

    def test_extract_terminal_values_draw(self) -> None:
        """Terminal state with no winner/scores returns all zeros."""
        config = SimultaneousMCTSConfig(num_simulations=5)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        values = duct._extract_terminal_values(info={}, active_players=[0, 1])
        assert values == {0: 0.0, 1: 0.0}

    def test_extract_terminal_values_winner(self) -> None:
        """Terminal state with winner returns +1/-1."""
        config = SimultaneousMCTSConfig(num_simulations=5)
        duct = DecoupledUCT(config, num_players=3, action_space_size=5)

        values = duct._extract_terminal_values(info={"winner": 1}, active_players=[0, 1, 2])
        assert values[1] == 1.0
        assert values[0] == -1.0
        assert values[2] == -1.0

    def test_action_resolver_is_wired(self) -> None:
        """DecoupledUCT creates an ActionResolver from config."""
        config = SimultaneousMCTSConfig(action_resolution="priority")
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)
        assert duct._action_resolver is not None
        assert duct._action_resolver._strategy.value == "priority"

    def test_expand_node_populates_children(self) -> None:
        """_expand_node populates per-player children and returns values."""
        config = SimultaneousMCTSConfig(num_simulations=5)
        duct = DecoupledUCT(config, num_players=2, action_space_size=10)

        env = _make_mock_env(num_players=2, action_space_size=10)
        network = _make_mock_network(num_players=2, action_space_size=10)

        node = SimultaneousNode(num_players=2)
        assert not node.is_expanded()

        values = duct._expand_node(node, env, network, active_players=[0, 1])

        # Node should now be expanded with per-player children
        assert node.is_expanded()
        assert 0 in node.per_player_children
        assert 1 in node.per_player_children
        # Each player should have action stats for all valid actions
        assert len(node.per_player_children[0]) == 10
        assert len(node.per_player_children[1]) == 10
        # Values should be returned as per-player dict
        assert isinstance(values, dict)
        assert 0 in values
        assert 1 in values

    def test_step_env_with_resolution_calls_resolver(self) -> None:
        """_step_env_with_resolution invokes ActionResolver and env.step."""
        config = SimultaneousMCTSConfig(num_simulations=5, action_resolution="priority")
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_mock_env(num_players=2, action_space_size=5)
        joint_action = {0: 2, 1: 3}

        is_terminal, info = duct._step_env_with_resolution(env, joint_action, active_players=[0, 1])

        # env.step should have been called exactly once
        env.step.assert_called_once()
        # Default mock env is non-terminal
        assert not is_terminal
        assert isinstance(info, dict)
