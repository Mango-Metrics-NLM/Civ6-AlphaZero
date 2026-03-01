"""Comprehensive tests for Decoupled UCT (simultaneous-move MCTS).

Targets alpha_zero/multiplayer/simultaneous_mcts.py for >= 70% coverage.
Tests all public methods, internal methods, edge cases, and code paths
that are not covered by the existing test_simultaneous_mcts.py at the
top-level tests/unit/ directory.

Focus areas:
- _run_simulation loop (selection, expansion, backpropagation at depth)
- _evaluate with scalar values, vector values, missing action_mask
- expand_simultaneous with zero-policy fallback, empty valid actions
- _step_env_with_resolution with empty resolver output (fallback)
- get_action_probabilities_for_player with fractional temperature
- Full search integration with terminal states at various depths
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import torch

from alpha_zero.multiplayer.simultaneous_mcts import (
    _INACTIVE_PLAYER_ACTION,
    ChildStats,
    DecoupledUCT,
    SimultaneousMCTSConfig,
    SimultaneousNode,
)

# =============================================================================
# Fixtures and helpers
# =============================================================================


def _make_env(
    num_players: int = 2,
    action_space_size: int = 5,
    terminated_after: int = -1,
    winner: int | None = None,
    player_scores: dict[int, float] | None = None,
) -> MagicMock:
    """Build a mock MultiplayerEnvironment.

    Args:
        num_players: Number of players in the game.
        action_space_size: Size of the action space.
        terminated_after: If >= 0, the env returns terminated=True after this many step() calls.
            If -1, the env never terminates.
        winner: Winner player id to report in terminal info.
        player_scores: Per-player scores to report in terminal info.
    """
    call_count = {"steps": 0}

    def _build() -> MagicMock:
        env = MagicMock()
        env.num_players = num_players
        env.current_player = 0

        action_space = MagicMock()
        action_space.n = action_space_size
        env.action_space = action_space

        spatial = np.zeros((4, 4, 4), dtype=np.float32)
        vector = np.zeros(8, dtype=np.float32)
        mask = np.ones(action_space_size, dtype=np.float32)

        env.get_observation.return_value = {
            "spatial": spatial,
            "vector": vector,
            "action_mask": mask,
        }
        env.get_valid_actions.return_value = mask

        def step_fn(action):
            call_count["steps"] += 1
            info = {}
            terminated = False
            if terminated_after >= 0 and call_count["steps"] >= terminated_after:
                terminated = True
                if player_scores is not None:
                    info["player_scores"] = player_scores
                elif winner is not None:
                    info["winner"] = winner
            return (
                {"spatial": spatial, "vector": vector, "action_mask": mask},
                0.0,
                terminated,
                False,
                info,
            )

        env.step.side_effect = step_fn
        env.clone.side_effect = lambda: _build()
        return env

    return _build()


def _make_network(
    num_players: int = 2,
    action_space_size: int = 5,
    scalar_value: bool = False,
    value_override: np.ndarray | None = None,
) -> MagicMock:
    """Build a mock MultiplayerNetwork.

    Args:
        num_players: Number of players.
        action_space_size: Size of action space.
        scalar_value: If True, return scalar value (legacy 2-player zero-sum).
        value_override: If provided, use this as the raw value output.
    """
    network = MagicMock()
    type(network).num_players = PropertyMock(return_value=num_players)

    def mock_predict(spatial, vector, action_mask):
        batch = spatial.shape[0]
        policy = torch.ones(batch, action_space_size) / action_space_size
        if value_override is not None:
            values = torch.from_numpy(value_override).float().unsqueeze(0)
        elif scalar_value:
            # Return shape (1,) so that squeeze(0) produces a scalar (ndim=0)
            values = torch.tensor([0.5])
        else:
            vals = [0.1 * (i + 1) for i in range(num_players)]
            values = torch.tensor([vals])
        return policy, values

    network.predict = mock_predict
    return network


@pytest.fixture
def default_config():
    return SimultaneousMCTSConfig(num_simulations=10, c_puct=1.5)


@pytest.fixture
def small_config():
    return SimultaneousMCTSConfig(num_simulations=3, c_puct=1.0)


# =============================================================================
# Test _INACTIVE_PLAYER_ACTION sentinel
# =============================================================================


class TestSentinel:
    """Verify the sentinel constant."""

    def test_inactive_player_action_is_negative(self):
        assert _INACTIVE_PLAYER_ACTION == -1
        assert _INACTIVE_PLAYER_ACTION < 0


# =============================================================================
# ChildStats additional tests
# =============================================================================


class TestChildStatsExtended:
    """Additional edge case tests for ChildStats."""

    def test_mutable_fields(self):
        """ChildStats fields are mutable (not frozen)."""
        stats = ChildStats()
        stats.visit_count = 42
        stats.value_sum = 3.14
        stats.prior = 0.99
        assert stats.visit_count == 42
        assert stats.value_sum == 3.14
        assert stats.prior == 0.99

    def test_negative_value_sum(self):
        """ChildStats can store negative values."""
        stats = ChildStats(value_sum=-10.0)
        assert stats.value_sum == -10.0


# =============================================================================
# SimultaneousNode: expand_simultaneous edge cases
# =============================================================================


class TestSimultaneousNodeExpansion:
    """Tests for expand_simultaneous edge cases."""

    def test_expand_with_zero_policy_sum(self):
        """When all valid action priors sum to ~0, fallback to uniform."""
        node = SimultaneousNode(num_players=2)
        # All priors are essentially zero
        policy = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        mask = np.array([1, 1, 1])

        node.expand_simultaneous(
            policies={0: policy},
            action_masks={0: mask},
        )

        # Should get uniform priors
        priors = [node.per_player_children[0][a].prior for a in [0, 1, 2]]
        expected = 1.0 / 3.0
        for p in priors:
            assert abs(p - expected) < 1e-6

    def test_expand_with_no_valid_actions(self):
        """When mask is all zeros, no children are created for that player."""
        node = SimultaneousNode(num_players=2)
        policy = np.array([0.5, 0.5])
        mask_all_invalid = np.array([0, 0])
        mask_valid = np.array([1, 1])

        node.expand_simultaneous(
            policies={0: policy, 1: policy.copy()},
            action_masks={0: mask_all_invalid, 1: mask_valid},
        )

        # Player 0 should have no children (warned, skipped)
        assert 0 not in node.per_player_children
        # Player 1 should be expanded normally
        assert 1 in node.per_player_children
        assert len(node.per_player_children[1]) == 2

    def test_expand_without_mask_defaults_to_all_valid(self):
        """When action_masks omits a player, defaults to ones_like(policy)."""
        node = SimultaneousNode(num_players=1)
        policy = np.array([0.3, 0.7])

        node.expand_simultaneous(
            policies={0: policy},
            action_masks={},  # No mask for player 0
        )

        assert 0 in node.per_player_children
        assert len(node.per_player_children[0]) == 2

    def test_expand_normalizes_policy_over_valid(self):
        """Policy is renormalized to only valid actions."""
        node = SimultaneousNode(num_players=1)
        policy = np.array([0.1, 0.2, 0.3, 0.4])
        mask = np.array([1, 0, 0, 1])  # Only actions 0, 3 valid

        node.expand_simultaneous(
            policies={0: policy},
            action_masks={0: mask},
        )

        p0 = node.per_player_children[0][0].prior
        p3 = node.per_player_children[0][3].prior
        assert abs(p0 + p3 - 1.0) < 1e-6
        # 0.1 / (0.1 + 0.4) = 0.2, 0.4 / 0.5 = 0.8
        assert abs(p0 - 0.2) < 1e-6
        assert abs(p3 - 0.8) < 1e-6

    def test_expand_multiple_players(self):
        """Expand with 4 players simultaneously."""
        node = SimultaneousNode(num_players=4)
        policy = np.array([0.25, 0.25, 0.25, 0.25])
        mask = np.ones(4)

        policies = {i: policy.copy() for i in range(4)}
        masks = {i: mask.copy() for i in range(4)}

        node.expand_simultaneous(policies, masks)

        for pid in range(4):
            assert pid in node.per_player_children
            assert len(node.per_player_children[pid]) == 4


# =============================================================================
# SimultaneousNode: select_action_for_player edge cases
# =============================================================================


class TestSimultaneousNodeSelection:
    """Tests for select_action_for_player edge cases."""

    def test_select_with_zero_visit_count(self):
        """Selection works when node has zero visits (sqrt_total=0)."""
        node = SimultaneousNode(num_players=1)
        node.visit_count = 0
        node.per_player_children[0] = {
            0: ChildStats(visit_count=0, value_sum=0.0, prior=0.8),
            1: ChildStats(visit_count=0, value_sum=0.0, prior=0.2),
        }

        # With zero node visits, exploration bonus is 0; Q is also 0
        # Both have score 0, but action 0 is iterated first so best_action stays 0
        # (dict iteration order in Python 3.7+)
        selected = node.select_action_for_player(0, c_puct=1.0)
        assert selected in [0, 1]

    def test_select_raises_on_empty_stats(self):
        """AssertionError when player has no stats."""
        node = SimultaneousNode(num_players=1)
        with pytest.raises(AssertionError, match="No action stats for player"):
            node.select_action_for_player(0, c_puct=1.0)

    def test_select_single_action(self):
        """When only one action is available, always select it."""
        node = SimultaneousNode(num_players=1)
        node.visit_count = 10
        node.per_player_children[0] = {
            7: ChildStats(visit_count=5, value_sum=2.5, prior=1.0),
        }
        selected = node.select_action_for_player(0, c_puct=1.0)
        assert selected == 7

    def test_select_prefers_high_prior_unvisited(self):
        """Unvisited action with high prior is preferred over visited ones."""
        node = SimultaneousNode(num_players=1)
        node.visit_count = 50
        node.per_player_children[0] = {
            0: ChildStats(visit_count=25, value_sum=10.0, prior=0.1),
            1: ChildStats(visit_count=25, value_sum=10.0, prior=0.1),
            2: ChildStats(visit_count=0, value_sum=0.0, prior=0.8),
        }
        selected = node.select_action_for_player(0, c_puct=2.0)
        assert selected == 2


# =============================================================================
# SimultaneousNode: update_stats edge cases
# =============================================================================


class TestSimultaneousNodeUpdateStats:
    """Tests for update_stats edge cases."""

    def test_update_with_missing_player_in_children(self):
        """update_stats handles player not in per_player_children gracefully."""
        node = SimultaneousNode(num_players=2)
        policy = np.array([0.5, 0.5])
        mask = np.array([1, 1])
        # Only expand player 0
        node.expand_simultaneous(
            policies={0: policy},
            action_masks={0: mask},
        )

        # Update with values for both players, but player 1 has no children
        node.update_stats({0: 0, 1: 1}, {0: 0.5, 1: -0.5})

        assert node.visit_count == 1
        assert node.value_sums[0] == 0.5
        assert node.value_sums[1] == -0.5
        assert node.per_player_children[0][0].visit_count == 1

    def test_update_with_action_not_in_stats(self):
        """update_stats handles action index not in per_player_children."""
        node = SimultaneousNode(num_players=1)
        policy = np.array([1.0, 0.0])
        mask = np.array([1, 0])  # Only action 0 valid
        node.expand_simultaneous(
            policies={0: policy},
            action_masks={0: mask},
        )

        # Try to update with invalid action 1
        node.update_stats({0: 1}, {0: 0.5})
        assert node.visit_count == 1
        assert node.value_sums[0] == 0.5
        # Action 0's stats should be unchanged
        assert node.per_player_children[0][0].visit_count == 0

    def test_update_with_empty_joint_action(self):
        """update_stats works with empty joint action."""
        node = SimultaneousNode(num_players=2)
        policy = np.array([0.5, 0.5])
        mask = np.array([1, 1])
        node.expand_simultaneous(
            policies={0: policy},
            action_masks={0: mask},
        )

        node.update_stats({}, {0: 0.3})
        assert node.visit_count == 1
        assert node.value_sums[0] == 0.3


# =============================================================================
# SimultaneousNode: get_action_probabilities_for_player edge cases
# =============================================================================


class TestSimultaneousNodeActionProbs:
    """Tests for get_action_probabilities_for_player edge cases."""

    def test_fractional_temperature(self):
        """Non-unity, non-zero temperature applies softmax-like transform."""
        node = SimultaneousNode(num_players=1)
        node.per_player_children[0] = {
            0: ChildStats(visit_count=10, value_sum=5.0, prior=0.5),
            1: ChildStats(visit_count=30, value_sum=15.0, prior=0.5),
            2: ChildStats(visit_count=60, value_sum=30.0, prior=0.0),
        }

        probs = node.get_action_probabilities_for_player(
            player_id=0, action_space_size=5, temperature=0.5
        )

        assert probs.shape == (5,)
        assert abs(probs.sum() - 1.0) < 1e-5
        # With low temperature, should concentrate more on high-visit action
        assert probs[2] > probs[1] > probs[0]

    def test_high_temperature_flattens_distribution(self):
        """High temperature makes distribution more uniform."""
        node = SimultaneousNode(num_players=1)
        node.per_player_children[0] = {
            0: ChildStats(visit_count=1, value_sum=0.5, prior=0.5),
            1: ChildStats(visit_count=100, value_sum=50.0, prior=0.5),
        }

        # Very high temperature
        probs = node.get_action_probabilities_for_player(
            player_id=0, action_space_size=3, temperature=10.0
        )

        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 1e-5
        # With high temperature, probabilities should be more similar
        ratio = probs[0] / (probs[1] + 1e-10)
        assert ratio > 0.1  # Much more uniform than visit ratio of 1:100

    def test_zero_visits_temperature_one(self):
        """All zero visits at temperature=1 gives uniform over valid actions."""
        node = SimultaneousNode(num_players=1)
        node.per_player_children[0] = {
            0: ChildStats(visit_count=0, value_sum=0.0, prior=0.5),
            1: ChildStats(visit_count=0, value_sum=0.0, prior=0.5),
        }

        probs = node.get_action_probabilities_for_player(
            player_id=0, action_space_size=4, temperature=1.0
        )

        assert probs.shape == (4,)
        assert abs(probs[0] - 0.5) < 1e-6
        assert abs(probs[1] - 0.5) < 1e-6
        assert probs[2] == 0.0
        assert probs[3] == 0.0

    def test_greedy_with_ties(self):
        """Greedy (temp=0) with tied visits picks one deterministically."""
        node = SimultaneousNode(num_players=1)
        node.per_player_children[0] = {
            0: ChildStats(visit_count=50, value_sum=25.0, prior=0.5),
            1: ChildStats(visit_count=50, value_sum=25.0, prior=0.5),
        }

        probs = node.get_action_probabilities_for_player(
            player_id=0, action_space_size=3, temperature=0.0
        )

        assert probs.sum() == 1.0
        # Exactly one action should have probability 1.0
        assert (probs == 1.0).sum() == 1

    def test_unknown_player(self):
        """Unknown player returns zeros."""
        node = SimultaneousNode(num_players=2)
        node.per_player_children[0] = {
            0: ChildStats(visit_count=10, value_sum=5.0, prior=1.0),
        }

        probs = node.get_action_probabilities_for_player(
            player_id=5, action_space_size=3
        )
        assert probs.shape == (3,)
        assert probs.sum() == 0.0


# =============================================================================
# SimultaneousMCTSConfig validation
# =============================================================================


class TestSimultaneousMCTSConfigExtended:
    """Extended config validation tests."""

    def test_valid_all_strategies(self):
        """All valid strategy strings are accepted."""
        for strategy in ["independent", "priority", "random", "simultaneous_apply"]:
            config = SimultaneousMCTSConfig(action_resolution=strategy)
            assert config.action_resolution == strategy

    def test_negative_dirichlet_epsilon(self):
        """Negative dirichlet_epsilon raises ValueError."""
        with pytest.raises(ValueError, match="dirichlet_epsilon must be in"):
            SimultaneousMCTSConfig(dirichlet_epsilon=-0.1)

    def test_custom_values(self):
        """Config stores all custom values."""
        config = SimultaneousMCTSConfig(
            num_simulations=500,
            c_puct=2.5,
            dirichlet_alpha=0.1,
            dirichlet_epsilon=0.5,
            temperature=0.5,
            temperature_threshold=20,
            action_resolution="priority",
        )
        assert config.num_simulations == 500
        assert config.c_puct == 2.5
        assert config.dirichlet_alpha == 0.1
        assert config.dirichlet_epsilon == 0.5
        assert config.temperature == 0.5
        assert config.temperature_threshold == 20
        assert config.action_resolution == "priority"

    def test_boundary_num_simulations(self):
        """num_simulations=1 is the minimum valid value."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        assert config.num_simulations == 1

    def test_boundary_dirichlet_epsilon_zero(self):
        """dirichlet_epsilon=0 (no noise) is valid."""
        config = SimultaneousMCTSConfig(dirichlet_epsilon=0.0)
        assert config.dirichlet_epsilon == 0.0

    def test_boundary_dirichlet_epsilon_one(self):
        """dirichlet_epsilon=1 (all noise) is valid."""
        config = SimultaneousMCTSConfig(dirichlet_epsilon=1.0)
        assert config.dirichlet_epsilon == 1.0


# =============================================================================
# DecoupledUCT: _evaluate edge cases
# =============================================================================


class TestDecoupledUCTEvaluate:
    """Tests for the _evaluate method."""

    def test_evaluate_scalar_value_2player(self):
        """Scalar value output is interpreted as zero-sum for 2 players."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_env(num_players=2, action_space_size=5)
        network = _make_network(num_players=2, action_space_size=5, scalar_value=True)

        policies, values = duct._evaluate(env, network)

        assert 0 in policies
        assert 1 in policies
        assert 0 in values
        assert 1 in values
        # Zero-sum: player 1 value = -player 0 value
        assert abs(values[0] + values[1]) < 1e-6

    def test_evaluate_vector_values(self):
        """Vector value output is distributed to players."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=3, action_space_size=5)

        env = _make_env(num_players=3, action_space_size=5)
        network = _make_network(num_players=3, action_space_size=5)

        policies, values = duct._evaluate(env, network)

        assert len(values) == 3
        for pid in range(3):
            assert pid in values
            assert pid in policies

    def test_evaluate_fewer_values_than_players(self):
        """When network returns fewer values than players, only available ones used."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=4, action_space_size=5)

        env = _make_env(num_players=4, action_space_size=5)
        # Network returns only 2 values for 4 players
        value_arr = np.array([0.3, -0.3])
        network = _make_network(
            num_players=4, action_space_size=5, value_override=value_arr
        )

        _policies, values = duct._evaluate(env, network)

        # Only 2 values should be in dict
        assert 0 in values
        assert 1 in values
        assert 2 not in values
        assert 3 not in values

    def test_evaluate_missing_action_mask_in_obs(self):
        """When observation has no action_mask, default to ones."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_env(num_players=2, action_space_size=5)
        # Remove action_mask from observation
        obs_without_mask = {
            "spatial": np.zeros((4, 4, 4), dtype=np.float32),
            "vector": np.zeros(8, dtype=np.float32),
        }
        env.get_observation.return_value = obs_without_mask
        network = _make_network(num_players=2, action_space_size=5)

        # Should not raise
        policies, values = duct._evaluate(env, network)
        assert 0 in policies
        assert 0 in values

    def test_evaluate_policies_are_independent_copies(self):
        """Each player gets an independent copy of the policy."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_env(num_players=2, action_space_size=5)
        network = _make_network(num_players=2, action_space_size=5)

        policies, _ = duct._evaluate(env, network)

        # Modify player 0's policy
        policies[0][0] = 999.0
        # Player 1's policy should be unaffected
        assert policies[1][0] != 999.0


# =============================================================================
# DecoupledUCT: _add_dirichlet_noise edge cases
# =============================================================================


class TestDecoupledUCTDirichletNoise:
    """Tests for the _add_dirichlet_noise method."""

    def test_no_noise_when_epsilon_zero(self):
        """With dirichlet_epsilon=0, priors are unchanged."""
        config = SimultaneousMCTSConfig(
            num_simulations=1, dirichlet_epsilon=0.0
        )
        duct = DecoupledUCT(config, num_players=1, action_space_size=3)

        node = SimultaneousNode(num_players=1)
        policy = np.array([0.2, 0.3, 0.5])
        mask = np.ones(3)
        node.expand_simultaneous(
            policies={0: policy}, action_masks={0: mask}
        )

        original_priors = [
            node.per_player_children[0][a].prior for a in [0, 1, 2]
        ]

        np.random.seed(42)
        duct._add_dirichlet_noise(node, active_players=[0])

        new_priors = [
            node.per_player_children[0][a].prior for a in [0, 1, 2]
        ]

        for orig, new in zip(original_priors, new_priors, strict=False):
            assert abs(orig - new) < 1e-10

    def test_full_noise_when_epsilon_one(self):
        """With dirichlet_epsilon=1, priors become pure Dirichlet samples."""
        config = SimultaneousMCTSConfig(
            num_simulations=1, dirichlet_epsilon=1.0, dirichlet_alpha=0.3
        )
        duct = DecoupledUCT(config, num_players=1, action_space_size=3)

        node = SimultaneousNode(num_players=1)
        policy = np.array([0.2, 0.3, 0.5])
        mask = np.ones(3)
        node.expand_simultaneous(
            policies={0: policy}, action_masks={0: mask}
        )

        np.random.seed(42)
        noise = np.random.dirichlet([0.3] * 3)

        # Reset node and re-expand to apply fresh noise
        node2 = SimultaneousNode(num_players=1)
        node2.expand_simultaneous(
            policies={0: policy.copy()}, action_masks={0: mask.copy()}
        )

        np.random.seed(42)
        duct._add_dirichlet_noise(node2, active_players=[0])

        for i, action in enumerate([0, 1, 2]):
            # eps=1 means prior = 0*(orig) + 1*(noise) = noise
            assert abs(node2.per_player_children[0][action].prior - noise[i]) < 1e-10

    def test_noise_skips_player_without_stats(self):
        """Dirichlet noise skips players with no action stats."""
        config = SimultaneousMCTSConfig(
            num_simulations=1, dirichlet_epsilon=0.5
        )
        duct = DecoupledUCT(config, num_players=2, action_space_size=3)

        node = SimultaneousNode(num_players=2)
        # Only expand player 0
        policy = np.array([0.5, 0.5, 0.0])
        mask = np.array([1, 1, 0])
        node.expand_simultaneous(
            policies={0: policy}, action_masks={0: mask}
        )

        # Should not raise even though player 1 has no stats
        duct._add_dirichlet_noise(node, active_players=[0, 1])


# =============================================================================
# DecoupledUCT: _extract_terminal_values
# =============================================================================


class TestDecoupledUCTTerminalValues:
    """Tests for _extract_terminal_values."""

    def test_player_scores_override_winner(self):
        """player_scores takes precedence over winner field."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        info = {
            "winner": 0,
            "player_scores": {0: 0.7, 1: 0.3},
        }
        values = duct._extract_terminal_values(info, active_players=[0, 1])
        assert values[0] == 0.7
        assert values[1] == 0.3

    def test_winner_gives_plus_minus_one(self):
        """Winner gets 1.0, losers get -1.0."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=3, action_space_size=5)

        values = duct._extract_terminal_values(
            {"winner": 2}, active_players=[0, 1, 2]
        )
        assert values[0] == -1.0
        assert values[1] == -1.0
        assert values[2] == 1.0

    def test_draw_gives_zeros(self):
        """No winner and no scores gives 0 for all players."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        values = duct._extract_terminal_values(
            {}, active_players=[0, 1]
        )
        assert values[0] == 0.0
        assert values[1] == 0.0

    def test_missing_player_in_scores(self):
        """Missing player in player_scores defaults to 0."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=3, action_space_size=5)

        info = {"player_scores": {0: 1.0}}  # Only player 0
        values = duct._extract_terminal_values(
            info, active_players=[0, 1, 2]
        )
        assert values[0] == 1.0
        assert values[1] == 0.0
        assert values[2] == 0.0


# =============================================================================
# DecoupledUCT: _backpropagate
# =============================================================================


class TestDecoupledUCTBackpropagate:
    """Tests for _backpropagate."""

    def test_backprop_multiple_nodes(self):
        """Backpropagation updates all nodes in the search path."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=3)

        root = SimultaneousNode(num_players=2)
        child = SimultaneousNode(num_players=2)

        policy = np.array([0.5, 0.3, 0.2])
        mask = np.ones(3)
        root.expand_simultaneous(
            policies={0: policy, 1: policy.copy()},
            action_masks={0: mask, 1: mask.copy()},
        )
        child.expand_simultaneous(
            policies={0: policy.copy(), 1: policy.copy()},
            action_masks={0: mask.copy(), 1: mask.copy()},
        )

        joint_actions = [{0: 1, 1: 2}]
        values = {0: 0.9, 1: -0.4}

        duct._backpropagate([root, child], joint_actions, values)

        # Root should be updated with joint_action
        assert root.visit_count == 1
        assert root.per_player_children[0][1].visit_count == 1
        assert root.per_player_children[1][2].visit_count == 1

        # Child should also be updated (but with empty joint_action since
        # joint_actions_path has length 1 but search_path has length 2)
        assert child.visit_count == 1
        assert child.value_sums[0] == 0.9

    def test_backprop_empty_path(self):
        """Empty search path doesn't crash."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=3)

        duct._backpropagate([], [], {0: 0.5, 1: -0.5})
        # No error means pass

    def test_backprop_single_root_no_actions(self):
        """Backprop with single root node and no actions."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=3)

        root = SimultaneousNode(num_players=2)
        policy = np.array([0.5, 0.5, 0.0])
        mask = np.array([1, 1, 0])
        root.expand_simultaneous(
            policies={0: policy}, action_masks={0: mask}
        )

        duct._backpropagate([root], [], {0: 1.0})
        assert root.visit_count == 1
        assert root.value_sums[0] == 1.0


# =============================================================================
# DecoupledUCT: _step_env_with_resolution edge cases
# =============================================================================


class TestDecoupledUCTStepEnv:
    """Tests for _step_env_with_resolution."""

    def test_terminal_detection(self):
        """Detects terminal state from step return."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_env(num_players=2, action_space_size=5)
        obs = env.get_observation()
        # Must clear side_effect before setting return_value
        env.step.side_effect = None
        env.step.return_value = (
            obs,
            0.0,
            True,  # terminated
            False,
            {"winner": 0},
        )

        is_terminal, info = duct._step_env_with_resolution(
            env, {0: 1, 1: 2}, active_players=[0, 1]
        )
        assert is_terminal
        assert info["winner"] == 0

    def test_truncated_counts_as_terminal(self):
        """Truncated state is also treated as terminal."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_env(num_players=2, action_space_size=5)
        obs = env.get_observation()
        env.step.side_effect = None
        env.step.return_value = (
            obs,
            0.0,
            False,
            True,  # truncated
            {},
        )

        is_terminal, _ = duct._step_env_with_resolution(
            env, {0: 1, 1: 2}, active_players=[0, 1]
        )
        assert is_terminal

    def test_fallback_when_resolver_returns_empty(self):
        """When resolver returns empty list, falls back to first active player."""
        config = SimultaneousMCTSConfig(
            num_simulations=1, action_resolution="simultaneous_apply"
        )
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_env(num_players=2, action_space_size=5)

        # Both players target the same action -> simultaneous_apply cancels both
        joint_action = {0: 3, 1: 3}
        _is_terminal, _ = duct._step_env_with_resolution(
            env, joint_action, active_players=[0, 1]
        )

        # env.step should still be called (with fallback action)
        env.step.assert_called_once()
        # The fallback action should be player 0's action (3)
        env.step.assert_called_with(3)


# =============================================================================
# DecoupledUCT: _run_simulation
# =============================================================================


class TestDecoupledUCTRunSimulation:
    """Tests for the _run_simulation method and the full simulation loop."""

    def test_single_simulation_expands_root(self):
        """A single simulation on an expanded root creates joint children."""
        config = SimultaneousMCTSConfig(num_simulations=1, c_puct=1.0)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_env(num_players=2, action_space_size=5)
        network = _make_network(num_players=2, action_space_size=5)

        # Expand root
        root = SimultaneousNode(num_players=2)
        duct._expand_node(root, env, network, active_players=[0, 1])

        # Run one simulation
        sim_env = env.clone()
        duct._run_simulation(sim_env, root, network, active_players=[0, 1])

        # Root should now have at least one joint child
        assert len(root.joint_children) >= 1
        assert root.visit_count >= 1

    def test_multiple_simulations_build_tree(self):
        """Multiple simulations build deeper tree structure."""
        np.random.seed(42)
        config = SimultaneousMCTSConfig(num_simulations=1, c_puct=1.5)
        duct = DecoupledUCT(config, num_players=2, action_space_size=3)

        env = _make_env(num_players=2, action_space_size=3)
        network = _make_network(num_players=2, action_space_size=3)

        root = SimultaneousNode(num_players=2)
        duct._expand_node(root, env, network, active_players=[0, 1])
        duct._add_dirichlet_noise(root, active_players=[0, 1])

        # Run multiple simulations
        for _ in range(10):
            sim_env = env.clone()
            duct._run_simulation(sim_env, root, network, active_players=[0, 1])

        assert root.visit_count >= 10
        assert len(root.joint_children) >= 1

    def test_simulation_with_terminal_on_first_step(self):
        """Simulation handles terminal state on first env step."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=3)

        env = _make_env(
            num_players=2,
            action_space_size=3,
            terminated_after=1,
            winner=0,
        )
        network = _make_network(num_players=2, action_space_size=3)

        root = SimultaneousNode(num_players=2)
        duct._expand_node(root, env, network, active_players=[0, 1])

        sim_env = env.clone()
        duct._run_simulation(sim_env, root, network, active_players=[0, 1])

        # Root should have been backpropagated
        assert root.visit_count >= 1

    def test_simulation_traverses_existing_child(self):
        """When joint child exists, simulation descends and continues."""
        np.random.seed(42)
        config = SimultaneousMCTSConfig(num_simulations=1, c_puct=1.0)
        duct = DecoupledUCT(config, num_players=2, action_space_size=2)

        env = _make_env(num_players=2, action_space_size=2)
        network = _make_network(num_players=2, action_space_size=2)

        # Expand root
        root = SimultaneousNode(num_players=2)
        duct._expand_node(root, env, network, active_players=[0, 1])

        # Run first simulation to create a joint child
        sim_env = env.clone()
        duct._run_simulation(sim_env, root, network, active_players=[0, 1])

        initial_children = len(root.joint_children)
        assert initial_children >= 1

        # Run another simulation; may visit existing child or create new one
        sim_env = env.clone()
        duct._run_simulation(sim_env, root, network, active_players=[0, 1])

        assert root.visit_count >= 2

    def test_simulation_terminal_at_existing_child(self):
        """Simulation hitting terminal at an existing child backtracks correctly."""
        config = SimultaneousMCTSConfig(num_simulations=1, c_puct=1.0)
        duct = DecoupledUCT(config, num_players=2, action_space_size=2)

        # First run with non-terminal env to build one level
        env_nonterminal = _make_env(num_players=2, action_space_size=2)
        network = _make_network(num_players=2, action_space_size=2)

        root = SimultaneousNode(num_players=2)
        duct._expand_node(root, env_nonterminal, network, active_players=[0, 1])

        # First sim to create a child
        sim_env = env_nonterminal.clone()
        duct._run_simulation(sim_env, root, network, active_players=[0, 1])

        first_visits = root.visit_count

        # Now use a terminal env for the second simulation
        env_terminal = _make_env(
            num_players=2,
            action_space_size=2,
            terminated_after=1,
            player_scores={0: 1.0, 1: -1.0},
        )

        sim_env = env_terminal.clone()
        duct._run_simulation(sim_env, root, network, active_players=[0, 1])

        assert root.visit_count == first_visits + 1


# =============================================================================
# DecoupledUCT: full search integration
# =============================================================================


class TestDecoupledUCTSearch:
    """Integration tests for the full search method."""

    def test_search_basic_2_player(self):
        """Basic 2-player search returns valid distributions."""
        np.random.seed(42)
        config = SimultaneousMCTSConfig(num_simulations=20, c_puct=1.5)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_env(num_players=2, action_space_size=5)
        network = _make_network(num_players=2, action_space_size=5)

        result = duct.search(env, network, active_players=[0, 1])

        assert len(result) == 2
        for pid in [0, 1]:
            assert result[pid].shape == (5,)
            assert abs(result[pid].sum() - 1.0) < 1e-5
            assert np.all(result[pid] >= 0)

    def test_search_single_player(self):
        """Search works with a single active player."""
        np.random.seed(42)
        config = SimultaneousMCTSConfig(num_simulations=10)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_env(num_players=2, action_space_size=5)
        network = _make_network(num_players=2, action_space_size=5)

        result = duct.search(env, network, active_players=[0])

        assert len(result) == 1
        assert 0 in result
        assert abs(result[0].sum() - 1.0) < 1e-5

    def test_search_many_players(self):
        """Search works with 8 active players."""
        np.random.seed(42)
        config = SimultaneousMCTSConfig(num_simulations=5, c_puct=1.0)
        duct = DecoupledUCT(config, num_players=8, action_space_size=4)

        env = _make_env(num_players=8, action_space_size=4)
        network = _make_network(num_players=8, action_space_size=4)

        result = duct.search(
            env, network, active_players=list(range(8))
        )

        assert len(result) == 8
        for pid in range(8):
            assert result[pid].shape == (4,)
            assert abs(result[pid].sum() - 1.0) < 1e-5

    def test_search_with_terminal_env(self):
        """Search works when env terminates immediately in simulations."""
        np.random.seed(42)
        config = SimultaneousMCTSConfig(num_simulations=5)
        duct = DecoupledUCT(config, num_players=2, action_space_size=3)

        env = _make_env(
            num_players=2,
            action_space_size=3,
            terminated_after=1,
            winner=1,
        )
        network = _make_network(num_players=2, action_space_size=3)

        result = duct.search(env, network, active_players=[0, 1])

        assert len(result) == 2
        for pid in [0, 1]:
            assert result[pid].shape == (3,)

    def test_search_greedy_after_threshold(self):
        """After temperature threshold, probabilities are greedy."""
        np.random.seed(42)
        config = SimultaneousMCTSConfig(
            num_simulations=15,
            temperature=1.0,
            temperature_threshold=10,
        )
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_env(num_players=2, action_space_size=5)
        network = _make_network(num_players=2, action_space_size=5)

        result = duct.search(
            env, network, active_players=[0, 1], turn_number=15
        )

        for pid in [0, 1]:
            assert result[pid].max() == 1.0
            assert np.count_nonzero(result[pid]) == 1

    def test_search_count_increments(self):
        """_search_count increments with each call."""
        config = SimultaneousMCTSConfig(num_simulations=3)
        duct = DecoupledUCT(config, num_players=2, action_space_size=3)

        env = _make_env(num_players=2, action_space_size=3)
        network = _make_network(num_players=2, action_space_size=3)

        assert duct._search_count == 0
        duct.search(env, network, active_players=[0, 1])
        assert duct._search_count == 1
        duct.search(env, network, active_players=[0, 1])
        assert duct._search_count == 2

    def test_search_with_scalar_value_network(self):
        """Full search works with a scalar-value network (2-player zero-sum)."""
        np.random.seed(42)
        config = SimultaneousMCTSConfig(num_simulations=10)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        env = _make_env(num_players=2, action_space_size=5)
        network = _make_network(
            num_players=2, action_space_size=5, scalar_value=True
        )

        result = duct.search(env, network, active_players=[0, 1])

        assert len(result) == 2
        for pid in [0, 1]:
            assert result[pid].shape == (5,)
            assert abs(result[pid].sum() - 1.0) < 1e-5

    def test_search_with_partial_action_mask(self):
        """Search works when only some actions are valid."""
        np.random.seed(42)
        config = SimultaneousMCTSConfig(num_simulations=10)
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)

        partial_mask = np.array([1, 0, 1, 0, 0], dtype=np.float32)

        def _build_masked_env():
            env = MagicMock()
            env.num_players = 2
            env.current_player = 0
            action_space = MagicMock()
            action_space.n = 5
            env.action_space = action_space

            spatial = np.zeros((4, 4, 4), dtype=np.float32)
            vector = np.zeros(8, dtype=np.float32)

            env.get_observation.return_value = {
                "spatial": spatial,
                "vector": vector,
                "action_mask": partial_mask.copy(),
            }
            env.get_valid_actions.return_value = partial_mask.copy()
            env.step.return_value = (
                {"spatial": spatial, "vector": vector, "action_mask": partial_mask.copy()},
                0.0,
                False,
                False,
                {},
            )
            env.clone.side_effect = lambda: _build_masked_env()
            return env

        env = _build_masked_env()
        network = _make_network(num_players=2, action_space_size=5)

        result = duct.search(env, network, active_players=[0, 1])

        for pid in [0, 1]:
            assert result[pid].shape == (5,)
            # Only actions 0 and 2 should have non-zero probability
            assert result[pid][1] == 0.0
            assert result[pid][3] == 0.0
            assert result[pid][4] == 0.0

    def test_search_with_player_scores_terminal(self):
        """Search with player_scores terminal info extracts correct values."""
        np.random.seed(42)
        config = SimultaneousMCTSConfig(num_simulations=5)
        duct = DecoupledUCT(config, num_players=2, action_space_size=3)

        env = _make_env(
            num_players=2,
            action_space_size=3,
            terminated_after=1,
            player_scores={0: 0.8, 1: 0.2},
        )
        network = _make_network(num_players=2, action_space_size=3)

        # Should not raise
        result = duct.search(env, network, active_players=[0, 1])
        assert len(result) == 2

    def test_search_with_different_resolution_strategies(self):
        """Search works with all action resolution strategies."""
        for strategy in ["independent", "priority", "random", "simultaneous_apply"]:
            np.random.seed(42)
            config = SimultaneousMCTSConfig(
                num_simulations=5, action_resolution=strategy
            )
            duct = DecoupledUCT(config, num_players=2, action_space_size=3)

            env = _make_env(num_players=2, action_space_size=3)
            network = _make_network(num_players=2, action_space_size=3)

            result = duct.search(env, network, active_players=[0, 1])

            assert len(result) == 2, f"Failed for strategy={strategy}"
            for pid in [0, 1]:
                assert result[pid].shape == (3,), f"Failed for strategy={strategy}"

    def test_search_at_temperature_threshold_boundary(self):
        """turn_number == temperature_threshold uses greedy."""
        np.random.seed(42)
        config = SimultaneousMCTSConfig(
            num_simulations=10,
            temperature=1.0,
            temperature_threshold=5,
        )
        duct = DecoupledUCT(config, num_players=2, action_space_size=3)

        env = _make_env(num_players=2, action_space_size=3)
        network = _make_network(num_players=2, action_space_size=3)

        # Exactly at threshold
        result = duct.search(
            env, network, active_players=[0, 1], turn_number=5
        )

        for pid in [0, 1]:
            assert result[pid].max() == 1.0


# =============================================================================
# DecoupledUCT: constructor and action resolver
# =============================================================================


class TestDecoupledUCTInit:
    """Tests for DecoupledUCT initialization."""

    def test_init_stores_attributes(self):
        config = SimultaneousMCTSConfig(num_simulations=50)
        duct = DecoupledUCT(config, num_players=3, action_space_size=20)

        assert duct.config is config
        assert duct.num_players == 3
        assert duct.action_space_size == 20
        assert duct._search_count == 0

    def test_action_resolver_created_with_config(self):
        """Action resolver strategy matches config."""
        config = SimultaneousMCTSConfig(action_resolution="random")
        duct = DecoupledUCT(config, num_players=2, action_space_size=5)
        assert duct._action_resolver._strategy.value == "random"

    def test_init_with_all_strategies(self):
        """DecoupledUCT can be initialized with all strategies."""
        for strategy in ["independent", "priority", "random", "simultaneous_apply"]:
            config = SimultaneousMCTSConfig(action_resolution=strategy)
            duct = DecoupledUCT(config, num_players=2, action_space_size=5)
            assert duct._action_resolver._strategy.value == strategy


# =============================================================================
# SimultaneousNode: is_expanded
# =============================================================================


class TestSimultaneousNodeIsExpanded:
    """Tests for the is_expanded method."""

    def test_not_expanded_initially(self):
        node = SimultaneousNode()
        assert not node.is_expanded()

    def test_expanded_after_expand_simultaneous(self):
        node = SimultaneousNode(num_players=1)
        policy = np.array([1.0])
        mask = np.array([1])
        node.expand_simultaneous(policies={0: policy}, action_masks={0: mask})
        assert node.is_expanded()

    def test_not_expanded_after_manual_joint_children(self):
        """Adding joint_children doesn't make node expanded."""
        node = SimultaneousNode(num_players=2)
        child = SimultaneousNode(num_players=2)
        node.joint_children[(0, 1)] = child
        assert not node.is_expanded()  # per_player_children is still empty


# =============================================================================
# _run_simulation: unexpanded non-root node path
# =============================================================================


class TestDecoupledUCTUnexpandedNode:
    """Test the final fallback path in _run_simulation for unexpanded nodes."""

    def test_unexpanded_node_evaluated_and_backpropagated(self):
        """If root is somehow unexpanded, _run_simulation evaluates and backtracks."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=3)

        env = _make_env(num_players=2, action_space_size=3)
        network = _make_network(num_players=2, action_space_size=3)

        # Create root but do NOT expand it
        root = SimultaneousNode(num_players=2)
        assert not root.is_expanded()

        sim_env = env.clone()
        duct._run_simulation(sim_env, root, network, active_players=[0, 1])

        # Node should have been backpropagated (visit_count incremented)
        assert root.visit_count == 1

    def test_simulation_on_deep_unexpanded(self):
        """
        Simulation reaching an unexpanded non-root node triggers the fallback
        evaluation path (the while loop exits because node.is_expanded() is False).
        """
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=2)

        env = _make_env(num_players=2, action_space_size=2)
        network = _make_network(num_players=2, action_space_size=2)

        # Create unexpanded root
        root = SimultaneousNode(num_players=2)

        sim_env = env.clone()
        duct._run_simulation(sim_env, root, network, active_players=[0, 1])

        # Should have been evaluated and backpropagated
        assert root.visit_count >= 1
        assert len(root.value_sums) > 0


# =============================================================================
# Integration: deterministic seed reproducibility
# =============================================================================


class TestDeterminism:
    """Verify deterministic behavior with fixed seeds."""

    def test_search_is_deterministic_with_seed(self):
        """Two searches with the same seed produce the same result."""
        results = []
        for _ in range(2):
            np.random.seed(123)
            torch.manual_seed(123)

            config = SimultaneousMCTSConfig(num_simulations=15, c_puct=1.5)
            duct = DecoupledUCT(config, num_players=2, action_space_size=5)

            env = _make_env(num_players=2, action_space_size=5)
            network = _make_network(num_players=2, action_space_size=5)

            result = duct.search(env, network, active_players=[0, 1])
            results.append(result)

        np.testing.assert_array_almost_equal(results[0][0], results[1][0])
        np.testing.assert_array_almost_equal(results[0][1], results[1][1])


# =============================================================================
# Integration: _expand_node specifics
# =============================================================================


class TestExpandNodeIntegration:
    """Integration tests for _expand_node."""

    def test_expand_node_returns_values(self):
        """_expand_node returns per-player value dict."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=3, action_space_size=4)

        env = _make_env(num_players=3, action_space_size=4)
        network = _make_network(num_players=3, action_space_size=4)

        node = SimultaneousNode(num_players=3)
        values = duct._expand_node(node, env, network, active_players=[0, 1, 2])

        assert isinstance(values, dict)
        assert 0 in values
        assert 1 in values
        assert 2 in values
        assert node.is_expanded()

    def test_expand_node_calls_get_valid_actions_per_player(self):
        """_expand_node calls get_valid_actions for each active player."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=2, action_space_size=3)

        env = _make_env(num_players=2, action_space_size=3)
        network = _make_network(num_players=2, action_space_size=3)

        node = SimultaneousNode(num_players=2)
        duct._expand_node(node, env, network, active_players=[0, 1])

        # get_valid_actions should be called for each active player
        assert env.get_valid_actions.call_count == 2

    def test_expand_node_subset_active_players(self):
        """_expand_node only expands active players."""
        config = SimultaneousMCTSConfig(num_simulations=1)
        duct = DecoupledUCT(config, num_players=4, action_space_size=3)

        env = _make_env(num_players=4, action_space_size=3)
        network = _make_network(num_players=4, action_space_size=3)

        node = SimultaneousNode(num_players=4)
        duct._expand_node(node, env, network, active_players=[1, 3])

        # Only players 1 and 3 should be expanded
        assert 1 in node.per_player_children
        assert 3 in node.per_player_children
        assert 0 not in node.per_player_children
        assert 2 not in node.per_player_children
