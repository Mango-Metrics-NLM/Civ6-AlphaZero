"""Tests for hierarchical MCTS components.

Tests cover:
- OptionTermination enum values
- Option dataclass and lifecycle
- MCTSConfig defaults and level-specific factories
- SearchStats policy calculation
- MCTSNode UCB scores, selection, expansion, backpropagation
- HierarchicalMCTS search basics
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from alpha_zero.hierarchical.config import HierarchyLevel
from alpha_zero.hierarchical.goals.goal_types import Goal, OperationalGoalType
from alpha_zero.hierarchical.mcts.hierarchical_mcts import (
    HierarchicalMCTS,
    MCTSConfig,
    MCTSNode,
    Option,
    OptionTermination,
    SearchStats,
)

# =============================================================================
# Constants
# =============================================================================

SEED = 42


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Create seeded random number generator."""
    return np.random.default_rng(SEED)


@pytest.fixture
def sample_state() -> dict[str, Any]:
    """Create sample game state."""
    return {
        "spatial": np.random.rand(13, 16, 16).astype(np.float32),
        "vector": np.random.rand(64).astype(np.float32),
        "action_mask": np.ones(100, dtype=np.float32),
        "turn": 50,
    }


@pytest.fixture
def sample_goal() -> Goal:
    """Create sample operational goal."""
    return Goal(
        level=HierarchyLevel.OPERATIONAL,
        goal_type=OperationalGoalType.MOVE_UNIT,
        progress=0.0,
    )


@pytest.fixture
def mcts_config() -> MCTSConfig:
    """Create minimal MCTS configuration."""
    return MCTSConfig(
        num_simulations=10,
        c_puct=1.5,
        temperature=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        discount=0.99,
        horizon=5,
    )


# =============================================================================
# OptionTermination Tests
# =============================================================================


class TestOptionTermination:
    """Tests for OptionTermination enum."""

    def test_termination_values(self) -> None:
        """Test all termination reasons exist."""
        assert OptionTermination.GOAL_ACHIEVED is not None
        assert OptionTermination.TIMEOUT is not None
        assert OptionTermination.FAILURE is not None
        assert OptionTermination.INTERRUPTED is not None
        assert OptionTermination.STILL_ACTIVE is not None

    def test_termination_distinct(self) -> None:
        """Test termination values are distinct."""
        values = [e.value for e in OptionTermination]
        assert len(values) == len(set(values))


# =============================================================================
# Option Tests
# =============================================================================


class TestOption:
    """Tests for Option dataclass."""

    def test_creation(self, sample_goal: Goal) -> None:
        """Test option creation."""
        option = Option(
            id="test_option",
            goal=sample_goal,
            level=HierarchyLevel.OPERATIONAL,
        )

        assert option.id == "test_option"
        assert option.goal is sample_goal
        assert option.level == HierarchyLevel.OPERATIONAL
        assert option.duration == 10
        assert option.reward_modifier == 1.0
        assert option.steps_active == 0
        assert option.cumulative_reward == 0.0
        assert option.status == OptionTermination.STILL_ACTIVE

    def test_can_initiate_default(self, sample_goal: Goal, sample_state: dict[str, Any]) -> None:
        """Test can_initiate with no initiation function."""
        option = Option(id="test", goal=sample_goal, level=HierarchyLevel.OPERATIONAL)

        # Default should allow initiation
        assert option.can_initiate(sample_state) is True

    def test_can_initiate_with_function(
        self, sample_goal: Goal, sample_state: dict[str, Any]
    ) -> None:
        """Test can_initiate with custom function."""
        # Only initiate if turn > 30
        option = Option(
            id="test",
            goal=sample_goal,
            level=HierarchyLevel.OPERATIONAL,
            initiation=lambda s: s.get("turn", 0) > 30,
        )

        sample_state["turn"] = 50
        assert option.can_initiate(sample_state) is True

        sample_state["turn"] = 10
        assert option.can_initiate(sample_state) is False

    def test_should_terminate_timeout(
        self, sample_goal: Goal, sample_state: dict[str, Any]
    ) -> None:
        """Test option terminates on timeout."""
        option = Option(
            id="test",
            goal=sample_goal,
            level=HierarchyLevel.OPERATIONAL,
            duration=5,
        )

        # Not at duration yet
        option.steps_active = 4
        assert option.should_terminate(sample_state) is False

        # At duration
        option.steps_active = 5
        assert option.should_terminate(sample_state) is True
        assert option.status == OptionTermination.TIMEOUT

    def test_should_terminate_goal_achieved(
        self, sample_goal: Goal, sample_state: dict[str, Any]
    ) -> None:
        """Test option terminates when goal achieved."""
        option = Option(
            id="test",
            goal=sample_goal,
            level=HierarchyLevel.OPERATIONAL,
            duration=100,
        )

        # Goal not complete
        sample_goal.progress = 0.5
        assert option.should_terminate(sample_state) is False

        # Goal complete
        sample_goal.progress = 1.0
        assert option.should_terminate(sample_state) is True
        assert option.status == OptionTermination.GOAL_ACHIEVED

    def test_get_action_default_random(self, sample_goal: Goal) -> None:
        """Test get_action with no policy defaults to random."""
        option = Option(id="test", goal=sample_goal, level=HierarchyLevel.OPERATIONAL)

        valid_actions = np.zeros(10)
        valid_actions[[2, 5, 7]] = 1

        np.random.seed(SEED)
        action = option.get_action({}, valid_actions)

        assert action in [2, 5, 7]

    def test_get_action_with_policy(self, sample_goal: Goal) -> None:
        """Test get_action with custom policy."""

        # Policy that always picks first valid action deterministically
        def policy(state: dict) -> np.ndarray:
            probs = np.zeros(10)
            probs[3] = 1.0  # Always pick action 3
            return probs

        option = Option(
            id="test",
            goal=sample_goal,
            level=HierarchyLevel.OPERATIONAL,
            policy=policy,
        )

        valid_actions = np.ones(10)

        action = option.get_action({}, valid_actions)
        assert action == 3

    def test_step_updates_state(self, sample_goal: Goal) -> None:
        """Test step updates option runtime state."""
        option = Option(
            id="test",
            goal=sample_goal,
            level=HierarchyLevel.OPERATIONAL,
            reward_modifier=2.0,
        )

        option.step(reward=1.5)

        assert option.steps_active == 1
        assert option.cumulative_reward == 3.0  # 1.5 * 2.0

        option.step(reward=0.5)

        assert option.steps_active == 2
        assert option.cumulative_reward == 4.0


# =============================================================================
# MCTSConfig Tests
# =============================================================================


class TestMCTSConfig:
    """Tests for MCTSConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MCTSConfig()

        assert config.num_simulations == 100
        assert config.c_puct == 1.5
        assert config.c_visit == 1.0
        assert config.temperature == 1.0
        assert config.dirichlet_alpha == 0.3
        assert config.dirichlet_epsilon == 0.25
        assert config.discount == 0.99
        assert config.horizon == 50
        assert config.use_options is True
        assert config.option_penalty == 0.1
        assert config.virtual_loss == 3.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = MCTSConfig(
            num_simulations=200,
            c_puct=2.0,
            temperature=0.5,
            use_options=False,
        )

        assert config.num_simulations == 200
        assert config.c_puct == 2.0
        assert config.temperature == 0.5
        assert config.use_options is False

    def test_for_level_strategic(self) -> None:
        """Test strategic level configuration."""
        config = MCTSConfig.for_level(HierarchyLevel.STRATEGIC)

        assert config.num_simulations == 50
        assert config.c_puct == 2.0
        assert config.horizon == 100
        assert config.temperature == 1.2
        assert config.use_options is True
        assert config.discount == 0.995

    def test_for_level_tactical(self) -> None:
        """Test tactical level configuration."""
        config = MCTSConfig.for_level(HierarchyLevel.TACTICAL)

        assert config.num_simulations == 100
        assert config.c_puct == 1.5
        assert config.horizon == 25
        assert config.temperature == 1.0
        assert config.use_options is True

    def test_for_level_operational(self) -> None:
        """Test operational level configuration."""
        config = MCTSConfig.for_level(HierarchyLevel.OPERATIONAL)

        assert config.num_simulations == 200
        assert config.c_puct == 1.5
        assert config.horizon == 5
        assert config.use_options is False


# =============================================================================
# SearchStats Tests
# =============================================================================


class TestSearchStats:
    """Tests for SearchStats dataclass."""

    def test_default_creation(self) -> None:
        """Test default stats creation."""
        stats = SearchStats()

        assert stats.root_value == 0.0
        assert len(stats.visit_counts) == 0
        assert len(stats.prior_probs) == 0
        assert len(stats.q_values) == 0
        assert stats.depth_reached == 0
        assert stats.simulations == 0
        assert stats.options_considered == 0

    def test_get_policy_with_temperature(self) -> None:
        """Test policy calculation with temperature."""
        stats = SearchStats(
            visit_counts=np.array([100, 50, 25, 25]),
        )

        policy = stats.get_policy(temperature=1.0)

        assert len(policy) == 4
        assert np.isclose(policy.sum(), 1.0)
        assert policy[0] > policy[1] > policy[2]  # Higher visits = higher prob

    def test_get_policy_zero_temperature(self) -> None:
        """Test deterministic policy with zero temperature."""
        stats = SearchStats(
            visit_counts=np.array([10, 100, 30, 50]),
        )

        policy = stats.get_policy(temperature=0)

        assert np.isclose(policy.sum(), 1.0)
        assert policy[1] == 1.0  # Most visited action
        assert policy[0] == 0.0
        assert policy[2] == 0.0
        assert policy[3] == 0.0

    def test_get_policy_high_temperature(self) -> None:
        """Test flatter policy with high temperature."""
        stats = SearchStats(
            visit_counts=np.array([100, 50, 25, 25]),
        )

        policy_low_temp = stats.get_policy(temperature=0.5)
        policy_high_temp = stats.get_policy(temperature=2.0)

        # High temperature should be more uniform
        assert np.std(policy_high_temp) < np.std(policy_low_temp)


# =============================================================================
# MCTSNode Tests
# =============================================================================


class TestMCTSNode:
    """Tests for MCTSNode class."""

    def test_initialization(self, sample_state: dict[str, Any]) -> None:
        """Test node initialization."""
        node = MCTSNode(state=sample_state)

        assert node.state is sample_state
        assert node.parent is None
        assert node.action is None
        assert node.prior == 1.0
        assert len(node.children) == 0
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.virtual_loss == 0.0

    def test_initialization_with_parent(self, sample_state: dict[str, Any]) -> None:
        """Test node initialization with parent."""
        parent = MCTSNode(state=sample_state)
        child = MCTSNode(state=sample_state, parent=parent, action=5, prior=0.3)

        assert child.parent is parent
        assert child.action == 5
        assert child.prior == 0.3

    def test_is_expanded(self, sample_state: dict[str, Any]) -> None:
        """Test is_expanded property."""
        node = MCTSNode(state=sample_state)

        assert node.is_expanded is False

        # Add a child
        node.children[0] = MCTSNode(state=sample_state, parent=node, action=0)

        assert node.is_expanded is True

    def test_q_value(self, sample_state: dict[str, Any]) -> None:
        """Test Q-value calculation."""
        node = MCTSNode(state=sample_state)

        # No visits yet
        assert node.q_value == 0.0

        # After visits
        node.visit_count = 4
        node.value_sum = 2.0

        assert node.q_value == 0.5

    def test_ucb_score(self, sample_state: dict[str, Any]) -> None:
        """Test UCB score calculation."""
        node = MCTSNode(state=sample_state, prior=0.5)
        node.visit_count = 10
        node.value_sum = 5.0  # Q = 0.5

        parent_visits = 100
        c_puct = 1.5

        # UCB = Q + c_puct * prior * sqrt(parent_visits) / (1 + visits)
        # UCB = 0.5 + 1.5 * 0.5 * 10 / 11 = 0.5 + 0.68 = 1.18 (approx)
        score = node.ucb_score(c_puct, parent_visits)

        expected = 0.5 + 1.5 * 0.5 * np.sqrt(100) / 11
        assert np.isclose(score, expected, rtol=0.01)

    def test_ucb_score_with_virtual_loss(self, sample_state: dict[str, Any]) -> None:
        """Test UCB score includes virtual loss."""
        node = MCTSNode(state=sample_state, prior=0.5)
        node.visit_count = 10
        node.value_sum = 5.0
        node.virtual_loss = 1.0

        score_with_vl = node.ucb_score(1.5, 100)

        # Reset virtual loss and compare
        node.virtual_loss = 0.0
        score_without_vl = node.ucb_score(1.5, 100)

        assert score_with_vl < score_without_vl

    def test_select_child(self, sample_state: dict[str, Any]) -> None:
        """Test child selection by UCB score."""
        parent = MCTSNode(state=sample_state)
        parent.visit_count = 100

        # Create children with different values
        for action in range(5):
            child = MCTSNode(state=sample_state, parent=parent, action=action, prior=0.2)
            child.visit_count = 10
            child.value_sum = action * 2  # Higher action = higher value
            parent.children[action] = child

        best_action, best_child = parent.select_child(c_puct=1.5)

        # Action 4 has highest Q-value
        assert best_action == 4
        assert best_child.action == 4

    def test_expand(self, sample_state: dict[str, Any]) -> None:
        """Test node expansion."""
        node = MCTSNode(state=sample_state)

        priors = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        valid_actions = np.array([1, 1, 1, 0, 1])  # Action 3 invalid

        def transition(state: dict, action: int) -> dict:
            return {"action_taken": action}

        node.expand(priors, valid_actions, transition)

        assert len(node.children) == 4  # Only valid actions
        assert 3 not in node.children
        assert node.children[0].prior == 0.1
        assert node.children[2].prior == 0.3

    def test_backpropagate(self, sample_state: dict[str, Any]) -> None:
        """Test value backpropagation."""
        # Create chain: root -> child -> grandchild
        root = MCTSNode(state=sample_state)
        child = MCTSNode(state=sample_state, parent=root, action=0)
        grandchild = MCTSNode(state=sample_state, parent=child, action=1)

        root.children[0] = child
        child.children[1] = grandchild

        # Backpropagate from grandchild
        grandchild.backpropagate(value=1.0, discount=0.9)

        assert grandchild.visit_count == 1
        assert grandchild.value_sum == 1.0

        assert child.visit_count == 1
        assert child.value_sum == 0.9  # Discounted

        assert root.visit_count == 1
        assert np.isclose(root.value_sum, 0.81)  # 0.9 * 0.9


# =============================================================================
# HierarchicalMCTS Tests
# =============================================================================


class TestHierarchicalMCTS:
    """Tests for HierarchicalMCTS class."""

    def test_initialization(self, mcts_config: MCTSConfig) -> None:
        """Test MCTS initialization."""
        mcts = HierarchicalMCTS(
            network=None,
            state_encoder=None,
            goal_encoder=None,
            config=mcts_config,
            level=HierarchyLevel.OPERATIONAL,
        )

        assert mcts.network is None
        assert mcts.config is mcts_config
        assert mcts.level == HierarchyLevel.OPERATIONAL
        assert len(mcts.options) == 0

    def test_initialization_default_config(self) -> None:
        """Test MCTS uses level-appropriate default config."""
        mcts = HierarchicalMCTS(
            network=None,
            state_encoder=None,
            goal_encoder=None,
            level=HierarchyLevel.STRATEGIC,
        )

        # Should use strategic-level defaults
        assert mcts.config.use_options is True
        assert mcts.config.num_simulations == 50

    def test_register_option(self, mcts_config: MCTSConfig, sample_goal: Goal) -> None:
        """Test option registration."""
        mcts = HierarchicalMCTS(
            network=None,
            state_encoder=None,
            goal_encoder=None,
            config=mcts_config,
        )

        option = Option(id="test", goal=sample_goal, level=HierarchyLevel.OPERATIONAL)
        mcts.register_option(option)

        assert len(mcts.options) == 1
        assert mcts.options[0] is option

    def test_add_dirichlet_noise(self, mcts_config: MCTSConfig) -> None:
        """Test Dirichlet noise addition."""
        mcts = HierarchicalMCTS(
            network=None,
            state_encoder=None,
            goal_encoder=None,
            config=mcts_config,
        )

        priors = np.array([0.4, 0.3, 0.2, 0.1])
        valid_actions = np.array([1, 1, 1, 1])

        np.random.seed(SEED)
        noisy = mcts._add_dirichlet_noise(priors, valid_actions)

        # Should still sum to ~1
        assert np.isclose(noisy.sum(), 1.0)
        # Should be different from original
        assert not np.allclose(noisy, priors)

    def test_add_dirichlet_noise_respects_mask(self, mcts_config: MCTSConfig) -> None:
        """Test Dirichlet noise respects action mask."""
        mcts = HierarchicalMCTS(
            network=None,
            state_encoder=None,
            goal_encoder=None,
            config=mcts_config,
        )

        priors = np.array([0.4, 0.3, 0.2, 0.1])
        valid_actions = np.array([1, 1, 0, 0])  # Only first two valid

        np.random.seed(SEED)
        noisy = mcts._add_dirichlet_noise(priors, valid_actions)

        # Invalid actions should be zero
        assert noisy[2] == 0.0
        assert noisy[3] == 0.0
        # Valid actions should have probability
        assert noisy[0] > 0.0
        assert noisy[1] > 0.0

    def test_add_dirichlet_noise_disabled(self) -> None:
        """Test no noise when epsilon is zero."""
        config = MCTSConfig(dirichlet_epsilon=0.0)
        mcts = HierarchicalMCTS(
            network=None,
            state_encoder=None,
            goal_encoder=None,
            config=config,
        )

        priors = np.array([0.4, 0.3, 0.2, 0.1])
        valid_actions = np.array([1, 1, 1, 1])

        noisy = mcts._add_dirichlet_noise(priors, valid_actions)

        np.testing.assert_array_equal(noisy, priors)

    def test_search_without_network(
        self,
        mcts_config: MCTSConfig,
        sample_state: dict[str, Any],
        sample_goal: Goal,
    ) -> None:
        """Test search works with fallback when network is None."""
        mcts = HierarchicalMCTS(
            network=None,
            state_encoder=None,
            goal_encoder=None,
            config=mcts_config,
        )

        # Set up valid actions
        sample_state["action_mask"] = np.ones(356)

        # This should not raise
        action, stats = mcts.search(
            state=sample_state,
            goals=[sample_goal],
            valid_actions=sample_state["action_mask"],
        )

        assert isinstance(action, int)
        assert action >= 0
        assert isinstance(stats, SearchStats)

    def test_default_state_transition(self, mcts_config: MCTSConfig) -> None:
        """Test default state transition returns copy."""
        mcts = HierarchicalMCTS(
            network=None,
            state_encoder=None,
            goal_encoder=None,
            config=mcts_config,
        )

        state = {"value": 42}
        next_state = mcts._default_state_transition(state, action=5)

        assert next_state is not state
        assert next_state["value"] == 42


# =============================================================================
# Integration Tests
# =============================================================================


class TestMCTSIntegration:
    """Integration tests for MCTS components."""

    def test_full_search_cycle(self) -> None:
        """Test complete search cycle with mocked components."""
        config = MCTSConfig(
            num_simulations=5,
            horizon=3,
            dirichlet_epsilon=0.0,  # No noise for determinism
        )

        mcts = HierarchicalMCTS(
            network=None,
            state_encoder=None,
            goal_encoder=None,
            config=config,
            level=HierarchyLevel.OPERATIONAL,
        )

        state = {
            "spatial": np.zeros((13, 16, 16)),
            "vector": np.zeros(64),
            "action_mask": np.ones(10),
        }

        goal = Goal(
            level=HierarchyLevel.OPERATIONAL,
            goal_type=OperationalGoalType.END_TURN,
        )

        # Custom state transition
        def transition(s: dict, a: int) -> dict:
            return {**s, "last_action": a}

        action, stats = mcts.search(
            state=state,
            goals=[goal],
            valid_actions=state["action_mask"],
            state_transition=transition,
        )

        assert 0 <= action < 10
        assert stats.simulations == 5

    def test_option_based_search(self) -> None:
        """Test search with options enabled."""
        config = MCTSConfig(
            num_simulations=5,
            use_options=True,
        )

        mcts = HierarchicalMCTS(
            network=None,
            state_encoder=None,
            goal_encoder=None,
            config=config,
            level=HierarchyLevel.TACTICAL,
        )

        goal = Goal(
            level=HierarchyLevel.TACTICAL,
            goal_type=OperationalGoalType.MOVE_UNIT,
        )

        # Register options
        for i in range(3):
            option = Option(
                id=f"option_{i}",
                goal=goal,
                level=HierarchyLevel.TACTICAL,
                duration=5,
            )
            mcts.register_option(option)

        state = {"turn": 50}

        def state_predictor(s: dict, opt: Option) -> dict:
            return {**s, "option_applied": opt.id}

        result, stats = mcts.search_with_options(
            state=state,
            goals=[goal],
            state_predictor=state_predictor,
        )

        # Should return either an option or fall back to action search
        assert result is not None or isinstance(result, int)
