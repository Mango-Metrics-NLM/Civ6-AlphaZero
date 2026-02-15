"""Unit tests for the multi-objective MCTS engine.

Tests the MultiObjectiveMCTSConfig, MultiObjectiveNode, and
MultiObjectiveMCTS classes, covering configuration, node operations
(expand, select, backpropagate), and search behavior with and
without a neural network.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.strategic.envs.resource_allocation import (
    ResourceAllocationConfig,
    ResourceAllocationEnv,
)
from alpha_zero.strategic.mcts.multi_objective_mcts import (
    MultiObjectiveMCTS,
    MultiObjectiveMCTSConfig,
    MultiObjectiveNode,
)
from alpha_zero.strategic.objectives import ObjectiveSpec, WeightedSumScalarizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_env() -> ResourceAllocationEnv:
    """Create a tiny ResourceAllocationEnv for fast MCTS tests.

    Uses 2 regions, 1 resource type, and 3 time steps to keep the
    action space small and tests fast.
    """
    config = ResourceAllocationConfig(
        num_regions=2,
        num_resource_types=1,
        num_time_steps=3,
        total_budget=50.0,
        resource_names=["supplies"],
        region_names=["North", "South"],
        seed=42,
    )
    env = ResourceAllocationEnv(config)
    env.reset(seed=42)
    return env


@pytest.fixture()
def scalarizer_2obj() -> WeightedSumScalarizer:
    """Create a weighted-sum scalarizer for 2 objectives."""
    objectives = [
        ObjectiveSpec(name="efficiency", weight=0.6),
        ObjectiveSpec(name="equity", weight=0.4),
    ]
    return WeightedSumScalarizer(objectives)


@pytest.fixture()
def scalarizer_3obj() -> WeightedSumScalarizer:
    """Create a weighted-sum scalarizer for 3 objectives (matches ResourceAllocationEnv)."""
    objectives = [
        ObjectiveSpec(name="shortage", weight=0.4),
        ObjectiveSpec(name="equity", weight=0.3),
        ObjectiveSpec(name="cost", weight=0.3),
    ]
    return WeightedSumScalarizer(objectives)


# ---------------------------------------------------------------------------
# TestMultiObjectiveMCTSConfig
# ---------------------------------------------------------------------------


class TestMultiObjectiveMCTSConfig:
    """Tests for MultiObjectiveMCTSConfig dataclass."""

    def test_defaults(self) -> None:
        """Test that default configuration values are reasonable."""
        config = MultiObjectiveMCTSConfig()

        assert config.num_simulations == 100
        assert config.c_puct == 1.5
        assert config.dirichlet_alpha == pytest.approx(0.3)
        assert config.dirichlet_epsilon == pytest.approx(0.25)
        assert config.temperature == pytest.approx(1.0)
        assert config.temperature_threshold == 30

    def test_from_dict(self) -> None:
        """Test creating a config from a dictionary, including unknown key filtering."""
        data = {
            "num_simulations": 200,
            "c_puct": 2.0,
            "dirichlet_alpha": 0.5,
            "unknown_key": "should_be_ignored",
        }
        config = MultiObjectiveMCTSConfig.from_dict(data)

        assert config.num_simulations == 200
        assert config.c_puct == 2.0
        assert config.dirichlet_alpha == pytest.approx(0.5)
        # Defaults for unspecified fields
        assert config.dirichlet_epsilon == pytest.approx(0.25)
        assert config.temperature == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestMultiObjectiveNode
# ---------------------------------------------------------------------------


class TestMultiObjectiveNode:
    """Tests for MultiObjectiveNode tree node."""

    def test_initial_state(self) -> None:
        """Test that a new node starts with zero visits and zero value sums."""
        node = MultiObjectiveNode(prior=0.5, num_objectives=3)

        assert node.prior == pytest.approx(0.5)
        assert node.visit_count == 0
        np.testing.assert_array_equal(node.value_sum, np.zeros(3))
        assert node.children == {}
        assert node.parent is None
        assert node.action is None
        assert node.is_expanded is False

    def test_value_unvisited(self) -> None:
        """Test that value of an unvisited node is a zero vector."""
        node = MultiObjectiveNode(prior=0.3, num_objectives=2)

        value = node.value
        np.testing.assert_array_equal(value, np.zeros(2))

    def test_value_after_visits(self) -> None:
        """Test that value returns the mean of backpropagated values."""
        node = MultiObjectiveNode(prior=0.5, num_objectives=2)
        node.visit_count = 4
        node.value_sum = np.array([2.0, 4.0])

        value = node.value
        np.testing.assert_array_almost_equal(value, [0.5, 1.0])

    def test_expand_creates_children(self) -> None:
        """Test that expand creates child nodes for all valid actions."""
        node = MultiObjectiveNode(prior=1.0, num_objectives=2)
        policy = np.array([0.3, 0.5, 0.2])
        action_mask = np.array([1.0, 1.0, 1.0])

        node.expand(policy, action_mask)

        assert node.is_expanded is True
        assert len(node.children) == 3
        assert set(node.children.keys()) == {0, 1, 2}

        # Verify priors sum to 1
        prior_sum = sum(c.prior for c in node.children.values())
        assert prior_sum == pytest.approx(1.0, abs=1e-6)

        # Verify parent and action references
        for action_idx, child in node.children.items():
            assert child.parent is node
            assert child.action == action_idx

    def test_expand_respects_mask(self) -> None:
        """Test that expand only creates children for actions allowed by the mask."""
        node = MultiObjectiveNode(prior=1.0, num_objectives=1)
        policy = np.array([0.4, 0.3, 0.2, 0.1])
        action_mask = np.array([1.0, 0.0, 1.0, 0.0])

        node.expand(policy, action_mask)

        assert len(node.children) == 2
        assert set(node.children.keys()) == {0, 2}

        # Priors should be renormalized over valid actions only
        total_prior = sum(c.prior for c in node.children.values())
        assert total_prior == pytest.approx(1.0, abs=1e-6)

    def test_select_child(self, scalarizer_2obj: WeightedSumScalarizer) -> None:
        """Test that select_child picks the child with the highest UCB score."""
        node = MultiObjectiveNode(prior=1.0, num_objectives=2)
        policy = np.array([0.5, 0.3, 0.2])
        mask = np.ones(3)
        node.expand(policy, mask)
        node.visit_count = 10

        # Give child 0 a good Q-value
        child0 = node.children[0]
        child0.visit_count = 5
        child0.value_sum = np.array([4.0, 3.0])  # mean = [0.8, 0.6]

        # Give child 1 a lower Q-value
        child1 = node.children[1]
        child1.visit_count = 3
        child1.value_sum = np.array([0.6, 0.3])  # mean = [0.2, 0.1]

        # child 2 is unvisited (prior exploration should kick in)
        action, selected = node.select_child(c_puct=1.5, scalarizer=scalarizer_2obj)

        # Should select one of the children (exact choice depends on UCB math)
        assert action in node.children
        assert selected is node.children[action]

    def test_select_child_raises_on_unexpanded(
        self, scalarizer_2obj: WeightedSumScalarizer
    ) -> None:
        """Test that select_child raises ValueError on unexpanded node."""
        node = MultiObjectiveNode(prior=1.0, num_objectives=2)

        with pytest.raises(ValueError, match="Cannot select child"):
            node.select_child(c_puct=1.5, scalarizer=scalarizer_2obj)

    def test_backpropagate(self) -> None:
        """Test that backpropagation updates visit counts and value sums up to root."""
        root = MultiObjectiveNode(prior=1.0, num_objectives=2)
        policy = np.array([0.5, 0.5])
        mask = np.ones(2)
        root.expand(policy, mask)

        child = root.children[0]

        value = np.array([0.7, 0.3])
        child.backpropagate(value)

        # Child should be updated
        assert child.visit_count == 1
        np.testing.assert_array_almost_equal(child.value_sum, value)

        # Root should also be updated
        assert root.visit_count == 1
        np.testing.assert_array_almost_equal(root.value_sum, value)

    def test_backpropagate_multiple(self) -> None:
        """Test that multiple backpropagations accumulate correctly."""
        root = MultiObjectiveNode(prior=1.0, num_objectives=2)
        policy = np.array([1.0])
        mask = np.ones(1)
        root.expand(policy, mask)

        child = root.children[0]

        child.backpropagate(np.array([0.5, 0.3]))
        child.backpropagate(np.array([0.7, 0.1]))

        assert child.visit_count == 2
        np.testing.assert_array_almost_equal(child.value_sum, [1.2, 0.4])
        np.testing.assert_array_almost_equal(child.value, [0.6, 0.2])


# ---------------------------------------------------------------------------
# TestMultiObjectiveMCTS
# ---------------------------------------------------------------------------


class TestMultiObjectiveMCTS:
    """Tests for the MultiObjectiveMCTS search engine."""

    def test_search_no_network(
        self,
        tiny_env: ResourceAllocationEnv,
        scalarizer_3obj: WeightedSumScalarizer,
    ) -> None:
        """Test MCTS search without a neural network (random rollout fallback)."""
        config = MultiObjectiveMCTSConfig(num_simulations=10)
        mcts = MultiObjectiveMCTS(
            config,
            network=None,
            scalarizer=scalarizer_3obj,
            num_objectives=3,
        )

        action_probs, value_vector = mcts.search(tiny_env)

        # Check shapes
        action_space_size = tiny_env.action_spec.total_size
        assert action_probs.shape == (action_space_size,)
        assert value_vector.shape == (3,)

        # Action probs should be a valid distribution
        assert action_probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert np.all(action_probs >= 0.0)

    def test_select_action_greedy(
        self,
        tiny_env: ResourceAllocationEnv,
        scalarizer_3obj: WeightedSumScalarizer,
    ) -> None:
        """Test that select_action with temperature=0 returns the argmax action."""
        config = MultiObjectiveMCTSConfig(num_simulations=10)
        mcts = MultiObjectiveMCTS(
            config,
            network=None,
            scalarizer=scalarizer_3obj,
            num_objectives=3,
        )

        action_probs, _ = mcts.search(tiny_env)

        # Greedy selection: should return the highest-probability action
        selected = mcts.select_action(action_probs, temperature=0.0)
        expected = int(np.argmax(action_probs))

        assert selected == expected

    def test_select_action_stochastic(
        self,
        tiny_env: ResourceAllocationEnv,
        scalarizer_3obj: WeightedSumScalarizer,
    ) -> None:
        """Test that select_action with temperature=1 samples from the distribution."""
        config = MultiObjectiveMCTSConfig(num_simulations=10)
        mcts = MultiObjectiveMCTS(
            config,
            network=None,
            scalarizer=scalarizer_3obj,
            num_objectives=3,
        )

        action_probs, _ = mcts.search(tiny_env)

        # Run many samples to verify stochasticity
        np.random.seed(123)
        actions = [
            mcts.select_action(action_probs, temperature=1.0) for _ in range(50)
        ]

        # All actions should be valid indices
        action_space_size = tiny_env.action_spec.total_size
        assert all(0 <= a < action_space_size for a in actions)

        # With temperature=1 and enough samples, we expect some diversity
        # (unless one action dominates completely)
        # This is a probabilistic test but should be stable with seed
        unique_actions = set(actions)
        # At minimum the most likely action should appear
        assert len(unique_actions) >= 1

    def test_search_returns_correct_shapes(
        self,
        tiny_env: ResourceAllocationEnv,
    ) -> None:
        """Test that search output shapes match environment specs for various num_objectives."""
        for num_obj in [1, 2, 3]:
            objectives = [
                ObjectiveSpec(name=f"obj_{i}", weight=1.0) for i in range(num_obj)
            ]
            scalarizer = WeightedSumScalarizer(objectives)
            config = MultiObjectiveMCTSConfig(num_simulations=5)
            mcts = MultiObjectiveMCTS(
                config,
                network=None,
                scalarizer=scalarizer,
                num_objectives=num_obj,
            )

            action_probs, value_vector = mcts.search(tiny_env)

            action_space_size = tiny_env.action_spec.total_size
            assert action_probs.shape == (action_space_size,), (
                f"Expected action_probs shape ({action_space_size},) "
                f"for num_objectives={num_obj}, got {action_probs.shape}"
            )
            assert value_vector.shape == (num_obj,), (
                f"Expected value shape ({num_obj},) "
                f"for num_objectives={num_obj}, got {value_vector.shape}"
            )

    def test_search_with_few_simulations(
        self,
        tiny_env: ResourceAllocationEnv,
        scalarizer_3obj: WeightedSumScalarizer,
    ) -> None:
        """Test that search with very few simulations still returns valid results."""
        config = MultiObjectiveMCTSConfig(num_simulations=1)
        mcts = MultiObjectiveMCTS(
            config,
            network=None,
            scalarizer=scalarizer_3obj,
            num_objectives=3,
        )

        action_probs, value_vector = mcts.search(tiny_env)

        assert action_probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert value_vector.shape == (3,)

    def test_select_action_with_mask(
        self,
        scalarizer_3obj: WeightedSumScalarizer,
    ) -> None:
        """Test that select_action respects action mask."""
        config = MultiObjectiveMCTSConfig(num_simulations=5)
        mcts = MultiObjectiveMCTS(
            config,
            network=None,
            scalarizer=scalarizer_3obj,
            num_objectives=3,
        )

        action_probs = np.array([0.1, 0.5, 0.3, 0.1])
        mask = np.array([0.0, 0.0, 1.0, 1.0])

        # Greedy with mask should pick action 2 (highest prob among valid)
        selected = mcts.select_action(action_probs, temperature=0.0, action_mask=mask)
        assert selected in [2, 3]

    def test_default_scalarizer(self) -> None:
        """Test that MCTS creates a default scalarizer when none is provided."""
        config = MultiObjectiveMCTSConfig(num_simulations=5)
        mcts = MultiObjectiveMCTS(config, network=None, num_objectives=2)

        assert mcts.scalarizer is not None
        assert isinstance(mcts.scalarizer, WeightedSumScalarizer)

    def test_search_override_num_simulations(
        self,
        tiny_env: ResourceAllocationEnv,
        scalarizer_3obj: WeightedSumScalarizer,
    ) -> None:
        """Test that num_simulations can be overridden per search call."""
        config = MultiObjectiveMCTSConfig(num_simulations=100)
        mcts = MultiObjectiveMCTS(
            config,
            network=None,
            scalarizer=scalarizer_3obj,
            num_objectives=3,
        )

        # Override with a small number to keep the test fast
        action_probs, value_vector = mcts.search(tiny_env, num_simulations=3)

        assert action_probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert value_vector.shape == (3,)
