"""Comprehensive tests for the SDK MCTS engine.

Covers:
- MCTSNode: construction, value properties, expand, backpropagate, virtual loss
- PUCTPolicy: c_puct, select_child with various conditions
- get_action_probabilities: temperature scaling, edge cases
- select_action: greedy, masked, stochastic selection
- MCTSEngine: search integration with mock environment
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.config import MCTSConfig
from src.core.mcts.engine import (
    MCTSEngine,
    get_action_probabilities,
    select_action,
)
from src.core.mcts.node import MCTSNode
from src.core.mcts.policies import PUCTPolicy
from src.core.types import DEFAULT_C_PUCT, ObservationSpec

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


class MockEnv:
    """Minimal environment that satisfies the StrategicEnvironment protocol."""

    def __init__(
        self,
        action_space_size: int = 4,
        terminal_after: int = 3,
    ) -> None:
        self.action_space_size = action_space_size
        self.observation_spec = ObservationSpec(spatial_shape=(1, 2, 2), vector_dim=2)
        self._terminal = False
        self._step_count = 0
        self._terminal_after = terminal_after

    def get_observation(self) -> dict[str, np.ndarray]:
        return {
            "spatial": np.zeros((1, 2, 2), dtype=np.float32),
            "vector": np.zeros(2, dtype=np.float32),
        }

    def get_valid_actions(self) -> np.ndarray:
        return np.ones(self.action_space_size, dtype=np.bool_)

    def is_terminal(self) -> bool:
        return self._terminal

    def step(self, action: int) -> tuple:
        self._step_count += 1
        if self._step_count >= self._terminal_after:
            self._terminal = True
        obs = self.get_observation()
        return obs, 0.0, self._terminal, False, {}

    def clone(self) -> MockEnv:
        c = MockEnv(
            action_space_size=self.action_space_size,
            terminal_after=self._terminal_after,
        )
        c._step_count = self._step_count
        c._terminal = self._terminal
        return c


def mock_eval_fn(
    spatial: np.ndarray,
    vector: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Uniform policy, zero value."""
    n = len(mask)
    policy = np.ones(n, dtype=np.float32) / n
    return policy, 0.0


def biased_eval_fn(
    spatial: np.ndarray,
    vector: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Policy heavily biased toward action 0, positive value."""
    n = len(mask)
    policy = np.zeros(n, dtype=np.float32)
    policy[0] = 0.9
    if n > 1:
        remaining = 0.1 / (n - 1)
        policy[1:] = remaining
    return policy, 0.5


@pytest.fixture
def root_node() -> MCTSNode:
    """A fresh root node."""
    return MCTSNode()


@pytest.fixture
def expanded_root() -> MCTSNode:
    """Root expanded with uniform policy over 4 actions."""
    root = MCTSNode()
    policy = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
    mask = np.ones(4, dtype=np.bool_)
    root.expand(policy, mask)
    return root


@pytest.fixture
def small_mcts_config() -> MCTSConfig:
    """Minimal MCTS config for fast tests."""
    return MCTSConfig(num_simulations=8, c_puct=1.5)


@pytest.fixture
def mock_env() -> MockEnv:
    """Standard mock environment with 4 actions."""
    return MockEnv(action_space_size=4, terminal_after=3)


# ============================================================================
# MCTSNode Tests
# ============================================================================


class TestMCTSNodeConstruction:
    """Tests for MCTSNode default construction and basic properties."""

    def test_default_construction_zeros(self, root_node: MCTSNode) -> None:
        """Default node has all zero numeric fields."""
        assert root_node.prior == 0.0
        assert root_node.visit_count == 0
        assert root_node.value_sum == 0.0
        assert root_node.virtual_loss == 0

    def test_default_construction_no_children(self, root_node: MCTSNode) -> None:
        """Default node has empty children dict."""
        assert root_node.children == {}
        assert len(root_node.children) == 0

    def test_default_construction_is_root(self, root_node: MCTSNode) -> None:
        """Default node is a root (no parent)."""
        assert root_node.is_root is True
        assert root_node.parent is None

    def test_default_construction_not_expanded(self, root_node: MCTSNode) -> None:
        """Default node is not expanded."""
        assert root_node.is_expanded is False

    def test_default_construction_action_none(self, root_node: MCTSNode) -> None:
        """Default node has no action."""
        assert root_node.action is None

    def test_custom_construction(self) -> None:
        """Node with custom prior and action."""
        node = MCTSNode(prior=0.7, action=3)
        assert node.prior == 0.7
        assert node.action == 3
        assert node.visit_count == 0


class TestMCTSNodeValue:
    """Tests for the value and effective_value properties."""

    def test_value_zero_when_unvisited(self, root_node: MCTSNode) -> None:
        """Value is 0 when visit_count is 0."""
        assert root_node.value == 0.0

    def test_value_correct_ratio(self) -> None:
        """Value is value_sum / visit_count."""
        node = MCTSNode(visit_count=4, value_sum=2.0)
        assert node.value == pytest.approx(0.5)

    def test_value_negative(self) -> None:
        """Value can be negative."""
        node = MCTSNode(visit_count=5, value_sum=-3.0)
        assert node.value == pytest.approx(-0.6)

    def test_effective_value_zero_when_unvisited(self, root_node: MCTSNode) -> None:
        """Effective value is 0 when both visit_count and virtual_loss are 0."""
        assert root_node.effective_value == 0.0

    def test_effective_value_adjusted_for_virtual_loss(self) -> None:
        """Effective value accounts for virtual loss."""
        node = MCTSNode(visit_count=10, value_sum=5.0, virtual_loss=2)
        # total = 10 + 2 = 12; adjusted = 5.0 - 2 = 3.0; effective = 3.0 / 12
        assert node.effective_value == pytest.approx(3.0 / 12.0)

    def test_effective_value_with_only_virtual_loss(self) -> None:
        """Effective value when only virtual loss is present (no real visits)."""
        node = MCTSNode(visit_count=0, value_sum=0.0, virtual_loss=3)
        # total = 0 + 3 = 3; adjusted = 0.0 - 3 = -3.0; effective = -3.0 / 3
        assert node.effective_value == pytest.approx(-1.0)

    def test_effective_value_equals_value_when_no_virtual_loss(self) -> None:
        """Effective value equals regular value when virtual_loss is 0."""
        node = MCTSNode(visit_count=10, value_sum=5.0, virtual_loss=0)
        assert node.effective_value == pytest.approx(node.value)


class TestMCTSNodeExpand:
    """Tests for expanding a node to create children."""

    def test_is_expanded_true_after_expand(self, expanded_root: MCTSNode) -> None:
        """Node reports is_expanded after expand is called."""
        assert expanded_root.is_expanded is True

    def test_expand_uniform_policy_child_count(self) -> None:
        """Expanding with all-valid mask creates one child per action."""
        node = MCTSNode()
        policy = np.ones(6, dtype=np.float32)
        mask = np.ones(6, dtype=np.bool_)
        node.expand(policy, mask)
        assert len(node.children) == 6

    def test_expand_uniform_policy_priors_normalized(self) -> None:
        """Child priors sum to 1 after normalization."""
        node = MCTSNode()
        policy = np.ones(4, dtype=np.float32)
        mask = np.ones(4, dtype=np.bool_)
        node.expand(policy, mask)
        priors = [child.prior for child in node.children.values()]
        assert sum(priors) == pytest.approx(1.0)
        for p in priors:
            assert p == pytest.approx(0.25)

    def test_expand_masked_policy_only_valid_children(self) -> None:
        """Only valid actions (True in mask) create children."""
        node = MCTSNode()
        policy = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        mask = np.array([True, False, True, False], dtype=np.bool_)
        node.expand(policy, mask)
        assert set(node.children.keys()) == {0, 2}

    def test_expand_masked_policy_priors_renormalized(self) -> None:
        """Priors are renormalized over valid actions only."""
        node = MCTSNode()
        policy = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        mask = np.array([True, False, True, False], dtype=np.bool_)
        node.expand(policy, mask)
        # valid policies: 0.1, 0.3 -> normalized: 0.25, 0.75
        assert node.children[0].prior == pytest.approx(0.25)
        assert node.children[2].prior == pytest.approx(0.75)

    def test_expand_zero_policy_fallback_uniform(self) -> None:
        """All-zero policy falls back to uniform distribution."""
        node = MCTSNode()
        policy = np.zeros(3, dtype=np.float32)
        mask = np.ones(3, dtype=np.bool_)
        node.expand(policy, mask)
        for child in node.children.values():
            assert child.prior == pytest.approx(1.0 / 3.0)

    def test_expand_no_valid_actions_no_children(self) -> None:
        """No children created when mask is all False."""
        node = MCTSNode()
        policy = np.array([0.5, 0.5], dtype=np.float32)
        mask = np.array([False, False], dtype=np.bool_)
        node.expand(policy, mask)
        assert len(node.children) == 0
        assert node.is_expanded is False

    def test_expand_parent_child_relationships(self, expanded_root: MCTSNode) -> None:
        """Children have correct parent and action references."""
        for action, child in expanded_root.children.items():
            assert child.parent is expanded_root
            assert child.action == action

    def test_expand_children_are_fresh_nodes(self, expanded_root: MCTSNode) -> None:
        """Children start with zero visits and no children of their own."""
        for child in expanded_root.children.values():
            assert child.visit_count == 0
            assert child.value_sum == 0.0
            assert child.is_expanded is False
            assert child.is_root is False

    def test_expand_single_valid_action(self) -> None:
        """Expanding with a single valid action creates one child with prior 1.0."""
        node = MCTSNode()
        policy = np.array([0.0, 0.0, 0.8, 0.0], dtype=np.float32)
        mask = np.array([False, False, True, False], dtype=np.bool_)
        node.expand(policy, mask)
        assert len(node.children) == 1
        assert 2 in node.children
        assert node.children[2].prior == pytest.approx(1.0)

    def test_expand_preserves_action_indices(self) -> None:
        """Child dict keys correspond to actual action indices, not offsets."""
        node = MCTSNode()
        policy = np.zeros(10, dtype=np.float32)
        policy[3] = 0.5
        policy[7] = 0.5
        mask = np.zeros(10, dtype=np.bool_)
        mask[3] = True
        mask[7] = True
        node.expand(policy, mask)
        assert set(node.children.keys()) == {3, 7}


class TestMCTSNodeBackpropagate:
    """Tests for backpropagation through the tree."""

    def test_backpropagate_single_node(self) -> None:
        """Backpropagating on a single root node."""
        root = MCTSNode()
        root.backpropagate(0.8)
        assert root.visit_count == 1
        assert root.value_sum == pytest.approx(0.8)

    def test_backpropagate_chain_sign_flipping(self) -> None:
        """Value flips sign at each level (negamax)."""
        root = MCTSNode()
        child = MCTSNode(parent=root, action=0)
        root.children[0] = child
        grandchild = MCTSNode(parent=child, action=1)
        child.children[1] = grandchild

        grandchild.backpropagate(1.0)

        # grandchild: +1.0, child: -1.0, root: +1.0
        assert grandchild.value_sum == pytest.approx(1.0)
        assert child.value_sum == pytest.approx(-1.0)
        assert root.value_sum == pytest.approx(1.0)

    def test_backpropagate_updates_all_visit_counts(self) -> None:
        """All nodes in the path get visit_count incremented."""
        root = MCTSNode()
        child = MCTSNode(parent=root, action=0)
        root.children[0] = child
        grandchild = MCTSNode(parent=child, action=1)
        child.children[1] = grandchild

        grandchild.backpropagate(0.5)

        assert grandchild.visit_count == 1
        assert child.visit_count == 1
        assert root.visit_count == 1

    def test_backpropagate_multiple_times(self) -> None:
        """Multiple backpropagations accumulate correctly."""
        root = MCTSNode()
        child = MCTSNode(parent=root, action=0)
        root.children[0] = child

        child.backpropagate(1.0)
        child.backpropagate(0.5)

        # child: 1.0 + 0.5 = 1.5, visits = 2
        assert child.visit_count == 2
        assert child.value_sum == pytest.approx(1.5)
        # root: -1.0 + -0.5 = -1.5, visits = 2
        assert root.visit_count == 2
        assert root.value_sum == pytest.approx(-1.5)

    def test_backpropagate_negative_value(self) -> None:
        """Backpropagating a negative value."""
        root = MCTSNode()
        child = MCTSNode(parent=root, action=0)
        root.children[0] = child

        child.backpropagate(-0.3)

        assert child.value_sum == pytest.approx(-0.3)
        assert root.value_sum == pytest.approx(0.3)

    def test_backpropagate_does_not_affect_siblings(self) -> None:
        """Backpropagation only affects nodes in the path."""
        root = MCTSNode()
        child_a = MCTSNode(parent=root, action=0)
        child_b = MCTSNode(parent=root, action=1)
        root.children[0] = child_a
        root.children[1] = child_b

        child_a.backpropagate(1.0)

        assert child_b.visit_count == 0
        assert child_b.value_sum == 0.0


class TestMCTSNodeVirtualLoss:
    """Tests for virtual loss add/remove."""

    def test_add_virtual_loss_single_node(self) -> None:
        """Adding virtual loss to a root node."""
        root = MCTSNode()
        root.add_virtual_loss()
        assert root.virtual_loss == 1

    def test_add_virtual_loss_chain(self) -> None:
        """Virtual loss propagates from leaf to root."""
        root = MCTSNode()
        child = MCTSNode(parent=root, action=0)
        root.children[0] = child
        grandchild = MCTSNode(parent=child, action=1)
        child.children[1] = grandchild

        grandchild.add_virtual_loss()

        assert grandchild.virtual_loss == 1
        assert child.virtual_loss == 1
        assert root.virtual_loss == 1

    def test_add_virtual_loss_multiple_times(self) -> None:
        """Multiple add_virtual_loss calls stack."""
        root = MCTSNode()
        root.add_virtual_loss()
        root.add_virtual_loss()
        assert root.virtual_loss == 2

    def test_remove_virtual_loss_decrements(self) -> None:
        """Removing virtual loss decrements the counter."""
        root = MCTSNode()
        child = MCTSNode(parent=root, action=0)
        root.children[0] = child

        child.add_virtual_loss()
        child.add_virtual_loss()
        child.remove_virtual_loss()

        assert child.virtual_loss == 1
        assert root.virtual_loss == 1

    def test_remove_virtual_loss_min_zero(self) -> None:
        """Virtual loss never goes below 0."""
        node = MCTSNode()
        node.remove_virtual_loss()
        assert node.virtual_loss == 0

    def test_remove_virtual_loss_chain_min_zero(self) -> None:
        """Remove virtual loss through chain, all clamped at 0."""
        root = MCTSNode()
        child = MCTSNode(parent=root, action=0)
        root.children[0] = child

        # Remove without adding
        child.remove_virtual_loss()
        assert child.virtual_loss == 0
        assert root.virtual_loss == 0


class TestMCTSNodeChildVisitCounts:
    """Tests for the child_visit_counts method."""

    def test_child_visit_counts_shape(self, expanded_root: MCTSNode) -> None:
        """Output has correct shape matching action_space_size."""
        counts = expanded_root.child_visit_counts(action_space_size=4)
        assert counts.shape == (4,)
        assert counts.dtype == np.float32

    def test_child_visit_counts_all_zeros_initially(
        self, expanded_root: MCTSNode
    ) -> None:
        """All counts are zero before any visits."""
        counts = expanded_root.child_visit_counts(action_space_size=4)
        np.testing.assert_array_equal(counts, np.zeros(4, dtype=np.float32))

    def test_child_visit_counts_after_visits(self, expanded_root: MCTSNode) -> None:
        """Counts reflect visit_count of each child."""
        expanded_root.children[0].visit_count = 5
        expanded_root.children[2].visit_count = 3
        counts = expanded_root.child_visit_counts(action_space_size=4)
        assert counts[0] == 5.0
        assert counts[1] == 0.0
        assert counts[2] == 3.0
        assert counts[3] == 0.0

    def test_child_visit_counts_larger_action_space(self) -> None:
        """Works when action_space_size is larger than number of children."""
        node = MCTSNode()
        policy = np.array([0.5, 0.5], dtype=np.float32)
        mask = np.array([True, True], dtype=np.bool_)
        node.expand(policy, mask)
        node.children[0].visit_count = 10
        counts = node.child_visit_counts(action_space_size=10)
        assert counts.shape == (10,)
        assert counts[0] == 10.0
        assert counts[1] == 0.0
        assert counts.sum() == 10.0

    def test_child_visit_counts_sparse_children(self) -> None:
        """Counts are correct when children have non-contiguous action indices."""
        root = MCTSNode()
        policy = np.zeros(8, dtype=np.float32)
        policy[2] = 0.5
        policy[7] = 0.5
        mask = np.zeros(8, dtype=np.bool_)
        mask[2] = True
        mask[7] = True
        root.expand(policy, mask)
        root.children[2].visit_count = 4
        root.children[7].visit_count = 6
        counts = root.child_visit_counts(action_space_size=8)
        assert counts[2] == 4.0
        assert counts[7] == 6.0
        assert counts.sum() == 10.0


# ============================================================================
# PUCTPolicy Tests
# ============================================================================


class TestPUCTPolicy:
    """Tests for the PUCT child selection policy."""

    def test_default_c_puct(self) -> None:
        """Default c_puct matches DEFAULT_C_PUCT constant."""
        policy = PUCTPolicy()
        assert policy.c_puct == DEFAULT_C_PUCT
        assert policy.c_puct == 1.5

    def test_custom_c_puct(self) -> None:
        """Custom c_puct value is stored correctly."""
        policy = PUCTPolicy(c_puct=3.0)
        assert policy.c_puct == 3.0

    def test_select_child_prefers_unvisited(self) -> None:
        """Unvisited children get high exploration bonus and are preferred."""
        policy = PUCTPolicy(c_puct=1.5)
        # 3 children: action 0 visited 10 times, action 1 visited 10 times,
        # action 2 never visited
        priors = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        values = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        visits = np.array([10, 10, 0], dtype=np.int64)
        valid = np.ones(3, dtype=np.bool_)

        selected = policy.select_child(20, priors, values, visits, valid)
        # Action 2 has infinite exploration bonus (sqrt(20)/(1+0)) and Q=0
        assert selected == 2

    def test_select_child_converges_to_best_q(self) -> None:
        """With many visits, selection converges to best Q-value child."""
        policy = PUCTPolicy(c_puct=1.5)
        # All heavily visited; action 1 has best Q-value (-values in formula)
        priors = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        values = np.array([0.2, -0.8, 0.3], dtype=np.float32)
        visits = np.array([1000, 1000, 1000], dtype=np.int64)
        valid = np.ones(3, dtype=np.bool_)

        selected = policy.select_child(3000, priors, values, visits, valid)
        # score = -Q + exploration. For action 1: -(-0.8) + small = 0.8 + small
        # For action 0: -(0.2) + small = -0.2 + small
        # For action 2: -(0.3) + small = -0.3 + small
        # Action 1 has the highest score due to its low (negative) Q-value
        assert selected == 1

    def test_select_child_respects_valid_mask(self) -> None:
        """Invalid actions are never selected."""
        policy = PUCTPolicy(c_puct=1.5)
        # Action 0 is invalid but would otherwise be the best choice (unvisited)
        priors = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        values = np.array([0.0, 0.1, 0.2], dtype=np.float32)
        visits = np.array([0, 5, 5], dtype=np.int64)
        valid = np.array([False, True, True], dtype=np.bool_)

        selected = policy.select_child(10, priors, values, visits, valid)
        assert selected != 0
        assert valid[selected] is np.True_

    def test_select_child_single_valid_action(self) -> None:
        """With only one valid action, it must be selected."""
        policy = PUCTPolicy(c_puct=1.5)
        priors = np.array([0.1, 0.8, 0.1], dtype=np.float32)
        values = np.array([0.9, -0.5, 0.7], dtype=np.float32)
        visits = np.array([100, 1, 100], dtype=np.int64)
        valid = np.array([False, True, False], dtype=np.bool_)

        selected = policy.select_child(201, priors, values, visits, valid)
        assert selected == 1

    def test_select_child_higher_c_puct_increases_exploration(self) -> None:
        """Higher c_puct tilts selection toward less-visited children."""
        priors = np.array([0.5, 0.5], dtype=np.float32)
        # Action 0 has worse Q but fewer visits
        values = np.array([0.3, -0.5], dtype=np.float32)
        visits = np.array([2, 50], dtype=np.int64)
        valid = np.ones(2, dtype=np.bool_)

        # With low c_puct, might prefer action 1 (better Q via negamax)
        low_policy = PUCTPolicy(c_puct=0.01)
        low_policy.select_child(52, priors, values, visits, valid)

        # With high c_puct, should prefer action 0 (fewer visits, exploration bonus)
        high_policy = PUCTPolicy(c_puct=100.0)
        high_choice = high_policy.select_child(52, priors, values, visits, valid)

        assert high_choice == 0  # high exploration -> less-visited child

    def test_select_child_zero_parent_visits(self) -> None:
        """With 0 parent visits, exploration term is 0 and -Q dominates."""
        policy = PUCTPolicy(c_puct=1.5)
        priors = np.array([0.5, 0.5], dtype=np.float32)
        values = np.array([0.8, -0.3], dtype=np.float32)
        visits = np.array([0, 0], dtype=np.int64)
        valid = np.ones(2, dtype=np.bool_)

        selected = policy.select_child(0, priors, values, visits, valid)
        # scores = -Q + 0 = [-0.8, 0.3]. Action 1 wins.
        assert selected == 1

    def test_select_child_prior_influence(self) -> None:
        """Higher prior boosts exploration term for a child."""
        policy = PUCTPolicy(c_puct=1.5)
        # Same Q and visits, but action 0 has much higher prior
        priors = np.array([0.9, 0.1], dtype=np.float32)
        values = np.array([0.0, 0.0], dtype=np.float32)
        visits = np.array([5, 5], dtype=np.int64)
        valid = np.ones(2, dtype=np.bool_)

        selected = policy.select_child(10, priors, values, visits, valid)
        # Exploration for action 0: 1.5 * 0.9 * sqrt(10) / 6 > 1.5 * 0.1 * sqrt(10) / 6
        assert selected == 0

    def test_select_child_returns_int(self) -> None:
        """Return type is a Python int."""
        policy = PUCTPolicy()
        priors = np.array([0.5, 0.5], dtype=np.float32)
        values = np.zeros(2, dtype=np.float32)
        visits = np.zeros(2, dtype=np.int64)
        valid = np.ones(2, dtype=np.bool_)

        result = policy.select_child(0, priors, values, visits, valid)
        assert isinstance(result, int)

    def test_select_child_all_equal_selects_first(self) -> None:
        """When all scores are equal, argmax returns first index."""
        policy = PUCTPolicy(c_puct=1.5)
        priors = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        values = np.zeros(4, dtype=np.float32)
        visits = np.zeros(4, dtype=np.int64)
        valid = np.ones(4, dtype=np.bool_)

        selected = policy.select_child(0, priors, values, visits, valid)
        # All scores identical -> argmax returns 0
        assert selected == 0


# ============================================================================
# get_action_probabilities Tests
# ============================================================================


class TestGetActionProbabilities:
    """Tests for converting visit counts to action probabilities."""

    def test_temperature_one_proportional(self) -> None:
        """temperature=1.0 gives probabilities proportional to visit counts."""
        root = MCTSNode()
        policy = np.ones(4, dtype=np.float32)
        mask = np.ones(4, dtype=np.bool_)
        root.expand(policy, mask)

        root.children[0].visit_count = 10
        root.children[1].visit_count = 20
        root.children[2].visit_count = 30
        root.children[3].visit_count = 40

        probs = get_action_probabilities(root, action_space_size=4, temperature=1.0)
        assert probs.shape == (4,)
        assert probs.sum() == pytest.approx(1.0)
        assert probs[0] == pytest.approx(0.1)
        assert probs[1] == pytest.approx(0.2)
        assert probs[2] == pytest.approx(0.3)
        assert probs[3] == pytest.approx(0.4)

    def test_temperature_zero_greedy(self) -> None:
        """temperature=0 puts all mass on the most-visited action."""
        root = MCTSNode()
        policy = np.ones(4, dtype=np.float32)
        mask = np.ones(4, dtype=np.bool_)
        root.expand(policy, mask)

        root.children[0].visit_count = 5
        root.children[1].visit_count = 15
        root.children[2].visit_count = 3
        root.children[3].visit_count = 7

        probs = get_action_probabilities(root, action_space_size=4, temperature=0)
        assert probs[1] == 1.0
        assert probs.sum() == 1.0
        # All other probs are 0
        assert probs[0] == 0.0
        assert probs[2] == 0.0
        assert probs[3] == 0.0

    def test_no_visits_returns_zeros(self) -> None:
        """If no children have been visited, returns zeros."""
        root = MCTSNode()
        policy = np.ones(4, dtype=np.float32)
        mask = np.ones(4, dtype=np.bool_)
        root.expand(policy, mask)

        probs = get_action_probabilities(root, action_space_size=4, temperature=1.0)
        np.testing.assert_array_equal(probs, np.zeros(4, dtype=np.float32))

    def test_single_child_visited(self) -> None:
        """If only one child was visited, all mass goes to that action."""
        root = MCTSNode()
        policy = np.ones(4, dtype=np.float32)
        mask = np.ones(4, dtype=np.bool_)
        root.expand(policy, mask)

        root.children[2].visit_count = 10

        probs = get_action_probabilities(root, action_space_size=4, temperature=1.0)
        assert probs[2] == pytest.approx(1.0)
        assert probs.sum() == pytest.approx(1.0)

    def test_equal_visits_uniform(self) -> None:
        """Equal visit counts produce uniform distribution."""
        root = MCTSNode()
        policy = np.ones(4, dtype=np.float32)
        mask = np.ones(4, dtype=np.bool_)
        root.expand(policy, mask)

        for child in root.children.values():
            child.visit_count = 10

        probs = get_action_probabilities(root, action_space_size=4, temperature=1.0)
        for i in range(4):
            assert probs[i] == pytest.approx(0.25)

    def test_low_temperature_sharpens_distribution(self) -> None:
        """Lower temperature makes the distribution more peaked."""
        root = MCTSNode()
        policy = np.ones(3, dtype=np.float32)
        mask = np.ones(3, dtype=np.bool_)
        root.expand(policy, mask)

        root.children[0].visit_count = 10
        root.children[1].visit_count = 20
        root.children[2].visit_count = 5

        probs_high_t = get_action_probabilities(root, action_space_size=3, temperature=2.0)
        probs_low_t = get_action_probabilities(root, action_space_size=3, temperature=0.5)

        # Lower temperature should concentrate more on the most-visited
        assert probs_low_t[1] > probs_high_t[1]

    def test_output_dtype_is_float32(self) -> None:
        """Output array has float32 dtype."""
        root = MCTSNode()
        policy = np.ones(3, dtype=np.float32)
        mask = np.ones(3, dtype=np.bool_)
        root.expand(policy, mask)
        root.children[0].visit_count = 5

        probs = get_action_probabilities(root, action_space_size=3, temperature=1.0)
        assert probs.dtype == np.float32

    def test_action_space_larger_than_children(self) -> None:
        """Unrepresented actions get probability 0."""
        root = MCTSNode()
        policy = np.array([0.5, 0.5], dtype=np.float32)
        mask = np.array([True, True], dtype=np.bool_)
        root.expand(policy, mask)
        root.children[0].visit_count = 5
        root.children[1].visit_count = 5

        probs = get_action_probabilities(root, action_space_size=10, temperature=1.0)
        assert probs.shape == (10,)
        assert probs[0] == pytest.approx(0.5)
        assert probs[1] == pytest.approx(0.5)
        for i in range(2, 10):
            assert probs[i] == 0.0


# ============================================================================
# select_action Tests
# ============================================================================


class TestSelectAction:
    """Tests for selecting a single action from a probability distribution."""

    def test_temperature_zero_returns_argmax(self) -> None:
        """temperature=0 always returns the action with highest probability."""
        probs = np.array([0.1, 0.5, 0.3, 0.1], dtype=np.float32)
        action = select_action(probs, temperature=0)
        assert action == 1

    def test_with_action_mask_only_selects_valid(self) -> None:
        """Masked actions are never selected."""
        probs = np.array([0.0, 0.8, 0.1, 0.1], dtype=np.float32)
        mask = np.array([True, False, True, True], dtype=np.bool_)

        # Run multiple times to check stochasticity
        for _ in range(50):
            action = select_action(probs, temperature=1.0, action_mask=mask)
            assert mask[action], f"Selected invalid action {action}"

    def test_all_zero_probs_with_mask_returns_first_valid(self) -> None:
        """When probs are all zero and mask is applied, returns first valid action."""
        probs = np.zeros(5, dtype=np.float32)
        mask = np.array([False, False, True, False, True], dtype=np.bool_)
        action = select_action(probs, temperature=1.0, action_mask=mask)
        assert action == 2  # first valid

    def test_temperature_one_deterministic_seed(self) -> None:
        """With a fixed seed, temperature=1.0 gives reproducible results."""
        probs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        np.random.seed(42)
        action1 = select_action(probs, temperature=1.0)
        np.random.seed(42)
        action2 = select_action(probs, temperature=1.0)
        assert action1 == action2

    def test_without_mask_works(self) -> None:
        """select_action works without an action_mask parameter."""
        probs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        action = select_action(probs, temperature=0)
        assert action == 2

    def test_returns_int(self) -> None:
        """Return type is a Python int."""
        probs = np.array([0.5, 0.5], dtype=np.float32)
        action = select_action(probs, temperature=0)
        assert isinstance(action, int)

    def test_mask_with_uniform_probs(self) -> None:
        """Mask correctly filters even with uniform probabilities."""
        probs = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        mask = np.array([False, False, False, True], dtype=np.bool_)
        action = select_action(probs, temperature=1.0, action_mask=mask)
        assert action == 3

    def test_all_mass_on_single_action(self) -> None:
        """When all probability mass is on a single action, it is always selected."""
        probs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        for _ in range(20):
            action = select_action(probs, temperature=1.0)
            assert action == 2

    def test_temperature_effect_on_sampling(self) -> None:
        """Higher temperature makes distribution more uniform."""
        probs = np.array([0.9, 0.1], dtype=np.float32)
        np.random.seed(12345)

        # With very low temperature, almost always picks action 0
        low_t_choices = [select_action(probs, temperature=0.01) for _ in range(100)]
        assert sum(1 for c in low_t_choices if c == 0) > 90

    def test_no_valid_actions_returns_zero(self) -> None:
        """Edge case: all-False mask returns 0."""
        probs = np.array([0.5, 0.5], dtype=np.float32)
        mask = np.array([False, False], dtype=np.bool_)
        action = select_action(probs, temperature=1.0, action_mask=mask)
        assert action == 0


# ============================================================================
# MCTSEngine Integration Tests
# ============================================================================


class TestMCTSEngine:
    """Integration tests for the full MCTS search engine."""

    def test_search_returns_valid_action_probs(
        self,
        small_mcts_config: MCTSConfig,
        mock_env: MockEnv,
    ) -> None:
        """search() returns probs that sum to ~1 with correct shape."""
        engine = MCTSEngine(small_mcts_config, mock_eval_fn)
        action_probs, _root_value = engine.search(mock_env, num_simulations=10)

        assert action_probs.shape == (mock_env.action_space_size,)
        assert action_probs.sum() == pytest.approx(1.0, abs=1e-5)
        assert np.all(action_probs >= 0)

    def test_search_with_num_simulations_override(
        self,
        small_mcts_config: MCTSConfig,
        mock_env: MockEnv,
    ) -> None:
        """num_simulations argument overrides config."""
        engine = MCTSEngine(small_mcts_config, mock_eval_fn)

        # Override with fewer simulations
        probs_few, _ = engine.search(mock_env, num_simulations=2)
        assert probs_few.shape == (mock_env.action_space_size,)
        # Should still produce valid output
        assert probs_few.sum() == pytest.approx(1.0, abs=1e-5)

    def test_search_with_no_noise(
        self,
        small_mcts_config: MCTSConfig,
        mock_env: MockEnv,
    ) -> None:
        """search(add_noise=False) produces valid results without Dirichlet noise."""
        engine = MCTSEngine(small_mcts_config, mock_eval_fn)
        probs, _value = engine.search(mock_env, num_simulations=10, add_noise=False)

        assert probs.shape == (mock_env.action_space_size,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_search_does_not_mutate_original_env(
        self,
        small_mcts_config: MCTSConfig,
    ) -> None:
        """The original environment state is unchanged after search."""
        env = MockEnv(action_space_size=4, terminal_after=10)
        original_step_count = env._step_count
        original_terminal = env._terminal

        engine = MCTSEngine(small_mcts_config, mock_eval_fn)
        engine.search(env, num_simulations=10)

        assert env._step_count == original_step_count
        assert env._terminal == original_terminal

    def test_search_root_value_is_float(
        self,
        small_mcts_config: MCTSConfig,
        mock_env: MockEnv,
    ) -> None:
        """root_value from search is a float."""
        engine = MCTSEngine(small_mcts_config, mock_eval_fn)
        _, root_value = engine.search(mock_env, num_simulations=10)
        assert isinstance(root_value, float)

    def test_search_with_biased_eval_fn(
        self,
        small_mcts_config: MCTSConfig,
        mock_env: MockEnv,
    ) -> None:
        """Search with a biased eval_fn concentrates visits on the favored action."""
        engine = MCTSEngine(small_mcts_config, biased_eval_fn)
        probs, _ = engine.search(mock_env, num_simulations=50, add_noise=False)

        # Action 0 should get the most probability mass
        assert probs[0] == probs.max()

    def test_search_uses_config_num_simulations(self) -> None:
        """When num_simulations is not overridden, config value is used."""
        config = MCTSConfig(num_simulations=5, c_puct=1.5)
        env = MockEnv(action_space_size=4, terminal_after=10)

        engine = MCTSEngine(config, mock_eval_fn)
        probs, _ = engine.search(env)

        assert probs.shape == (4,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_engine_config_property(self, small_mcts_config: MCTSConfig) -> None:
        """config property returns the stored MCTSConfig."""
        engine = MCTSEngine(small_mcts_config, mock_eval_fn)
        assert engine.config is small_mcts_config

    def test_engine_with_custom_selection_policy(
        self,
        small_mcts_config: MCTSConfig,
        mock_env: MockEnv,
    ) -> None:
        """Engine accepts a custom PUCTPolicy."""
        custom_policy = PUCTPolicy(c_puct=5.0)
        engine = MCTSEngine(small_mcts_config, mock_eval_fn, selection_policy=custom_policy)

        probs, _ = engine.search(mock_env, num_simulations=10)
        assert probs.shape == (mock_env.action_space_size,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-5)

    def test_engine_select_child_raises_on_unexpanded(
        self,
        small_mcts_config: MCTSConfig,
    ) -> None:
        """_select_child raises ValueError on unexpanded node."""
        engine = MCTSEngine(small_mcts_config, mock_eval_fn)
        empty_node = MCTSNode()
        with pytest.raises(ValueError, match="Cannot select child"):
            engine._select_child(empty_node)

    def test_search_terminal_env_returns_valid_output(self) -> None:
        """Searching from an already-terminal-after-1 env still works."""
        config = MCTSConfig(num_simulations=5, c_puct=1.5)
        env = MockEnv(action_space_size=4, terminal_after=1)

        engine = MCTSEngine(config, mock_eval_fn)
        probs, value = engine.search(env, num_simulations=5)

        assert probs.shape == (4,)
        assert isinstance(value, float)


class TestMCTSEngineNoValidActions:
    """Tests for MCTS when leaf node has no valid actions."""

    def test_search_no_valid_actions_at_leaf_does_not_expand(self) -> None:
        """When leaf has all-zero action mask, node is not expanded."""

        class NoValidActionsEnv(MockEnv):
            """Env where get_valid_actions always returns all-False."""

            def get_valid_actions(self) -> np.ndarray:
                return np.zeros(self.action_space_size, dtype=np.bool_)

        config = MCTSConfig(num_simulations=5, c_puct=1.5)
        env = NoValidActionsEnv(action_space_size=4, terminal_after=10)
        engine = MCTSEngine(config, mock_eval_fn)
        probs, _value = engine.search(env, num_simulations=5)

        # Should return valid output without crashing
        assert probs.shape == (4,)
        assert not np.isnan(probs).any()

    def test_select_action_empty_mask_returns_zero(self) -> None:
        """select_action with all-zero mask and zero probs returns 0."""
        probs = np.zeros(5, dtype=np.float32)
        mask = np.zeros(5, dtype=np.bool_)
        action = select_action(probs, temperature=1.0, action_mask=mask)
        assert action == 0

    def test_select_action_zero_probs_no_mask_returns_argmax(self) -> None:
        """select_action with zero probs and no mask falls through to argmax."""
        probs = np.zeros(5, dtype=np.float32)
        action = select_action(probs, temperature=1.0)
        # argmax of all zeros returns 0
        assert action == 0

    def test_get_action_probabilities_zero_visits(self) -> None:
        """get_action_probabilities with zero visits returns zero array."""
        node = MCTSNode()
        probs = get_action_probabilities(node, action_space_size=5)
        assert probs.shape == (5,)
        assert probs.sum() == 0.0
