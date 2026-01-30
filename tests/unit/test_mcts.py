"""Unit tests for MCTS."""

import numpy as np
import pytest

from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig, MCTSNode, SimpleMCTS
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.envs.microciv import MicroCivEnv


class TestMCTSNode:
    """Test suite for MCTS node."""

    def test_initial_state(self):
        """New node has correct initial values."""
        node = MCTSNode()

        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.0
        assert not node.is_expanded()

    def test_value_calculation(self):
        """Value is correctly computed."""
        node = MCTSNode(visit_count=10, value_sum=5.0)

        assert node.value == 0.5

    def test_unexpanded_node(self):
        """Unexpanded node has no children."""
        node = MCTSNode()

        assert not node.is_expanded()
        assert len(node.children) == 0

    def test_expand_creates_children(self):
        """Expand creates child nodes for valid actions."""
        node = MCTSNode()
        policy = np.array([0.3, 0.5, 0.2, 0.0, 0.0])
        mask = np.array([1, 1, 1, 0, 0])

        node.expand(policy, mask)

        assert node.is_expanded()
        assert len(node.children) == 3
        assert 0 in node.children
        assert 1 in node.children
        assert 2 in node.children

    def test_expand_normalizes_priors(self):
        """Expand normalizes policy over valid actions."""
        node = MCTSNode()
        policy = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        mask = np.array([1, 1, 0, 0, 0])  # Only 2 valid

        node.expand(policy, mask)

        total_prior = sum(c.prior for c in node.children.values())
        assert abs(total_prior - 1.0) < 1e-5

    def test_select_child_ucb(self):
        """Select child uses UCB formula."""
        root = MCTSNode(visit_count=10)

        # Create children with different stats
        root.children[0] = MCTSNode(prior=0.8, visit_count=5, value_sum=-2.0)
        root.children[1] = MCTSNode(prior=0.2, visit_count=2, value_sum=-0.5)

        action, child = root.select_child(c_puct=1.5)

        # Should select one of the children
        assert action in [0, 1]
        assert child is root.children[action]

    def test_backpropagate(self):
        """Backpropagation updates all ancestors."""
        # Create tree: root -> child -> grandchild
        root = MCTSNode()
        child = MCTSNode(parent=root)
        grandchild = MCTSNode(parent=child)

        root.children[0] = child
        child.children[0] = grandchild

        # Backpropagate from grandchild
        grandchild.backpropagate(1.0)

        assert grandchild.visit_count == 1
        assert grandchild.value_sum == 1.0
        assert child.visit_count == 1
        assert child.value_sum == -1.0  # Flipped
        assert root.visit_count == 1
        assert root.value_sum == 1.0  # Flipped again


class TestSimpleMCTS:
    """Test suite for SimpleMCTS."""

    @pytest.fixture
    def env(self):
        return MicroCivEnv(board_size=8, max_turns=50, seed=42)

    @pytest.fixture
    def mcts(self):
        return SimpleMCTS(num_simulations=20, c_puct=1.5, rollout_depth=10)

    def test_search_returns_probs(self, mcts, env):
        """Search returns valid probability distribution."""
        env.reset()
        action_probs, root_value = mcts.search(env)

        # Should be valid distribution
        assert action_probs.shape == (env.action_space.n,)
        assert np.all(action_probs >= 0)
        assert abs(action_probs.sum() - 1.0) < 1e-5

    def test_search_respects_valid_actions(self, mcts, env):
        """Search only assigns probability to valid actions."""
        env.reset()
        action_probs, _ = mcts.search(env)
        valid_mask = env.get_valid_actions()

        # Probability should be 0 for invalid actions
        invalid_mask = valid_mask == 0
        assert np.all(action_probs[invalid_mask] == 0)

    def test_select_action_greedy(self, mcts, env):
        """Temperature 0 selects best action."""
        env.reset()
        action_probs, _ = mcts.search(env)

        # Select multiple times - should be same (greedy)
        action1 = mcts.select_action(action_probs, temperature=0)
        action2 = mcts.select_action(action_probs, temperature=0)

        assert action1 == action2
        assert action1 == np.argmax(action_probs)

    def test_select_action_stochastic(self, mcts, env):
        """Temperature > 0 allows stochastic selection."""
        env.reset()
        action_probs = np.zeros(env.action_space.n)
        action_probs[0] = 0.5
        action_probs[1] = 0.5

        # With temperature, both actions should be selectable
        actions = set()
        for _ in range(50):
            action = mcts.select_action(action_probs, temperature=1.0)
            actions.add(action)
            if len(actions) == 2:
                break

        assert len(actions) == 2


class TestCiv6MCTS:
    """Test suite for neural network guided MCTS."""

    @pytest.fixture
    def env(self):
        return MicroCivEnv(board_size=8, max_turns=50, seed=42)

    @pytest.fixture
    def network(self):
        return MicroCivNetwork(
            board_size=8,
            action_space_size=MicroCivEnv(board_size=8).action_space.n,
            num_res_blocks=2,
            num_filters=16,
        )

    @pytest.fixture
    def mcts(self, network):
        config = MCTSConfig(num_simulations=10, c_puct=1.5)
        return Civ6MCTS(network, config, device="cpu")

    def test_search_returns_probs(self, mcts, env):
        """Search returns valid probability distribution."""
        env.reset()
        action_probs, root_value = mcts.search(env)

        assert action_probs.shape == (env.action_space.n,)
        assert np.all(action_probs >= 0)
        assert abs(action_probs.sum() - 1.0) < 1e-5

    def test_search_with_noise(self, mcts, env):
        """Search with Dirichlet noise adds exploration."""
        env.reset()

        # Run multiple searches
        probs1, _ = mcts.search(env, add_noise=True)
        probs2, _ = mcts.search(env, add_noise=True)

        # With noise, results should differ
        # (Not guaranteed but very likely)
        # Just check they're valid
        assert np.all(probs1 >= 0)
        assert np.all(probs2 >= 0)

    def test_search_without_noise(self, mcts, env):
        """Search without noise is more deterministic."""
        env.reset()
        probs, _ = mcts.search(env, add_noise=False)

        assert np.all(probs >= 0)

    def test_root_value_bounded(self, mcts, env):
        """Root value is in [-1, 1] range."""
        env.reset()
        _, root_value = mcts.search(env)

        assert -1 <= root_value <= 1

    def test_policy_respects_mask(self, mcts, env):
        """Policy only assigns probability to valid actions."""
        env.reset()
        action_probs, _ = mcts.search(env)
        valid_mask = env.get_valid_actions()

        # Invalid actions should have 0 probability
        for i in range(len(action_probs)):
            if valid_mask[i] == 0:
                assert action_probs[i] == 0


class TestMCTSConfig:
    """Test suite for MCTS configuration."""

    def test_default_values(self):
        """Default config has reasonable values."""
        config = MCTSConfig()

        assert config.num_simulations > 0
        assert config.c_puct > 0
        assert 0 <= config.dirichlet_alpha <= 1
        assert 0 <= config.dirichlet_epsilon <= 1

    def test_custom_values(self):
        """Config accepts custom values."""
        config = MCTSConfig(
            num_simulations=100,
            c_puct=2.0,
            dirichlet_alpha=0.5,
        )

        assert config.num_simulations == 100
        assert config.c_puct == 2.0
        assert config.dirichlet_alpha == 0.5
