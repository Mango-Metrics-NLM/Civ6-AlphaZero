"""Unit tests for simplified MCTS module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from alpha_zero.utils.simple_mcts import (
    MCTSConfig,
    MCTSNode,
    MCTS,
    SimpleMCTS,
    create_network_fn,
    create_mcts,
)


class TestMCTSConfig:
    """Tests for MCTSConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MCTSConfig()

        assert config.num_simulations == 50
        assert config.c_puct == 1.5
        assert config.temperature == 1.0
        assert config.dirichlet_alpha == 0.3
        assert config.dirichlet_epsilon == 0.25
        assert config.max_depth == 50

    def test_custom_values(self):
        """Test custom configuration."""
        config = MCTSConfig(
            num_simulations=100,
            c_puct=2.0,
            temperature=0.5,
        )

        assert config.num_simulations == 100
        assert config.c_puct == 2.0
        assert config.temperature == 0.5


class TestMCTSNode:
    """Tests for MCTSNode class."""

    def test_initialization(self):
        """Test node initialization."""
        node = MCTSNode(prior=0.5, parent=None, action=3)

        assert node.prior == 0.5
        assert node.parent is None
        assert node.action == 3
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.children == {}
        assert node.is_terminal is False

    def test_default_initialization(self):
        """Test default node initialization."""
        node = MCTSNode()

        assert node.prior == 0.0
        assert node.parent is None
        assert node.action is None

    def test_value_property_zero_visits(self):
        """Test value property with zero visits."""
        node = MCTSNode()
        assert node.value == 0.0

    def test_value_property_with_visits(self):
        """Test value property with visits."""
        node = MCTSNode()
        node.visit_count = 5
        node.value_sum = 2.5

        assert node.value == 0.5

    def test_is_expanded_property(self):
        """Test is_expanded property."""
        node = MCTSNode()
        assert node.is_expanded is False

        node.children[0] = MCTSNode()
        assert node.is_expanded is True

    def test_ucb_score(self):
        """Test UCB score calculation."""
        node = MCTSNode(prior=0.5)
        node.visit_count = 5
        node.value_sum = 2.5  # value = 0.5

        parent_visits = 100
        c_puct = 1.5

        score = node.ucb_score(parent_visits, c_puct)

        # UCB = Q + c * P * sqrt(N_parent) / (1 + N)
        expected_q = 0.5
        expected_exploration = 1.5 * 0.5 * np.sqrt(100) / 6

        assert score == pytest.approx(expected_q + expected_exploration)

    def test_ucb_score_unvisited(self):
        """Test UCB score for unvisited node (exploration bonus high)."""
        node = MCTSNode(prior=0.5)
        # visit_count = 0

        parent_visits = 100
        c_puct = 1.5

        score = node.ucb_score(parent_visits, c_puct)

        # UCB = 0 + c * P * sqrt(N_parent) / 1
        expected = 1.5 * 0.5 * np.sqrt(100)
        assert score == pytest.approx(expected)

    def test_select_child(self):
        """Test child selection by UCB score."""
        parent = MCTSNode()
        parent.visit_count = 100

        # Create children with different values
        child1 = MCTSNode(prior=0.3)
        child1.visit_count = 10
        child1.value_sum = 3.0  # value = 0.3

        child2 = MCTSNode(prior=0.5)
        child2.visit_count = 5
        child2.value_sum = 4.0  # value = 0.8

        parent.children[0] = child1
        parent.children[1] = child2

        selected = parent.select_child(c_puct=1.5)

        # Child2 should have higher UCB due to higher value and prior
        assert selected == child2

    def test_expand(self):
        """Test node expansion."""
        node = MCTSNode()
        policy = np.array([0.2, 0.3, 0.3, 0.2])
        mask = np.array([1, 1, 0, 1])  # action 2 invalid

        node.expand(policy, mask)

        assert len(node.children) == 3
        assert 0 in node.children
        assert 1 in node.children
        assert 2 not in node.children  # masked out
        assert 3 in node.children

        # Check priors are normalized
        total_prior = sum(c.prior for c in node.children.values())
        assert total_prior == pytest.approx(1.0)

    def test_expand_all_masked(self):
        """Test expansion with all actions masked."""
        node = MCTSNode()
        policy = np.array([0.25, 0.25, 0.25, 0.25])
        mask = np.array([0, 0, 0, 0])

        node.expand(policy, mask)

        assert node.is_terminal is True
        assert len(node.children) == 0

    def test_expand_zero_policy(self):
        """Test expansion with zero policy (uses uniform)."""
        node = MCTSNode()
        policy = np.array([0.0, 0.0, 0.0, 0.0])
        mask = np.array([1, 1, 1, 0])

        node.expand(policy, mask)

        # Should use uniform distribution
        for child in node.children.values():
            assert child.prior == pytest.approx(1.0 / 3)

    def test_backpropagate(self):
        """Test value backpropagation."""
        root = MCTSNode()
        child = MCTSNode(parent=root, action=0)
        root.children[0] = child

        grandchild = MCTSNode(parent=child, action=1)
        child.children[1] = grandchild

        # Backpropagate value from grandchild
        grandchild.backpropagate(0.5)

        # Grandchild: +0.5
        assert grandchild.visit_count == 1
        assert grandchild.value_sum == 0.5

        # Child: -0.5 (negated for opponent)
        assert child.visit_count == 1
        assert child.value_sum == -0.5

        # Root: +0.5 (negated again)
        assert root.visit_count == 1
        assert root.value_sum == 0.5


class TestMCTS:
    """Tests for full MCTS class."""

    def test_initialization(self):
        """Test MCTS initialization."""
        mcts = MCTS()

        assert mcts.config is not None
        assert mcts.network_fn is None
        assert mcts.root is None

    def test_custom_config(self):
        """Test MCTS with custom config."""
        config = MCTSConfig(num_simulations=25)
        mcts = MCTS(config=config)

        assert mcts.config.num_simulations == 25

    def test_set_network(self):
        """Test setting network function."""
        mcts = MCTS()

        def mock_network(obs):
            return np.ones(4) / 4, 0.5

        mcts.set_network(mock_network)
        assert mcts.network_fn is not None

    def test_search_without_network_raises(self):
        """Test that search raises without network."""
        mcts = MCTS()
        obs = {
            "spatial": np.zeros((13, 8, 8)),
            "vector": np.zeros(64),
            "action_mask": np.ones(4),
        }

        with pytest.raises(ValueError, match="Network function not set"):
            mcts.search(obs, lambda o, a: (o, 0, False))

    def test_search_returns_action_probs(self):
        """Test that search returns valid action probabilities."""
        config = MCTSConfig(num_simulations=5)
        mcts = MCTS(config=config)

        def mock_network(obs):
            return np.array([0.25, 0.25, 0.25, 0.25]), 0.0

        mcts.set_network(mock_network)

        obs = {
            "spatial": np.zeros((13, 8, 8)),
            "vector": np.zeros(64),
            "action_mask": np.array([1, 1, 1, 0]),
        }

        def mock_step(obs, action):
            return obs.copy(), 0.0, False

        action_probs = mcts.search(obs, mock_step, add_noise=False)

        assert action_probs.shape == (4,)
        assert action_probs.sum() == pytest.approx(1.0)
        # Masked action should have 0 probability
        assert action_probs[3] == 0.0

    def test_search_with_terminal_state(self):
        """Test search handles terminal states."""
        config = MCTSConfig(num_simulations=3)
        mcts = MCTS(config=config)

        def mock_network(obs):
            return np.array([0.5, 0.5]), 0.0

        mcts.set_network(mock_network)

        obs = {
            "spatial": np.zeros((13, 8, 8)),
            "vector": np.zeros(64),
            "action_mask": np.array([1, 1]),
        }

        # Step function that returns terminal immediately
        def mock_step(obs, action):
            return obs.copy(), 1.0, True

        action_probs = mcts.search(obs, mock_step, add_noise=False)

        assert action_probs.shape == (2,)
        assert action_probs.sum() == pytest.approx(1.0)

    def test_get_action_probs_temperature(self):
        """Test action probability extraction with temperature."""
        mcts = MCTS()
        mcts.root = MCTSNode()

        # Add children with different visit counts
        child1 = MCTSNode(prior=0.5)
        child1.visit_count = 10

        child2 = MCTSNode(prior=0.5)
        child2.visit_count = 5

        mcts.root.children[0] = child1
        mcts.root.children[1] = child2

        # With temperature=1
        mcts.config.temperature = 1.0
        probs = mcts._get_action_probs(2)
        assert probs[0] > probs[1]  # Higher visit count = higher prob

        # With temperature=0 (greedy)
        mcts.config.temperature = 0
        probs_greedy = mcts._get_action_probs(2)
        assert probs_greedy[0] == 1.0
        assert probs_greedy[1] == 0.0

    def test_select_action(self):
        """Test action selection from probabilities."""
        mcts = MCTS()

        np.random.seed(42)
        action_probs = np.array([0.7, 0.2, 0.1, 0.0])

        # Should select valid actions
        for _ in range(10):
            action = mcts.select_action(action_probs)
            assert action in [0, 1, 2]
            assert action_probs[action] > 0

    def test_select_action_all_zero(self):
        """Test action selection with zero probabilities."""
        mcts = MCTS()
        action_probs = np.array([0.0, 0.0, 0.0, 0.0])

        action = mcts.select_action(action_probs)
        # Should return fallback
        assert action == 0


class TestSimpleMCTS:
    """Tests for SimpleMCTS class."""

    def test_initialization(self):
        """Test SimpleMCTS initialization."""
        mcts = SimpleMCTS()
        assert mcts.config is not None

    def test_custom_config(self):
        """Test SimpleMCTS with custom config."""
        config = MCTSConfig(temperature=0.5)
        mcts = SimpleMCTS(config=config)
        assert mcts.config.temperature == 0.5

    def test_get_action_probs(self):
        """Test action probability generation."""
        mcts = SimpleMCTS()

        policy = np.array([0.4, 0.3, 0.2, 0.1])
        mask = np.array([1, 1, 1, 0])

        probs = mcts.get_action_probs(policy, mask, add_noise=False)

        assert probs.shape == (4,)
        assert probs.sum() == pytest.approx(1.0)
        assert probs[3] == 0.0  # Masked action

    def test_get_action_probs_with_noise(self):
        """Test action probabilities with Dirichlet noise."""
        config = MCTSConfig(dirichlet_epsilon=0.25)
        mcts = SimpleMCTS(config=config)

        np.random.seed(42)
        policy = np.array([1.0, 0.0, 0.0, 0.0])
        mask = np.array([1, 1, 1, 0])

        probs = mcts.get_action_probs(policy, mask, add_noise=True)

        # With noise, previously zero actions should have some probability
        assert probs[1] > 0
        assert probs[2] > 0

    def test_get_action_probs_all_masked(self):
        """Test with all actions masked."""
        mcts = SimpleMCTS()

        policy = np.array([0.25, 0.25, 0.25, 0.25])
        mask = np.array([0, 0, 0, 0])

        probs = mcts.get_action_probs(policy, mask)

        assert probs.sum() == 0.0

    def test_get_action_probs_zero_policy(self):
        """Test with zero policy."""
        mcts = SimpleMCTS()

        policy = np.array([0.0, 0.0, 0.0, 0.0])
        mask = np.array([1, 1, 0, 0])

        probs = mcts.get_action_probs(policy, mask, add_noise=False)

        # Should use uniform distribution
        assert probs[0] == pytest.approx(0.5)
        assert probs[1] == pytest.approx(0.5)

    def test_select_action(self):
        """Test action selection."""
        mcts = SimpleMCTS()

        np.random.seed(42)
        policy = np.array([0.8, 0.1, 0.1, 0.0])
        mask = np.array([1, 1, 1, 0])

        action, probs = mcts.select_action(policy, mask, add_noise=False)

        assert action in [0, 1, 2]
        assert mask[action] > 0
        assert probs.shape == (4,)

    def test_select_action_all_masked(self):
        """Test action selection with all masked."""
        mcts = SimpleMCTS()

        policy = np.array([0.25, 0.25, 0.25, 0.25])
        mask = np.array([0, 0, 0, 0])

        action, probs = mcts.select_action(policy, mask)

        # Should return 0 as fallback
        assert action == 0

    def test_temperature_effect(self):
        """Test temperature effect on action probabilities."""
        policy = np.array([0.6, 0.3, 0.1])
        mask = np.array([1, 1, 1])

        # Low temperature (more deterministic)
        mcts_low = SimpleMCTS(MCTSConfig(temperature=0.1))
        probs_low = mcts_low.get_action_probs(policy, mask, add_noise=False)

        # High temperature (more uniform)
        mcts_high = SimpleMCTS(MCTSConfig(temperature=2.0))
        probs_high = mcts_high.get_action_probs(policy, mask, add_noise=False)

        # Low temperature should be more peaked
        assert probs_low[0] > probs_high[0]


class TestCreateNetworkFn:
    """Tests for create_network_fn helper."""

    def test_create_network_fn(self):
        """Test creating network function wrapper."""
        # Create a mock network
        mock_network = MagicMock()
        mock_network.return_value = (
            torch.tensor([[1.0, 2.0, 3.0]]),  # logits
            torch.tensor([[0.5]]),  # value
        )

        network_fn = create_network_fn(mock_network, device="cpu")

        obs = {
            "spatial": np.zeros((13, 8, 8), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
        }

        policy, value = network_fn(obs)

        # Check shapes
        assert policy.shape == (3,)
        assert isinstance(value, float)

        # Check policy is softmax of logits
        expected_policy = torch.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=-1).numpy()
        np.testing.assert_array_almost_equal(policy, expected_policy)

        assert value == pytest.approx(0.5)


class TestCreateMcts:
    """Tests for create_mcts factory function."""

    def test_create_simple_mcts(self):
        """Test creating SimpleMCTS."""
        mcts = create_mcts(use_full_mcts=False)
        assert isinstance(mcts, SimpleMCTS)

    def test_create_full_mcts(self):
        """Test creating full MCTS."""
        mcts = create_mcts(use_full_mcts=True)
        assert isinstance(mcts, MCTS)

    def test_create_with_custom_params(self):
        """Test creating with custom parameters."""
        mcts = create_mcts(
            num_simulations=100,
            c_puct=2.0,
            temperature=0.5,
            use_full_mcts=False,
        )

        assert mcts.config.num_simulations == 100
        assert mcts.config.c_puct == 2.0
        assert mcts.config.temperature == 0.5

    def test_create_full_mcts_with_network(self):
        """Test creating full MCTS with network."""
        mock_network = MagicMock()
        mock_network.return_value = (
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[0.0]]),
        )

        mcts = create_mcts(use_full_mcts=True, network=mock_network)

        assert isinstance(mcts, MCTS)
        assert mcts.network_fn is not None


class TestMCTSIntegration:
    """Integration tests for MCTS components."""

    def test_simple_mcts_exploration(self):
        """Test that SimpleMCTS provides exploration."""
        mcts = SimpleMCTS(MCTSConfig(dirichlet_epsilon=0.25))

        # Deterministic policy
        policy = np.array([1.0, 0.0, 0.0, 0.0])
        mask = np.array([1, 1, 1, 1])

        np.random.seed(42)
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for _ in range(100):
            action, _ = mcts.select_action(policy, mask, add_noise=True)
            action_counts[action] += 1

        # Should explore all actions due to noise
        assert all(count > 0 for count in action_counts.values())

    def test_mcts_node_tree_structure(self):
        """Test building MCTS tree structure."""
        root = MCTSNode()
        policy = np.array([0.4, 0.3, 0.2, 0.1])
        mask = np.array([1, 1, 1, 1])

        root.expand(policy, mask)
        assert len(root.children) == 4

        # Expand a child
        child = root.children[0]
        child.expand(policy, mask)
        assert len(child.children) == 4

        # Backpropagate
        grandchild = child.children[0]
        grandchild.backpropagate(1.0)

        assert grandchild.visit_count == 1
        assert child.visit_count == 1
        assert root.visit_count == 1

    def test_dirichlet_noise_distribution(self):
        """Test that Dirichlet noise follows expected properties."""
        mcts = SimpleMCTS(MCTSConfig(dirichlet_alpha=0.3, dirichlet_epsilon=1.0))

        policy = np.zeros(4)
        mask = np.array([1, 1, 1, 1])

        samples = []
        for seed in range(100):
            np.random.seed(seed)
            probs = mcts.get_action_probs(policy, mask, add_noise=True)
            samples.append(probs)

        samples = np.array(samples)

        # Mean should be approximately uniform
        mean_probs = samples.mean(axis=0)
        np.testing.assert_array_almost_equal(mean_probs, [0.25] * 4, decimal=1)
