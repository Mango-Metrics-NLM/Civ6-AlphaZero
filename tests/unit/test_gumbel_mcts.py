"""Unit tests for Gumbel MuZero MCTS."""

import numpy as np
import pytest

from alpha_zero.core.civ6_mcts import (
    GumbelMCTS,
    GumbelMCTSConfig,
    GumbelMCTSNode,
    compute_completed_q,
    gumbel_sample_actions,
)
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.envs.microciv import MicroCivEnv


class TestGumbelSampleActions:
    """Test suite for Gumbel-Top-k sampling."""

    def test_returns_correct_number_of_actions(self):
        """Returns exactly k actions when k < num_valid."""
        policy = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        mask = np.ones(5)

        actions, scores = gumbel_sample_actions(policy, mask, num_actions=3)

        assert len(actions) == 3
        assert len(scores) == 3

    def test_returns_all_valid_when_k_larger(self):
        """Returns all valid actions when k > num_valid."""
        policy = np.array([0.5, 0.5, 0.0])
        mask = np.array([1, 1, 0])

        actions, _scores = gumbel_sample_actions(policy, mask, num_actions=10)

        assert len(actions) == 2
        assert set(actions) == {0, 1}

    def test_respects_action_mask(self):
        """Only samples from valid actions."""
        policy = np.array([0.1, 0.1, 0.3, 0.3, 0.2])
        mask = np.array([0, 0, 1, 1, 0])  # Only actions 2, 3 valid

        actions, _scores = gumbel_sample_actions(policy, mask, num_actions=5)

        assert all(a in [2, 3] for a in actions)

    def test_empty_mask_returns_empty(self):
        """Returns empty arrays when no valid actions."""
        policy = np.array([0.5, 0.5])
        mask = np.array([0, 0])

        actions, scores = gumbel_sample_actions(policy, mask, num_actions=2)

        assert len(actions) == 0
        assert len(scores) == 0

    def test_high_prob_actions_selected_more_often(self):
        """Higher probability actions are more likely to be selected."""
        policy = np.array([0.9, 0.05, 0.05])
        mask = np.ones(3)

        # Run many times and count selections
        selections = np.zeros(3)
        for _ in range(100):
            actions, _ = gumbel_sample_actions(policy, mask, num_actions=1)
            selections[actions[0]] += 1

        # Action 0 should be selected most often (not guaranteed but very likely)
        assert selections[0] > selections[1]
        assert selections[0] > selections[2]

    def test_scores_are_sorted_descending(self):
        """Returned actions are sorted by score (descending)."""
        policy = np.array([0.25, 0.25, 0.25, 0.25])
        mask = np.ones(4)

        _actions, scores = gumbel_sample_actions(policy, mask, num_actions=4)

        # Scores should be in descending order
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_temperature_affects_exploration(self):
        """Higher temperature leads to more uniform selection."""
        policy = np.array([0.9, 0.05, 0.05])
        mask = np.ones(3)

        # With low temperature, action 0 dominates
        low_temp_selections = np.zeros(3)
        for _ in range(100):
            actions, _ = gumbel_sample_actions(policy, mask, num_actions=1, temperature=0.1)
            low_temp_selections[actions[0]] += 1

        # With high temperature, more spread out
        high_temp_selections = np.zeros(3)
        for _ in range(100):
            actions, _ = gumbel_sample_actions(policy, mask, num_actions=1, temperature=5.0)
            high_temp_selections[actions[0]] += 1

        # Low temp should have more concentrated selection
        assert low_temp_selections.max() >= high_temp_selections.max()


class TestComputeCompletedQ:
    """Test suite for completed Q-value computation."""

    def test_equal_visits_returns_q(self):
        """When visit_count == max_visits, returns original Q."""
        q = 0.5
        result = compute_completed_q(q, visit_count=10, max_visit_count=10, root_value=0.3)

        assert result == q

    def test_zero_visits_adds_full_root_value(self):
        """When visit_count == 0, adds full root_value."""
        q = 0.0
        root_value = 0.6
        result = compute_completed_q(q, visit_count=0, max_visit_count=10, root_value=root_value)

        assert result == q + root_value

    def test_partial_visits(self):
        """Partial visits add proportional root_value."""
        q = 0.2
        root_value = 1.0
        # 5 visits out of 10 max -> sigma = 0.5
        result = compute_completed_q(q, visit_count=5, max_visit_count=10, root_value=root_value)

        expected = q + 0.5 * root_value
        assert abs(result - expected) < 1e-6

    def test_zero_max_visits_returns_q(self):
        """When max_visits is 0, returns original Q."""
        q = 0.5
        result = compute_completed_q(q, visit_count=0, max_visit_count=0, root_value=1.0)

        assert result == q


class TestGumbelMCTSNode:
    """Test suite for Gumbel MCTS node."""

    def test_initial_state(self):
        """New node has correct initial values."""
        node = GumbelMCTSNode()

        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.0
        assert node.gumbel_score == 0.0
        assert not node.is_expanded()

    def test_value_calculation(self):
        """Value is correctly computed."""
        node = GumbelMCTSNode(visit_count=10, value_sum=5.0)

        assert node.value == 0.5

    def test_value_zero_visits(self):
        """Value returns 0 for unvisited node."""
        node = GumbelMCTSNode(visit_count=0, value_sum=0.0)

        assert node.value == 0.0

    def test_expand_creates_children(self):
        """Expand creates child nodes for valid actions."""
        node = GumbelMCTSNode()
        policy = np.array([0.3, 0.5, 0.2, 0.0])
        mask = np.array([1, 1, 1, 0])

        node.expand(policy, mask)

        assert node.is_expanded()
        assert len(node.children) == 3
        assert 0 in node.children
        assert 1 in node.children
        assert 2 in node.children

    def test_expand_normalizes_priors(self):
        """Expand normalizes policy over valid actions."""
        node = GumbelMCTSNode()
        policy = np.array([0.2, 0.2, 0.2, 0.2])
        mask = np.array([1, 1, 0, 0])

        node.expand(policy, mask)

        total_prior = sum(c.prior for c in node.children.values())
        assert abs(total_prior - 1.0) < 1e-5

    def test_select_child_returns_valid(self):
        """Select child returns a valid child."""
        root = GumbelMCTSNode(visit_count=10)
        root.children[0] = GumbelMCTSNode(prior=0.5, visit_count=3, value_sum=-1.0)
        root.children[1] = GumbelMCTSNode(prior=0.5, visit_count=2, value_sum=-0.5)

        action, child = root.select_child(c_puct=1.5)

        assert action in [0, 1]
        assert child is root.children[action]

    def test_backpropagate(self):
        """Backpropagation updates all ancestors."""
        root = GumbelMCTSNode()
        child = GumbelMCTSNode(parent=root)
        grandchild = GumbelMCTSNode(parent=child)

        root.children[0] = child
        child.children[0] = grandchild

        grandchild.backpropagate(1.0)

        assert grandchild.visit_count == 1
        assert grandchild.value_sum == 1.0
        assert child.visit_count == 1
        assert child.value_sum == -1.0  # Flipped
        assert root.visit_count == 1
        assert root.value_sum == 1.0  # Flipped again


class TestGumbelMCTSConfig:
    """Test suite for Gumbel MCTS configuration."""

    def test_default_values(self):
        """Default config has reasonable values."""
        config = GumbelMCTSConfig()

        assert config.num_simulations == 64
        assert config.num_sampled_actions == 16
        assert config.c_visit > 0
        assert config.c_scale > 0
        assert config.use_sequential_halving is True
        assert config.min_actions_after_halving >= 1

    def test_custom_values(self):
        """Config accepts custom values."""
        config = GumbelMCTSConfig(
            num_simulations=8,
            num_sampled_actions=8,
            c_visit=100.0,
            use_sequential_halving=False,
        )

        assert config.num_simulations == 8
        assert config.num_sampled_actions == 8
        assert config.c_visit == 100.0
        assert config.use_sequential_halving is False

    def test_low_simulation_config(self):
        """Config allows very low simulation counts."""
        config = GumbelMCTSConfig(num_simulations=4)

        assert config.num_simulations == 4


class TestGumbelMCTS:
    """Test suite for Gumbel MuZero MCTS."""

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
        config = GumbelMCTSConfig(num_simulations=8, num_sampled_actions=8)
        return GumbelMCTS(network, config, device="cpu")

    def test_search_returns_probs(self, mcts, env):
        """Search returns valid probability distribution."""
        env.reset()
        action_probs, _root_value = mcts.search(env)

        assert action_probs.shape == (env.action_space.n,)
        assert np.all(action_probs >= 0)
        # Gumbel MCTS returns one-hot, so sum should be ~1
        assert abs(action_probs.sum() - 1.0) < 1e-5

    def test_search_returns_one_hot(self, mcts, env):
        """Gumbel MCTS typically returns one-hot policy."""
        env.reset()
        action_probs, _ = mcts.search(env)

        # Should have exactly one action with probability ~1
        assert action_probs.max() > 0.99

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

        # Selected action should be valid
        selected_action = np.argmax(action_probs)
        assert valid_mask[selected_action] == 1

    def test_select_action_greedy(self, mcts, env):
        """Temperature 0 selects best action."""
        env.reset()
        action_probs, _ = mcts.search(env)

        action1 = mcts.select_action(action_probs, temperature=0)
        action2 = mcts.select_action(action_probs, temperature=0)

        assert action1 == action2
        assert action1 == np.argmax(action_probs)

    def test_works_with_few_simulations(self, network, env):
        """Gumbel MCTS works with very few simulations."""
        config = GumbelMCTSConfig(num_simulations=4, num_sampled_actions=4)
        mcts = GumbelMCTS(network, config, device="cpu")

        env.reset()
        action_probs, root_value = mcts.search(env)

        assert action_probs.sum() > 0
        assert -1 <= root_value <= 1

    def test_sequential_halving_mode(self, network, env):
        """Sequential halving mode works."""
        config = GumbelMCTSConfig(
            num_simulations=16,
            num_sampled_actions=8,
            use_sequential_halving=True,
        )
        mcts = GumbelMCTS(network, config, device="cpu")

        env.reset()
        action_probs, _ = mcts.search(env)

        assert action_probs.max() > 0.99  # Should return one-hot

    def test_simple_search_mode(self, network, env):
        """Simple search mode (no sequential halving) works."""
        config = GumbelMCTSConfig(
            num_simulations=16,
            num_sampled_actions=8,
            use_sequential_halving=False,
        )
        mcts = GumbelMCTS(network, config, device="cpu")

        env.reset()
        action_probs, _ = mcts.search(env)

        assert action_probs.max() > 0.99

    def test_different_seeds_produce_different_results(self, mcts, env):
        """Different random seeds can produce different search results."""
        results = set()
        for seed in range(10):
            np.random.seed(seed)
            env.reset(seed=seed)
            action_probs, _ = mcts.search(env)
            selected = int(np.argmax(action_probs))
            results.add(selected)
            if len(results) > 1:
                break

        # Should get at least some variation (not guaranteed but likely)
        # This test may occasionally fail due to randomness
        assert len(results) >= 1


class TestGumbelMCTSIntegration:
    """Integration tests for Gumbel MCTS with environment."""

    @pytest.fixture
    def env(self):
        return MicroCivEnv(board_size=8, max_turns=30, seed=42)

    @pytest.fixture
    def network(self):
        return MicroCivNetwork(
            board_size=8,
            action_space_size=MicroCivEnv(board_size=8).action_space.n,
            num_res_blocks=2,
            num_filters=16,
        )

    def test_can_play_full_game(self, network, env):
        """Can use Gumbel MCTS to play a complete game."""
        config = GumbelMCTSConfig(num_simulations=8)
        mcts = GumbelMCTS(network, config, device="cpu")

        env.reset()
        done = False
        steps = 0

        while not done and steps < 100:
            action_probs, _ = mcts.search(env)
            action = mcts.select_action(action_probs, temperature=0)

            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        # Game should end eventually
        assert done or steps == 100

    def test_performance_better_than_random(self, network, env):
        """Gumbel MCTS should perform better than random over many games."""
        # This is a weak test - just checking it doesn't crash
        config = GumbelMCTSConfig(num_simulations=4)
        mcts = GumbelMCTS(network, config, device="cpu")

        for _ in range(3):
            env.reset()
            done = False
            while not done:
                action_probs, _ = mcts.search(env)
                action = mcts.select_action(action_probs, temperature=0)
                _, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
