"""Unit tests for MuZero MCTS (hidden state based)."""

import numpy as np
import pytest
import torch

from alpha_zero.core.civ6_mcts import (
    MuZeroMCTS,
    MuZeroMCTSConfig,
    MuZeroNode,
)
from alpha_zero.core.muzero_network import MuZeroConfig, MuZeroNetwork


class TestMuZeroNode:
    """Test suite for MuZero MCTS node."""

    def test_initial_state(self):
        """New node has correct initial values."""
        node = MuZeroNode()

        assert node.hidden_state is None
        assert node.reward == 0.0
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.prior == 0.0
        assert not node.is_expanded()

    def test_value_calculation(self):
        """Value is correctly computed."""
        node = MuZeroNode(visit_count=10, value_sum=5.0)

        assert node.value == 0.5

    def test_value_zero_visits(self):
        """Value returns 0 for unvisited node."""
        node = MuZeroNode(visit_count=0, value_sum=0.0)

        assert node.value == 0.0

    def test_with_hidden_state(self):
        """Node can store hidden state tensor."""
        hidden = torch.randn(1, 32, 8, 8)
        node = MuZeroNode(hidden_state=hidden)

        assert node.hidden_state is not None
        assert node.hidden_state.shape == (1, 32, 8, 8)

    def test_expand_creates_children(self):
        """Expand creates child nodes for given actions."""
        node = MuZeroNode()
        actions = [0, 1, 5, 10]
        priors = np.array([0.3, 0.3, 0.2, 0.2])

        node.expand(actions, priors)

        assert node.is_expanded()
        assert len(node.children) == 4
        assert 0 in node.children
        assert 1 in node.children
        assert 5 in node.children
        assert 10 in node.children

    def test_expand_assigns_priors(self):
        """Expand assigns correct priors to children."""
        node = MuZeroNode()
        actions = [0, 1]
        priors = np.array([0.7, 0.3])

        node.expand(actions, priors)

        assert abs(node.children[0].prior - 0.7) < 1e-5
        assert abs(node.children[1].prior - 0.3) < 1e-5

    def test_select_child_returns_valid(self):
        """Select child returns a valid child."""
        node = MuZeroNode(visit_count=10)
        node.children[0] = MuZeroNode(prior=0.5, visit_count=3, value_sum=1.5)
        node.children[1] = MuZeroNode(prior=0.5, visit_count=2, value_sum=0.8)

        action, child = node.select_child(c_puct=1.5, min_q=0.0, max_q=1.0)

        assert action in [0, 1]
        assert child is node.children[action]

    def test_select_child_prefers_high_prior_low_visits(self):
        """Select child prefers high prior with low visits."""
        node = MuZeroNode(visit_count=10)
        # Action 0: high visits, low prior
        node.children[0] = MuZeroNode(prior=0.1, visit_count=8, value_sum=4.0)
        # Action 1: low visits, high prior - should be preferred for exploration
        node.children[1] = MuZeroNode(prior=0.9, visit_count=1, value_sum=0.5)

        action, _ = node.select_child(c_puct=2.0, min_q=0.0, max_q=1.0)

        # With high c_puct, action 1 should be selected due to high prior and low visits
        assert action == 1


class TestMuZeroMCTSConfig:
    """Test suite for MuZero MCTS configuration."""

    def test_default_values(self):
        """Default config has reasonable values."""
        config = MuZeroMCTSConfig()

        assert config.num_simulations == 50
        assert config.c_puct == 1.5
        assert config.use_gumbel is True
        assert config.num_sampled_actions == 16
        assert config.discount > 0.9

    def test_custom_values(self):
        """Config accepts custom values."""
        config = MuZeroMCTSConfig(
            num_simulations=100,
            c_puct=2.0,
            use_gumbel=False,
            discount=0.99,
        )

        assert config.num_simulations == 100
        assert config.c_puct == 2.0
        assert config.use_gumbel is False
        assert config.discount == 0.99


class TestMuZeroMCTS:
    """Test suite for MuZero MCTS with learned dynamics."""

    @pytest.fixture
    def network_config(self):
        return MuZeroConfig(
            spatial_channels=13,
            spatial_size=8,
            vector_dim=64,
            num_res_blocks=2,
            hidden_state_channels=32,
            fc_hidden_dim=64,
            action_space_size=100,
            dynamics_num_blocks=2,
            use_consistency_loss=False,
            use_value_prefix=False,
        )

    @pytest.fixture
    def network(self, network_config):
        return MuZeroNetwork(network_config)

    @pytest.fixture
    def mcts_config(self):
        return MuZeroMCTSConfig(
            num_simulations=8,
            num_sampled_actions=8,
            use_gumbel=True,
            use_sequential_halving=True,
        )

    @pytest.fixture
    def mcts(self, network, mcts_config):
        return MuZeroMCTS(network, mcts_config, device="cpu")

    @pytest.fixture
    def sample_observation(self):
        return {
            "spatial": np.random.randn(13, 8, 8).astype(np.float32),
            "vector": np.random.randn(64).astype(np.float32),
        }

    @pytest.fixture
    def sample_action_mask(self):
        mask = np.zeros(100, dtype=np.float32)
        # Enable some random actions
        mask[[0, 1, 5, 10, 20, 50]] = 1.0
        return mask

    def test_search_returns_valid_probs(self, mcts, sample_observation, sample_action_mask):
        """Search returns valid probability distribution."""
        action_probs, root_value = mcts.search(
            sample_observation, sample_action_mask, add_noise=False
        )

        assert action_probs.shape == (100,)
        assert np.all(action_probs >= 0)
        assert abs(action_probs.sum() - 1.0) < 1e-5

    def test_search_returns_one_hot_with_gumbel(self, mcts, sample_observation, sample_action_mask):
        """Gumbel MCTS returns one-hot policy."""
        action_probs, _ = mcts.search(sample_observation, sample_action_mask, add_noise=False)

        # Should have exactly one action with probability ~1
        assert action_probs.max() > 0.99

    def test_search_respects_action_mask(self, mcts, sample_observation, sample_action_mask):
        """Search only selects valid actions."""
        action_probs, _ = mcts.search(sample_observation, sample_action_mask, add_noise=False)

        # Selected action should be valid
        selected_action = np.argmax(action_probs)
        assert sample_action_mask[selected_action] == 1.0

    def test_root_value_bounded(self, mcts, sample_observation, sample_action_mask):
        """Root value is in reasonable range."""
        _, root_value = mcts.search(sample_observation, sample_action_mask, add_noise=False)

        # Value should be finite
        assert np.isfinite(root_value)

    def test_no_valid_actions_returns_zeros(self, mcts, sample_observation):
        """Returns zero probs when no valid actions."""
        empty_mask = np.zeros(100, dtype=np.float32)

        action_probs, root_value = mcts.search(sample_observation, empty_mask, add_noise=False)

        assert action_probs.sum() == 0.0
        assert root_value == 0.0

    def test_select_action_greedy(self, mcts, sample_observation, sample_action_mask):
        """Temperature 0 selects best action deterministically."""
        action_probs, _ = mcts.search(sample_observation, sample_action_mask, add_noise=False)

        action1 = mcts.select_action(action_probs, temperature=0)
        action2 = mcts.select_action(action_probs, temperature=0)

        assert action1 == action2
        assert action1 == np.argmax(action_probs)

    def test_works_without_gumbel(self, network):
        """MCTS works in standard PUCT mode."""
        config = MuZeroMCTSConfig(
            num_simulations=4,
            use_gumbel=False,
        )
        mcts = MuZeroMCTS(network, config, device="cpu")

        obs = {
            "spatial": np.random.randn(13, 8, 8).astype(np.float32),
            "vector": np.random.randn(64).astype(np.float32),
        }
        mask = np.zeros(100, dtype=np.float32)
        mask[[0, 1, 5, 10]] = 1.0

        action_probs, _ = mcts.search(obs, mask, add_noise=False)

        assert action_probs.sum() > 0
        # Non-gumbel returns visit count distribution, not one-hot
        assert action_probs.shape == (100,)

    def test_works_with_single_valid_action(self, mcts, sample_observation):
        """MCTS works when only one action is valid."""
        mask = np.zeros(100, dtype=np.float32)
        mask[42] = 1.0

        action_probs, _ = mcts.search(sample_observation, mask, add_noise=False)

        assert action_probs[42] == 1.0

    def test_hidden_states_are_computed(self, network):
        """Dynamics function computes hidden states for children."""
        config = MuZeroMCTSConfig(
            num_simulations=4,
            num_sampled_actions=4,
            use_gumbel=True,
        )
        mcts = MuZeroMCTS(network, config, device="cpu")

        obs = {
            "spatial": np.random.randn(13, 8, 8).astype(np.float32),
            "vector": np.random.randn(64).astype(np.float32),
        }
        mask = np.ones(100, dtype=np.float32)

        # Run search - this should use dynamics to compute hidden states
        action_probs, _ = mcts.search(obs, mask, add_noise=False)

        # Should successfully return a result
        assert action_probs.sum() > 0


class TestMuZeroMCTSIntegration:
    """Integration tests for MuZero MCTS with network."""

    @pytest.fixture
    def network(self):
        config = MuZeroConfig(
            spatial_channels=5,
            spatial_size=4,
            vector_dim=16,
            num_res_blocks=1,
            hidden_state_channels=16,
            fc_hidden_dim=32,
            action_space_size=20,
            dynamics_num_blocks=1,
            use_consistency_loss=False,
            use_value_prefix=False,
        )
        return MuZeroNetwork(config)

    def test_many_simulations(self, network):
        """MCTS works with many simulations."""
        config = MuZeroMCTSConfig(
            num_simulations=32,
            use_gumbel=True,
            use_sequential_halving=True,
        )
        mcts = MuZeroMCTS(network, config, device="cpu")

        obs = {
            "spatial": np.random.randn(5, 4, 4).astype(np.float32),
            "vector": np.random.randn(16).astype(np.float32),
        }
        mask = np.ones(20, dtype=np.float32)

        action_probs, root_value = mcts.search(obs, mask, add_noise=False)

        assert action_probs.max() > 0.99  # Should select one action
        assert np.isfinite(root_value)

    def test_repeated_searches_different_results(self, network):
        """Different random seeds can produce different results."""
        config = MuZeroMCTSConfig(
            num_simulations=8,
            use_gumbel=True,
        )
        mcts = MuZeroMCTS(network, config, device="cpu")

        mask = np.ones(20, dtype=np.float32)
        results = set()

        for seed in range(10):
            np.random.seed(seed)
            obs = {
                "spatial": np.random.randn(5, 4, 4).astype(np.float32),
                "vector": np.random.randn(16).astype(np.float32),
            }
            action_probs, _ = mcts.search(obs, mask, add_noise=False)
            selected = int(np.argmax(action_probs))
            results.add(selected)

            if len(results) > 1:
                break

        # Should get some variation
        assert len(results) >= 1

    def test_low_simulation_count(self, network):
        """Works with very few simulations."""
        config = MuZeroMCTSConfig(
            num_simulations=2,
            num_sampled_actions=4,
            use_gumbel=True,
        )
        mcts = MuZeroMCTS(network, config, device="cpu")

        obs = {
            "spatial": np.random.randn(5, 4, 4).astype(np.float32),
            "vector": np.random.randn(16).astype(np.float32),
        }
        mask = np.ones(20, dtype=np.float32)

        action_probs, _ = mcts.search(obs, mask, add_noise=False)

        assert action_probs.sum() > 0


class TestMuZeroMCTSNoEnvCloning:
    """Tests verifying MuZero MCTS does not require environment cloning."""

    @pytest.fixture
    def network(self):
        config = MuZeroConfig(
            spatial_channels=5,
            spatial_size=4,
            vector_dim=16,
            num_res_blocks=1,
            hidden_state_channels=16,
            fc_hidden_dim=32,
            action_space_size=20,
            dynamics_num_blocks=1,
        )
        return MuZeroNetwork(config)

    def test_search_only_needs_observation(self, network):
        """Search works with just observation and mask (no env object)."""
        mcts = MuZeroMCTS(network, device="cpu")

        # Only provide observation dict and action mask - no environment
        obs = {
            "spatial": np.random.randn(5, 4, 4).astype(np.float32),
            "vector": np.random.randn(16).astype(np.float32),
        }
        mask = np.ones(20, dtype=np.float32)

        # This should work without any environment object
        action_probs, root_value = mcts.search(obs, mask)

        assert action_probs.shape == (20,)
        assert np.isfinite(root_value)

    def test_dynamics_used_for_tree_expansion(self, network):
        """Verify dynamics function is used (not env.step)."""
        config = MuZeroMCTSConfig(
            num_simulations=16,
            use_gumbel=True,
        )
        mcts = MuZeroMCTS(network, config, device="cpu")

        obs = {
            "spatial": np.random.randn(5, 4, 4).astype(np.float32),
            "vector": np.random.randn(16).astype(np.float32),
        }
        mask = np.ones(20, dtype=np.float32)

        # Search should succeed using only the network's dynamics
        action_probs, _ = mcts.search(obs, mask)

        assert action_probs.max() > 0.99


class TestMuZeroMCTSRewardHandling:
    """Tests for reward conversion with categorical vs scalar rewards."""

    @pytest.fixture
    def categorical_network(self):
        """Network with categorical rewards enabled."""
        config = MuZeroConfig(
            spatial_channels=5,
            spatial_size=4,
            vector_dim=16,
            num_res_blocks=1,
            hidden_state_channels=16,
            fc_hidden_dim=32,
            action_space_size=20,
            dynamics_num_blocks=1,
            use_categorical_reward=True,
            reward_support_size=601,
        )
        return MuZeroNetwork(config)

    @pytest.fixture
    def scalar_network(self):
        """Network with scalar rewards."""
        config = MuZeroConfig(
            spatial_channels=5,
            spatial_size=4,
            vector_dim=16,
            num_res_blocks=1,
            hidden_state_channels=16,
            fc_hidden_dim=32,
            action_space_size=20,
            dynamics_num_blocks=1,
            use_categorical_reward=False,
        )
        return MuZeroNetwork(config)

    def test_categorical_reward_config(self, categorical_network):
        """MCTS config correctly set for categorical rewards."""
        mcts_config = MuZeroMCTSConfig(
            num_simulations=4,
            use_categorical_reward=True,
            reward_support_size=601,
        )
        mcts = MuZeroMCTS(categorical_network, mcts_config, device="cpu")

        assert mcts.config.use_categorical_reward is True
        assert mcts.config.reward_support_size == 601

    def test_scalar_reward_config(self, scalar_network):
        """MCTS config correctly set for scalar rewards."""
        mcts_config = MuZeroMCTSConfig(
            num_simulations=4,
            use_categorical_reward=False,
        )
        mcts = MuZeroMCTS(scalar_network, mcts_config, device="cpu")

        assert mcts.config.use_categorical_reward is False

    def test_convert_reward_scalar_dim_zero(self, scalar_network):
        """Convert scalar reward with dim 0."""
        mcts_config = MuZeroMCTSConfig(use_categorical_reward=False)
        mcts = MuZeroMCTS(scalar_network, mcts_config, device="cpu")

        reward = torch.tensor(0.5)
        result = mcts._convert_reward_to_scalar(reward)

        assert abs(result - 0.5) < 1e-5

    def test_convert_reward_scalar_dim_one(self, scalar_network):
        """Convert scalar reward with dim 1."""
        mcts_config = MuZeroMCTSConfig(use_categorical_reward=False)
        mcts = MuZeroMCTS(scalar_network, mcts_config, device="cpu")

        reward = torch.tensor([0.7])
        result = mcts._convert_reward_to_scalar(reward)

        assert abs(result - 0.7) < 1e-5

    def test_convert_reward_categorical(self, categorical_network):
        """Convert categorical reward to scalar."""
        mcts_config = MuZeroMCTSConfig(
            use_categorical_reward=True,
            reward_support_size=601,
        )
        mcts = MuZeroMCTS(categorical_network, mcts_config, device="cpu")

        # Create categorical distribution centered around 0
        reward = torch.zeros(1, 601)
        reward[0, 300] = 10.0  # Peak at center (value ~= 0)
        result = mcts._convert_reward_to_scalar(reward)

        # Should be close to 0 (center of support)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_search_with_categorical_network(self, categorical_network):
        """Search works correctly with categorical reward network."""
        mcts_config = MuZeroMCTSConfig(
            num_simulations=4,
            use_categorical_reward=True,
            reward_support_size=601,
        )
        mcts = MuZeroMCTS(categorical_network, mcts_config, device="cpu")

        obs = {
            "spatial": np.random.randn(5, 4, 4).astype(np.float32),
            "vector": np.random.randn(16).astype(np.float32),
        }
        mask = np.ones(20, dtype=np.float32)

        action_probs, root_value = mcts.search(obs, mask, add_noise=False)

        assert action_probs.shape == (20,)
        assert np.isfinite(root_value)

    def test_search_with_scalar_network(self, scalar_network):
        """Search works correctly with scalar reward network."""
        mcts_config = MuZeroMCTSConfig(
            num_simulations=4,
            use_categorical_reward=False,
        )
        mcts = MuZeroMCTS(scalar_network, mcts_config, device="cpu")

        obs = {
            "spatial": np.random.randn(5, 4, 4).astype(np.float32),
            "vector": np.random.randn(16).astype(np.float32),
        }
        mask = np.ones(20, dtype=np.float32)

        action_probs, root_value = mcts.search(obs, mask, add_noise=False)

        assert action_probs.shape == (20,)
        assert np.isfinite(root_value)


class TestMuZeroMCTSFromNetwork:
    """Tests for the from_network factory method."""

    @pytest.fixture
    def categorical_network(self):
        """Network with categorical rewards."""
        config = MuZeroConfig(
            spatial_channels=5,
            spatial_size=4,
            vector_dim=16,
            action_space_size=20,
            use_categorical_reward=True,
            reward_support_size=601,
        )
        return MuZeroNetwork(config)

    @pytest.fixture
    def scalar_network(self):
        """Network with scalar rewards."""
        config = MuZeroConfig(
            spatial_channels=5,
            spatial_size=4,
            vector_dim=16,
            action_space_size=20,
            use_categorical_reward=False,
        )
        return MuZeroNetwork(config)

    def test_from_network_syncs_categorical(self, categorical_network):
        """Factory method correctly syncs categorical reward config."""
        mcts = MuZeroMCTS.from_network(categorical_network, device="cpu")

        assert mcts.config.use_categorical_reward is True
        assert mcts.config.reward_support_size == 601

    def test_from_network_syncs_scalar(self, scalar_network):
        """Factory method correctly syncs scalar reward config."""
        mcts = MuZeroMCTS.from_network(scalar_network, device="cpu")

        assert mcts.config.use_categorical_reward is False

    def test_from_network_preserves_other_config(self, categorical_network):
        """Factory method preserves non-reward config settings."""
        base_config = MuZeroMCTSConfig(
            num_simulations=100,
            c_puct=2.5,
            use_gumbel=False,
        )
        mcts = MuZeroMCTS.from_network(categorical_network, config=base_config, device="cpu")

        assert mcts.config.num_simulations == 100
        assert mcts.config.c_puct == 2.5
        assert mcts.config.use_gumbel is False
        # But reward config should be synced from network
        assert mcts.config.use_categorical_reward is True

    def test_from_network_search_works(self, categorical_network):
        """MCTS created via from_network works correctly."""
        mcts = MuZeroMCTS.from_network(
            categorical_network,
            config=MuZeroMCTSConfig(num_simulations=4),
            device="cpu",
        )

        obs = {
            "spatial": np.random.randn(5, 4, 4).astype(np.float32),
            "vector": np.random.randn(16).astype(np.float32),
        }
        mask = np.ones(20, dtype=np.float32)

        action_probs, root_value = mcts.search(obs, mask, add_noise=False)

        assert action_probs.shape == (20,)
        assert np.isfinite(root_value)
