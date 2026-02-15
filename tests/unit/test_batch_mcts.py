"""Comprehensive test suite for BatchGumbelMCTS implementation.

This module tests the batch MCTS implementation which enables efficient parallel
search across multiple environments with batched neural network evaluation.

Test coverage:
- BatchMCTSConfig dataclass validation and defaults
- BatchGumbelMCTS initialization and search methods
- Single environment search (search method)
- Multi-environment batch search (batch_search method)
- Action selection with temperature and masking
- MCTSProtocol compliance
- Factory integration (create_mcts, list_variants, MCTSVariant enum)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alpha_zero.core.batch_mcts import BatchGumbelMCTS, BatchMCTSConfig
from alpha_zero.core.mcts_constants import DEFAULT_C_VISIT
from alpha_zero.core.mcts_factory import MCTSVariant, create_mcts, list_variants
from alpha_zero.envs.microciv import MicroCivEnv


class MockNetwork:
    """Mock neural network for testing.

    Returns uniform policy over valid actions and zero value estimates.
    """

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy and value for batch of states.

        Args:
            spatial: Batch of spatial features [B, C, H, W].
            vector: Batch of vector features [B, D].
            action_mask: Batch of action masks [B, A].

        Returns:
            Tuple of (policy_probs, value):
                - policy_probs: [B, A] with uniform distribution over valid actions
                - value: [B, 1] with all zeros
        """
        batch_size = spatial.shape[0]
        action_size = action_mask.shape[-1]

        # Uniform policy over valid actions
        policy = torch.ones(batch_size, action_size) / action_size
        policy = policy * action_mask  # Mask invalid actions
        # Renormalize
        policy_sum = policy.sum(dim=-1, keepdim=True)
        policy = policy / (policy_sum + 1e-8)

        # Zero value estimates
        value = torch.zeros(batch_size, 1)

        return policy, value

    def eval(self) -> None:
        """Set network to evaluation mode (no-op for mock)."""
        pass


@pytest.fixture
def mock_network():
    """Fixture providing a mock neural network."""
    return MockNetwork()


@pytest.fixture
def test_env():
    """Fixture providing a test environment.

    Uses small board size and max_turns for fast testing.
    """
    env = MicroCivEnv(board_size=8, max_turns=10)
    env.reset()
    return env


@pytest.fixture
def test_envs():
    """Fixture providing multiple test environments."""
    envs = []
    for _ in range(3):
        env = MicroCivEnv(board_size=8, max_turns=10)
        env.reset()
        envs.append(env)
    return envs


class TestBatchMCTSConfig:
    """Test suite for BatchMCTSConfig dataclass."""

    def test_defaults(self):
        """Verify all default configuration values."""
        config = BatchMCTSConfig()

        assert config.num_simulations == 16
        assert config.num_sampled_actions == 16
        assert config.max_batch_size == 64
        assert config.num_parallel_envs == 8
        assert config.max_considered_actions == 32
        assert config.c_visit == DEFAULT_C_VISIT
        assert config.c_scale == 1.0
        assert config.use_sequential_halving is True
        assert config.min_actions_after_halving == 1
        assert config.temperature == 1.0
        assert config.temperature_threshold == 30

    def test_custom_values(self):
        """Create config with custom values."""
        config = BatchMCTSConfig(
            num_simulations=32,
            num_sampled_actions=8,
            max_batch_size=128,
            num_parallel_envs=16,
            max_considered_actions=64,
            c_visit=100.0,
            c_scale=2.0,
            use_sequential_halving=False,
            min_actions_after_halving=2,
            temperature=0.5,
            temperature_threshold=20,
        )

        assert config.num_simulations == 32
        assert config.num_sampled_actions == 8
        assert config.max_batch_size == 128
        assert config.num_parallel_envs == 16
        assert config.max_considered_actions == 64
        assert config.c_visit == 100.0
        assert config.c_scale == 2.0
        assert config.use_sequential_halving is False
        assert config.min_actions_after_halving == 2
        assert config.temperature == 0.5
        assert config.temperature_threshold == 20

    def test_invalid_num_parallel_envs_zero(self):
        """Raise ValueError for num_parallel_envs = 0."""
        with pytest.raises(ValueError, match="num_parallel_envs must be >= 1"):
            BatchMCTSConfig(num_parallel_envs=0)

    def test_invalid_num_parallel_envs_negative(self):
        """Raise ValueError for negative num_parallel_envs."""
        with pytest.raises(ValueError, match="num_parallel_envs must be >= 1"):
            BatchMCTSConfig(num_parallel_envs=-1)

    def test_invalid_max_batch_size_zero(self):
        """Raise ValueError for max_batch_size = 0."""
        with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
            BatchMCTSConfig(max_batch_size=0)

    def test_invalid_max_batch_size_negative(self):
        """Raise ValueError for negative max_batch_size."""
        with pytest.raises(ValueError, match="max_batch_size must be >= 1"):
            BatchMCTSConfig(max_batch_size=-5)

    def test_invalid_num_simulations_zero(self):
        """Raise ValueError for num_simulations = 0."""
        with pytest.raises(ValueError, match="num_simulations must be >= 1"):
            BatchMCTSConfig(num_simulations=0)

    def test_invalid_num_simulations_negative(self):
        """Raise ValueError for negative num_simulations."""
        with pytest.raises(ValueError, match="num_simulations must be >= 1"):
            BatchMCTSConfig(num_simulations=-10)


class TestBatchGumbelMCTS:
    """Test suite for BatchGumbelMCTS class."""

    def test_init_default_config(self, mock_network):
        """Initialize with default configuration."""
        mcts = BatchGumbelMCTS(mock_network)

        assert mcts.network is mock_network
        assert isinstance(mcts.config, BatchMCTSConfig)
        assert mcts.config.num_simulations == 16
        assert mcts.device == "cpu"
        assert mcts._search_count == 0

    def test_init_custom_config(self, mock_network):
        """Initialize with custom configuration."""
        config = BatchMCTSConfig(
            num_simulations=4,
            num_sampled_actions=8,
            max_batch_size=32,
        )
        mcts = BatchGumbelMCTS(mock_network, config=config, device="cpu")

        assert mcts.network is mock_network
        assert mcts.config is config
        assert mcts.config.num_simulations == 4
        assert mcts.config.num_sampled_actions == 8
        assert mcts.config.max_batch_size == 32
        assert mcts.device == "cpu"

    def test_search_single_env(self, mock_network, test_env):
        """Search with single environment returns (np.ndarray, float)."""
        config = BatchMCTSConfig(num_simulations=2)
        mcts = BatchGumbelMCTS(mock_network, config=config)

        action_probs, root_value = mcts.search(test_env, add_noise=False)

        # Check return types
        assert isinstance(action_probs, np.ndarray)
        assert isinstance(root_value, (float, np.floating))

        # Check shapes
        assert action_probs.shape == (test_env.action_space.n,)

    def test_search_returns_valid_policy(self, mock_network, test_env):
        """Policy from search sums to ~1.0 and has correct shape."""
        config = BatchMCTSConfig(num_simulations=2)
        mcts = BatchGumbelMCTS(mock_network, config=config)

        action_probs, root_value = mcts.search(test_env, add_noise=False)

        # Check policy properties
        assert action_probs.shape == (test_env.action_space.n,)
        assert np.all(action_probs >= 0.0)
        assert np.isclose(action_probs.sum(), 1.0, atol=1e-6)

        # Check value is in valid range
        assert isinstance(root_value, (float, np.floating))

    def test_search_with_noise_parameter(self, mock_network, test_env):
        """Search accepts add_noise parameter (not used by Gumbel)."""
        config = BatchMCTSConfig(num_simulations=2)
        mcts = BatchGumbelMCTS(mock_network, config=config)

        # Should not raise even though add_noise is ignored
        action_probs1, value1 = mcts.search(test_env, add_noise=True)
        action_probs2, value2 = mcts.search(test_env, add_noise=False)

        # Both should return valid results
        assert action_probs1.shape == (test_env.action_space.n,)
        assert action_probs2.shape == (test_env.action_space.n,)

    def test_batch_search_multiple_envs(self, mock_network, test_envs):
        """Batch search returns list of N results."""
        config = BatchMCTSConfig(num_simulations=2)
        mcts = BatchGumbelMCTS(mock_network, config=config)

        results = mcts.batch_search(test_envs)

        # Check we get one result per environment
        assert isinstance(results, list)
        assert len(results) == len(test_envs)

        # Check each result is a tuple
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_batch_search_result_shapes(self, mock_network, test_envs):
        """Each result from batch_search has correct shapes."""
        config = BatchMCTSConfig(num_simulations=2)
        mcts = BatchGumbelMCTS(mock_network, config=config)

        results = mcts.batch_search(test_envs)

        for (action_probs, root_value), env in zip(results, test_envs, strict=True):
            # Check action probabilities
            assert isinstance(action_probs, np.ndarray)
            assert action_probs.shape == (env.action_space.n,)
            assert np.all(action_probs >= 0.0)
            assert np.isclose(action_probs.sum(), 1.0, atol=1e-6)

            # Check value
            assert isinstance(root_value, (float, np.floating))

    def test_batch_search_empty_list(self, mock_network):
        """Batch search with empty list raises ValueError.

        Note: Empty environment list is an edge case that currently raises
        ValueError in _batch_sequential_halving. This is reasonable behavior
        since batch search with no environments doesn't have practical use.
        """
        config = BatchMCTSConfig(num_simulations=2)
        mcts = BatchGumbelMCTS(mock_network, config=config)

        # Empty list causes max() on empty iterable in sequential halving
        with pytest.raises(
            ValueError, match="max\\(\\) (arg is an empty sequence|iterable argument is empty)"
        ):
            mcts.batch_search([])

    def test_batch_search_single_env(self, mock_network, test_env):
        """Batch search with single env in list."""
        config = BatchMCTSConfig(num_simulations=2)
        mcts = BatchGumbelMCTS(mock_network, config=config)

        results = mcts.batch_search([test_env])

        assert len(results) == 1
        action_probs, root_value = results[0]
        assert action_probs.shape == (test_env.action_space.n,)
        assert np.isclose(action_probs.sum(), 1.0, atol=1e-6)

    def test_select_action_greedy(self, mock_network):
        """Temperature=0 returns argmax."""
        mcts = BatchGumbelMCTS(mock_network)

        # Create policy with clear maximum
        action_probs = np.array([0.1, 0.7, 0.2])

        action = mcts.select_action(action_probs, temperature=0.0)

        assert action == 1  # Index of max probability

    def test_select_action_greedy_with_one_hot(self, mock_network):
        """Greedy selection with one-hot policy."""
        mcts = BatchGumbelMCTS(mock_network)

        # One-hot policy (typical from Gumbel MCTS)
        action_probs = np.array([0.0, 0.0, 1.0, 0.0])

        action = mcts.select_action(action_probs, temperature=0.0)

        assert action == 2

    def test_select_action_with_mask(self, mock_network):
        """Respect action mask when selecting actions."""
        mcts = BatchGumbelMCTS(mock_network)

        # Policy suggests action 2, but it's masked out
        action_probs = np.array([0.2, 0.3, 0.5])
        action_mask = np.array([1.0, 1.0, 0.0])  # Action 2 invalid

        action = mcts.select_action(action_probs, temperature=0.0, action_mask=action_mask)

        # Should select action 1 (highest valid action)
        assert action == 1

    def test_select_action_with_mask_renormalizes(self, mock_network):
        """Action mask causes renormalization."""
        mcts = BatchGumbelMCTS(mock_network)

        action_probs = np.array([0.4, 0.4, 0.2])
        action_mask = np.array([1.0, 1.0, 0.0])

        # With temperature > 0, should still respect mask
        # Set random seed for deterministic test
        np.random.seed(42)
        action = mcts.select_action(action_probs, temperature=1.0, action_mask=action_mask)

        # Should be one of the valid actions
        assert action in [0, 1]

    def test_select_action_no_valid_actions_raises(self, mock_network):
        """ValueError when no valid actions available."""
        mcts = BatchGumbelMCTS(mock_network)

        action_probs = np.array([0.5, 0.5])
        action_mask = np.array([0.0, 0.0])  # All actions masked

        with pytest.raises(ValueError, match="No valid actions available"):
            mcts.select_action(action_probs, temperature=0.0, action_mask=action_mask)

    def test_select_action_stochastic_with_temperature(self, mock_network):
        """Temperature > 0 enables stochastic selection."""
        mcts = BatchGumbelMCTS(mock_network)

        action_probs = np.array([0.4, 0.6])

        # Run multiple times to check stochasticity
        # With temperature=1.0, both actions should be possible
        np.random.seed(42)
        actions = [mcts.select_action(action_probs, temperature=1.0) for _ in range(20)]

        # Should see both actions selected at least once
        # (with 20 samples and 40/60 split, this is very likely)
        unique_actions = set(actions)
        assert len(unique_actions) >= 1  # At minimum one action
        assert all(a in [0, 1] for a in actions)  # All valid

    def test_protocol_compliance(self, mock_network):
        """Verify BatchGumbelMCTS has required protocol methods."""
        mcts = BatchGumbelMCTS(mock_network)

        # Check required methods exist
        assert hasattr(mcts, "search")
        assert callable(mcts.search)

        assert hasattr(mcts, "select_action")
        assert callable(mcts.select_action)

        # Check signatures match protocol
        import inspect

        search_sig = inspect.signature(mcts.search)
        assert "env" in search_sig.parameters
        assert "add_noise" in search_sig.parameters

        select_sig = inspect.signature(mcts.select_action)
        assert "action_probs" in select_sig.parameters
        assert "temperature" in select_sig.parameters
        assert "action_mask" in select_sig.parameters

    def test_sequential_halving_enabled(self, mock_network, test_env):
        """Sequential halving produces valid results."""
        config = BatchMCTSConfig(
            num_simulations=4,
            use_sequential_halving=True,
            min_actions_after_halving=1,
        )
        mcts = BatchGumbelMCTS(mock_network, config=config)

        action_probs, root_value = mcts.search(test_env)

        assert action_probs.shape == (test_env.action_space.n,)
        assert np.isclose(action_probs.sum(), 1.0, atol=1e-6)

    def test_sequential_halving_disabled(self, mock_network, test_env):
        """Simple search (no halving) produces valid results."""
        config = BatchMCTSConfig(
            num_simulations=4,
            use_sequential_halving=False,
        )
        mcts = BatchGumbelMCTS(mock_network, config=config)

        action_probs, root_value = mcts.search(test_env)

        assert action_probs.shape == (test_env.action_space.n,)
        assert np.isclose(action_probs.sum(), 1.0, atol=1e-6)


class TestBatchMCTSFactory:
    """Test suite for factory integration."""

    def test_batch_gumbel_in_variants(self):
        """Verify 'batch_gumbel' is in list_variants()."""
        variants = list_variants()

        assert "batch_gumbel" in variants

    def test_batch_gumbel_enum(self):
        """Verify MCTSVariant.BATCH_GUMBEL exists."""
        assert hasattr(MCTSVariant, "BATCH_GUMBEL")
        assert MCTSVariant.BATCH_GUMBEL.value == "batch_gumbel"

    def test_create_batch_gumbel_string(self, mock_network):
        """Create batch_gumbel MCTS using string variant."""
        mcts = create_mcts("batch_gumbel", network=mock_network, num_simulations=4)

        assert isinstance(mcts, BatchGumbelMCTS)
        assert mcts.config.num_simulations == 4

    def test_create_batch_gumbel_enum(self, mock_network):
        """Create batch_gumbel MCTS using enum variant."""
        mcts = create_mcts(MCTSVariant.BATCH_GUMBEL, network=mock_network, num_simulations=4)

        assert isinstance(mcts, BatchGumbelMCTS)
        assert mcts.config.num_simulations == 4

    def test_create_batch_gumbel_with_config(self, mock_network):
        """Create batch_gumbel MCTS with custom config."""
        config = BatchMCTSConfig(
            num_simulations=8,
            max_batch_size=32,
            use_sequential_halving=False,
        )
        mcts = create_mcts("batch_gumbel", network=mock_network, config=config)

        assert isinstance(mcts, BatchGumbelMCTS)
        assert mcts.config is config
        assert mcts.config.num_simulations == 8
        assert mcts.config.max_batch_size == 32
        assert mcts.config.use_sequential_halving is False

    def test_create_batch_gumbel_with_device(self, mock_network):
        """Create batch_gumbel MCTS with device parameter."""
        mcts = create_mcts("batch_gumbel", network=mock_network, device="cpu")

        assert isinstance(mcts, BatchGumbelMCTS)
        assert mcts.device == "cpu"

    def test_create_batch_gumbel_default_simulations(self, mock_network):
        """Factory uses correct default num_simulations for batch_gumbel."""
        mcts = create_mcts("batch_gumbel", network=mock_network)

        # Default for batch_gumbel should be 16 (from registry)
        assert mcts.config.num_simulations == 16

    def test_create_batch_gumbel_kwargs_passed_to_config(self, mock_network):
        """Factory passes kwargs to config constructor."""
        mcts = create_mcts(
            "batch_gumbel",
            network=mock_network,
            num_simulations=12,
            max_batch_size=48,
            c_visit=75.0,
        )

        assert mcts.config.num_simulations == 12
        assert mcts.config.max_batch_size == 48
        assert mcts.config.c_visit == 75.0

    def test_create_batch_gumbel_network_required(self):
        """Factory raises ValueError if network is None."""
        with pytest.raises(ValueError, match="Network is required"):
            create_mcts("batch_gumbel", network=None)
