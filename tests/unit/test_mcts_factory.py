"""Unit tests for the MCTS factory module."""

from __future__ import annotations

import pytest

from alpha_zero.core.civ6_mcts import (
    Civ6MCTS,
    GumbelMCTS,
    MuZeroMCTS,
)
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.core.mcts_factory import create_mcts, get_available_mcts_types
from alpha_zero.envs.microciv import MicroCivEnv

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def network() -> MicroCivNetwork:
    """Create a small test network."""
    return MicroCivNetwork(
        board_size=8,
        action_space_size=376,
        num_res_blocks=2,
        num_filters=16,
    )


@pytest.fixture
def env() -> MicroCivEnv:
    """Create a test environment."""
    return MicroCivEnv(board_size=8, max_turns=5)


# =============================================================================
# get_available_mcts_types Tests
# =============================================================================


class TestGetAvailableTypes:
    """Tests for get_available_mcts_types."""

    def test_returns_list(self) -> None:
        """Returns a list of strings."""
        types = get_available_mcts_types()
        assert isinstance(types, list)
        assert all(isinstance(t, str) for t in types)

    def test_contains_known_types(self) -> None:
        """Contains all expected MCTS variants."""
        types = get_available_mcts_types()
        assert "civ6" in types
        assert "gumbel" in types
        assert "muzero" in types

    def test_returns_at_least_three(self) -> None:
        """Returns at least 3 types."""
        assert len(get_available_mcts_types()) >= 3


# =============================================================================
# create_mcts Tests - Type Dispatch
# =============================================================================


class TestCreateMCTSTypeDispatch:
    """Tests for correct class instantiation by type name."""

    def test_creates_civ6_mcts(self, network: MicroCivNetwork) -> None:
        """'civ6' creates a Civ6MCTS instance."""
        mcts = create_mcts("civ6", network=network)
        assert isinstance(mcts, Civ6MCTS)

    def test_creates_gumbel_mcts(self, network: MicroCivNetwork) -> None:
        """'gumbel' creates a GumbelMCTS instance."""
        mcts = create_mcts("gumbel", network=network)
        assert isinstance(mcts, GumbelMCTS)

    def test_creates_muzero_mcts(self, network: MicroCivNetwork) -> None:
        """'muzero' creates a MuZeroMCTS instance."""
        mcts = create_mcts("muzero", network=network)
        assert isinstance(mcts, MuZeroMCTS)

    def test_default_is_civ6(self, network: MicroCivNetwork) -> None:
        """Default mcts_type is 'civ6'."""
        mcts = create_mcts(network=network)
        assert isinstance(mcts, Civ6MCTS)

    def test_unknown_type_raises_value_error(self, network: MicroCivNetwork) -> None:
        """Unknown type raises ValueError with helpful message."""
        with pytest.raises(ValueError, match="Unknown MCTS type"):
            create_mcts("invalid_type", network=network)

    def test_error_message_lists_available_types(self, network: MicroCivNetwork) -> None:
        """ValueError message lists all available types."""
        with pytest.raises(ValueError, match="civ6.*gumbel.*muzero"):
            create_mcts("bad", network=network)


# =============================================================================
# create_mcts Tests - Configuration
# =============================================================================


class TestCreateMCTSConfig:
    """Tests for config passthrough and defaults."""

    def test_default_simulations_civ6(self, network: MicroCivNetwork) -> None:
        """civ6 defaults to 200 simulations."""
        mcts = create_mcts("civ6", network=network)
        assert mcts.config.num_simulations == 200

    def test_default_simulations_gumbel(self, network: MicroCivNetwork) -> None:
        """gumbel defaults to 16 simulations."""
        mcts = create_mcts("gumbel", network=network)
        assert mcts.config.num_simulations == 16

    def test_override_simulations(self, network: MicroCivNetwork) -> None:
        """num_simulations override is respected."""
        mcts = create_mcts("civ6", network=network, num_simulations=50)
        assert mcts.config.num_simulations == 50

    def test_gumbel_extra_kwargs(self, network: MicroCivNetwork) -> None:
        """Gumbel-specific kwargs are passed to config."""
        mcts = create_mcts(
            "gumbel",
            network=network,
            num_sampled_actions=8,
        )
        assert isinstance(mcts, GumbelMCTS)
        assert mcts.config.num_sampled_actions == 8

    def test_unknown_kwargs_ignored(self, network: MicroCivNetwork) -> None:
        """Unknown kwargs are silently ignored (no crash)."""
        mcts = create_mcts(
            "civ6",
            network=network,
            totally_fake_kwarg=42,
        )
        assert isinstance(mcts, Civ6MCTS)

    def test_device_string(self, network: MicroCivNetwork) -> None:
        """Device string is passed correctly."""
        mcts = create_mcts("civ6", network=network, device="cpu")
        assert isinstance(mcts, Civ6MCTS)


# =============================================================================
# create_mcts Tests - Functional (search + select_action)
# =============================================================================


class TestCreateMCTSFunctional:
    """Tests that factory-created instances work end-to-end."""

    def test_civ6_search_returns_valid_output(
        self, network: MicroCivNetwork, env: MicroCivEnv
    ) -> None:
        """Civ6MCTS search returns valid action probs and value."""
        mcts = create_mcts("civ6", network=network, num_simulations=4)
        env.reset()
        action_probs, root_value = mcts.search(env, add_noise=False)

        assert action_probs.shape == (376,)
        assert abs(action_probs.sum() - 1.0) < 0.01
        assert isinstance(root_value, float)

    def test_gumbel_search_returns_valid_output(
        self, network: MicroCivNetwork, env: MicroCivEnv
    ) -> None:
        """GumbelMCTS search returns valid action probs and value."""
        mcts = create_mcts("gumbel", network=network, num_simulations=4)
        env.reset()
        action_probs, root_value = mcts.search(env, add_noise=False)

        assert action_probs.shape == (376,)
        assert abs(action_probs.sum() - 1.0) < 0.01
        assert isinstance(root_value, float)

    def test_select_action_returns_valid_int(
        self, network: MicroCivNetwork, env: MicroCivEnv
    ) -> None:
        """select_action returns a valid integer action."""
        mcts = create_mcts("gumbel", network=network, num_simulations=4)
        env.reset()
        action_probs, _ = mcts.search(env, add_noise=False)
        action = mcts.select_action(action_probs, temperature=1.0)

        assert isinstance(action, (int,))
        assert 0 <= action < 376

    def test_greedy_select_action(self, network: MicroCivNetwork, env: MicroCivEnv) -> None:
        """temperature=0 selects the highest probability action."""
        mcts = create_mcts("civ6", network=network, num_simulations=4)
        env.reset()
        action_probs, _ = mcts.search(env, add_noise=False)
        action = mcts.select_action(action_probs, temperature=0.0)

        # Should pick the argmax action
        import numpy as np

        assert action == int(np.argmax(action_probs))
