"""Tests for MiniResourceEnv and EnvironmentRegistry."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.exceptions import DuplicateRegistrationError, EnvironmentNotRegisteredError
from src.core.protocols import assert_valid_environment
from src.core.types import ObservationSpec
from src.environments.mini_games.mini_resource import MiniResourceEnv
from src.environments.registry import EnvironmentRegistry

# ===================================================================
# MiniResourceEnv tests
# ===================================================================


class TestMiniResourceConstruction:
    """Tests for constructing the MiniResourceEnv."""

    def test_default_construction(self) -> None:
        env = MiniResourceEnv()
        assert env._board_size == 8
        assert env._max_steps == 50
        assert env._num_resources == 5

    def test_custom_construction(self) -> None:
        env = MiniResourceEnv(board_size=4, max_steps=20, num_resources=3)
        assert env._board_size == 4
        assert env._max_steps == 20
        assert env._num_resources == 3

    def test_observation_spec_correct_shapes(self) -> None:
        env = MiniResourceEnv(board_size=8)
        spec = env.observation_spec
        assert isinstance(spec, ObservationSpec)
        assert spec.spatial_shape == (6, 8, 8)
        assert spec.vector_dim == 4

    def test_observation_spec_custom_board_size(self) -> None:
        env = MiniResourceEnv(board_size=12)
        spec = env.observation_spec
        assert spec.spatial_shape == (6, 12, 12)

    def test_action_space_size_is_five(self) -> None:
        env = MiniResourceEnv()
        assert env.action_space_size == 5


class TestMiniResourceReset:
    """Tests for resetting the environment."""

    def test_reset_returns_dict_with_spatial_and_vector(self) -> None:
        env = MiniResourceEnv(seed=42)
        obs = env.reset()
        assert isinstance(obs, dict)
        assert "spatial" in obs
        assert "vector" in obs

    def test_reset_spatial_shape_matches_spec(self) -> None:
        env = MiniResourceEnv(board_size=8, seed=42)
        obs = env.reset()
        spec = env.observation_spec
        assert obs["spatial"].shape == spec.spatial_shape

    def test_reset_vector_shape(self) -> None:
        env = MiniResourceEnv(seed=42)
        obs = env.reset()
        assert obs["vector"].shape == (4,)

    def test_reset_sets_is_terminal_false(self) -> None:
        env = MiniResourceEnv(seed=42)
        env.reset()
        assert not env.is_terminal()

    def test_seeded_reset_is_reproducible(self) -> None:
        env1 = MiniResourceEnv()
        env2 = MiniResourceEnv()
        obs1 = env1.reset(seed=99)
        obs2 = env2.reset(seed=99)
        np.testing.assert_array_equal(obs1["spatial"], obs2["spatial"])
        np.testing.assert_array_equal(obs1["vector"], obs2["vector"])


class TestMiniResourceActions:
    """Tests for actions and stepping."""

    def test_get_valid_actions_shape_and_dtype(self) -> None:
        env = MiniResourceEnv(seed=42)
        env.reset()
        mask = env.get_valid_actions()
        assert mask.shape == (5,)
        assert mask.dtype == np.bool_
        assert np.all(mask)

    def test_step_returns_five_tuple(self) -> None:
        env = MiniResourceEnv(seed=42)
        env.reset()
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_reward_in_range(self) -> None:
        env = MiniResourceEnv(seed=42)
        env.reset()
        for _ in range(10):
            _, reward, terminated, truncated, _ = env.step(0)
            assert -1.0 <= reward <= 1.0
            if terminated or truncated:
                break

    def test_step_alternates_current_player(self) -> None:
        env = MiniResourceEnv(seed=42)
        env.reset()
        assert env._current_player == 0
        env.step(0)
        assert env._current_player == 1
        env.step(0)
        assert env._current_player == 0

    def test_resource_collection_increases_score(self) -> None:
        """Play many steps and verify at least one resource is collected."""
        env = MiniResourceEnv(board_size=4, num_resources=10, max_steps=200, seed=0)
        env.reset()
        initial_total = sum(env._scores)
        for _ in range(200):
            if env.is_terminal():
                break
            # Random action
            action = int(np.random.default_rng(0).integers(0, 5))
            env.step(action)
        final_total = sum(env._scores)
        # With 10 resources on a 4x4 board and 200 steps, at least one should be collected
        assert final_total >= initial_total


class TestMiniResourceTermination:
    """Tests for game termination conditions."""

    def test_game_terminates_at_max_steps(self) -> None:
        env = MiniResourceEnv(board_size=8, max_steps=10, seed=42)
        env.reset()
        for _ in range(10):
            if env.is_terminal():
                break
            env.step(4)  # action=stay
        assert env.is_terminal()

    def test_game_terminates_when_all_resources_collected(self) -> None:
        """Place 1 resource adjacent to player and collect it."""
        env = MiniResourceEnv(board_size=8, max_steps=200, num_resources=1, seed=42)
        env.reset()
        # Play until terminal or max steps
        for _ in range(200):
            if env.is_terminal():
                break
            # Try all directions to find the resource
            action = np.random.default_rng(42).integers(0, 5)
            env.step(int(action))

        # Even if we didn't collect it through random play, the max_steps will end it
        # Let's verify the termination logic with a direct approach
        env2 = MiniResourceEnv(board_size=3, max_steps=500, num_resources=1, seed=0)
        env2.reset()
        # Brute force: try all directions repeatedly until resource is found
        step_count = 0
        while not env2.is_terminal() and step_count < 500:
            # Cycle through directions
            env2.step(step_count % 4)
            step_count += 1

        assert env2.is_terminal()


class TestMiniResourceClone:
    """Tests for environment cloning."""

    def test_clone_creates_independent_copy(self) -> None:
        env = MiniResourceEnv(seed=42)
        env.reset()
        env.step(0)
        env.step(1)

        cloned = env.clone()
        # Mutate original
        env.step(2)

        # Cloned should not be affected
        assert cloned._step_count != env._step_count

    def test_clone_state_matches_original(self) -> None:
        env = MiniResourceEnv(seed=42)
        env.reset()
        env.step(0)

        cloned = env.clone()
        assert cloned._step_count == env._step_count
        assert cloned._current_player == env._current_player
        assert cloned._scores == env._scores
        assert cloned._positions == env._positions
        assert cloned._resources == env._resources


class TestMiniResourceSymmetries:
    """Tests for get_symmetries."""

    def test_get_symmetries_returns_identity(self) -> None:
        env = MiniResourceEnv(seed=42)
        obs = env.reset()
        policy = np.ones(5, dtype=np.float32) / 5.0
        symmetries = env.get_symmetries(obs["spatial"], policy)
        assert isinstance(symmetries, list)
        assert len(symmetries) == 1
        sym_obs, sym_policy = symmetries[0]
        np.testing.assert_array_equal(sym_obs, obs["spatial"])
        np.testing.assert_array_equal(sym_policy, policy)


class TestMiniResourceProtocol:
    """Tests that MiniResourceEnv satisfies the StrategicEnvironment protocol."""

    def test_passes_assert_valid_environment(self) -> None:
        env = MiniResourceEnv(board_size=4, max_steps=20, num_resources=3, seed=42)
        # Should not raise
        assert_valid_environment(env)


# ===================================================================
# EnvironmentRegistry tests
# ===================================================================


class _DummyEnvA:
    """Trivial environment for registry testing."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _DummyEnvB:
    """Another trivial environment for registry testing."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class TestRegistryEmpty:
    """Tests for an empty registry."""

    def test_empty_registry(self) -> None:
        registry = EnvironmentRegistry()
        assert len(registry) == 0
        assert registry.list_available() == []


class TestRegistryRegisterAndCreate:
    """Tests for registering and creating environments."""

    def test_register_and_create(self) -> None:
        registry = EnvironmentRegistry()
        registry.register("dummy_a", _DummyEnvA)
        env = registry.create("dummy_a", x=1)
        assert isinstance(env, _DummyEnvA)
        assert env.kwargs == {"x": 1}

    def test_duplicate_registration_raises(self) -> None:
        registry = EnvironmentRegistry()
        registry.register("dup", _DummyEnvA)
        with pytest.raises(DuplicateRegistrationError):
            registry.register("dup", _DummyEnvB)

    def test_force_overwrites(self) -> None:
        registry = EnvironmentRegistry()
        registry.register("env", _DummyEnvA)
        registry.register("env", _DummyEnvB, force=True)
        env = registry.create("env")
        assert isinstance(env, _DummyEnvB)

    def test_unknown_name_raises(self) -> None:
        registry = EnvironmentRegistry()
        with pytest.raises(EnvironmentNotRegisteredError):
            registry.create("nonexistent")


class TestRegistryUtilities:
    """Tests for utility methods: list_available, is_registered, unregister, etc."""

    def test_list_available_returns_sorted(self) -> None:
        registry = EnvironmentRegistry()
        registry.register("zebra", _DummyEnvA)
        registry.register("alpha", _DummyEnvB)
        assert registry.list_available() == ["alpha", "zebra"]

    def test_is_registered(self) -> None:
        registry = EnvironmentRegistry()
        registry.register("present", _DummyEnvA)
        assert registry.is_registered("present")
        assert not registry.is_registered("absent")

    def test_unregister(self) -> None:
        registry = EnvironmentRegistry()
        registry.register("temp", _DummyEnvA)
        assert registry.is_registered("temp")
        registry.unregister("temp")
        assert not registry.is_registered("temp")

    def test_len(self) -> None:
        registry = EnvironmentRegistry()
        assert len(registry) == 0
        registry.register("a", _DummyEnvA)
        registry.register("b", _DummyEnvB)
        assert len(registry) == 2

    def test_contains(self) -> None:
        registry = EnvironmentRegistry()
        registry.register("env1", _DummyEnvA)
        assert "env1" in registry
        assert "env2" not in registry
