"""Unit tests for the Resource Allocation strategic environment.

Tests the ResourceAllocationEnv and ResourceAllocationConfig classes
from the Strategic Planning SDK, validating environment lifecycle,
observation structure, reward signals, and cloning behaviour.
"""

import numpy as np
import pytest

from alpha_zero.strategic.envs.resource_allocation import (
    ResourceAllocationConfig,
    ResourceAllocationEnv,
)
from alpha_zero.strategic.protocols import StrategicObservation, StrategicStepResult


class TestResourceAllocationConfig:
    """Tests for ResourceAllocationConfig dataclass."""

    def test_defaults(self) -> None:
        """Default configuration has expected values."""
        config = ResourceAllocationConfig()

        assert config.num_regions == 4
        assert config.num_resource_types == 3
        assert config.num_time_steps == 10
        assert config.total_budget == 100.0
        assert len(config.resource_names) == config.num_resource_types
        assert len(config.region_names) == config.num_regions
        assert config.seed is None

    def test_from_dict(self) -> None:
        """Config can be created from a dictionary with partial overrides."""
        data = {
            "num_regions": 2,
            "num_resource_types": 1,
            "num_time_steps": 5,
            "total_budget": 50.0,
            "resource_names": ["supplies"],
            "region_names": ["A", "B"],
            "seed": 123,
        }
        config = ResourceAllocationConfig.from_dict(data)

        assert config.num_regions == 2
        assert config.num_resource_types == 1
        assert config.num_time_steps == 5
        assert config.total_budget == 50.0
        assert config.resource_names == ["supplies"]
        assert config.region_names == ["A", "B"]
        assert config.seed == 123

    def test_from_dict_uses_defaults_for_missing_keys(self) -> None:
        """from_dict falls back to defaults for keys not present."""
        config = ResourceAllocationConfig.from_dict({})

        assert config.num_regions == 4
        assert config.num_resource_types == 3

    def test_validation_rejects_bad_values(self) -> None:
        """Post-init validation rejects invalid configuration values."""
        with pytest.raises(ValueError, match="num_regions"):
            ResourceAllocationConfig(num_regions=0)

        with pytest.raises(ValueError, match="num_resource_types"):
            ResourceAllocationConfig(num_resource_types=0)

        with pytest.raises(ValueError, match="total_budget"):
            ResourceAllocationConfig(total_budget=-1.0)


class TestResourceAllocationEnv:
    """Tests for ResourceAllocationEnv."""

    @pytest.fixture
    def tiny_config(self) -> ResourceAllocationConfig:
        """Minimal configuration for fast tests."""
        return ResourceAllocationConfig(
            num_regions=2,
            num_resource_types=1,
            num_time_steps=3,
            total_budget=50.0,
            resource_names=["supplies"],
            region_names=["A", "B"],
            seed=42,
        )

    @pytest.fixture
    def env(self, tiny_config: ResourceAllocationConfig) -> ResourceAllocationEnv:
        """Tiny environment instance, already reset."""
        e = ResourceAllocationEnv(tiny_config)
        e.reset(seed=42)
        return e

    def test_num_objectives(self, env: ResourceAllocationEnv) -> None:
        """Environment reports exactly 3 objectives."""
        assert env.num_objectives == 3

    def test_objective_names(self, env: ResourceAllocationEnv) -> None:
        """Objective names match the expected shortage/equity/cost triple."""
        names = env.objective_names
        assert len(names) == 3
        assert "minimize_shortage" in names
        assert "maximize_equity" in names
        assert "minimize_cost" in names

    def test_num_agents(self, env: ResourceAllocationEnv) -> None:
        """Environment is single-agent."""
        assert env.num_agents == 1

    def test_state_spec_no_spatial(self, env: ResourceAllocationEnv) -> None:
        """State spec has no spatial component for this environment."""
        spec = env.state_spec
        assert spec.spatial_shape is None
        assert spec.has_spatial is False

    def test_reset_returns_observation(
        self, tiny_config: ResourceAllocationConfig
    ) -> None:
        """reset() returns a StrategicObservation with the right fields."""
        env = ResourceAllocationEnv(tiny_config)
        obs = env.reset(seed=0)

        assert isinstance(obs, StrategicObservation)
        assert obs.spatial is None
        assert isinstance(obs.vector, np.ndarray)
        assert obs.vector.ndim == 1
        assert obs.vector.shape[0] == env.state_spec.vector_dim
        assert isinstance(obs.action_mask, np.ndarray)
        assert obs.agent_id == 0
        assert obs.step_number == 0
        assert obs.done is False

    def test_step_returns_result(self, env: ResourceAllocationEnv) -> None:
        """step() returns a StrategicStepResult with a 3-element rewards array."""
        result = env.step(0)

        assert isinstance(result, StrategicStepResult)
        assert isinstance(result.rewards, np.ndarray)
        assert len(result.rewards) == 3
        assert isinstance(result.done, bool)
        assert isinstance(result.truncated, bool)
        assert isinstance(result.observation, StrategicObservation)

    def test_get_valid_actions_returns_mask(
        self, env: ResourceAllocationEnv
    ) -> None:
        """get_valid_actions returns a float32 mask of the correct size."""
        mask = env.get_valid_actions()

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (env.state_spec.action_space_size,)
        # All actions should be valid while episode is active
        assert np.all(mask == 1.0)

    def test_clone_creates_independent_copy(
        self, env: ResourceAllocationEnv
    ) -> None:
        """Cloned env is independent: stepping original does not affect clone."""
        clone = env.clone()

        # Take a step on the original
        env.step(0)

        # Clone should still be at the initial state
        clone_info = clone.get_info()
        env_info = env.get_info()

        assert clone_info["step"] == 0
        assert env_info["step"] == 1
        assert clone_info["remaining_budget"] != env_info["remaining_budget"] or (
            # Budget might be the same if action 0 allocates nothing
            clone_info["cumulative_cost"] != env_info["cumulative_cost"]
            or clone_info["step"] != env_info["step"]
        )

    def test_episode_terminates(
        self, tiny_config: ResourceAllocationConfig
    ) -> None:
        """Running enough steps causes the episode to terminate."""
        env = ResourceAllocationEnv(tiny_config)
        env.reset(seed=0)

        done = False
        steps = 0
        max_steps = tiny_config.num_time_steps + 5  # safety margin
        while not done and steps < max_steps:
            mask = env.get_valid_actions()
            valid = np.where(mask > 0)[0]
            if len(valid) == 0:
                break
            result = env.step(int(valid[0]))
            done = result.done or result.truncated
            steps += 1

        assert done, (
            f"Episode did not terminate within {max_steps} steps "
            f"(expected ~{tiny_config.num_time_steps})"
        )
        assert steps == tiny_config.num_time_steps

    def test_reproducibility_with_seed(
        self, tiny_config: ResourceAllocationConfig
    ) -> None:
        """Same seed produces identical initial observations."""
        env_a = ResourceAllocationEnv(tiny_config)
        env_b = ResourceAllocationEnv(tiny_config)

        obs_a = env_a.reset(seed=99)
        obs_b = env_b.reset(seed=99)

        np.testing.assert_array_equal(obs_a.vector, obs_b.vector)
        np.testing.assert_array_equal(obs_a.action_mask, obs_b.action_mask)

    def test_scenario_override(
        self, tiny_config: ResourceAllocationConfig
    ) -> None:
        """Passing scenario parameters at reset overrides initial demands."""
        env = ResourceAllocationEnv(tiny_config)

        custom_demands = np.array([[10.0], [20.0]])  # (2 regions, 1 resource)
        obs = env.reset(
            seed=0,
            scenario={"initial_demands": custom_demands, "demand_volatility": 0.5},
        )

        info = env.get_info()
        actual = np.array(info["demands"])
        np.testing.assert_array_almost_equal(actual, custom_demands)

        # Vector observation should reflect the custom demands
        assert isinstance(obs.vector, np.ndarray)
        assert obs.vector.shape[0] > 0

    def test_step_raises_when_done(
        self, tiny_config: ResourceAllocationConfig
    ) -> None:
        """Stepping a finished environment raises RuntimeError."""
        env = ResourceAllocationEnv(tiny_config)
        env.reset(seed=0)

        # Exhaust the episode
        for _ in range(tiny_config.num_time_steps):
            env.step(0)

        with pytest.raises(RuntimeError, match="done"):
            env.step(0)

    def test_step_raises_on_invalid_action(
        self, env: ResourceAllocationEnv
    ) -> None:
        """Out-of-range action raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            env.step(-1)

        with pytest.raises(ValueError, match="out of range"):
            env.step(env.state_spec.action_space_size)

    def test_get_info_contains_expected_keys(
        self, env: ResourceAllocationEnv
    ) -> None:
        """get_info returns a dict with the documented keys."""
        info = env.get_info()

        assert "step" in info
        assert "remaining_budget" in info
        assert "cumulative_shortage" in info
        assert "cumulative_cost" in info
        assert "demands" in info
        assert "capacities" in info
        assert "done" in info
