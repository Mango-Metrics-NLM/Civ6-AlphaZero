"""Unit tests for the Pandemic Response strategic environment.

Tests the PandemicResponseEnv and PandemicConfig classes from the
Strategic Planning SDK, validating SIR dynamics, observation structure,
reward signals, resource masking, and cloning behaviour.
"""

import numpy as np
import pytest

from alpha_zero.strategic.envs.pandemic_response import (
    PandemicConfig,
    PandemicResponseEnv,
)
from alpha_zero.strategic.protocols import StrategicObservation, StrategicStepResult


class TestPandemicConfig:
    """Tests for PandemicConfig dataclass."""

    def test_defaults(self) -> None:
        """Default configuration has expected values."""
        config = PandemicConfig()

        assert config.num_regions == 4
        assert config.num_time_steps == 30
        assert config.initial_infected == 100
        assert config.r0 == pytest.approx(2.5)
        assert config.recovery_rate == pytest.approx(0.1)
        assert config.total_resources == pytest.approx(1000.0)
        assert config.resource_types == ["vaccines", "ppe", "ventilators"]
        assert config.region_populations is None
        assert config.base_mortality_rate == pytest.approx(0.02)
        assert config.seed is None

    def test_from_dict(self) -> None:
        """Config can be created from a dictionary with overrides."""
        data = {
            "num_regions": 3,
            "num_time_steps": 10,
            "initial_infected": 50,
            "r0": 3.0,
            "recovery_rate": 0.2,
            "total_resources": 500.0,
            "resource_types": ["vaccines", "ppe"],
            "region_populations": [500_000, 800_000, 200_000],
            "base_mortality_rate": 0.05,
            "seed": 7,
        }
        config = PandemicConfig.from_dict(data)

        assert config.num_regions == 3
        assert config.num_time_steps == 10
        assert config.initial_infected == 50
        assert config.r0 == pytest.approx(3.0)
        assert config.recovery_rate == pytest.approx(0.2)
        assert config.total_resources == pytest.approx(500.0)
        assert config.resource_types == ["vaccines", "ppe"]
        assert config.region_populations == [500_000, 800_000, 200_000]
        assert config.base_mortality_rate == pytest.approx(0.05)
        assert config.seed == 7

    def test_from_dict_uses_defaults_for_missing_keys(self) -> None:
        """from_dict falls back to defaults for keys not present."""
        config = PandemicConfig.from_dict({})

        assert config.num_regions == 4
        assert config.num_time_steps == 30
        assert config.r0 == pytest.approx(2.5)

    def test_validation_rejects_bad_values(self) -> None:
        """Post-init validation rejects invalid configuration values."""
        with pytest.raises(ValueError, match="num_regions"):
            PandemicConfig(num_regions=0)

        with pytest.raises(ValueError, match="num_time_steps"):
            PandemicConfig(num_time_steps=0)

        with pytest.raises(ValueError, match="r0"):
            PandemicConfig(r0=-1.0)

        with pytest.raises(ValueError, match="recovery_rate"):
            PandemicConfig(recovery_rate=0.0)

        with pytest.raises(ValueError, match="recovery_rate"):
            PandemicConfig(recovery_rate=1.5)

        with pytest.raises(ValueError, match="total_resources"):
            PandemicConfig(total_resources=-10.0)

        with pytest.raises(ValueError, match="base_mortality_rate"):
            PandemicConfig(base_mortality_rate=-0.1)

        with pytest.raises(ValueError, match="base_mortality_rate"):
            PandemicConfig(base_mortality_rate=1.5)

    def test_validation_rejects_length_mismatch(self) -> None:
        """Post-init validation catches region_populations length mismatch."""
        with pytest.raises(ValueError, match="region_populations"):
            PandemicConfig(num_regions=2, region_populations=[100, 200, 300])

    def test_validation_rejects_empty_resource_types(self) -> None:
        """Post-init validation rejects empty resource_types list."""
        with pytest.raises(ValueError, match="resource type"):
            PandemicConfig(resource_types=[])


class TestPandemicResponseEnv:
    """Tests for PandemicResponseEnv."""

    @pytest.fixture
    def small_config(self) -> PandemicConfig:
        """Small configuration for fast tests."""
        return PandemicConfig(
            num_regions=2,
            num_time_steps=5,
            initial_infected=100,
            r0=2.0,
            recovery_rate=0.1,
            total_resources=500.0,
            region_populations=[100_000, 200_000],
            seed=42,
        )

    @pytest.fixture
    def env(self, small_config: PandemicConfig) -> PandemicResponseEnv:
        """Small environment instance, already reset."""
        e = PandemicResponseEnv(small_config)
        e.reset(seed=42)
        return e

    def test_num_objectives(self, env: PandemicResponseEnv) -> None:
        """Environment reports exactly 3 objectives."""
        assert env.num_objectives == 3

    def test_objective_names(self, env: PandemicResponseEnv) -> None:
        """Objective names match the expected deaths/economic/equity triple."""
        names = env.objective_names
        assert len(names) == 3
        assert "minimize_deaths" in names
        assert "minimize_economic_impact" in names
        assert "maximize_equity" in names

    def test_num_agents(self, env: PandemicResponseEnv) -> None:
        """Environment is single-agent."""
        assert env.num_agents == 1

    def test_current_agent_always_zero(self, env: PandemicResponseEnv) -> None:
        """current_agent is always 0 for single-agent environment."""
        assert env.current_agent == 0
        env.step(0)
        assert env.current_agent == 0

    def test_state_spec_no_spatial(self, env: PandemicResponseEnv) -> None:
        """State spec has no spatial component."""
        spec = env.state_spec
        assert spec.spatial_shape is None
        assert spec.has_spatial is False

    def test_state_spec_vector_dim(
        self, env: PandemicResponseEnv, small_config: PandemicConfig
    ) -> None:
        """State spec vector dimension matches expected formula."""
        spec = env.state_spec
        nr = small_config.num_regions
        nrt = len(small_config.resource_types)
        expected_dim = nr * 5 + nrt + 3
        assert spec.vector_dim == expected_dim

    def test_action_spec_size(
        self, env: PandemicResponseEnv, small_config: PandemicConfig
    ) -> None:
        """Action spec total_size matches num_pairs * 5."""
        spec = env.action_spec
        nr = small_config.num_regions
        nrt = len(small_config.resource_types)
        expected_size = nrt * nr * 5
        assert spec.total_size == expected_size
        # Single hierarchy level means is_hierarchical is False
        assert spec.is_hierarchical is False
        assert len(spec.hierarchy_levels) == 1

    def test_reset_returns_observation(
        self, small_config: PandemicConfig
    ) -> None:
        """reset() returns a StrategicObservation with the right fields."""
        env = PandemicResponseEnv(small_config)
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

    def test_step_returns_result(self, env: PandemicResponseEnv) -> None:
        """step() returns a StrategicStepResult with a 3-element rewards array."""
        result = env.step(0)

        assert isinstance(result, StrategicStepResult)
        assert isinstance(result.rewards, np.ndarray)
        assert len(result.rewards) == 3
        assert isinstance(result.done, bool)
        assert isinstance(result.truncated, bool)
        assert isinstance(result.observation, StrategicObservation)

    def test_step_reward_shape(self, env: PandemicResponseEnv) -> None:
        """Rewards array always has exactly 3 elements after each step."""
        for _ in range(3):
            result = env.step(0)
            assert result.rewards.shape == (3,)

    def test_get_valid_actions(
        self, env: PandemicResponseEnv, small_config: PandemicConfig
    ) -> None:
        """get_valid_actions returns a mask of the correct size."""
        mask = env.get_valid_actions()

        assert isinstance(mask, np.ndarray)
        nr = small_config.num_regions
        nrt = len(small_config.resource_types)
        expected_size = nrt * nr * 5
        assert mask.shape == (expected_size,)
        # At least zero-allocation actions should be valid
        assert mask.sum() > 0

    def test_get_valid_actions_zero_allocation_always_valid(
        self, env: PandemicResponseEnv, small_config: PandemicConfig
    ) -> None:
        """Zero-allocation actions (level 0) are always valid."""
        mask = env.get_valid_actions()

        nr = small_config.num_regions
        nrt = len(small_config.resource_types)
        num_pairs = nrt * nr
        for pair_idx in range(num_pairs):
            zero_action = pair_idx * 5 + 0
            assert mask[zero_action] == 1.0

    def test_clone(self, env: PandemicResponseEnv) -> None:
        """Cloned env is independent: stepping original does not affect clone."""
        clone = env.clone()

        # Step original a few times
        env.step(0)
        env.step(0)

        # Clone should still be at the beginning
        clone_info = clone.get_info()
        env_info = env.get_info()

        assert clone_info["step"] == 0
        assert env_info["step"] == 2

    def test_clone_state_independence(self, env: PandemicResponseEnv) -> None:
        """Mutating clone state does not affect original."""
        clone = env.clone()
        clone.step(0)

        original_info = env.get_info()
        assert original_info["step"] == 0

    def test_episode_terminates(
        self, small_config: PandemicConfig
    ) -> None:
        """Running through all time steps causes the episode to terminate."""
        env = PandemicResponseEnv(small_config)
        env.reset(seed=0)

        done = False
        steps = 0
        max_steps = small_config.num_time_steps + 5
        while not done and steps < max_steps:
            result = env.step(0)
            done = result.done or result.truncated
            steps += 1

        assert done, (
            f"Episode did not terminate within {max_steps} steps "
            f"(expected at most {small_config.num_time_steps})"
        )
        assert steps <= small_config.num_time_steps

    def test_scenario_r0_override(
        self, small_config: PandemicConfig
    ) -> None:
        """Passing r0 in scenario overrides the config value."""
        env = PandemicResponseEnv(small_config)

        obs = env.reset(
            seed=0,
            scenario={"r0": 5.0},
        )

        info = env.get_info()
        # Effective R0 should reflect the override
        for r0_val in info["effective_r0"]:
            assert r0_val == pytest.approx(5.0)
        assert isinstance(obs, StrategicObservation)

    def test_scenario_initial_infected_override(
        self, small_config: PandemicConfig
    ) -> None:
        """Passing initial_infected in scenario overrides the config value."""
        env = PandemicResponseEnv(small_config)

        env.reset(seed=0, scenario={"initial_infected": 500})

        info = env.get_info()
        for infected in info["infected"]:
            assert infected == pytest.approx(500.0)

    def test_scenario_mortality_override(
        self, small_config: PandemicConfig
    ) -> None:
        """Passing mortality_rate in scenario overrides base_mortality_rate."""
        env = PandemicResponseEnv(small_config)

        obs = env.reset(
            seed=0,
            scenario={"mortality_rate": 0.1},
        )

        assert isinstance(obs, StrategicObservation)

    def test_step_raises_when_done(
        self, small_config: PandemicConfig
    ) -> None:
        """Stepping a finished environment raises RuntimeError."""
        env = PandemicResponseEnv(small_config)
        env.reset(seed=0)

        # Run until done
        for _ in range(small_config.num_time_steps):
            result = env.step(0)
            if result.done or result.truncated:
                break

        with pytest.raises(RuntimeError, match="done"):
            env.step(0)

    def test_step_raises_on_invalid_action(
        self, env: PandemicResponseEnv
    ) -> None:
        """Out-of-range action raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            env.step(-1)

        action_size = env.action_spec.total_size
        with pytest.raises(ValueError, match="out of range"):
            env.step(action_size)

    def test_get_info_contains_expected_keys(
        self, env: PandemicResponseEnv
    ) -> None:
        """get_info returns a dict with the documented keys."""
        info = env.get_info()

        assert "step" in info
        assert "susceptible" in info
        assert "infected" in info
        assert "recovered" in info
        assert "dead" in info
        assert "effective_r0" in info
        assert "resources_remaining" in info
        assert "cumulative_deaths" in info
        assert "cumulative_economic_impact" in info
        assert "done" in info

    def test_sir_dynamics_infected_changes(
        self, small_config: PandemicConfig
    ) -> None:
        """SIR dynamics cause the infected count to change over steps."""
        env = PandemicResponseEnv(small_config)
        env.reset(seed=42)

        initial_info = env.get_info()
        initial_infected = sum(initial_info["infected"])

        # Step through several time steps
        for _ in range(3):
            env.step(0)

        after_info = env.get_info()
        after_infected = sum(after_info["infected"])

        # With R0=2.0 and no intervention (action 0 is zero allocation),
        # infected count should change (typically increase initially)
        assert after_infected != pytest.approx(initial_infected, abs=1.0), (
            "Infected count should change due to SIR dynamics"
        )

    def test_sir_compartments_conserve_population(
        self, small_config: PandemicConfig
    ) -> None:
        """S + I + R + D should approximately equal the total population."""
        env = PandemicResponseEnv(small_config)
        env.reset(seed=42)

        total_pop = sum(small_config.region_populations)  # type: ignore[arg-type]

        for _ in range(small_config.num_time_steps):
            result = env.step(0)
            if result.done:
                break

        info = env.get_info()
        compartment_total = (
            sum(info["susceptible"])
            + sum(info["infected"])
            + sum(info["recovered"])
            + sum(info["dead"])
        )

        # Allow small floating point drift but population should be conserved
        assert compartment_total == pytest.approx(total_pop, rel=0.01), (
            f"Population not conserved: S+I+R+D={compartment_total}, expected {total_pop}"
        )

    def test_deaths_accumulate(
        self, small_config: PandemicConfig
    ) -> None:
        """Cumulative deaths should be non-decreasing over steps."""
        env = PandemicResponseEnv(small_config)
        env.reset(seed=42)

        prev_deaths = 0.0
        for _ in range(small_config.num_time_steps):
            result = env.step(0)
            info = env.get_info()
            assert info["cumulative_deaths"] >= prev_deaths
            prev_deaths = info["cumulative_deaths"]
            if result.done:
                break

    def test_resources_deplete(self, env: PandemicResponseEnv) -> None:
        """Using non-zero allocation actions depletes resources."""
        initial_info = env.get_info()
        initial_resources = sum(initial_info["resources_remaining"])

        # Use a non-zero allocation action (level > 0)
        # Action 1 = pair_index 0, level 1 (25% allocation)
        env.step(1)

        after_info = env.get_info()
        after_resources = sum(after_info["resources_remaining"])

        assert after_resources < initial_resources
