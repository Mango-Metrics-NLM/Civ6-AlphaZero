"""Unit tests for the Climate Negotiation strategic environment.

Tests the ClimateNegotiationEnv and ClimateConfig classes from the
Strategic Planning SDK, validating multi-agent turn taking, observation
structure, reward signals, and cloning behaviour.
"""

import numpy as np
import pytest

from alpha_zero.strategic.envs.climate_negotiation import (
    ClimateConfig,
    ClimateNegotiationEnv,
)
from alpha_zero.strategic.protocols import StrategicObservation, StrategicStepResult


class TestClimateConfig:
    """Tests for ClimateConfig dataclass."""

    def test_defaults(self) -> None:
        """Default configuration has expected values."""
        config = ClimateConfig()

        assert config.num_nations == 3
        assert config.num_periods == 20
        assert config.initial_temperature == pytest.approx(1.1)
        assert config.climate_sensitivity == pytest.approx(3.0)
        assert len(config.baseline_emissions) == config.num_nations
        assert len(config.gdp) == config.num_nations
        assert len(config.vulnerability) == config.num_nations
        assert config.seed is None

    def test_from_dict(self) -> None:
        """Config can be created from a dictionary with overrides."""
        data = {
            "num_nations": 2,
            "num_periods": 5,
            "initial_temperature": 1.5,
            "climate_sensitivity": 4.0,
            "baseline_emissions": [12.0, 6.0],
            "gdp": [25.0, 10.0],
            "vulnerability": [0.1, 0.8],
            "seed": 7,
        }
        config = ClimateConfig.from_dict(data)

        assert config.num_nations == 2
        assert config.num_periods == 5
        assert config.initial_temperature == pytest.approx(1.5)
        assert config.climate_sensitivity == pytest.approx(4.0)
        assert config.baseline_emissions == [12.0, 6.0]
        assert config.gdp == [25.0, 10.0]
        assert config.vulnerability == [0.1, 0.8]
        assert config.seed == 7

    def test_from_dict_uses_defaults_for_missing_keys(self) -> None:
        """from_dict falls back to defaults for keys not present."""
        config = ClimateConfig.from_dict({})

        assert config.num_nations == 3
        assert config.num_periods == 20

    def test_validation_rejects_bad_values(self) -> None:
        """Post-init validation rejects invalid configuration values."""
        with pytest.raises(ValueError, match="num_nations"):
            ClimateConfig(num_nations=0)

        with pytest.raises(ValueError, match="num_periods"):
            ClimateConfig(num_periods=0)

        with pytest.raises(ValueError, match="climate_sensitivity"):
            ClimateConfig(climate_sensitivity=-1.0)

    def test_validation_rejects_length_mismatch(self) -> None:
        """Post-init validation catches list length mismatches."""
        with pytest.raises(ValueError, match="baseline_emissions"):
            ClimateConfig(num_nations=2, baseline_emissions=[1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="gdp"):
            ClimateConfig(
                num_nations=2,
                baseline_emissions=[1.0, 2.0],
                gdp=[1.0],
            )

        with pytest.raises(ValueError, match="vulnerability"):
            ClimateConfig(
                num_nations=2,
                baseline_emissions=[1.0, 2.0],
                gdp=[1.0, 2.0],
                vulnerability=[0.5],
            )


class TestClimateNegotiationEnv:
    """Tests for ClimateNegotiationEnv."""

    @pytest.fixture
    def small_config(self) -> ClimateConfig:
        """Small configuration for fast tests."""
        return ClimateConfig(
            num_nations=2,
            num_periods=3,
            baseline_emissions=[8.0, 5.0],
            gdp=[15.0, 10.0],
            vulnerability=[0.3, 0.6],
            seed=42,
        )

    @pytest.fixture
    def env(self, small_config: ClimateConfig) -> ClimateNegotiationEnv:
        """Small environment instance, already reset."""
        e = ClimateNegotiationEnv(small_config)
        e.reset(seed=42)
        return e

    def test_num_objectives(self, env: ClimateNegotiationEnv) -> None:
        """Environment reports exactly 3 objectives."""
        assert env.num_objectives == 3

    def test_objective_names(self, env: ClimateNegotiationEnv) -> None:
        """Objective names match the expected warming/gdp/equity triple."""
        names = env.objective_names
        assert len(names) == 3
        assert "minimize_warming" in names
        assert "maximize_total_gdp" in names
        assert "maximize_equity" in names

    def test_num_agents(
        self, env: ClimateNegotiationEnv, small_config: ClimateConfig
    ) -> None:
        """Number of agents equals the number of nations."""
        assert env.num_agents == small_config.num_nations

    def test_multi_agent_turns(self, env: ClimateNegotiationEnv) -> None:
        """current_agent advances after each step within a period."""
        assert env.current_agent == 0

        # Nation 0 acts
        env.step(0)
        assert env.current_agent == 1

        # Nation 1 acts (completes the period)
        env.step(0)
        # After all nations acted, should wrap back to nation 0
        assert env.current_agent == 0

    def test_reset_returns_observation(
        self, small_config: ClimateConfig
    ) -> None:
        """reset() returns a StrategicObservation with the right fields."""
        env = ClimateNegotiationEnv(small_config)
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

    def test_step_returns_result(self, env: ClimateNegotiationEnv) -> None:
        """step() returns a StrategicStepResult with a 3-element rewards array."""
        result = env.step(0)

        assert isinstance(result, StrategicStepResult)
        assert isinstance(result.rewards, np.ndarray)
        assert len(result.rewards) == 3
        assert isinstance(result.done, bool)
        assert isinstance(result.truncated, bool)
        assert isinstance(result.observation, StrategicObservation)

    def test_rewards_zero_until_period_ends(
        self, env: ClimateNegotiationEnv
    ) -> None:
        """Rewards are zero until all nations have acted in a period."""
        # With 2 nations, first step should yield zero rewards
        result = env.step(0)
        np.testing.assert_array_equal(result.rewards, np.zeros(3))

        # Second step completes the period; rewards may be non-zero
        result = env.step(0)
        # Just verify shape, rewards can be any value after period ends
        assert result.rewards.shape == (3,)

    def test_get_valid_actions(self, env: ClimateNegotiationEnv) -> None:
        """get_valid_actions returns a mask of the correct size."""
        mask = env.get_valid_actions()

        assert isinstance(mask, np.ndarray)
        # Action space for climate env is 25 (5 carbon prices * 5 tech levels)
        assert mask.shape == (25,)
        assert np.all(mask == 1.0)

    def test_clone(
        self, env: ClimateNegotiationEnv
    ) -> None:
        """Cloned env is independent: stepping original does not affect clone."""
        clone = env.clone()

        # Step original through one full period
        env.step(0)
        env.step(0)

        # Clone should still be at the beginning
        clone_info = clone.get_info()
        env_info = env.get_info()

        assert clone_info["period"] == 0
        assert env_info["period"] == 1
        assert clone_info["current_nation"] == 0

    def test_episode_terminates(
        self, small_config: ClimateConfig
    ) -> None:
        """Running through all periods causes the episode to terminate."""
        env = ClimateNegotiationEnv(small_config)
        env.reset(seed=0)

        done = False
        steps = 0
        # Total steps = num_periods * num_nations
        max_steps = small_config.num_periods * small_config.num_nations + 5
        while not done and steps < max_steps:
            result = env.step(0)
            done = result.done or result.truncated
            steps += 1

        assert done, (
            f"Episode did not terminate within {max_steps} steps "
            f"(expected {small_config.num_periods * small_config.num_nations})"
        )
        expected_steps = small_config.num_periods * small_config.num_nations
        assert steps == expected_steps

    def test_scenario_override(
        self, small_config: ClimateConfig
    ) -> None:
        """Passing scenario parameters at reset overrides environment settings."""
        env = ClimateNegotiationEnv(small_config)

        obs = env.reset(
            seed=0,
            scenario={"initial_temperature": 2.0, "climate_sensitivity": 5.0},
        )

        info = env.get_info()
        assert info["temperature"] == pytest.approx(2.0)
        assert isinstance(obs, StrategicObservation)

    def test_step_raises_when_done(
        self, small_config: ClimateConfig
    ) -> None:
        """Stepping a finished environment raises RuntimeError."""
        env = ClimateNegotiationEnv(small_config)
        env.reset(seed=0)

        total_steps = small_config.num_periods * small_config.num_nations
        for _ in range(total_steps):
            env.step(0)

        with pytest.raises(RuntimeError, match="done"):
            env.step(0)

    def test_step_raises_on_invalid_action(
        self, env: ClimateNegotiationEnv
    ) -> None:
        """Out-of-range action raises ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            env.step(-1)

        with pytest.raises(ValueError, match="out of range"):
            env.step(25)

    def test_get_info_contains_expected_keys(
        self, env: ClimateNegotiationEnv
    ) -> None:
        """get_info returns a dict with the documented keys."""
        info = env.get_info()

        assert "period" in info
        assert "current_nation" in info
        assert "temperature" in info
        assert "cumulative_emissions" in info
        assert "emissions" in info
        assert "gdp" in info
        assert "tech_levels" in info
        assert "done" in info

    def test_state_spec_no_spatial(self, env: ClimateNegotiationEnv) -> None:
        """State spec has no spatial component."""
        spec = env.state_spec
        assert spec.spatial_shape is None
        assert spec.has_spatial is False

    def test_action_spec_size(self, env: ClimateNegotiationEnv) -> None:
        """Action spec reports 25 total actions."""
        spec = env.action_spec
        assert spec.total_size == 25
        assert spec.is_hierarchical is True
        assert len(spec.hierarchy_levels) == 2
