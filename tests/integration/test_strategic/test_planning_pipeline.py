"""Integration tests for the end-to-end strategic planning pipeline.

Tests the full flow: environment -> MCTS search -> decision traces -> results.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.strategic.config import StrategicConfig
from alpha_zero.strategic.envs.resource_allocation import (
    ResourceAllocationConfig,
    ResourceAllocationEnv,
)
from alpha_zero.strategic.objectives import ObjectiveSpec
from alpha_zero.strategic.planner import StrategicPlanner
from alpha_zero.strategic.scenarios import Scenario


class TestEndToEndPlanning:
    """Integration tests for the full planning pipeline."""

    @pytest.fixture
    def fast_config(self) -> StrategicConfig:
        """Create a minimal config for fast integration testing."""
        config = StrategicConfig.from_preset("fast")
        config.planning.num_simulations = 5
        config.planning.num_planning_iterations = 1
        config.planning.strategies_per_iteration = 3
        config.random_seed = 42
        return config

    @pytest.fixture
    def tiny_env(self) -> ResourceAllocationEnv:
        """Create a tiny environment for fast testing."""
        return ResourceAllocationEnv(
            config=ResourceAllocationConfig(
                num_regions=2,
                num_resource_types=1,
                num_time_steps=3,
                resource_names=["medical"],
                region_names=["North", "South"],
                seed=42,
            )
        )

    def test_plan_produces_valid_result(
        self, fast_config: StrategicConfig, tiny_env: ResourceAllocationEnv
    ) -> None:
        """Full planning pipeline produces valid PlanningResult."""
        planner = StrategicPlanner(fast_config)
        result = planner.plan(environment=tiny_env, num_episodes=3)

        assert result.total_episodes == 3
        assert result.num_strategies == 3
        assert len(result.objective_names) == 3
        assert result.pareto_frontier is not None
        assert result.pareto_frontier.size >= 1
        assert result.elapsed_seconds > 0
        assert result.best_trace is not None

    def test_plan_with_custom_objectives(
        self, fast_config: StrategicConfig, tiny_env: ResourceAllocationEnv
    ) -> None:
        """Planning with explicitly specified objectives."""
        objectives = [
            ObjectiveSpec(name="shortage", direction="minimize", weight=0.5),
            ObjectiveSpec(name="equity", direction="maximize", weight=0.3),
            ObjectiveSpec(name="cost", direction="minimize", weight=0.2),
        ]
        planner = StrategicPlanner(fast_config)
        result = planner.plan(
            environment=tiny_env,
            objectives=objectives,
            num_episodes=2,
        )

        assert result.num_strategies == 2
        assert result.best_trace is not None
        assert len(result.best_trace.final_outcomes) == 3

    def test_plan_with_scenarios(
        self, fast_config: StrategicConfig, tiny_env: ResourceAllocationEnv
    ) -> None:
        """Planning with uncertainty scenarios."""
        scenarios = [
            Scenario(name="baseline", parameters={}, probability=0.5),
            Scenario(
                name="high_demand",
                parameters={"demand_volatility": 0.5},
                probability=0.5,
            ),
        ]
        planner = StrategicPlanner(fast_config)
        result = planner.plan(
            environment=tiny_env,
            scenarios=scenarios,
            num_episodes=4,
        )

        assert result.total_episodes == 4
        assert len(result.scenarios_evaluated) >= 1

    def test_decision_traces_have_content(
        self, fast_config: StrategicConfig, tiny_env: ResourceAllocationEnv
    ) -> None:
        """Decision traces contain meaningful data."""
        fast_config.explainability.enabled = True
        fast_config.explainability.record_alternatives = True
        planner = StrategicPlanner(fast_config)
        result = planner.plan(environment=tiny_env, num_episodes=2)

        for trace in result.traces:
            assert trace.num_steps >= 0
            assert len(trace.objective_names) == 3
            assert len(trace.final_outcomes) == 3
            assert all(np.isfinite(trace.final_outcomes))

            # Decision points should exist if steps were taken
            if trace.num_steps > 0:
                for dp in trace.decision_points:
                    assert dp.step >= 0
                    assert isinstance(dp.action_name, str)
                    assert dp.confidence >= 0.0

    def test_traces_serializable(
        self, fast_config: StrategicConfig, tiny_env: ResourceAllocationEnv
    ) -> None:
        """Traces can be serialized to JSON and back."""
        planner = StrategicPlanner(fast_config)
        result = planner.plan(environment=tiny_env, num_episodes=2)

        for trace in result.traces:
            json_str = trace.to_json()
            assert isinstance(json_str, str)
            assert len(json_str) > 0

            # Round-trip
            from alpha_zero.strategic.trace.decision_trace import DecisionTrace

            restored = DecisionTrace.from_json(json_str)
            assert restored.environment_name == trace.environment_name
            assert len(restored.final_outcomes) == len(trace.final_outcomes)

    def test_evaluate_across_scenarios(
        self, fast_config: StrategicConfig, tiny_env: ResourceAllocationEnv
    ) -> None:
        """Evaluate strategy across multiple scenarios."""
        scenarios = [
            Scenario(name="low", parameters={}, probability=0.5),
            Scenario(name="high", parameters={"demand_volatility": 0.5}, probability=0.5),
        ]
        planner = StrategicPlanner(fast_config)
        result = planner.evaluate(
            environment=tiny_env,
            scenarios=scenarios,
            num_episodes_per_scenario=1,
        )

        assert len(result.outcomes_by_scenario) >= 1
        assert len(result.expected_outcomes) == 3
        assert 0.0 <= result.robustness_score <= 1.0

    def test_compare_strategies(
        self, fast_config: StrategicConfig, tiny_env: ResourceAllocationEnv
    ) -> None:
        """Compare multiple planning results."""
        planner = StrategicPlanner(fast_config)

        # Run two separate plans
        result_a = planner.plan(environment=tiny_env, num_episodes=2)
        result_b = planner.plan(environment=tiny_env, num_episodes=2)

        comparison = planner.compare(
            [result_a, result_b],
            strategy_names=["strategy_a", "strategy_b"],
        )

        assert len(comparison.strategy_names) == 2
        assert len(comparison.outcomes) == 2
        assert comparison.pareto_frontier is not None


class TestMultiEnvironmentPlanning:
    """Test planning across different environment types."""

    @pytest.fixture
    def fast_config(self) -> StrategicConfig:
        """Minimal config for testing."""
        config = StrategicConfig.from_preset("fast")
        config.planning.num_simulations = 3
        config.planning.num_planning_iterations = 1
        config.planning.strategies_per_iteration = 1
        config.random_seed = 42
        return config

    def test_plan_resource_allocation(self, fast_config: StrategicConfig) -> None:
        """Plan on resource allocation environment."""
        from alpha_zero.strategic.envs.resource_allocation import (
            ResourceAllocationConfig,
            ResourceAllocationEnv,
        )

        env = ResourceAllocationEnv(
            config=ResourceAllocationConfig(
                num_regions=2,
                num_resource_types=1,
                num_time_steps=3,
                resource_names=["medical"],
                region_names=["North", "South"],
            )
        )
        planner = StrategicPlanner(fast_config)
        result = planner.plan(environment=env, num_episodes=1)
        assert result.num_strategies == 1
        assert len(result.objective_names) == 3

    def test_plan_climate_negotiation(self, fast_config: StrategicConfig) -> None:
        """Plan on climate negotiation environment."""
        from alpha_zero.strategic.envs.climate_negotiation import (
            ClimateConfig,
            ClimateNegotiationEnv,
        )

        env = ClimateNegotiationEnv(
            config=ClimateConfig(
                num_nations=2,
                num_periods=3,
                baseline_emissions=[10.0, 8.0],
                gdp=[20.0, 15.0],
                vulnerability=[0.2, 0.3],
            )
        )
        planner = StrategicPlanner(fast_config)
        result = planner.plan(environment=env, num_episodes=1)
        assert result.num_strategies == 1
        assert len(result.objective_names) == 3

    def test_plan_supply_chain(self, fast_config: StrategicConfig) -> None:
        """Plan on supply chain environment."""
        from alpha_zero.strategic.envs.supply_chain import (
            SupplyChainConfig,
            SupplyChainEnv,
        )

        env = SupplyChainEnv(
            config=SupplyChainConfig(num_suppliers=2, num_products=1, num_periods=3)
        )
        planner = StrategicPlanner(fast_config)
        result = planner.plan(environment=env, num_episodes=1)
        assert result.num_strategies == 1
        assert len(result.objective_names) == 3
