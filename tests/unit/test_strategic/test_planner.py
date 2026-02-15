"""Unit tests for the Strategic Planner.

Tests the StrategicPlanner and PlanningResult classes from the
Strategic Planning SDK, validating the planning loop, scenario
handling, decision trace recording, and Pareto frontier construction.
"""

import numpy as np
import pytest

from alpha_zero.strategic.config import StrategicConfig
from alpha_zero.strategic.envs.resource_allocation import (
    ResourceAllocationConfig,
    ResourceAllocationEnv,
)
from alpha_zero.strategic.planner import PlanningResult, StrategicPlanner
from alpha_zero.strategic.scenarios import Scenario
from alpha_zero.strategic.trace.decision_trace import DecisionTrace


class TestPlanningResult:
    """Tests for the PlanningResult dataclass."""

    def test_empty_result(self) -> None:
        """An empty PlanningResult has sensible defaults."""
        result = PlanningResult()

        assert result.traces == []
        assert result.pareto_frontier is None
        assert result.best_trace is None
        assert result.num_strategies == 0
        assert result.total_episodes == 0
        assert result.elapsed_seconds == 0.0

    def test_best_trace(self) -> None:
        """best_trace returns the first trace in the list."""
        trace_a = DecisionTrace(environment_name="env_a")
        trace_a.total_reward = 1.0
        trace_b = DecisionTrace(environment_name="env_b")
        trace_b.total_reward = 2.0

        result = PlanningResult(traces=[trace_b, trace_a])

        assert result.best_trace is trace_b

    def test_num_strategies(self) -> None:
        """num_strategies equals the number of traces."""
        traces = [
            DecisionTrace(environment_name=f"env_{i}") for i in range(5)
        ]
        result = PlanningResult(traces=traces)

        assert result.num_strategies == 5

    def test_summary(self) -> None:
        """summary() returns a non-empty string with key information."""
        trace = DecisionTrace(
            environment_name="test_env",
            objective_names=["a", "b"],
        )
        trace.final_outcomes = np.array([0.5, 0.8])
        trace.total_reward = 1.3

        result = PlanningResult(
            traces=[trace],
            objective_names=["a", "b"],
            total_episodes=10,
            elapsed_seconds=1.23,
        )
        summary = result.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "10 episodes" in summary
        assert "1.23" in summary
        assert "Strategies found: 1" in summary

    def test_summary_empty(self) -> None:
        """summary() works on an empty result without errors."""
        result = PlanningResult()
        summary = result.summary()
        assert isinstance(summary, str)
        assert "0 episodes" in summary


class TestStrategicPlanner:
    """Tests for the StrategicPlanner class."""

    @pytest.fixture
    def tiny_env_config(self) -> ResourceAllocationConfig:
        """Minimal ResourceAllocationConfig for fast tests."""
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
    def tiny_env(
        self, tiny_env_config: ResourceAllocationConfig
    ) -> ResourceAllocationEnv:
        """Tiny ResourceAllocationEnv instance (not yet reset)."""
        return ResourceAllocationEnv(tiny_env_config)

    @pytest.fixture
    def fast_config(self) -> StrategicConfig:
        """Fast planner config with minimal simulations."""
        config = StrategicConfig.from_preset("fast")
        config.planning.num_simulations = 5
        config.planning.num_planning_iterations = 1
        config.planning.strategies_per_iteration = 2
        config.random_seed = 123
        return config

    @pytest.fixture
    def planner(self, fast_config: StrategicConfig) -> StrategicPlanner:
        """Planner instance with fast configuration."""
        return StrategicPlanner(fast_config)

    def test_init_with_config(self, fast_config: StrategicConfig) -> None:
        """Planner can be initialised with a StrategicConfig."""
        planner = StrategicPlanner(fast_config)

        assert planner.config is fast_config
        assert planner.config.planning.num_simulations == 5
        assert planner.config.planning.num_planning_iterations == 1
        assert planner.config.planning.strategies_per_iteration == 2

    def test_plan_basic(
        self,
        planner: StrategicPlanner,
        tiny_env: ResourceAllocationEnv,
    ) -> None:
        """Basic planning run returns a well-formed PlanningResult."""
        result = planner.plan(environment=tiny_env, num_episodes=2)

        assert isinstance(result, PlanningResult)
        assert result.total_episodes == 2
        assert result.num_strategies == 2
        assert result.elapsed_seconds > 0.0
        assert len(result.traces) == 2
        assert result.best_trace is not None

        # Objective names should come from the environment
        assert "minimize_shortage" in result.objective_names
        assert "maximize_equity" in result.objective_names
        assert "minimize_cost" in result.objective_names

        # Each trace should have final outcomes
        for trace in result.traces:
            assert isinstance(trace, DecisionTrace)
            assert len(trace.final_outcomes) == 3

    def test_plan_with_scenarios(
        self,
        planner: StrategicPlanner,
        tiny_env: ResourceAllocationEnv,
    ) -> None:
        """Planning with explicit Scenario objects populates scenarios_evaluated."""
        scenarios = [
            Scenario(
                name="low_demand",
                parameters={"demand_volatility": 0.5},
                probability=0.5,
            ),
            Scenario(
                name="high_demand",
                parameters={"demand_volatility": 5.0},
                probability=0.5,
            ),
        ]

        result = planner.plan(
            environment=tiny_env,
            scenarios=scenarios,
            num_episodes=2,
        )

        assert isinstance(result, PlanningResult)
        assert len(result.scenarios_evaluated) > 0
        # At least one of our scenario names should appear
        evaluated_set = set(result.scenarios_evaluated)
        assert evaluated_set & {"low_demand", "high_demand"}

    def test_plan_records_traces(
        self,
        planner: StrategicPlanner,
        tiny_env: ResourceAllocationEnv,
    ) -> None:
        """Traces are non-empty and contain decision points."""
        # Enable explainability so decision points are recorded
        planner.config.explainability.enabled = True
        planner.config.explainability.record_alternatives = True

        result = planner.plan(environment=tiny_env, num_episodes=2)

        assert len(result.traces) > 0
        for trace in result.traces:
            assert isinstance(trace, DecisionTrace)
            # Each trace should have at least one decision point
            assert trace.num_steps > 0, (
                "Expected decision points but trace was empty"
            )

            # Each decision point should have basic data
            for dp in trace.decision_points:
                assert dp.step >= 0
                assert isinstance(dp.action_name, str)
                assert 0.0 <= dp.confidence <= 1.0

    def test_plan_builds_pareto_frontier(
        self,
        planner: StrategicPlanner,
        tiny_env: ResourceAllocationEnv,
    ) -> None:
        """Planning builds a Pareto frontier with at least one point."""
        result = planner.plan(environment=tiny_env, num_episodes=2)

        assert result.pareto_frontier is not None
        # With 2 episodes there should be at least 1 Pareto-optimal point
        assert result.pareto_frontier.size >= 1

        frontier_points = result.pareto_frontier.get_frontier()
        for point in frontier_points:
            assert len(point.values) == 3  # 3 objectives

    def test_plan_returns_finite_outcomes(
        self,
        planner: StrategicPlanner,
        tiny_env: ResourceAllocationEnv,
    ) -> None:
        """Planning produces finite, non-NaN outcome values."""
        result = planner.plan(environment=tiny_env, num_episodes=2)

        for trace in result.traces:
            assert len(trace.final_outcomes) == 3
            assert np.all(np.isfinite(trace.final_outcomes)), (
                f"Non-finite outcomes: {trace.final_outcomes}"
            )

    def test_plan_metadata(
        self,
        planner: StrategicPlanner,
        tiny_env: ResourceAllocationEnv,
    ) -> None:
        """Planning result metadata contains config and objective count."""
        result = planner.plan(environment=tiny_env, num_episodes=1)

        assert "config" in result.metadata
        assert "num_objectives" in result.metadata
        assert result.metadata["num_objectives"] == 3
