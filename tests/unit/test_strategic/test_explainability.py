"""Unit tests for the explainability module.

Tests explanation generation utilities including:
- TraceExplainer for single decisions and full traces
- TradeoffAnalyzer for multi-trace tradeoff analysis
- SensitivityAnalyzer for scenario-based sensitivity
- ComparisonExplainer for side-by-side strategy comparison
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.strategic.trace.decision_trace import (
    ActionAlternative,
    DecisionPoint,
    DecisionTrace,
)
from alpha_zero.strategic.trace.explainability import (
    ComparisonExplainer,
    SensitivityAnalyzer,
    TraceExplainer,
    TradeoffAnalyzer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alternative(
    index: int, name: str, visits: int, prior: float, values: list[float]
) -> ActionAlternative:
    """Shorthand factory for ActionAlternative."""
    return ActionAlternative(
        action_index=index,
        action_name=name,
        visit_count=visits,
        prior_probability=prior,
        value_estimate=np.array(values),
    )


def _make_decision_point(
    step: int,
    action_index: int,
    action_name: str,
    objective_values: list[float],
    confidence: float,
    alternatives: list[ActionAlternative] | None = None,
) -> DecisionPoint:
    """Shorthand factory for DecisionPoint."""
    return DecisionPoint(
        step=step,
        agent_id=0,
        state_summary={"step": step},
        action_taken=action_index,
        action_name=action_name,
        alternatives_considered=alternatives or [],
        objective_values=np.array(objective_values),
        mcts_visit_counts=np.array([100, 80, 20]),
        value_estimate=np.array(objective_values),
        confidence=confidence,
    )


def _make_trace(
    env_name: str,
    scenario: str,
    objective_names: list[str],
    decision_points: list[DecisionPoint],
    final_outcomes: list[float],
    total_reward: float = 0.0,
    pareto_rank: int = -1,
) -> DecisionTrace:
    """Shorthand factory for DecisionTrace."""
    trace = DecisionTrace(
        environment_name=env_name,
        scenario_name=scenario,
        objective_names=objective_names,
        final_outcomes=np.array(final_outcomes),
        total_reward=total_reward,
        pareto_rank=pareto_rank,
    )
    for dp in decision_points:
        trace.add_decision_point(dp)
    return trace


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def objective_names() -> list[str]:
    """Standard two-objective names."""
    return ["efficiency", "equity"]


@pytest.fixture
def explainer(objective_names: list[str]) -> TraceExplainer:
    """Create a TraceExplainer with standard objectives."""
    return TraceExplainer(objective_names)


@pytest.fixture
def sample_decision_point() -> DecisionPoint:
    """Create a representative DecisionPoint with alternatives."""
    alternatives = [
        _make_alternative(0, "invest_health", 150, 0.4, [0.8, 0.6]),
        _make_alternative(1, "invest_defense", 80, 0.2, [0.3, 0.9]),
        _make_alternative(2, "invest_education", 120, 0.35, [0.7, 0.5]),
    ]
    return DecisionPoint(
        step=3,
        agent_id=0,
        state_summary={"budget": 85.0},
        action_taken=0,
        action_name="invest_health",
        alternatives_considered=alternatives,
        objective_values=np.array([0.8, 0.6]),
        mcts_visit_counts=np.array([150, 80, 120]),
        value_estimate=np.array([0.75, 0.55]),
        confidence=0.85,
        reasoning="High efficiency return.",
    )


# ---------------------------------------------------------------------------
# TestTraceExplainer
# ---------------------------------------------------------------------------


class TestTraceExplainer:
    """Tests for TraceExplainer."""

    def test_explain_decision(
        self, explainer: TraceExplainer, sample_decision_point: DecisionPoint
    ) -> None:
        """Test explanation of a single decision point is non-empty."""
        explanation = explainer.explain_decision(sample_decision_point)

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        # Should mention the action and confidence
        assert "invest_health" in explanation
        assert "85.0%" in explanation
        # Should mention objective values
        assert "efficiency" in explanation
        assert "equity" in explanation
        # Should mention alternatives (excluding the chosen action)
        assert "invest_defense" in explanation or "invest_education" in explanation
        # Should include confidence characterization
        assert "high-confidence" in explanation.lower()
        # Should include reasoning
        assert "High efficiency return." in explanation

    def test_explain_trace(
        self, explainer: TraceExplainer, objective_names: list[str]
    ) -> None:
        """Test explanation of a full trace with 3 decision points."""
        dps = [
            _make_decision_point(0, 0, "expand", [0.5, 0.4], 0.9),
            _make_decision_point(1, 1, "build", [0.6, 0.5], 0.6),
            _make_decision_point(2, 2, "attack", [0.7, 0.3], 0.3),
        ]
        trace = _make_trace(
            env_name="resource_allocation",
            scenario="optimistic",
            objective_names=objective_names,
            decision_points=dps,
            final_outcomes=[0.7, 0.4],
            total_reward=1.1,
            pareto_rank=0,
        )

        explanation = explainer.explain_trace(trace)

        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "resource_allocation" in explanation
        assert "optimistic" in explanation
        assert "Total steps: 3" in explanation
        assert "Pareto rank: 0" in explanation
        assert "Pareto frontier" in explanation  # rank 0 note
        assert "expand" in explanation or "build" in explanation or "attack" in explanation


# ---------------------------------------------------------------------------
# TestTradeoffAnalyzer
# ---------------------------------------------------------------------------


class TestTradeoffAnalyzer:
    """Tests for TradeoffAnalyzer."""

    def test_analyze_tradeoffs(self, objective_names: list[str]) -> None:
        """Test tradeoff analysis across 5 traces with varied outcomes."""
        analyzer = TradeoffAnalyzer(objective_names)

        # Create traces where efficiency and equity are anti-correlated
        traces = [
            _make_trace("env", "s1", objective_names, [], [0.9, 0.1], 1.0),
            _make_trace("env", "s2", objective_names, [], [0.8, 0.3], 1.1),
            _make_trace("env", "s3", objective_names, [], [0.5, 0.5], 1.0),
            _make_trace("env", "s4", objective_names, [], [0.3, 0.8], 1.1),
            _make_trace("env", "s5", objective_names, [], [0.1, 0.9], 1.0),
        ]

        result = analyzer.analyze_tradeoffs(traces)

        assert isinstance(result, dict)
        assert "tradeoff_pairs" in result
        assert "correlation_matrix" in result
        assert "dominant_objectives" in result

        # With anti-correlated outcomes, we expect a tradeoff pair
        assert len(result["tradeoff_pairs"]) >= 1
        pair = result["tradeoff_pairs"][0]
        # Each pair is (obj_a, obj_b, correlation)
        assert len(pair) == 3
        assert pair[2] < -0.1  # Negative correlation

        # Correlation matrix should be 2x2
        corr = np.array(result["correlation_matrix"])
        assert corr.shape == (2, 2)
        # Diagonal should be 1.0
        np.testing.assert_allclose(np.diag(corr), [1.0, 1.0], atol=1e-6)

        # Dominant objectives should be a non-empty list of strings
        assert isinstance(result["dominant_objectives"], list)
        assert len(result["dominant_objectives"]) > 0


# ---------------------------------------------------------------------------
# TestSensitivityAnalyzer
# ---------------------------------------------------------------------------


class TestSensitivityAnalyzer:
    """Tests for SensitivityAnalyzer."""

    def test_analyze_sensitivity(self, objective_names: list[str]) -> None:
        """Test sensitivity analysis across grouped scenarios."""
        analyzer = SensitivityAnalyzer(objective_names)

        # Scenario A: efficiency is stable, equity varies
        traces_a = [
            _make_trace("env", "A", objective_names, [], [0.8, 0.2]),
            _make_trace("env", "A", objective_names, [], [0.8, 0.3]),
        ]
        # Scenario B: efficiency drops, equity improves
        traces_b = [
            _make_trace("env", "B", objective_names, [], [0.4, 0.7]),
            _make_trace("env", "B", objective_names, [], [0.5, 0.8]),
        ]
        # Scenario C: middle ground
        traces_c = [
            _make_trace("env", "C", objective_names, [], [0.6, 0.5]),
            _make_trace("env", "C", objective_names, [], [0.6, 0.5]),
        ]

        traces_by_scenario = {
            "scenario_A": traces_a,
            "scenario_B": traces_b,
            "scenario_C": traces_c,
        }

        result = analyzer.analyze_sensitivity(traces_by_scenario)

        assert isinstance(result, dict)
        assert "scenario_impacts" in result
        assert "robust_objectives" in result
        assert "sensitive_objectives" in result

        # Should have impacts for all 3 scenarios
        assert len(result["scenario_impacts"]) == 3
        assert "scenario_A" in result["scenario_impacts"]
        assert "scenario_B" in result["scenario_impacts"]
        assert "scenario_C" in result["scenario_impacts"]

        # Each scenario impact should be a list of 2 floats
        for name, impacts in result["scenario_impacts"].items():
            assert len(impacts) == 2

        # Robust + sensitive should cover all objectives
        all_classified = result["robust_objectives"] + result["sensitive_objectives"]
        assert set(all_classified) == set(objective_names)


# ---------------------------------------------------------------------------
# TestComparisonExplainer
# ---------------------------------------------------------------------------


class TestComparisonExplainer:
    """Tests for ComparisonExplainer."""

    def test_compare_strategies(self, objective_names: list[str]) -> None:
        """Test side-by-side comparison of two strategy traces."""
        comparer = ComparisonExplainer(objective_names)

        dps_a = [
            _make_decision_point(0, 0, "expand", [0.8, 0.3], 0.9),
            _make_decision_point(1, 1, "build", [0.7, 0.4], 0.7),
        ]
        dps_b = [
            _make_decision_point(0, 2, "defend", [0.4, 0.8], 0.6),
            _make_decision_point(1, 3, "trade", [0.5, 0.7], 0.8),
        ]

        trace_a = _make_trace(
            "env", "optimistic", objective_names, dps_a, [0.75, 0.35],
            total_reward=1.1, pareto_rank=1,
        )
        trace_b = _make_trace(
            "env", "pessimistic", objective_names, dps_b, [0.45, 0.75],
            total_reward=1.2, pareto_rank=0,
        )

        comparison = comparer.compare_strategies(trace_a, trace_b)

        assert isinstance(comparison, str)
        assert len(comparison) > 0

        # Should contain strategy labels and reward info
        assert "Strategy A" in comparison
        assert "Strategy B" in comparison

        # Should contain objective comparison
        assert "efficiency" in comparison
        assert "equity" in comparison

        # Should detect tradeoff (A better at efficiency, B better at equity)
        assert "Tradeoff" in comparison or "tradeoff" in comparison.lower()

        # Should mention Pareto ranks
        assert "Pareto" in comparison
