"""Unit tests for the decision trace recording system.

Tests decision trace components including:
- ActionAlternative creation and serialization
- DecisionPoint creation and serialization
- DecisionTrace lifecycle, JSON serialization, and analysis utilities
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from alpha_zero.strategic.trace.decision_trace import (
    ActionAlternative,
    DecisionPoint,
    DecisionTrace,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_alternative() -> ActionAlternative:
    """Create a sample ActionAlternative for testing."""
    return ActionAlternative(
        action_index=2,
        action_name="invest_education",
        visit_count=120,
        prior_probability=0.35,
        value_estimate=np.array([0.7, 0.5]),
    )


@pytest.fixture
def sample_alternatives() -> list[ActionAlternative]:
    """Create a list of sample ActionAlternatives for testing."""
    return [
        ActionAlternative(
            action_index=0,
            action_name="invest_health",
            visit_count=150,
            prior_probability=0.4,
            value_estimate=np.array([0.8, 0.6]),
        ),
        ActionAlternative(
            action_index=1,
            action_name="invest_defense",
            visit_count=80,
            prior_probability=0.2,
            value_estimate=np.array([0.3, 0.9]),
        ),
        ActionAlternative(
            action_index=2,
            action_name="invest_education",
            visit_count=120,
            prior_probability=0.35,
            value_estimate=np.array([0.7, 0.5]),
        ),
    ]


@pytest.fixture
def sample_decision_point(sample_alternatives: list[ActionAlternative]) -> DecisionPoint:
    """Create a sample DecisionPoint for testing."""
    return DecisionPoint(
        step=3,
        agent_id=0,
        state_summary={"budget": 85.0, "turn": 3},
        action_taken=0,
        action_name="invest_health",
        alternatives_considered=sample_alternatives,
        objective_values=np.array([0.8, 0.6]),
        mcts_visit_counts=np.array([150, 80, 120]),
        value_estimate=np.array([0.75, 0.55]),
        confidence=0.85,
        reasoning="Health investment has highest expected efficiency.",
    )


@pytest.fixture
def sample_trace(sample_decision_point: DecisionPoint) -> DecisionTrace:
    """Create a sample DecisionTrace with one decision point."""
    trace = DecisionTrace(
        environment_name="resource_allocation",
        scenario_name="baseline",
        objective_names=["efficiency", "equity"],
        final_outcomes=np.array([0.82, 0.61]),
        pareto_rank=0,
        total_reward=1.43,
        metadata={"config": "fast", "seed": 42},
    )
    trace.add_decision_point(sample_decision_point)
    return trace


# ---------------------------------------------------------------------------
# TestActionAlternative
# ---------------------------------------------------------------------------


class TestActionAlternative:
    """Tests for ActionAlternative dataclass."""

    def test_creation(self, sample_alternative: ActionAlternative) -> None:
        """Test creating an ActionAlternative with all fields."""
        alt = sample_alternative
        assert alt.action_index == 2
        assert alt.action_name == "invest_education"
        assert alt.visit_count == 120
        assert alt.prior_probability == pytest.approx(0.35)
        assert alt.value_estimate.shape == (2,)
        np.testing.assert_allclose(alt.value_estimate, [0.7, 0.5])

    def test_to_from_dict(self, sample_alternative: ActionAlternative) -> None:
        """Test round-trip serialization through dictionary."""
        d = sample_alternative.to_dict()

        assert isinstance(d, dict)
        assert d["action_index"] == 2
        assert d["action_name"] == "invest_education"
        assert d["visit_count"] == 120
        assert d["prior_probability"] == pytest.approx(0.35)
        assert isinstance(d["value_estimate"], list)

        restored = ActionAlternative.from_dict(d)
        assert restored.action_index == sample_alternative.action_index
        assert restored.action_name == sample_alternative.action_name
        assert restored.visit_count == sample_alternative.visit_count
        assert restored.prior_probability == pytest.approx(
            sample_alternative.prior_probability
        )
        np.testing.assert_allclose(
            restored.value_estimate, sample_alternative.value_estimate
        )


# ---------------------------------------------------------------------------
# TestDecisionPoint
# ---------------------------------------------------------------------------


class TestDecisionPoint:
    """Tests for DecisionPoint dataclass."""

    def test_creation(self, sample_decision_point: DecisionPoint) -> None:
        """Test creating a DecisionPoint with all fields."""
        dp = sample_decision_point
        assert dp.step == 3
        assert dp.agent_id == 0
        assert dp.state_summary == {"budget": 85.0, "turn": 3}
        assert dp.action_taken == 0
        assert dp.action_name == "invest_health"
        assert len(dp.alternatives_considered) == 3
        assert dp.objective_values.shape == (2,)
        assert dp.mcts_visit_counts.shape == (3,)
        assert dp.value_estimate.shape == (2,)
        assert dp.confidence == pytest.approx(0.85)
        assert dp.reasoning == "Health investment has highest expected efficiency."

    def test_to_from_dict(self, sample_decision_point: DecisionPoint) -> None:
        """Test round-trip serialization through dictionary."""
        d = sample_decision_point.to_dict()

        assert isinstance(d, dict)
        assert d["step"] == 3
        assert d["agent_id"] == 0
        assert d["action_taken"] == 0
        assert d["action_name"] == "invest_health"
        assert len(d["alternatives_considered"]) == 3
        assert isinstance(d["objective_values"], list)
        assert isinstance(d["mcts_visit_counts"], list)
        assert d["confidence"] == pytest.approx(0.85)
        assert d["reasoning"] == "Health investment has highest expected efficiency."

        restored = DecisionPoint.from_dict(d)
        assert restored.step == sample_decision_point.step
        assert restored.agent_id == sample_decision_point.agent_id
        assert restored.action_taken == sample_decision_point.action_taken
        assert restored.action_name == sample_decision_point.action_name
        assert len(restored.alternatives_considered) == 3
        np.testing.assert_allclose(
            restored.objective_values, sample_decision_point.objective_values
        )
        np.testing.assert_allclose(
            restored.mcts_visit_counts, sample_decision_point.mcts_visit_counts
        )
        np.testing.assert_allclose(
            restored.value_estimate, sample_decision_point.value_estimate
        )
        assert restored.confidence == pytest.approx(sample_decision_point.confidence)
        assert restored.reasoning == sample_decision_point.reasoning


# ---------------------------------------------------------------------------
# TestDecisionTrace
# ---------------------------------------------------------------------------


class TestDecisionTrace:
    """Tests for DecisionTrace dataclass."""

    def test_empty_trace(self) -> None:
        """Test creating an empty DecisionTrace with defaults."""
        trace = DecisionTrace(environment_name="test_env")

        assert trace.environment_name == "test_env"
        assert trace.scenario_name == "default"
        assert trace.objective_names == []
        assert trace.decision_points == []
        assert len(trace.final_outcomes) == 0
        assert trace.pareto_rank == -1
        assert trace.total_reward == 0.0
        assert trace.metadata == {}
        assert trace.num_steps == 0

    def test_add_decision_point(
        self, sample_decision_point: DecisionPoint
    ) -> None:
        """Test adding decision points incrementally."""
        trace = DecisionTrace(
            environment_name="test_env",
            objective_names=["efficiency", "equity"],
        )
        assert trace.num_steps == 0

        trace.add_decision_point(sample_decision_point)
        assert trace.num_steps == 1
        assert trace.decision_points[0] is sample_decision_point

        # Add a second point
        dp2 = DecisionPoint(
            step=4,
            agent_id=0,
            state_summary={"budget": 70.0},
            action_taken=1,
            action_name="invest_defense",
            alternatives_considered=[],
            objective_values=np.array([0.6, 0.7]),
            mcts_visit_counts=np.array([100, 90]),
            value_estimate=np.array([0.65, 0.60]),
            confidence=0.55,
        )
        trace.add_decision_point(dp2)
        assert trace.num_steps == 2

    def test_num_steps(self, sample_trace: DecisionTrace) -> None:
        """Test num_steps property reflects actual decision point count."""
        assert sample_trace.num_steps == 1
        assert sample_trace.num_steps == len(sample_trace.decision_points)

    def test_to_from_json(self, sample_trace: DecisionTrace) -> None:
        """Test round-trip JSON serialization and deserialization."""
        json_str = sample_trace.to_json()

        # Verify it is valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert parsed["environment_name"] == "resource_allocation"

        # Restore and compare
        restored = DecisionTrace.from_json(json_str)
        assert restored.environment_name == sample_trace.environment_name
        assert restored.scenario_name == sample_trace.scenario_name
        assert restored.objective_names == sample_trace.objective_names
        assert restored.num_steps == sample_trace.num_steps
        assert restored.pareto_rank == sample_trace.pareto_rank
        assert restored.total_reward == pytest.approx(sample_trace.total_reward)
        np.testing.assert_allclose(
            restored.final_outcomes, sample_trace.final_outcomes
        )
        assert restored.metadata == sample_trace.metadata

        # Verify nested decision point survived
        orig_dp = sample_trace.decision_points[0]
        rest_dp = restored.decision_points[0]
        assert rest_dp.step == orig_dp.step
        assert rest_dp.action_name == orig_dp.action_name
        assert len(rest_dp.alternatives_considered) == len(
            orig_dp.alternatives_considered
        )

    def test_to_dict_from_dict(self, sample_trace: DecisionTrace) -> None:
        """Test round-trip dictionary serialization and deserialization."""
        d = sample_trace.to_dict()

        assert isinstance(d, dict)
        assert d["environment_name"] == "resource_allocation"
        assert d["scenario_name"] == "baseline"
        assert d["pareto_rank"] == 0
        assert len(d["decision_points"]) == 1

        restored = DecisionTrace.from_dict(d)
        assert restored.environment_name == sample_trace.environment_name
        assert restored.scenario_name == sample_trace.scenario_name
        assert restored.num_steps == sample_trace.num_steps
        np.testing.assert_allclose(
            restored.final_outcomes, sample_trace.final_outcomes
        )

    def test_summary(self, sample_trace: DecisionTrace) -> None:
        """Test summary produces a non-empty human-readable string."""
        summary = sample_trace.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should contain key information
        assert "resource_allocation" in summary
        assert "baseline" in summary
        assert "Steps: 1" in summary
        assert "Total reward:" in summary
        assert "Pareto rank:" in summary
        # Should mention objectives
        assert "efficiency" in summary
        assert "equity" in summary

    def test_to_dataframe(self, sample_trace: DecisionTrace) -> None:
        """Test conversion to flat dictionaries for DataFrame creation."""
        rows = sample_trace.to_dataframe()

        assert isinstance(rows, list)
        assert len(rows) == 1  # One decision point -> one row

        row = rows[0]
        assert row["environment"] == "resource_allocation"
        assert row["scenario"] == "baseline"
        assert row["step"] == 3
        assert row["agent_id"] == 0
        assert row["action_taken"] == 0
        assert row["action_name"] == "invest_health"
        assert row["confidence"] == pytest.approx(0.85)
        assert row["num_alternatives"] == 3

        # Per-objective columns should be expanded
        assert "objective_efficiency" in row
        assert "objective_equity" in row
        assert row["objective_efficiency"] == pytest.approx(0.8)
        assert row["objective_equity"] == pytest.approx(0.6)

        # Value estimate columns
        assert "value_estimate_efficiency" in row
        assert "value_estimate_equity" in row

        # Total visits
        assert row["total_visits"] == 350  # 150 + 80 + 120

        # Flattened state summary
        assert row["state_budget"] == pytest.approx(85.0)
        assert row["state_turn"] == 3

    def test_to_dataframe_empty_trace(self) -> None:
        """Test to_dataframe returns empty list for a trace with no decisions."""
        trace = DecisionTrace(environment_name="empty")
        rows = trace.to_dataframe()
        assert rows == []
