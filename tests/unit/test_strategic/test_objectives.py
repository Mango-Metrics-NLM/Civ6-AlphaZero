"""Tests for the multi-objective optimization module.

Validates ObjectiveSpec validation, scalarization strategies
(WeightedSum, Chebyshev, EpsilonConstraint), and the
MultiObjectiveResult container.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.strategic.objectives import (
    ChebyshevScalarizer,
    EpsilonConstraintScalarizer,
    MultiObjectiveResult,
    ObjectiveSpec,
    WeightedSumScalarizer,
)


class TestObjectiveSpec:
    """Test suite for ObjectiveSpec dataclass."""

    def test_creation(self) -> None:
        """ObjectiveSpec stores name, direction, weight, and optional bounds."""
        spec = ObjectiveSpec(
            name="efficiency",
            direction="maximize",
            weight=0.6,
            lower_bound=0.0,
            upper_bound=1.0,
            description="Resource utilization efficiency",
        )

        assert spec.name == "efficiency"
        assert spec.direction == "maximize"
        assert spec.weight == 0.6
        assert spec.lower_bound == 0.0
        assert spec.upper_bound == 1.0
        assert spec.description == "Resource utilization efficiency"

    def test_invalid_direction_raises(self) -> None:
        """ObjectiveSpec raises ValueError for invalid direction."""
        with pytest.raises(ValueError, match="Invalid direction"):
            ObjectiveSpec(name="bad", direction="up")

    def test_negative_weight_raises(self) -> None:
        """ObjectiveSpec raises ValueError for negative weight."""
        with pytest.raises(ValueError, match="Weight must be non-negative"):
            ObjectiveSpec(name="bad", weight=-0.5)

    def test_sign_maximize(self) -> None:
        """sign returns 1.0 for maximize objectives."""
        spec = ObjectiveSpec(name="profit", direction="maximize")
        assert spec.sign == 1.0

    def test_sign_minimize(self) -> None:
        """sign returns -1.0 for minimize objectives."""
        spec = ObjectiveSpec(name="cost", direction="minimize")
        assert spec.sign == -1.0

    def test_normalize_with_bounds(self) -> None:
        """normalize maps values to [0, 1] range using lower/upper bounds."""
        spec = ObjectiveSpec(
            name="temp",
            lower_bound=0.0,
            upper_bound=100.0,
        )

        assert spec.normalize(0.0) == pytest.approx(0.0)
        assert spec.normalize(50.0) == pytest.approx(0.5)
        assert spec.normalize(100.0) == pytest.approx(1.0)
        # Values outside bounds are clipped
        assert spec.normalize(-10.0) == pytest.approx(0.0)
        assert spec.normalize(200.0) == pytest.approx(1.0)

    def test_normalize_without_bounds(self) -> None:
        """normalize returns raw value when bounds are not set."""
        spec = ObjectiveSpec(name="raw")

        assert spec.normalize(42.0) == pytest.approx(42.0)
        assert spec.normalize(-5.0) == pytest.approx(-5.0)

    def test_normalize_partial_bounds(self) -> None:
        """normalize returns raw value when only one bound is set."""
        spec_lower_only = ObjectiveSpec(name="partial", lower_bound=0.0)
        assert spec_lower_only.normalize(50.0) == pytest.approx(50.0)

        spec_upper_only = ObjectiveSpec(name="partial", upper_bound=100.0)
        assert spec_upper_only.normalize(50.0) == pytest.approx(50.0)

    def test_to_from_dict(self) -> None:
        """ObjectiveSpec round-trips through dictionary serialization."""
        original = ObjectiveSpec(
            name="equity",
            direction="minimize",
            weight=0.4,
            lower_bound=-1.0,
            upper_bound=1.0,
            constraint_value=0.5,
            description="Equity index",
        )

        data = original.to_dict()
        restored = ObjectiveSpec.from_dict(data)

        assert restored.name == original.name
        assert restored.direction == original.direction
        assert restored.weight == pytest.approx(original.weight)
        assert restored.lower_bound == pytest.approx(original.lower_bound)
        assert restored.upper_bound == pytest.approx(original.upper_bound)
        assert restored.constraint_value == pytest.approx(original.constraint_value)
        assert restored.description == original.description

    def test_from_dict_defaults(self) -> None:
        """ObjectiveSpec.from_dict uses sensible defaults for missing keys."""
        data = {"name": "minimal"}
        spec = ObjectiveSpec.from_dict(data)

        assert spec.name == "minimal"
        assert spec.direction == "maximize"
        assert spec.weight == 1.0
        assert spec.lower_bound is None
        assert spec.upper_bound is None
        assert spec.constraint_value is None
        assert spec.description == ""


class TestWeightedSumScalarizer:
    """Test suite for WeightedSumScalarizer."""

    def test_equal_weights(self) -> None:
        """Equal-weight scalarization produces the mean of values."""
        objectives = [
            ObjectiveSpec(name="a", direction="maximize", weight=1.0),
            ObjectiveSpec(name="b", direction="maximize", weight=1.0),
        ]
        scalarizer = WeightedSumScalarizer(objectives)

        values = np.array([0.8, 0.4])
        result = scalarizer.scalarize(values)

        # Equal weights normalized: [0.5, 0.5]
        # scalar = 0.5 * 0.8 + 0.5 * 0.4 = 0.6
        assert result == pytest.approx(0.6)

    def test_unequal_weights(self) -> None:
        """Unequal weights shift the scalarized value toward the heavier objective."""
        objectives = [
            ObjectiveSpec(name="a", direction="maximize", weight=0.6),
            ObjectiveSpec(name="b", direction="maximize", weight=0.4),
        ]
        scalarizer = WeightedSumScalarizer(objectives)

        values = np.array([1.0, 0.0])
        result = scalarizer.scalarize(values)

        # Normalized weights: [0.6, 0.4]
        # scalar = 0.6 * 1.0 + 0.4 * 0.0 = 0.6
        assert result == pytest.approx(0.6)

    def test_minimize_direction(self) -> None:
        """Minimize objectives are negated before weighting."""
        objectives = [
            ObjectiveSpec(name="cost", direction="minimize", weight=1.0),
        ]
        scalarizer = WeightedSumScalarizer(objectives)

        # For minimize, sign = -1, so value 5.0 becomes -5.0
        result = scalarizer.scalarize(np.array([5.0]))
        assert result == pytest.approx(-5.0)

    def test_mixed_directions(self, objectives_3d: list[ObjectiveSpec]) -> None:
        """Mixed maximize/minimize objectives are handled correctly."""
        # objectives_3d: efficiency(max, 0.5), equity(max, 0.3), cost(min, 0.2)
        scalarizer = WeightedSumScalarizer(objectives_3d)

        values = np.array([0.8, 0.6, 2.0])
        result = scalarizer.scalarize(values)

        # Normalized weights: [0.5, 0.3, 0.2]
        # Signs: [1.0, 1.0, -1.0]
        # scalar = 0.5 * 0.8 + 0.3 * 0.6 + 0.2 * (-2.0)
        #        = 0.4 + 0.18 + (-0.4)
        #        = 0.18
        assert result == pytest.approx(0.18)

    def test_weights_are_normalized(self) -> None:
        """Scalarizer normalizes weights so they sum to 1."""
        objectives = [
            ObjectiveSpec(name="a", direction="maximize", weight=3.0),
            ObjectiveSpec(name="b", direction="maximize", weight=7.0),
        ]
        scalarizer = WeightedSumScalarizer(objectives)

        np.testing.assert_array_almost_equal(scalarizer.weights, [0.3, 0.7])

    def test_properties(self, objectives_2d: list[ObjectiveSpec]) -> None:
        """Scalarizer exposes objectives and num_objectives properties."""
        scalarizer = WeightedSumScalarizer(objectives_2d)

        assert scalarizer.num_objectives == 2
        assert len(scalarizer.objectives) == 2
        assert scalarizer.objectives[0].name == "efficiency"


class TestChebyshevScalarizer:
    """Test suite for ChebyshevScalarizer."""

    def test_ideal_point(self) -> None:
        """Chebyshev scalarizer uses provided ideal point."""
        objectives = [
            ObjectiveSpec(name="a", direction="maximize", weight=1.0),
            ObjectiveSpec(name="b", direction="maximize", weight=1.0),
        ]
        ideal = np.array([1.0, 1.0])
        scalarizer = ChebyshevScalarizer(objectives, ideal_point=ideal)

        np.testing.assert_array_equal(scalarizer.ideal_point, ideal)

        # At ideal point, all deviations are 0, so result is -0.0
        result = scalarizer.scalarize(np.array([1.0, 1.0]))
        assert result == pytest.approx(0.0, abs=1e-10)

        # Away from ideal: deviations weighted by normalized weights [0.5, 0.5]
        result_off = scalarizer.scalarize(np.array([0.5, 0.8]))
        # deviations = [0.5 * |1.0 - 0.5|, 0.5 * |1.0 - 0.8|] = [0.25, 0.1]
        # max deviation = 0.25, result = -0.25
        assert result_off == pytest.approx(-0.25)

    def test_default_ideal(self) -> None:
        """Default ideal point is ones vector."""
        objectives = [
            ObjectiveSpec(name="a", direction="maximize", weight=1.0),
            ObjectiveSpec(name="b", direction="maximize", weight=1.0),
        ]
        scalarizer = ChebyshevScalarizer(objectives)

        np.testing.assert_array_equal(
            scalarizer.ideal_point, np.ones(2, dtype=np.float64)
        )

    def test_update_ideal(self) -> None:
        """update_ideal_point changes the reference point."""
        objectives = [
            ObjectiveSpec(name="a", direction="maximize", weight=1.0),
        ]
        scalarizer = ChebyshevScalarizer(objectives)

        assert scalarizer.ideal_point[0] == pytest.approx(1.0)

        scalarizer.update_ideal_point(np.array([2.0]))
        assert scalarizer.ideal_point[0] == pytest.approx(2.0)

    def test_minimize_objective_in_chebyshev(self) -> None:
        """Chebyshev handles minimize objectives by negating values."""
        objectives = [
            ObjectiveSpec(name="cost", direction="minimize", weight=1.0),
        ]
        ideal = np.array([0.0])  # ideal is in maximization form: sign * value
        scalarizer = ChebyshevScalarizer(objectives, ideal_point=ideal)

        # For minimize: sign = -1, so value 0.0 -> maximized = 0.0
        result_at_ideal = scalarizer.scalarize(np.array([0.0]))
        assert result_at_ideal == pytest.approx(0.0, abs=1e-10)


class TestEpsilonConstraintScalarizer:
    """Test suite for EpsilonConstraintScalarizer."""

    def test_no_violations(self) -> None:
        """When no constraints are violated, returns the primary objective value."""
        objectives = [
            ObjectiveSpec(name="primary", direction="maximize", weight=1.0),
            ObjectiveSpec(
                name="secondary",
                direction="maximize",
                weight=1.0,
                constraint_value=0.3,
            ),
        ]
        scalarizer = EpsilonConstraintScalarizer(objectives, primary_index=0)

        # Primary = 0.8, secondary = 0.5 (above constraint 0.3)
        values = np.array([0.8, 0.5])
        result = scalarizer.scalarize(values)

        # No violation, so result = primary * sign = 0.8
        assert result == pytest.approx(0.8)

    def test_with_violations(self) -> None:
        """Constraint violations reduce the scalarized value by the penalty amount."""
        objectives = [
            ObjectiveSpec(name="primary", direction="maximize", weight=1.0),
            ObjectiveSpec(
                name="constrained",
                direction="maximize",
                weight=1.0,
                constraint_value=0.8,
            ),
        ]
        scalarizer = EpsilonConstraintScalarizer(
            objectives, primary_index=0, constraint_penalty=10.0
        )

        # Primary = 1.0, constrained = 0.5 (below constraint 0.8)
        values = np.array([1.0, 0.5])
        result = scalarizer.scalarize(values)

        # Violation = 0.8 * 1.0 - 0.5 = 0.3 (constraint_value * sign - maximized)
        # Penalty = 10.0 * 0.3 = 3.0
        # Result = 1.0 - 3.0 = -2.0
        assert result == pytest.approx(-2.0)

    def test_invalid_primary_index(self) -> None:
        """EpsilonConstraintScalarizer raises on out-of-range primary_index."""
        objectives = [
            ObjectiveSpec(name="a", direction="maximize"),
            ObjectiveSpec(name="b", direction="maximize"),
        ]

        with pytest.raises(ValueError, match="primary_index"):
            EpsilonConstraintScalarizer(objectives, primary_index=2)

        with pytest.raises(ValueError, match="primary_index"):
            EpsilonConstraintScalarizer(objectives, primary_index=-1)

    def test_primary_index_property(self) -> None:
        """primary_index and constraint_penalty are accessible as properties."""
        objectives = [
            ObjectiveSpec(name="a", direction="maximize"),
            ObjectiveSpec(name="b", direction="maximize"),
        ]
        scalarizer = EpsilonConstraintScalarizer(
            objectives, primary_index=1, constraint_penalty=5.0
        )

        assert scalarizer.primary_index == 1
        assert scalarizer.constraint_penalty == pytest.approx(5.0)

    def test_no_constraint_value_means_no_penalty(self) -> None:
        """Objectives without constraint_value are not penalized."""
        objectives = [
            ObjectiveSpec(name="primary", direction="maximize"),
            ObjectiveSpec(
                name="unconstrained",
                direction="maximize",
                # constraint_value is None (default)
            ),
        ]
        scalarizer = EpsilonConstraintScalarizer(objectives, primary_index=0)

        values = np.array([0.5, -100.0])
        result = scalarizer.scalarize(values)

        # No constraint on secondary, so result is just the primary value
        assert result == pytest.approx(0.5)


class TestMultiObjectiveResult:
    """Test suite for MultiObjectiveResult container."""

    def test_to_dict(self) -> None:
        """MultiObjectiveResult serializes to a well-structured dictionary."""
        result = MultiObjectiveResult(
            objective_values=np.array([0.8, 0.6, -0.3]),
            scalarized_value=0.45,
            objective_names=["efficiency", "equity", "cost"],
            constraint_violations=np.array([0.0, 0.0, 0.1]),
        )

        data = result.to_dict()

        assert data["scalarized_value"] == pytest.approx(0.45)
        assert "objectives" in data
        assert data["objectives"]["efficiency"] == pytest.approx(0.8)
        assert data["objectives"]["equity"] == pytest.approx(0.6)
        assert data["objectives"]["cost"] == pytest.approx(-0.3)
        assert "constraint_violations" in data
        assert len(data["constraint_violations"]) == 3

    def test_to_dict_no_violations(self) -> None:
        """to_dict omits constraint_violations when empty."""
        result = MultiObjectiveResult(
            objective_values=np.array([0.5, 0.5]),
            scalarized_value=0.5,
            objective_names=["a", "b"],
        )

        data = result.to_dict()

        assert data["scalarized_value"] == pytest.approx(0.5)
        assert data["objectives"]["a"] == pytest.approx(0.5)
        assert "constraint_violations" not in data

    def test_to_dict_empty_names(self) -> None:
        """to_dict handles empty objective_names gracefully."""
        result = MultiObjectiveResult(
            objective_values=np.array([1.0]),
            scalarized_value=1.0,
            objective_names=[],
        )

        data = result.to_dict()
        assert data["objectives"] == {}
        assert data["scalarized_value"] == pytest.approx(1.0)
