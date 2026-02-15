"""Unit tests for Pareto frontier utilities.

Tests the ParetoPoint and ParetoFrontier classes from the
multi-objective optimization module, covering point creation,
serialization, dominance checks, frontier management, hypervolume
computation, and crowding distance calculation.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.strategic.mcts.pareto import ParetoFrontier, ParetoPoint


class TestParetoPoint:
    """Tests for the ParetoPoint data class."""

    def test_creation(self) -> None:
        """Test that a ParetoPoint can be created with values, metadata, and strategy_id."""
        values = np.array([0.8, 0.3])
        point = ParetoPoint(
            values=values,
            metadata={"name": "aggressive"},
            strategy_id="strat_001",
        )

        np.testing.assert_array_almost_equal(point.values, [0.8, 0.3])
        assert point.metadata == {"name": "aggressive"}
        assert point.strategy_id == "strat_001"

    def test_creation_auto_converts_list_to_array(self) -> None:
        """Test that list values are automatically converted to numpy arrays."""
        point = ParetoPoint(values=[1.0, 2.0, 3.0])

        assert isinstance(point.values, np.ndarray)
        assert point.values.dtype == np.float64
        np.testing.assert_array_equal(point.values, [1.0, 2.0, 3.0])

    def test_creation_generates_uuid_strategy_id(self) -> None:
        """Test that a UUID is generated for strategy_id when not provided."""
        point = ParetoPoint(values=np.array([0.5]))
        assert isinstance(point.strategy_id, str)
        assert len(point.strategy_id) > 0

    def test_to_from_dict(self) -> None:
        """Test round-trip serialization and deserialization of a ParetoPoint."""
        original = ParetoPoint(
            values=np.array([0.9, 0.4, 0.7]),
            metadata={"score": 42, "label": "test"},
            strategy_id="round_trip_test",
        )

        data = original.to_dict()

        assert isinstance(data, dict)
        assert "values" in data
        assert "metadata" in data
        assert "strategy_id" in data
        assert data["strategy_id"] == "round_trip_test"
        assert data["values"] == pytest.approx([0.9, 0.4, 0.7])

        restored = ParetoPoint.from_dict(data)

        np.testing.assert_array_almost_equal(restored.values, original.values)
        assert restored.metadata == original.metadata
        assert restored.strategy_id == original.strategy_id


class TestParetoFrontier:
    """Tests for the ParetoFrontier class."""

    def test_empty_frontier(self) -> None:
        """Test that a newly created frontier is empty."""
        frontier = ParetoFrontier(num_objectives=2)

        assert frontier.size == 0
        assert frontier.num_objectives == 2
        assert frontier.get_frontier() == []

    def test_invalid_num_objectives_raises(self) -> None:
        """Test that creating a frontier with num_objectives < 1 raises ValueError."""
        with pytest.raises(ValueError, match="num_objectives must be >= 1"):
            ParetoFrontier(num_objectives=0)

    def test_add_non_dominated(self) -> None:
        """Test that non-dominated points are added to the frontier."""
        frontier = ParetoFrontier(num_objectives=2)

        p1 = ParetoPoint(values=np.array([0.8, 0.3]), strategy_id="s1")
        p2 = ParetoPoint(values=np.array([0.5, 0.7]), strategy_id="s2")

        assert frontier.add(p1) is True
        assert frontier.size == 1

        assert frontier.add(p2) is True
        assert frontier.size == 2

    def test_dominated_not_added(self) -> None:
        """Test that a dominated point is rejected from the frontier."""
        frontier = ParetoFrontier(num_objectives=2)

        # s1 dominates s3 because (0.8 > 0.4) and (0.3 > 0.2)
        p1 = ParetoPoint(values=np.array([0.8, 0.3]), strategy_id="s1")
        p3 = ParetoPoint(values=np.array([0.4, 0.2]), strategy_id="s3")

        assert frontier.add(p1) is True
        assert frontier.add(p3) is False
        assert frontier.size == 1

    def test_dominates_static(self) -> None:
        """Test the static dominates method for pairwise dominance checking."""
        # a dominates b: a >= b in all, a > b in at least one
        a = np.array([0.8, 0.5])
        b = np.array([0.7, 0.4])
        assert ParetoFrontier.dominates(a, b) is True
        assert ParetoFrontier.dominates(b, a) is False

        # Equal points: neither dominates the other
        c = np.array([0.5, 0.5])
        assert ParetoFrontier.dominates(c, c) is False

        # Trade-off: neither dominates
        d = np.array([0.9, 0.1])
        e = np.array([0.1, 0.9])
        assert ParetoFrontier.dominates(d, e) is False
        assert ParetoFrontier.dominates(e, d) is False

    def test_get_frontier_is_copy(self) -> None:
        """Test that get_frontier returns a copy, not the internal list."""
        frontier = ParetoFrontier(num_objectives=2)
        frontier.add(ParetoPoint(values=np.array([0.5, 0.5]), strategy_id="s1"))

        points = frontier.get_frontier()
        assert len(points) == 1

        # Modifying the returned list should not affect the frontier
        points.clear()
        assert frontier.size == 1

    def test_size_property(self) -> None:
        """Test that the size property correctly tracks the number of frontier points."""
        frontier = ParetoFrontier(num_objectives=2)
        assert frontier.size == 0

        frontier.add(ParetoPoint(values=np.array([0.1, 0.9])))
        assert frontier.size == 1

        frontier.add(ParetoPoint(values=np.array([0.9, 0.1])))
        assert frontier.size == 2

        frontier.add(ParetoPoint(values=np.array([0.5, 0.5])))
        assert frontier.size == 3

    def test_add_removes_newly_dominated(self) -> None:
        """Test that adding a dominating point removes previously non-dominated points."""
        frontier = ParetoFrontier(num_objectives=2)

        # Add two non-dominating points
        frontier.add(ParetoPoint(values=np.array([0.3, 0.2]), strategy_id="weak1"))
        frontier.add(ParetoPoint(values=np.array([0.2, 0.3]), strategy_id="weak2"))
        assert frontier.size == 2

        # Add a point that dominates both
        frontier.add(ParetoPoint(values=np.array([0.9, 0.9]), strategy_id="strong"))
        assert frontier.size == 1

        remaining = frontier.get_frontier()
        assert remaining[0].strategy_id == "strong"

    def test_hypervolume_2d(self) -> None:
        """Test 2D hypervolume computation with known points and reference."""
        frontier = ParetoFrontier(num_objectives=2)

        # Add two points forming a step-shaped frontier
        frontier.add(ParetoPoint(values=np.array([3.0, 1.0]), strategy_id="a"))
        frontier.add(ParetoPoint(values=np.array([1.0, 3.0]), strategy_id="b"))

        reference_point = np.array([0.0, 0.0])
        hv = frontier.hypervolume(reference_point)

        # Expected hypervolume:
        # The frontier has two points: (3,1) and (1,3)
        # Sort by first obj descending: [(3,1), (1,3)]
        # Sweep: prev_y starts at 0
        #   Point (3,1): x_contrib = 3-0 = 3, y_contrib = 1-0 = 1 -> area = 3
        #   Point (1,3): x_contrib = 1-0 = 1, y_contrib = 3-1 = 2 -> area = 2
        # Total = 5
        assert hv == pytest.approx(5.0)

    def test_hypervolume_empty_frontier(self) -> None:
        """Test that hypervolume of an empty frontier is zero."""
        frontier = ParetoFrontier(num_objectives=2)
        hv = frontier.hypervolume(np.array([0.0, 0.0]))
        assert hv == 0.0

    def test_hypervolume_single_point(self) -> None:
        """Test hypervolume with a single point."""
        frontier = ParetoFrontier(num_objectives=2)
        frontier.add(ParetoPoint(values=np.array([2.0, 3.0])))

        hv = frontier.hypervolume(np.array([0.0, 0.0]))
        # Area = (2-0) * (3-0) = 6
        assert hv == pytest.approx(6.0)

    def test_hypervolume_returns_zero_for_high_dimensions(self) -> None:
        """Test that hypervolume returns 0 for >2 objectives (not implemented)."""
        frontier = ParetoFrontier(num_objectives=3)
        frontier.add(ParetoPoint(values=np.array([1.0, 1.0, 1.0])))
        hv = frontier.hypervolume(np.array([0.0, 0.0, 0.0]))
        assert hv == 0.0

    def test_crowding_distances(self) -> None:
        """Test NSGA-II crowding distance computation."""
        frontier = ParetoFrontier(num_objectives=2)

        # Add three points along the frontier
        frontier.add(ParetoPoint(values=np.array([0.0, 1.0]), strategy_id="p1"))
        frontier.add(ParetoPoint(values=np.array([0.5, 0.5]), strategy_id="p2"))
        frontier.add(ParetoPoint(values=np.array([1.0, 0.0]), strategy_id="p3"))

        distances = frontier.crowding_distances()
        assert len(distances) == 3

        # Boundary points (p1 and p3) should have infinite distance
        # p1 and p3 are at the extremes of at least one objective
        assert distances[0] == float("inf")  # p1 is extreme in obj2
        assert distances[2] == float("inf")  # p3 is extreme in obj1

        # Interior point (p2) should have finite distance
        assert distances[1] < float("inf")
        assert distances[1] > 0.0

    def test_crowding_distances_empty(self) -> None:
        """Test that crowding distances for an empty frontier returns empty list."""
        frontier = ParetoFrontier(num_objectives=2)
        assert frontier.crowding_distances() == []

    def test_crowding_distances_one_point(self) -> None:
        """Test that a single point gets infinite crowding distance."""
        frontier = ParetoFrontier(num_objectives=2)
        frontier.add(ParetoPoint(values=np.array([0.5, 0.5])))
        distances = frontier.crowding_distances()
        assert len(distances) == 1
        assert distances[0] == float("inf")

    def test_crowding_distances_two_points(self) -> None:
        """Test that two points both get infinite crowding distance."""
        frontier = ParetoFrontier(num_objectives=2)
        frontier.add(ParetoPoint(values=np.array([0.0, 1.0])))
        frontier.add(ParetoPoint(values=np.array([1.0, 0.0])))
        distances = frontier.crowding_distances()
        assert len(distances) == 2
        assert all(d == float("inf") for d in distances)

    def test_to_from_dict(self) -> None:
        """Test round-trip serialization and deserialization of a ParetoFrontier."""
        frontier = ParetoFrontier(num_objectives=2)
        frontier.add(ParetoPoint(values=np.array([0.8, 0.3]), strategy_id="s1"))
        frontier.add(ParetoPoint(values=np.array([0.5, 0.7]), strategy_id="s2"))

        data = frontier.to_dict()
        assert data["num_objectives"] == 2
        assert len(data["points"]) == 2

        restored = ParetoFrontier.from_dict(data)
        assert restored.num_objectives == 2
        assert restored.size == 2

        original_points = frontier.get_frontier()
        restored_points = restored.get_frontier()

        for orig, rest in zip(original_points, restored_points):
            np.testing.assert_array_almost_equal(rest.values, orig.values)
            assert rest.strategy_id == orig.strategy_id

    def test_dimension_mismatch_raises(self) -> None:
        """Test that adding a point with wrong dimension raises ValueError."""
        frontier = ParetoFrontier(num_objectives=2)
        point = ParetoPoint(values=np.array([0.5, 0.5, 0.5]))

        with pytest.raises(ValueError, match="3 objectives.*expected 2"):
            frontier.add(point)

    def test_repr(self) -> None:
        """Test the string representation of a ParetoFrontier."""
        frontier = ParetoFrontier(num_objectives=3)
        assert "num_objectives=3" in repr(frontier)
        assert "size=0" in repr(frontier)
