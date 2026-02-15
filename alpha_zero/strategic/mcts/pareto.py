"""Pareto frontier utilities for multi-objective optimization.

This module provides data structures and algorithms for maintaining and
analyzing Pareto frontiers during multi-objective MCTS search. A Pareto
frontier is the set of non-dominated solutions -- points where no other
solution is better in all objectives simultaneously.

Key concepts:
- **Dominance**: Solution A dominates B if A is at least as good in all
  objectives and strictly better in at least one.
- **Pareto frontier**: The set of all non-dominated solutions.
- **Hypervolume**: The volume of objective space dominated by the frontier
  (measures frontier quality).
- **Crowding distance**: NSGA-II metric for diversity along the frontier.

Example:
    >>> frontier = ParetoFrontier(num_objectives=2)
    >>> frontier.add(ParetoPoint(values=np.array([0.8, 0.3]), metadata={}, strategy_id="s1"))
    True
    >>> frontier.add(ParetoPoint(values=np.array([0.5, 0.7]), metadata={}, strategy_id="s2"))
    True
    >>> frontier.add(ParetoPoint(values=np.array([0.4, 0.2]), metadata={}, strategy_id="s3"))
    False  # dominated by s1
    >>> frontier.size
    2
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ParetoPoint:
    """A single point in the multi-objective value space.

    Represents one candidate solution with its objective values,
    arbitrary metadata, and a unique strategy identifier.

    Attributes:
        values: Array of objective values, one per objective.
            Higher values are assumed to be better (maximization).
        metadata: Arbitrary metadata associated with this point
            (e.g., the strategy parameters, search statistics).
        strategy_id: Unique identifier for the strategy that
            produced this point.
    """

    values: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        """Ensure values is a numpy array."""
        if not isinstance(self.values, np.ndarray):
            self.values = np.asarray(self.values, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "values": self.values.tolist(),
            "metadata": self.metadata,
            "strategy_id": self.strategy_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParetoPoint:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with "values", "metadata", and "strategy_id" keys.

        Returns:
            Reconstructed ParetoPoint.
        """
        return cls(
            values=np.asarray(data["values"], dtype=np.float64),
            metadata=data.get("metadata", {}),
            strategy_id=data.get("strategy_id", str(uuid.uuid4())),
        )


class ParetoFrontier:
    """Maintains and analyzes a Pareto frontier of non-dominated solutions.

    The frontier automatically removes dominated points when new
    non-dominated points are added, ensuring it always contains
    only the Pareto-optimal set.

    All objectives are assumed to be maximized. For minimization
    objectives, negate the values before adding them.

    Args:
        num_objectives: Number of objectives (dimensionality of value space).

    Example:
        >>> frontier = ParetoFrontier(num_objectives=2)
        >>> added = frontier.add(ParetoPoint(
        ...     values=np.array([0.9, 0.4]),
        ...     metadata={"name": "aggressive"},
        ...     strategy_id="strat_001",
        ... ))
        >>> print(f"Added: {added}, Size: {frontier.size}")
        Added: True, Size: 1
    """

    def __init__(self, num_objectives: int) -> None:
        """Initialize an empty Pareto frontier.

        Args:
            num_objectives: Number of objectives. Must be >= 1.

        Raises:
            ValueError: If num_objectives < 1.
        """
        if num_objectives < 1:
            raise ValueError(
                f"num_objectives must be >= 1, got {num_objectives}"
            )
        self._num_objectives = num_objectives
        self._points: list[ParetoPoint] = []

    @property
    def num_objectives(self) -> int:
        """Number of objectives in the value space."""
        return self._num_objectives

    @property
    def size(self) -> int:
        """Number of points currently on the frontier."""
        return len(self._points)

    def add(self, point: ParetoPoint) -> bool:
        """Add a point to the frontier if it is not dominated.

        If the new point dominates any existing points, those dominated
        points are removed from the frontier. If the new point is
        dominated by any existing point, it is not added.

        Args:
            point: The candidate point to add.

        Returns:
            True if the point was added to the frontier, False if it
            was dominated by an existing point.

        Raises:
            ValueError: If the point's dimension does not match num_objectives.
        """
        if len(point.values) != self._num_objectives:
            raise ValueError(
                f"Point has {len(point.values)} objectives, "
                f"expected {self._num_objectives}"
            )

        # Check if the new point is dominated by any existing point
        if self.is_dominated(point.values):
            logger.debug(
                "Point %s is dominated, not added to frontier",
                point.strategy_id,
            )
            return False

        # Remove any existing points that the new point dominates
        non_dominated = [
            p for p in self._points
            if not ParetoFrontier.dominates(point.values, p.values)
        ]
        removed_count = len(self._points) - len(non_dominated)
        if removed_count > 0:
            logger.debug(
                "New point %s dominates %d existing points",
                point.strategy_id,
                removed_count,
            )

        self._points = non_dominated
        self._points.append(point)
        logger.debug(
            "Added point %s to frontier (size: %d)",
            point.strategy_id,
            self.size,
        )
        return True

    def is_dominated(self, values: np.ndarray) -> bool:
        """Check if a value vector is dominated by any point on the frontier.

        A value vector is dominated if there exists a point on the frontier
        that is at least as good in all objectives and strictly better in
        at least one.

        Args:
            values: Array of objective values to check.

        Returns:
            True if the values are dominated by an existing frontier point.
        """
        values = np.asarray(values, dtype=np.float64)
        for point in self._points:
            if ParetoFrontier.dominates(point.values, values):
                return True
        return False

    @staticmethod
    def dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """Check if solution a dominates solution b.

        Solution a dominates b if a is at least as good as b in all
        objectives and strictly better in at least one objective.
        (Assuming maximization for all objectives.)

        Args:
            a: Objective values of the potentially dominating solution.
            b: Objective values of the potentially dominated solution.

        Returns:
            True if a dominates b.
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return bool(np.all(a >= b) and np.any(a > b))

    def get_frontier(self) -> list[ParetoPoint]:
        """Return a copy of the current Pareto frontier points.

        Returns:
            List of ParetoPoint objects on the frontier, in insertion order.
        """
        return list(self._points)

    def hypervolume(self, reference_point: np.ndarray) -> float:
        """Compute the hypervolume indicator for the frontier.

        The hypervolume indicator measures the volume of objective space
        that is dominated by the frontier and bounded by a reference point.
        Larger hypervolume indicates a better frontier.

        Currently implements exact computation for 2D objectives.
        For higher dimensions, returns 0.0 with a warning (a proper
        high-dimensional algorithm like WFG would be needed).

        Args:
            reference_point: The reference (anti-ideal) point that bounds
                the hypervolume computation. Should be dominated by all
                frontier points (i.e., worse than all points in all objectives).

        Returns:
            The hypervolume indicator value. Returns 0.0 if the frontier
            is empty or if dimensions > 2.
        """
        reference_point = np.asarray(reference_point, dtype=np.float64)

        if self.size == 0:
            return 0.0

        if self._num_objectives != 2:
            logger.warning(
                "Hypervolume computation is only implemented for 2 objectives. "
                "Got %d objectives; returning 0.0.",
                self._num_objectives,
            )
            return 0.0

        # 2D hypervolume: sort points by first objective descending,
        # sweep along second objective
        points = [p.values for p in self._points]

        # Filter points that actually dominate the reference point
        valid_points = [
            p for p in points
            if p[0] > reference_point[0] and p[1] > reference_point[1]
        ]

        if not valid_points:
            return 0.0

        # Sort by first objective in descending order
        sorted_points = sorted(valid_points, key=lambda p: -p[0])

        hv = 0.0
        prev_y = reference_point[1]

        for point in sorted_points:
            x_contrib = point[0] - reference_point[0]
            y_contrib = point[1] - prev_y
            if y_contrib > 0:
                hv += x_contrib * y_contrib
                prev_y = point[1]

        return float(hv)

    def crowding_distances(self) -> list[float]:
        """Compute NSGA-II crowding distances for all frontier points.

        Crowding distance measures how spread out each point is relative
        to its neighbors along each objective. Points at the extremes of
        each objective receive infinite distance. Used for diversity
        preservation in selection.

        Returns:
            List of crowding distances, one per frontier point, in the
            same order as get_frontier(). Returns empty list if frontier
            has fewer than 2 points.
        """
        n = self.size
        if n == 0:
            return []
        if n <= 2:
            return [float("inf")] * n

        distances = [0.0] * n

        for obj_idx in range(self._num_objectives):
            # Sort indices by this objective's value
            sorted_indices = sorted(
                range(n),
                key=lambda i: self._points[i].values[obj_idx],
            )

            # Boundary points get infinite distance
            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            # Compute range for normalization
            obj_min = self._points[sorted_indices[0]].values[obj_idx]
            obj_max = self._points[sorted_indices[-1]].values[obj_idx]
            obj_range = obj_max - obj_min

            if obj_range < 1e-12:
                # All values are the same for this objective; skip
                continue

            # Interior points: distance proportional to neighbor spread
            for k in range(1, n - 1):
                prev_val = self._points[sorted_indices[k - 1]].values[obj_idx]
                next_val = self._points[sorted_indices[k + 1]].values[obj_idx]
                distances[sorted_indices[k]] += (next_val - prev_val) / obj_range

        return distances

    def to_dict(self) -> dict[str, Any]:
        """Serialize the frontier to a dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "num_objectives": self._num_objectives,
            "points": [p.to_dict() for p in self._points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParetoFrontier:
        """Deserialize a frontier from a dictionary.

        Reconstructs the frontier by adding points one at a time,
        which automatically enforces dominance filtering.

        Args:
            data: Dictionary with "num_objectives" and "points" keys,
                as produced by to_dict().

        Returns:
            Reconstructed ParetoFrontier.
        """
        num_objectives = data["num_objectives"]
        frontier = cls(num_objectives=num_objectives)

        for point_data in data.get("points", []):
            point = ParetoPoint.from_dict(point_data)
            frontier.add(point)

        return frontier

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"ParetoFrontier(num_objectives={self._num_objectives}, "
            f"size={self.size})"
        )
