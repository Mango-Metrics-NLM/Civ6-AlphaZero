"""Multi-objective optimization support for the Strategic Planning SDK.

Provides objective specification, scalarization strategies, and utilities
for handling multiple competing objectives in strategic planning.

Example:
    >>> objectives = [
    ...     ObjectiveSpec(name="efficiency", direction="maximize", weight=0.6),
    ...     ObjectiveSpec(name="equity", direction="maximize", weight=0.4),
    ... ]
    >>> scalarizer = WeightedSumScalarizer(objectives)
    >>> scalar_value = scalarizer.scalarize(np.array([0.8, 0.6]))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveSpec:
    """Specification for a single optimization objective.

    Attributes:
        name: Human-readable name for the objective.
        direction: Optimization direction ("maximize" or "minimize").
        weight: Relative importance weight (used in scalarization).
        lower_bound: Optional lower bound for the objective value.
        upper_bound: Optional upper bound for the objective value.
        constraint_value: Optional constraint (for epsilon-constraint method).
        description: Optional longer description.
    """

    name: str
    direction: str = "maximize"
    weight: float = 1.0
    lower_bound: float | None = None
    upper_bound: float | None = None
    constraint_value: float | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate objective specification."""
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(
                f"Invalid direction '{self.direction}' for objective '{self.name}'. "
                f"Must be 'maximize' or 'minimize'."
            )
        if self.weight < 0:
            raise ValueError(
                f"Weight must be non-negative for objective '{self.name}', "
                f"got {self.weight}."
            )

    @property
    def sign(self) -> float:
        """Multiplier for converting to maximization form.

        Returns 1.0 for maximize, -1.0 for minimize.
        """
        return 1.0 if self.direction == "maximize" else -1.0

    def normalize(self, value: float) -> float:
        """Normalize value to [0, 1] range using bounds.

        Args:
            value: Raw objective value.

        Returns:
            Normalized value in [0, 1], or the raw value if bounds not set.
        """
        if self.lower_bound is not None and self.upper_bound is not None:
            range_val = self.upper_bound - self.lower_bound
            if range_val > 0:
                return float(np.clip(
                    (value - self.lower_bound) / range_val, 0.0, 1.0
                ))
        return value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "direction": self.direction,
            "weight": self.weight,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "constraint_value": self.constraint_value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectiveSpec:
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            direction=data.get("direction", "maximize"),
            weight=float(data.get("weight", 1.0)),
            lower_bound=data.get("lower_bound"),
            upper_bound=data.get("upper_bound"),
            constraint_value=data.get("constraint_value"),
            description=data.get("description", ""),
        )


class BaseScalarizer:
    """Base class for scalarization strategies.

    Scalarizers convert multi-objective value vectors into single
    scalar values for use in MCTS node selection and backpropagation.
    """

    def __init__(self, objectives: list[ObjectiveSpec]) -> None:
        """Initialize scalarizer.

        Args:
            objectives: List of objective specifications.
        """
        self._objectives = list(objectives)
        self._weights = np.array(
            [obj.weight for obj in self._objectives], dtype=np.float64
        )
        total = self._weights.sum()
        if total > 0:
            self._normalized_weights = self._weights / total
        else:
            self._normalized_weights = np.ones(len(self._objectives)) / max(
                len(self._objectives), 1
            )
        self._signs = np.array(
            [obj.sign for obj in self._objectives], dtype=np.float64
        )

    @property
    def objectives(self) -> list[ObjectiveSpec]:
        """Get objective specifications."""
        return self._objectives

    @property
    def num_objectives(self) -> int:
        """Number of objectives."""
        return len(self._objectives)

    @property
    def weights(self) -> np.ndarray:
        """Normalized objective weights."""
        return self._normalized_weights

    def _to_maximization_form(self, values: np.ndarray) -> np.ndarray:
        """Convert objective values to maximization form.

        Multiplies minimize objectives by -1 so all objectives
        are expressed as maximization.

        Args:
            values: Raw objective values.

        Returns:
            Values in maximization form.
        """
        return values * self._signs

    def scalarize(self, values: np.ndarray) -> float:
        """Convert multi-objective values to scalar. Override in subclasses."""
        raise NotImplementedError


class WeightedSumScalarizer(BaseScalarizer):
    """Weighted sum scalarization.

    The simplest approach: linear combination of objectives weighted
    by importance. Effective when objectives are roughly commensurable.

    scalar_value = sum(weight_i * sign_i * value_i)
    """

    def scalarize(self, values: np.ndarray) -> float:
        """Compute weighted sum of objective values.

        Args:
            values: Per-objective values, shape (num_objectives,).

        Returns:
            Weighted scalar value.
        """
        maximized = self._to_maximization_form(values)
        return float(np.dot(self._normalized_weights, maximized))


class ChebyshevScalarizer(BaseScalarizer):
    """Chebyshev (minimax) scalarization.

    Minimizes the worst weighted deviation from an ideal point.
    Better at finding solutions on non-convex Pareto frontiers.

    scalar_value = -max(weight_i * |ideal_i - value_i|)

    Attributes:
        ideal_point: The ideal objective vector (best achievable per objective).
    """

    def __init__(
        self,
        objectives: list[ObjectiveSpec],
        ideal_point: np.ndarray | None = None,
    ) -> None:
        """Initialize Chebyshev scalarizer.

        Args:
            objectives: List of objective specifications.
            ideal_point: Ideal objective vector. If None, uses ones.
        """
        super().__init__(objectives)
        if ideal_point is not None:
            self._ideal = np.asarray(ideal_point, dtype=np.float64)
        else:
            self._ideal = np.ones(self.num_objectives, dtype=np.float64)

    @property
    def ideal_point(self) -> np.ndarray:
        """Get the ideal point."""
        return self._ideal

    def update_ideal_point(self, ideal_point: np.ndarray) -> None:
        """Update the ideal point with new values.

        Args:
            ideal_point: New ideal objective vector.
        """
        self._ideal = np.asarray(ideal_point, dtype=np.float64)

    def scalarize(self, values: np.ndarray) -> float:
        """Compute Chebyshev scalarization.

        Args:
            values: Per-objective values, shape (num_objectives,).

        Returns:
            Negative max weighted deviation (higher is better).
        """
        maximized = self._to_maximization_form(values)
        deviations = self._normalized_weights * np.abs(self._ideal - maximized)
        return -float(np.max(deviations))


class EpsilonConstraintScalarizer(BaseScalarizer):
    """Epsilon-constraint scalarization.

    Optimizes the primary objective while constraining all others
    to be above threshold values.

    If constraints are violated, applies a penalty proportional
    to the violation magnitude.

    Attributes:
        primary_index: Index of the primary objective to optimize.
        constraint_penalty: Penalty multiplier for constraint violations.
    """

    def __init__(
        self,
        objectives: list[ObjectiveSpec],
        primary_index: int = 0,
        constraint_penalty: float = 10.0,
    ) -> None:
        """Initialize epsilon-constraint scalarizer.

        Args:
            objectives: List of objective specifications.
            primary_index: Index of the primary objective.
            constraint_penalty: Penalty for constraint violations.
        """
        super().__init__(objectives)
        if primary_index < 0 or primary_index >= self.num_objectives:
            raise ValueError(
                f"primary_index {primary_index} out of range "
                f"[0, {self.num_objectives})"
            )
        self._primary_index = primary_index
        self._constraint_penalty = constraint_penalty

    @property
    def primary_index(self) -> int:
        """Index of the primary objective."""
        return self._primary_index

    @property
    def constraint_penalty(self) -> float:
        """Penalty multiplier for constraint violations."""
        return self._constraint_penalty

    def scalarize(self, values: np.ndarray) -> float:
        """Compute epsilon-constraint scalarization.

        Args:
            values: Per-objective values, shape (num_objectives,).

        Returns:
            Primary objective value minus constraint violation penalties.
        """
        maximized = self._to_maximization_form(values)
        primary_value = float(maximized[self._primary_index])

        # Check constraints on non-primary objectives
        penalty = 0.0
        for i, obj in enumerate(self._objectives):
            if i == self._primary_index:
                continue
            if obj.constraint_value is not None:
                threshold = obj.constraint_value * obj.sign
                violation = threshold - maximized[i]
                if violation > 0:
                    penalty += self._constraint_penalty * violation

        return primary_value - penalty


@dataclass
class MultiObjectiveResult:
    """Container for multi-objective evaluation results.

    Attributes:
        objective_values: Per-objective values, shape (num_objectives,).
        scalarized_value: Combined scalar value.
        objective_names: Names of each objective.
        constraint_violations: Per-objective constraint violations (0 if met).
    """

    objective_values: np.ndarray
    scalarized_value: float
    objective_names: list[str] = field(default_factory=list)
    constraint_violations: np.ndarray = field(
        default_factory=lambda: np.array([])
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "scalarized_value": self.scalarized_value,
            "objectives": {},
        }
        for i, name in enumerate(self.objective_names):
            result["objectives"][name] = float(self.objective_values[i])
        if len(self.constraint_violations) > 0:
            result["constraint_violations"] = self.constraint_violations.tolist()
        return result


__all__ = [
    "BaseScalarizer",
    "ChebyshevScalarizer",
    "EpsilonConstraintScalarizer",
    "MultiObjectiveResult",
    "ObjectiveSpec",
    "WeightedSumScalarizer",
]
