"""Scenario engine for uncertainty handling in strategic planning.

Manages multiple scenarios representing different assumptions about
uncertain parameters. Supports probability-weighted analysis, robustness
scoring, and scenario sampling for Monte Carlo evaluation.

Example:
    >>> engine = ScenarioEngine()
    >>> engine.add_scenario(Scenario(
    ...     name="optimistic",
    ...     parameters={"growth_rate": 0.05},
    ...     probability=0.3,
    ...     description="High growth scenario",
    ... ))
    >>> engine.add_scenario(Scenario(
    ...     name="pessimistic",
    ...     parameters={"growth_rate": 0.01},
    ...     probability=0.7,
    ...     description="Low growth scenario",
    ... ))
    >>> engine.normalize_probabilities()
    >>> scenario = engine.sample_scenario()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """A single uncertainty scenario with associated parameters and probability.

    Represents a specific set of assumptions about uncertain parameters
    (e.g., climate sensitivity, market conditions, disruption severity).
    Satisfies the ``ScenarioProtocol`` via structural subtyping.

    Attributes:
        name: Human-readable scenario name.
        parameters: Scenario-specific parameter overrides applied to the
            environment when simulating under this scenario.
        probability: Prior probability of this scenario occurring (0-1).
            Probabilities across all scenarios in an engine should sum to 1.
        description: Optional longer description of the scenario.
    """

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        """Validate scenario parameters."""
        if self.probability < 0.0 or self.probability > 1.0:
            raise ValueError(
                f"Scenario probability must be in [0, 1], "
                f"got {self.probability} for scenario '{self.name}'."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of the scenario.
        """
        return {
            "name": self.name,
            "parameters": self.parameters,
            "probability": self.probability,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scenario:
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Reconstructed Scenario.
        """
        return cls(
            name=str(data["name"]),
            parameters=data.get("parameters", {}),
            probability=float(data.get("probability", 1.0)),
            description=str(data.get("description", "")),
        )


class ScenarioEngine:
    """Engine for managing and analyzing uncertainty scenarios.

    Provides scenario management (add/remove/get), probability
    normalization, sampling, and analytical tools like expected
    value computation and robustness scoring.

    Args:
        scenarios: Optional initial list of scenarios.
    """

    def __init__(self, scenarios: list[Scenario] | None = None) -> None:
        self._scenarios: list[Scenario] = []
        self._scenario_map: dict[str, Scenario] = {}

        if scenarios is not None:
            for scenario in scenarios:
                self.add_scenario(scenario)

    def add_scenario(self, scenario: Scenario) -> None:
        """Add a scenario to the engine.

        Validates that the probability is in [0, 1] and the name is unique.

        Args:
            scenario: The scenario to add.

        Raises:
            ValueError: If probability is invalid or name already exists.
        """
        if scenario.probability < 0.0 or scenario.probability > 1.0:
            raise ValueError(
                f"Scenario probability must be in [0, 1], "
                f"got {scenario.probability} for '{scenario.name}'."
            )
        if scenario.name in self._scenario_map:
            raise ValueError(
                f"Scenario with name '{scenario.name}' already exists. "
                f"Use remove_scenario first to replace."
            )
        self._scenarios.append(scenario)
        self._scenario_map[scenario.name] = scenario
        logger.debug(
            "Added scenario '%s' (probability=%.3f)",
            scenario.name,
            scenario.probability,
        )

    def remove_scenario(self, name: str) -> None:
        """Remove a scenario by name.

        Args:
            name: Name of the scenario to remove.

        Raises:
            KeyError: If no scenario with the given name exists.
        """
        if name not in self._scenario_map:
            raise KeyError(
                f"No scenario with name '{name}' found. "
                f"Available: {list(self._scenario_map.keys())}"
            )
        scenario = self._scenario_map.pop(name)
        self._scenarios.remove(scenario)
        logger.debug("Removed scenario '%s'", name)

    def get_scenario(self, name: str) -> Scenario:
        """Get a scenario by name.

        Args:
            name: Name of the scenario.

        Returns:
            The matching scenario.

        Raises:
            KeyError: If no scenario with the given name exists.
        """
        if name not in self._scenario_map:
            raise KeyError(
                f"No scenario with name '{name}' found. "
                f"Available: {list(self._scenario_map.keys())}"
            )
        return self._scenario_map[name]

    @property
    def scenarios(self) -> list[Scenario]:
        """Get all scenarios in the engine.

        Returns:
            Copy of the internal scenario list.
        """
        return list(self._scenarios)

    @property
    def num_scenarios(self) -> int:
        """Number of scenarios in the engine."""
        return len(self._scenarios)

    def normalize_probabilities(self) -> None:
        """Normalize scenario probabilities to sum to 1.0.

        If all probabilities are zero, assigns equal probability
        to each scenario.

        Raises:
            ValueError: If there are no scenarios.
        """
        if not self._scenarios:
            raise ValueError("Cannot normalize: no scenarios in engine.")

        total = sum(s.probability for s in self._scenarios)

        if total <= 0.0:
            # All zero probabilities: assign equal weight
            equal_prob = 1.0 / len(self._scenarios)
            for scenario in self._scenarios:
                scenario.probability = equal_prob
            logger.warning(
                "All scenario probabilities were zero. "
                "Assigned equal probability %.4f to each.",
                equal_prob,
            )
        else:
            for scenario in self._scenarios:
                scenario.probability = scenario.probability / total

        logger.debug(
            "Normalized probabilities for %d scenarios (sum=%.6f)",
            len(self._scenarios),
            sum(s.probability for s in self._scenarios),
        )

    def sample_scenario(
        self, rng: np.random.Generator | None = None
    ) -> Scenario:
        """Sample a scenario according to probability weights.

        Args:
            rng: Optional numpy random generator for reproducibility.
                If None, uses numpy's default random state.

        Returns:
            A sampled scenario.

        Raises:
            ValueError: If there are no scenarios.
        """
        if not self._scenarios:
            raise ValueError("Cannot sample: no scenarios in engine.")

        probabilities = np.array(
            [s.probability for s in self._scenarios], dtype=np.float64
        )

        # Normalize in case probabilities don't sum to 1
        prob_sum = probabilities.sum()
        if prob_sum <= 0.0:
            # Uniform fallback
            probabilities = np.ones(len(self._scenarios)) / len(self._scenarios)
        else:
            probabilities = probabilities / prob_sum

        if rng is not None:
            idx = int(rng.choice(len(self._scenarios), p=probabilities))
        else:
            idx = int(
                np.random.choice(len(self._scenarios), p=probabilities)
            )

        return self._scenarios[idx]

    def expected_value(
        self, values_by_scenario: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute probability-weighted expected value across scenarios.

        Args:
            values_by_scenario: Mapping from scenario name to objective
                value vector. Scenarios not in this dict are skipped
                (with their probability redistributed).

        Returns:
            Probability-weighted average value vector.

        Raises:
            ValueError: If no matching scenarios found or vectors
                have inconsistent shapes.
        """
        matched_scenarios: list[tuple[Scenario, np.ndarray]] = []
        for scenario in self._scenarios:
            if scenario.name in values_by_scenario:
                matched_scenarios.append(
                    (scenario, np.asarray(values_by_scenario[scenario.name],
                                          dtype=np.float64))
                )

        if not matched_scenarios:
            raise ValueError(
                "No matching scenarios found. "
                f"Engine has: {[s.name for s in self._scenarios]}, "
                f"values provided for: {list(values_by_scenario.keys())}"
            )

        # Validate consistent shapes
        shapes = {v.shape for _, v in matched_scenarios}
        if len(shapes) > 1:
            raise ValueError(
                f"Inconsistent value vector shapes across scenarios: {shapes}"
            )

        # Compute weighted average with renormalized probabilities
        total_prob = sum(s.probability for s, _ in matched_scenarios)
        if total_prob <= 0.0:
            # Equal weight fallback
            total_prob = float(len(matched_scenarios))
            weights = [1.0] * len(matched_scenarios)
        else:
            weights = [s.probability for s, _ in matched_scenarios]

        result = np.zeros_like(matched_scenarios[0][1])
        for (_, values), weight in zip(matched_scenarios, weights, strict=False):
            result += (weight / total_prob) * values

        return result

    def robustness_score(
        self, values_by_scenario: dict[str, np.ndarray]
    ) -> float:
        """Compute robustness score: worst-case relative to best-case.

        The robustness score measures how well outcomes hold up under
        the worst scenario compared to the best. A score of 1.0 means
        outcomes are identical across scenarios (perfectly robust).
        A score near 0.0 means the worst case is far worse than the best.

        Uses the scalarized (summed) objective value for comparison.

        Args:
            values_by_scenario: Mapping from scenario name to objective
                value vector.

        Returns:
            Robustness score in [0, 1]. Higher is more robust.
        """
        matched_values: list[float] = []
        for scenario in self._scenarios:
            if scenario.name in values_by_scenario:
                values = np.asarray(
                    values_by_scenario[scenario.name], dtype=np.float64
                )
                matched_values.append(float(values.sum()))

        if len(matched_values) < 2:
            logger.warning(
                "Need at least 2 matched scenarios for robustness score, "
                "got %d. Returning 1.0 (trivially robust).",
                len(matched_values),
            )
            return 1.0

        worst = min(matched_values)
        best = max(matched_values)

        if abs(best) < 1e-10:
            # Avoid division by zero; if best is near zero, check if
            # worst is also near zero
            if abs(worst) < 1e-10:
                return 1.0  # All values near zero
            return 0.0

        # Handle negative values: shift so best is positive
        if best <= 0.0:
            # All non-positive: ratio of magnitudes (inverted)
            # worst <= best <= 0
            # Score = |best| / |worst| = best / worst (both negative)
            if abs(worst) < 1e-10:
                return 1.0
            return float(np.clip(best / worst, 0.0, 1.0))

        # Mixed or all positive: worst/best ratio
        if worst < 0.0:
            # Worst is negative, best is positive - significant gap
            return 0.0

        # Both positive: simple ratio
        return float(np.clip(worst / best, 0.0, 1.0))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the engine to a dictionary.

        Returns:
            Dictionary representation of all scenarios.
        """
        return {
            "scenarios": [s.to_dict() for s in self._scenarios],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioEngine:
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Reconstructed ScenarioEngine.
        """
        scenarios = [
            Scenario.from_dict(s_data)
            for s_data in data.get("scenarios", [])
        ]
        return cls(scenarios=scenarios)


__all__ = [
    "Scenario",
    "ScenarioEngine",
]
