"""Decision trace recording system for strategic planning.

Records every decision made during planning, including alternatives
considered, objective values, MCTS statistics, and reasoning. Traces
can be serialized to JSON for analysis, visualization, and explainability.

Example:
    >>> trace = DecisionTrace(
    ...     environment_name="resource_allocation",
    ...     objective_names=["efficiency", "equity"],
    ... )
    >>> alt = ActionAlternative(
    ...     action_index=0, action_name="invest",
    ...     visit_count=150, prior_probability=0.3,
    ...     value_estimate=np.array([0.8, 0.6]),
    ... )
    >>> dp = DecisionPoint(
    ...     step=0, agent_id=0, state_summary={"budget": 100},
    ...     action_taken=0, action_name="invest",
    ...     alternatives_considered=[alt],
    ...     objective_values=np.array([0.8, 0.6]),
    ...     mcts_visit_counts=np.array([150, 80, 20]),
    ...     value_estimate=np.array([0.75, 0.55]),
    ...     confidence=0.85,
    ... )
    >>> trace.add_decision_point(dp)
    >>> json_str = trace.to_json()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ActionAlternative:
    """An alternative action that was considered at a decision point.

    Captures MCTS visit statistics and value estimates for an action
    that was evaluated but not necessarily chosen.

    Attributes:
        action_index: Discrete action index in the action space.
        action_name: Human-readable name for the action.
        visit_count: Number of MCTS visits to this action.
        prior_probability: Neural network prior probability.
        value_estimate: Per-objective value estimate from MCTS.
    """

    action_index: int
    action_name: str
    visit_count: int
    prior_probability: float
    value_estimate: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation with numpy arrays converted to lists.
        """
        return {
            "action_index": self.action_index,
            "action_name": self.action_name,
            "visit_count": self.visit_count,
            "prior_probability": self.prior_probability,
            "value_estimate": self.value_estimate.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionAlternative:
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Reconstructed ActionAlternative.
        """
        return cls(
            action_index=int(data["action_index"]),
            action_name=str(data["action_name"]),
            visit_count=int(data["visit_count"]),
            prior_probability=float(data["prior_probability"]),
            value_estimate=np.array(data["value_estimate"], dtype=np.float64),
        )


@dataclass
class DecisionPoint:
    """A single decision recorded during planning.

    Captures the full context of a decision: what state the agent was in,
    what action it chose, what alternatives it considered, and the
    objective values and MCTS statistics at the time.

    Attributes:
        step: Step number in the episode.
        agent_id: Index of the agent making the decision.
        state_summary: Domain-specific state summary dictionary.
        action_taken: Index of the action that was chosen.
        action_name: Human-readable name of the chosen action.
        alternatives_considered: List of alternative actions evaluated.
        objective_values: Per-objective values at this step.
        mcts_visit_counts: Raw MCTS visit counts across all actions.
        value_estimate: Per-objective value estimate from the network/MCTS.
        confidence: Confidence in the decision (0-1), typically derived
            from the visit count ratio of the chosen action.
        reasoning: Optional natural language reasoning for the decision.
    """

    step: int
    agent_id: int
    state_summary: dict[str, Any]
    action_taken: int
    action_name: str
    alternatives_considered: list[ActionAlternative]
    objective_values: np.ndarray
    mcts_visit_counts: np.ndarray
    value_estimate: np.ndarray
    confidence: float
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation with numpy arrays converted to lists.
        """
        return {
            "step": self.step,
            "agent_id": self.agent_id,
            "state_summary": self.state_summary,
            "action_taken": self.action_taken,
            "action_name": self.action_name,
            "alternatives_considered": [
                alt.to_dict() for alt in self.alternatives_considered
            ],
            "objective_values": self.objective_values.tolist(),
            "mcts_visit_counts": self.mcts_visit_counts.tolist(),
            "value_estimate": self.value_estimate.tolist(),
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionPoint:
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Reconstructed DecisionPoint.
        """
        alternatives = [
            ActionAlternative.from_dict(alt_data)
            for alt_data in data.get("alternatives_considered", [])
        ]
        return cls(
            step=int(data["step"]),
            agent_id=int(data["agent_id"]),
            state_summary=data.get("state_summary", {}),
            action_taken=int(data["action_taken"]),
            action_name=str(data["action_name"]),
            alternatives_considered=alternatives,
            objective_values=np.array(
                data["objective_values"], dtype=np.float64
            ),
            mcts_visit_counts=np.array(
                data["mcts_visit_counts"], dtype=np.float64
            ),
            value_estimate=np.array(
                data["value_estimate"], dtype=np.float64
            ),
            confidence=float(data["confidence"]),
            reasoning=str(data.get("reasoning", "")),
        )


@dataclass
class DecisionTrace:
    """Complete trace of decisions made during a planning episode.

    A trace captures the full sequence of decisions, their contexts,
    and the final outcomes. It supports JSON serialization for storage,
    analysis, and explainability.

    Attributes:
        environment_name: Name of the environment that produced this trace.
        scenario_name: Name of the scenario (for uncertainty analysis).
        objective_names: Names of each optimization objective.
        decision_points: Ordered list of decisions made.
        final_outcomes: Per-objective final outcome values.
        pareto_rank: Pareto rank of this trace (-1 if not ranked).
        total_reward: Total scalarized reward across the episode.
        metadata: Additional metadata (configuration, timing, etc.).
    """

    environment_name: str
    scenario_name: str = "default"
    objective_names: list[str] = field(default_factory=list)
    decision_points: list[DecisionPoint] = field(default_factory=list)
    final_outcomes: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    pareto_rank: int = -1
    total_reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        """Number of decision points in the trace."""
        return len(self.decision_points)

    def add_decision_point(self, decision_point: DecisionPoint) -> None:
        """Add a decision point to the trace.

        Args:
            decision_point: The decision point to record.
        """
        self.decision_points.append(decision_point)
        logger.debug(
            "Added decision point at step %d (action: %s, confidence: %.3f)",
            decision_point.step,
            decision_point.action_name,
            decision_point.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire trace to a dictionary.

        Returns:
            Dictionary representation with all numpy arrays converted to lists.
        """
        return {
            "environment_name": self.environment_name,
            "scenario_name": self.scenario_name,
            "objective_names": self.objective_names,
            "decision_points": [dp.to_dict() for dp in self.decision_points],
            "final_outcomes": self.final_outcomes.tolist(),
            "pareto_rank": self.pareto_rank,
            "total_reward": self.total_reward,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionTrace:
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            Reconstructed DecisionTrace.
        """
        decision_points = [
            DecisionPoint.from_dict(dp_data)
            for dp_data in data.get("decision_points", [])
        ]
        final_outcomes = np.array(
            data.get("final_outcomes", []), dtype=np.float64
        )
        return cls(
            environment_name=str(data["environment_name"]),
            scenario_name=str(data.get("scenario_name", "default")),
            objective_names=data.get("objective_names", []),
            decision_points=decision_points,
            final_outcomes=final_outcomes,
            pareto_rank=int(data.get("pareto_rank", -1)),
            total_reward=float(data.get("total_reward", 0.0)),
            metadata=data.get("metadata", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize the trace to a JSON string.

        Args:
            indent: JSON indentation level. Use None for compact output.

        Returns:
            JSON string representation of the trace.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> DecisionTrace:
        """Deserialize from a JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            Reconstructed DecisionTrace.

        Raises:
            json.JSONDecodeError: If the string is not valid JSON.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def summary(self) -> str:
        """Generate a human-readable summary of the trace.

        Returns:
            Multi-line string summarizing the trace.
        """
        lines: list[str] = []
        lines.append(f"=== Decision Trace: {self.environment_name} ===")
        lines.append(f"Scenario: {self.scenario_name}")
        lines.append(f"Steps: {self.num_steps}")
        lines.append(f"Total reward: {self.total_reward:.4f}")

        if self.pareto_rank >= 0:
            lines.append(f"Pareto rank: {self.pareto_rank}")

        if len(self.final_outcomes) > 0 and len(self.objective_names) > 0:
            lines.append("Final outcomes:")
            for i, name in enumerate(self.objective_names):
                if i < len(self.final_outcomes):
                    lines.append(f"  {name}: {self.final_outcomes[i]:.4f}")

        if self.decision_points:
            # Summarize confidence statistics
            confidences = [dp.confidence for dp in self.decision_points]
            avg_confidence = sum(confidences) / len(confidences)
            min_confidence = min(confidences)
            lines.append(
                f"Confidence: avg={avg_confidence:.3f}, "
                f"min={min_confidence:.3f}"
            )

            # Show first and last decisions
            first_dp = self.decision_points[0]
            lines.append(
                f"First action: {first_dp.action_name} "
                f"(confidence={first_dp.confidence:.3f})"
            )
            if self.num_steps > 1:
                last_dp = self.decision_points[-1]
                lines.append(
                    f"Last action: {last_dp.action_name} "
                    f"(confidence={last_dp.confidence:.3f})"
                )

        if self.metadata:
            lines.append(f"Metadata keys: {list(self.metadata.keys())}")

        return "\n".join(lines)

    def to_dataframe(self) -> list[dict[str, Any]]:
        """Convert trace to a list of flat dictionaries for DataFrame creation.

        Each decision point becomes one row. Objective values and visit counts
        are expanded into separate columns for easy analysis.

        Returns:
            List of dictionaries, one per decision point, suitable for
            ``pandas.DataFrame(trace.to_dataframe())``.
        """
        rows: list[dict[str, Any]] = []
        for dp in self.decision_points:
            row: dict[str, Any] = {
                "environment": self.environment_name,
                "scenario": self.scenario_name,
                "step": dp.step,
                "agent_id": dp.agent_id,
                "action_taken": dp.action_taken,
                "action_name": dp.action_name,
                "confidence": dp.confidence,
                "reasoning": dp.reasoning,
                "num_alternatives": len(dp.alternatives_considered),
            }

            # Expand per-objective values into separate columns
            for i, name in enumerate(self.objective_names):
                if i < len(dp.objective_values):
                    row[f"objective_{name}"] = float(dp.objective_values[i])
                if i < len(dp.value_estimate):
                    row[f"value_estimate_{name}"] = float(
                        dp.value_estimate[i]
                    )

            # Add total visit count
            if len(dp.mcts_visit_counts) > 0:
                row["total_visits"] = int(dp.mcts_visit_counts.sum())

            # Add state summary fields (flattened)
            for key, value in dp.state_summary.items():
                if isinstance(value, (int, float, str, bool)):
                    row[f"state_{key}"] = value

            rows.append(row)

        return rows


__all__ = [
    "ActionAlternative",
    "DecisionPoint",
    "DecisionTrace",
]
