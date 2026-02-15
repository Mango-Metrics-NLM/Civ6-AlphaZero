"""Explanation generation utilities for strategic planning decision traces.

Provides natural language explanations, tradeoff analysis, sensitivity
analysis, and strategy comparison for decision traces. These tools help
domain experts understand and communicate planning outcomes.

Example:
    >>> explainer = TraceExplainer(objective_names=["efficiency", "equity"])
    >>> explanation = explainer.explain_decision(decision_point)
    >>> print(explanation)
    At step 3, agent 0 chose 'invest_education' with 85.0% confidence.
    ...

    >>> analyzer = TradeoffAnalyzer(objective_names=["efficiency", "equity"])
    >>> tradeoffs = analyzer.analyze_tradeoffs(traces)
    >>> print(tradeoffs["dominant_objectives"])
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from alpha_zero.strategic.trace.decision_trace import (
    DecisionPoint,
    DecisionTrace,
)

logger = logging.getLogger(__name__)


class TraceExplainer:
    """Generates natural language explanations for decision traces.

    Translates MCTS statistics, objective values, and action alternatives
    into human-readable narratives suitable for domain expert review.

    Args:
        objective_names: Names of each optimization objective.
    """

    def __init__(self, objective_names: list[str]) -> None:
        self._objective_names = list(objective_names)

    def explain_decision(self, decision_point: DecisionPoint) -> str:
        """Generate a natural language explanation of a single decision.

        Describes what action was taken, why it was preferred over
        alternatives, and what the expected objective impacts are.

        Args:
            decision_point: The decision point to explain.

        Returns:
            Human-readable explanation string.
        """
        dp = decision_point
        lines: list[str] = []

        # Opening: what action was taken
        lines.append(
            f"At step {dp.step}, agent {dp.agent_id} chose "
            f"'{dp.action_name}' with {dp.confidence * 100:.1f}% confidence."
        )

        # Objective impact
        if len(dp.objective_values) > 0 and len(self._objective_names) > 0:
            obj_parts: list[str] = []
            for i, name in enumerate(self._objective_names):
                if i < len(dp.objective_values):
                    obj_parts.append(f"{name}={dp.objective_values[i]:.3f}")
            if obj_parts:
                lines.append(f"Objective values: {', '.join(obj_parts)}.")

        # Value estimate
        if len(dp.value_estimate) > 0 and len(self._objective_names) > 0:
            est_parts: list[str] = []
            for i, name in enumerate(self._objective_names):
                if i < len(dp.value_estimate):
                    est_parts.append(f"{name}={dp.value_estimate[i]:.3f}")
            if est_parts:
                lines.append(
                    f"Expected future values: {', '.join(est_parts)}."
                )

        # Alternatives considered
        if dp.alternatives_considered:
            sorted_alts = sorted(
                dp.alternatives_considered,
                key=lambda a: a.visit_count,
                reverse=True,
            )
            # Show top alternatives (up to 3, excluding the chosen action
            # if it appears in alternatives)
            shown = 0
            alt_descriptions: list[str] = []
            for alt in sorted_alts:
                if alt.action_index == dp.action_taken:
                    continue
                alt_descriptions.append(
                    f"'{alt.action_name}' (visits={alt.visit_count}, "
                    f"prior={alt.prior_probability:.3f})"
                )
                shown += 1
                if shown >= 3:
                    break

            if alt_descriptions:
                lines.append(
                    "Top alternatives considered: "
                    + "; ".join(alt_descriptions) + "."
                )

        # Confidence interpretation
        if dp.confidence >= 0.8:
            lines.append("This was a high-confidence decision.")
        elif dp.confidence >= 0.5:
            lines.append(
                "This was a moderate-confidence decision with "
                "viable alternatives."
            )
        else:
            lines.append(
                "This was a low-confidence decision - multiple "
                "actions appeared nearly equally promising."
            )

        # Optional reasoning
        if dp.reasoning:
            lines.append(f"Reasoning: {dp.reasoning}")

        return "\n".join(lines)

    def explain_trace(self, trace: DecisionTrace) -> str:
        """Generate a high-level summary explanation of an entire trace.

        Provides an overview of the planning episode including strategy
        characteristics, key turning points, and final outcomes.

        Args:
            trace: The complete decision trace to explain.

        Returns:
            Human-readable summary string.
        """
        lines: list[str] = []

        lines.append(
            f"Planning episode in '{trace.environment_name}' "
            f"(scenario: {trace.scenario_name}):"
        )
        lines.append(f"Total steps: {trace.num_steps}")
        lines.append(f"Total reward: {trace.total_reward:.4f}")

        # Final outcomes
        if (
            len(trace.final_outcomes) > 0
            and len(self._objective_names) > 0
        ):
            outcome_parts: list[str] = []
            for i, name in enumerate(self._objective_names):
                if i < len(trace.final_outcomes):
                    outcome_parts.append(
                        f"{name}={trace.final_outcomes[i]:.4f}"
                    )
            lines.append(f"Final outcomes: {', '.join(outcome_parts)}.")

        if trace.pareto_rank >= 0:
            lines.append(f"Pareto rank: {trace.pareto_rank}")
            if trace.pareto_rank == 0:
                lines.append(
                    "This trace is on the Pareto frontier (non-dominated)."
                )

        if not trace.decision_points:
            lines.append("No decision points recorded.")
            return "\n".join(lines)

        # Confidence analysis
        confidences = [dp.confidence for dp in trace.decision_points]
        avg_conf = float(np.mean(confidences))
        min_conf = float(np.min(confidences))
        max_conf = float(np.max(confidences))
        lines.append(
            f"Confidence: avg={avg_conf:.3f}, "
            f"min={min_conf:.3f}, max={max_conf:.3f}"
        )

        # Identify key turning points (low confidence decisions)
        low_conf_threshold = avg_conf - float(np.std(confidences))
        turning_points = [
            dp for dp in trace.decision_points
            if dp.confidence < low_conf_threshold
        ]
        if turning_points:
            tp_steps = [str(dp.step) for dp in turning_points[:5]]
            lines.append(
                f"Key decision points (low confidence): "
                f"steps {', '.join(tp_steps)}"
            )

        # Action diversity
        unique_actions = {dp.action_name for dp in trace.decision_points}
        lines.append(
            f"Unique actions used: {len(unique_actions)} "
            f"({', '.join(sorted(unique_actions)[:5])}"
            + ("..." if len(unique_actions) > 5 else "")
            + ")"
        )

        # Strategy characterization
        if avg_conf >= 0.7:
            lines.append(
                "Strategy characterization: Decisive - the planner "
                "consistently found clear best actions."
            )
        elif avg_conf >= 0.4:
            lines.append(
                "Strategy characterization: Balanced - the planner "
                "navigated meaningful tradeoffs throughout."
            )
        else:
            lines.append(
                "Strategy characterization: Exploratory - many decisions "
                "were close calls with near-equal alternatives."
            )

        return "\n".join(lines)


class TradeoffAnalyzer:
    """Analyzes tradeoff patterns across multiple planning traces.

    Identifies which objectives tend to compete, which are correlated,
    and which dominate the planning outcomes.

    Args:
        objective_names: Names of each optimization objective.
    """

    def __init__(self, objective_names: list[str]) -> None:
        self._objective_names = list(objective_names)

    def analyze_tradeoffs(self, traces: list[DecisionTrace]) -> dict[str, Any]:
        """Identify tradeoff patterns across multiple traces.

        Computes correlation between objective outcomes across traces
        to identify competing and complementary objectives.

        Args:
            traces: List of decision traces to analyze.

        Returns:
            Dictionary with keys:
                - "tradeoff_pairs": List of (obj_a, obj_b, correlation)
                    tuples for negatively correlated objective pairs.
                - "correlation_matrix": Full correlation matrix as nested
                    list (num_objectives x num_objectives).
                - "dominant_objectives": List of objective names that
                    tend to have the highest normalized values.
        """
        num_objectives = len(self._objective_names)

        if not traces or num_objectives == 0:
            logger.warning(
                "Cannot analyze tradeoffs: no traces or no objectives."
            )
            return {
                "tradeoff_pairs": [],
                "correlation_matrix": [],
                "dominant_objectives": [],
            }

        # Collect final outcome vectors
        outcome_matrix: list[list[float]] = []
        for trace in traces:
            if len(trace.final_outcomes) >= num_objectives:
                outcome_matrix.append(
                    [float(trace.final_outcomes[i]) for i in range(num_objectives)]
                )

        if len(outcome_matrix) < 2:
            logger.warning(
                "Need at least 2 traces with complete outcomes "
                "for tradeoff analysis, got %d.",
                len(outcome_matrix),
            )
            return {
                "tradeoff_pairs": [],
                "correlation_matrix": np.eye(num_objectives).tolist(),
                "dominant_objectives": list(self._objective_names),
            }

        outcomes = np.array(outcome_matrix, dtype=np.float64)

        # Compute correlation matrix
        # Handle constant columns (zero std) to avoid NaN
        stds = np.std(outcomes, axis=0)
        valid_cols = stds > 1e-10
        corr_matrix = np.eye(num_objectives, dtype=np.float64)

        if np.sum(valid_cols) >= 2:
            valid_indices = np.where(valid_cols)[0]
            valid_outcomes = outcomes[:, valid_cols]
            valid_corr = np.corrcoef(valid_outcomes, rowvar=False)
            # Map back into full correlation matrix
            for i_local, i_global in enumerate(valid_indices):
                for j_local, j_global in enumerate(valid_indices):
                    corr_matrix[i_global, j_global] = valid_corr[
                        i_local, j_local
                    ]

        # Identify tradeoff pairs (negatively correlated)
        tradeoff_pairs: list[tuple[str, str, float]] = []
        for i in range(num_objectives):
            for j in range(i + 1, num_objectives):
                correlation = float(corr_matrix[i, j])
                if correlation < -0.1:
                    tradeoff_pairs.append((
                        self._objective_names[i],
                        self._objective_names[j],
                        correlation,
                    ))

        # Sort by strength of negative correlation
        tradeoff_pairs.sort(key=lambda x: x[2])

        # Identify dominant objectives (highest mean normalized value)
        means = np.mean(outcomes, axis=0)
        # Normalize to [0, 1] range per objective for comparison
        ranges = np.ptp(outcomes, axis=0)
        normalized_means = np.zeros(num_objectives)
        for i in range(num_objectives):
            if ranges[i] > 1e-10:
                mins = np.min(outcomes[:, i])
                normalized_means[i] = (means[i] - mins) / ranges[i]
            else:
                normalized_means[i] = 0.5  # Constant objective

        # Objectives with above-median normalized mean are "dominant"
        median_norm = float(np.median(normalized_means))
        dominant_objectives = [
            self._objective_names[i]
            for i in range(num_objectives)
            if normalized_means[i] >= median_norm
        ]

        return {
            "tradeoff_pairs": tradeoff_pairs,
            "correlation_matrix": corr_matrix.tolist(),
            "dominant_objectives": dominant_objectives,
        }


class SensitivityAnalyzer:
    """Analyzes how different scenarios affect planning outcomes.

    Identifies which objectives are robust across scenarios and which
    are sensitive to specific scenario parameters.

    Args:
        objective_names: Names of each optimization objective.
    """

    def __init__(self, objective_names: list[str]) -> None:
        self._objective_names = list(objective_names)

    def analyze_sensitivity(
        self, traces_by_scenario: dict[str, list[DecisionTrace]]
    ) -> dict[str, Any]:
        """Analyze how scenarios affect outcomes.

        Compares objective outcomes across different scenarios to identify
        which objectives are robust and which are sensitive.

        Args:
            traces_by_scenario: Mapping from scenario name to list of
                traces produced under that scenario.

        Returns:
            Dictionary with keys:
                - "scenario_impacts": Dict mapping scenario name to
                    per-objective mean outcome values.
                - "robust_objectives": List of objective names with low
                    variance across scenarios.
                - "sensitive_objectives": List of objective names with high
                    variance across scenarios.
        """
        num_objectives = len(self._objective_names)

        if not traces_by_scenario or num_objectives == 0:
            logger.warning(
                "Cannot analyze sensitivity: no scenarios or no objectives."
            )
            return {
                "scenario_impacts": {},
                "robust_objectives": [],
                "sensitive_objectives": [],
            }

        # Compute per-scenario mean outcomes
        scenario_impacts: dict[str, list[float]] = {}
        for scenario_name, traces in traces_by_scenario.items():
            outcomes: list[list[float]] = []
            for trace in traces:
                if len(trace.final_outcomes) >= num_objectives:
                    outcomes.append(
                        [
                            float(trace.final_outcomes[i])
                            for i in range(num_objectives)
                        ]
                    )
            if outcomes:
                mean_outcomes = np.mean(
                    np.array(outcomes, dtype=np.float64), axis=0
                )
                scenario_impacts[scenario_name] = mean_outcomes.tolist()

        if len(scenario_impacts) < 2:
            logger.warning(
                "Need at least 2 scenarios for sensitivity analysis, "
                "got %d.",
                len(scenario_impacts),
            )
            return {
                "scenario_impacts": scenario_impacts,
                "robust_objectives": list(self._objective_names),
                "sensitive_objectives": [],
            }

        # Stack scenario means into a matrix (num_scenarios x num_objectives)
        scenario_matrix = np.array(
            list(scenario_impacts.values()), dtype=np.float64
        )

        # Compute coefficient of variation per objective across scenarios
        means = np.mean(scenario_matrix, axis=0)
        stds = np.std(scenario_matrix, axis=0)
        cvs = np.zeros(num_objectives, dtype=np.float64)
        for i in range(num_objectives):
            if abs(means[i]) > 1e-10:
                cvs[i] = stds[i] / abs(means[i])
            else:
                cvs[i] = stds[i]  # Use raw std if mean is near zero

        # Classify objectives by sensitivity
        # Use median CV as the threshold
        median_cv = float(np.median(cvs))

        robust_objectives: list[str] = []
        sensitive_objectives: list[str] = []
        for i, name in enumerate(self._objective_names):
            if cvs[i] <= median_cv:
                robust_objectives.append(name)
            else:
                sensitive_objectives.append(name)

        return {
            "scenario_impacts": scenario_impacts,
            "robust_objectives": robust_objectives,
            "sensitive_objectives": sensitive_objectives,
        }


class ComparisonExplainer:
    """Generates natural language comparisons between two planning strategies.

    Compares decision traces side-by-side, highlighting differences in
    actions, objective outcomes, and confidence levels.

    Args:
        objective_names: Names of each optimization objective.
    """

    def __init__(self, objective_names: list[str]) -> None:
        self._objective_names = list(objective_names)

    def compare_strategies(
        self, trace_a: DecisionTrace, trace_b: DecisionTrace
    ) -> str:
        """Generate a natural language comparison of two strategies.

        Compares final outcomes, action choices, and confidence patterns
        between two decision traces.

        Args:
            trace_a: First strategy trace.
            trace_b: Second strategy trace.

        Returns:
            Human-readable comparison string.
        """
        lines: list[str] = []
        num_objectives = len(self._objective_names)

        label_a = (
            f"Strategy A ({trace_a.scenario_name})"
            if trace_a.scenario_name != trace_b.scenario_name
            else "Strategy A"
        )
        label_b = (
            f"Strategy B ({trace_b.scenario_name})"
            if trace_a.scenario_name != trace_b.scenario_name
            else "Strategy B"
        )

        lines.append("=== Strategy Comparison ===")
        lines.append(
            f"{label_a}: {trace_a.num_steps} steps, "
            f"reward={trace_a.total_reward:.4f}"
        )
        lines.append(
            f"{label_b}: {trace_b.num_steps} steps, "
            f"reward={trace_b.total_reward:.4f}"
        )

        # Compare final outcomes
        if (
            len(trace_a.final_outcomes) >= num_objectives
            and len(trace_b.final_outcomes) >= num_objectives
            and num_objectives > 0
        ):
            lines.append("")
            lines.append("Objective comparison:")
            a_wins: list[str] = []
            b_wins: list[str] = []
            ties: list[str] = []
            for i, name in enumerate(self._objective_names):
                val_a = float(trace_a.final_outcomes[i])
                val_b = float(trace_b.final_outcomes[i])
                diff = val_a - val_b
                if abs(diff) < 1e-6:
                    ties.append(name)
                    lines.append(
                        f"  {name}: A={val_a:.4f}, B={val_b:.4f} (tied)"
                    )
                elif diff > 0:
                    a_wins.append(name)
                    lines.append(
                        f"  {name}: A={val_a:.4f}, B={val_b:.4f} "
                        f"(A better by {diff:.4f})"
                    )
                else:
                    b_wins.append(name)
                    lines.append(
                        f"  {name}: A={val_a:.4f}, B={val_b:.4f} "
                        f"(B better by {-diff:.4f})"
                    )

            if a_wins and b_wins:
                lines.append(
                    f"\nTradeoff detected: {label_a} excels at "
                    f"{', '.join(a_wins)}, while {label_b} excels at "
                    f"{', '.join(b_wins)}."
                )
            elif a_wins and not b_wins:
                lines.append(
                    f"\n{label_a} dominates {label_b} across all objectives."
                )
            elif b_wins and not a_wins:
                lines.append(
                    f"\n{label_b} dominates {label_a} across all objectives."
                )

        # Compare confidence
        if trace_a.decision_points and trace_b.decision_points:
            conf_a = [dp.confidence for dp in trace_a.decision_points]
            conf_b = [dp.confidence for dp in trace_b.decision_points]
            avg_a = float(np.mean(conf_a))
            avg_b = float(np.mean(conf_b))
            lines.append("")
            lines.append(
                f"Average confidence: A={avg_a:.3f}, B={avg_b:.3f}"
            )
            if avg_a > avg_b + 0.05:
                lines.append(
                    f"{label_a} was generally more confident "
                    f"in its decisions."
                )
            elif avg_b > avg_a + 0.05:
                lines.append(
                    f"{label_b} was generally more confident "
                    f"in its decisions."
                )
            else:
                lines.append("Both strategies showed similar confidence.")

        # Compare action diversity
        if trace_a.decision_points and trace_b.decision_points:
            actions_a = {dp.action_name for dp in trace_a.decision_points}
            actions_b = {dp.action_name for dp in trace_b.decision_points}
            shared = actions_a & actions_b
            only_a = actions_a - actions_b
            only_b = actions_b - actions_a

            lines.append("")
            lines.append(
                f"Shared actions: {len(shared)}, "
                f"unique to A: {len(only_a)}, "
                f"unique to B: {len(only_b)}"
            )
            if only_a:
                lines.append(
                    f"Actions unique to {label_a}: "
                    + ", ".join(sorted(only_a)[:5])
                    + ("..." if len(only_a) > 5 else "")
                )
            if only_b:
                lines.append(
                    f"Actions unique to {label_b}: "
                    + ", ".join(sorted(only_b)[:5])
                    + ("..." if len(only_b) > 5 else "")
                )

        # Pareto rank comparison
        if trace_a.pareto_rank >= 0 and trace_b.pareto_rank >= 0:
            lines.append("")
            lines.append(
                f"Pareto ranks: A={trace_a.pareto_rank}, "
                f"B={trace_b.pareto_rank}"
            )
            if trace_a.pareto_rank < trace_b.pareto_rank:
                lines.append(f"{label_a} has a better Pareto rank.")
            elif trace_b.pareto_rank < trace_a.pareto_rank:
                lines.append(f"{label_b} has a better Pareto rank.")
            else:
                lines.append("Both strategies have the same Pareto rank.")

        return "\n".join(lines)


__all__ = [
    "ComparisonExplainer",
    "SensitivityAnalyzer",
    "TraceExplainer",
    "TradeoffAnalyzer",
]
