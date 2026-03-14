"""Loss analysis engine for training run diagnostics.

Extracts loss curves, computes convergence rates via linear regression,
detects learning stalls, and decomposes policy/value loss contributions.
Delegates stall detection and trend computation to TrainingDiagnostics.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from alpha_zero.analysis.data_model import RunData
from alpha_zero.training.diagnostics import TrainingDiagnostics


@dataclass
class LossAnalysisResult:
    """Result of analyzing loss curves from a training run."""

    policy_losses: list[float]
    value_losses: list[float]
    total_losses: list[float]
    iterations: list[int]

    policy_loss_initial: float | None
    policy_loss_final: float | None
    policy_convergence_rate: float

    value_loss_initial: float | None
    value_loss_final: float | None
    value_convergence_rate: float

    policy_fraction_of_total: float | None
    value_fraction_of_total: float | None

    policy_stalled: bool
    value_stalled: bool
    stall_iteration: int | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return dataclasses.asdict(self)


class LossAnalyzer:
    """Analyzes loss curves from a training run.

    Uses TrainingDiagnostics for stall detection (detect_learning_stall)
    and convergence rate computation (compute_win_rate_trend, which is
    a general-purpose linear regression helper).
    """

    def __init__(
        self, stall_lookback: int = 5, stall_threshold: float = 0.02
    ) -> None:
        self._stall_lookback = stall_lookback
        self._stall_threshold = stall_threshold
        self._diagnostics = TrainingDiagnostics(stall_threshold=stall_threshold)

    def analyze(self, run_data: RunData) -> LossAnalysisResult:
        """Analyze loss curves from *run_data* and return a structured result."""
        # Extract non-None loss values with their iteration numbers
        policy_losses: list[float] = []
        value_losses: list[float] = []
        total_losses: list[float] = []
        iterations: list[int] = []

        for it in run_data.iterations:
            if it.policy_loss is not None:
                policy_losses.append(it.policy_loss)
                if it.iteration not in iterations:
                    iterations.append(it.iteration)
            if it.value_loss is not None:
                value_losses.append(it.value_loss)
            if it.total_loss is not None:
                total_losses.append(it.total_loss)

        # For iterations list, use iterations that have at least one non-None loss.
        # Rebuild properly: track which iterations have any loss data.
        iterations = []
        policy_losses = []
        value_losses = []
        total_losses = []
        for it in run_data.iterations:
            has_data = (
                it.policy_loss is not None
                or it.value_loss is not None
                or it.total_loss is not None
            )
            if has_data:
                iterations.append(it.iteration)
                if it.policy_loss is not None:
                    policy_losses.append(it.policy_loss)
                if it.value_loss is not None:
                    value_losses.append(it.value_loss)
                if it.total_loss is not None:
                    total_losses.append(it.total_loss)

        # Initial and final values
        policy_loss_initial = policy_losses[0] if policy_losses else None
        policy_loss_final = policy_losses[-1] if policy_losses else None
        value_loss_initial = value_losses[0] if value_losses else None
        value_loss_final = value_losses[-1] if value_losses else None

        # Convergence rates via linear regression (compute_win_rate_trend works
        # on any numeric sequence — it just computes slope).
        policy_convergence_rate = self._diagnostics.compute_win_rate_trend(
            policy_losses, lookback=len(policy_losses)
        )
        value_convergence_rate = self._diagnostics.compute_win_rate_trend(
            value_losses, lookback=len(value_losses)
        )

        # Loss decomposition: average fraction of total
        policy_fraction = self._compute_fraction(run_data, "policy")
        value_fraction = self._compute_fraction(run_data, "value")

        # Stall detection
        policy_stalled = self._diagnostics.detect_learning_stall(
            policy_losses,
            lookback=self._stall_lookback,
            threshold=self._stall_threshold,
        )
        value_stalled = self._diagnostics.detect_learning_stall(
            value_losses,
            lookback=self._stall_lookback,
            threshold=self._stall_threshold,
        )

        # Find first stall iteration (scan with a sliding window)
        stall_iteration = self._find_first_stall(
            policy_losses, value_losses, iterations
        )

        return LossAnalysisResult(
            policy_losses=policy_losses,
            value_losses=value_losses,
            total_losses=total_losses,
            iterations=iterations,
            policy_loss_initial=policy_loss_initial,
            policy_loss_final=policy_loss_final,
            policy_convergence_rate=policy_convergence_rate,
            value_loss_initial=value_loss_initial,
            value_loss_final=value_loss_final,
            value_convergence_rate=value_convergence_rate,
            policy_fraction_of_total=policy_fraction,
            value_fraction_of_total=value_fraction,
            policy_stalled=policy_stalled,
            value_stalled=value_stalled,
            stall_iteration=stall_iteration,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_fraction(
        self, run_data: RunData, loss_type: str
    ) -> float | None:
        """Compute average fraction of total loss for *loss_type* ('policy' or 'value')."""
        fractions: list[float] = []
        for it in run_data.iterations:
            numerator = it.policy_loss if loss_type == "policy" else it.value_loss
            if numerator is not None and it.total_loss is not None and it.total_loss > 0:
                fractions.append(numerator / it.total_loss)
        if not fractions:
            return None
        return sum(fractions) / len(fractions)

    def _find_first_stall(
        self,
        policy_losses: list[float],
        value_losses: list[float],
        iterations: list[int],
    ) -> int | None:
        """Return the iteration number where the first stall window ends, or None."""
        lb = self._stall_lookback
        for series in (policy_losses, value_losses):
            if len(series) < lb:
                continue
            for end in range(lb, len(series) + 1):
                window = series[end - lb : end]
                if max(window) - min(window) < self._stall_threshold:
                    # Map back to iteration number
                    idx = end - 1
                    if idx < len(iterations):
                        return iterations[idx]
        return None
