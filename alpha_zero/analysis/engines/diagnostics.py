"""Diagnostics engine for training run health analysis.

Detects value head instability, win rate stalls, and loss stalls
using the existing TrainingDiagnostics module. Produces a human-readable
diagnosis string summarizing findings.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np

from alpha_zero.analysis.data_model import RunData
from alpha_zero.training.diagnostics import TrainingDiagnostics


@dataclass
class DiagnosticsResult:
    """Result of diagnostics analysis on a training run."""

    value_correlation_series: list[float]
    value_correlation_min: float | None
    value_correlation_max: float | None
    value_correlation_std: float | None
    value_head_unstable: bool  # std > threshold
    win_rate_series: list[float]
    win_rate_stalled: bool
    win_rate_trend: float  # Slope from linear regression
    loss_stalled: bool
    diagnosis: str  # Human-readable multi-line summary

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return dataclasses.asdict(self)


class DiagnosticsAnalyzer:
    """Analyze training run health: value stability, stalls, trends."""

    def __init__(
        self,
        value_instability_threshold: float = 0.3,
        stall_lookback: int = 5,
        stall_threshold: float = 0.02,
    ) -> None:
        self._value_threshold = value_instability_threshold
        self._stall_lookback = stall_lookback
        self._stall_threshold = stall_threshold
        self._diagnostics = TrainingDiagnostics(stall_threshold=stall_threshold)

    def analyze(self, run_data: RunData) -> DiagnosticsResult:
        """Analyze diagnostics from a training run.

        Args:
            run_data: Normalized run data with iteration-level metrics.

        Returns:
            DiagnosticsResult with stability analysis and human-readable diagnosis.
        """
        # Extract value correlation series (skip None)
        value_corrs = [
            it.value_correlation
            for it in run_data.iterations
            if it.value_correlation is not None
        ]

        if value_corrs:
            arr = np.array(value_corrs, dtype=np.float64)
            vc_min = float(np.min(arr))
            vc_max = float(np.max(arr))
            vc_std = float(np.std(arr))
            value_head_unstable = vc_std > self._value_threshold
        else:
            vc_min = None
            vc_max = None
            vc_std = None
            value_head_unstable = False

        # Extract win rate series (skip None)
        win_rates = [
            it.eval_win_rate
            for it in run_data.iterations
            if it.eval_win_rate is not None
        ]

        # Win rate stall detection
        win_rate_stalled = self._diagnostics.detect_learning_stall(
            win_rates,
            lookback=self._stall_lookback,
            threshold=self._stall_threshold,
        ) if win_rates else False

        # Win rate trend
        win_rate_trend = self._diagnostics.compute_win_rate_trend(
            win_rates,
            lookback=self._stall_lookback,
        ) if win_rates else 0.0

        # Loss stall detection on total_loss
        total_losses = [
            it.total_loss
            for it in run_data.iterations
            if it.total_loss is not None
        ]
        loss_stalled = self._diagnostics.detect_learning_stall(
            total_losses,
            lookback=self._stall_lookback,
            threshold=self._stall_threshold,
        ) if total_losses else False

        # Generate human-readable diagnosis
        diagnosis = self._build_diagnosis(
            value_corrs=value_corrs,
            vc_std=vc_std,
            value_head_unstable=value_head_unstable,
            win_rates=win_rates,
            win_rate_stalled=win_rate_stalled,
            win_rate_trend=win_rate_trend,
            loss_stalled=loss_stalled,
        )

        return DiagnosticsResult(
            value_correlation_series=value_corrs,
            value_correlation_min=vc_min,
            value_correlation_max=vc_max,
            value_correlation_std=vc_std,
            value_head_unstable=value_head_unstable,
            win_rate_series=win_rates,
            win_rate_stalled=win_rate_stalled,
            win_rate_trend=win_rate_trend,
            loss_stalled=loss_stalled,
            diagnosis=diagnosis,
        )

    def _build_diagnosis(
        self,
        *,
        value_corrs: list[float],
        vc_std: float | None,
        value_head_unstable: bool,
        win_rates: list[float],
        win_rate_stalled: bool,
        win_rate_trend: float,
        loss_stalled: bool,
    ) -> str:
        """Build a human-readable diagnosis summary."""
        lines: list[str] = []

        # Value head analysis
        if not value_corrs:
            lines.append("Value head: No correlation data available.")
        elif value_head_unstable:
            lines.append(
                f"Value head: UNSTABLE (std={vc_std:.3f}, threshold={self._value_threshold:.2f}). "
                f"Range [{min(value_corrs):.3f}, {max(value_corrs):.3f}]."
            )
        else:
            lines.append(
                f"Value head: Stable (std={vc_std:.3f}). "
                f"Range [{min(value_corrs):.3f}, {max(value_corrs):.3f}]."
            )

        # Win rate analysis
        if not win_rates:
            lines.append("Win rate: No evaluation data available.")
        else:
            trend_desc = "improving" if win_rate_trend > 0.01 else ("declining" if win_rate_trend < -0.01 else "flat")
            lines.append(
                f"Win rate: {trend_desc} (trend={win_rate_trend:.4f}/iter). "
                f"Latest={win_rates[-1]:.3f}."
            )
            if win_rate_stalled:
                lines.append("Win rate: STALLED over recent iterations.")

        # Loss analysis
        if loss_stalled:
            lines.append("Loss: STALLED - training loss not decreasing.")

        return "\n".join(lines)
