"""Tests for diagnostics analysis engine."""

from __future__ import annotations

import pytest

from alpha_zero.analysis.data_model import (
    IterationData,
    RunData,
    RunIdentifier,
)
from alpha_zero.analysis.engines.diagnostics import DiagnosticsAnalyzer, DiagnosticsResult


def _make_run(iterations: list[IterationData]) -> RunData:
    return RunData(
        identifier=RunIdentifier(wandb_run_id="test-run"),
        iterations=iterations,
    )


class TestDiagnosticsAnalyzer:
    def test_value_head_instability_detected(self) -> None:
        """Correlations [0.1, 0.9, 0.2, 0.8] -> std > 0.3 -> unstable=True."""
        iters = [
            IterationData(iteration=i, value_correlation=v)
            for i, v in enumerate([0.1, 0.9, 0.2, 0.8])
        ]
        result = DiagnosticsAnalyzer(value_instability_threshold=0.3).analyze(_make_run(iters))
        assert result.value_head_unstable is True
        assert result.value_correlation_std is not None
        assert result.value_correlation_std > 0.3

    def test_value_head_stable(self) -> None:
        """Correlations [0.7, 0.72, 0.71, 0.73] -> std < 0.3 -> unstable=False."""
        iters = [
            IterationData(iteration=i, value_correlation=v)
            for i, v in enumerate([0.7, 0.72, 0.71, 0.73])
        ]
        result = DiagnosticsAnalyzer(value_instability_threshold=0.3).analyze(_make_run(iters))
        assert result.value_head_unstable is False

    def test_win_rate_stall_detected(self) -> None:
        """Flat win rates [0.5, 0.5, 0.5, 0.5, 0.5] -> stalled=True."""
        iters = [
            IterationData(iteration=i, eval_win_rate=0.5)
            for i in range(5)
        ]
        result = DiagnosticsAnalyzer(stall_lookback=5, stall_threshold=0.02).analyze(_make_run(iters))
        assert result.win_rate_stalled is True

    def test_human_readable_diagnosis(self) -> None:
        """Non-empty string with useful info."""
        iters = [
            IterationData(iteration=i, value_correlation=0.5, eval_win_rate=0.5 + i * 0.01, total_loss=1.0 - i * 0.05)
            for i in range(5)
        ]
        result = DiagnosticsAnalyzer().analyze(_make_run(iters))
        assert isinstance(result.diagnosis, str)
        assert len(result.diagnosis) > 0

    def test_empty_data_handled(self) -> None:
        """No metrics -> safe defaults, no crash."""
        result = DiagnosticsAnalyzer().analyze(_make_run([]))
        assert result.value_correlation_series == []
        assert result.value_correlation_min is None
        assert result.value_correlation_max is None
        assert result.value_correlation_std is None
        assert result.value_head_unstable is False
        assert result.win_rate_series == []
        assert result.win_rate_stalled is False
        assert result.win_rate_trend == 0.0
        assert result.loss_stalled is False
        assert isinstance(result.diagnosis, str)

    def test_win_rate_trend_positive(self) -> None:
        """Increasing win rates -> positive trend."""
        iters = [
            IterationData(iteration=i, eval_win_rate=0.4 + i * 0.05)
            for i in range(5)
        ]
        result = DiagnosticsAnalyzer().analyze(_make_run(iters))
        assert result.win_rate_trend > 0.0

    def test_none_correlations_skipped(self) -> None:
        """None value_correlation entries are skipped."""
        iters = [
            IterationData(iteration=0, value_correlation=0.5),
            IterationData(iteration=1, value_correlation=None),
            IterationData(iteration=2, value_correlation=0.6),
        ]
        result = DiagnosticsAnalyzer().analyze(_make_run(iters))
        assert len(result.value_correlation_series) == 2

    def test_to_dict(self) -> None:
        """DiagnosticsResult.to_dict() works."""
        result = DiagnosticsAnalyzer().analyze(_make_run([]))
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "diagnosis" in d
