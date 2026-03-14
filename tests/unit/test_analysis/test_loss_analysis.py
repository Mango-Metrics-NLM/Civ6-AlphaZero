"""Tests for the loss analysis engine."""

from __future__ import annotations

import pytest

from alpha_zero.analysis.data_model import IterationData, RunData, RunIdentifier
from alpha_zero.analysis.engines.loss_analysis import LossAnalyzer, LossAnalysisResult


def _make_run(iterations: list[IterationData]) -> RunData:
    return RunData(identifier=RunIdentifier(), iterations=iterations)


class TestLossAnalyzer:
    def test_convergence_rate_negative_for_decreasing_loss(self) -> None:
        """Losses [1.0, 0.8, 0.6, 0.4] should yield a negative convergence rate."""
        iters = [
            IterationData(iteration=i, policy_loss=v, value_loss=v * 0.5, total_loss=v * 1.5)
            for i, v in enumerate([1.0, 0.8, 0.6, 0.4])
        ]
        result = LossAnalyzer().analyze(_make_run(iters))

        assert result.policy_convergence_rate < 0.0
        assert result.value_convergence_rate < 0.0

    def test_stall_detection(self) -> None:
        """Flat losses [0.5]*5 should be detected as stalled."""
        iters = [
            IterationData(iteration=i, policy_loss=0.5, value_loss=0.3, total_loss=0.8)
            for i in range(5)
        ]
        analyzer = LossAnalyzer(stall_lookback=5, stall_threshold=0.02)
        result = analyzer.analyze(_make_run(iters))

        assert result.policy_stalled is True
        assert result.value_stalled is True

    def test_policy_value_decomposition(self) -> None:
        """When policy=0.6, value=0.4, total=1.0 fractions should be ~0.6 and ~0.4."""
        iters = [
            IterationData(iteration=i, policy_loss=0.6, value_loss=0.4, total_loss=1.0)
            for i in range(3)
        ]
        result = LossAnalyzer().analyze(_make_run(iters))

        assert result.policy_fraction_of_total is not None
        assert result.value_fraction_of_total is not None
        assert abs(result.policy_fraction_of_total - 0.6) < 1e-6
        assert abs(result.value_fraction_of_total - 0.4) < 1e-6

    def test_empty_iterations_handled(self) -> None:
        """RunData with no iterations should not crash and return safe defaults."""
        result = LossAnalyzer().analyze(_make_run([]))

        assert result.policy_losses == []
        assert result.value_losses == []
        assert result.total_losses == []
        assert result.iterations == []
        assert result.policy_loss_initial is None
        assert result.policy_loss_final is None
        assert result.policy_convergence_rate == 0.0
        assert result.value_convergence_rate == 0.0
        assert result.policy_fraction_of_total is None
        assert result.value_fraction_of_total is None
        assert result.policy_stalled is False
        assert result.value_stalled is False
        assert result.stall_iteration is None

    def test_none_losses_skipped(self) -> None:
        """Iterations with None losses should be excluded from results."""
        iters = [
            IterationData(iteration=0, policy_loss=1.0, value_loss=0.5, total_loss=1.5),
            IterationData(iteration=1),  # all None
            IterationData(iteration=2, policy_loss=0.8, value_loss=0.4, total_loss=1.2),
        ]
        result = LossAnalyzer().analyze(_make_run(iters))

        assert len(result.policy_losses) == 2
        assert len(result.value_losses) == 2
        assert result.iterations == [0, 2]

    def test_initial_and_final_values(self) -> None:
        """First and last non-None values should be extracted."""
        iters = [
            IterationData(iteration=0),  # None
            IterationData(iteration=1, policy_loss=1.2, value_loss=0.9, total_loss=2.1),
            IterationData(iteration=2, policy_loss=0.8, value_loss=0.5, total_loss=1.3),
            IterationData(iteration=3),  # None
        ]
        result = LossAnalyzer().analyze(_make_run(iters))

        assert result.policy_loss_initial == 1.2
        assert result.policy_loss_final == 0.8
        assert result.value_loss_initial == 0.9
        assert result.value_loss_final == 0.5

    def test_to_dict_returns_dict(self) -> None:
        """LossAnalysisResult.to_dict() should return a plain dict."""
        result = LossAnalyzer().analyze(_make_run([]))
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "policy_convergence_rate" in d
