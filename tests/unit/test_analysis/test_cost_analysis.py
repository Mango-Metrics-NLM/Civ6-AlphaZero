"""Tests for cost analysis engine."""

from __future__ import annotations

import pytest

from alpha_zero.analysis.data_model import (
    IterationData,
    ModelVersionData,
    RunData,
    RunIdentifier,
)
from alpha_zero.analysis.engines.cost_analysis import CostAnalyzer, CostAnalysisResult


def _make_run(iterations: list[IterationData], models: list[ModelVersionData] | None = None) -> RunData:
    return RunData(
        identifier=RunIdentifier(wandb_run_id="test-run"),
        iterations=iterations,
        models=models or [],
    )


class TestCostAnalyzer:
    def test_total_cost_summed(self) -> None:
        """3 iterations with costs -> correct total."""
        iters = [
            IterationData(iteration=0, selfplay_cost=2.0, training_cost=1.0, eval_cost=0.5),
            IterationData(iteration=1, selfplay_cost=3.0, training_cost=1.5, eval_cost=0.5),
            IterationData(iteration=2, selfplay_cost=1.0, training_cost=0.5, eval_cost=0.0),
        ]
        result = CostAnalyzer(gpu_costs={}).analyze(_make_run(iters))
        assert result.total_cost == pytest.approx(10.0)
        assert result.selfplay_cost == pytest.approx(6.0)
        assert result.training_cost == pytest.approx(3.0)
        assert result.eval_cost == pytest.approx(1.0)

    def test_cost_breakdown_percentages(self) -> None:
        """selfplay=6, training=3, eval=1 -> 60%, 30%, 10%."""
        iters = [
            IterationData(iteration=0, selfplay_cost=6.0, training_cost=3.0, eval_cost=1.0),
        ]
        result = CostAnalyzer(gpu_costs={}).analyze(_make_run(iters))
        assert result.cost_breakdown_pct["selfplay"] == pytest.approx(60.0)
        assert result.cost_breakdown_pct["training"] == pytest.approx(30.0)
        assert result.cost_breakdown_pct["eval"] == pytest.approx(10.0)

    def test_cost_per_promotion_with_promotions(self) -> None:
        """total=10, 2 promotions -> 5.0."""
        iters = [
            IterationData(iteration=0, selfplay_cost=3.0, training_cost=1.0, eval_cost=1.0, promoted=True),
            IterationData(iteration=1, selfplay_cost=3.0, training_cost=1.0, eval_cost=1.0, promoted=True),
        ]
        result = CostAnalyzer(gpu_costs={}).analyze(_make_run(iters))
        assert result.cost_per_promotion == pytest.approx(5.0)

    def test_cost_per_promotion_zero_promotions(self) -> None:
        """Zero promotions -> None."""
        iters = [
            IterationData(iteration=0, selfplay_cost=5.0, training_cost=3.0, eval_cost=2.0),
        ]
        result = CostAnalyzer(gpu_costs={}).analyze(_make_run(iters))
        assert result.cost_per_promotion is None

    def test_none_costs_treated_as_zero(self) -> None:
        """Some costs None -> treated as 0."""
        iters = [
            IterationData(iteration=0, selfplay_cost=2.0, training_cost=None, eval_cost=None),
            IterationData(iteration=1, selfplay_cost=None, training_cost=1.0, eval_cost=None),
        ]
        result = CostAnalyzer(gpu_costs={}).analyze(_make_run(iters))
        assert result.total_cost == pytest.approx(3.0)
        assert result.selfplay_cost == pytest.approx(2.0)
        assert result.training_cost == pytest.approx(1.0)
        assert result.eval_cost == pytest.approx(0.0)

    def test_cost_per_iteration(self) -> None:
        """cost_per_iteration = total / num_iterations."""
        iters = [
            IterationData(iteration=0, selfplay_cost=2.0, training_cost=1.0, eval_cost=1.0),
            IterationData(iteration=1, selfplay_cost=2.0, training_cost=1.0, eval_cost=1.0),
        ]
        result = CostAnalyzer(gpu_costs={}).analyze(_make_run(iters))
        assert result.cost_per_iteration == pytest.approx(4.0)

    def test_empty_iterations(self) -> None:
        """No iterations -> safe defaults."""
        result = CostAnalyzer(gpu_costs={}).analyze(_make_run([]))
        assert result.total_cost == 0.0
        assert result.cost_per_iteration == 0.0
        assert result.cost_per_promotion is None

    def test_cost_per_elo_point(self) -> None:
        """ELO spread exists -> cost_per_elo_point computed."""
        iters = [
            IterationData(iteration=0, selfplay_cost=5.0, training_cost=3.0, eval_cost=2.0),
        ]
        models = [
            ModelVersionData(version=0, iteration=0, elo=1400.0),
            ModelVersionData(version=1, iteration=1, elo=1500.0),
        ]
        result = CostAnalyzer(gpu_costs={}).analyze(_make_run(iters, models))
        assert result.cost_per_elo_point == pytest.approx(0.1)

    def test_cost_per_elo_point_no_elo(self) -> None:
        """No ELO data -> None."""
        iters = [
            IterationData(iteration=0, selfplay_cost=5.0, training_cost=3.0, eval_cost=2.0),
        ]
        result = CostAnalyzer(gpu_costs={}).analyze(_make_run(iters))
        assert result.cost_per_elo_point is None

    def test_per_iteration_costs(self) -> None:
        """per_iteration_costs list has correct structure."""
        iters = [
            IterationData(iteration=0, selfplay_cost=2.0, training_cost=1.0, eval_cost=0.5),
            IterationData(iteration=1, selfplay_cost=3.0, training_cost=1.5, eval_cost=0.5),
        ]
        result = CostAnalyzer(gpu_costs={}).analyze(_make_run(iters))
        assert len(result.per_iteration_costs) == 2
        assert result.per_iteration_costs[0]["iteration"] == 0
        assert result.per_iteration_costs[0]["total"] == pytest.approx(3.5)
        assert result.per_iteration_costs[1]["total"] == pytest.approx(5.0)

    def test_to_dict(self) -> None:
        """CostAnalysisResult.to_dict() works."""
        iters = [
            IterationData(iteration=0, selfplay_cost=1.0, training_cost=1.0, eval_cost=1.0),
        ]
        result = CostAnalyzer(gpu_costs={}).analyze(_make_run(iters))
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "total_cost" in d
