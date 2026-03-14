"""Cost analysis engine for training runs.

Computes per-iteration and aggregate cost breakdowns across selfplay,
training, and evaluation phases. Supports configurable GPU pricing.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from alpha_zero.analysis.data_model import RunData


@dataclass
class CostAnalysisResult:
    """Result of cost analysis on a training run."""

    total_cost: float
    selfplay_cost: float
    training_cost: float
    eval_cost: float
    cost_per_iteration: float  # total / num_iterations (0 if no iterations)
    cost_per_promotion: float | None  # total / promotions (None if 0 promotions)
    cost_breakdown_pct: dict[str, float]  # {"selfplay": %, "training": %, "eval": %}
    per_iteration_costs: list[dict[str, float]]  # [{iteration, selfplay, training, eval, total}]
    cost_per_elo_point: float | None  # total / elo_spread (None if no ELO data)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return dataclasses.asdict(self)


class CostAnalyzer:
    """Analyze training run costs across selfplay, training, and evaluation."""

    def __init__(self, gpu_costs: dict[str, float] | None = None) -> None:
        if gpu_costs is None:
            try:
                from alpha_zero.cloud.gcp.config import get_gpu_costs

                self._gpu_costs = get_gpu_costs()
            except ImportError:
                self._gpu_costs = {}
        else:
            self._gpu_costs = gpu_costs

    def analyze(self, run_data: RunData) -> CostAnalysisResult:
        """Analyze costs from a training run.

        Args:
            run_data: Normalized run data with iteration-level costs.

        Returns:
            CostAnalysisResult with aggregate and per-iteration cost breakdowns.
        """
        selfplay_total = 0.0
        training_total = 0.0
        eval_total = 0.0
        per_iteration_costs: list[dict[str, float]] = []

        for it in run_data.iterations:
            sp = it.selfplay_cost if it.selfplay_cost is not None else 0.0
            tr = it.training_cost if it.training_cost is not None else 0.0
            ev = it.eval_cost if it.eval_cost is not None else 0.0

            selfplay_total += sp
            training_total += tr
            eval_total += ev

            per_iteration_costs.append({
                "iteration": it.iteration,
                "selfplay": sp,
                "training": tr,
                "eval": ev,
                "total": sp + tr + ev,
            })

        total_cost = selfplay_total + training_total + eval_total

        # Breakdown percentages
        if total_cost > 0:
            cost_breakdown_pct = {
                "selfplay": (selfplay_total / total_cost) * 100.0,
                "training": (training_total / total_cost) * 100.0,
                "eval": (eval_total / total_cost) * 100.0,
            }
        else:
            cost_breakdown_pct = {"selfplay": 0.0, "training": 0.0, "eval": 0.0}

        # Cost per iteration
        num_iterations = run_data.total_iterations
        cost_per_iteration = total_cost / num_iterations if num_iterations > 0 else 0.0

        # Cost per promotion
        promotions = run_data.promotions
        cost_per_promotion = total_cost / promotions if promotions > 0 else None

        # Cost per ELO point
        cost_per_elo_point = self._compute_cost_per_elo(run_data, total_cost)

        return CostAnalysisResult(
            total_cost=total_cost,
            selfplay_cost=selfplay_total,
            training_cost=training_total,
            eval_cost=eval_total,
            cost_per_iteration=cost_per_iteration,
            cost_per_promotion=cost_per_promotion,
            cost_breakdown_pct=cost_breakdown_pct,
            per_iteration_costs=per_iteration_costs,
            cost_per_elo_point=cost_per_elo_point,
        )

    def _compute_cost_per_elo(self, run_data: RunData, total_cost: float) -> float | None:
        """Compute cost per ELO point from model ELO spread."""
        elos = [m.elo for m in run_data.models if m.elo is not None]
        if len(elos) < 2:
            return None
        elo_spread = max(elos) - min(elos)
        if elo_spread <= 0:
            return None
        return total_cost / elo_spread
