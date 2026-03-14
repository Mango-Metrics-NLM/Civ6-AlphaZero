"""Selfplay quality analysis engine.

Extracts and aggregates selfplay metrics (games, trajectories, MCTS stats)
from training run data.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from alpha_zero.analysis.data_model import RunData


@dataclass
class SelfplayQualityResult:
    """Result of selfplay quality analysis."""

    games_per_iteration: list[int]
    total_games: int
    avg_games_per_iteration: float
    trajectory_counts_per_iteration: list[int]
    total_trajectories: int
    avg_mcts_depth: float | None
    avg_mcts_visits: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return dataclasses.asdict(self)


class SelfplayQualityAnalyzer:
    """Analyze selfplay quality metrics from a training run."""

    def analyze(self, run_data: RunData) -> SelfplayQualityResult:
        """Analyze selfplay quality from a training run.

        Args:
            run_data: Normalized run data with iteration-level metrics.

        Returns:
            SelfplayQualityResult with game counts, trajectory counts, and MCTS stats.
        """
        games_per_iteration = [
            it.games_completed if it.games_completed is not None else 0
            for it in run_data.iterations
        ]
        trajectory_counts = [
            it.trajectory_count if it.trajectory_count is not None else 0
            for it in run_data.iterations
        ]

        total_games = sum(games_per_iteration)
        total_trajectories = sum(trajectory_counts)

        num_iters = len(run_data.iterations)
        avg_games = total_games / num_iters if num_iters > 0 else 0.0

        # MCTS depth/visits are not currently stored in IterationData
        avg_mcts_depth: float | None = None
        avg_mcts_visits: float | None = None

        return SelfplayQualityResult(
            games_per_iteration=games_per_iteration,
            total_games=total_games,
            avg_games_per_iteration=avg_games,
            trajectory_counts_per_iteration=trajectory_counts,
            total_trajectories=total_trajectories,
            avg_mcts_depth=avg_mcts_depth,
            avg_mcts_visits=avg_mcts_visits,
        )
