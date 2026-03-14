"""Tests for selfplay quality analysis engine."""

from __future__ import annotations

import pytest

from alpha_zero.analysis.data_model import (
    IterationData,
    RunData,
    RunIdentifier,
)
from alpha_zero.analysis.engines.selfplay_quality import (
    SelfplayQualityAnalyzer,
    SelfplayQualityResult,
)


def _make_run(iterations: list[IterationData]) -> RunData:
    return RunData(
        identifier=RunIdentifier(wandb_run_id="test-run"),
        iterations=iterations,
    )


class TestSelfplayQualityAnalyzer:
    def test_games_per_iteration_extracted(self) -> None:
        """3 iterations with games -> correct list."""
        iters = [
            IterationData(iteration=0, games_completed=50),
            IterationData(iteration=1, games_completed=60),
            IterationData(iteration=2, games_completed=40),
        ]
        result = SelfplayQualityAnalyzer().analyze(_make_run(iters))
        assert result.games_per_iteration == [50, 60, 40]

    def test_total_games_summed(self) -> None:
        """Sum matches."""
        iters = [
            IterationData(iteration=0, games_completed=50),
            IterationData(iteration=1, games_completed=60),
            IterationData(iteration=2, games_completed=40),
        ]
        result = SelfplayQualityAnalyzer().analyze(_make_run(iters))
        assert result.total_games == 150

    def test_trajectory_counts(self) -> None:
        """trajectory_count values extracted."""
        iters = [
            IterationData(iteration=0, trajectory_count=100),
            IterationData(iteration=1, trajectory_count=200),
            IterationData(iteration=2, trajectory_count=150),
        ]
        result = SelfplayQualityAnalyzer().analyze(_make_run(iters))
        assert result.trajectory_counts_per_iteration == [100, 200, 150]
        assert result.total_trajectories == 450

    def test_empty_data_handled(self) -> None:
        """Empty iterations -> total=0, avg=0.0, no crash."""
        result = SelfplayQualityAnalyzer().analyze(_make_run([]))
        assert result.total_games == 0
        assert result.avg_games_per_iteration == 0.0
        assert result.games_per_iteration == []
        assert result.trajectory_counts_per_iteration == []
        assert result.total_trajectories == 0

    def test_none_games_treated_as_zero(self) -> None:
        """None games_completed -> 0 in list."""
        iters = [
            IterationData(iteration=0, games_completed=50),
            IterationData(iteration=1, games_completed=None),
            IterationData(iteration=2, games_completed=30),
        ]
        result = SelfplayQualityAnalyzer().analyze(_make_run(iters))
        assert result.games_per_iteration == [50, 0, 30]
        assert result.total_games == 80

    def test_avg_games_per_iteration(self) -> None:
        """Average computed correctly."""
        iters = [
            IterationData(iteration=0, games_completed=30),
            IterationData(iteration=1, games_completed=60),
        ]
        result = SelfplayQualityAnalyzer().analyze(_make_run(iters))
        assert result.avg_games_per_iteration == pytest.approx(45.0)

    def test_mcts_depth_and_visits_none(self) -> None:
        """MCTS depth/visits not in iteration data -> None."""
        iters = [IterationData(iteration=0, games_completed=10)]
        result = SelfplayQualityAnalyzer().analyze(_make_run(iters))
        assert result.avg_mcts_depth is None
        assert result.avg_mcts_visits is None

    def test_to_dict(self) -> None:
        """SelfplayQualityResult.to_dict() works."""
        result = SelfplayQualityAnalyzer().analyze(_make_run([]))
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "total_games" in d
