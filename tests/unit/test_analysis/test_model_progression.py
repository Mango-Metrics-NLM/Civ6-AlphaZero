"""Tests for the model progression engine."""

from __future__ import annotations

import pytest

from alpha_zero.analysis.data_model import (
    IterationData,
    ModelVersionData,
    RunData,
    RunIdentifier,
)
from alpha_zero.analysis.engines.model_progression import (
    ModelProgressionAnalyzer,
    ModelProgressionResult,
)


def _make_run(
    iterations: list[IterationData] | None = None,
    models: list[ModelVersionData] | None = None,
) -> RunData:
    return RunData(
        identifier=RunIdentifier(),
        iterations=iterations or [],
        models=models or [],
    )


class TestModelProgressionAnalyzer:
    def test_promotion_chain_extracted(self) -> None:
        """2 iterations with promoted=True should yield a chain of length 2."""
        iters = [
            IterationData(iteration=0, model_version=1, promoted=False),
            IterationData(iteration=1, model_version=2, promoted=True),
            IterationData(iteration=2, model_version=3, promoted=False),
            IterationData(iteration=3, model_version=4, promoted=True),
        ]
        result = ModelProgressionAnalyzer().analyze(_make_run(iters))

        assert result.total_promotions == 2
        assert len(result.promotion_chain) == 2
        assert result.promotion_chain[0]["iteration"] == 1
        assert result.promotion_chain[0]["model_version"] == 2
        assert result.promotion_chain[1]["iteration"] == 3
        assert result.promotion_chain[1]["model_version"] == 4
        assert result.promotion_iterations == [1, 3]

    def test_training_from_scratch_detected(self) -> None:
        """10 iterations all with total_loss=1.2, 0 promotions -> detected=True."""
        iters = [
            IterationData(iteration=i, total_loss=1.2, promoted=False)
            for i in range(10)
        ]
        analyzer = ModelProgressionAnalyzer(fresh_model_loss_threshold=0.8)
        result = analyzer.analyze(_make_run(iters))

        assert result.training_from_scratch_detected is True

    def test_healthy_run_no_bug_detected(self) -> None:
        """Losses decrease and promotions happen -> no bug detected."""
        losses = [1.5, 1.2, 1.0, 0.8, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25]
        iters = [
            IterationData(
                iteration=i,
                total_loss=loss,
                promoted=(i == 4 or i == 8),
                model_version=i,
            )
            for i, loss in enumerate(losses)
        ]
        result = ModelProgressionAnalyzer().analyze(_make_run(iters))

        assert result.training_from_scratch_detected is False
        assert result.total_promotions == 2

    def test_elo_trajectory_from_models(self) -> None:
        """3 models with ELO should produce elo_trajectory with 3 entries."""
        models = [
            ModelVersionData(version=1, iteration=0, elo=1500.0),
            ModelVersionData(version=2, iteration=5, elo=1520.0),
            ModelVersionData(version=3, iteration=10, elo=1550.0),
        ]
        result = ModelProgressionAnalyzer().analyze(_make_run(models=models))

        assert len(result.elo_trajectory) == 3
        assert result.elo_trajectory[0]["version"] == 1
        assert result.elo_trajectory[0]["elo"] == 1500.0
        assert result.elo_trajectory[2]["elo"] == 1550.0

    def test_promotion_rate_calculation(self) -> None:
        """3 promotions in 50 iterations -> rate=0.06."""
        iters = [
            IterationData(iteration=i, promoted=(i in {10, 25, 40}))
            for i in range(50)
        ]
        result = ModelProgressionAnalyzer().analyze(_make_run(iters))

        assert result.total_promotions == 3
        assert abs(result.promotion_rate - 3.0 / 50.0) < 1e-9

    def test_zero_promotions_handled(self) -> None:
        """0 promotions should yield rate=0.0 with no crash."""
        iters = [IterationData(iteration=i) for i in range(10)]
        result = ModelProgressionAnalyzer().analyze(_make_run(iters))

        assert result.total_promotions == 0
        assert result.promotion_rate == 0.0
        assert result.promotion_chain == []
        assert result.promotion_iterations == []

    def test_diagnosis_string_not_empty(self) -> None:
        """Diagnosis should always be a non-empty string."""
        # Empty run
        result1 = ModelProgressionAnalyzer().analyze(_make_run())
        assert isinstance(result1.diagnosis, str)
        assert len(result1.diagnosis) > 0

        # Run with data
        iters = [IterationData(iteration=i, promoted=(i == 3)) for i in range(5)]
        result2 = ModelProgressionAnalyzer().analyze(_make_run(iters))
        assert isinstance(result2.diagnosis, str)
        assert len(result2.diagnosis) > 0

    def test_to_dict_returns_dict(self) -> None:
        """ModelProgressionResult.to_dict() should return a plain dict."""
        result = ModelProgressionAnalyzer().analyze(_make_run())
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "training_from_scratch_detected" in d

    def test_empty_run_handled(self) -> None:
        """Empty RunData should not crash."""
        result = ModelProgressionAnalyzer().analyze(_make_run())

        assert result.versions == []
        assert result.total_promotions == 0
        assert result.promotion_rate == 0.0
        assert result.training_from_scratch_detected is False
