"""Tests for promotion-stagnation detection in the cloud evaluator.

Covers the pure ``count_consecutive_non_promotions`` helper and the
``CloudEvaluator._check_promotion_stagnation`` wiring (patience gating, storage
errors, and the streak-reset-on-promotion semantics). This is the highest-
fidelity guard for the v1.2.0 "0 promotions" regression (CLAUDE.md Section 13).
"""

from __future__ import annotations

from alpha_zero.cloud.evaluator import (
    CloudEvaluator,
    EvaluatorConfig,
    count_consecutive_non_promotions,
)
from alpha_zero.storage.models import EvaluationResult


def _eval(iteration: int, promoted: bool) -> EvaluationResult:
    """Build a minimal EvaluationResult stub for the given iteration."""
    return EvaluationResult(
        iteration=iteration,
        candidate_version=iteration,
        champion_version=0,
        num_games=10,
        candidate_wins=5,
        champion_wins=5,
        draws=0,
        win_rate=0.5,
        promoted=promoted,
    )


class _FakeStorage:
    """Storage stub exposing only ``list_evaluations``."""

    def __init__(self, evaluations: list[EvaluationResult], raise_exc: bool = False) -> None:
        self._evaluations = evaluations
        self._raise = raise_exc

    def list_evaluations(self) -> list[EvaluationResult]:
        if self._raise:
            raise RuntimeError("storage offline")
        return self._evaluations


class TestCountConsecutiveNonPromotions:
    def test_empty_history_is_zero(self) -> None:
        assert count_consecutive_non_promotions([]) == 0

    def test_all_non_promotions(self) -> None:
        evals = [_eval(i, promoted=False) for i in range(5)]
        assert count_consecutive_non_promotions(evals) == 5

    def test_resets_at_most_recent_promotion(self) -> None:
        evals = [
            _eval(0, promoted=False),
            _eval(1, promoted=True),  # promotion resets the trailing streak
            _eval(2, promoted=False),
            _eval(3, promoted=False),
        ]
        assert count_consecutive_non_promotions(evals) == 2

    def test_counts_from_most_recent_regardless_of_input_order(self) -> None:
        # Unsorted input must still be ordered by iteration internally.
        evals = [
            _eval(3, promoted=False),
            _eval(0, promoted=False),
            _eval(2, promoted=False),
            _eval(1, promoted=True),
        ]
        assert count_consecutive_non_promotions(evals) == 2

    def test_trailing_promotion_yields_zero(self) -> None:
        evals = [_eval(0, promoted=False), _eval(1, promoted=True)]
        assert count_consecutive_non_promotions(evals) == 0


class TestCheckPromotionStagnation:
    def _evaluator(self, storage: _FakeStorage, patience: int) -> CloudEvaluator:
        config = EvaluatorConfig(device="cpu", stagnation_patience=patience)
        return CloudEvaluator(config=config, storage=storage)  # type: ignore[arg-type]

    def test_disabled_by_default(self) -> None:
        storage = _FakeStorage([_eval(i, promoted=False) for i in range(10)])
        evaluator = self._evaluator(storage, patience=0)
        result = evaluator._check_promotion_stagnation()
        assert result == {"stagnation_detected": False, "consecutive_non_promotions": 0}

    def test_triggers_at_patience(self) -> None:
        storage = _FakeStorage([_eval(i, promoted=False) for i in range(3)])
        evaluator = self._evaluator(storage, patience=3)
        result = evaluator._check_promotion_stagnation()
        assert result["stagnation_detected"] is True
        assert result["consecutive_non_promotions"] == 3

    def test_below_patience_not_detected(self) -> None:
        storage = _FakeStorage([_eval(i, promoted=False) for i in range(2)])
        evaluator = self._evaluator(storage, patience=3)
        result = evaluator._check_promotion_stagnation()
        assert result["stagnation_detected"] is False
        assert result["consecutive_non_promotions"] == 2

    def test_storage_error_is_non_fatal(self) -> None:
        storage = _FakeStorage([], raise_exc=True)
        evaluator = self._evaluator(storage, patience=3)
        result = evaluator._check_promotion_stagnation()
        assert result == {"stagnation_detected": False, "consecutive_non_promotions": 0}

    def test_from_env_reads_patience(self, monkeypatch) -> None:
        monkeypatch.setenv("CIV6_STAGNATION_PATIENCE", "7")
        config = EvaluatorConfig.from_env()
        assert config.stagnation_patience == 7
