"""Tests for the promotion decision in the legacy cloud evaluator.

``alpha_zero/cloud/evaluator.py`` is not the deployed evaluation path (see the
module's own warning), but it decides promotion independently and is a
divergence risk if revived. These tests pin the two properties that matter:

1. **The default reproduces the historical gate exactly.** Anything else would
   silently change which candidates get promoted.
2. **Opting into a draw-aware policy actually changes the outcome** -- and a
   promotion is asserted, not just a rejection. ``test_promotion_chain.py``
   records that no test in this repository ever asserted a *promotion*, so a
   gate that could never promote would have looked healthy.
"""

from __future__ import annotations

import pytest

from alpha_zero.cloud.evaluator import CloudEvaluator, EvaluatorConfig
from alpha_zero.evaluation.scoring import score_rate

# A real draw-heavy record (CLAUDE.md Section 5.2). Under the legacy statistic
# the candidate scores 0.1775 and can never reach a 0.55 threshold, because
# draws are counted as losses; scored draw-aware it is 0.545.
_DRAW_HEAVY = (142, 588, 70)  # (candidate_wins, draws, champion_wins)


@pytest.fixture(autouse=True)
def _clear_promotion_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep ambient environment out of these tests."""
    for var in (
        "CIV6AZ_PROMOTION_POLICY",
        "CIV6AZ_PROMOTION_THRESHOLD",
        "CIV6AZ_PROMOTION_CONFIDENCE",
        "CIV6AZ_PROMOTION_MIN_GAMES",
        "CIV6_PROMOTION_THRESHOLD",
    ):
        monkeypatch.delenv(var, raising=False)


class _CountingStorage:
    """Storage stub that captures whatever the evaluator saves."""

    def __init__(self) -> None:
        self.saved: list = []

    def save_evaluation(self, result) -> None:
        self.saved.append(result)

    def list_evaluations(self) -> list:
        return list(self.saved)


def _evaluator(threshold: float = 0.55, **overrides) -> CloudEvaluator:
    config = EvaluatorConfig(promotion_threshold=threshold, **overrides)
    return CloudEvaluator(config=config, storage=_CountingStorage())


class TestDefaultPolicyIsUnchanged:
    """The default must remain byte-identical to the historical gate."""

    @pytest.mark.parametrize(
        "wins,draws,losses",
        [_DRAW_HEAVY, (300, 0, 100), (0, 0, 10), (5, 5, 5)],
    )
    def test_statistic_matches_wins_over_total(self, wins, draws, losses) -> None:
        evaluator = _evaluator()
        verdict = evaluator._promotion_verdict(wins, losses, draws)
        total = wins + draws + losses
        assert verdict.statistic == pytest.approx(wins / total)

    def test_draw_heavy_record_is_not_promoted(self) -> None:
        wins, draws, losses = _DRAW_HEAVY
        verdict = _evaluator()._promotion_verdict(wins, losses, draws)
        assert verdict.promoted is False

    def test_clear_winner_is_promoted(self) -> None:
        """A genuine promotion under the default policy."""
        verdict = _evaluator()._promotion_verdict(300, 100, 0)
        assert verdict.promoted is True


class TestPolicyOptIn:
    """Opting into draw-aware scoring changes the decision."""

    def test_score_rate_promotes_the_draw_heavy_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CIV6AZ_PROMOTION_POLICY", "score_rate")
        wins, draws, losses = _DRAW_HEAVY

        verdict = _evaluator(threshold=0.52)._promotion_verdict(wins, losses, draws)

        assert verdict.policy == "score_rate"
        assert verdict.statistic == pytest.approx(score_rate(wins, draws, losses))
        # The assertion this repository never had: a promotion actually happens.
        assert verdict.promoted is True

    def test_same_record_rejected_under_default_policy(self) -> None:
        """Control: the flip above comes from the policy, not the threshold."""
        wins, draws, losses = _DRAW_HEAVY
        verdict = _evaluator(threshold=0.52)._promotion_verdict(wins, losses, draws)
        assert verdict.promoted is False


class TestThresholdPrecedence:
    """The evaluator's own threshold must not be orphaned by the registry."""

    def test_config_threshold_governs(self) -> None:
        # 0.60 win rate: promoted at threshold 0.55, rejected at 0.75.
        assert _evaluator(threshold=0.55)._promotion_verdict(60, 40, 0).promoted is True
        assert _evaluator(threshold=0.75)._promotion_verdict(60, 40, 0).promoted is False

    def test_env_threshold_does_not_override_explicit_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicitly configured threshold wins over the ambient env var."""
        monkeypatch.setenv("CIV6AZ_PROMOTION_THRESHOLD", "0.05")
        assert _evaluator(threshold=0.75)._promotion_verdict(60, 40, 0).promoted is False


class TestEarlyStopIsPolicyConsistent:
    """Early stopping must measure what the final gate measures."""

    def test_legacy_policy_early_stop_unchanged(self) -> None:
        """Under the default policy the statistic is still wins/total."""
        evaluator = _evaluator(threshold=0.55, early_stop_threshold=10, early_stop_margin=0.15)
        # 90% win rate over 100 games: a decisive lead under either reading.
        should_stop, reason = evaluator._should_early_stop(90, 10, 0)
        assert should_stop is True
        assert reason == "decisive_lead"

        # 10% win rate: decisive loss.
        should_stop, reason = evaluator._should_early_stop(10, 90, 0)
        assert should_stop is True
        assert reason == "decisive_loss"

    def test_below_threshold_never_stops(self) -> None:
        evaluator = _evaluator(early_stop_threshold=500)
        assert evaluator._should_early_stop(90, 10, 0) == (False, "")

    def test_draw_heavy_stop_follows_the_configured_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two policies disagree on this record -- early stop must follow suit.

        Legacy scores it 0.1775 (a decisive loss); draw-aware scores it 0.545
        (no decision). Mixing the two would let a heuristic stop a run that the
        final gate would have promoted.
        """
        wins, draws, losses = _DRAW_HEAVY
        kwargs = {"threshold": 0.55, "early_stop_threshold": 10, "early_stop_margin": 0.15}

        legacy_stop, legacy_reason = _evaluator(**kwargs)._should_early_stop(wins, losses, draws)
        assert (legacy_stop, legacy_reason) == (True, "decisive_loss")

        monkeypatch.setenv("CIV6AZ_PROMOTION_POLICY", "score_rate")
        aware_stop, _ = _evaluator(**kwargs)._should_early_stop(wins, losses, draws)
        assert aware_stop is False
