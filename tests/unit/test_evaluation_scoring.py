"""Tests for the score-rate single source of truth.

These pin the two defects that motivated the module: draws scored as losses, and
a binomial variance applied to a ternary score.
"""

from __future__ import annotations

import math

import pytest

from alpha_zero.evaluation.scoring import (
    PromotionDecision,
    decisive_win_rate,
    effective_n,
    minimum_detectable_effect,
    promotion_decision,
    score_interval,
    score_rate,
    score_variance,
    summarize,
)

# The committed A/B artifact: docs/evidence/hexcrystal_ab/arena_result.json
AB_WINS, AB_DRAWS, AB_LOSSES = 142, 588, 70


class TestScoreRate:
    def test_draw_is_half_a_point_not_a_loss(self) -> None:
        """The defect this module exists to prevent."""
        assert score_rate(0, 10, 0) == 0.5
        # The old production statistic, wins / total, would score this 0.0.
        assert score_rate(5, 10, 5) == 0.5

    def test_matches_published_ab_score_rate(self) -> None:
        assert score_rate(AB_WINS, AB_DRAWS, AB_LOSSES) == pytest.approx(0.545, abs=5e-4)

    def test_all_wins_and_all_losses(self) -> None:
        assert score_rate(10, 0, 0) == 1.0
        assert score_rate(0, 0, 10) == 0.0

    def test_empty_record_is_no_information_not_a_loss(self) -> None:
        assert score_rate(0, 0, 0) == 0.5

    def test_negative_counts_rejected(self) -> None:
        with pytest.raises(ValueError):
            score_rate(-1, 0, 0)


class TestVariance:
    def test_binomial_overstates_ternary_variance_by_exactly_draw_rate_over_four(self) -> None:
        """The identity behind the ~2x-too-wide published interval."""
        for wins, draws, losses in [(142, 588, 70), (10, 5, 5), (1, 98, 1), (30, 0, 20)]:
            total = wins + draws + losses
            mu = score_rate(wins, draws, losses)
            binomial = mu * (1 - mu)
            ternary = score_variance(wins, draws, losses)
            assert binomial - ternary == pytest.approx(draws / total / 4.0, abs=1e-12)

    def test_draws_reduce_variance(self) -> None:
        """Draws sit near the mean, so they shrink variance on the score scale."""
        heavy_draws = score_variance(142, 588, 70)
        no_draws = score_variance(436, 0, 364)  # same score rate, zero draws
        assert score_rate(142, 588, 70) == pytest.approx(score_rate(436, 0, 364), abs=1e-3)
        assert heavy_draws < no_draws

    def test_zero_variance_when_every_game_is_a_draw(self) -> None:
        assert score_variance(0, 50, 0) == pytest.approx(0.0, abs=1e-12)


class TestInterval:
    def test_corrected_ab_interval(self) -> None:
        """The published interval was [0.5104, 0.5792]; the ternary one is ~2x tighter."""
        low, high = score_interval(AB_WINS, AB_DRAWS, AB_LOSSES)
        assert low == pytest.approx(0.5274, abs=1e-3)
        assert high == pytest.approx(0.5626, abs=1e-3)

        total = AB_WINS + AB_DRAWS + AB_LOSSES
        mu = score_rate(AB_WINS, AB_DRAWS, AB_LOSSES)
        binomial_half = 1.959963984540054 * math.sqrt(mu * (1 - mu) / total)
        assert binomial_half / ((high - low) / 2) == pytest.approx(1.96, abs=0.02)

    def test_interval_is_ordered_and_brackets_the_estimate(self) -> None:
        for wins, draws, losses in [(142, 588, 70), (1, 1, 1), (30, 10, 20), (0, 0, 9)]:
            low, high = score_interval(wins, draws, losses)
            assert low <= score_rate(wins, draws, losses) <= high
            assert low < high, f"zero-width interval for {(wins, draws, losses)}"

    def test_all_draw_record_keeps_positive_width_around_half(self) -> None:
        """Deliberate: N observed draws do not prove the population is draw-only.

        The pseudo-count adjustment keeps positive width so a finite all-draw
        sample never claims perfect certainty; the width shrinks with n, and
        such a record still can never promote (its lower bound sits below any
        threshold above 0.5).
        """
        low, high = score_interval(0, 50, 0)
        assert low < 0.5 < high
        big_low, big_high = score_interval(0, 400, 0)
        assert (big_high - big_low) < (high - low)
        assert promotion_decision(0, 400, 0, threshold=0.55) is not PromotionDecision.PROMOTE

    def test_bounds_clipped_to_unit_interval(self) -> None:
        low, high = score_interval(50, 0, 0)
        assert low >= 0.0 and high <= 1.0

    def test_unknown_confidence_rejected(self) -> None:
        with pytest.raises(ValueError):
            score_interval(1, 1, 1, confidence=0.5)


class TestPromotionDecision:
    def test_all_draws_never_promotes(self) -> None:
        assert promotion_decision(0, 400, 0) is PromotionDecision.INCONCLUSIVE

    def test_even_record_never_promotes_at_any_n(self) -> None:
        for n in (10, 100, 1000):
            assert promotion_decision(n, 0, n) is not PromotionDecision.PROMOTE

    def test_gate_returns_all_three_states(self) -> None:
        """The live gate cannot currently return PROMOTE at all; this proves it can."""
        assert promotion_decision(400, 0, 100) is PromotionDecision.PROMOTE
        assert promotion_decision(100, 0, 400) is PromotionDecision.REJECT
        assert promotion_decision(3, 0, 2) is PromotionDecision.INCONCLUSIVE

    def test_requires_lower_bound_above_threshold_not_point_estimate(self) -> None:
        """6W/4L has a point estimate of 0.6 but is far too noisy to promote."""
        assert score_rate(6, 0, 4) == 0.6
        assert promotion_decision(6, 0, 4) is PromotionDecision.INCONCLUSIVE

    def test_min_games_floor(self) -> None:
        assert promotion_decision(5, 0, 0, min_games=10) is PromotionDecision.INCONCLUSIVE


class TestDiagnostics:
    def test_decisive_win_rate_excludes_draws(self) -> None:
        assert decisive_win_rate(AB_WINS, AB_LOSSES) == pytest.approx(142 / 212, abs=1e-9)

    def test_decisive_win_rate_is_none_without_decisive_games(self) -> None:
        assert decisive_win_rate(0, 0) is None

    def test_effective_n_counts_decisive_games(self) -> None:
        assert effective_n(AB_WINS, AB_DRAWS, AB_LOSSES) == 212

    def test_mde_widens_as_draw_rate_falls(self) -> None:
        """Counter-intuitive, and the reason a draw-rate ceiling is the wrong gate."""
        assert minimum_detectable_effect(400, 0.20) > minimum_detectable_effect(400, 0.735)

    def test_mde_shrinks_with_sample_size(self) -> None:
        assert minimum_detectable_effect(400, 0.5) < minimum_detectable_effect(20, 0.5)

    def test_summary_round_trips(self) -> None:
        d = summarize(AB_WINS, AB_DRAWS, AB_LOSSES).to_dict()
        assert d["total_games"] == 800
        assert d["decisive_games"] == 212
        assert d["ci_low"] < d["score_rate"] < d["ci_high"]
