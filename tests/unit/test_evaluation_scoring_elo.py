"""Elo transforms, the legacy ``win_rate`` helper, and promotion feasibility.

Numbers quoted in ``docs/runbooks/promotion-gate.md`` are pinned here so the
runbook and the code cannot drift: a score rate of 0.52 is about +13.9 Elo,
0.55 about +34.9, and 20 games at a 70% draw rate cannot see less than roughly
+100 Elo at 80% power.

Negative controls performed when this file was written (each reverted):

* ``elo_from_score_rate`` with the sign dropped (``400 * log10``) ->
  ``test_known_values`` and ``test_symmetry`` fail;
* ``feasibility`` using ``0.5 - mde`` -> ``test_feasibility_small_budget_is_infeasible``
  fails (implied Elo goes negative);
* removing the WARNING in ``_log_feasibility`` -> ``test_warns_when_under_powered`` fails.
"""

from __future__ import annotations

import itertools
import logging
import math

import pytest

from alpha_zero.evaluation.promotion import (
    DEFAULT_MAX_IMPLIED_ELO,
    FeasibilityReport,
    PromotionConfig,
    evaluate_promotion,
)
from alpha_zero.evaluation.scoring import (
    elo_from_score_rate,
    elo_interval,
    games_for_score_rate,
    score_interval,
    score_rate_from_elo,
    win_rate,
)


@pytest.fixture(autouse=True)
def _clear_promotion_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "CIV6AZ_PROMOTION_POLICY",
        "CIV6AZ_PROMOTION_THRESHOLD",
        "CIV6AZ_PROMOTION_CONFIDENCE",
        "CIV6AZ_PROMOTION_MIN_GAMES",
        "CIV6AZ_PROMOTION_MAX_IMPLIED_ELO",
        "CIV6_PROMOTION_THRESHOLD",
    ):
        monkeypatch.delenv(var, raising=False)


class TestEloFromScoreRate:
    def test_known_values(self) -> None:
        assert elo_from_score_rate(0.5) == pytest.approx(0.0, abs=1e-9)
        assert elo_from_score_rate(0.52) == pytest.approx(13.9, abs=0.05)
        assert elo_from_score_rate(0.55) == pytest.approx(34.9, abs=0.05)
        assert elo_from_score_rate(0.75) == pytest.approx(190.8, abs=0.1)

    def test_symmetry(self) -> None:
        for p in (0.51, 0.6, 0.75, 0.9):
            assert elo_from_score_rate(p) == pytest.approx(-elo_from_score_rate(1.0 - p))

    def test_monotone(self) -> None:
        grid = [i / 100 for i in range(1, 100)]
        values = [elo_from_score_rate(p) for p in grid]
        assert all(a < b for a, b in itertools.pairwise(values))

    def test_endpoints_are_finite_and_extreme(self) -> None:
        assert math.isfinite(elo_from_score_rate(0.0))
        assert math.isfinite(elo_from_score_rate(1.0))
        assert elo_from_score_rate(1.0) > 3000
        assert elo_from_score_rate(0.0) < -3000

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            elo_from_score_rate(1.2)

    def test_inverse(self) -> None:
        for elo in (-300.0, -14.0, 0.0, 13.9, 100.0):
            assert elo_from_score_rate(score_rate_from_elo(elo)) == pytest.approx(elo, abs=1e-6)


class TestEloInterval:
    def test_symmetric_record_gives_symmetric_interval(self) -> None:
        low, high = elo_interval(10, 20, 10)
        assert low == pytest.approx(-high, abs=1e-9)
        assert low < 0 < high

    def test_shrinks_with_n(self) -> None:
        small = elo_interval(12, 10, 8)
        large = elo_interval(120, 100, 80)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_is_the_monotone_image_of_the_score_interval(self) -> None:
        lo, hi = score_interval(142, 588, 70)
        assert elo_interval(142, 588, 70) == (elo_from_score_rate(lo), elo_from_score_rate(hi))

    def test_the_recorded_ab_arm(self) -> None:
        """800 games, 73.5% draws, +31 Elo arm: the interval contains +31 and excludes 0."""
        low, high = elo_interval(142, 588, 70)
        assert low > 0
        assert low < 31 < high


class TestLegacyWinRate:
    def test_draws_count_as_losses(self) -> None:
        assert win_rate(3, 4, 3) == pytest.approx(0.3)

    def test_empty_record_is_zero_like_every_legacy_site(self) -> None:
        assert win_rate(0, 0, 0) == 0.0

    def test_rejects_negative_counts(self) -> None:
        with pytest.raises(ValueError):
            win_rate(-1, 0, 0)


class TestGamesForScoreRate:
    def test_inverts_the_mde(self) -> None:
        from alpha_zero.evaluation.scoring import minimum_detectable_effect

        n = games_for_score_rate(0.52, 0.735)
        # At n games the MDE is at most 0.02; at n - 1 it is more.
        assert minimum_detectable_effect(n, 0.735) <= 0.02 + 1e-12
        assert minimum_detectable_effect(n - 1, 0.735) > 0.02

    def test_fewer_draws_need_more_games(self) -> None:
        assert games_for_score_rate(0.52, 0.3) > games_for_score_rate(0.52, 0.735)

    def test_rejects_parity_or_worse(self) -> None:
        with pytest.raises(ValueError):
            games_for_score_rate(0.5, 0.5)


class TestFeasibility:
    def test_small_budget_is_infeasible(self) -> None:
        report = PromotionConfig().feasibility(20, 0.7)
        assert isinstance(report, FeasibilityReport)
        assert report.implied_elo > 90
        assert not report.feasible
        assert report.min_observed_score == pytest.approx(0.55)  # point-estimate policy

    def test_large_budget_is_feasible(self) -> None:
        report = PromotionConfig().feasibility(1500, 0.7)
        assert report.implied_elo < 20
        assert report.feasible

    def test_interval_policy_needs_more_than_the_threshold(self) -> None:
        point = PromotionConfig(policy="score_rate", threshold=0.52).feasibility(200, 0.7)
        interval = PromotionConfig(policy="score_rate_lower_bound", threshold=0.52).feasibility(
            200, 0.7
        )
        assert point.min_observed_score == pytest.approx(0.52)
        assert interval.min_observed_score > 0.52

    def test_games_for_target_elo_matches_the_scoring_helper(self) -> None:
        report = PromotionConfig().feasibility(20, 0.735, target_elo=14.0)
        assert report.games_for_target_elo == games_for_score_rate(score_rate_from_elo(14.0), 0.735)

    def test_one_game_does_not_raise(self) -> None:
        report = PromotionConfig().feasibility(1, 0.0)
        assert math.isfinite(report.implied_elo)
        assert not report.feasible

    def test_rejects_non_positive_n(self) -> None:
        with pytest.raises(ValueError):
            PromotionConfig().feasibility(0, 0.5)

    def test_to_dict_and_describe(self) -> None:
        report = PromotionConfig().feasibility(400, 0.5)
        payload = report.to_dict()
        assert payload["n_games"] == 400
        assert payload["feasible"] is report.feasible
        assert "MDE" in report.describe()


class TestMaxImpliedEloConfig:
    def test_default(self) -> None:
        assert PromotionConfig().max_implied_elo == DEFAULT_MAX_IMPLIED_ELO

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6AZ_PROMOTION_MAX_IMPLIED_ELO", "120")
        assert PromotionConfig().max_implied_elo == 120.0
        assert PromotionConfig().feasibility(20, 0.7).feasible

    def test_round_trips_through_to_dict(self) -> None:
        config = PromotionConfig(max_implied_elo=33.0)
        assert PromotionConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["max_implied_elo"] == 33.0

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="max_implied_elo"):
            PromotionConfig(max_implied_elo=0.0)


class TestFeasibilityLogging:
    def test_logs_feasibility_at_info_when_the_policy_resolves(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="alpha_zero.evaluation.promotion"):
            evaluate_promotion(300, 700, 500)
        assert any("promotion feasibility" in r.getMessage() for r in caplog.records)

    def test_warns_when_under_powered(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="alpha_zero.evaluation.promotion"):
            evaluate_promotion(3, 14, 3)  # 20 games, 70% draws -> ~+109 Elo needed
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("under-powered" in r.getMessage() for r in warnings)

    def test_does_not_warn_when_powered(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="alpha_zero.evaluation.promotion"):
            evaluate_promotion(300, 1050, 150)  # 1500 games, 70% draws -> ~+12 Elo
        assert not any("under-powered" in r.getMessage() for r in caplog.records)

    def test_empty_record_logs_nothing_and_does_not_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="alpha_zero.evaluation.promotion"):
            evaluate_promotion(0, 0, 0)
        assert not any("promotion feasibility" in r.getMessage() for r in caplog.records)
