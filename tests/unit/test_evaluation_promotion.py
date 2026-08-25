"""Tests for config-driven promotion policies.

The load-bearing test in this file is
``TestLegacyPolicyReproducesDeployedBehaviour``: adopting this module must not
change a single promotion decision until an operator opts in. Everything else
is secondary to that guarantee.
"""

from __future__ import annotations

import logging

import pytest

from alpha_zero.evaluation.promotion import (
    DEFAULT_CONFIDENCE,
    DEFAULT_POLICY,
    DEFAULT_PROMOTION_THRESHOLD,
    PromotionConfig,
    PromotionVerdict,
    create_promotion_policy,
    evaluate_promotion,
    list_promotion_policies,
    log_promotion_decision,
    register_promotion_policy,
)
from alpha_zero.evaluation.scoring import PromotionDecision, summarize

# (wins, draws, losses) records spanning draw-free, draw-heavy, and degenerate.
RECORDS = [
    (400, 0, 100),
    (100, 0, 400),
    (142, 588, 70),
    (0, 400, 0),
    (10, 10, 10),
    (1, 0, 0),
    (0, 0, 1),
    (250, 0, 250),
]


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


class TestBackwardsCompatibility:
    def test_default_policy_is_the_legacy_one(self) -> None:
        """Adopting this module must not change behaviour by default."""
        assert DEFAULT_POLICY == "legacy_win_rate"
        assert PromotionConfig().policy == "legacy_win_rate"

    def test_default_threshold_matches_the_deployed_constant(self) -> None:
        assert DEFAULT_PROMOTION_THRESHOLD == 0.55
        assert PromotionConfig().threshold == pytest.approx(0.55)

    @pytest.mark.parametrize(("wins", "draws", "losses"), RECORDS)
    def test_legacy_policy_reproduces_deployed_formula_exactly(
        self, wins: int, draws: int, losses: int
    ) -> None:
        """`wins / max(total, 1) >= threshold` -- arena.py's exact comparison."""
        config = PromotionConfig()
        verdict = evaluate_promotion(wins, draws, losses, config)

        total = wins + draws + losses
        expected_statistic = wins / max(total, 1)
        expected_promoted = expected_statistic >= config.threshold

        assert verdict.statistic == pytest.approx(expected_statistic)
        assert verdict.promoted is expected_promoted

    def test_legacy_policy_still_scores_draws_as_losses(self) -> None:
        """The defect is preserved deliberately, so the default is faithful."""
        verdict = evaluate_promotion(0, 100, 0, PromotionConfig(policy="legacy_win_rate"))
        assert verdict.statistic == 0.0
        assert verdict.promoted is False


class TestRegistry:
    def test_builtin_policies_registered(self) -> None:
        assert list_promotion_policies() == [
            "legacy_win_rate",
            "score_rate",
            "score_rate_lower_bound",
        ]

    def test_unknown_policy_raises_with_valid_options(self) -> None:
        with pytest.raises(ValueError, match="Unknown promotion policy"):
            create_promotion_policy("does_not_exist")

    def test_third_party_policy_can_be_registered(self) -> None:
        """Extension through the seam, without editing this module."""

        class AlwaysPromote:
            name = "test_always_promote"

            def evaluate(self, wins, draws, losses, config):
                return PromotionVerdict(
                    decision=PromotionDecision.PROMOTE,
                    policy=self.name,
                    statistic_name="always",
                    statistic=1.0,
                    threshold=config.threshold,
                    summary=summarize(wins, draws, losses),
                    reason="test policy",
                )

        register_promotion_policy(AlwaysPromote())
        try:
            assert "test_always_promote" in list_promotion_policies()
            verdict = evaluate_promotion(0, 0, 100, PromotionConfig(policy="test_always_promote"))
            assert verdict.promoted is True
        finally:
            from alpha_zero.evaluation import promotion as module

            module._POLICY_REGISTRY.pop("test_always_promote", None)

    def test_duplicate_registration_requires_replace(self) -> None:
        policy = create_promotion_policy("score_rate")
        with pytest.raises(ValueError, match="already registered"):
            register_promotion_policy(policy)
        register_promotion_policy(policy, replace=True)  # explicit replace is fine


class TestConfig:
    def test_env_overrides_every_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6AZ_PROMOTION_POLICY", "score_rate")
        monkeypatch.setenv("CIV6AZ_PROMOTION_THRESHOLD", "0.6")
        monkeypatch.setenv("CIV6AZ_PROMOTION_CONFIDENCE", "0.99")
        monkeypatch.setenv("CIV6AZ_PROMOTION_MIN_GAMES", "42")
        config = PromotionConfig.from_env()
        assert config.policy == "score_rate"
        assert config.threshold == pytest.approx(0.6)
        assert config.confidence == pytest.approx(0.99)
        assert config.min_games == 42

    def test_legacy_env_var_honoured_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """cloud/ predates CIV6AZ_; deployments still set the old name."""
        monkeypatch.setenv("CIV6_PROMOTION_THRESHOLD", "0.7")
        with caplog.at_level(logging.WARNING):
            config = PromotionConfig.from_env()
        assert config.threshold == pytest.approx(0.7)
        assert "CIV6_PROMOTION_THRESHOLD is deprecated" in caplog.text

    def test_modern_env_var_wins_over_legacy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6AZ_PROMOTION_THRESHOLD", "0.6")
        monkeypatch.setenv("CIV6_PROMOTION_THRESHOLD", "0.9")
        assert PromotionConfig.from_env().threshold == pytest.approx(0.6)

    def test_malformed_env_value_falls_back_with_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("CIV6AZ_PROMOTION_THRESHOLD", "not-a-number")
        with caplog.at_level(logging.WARNING):
            config = PromotionConfig.from_env()
        assert config.threshold == pytest.approx(DEFAULT_PROMOTION_THRESHOLD)
        assert "Invalid float" in caplog.text

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"policy": "nope"}, "unknown policy"),
            ({"threshold": 1.5}, "threshold must be in"),
            ({"confidence": 0.0}, "confidence must be in"),
            ({"min_games": -1}, "min_games must be non-negative"),
        ],
    )
    def test_validation_rejects_bad_values(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            PromotionConfig(**kwargs)

    def test_round_trips_through_dict(self) -> None:
        config = PromotionConfig(policy="score_rate", threshold=0.6)
        assert PromotionConfig.from_dict(config.to_dict()) == config

    def test_is_frozen(self) -> None:
        with pytest.raises(Exception):  # FrozenInstanceError subclasses it
            PromotionConfig().threshold = 0.9  # type: ignore[misc]


class TestScoreRatePolicies:
    def test_score_rate_counts_a_draw_as_half(self) -> None:
        verdict = evaluate_promotion(0, 100, 0, PromotionConfig(policy="score_rate"))
        assert verdict.statistic == pytest.approx(0.5)

    def test_lower_bound_policy_is_inconclusive_on_all_draws(self) -> None:
        verdict = evaluate_promotion(
            0, 400, 0, PromotionConfig(policy="score_rate_lower_bound", threshold=0.5)
        )
        assert verdict.decision is PromotionDecision.INCONCLUSIVE
        assert verdict.promoted is False

    def test_lower_bound_policy_never_promotes_an_even_record(self) -> None:
        for n in (10, 100, 1000):
            verdict = evaluate_promotion(
                n, 0, n, PromotionConfig(policy="score_rate_lower_bound", threshold=0.5)
            )
            assert verdict.promoted is False

    def test_lower_bound_policy_returns_all_three_states(self) -> None:
        config = PromotionConfig(policy="score_rate_lower_bound", threshold=0.5)
        assert evaluate_promotion(400, 0, 100, config).decision is PromotionDecision.PROMOTE
        assert evaluate_promotion(100, 0, 400, config).decision is PromotionDecision.REJECT
        assert evaluate_promotion(3, 0, 2, config).decision is PromotionDecision.INCONCLUSIVE

    def test_lower_bound_is_stricter_than_the_point_estimate(self) -> None:
        """6W/4L looks like 0.6 but is far too noisy to act on."""
        point = evaluate_promotion(6, 0, 4, PromotionConfig(policy="score_rate", threshold=0.5))
        bound = evaluate_promotion(
            6, 0, 4, PromotionConfig(policy="score_rate_lower_bound", threshold=0.5)
        )
        assert point.promoted is True
        assert bound.promoted is False

    def test_min_games_floor_is_respected(self) -> None:
        verdict = evaluate_promotion(
            5, 0, 0, PromotionConfig(policy="score_rate_lower_bound", min_games=100)
        )
        assert verdict.decision is PromotionDecision.INCONCLUSIVE
        assert "below min_games" in verdict.reason


class TestVerdict:
    def test_inconclusive_is_not_promoted(self) -> None:
        """Absence of evidence must never advance a champion."""
        verdict = evaluate_promotion(
            1, 0, 1, PromotionConfig(policy="score_rate_lower_bound", min_games=1000)
        )
        assert verdict.decision is PromotionDecision.INCONCLUSIVE
        assert verdict.promoted is False

    def test_to_dict_carries_the_full_audit_trail(self) -> None:
        payload = evaluate_promotion(142, 588, 70).to_dict()
        for key in (
            "decision",
            "promoted",
            "policy",
            "statistic_name",
            "statistic",
            "threshold",
            "reason",
            "wins",
            "draws",
            "losses",
            "total_games",
            "draw_rate",
            "decisive_games",
            "score_rate",
        ):
            assert key in payload, f"missing {key}"

    def test_to_dict_ci_mirrors_the_verdict_fields_per_policy(self) -> None:
        """The serialized ci fields mean *what the verdict carries*, per policy.

        Before the merge order was fixed, the summary's always-computed
        diagnostic interval overwrote the verdict fields, so the legacy
        policy's deliberate ``None`` serialized as if an interval had been
        consulted. The legacy verdict carries no interval; the score-rate
        policies carry theirs, and the payload must agree with the verdict
        either way.
        """
        legacy = evaluate_promotion(142, 588, 70, PromotionConfig(policy="legacy_win_rate"))
        assert legacy.ci_low is None and legacy.ci_high is None
        payload = legacy.to_dict()
        assert payload["ci_low"] is None
        assert payload["ci_high"] is None

        for policy in ("score_rate", "score_rate_lower_bound"):
            verdict = evaluate_promotion(142, 588, 70, PromotionConfig(policy=policy))
            payload = verdict.to_dict()
            assert payload["ci_low"] == verdict.ci_low is not None, policy
            assert payload["ci_high"] == verdict.ci_high is not None, policy

    def test_carries_the_confidence_interval_for_interval_policies(self) -> None:
        verdict = evaluate_promotion(142, 588, 70, PromotionConfig(policy="score_rate_lower_bound"))
        assert verdict.ci_low is not None and verdict.ci_high is not None
        assert verdict.ci_low < verdict.statistic < verdict.ci_high

    def test_confidence_flows_from_config_into_the_interval(self) -> None:
        wide = evaluate_promotion(
            142, 588, 70, PromotionConfig(policy="score_rate_lower_bound", confidence=0.99)
        )
        narrow = evaluate_promotion(
            142, 588, 70, PromotionConfig(policy="score_rate_lower_bound", confidence=0.90)
        )
        assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)
        assert DEFAULT_CONFIDENCE == 0.95


class TestLogging:
    def test_decision_is_logged_at_info_with_the_key_numbers(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        verdict = evaluate_promotion(142, 588, 70)
        with caplog.at_level(logging.INFO):
            log_promotion_decision(verdict)
        assert "Promotion" in caplog.text
        assert "legacy_win_rate" in caplog.text
        assert "800 games" in caplog.text
        assert "142W/588D/70L" in caplog.text

    def test_full_record_is_available_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        verdict = evaluate_promotion(142, 588, 70)
        with caplog.at_level(logging.DEBUG):
            log_promotion_decision(verdict, context={"iteration": 7})
        assert "Promotion decision record" in caplog.text
        assert "iteration" in caplog.text

    def test_legacy_policy_warns_when_promotion_is_impossible(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The condition that previously passed unnoticed for 50 iterations."""
        with caplog.at_level(logging.WARNING):
            evaluate_promotion(142, 588, 70, PromotionConfig(policy="legacy_win_rate"))
        assert "arithmetically impossible" in caplog.text
        assert "CIV6AZ_PROMOTION_POLICY=score_rate" in caplog.text

    def test_no_warning_when_promotion_is_reachable(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            evaluate_promotion(400, 0, 100, PromotionConfig(policy="legacy_win_rate"))
        assert "arithmetically impossible" not in caplog.text
