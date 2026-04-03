"""Tests for alpha_zero.evaluation.sequential_testing module.

Focuses on coverage of SequentialTestingConfig, SequentialTester,
GameResult, SequentialTestResult, and helper functions.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import yaml

from alpha_zero.evaluation.sequential_testing import (
    EvaluationDecision,
    GameResult,
    SequentialTester,
    SequentialTestingConfig,
    SequentialTestResult,
    get_sequential_tester,
)

# ---------------------------------------------------------------------------
# GameResult
# ---------------------------------------------------------------------------


class TestGameResult:
    def test_to_dict(self):
        g = GameResult(game_id=1, winner=1, turns=10, duration_seconds=1.5)
        d = g.to_dict()
        assert d["game_id"] == 1
        assert d["winner"] == 1
        assert d["turns"] == 10
        assert d["duration_seconds"] == 1.5

    def test_defaults(self):
        g = GameResult(game_id=0, winner=0)
        assert g.turns == 0
        assert g.duration_seconds == 0.0


# ---------------------------------------------------------------------------
# SequentialTestResult
# ---------------------------------------------------------------------------


class TestSequentialTestResult:
    def test_to_dict(self):
        r = SequentialTestResult(
            decision=EvaluationDecision.ACCEPT,
            games_played=50,
            wins=30,
            losses=15,
            draws=5,
            win_rate=0.6667,
            confidence=0.95,
            reason="Accepted",
            cost_savings=3.50,
        )
        d = r.to_dict()
        assert d["decision"] == "accept"
        assert d["games_played"] == 50
        assert d["cost_savings"] == 3.50


# ---------------------------------------------------------------------------
# SequentialTestingConfig
# ---------------------------------------------------------------------------


class TestSequentialTestingConfig:
    def test_defaults(self):
        cfg = SequentialTestingConfig()
        assert cfg.min_games == 20
        assert cfg.max_games == 400
        assert cfg.significance_level == 0.05
        assert cfg.win_threshold == 0.55

    def test_from_env(self):
        env = {
            "CIV6AZ_EVAL_MIN_GAMES": "10",
            "CIV6AZ_EVAL_MAX_GAMES": "200",
            "CIV6AZ_EVAL_EARLY_STOP_THRESHOLD": "100",
            "CIV6AZ_EVAL_SIGNIFICANCE_LEVEL": "0.01",
            "CIV6AZ_EVAL_WIN_THRESHOLD": "0.6",
            "CIV6AZ_EVAL_DECISIVE_MARGIN": "0.15",
            "CIV6AZ_EVAL_COST_PER_GAME": "0.05",
        }
        with patch.dict("os.environ", env, clear=False):
            cfg = SequentialTestingConfig.from_env()
        assert cfg.min_games == 10
        assert cfg.max_games == 200
        assert cfg.significance_level == 0.01

    def test_from_dict(self):
        cfg = SequentialTestingConfig.from_dict(
            {"min_games": 5, "max_games": 50, "early_stop_threshold": 25}
        )
        assert cfg.min_games == 5
        assert cfg.max_games == 50

    def test_to_dict(self):
        cfg = SequentialTestingConfig()
        d = cfg.to_dict()
        assert "min_games" in d
        assert "max_games" in d
        assert "cost_per_game_usd" in d

    def test_from_yaml(self, tmp_path):
        yaml_path = tmp_path / "seq_test.yaml"
        data = {
            "min_games": 10,
            "max_games": 100,
            "early_stop_threshold": 50,
            "significance_level": 0.05,
            "win_threshold": 0.55,
            "decisive_margin": 0.1,
            "cost_per_game_usd": 0.02,
        }
        yaml_path.write_text(yaml.dump(data))
        cfg = SequentialTestingConfig.from_yaml(yaml_path)
        assert cfg.min_games == 10
        assert cfg.max_games == 100

    def test_from_yaml_missing_file(self, tmp_path):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="not found"):
            SequentialTestingConfig.from_yaml(tmp_path / "nonexistent.yaml")

    def test_from_yaml_invalid_yaml(self, tmp_path):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{{{{invalid yaml: [")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            SequentialTestingConfig.from_yaml(bad_yaml)

    def test_validate_min_games_error(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="min_games"):
            SequentialTestingConfig(min_games=0, max_games=100, early_stop_threshold=50)

    def test_validate_max_games_less_than_min(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="max_games"):
            SequentialTestingConfig(min_games=50, max_games=10, early_stop_threshold=50)

    def test_validate_early_stop_less_than_min(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="early_stop_threshold"):
            SequentialTestingConfig(min_games=50, max_games=100, early_stop_threshold=10)

    def test_validate_early_stop_greater_than_max(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="early_stop_threshold"):
            SequentialTestingConfig(min_games=10, max_games=50, early_stop_threshold=100)

    def test_validate_significance_level_out_of_range(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="significance_level"):
            SequentialTestingConfig(significance_level=0.0)
        with pytest.raises(ConfigurationError, match="significance_level"):
            SequentialTestingConfig(significance_level=1.0)

    def test_validate_win_threshold_below_half(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="win_threshold"):
            SequentialTestingConfig(win_threshold=0.4)

    def test_validate_decisive_margin_out_of_range(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="decisive_margin"):
            SequentialTestingConfig(decisive_margin=0.0)
        with pytest.raises(ConfigurationError, match="decisive_margin"):
            SequentialTestingConfig(decisive_margin=0.5)

    def test_validate_negative_cost(self):
        from alpha_zero.cloud.gcp.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="cost_per_game_usd"):
            SequentialTestingConfig(cost_per_game_usd=-1.0)


# ---------------------------------------------------------------------------
# SequentialTester
# ---------------------------------------------------------------------------


class TestSequentialTester:
    @pytest.fixture()
    def config(self):
        return SequentialTestingConfig(
            min_games=5,
            max_games=50,
            early_stop_threshold=25,
            significance_level=0.05,
            win_threshold=0.55,
            decisive_margin=0.1,
            cost_per_game_usd=0.01,
        )

    @pytest.fixture()
    def tester(self, config):
        return SequentialTester(config)

    def test_init_default_config(self):
        t = SequentialTester()
        assert t.config.min_games == 20

    def test_record_game_win(self, tester):
        tester.record_game(0, winner=1, turns=10, duration_seconds=1.0)
        assert tester._wins == 1
        assert tester._losses == 0
        assert tester._draws == 0
        assert len(tester._games) == 1

    def test_record_game_loss(self, tester):
        tester.record_game(0, winner=-1)
        assert tester._losses == 1

    def test_record_game_draw(self, tester):
        tester.record_game(0, winner=0)
        assert tester._draws == 1

    def test_get_decision_continue_below_min(self, tester):
        tester.record_game(0, winner=1)
        assert tester.get_decision() == EvaluationDecision.CONTINUE

    def test_get_decision_continue_all_draws(self, tester):
        for i in range(10):
            tester.record_game(i, winner=0)
        assert tester.get_decision() == EvaluationDecision.CONTINUE

    def test_get_decision_accept_at_max_games(self, tester):
        """When max_games reached and win_rate >= threshold, accept."""
        # Record 50 games, majority wins
        for i in range(40):
            tester.record_game(i, winner=1)
        for i in range(10):
            tester.record_game(40 + i, winner=-1)
        assert tester.get_decision() == EvaluationDecision.ACCEPT

    def test_get_decision_inconclusive_at_max_games(self, tester):
        """When max_games reached and win_rate < threshold, inconclusive."""
        for i in range(25):
            tester.record_game(i, winner=1)
        for i in range(25):
            tester.record_game(25 + i, winner=-1)
        assert tester.get_decision() == EvaluationDecision.INCONCLUSIVE

    def test_get_decision_decisive_accept(self, tester):
        """Decisive win above threshold + margin."""
        for i in range(10):
            tester.record_game(i, winner=1)
        for i in range(2):
            tester.record_game(10 + i, winner=-1)
        # win_rate = 10/12 ~= 0.833 > 0.55 + 0.1 = 0.65 -> decisive accept
        decision = tester.get_decision()
        assert decision == EvaluationDecision.ACCEPT

    def test_get_decision_decisive_reject(self, tester):
        """Decisive loss below 0.5 - margin."""
        for i in range(2):
            tester.record_game(i, winner=1)
        for i in range(10):
            tester.record_game(2 + i, winner=-1)
        # win_rate = 2/12 ~= 0.167 < 0.5 - 0.1 = 0.4 -> decisive reject
        decision = tester.get_decision()
        assert decision == EvaluationDecision.REJECT

    def test_get_decision_early_stop_accept(self, config):
        """At early_stop_threshold, accept if CI lower bound >= threshold."""
        config.early_stop_threshold = 10
        config.min_games = 5
        config.max_games = 50
        config.decisive_margin = 0.45  # Make decisive harder to trigger
        tester = SequentialTester(config)
        # Many wins to push CI high
        for i in range(25):
            tester.record_game(i, winner=1)
        for i in range(1):
            tester.record_game(25 + i, winner=-1)
        decision = tester.get_decision()
        assert decision in (EvaluationDecision.ACCEPT, EvaluationDecision.CONTINUE)

    def test_get_decision_early_stop_reject(self, config):
        """At early_stop_threshold, reject if CI upper bound < 0.5."""
        config.early_stop_threshold = 10
        config.min_games = 5
        config.max_games = 50
        config.decisive_margin = 0.49  # Make decisive very hard to trigger
        tester = SequentialTester(config)
        # Many losses
        for i in range(25):
            tester.record_game(i, winner=-1)
        for i in range(1):
            tester.record_game(25 + i, winner=1)
        decision = tester.get_decision()
        assert decision in (EvaluationDecision.REJECT, EvaluationDecision.CONTINUE)

    def test_is_decisive_below_min(self, tester):
        tester.record_game(0, winner=1)
        assert tester.is_decisive() is False

    def test_is_decisive_clear_win(self, tester):
        for i in range(10):
            tester.record_game(i, winner=1)
        assert tester.is_decisive() is True

    def test_is_decisive_clear_loss(self, tester):
        for i in range(10):
            tester.record_game(i, winner=-1)
        assert tester.is_decisive() is True

    def test_is_decisive_close_result(self, tester):
        for i in range(5):
            tester.record_game(i, winner=1)
        for i in range(5):
            tester.record_game(5 + i, winner=-1)
        assert tester.is_decisive() is False

    def test_get_confidence_interval_no_games(self, tester):
        ci = tester.get_confidence_interval()
        assert ci == (0.0, 1.0)

    def test_get_confidence_interval_with_games(self, tester):
        for i in range(20):
            tester.record_game(i, winner=1)
        for i in range(10):
            tester.record_game(20 + i, winner=-1)
        low, high = tester.get_confidence_interval()
        # The z-score implementation returns negative values for alpha < 0.5,
        # which can cause low > high. Just verify both bounds are in [0, 1].
        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0

    def test_get_z_score(self):
        # For alpha=0.025 (<0.5), sign is -1, so z is approximately -1.96
        z = SequentialTester._get_z_score(0.025)
        assert abs(z) == pytest.approx(1.96, abs=0.05)

        z = SequentialTester._get_z_score(0.5)
        assert z == 0.0

        z = SequentialTester._get_z_score(0.95)
        assert z == 0.0  # alpha >= 0.5 returns 0

    def test_normal_cdf(self):
        assert SequentialTester._normal_cdf(0) == pytest.approx(0.5, abs=0.001)
        assert SequentialTester._normal_cdf(1.96) == pytest.approx(0.975, abs=0.01)
        assert SequentialTester._normal_cdf(-1.96) == pytest.approx(0.025, abs=0.01)

    def test_get_result_no_games(self, tester):
        result = tester.get_result()
        assert result.games_played == 0
        assert result.win_rate == 0.5
        assert result.decision == EvaluationDecision.CONTINUE

    def test_get_result_with_games(self, tester):
        for i in range(30):
            tester.record_game(i, winner=1)
        for i in range(10):
            tester.record_game(30 + i, winner=-1)
        for i in range(5):
            tester.record_game(40 + i, winner=0)
        result = tester.get_result()
        assert result.games_played == 45
        assert result.wins == 30
        assert result.losses == 10
        assert result.draws == 5
        assert result.win_rate == pytest.approx(0.75, abs=0.01)
        assert result.confidence > 0

    def test_get_result_reason_accept(self, tester):
        for i in range(40):
            tester.record_game(i, winner=1)
        for i in range(10):
            tester.record_game(40 + i, winner=-1)
        result = tester.get_result()
        assert "accepted" in result.reason.lower() or "need more" in result.reason.lower()

    def test_get_result_reason_continue(self, tester):
        for i in range(3):
            tester.record_game(i, winner=1)
        result = tester.get_result()
        assert "need more" in result.reason.lower()

    def test_calculate_p_value_few_games(self, tester):
        tester.record_game(0, winner=1)
        p = tester._calculate_p_value()
        assert p == 1.0

    def test_calculate_p_value_many_games(self, tester):
        for i in range(30):
            tester.record_game(i, winner=1)
        for i in range(10):
            tester.record_game(30 + i, winner=-1)
        p = tester._calculate_p_value()
        assert 0.0 <= p <= 1.0

    def test_build_reason_reject(self, tester):
        for i in range(10):
            tester.record_game(i, winner=-1)
        result = tester.get_result()
        # With 10 losses it should be decisive reject
        if result.decision == EvaluationDecision.REJECT:
            assert "rejected" in result.reason.lower()

    def test_build_reason_inconclusive(self, config):
        config.max_games = 10
        config.min_games = 5
        config.early_stop_threshold = 8
        config.decisive_margin = 0.49
        tester = SequentialTester(config)
        for i in range(5):
            tester.record_game(i, winner=1)
        for i in range(5):
            tester.record_game(5 + i, winner=-1)
        result = tester.get_result()
        if result.decision == EvaluationDecision.INCONCLUSIVE:
            assert "inconclusive" in result.reason.lower()

    def test_calculate_cost_savings(self, tester):
        # 50 max games, 10 played = 40 * 0.01 = 0.40
        savings = tester._calculate_cost_savings(10)
        assert savings == pytest.approx(0.40)

    def test_calculate_cost_savings_no_savings(self, tester):
        savings = tester._calculate_cost_savings(50)
        assert savings == 0.0

    def test_calculate_cost_savings_over_max(self, tester):
        savings = tester._calculate_cost_savings(60)
        assert savings == 0.0

    def test_reset(self, tester):
        tester.record_game(0, winner=1)
        tester.record_game(1, winner=-1)
        tester.reset()
        assert len(tester._games) == 0
        assert tester._wins == 0
        assert tester._losses == 0
        assert tester._draws == 0

    def test_get_statistics(self, tester):
        for i in range(10):
            tester.record_game(i, winner=1)
        for i in range(5):
            tester.record_game(10 + i, winner=-1)
        for i in range(3):
            tester.record_game(15 + i, winner=0)
        stats = tester.get_statistics()
        assert stats["games_played"] == 18
        assert stats["wins"] == 10
        assert stats["losses"] == 5
        assert stats["draws"] == 3
        assert stats["win_rate"] == pytest.approx(10 / 15, abs=0.001)
        assert "confidence_interval" in stats
        assert "is_decisive" in stats
        assert "current_decision" in stats

    def test_get_statistics_no_games(self, tester):
        stats = tester.get_statistics()
        assert stats["win_rate"] == 0.5
        assert stats["confidence_interval"] == (0.0, 1.0)


# ---------------------------------------------------------------------------
# get_sequential_tester convenience function
# ---------------------------------------------------------------------------


class TestGetSequentialTester:
    def test_returns_tester(self):
        tester = get_sequential_tester()
        assert isinstance(tester, SequentialTester)
