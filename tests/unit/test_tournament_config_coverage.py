"""Comprehensive tests for alpha_zero.tournament.config module.

Covers uncovered lines in:
- ELOConfig (from_dict, get_k_factor decay, to_dict)
- TierConfig (contains_rating, from_dict)
- MatchmakingConfig (get_rating_range, from_dict, to_dict)
- TournamentConfig (from_yaml, from_env, _convert_value, _from_dict, save,
  get_tier_name, get_tier_by_name, to_dict)
- PlayerStats (record_game, win_rate, loss_rate, from_dict, save, load)
- MatchResult (is_draw, get_outcome_for_player, from_dict)
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import yaml

from alpha_zero.tournament.config import (
    ELOConfig,
    MatchmakingConfig,
    MatchOutcome,
    MatchResult,
    PlayerStats,
    TierConfig,
    TournamentConfig,
    default_tiers,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def elo_config():
    return ELOConfig()


@pytest.fixture()
def tier_bronze():
    return TierConfig(name="Bronze", rating_min=0, rating_max=1200)


@pytest.fixture()
def tier_silver():
    return TierConfig(name="Silver", rating_min=1200, rating_max=1500)


@pytest.fixture()
def tournament_config():
    return TournamentConfig()


@pytest.fixture()
def sample_player():
    return PlayerStats(player_id="test_player")


# ---------------------------------------------------------------------------
# ELOConfig
# ---------------------------------------------------------------------------


class TestELOConfig:
    def test_get_k_factor_provisional(self, elo_config):
        k = elo_config.get_k_factor(games_played=5, is_provisional=True)
        assert k == elo_config.k_factor_base

    def test_get_k_factor_below_threshold(self, elo_config):
        k = elo_config.get_k_factor(games_played=10, is_provisional=False)
        assert k == elo_config.k_factor_base  # still under provisional_threshold=30

    def test_get_k_factor_at_threshold(self, elo_config):
        k = elo_config.get_k_factor(games_played=30, is_provisional=False)
        assert k == elo_config.k_factor_base  # decay=0 at exactly threshold

    def test_get_k_factor_well_past_threshold(self, elo_config):
        k = elo_config.get_k_factor(games_played=130, is_provisional=False)
        assert k == elo_config.k_factor_min

    def test_get_k_factor_partial_decay(self, elo_config):
        # 80 games: (80 - 30)/100 = 0.5 decay
        k = elo_config.get_k_factor(games_played=80, is_provisional=False)
        expected = 32.0 - 0.5 * (32.0 - 10.0)
        assert abs(k - expected) < 1e-9

    def test_to_dict_roundtrip(self, elo_config):
        d = elo_config.to_dict()
        restored = ELOConfig.from_dict(d)
        assert restored.initial_rating == elo_config.initial_rating
        assert restored.k_factor_base == elo_config.k_factor_base

    def test_from_dict_ignores_extra_keys(self):
        data = {"initial_rating": 1200.0, "nonexistent_field": 42}
        config = ELOConfig.from_dict(data)
        assert config.initial_rating == 1200.0


# ---------------------------------------------------------------------------
# TierConfig
# ---------------------------------------------------------------------------


class TestTierConfig:
    def test_contains_rating_inside(self, tier_bronze):
        assert tier_bronze.contains_rating(600) is True

    def test_contains_rating_at_min(self, tier_bronze):
        assert tier_bronze.contains_rating(0) is True

    def test_contains_rating_at_max_excluded(self, tier_bronze):
        assert tier_bronze.contains_rating(1200) is False

    def test_contains_rating_outside(self, tier_bronze):
        assert tier_bronze.contains_rating(1500) is False

    def test_to_dict(self, tier_bronze):
        d = tier_bronze.to_dict()
        assert d["name"] == "Bronze"
        assert d["rating_min"] == 0
        assert d["rating_max"] == 1200

    def test_from_dict_roundtrip(self, tier_bronze):
        d = tier_bronze.to_dict()
        restored = TierConfig.from_dict(d)
        assert restored.name == tier_bronze.name
        assert restored.rating_min == tier_bronze.rating_min

    def test_from_dict_ignores_extra(self):
        data = {"name": "X", "rating_min": 0, "rating_max": 100, "extra": True}
        t = TierConfig.from_dict(data)
        assert t.name == "X"


# ---------------------------------------------------------------------------
# MatchmakingConfig
# ---------------------------------------------------------------------------


class TestMatchmakingConfig:
    def test_get_rating_range_zero_wait(self):
        mc = MatchmakingConfig()
        assert mc.get_rating_range(0.0) == mc.rating_range_initial

    def test_get_rating_range_expands(self):
        mc = MatchmakingConfig(rating_range_initial=200, rating_range_expand_rate=10)
        assert mc.get_rating_range(5.0) == 250.0

    def test_get_rating_range_capped(self):
        mc = MatchmakingConfig(rating_range_initial=200, rating_range_max=500)
        assert mc.get_rating_range(9999.0) == 500.0

    def test_to_dict_roundtrip(self):
        mc = MatchmakingConfig(rating_range_initial=150)
        d = mc.to_dict()
        restored = MatchmakingConfig.from_dict(d)
        assert restored.rating_range_initial == 150

    def test_from_dict_ignores_extra(self):
        data = {"rating_range_initial": 100, "bogus": True}
        mc = MatchmakingConfig.from_dict(data)
        assert mc.rating_range_initial == 100


# ---------------------------------------------------------------------------
# default_tiers helper
# ---------------------------------------------------------------------------


class TestDefaultTiers:
    def test_returns_six_tiers(self):
        tiers = default_tiers()
        assert len(tiers) == 6

    def test_tier_names(self):
        names = [t.name for t in default_tiers()]
        assert names == ["Bronze", "Silver", "Gold", "Platinum", "Diamond", "Master"]


# ---------------------------------------------------------------------------
# TournamentConfig
# ---------------------------------------------------------------------------


class TestTournamentConfig:
    def test_get_tier_for_rating_found(self, tournament_config):
        tier = tournament_config.get_tier_for_rating(1600)
        assert tier is not None
        assert tier.name == "Gold"

    def test_get_tier_for_rating_none(self):
        config = TournamentConfig(tiers=[TierConfig("Only", 1000, 2000)])
        assert config.get_tier_for_rating(500) is None

    def test_get_tier_name(self, tournament_config):
        assert tournament_config.get_tier_name(1600) == "Gold"

    def test_get_tier_name_unknown(self):
        config = TournamentConfig(tiers=[TierConfig("Only", 1000, 2000)])
        assert config.get_tier_name(500) == "Unknown"

    def test_get_tier_by_name_found(self, tournament_config):
        tier = tournament_config.get_tier_by_name("Silver")
        assert tier is not None
        assert tier.name == "Silver"

    def test_get_tier_by_name_not_found(self, tournament_config):
        assert tournament_config.get_tier_by_name("Mythic") is None

    def test_from_yaml(self, tmp_path):
        data = {
            "elo": {"initial_rating": 1200},
            "matchmaking": {"rating_range_initial": 100},
            "tiers": [{"name": "Low", "rating_min": 0, "rating_max": 2000}],
            "games_per_match": 3,
        }
        path = tmp_path / "config.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)

        config = TournamentConfig.from_yaml(path)
        assert config.elo.initial_rating == 1200
        assert config.matchmaking.rating_range_initial == 100
        assert len(config.tiers) == 1
        assert config.games_per_match == 3

    def test_from_yaml_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TournamentConfig.from_yaml(tmp_path / "missing.yaml")

    def test_from_yaml_empty_file(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        config = TournamentConfig.from_yaml(path)
        # Should use defaults
        assert config.elo.initial_rating == 1500.0

    def test_from_env_elo_override(self, tournament_config):
        with patch.dict(os.environ, {"CIV6AZ_TOURNAMENT_ELO_INITIAL_RATING": "1200"}):
            config = TournamentConfig.from_env(tournament_config)
            assert config.elo.initial_rating == 1200.0

    def test_from_env_matchmaking_override(self, tournament_config):
        with patch.dict(os.environ, {"CIV6AZ_TOURNAMENT_MATCHMAKING_RATING_RANGE_INITIAL": "300"}):
            config = TournamentConfig.from_env(tournament_config)
            assert config.matchmaking.rating_range_initial == 300.0

    def test_from_env_top_level_override(self, tournament_config):
        with patch.dict(os.environ, {"CIV6AZ_TOURNAMENT_GAMES_PER_MATCH": "5"}):
            config = TournamentConfig.from_env(tournament_config)
            assert config.games_per_match == 5

    def test_from_env_bool_conversion(self, tournament_config):
        with patch.dict(os.environ, {"CIV6AZ_TOURNAMENT_LOG_MATCHES": "false"}):
            config = TournamentConfig.from_env(tournament_config)
            assert config.log_matches is False

    def test_from_env_string_conversion(self, tournament_config):
        with patch.dict(os.environ, {"CIV6AZ_TOURNAMENT_LOG_DIR": "/tmp/logs"}):
            config = TournamentConfig.from_env(tournament_config)
            assert config.log_dir == "/tmp/logs"

    def test_from_env_no_base(self):
        with patch.dict(os.environ, {}, clear=False):
            config = TournamentConfig.from_env()
            assert config.elo.initial_rating == 1500.0

    def test_convert_value_bool_true(self):
        assert TournamentConfig._convert_value("true", False) is True
        assert TournamentConfig._convert_value("1", False) is True
        assert TournamentConfig._convert_value("yes", False) is True

    def test_convert_value_bool_false(self):
        assert TournamentConfig._convert_value("no", True) is False

    def test_convert_value_int(self):
        assert TournamentConfig._convert_value("42", 0) == 42

    def test_convert_value_float(self):
        assert TournamentConfig._convert_value("3.14", 0.0) == pytest.approx(3.14)

    def test_convert_value_string(self):
        assert TournamentConfig._convert_value("hello", "default") == "hello"

    def test_to_dict(self, tournament_config):
        d = tournament_config.to_dict()
        assert "elo" in d
        assert "matchmaking" in d
        assert "tiers" in d
        assert d["games_per_match"] == 2

    def test_save(self, tmp_path, tournament_config):
        path = tmp_path / "sub" / "saved.yaml"
        tournament_config.save(path)
        assert path.exists()
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["games_per_match"] == 2

    def test_from_dict_with_remaining_fields(self):
        data = {
            "elo": {"initial_rating": 1300},
            "matchmaking": {},
            "games_per_match": 4,
            "time_control_seconds": 600.0,
            "ENV_PREFIX": "IGNORED",  # Should not overwrite
        }
        config = TournamentConfig._from_dict(data)
        assert config.elo.initial_rating == 1300
        assert config.games_per_match == 4
        assert config.time_control_seconds == 600.0


# ---------------------------------------------------------------------------
# PlayerStats
# ---------------------------------------------------------------------------


class TestPlayerStats:
    def test_post_init_display_name(self):
        p = PlayerStats(player_id="abc")
        assert p.display_name == "abc"

    def test_post_init_display_name_provided(self):
        p = PlayerStats(player_id="abc", display_name="Alice")
        assert p.display_name == "Alice"

    def test_win_rate_zero_games(self, sample_player):
        assert sample_player.win_rate == 0.0

    def test_loss_rate_zero_games(self, sample_player):
        assert sample_player.loss_rate == 0.0

    def test_win_rate_nonzero(self):
        p = PlayerStats(player_id="x", games_played=10, wins=7)
        assert abs(p.win_rate - 0.7) < 1e-9

    def test_loss_rate_nonzero(self):
        p = PlayerStats(player_id="x", games_played=10, losses=3)
        assert abs(p.loss_rate - 0.3) < 1e-9

    def test_record_game_win(self, sample_player):
        sample_player.record_game(MatchOutcome.WIN, rating_change=16.0)
        assert sample_player.wins == 1
        assert sample_player.games_played == 1
        assert sample_player.win_streak == 1
        assert sample_player.rating == 1516.0
        assert sample_player.best_rating == 1516.0
        assert sample_player.last_game_at is not None

    def test_record_game_loss(self, sample_player):
        sample_player.record_game(MatchOutcome.WIN, rating_change=16.0)
        sample_player.record_game(MatchOutcome.LOSS, rating_change=-16.0)
        assert sample_player.losses == 1
        assert sample_player.win_streak == 0

    def test_record_game_draw(self, sample_player):
        sample_player.record_game(MatchOutcome.DRAW, rating_change=0.0)
        assert sample_player.draws == 1

    def test_record_game_best_streak(self, sample_player):
        for _ in range(5):
            sample_player.record_game(MatchOutcome.WIN, rating_change=1.0)
        assert sample_player.best_streak == 5
        sample_player.record_game(MatchOutcome.LOSS, rating_change=-1.0)
        assert sample_player.best_streak == 5  # preserved

    def test_record_game_with_tier_config(self, sample_player):
        tier = TierConfig(name="Gold", rating_min=1500, rating_max=1800)
        sample_player.record_game(MatchOutcome.WIN, rating_change=10.0, tier_config=tier)
        assert sample_player.tier == "Gold"

    def test_record_game_tier_config_not_matching(self, sample_player):
        tier = TierConfig(name="Gold", rating_min=1600, rating_max=1800)
        sample_player.record_game(MatchOutcome.WIN, rating_change=10.0, tier_config=tier)
        # Rating 1510 not in Gold range, tier stays as default
        assert sample_player.tier == "Bronze"

    def test_provisional_becomes_false_after_30_games(self, sample_player):
        assert sample_player.is_provisional is True
        for _ in range(30):
            sample_player.record_game(MatchOutcome.WIN, rating_change=1.0)
        assert sample_player.is_provisional is False

    def test_rating_history_appended(self, sample_player):
        sample_player.record_game(MatchOutcome.WIN, rating_change=10.0)
        assert len(sample_player.rating_history) == 1
        assert sample_player.rating_history[0][1] == 1510.0

    def test_to_dict(self, sample_player):
        d = sample_player.to_dict()
        assert d["player_id"] == "test_player"
        assert "win_rate" in d

    def test_from_dict_roundtrip(self, sample_player):
        sample_player.record_game(MatchOutcome.WIN, rating_change=10.0)
        d = sample_player.to_dict()
        restored = PlayerStats.from_dict(d)
        assert restored.player_id == sample_player.player_id
        assert restored.rating == sample_player.rating
        assert restored.wins == sample_player.wins

    def test_save_and_load(self, tmp_path, sample_player):
        sample_player.record_game(MatchOutcome.WIN, rating_change=15.0)
        path = tmp_path / "player.json"
        sample_player.save(path)
        loaded = PlayerStats.load(path)
        assert loaded.player_id == sample_player.player_id
        assert loaded.rating == sample_player.rating


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------


class TestMatchResult:
    def test_is_draw_true(self):
        r = MatchResult(
            match_id="m1",
            player1_id="a",
            player2_id="b",
            player1_score=1.0,
            player2_score=1.0,
            winner_id=None,
        )
        assert r.is_draw is True

    def test_is_draw_false_winner(self):
        r = MatchResult(
            match_id="m2",
            player1_id="a",
            player2_id="b",
            winner_id="a",
        )
        assert r.is_draw is False

    def test_is_draw_false_unequal_scores(self):
        r = MatchResult(
            match_id="m3",
            player1_id="a",
            player2_id="b",
            player1_score=2.0,
            player2_score=1.0,
            winner_id=None,
        )
        assert r.is_draw is False

    def test_get_outcome_for_winner(self):
        r = MatchResult(match_id="m4", player1_id="a", player2_id="b", winner_id="a")
        assert r.get_outcome_for_player("a") == MatchOutcome.WIN

    def test_get_outcome_for_loser(self):
        r = MatchResult(match_id="m5", player1_id="a", player2_id="b", winner_id="a")
        assert r.get_outcome_for_player("b") == MatchOutcome.LOSS

    def test_get_outcome_for_draw(self):
        r = MatchResult(
            match_id="m6",
            player1_id="a",
            player2_id="b",
            winner_id=None,
            player1_score=1.0,
            player2_score=1.0,
        )
        assert r.get_outcome_for_player("a") == MatchOutcome.DRAW

    def test_to_dict(self):
        r = MatchResult(match_id="m7", player1_id="a", player2_id="b", winner_id="a")
        d = r.to_dict()
        assert d["match_id"] == "m7"
        assert "is_draw" in d

    def test_from_dict_roundtrip(self):
        r = MatchResult(
            match_id="m8",
            player1_id="a",
            player2_id="b",
            winner_id="a",
            player1_score=2.0,
            player2_score=0.0,
            duration_seconds=120.0,
        )
        d = r.to_dict()
        restored = MatchResult.from_dict(d)
        assert restored.match_id == "m8"
        assert restored.winner_id == "a"
        assert restored.duration_seconds == 120.0


# ---------------------------------------------------------------------------
# MatchOutcome enum
# ---------------------------------------------------------------------------


class TestMatchOutcome:
    def test_values(self):
        assert MatchOutcome.WIN.value == "win"
        assert MatchOutcome.LOSS.value == "loss"
        assert MatchOutcome.DRAW.value == "draw"

    def test_string_comparison(self):
        assert MatchOutcome.WIN == "win"
