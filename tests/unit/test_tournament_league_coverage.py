"""Comprehensive tests for alpha_zero.tournament.league module.

Covers:
- TierProgress dataclass (properties, record methods, serialization)
- SeasonInfo dataclass (serialization, from_dict)
- LeagueManager (tier lookup, registration, promotion/demotion, seasons,
  match processing, leaderboards, statistics)
"""

from __future__ import annotations

import pytest

from alpha_zero.tournament.config import (
    MatchResult,
    PlayerStats,
    TierConfig,
    TournamentConfig,
)
from alpha_zero.tournament.league import (
    LeagueManager,
    SeasonInfo,
    TierProgress,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_config():
    """A TournamentConfig with standard tiers."""
    return TournamentConfig()


@pytest.fixture()
def league(default_config):
    """A LeagueManager with the default configuration."""
    return LeagueManager(default_config)


@pytest.fixture()
def bronze_player():
    """A player in the Bronze tier."""
    return PlayerStats(player_id="bronze_p", rating=1100.0)


@pytest.fixture()
def gold_player():
    """A player in the Gold tier."""
    return PlayerStats(player_id="gold_p", rating=1600.0)


@pytest.fixture()
def silver_player():
    """A player in the Silver tier."""
    return PlayerStats(player_id="silver_p", rating=1300.0)


# ---------------------------------------------------------------------------
# TierProgress
# ---------------------------------------------------------------------------


class TestTierProgress:
    """Tests for the TierProgress dataclass."""

    def test_default_values(self):
        tp = TierProgress()
        assert tp.tier_name == "Bronze"
        assert tp.wins_in_tier == 0
        assert tp.losses_in_tier == 0
        assert tp.streak == 0
        assert tp.at_promotion_threshold is False
        assert tp.at_demotion_threshold is False

    def test_games_in_tier_property(self):
        tp = TierProgress(wins_in_tier=5, losses_in_tier=3)
        assert tp.games_in_tier == 8

    def test_win_rate_in_tier_zero_games(self):
        tp = TierProgress()
        assert tp.win_rate_in_tier == 0.0

    def test_win_rate_in_tier_nonzero(self):
        tp = TierProgress(wins_in_tier=7, losses_in_tier=3)
        assert abs(tp.win_rate_in_tier - 0.7) < 1e-9

    def test_record_win_increments_and_positive_streak(self):
        tp = TierProgress()
        tp.record_win()
        assert tp.wins_in_tier == 1
        assert tp.streak == 1
        tp.record_win()
        assert tp.streak == 2

    def test_record_win_resets_negative_streak(self):
        tp = TierProgress(streak=-3)
        tp.record_win()
        assert tp.streak == 1

    def test_record_loss_increments_and_negative_streak(self):
        tp = TierProgress()
        tp.record_loss()
        assert tp.losses_in_tier == 1
        assert tp.streak == -1
        tp.record_loss()
        assert tp.streak == -2

    def test_record_loss_resets_positive_streak(self):
        tp = TierProgress(streak=5)
        tp.record_loss()
        assert tp.streak == -1

    def test_record_win_during_promotion(self):
        tp = TierProgress(at_promotion_threshold=True)
        tp.record_win()
        assert tp.promotion_wins == 1

    def test_record_loss_during_promotion(self):
        tp = TierProgress(at_promotion_threshold=True)
        tp.record_loss()
        assert tp.promotion_losses == 1

    def test_reset_for_new_tier(self):
        tp = TierProgress(
            tier_name="Bronze",
            wins_in_tier=10,
            losses_in_tier=5,
            streak=3,
            at_promotion_threshold=True,
            at_demotion_threshold=True,
            promotion_wins=2,
            promotion_losses=1,
        )
        tp.reset_for_new_tier("Silver")
        assert tp.tier_name == "Silver"
        assert tp.wins_in_tier == 0
        assert tp.losses_in_tier == 0
        assert tp.at_promotion_threshold is False
        assert tp.at_demotion_threshold is False
        assert tp.promotion_wins == 0
        assert tp.promotion_losses == 0
        # streak is preserved
        assert tp.streak == 3

    def test_to_dict(self):
        tp = TierProgress(tier_name="Gold", wins_in_tier=4, losses_in_tier=2, streak=2)
        d = tp.to_dict()
        assert d["tier_name"] == "Gold"
        assert d["wins_in_tier"] == 4
        assert d["losses_in_tier"] == 2
        assert d["streak"] == 2

    def test_from_dict_roundtrip(self):
        tp = TierProgress(tier_name="Platinum", wins_in_tier=3, losses_in_tier=1, streak=-1)
        d = tp.to_dict()
        tp2 = TierProgress.from_dict(d)
        assert tp2.tier_name == tp.tier_name
        assert tp2.wins_in_tier == tp.wins_in_tier
        assert tp2.streak == tp.streak


# ---------------------------------------------------------------------------
# SeasonInfo
# ---------------------------------------------------------------------------


class TestSeasonInfo:
    """Tests for the SeasonInfo dataclass."""

    def test_defaults(self):
        s = SeasonInfo(season_id="s1")
        assert s.season_id == "s1"
        assert s.name == "Season 1"
        assert s.is_active is True
        assert s.end_date is None
        assert s.rewards == {}

    def test_to_dict(self):
        s = SeasonInfo(season_id="s2", name="Autumn", rewards={"gold": 100})
        d = s.to_dict()
        assert d["season_id"] == "s2"
        assert d["name"] == "Autumn"
        assert d["rewards"]["gold"] == 100

    def test_from_dict_roundtrip(self):
        s = SeasonInfo(season_id="s3", name="Winter", is_active=False, end_date="2026-01-01")
        d = s.to_dict()
        s2 = SeasonInfo.from_dict(d)
        assert s2.season_id == s.season_id
        assert s2.is_active is False
        assert s2.end_date == "2026-01-01"


# ---------------------------------------------------------------------------
# LeagueManager
# ---------------------------------------------------------------------------


class TestLeagueManagerInit:
    def test_initialization(self, league, default_config):
        assert league.config is default_config

    def test_empty_tiers_config(self):
        config = TournamentConfig(tiers=[])
        lm = LeagueManager(config)
        default_tier = lm._get_default_tier()
        assert default_tier.name == "Bronze"


class TestLeagueManagerTierLookup:
    def test_get_tier_for_rating_bronze(self, league):
        tier = league.get_tier_for_rating(1000)
        assert tier is not None
        assert tier.name == "Bronze"

    def test_get_tier_for_rating_gold(self, league):
        tier = league.get_tier_for_rating(1600)
        assert tier is not None
        assert tier.name == "Gold"

    def test_get_tier_for_rating_master(self, league):
        tier = league.get_tier_for_rating(2500)
        assert tier is not None
        assert tier.name == "Master"

    def test_get_tier_for_rating_none(self):
        config = TournamentConfig(tiers=[TierConfig("Only", 1000, 2000)])
        lm = LeagueManager(config)
        assert lm.get_tier_for_rating(500) is None

    def test_get_player_tier(self, league, gold_player):
        tier = league.get_player_tier(gold_player)
        assert tier is not None
        assert tier.name == "Gold"


class TestLeagueManagerRegistration:
    def test_register_player(self, league, bronze_player):
        progress = league.register_player(bronze_player)
        assert progress.tier_name == "Bronze"
        assert bronze_player.player_id in league._tier_populations["Bronze"]

    def test_register_player_unknown_rating(self):
        config = TournamentConfig(tiers=[TierConfig("Only", 1000, 2000)])
        lm = LeagueManager(config)
        player = PlayerStats(player_id="low", rating=500)
        progress = lm.register_player(player)
        # Falls back to default tier since no tier matches
        assert progress.tier_name == "Only"

    def test_get_player_progress_creates_default(self, league):
        progress = league.get_player_progress("new_player")
        assert progress.tier_name == "Bronze"


class TestLeagueManagerPromotion:
    def test_promotion_on_rating_increase(self, league):
        player = PlayerStats(player_id="p1", rating=1100.0)
        league.register_player(player)
        # Simulate rating increase past Silver threshold
        player.rating = 1250.0
        promoted, demoted = league.update_player_tier(player, won_last_game=True)
        assert promoted is True
        assert demoted is False

    def test_demotion_on_rating_decrease(self, league):
        player = PlayerStats(player_id="p2", rating=1300.0)
        league.register_player(player)
        # Simulate rating drop into Bronze
        player.rating = 1100.0
        promoted, demoted = league.update_player_tier(player, won_last_game=False)
        assert promoted is False
        assert demoted is True

    def test_no_demotion_from_lowest_tier(self, league):
        player = PlayerStats(player_id="p3", rating=1100.0)
        league.register_player(player)
        # Even if somehow tier lookup says a lower tier, can't demote from Bronze
        # Force the player progress to Bronze (index 0)
        progress = league.get_player_progress("p3")
        assert progress.tier_name == "Bronze"
        # Rating stays in Bronze, no change
        promoted, demoted = league.update_player_tier(player, won_last_game=False)
        assert promoted is False
        assert demoted is False

    def test_no_change_when_same_tier(self, league, bronze_player):
        league.register_player(bronze_player)
        promoted, demoted = league.update_player_tier(bronze_player, won_last_game=True)
        assert promoted is False
        assert demoted is False


class TestLeagueManagerThresholds:
    def test_promotion_threshold_set_after_enough_games(self, league):
        player = PlayerStats(player_id="t1", rating=1100.0)
        league.register_player(player)
        progress = league.get_player_progress("t1")
        # Simulate 10+ games with high win rate
        progress.wins_in_tier = 8
        progress.losses_in_tier = 2
        league._update_thresholds(progress, league._get_default_tier())
        assert progress.at_promotion_threshold is True

    def test_demotion_threshold_set_after_enough_games(self, league):
        player = PlayerStats(player_id="t2", rating=1100.0)
        league.register_player(player)
        progress = league.get_player_progress("t2")
        progress.wins_in_tier = 3
        progress.losses_in_tier = 8
        league._update_thresholds(progress, league._get_default_tier())
        assert progress.at_demotion_threshold is True

    def test_no_threshold_update_under_10_games(self, league):
        player = PlayerStats(player_id="t3", rating=1100.0)
        league.register_player(player)
        progress = league.get_player_progress("t3")
        progress.wins_in_tier = 5
        progress.losses_in_tier = 3
        league._update_thresholds(progress, league._get_default_tier())
        assert progress.at_promotion_threshold is False
        assert progress.at_demotion_threshold is False

    def test_promotion_threshold_resets_when_below(self, league):
        player = PlayerStats(player_id="t4", rating=1100.0)
        league.register_player(player)
        progress = league.get_player_progress("t4")
        progress.wins_in_tier = 6
        progress.losses_in_tier = 5
        progress.at_promotion_threshold = True
        progress.promotion_wins = 2
        progress.promotion_losses = 1
        league._update_thresholds(progress, league._get_default_tier())
        assert progress.at_promotion_threshold is False
        assert progress.promotion_wins == 0
        assert progress.promotion_losses == 0


class TestLeagueManagerMatchResult:
    def test_process_match_result_with_winner(self, league):
        p1 = PlayerStats(player_id="m1", rating=1100.0)
        p2 = PlayerStats(player_id="m2", rating=1100.0)
        league.register_player(p1)
        league.register_player(p2)
        result = MatchResult(
            match_id="match_1",
            player1_id="m1",
            player2_id="m2",
            winner_id="m1",
        )
        changes = league.process_match_result(p1, p2, result)
        assert "m1" in changes
        assert "m2" in changes

    def test_process_match_result_draw_no_winner(self, league):
        p1 = PlayerStats(player_id="d1", rating=1100.0)
        p2 = PlayerStats(player_id="d2", rating=1100.0)
        league.register_player(p1)
        league.register_player(p2)
        result = MatchResult(
            match_id="match_d",
            player1_id="d1",
            player2_id="d2",
            winner_id=None,
            player1_score=0.5,
            player2_score=0.5,
        )
        changes = league.process_match_result(p1, p2, result)
        # Both are neither promoted nor demoted on draw in same tier
        assert changes["d1"] == (False, False)
        assert changes["d2"] == (False, False)

    def test_process_match_result_score_comparison_fallback(self, league):
        p1 = PlayerStats(player_id="s1", rating=1100.0)
        p2 = PlayerStats(player_id="s2", rating=1100.0)
        league.register_player(p1)
        league.register_player(p2)
        result = MatchResult(
            match_id="match_s",
            player1_id="s1",
            player2_id="s2",
            winner_id=None,
            player1_score=3.0,
            player2_score=1.0,
        )
        changes = league.process_match_result(p1, p2, result)
        assert "s1" in changes
        assert "s2" in changes


class TestLeagueManagerStatistics:
    def test_get_tier_statistics(self, league, bronze_player):
        league.register_player(bronze_player)
        stats = league.get_tier_statistics()
        assert "Bronze" in stats
        assert stats["Bronze"]["population"] >= 1
        assert "color" in stats["Bronze"]

    def test_get_tier_leaderboard(self, league):
        p1 = PlayerStats(player_id="lb1", rating=1100.0)
        p2 = PlayerStats(player_id="lb2", rating=1150.0)
        league.register_player(p1)
        league.register_player(p2)
        players = {"lb1": p1, "lb2": p2}
        board = league.get_tier_leaderboard("Bronze", players, limit=5)
        assert len(board) == 2
        # Higher rated first
        assert board[0][0] == "lb2"
        assert board[0][2] == 1  # rank 1

    def test_get_tier_leaderboard_empty(self, league):
        board = league.get_tier_leaderboard("Master", {}, limit=5)
        assert board == []

    def test_get_tier_leaderboard_missing_player(self, league):
        p1 = PlayerStats(player_id="lb3", rating=1100.0)
        league.register_player(p1)
        # Player in population but not in players dict
        board = league.get_tier_leaderboard("Bronze", {}, limit=5)
        assert board == []

    def test_get_global_statistics(self, league, bronze_player):
        league.register_player(bronze_player)
        stats = league.get_global_statistics()
        assert stats["total_players"] >= 1
        assert stats["num_tiers"] == len(league.config.tiers)
        assert stats["season_active"] is False
        assert stats["current_season"] is None


class TestLeagueManagerSeasons:
    def test_start_season(self, league):
        season = league.start_season("s1", name="Alpha")
        assert season.is_active is True
        assert season.name == "Alpha"
        assert league.get_current_season() is season

    def test_start_season_ends_previous(self, league):
        s1 = league.start_season("s1", name="First")
        s2 = league.start_season("s2", name="Second")
        assert s1.is_active is False
        assert s1.end_date is not None
        assert s2.is_active is True

    def test_start_season_resets_progress(self, league, bronze_player):
        league.register_player(bronze_player)
        progress = league.get_player_progress(bronze_player.player_id)
        progress.wins_in_tier = 10
        progress.losses_in_tier = 5
        progress.at_promotion_threshold = True
        league.start_season("s1")
        assert progress.wins_in_tier == 0
        assert progress.losses_in_tier == 0
        assert progress.at_promotion_threshold is False

    def test_end_season(self, league):
        league.start_season("s1", name="One")
        ended = league.end_season()
        assert ended is not None
        assert ended.is_active is False
        assert ended.end_date is not None
        assert league.get_current_season() is None

    def test_end_season_no_active(self, league):
        result = league.end_season()
        assert result is None

    def test_get_season_history(self, league):
        league.start_season("s1", name="One")
        league.start_season("s2", name="Two")
        league.end_season()
        history = league.get_season_history()
        # s1 was ended when s2 started, s2 was ended explicitly
        assert len(history) == 2

    def test_global_stats_with_active_season(self, league):
        league.start_season("s1", name="Active")
        stats = league.get_global_statistics()
        assert stats["season_active"] is True
        assert stats["current_season"] == "Active"
