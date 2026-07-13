"""Comprehensive tests for alpha_zero.tournament.elo module.

Focuses on lines not covered by existing test_tournament_elo.py:
- ELORatingSystem.process_game_result (all outcome branches)
- ELORatingSystem.simulate_rating_change (LOSS, DRAW branches)
- ELORatingSystem.get_rating_needed_for_probability (edge cases)
- ELORatingSystem.clear_history
- RatingLeaderboard.get_rank (not found case)
- RatingLeaderboard.size property
- RatingUpdate.to_dict (already partially covered, edge cases)
"""

from __future__ import annotations

import pytest

from alpha_zero.tournament.config import (
    ELOConfig,
    MatchOutcome,
    PlayerStats,
    TournamentConfig,
)
from alpha_zero.tournament.elo import (
    ELORatingSystem,
    RatingLeaderboard,
    calculate_expected_score,
    calculate_rating_change,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def elo_system():
    return ELORatingSystem()


@pytest.fixture()
def player1():
    return PlayerStats(player_id="p1", rating=1500.0)


@pytest.fixture()
def player2():
    return PlayerStats(player_id="p2", rating=1500.0)


@pytest.fixture()
def tournament_config():
    return TournamentConfig()


# ---------------------------------------------------------------------------
# process_game_result
# ---------------------------------------------------------------------------


class TestProcessGameResult:
    def test_win_outcome(self, elo_system, player1, player2):
        u1, u2 = elo_system.process_game_result(player1, player2, MatchOutcome.WIN)
        assert u1.actual_score == 1.0
        assert u2.actual_score == 0.0
        assert player1.wins == 1
        assert player2.losses == 1

    def test_loss_outcome(self, elo_system, player1, player2):
        u1, u2 = elo_system.process_game_result(player1, player2, MatchOutcome.LOSS)
        assert u1.actual_score == 0.0
        assert u2.actual_score == 1.0
        assert player1.losses == 1
        assert player2.wins == 1

    def test_draw_outcome(self, elo_system, player1, player2):
        u1, u2 = elo_system.process_game_result(player1, player2, MatchOutcome.DRAW)
        assert u1.actual_score == 0.5
        assert u2.actual_score == 0.5
        assert player1.draws == 1
        assert player2.draws == 1

    def test_with_tournament_config(self, elo_system, player1, player2, tournament_config):
        _u1, _u2 = elo_system.process_game_result(
            player1, player2, MatchOutcome.WIN, config=tournament_config
        )
        # Player stats should be updated including tier
        assert player1.games_played == 1
        assert player2.games_played == 1

    def test_updates_player_ratings(self, elo_system, player1, player2):
        elo_system.process_game_result(player1, player2, MatchOutcome.WIN)
        assert player1.rating > 1500.0
        assert player2.rating < 1500.0

    def test_config_none_no_tier_update(self, elo_system, player1, player2):
        original_tier = player1.tier
        elo_system.process_game_result(player1, player2, MatchOutcome.WIN, config=None)
        # Tier not updated when config is None (no tier_config passed)
        assert player1.tier == original_tier


# ---------------------------------------------------------------------------
# simulate_rating_change
# ---------------------------------------------------------------------------


class TestSimulateRatingChange:
    def test_simulate_win(self, elo_system):
        change = elo_system.simulate_rating_change(1500, 1500, MatchOutcome.WIN)
        assert change > 0

    def test_simulate_loss(self, elo_system):
        change = elo_system.simulate_rating_change(1500, 1500, MatchOutcome.LOSS)
        assert change < 0

    def test_simulate_draw_equal_ratings(self, elo_system):
        change = elo_system.simulate_rating_change(1500, 1500, MatchOutcome.DRAW)
        assert abs(change) < 0.01

    def test_simulate_draw_different_ratings(self, elo_system):
        # Higher-rated player drawing with lower-rated loses rating
        change = elo_system.simulate_rating_change(1600, 1400, MatchOutcome.DRAW)
        assert change < 0

    def test_simulate_does_not_record(self, elo_system):
        elo_system.simulate_rating_change(1500, 1500, MatchOutcome.WIN)
        assert len(elo_system.get_update_history()) == 0

    def test_simulate_with_games_played(self, elo_system):
        change_new = elo_system.simulate_rating_change(1500, 1500, MatchOutcome.WIN, games_played=0)
        change_exp = elo_system.simulate_rating_change(
            1500, 1500, MatchOutcome.WIN, games_played=200
        )
        # Experienced player has lower K-factor, smaller change
        assert abs(change_exp) < abs(change_new)


# ---------------------------------------------------------------------------
# get_rating_needed_for_probability
# ---------------------------------------------------------------------------


class TestGetRatingNeededForProbability:
    def test_50_percent(self, elo_system):
        rating = elo_system.get_rating_needed_for_probability(1500, 0.5)
        assert abs(rating - 1500) < 1.0

    def test_75_percent(self, elo_system):
        rating = elo_system.get_rating_needed_for_probability(1500, 0.75)
        assert rating > 1500

    def test_25_percent(self, elo_system):
        rating = elo_system.get_rating_needed_for_probability(1500, 0.25)
        assert rating < 1500

    def test_invalid_probability_zero(self, elo_system):
        with pytest.raises(ValueError, match="Win probability must be between 0 and 1"):
            elo_system.get_rating_needed_for_probability(1500, 0.0)

    def test_invalid_probability_one(self, elo_system):
        with pytest.raises(ValueError, match="Win probability must be between 0 and 1"):
            elo_system.get_rating_needed_for_probability(1500, 1.0)

    def test_invalid_probability_negative(self, elo_system):
        with pytest.raises(ValueError):
            elo_system.get_rating_needed_for_probability(1500, -0.1)

    def test_invalid_probability_over_one(self, elo_system):
        with pytest.raises(ValueError):
            elo_system.get_rating_needed_for_probability(1500, 1.5)

    def test_roundtrip_with_expected_score(self, elo_system):
        """Rating computed for 0.9 probability should yield ~0.9 expected score."""
        target_prob = 0.9
        rating = elo_system.get_rating_needed_for_probability(1500, target_prob)
        actual_prob = calculate_expected_score(rating, 1500)
        assert abs(actual_prob - target_prob) < 0.001


# ---------------------------------------------------------------------------
# clear_history
# ---------------------------------------------------------------------------


class TestClearHistory:
    def test_clear_history(self, elo_system):
        elo_system.process_match("a", "b", 1500, 1500, 1.0, 0.0)
        assert len(elo_system.get_update_history()) > 0
        elo_system.clear_history()
        assert len(elo_system.get_update_history()) == 0


# ---------------------------------------------------------------------------
# get_update_history filtering & limit
# ---------------------------------------------------------------------------


class TestGetUpdateHistory:
    def test_filter_by_player(self, elo_system):
        elo_system.process_match("a", "b", 1500, 1500, 1.0, 0.0)
        elo_system.process_match("c", "b", 1500, 1500, 1.0, 0.0)
        a_updates = elo_system.get_update_history("a")
        assert all(u.player_id == "a" for u in a_updates)
        assert len(a_updates) == 1

    def test_limit(self, elo_system):
        for i in range(10):
            elo_system.process_match(f"p{i}", f"q{i}", 1500, 1500, 1.0, 0.0)
        history = elo_system.get_update_history(limit=5)
        assert len(history) == 5

    def test_all_updates_returned(self, elo_system):
        elo_system.process_match("a", "b", 1500, 1500, 1.0, 0.0)
        all_updates = elo_system.get_update_history()
        assert len(all_updates) == 2


# ---------------------------------------------------------------------------
# apply_rating_bounds edge cases
# ---------------------------------------------------------------------------


class TestApplyRatingBounds:
    def test_at_floor(self):
        config = ELOConfig(rating_floor=100, rating_ceiling=3000)
        system = ELORatingSystem(config)
        assert system.apply_rating_bounds(100) == 100

    def test_at_ceiling(self):
        config = ELOConfig(rating_floor=100, rating_ceiling=3000)
        system = ELORatingSystem(config)
        assert system.apply_rating_bounds(3000) == 3000

    def test_negative_value(self):
        config = ELOConfig(rating_floor=100, rating_ceiling=3000)
        system = ELORatingSystem(config)
        assert system.apply_rating_bounds(-500) == 100


# ---------------------------------------------------------------------------
# process_match with bounds clamping
# ---------------------------------------------------------------------------


class TestProcessMatchBounds:
    def test_rating_clamped_at_floor(self):
        config = ELOConfig(rating_floor=100, rating_ceiling=3000, k_factor_base=200)
        system = ELORatingSystem(config)
        # Very low rated player loses to high rated player with huge K
        u1, _u2 = system.process_match("low", "high", 105, 2900, 0.0, 1.0)
        assert u1.rating_after >= 100

    def test_rating_clamped_at_ceiling(self):
        config = ELOConfig(rating_floor=100, rating_ceiling=3000, k_factor_base=200)
        system = ELORatingSystem(config)
        u1, _u2 = system.process_match("high", "low", 2995, 200, 1.0, 0.0)
        assert u1.rating_after <= 3000


# ---------------------------------------------------------------------------
# RatingLeaderboard additional coverage
# ---------------------------------------------------------------------------


class TestRatingLeaderboardCoverage:
    def test_size_property(self):
        lb = RatingLeaderboard()
        assert lb.size == 0
        lb.update("a", 1500)
        assert lb.size == 1

    def test_get_rank_not_found(self):
        lb = RatingLeaderboard()
        assert lb.get_rank("ghost") is None

    def test_remove_nonexistent(self):
        lb = RatingLeaderboard()
        lb.remove("nobody")  # Should not raise
        assert lb.size == 0

    def test_get_top_empty(self):
        lb = RatingLeaderboard()
        assert lb.get_top(5) == []

    def test_get_top_fewer_than_n(self):
        lb = RatingLeaderboard()
        lb.update("a", 1500)
        top = lb.get_top(5)
        assert len(top) == 1

    def test_get_players_in_range_empty(self):
        lb = RatingLeaderboard()
        assert lb.get_players_in_range(1000, 2000) == []

    def test_get_players_in_range_boundary(self):
        lb = RatingLeaderboard()
        lb.update("a", 1000)
        lb.update("b", 2000)
        result = lb.get_players_in_range(1000, 2000)
        assert len(result) == 2

    def test_update_overwrites(self):
        lb = RatingLeaderboard()
        lb.update("a", 1500)
        lb.update("a", 1600)
        assert lb.get_rating("a") == 1600
        assert lb.size == 1


# ---------------------------------------------------------------------------
# calculate_rating_change edge cases
# ---------------------------------------------------------------------------


class TestCalculateRatingChangeEdge:
    def test_zero_k_factor(self):
        change = calculate_rating_change(1500, 0.5, 1.0, 0.0)
        assert change == 0.0

    def test_perfect_expected_score_loss(self):
        # Expected to win (1.0) but lost (0.0) — maximum negative change
        change = calculate_rating_change(1500, 1.0, 0.0, 32)
        assert change == -32.0
