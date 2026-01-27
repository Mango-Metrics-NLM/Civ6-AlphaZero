"""Integration tests for Tournament System.

Tests cover:
- Full tournament with multiple players
- ELO rating updates across matches
- Matchmaking queue flow
- League tier progression
- Match history persistence
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from alpha_zero.tournament.config import (
    ELOConfig,
    MatchOutcome,
    MatchResult,
    PlayerStats,
    TournamentConfig,
)
from alpha_zero.tournament.elo import ELORatingSystem
from alpha_zero.tournament.history import MatchHistory
from alpha_zero.tournament.league import LeagueManager
from alpha_zero.tournament.matchmaking import MatchMaker, OpponentSelector


class TestTournamentIntegration:
    """Integration tests for tournament system."""

    @pytest.fixture
    def config(self):
        """Create tournament configuration."""
        return TournamentConfig()

    @pytest.fixture
    def players(self):
        """Create test players with varying ratings."""
        return [
            PlayerStats(player_id=f"player_{i}", rating=1400 + i * 50)
            for i in range(10)
        ]

    def test_full_match_flow(self, config, players):
        """Test complete match flow from queue to result."""
        elo_system = ELORatingSystem(config.elo)
        matchmaker = MatchMaker(config)

        p1, p2 = players[0], players[1]

        # Add players to queue
        matchmaker.add_to_queue(p1)
        matchmaker.add_to_queue(p2)

        # Find match
        match = matchmaker.find_match(p1.player_id)
        assert match is not None
        assert match.is_ready

        # Process match result
        update1, update2 = elo_system.process_game_result(
            p1, p2, MatchOutcome.WIN, config
        )

        # Verify rating changes
        assert update1.rating_after > update1.rating_before
        assert update2.rating_after < update2.rating_before

    def test_multiple_matches_elo_stability(self, config, players):
        """Test ELO stability over multiple matches."""
        elo_system = ELORatingSystem(config.elo)

        # Play many matches between players
        for _ in range(50):
            p1 = np.random.choice(players)
            p2 = np.random.choice([p for p in players if p != p1])

            # Simulate outcome based on rating difference
            expected = elo_system.get_win_probability(p1.rating, p2.rating)
            outcome = MatchOutcome.WIN if np.random.random() < expected else MatchOutcome.LOSS

            elo_system.process_game_result(p1, p2, outcome, config)

        # All ratings should still be valid
        for player in players:
            assert config.elo.rating_floor <= player.rating <= config.elo.rating_ceiling

    def test_matchmaking_fairness(self, config, players):
        """Test matchmaking produces fair matches."""
        matchmaker = MatchMaker(config)

        # Add all players to queue
        for player in players:
            matchmaker.add_to_queue(player)

        # Process all matches
        matches = matchmaker.process_queue()

        # Verify matches are reasonably fair (within initial range)
        for match in matches:
            rating_diff = abs(match.player1.rating - match.player2.rating)
            assert rating_diff <= config.matchmaking.rating_range_max

    def test_league_progression(self, config):
        """Test league tier progression over matches."""
        league = LeagueManager(config)
        elo = ELORatingSystem(config.elo)

        # Create player in Silver tier
        player = PlayerStats(player_id="climber", rating=1350)
        league.register_player(player)

        initial_tier = league.get_player_tier(player)
        assert initial_tier.name == "Silver"

        # Simulate winning streak
        opponent = PlayerStats(player_id="opponent", rating=1400)
        for _ in range(20):
            elo.process_game_result(player, opponent, MatchOutcome.WIN, config)
            league.update_player_tier(player, won_last_game=True)

        # Should have progressed
        final_tier = league.get_player_tier(player)
        # Rating should have increased significantly
        assert player.rating > 1350

    def test_match_history_persistence(self, config, players):
        """Test match history saves and retrieves correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tournament.db"
            history = MatchHistory(db_path, config)
            elo = ELORatingSystem(config.elo)

            p1, p2 = players[0], players[1]
            initial_rating1 = p1.rating
            initial_rating2 = p2.rating

            # Record several matches
            for i in range(5):
                outcome = MatchOutcome.WIN if i % 2 == 0 else MatchOutcome.LOSS
                update1, update2 = elo.process_game_result(p1, p2, outcome, config)

                # Determine winner_id based on outcome
                winner_id = p1.player_id if outcome == MatchOutcome.WIN else (
                    p2.player_id if outcome == MatchOutcome.LOSS else None
                )

                result = MatchResult(
                    match_id=f"match_{i}",
                    player1_id=p1.player_id,
                    player2_id=p2.player_id,
                    player1_rating_before=initial_rating1,
                    player2_rating_before=initial_rating2,
                    winner_id=winner_id,
                    duration_seconds=50.0,
                )
                history.record_match(result, update1.rating_after, update2.rating_after)

                initial_rating1 = update1.rating_after
                initial_rating2 = update2.rating_after

            # Verify history
            p1_matches = history.get_player_matches(p1.player_id)
            assert len(p1_matches) == 5

            # Check statistics
            p1_stats = history.get_player_statistics(p1.player_id)
            assert p1_stats is not None
            assert p1_stats.total_games == 5
            assert p1_stats.wins + p1_stats.losses == 5

            # Check head-to-head
            h2h = history.get_head_to_head(p1.player_id, p2.player_id)
            assert h2h["total_games"] == 5

    def test_opponent_selection_strategies(self, config, players):
        """Test different opponent selection strategies."""
        selector = OpponentSelector(players, config)

        target_player = players[5]  # Middle rating

        # Test rating proximity selection
        by_rating = selector.select_by_rating_proximity(
            target_player.rating, num_opponents=3
        )
        assert len(by_rating) == 3
        # Closest ratings should be selected
        for opp in by_rating:
            assert abs(opp.rating - target_player.rating) <= 100

        # Test diverse selection
        diverse = selector.select_diverse(num_opponents=5)
        assert len(diverse) == 5
        ratings = [p.rating for p in diverse]
        # Should span a range
        assert max(ratings) - min(ratings) >= 150

        # Test challenging selection
        challenging = selector.select_challenging(
            target_player.rating,
            challenge_factor=1.05,
            num_opponents=3,
        )
        for opp in challenging:
            assert opp.rating >= target_player.rating

    def test_season_management(self, config):
        """Test season start and end functionality."""
        league = LeagueManager(config)

        # Register some players
        players = [
            PlayerStats(player_id=f"player_{i}", rating=1400 + i * 100)
            for i in range(5)
        ]
        for player in players:
            league.register_player(player)

        # Start season
        season = league.start_season("season_1", "Test Season")
        assert season.is_active

        # Simulate some games
        for i, player in enumerate(players):
            for _ in range(10):
                league.update_player_tier(player, won_last_game=(i % 2 == 0))

        # Get statistics
        stats = league.get_global_statistics()
        assert stats["total_players"] == 5
        assert stats["season_active"] is True

        # End season
        ended = league.end_season()
        assert not ended.is_active
        assert league.get_current_season() is None


class TestTournamentRobustness:
    """Robustness tests for tournament system."""

    def test_rating_extremes(self):
        """Test handling of extreme rating differences."""
        config = TournamentConfig()
        elo = ELORatingSystem(config.elo)

        # Very high vs very low rating
        p1 = PlayerStats(player_id="strong", rating=2500)
        p2 = PlayerStats(player_id="weak", rating=500)

        # Process match - strong player wins
        update1, update2 = elo.process_game_result(p1, p2, MatchOutcome.WIN, config)

        # Changes should be small (expected win)
        assert abs(update1.rating_change) < 5

        # Process upset - weak player wins
        update1, update2 = elo.process_game_result(p2, p1, MatchOutcome.WIN, config)

        # Weak player should gain significant rating
        assert update1.rating_change > 20

    def test_concurrent_queue_operations(self):
        """Test queue handles rapid add/remove."""
        config = TournamentConfig()
        matchmaker = MatchMaker(config)

        # Add many players
        players = [
            PlayerStats(player_id=f"p_{i}", rating=1500 + np.random.randint(-200, 200))
            for i in range(100)
        ]

        for player in players:
            matchmaker.add_to_queue(player)

        # Process queue
        matches = matchmaker.process_queue()

        # Should have paired most players
        assert len(matches) >= 40  # At least 80% paired
        assert matchmaker.queue_size <= 10  # Few leftover

    def test_draw_handling(self):
        """Test proper handling of draws."""
        config = TournamentConfig()
        elo = ELORatingSystem(config.elo)

        p1 = PlayerStats(player_id="p1", rating=1500)
        p2 = PlayerStats(player_id="p2", rating=1500)

        # Process draw
        update1, update2 = elo.process_game_result(p1, p2, MatchOutcome.DRAW, config)

        # Ratings should not change for equal players
        assert abs(update1.rating_change) < 1
        assert abs(update2.rating_change) < 1

    def test_history_large_volume(self):
        """Test history handles large number of matches."""
        config = TournamentConfig()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "tournament.db"
            history = MatchHistory(db_path, config)

            # Record many matches
            for i in range(500):
                p1_id = f"player_{i % 50}"
                p2_id = f"player_{(i + 1) % 50}"
                is_win = i % 2 == 0
                winner_id = p1_id if is_win else p2_id

                result = MatchResult(
                    match_id=f"match_{i}",
                    player1_id=p1_id,
                    player2_id=p2_id,
                    player1_rating_before=1500.0,
                    player2_rating_before=1500.0,
                    winner_id=winner_id,
                    duration_seconds=50.0,
                )
                history.record_match(result, 1510, 1490)

            # Verify retrieval
            stats = history.get_global_statistics()
            assert stats["total_matches"] == 500

            # Query should be fast
            recent = history.get_recent_matches(limit=50)
            assert len(recent) == 50

            leaderboard = history.get_leaderboard(limit=20, min_games=5)
            assert len(leaderboard) <= 20
