"""Unit tests for tournament W&B dashboards module."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from alpha_zero.tournament.config import MatchResult
from alpha_zero.tournament.elo import RatingLeaderboard
from alpha_zero.tournament.wandb_dashboards import TournamentWandbLogger
from alpha_zero.utils.wandb_logger import WandbLogger


@pytest.fixture
def mock_wandb_logger():
    """Create a mock WandbLogger."""
    logger = Mock(spec=WandbLogger)
    logger.is_initialized = True
    logger.log = Mock()
    logger.log_table = Mock()
    logger.define_metric = Mock()
    return logger


@pytest.fixture
def tournament_logger(mock_wandb_logger):
    """Create a TournamentWandbLogger with mocked WandbLogger."""
    return TournamentWandbLogger(mock_wandb_logger)


@pytest.fixture
def sample_match():
    """Create a sample match result."""
    return MatchResult(
        match_id="match_001",
        player1_id="player_a",
        player2_id="player_b",
        player1_score=1.0,
        player2_score=0.0,
        player1_rating_before=1500.0,
        player2_rating_before=1600.0,
        player1_rating_change=20.0,
        player2_rating_change=-20.0,
        winner_id="player_a",
        duration_seconds=300.0,
    )


class TestTournamentWandbLogger:
    """Test TournamentWandbLogger class."""

    def test_initialization(self, mock_wandb_logger):
        """Test logger initialization."""
        logger = TournamentWandbLogger(mock_wandb_logger)
        assert logger.wandb_logger is mock_wandb_logger

    def test_define_tournament_charts(self, tournament_logger, mock_wandb_logger):
        """Test chart definition."""
        tournament_logger.define_tournament_charts()

        # Should call define_metric for each metric
        assert mock_wandb_logger.define_metric.call_count >= 5

        # Check that round metric was defined
        calls = [call[0][0] for call in mock_wandb_logger.define_metric.call_args_list]
        assert "tournament/round" in calls

    def test_define_tournament_charts_not_initialized(self, mock_wandb_logger):
        """Test chart definition when W&B not initialized."""
        mock_wandb_logger.is_initialized = False
        logger = TournamentWandbLogger(mock_wandb_logger)

        logger.define_tournament_charts()

        # Should not call define_metric
        mock_wandb_logger.define_metric.assert_not_called()

    def test_log_match_result(self, tournament_logger, mock_wandb_logger, sample_match):
        """Test logging individual match result."""
        tournament_logger.log_match_result(sample_match, tournament_id="season_1")

        # Should log data
        mock_wandb_logger.log.assert_called_once()

        # Check logged data
        logged_data = mock_wandb_logger.log.call_args[0][0]
        assert "tournament/match/player1_rating_before" in logged_data
        assert logged_data["tournament/match/player1_rating_before"] == 1500.0
        assert "tournament/match/is_upset" in logged_data
        assert logged_data["tournament/match/is_upset"] == 1  # Lower rated player won

    def test_log_match_result_not_initialized(self, mock_wandb_logger, sample_match):
        """Test match logging when W&B not initialized."""
        mock_wandb_logger.is_initialized = False
        logger = TournamentWandbLogger(mock_wandb_logger)

        logger.log_match_result(sample_match)

        # Should not log
        mock_wandb_logger.log.assert_not_called()

    def test_log_match_result_draw(self, tournament_logger, mock_wandb_logger):
        """Test logging a draw match."""
        draw_match = MatchResult(
            match_id="match_002",
            player1_id="player_a",
            player2_id="player_b",
            player1_score=0.5,
            player2_score=0.5,
            player1_rating_before=1500.0,
            player2_rating_before=1500.0,
            player1_rating_change=0.0,
            player2_rating_change=0.0,
            winner_id=None,
            duration_seconds=250.0,
        )

        tournament_logger.log_match_result(draw_match)

        logged_data = mock_wandb_logger.log.call_args[0][0]
        assert logged_data["tournament/match/is_draw"] == 1
        assert logged_data["tournament/match/is_upset"] == 0

    def test_log_round_results(self, tournament_logger, mock_wandb_logger, sample_match):
        """Test logging round summary."""
        # Create multiple matches
        matches = [sample_match for _ in range(5)]

        tournament_logger.log_round_results(round_num=3, results=matches)

        # Should log data
        mock_wandb_logger.log.assert_called_once()

        logged_data = mock_wandb_logger.log.call_args[0][0]
        assert logged_data["tournament/round"] == 3
        assert logged_data["tournament/match_count"] == 5
        assert logged_data["tournament/avg_game_length"] == 300.0
        assert logged_data["tournament/upset_rate"] == 1.0  # All were upsets

    def test_log_round_results_empty(self, tournament_logger, mock_wandb_logger):
        """Test logging empty round."""
        tournament_logger.log_round_results(round_num=1, results=[])

        # Should not log
        mock_wandb_logger.log.assert_not_called()

    def test_log_tournament_summary(self, tournament_logger, mock_wandb_logger):
        """Test logging tournament summary with TournamentResult-style object."""
        from alpha_zero.tournament.orchestrator import PlayerStanding

        # Create mock tournament result using TournamentResult field names
        result = MagicMock(
            spec=["winner", "total_games", "total_rounds", "duration_seconds", "final_standings"]
        )
        result.winner = "champion"
        result.total_games = 100
        result.total_rounds = 5
        result.duration_seconds = 3600.0
        result.final_standings = [
            PlayerStanding("p1", rank=1, wins=5, losses=0, draws=0, elo_rating=1800.0, points=5.0),
            PlayerStanding("p2", rank=2, wins=3, losses=2, draws=0, elo_rating=1700.0, points=3.0),
            PlayerStanding("p3", rank=3, wins=1, losses=4, draws=0, elo_rating=1600.0, points=1.0),
        ]

        tournament_logger.log_tournament_summary(result)

        # Should log data
        mock_wandb_logger.log.assert_called_once()

        logged_data = mock_wandb_logger.log.call_args[0][0]
        assert logged_data["tournament/summary/winner"] == "champion"
        assert logged_data["tournament/summary/total_games"] == 100
        assert logged_data["tournament/summary/first_place_rating"] == 1800.0

    def test_log_elo_leaderboard(self, tournament_logger, mock_wandb_logger):
        """Test logging ELO leaderboard."""
        leaderboard = RatingLeaderboard()
        leaderboard.update("player_a", 1800.0)
        leaderboard.update("player_b", 1700.0)
        leaderboard.update("player_c", 1600.0)

        tournament_logger.log_elo_leaderboard(leaderboard)

        # Should log table and summary
        mock_wandb_logger.log_table.assert_called_once()
        mock_wandb_logger.log.assert_called_once()

        # Check table
        table_call = mock_wandb_logger.log_table.call_args
        assert table_call[1]["name"] == "tournament/leaderboard"
        assert table_call[1]["columns"] == ["Rank", "Player ID", "Rating"]
        assert len(table_call[1]["data"]) == 3

        # Check summary data
        logged_data = mock_wandb_logger.log.call_args[0][0]
        assert logged_data["tournament/elo_spread"] == 200.0  # 1800 - 1600
        assert logged_data["tournament/elo_max"] == 1800.0

    def test_log_elo_leaderboard_empty(self, tournament_logger, mock_wandb_logger):
        """Test logging empty leaderboard."""
        leaderboard = RatingLeaderboard()

        tournament_logger.log_elo_leaderboard(leaderboard)

        # Should not log
        mock_wandb_logger.log_table.assert_not_called()
        mock_wandb_logger.log.assert_not_called()

    def test_log_head_to_head_matrix(self, tournament_logger, mock_wandb_logger):
        """Test logging head-to-head matrix."""
        # Create 3x3 win rate matrix
        matrix = np.array(
            [
                [0.5, 0.6, 0.7],
                [0.4, 0.5, 0.6],
                [0.3, 0.4, 0.5],
            ]
        )
        players = ["player_a", "player_b", "player_c"]

        tournament_logger.log_head_to_head_matrix(matrix, players)

        # Should log table and summary
        mock_wandb_logger.log_table.assert_called_once()
        mock_wandb_logger.log.assert_called_once()

        # Check table
        table_call = mock_wandb_logger.log_table.call_args
        assert table_call[1]["name"] == "tournament/head_to_head_matrix"
        assert table_call[1]["columns"] == ["Player", "player_a", "player_b", "player_c"]
        assert len(table_call[1]["data"]) == 3

    def test_log_head_to_head_matrix_mismatch(self, tournament_logger, mock_wandb_logger):
        """Test matrix size mismatch handling."""
        matrix = np.array([[0.5, 0.6], [0.4, 0.5]])
        players = ["player_a", "player_b", "player_c"]  # 3 players but 2x2 matrix

        tournament_logger.log_head_to_head_matrix(matrix, players)

        # Should not log due to mismatch
        mock_wandb_logger.log_table.assert_not_called()
        mock_wandb_logger.log.assert_not_called()

    def test_log_elo_history(self, tournament_logger, mock_wandb_logger):
        """Test logging ELO history."""
        # Create mock trends
        trends = {
            "player_a": MagicMock(
                rating_history=[
                    ("2026-01-01T00:00:00", 1500.0),
                    ("2026-01-02T00:00:00", 1520.0),
                    ("2026-01-03T00:00:00", 1550.0),
                ]
            ),
            "player_b": MagicMock(
                rating_history=[
                    ("2026-01-01T00:00:00", 1600.0),
                    ("2026-01-02T00:00:00", 1580.0),
                ]
            ),
        }

        tournament_logger.log_elo_history(trends)

        # Should log table
        mock_wandb_logger.log_table.assert_called_once()

        table_call = mock_wandb_logger.log_table.call_args
        assert table_call[1]["name"] == "tournament/elo_history"
        assert table_call[1]["columns"] == ["Player ID", "Timestamp", "Rating"]
        assert len(table_call[1]["data"]) == 5  # 3 + 2 history entries

    def test_log_elo_history_dict_format(self, tournament_logger, mock_wandb_logger):
        """Test logging ELO history with dict format."""
        # Dict format instead of object
        trends = {
            "player_a": {
                "rating_history": [
                    ("2026-01-01T00:00:00", 1500.0),
                    ("2026-01-02T00:00:00", 1520.0),
                ]
            }
        }

        tournament_logger.log_elo_history(trends)

        # Should log table
        mock_wandb_logger.log_table.assert_called_once()

        table_call = mock_wandb_logger.log_table.call_args
        assert len(table_call[1]["data"]) == 2

    def test_log_elo_history_empty(self, tournament_logger, mock_wandb_logger):
        """Test logging empty ELO history."""
        tournament_logger.log_elo_history({})

        # Should not log
        mock_wandb_logger.log_table.assert_not_called()


class TestIntegrationWithWandbLogger:
    """Integration tests with real WandbLogger (disabled mode)."""

    def test_graceful_degradation(self):
        """Test that all methods work when W&B is disabled."""
        # Create disabled WandbLogger
        from alpha_zero.utils.wandb_logger import WandbConfig

        wandb_logger = WandbLogger(WandbConfig(enabled=False))
        tournament_logger = TournamentWandbLogger(wandb_logger)

        # All methods should work without errors
        tournament_logger.define_tournament_charts()

        match = MatchResult(
            match_id="test",
            player1_id="a",
            player2_id="b",
            player1_score=1.0,
            player2_score=0.0,
        )

        tournament_logger.log_match_result(match)
        tournament_logger.log_round_results(1, [match])
        tournament_logger.log_tournament_summary(MagicMock())
        tournament_logger.log_elo_leaderboard(RatingLeaderboard())
        tournament_logger.log_head_to_head_matrix(np.zeros((2, 2)), ["a", "b"])
        tournament_logger.log_elo_history({})

        # No errors should occur
