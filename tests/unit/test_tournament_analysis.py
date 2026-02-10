"""Tests for tournament analysis module.

Tests for alpha_zero.tournament.analysis:
- PlayerTrend dataclass
- MetaAnalysis dataclass
- GenerationComparison dataclass
- TournamentAnalyzer initialization
- Head-to-head matrix generation
- Player trend analysis
- Meta-analysis of top players
- Upset frequency calculation
- Generation comparison
- Report export
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from alpha_zero.tournament.analysis import (
    GenerationComparison,
    MetaAnalysis,
    PlayerTrend,
    TournamentAnalyzer,
)


class TestPlayerTrend:
    """Tests for PlayerTrend dataclass."""

    def test_initialization(self):
        """Test PlayerTrend initialization with defaults."""
        trend = PlayerTrend(player_id="player_1")

        assert trend.player_id == "player_1"
        assert trend.elo_history == []
        assert trend.win_rate_history == []
        assert trend.games_played == 0
        assert trend.peak_elo == 0.0
        assert trend.current_streak == 0

    def test_initialization_with_values(self):
        """Test PlayerTrend initialization with values."""
        trend = PlayerTrend(
            player_id="player_1",
            elo_history=[1500.0, 1520.0, 1550.0],
            win_rate_history=[0.5, 0.55, 0.6],
            games_played=30,
            peak_elo=1600.0,
            current_streak=5,
        )

        assert trend.player_id == "player_1"
        assert len(trend.elo_history) == 3
        assert len(trend.win_rate_history) == 3
        assert trend.games_played == 30
        assert trend.peak_elo == 1600.0
        assert trend.current_streak == 5

    def test_to_dict(self):
        """Test PlayerTrend.to_dict conversion."""
        trend = PlayerTrend(
            player_id="player_1",
            elo_history=[1500.0, 1520.0],
            win_rate_history=[0.5, 0.6],
            games_played=20,
            peak_elo=1550.0,
            current_streak=3,
        )

        data = trend.to_dict()

        assert data["player_id"] == "player_1"
        assert data["elo_history"] == [1500.0, 1520.0]
        assert data["win_rate_history"] == [0.5, 0.6]
        assert data["games_played"] == 20
        assert data["peak_elo"] == 1550.0
        assert data["current_streak"] == 3


class TestMetaAnalysis:
    """Tests for MetaAnalysis dataclass."""

    def test_initialization_defaults(self):
        """Test MetaAnalysis initialization with defaults."""
        meta = MetaAnalysis()

        assert meta.top_hyperparameters == {}
        assert meta.strongest_strategies == []
        assert meta.diversity_index == 0.0

    def test_initialization_with_values(self):
        """Test MetaAnalysis initialization with values."""
        meta = MetaAnalysis(
            top_hyperparameters={"learning_rate": {"mean": 0.001, "std": 0.0001}},
            strongest_strategies=["aggressive", "defensive"],
            diversity_index=0.75,
        )

        assert "learning_rate" in meta.top_hyperparameters
        assert len(meta.strongest_strategies) == 2
        assert meta.diversity_index == 0.75

    def test_to_dict(self):
        """Test MetaAnalysis.to_dict conversion."""
        meta = MetaAnalysis(
            top_hyperparameters={"lr": {"mean": 0.001, "std": 0.0001}},
            strongest_strategies=["strategy_a"],
            diversity_index=0.5,
        )

        data = meta.to_dict()

        assert data["top_hyperparameters"]["lr"]["mean"] == 0.001
        assert data["strongest_strategies"] == ["strategy_a"]
        assert data["diversity_index"] == 0.5


class TestGenerationComparison:
    """Tests for GenerationComparison dataclass."""

    def test_initialization_defaults(self):
        """Test GenerationComparison initialization with defaults."""
        comparison = GenerationComparison(gen_a=0, gen_b=1)

        assert comparison.gen_a == 0
        assert comparison.gen_b == 1
        assert comparison.gen_a_win_rate == 0.0
        assert comparison.gen_b_win_rate == 0.0
        assert comparison.elo_difference == 0.0
        assert comparison.significant is False
        assert comparison.sample_size == 0

    def test_initialization_with_values(self):
        """Test GenerationComparison initialization with values."""
        comparison = GenerationComparison(
            gen_a=0,
            gen_b=1,
            gen_a_win_rate=0.4,
            gen_b_win_rate=0.6,
            elo_difference=50.0,
            significant=True,
            sample_size=100,
        )

        assert comparison.gen_a == 0
        assert comparison.gen_b == 1
        assert comparison.gen_a_win_rate == 0.4
        assert comparison.gen_b_win_rate == 0.6
        assert comparison.elo_difference == 50.0
        assert comparison.significant is True
        assert comparison.sample_size == 100

    def test_to_dict(self):
        """Test GenerationComparison.to_dict conversion."""
        comparison = GenerationComparison(
            gen_a=0,
            gen_b=1,
            gen_a_win_rate=0.45,
            gen_b_win_rate=0.55,
            elo_difference=30.0,
            significant=True,
            sample_size=50,
        )

        data = comparison.to_dict()

        assert data["gen_a"] == 0
        assert data["gen_b"] == 1
        assert data["gen_a_win_rate"] == 0.45
        assert data["gen_b_win_rate"] == 0.55
        assert data["elo_difference"] == 30.0
        assert data["significant"] is True
        assert data["sample_size"] == 50


class TestTournamentAnalyzer:
    """Tests for TournamentAnalyzer class."""

    def test_initialization_defaults(self):
        """Test TournamentAnalyzer initialization with defaults."""
        analyzer = TournamentAnalyzer()

        assert analyzer.match_history is None
        assert analyzer.elo_system is not None

    def test_initialization_with_history(self):
        """Test TournamentAnalyzer initialization with match history."""
        mock_history = Mock()
        analyzer = TournamentAnalyzer(match_history=mock_history)

        assert analyzer.match_history is mock_history

    def test_head_to_head_matrix_shape(self):
        """Test head_to_head_matrix returns correct shape."""
        mock_history = Mock()
        mock_history.get_head_to_head.return_value = {
            "total_games": 0,
            "player1_win_rate": 0.5,
        }

        analyzer = TournamentAnalyzer(match_history=mock_history)
        players = ["p1", "p2", "p3"]

        matrix = analyzer.head_to_head_matrix(players)

        assert matrix.shape == (3, 3)

    def test_head_to_head_matrix_diagonal(self):
        """Test head_to_head_matrix diagonal values are 0.5."""
        mock_history = Mock()
        mock_history.get_head_to_head.return_value = {
            "total_games": 0,
            "player1_win_rate": 0.5,
        }

        analyzer = TournamentAnalyzer(match_history=mock_history)
        players = ["p1", "p2", "p3"]

        matrix = analyzer.head_to_head_matrix(players)

        # Diagonal should be 0.5 (player vs self)
        assert matrix[0, 0] == 0.5
        assert matrix[1, 1] == 0.5
        assert matrix[2, 2] == 0.5

    def test_head_to_head_matrix_symmetry(self):
        """Test head_to_head_matrix symmetry."""
        mock_history = Mock()

        def mock_h2h(p1, p2):
            if p1 == "p1" and p2 == "p2":
                return {"total_games": 10, "player1_win_rate": 0.6}
            elif p1 == "p2" and p2 == "p1":
                return {"total_games": 10, "player1_win_rate": 0.4}
            return {"total_games": 0, "player1_win_rate": 0.5}

        mock_history.get_head_to_head.side_effect = mock_h2h

        analyzer = TournamentAnalyzer(match_history=mock_history)
        players = ["p1", "p2"]

        matrix = analyzer.head_to_head_matrix(players)

        # matrix[0,1] + matrix[1,0] should equal 1.0 (complementary win rates)
        assert abs((matrix[0, 1] + matrix[1, 0]) - 1.0) < 0.01

    def test_head_to_head_matrix_no_history(self):
        """Test head_to_head_matrix without match history."""
        analyzer = TournamentAnalyzer()
        players = ["p1", "p2", "p3"]

        matrix = analyzer.head_to_head_matrix(players)

        # Should return default matrix with 0.5 values
        assert matrix.shape == (3, 3)
        assert np.all(matrix == 0.5)

    def test_get_player_trends_returns_trend(self):
        """Test get_player_trends returns PlayerTrend."""
        analyzer = TournamentAnalyzer()

        trend = analyzer.get_player_trends("player_1")

        assert isinstance(trend, PlayerTrend)
        assert trend.player_id == "player_1"

    def test_get_player_trends_with_window(self):
        """Test get_player_trends with custom window parameter."""
        mock_history = Mock()
        mock_history.get_player_matches.return_value = []
        mock_history.get_player_statistics.return_value = None

        analyzer = TournamentAnalyzer(match_history=mock_history)

        trend = analyzer.get_player_trends("player_1", window=10)

        assert isinstance(trend, PlayerTrend)
        mock_history.get_player_matches.assert_called_once_with("player_1", limit=10000)

    def test_get_player_trends_no_matches(self):
        """Test get_player_trends with no match history."""
        mock_history = Mock()
        mock_history.get_player_matches.return_value = []
        mock_history.get_player_statistics.return_value = None

        analyzer = TournamentAnalyzer(match_history=mock_history)

        trend = analyzer.get_player_trends("player_1")

        assert trend.games_played == 0
        assert trend.elo_history == []

    def test_get_meta_analysis_with_top_k(self):
        """Test get_meta_analysis with top_k parameter."""
        mock_history = Mock()
        mock_history.get_leaderboard.return_value = [
            ("p1", 1600.0, 10, 0.6),
            ("p2", 1550.0, 12, 0.55),
        ]

        analyzer = TournamentAnalyzer(match_history=mock_history)

        meta = analyzer.get_meta_analysis(top_k=2)

        assert isinstance(meta, MetaAnalysis)
        mock_history.get_leaderboard.assert_called_once_with(limit=2, min_games=10)

    def test_get_meta_analysis_with_hyperparameters(self):
        """Test get_meta_analysis with hyperparameters."""
        mock_history = Mock()
        mock_history.get_leaderboard.return_value = [
            ("p1", 1600.0, 10, 0.6),
            ("p2", 1550.0, 12, 0.55),
        ]

        analyzer = TournamentAnalyzer(match_history=mock_history)

        hyperparams = {
            "p1": {"learning_rate": 0.001, "batch_size": 64},
            "p2": {"learning_rate": 0.002, "batch_size": 128},
        }

        meta = analyzer.get_meta_analysis(top_k=2, hyperparameters=hyperparams)

        assert isinstance(meta, MetaAnalysis)
        assert "learning_rate" in meta.top_hyperparameters
        assert "batch_size" in meta.top_hyperparameters

    def test_get_meta_analysis_no_history(self):
        """Test get_meta_analysis without match history."""
        analyzer = TournamentAnalyzer()

        meta = analyzer.get_meta_analysis(top_k=5)

        assert isinstance(meta, MetaAnalysis)
        assert meta.top_hyperparameters == {}
        assert meta.strongest_strategies == []

    def test_get_upset_frequency_calculation(self):
        """Test get_upset_frequency calculation."""
        mock_match = Mock()
        mock_match.player1_rating_before = 1400.0
        mock_match.player2_rating_before = 1600.0
        mock_match.outcome = "win"  # Underdog wins

        mock_history = Mock()
        mock_history.get_recent_matches.return_value = [mock_match] * 10

        analyzer = TournamentAnalyzer(match_history=mock_history)

        upset_rate = analyzer.get_upset_frequency(elo_diff_threshold=100.0)

        assert 0.0 <= upset_rate <= 1.0
        assert upset_rate == 1.0  # All matches were upsets

    def test_get_upset_frequency_no_matches(self):
        """Test get_upset_frequency with no matches."""
        mock_history = Mock()
        mock_history.get_recent_matches.return_value = []

        analyzer = TournamentAnalyzer(match_history=mock_history)

        upset_rate = analyzer.get_upset_frequency()

        assert upset_rate == 0.0

    def test_compare_generations_returns_comparison(self):
        """Test compare_generations returns GenerationComparison."""
        analyzer = TournamentAnalyzer()

        gen_a_players = ["gen_0_p1", "gen_0_p2"]
        gen_b_players = ["gen_1_p1", "gen_1_p2"]

        comparison = analyzer.compare_generations(gen_a_players, gen_b_players)

        assert isinstance(comparison, GenerationComparison)
        assert comparison.gen_a == 0
        assert comparison.gen_b == 1

    def test_compare_generations_with_history(self):
        """Test compare_generations with match history."""
        mock_history = Mock()
        mock_history.get_head_to_head.return_value = {
            "total_games": 10,
            "player1_wins": 4,
            "player2_wins": 6,
        }
        mock_history.get_player_statistics.return_value = Mock(current_rating=1500.0)

        analyzer = TournamentAnalyzer(match_history=mock_history)

        gen_a_players = ["gen_0_p1"]
        gen_b_players = ["gen_1_p1"]

        comparison = analyzer.compare_generations(gen_a_players, gen_b_players)

        assert comparison.sample_size == 10
        assert 0.0 <= comparison.gen_b_win_rate <= 1.0

    def test_export_report_json_format(self, tmp_path):
        """Test export_report with JSON format."""
        mock_history = Mock()
        mock_history.get_global_statistics.return_value = {"total_matches": 100}
        mock_history.get_leaderboard.return_value = []
        mock_history.get_recent_matches.return_value = []

        analyzer = TournamentAnalyzer(match_history=mock_history)

        output_path = tmp_path / "report.json"
        analyzer.export_report(output_path, format="json")

        assert output_path.exists()

    def test_export_report_markdown_format(self, tmp_path):
        """Test export_report with markdown format."""
        mock_history = Mock()
        mock_history.get_global_statistics.return_value = {
            "total_matches": 100,
            "total_players": 10,
            "avg_rating": 1500.0,
            "highest_rating_ever": 1800.0,
            "avg_game_length": 300.0,
        }
        mock_history.get_leaderboard.return_value = []
        mock_history.get_recent_matches.return_value = []

        analyzer = TournamentAnalyzer(match_history=mock_history)

        output_path = tmp_path / "report.md"
        analyzer.export_report(output_path, format="markdown")

        assert output_path.exists()
        content = output_path.read_text()
        assert "Tournament Analysis Report" in content

    def test_export_report_invalid_format(self, tmp_path):
        """Test export_report with invalid format raises error."""
        analyzer = TournamentAnalyzer()

        output_path = tmp_path / "report.txt"

        with pytest.raises(ValueError, match="Unsupported format"):
            analyzer.export_report(output_path, format="txt")
