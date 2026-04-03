"""Comprehensive tests for alpha_zero.tournament.analysis module.

Focuses on lines not covered by existing test_tournament_analysis.py:
- get_player_trends with actual match data (elo/win-rate history building,
  player2 outcome flipping, rolling window, streak from stats)
- get_meta_analysis with strategy/style string hyperparameters, diversity index
- get_upset_frequency with non-qualifying games, player2 underdog upsets
- compare_generations with ELO stats, significance test, bad player ID format
- export_report JSON with leaderboard entries
- export_report markdown with leaderboard entries
"""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from alpha_zero.tournament.analysis import (
    TournamentAnalyzer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_history():
    return Mock()


@pytest.fixture()
def analyzer(mock_history):
    return TournamentAnalyzer(match_history=mock_history)


def _make_match(
    player1_id="alice",
    player2_id="bob",
    outcome="win",
    p1_rating_before=1500.0,
    p2_rating_before=1500.0,
    p1_rating_after=1516.0,
    p2_rating_after=1484.0,
):
    m = Mock()
    m.player1_id = player1_id
    m.player2_id = player2_id
    m.outcome = outcome
    m.player1_rating_before = p1_rating_before
    m.player2_rating_before = p2_rating_before
    m.player1_rating_after = p1_rating_after
    m.player2_rating_after = p2_rating_after
    return m


# ---------------------------------------------------------------------------
# get_player_trends — detailed coverage
# ---------------------------------------------------------------------------


class TestGetPlayerTrendsDetailed:
    def test_player1_win_builds_elo_history(self, analyzer, mock_history):
        match = _make_match(outcome="win", p1_rating_after=1520.0)
        mock_history.get_player_matches.return_value = [match]
        mock_history.get_player_statistics.return_value = Mock(current_streak=1)

        trend = analyzer.get_player_trends("alice")

        assert trend.elo_history == [1520.0]
        assert trend.games_played == 1
        assert trend.peak_elo == 1520.0
        assert trend.current_streak == 1

    def test_player2_outcome_flipped(self, analyzer, mock_history):
        """When querying player2's trends, outcomes should be flipped."""
        match = _make_match(
            player1_id="alice",
            player2_id="bob",
            outcome="win",  # alice won => bob lost
            p2_rating_after=1484.0,
        )
        mock_history.get_player_matches.return_value = [match]
        mock_history.get_player_statistics.return_value = Mock(current_streak=-1)

        trend = analyzer.get_player_trends("bob")

        assert trend.elo_history == [1484.0]
        # bob lost, so win_rate_history should reflect 0.0
        assert len(trend.win_rate_history) == 1
        assert trend.win_rate_history[0] == 0.0

    def test_player2_loss_outcome_flipped_to_win(self, analyzer, mock_history):
        """Player2 perspective: if outcome='loss', player2 actually won."""
        match = _make_match(
            player1_id="alice",
            player2_id="bob",
            outcome="loss",  # alice lost => bob won
            p2_rating_after=1520.0,
        )
        mock_history.get_player_matches.return_value = [match]
        mock_history.get_player_statistics.return_value = None

        trend = analyzer.get_player_trends("bob")

        assert trend.win_rate_history == [1.0]

    def test_player2_draw_stays_draw(self, analyzer, mock_history):
        match = _make_match(outcome="draw", p2_rating_after=1500.0)
        mock_history.get_player_matches.return_value = [match]
        mock_history.get_player_statistics.return_value = None

        trend = analyzer.get_player_trends("bob")

        assert trend.win_rate_history == [0.5]

    def test_rolling_window_calculation(self, analyzer, mock_history):
        """With enough matches, rolling window should produce multiple data points."""
        matches = []
        for i in range(25):
            outcome = "win" if i % 2 == 0 else "loss"
            m = _make_match(outcome=outcome, p1_rating_after=1500.0 + i)
            matches.append(m)

        # Reversed because get_player_trends reverses the list
        mock_history.get_player_matches.return_value = list(reversed(matches))
        mock_history.get_player_statistics.return_value = Mock(current_streak=0)

        trend = analyzer.get_player_trends("alice", window=20)

        assert len(trend.elo_history) == 25
        # rolling window of 20 on 25 items => 6 entries (indices 20..25)
        assert len(trend.win_rate_history) == 6

    def test_fewer_matches_than_window(self, analyzer, mock_history):
        """With fewer matches than window, single average is used."""
        matches = [_make_match(outcome="win", p1_rating_after=1520.0)]
        mock_history.get_player_matches.return_value = matches
        mock_history.get_player_statistics.return_value = None

        trend = analyzer.get_player_trends("alice", window=20)

        assert len(trend.win_rate_history) == 1
        assert trend.win_rate_history[0] == 1.0

    def test_stats_none_streak_zero(self, analyzer, mock_history):
        match = _make_match(outcome="win", p1_rating_after=1520.0)
        mock_history.get_player_matches.return_value = [match]
        mock_history.get_player_statistics.return_value = None

        trend = analyzer.get_player_trends("alice")
        assert trend.current_streak == 0


# ---------------------------------------------------------------------------
# get_meta_analysis — strategy strings and diversity
# ---------------------------------------------------------------------------


class TestGetMetaAnalysisDetailed:
    def test_strategy_string_hyperparams(self, analyzer, mock_history):
        mock_history.get_leaderboard.return_value = [
            ("p1", 1600.0, 20, 0.6),
            ("p2", 1550.0, 15, 0.55),
        ]
        hyperparams = {
            "p1": {"strategy": "aggressive", "lr": 0.001},
            "p2": {"strategy": "defensive", "lr": 0.002},
        }

        meta = analyzer.get_meta_analysis(top_k=2, hyperparameters=hyperparams)

        assert "aggressive" in meta.strongest_strategies or "defensive" in meta.strongest_strategies
        assert "lr" in meta.top_hyperparameters

    def test_style_string_hyperparams(self, analyzer, mock_history):
        mock_history.get_leaderboard.return_value = [("p1", 1600.0, 20, 0.6)]
        hyperparams = {"p1": {"style": "rush", "batch_size": 64}}

        meta = analyzer.get_meta_analysis(top_k=1, hyperparameters=hyperparams)

        assert "rush" in meta.strongest_strategies

    def test_diversity_index_calculation(self, analyzer, mock_history):
        mock_history.get_leaderboard.return_value = [
            ("p1", 1600.0, 20, 0.6),
            ("p2", 1550.0, 15, 0.55),
        ]
        hyperparams = {
            "p1": {"lr": 0.001, "layers": 6},
            "p2": {"lr": 0.003, "layers": 6},
        }

        meta = analyzer.get_meta_analysis(top_k=2, hyperparameters=hyperparams)

        assert meta.diversity_index >= 0.0
        assert meta.diversity_index <= 1.0

    def test_diversity_index_zero_mean(self, analyzer, mock_history):
        """When mean is 0, that key is skipped in CV calculation."""
        mock_history.get_leaderboard.return_value = [("p1", 1600.0, 20, 0.6)]
        hyperparams = {"p1": {"offset": 0, "lr": 0.001}}

        meta = analyzer.get_meta_analysis(top_k=1, hyperparameters=hyperparams)

        # Should not crash; diversity calculated from non-zero-mean params only
        assert isinstance(meta.diversity_index, float)

    def test_no_leaderboard(self, analyzer, mock_history):
        mock_history.get_leaderboard.return_value = []

        meta = analyzer.get_meta_analysis(top_k=5)

        assert meta.top_hyperparameters == {}

    def test_hyperparams_player_not_in_top(self, analyzer, mock_history):
        mock_history.get_leaderboard.return_value = [("p1", 1600.0, 20, 0.6)]
        # hyperparams only for p2 who is not on leaderboard
        hyperparams = {"p2": {"lr": 0.001}}

        meta = analyzer.get_meta_analysis(top_k=1, hyperparameters=hyperparams)

        assert meta.top_hyperparameters == {}

    def test_deduplication_of_strategies(self, analyzer, mock_history):
        mock_history.get_leaderboard.return_value = [
            ("p1", 1600.0, 20, 0.6),
            ("p2", 1550.0, 15, 0.55),
        ]
        hyperparams = {
            "p1": {"strategy": "rush"},
            "p2": {"strategy": "rush"},
        }

        meta = analyzer.get_meta_analysis(top_k=2, hyperparameters=hyperparams)

        assert meta.strongest_strategies.count("rush") == 1


# ---------------------------------------------------------------------------
# get_upset_frequency — detailed
# ---------------------------------------------------------------------------


class TestGetUpsetFrequencyDetailed:
    def test_no_qualifying_games(self, analyzer, mock_history):
        """Games with small ELO diff don't count."""
        match = _make_match(p1_rating_before=1500.0, p2_rating_before=1510.0, outcome="win")
        mock_history.get_recent_matches.return_value = [match]

        rate = analyzer.get_upset_frequency(elo_diff_threshold=100.0)
        assert rate == 0.0

    def test_player2_underdog_upset(self, analyzer, mock_history):
        """Player2 is underdog and wins (outcome='loss' means p1 lost)."""
        match = _make_match(
            p1_rating_before=1600.0,
            p2_rating_before=1400.0,
            outcome="loss",  # p1 lost, so p2 (underdog) won
        )
        mock_history.get_recent_matches.return_value = [match]

        rate = analyzer.get_upset_frequency(elo_diff_threshold=100.0)
        assert rate == 1.0

    def test_no_upset_favorite_wins(self, analyzer, mock_history):
        """Favorite wins — not an upset."""
        match = _make_match(
            p1_rating_before=1600.0,
            p2_rating_before=1400.0,
            outcome="win",  # p1 (favorite) wins
        )
        mock_history.get_recent_matches.return_value = [match]

        rate = analyzer.get_upset_frequency(elo_diff_threshold=100.0)
        assert rate == 0.0

    def test_mixed_results(self, analyzer, mock_history):
        m1 = _make_match(p1_rating_before=1400.0, p2_rating_before=1600.0, outcome="win")
        m2 = _make_match(p1_rating_before=1400.0, p2_rating_before=1600.0, outcome="loss")
        mock_history.get_recent_matches.return_value = [m1, m2]

        rate = analyzer.get_upset_frequency(elo_diff_threshold=100.0)
        assert abs(rate - 0.5) < 1e-9

    def test_no_history(self):
        analyzer = TournamentAnalyzer()
        assert analyzer.get_upset_frequency() == 0.0


# ---------------------------------------------------------------------------
# compare_generations — detailed
# ---------------------------------------------------------------------------


class TestCompareGenerationsDetailed:
    def test_no_history(self):
        analyzer = TournamentAnalyzer()
        cmp = analyzer.compare_generations(["gen_0_p1"], ["gen_1_p1"])
        assert cmp.sample_size == 0

    def test_elo_difference(self, analyzer, mock_history):
        mock_history.get_head_to_head.return_value = {
            "total_games": 20,
            "player1_wins": 8,
            "player2_wins": 12,
        }
        mock_history.get_player_statistics.side_effect = lambda pid: (
            Mock(current_rating=1400.0) if "0" in pid else Mock(current_rating=1600.0)
        )

        cmp = analyzer.compare_generations(["gen_0_p1"], ["gen_1_p1"])

        assert cmp.elo_difference == pytest.approx(200.0)

    def test_significance_large_sample(self, analyzer, mock_history):
        mock_history.get_head_to_head.return_value = {
            "total_games": 100,
            "player1_wins": 30,
            "player2_wins": 70,
        }
        mock_history.get_player_statistics.return_value = Mock(current_rating=1500.0)

        cmp = analyzer.compare_generations(["gen_0_p1"], ["gen_1_p1"])

        assert cmp.sample_size == 100
        assert cmp.significant == True  # noqa: E712 — np.bool_ vs bool

    def test_significance_small_sample(self, analyzer, mock_history):
        mock_history.get_head_to_head.return_value = {
            "total_games": 5,
            "player1_wins": 2,
            "player2_wins": 3,
        }
        mock_history.get_player_statistics.return_value = Mock(current_rating=1500.0)

        cmp = analyzer.compare_generations(["gen_0_p1"], ["gen_1_p1"])

        assert not cmp.significant

    def test_bad_player_id_format(self, analyzer, mock_history):
        """Player IDs that don't match gen_N_xxx pattern use defaults."""
        mock_history.get_head_to_head.return_value = {"total_games": 0}
        mock_history.get_player_statistics.return_value = None

        cmp = analyzer.compare_generations(["randomname"], ["othername"])

        assert cmp.gen_a == 0
        assert cmp.gen_b == 1

    def test_no_stats_for_players(self, analyzer, mock_history):
        mock_history.get_head_to_head.return_value = {
            "total_games": 10,
            "player1_wins": 5,
            "player2_wins": 5,
        }
        mock_history.get_player_statistics.return_value = None

        cmp = analyzer.compare_generations(["gen_0_p1"], ["gen_1_p1"])

        assert cmp.elo_difference == 0.0

    def test_significance_even_split(self, analyzer, mock_history):
        """Exactly 50/50 should not be significant."""
        mock_history.get_head_to_head.return_value = {
            "total_games": 100,
            "player1_wins": 50,
            "player2_wins": 50,
        }
        mock_history.get_player_statistics.return_value = Mock(current_rating=1500.0)

        cmp = analyzer.compare_generations(["gen_0_p1"], ["gen_1_p1"])

        assert not cmp.significant

    def test_multiple_players_per_generation(self, analyzer, mock_history):
        """Cross-generation matches across multiple players."""

        def h2h_side_effect(pa, pb):
            return {"total_games": 5, "player1_wins": 2, "player2_wins": 3}

        mock_history.get_head_to_head.side_effect = h2h_side_effect
        mock_history.get_player_statistics.return_value = Mock(current_rating=1500.0)

        cmp = analyzer.compare_generations(
            ["gen_0_p1", "gen_0_p2"],
            ["gen_1_p1", "gen_1_p2"],
        )

        # 4 matchups x 5 games each = 20 total
        assert cmp.sample_size == 20


# ---------------------------------------------------------------------------
# export_report — detailed coverage
# ---------------------------------------------------------------------------


class TestExportReportDetailed:
    def test_json_with_leaderboard(self, tmp_path, mock_history):
        mock_history.get_global_statistics.return_value = {
            "total_matches": 50,
            "total_players": 5,
        }
        mock_history.get_leaderboard.return_value = [
            ("p1", 1600.0, 20, 0.65),
        ]
        mock_history.get_player_matches.return_value = []
        mock_history.get_player_statistics.return_value = None
        mock_history.get_recent_matches.return_value = []

        analyzer = TournamentAnalyzer(match_history=mock_history)
        path = tmp_path / "report.json"
        analyzer.export_report(path, format="json")

        with open(path) as f:
            data = json.load(f)
        assert len(data["top_players"]) == 1
        assert data["top_players"][0]["player_id"] == "p1"
        assert data["top_players"][0]["rating"] == 1600.0

    def test_json_no_history(self, tmp_path):
        analyzer = TournamentAnalyzer()
        path = tmp_path / "report.json"
        analyzer.export_report(path, format="json")

        with open(path) as f:
            data = json.load(f)
        assert data["top_players"] == []

    def test_markdown_with_leaderboard(self, tmp_path, mock_history):
        mock_history.get_global_statistics.return_value = {
            "total_matches": 100,
            "total_players": 10,
            "avg_rating": 1500.0,
            "highest_rating_ever": 1800.0,
            "avg_game_length": 250.0,
        }
        mock_history.get_leaderboard.return_value = [
            ("p1", 1700.0, 30, 0.7),
            ("p2", 1600.0, 25, 0.6),
        ]
        mock_history.get_recent_matches.return_value = []

        analyzer = TournamentAnalyzer(match_history=mock_history)
        path = tmp_path / "report.md"
        analyzer.export_report(path, format="markdown")

        content = path.read_text()
        assert "| 1 | p1 |" in content
        assert "| 2 | p2 |" in content
        assert "Upset Frequency" in content

    def test_markdown_no_history(self, tmp_path):
        analyzer = TournamentAnalyzer()
        path = tmp_path / "report.md"
        analyzer.export_report(path, format="markdown")

        content = path.read_text()
        assert "Tournament Analysis Report" in content

    def test_invalid_format_raises(self, tmp_path):
        analyzer = TournamentAnalyzer()
        with pytest.raises(ValueError, match="Unsupported format"):
            analyzer.export_report(tmp_path / "report.csv", format="csv")

    def test_creates_parent_directory(self, tmp_path, mock_history):
        mock_history.get_global_statistics.return_value = {"total_matches": 0}
        mock_history.get_leaderboard.return_value = []
        mock_history.get_recent_matches.return_value = []

        analyzer = TournamentAnalyzer(match_history=mock_history)
        path = tmp_path / "sub" / "dir" / "report.json"
        analyzer.export_report(path, format="json")
        assert path.exists()


# ---------------------------------------------------------------------------
# head_to_head_matrix — with actual data
# ---------------------------------------------------------------------------


class TestHeadToHeadMatrixDetailed:
    def test_with_game_data(self, analyzer, mock_history):
        def h2h(p1, p2):
            if p1 == "a" and p2 == "b":
                return {"total_games": 10, "player1_win_rate": 0.7}
            if p1 == "b" and p2 == "a":
                return {"total_games": 10, "player1_win_rate": 0.3}
            return {"total_games": 0, "player1_win_rate": 0.5}

        mock_history.get_head_to_head.side_effect = h2h

        matrix = analyzer.head_to_head_matrix(["a", "b"])

        assert matrix[0, 0] == 0.5  # diagonal
        assert matrix[0, 1] == 0.7
        assert matrix[1, 0] == 0.3
