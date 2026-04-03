"""Comprehensive tests for alpha_zero.tournament.history module.

Covers:
- MatchRecord (from_match_result, to_dict, from_dict, from_row)
- PlayerStatistics (properties, to_dict)
- MatchHistory (record_match, get_match, get_player_matches,
  get_player_statistics, get_head_to_head, get_recent_matches,
  get_leaderboard, get_rating_history, get_global_statistics,
  delete_player, clear_all, export_to_json)
"""

from __future__ import annotations

import json

import pytest

from alpha_zero.tournament.config import MatchResult, TournamentConfig
from alpha_zero.tournament.history import (
    MatchHistory,
    MatchRecord,
    PlayerStatistics,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "test_tournament.db"


@pytest.fixture()
def history(db_path):
    return MatchHistory(database_path=db_path)


@pytest.fixture()
def win_result():
    return MatchResult(
        match_id="m1",
        player1_id="alice",
        player2_id="bob",
        player1_score=1.0,
        player2_score=0.0,
        player1_rating_before=1500.0,
        player2_rating_before=1500.0,
        winner_id="alice",
        duration_seconds=120.0,
        game_details=[{"turn": 50}],
    )


@pytest.fixture()
def draw_result():
    return MatchResult(
        match_id="m2",
        player1_id="alice",
        player2_id="bob",
        player1_score=0.5,
        player2_score=0.5,
        player1_rating_before=1510.0,
        player2_rating_before=1490.0,
        winner_id=None,
    )


@pytest.fixture()
def loss_result():
    """Player1 loses."""
    return MatchResult(
        match_id="m3",
        player1_id="alice",
        player2_id="bob",
        player1_score=0.0,
        player2_score=1.0,
        player1_rating_before=1500.0,
        player2_rating_before=1500.0,
        winner_id="bob",
    )


# ---------------------------------------------------------------------------
# MatchRecord
# ---------------------------------------------------------------------------


class TestMatchRecord:
    def test_from_match_result_win(self, win_result):
        record = MatchRecord.from_match_result(win_result, 1516.0, 1484.0)
        assert record.match_id == "m1"
        assert record.outcome == "win"
        assert record.player1_rating_after == 1516.0
        assert record.player2_rating_after == 1484.0
        assert record.game_length == 120
        assert "game_details" in record.metadata

    def test_from_match_result_draw(self, draw_result):
        record = MatchRecord.from_match_result(draw_result, 1510.0, 1490.0)
        assert record.outcome == "draw"
        assert record.metadata == {}

    def test_from_match_result_loss(self, loss_result):
        record = MatchRecord.from_match_result(loss_result, 1484.0, 1516.0)
        assert record.outcome == "loss"

    def test_to_dict(self, win_result):
        record = MatchRecord.from_match_result(win_result, 1516.0, 1484.0)
        d = record.to_dict()
        assert d["match_id"] == "m1"
        assert d["outcome"] == "win"

    def test_from_dict_roundtrip(self, win_result):
        record = MatchRecord.from_match_result(win_result, 1516.0, 1484.0)
        d = record.to_dict()
        restored = MatchRecord.from_dict(d)
        assert restored.match_id == record.match_id
        assert restored.outcome == record.outcome

    def test_from_row_full(self):
        row = (
            "m1",
            "alice",
            "bob",
            1500.0,
            1500.0,
            1516.0,
            1484.0,
            "win",
            "2026-01-01T00:00:00",
            50,
            '{"key": "value"}',
        )
        record = MatchRecord.from_row(row)
        assert record.match_id == "m1"
        assert record.game_length == 50
        assert record.metadata == {"key": "value"}

    def test_from_row_minimal(self):
        row = (
            "m2",
            "alice",
            "bob",
            1500.0,
            1500.0,
            1510.0,
            1490.0,
            "draw",
            "2026-01-01T00:00:00",
        )
        record = MatchRecord.from_row(row)
        assert record.game_length == 0
        assert record.metadata == {}

    def test_from_row_none_metadata(self):
        row = (
            "m3",
            "alice",
            "bob",
            1500.0,
            1500.0,
            1510.0,
            1490.0,
            "win",
            "2026-01-01T00:00:00",
            30,
            None,
        )
        record = MatchRecord.from_row(row)
        assert record.metadata == {}


# ---------------------------------------------------------------------------
# PlayerStatistics
# ---------------------------------------------------------------------------


class TestPlayerStatistics:
    def test_win_rate_zero_games(self):
        ps = PlayerStatistics(player_id="x")
        assert ps.win_rate == 0.0

    def test_win_rate_nonzero(self):
        ps = PlayerStatistics(player_id="x", total_games=10, wins=7)
        assert abs(ps.win_rate - 0.7) < 1e-9

    def test_rating_volatility(self):
        ps = PlayerStatistics(player_id="x", peak_rating=1800.0, lowest_rating=1200.0)
        assert ps.rating_volatility == 600.0

    def test_to_dict(self):
        ps = PlayerStatistics(player_id="x", total_games=5, wins=3, losses=1, draws=1)
        d = ps.to_dict()
        assert d["player_id"] == "x"
        assert d["total_games"] == 5
        assert "win_rate" in d


# ---------------------------------------------------------------------------
# MatchHistory
# ---------------------------------------------------------------------------


class TestMatchHistoryInit:
    def test_creates_database(self, db_path):
        MatchHistory(database_path=db_path)
        assert db_path.exists()

    def test_with_config(self, db_path):
        config = TournamentConfig(games_per_match=4)
        h = MatchHistory(database_path=db_path, config=config)
        assert h.config.games_per_match == 4


class TestMatchHistoryRecordAndGet:
    def test_record_match_win(self, history, win_result):
        record = history.record_match(win_result, 1516.0, 1484.0)
        assert record.match_id == "m1"
        assert record.outcome == "win"

    def test_get_match_found(self, history, win_result):
        history.record_match(win_result, 1516.0, 1484.0)
        match = history.get_match("m1")
        assert match is not None
        assert match.match_id == "m1"

    def test_get_match_not_found(self, history):
        assert history.get_match("nonexistent") is None

    def test_record_and_get_draw(self, history, draw_result):
        history.record_match(draw_result, 1510.0, 1490.0)
        match = history.get_match("m2")
        assert match is not None
        assert match.outcome == "draw"

    def test_record_and_get_loss(self, history, loss_result):
        history.record_match(loss_result, 1484.0, 1516.0)
        match = history.get_match("m3")
        assert match is not None
        assert match.outcome == "loss"


class TestMatchHistoryPlayerMatches:
    def test_get_player_matches(self, history, win_result, draw_result):
        history.record_match(win_result, 1516.0, 1484.0)
        history.record_match(draw_result, 1510.0, 1490.0)
        matches = history.get_player_matches("alice")
        assert len(matches) == 2

    def test_get_player_matches_limit(self, history, win_result):
        history.record_match(win_result, 1516.0, 1484.0)
        matches = history.get_player_matches("alice", limit=1)
        assert len(matches) == 1

    def test_get_player_matches_with_opponent(self, history, win_result):
        history.record_match(win_result, 1516.0, 1484.0)
        matches = history.get_player_matches("alice", opponent_id="bob")
        assert len(matches) == 1

    def test_get_player_matches_opponent_not_found(self, history, win_result):
        history.record_match(win_result, 1516.0, 1484.0)
        matches = history.get_player_matches("alice", opponent_id="charlie")
        assert len(matches) == 0

    def test_get_player_matches_offset(self, history, win_result, draw_result):
        history.record_match(win_result, 1516.0, 1484.0)
        history.record_match(draw_result, 1510.0, 1490.0)
        matches = history.get_player_matches("alice", limit=1, offset=1)
        assert len(matches) == 1


class TestMatchHistoryPlayerStatistics:
    def test_get_player_statistics_exists(self, history, win_result):
        history.record_match(win_result, 1516.0, 1484.0)
        stats = history.get_player_statistics("alice")
        assert stats is not None
        assert stats.wins == 1
        assert stats.current_rating == 1516.0
        assert stats.peak_rating == 1516.0

    def test_get_player_statistics_not_found(self, history):
        assert history.get_player_statistics("nobody") is None

    def test_player_stats_updated_on_multiple_matches(self, history, win_result, loss_result):
        history.record_match(win_result, 1516.0, 1484.0)
        history.record_match(loss_result, 1500.0, 1500.0)
        stats = history.get_player_statistics("alice")
        assert stats is not None
        assert stats.total_games == 2
        assert stats.wins == 1
        assert stats.losses == 1

    def test_streak_tracking(self, history):
        # Two wins in a row
        r1 = MatchResult(
            match_id="s1",
            player1_id="p1",
            player2_id="p2",
            winner_id="p1",
            player1_rating_before=1500.0,
            player2_rating_before=1500.0,
        )
        r2 = MatchResult(
            match_id="s2",
            player1_id="p1",
            player2_id="p2",
            winner_id="p1",
            player1_rating_before=1516.0,
            player2_rating_before=1484.0,
        )
        history.record_match(r1, 1516.0, 1484.0)
        history.record_match(r2, 1530.0, 1470.0)
        stats = history.get_player_statistics("p1")
        assert stats is not None
        assert stats.current_streak >= 2
        assert stats.longest_win_streak >= 2


class TestMatchHistoryHeadToHead:
    def test_head_to_head_no_matches(self, history):
        h2h = history.get_head_to_head("alice", "bob")
        assert h2h["total_games"] == 0

    def test_head_to_head_with_matches(self, history, win_result, loss_result):
        history.record_match(win_result, 1516.0, 1484.0)
        history.record_match(loss_result, 1500.0, 1500.0)
        h2h = history.get_head_to_head("alice", "bob")
        assert h2h["total_games"] == 2
        assert h2h["player1_wins"] == 1
        assert h2h["player2_wins"] == 1

    def test_head_to_head_draw(self, history, draw_result):
        history.record_match(draw_result, 1510.0, 1490.0)
        h2h = history.get_head_to_head("alice", "bob")
        assert h2h["draws"] == 1

    def test_head_to_head_reversed_order(self, history, win_result):
        """h2h should work regardless of argument order."""
        history.record_match(win_result, 1516.0, 1484.0)
        h2h = history.get_head_to_head("bob", "alice")
        assert h2h["total_games"] == 1
        # From bob's perspective: alice won means bob lost
        assert h2h["player2_wins"] == 1


class TestMatchHistoryRecentMatches:
    def test_get_recent_matches_empty(self, history):
        assert history.get_recent_matches() == []

    def test_get_recent_matches(self, history, win_result):
        history.record_match(win_result, 1516.0, 1484.0)
        recent = history.get_recent_matches(limit=5)
        assert len(recent) == 1

    def test_get_recent_matches_with_tier_filter(self, history, win_result):
        history.record_match(win_result, 1516.0, 1484.0)
        # No matches have tier metadata, so filtering returns empty
        recent = history.get_recent_matches(limit=5, tier="Gold")
        assert len(recent) == 0


class TestMatchHistoryLeaderboard:
    def test_get_leaderboard_empty(self, history):
        assert history.get_leaderboard() == []

    def test_get_leaderboard_min_games(self, history):
        # Record enough games for a player
        for i in range(12):
            r = MatchResult(
                match_id=f"lb_{i}",
                player1_id="top_player",
                player2_id=f"opp_{i}",
                winner_id="top_player",
                player1_rating_before=1500.0 + i,
                player2_rating_before=1500.0,
            )
            history.record_match(r, 1516.0 + i, 1484.0)
        board = history.get_leaderboard(limit=10, min_games=10)
        assert len(board) >= 1
        assert board[0][0] == "top_player"


class TestMatchHistoryRatingHistory:
    def test_get_rating_history_empty(self, history):
        assert history.get_rating_history("nobody") == []

    def test_get_rating_history(self, history, win_result, draw_result):
        history.record_match(win_result, 1516.0, 1484.0)
        history.record_match(draw_result, 1510.0, 1490.0)
        rh = history.get_rating_history("alice")
        assert len(rh) == 2
        # Each entry is (timestamp, rating)
        assert rh[0][1] == 1516.0


class TestMatchHistoryGlobalStats:
    def test_get_global_statistics_empty(self, history):
        stats = history.get_global_statistics()
        assert stats["total_matches"] == 0
        assert stats["total_players"] == 0

    def test_get_global_statistics_with_data(self, history, win_result):
        history.record_match(win_result, 1516.0, 1484.0)
        stats = history.get_global_statistics()
        assert stats["total_matches"] == 1
        assert stats["total_players"] == 2
        assert stats["avg_rating"] > 0


class TestMatchHistoryDeletion:
    def test_delete_player(self, history, win_result):
        history.record_match(win_result, 1516.0, 1484.0)
        deleted = history.delete_player("alice")
        assert deleted is True
        assert history.get_player_statistics("alice") is None

    def test_delete_player_not_found(self, history):
        deleted = history.delete_player("ghost")
        assert deleted is False

    def test_clear_all(self, history, win_result):
        history.record_match(win_result, 1516.0, 1484.0)
        history.clear_all()
        stats = history.get_global_statistics()
        assert stats["total_matches"] == 0
        assert stats["total_players"] == 0


class TestMatchHistoryExport:
    def test_export_to_json(self, history, win_result, tmp_path):
        history.record_match(win_result, 1516.0, 1484.0)
        output = tmp_path / "export.json"
        history.export_to_json(output)
        assert output.exists()
        with open(output) as f:
            data = json.load(f)
        assert len(data["matches"]) == 1
        assert len(data["players"]) == 2
        assert "exported_at" in data

    def test_export_to_json_creates_parent(self, history, tmp_path):
        output = tmp_path / "sub" / "dir" / "export.json"
        history.export_to_json(output)
        assert output.exists()
