"""Tests for ``alpha_zero.monitoring.tournament_dashboards``."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from alpha_zero.monitoring.config import MLflowConfig
from alpha_zero.monitoring.mlflow_logger import MLflowLogger
from alpha_zero.monitoring.tournament_dashboards import TournamentMlflowLogger


class _FakeLeaderboard:
    def __init__(self, ratings):
        self._ratings = ratings

    def get_top(self, n=10):
        items = sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)
        return [(pid, rating, i + 1) for i, (pid, rating) in enumerate(items[:n])]


@pytest.fixture
def initialized_logger():
    """Return an MLflowLogger whose underlying mlflow calls are mocked."""
    base = MLflowLogger(MLflowConfig(enabled=True))
    base._initialized = True  # bypass init() to skip the mlflow handshake
    base._parent_run_id = "fake-id"
    base._client = MagicMock()
    # Replace public methods so the tournament layer can be tested in isolation.
    base.log_metrics = MagicMock()  # type: ignore[method-assign]
    base.log_text = MagicMock()  # type: ignore[method-assign]
    return base


def test_log_match_result_when_initialized(initialized_logger):
    tlog = TournamentMlflowLogger(initialized_logger)
    tlog.log_match_result(
        round_index=2,
        match_index=5,
        player1_id="alice",
        player2_id="bob",
        score1=1.0,
        score2=0.0,
        game_length=42,
    )
    args, _ = initialized_logger.log_metrics.call_args
    payload = args[0]
    assert "tournament/round_002/match_005/score_alice" in payload
    assert "tournament/round_002/match_005/game_length" in payload


def test_log_match_result_no_op_when_uninitialized():
    base = MLflowLogger(MLflowConfig(enabled=False))
    base.log_metrics = MagicMock()  # type: ignore[method-assign]
    tlog = TournamentMlflowLogger(base)
    tlog.log_match_result(0, 0, "a", "b", 1.0, 0.0)
    base.log_metrics.assert_not_called()


def test_log_round_results(initialized_logger):
    tlog = TournamentMlflowLogger(initialized_logger)
    tlog.log_round_results(
        round_index=3,
        match_count=10,
        avg_game_length=50.5,
        upset_rate=0.25,
    )
    args, _ = initialized_logger.log_metrics.call_args
    payload = args[0]
    assert payload["tournament/round"] == 3.0
    assert payload["tournament/match_count"] == 10.0
    assert payload["tournament/avg_game_length"] == 50.5
    assert payload["tournament/upset_rate"] == 0.25


def test_log_elo_leaderboard(initialized_logger):
    tlog = TournamentMlflowLogger(initialized_logger)
    leaderboard = _FakeLeaderboard({"alice": 1700.0, "bob": 1500.0, "carol": 1400.0})
    tlog.log_elo_leaderboard(leaderboard, round_index=4, top_n=3)
    args, _ = initialized_logger.log_metrics.call_args
    payload = args[0]
    assert payload["elo/spread"] == 300.0
    assert "elo/leaderboard/alice/rating" in payload
    assert "elo/leaderboard/alice/rank" in payload
    assert payload["elo/leaderboard/alice/rank"] == 1.0


def test_log_elo_history(initialized_logger):
    update = MagicMock()
    update.player_id = "current"
    update.rating_before = 1500.0
    update.rating_after = 1520.0
    update.rating_change = 20.0
    tlog = TournamentMlflowLogger(initialized_logger)
    tlog.log_elo_history([update], round_index=7)
    args, _ = initialized_logger.log_metrics.call_args
    payload = args[0]
    assert payload["elo/history/current/rating_before"] == 1500.0
    assert payload["elo/history/current/rating_after"] == 1520.0
    assert payload["elo/history/current/rating_change"] == 20.0


def test_log_head_to_head_matrix(initialized_logger):
    tlog = TournamentMlflowLogger(initialized_logger)
    matrix = {"alice": {"bob": 5, "carol": 3}, "bob": {"alice": 1, "carol": 2}}
    tlog.log_head_to_head_matrix(matrix, round_index=1)
    initialized_logger.log_text.assert_called_once()
    args, _ = initialized_logger.log_text.call_args
    assert "alice" in args[0]
    assert args[1] == "head_to_head.json"


def test_log_head_to_head_matrix_swallows(initialized_logger):
    initialized_logger.log_text = MagicMock(side_effect=RuntimeError("boom"))
    tlog = TournamentMlflowLogger(initialized_logger)
    # Must not raise.
    tlog.log_head_to_head_matrix({"a": {"b": 1}}, round_index=0)


def test_no_op_methods_when_uninitialized():
    base = MLflowLogger(MLflowConfig(enabled=False))
    base.log_metrics = MagicMock()  # type: ignore[method-assign]
    base.log_text = MagicMock()  # type: ignore[method-assign]
    tlog = TournamentMlflowLogger(base)
    tlog.log_round_results(0, 0, 0.0)
    tlog.log_elo_leaderboard(_FakeLeaderboard({}), 0)
    tlog.log_elo_history([], 0)
    tlog.log_head_to_head_matrix({}, 0)
    base.log_metrics.assert_not_called()
    base.log_text.assert_not_called()
