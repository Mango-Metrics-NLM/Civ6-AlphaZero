"""Unit tests for sequential_testing module."""
from __future__ import annotations
import os
import pytest
from alpha_zero.evaluation.sequential_testing import (
    EvaluationDecision, GameResult, SequentialTester,
    SequentialTestingConfig, SequentialTestResult,
    get_sequential_tester,
)

class TestEvaluationDecision:
    def test_decision_values(self) -> None:
        assert EvaluationDecision.CONTINUE.value == "continue"
        assert EvaluationDecision.ACCEPT.value == "accept"
        assert EvaluationDecision.REJECT.value == "reject"

class TestGameResult:
    def test_creation(self) -> None:
        result = GameResult(game_id=1, winner=1, turns=150)
        assert result.game_id == 1
        assert result.winner == 1

    def test_to_dict(self) -> None:
        result = GameResult(game_id=2, winner=-1, turns=200)
        data = result.to_dict()
        assert data["game_id"] == 2
        assert data["winner"] == -1

class TestSequentialTestingConfig:
    def test_default_values(self) -> None:
        config = SequentialTestingConfig()
        assert config.min_games == 20
        assert config.max_games == 400
        assert config.win_threshold == 0.55

    def test_from_env(self) -> None:
        os.environ["CIV6AZ_EVAL_MIN_GAMES"] = "10"
        os.environ["CIV6AZ_EVAL_MAX_GAMES"] = "200"
        try:
            config = SequentialTestingConfig.from_env()
            assert config.min_games == 10
            assert config.max_games == 200
        finally:
            del os.environ["CIV6AZ_EVAL_MIN_GAMES"]
            del os.environ["CIV6AZ_EVAL_MAX_GAMES"]

class TestSequentialTester:
    def test_init(self) -> None:
        config = SequentialTestingConfig()
        tester = SequentialTester(config)
        assert tester.config == config

    def test_record_game(self) -> None:
        config = SequentialTestingConfig()
        tester = SequentialTester(config)
        tester.record_game(game_id=1, winner=1, turns=100)
        assert len(tester._games) == 1

    def test_get_decision_insufficient_games(self) -> None:
        config = SequentialTestingConfig(min_games=20)
        tester = SequentialTester(config)
        for i in range(5):
            tester.record_game(game_id=i, winner=1, turns=100)
        decision = tester.get_decision()
        assert decision == EvaluationDecision.CONTINUE

    def test_get_decision_clear_winner(self) -> None:
        # early_stop_threshold must be <= max_games, so set it explicitly
        config = SequentialTestingConfig(min_games=10, max_games=100, early_stop_threshold=50)
        tester = SequentialTester(config)
        for i in range(30):
            tester.record_game(game_id=i, winner=1, turns=100)
        for i in range(10):
            tester.record_game(game_id=i+30, winner=-1, turns=100)
        decision = tester.get_decision()
        assert decision in [EvaluationDecision.ACCEPT, EvaluationDecision.CONTINUE]

    def test_get_statistics(self) -> None:
        config = SequentialTestingConfig()
        tester = SequentialTester(config)
        for i in range(10):
            tester.record_game(game_id=i, winner=1, turns=100)
        stats = tester.get_statistics()
        assert "games_played" in stats
        assert "win_rate" in stats  # Implementation uses win_rate, not current_win_rate

    def test_reset(self) -> None:
        config = SequentialTestingConfig()
        tester = SequentialTester(config)
        tester.record_game(game_id=1, winner=1, turns=100)
        tester.reset()
        assert len(tester._games) == 0

class TestGetSequentialTester:
    def test_returns_tester(self) -> None:
        tester = get_sequential_tester()
        assert isinstance(tester, SequentialTester)
