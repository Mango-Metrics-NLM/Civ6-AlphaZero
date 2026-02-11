"""Unit tests for model evaluator."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alpha_zero.evaluation.baselines import RandomPlayer
from alpha_zero.evaluation.config import BaselineConfig, BaselineType, EvaluationConfig
from alpha_zero.evaluation.evaluator import (
    EvaluationResult,
    GameResult,
    ModelEvaluator,
)


class TestGameResult:
    """Tests for GameResult dataclass."""

    def test_model_win(self):
        """Test game result for model win."""
        result = GameResult(
            winner=0,
            num_turns=50,
            model_score=1.0,
            opponent_score=0.0,
        )
        assert result.winner == 0
        assert result.model_score == 1.0
        assert result.opponent_score == 0.0

    def test_opponent_win(self):
        """Test game result for opponent win."""
        result = GameResult(
            winner=1,
            num_turns=30,
            model_score=0.0,
            opponent_score=1.0,
        )
        assert result.winner == 1
        assert result.model_score == 0.0

    def test_draw(self):
        """Test game result for draw."""
        result = GameResult(
            winner=None,
            num_turns=100,
            model_score=0.5,
            opponent_score=0.5,
        )
        assert result.winner is None
        assert result.model_score == 0.5

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = GameResult(
            winner=0,
            num_turns=50,
            model_score=1.0,
            opponent_score=0.0,
            metadata={"model_played_first": True},
        )
        data = result.to_dict()

        assert data["winner"] == 0
        assert data["num_turns"] == 50
        assert data["metadata"]["model_played_first"] is True


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        result = EvaluationResult(
            baseline_name="random",
            games_played=10,
            wins=7,
            losses=2,
            draws=1,
            avg_game_length=50.0,
        )
        assert result.win_rate == 0.7
        assert result.loss_rate == 0.2
        assert result.draw_rate == 0.1

    def test_win_rate_zero_games(self):
        """Test win rate with zero games."""
        result = EvaluationResult(
            baseline_name="random",
            games_played=0,
            wins=0,
            losses=0,
            draws=0,
            avg_game_length=0.0,
        )
        assert result.win_rate == 0.0
        assert result.loss_rate == 0.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = EvaluationResult(
            baseline_name="random",
            games_played=5,
            wins=3,
            losses=1,
            draws=1,
            avg_game_length=45.0,
        )
        data = result.to_dict()

        assert data["baseline_name"] == "random"
        assert data["games_played"] == 5
        assert data["win_rate"] == 0.6
        assert data["avg_game_length"] == 45.0

    def test_summary(self):
        """Test summary generation."""
        result = EvaluationResult(
            baseline_name="random",
            games_played=10,
            wins=7,
            losses=2,
            draws=1,
            avg_game_length=50.0,
        )
        summary = result.summary()

        assert "random" in summary
        assert "Games: 10" in summary
        assert "7W-2L-1D" in summary
        assert "70.0%" in summary


class TestModelEvaluator:
    """Tests for ModelEvaluator."""

    @pytest.fixture
    def quick_config(self):
        """Create quick evaluation config for testing."""
        return EvaluationConfig(
            experiment_name="test_eval",
            baselines=[
                BaselineConfig(
                    type=BaselineType.RANDOM,
                    name="random",
                    num_games=2,  # Very few games for quick tests
                ),
            ],
            elo_enabled=False,  # Disable ELO for simpler tests
            diversity=MagicMock(enabled=False),
            output_dir="logs/test_evaluation",
            tensorboard_enabled=False,
        )

    @pytest.fixture
    def evaluator(self, quick_config):
        """Create evaluator with quick config."""
        return ModelEvaluator(quick_config)

    def test_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert evaluator.config is not None
        assert evaluator._baselines == {}

    def test_get_baseline(self, evaluator):
        """Test getting/creating baseline player."""
        config = BaselineConfig(type=BaselineType.RANDOM, name="test")
        player = evaluator.get_baseline(config)

        assert isinstance(player, RandomPlayer)
        assert player.name == "test"

        # Should return same instance on second call
        player2 = evaluator.get_baseline(config)
        assert player is player2

    @patch("alpha_zero.evaluation.evaluator.ModelEvaluator._create_environment")
    def test_play_game_model_wins(self, mock_create_env, evaluator):
        """Test playing a game where model wins."""
        # Create mock environment
        mock_env = MagicMock()
        mock_env.done = False
        mock_env.current_player = 0
        mock_env.winner = 0  # Model wins
        mock_env.turn = 0

        # Setup step to end game after a few moves
        call_count = [0]

        def step_side_effect(action):
            call_count[0] += 1
            mock_env.turn = call_count[0]
            if call_count[0] >= 3:
                mock_env.done = True
            obs = {"action_mask": np.array([1, 1, 1])}
            return obs, 0, call_count[0] >= 3, False, {}

        mock_env.step = step_side_effect
        mock_env.reset.return_value = ({"action_mask": np.array([1, 1, 1])}, {})
        mock_env.get_action_mask.return_value = np.array([1, 1, 1])

        mock_create_env.return_value = mock_env

        # Create players
        model_player = RandomPlayer(name="model", seed=42)
        opponent = RandomPlayer(name="opponent", seed=43)

        result = evaluator.play_game(model_player, opponent, model_plays_first=True)

        assert result.winner == 0  # Model won
        assert result.model_score == 1.0
        assert result.num_turns > 0

    @patch("alpha_zero.evaluation.evaluator.ModelEvaluator._create_environment")
    def test_play_game_opponent_wins(self, mock_create_env, evaluator):
        """Test playing a game where opponent wins."""
        mock_env = MagicMock()
        mock_env.done = False
        mock_env.current_player = 0
        mock_env.winner = 1  # Opponent wins (player 1)
        mock_env.turn = 0

        call_count = [0]

        def step_side_effect(action):
            call_count[0] += 1
            mock_env.turn = call_count[0]
            if call_count[0] >= 3:
                mock_env.done = True
            return {"action_mask": np.array([1, 1, 1])}, 0, call_count[0] >= 3, False, {}

        mock_env.step = step_side_effect
        mock_env.reset.return_value = ({"action_mask": np.array([1, 1, 1])}, {})
        mock_env.get_action_mask.return_value = np.array([1, 1, 1])

        mock_create_env.return_value = mock_env

        model_player = RandomPlayer(name="model")
        opponent = RandomPlayer(name="opponent")

        result = evaluator.play_game(model_player, opponent, model_plays_first=True)

        assert result.winner == 1  # Opponent won
        assert result.model_score == 0.0

    @patch("alpha_zero.evaluation.evaluator.ModelEvaluator._create_environment")
    def test_play_game_draw(self, mock_create_env, evaluator):
        """Test playing a game that ends in draw."""
        mock_env = MagicMock()
        mock_env.done = False
        mock_env.current_player = 0
        mock_env.winner = None  # Draw
        mock_env.turn = 0

        call_count = [0]

        def step_side_effect(action):
            call_count[0] += 1
            mock_env.turn = call_count[0]
            if call_count[0] >= 3:
                mock_env.done = True
            return {"action_mask": np.array([1, 1, 1])}, 0, call_count[0] >= 3, False, {}

        mock_env.step = step_side_effect
        mock_env.reset.return_value = ({"action_mask": np.array([1, 1, 1])}, {})
        mock_env.get_action_mask.return_value = np.array([1, 1, 1])

        mock_create_env.return_value = mock_env

        model_player = RandomPlayer(name="model")
        opponent = RandomPlayer(name="opponent")

        result = evaluator.play_game(model_player, opponent)

        assert result.winner is None
        assert result.model_score == 0.5

    @patch("alpha_zero.evaluation.evaluator.ModelEvaluator.play_game")
    def test_evaluate_against_baseline(self, mock_play_game, evaluator):
        """Test evaluating against a single baseline."""
        # Mock game results: 2 wins, 0 losses
        mock_play_game.side_effect = [
            GameResult(winner=0, num_turns=50, model_score=1.0, opponent_score=0.0),
            GameResult(winner=0, num_turns=60, model_score=1.0, opponent_score=0.0),
        ]

        baseline_config = BaselineConfig(
            type=BaselineType.RANDOM,
            name="random",
            num_games=2,
        )
        model_player = RandomPlayer(name="model")

        result = evaluator.evaluate_against_baseline(model_player, baseline_config)

        assert result.baseline_name == "random"
        assert result.games_played == 2
        assert result.wins == 2
        assert result.losses == 0
        assert result.win_rate == 1.0
        assert result.avg_game_length == 55.0

    def test_get_diversity_metrics_disabled(self, evaluator):
        """Test getting diversity metrics when disabled."""
        metrics = evaluator.get_diversity_metrics()
        assert metrics is None

    def test_get_model_elo_disabled(self, evaluator):
        """Test getting ELO when disabled."""
        elo = evaluator.get_model_elo()
        assert elo is None

    @patch("alpha_zero.evaluation.evaluator.ModelEvaluator.evaluate_against_baseline")
    def test_summary(self, mock_evaluate, evaluator):
        """Test generating evaluation summary."""
        results = {
            "random": EvaluationResult(
                baseline_name="random",
                games_played=10,
                wins=7,
                losses=2,
                draws=1,
                avg_game_length=50.0,
            ),
            "heuristic": EvaluationResult(
                baseline_name="heuristic",
                games_played=10,
                wins=5,
                losses=4,
                draws=1,
                avg_game_length=60.0,
            ),
        }

        summary = evaluator.summary(results)

        assert "test_eval" in summary
        assert "Total Games: 20" in summary
        assert "12W-6L-2D" in summary
        assert "random" in summary
        assert "heuristic" in summary
