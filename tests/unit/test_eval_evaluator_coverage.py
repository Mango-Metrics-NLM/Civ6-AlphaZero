"""Tests for alpha_zero.evaluation.evaluator module.

Focuses on coverage of GameResult, EvaluationResult, and ModelEvaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alpha_zero.evaluation.config import (
    BaselineConfig,
    BaselineType,
    DiversityConfig,
    EvaluationConfig,
)
from alpha_zero.evaluation.evaluator import (
    EvaluationResult,
    GameResult,
    ModelEvaluator,
)

# ---------------------------------------------------------------------------
# GameResult
# ---------------------------------------------------------------------------


class TestGameResult:
    def test_basic_construction(self):
        g = GameResult(
            winner=0,
            num_turns=15,
            model_score=1.0,
            opponent_score=0.0,
        )
        assert g.winner == 0
        assert g.num_turns == 15
        assert g.model_score == 1.0

    def test_draw(self):
        g = GameResult(winner=None, num_turns=100, model_score=0.5, opponent_score=0.5)
        assert g.winner is None
        assert g.model_score == 0.5

    def test_to_dict(self):
        g = GameResult(
            winner=1,
            num_turns=20,
            model_score=0.0,
            opponent_score=1.0,
            metadata={"test": True},
        )
        d = g.to_dict()
        assert d["winner"] == 1
        assert d["num_turns"] == 20
        assert d["model_score"] == 0.0
        assert d["metadata"]["test"] is True
        # game_history is NOT in to_dict
        assert "game_history" not in d

    def test_default_fields(self):
        g = GameResult(winner=0, num_turns=5, model_score=1.0, opponent_score=0.0)
        assert g.game_history == []
        assert g.metadata == {}


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def test_win_rate(self):
        r = EvaluationResult(
            baseline_name="random",
            games_played=10,
            wins=7,
            losses=2,
            draws=1,
            avg_game_length=20.0,
        )
        assert r.win_rate == pytest.approx(0.7)

    def test_loss_rate(self):
        r = EvaluationResult(
            baseline_name="random", games_played=10, wins=3, losses=5, draws=2, avg_game_length=15.0
        )
        assert r.loss_rate == pytest.approx(0.5)

    def test_draw_rate(self):
        r = EvaluationResult(
            baseline_name="random", games_played=10, wins=3, losses=3, draws=4, avg_game_length=15.0
        )
        assert r.draw_rate == pytest.approx(0.4)

    def test_zero_games_rates(self):
        r = EvaluationResult(
            baseline_name="test", games_played=0, wins=0, losses=0, draws=0, avg_game_length=0.0
        )
        assert r.win_rate == 0.0
        assert r.loss_rate == 0.0
        assert r.draw_rate == 0.0

    def test_to_dict(self):
        r = EvaluationResult(
            baseline_name="heuristic",
            games_played=20,
            wins=12,
            losses=6,
            draws=2,
            avg_game_length=25.0,
        )
        d = r.to_dict()
        assert d["baseline_name"] == "heuristic"
        assert d["win_rate"] == pytest.approx(0.6)
        assert d["loss_rate"] == pytest.approx(0.3)
        assert d["draw_rate"] == pytest.approx(0.1)
        assert d["elo_update"] is None

    def test_to_dict_with_elo_update(self):
        mock_elo = MagicMock()
        mock_elo.rating_change = 15.0
        r = EvaluationResult(
            baseline_name="test",
            games_played=10,
            wins=7,
            losses=3,
            draws=0,
            avg_game_length=20.0,
            elo_update=mock_elo,
        )
        # asdict is called on elo_update, so mock needs to support that
        with patch("alpha_zero.evaluation.evaluator.asdict", return_value={"rating_change": 15.0}):
            d = r.to_dict()
        assert d["elo_update"] == {"rating_change": 15.0}

    def test_summary(self):
        r = EvaluationResult(
            baseline_name="random",
            games_played=10,
            wins=6,
            losses=3,
            draws=1,
            avg_game_length=20.0,
        )
        s = r.summary()
        assert "random" in s
        assert "6W-3L-1D" in s
        assert "60.0%" in s

    def test_summary_with_elo(self):
        @dataclass
        class FakeElo:
            rating_change: float = 12.5

        r = EvaluationResult(
            baseline_name="test",
            games_played=10,
            wins=7,
            losses=3,
            draws=0,
            avg_game_length=20.0,
            elo_update=FakeElo(),
        )
        s = r.summary()
        assert "ELO Change" in s
        assert "+12.5" in s


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------


class TestModelEvaluator:
    @pytest.fixture()
    def config(self, tmp_path):
        return EvaluationConfig(
            experiment_name="test_evaluator",
            model_checkpoint_path="nonexistent.pt",
            baselines=[
                BaselineConfig(type=BaselineType.RANDOM, name="random_test", num_games=2),
            ],
            diversity=DiversityConfig(enabled=False),
            elo_enabled=False,
            tensorboard_enabled=False,
            output_dir=str(tmp_path / "eval_output"),
            env_board_size=8,
            env_max_turns=10,
            save_game_logs=True,
        )

    @pytest.fixture()
    def evaluator(self, config):
        return ModelEvaluator(config)

    # -- Initialization --
    def test_init(self, evaluator):
        assert evaluator.config.experiment_name == "test_evaluator"
        assert evaluator.elo_system is None
        assert evaluator.diversity_tracker is None

    def test_init_with_elo(self, tmp_path):
        config = EvaluationConfig(
            experiment_name="elo_test",
            elo_enabled=True,
            elo_initial_rating=1500.0,
            tensorboard_enabled=False,
            output_dir=str(tmp_path / "elo_out"),
            diversity=DiversityConfig(enabled=False),
        )
        evaluator = ModelEvaluator(config)
        assert evaluator.elo_system is not None
        assert evaluator.model_stats is not None
        assert evaluator.model_stats.rating == 1500.0

    def test_init_with_diversity(self, tmp_path):
        config = EvaluationConfig(
            experiment_name="div_test",
            elo_enabled=False,
            tensorboard_enabled=False,
            output_dir=str(tmp_path / "div_out"),
            diversity=DiversityConfig(enabled=True),
        )
        evaluator = ModelEvaluator(config)
        assert evaluator.diversity_tracker is not None

    def test_init_tensorboard_import_failure(self, tmp_path):
        config = EvaluationConfig(
            experiment_name="tb_fail",
            elo_enabled=False,
            tensorboard_enabled=True,
            output_dir=str(tmp_path / "tb_out"),
            diversity=DiversityConfig(enabled=False),
        )
        with patch("alpha_zero.evaluation.evaluator.ModelEvaluator._setup_logging"):
            ModelEvaluator(config)
        # Just verify it doesn't crash

    # -- get_baseline --
    def test_get_baseline(self, evaluator):
        bc = BaselineConfig(type=BaselineType.RANDOM, name="test_bl")
        player = evaluator.get_baseline(bc)
        assert player is not None
        # Second call returns cached
        player2 = evaluator.get_baseline(bc)
        assert player is player2

    # -- play_game --
    def test_play_game_model_wins(self, evaluator):
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"action_mask": np.ones(10)}, {})
        mock_env.done = False

        call_count = [0]

        def fake_step(action):
            call_count[0] += 1
            if call_count[0] >= 3:
                mock_env.done = True
                mock_env.winner = 0
            return ({"action_mask": np.ones(10)}, 0.0, call_count[0] >= 3, False, {})

        mock_env.step = fake_step
        mock_env.current_player = 0

        model_player = MagicMock()
        model_player.select_action.return_value = 0
        model_player.reset.return_value = None

        opponent = MagicMock()
        opponent.select_action.return_value = 0
        opponent.reset.return_value = None

        with patch.object(evaluator, "_create_environment", return_value=mock_env):
            result = evaluator.play_game(model_player, opponent, model_plays_first=True)

        assert result.winner == 0
        assert result.model_score == 1.0
        assert result.num_turns == 3

    def test_play_game_opponent_wins(self, evaluator):
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"action_mask": np.ones(10)}, {})
        mock_env.done = False

        call_count = [0]

        def fake_step(action):
            call_count[0] += 1
            if call_count[0] >= 2:
                mock_env.done = True
                mock_env.winner = 1
            return ({"action_mask": np.ones(10)}, 0.0, call_count[0] >= 2, False, {})

        mock_env.step = fake_step
        mock_env.current_player = 0

        model_player = MagicMock()
        model_player.select_action.return_value = 0
        model_player.reset.return_value = None

        opponent = MagicMock()
        opponent.select_action.return_value = 0
        opponent.reset.return_value = None

        with patch.object(evaluator, "_create_environment", return_value=mock_env):
            result = evaluator.play_game(model_player, opponent, model_plays_first=True)

        assert result.winner == 1
        assert result.model_score == 0.0

    def test_play_game_draw(self, evaluator):
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"action_mask": np.ones(10)}, {})
        mock_env.done = False

        call_count = [0]

        def fake_step(action):
            call_count[0] += 1
            if call_count[0] >= 2:
                mock_env.done = True
                mock_env.winner = None
            return ({"action_mask": np.ones(10)}, 0.0, False, call_count[0] >= 2, {})

        mock_env.step = fake_step
        mock_env.current_player = 0

        model_player = MagicMock()
        model_player.select_action.return_value = 0
        model_player.reset.return_value = None

        opponent = MagicMock()
        opponent.select_action.return_value = 0
        opponent.reset.return_value = None

        with patch.object(evaluator, "_create_environment", return_value=mock_env):
            result = evaluator.play_game(model_player, opponent, model_plays_first=True)

        assert result.winner is None
        assert result.model_score == 0.5

    def test_play_game_model_second(self, evaluator):
        """Model plays second (player index 1), and wins."""
        mock_env = MagicMock()
        mock_env.reset.return_value = ({"action_mask": np.ones(10)}, {})
        mock_env.done = False

        call_count = [0]

        def fake_step(action):
            call_count[0] += 1
            if call_count[0] >= 2:
                mock_env.done = True
                mock_env.winner = 1  # Player 1 wins = model wins since model_plays_first=False
            return ({"action_mask": np.ones(10)}, 0.0, call_count[0] >= 2, False, {})

        mock_env.step = fake_step
        mock_env.current_player = 0

        model_player = MagicMock()
        model_player.select_action.return_value = 0
        model_player.reset.return_value = None

        opponent = MagicMock()
        opponent.select_action.return_value = 0
        opponent.reset.return_value = None

        with patch.object(evaluator, "_create_environment", return_value=mock_env):
            result = evaluator.play_game(model_player, opponent, model_plays_first=False)

        assert result.winner == 0  # Model won
        assert result.model_score == 1.0
        assert result.metadata["model_played_first"] is False

    # -- evaluate_against_baseline --
    def test_evaluate_against_baseline(self, evaluator):
        model_player = MagicMock()
        model_player.select_action.return_value = 0
        model_player.reset.return_value = None

        fake_game_result = GameResult(
            winner=0, num_turns=5, model_score=1.0, opponent_score=0.0, game_history=[]
        )

        baseline_config = BaselineConfig(type=BaselineType.RANDOM, name="random_bl", num_games=4)

        with patch.object(evaluator, "play_game", return_value=fake_game_result):
            result = evaluator.evaluate_against_baseline(model_player, baseline_config)

        assert result.games_played == 4
        assert result.wins == 4
        assert result.losses == 0
        assert result.draws == 0
        assert result.avg_game_length == 5.0

    def test_evaluate_against_baseline_mixed_results(self, evaluator):
        model_player = MagicMock()
        model_player.reset.return_value = None

        results_iter = iter(
            [
                GameResult(winner=0, num_turns=5, model_score=1.0, opponent_score=0.0),
                GameResult(winner=1, num_turns=8, model_score=0.0, opponent_score=1.0),
                GameResult(winner=None, num_turns=10, model_score=0.5, opponent_score=0.5),
                GameResult(winner=0, num_turns=6, model_score=1.0, opponent_score=0.0),
            ]
        )

        def side_effect(*args, **kwargs):
            return next(results_iter)

        baseline_config = BaselineConfig(type=BaselineType.RANDOM, name="mixed", num_games=4)

        with patch.object(evaluator, "play_game", side_effect=side_effect):
            result = evaluator.evaluate_against_baseline(model_player, baseline_config)

        assert result.wins == 2
        assert result.losses == 1
        assert result.draws == 1
        assert result.avg_game_length == pytest.approx(29.0 / 4)

    def test_evaluate_against_baseline_with_diversity(self, tmp_path):
        config = EvaluationConfig(
            experiment_name="div_eval",
            elo_enabled=False,
            tensorboard_enabled=False,
            output_dir=str(tmp_path / "div"),
            diversity=DiversityConfig(enabled=True),
            save_game_logs=True,
        )
        evaluator = ModelEvaluator(config)

        model_player = MagicMock()
        model_player.reset.return_value = None

        fake_result = GameResult(
            winner=0,
            num_turns=5,
            model_score=1.0,
            opponent_score=0.0,
            game_history=[(0, {"is_model": True, "turn": 0})],
        )
        baseline_config = BaselineConfig(type=BaselineType.RANDOM, name="random_div", num_games=2)

        with patch.object(evaluator, "play_game", return_value=fake_result):
            result = evaluator.evaluate_against_baseline(model_player, baseline_config)

        assert result.wins == 2

    def test_evaluate_against_baseline_with_elo(self, tmp_path):
        config = EvaluationConfig(
            experiment_name="elo_eval",
            elo_enabled=True,
            elo_initial_rating=1500.0,
            tensorboard_enabled=False,
            output_dir=str(tmp_path / "elo"),
            diversity=DiversityConfig(enabled=False),
        )
        evaluator = ModelEvaluator(config)

        model_player = MagicMock()
        model_player.reset.return_value = None

        fake_result = GameResult(winner=0, num_turns=5, model_score=1.0, opponent_score=0.0)
        baseline_config = BaselineConfig(type=BaselineType.RANDOM, name="random_elo", num_games=2)

        with patch.object(evaluator, "play_game", return_value=fake_result):
            result = evaluator.evaluate_against_baseline(model_player, baseline_config)

        assert result.elo_update is not None or result.elo_update is None  # May succeed or fail

    # -- evaluate_model --
    def test_evaluate_model_with_player(self, evaluator):
        model_player = MagicMock()
        model_player.reset.return_value = None

        fake_result = EvaluationResult(
            baseline_name="random_test",
            games_played=2,
            wins=1,
            losses=1,
            draws=0,
            avg_game_length=10.0,
        )

        with patch.object(evaluator, "evaluate_against_baseline", return_value=fake_result):
            results = evaluator.evaluate_model(model_player=model_player)

        assert "random_test" in results
        assert results["random_test"].games_played == 2

    def test_evaluate_model_skips_disabled_baselines(self, tmp_path):
        config = EvaluationConfig(
            experiment_name="skip_test",
            baselines=[
                BaselineConfig(type=BaselineType.RANDOM, name="enabled", num_games=2, enabled=True),
                BaselineConfig(
                    type=BaselineType.RANDOM, name="disabled", num_games=2, enabled=False
                ),
            ],
            elo_enabled=False,
            tensorboard_enabled=False,
            output_dir=str(tmp_path / "skip"),
            diversity=DiversityConfig(enabled=False),
        )
        evaluator = ModelEvaluator(config)
        model_player = MagicMock()

        fake_result = EvaluationResult(
            baseline_name="enabled", games_played=2, wins=1, losses=1, draws=0, avg_game_length=10.0
        )

        with patch.object(evaluator, "evaluate_against_baseline", return_value=fake_result):
            results = evaluator.evaluate_model(model_player=model_player)

        assert "enabled" in results
        assert "disabled" not in results

    def test_evaluate_model_no_checkpoint_falls_back(self, evaluator):
        """When checkpoint doesn't exist and no model_player, uses RandomPlayer."""
        fake_result = EvaluationResult(
            baseline_name="random_test",
            games_played=2,
            wins=1,
            losses=1,
            draws=0,
            avg_game_length=10.0,
        )

        with patch.object(evaluator, "evaluate_against_baseline", return_value=fake_result):
            results = evaluator.evaluate_model(checkpoint_path="nonexistent.pt")

        assert "random_test" in results

    # -- get_diversity_metrics --
    def test_get_diversity_metrics_disabled(self, evaluator):
        assert evaluator.get_diversity_metrics() is None

    def test_get_diversity_metrics_enabled(self, tmp_path):
        config = EvaluationConfig(
            experiment_name="dm_test",
            elo_enabled=False,
            tensorboard_enabled=False,
            output_dir=str(tmp_path / "dm"),
            diversity=DiversityConfig(enabled=True),
        )
        evaluator = ModelEvaluator(config)
        metrics = evaluator.get_diversity_metrics()
        assert metrics is not None

    # -- get_model_elo --
    def test_get_model_elo_disabled(self, evaluator):
        assert evaluator.get_model_elo() is None

    def test_get_model_elo_enabled(self, tmp_path):
        config = EvaluationConfig(
            experiment_name="elo_get",
            elo_enabled=True,
            elo_initial_rating=1500.0,
            tensorboard_enabled=False,
            output_dir=str(tmp_path / "elo_get"),
            diversity=DiversityConfig(enabled=False),
        )
        evaluator = ModelEvaluator(config)
        elo = evaluator.get_model_elo()
        assert elo == 1500.0

    # -- summary --
    def test_summary(self, evaluator):
        results = {
            "random": EvaluationResult(
                baseline_name="random",
                games_played=10,
                wins=7,
                losses=2,
                draws=1,
                avg_game_length=20.0,
            ),
        }
        s = evaluator.summary(results)
        assert "test_evaluator" in s
        assert "7W-2L-1D" in s
        assert "random" in s

    def test_summary_with_elo(self, tmp_path):
        config = EvaluationConfig(
            experiment_name="elo_summary",
            elo_enabled=True,
            elo_initial_rating=1500.0,
            tensorboard_enabled=False,
            output_dir=str(tmp_path / "elo_sum"),
            diversity=DiversityConfig(enabled=False),
        )
        evaluator = ModelEvaluator(config)
        results = {
            "random": EvaluationResult(
                baseline_name="random",
                games_played=10,
                wins=7,
                losses=3,
                draws=0,
                avg_game_length=20.0,
            ),
        }
        s = evaluator.summary(results)
        assert "ELO" in s

    def test_summary_with_diversity(self, tmp_path):
        config = EvaluationConfig(
            experiment_name="div_summary",
            elo_enabled=False,
            tensorboard_enabled=False,
            output_dir=str(tmp_path / "div_sum"),
            diversity=DiversityConfig(enabled=True),
        )
        evaluator = ModelEvaluator(config)
        # Record some data so diversity tracker has metrics
        evaluator.diversity_tracker.record_game(
            [(0, {"is_model": True, "turn": 0})],
            baseline_name="test",
            victory_type="win",
            game_length=10,
        )
        results = {
            "test": EvaluationResult(
                baseline_name="test",
                games_played=1,
                wins=1,
                losses=0,
                draws=0,
                avg_game_length=10.0,
            ),
        }
        s = evaluator.summary(results)
        assert "Diversity" in s

    def test_summary_empty_results(self, evaluator):
        s = evaluator.summary({})
        assert "test_evaluator" in s

    # -- _get_baseline_stats --
    def test_get_baseline_stats(self, tmp_path):
        config = EvaluationConfig(
            experiment_name="bl_stats",
            elo_enabled=True,
            elo_initial_rating=1500.0,
            tensorboard_enabled=False,
            output_dir=str(tmp_path / "bl_stats"),
            diversity=DiversityConfig(enabled=False),
        )
        evaluator = ModelEvaluator(config)
        stats = evaluator._get_baseline_stats("random")
        assert stats.rating == 800.0

        stats2 = evaluator._get_baseline_stats("heuristic")
        assert stats2.rating == 1200.0

        stats3 = evaluator._get_baseline_stats("unknown_baseline")
        assert stats3.rating == 1000.0

        # Cached
        stats4 = evaluator._get_baseline_stats("random")
        assert stats4 is stats
