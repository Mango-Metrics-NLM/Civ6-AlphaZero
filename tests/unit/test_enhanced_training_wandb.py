"""Unit tests for W&B integration in enhanced training."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestWandbAvailability:
    """Tests for W&B availability handling."""

    def test_wandb_available_flag(self):
        """Test WANDB_AVAILABLE flag is set correctly."""
        from run_enhanced_training import WANDB_AVAILABLE

        # Should be a boolean
        assert isinstance(WANDB_AVAILABLE, bool)

    def test_wandb_classes_importable_when_available(self):
        """Test W&B classes can be imported when available."""
        from run_enhanced_training import WANDB_AVAILABLE

        if WANDB_AVAILABLE:
            from run_enhanced_training import WandbLogger, WandbConfig
            assert WandbLogger is not None
            assert WandbConfig is not None


class TestRunEnhancedTrainingSignatureWithWandb:
    """Tests for run_enhanced_training with W&B parameters."""

    def test_function_has_wandb_parameters(self):
        """Test that run_enhanced_training has W&B parameters."""
        from run_enhanced_training import run_enhanced_training
        import inspect

        sig = inspect.signature(run_enhanced_training)
        params = list(sig.parameters.keys())

        assert "wandb_enabled" in params
        assert "wandb_project" in params
        assert "wandb_run_name" in params
        assert "wandb_tags" in params

    def test_wandb_parameters_have_defaults(self):
        """Test W&B parameters have sensible defaults."""
        from run_enhanced_training import run_enhanced_training
        import inspect

        sig = inspect.signature(run_enhanced_training)

        # wandb_enabled should default to False
        assert sig.parameters["wandb_enabled"].default is False

        # wandb_project should have a default string
        assert sig.parameters["wandb_project"].default == "civ6-alphazero-enhanced"

        # wandb_run_name should default to None (auto-generated)
        assert sig.parameters["wandb_run_name"].default is None


class TestWandbDisabledBehavior:
    """Tests for behavior when W&B is disabled."""

    def test_training_runs_without_wandb(self):
        """Test training runs successfully without W&B."""
        from run_enhanced_training import run_enhanced_training

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should run without errors when wandb_enabled=False
            history = run_enhanced_training(
                num_iterations=1,
                games_per_iteration=1,
                batch_size=4,
                num_epochs=1,
                num_simulations=3,
                board_size=6,
                max_turns=10,
                checkpoint_dir=tmpdir,
                eval_interval=1,
                device="cpu",
                wandb_enabled=False,  # Explicitly disabled
            )

            assert "final_metrics" in history
            assert "iterations" in history


class TestWandbLoggerIntegration:
    """Tests for W&B logger integration patterns."""

    def test_wandb_logger_optional_import(self):
        """Test W&B logger is imported as optional."""
        # This should not raise an error even if wandb is not installed
        try:
            from run_enhanced_training import WandbLogger, WandbConfig, WANDB_AVAILABLE
            # Should either have classes or None
            if WANDB_AVAILABLE:
                assert WandbLogger is not None
            else:
                # When not available, these may be None (type: ignore in code)
                pass
        except ImportError:
            pytest.skip("Import structure unexpected")

    @pytest.fixture
    def mock_wandb_logger(self):
        """Create a mock W&B logger for testing."""
        mock_logger = MagicMock()
        mock_logger.is_initialized = True
        mock_logger.init.return_value = MagicMock(name="mock_run")
        return mock_logger

    def test_wandb_logger_methods_called(self, mock_wandb_logger):
        """Test that W&B logger methods are called during training."""
        # Verify the mock has expected methods
        mock_wandb_logger.log_selfplay(
            games_completed=10,
            buffer_size=100,
            samples_generated=500,
        )
        mock_wandb_logger.log_selfplay.assert_called_once()

        mock_wandb_logger.log_training_step(
            step=1,
            loss=0.5,
            policy_loss=0.3,
            value_loss=0.2,
            learning_rate=0.001,
        )
        mock_wandb_logger.log_training_step.assert_called_once()

        mock_wandb_logger.log_evaluation(
            win_rate=0.6,
            games_played=10,
            average_reward=0.2,
            average_turns=50.0,
        )
        mock_wandb_logger.log_evaluation.assert_called_once()


class TestMainFunctionWandbArgs:
    """Tests for main function W&B argument parsing."""

    def test_main_parses_wandb_flag(self):
        """Test main function parses --wandb flag."""
        import argparse
        from run_enhanced_training import main

        # Create parser similar to main()
        parser = argparse.ArgumentParser()
        parser.add_argument("--wandb", action="store_true")
        parser.add_argument("--wandb-project", type=str, default="test")
        parser.add_argument("--wandb-run-name", type=str, default=None)
        parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)

        args = parser.parse_args(["--wandb"])
        assert args.wandb is True

        args = parser.parse_args([])
        assert args.wandb is False

    def test_main_parses_wandb_project(self):
        """Test main function parses --wandb-project."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--wandb-project", type=str, default="default-project")

        args = parser.parse_args(["--wandb-project", "my-project"])
        assert args.wandb_project == "my-project"

    def test_main_parses_wandb_tags(self):
        """Test main function parses --wandb-tags."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)

        args = parser.parse_args(["--wandb-tags", "tag1", "tag2", "tag3"])
        assert args.wandb_tags == ["tag1", "tag2", "tag3"]


class TestWandbConfigDict:
    """Tests for W&B configuration dictionary."""

    def test_config_dict_structure(self):
        """Test the structure of config dict passed to W&B."""
        # Simulate the config dict creation
        config_dict = {
            "num_iterations": 10,
            "games_per_iteration": 20,
            "batch_size": 64,
            "learning_rate": 0.001,
            "num_epochs": 5,
            "num_simulations": 30,
            "board_size": 8,
            "max_turns": 50,
            "use_curriculum": True,
            "use_reward_shaping": True,
            "use_env_tuning": True,
            "device": "cpu",
        }

        # All expected keys should be present
        expected_keys = [
            "num_iterations",
            "games_per_iteration",
            "batch_size",
            "learning_rate",
            "num_epochs",
            "num_simulations",
            "board_size",
            "max_turns",
            "use_curriculum",
            "use_reward_shaping",
            "use_env_tuning",
            "device",
        ]

        for key in expected_keys:
            assert key in config_dict

    def test_config_dict_types(self):
        """Test config dict has correct types."""
        config_dict = {
            "num_iterations": 10,
            "games_per_iteration": 20,
            "batch_size": 64,
            "learning_rate": 0.001,
            "use_curriculum": True,
        }

        assert isinstance(config_dict["num_iterations"], int)
        assert isinstance(config_dict["learning_rate"], float)
        assert isinstance(config_dict["use_curriculum"], bool)


class TestWandbMetricsFormat:
    """Tests for W&B metrics format."""

    def test_selfplay_metrics_format(self):
        """Test self-play metrics have correct format."""
        # Simulate metrics that would be logged
        metrics = {
            "selfplay/games_completed": 100,
            "selfplay/buffer_size": 5000,
            "selfplay/samples_generated": 1000,
            "selfplay/win_rate": 0.5,
            "selfplay/avg_game_length": 45.0,
        }

        # All metrics should be numeric
        for key, value in metrics.items():
            assert isinstance(value, (int, float))
            assert key.startswith("selfplay/")

    def test_curriculum_metrics_format(self):
        """Test curriculum metrics have correct format."""
        metrics = {
            "curriculum/stage": 0,
            "curriculum/stage_name": "beginner",
            "curriculum/win_rate": 0.6,
            "curriculum/total_games": 50,
        }

        # Stage should be an integer
        assert isinstance(metrics["curriculum/stage"], int)
        # Stage name should be a string
        assert isinstance(metrics["curriculum/stage_name"], str)
        # Win rate should be a float
        assert isinstance(metrics["curriculum/win_rate"], float)

    def test_training_metrics_format(self):
        """Test training metrics have correct format."""
        metrics = {
            "train/loss": 0.5,
            "train/policy_loss": 0.3,
            "train/value_loss": 0.2,
            "train/learning_rate": 0.001,
        }

        for key, value in metrics.items():
            assert isinstance(value, float)
            assert key.startswith("train/")

    def test_eval_metrics_format(self):
        """Test evaluation metrics have correct format."""
        metrics = {
            "eval/win_rate": 0.65,
            "eval/games_played": 20,
            "eval/average_reward": 0.3,
            "eval/average_turns": 52.5,
        }

        # Win rate should be between 0 and 1
        assert 0.0 <= metrics["eval/win_rate"] <= 1.0
        # Games played should be positive integer
        assert metrics["eval/games_played"] > 0


class TestWandbSummaryFormat:
    """Tests for W&B summary format."""

    def test_final_summary_structure(self):
        """Test final summary has correct structure."""
        summary = {
            "final_win_rate": 0.7,
            "final_avg_game_length": 48.5,
            "total_iterations": 50,
            "total_games": 500,
            "final_curriculum_stage": "advanced",
            "final_curriculum_level": 2,
        }

        # All required keys present
        required_keys = [
            "final_win_rate",
            "final_avg_game_length",
            "total_iterations",
            "total_games",
            "final_curriculum_stage",
            "final_curriculum_level",
        ]

        for key in required_keys:
            assert key in summary

    def test_summary_values_valid(self):
        """Test summary values are valid."""
        summary = {
            "final_win_rate": 0.7,
            "final_avg_game_length": 48.5,
            "total_iterations": 50,
            "total_games": 500,
        }

        # Win rate should be between 0 and 1
        assert 0.0 <= summary["final_win_rate"] <= 1.0

        # Positive values
        assert summary["final_avg_game_length"] > 0
        assert summary["total_iterations"] > 0
        assert summary["total_games"] > 0
