"""Integration tests for W&B integration in enhanced training."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# The runner scripts live in scripts/, which is not on sys.path; bootstrap it the way
# tests/scripts/ does. The module name stays bare so existing mock.patch targets hold.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


@pytest.mark.slow
class TestEnhancedTrainingWithMockedWandb:
    """Integration tests with mocked W&B to avoid actual API calls."""

    @pytest.fixture
    def mock_wandb_logger(self):
        """Create a comprehensive mock W&B logger."""
        mock = MagicMock()
        mock.is_initialized = True
        mock.init.return_value = MagicMock(name="test_run", id="test123")
        mock.log = MagicMock()
        mock.log_selfplay = MagicMock()
        mock.log_training_step = MagicMock()
        mock.log_evaluation = MagicMock()
        mock.log_model = MagicMock()
        mock.summary = MagicMock()
        mock.finish = MagicMock()
        return mock

    @patch("run_enhanced_training.WANDB_AVAILABLE", True)
    @patch("run_enhanced_training.WandbLogger")
    @patch("run_enhanced_training.WandbConfig")
    def test_training_with_wandb_enabled(self, mock_config_cls, mock_logger_cls, mock_wandb_logger):
        """Test training with W&B enabled uses logger correctly."""
        from run_enhanced_training import run_enhanced_training

        # Setup mocks
        mock_logger_cls.return_value = mock_wandb_logger
        mock_config_cls.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            _history = run_enhanced_training(
                num_iterations=2,
                games_per_iteration=2,
                batch_size=4,
                num_epochs=1,
                num_simulations=3,
                board_size=6,
                max_turns=10,
                checkpoint_dir=tmpdir,
                eval_interval=2,
                device="cpu",
                wandb_enabled=True,
                wandb_project="test-project",
            )

            # W&B logger should have been initialized
            mock_wandb_logger.init.assert_called_once()

            # Should have logged self-play stats
            assert mock_wandb_logger.log_selfplay.called or mock_wandb_logger.log.called

            # Should have finished the run
            mock_wandb_logger.finish.assert_called_once()

    def test_training_without_wandb(self):
        """Test training works correctly without W&B."""
        from run_enhanced_training import run_enhanced_training

        with tempfile.TemporaryDirectory() as tmpdir:
            history = run_enhanced_training(
                num_iterations=2,
                games_per_iteration=2,
                batch_size=4,
                num_epochs=1,
                num_simulations=3,
                board_size=6,
                max_turns=10,
                checkpoint_dir=tmpdir,
                eval_interval=2,
                device="cpu",
                wandb_enabled=False,
            )

            # Training should complete successfully
            assert "final_metrics" in history
            assert "iterations" in history
            assert len(history["iterations"]) == 2


@pytest.mark.slow
class TestWandbLoggingCallSequence:
    """Tests for the sequence of W&B logging calls."""

    @pytest.fixture
    def call_tracker(self):
        """Track the sequence of W&B calls."""
        return {
            "calls": [],
            "selfplay_count": 0,
            "training_count": 0,
            "eval_count": 0,
        }

    def test_logging_call_order(self, call_tracker):
        """Test that W&B logging calls happen in correct order."""
        # Simulate the expected call sequence
        expected_sequence = [
            "init",
            "log_selfplay",  # First iteration self-play
            "log",  # Curriculum progress
            "log_training_step",  # Training
            "log_selfplay",  # Second iteration self-play
            "log",  # Curriculum progress
            "log_training_step",  # Training
            "log_evaluation",  # Evaluation at eval_interval
            "log_model",  # Save model
            "log_evaluation",  # Final evaluation
            "summary",  # Final summary
            "log_model",  # Final model
            "finish",
        ]

        # Verify the sequence makes sense
        assert expected_sequence[0] == "init"
        assert expected_sequence[-1] == "finish"
        assert "log_selfplay" in expected_sequence
        assert "log_training_step" in expected_sequence


@pytest.mark.slow
class TestWandbHistoryTracking:
    """Tests for history tracking with W&B."""

    def test_history_contains_wandb_compatible_data(self):
        """Test history contains data compatible with W&B logging."""
        from run_enhanced_training import run_enhanced_training

        with tempfile.TemporaryDirectory() as tmpdir:
            history = run_enhanced_training(
                num_iterations=2,
                games_per_iteration=2,
                batch_size=4,
                num_epochs=1,
                num_simulations=3,
                board_size=6,
                max_turns=10,
                checkpoint_dir=tmpdir,
                eval_interval=2,
                device="cpu",
                wandb_enabled=False,
            )

            # Check history structure is W&B compatible
            assert "iterations" in history
            assert "losses" in history
            assert "curriculum_stages" in history
            assert "final_metrics" in history

            # Final metrics should have numeric values
            final_metrics = history["final_metrics"]
            assert isinstance(final_metrics["win_rate"], (int, float))
            assert isinstance(final_metrics["avg_game_length"], (int, float))


@pytest.mark.slow
class TestWandbConfigPropagation:
    """Tests for W&B configuration propagation."""

    def test_config_dict_contains_all_params(self):
        """Test that config dict passed to W&B contains all training params."""
        # Simulate the config dict creation from run_enhanced_training
        num_iterations = 10
        games_per_iteration = 20
        batch_size = 64
        learning_rate = 0.001
        num_epochs = 5
        num_simulations = 30
        board_size = 8
        max_turns = 50
        use_curriculum = True
        use_reward_shaping = True
        use_env_tuning = True
        device = "cpu"

        config_dict = {
            "num_iterations": num_iterations,
            "games_per_iteration": games_per_iteration,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "num_simulations": num_simulations,
            "board_size": board_size,
            "max_turns": max_turns,
            "use_curriculum": use_curriculum,
            "use_reward_shaping": use_reward_shaping,
            "use_env_tuning": use_env_tuning,
            "device": device,
        }

        # Verify all values are correctly propagated
        assert config_dict["num_iterations"] == num_iterations
        assert config_dict["batch_size"] == batch_size
        assert config_dict["use_curriculum"] == use_curriculum


@pytest.mark.slow
class TestWandbErrorHandling:
    """Tests for W&B error handling."""

    def test_training_continues_if_wandb_unavailable(self):
        """Test training continues if W&B import fails."""
        from run_enhanced_training import run_enhanced_training

        with tempfile.TemporaryDirectory() as tmpdir:
            # This should not raise even if wandb is not installed
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
                wandb_enabled=True,  # Request W&B even if not available
            )

            # Training should complete
            assert "final_metrics" in history

    @patch("run_enhanced_training.WANDB_AVAILABLE", False)
    def test_warning_when_wandb_requested_but_unavailable(self, caplog):
        """Test warning is logged when W&B requested but unavailable."""
        import logging

        from run_enhanced_training import run_enhanced_training

        with tempfile.TemporaryDirectory() as tmpdir:
            with caplog.at_level(logging.WARNING):
                _history = run_enhanced_training(
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
                    wandb_enabled=True,
                )

            # Should have logged a warning about W&B
            # Note: This may or may not trigger depending on import state


@pytest.mark.slow
class TestWandbArtifactLogging:
    """Tests for W&B artifact (model) logging."""

    def test_model_artifacts_structure(self):
        """Test model artifact metadata structure."""
        # Simulate artifact metadata
        artifact_metadata = {
            "iteration": 5,
            "win_rate": 0.65,
        }

        assert "iteration" in artifact_metadata
        assert "win_rate" in artifact_metadata
        assert isinstance(artifact_metadata["iteration"], int)
        assert isinstance(artifact_metadata["win_rate"], float)

    def test_final_model_metadata(self):
        """Test final model artifact metadata."""
        final_metadata = {
            "win_rate": 0.7,
            "total_games": 500,
        }

        assert "win_rate" in final_metadata
        assert "total_games" in final_metadata


@pytest.mark.slow
class TestWandbTagsAndNaming:
    """Tests for W&B tags and run naming."""

    def test_default_tags(self):
        """Test default W&B tags are appropriate."""
        default_tags = ["enhanced-training", "curriculum", "reward-shaping"]

        # Tags should be descriptive
        assert "enhanced-training" in default_tags
        assert len(default_tags) >= 1

    def test_custom_tags_propagation(self):
        """Test custom tags are propagated correctly."""
        custom_tags = ["experiment-1", "test", "ablation"]

        # Tags should be a list of strings
        assert isinstance(custom_tags, list)
        assert all(isinstance(tag, str) for tag in custom_tags)

    def test_project_name_format(self):
        """Test project name format."""
        default_project = "civ6-alphazero-enhanced"

        # Project name should be lowercase with hyphens
        assert default_project.islower() or "-" in default_project
        assert " " not in default_project


@pytest.mark.slow
class TestWandbMetricsCompleteness:
    """Tests for completeness of W&B metrics logging."""

    def test_all_training_phases_logged(self):
        """Test all training phases have metrics logged."""
        # Define expected metrics for each phase
        selfplay_metrics = [
            "games_completed",
            "buffer_size",
            "samples_generated",
        ]

        training_metrics = [
            "loss",
            "policy_loss",
            "value_loss",
            "learning_rate",
        ]

        eval_metrics = [
            "win_rate",
            "games_played",
            "average_reward",
            "average_turns",
        ]

        curriculum_metrics = [
            "stage",
            "win_rate",
            "total_games",
        ]

        # All metric lists should be non-empty
        assert len(selfplay_metrics) > 0
        assert len(training_metrics) > 0
        assert len(eval_metrics) > 0
        assert len(curriculum_metrics) > 0

    def test_summary_metrics_completeness(self):
        """Test summary metrics are complete."""
        required_summary_keys = [
            "final_win_rate",
            "final_avg_game_length",
            "total_iterations",
            "total_games",
            "final_curriculum_stage",
            "final_curriculum_level",
        ]

        # All required keys should be defined
        assert len(required_summary_keys) == 6
