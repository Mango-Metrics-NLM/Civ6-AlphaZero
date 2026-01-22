"""Tests for W&B logger module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from alpha_zero.utils.wandb_logger import (
    WandbConfig,
    WandbLogger,
    get_wandb_logger,
    WANDB_AVAILABLE,
)


class TestWandbConfig:
    """Tests for WandbConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WandbConfig()
        assert config.enabled is False
        assert config.project == "civ6-alphazero"
        assert config.mode == "online"
        assert config.log_frequency == 10
        assert config.log_model is True

    def test_from_env_disabled(self):
        """Test from_env with W&B disabled."""
        with patch.dict(os.environ, {"WANDB_ENABLED": "false"}, clear=True):
            config = WandbConfig.from_env()
            assert config.enabled is False

    def test_from_env_enabled(self):
        """Test from_env with W&B enabled."""
        env_vars = {
            "WANDB_ENABLED": "true",
            "WANDB_PROJECT": "test-project",
            "WANDB_ENTITY": "test-entity",
            "WANDB_MODE": "offline",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = WandbConfig.from_env()
            assert config.enabled is True
            assert config.project == "test-project"
            assert config.entity == "test-entity"
            assert config.mode == "offline"

    def test_from_env_with_api_key(self):
        """Test from_env enables when API key is set."""
        env_vars = {
            "WANDB_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = WandbConfig.from_env()
            # Should be enabled when API key is present
            assert config.enabled is True

    def test_from_env_with_tags(self):
        """Test from_env with tags."""
        env_vars = {
            "WANDB_ENABLED": "true",
            "WANDB_TAGS": "tag1,tag2,tag3",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = WandbConfig.from_env()
            assert config.tags == ["tag1", "tag2", "tag3"]

    def test_from_env_numeric_settings(self):
        """Test from_env with numeric settings."""
        env_vars = {
            "WANDB_ENABLED": "true",
            "WANDB_LOG_FREQUENCY": "50",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            config = WandbConfig.from_env()
            assert config.log_frequency == 50


class TestWandbLoggerDisabled:
    """Tests for WandbLogger when disabled."""

    def test_disabled_by_default(self):
        """Test logger is disabled by default."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)
        assert logger.is_enabled is False
        assert logger.is_initialized is False

    def test_init_when_disabled(self):
        """Test init returns None when disabled."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)
        result = logger.init()
        assert result is None
        assert logger.is_initialized is False

    def test_log_when_disabled(self):
        """Test log does nothing when disabled."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)
        # Should not raise
        logger.log({"test": 1.0})
        logger.log_training_step(step=1, loss=0.5, policy_loss=0.3, value_loss=0.2)
        logger.log_game(game_id="test", winner=0, total_turns=100, reward=1.0)


class TestWandbLoggerEnabled:
    """Tests for WandbLogger when enabled (with mocking)."""

    @pytest.fixture
    def mock_wandb(self):
        """Fixture to mock wandb module."""
        with patch("alpha_zero.utils.wandb_logger.WANDB_AVAILABLE", True):
            with patch("alpha_zero.utils.wandb_logger.wandb") as mock:
                # Setup mock run
                mock_run = MagicMock()
                mock_run.name = "test-run"
                mock_run.id = "abc123"
                mock_run.summary = {}
                mock.init.return_value = mock_run
                mock.run = mock_run
                yield mock

    def test_init_success(self, mock_wandb):
        """Test successful initialization."""
        config = WandbConfig(enabled=True)

        # Create logger and force enabled state by setting config
        with patch("alpha_zero.utils.wandb_logger.WANDB_AVAILABLE", True):
            logger = WandbLogger(config)
            logger.init(config_dict={"lr": 0.001})

        mock_wandb.init.assert_called_once()

    def test_log_training_step(self, mock_wandb):
        """Test logging training step."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._initialized = True
        logger._run = mock_wandb.init.return_value

        logger.log_training_step(
            step=100,
            loss=0.5,
            policy_loss=0.3,
            value_loss=0.2,
            learning_rate=0.001,
        )

        mock_wandb.log.assert_called()
        call_args = mock_wandb.log.call_args
        data = call_args[0][0]
        assert "train/loss" in data
        assert "train/policy_loss" in data
        assert "train/value_loss" in data
        assert "train/learning_rate" in data

    def test_log_game(self, mock_wandb):
        """Test logging game results."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._initialized = True
        logger._run = mock_wandb.init.return_value

        logger.log_game(
            game_id="game_001",
            winner=0,
            total_turns=150,
            reward=1.0,
            game_length_seconds=60.0,
        )

        mock_wandb.log.assert_called()
        call_args = mock_wandb.log.call_args
        data = call_args[0][0]
        assert "game/winner" in data
        assert "game/total_turns" in data
        assert "game/reward" in data
        assert "game/duration_seconds" in data

    def test_log_evaluation(self, mock_wandb):
        """Test logging evaluation metrics."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._initialized = True
        logger._run = mock_wandb.init.return_value

        logger.log_evaluation(
            win_rate=0.75,
            games_played=100,
            average_reward=0.5,
            average_turns=120,
        )

        mock_wandb.log.assert_called()
        call_args = mock_wandb.log.call_args
        data = call_args[0][0]
        assert "eval/win_rate" in data
        assert "eval/games_played" in data

    def test_log_selfplay(self, mock_wandb):
        """Test logging self-play metrics."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._initialized = True
        logger._run = mock_wandb.init.return_value

        logger.log_selfplay(
            games_completed=500,
            buffer_size=10000,
            samples_generated=50000,
        )

        mock_wandb.log.assert_called()
        call_args = mock_wandb.log.call_args
        data = call_args[0][0]
        assert "selfplay/games_completed" in data
        assert "selfplay/buffer_size" in data

    def test_log_mcts(self, mock_wandb):
        """Test logging MCTS statistics."""
        config = WandbConfig(enabled=True)
        logger = WandbLogger(config)
        logger._initialized = True
        logger._run = mock_wandb.init.return_value

        logger.log_mcts(
            average_depth=15.5,
            average_visits=200.0,
            tree_reuse_rate=0.3,
        )

        mock_wandb.log.assert_called()
        call_args = mock_wandb.log.call_args
        data = call_args[0][0]
        assert "mcts/average_depth" in data
        assert "mcts/average_visits" in data
        assert "mcts/tree_reuse_rate" in data


class TestWandbLoggerDataCleaning:
    """Tests for data cleaning functionality."""

    def test_clean_numpy_scalar(self):
        """Test cleaning numpy scalar values."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)

        data = {
            "float32": np.float32(1.5),
            "int64": np.int64(42),
            "float64": np.float64(3.14),
        }
        cleaned = logger._clean_data(data)

        assert isinstance(cleaned["float32"], float)
        assert isinstance(cleaned["int64"], float)
        assert isinstance(cleaned["float64"], float)

    def test_clean_numpy_array_single(self):
        """Test cleaning single-element numpy arrays."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)

        data = {"single": np.array([1.5])}
        cleaned = logger._clean_data(data)

        assert isinstance(cleaned["single"], float)
        assert cleaned["single"] == 1.5

    def test_clean_numpy_array_multi(self):
        """Test cleaning multi-element numpy arrays."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)

        data = {"multi": np.array([1.0, 2.0, 3.0])}
        cleaned = logger._clean_data(data)

        assert isinstance(cleaned["multi"], list)
        assert cleaned["multi"] == [1.0, 2.0, 3.0]

    def test_clean_nested_dict(self):
        """Test cleaning nested dictionaries."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)

        data = {
            "outer": {
                "inner": np.float32(1.5),
            }
        }
        cleaned = logger._clean_data(data)

        assert isinstance(cleaned["outer"]["inner"], float)

    def test_clean_python_types(self):
        """Test that Python types pass through unchanged."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)

        data = {
            "str": "test",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
        }
        cleaned = logger._clean_data(data)

        assert cleaned == data


class TestWandbLoggerProperties:
    """Tests for WandbLogger properties."""

    def test_step_property(self):
        """Test step property."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)

        assert logger.step == 0
        logger._step = 100
        assert logger.step == 100

    def test_run_property(self):
        """Test run property."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)

        assert logger.run is None

    def test_is_enabled_property(self):
        """Test is_enabled property."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)
        assert logger.is_enabled is False

        config = WandbConfig(enabled=True)
        # Without WANDB_AVAILABLE, should still be False
        with patch("alpha_zero.utils.wandb_logger.WANDB_AVAILABLE", False):
            logger = WandbLogger(config)
            assert logger.is_enabled is False

    def test_is_initialized_property(self):
        """Test is_initialized property."""
        config = WandbConfig(enabled=False)
        logger = WandbLogger(config)

        assert logger.is_initialized is False
        logger._initialized = True
        logger._run = MagicMock()
        assert logger.is_initialized is True


class TestWandbLoggerGlobal:
    """Tests for global logger functions."""

    def test_get_wandb_logger(self):
        """Test getting global logger."""
        # Reset global state
        import alpha_zero.utils.wandb_logger as wandb_module
        wandb_module._global_logger = None

        logger = get_wandb_logger()
        assert isinstance(logger, WandbLogger)

        # Should return same instance
        logger2 = get_wandb_logger()
        assert logger is logger2

    def test_from_env(self):
        """Test creating logger from environment."""
        with patch.dict(os.environ, {"WANDB_PROJECT": "test-from-env"}, clear=True):
            logger = WandbLogger.from_env()
            assert logger.config.project == "test-from-env"


@pytest.mark.skipif(not WANDB_AVAILABLE, reason="W&B not installed")
class TestWandbLoggerIntegration:
    """Integration tests that require W&B package."""

    def test_import_wandb(self):
        """Test that wandb can be imported."""
        import wandb
        assert wandb is not None

    def test_config_creation(self):
        """Test creating config with real environment."""
        config = WandbConfig.from_env()
        assert isinstance(config, WandbConfig)

    def test_logger_creation(self):
        """Test creating logger."""
        logger = WandbLogger.from_env()
        assert isinstance(logger, WandbLogger)
