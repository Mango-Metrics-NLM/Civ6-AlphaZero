"""Coverage tests for PaddedEnvironmentWrapper and TrainingOrchestrator helpers.

Targets uncovered paths in alpha_zero/orchestrator/orchestrator.py:
- PaddedEnvironmentWrapper: pad_spatial, pad_actions, get_observation, reset, step, clone
- TrainingOrchestrator: __init__, _setup_logging, _handle_shutdown, set_training_safeguard,
  set_muzero_bridge
- TrainingState and GameResult dataclasses
"""

from __future__ import annotations

import signal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alpha_zero.orchestrator.orchestrator import (
    GameResult,
    PaddedEnvironmentWrapper,
    TrainingOrchestrator,
    TrainingState,
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


class TestTrainingState:
    def test_defaults(self):
        state = TrainingState()
        assert state.iteration == 0
        assert state.total_games == 0
        assert state.is_running is False

    def test_custom_values(self):
        state = TrainingState(iteration=5, total_games=100, best_win_rate=0.75)
        assert state.iteration == 5
        assert state.best_win_rate == 0.75


class TestGameResult:
    def test_basic(self):
        result = GameResult(
            trajectory=[{"state": "s1"}],
            winner=1,
            total_reward=1.0,
            turns=50,
        )
        assert result.winner == 1
        assert result.turns == 50
        assert result.metrics == {}

    def test_with_metrics(self):
        result = GameResult(
            trajectory=[],
            winner=0,
            total_reward=0.0,
            turns=10,
            metrics={"entropy": 1.5},
        )
        assert result.metrics["entropy"] == 1.5


# ---------------------------------------------------------------------------
# PaddedEnvironmentWrapper
# ---------------------------------------------------------------------------


class TestPaddedEnvironmentWrapper:
    @pytest.fixture
    def mock_env(self):
        env = MagicMock()
        env.action_space = MagicMock()
        env.action_space.n = 50
        return env

    def test_getattr_forwards(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=16, expected_action_size=100)
        mock_env.some_attr = 42
        assert wrapper.some_attr == 42

    def test_pad_spatial_no_pad_needed(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=8, expected_action_size=50)
        spatial = np.random.rand(13, 8, 8).astype(np.float32)
        result = wrapper._pad_spatial(spatial)
        assert result.shape == (13, 8, 8)

    def test_pad_spatial_pads(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=16, expected_action_size=50)
        spatial = np.random.rand(13, 8, 8).astype(np.float32)
        result = wrapper._pad_spatial(spatial)
        assert result.shape == (13, 16, 16)
        # Center should contain original data
        assert np.allclose(result[:, 4:12, 4:12], spatial)

    def test_pad_actions_no_pad_needed(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=8, expected_action_size=50)
        actions = np.ones(50, dtype=np.float32)
        result = wrapper._pad_actions(actions)
        assert result.shape == (50,)

    def test_pad_actions_pads(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=8, expected_action_size=100)
        actions = np.ones(50, dtype=np.float32)
        result = wrapper._pad_actions(actions)
        assert result.shape == (100,)
        assert np.all(result[:50] == 1.0)
        assert np.all(result[50:] == 0.0)

    def test_get_observation_dict(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=16, expected_action_size=100)
        mock_env.get_observation.return_value = {
            "spatial": np.random.rand(13, 8, 8).astype(np.float32),
            "vector": np.random.rand(64).astype(np.float32),
            "action_mask": np.ones(50, dtype=np.float32),
        }
        obs = wrapper.get_observation()
        assert obs["spatial"].shape == (13, 16, 16)
        assert obs["action_mask"].shape == (100,)
        assert obs["vector"].shape == (64,)  # Unchanged

    def test_get_observation_non_dict(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=16, expected_action_size=100)
        mock_env.get_observation.return_value = np.array([1.0, 2.0])
        obs = wrapper.get_observation()
        assert isinstance(obs, np.ndarray)

    def test_deprecated_get_observation(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=8, expected_action_size=50)
        mock_env.get_observation.return_value = {"spatial": np.zeros((13, 8, 8))}
        with pytest.warns(DeprecationWarning, match="_get_observation"):
            wrapper._get_observation()

    def test_get_valid_actions(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=8, expected_action_size=100)
        mock_env.get_valid_actions.return_value = np.ones(50, dtype=np.float32)
        result = wrapper.get_valid_actions()
        assert result.shape == (100,)

    def test_reset_tuple(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=16, expected_action_size=100)
        mock_env.reset.return_value = (
            {
                "spatial": np.random.rand(13, 8, 8).astype(np.float32),
                "action_mask": np.ones(50, dtype=np.float32),
            },
            {"info_key": "val"},
        )
        obs, info = wrapper.reset()
        assert obs["spatial"].shape == (13, 16, 16)
        assert obs["action_mask"].shape == (100,)
        assert info["info_key"] == "val"

    def test_reset_non_tuple(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=8, expected_action_size=50)
        mock_env.reset.return_value = np.array([1.0])
        obs, info = wrapper.reset()
        assert isinstance(obs, np.ndarray)
        assert info == {}

    def test_step_dict_obs(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=16, expected_action_size=100)
        mock_env.step.return_value = (
            {"spatial": np.random.rand(13, 8, 8).astype(np.float32)},
            1.0,
            False,
            False,
            {},
        )
        obs, reward, terminated, truncated, info = wrapper.step(0)
        assert obs["spatial"].shape == (13, 16, 16)
        assert reward == 1.0

    def test_step_non_dict_obs(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=8, expected_action_size=50)
        mock_env.step.return_value = (np.array([1.0]), 0.5, True, False, {})
        obs, reward, terminated, truncated, info = wrapper.step(0)
        assert isinstance(obs, np.ndarray)
        assert terminated is True

    def test_clone(self, mock_env):
        wrapper = PaddedEnvironmentWrapper(mock_env, expected_board_size=16, expected_action_size=100)
        cloned_env = MagicMock()
        mock_env.clone.return_value = cloned_env
        cloned = wrapper.clone()
        assert isinstance(cloned, PaddedEnvironmentWrapper)
        assert cloned._expected_board_size == 16
        assert cloned._expected_action_size == 100


# ---------------------------------------------------------------------------
# TrainingOrchestrator init helpers
# ---------------------------------------------------------------------------


class TestTrainingOrchestratorInit:
    @pytest.fixture
    def config(self):
        cfg = MagicMock()
        cfg.logging = MagicMock()
        cfg.logging.level = "INFO"
        cfg.log_dir = ""
        cfg.experiment_name = "test_experiment"
        return cfg

    def test_init(self, config):
        with patch("alpha_zero.orchestrator.orchestrator.signal.signal"):
            orch = TrainingOrchestrator(config)
        assert orch.state.iteration == 0
        assert orch._shutdown_requested is False
        assert orch._training_mode == "alphazero"

    def test_setup_logging_with_log_dir(self, config, tmp_path):
        config.log_dir = str(tmp_path)
        with patch("alpha_zero.orchestrator.orchestrator.signal.signal"):
            orch = TrainingOrchestrator(config)
        # Should have created a log file
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) >= 1

    def test_handle_shutdown(self, config):
        with patch("alpha_zero.orchestrator.orchestrator.signal.signal"):
            orch = TrainingOrchestrator(config)
        orch._handle_shutdown(signal.SIGINT, None)
        assert orch._shutdown_requested is True

    def test_set_training_safeguard(self, config):
        with patch("alpha_zero.orchestrator.orchestrator.signal.signal"):
            orch = TrainingOrchestrator(config)
        mock_safeguard = MagicMock()
        orch.set_training_safeguard(mock_safeguard)
        assert orch._training_safeguard is mock_safeguard

    def test_set_muzero_bridge(self, config):
        with patch("alpha_zero.orchestrator.orchestrator.signal.signal"):
            orch = TrainingOrchestrator(config)
        mock_bridge = MagicMock()
        orch.set_muzero_bridge(mock_bridge, training_mode="muzero")
        assert orch._muzero_bridge is mock_bridge
        assert orch._training_mode == "muzero"

    def test_set_muzero_bridge_none(self, config):
        with patch("alpha_zero.orchestrator.orchestrator.signal.signal"):
            orch = TrainingOrchestrator(config)
        orch.set_muzero_bridge(None)
        assert orch._muzero_bridge is None
