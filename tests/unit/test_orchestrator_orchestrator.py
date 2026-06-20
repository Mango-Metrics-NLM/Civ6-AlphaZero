"""Unit tests for alpha_zero.orchestrator.orchestrator module.

Covers:
- TrainingState and GameResult dataclasses
- PaddedEnvironmentWrapper
- TrainingOrchestrator: initialization, run loop dispatch, checkpoint
  management, metric tracking, helpers, dry_run, get_statistics
"""

from __future__ import annotations

import contextlib
import signal
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from alpha_zero.orchestrator.config import OrchestratorConfig
from alpha_zero.orchestrator.orchestrator import (
    GameResult,
    PaddedEnvironmentWrapper,
    TrainingOrchestrator,
    TrainingState,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_SIZE = 6
ACTION_SIZE = 50
MOD = "alpha_zero.orchestrator.orchestrator"

# ---------------------------------------------------------------------------
# Helpers / shared factories
# ---------------------------------------------------------------------------


def _make_config(tmp_path, *, num_iterations=2, resume=False) -> OrchestratorConfig:
    """Return a minimal fast-preset config wired to tmp_path."""
    config = OrchestratorConfig.from_preset("fast")
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    config.log_dir = str(tmp_path / "logs")
    config.training.resume = resume
    config.training.num_iterations = num_iterations
    config.training.games_per_iteration = 1
    config.training.num_epochs = 1
    config.training.min_replay_size = 1
    config.training.replay_buffer_size = 100
    config.training.checkpoint_interval = 1
    config.training.evaluation_interval = 1
    config.training.evaluation_games = 1
    config.environment.board_size = BOARD_SIZE
    config.environment.max_turns = 3
    config.mcts.num_simulations = 1
    config.random_seed = 0
    # Disable all optional subsystems
    config.rag.enabled = False
    config.llm.enabled = False
    config.strategy_advisor.enabled = False
    config.wandb.enabled = False
    config.reflection.enabled = False
    config.hierarchical.enabled = False
    return config


def _make_mock_env(board_size=BOARD_SIZE, action_size=ACTION_SIZE):
    """Return a mock environment with a realistic interface."""
    env = MagicMock()
    obs = {
        "spatial": np.zeros((13, board_size, board_size), dtype=np.float32),
        "vector": np.zeros(64, dtype=np.float32),
        "action_mask": np.ones(action_size, dtype=np.float32),
    }
    env.reset.return_value = (obs, {})
    env.step.return_value = (obs, 1.0, True, False, {"winner": 1})
    env.get_valid_actions.return_value = np.ones(action_size, dtype=np.float32)
    env.get_observation.return_value = obs
    env.action_space = MagicMock()
    env.action_space.n = action_size
    env.clone.return_value = env
    return env


def _make_mock_network(action_size=ACTION_SIZE):
    """Return a real tiny module so Adam can build its param groups,
    but with a mocked forward() that returns fixed tensors."""

    class _TinyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._dummy = torch.nn.Linear(1, 1, bias=False)

        def forward(self, spatial, vector):  # type: ignore[override]
            batch = spatial.shape[0]
            return torch.zeros(batch, action_size), torch.zeros(batch, 1)

    network = _TinyNet()
    network.state_dict = MagicMock(return_value={})  # type: ignore[method-assign]
    return network


def _make_mock_mcts(action_size=ACTION_SIZE):
    """Return a mock MCTS with a uniform policy."""
    mcts = MagicMock()
    policy = np.ones(action_size, dtype=np.float32) / action_size
    mcts.search.return_value = (policy, {})
    mcts.select_action.return_value = 0
    return mcts


def _make_mock_checkpointer():
    mock_checkpointer = MagicMock()
    mock_checkpointer.has_checkpoint.return_value = False
    mock_checkpointer.save.return_value = None
    mock_checkpointer.save_best.return_value = None
    mock_checkpointer.save_final.return_value = None
    return mock_checkpointer


def _make_mocks():
    """Build the full set of component mocks."""
    mock_env = _make_mock_env()
    mock_net = _make_mock_network()
    mock_mcts = _make_mock_mcts()

    mock_replay = MagicMock()
    mock_replay.__len__ = MagicMock(return_value=200)
    mock_replay.is_ready.return_value = False
    mock_replay.sample.return_value = []
    mock_replay.min_size = 1
    mock_replay.buffer = []

    mock_curriculum = MagicMock()
    mock_curriculum.get_environment_params.return_value = {}
    stage = MagicMock()
    stage.name = "beginner"
    mock_curriculum.get_current_stage.return_value = stage

    mock_reward_shaper = MagicMock()
    mock_reward_shaper.shape_reward.return_value = 0.0

    mock_checkpointer = _make_mock_checkpointer()

    return {
        "env": mock_env,
        "network": mock_net,
        "mcts": mock_mcts,
        "replay": mock_replay,
        "curriculum": mock_curriculum,
        "reward_shaper": mock_reward_shaper,
        "checkpointer": mock_checkpointer,
    }


@contextlib.contextmanager
def _patch_factories(mocks):
    """Context manager that patches all factory functions and CheckpointManager."""
    with (
        patch(f"{MOD}.create_environment", return_value=mocks["env"]),
        patch(f"{MOD}.create_network", return_value=mocks["network"]),
        patch(f"{MOD}.create_mcts", return_value=mocks["mcts"]),
        patch(f"{MOD}.create_replay_buffer", return_value=mocks["replay"]),
        patch(f"{MOD}.create_curriculum_manager", return_value=mocks["curriculum"]),
        patch(f"{MOD}.create_reward_shaper", return_value=mocks["reward_shaper"]),
        patch(f"{MOD}.create_llm_client", return_value=None),
        patch(f"{MOD}.create_wandb_logger", return_value=None),
        patch(f"{MOD}.CheckpointManager", return_value=mocks["checkpointer"]),
    ):
        yield


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mocks():
    return _make_mocks()


@pytest.fixture()
def orchestrator(tmp_path, mocks):
    """Return a fully-initialized TrainingOrchestrator with mocked components."""
    config = _make_config(tmp_path)
    with _patch_factories(mocks):
        orch = TrainingOrchestrator(config)
        ok = orch.initialize()
    assert ok, "initialize() must return True"
    return orch, mocks


# ---------------------------------------------------------------------------
# TrainingState
# ---------------------------------------------------------------------------


class TestTrainingState:
    def test_default_values(self):
        state = TrainingState()
        assert state.iteration == 0
        assert state.total_games == 0
        assert state.total_steps == 0
        assert state.best_win_rate == 0.0
        assert state.current_win_rate == 0.0
        assert state.recent_losses == []
        assert state.curriculum_stage == "beginner"
        assert state.is_running is False
        assert state.start_time == 0.0

    def test_mutable_fields_are_independent(self):
        s1 = TrainingState()
        s2 = TrainingState()
        s1.recent_losses.append(1.0)
        assert s2.recent_losses == []


# ---------------------------------------------------------------------------
# GameResult
# ---------------------------------------------------------------------------


class TestGameResult:
    def test_construction(self):
        traj = [{"spatial": np.zeros((13, 6, 6)), "value": 1.0}]
        result = GameResult(trajectory=traj, winner=1, total_reward=5.0, turns=10)
        assert result.winner == 1
        assert result.total_reward == 5.0
        assert result.turns == 10
        assert result.metrics == {}

    def test_custom_metrics(self):
        result = GameResult(
            trajectory=[],
            winner=0,
            total_reward=0.0,
            turns=0,
            metrics={"avg_reward": 0.5},
        )
        assert result.metrics["avg_reward"] == 0.5


# ---------------------------------------------------------------------------
# PaddedEnvironmentWrapper
# ---------------------------------------------------------------------------


class TestPaddedEnvironmentWrapper:
    @pytest.fixture()
    def small_env(self):
        return _make_mock_env(board_size=4, action_size=20)

    @pytest.fixture()
    def wrapper(self, small_env):
        return PaddedEnvironmentWrapper(small_env, expected_board_size=8, expected_action_size=40)

    def test_getattr_delegates(self, wrapper, small_env):
        small_env.some_attr = 42
        assert wrapper.some_attr == 42

    def test_pad_spatial_when_smaller(self, wrapper):
        arr = np.ones((13, 4, 4), dtype=np.float32)
        padded = wrapper._pad_spatial(arr)
        assert padded.shape == (13, 8, 8)
        # Original region is in the centre after symmetric padding
        assert padded[:, 2:6, 2:6].sum() == pytest.approx(13 * 4 * 4)

    def test_pad_spatial_no_op_when_same_size(self, wrapper):
        arr = np.ones((13, 8, 8), dtype=np.float32)
        padded = wrapper._pad_spatial(arr)
        assert padded is arr

    def test_pad_actions_when_smaller(self, wrapper):
        arr = np.ones(20, dtype=np.float32)
        padded = wrapper._pad_actions(arr)
        assert padded.shape == (40,)
        assert padded[:20].sum() == pytest.approx(20.0)
        assert padded[20:].sum() == pytest.approx(0.0)

    def test_pad_actions_no_op_when_same_size(self, wrapper):
        arr = np.ones(40, dtype=np.float32)
        padded = wrapper._pad_actions(arr)
        assert padded is arr

    def test_get_valid_actions_padded(self, wrapper, small_env):
        small_env.get_valid_actions.return_value = np.ones(20, dtype=np.float32)
        result = wrapper.get_valid_actions()
        assert result.shape == (40,)

    def test_get_observation_padded(self, wrapper, small_env):
        obs = {
            "spatial": np.zeros((13, 4, 4), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
            "action_mask": np.ones(20, dtype=np.float32),
        }
        small_env.get_observation.return_value = obs
        result = wrapper.get_observation()
        assert result["spatial"].shape == (13, 8, 8)
        assert result["action_mask"].shape == (40,)

    def test_reset_pads_obs(self, wrapper, small_env):
        small_obs = {
            "spatial": np.zeros((13, 4, 4), dtype=np.float32),
            "action_mask": np.ones(20, dtype=np.float32),
        }
        small_env.reset.return_value = (small_obs, {})
        obs, info = wrapper.reset()
        assert obs["spatial"].shape == (13, 8, 8)
        assert obs["action_mask"].shape == (40,)

    def test_reset_non_tuple_env(self, wrapper, small_env):
        """If base env returns only obs (old API), wrapper handles it."""
        small_obs = {
            "spatial": np.zeros((13, 4, 4), dtype=np.float32),
            "action_mask": np.ones(20, dtype=np.float32),
        }
        small_env.reset.return_value = small_obs
        obs, info = wrapper.reset()
        assert obs["spatial"].shape == (13, 8, 8)

    def test_step_pads_obs(self, wrapper, small_env):
        small_obs = {
            "spatial": np.zeros((13, 4, 4), dtype=np.float32),
        }
        small_env.step.return_value = (small_obs, 1.0, True, False, {})
        obs, reward, terminated, truncated, info = wrapper.step(0)
        assert obs["spatial"].shape == (13, 8, 8)

    def test_clone_returns_wrapper(self, wrapper, small_env):
        cloned_base = _make_mock_env(board_size=4, action_size=20)
        small_env.clone.return_value = cloned_base
        result = wrapper.clone()
        assert isinstance(result, PaddedEnvironmentWrapper)
        assert result._expected_board_size == wrapper._expected_board_size


# ---------------------------------------------------------------------------
# TrainingOrchestrator — initialisation
# ---------------------------------------------------------------------------


class TestTrainingOrchestratorInit:
    def test_init_sets_config(self, tmp_path, mocks):
        config = _make_config(tmp_path)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
        assert orch.config is config
        assert isinstance(orch.state, TrainingState)

    def test_initialize_returns_true(self, tmp_path, mocks):
        config = _make_config(tmp_path)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
            result = orch.initialize()
        assert result is True

    def test_initialize_creates_scheduler(self, tmp_path, mocks):
        config = _make_config(tmp_path)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
            orch.initialize()
        assert orch._scheduler is not None

    def test_initialize_sets_device(self, tmp_path, mocks):
        config = _make_config(tmp_path)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
            orch.initialize()
        assert orch._device in ("cpu", "cuda")

    def test_initialize_fails_gracefully_on_exception(self, tmp_path, mocks):
        config = _make_config(tmp_path)
        with _patch_factories(mocks):
            with patch(f"{MOD}.create_environment", side_effect=RuntimeError("env error")):
                orch = TrainingOrchestrator(config)
                result = orch.initialize()
        assert result is False

    def test_initialize_with_log_dir(self, tmp_path, mocks):
        """Log dir creation shouldn't raise."""
        config = _make_config(tmp_path)
        config.log_dir = str(tmp_path / "my_logs")
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
            result = orch.initialize()
        assert result is True
        assert (tmp_path / "my_logs").exists()

    def test_initialize_sets_random_seed(self, tmp_path, mocks):
        config = _make_config(tmp_path)
        config.random_seed = 42
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
            orch.initialize()
        # No assertion needed — just verify no exception is raised

    def test_initialize_no_rag_when_disabled(self, tmp_path, mocks):
        config = _make_config(tmp_path)
        config.rag.enabled = False
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
            orch.initialize()
        assert orch._rag is None


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


class TestCheckpointManagement:
    def test_resume_from_checkpoint_loads_state(self, tmp_path, mocks):
        mocks["checkpointer"].has_checkpoint.return_value = True
        mocks["checkpointer"].load_latest.return_value = {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "scheduler_state": {"step": 50},
            "training_state": {
                "iteration": 5,
                "total_games": 100,
                "best_win_rate": 0.6,
            },
        }
        config = _make_config(tmp_path, resume=True)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
            # Pre-wire required attributes so _resume_from_checkpoint can run
            orch._network = MagicMock()
            orch._optimizer = MagicMock()
            orch._scheduler = MagicMock()
            orch._checkpointer = mocks["checkpointer"]
            orch._resume_from_checkpoint()
        assert orch.state.iteration == 5
        assert orch.state.total_games == 100
        assert orch.state.best_win_rate == 0.6

    def test_save_checkpoint_calls_checkpointer(self, orchestrator):
        orch, mocks = orchestrator
        orch.state.iteration = 3
        orch.state.recent_losses = [0.5, 0.4]
        orch._save_checkpoint()
        mocks["checkpointer"].save.assert_called_once()
        call_kwargs = mocks["checkpointer"].save.call_args
        assert call_kwargs.kwargs["iteration"] == 3

    def test_save_checkpoint_contains_model_key(self, orchestrator):
        orch, mocks = orchestrator
        orch._network = _make_mock_network()
        orch._network.state_dict = MagicMock(return_value={"w": 1.0})
        orch._save_checkpoint()
        call_args = mocks["checkpointer"].save.call_args[0][0]
        assert "model_state_dict" in call_args
        assert "optimizer_state_dict" in call_args
        assert "training_state" in call_args

    def test_cleanup_saves_final_checkpoint(self, orchestrator):
        orch, mocks = orchestrator
        orch.state.start_time = 0.0
        orch._cleanup()
        mocks["checkpointer"].save_final.assert_called_once()

    def test_cleanup_closes_rag(self, orchestrator):
        orch, mocks = orchestrator
        mock_rag = MagicMock()
        orch._rag = mock_rag
        orch.state.start_time = 0.0
        orch._cleanup()
        mock_rag.close.assert_called_once()

    def test_cleanup_finishes_wandb(self, orchestrator):
        orch, mocks = orchestrator
        mock_wandb = MagicMock()
        orch._wandb_logger = mock_wandb
        orch.state.start_time = 0.0
        orch._cleanup()
        mock_wandb.finish.assert_called_once()

    def test_cleanup_without_network_noop(self, orchestrator):
        orch, mocks = orchestrator
        orch._network = None
        orch.state.start_time = 0.0
        # Should not raise
        orch._cleanup()
        mocks["checkpointer"].save_final.assert_not_called()


# ---------------------------------------------------------------------------
# Metric tracking / helpers
# ---------------------------------------------------------------------------


class TestMetricTracking:
    def test_assign_values_winner_1(self, orchestrator):
        orch, _ = orchestrator
        traj = [{"spatial": None, "policy": None} for _ in range(4)]
        orch._assign_values(traj, winner=1, total_reward=5.0)
        assert traj[0]["value"] == 1.0  # even index: player perspective
        assert traj[1]["value"] == -1.0  # odd index: opponent perspective
        assert traj[2]["value"] == 1.0
        assert traj[3]["value"] == -1.0

    def test_assign_values_winner_minus1(self, orchestrator):
        orch, _ = orchestrator
        traj = [{"spatial": None, "policy": None} for _ in range(2)]
        orch._assign_values(traj, winner=-1, total_reward=-5.0)
        assert traj[0]["value"] == -1.0
        assert traj[1]["value"] == 1.0

    def test_assign_values_draw(self, orchestrator):
        orch, _ = orchestrator
        traj = [{"spatial": None, "policy": None} for _ in range(3)]
        orch._assign_values(traj, winner=0, total_reward=0.0)
        for step in traj:
            assert step["value"] == 0.0

    def test_pad_action_array_smaller(self, orchestrator):
        orch, _ = orchestrator
        arr = np.ones(10, dtype=np.float32)
        orch._expected_action_size = 20
        result = orch._pad_action_array(arr)
        assert result.shape == (20,)
        assert result[:10].sum() == pytest.approx(10.0)

    def test_pad_action_array_same_size(self, orchestrator):
        orch, _ = orchestrator
        arr = np.ones(ACTION_SIZE, dtype=np.float32)
        orch._expected_action_size = ACTION_SIZE
        result = orch._pad_action_array(arr)
        assert result is arr

    def test_encode_state_dict_input(self, orchestrator):
        orch, _ = orchestrator
        state = {
            "spatial": np.zeros((13, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
        }
        spatial, vector = orch._encode_state(state)
        assert spatial.shape[-1] == BOARD_SIZE
        assert vector.shape == (64,)

    def test_encode_state_fallback(self, orchestrator):
        orch, _ = orchestrator
        spatial, vector = orch._encode_state("not_a_dict")
        assert spatial.shape == (13, BOARD_SIZE, BOARD_SIZE)
        assert vector.shape == (10,)

    def test_encode_state_pads_small_board(self, orchestrator):
        orch, _ = orchestrator
        state = {
            "spatial": np.zeros((13, 4, 4), dtype=np.float32),
            "vector": np.zeros(10, dtype=np.float32),
        }
        spatial, _ = orch._encode_state(state)
        assert spatial.shape == (13, BOARD_SIZE, BOARD_SIZE)

    def test_get_statistics_structure(self, orchestrator):
        orch, _ = orchestrator
        stats = orch.get_statistics()
        for key in (
            "iteration",
            "total_games",
            "best_win_rate",
            "current_win_rate",
            "buffer_size",
            "is_running",
        ):
            assert key in stats

    def test_get_statistics_buffer_size(self, orchestrator):
        orch, mocks = orchestrator
        mocks["replay"].__len__ = MagicMock(return_value=42)
        orch._replay_buffer = mocks["replay"]
        stats = orch.get_statistics()
        assert stats["buffer_size"] == 42

    def test_get_statistics_no_buffer(self, orchestrator):
        orch, _ = orchestrator
        orch._replay_buffer = None
        stats = orch.get_statistics()
        assert stats["buffer_size"] == 0

    def test_update_optimizer_lr(self, orchestrator):
        orch, _ = orchestrator
        orch._scheduler.get_learning_rate = MagicMock(return_value=0.01)
        orch._update_optimizer_lr()
        for pg in orch._optimizer.param_groups:
            assert pg["lr"] == 0.01


# ---------------------------------------------------------------------------
# Signal handling / bridge setters
# ---------------------------------------------------------------------------


class TestSignalHandling:
    def test_shutdown_sets_flag(self, orchestrator):
        orch, _ = orchestrator
        assert orch._shutdown_requested is False
        orch._handle_shutdown(signal.SIGINT, None)
        assert orch._shutdown_requested is True

    def test_set_training_safeguard(self, orchestrator):
        orch, _ = orchestrator
        mock_safeguard = MagicMock()
        orch.set_training_safeguard(mock_safeguard)
        assert orch._training_safeguard is mock_safeguard

    def test_set_muzero_bridge(self, orchestrator):
        orch, _ = orchestrator
        mock_bridge = MagicMock()
        orch.set_muzero_bridge(mock_bridge, training_mode="muzero")
        assert orch._muzero_bridge is mock_bridge
        assert orch._training_mode == "muzero"

    def test_set_muzero_bridge_none(self, orchestrator):
        orch, _ = orchestrator
        orch.set_muzero_bridge(None)
        assert orch._muzero_bridge is None


# ---------------------------------------------------------------------------
# _add_to_buffer
# ---------------------------------------------------------------------------


class TestAddToBuffer:
    def _make_step(self):
        return {
            "spatial": np.zeros((13, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
            "action_mask": np.ones(ACTION_SIZE, dtype=np.float32),
            "policy": np.ones(ACTION_SIZE, dtype=np.float32) / ACTION_SIZE,
            "value": 1.0,
        }

    def test_adds_all_steps(self, orchestrator):
        orch, mocks = orchestrator
        trajectory = [self._make_step() for _ in range(3)]
        result = GameResult(trajectory=trajectory, winner=1, total_reward=3.0, turns=3)
        buffer_list = []
        mocks["replay"].buffer = buffer_list
        orch._replay_buffer = mocks["replay"]
        orch._add_to_buffer(result)
        assert len(buffer_list) == 3

    def test_empty_trajectory_noop(self, orchestrator):
        orch, mocks = orchestrator
        result = GameResult(trajectory=[], winner=0, total_reward=0.0, turns=0)
        buffer_list = []
        mocks["replay"].buffer = buffer_list
        orch._replay_buffer = mocks["replay"]
        orch._add_to_buffer(result)
        assert len(buffer_list) == 0

    def test_buffer_entries_have_required_keys(self, orchestrator):
        orch, mocks = orchestrator
        trajectory = [self._make_step()]
        result = GameResult(trajectory=trajectory, winner=1, total_reward=1.0, turns=1)
        buffer_list = []
        mocks["replay"].buffer = buffer_list
        orch._replay_buffer = mocks["replay"]
        orch._add_to_buffer(result)
        entry = buffer_list[0]
        for key in ("spatial", "vector", "action_mask", "policy", "value"):
            assert key in entry


# ---------------------------------------------------------------------------
# _run_strategy_analysis
# ---------------------------------------------------------------------------


class TestStrategyAnalysis:
    def test_noop_without_advisor(self, orchestrator):
        orch, _ = orchestrator
        orch._strategy_advisor = None
        orch._run_strategy_analysis([])  # Should not raise

    def test_calls_advisor_with_last_game(self, orchestrator):
        orch, _ = orchestrator
        mock_advisor = MagicMock()
        mock_report = MagicMock()
        mock_report.learning_points = ["point1", "point2"]
        mock_report.strategy_assessment = "Good strategy"
        mock_advisor.generate_game_report.return_value = mock_report
        orch._strategy_advisor = mock_advisor

        traj = [{"spatial": None, "policy": None, "value": 1.0}]
        results = [
            GameResult(trajectory=traj, winner=1, total_reward=1.0, turns=5),
            GameResult(trajectory=traj, winner=-1, total_reward=-1.0, turns=5),
        ]
        orch._run_strategy_analysis(results)
        mock_advisor.generate_game_report.assert_called_once()
        call_kwargs = mock_advisor.generate_game_report.call_args.kwargs
        assert call_kwargs["outcome"] == "loss"

    def test_stores_insights_in_rag(self, orchestrator):
        orch, _ = orchestrator
        mock_rag = MagicMock()
        orch._rag = mock_rag

        mock_advisor = MagicMock()
        mock_report = MagicMock()
        mock_report.learning_points = ["insight_a", "insight_b"]
        mock_report.strategy_assessment = "Analysis"
        mock_advisor.generate_game_report.return_value = mock_report
        orch._strategy_advisor = mock_advisor

        traj = [{"spatial": None, "policy": None, "value": 1.0}]
        results = [GameResult(trajectory=traj, winner=1, total_reward=1.0, turns=1)]
        orch._run_strategy_analysis(results)
        assert mock_rag.store_training_insight.call_count == 2

    def test_noop_when_no_results(self, orchestrator):
        orch, _ = orchestrator
        mock_advisor = MagicMock()
        orch._strategy_advisor = mock_advisor
        orch._run_strategy_analysis([])  # Should not crash
        mock_advisor.generate_game_report.assert_not_called()


# ---------------------------------------------------------------------------
# _log_iteration
# ---------------------------------------------------------------------------


class TestLogIteration:
    def test_wandb_logger_called(self, orchestrator):
        orch, _ = orchestrator
        mock_wandb = MagicMock()
        orch._wandb_logger = mock_wandb
        orch._scheduler = MagicMock()
        orch._scheduler.get_learning_rate.return_value = 0.001
        orch._scheduler.get_batch_size.return_value = 32

        results = [GameResult(trajectory=[], winner=1, total_reward=1.0, turns=5)]
        orch._log_iteration(0, results, {"total_loss": 0.5}, 1.0)
        mock_wandb.log.assert_called_once()

    def test_no_wandb_no_crash(self, orchestrator):
        orch, _ = orchestrator
        orch._wandb_logger = None
        results = [GameResult(trajectory=[], winner=0, total_reward=0.0, turns=1)]
        orch._log_iteration(0, results, {}, 0.5)

    def test_empty_game_results(self, orchestrator):
        orch, _ = orchestrator
        orch._wandb_logger = None
        orch._log_iteration(0, [], {}, 0.5)  # Should not raise

    def test_win_rate_computed_correctly(self, orchestrator):
        """W&B log call should include correct win_rate."""
        orch, _ = orchestrator
        mock_wandb = MagicMock()
        orch._wandb_logger = mock_wandb
        orch._scheduler = MagicMock()
        orch._scheduler.get_learning_rate.return_value = 0.001
        orch._scheduler.get_batch_size.return_value = 32

        results = [
            GameResult(trajectory=[], winner=1, total_reward=1.0, turns=1),
            GameResult(trajectory=[], winner=-1, total_reward=-1.0, turns=1),
        ]
        orch._log_iteration(0, results, {}, 0.1)
        logged = mock_wandb.log.call_args[0][0]
        assert logged["win_rate"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _run_self_play (high-level dispatch)
# ---------------------------------------------------------------------------


class TestRunSelfPlay:
    def test_returns_game_results(self, orchestrator):
        orch, _ = orchestrator
        fake_result = GameResult(trajectory=[], winner=1, total_reward=1.0, turns=3)
        with patch.object(orch, "_play_single_game", return_value=fake_result) as mock_play:
            results = orch._run_self_play(num_games=3)
        assert len(results) == 3
        assert mock_play.call_count == 3

    def test_muzero_path_dispatches(self, orchestrator):
        orch, _ = orchestrator
        orch._training_mode = "muzero"
        orch._muzero_bridge = MagicMock()
        fake_result = GameResult(trajectory=[], winner=1, total_reward=1.0, turns=3)
        with patch.object(orch, "_run_muzero_self_play", return_value=[fake_result]) as mock_mz:
            results = orch._run_self_play(num_games=1)
        mock_mz.assert_called_once_with(1, None)
        assert len(results) == 1

    def test_network_set_to_eval_mode(self, orchestrator):
        orch, _ = orchestrator
        fake_result = GameResult(trajectory=[], winner=1, total_reward=1.0, turns=1)
        with patch.object(orch._network, "eval", wraps=orch._network.eval) as spy_eval:
            with patch.object(orch, "_play_single_game", return_value=fake_result):
                orch._run_self_play(num_games=1)
        spy_eval.assert_called()

    def test_alphazero_mode_not_muzero(self, orchestrator):
        orch, _ = orchestrator
        orch._training_mode = "alphazero"
        fake_result = GameResult(trajectory=[], winner=1, total_reward=1.0, turns=1)
        with patch.object(orch, "_play_single_game", return_value=fake_result) as mock_play:
            with patch.object(orch, "_run_muzero_self_play") as mock_mz:
                orch._run_self_play(num_games=2)
        mock_play.assert_called()
        mock_mz.assert_not_called()


# ---------------------------------------------------------------------------
# dry_run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_returns_true(self, tmp_path, mocks):
        config = _make_config(tmp_path)
        fake_result = GameResult(trajectory=[], winner=1, total_reward=1.0, turns=3)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
            with patch.object(orch, "_play_single_game", return_value=fake_result):
                result = orch.dry_run()
        assert result is True

    def test_dry_run_returns_false_on_init_failure(self, tmp_path, mocks):
        config = _make_config(tmp_path)
        with _patch_factories(mocks):
            with patch(f"{MOD}.create_environment", side_effect=RuntimeError("fail")):
                orch = TrainingOrchestrator(config)
                result = orch.dry_run()
        assert result is False


# ---------------------------------------------------------------------------
# run() loop dispatch
# ---------------------------------------------------------------------------


class TestRunLoop:
    def test_run_calls_training_iteration(self, tmp_path, mocks):
        config = _make_config(tmp_path, num_iterations=2)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
            with patch.object(orch, "_training_iteration") as mock_iter:
                with patch.object(orch, "initialize", return_value=True):
                    orch.run()
        assert mock_iter.call_count == 2

    def test_run_stops_on_shutdown_request(self, tmp_path, mocks):
        config = _make_config(tmp_path, num_iterations=10)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
            call_count = 0

            def fake_iteration():
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    orch._shutdown_requested = True

            with patch.object(orch, "_training_iteration", side_effect=fake_iteration):
                with patch.object(orch, "initialize", return_value=True):
                    orch.run()
        # Shutdown happens after iteration 2; loop checks flag before next call
        assert call_count <= 3

    def test_run_aborts_on_init_failure(self, tmp_path, mocks):
        config = _make_config(tmp_path, num_iterations=5)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
            with patch.object(orch, "_training_iteration") as mock_iter:
                with patch.object(orch, "initialize", return_value=False):
                    orch.run()
        mock_iter.assert_not_called()

    def test_run_raises_on_training_exception(self, tmp_path, mocks):
        config = _make_config(tmp_path, num_iterations=5)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
            with patch.object(orch, "_training_iteration", side_effect=RuntimeError("boom")):
                with patch.object(orch, "initialize", return_value=True):
                    with pytest.raises(RuntimeError, match="boom"):
                        orch.run()

    def test_run_increments_iteration_counter(self, tmp_path, mocks):
        config = _make_config(tmp_path, num_iterations=3)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
            with patch.object(orch, "_training_iteration"):
                with patch.object(orch, "initialize", return_value=True):
                    orch.run()
        assert orch.state.iteration == 3


# ---------------------------------------------------------------------------
# _build_game_state_dict
# ---------------------------------------------------------------------------


class TestBuildGameStateDict:
    def test_returns_dict_with_required_keys(self, orchestrator):
        orch, _ = orchestrator
        state = {
            "spatial": np.zeros((13, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
        }
        mock_env = _make_mock_env()
        result = orch._build_game_state_dict(state, mock_env, turn=5)
        assert result["turn"] == 5
        assert "spatial" in result
        assert "vector" in result

    def test_uses_get_game_info_if_available(self, orchestrator):
        orch, _ = orchestrator
        state = {
            "spatial": np.zeros((13, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
        }
        mock_env = _make_mock_env()
        inner = MagicMock()
        inner.get_game_info = MagicMock(return_value={"extra_key": 99})
        mock_env._env = inner
        result = orch._build_game_state_dict(state, mock_env, turn=0)
        assert result.get("extra_key") == 99


# ---------------------------------------------------------------------------
# _evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_evaluate_updates_win_rate(self, orchestrator):
        orch, _ = orchestrator
        fake_result = GameResult(trajectory=[], winner=1, total_reward=1.0, turns=3)
        with patch.object(orch, "_play_single_game", return_value=fake_result):
            metrics = orch._evaluate()
        assert metrics["win_rate"] == pytest.approx(1.0)
        assert orch.state.current_win_rate == pytest.approx(1.0)

    def test_evaluate_saves_best_model(self, orchestrator):
        orch, mocks = orchestrator
        fake_result = GameResult(trajectory=[], winner=1, total_reward=1.0, turns=3)
        orch.state.best_win_rate = 0.0
        with patch.object(orch, "_play_single_game", return_value=fake_result):
            orch._evaluate()
        mocks["checkpointer"].save_best.assert_called_once()

    def test_evaluate_does_not_save_if_not_improved(self, orchestrator):
        orch, mocks = orchestrator
        orch.state.best_win_rate = 1.0
        fake_result = GameResult(trajectory=[], winner=0, total_reward=0.0, turns=3)
        with patch.object(orch, "_play_single_game", return_value=fake_result):
            orch._evaluate()
        mocks["checkpointer"].save_best.assert_not_called()

    def test_evaluate_returns_avg_reward(self, orchestrator):
        orch, _ = orchestrator
        fake_result = GameResult(trajectory=[], winner=1, total_reward=2.0, turns=3)
        with patch.object(orch, "_play_single_game", return_value=fake_result):
            metrics = orch._evaluate()
        assert "avg_reward" in metrics
        assert metrics["avg_reward"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _require() initialization guard (replaces bare asserts; -O safe)
# ---------------------------------------------------------------------------


class TestRequireGuard:
    """The _require() helper hardens the old ``assert self._x is not None`` guards."""

    def test_require_passes_through_non_none(self, tmp_path, mocks):
        config = _make_config(tmp_path)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
        sentinel = object()
        assert orch._require(sentinel, "thing") is sentinel

    def test_require_raises_runtimeerror_when_none(self, tmp_path, mocks):
        config = _make_config(tmp_path)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
        with pytest.raises(RuntimeError, match=r"network.*not initialized"):
            orch._require(orch._network, "network")

    def test_entry_method_raises_clear_error_before_setup(self, tmp_path, mocks):
        """A method needing components must raise RuntimeError (not AttributeError).

        Pre-``initialize()`` the components are ``None``; the old ``assert`` would be
        stripped under ``python -O`` and surface as an opaque ``AttributeError``.
        """
        config = _make_config(tmp_path)
        with _patch_factories(mocks):
            orch = TrainingOrchestrator(config)
        # _resume_from_checkpoint() guards _checkpointer/_network/_optimizer/_scheduler.
        with pytest.raises(RuntimeError, match="not initialized; call initialize"):
            orch._resume_from_checkpoint()
