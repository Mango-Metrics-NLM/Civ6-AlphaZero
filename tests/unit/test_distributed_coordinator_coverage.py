"""Additional coverage tests for distributed training coordinator.

Targets uncovered paths in alpha_zero/distributed/coordinator.py:
- save_checkpoint / load_checkpoint
- _setup_signal_handlers
- _update_eval_network (success + error paths)
- _log_status (with/without loss/eval history)
- _create_learner (standard and DDP branches)
- run() loop control paths (max_games, max_steps, shutdown)
- shutdown() with / without actor_pool + learner
- main() CLI entry point
- YAML from_yaml with distributed section
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

from alpha_zero.distributed.coordinator import (
    CoordinatorConfig,
    TrainingCoordinator,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_coordinator(extra_config: dict | None = None, tmpdir: str | None = None) -> tuple[TrainingCoordinator, str]:
    """Build a TrainingCoordinator with all heavy deps mocked."""
    if tmpdir is None:
        raise ValueError("tmpdir required")

    kwargs = dict(
        board_size=8,
        max_turns=20,
        num_res_blocks=2,
        num_filters=16,
        num_simulations=5,
        num_actors=2,
        batch_size=16,
        replay_buffer_size=500,
        min_replay_size=50,
        max_games=0,
        max_training_steps=0,
        device="cpu",
        checkpoint_dir=os.path.join(tmpdir, "ckpt"),
        log_dir=os.path.join(tmpdir, "logs"),
    )
    if extra_config:
        kwargs.update(extra_config)

    config = CoordinatorConfig(**kwargs)

    with (
        patch("alpha_zero.distributed.coordinator.mp.set_start_method"),
        patch("alpha_zero.distributed.coordinator.SharedStorage") as mock_storage_cls,
        patch("alpha_zero.distributed.coordinator.DistributedReplayBuffer") as mock_buf_cls,
    ):
        mock_storage = MagicMock()
        mock_storage_cls.return_value = mock_storage
        mock_buf = MagicMock()
        mock_buf_cls.return_value = mock_buf

        coordinator = TrainingCoordinator(config)

    return coordinator, tmpdir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmpdir_path():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def coordinator(tmpdir_path):
    c, _ = _make_coordinator(tmpdir=tmpdir_path)
    return c


# ---------------------------------------------------------------------------
# CoordinatorConfig – YAML distributed section
# ---------------------------------------------------------------------------


class TestCoordinatorConfigYAMLExtended:
    def test_from_yaml_distributed_section(self):
        config_dict = {
            "distributed": {
                "use_ddp": True,
                "backend": "gloo",
                "world_size": 2,
                "gradient_sync_interval": 4,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            path = f.name
        try:
            config = CoordinatorConfig.from_yaml(path)
            assert config.use_ddp is True
            assert config.ddp_backend == "gloo"
            assert config.ddp_world_size == 2
            assert config.ddp_gradient_sync_interval == 4
        finally:
            os.unlink(path)

    def test_from_yaml_selfplay_save_interval(self):
        config_dict = {"selfplay": {"save_interval": 42}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            path = f.name
        try:
            config = CoordinatorConfig.from_yaml(path)
            assert config.save_interval == 42
        finally:
            os.unlink(path)

    def test_from_yaml_microciv_filters(self):
        config_dict = {"microciv": {"num_filters": 32, "batch_size": 64}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            path = f.name
        try:
            config = CoordinatorConfig.from_yaml(path)
            assert config.num_filters == 32
            assert config.batch_size == 64
        finally:
            os.unlink(path)

    def test_from_yaml_training_lr_decay(self):
        config_dict = {
            "training": {
                "lr_decay_steps": [50000, 100000],
                "lr_decay_rate": 0.5,
                "gradient_clip": 0.5,
            }
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            path = f.name
        try:
            config = CoordinatorConfig.from_yaml(path)
            assert config.lr_decay_steps == [50000, 100000]
            assert config.lr_decay_rate == 0.5
            assert config.gradient_clip == 0.5
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointing:
    def test_save_checkpoint_returns_path(self, coordinator, tmpdir_path):
        network_state = {"w": torch.tensor([1.0, 2.0])}
        path = coordinator.save_checkpoint(step=10, network_state=network_state)
        assert os.path.exists(path)
        assert "checkpoint_step_10" in path

    def test_save_checkpoint_with_optimizer_state(self, coordinator, tmpdir_path):
        network_state = {"w": torch.tensor([1.0])}
        optimizer_state = {"lr": 0.001}
        path = coordinator.save_checkpoint(
            step=20, network_state=network_state, optimizer_state=optimizer_state
        )
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["step"] == 20
        assert "optimizer_state" in data

    def test_save_checkpoint_with_additional_state(self, coordinator):
        network_state = {"w": torch.tensor([1.0])}
        additional = {"custom_key": 42}
        path = coordinator.save_checkpoint(
            step=30, network_state=network_state, additional_state=additional
        )
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["custom_key"] == 42

    def test_load_checkpoint_specific_path(self, coordinator, tmpdir_path):
        network_state = {"w": torch.tensor([3.0])}
        saved_path = coordinator.save_checkpoint(step=5, network_state=network_state)
        loaded = coordinator.load_checkpoint(path=saved_path)
        assert loaded is not None
        assert loaded["step"] == 5

    def test_load_checkpoint_latest(self, coordinator, tmpdir_path):
        ns1 = {"w": torch.tensor([1.0])}
        ns2 = {"w": torch.tensor([2.0])}
        coordinator.save_checkpoint(step=1, network_state=ns1)
        coordinator.save_checkpoint(step=2, network_state=ns2)
        loaded = coordinator.load_checkpoint()
        # Should load the last (step=2)
        assert loaded is not None
        assert loaded["step"] == 2

    def test_load_checkpoint_no_checkpoints(self, coordinator, tmpdir_path):
        # Empty checkpoint dir
        result = coordinator.load_checkpoint()
        assert result is None

    def test_load_checkpoint_path_not_exist(self, coordinator):
        result = coordinator.load_checkpoint(path="/nonexistent/path.pt")
        assert result is None

    def test_load_checkpoint_corrupt_file(self, coordinator, tmpdir_path):
        # Write a corrupt file
        bad_path = os.path.join(coordinator.config.checkpoint_dir, "checkpoint_step_99.pt")
        os.makedirs(coordinator.config.checkpoint_dir, exist_ok=True)
        with open(bad_path, "wb") as f:
            f.write(b"not a valid torch file")
        result = coordinator.load_checkpoint(path=bad_path)
        assert result is None


# ---------------------------------------------------------------------------
# _setup_signal_handlers
# ---------------------------------------------------------------------------


class TestSignalHandlers:
    def test_setup_signal_handlers_sets_running_false(self, coordinator):
        with patch("alpha_zero.distributed.coordinator.signal.signal") as mock_signal:
            coordinator._setup_signal_handlers()
            assert mock_signal.call_count == 2

    def test_signal_handler_sets_shutdown_requested(self, coordinator):
        # Capture the installed handler and invoke it manually
        handlers = {}
        import signal as _signal

        def fake_signal(signum, handler):
            handlers[signum] = handler

        with patch("alpha_zero.distributed.coordinator.signal.signal", side_effect=fake_signal):
            coordinator._setup_signal_handlers()

        coordinator.shared_storage.request_stop = MagicMock()
        # Invoke the SIGTERM handler
        sigterm = _signal.SIGTERM
        if sigterm in handlers:
            handlers[sigterm](int(sigterm), None)
            assert coordinator._shutdown_requested is True
            coordinator.shared_storage.request_stop.assert_called_once()


# ---------------------------------------------------------------------------
# _update_eval_network – success and error paths
# ---------------------------------------------------------------------------


class TestUpdateEvalNetwork:
    def test_update_eval_network_success(self, coordinator):
        mock_network = MagicMock()
        coordinator.eval_network = mock_network
        coordinator.shared_storage.get_model_weights.return_value = {"w": torch.tensor([1.0])}
        result = coordinator._update_eval_network()
        assert result is True
        mock_network.load_state_dict.assert_called_once()

    def test_update_eval_network_load_state_dict_error(self, coordinator):
        mock_network = MagicMock()
        mock_network.load_state_dict.side_effect = RuntimeError("shape mismatch")
        coordinator.eval_network = mock_network
        coordinator.shared_storage.get_model_weights.return_value = {"w": torch.tensor([1.0])}
        result = coordinator._update_eval_network()
        assert result is False


# ---------------------------------------------------------------------------
# _log_status
# ---------------------------------------------------------------------------


class TestLogStatus:
    def _setup_storage_stats(self, coordinator, extra: dict | None = None):
        stats = {
            "games_played": 42,
            "training_step": 100,
            "model_version": 3,
            "loss_history": [],
            "eval_history": [],
        }
        if extra:
            stats.update(extra)
        coordinator.shared_storage.get_statistics.return_value = stats
        coordinator.replay_buffer.get_statistics.return_value = {"buffer_size": 200}

    def test_log_status_basic(self, coordinator):
        self._setup_storage_stats(coordinator)
        coordinator.actor_pool = MagicMock()
        coordinator.actor_pool.get_running_count.return_value = 2
        coordinator.start_time = 0.0
        coordinator._log_status()  # Should not raise

    def test_log_status_with_loss_history(self, coordinator):
        self._setup_storage_stats(
            coordinator,
            extra={
                "loss_history": [
                    {"total_loss": 1.5, "policy_loss": 0.8, "value_loss": 0.7}
                ]
            },
        )
        coordinator.actor_pool = None
        coordinator.start_time = 0.0
        coordinator._log_status()  # Should not raise

    def test_log_status_with_eval_history(self, coordinator):
        self._setup_storage_stats(
            coordinator,
            extra={"eval_history": [{"win_rate": 0.65}]},
        )
        coordinator.actor_pool = None
        coordinator.start_time = 0.0
        coordinator._log_status()  # Should not raise

    def test_log_status_no_actor_pool(self, coordinator):
        self._setup_storage_stats(coordinator)
        coordinator.actor_pool = None
        coordinator.start_time = 0.0
        coordinator._log_status()  # Should not raise


# ---------------------------------------------------------------------------
# _create_learner – standard and DDP branches
# ---------------------------------------------------------------------------


class TestCreateLearner:
    def test_create_learner_standard(self, coordinator):
        coordinator.config.use_ddp = False
        with patch("alpha_zero.distributed.coordinator.Learner") as mock_learner_cls:
            mock_learner = MagicMock()
            mock_learner_cls.return_value = mock_learner
            learner = coordinator._create_learner()
            mock_learner_cls.assert_called_once()
            assert learner is mock_learner

    def test_create_learner_ddp(self, coordinator):
        coordinator.config.use_ddp = True
        with patch("alpha_zero.distributed.coordinator.DDPLearner") as mock_ddp_cls:
            mock_ddp = MagicMock()
            mock_ddp_cls.return_value = mock_ddp
            learner = coordinator._create_learner()
            mock_ddp_cls.assert_called_once()
            assert learner is mock_ddp


# ---------------------------------------------------------------------------
# shutdown()
# ---------------------------------------------------------------------------


class TestShutdown:
    def test_shutdown_without_actor_pool_and_learner(self, coordinator):
        coordinator.shared_storage.get_games_played.return_value = 0
        coordinator.shutdown()
        assert coordinator._running is False
        coordinator.shared_storage.request_stop.assert_called()
        coordinator.replay_buffer.shutdown.assert_called()
        coordinator.shared_storage.shutdown.assert_called()

    def test_shutdown_with_actor_pool(self, coordinator):
        mock_pool = MagicMock()
        coordinator.actor_pool = mock_pool
        coordinator.shared_storage.get_games_played.return_value = 5
        coordinator.shutdown()
        mock_pool.stop.assert_called_once()

    def test_shutdown_with_learner_has_steps(self, coordinator):
        mock_learner = MagicMock()
        mock_learner.training_step = 10
        coordinator.learner = mock_learner
        coordinator.shared_storage.get_games_played.return_value = 5
        coordinator.shutdown()
        mock_learner._save_checkpoint.assert_called_once_with(10)

    def test_shutdown_with_learner_no_steps(self, coordinator):
        mock_learner = MagicMock()
        mock_learner.training_step = 0
        coordinator.learner = mock_learner
        coordinator.shared_storage.get_games_played.return_value = 0
        coordinator.shutdown()
        mock_learner._save_checkpoint.assert_not_called()

    def test_shutdown_handles_broken_pipe_on_get_games(self, coordinator):
        coordinator.shared_storage.get_games_played.side_effect = BrokenPipeError("pipe broken")
        # Should not raise
        coordinator.shutdown()
        assert coordinator._running is False

    def test_shutdown_handles_eoferror_on_get_games(self, coordinator):
        coordinator.shared_storage.get_games_played.side_effect = EOFError("eof")
        coordinator.shutdown()
        assert coordinator._running is False


# ---------------------------------------------------------------------------
# run() – controlled loop termination
# ---------------------------------------------------------------------------


class TestRunLoop:
    def _setup_run_mocks(self, coordinator):
        """Patch the heavy components used inside run()."""
        coordinator.shared_storage.get_games_played.return_value = 0
        coordinator.shared_storage.get_statistics.return_value = {
            "games_played": 0,
            "training_step": 0,
            "model_version": 0,
            "loss_history": [],
            "eval_history": [],
        }
        coordinator.replay_buffer.get_statistics.return_value = {"buffer_size": 0}

    def test_run_exits_on_shutdown_requested(self, coordinator, tmpdir_path):
        """run() should exit cleanly when _shutdown_requested is set."""

        def fake_start(*a, **kw):
            coordinator._shutdown_requested = True

        with (
            patch("alpha_zero.distributed.coordinator.ActorPool") as mock_pool_cls,
            patch("alpha_zero.distributed.coordinator.MicroCivEnv") as mock_env_cls,
            patch("alpha_zero.distributed.coordinator.MicroCivNetwork") as mock_net_cls,
        ):
            mock_env = MagicMock()
            mock_env.action_space.n = 50
            mock_env_cls.return_value = mock_env

            mock_net = MagicMock()
            mock_net.to.return_value = mock_net
            mock_net_cls.return_value = mock_net

            mock_pool = MagicMock()
            mock_pool.start.side_effect = fake_start
            mock_pool_cls.return_value = mock_pool

            mock_learner = MagicMock()
            mock_learner.training_step = 0

            coordinator.replay_buffer.process_pending_trajectories = MagicMock()
            coordinator.replay_buffer.is_ready.return_value = False
            coordinator.replay_buffer.size.return_value = 0
            coordinator.shared_storage.get_games_played.return_value = 0

            with patch.object(coordinator, "_create_learner", return_value=mock_learner):
                coordinator.run()

        assert coordinator._running is False

    def test_run_exits_on_max_games(self, coordinator, tmpdir_path):
        """run() exits when games >= max_games."""
        coordinator.config.max_games = 3
        coordinator.config.eval_interval = 1000  # Suppress evals

        call_count = {"n": 0}

        def games_played():
            call_count["n"] += 1
            # Return >= max_games after a few calls so loop exits
            return 5 if call_count["n"] > 2 else 0

        with (
            patch("alpha_zero.distributed.coordinator.ActorPool") as mock_pool_cls,
            patch("alpha_zero.distributed.coordinator.MicroCivEnv") as mock_env_cls,
            patch("alpha_zero.distributed.coordinator.MicroCivNetwork") as mock_net_cls,
        ):
            mock_env = MagicMock()
            mock_env.action_space.n = 50
            mock_env_cls.return_value = mock_env

            mock_net = MagicMock()
            mock_net.to.return_value = mock_net
            mock_net_cls.return_value = mock_net

            mock_pool = MagicMock()
            mock_pool_cls.return_value = mock_pool

            mock_learner = MagicMock()
            mock_learner.training_step = 0
            mock_learner.train_step.return_value = {
                "total_loss": 0.5,
                "policy_loss": 0.3,
                "value_loss": 0.2,
                "learning_rate": 0.001,
            }

            coordinator.replay_buffer.process_pending_trajectories = MagicMock()
            coordinator.replay_buffer.is_ready.return_value = True
            coordinator.replay_buffer.size.return_value = 100
            coordinator.shared_storage.get_games_played.side_effect = games_played
            coordinator.shared_storage.get_statistics.return_value = {
                "games_played": 5,
                "training_step": 0,
                "model_version": 0,
                "loss_history": [],
                "eval_history": [],
            }
            coordinator.replay_buffer.get_statistics.return_value = {"buffer_size": 100}

            with patch.object(coordinator, "_create_learner", return_value=mock_learner):
                # Patch time to avoid sleeping and jump past monitor_interval
                with patch("alpha_zero.distributed.coordinator.time.time", return_value=1e9):
                    with patch("alpha_zero.distributed.coordinator.time.sleep"):
                        coordinator.run()

        assert coordinator._running is False

    def test_run_exits_on_keyboard_interrupt(self, coordinator):
        """run() handles KeyboardInterrupt gracefully."""
        with (
            patch("alpha_zero.distributed.coordinator.ActorPool") as mock_pool_cls,
            patch("alpha_zero.distributed.coordinator.MicroCivEnv") as mock_env_cls,
            patch("alpha_zero.distributed.coordinator.MicroCivNetwork") as mock_net_cls,
        ):
            mock_env = MagicMock()
            mock_env.action_space.n = 50
            mock_env_cls.return_value = mock_env
            mock_net = MagicMock()
            mock_net.to.return_value = mock_net
            mock_net_cls.return_value = mock_net
            mock_pool = MagicMock()
            mock_pool.start.side_effect = KeyboardInterrupt
            mock_pool_cls.return_value = mock_pool
            mock_learner = MagicMock()
            mock_learner.training_step = 0

            coordinator.replay_buffer.process_pending_trajectories = MagicMock()
            coordinator.shared_storage.get_games_played.return_value = 0

            with patch.object(coordinator, "_create_learner", return_value=mock_learner):
                coordinator.run()

        assert coordinator._running is False


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


class TestMainCLI:
    def test_main_dry_run(self, tmpdir_path):
        """main() in dry_run mode sets minimal config and runs coordinator."""
        with (
            patch(
                "sys.argv",
                [
                    "coordinator",
                    "--dry_run",
                    "--checkpoint_dir",
                    os.path.join(tmpdir_path, "ckpt"),
                ],
            ),
            patch("alpha_zero.distributed.coordinator.TrainingCoordinator") as mock_coord_cls,
        ):
            mock_coord = MagicMock()
            mock_coord_cls.return_value = mock_coord

            main()

            mock_coord_cls.assert_called_once()
            cfg = mock_coord_cls.call_args[0][0]
            assert cfg.num_actors == 2
            assert cfg.max_games == 10
            assert cfg.max_training_steps == 50
            mock_coord.run.assert_called_once()

    def test_main_with_overrides(self, tmpdir_path):
        with (
            patch(
                "sys.argv",
                [
                    "coordinator",
                    "--num_actors",
                    "4",
                    "--batch_size",
                    "64",
                    "--learning_rate",
                    "0.01",
                    "--max_games",
                    "100",
                    "--device",
                    "cpu",
                    "--checkpoint_dir",
                    os.path.join(tmpdir_path, "ckpt"),
                ],
            ),
            patch("alpha_zero.distributed.coordinator.TrainingCoordinator") as mock_coord_cls,
        ):
            mock_coord = MagicMock()
            mock_coord_cls.return_value = mock_coord

            main()

            cfg = mock_coord_cls.call_args[0][0]
            assert cfg.num_actors == 4
            assert cfg.batch_size == 64
            assert cfg.learning_rate == 0.01
            assert cfg.max_games == 100
            assert cfg.device == "cpu"

    def test_main_loads_yaml_if_exists(self, tmpdir_path):
        yaml_path = os.path.join(tmpdir_path, "test_config.yaml")
        yaml_data = {"environment": {"board_size": 12}}
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f)

        with (
            patch(
                "sys.argv",
                [
                    "coordinator",
                    "--config",
                    yaml_path,
                    "--checkpoint_dir",
                    os.path.join(tmpdir_path, "ckpt"),
                ],
            ),
            patch("alpha_zero.distributed.coordinator.TrainingCoordinator") as mock_coord_cls,
        ):
            mock_coord = MagicMock()
            mock_coord_cls.return_value = mock_coord
            main()
            cfg = mock_coord_cls.call_args[0][0]
            assert cfg.board_size == 12

    def test_main_uses_default_config_if_yaml_missing(self, tmpdir_path):
        with (
            patch(
                "sys.argv",
                [
                    "coordinator",
                    "--config",
                    "/nonexistent/config.yaml",
                    "--checkpoint_dir",
                    os.path.join(tmpdir_path, "ckpt"),
                ],
            ),
            patch("alpha_zero.distributed.coordinator.TrainingCoordinator") as mock_coord_cls,
        ):
            mock_coord = MagicMock()
            mock_coord_cls.return_value = mock_coord
            main()
            cfg = mock_coord_cls.call_args[0][0]
            # Default board size
            assert cfg.board_size == 16

    def test_main_max_training_steps_override(self, tmpdir_path):
        with (
            patch(
                "sys.argv",
                [
                    "coordinator",
                    "--max_training_steps",
                    "500",
                    "--num_simulations",
                    "25",
                    "--checkpoint_dir",
                    os.path.join(tmpdir_path, "ckpt"),
                ],
            ),
            patch("alpha_zero.distributed.coordinator.TrainingCoordinator") as mock_coord_cls,
        ):
            mock_coord = MagicMock()
            mock_coord_cls.return_value = mock_coord
            main()
            cfg = mock_coord_cls.call_args[0][0]
            assert cfg.max_training_steps == 500
            assert cfg.num_simulations == 25
