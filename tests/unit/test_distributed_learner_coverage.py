"""Additional coverage tests for distributed learner.

Targets uncovered paths in alpha_zero/distributed/learner.py:
- LearningRateScheduler: warmup, milestone decay, get_lr
- GradientAccumulator: partial accumulation, full step, gradient clipping
- Learner: _publish_weights, _save_checkpoint, load_checkpoint, train_step,
           run() loop (max steps, stop signal, buffer wait, logging, publish),
           stop/is_running, get_statistics
- learner_process() entry point
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from alpha_zero.distributed.learner import (
    GradientAccumulator,
    Learner,
    LearnerConfig,
    LearningRateScheduler,
    learner_process,
)
from alpha_zero.distributed.shared_storage import (
    DistributedReplayBuffer,
    ReplayBufferConfig,
    SharedStorage,
    SharedStorageConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def learner_config(tmpdir):
    return LearnerConfig(
        board_size=8,
        num_players=2,
        num_res_blocks=2,
        num_filters=16,
        batch_size=16,
        learning_rate=0.01,
        weight_decay=0.0001,
        gradient_clip=1.0,
        lr_decay_steps=[100, 200],
        lr_decay_rate=0.5,
        gradient_accumulation_steps=1,
        min_buffer_size=10,
        max_training_steps=0,
        publish_interval=5,
        checkpoint_dir=os.path.join(tmpdir, "ckpt"),
        checkpoint_interval=10,
        device="cpu",
        log_interval=5,
    )


@pytest.fixture
def shared_storage():
    cfg = SharedStorageConfig(compress_weights=False)
    s = SharedStorage(config=cfg)
    yield s
    s.shutdown()


@pytest.fixture
def replay_buffer():
    cfg = ReplayBufferConfig(max_size=500, min_size=10)
    b = DistributedReplayBuffer(config=cfg)
    yield b
    b.shutdown()


@pytest.fixture
def learner(learner_config, shared_storage, replay_buffer):
    return Learner(learner_config, shared_storage, replay_buffer)


# ---------------------------------------------------------------------------
# LearningRateScheduler
# ---------------------------------------------------------------------------


class TestLearningRateScheduler:
    def _make_scheduler(self, decay_steps=None, warmup_steps=0, base_lr=0.1):
        param = nn.Parameter(torch.tensor([1.0]))
        opt = optim.SGD([param], lr=base_lr)
        scheduler = LearningRateScheduler(
            opt,
            decay_steps=decay_steps,
            decay_rate=0.5,
            warmup_steps=warmup_steps,
            base_lr=base_lr,
        )
        return scheduler, opt

    def test_step_returns_lr(self):
        sched, _ = self._make_scheduler()
        lr = sched.step()
        assert isinstance(lr, float)
        assert lr == pytest.approx(0.1)

    def test_warmup_increases_lr(self):
        sched, _ = self._make_scheduler(warmup_steps=10, base_lr=0.1)
        # Step 1 of 10 should give lr = 0.1 * 1/10 = 0.01
        lr = sched.step()
        assert lr == pytest.approx(0.01, rel=1e-4)

    def test_warmup_reaches_base_lr_at_end(self):
        sched, _ = self._make_scheduler(warmup_steps=4, base_lr=0.1)
        for _ in range(4):
            lr = sched.step()
        # At step 4 (== warmup_steps), we exit warmup; lr == base_lr
        assert lr == pytest.approx(0.1, rel=1e-4)

    def test_decay_at_milestone(self):
        sched, _ = self._make_scheduler(decay_steps=[3], base_lr=0.1)
        for _ in range(3):
            lr = sched.step()
        # At step 3 >= milestone 3, lr = base_lr * decay_rate = 0.05
        assert lr == pytest.approx(0.05, rel=1e-4)

    def test_multiple_milestones_stack(self):
        sched, _ = self._make_scheduler(decay_steps=[2, 4], base_lr=0.1)
        for _ in range(4):
            lr = sched.step()
        # 0.1 * 0.5 * 0.5 = 0.025
        assert lr == pytest.approx(0.025, rel=1e-4)

    def test_get_lr_returns_current(self):
        sched, _ = self._make_scheduler(base_lr=0.1)
        sched.step()
        assert sched.get_lr() == pytest.approx(0.1)

    def test_no_decay_steps_constant_lr(self):
        sched, _ = self._make_scheduler(decay_steps=[], base_lr=0.2)
        for _ in range(10):
            sched.step()
        assert sched.get_lr() == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# GradientAccumulator
# ---------------------------------------------------------------------------


class TestGradientAccumulator:
    def _make_accumulator(self, steps=1, clip=1.0):
        net = nn.Linear(4, 2)
        opt = optim.SGD(net.parameters(), lr=0.01)
        opt.zero_grad()
        acc = GradientAccumulator(net, opt, accumulation_steps=steps, gradient_clip=clip)
        return acc, net, opt

    def test_single_step_accumulation_triggers_update(self):
        acc, net, opt = self._make_accumulator(steps=1)
        x = torch.randn(2, 4)
        loss = net(x).sum()
        did_step = acc.accumulate(loss)
        assert did_step is True
        assert acc.step_count == 0  # Reset after step

    def test_multi_step_no_update_until_threshold(self):
        acc, net, opt = self._make_accumulator(steps=3)
        for i in range(2):
            x = torch.randn(2, 4)
            loss = net(x).sum()
            did_step = acc.accumulate(loss)
            assert did_step is False

        # Third accumulation should trigger
        x = torch.randn(2, 4)
        loss = net(x).sum()
        did_step = acc.accumulate(loss)
        assert did_step is True

    def test_gradient_clip_applied(self):
        acc, net, opt = self._make_accumulator(steps=1, clip=0.001)
        x = torch.randn(2, 4) * 100  # Large input → large gradients
        loss = net(x).sum()
        acc.accumulate(loss)
        # After clipping, all gradients should have norm <= clip
        total_norm = sum(
            p.grad.norm().item() ** 2 for p in net.parameters() if p.grad is not None
        ) ** 0.5
        assert total_norm <= 0.001 + 1e-6

    def test_gradient_clip_zero_skips_clipping(self):
        acc, net, opt = self._make_accumulator(steps=1, clip=0.0)
        x = torch.randn(2, 4)
        loss = net(x).sum()
        # Should not raise even with clip=0
        acc.accumulate(loss)


# ---------------------------------------------------------------------------
# Learner – unit-level helpers
# ---------------------------------------------------------------------------


class TestLearnerHelpers:
    def test_publish_weights(self, learner, shared_storage):
        learner._publish_weights()
        weights = shared_storage.get_model_weights()
        assert weights is not None

    def test_save_checkpoint_creates_file(self, learner, tmpdir):
        path = learner._save_checkpoint(step=42)
        assert os.path.exists(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["step"] == 42
        assert "network_state_dict" in data
        assert "optimizer_state_dict" in data
        assert "lr_scheduler_step" in data

    def test_load_checkpoint_restores_step(self, learner, tmpdir):
        saved_path = learner._save_checkpoint(step=7)
        step = learner.load_checkpoint(saved_path)
        assert step == 7

    def test_stop_and_is_running(self, learner):
        learner._running = True
        assert learner.is_running() is True
        learner.stop()
        assert learner._running is False
        assert learner.is_running() is False

    def test_get_statistics_initial(self, learner, shared_storage):
        stats = learner.get_statistics()
        assert stats["training_step"] == 0
        assert stats["is_running"] is False
        assert "learning_rate" in stats
        assert "buffer_size" in stats

    def test_get_statistics_after_start(self, learner):
        learner._running = True
        learner.training_step = 5
        import time

        learner.start_time = time.time() - 10
        stats = learner.get_statistics()
        assert stats["training_step"] == 5
        assert stats["steps_per_second"] > 0


# ---------------------------------------------------------------------------
# Learner.train_step
# ---------------------------------------------------------------------------


class TestLearnerTrainStep:
    def _make_batch(self, board_size=8, action_space=50, batch_size=4):
        """Build a minimal fake batch dict with tensors."""
        return {
            "spatial": torch.randn(batch_size, 13, board_size, board_size),
            "vector": torch.randn(batch_size, 64),
            "action_mask": torch.ones(batch_size, action_space),
            "policy": torch.softmax(torch.randn(batch_size, action_space), dim=-1),
            "value": torch.rand(batch_size, 1) * 2 - 1,
        }

    def test_train_step_returns_none_when_buffer_not_ready(self, learner):
        learner.replay_buffer.is_ready = MagicMock(return_value=False)
        result = learner.train_step()
        assert result is None

    def test_train_step_returns_none_when_sample_empty(self, learner):
        learner.replay_buffer.is_ready = MagicMock(return_value=True)
        learner.replay_buffer.sample_batch = MagicMock(return_value=None)
        result = learner.train_step()
        assert result is None

    def test_train_step_returns_losses(self, learner):
        batch = self._make_batch(
            board_size=learner.config.board_size,
            action_space=learner.action_space_size,
            batch_size=4,
        )
        learner.replay_buffer.is_ready = MagicMock(return_value=True)
        learner.replay_buffer.sample_batch = MagicMock(return_value=batch)
        learner.replay_buffer.process_pending_trajectories = MagicMock()

        result = learner.train_step()
        assert result is not None
        assert "total_loss" in result
        assert "policy_loss" in result
        assert "value_loss" in result
        assert "learning_rate" in result

    def test_train_step_increments_training_step(self, learner):
        batch = self._make_batch(
            board_size=learner.config.board_size,
            action_space=learner.action_space_size,
            batch_size=4,
        )
        learner.replay_buffer.is_ready = MagicMock(return_value=True)
        learner.replay_buffer.sample_batch = MagicMock(return_value=batch)
        learner.replay_buffer.process_pending_trajectories = MagicMock()

        initial_step = learner.training_step
        learner.train_step()
        assert learner.training_step == initial_step + 1

    def test_train_step_with_gradient_accumulation(self, learner_config, shared_storage, replay_buffer):
        """With accumulation_steps=2, training_step only increments every 2 calls."""
        learner_config.gradient_accumulation_steps = 2
        learner = Learner(learner_config, shared_storage, replay_buffer)

        def make_batch(*args, **kwargs):
            return {
                "spatial": torch.randn(4, 13, 8, 8),
                "vector": torch.randn(4, 64),
                "action_mask": torch.ones(4, learner.action_space_size),
                "policy": torch.softmax(torch.randn(4, learner.action_space_size), dim=-1),
                "value": torch.rand(4, 1) * 2 - 1,
            }

        learner.replay_buffer.is_ready = MagicMock(return_value=True)
        learner.replay_buffer.sample_batch = MagicMock(side_effect=make_batch)
        learner.replay_buffer.process_pending_trajectories = MagicMock()

        # First step: no optimizer update yet
        learner.train_step()
        assert learner.training_step == 0

        # Second step: optimizer fires
        learner.train_step()
        assert learner.training_step == 1


# ---------------------------------------------------------------------------
# Learner.run – controlled loop
# ---------------------------------------------------------------------------


class TestLearnerRun:
    def _fake_batch(self, learner):
        return {
            "spatial": torch.randn(4, 13, 8, 8),
            "vector": torch.randn(4, 64),
            "action_mask": torch.ones(4, learner.action_space_size),
            "policy": torch.softmax(torch.randn(4, learner.action_space_size), dim=-1),
            "value": torch.rand(4, 1) * 2 - 1,
        }

    def test_run_stops_on_max_training_steps(self, learner):
        learner.config.max_training_steps = 3
        learner.config.log_interval = 1
        learner.config.publish_interval = 1
        learner.config.checkpoint_interval = 0  # Disable checkpoint during run

        call_count = {"n": 0}

        def fake_train_step():
            call_count["n"] += 1
            learner.training_step += 1
            return {
                "total_loss": 0.5,
                "policy_loss": 0.3,
                "value_loss": 0.2,
                "learning_rate": 0.001,
            }

        learner.replay_buffer.is_ready = MagicMock(return_value=True)
        learner.replay_buffer.process_pending_trajectories = MagicMock()

        with patch.object(learner, "train_step", side_effect=fake_train_step):
            with patch.object(learner, "_publish_weights"):
                with patch.object(learner, "_save_checkpoint"):
                    learner.run()

        assert learner.training_step >= 3
        assert learner._running is False

    def test_run_waits_for_buffer_then_stops(self, learner):
        """When buffer never becomes ready and stop is requested, run exits."""
        learner.replay_buffer.is_ready = MagicMock(return_value=False)
        learner.replay_buffer.process_pending_trajectories = MagicMock()
        learner.replay_buffer.size = MagicMock(return_value=0)

        # Simulate stop signal after one wait iteration
        call_count = {"n": 0}

        def fake_should_stop():
            call_count["n"] += 1
            return call_count["n"] > 1

        learner.storage.should_stop = MagicMock(side_effect=fake_should_stop)

        with patch("alpha_zero.distributed.learner.time.sleep"):
            learner.run()

        assert learner._running is False

    def test_run_publishes_weights_at_interval(self, learner):
        learner.config.max_training_steps = 10
        learner.config.publish_interval = 1
        learner.config.checkpoint_interval = 0
        learner.config.log_interval = 100

        call_count = {"n": 0}

        def fake_train_step():
            call_count["n"] += 1
            learner.training_step += 1
            return {
                "total_loss": 0.1,
                "policy_loss": 0.05,
                "value_loss": 0.05,
                "learning_rate": 0.001,
            }

        learner.replay_buffer.is_ready = MagicMock(return_value=True)
        learner.replay_buffer.process_pending_trajectories = MagicMock()

        publish_calls = []
        with patch.object(learner, "train_step", side_effect=fake_train_step):
            with patch.object(learner, "_publish_weights", side_effect=lambda: publish_calls.append(1)):
                with patch.object(learner, "_save_checkpoint"):
                    learner.run()

        # Should have published weights at least once
        assert len(publish_calls) > 0

    def test_run_checkpoints_at_interval(self, learner, tmpdir):
        learner.config.max_training_steps = 10
        learner.config.publish_interval = 100
        learner.config.checkpoint_interval = 5
        learner.config.log_interval = 100

        call_count = {"n": 0}

        def fake_train_step():
            call_count["n"] += 1
            learner.training_step += 1
            return {
                "total_loss": 0.1,
                "policy_loss": 0.05,
                "value_loss": 0.05,
                "learning_rate": 0.001,
            }

        learner.replay_buffer.is_ready = MagicMock(return_value=True)
        learner.replay_buffer.process_pending_trajectories = MagicMock()

        ckpt_calls = []
        with patch.object(learner, "train_step", side_effect=fake_train_step):
            with patch.object(learner, "_publish_weights"):
                with patch.object(
                    learner, "_save_checkpoint", side_effect=lambda s: ckpt_calls.append(s)
                ):
                    learner.run()

        # At steps 5, 10, and final → at least 2 checkpoint saves
        assert len(ckpt_calls) >= 2

    def test_run_logs_at_interval(self, learner):
        learner.config.max_training_steps = 10
        learner.config.log_interval = 5
        learner.config.publish_interval = 100
        learner.config.checkpoint_interval = 0

        call_count = {"n": 0}

        def fake_train_step():
            call_count["n"] += 1
            learner.training_step += 1
            return {
                "total_loss": 0.1,
                "policy_loss": 0.05,
                "value_loss": 0.05,
                "learning_rate": 0.001,
            }

        learner.replay_buffer.is_ready = MagicMock(return_value=True)
        learner.replay_buffer.process_pending_trajectories = MagicMock()

        log_calls = []
        with patch.object(learner, "train_step", side_effect=fake_train_step):
            with patch.object(learner, "_publish_weights"):
                with patch.object(learner, "_save_checkpoint"):
                    with patch.object(
                        learner.storage,
                        "add_loss_entry",
                        side_effect=lambda entry: log_calls.append(entry),
                    ):
                        learner.run()

        # Verify add_loss_entry was called (storage logging path)
        assert len(log_calls) >= 1

    def test_run_handles_train_step_none(self, learner):
        """When train_step returns None, learner sleeps and continues."""
        learner.config.max_training_steps = 0
        learner.replay_buffer.is_ready = MagicMock(return_value=True)
        learner.replay_buffer.process_pending_trajectories = MagicMock()

        none_count = {"n": 0}

        def fake_train_step():
            none_count["n"] += 1
            if none_count["n"] > 3:
                learner._running = False
            return None

        with patch.object(learner, "train_step", side_effect=fake_train_step):
            with patch("alpha_zero.distributed.learner.time.sleep") as mock_sleep:
                learner.run()

        assert mock_sleep.call_count > 0


# ---------------------------------------------------------------------------
# learner_process entry point
# ---------------------------------------------------------------------------


class TestLearnerProcess:
    def test_learner_process_runs_and_exits(self, learner_config, shared_storage, replay_buffer):
        """learner_process should create Learner and call run()."""
        config_dict = {
            "board_size": learner_config.board_size,
            "num_players": learner_config.num_players,
            "num_res_blocks": learner_config.num_res_blocks,
            "num_filters": learner_config.num_filters,
            "batch_size": learner_config.batch_size,
            "learning_rate": learner_config.learning_rate,
            "weight_decay": learner_config.weight_decay,
            "gradient_clip": learner_config.gradient_clip,
            "min_buffer_size": learner_config.min_buffer_size,
            "max_training_steps": 2,
            "checkpoint_dir": learner_config.checkpoint_dir,
            "checkpoint_interval": 0,
            "device": "cpu",
            "log_interval": 1,
            "publish_interval": 1,
        }

        with patch("alpha_zero.distributed.learner.Learner") as mock_learner_cls:
            mock_learner = MagicMock()
            mock_learner_cls.return_value = mock_learner

            learner_process(config_dict, shared_storage, replay_buffer)

            mock_learner.run.assert_called_once()

    def test_learner_process_handles_exception(self, learner_config, shared_storage, replay_buffer):
        """learner_process should not re-raise exceptions from run()."""
        config_dict = {
            "board_size": learner_config.board_size,
            "num_players": learner_config.num_players,
            "num_res_blocks": learner_config.num_res_blocks,
            "num_filters": learner_config.num_filters,
            "batch_size": learner_config.batch_size,
            "learning_rate": learner_config.learning_rate,
            "weight_decay": learner_config.weight_decay,
            "gradient_clip": learner_config.gradient_clip,
            "min_buffer_size": learner_config.min_buffer_size,
            "max_training_steps": 0,
            "checkpoint_dir": learner_config.checkpoint_dir,
            "checkpoint_interval": 0,
            "device": "cpu",
            "log_interval": 1,
            "publish_interval": 1,
        }

        with patch("alpha_zero.distributed.learner.Learner") as mock_learner_cls:
            mock_learner = MagicMock()
            mock_learner.run.side_effect = RuntimeError("simulated crash")
            mock_learner_cls.return_value = mock_learner

            # Should not raise
            learner_process(config_dict, shared_storage, replay_buffer)
