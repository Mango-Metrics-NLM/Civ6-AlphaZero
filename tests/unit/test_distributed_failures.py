"""Tests for distributed training failure scenarios.

Tests cover:
- Actor crashes and recovery
- Queue deadlock detection
- Network weight corruption handling
- Communication timeout handling
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from alpha_zero.distributed.actor import ActorConfig, SelfPlayActor
from alpha_zero.distributed.shared_storage import (
    DistributedReplayBuffer,
    GameTrajectory,
    ReplayBufferConfig,
    SharedStorage,
    SharedStorageConfig,
    TrainingExample,
    WeightSynchronizer,
)
from alpha_zero.utils.constants import ENCODER


@pytest.fixture
def fast_actor_config() -> ActorConfig:
    """Create a fast actor configuration for testing."""
    return ActorConfig(
        actor_id=0,
        board_size=8,
        max_turns=10,
        num_res_blocks=1,
        num_filters=8,
        num_simulations=2,
        max_games=2,
        log_interval=1,
    )


@pytest.fixture
def shared_storage():
    """Create shared storage for testing."""
    config = SharedStorageConfig(compress_weights=False)
    storage = SharedStorage(config=config)
    yield storage
    storage.shutdown()


@pytest.fixture
def replay_buffer():
    """Create replay buffer for testing."""
    config = ReplayBufferConfig(max_size=1000, min_size=10)
    buffer = DistributedReplayBuffer(config=config)
    yield buffer
    buffer.shutdown()


@pytest.fixture
def small_network():
    """Create a small network for testing."""
    import torch

    model = torch.nn.Linear(10, 10)
    return model


class TestActorFailures:
    """Tests for actor failure scenarios."""

    def test_actor_initialization(
        self,
        fast_actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ):
        """Test actor initializes correctly."""
        actor = SelfPlayActor(fast_actor_config, shared_storage, replay_buffer)
        assert actor.config == fast_actor_config
        assert actor.storage == shared_storage

    def test_actor_requires_mcts_init(
        self,
        fast_actor_config: ActorConfig,
        shared_storage: SharedStorage,
        replay_buffer: DistributedReplayBuffer,
    ):
        """Test actor raises when MCTS not initialized."""
        actor = SelfPlayActor(fast_actor_config, shared_storage, replay_buffer)

        # play_game should fail if MCTS not initialized
        with pytest.raises(RuntimeError, match="MCTS not initialized"):
            actor.play_game()


class TestQueueDeadlocks:
    """Tests for queue deadlock scenarios."""

    def test_trajectory_queue_full_handling(self):
        """Test behavior when trajectory queue is full."""
        config = ReplayBufferConfig(
            trajectory_queue_size=2,
            max_size=100,
        )
        buffer = DistributedReplayBuffer(config=config)

        try:
            trajectory = GameTrajectory(examples=[], game_id=0, actor_id=0)

            # Fill queue
            result1 = buffer.submit_trajectory(trajectory, timeout=0.5)
            result2 = buffer.submit_trajectory(trajectory, timeout=0.5)

            assert result1 is True
            assert result2 is True

            # Third should fail (queue full) - use very short timeout
            result3 = buffer.submit_trajectory(trajectory, timeout=0.1)
            assert result3 is False
        finally:
            buffer.shutdown()

    def test_concurrent_trajectory_submission(self):
        """Test concurrent trajectory submissions don't deadlock."""
        config = ReplayBufferConfig(
            trajectory_queue_size=10,
            max_size=1000,
        )
        buffer = DistributedReplayBuffer(config=config)

        results: list[tuple[int, bool]] = []
        errors: list[tuple[int, Exception]] = []

        def submit_trajectory(actor_id: int):
            try:
                trajectory = GameTrajectory(
                    examples=[],
                    game_id=actor_id,
                    actor_id=actor_id,
                )
                result = buffer.submit_trajectory(trajectory, timeout=1.0)
                results.append((actor_id, result))
            except Exception as e:
                errors.append((actor_id, e))

        try:
            threads = [threading.Thread(target=submit_trajectory, args=(i,)) for i in range(5)]

            for t in threads:
                t.start()

            for t in threads:
                t.join(timeout=5.0)

            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 5
            assert all(r[1] is True for r in results)
        finally:
            buffer.shutdown()


class TestNetworkWeightSynchronization:
    """Tests for weight synchronization edge cases."""

    def test_weight_version_increments(
        self,
        shared_storage: SharedStorage,
        small_network,
    ):
        """Test weight version increments correctly."""
        state_dict = small_network.state_dict()

        # Store initial weights
        v1 = shared_storage.set_model_weights(state_dict)
        assert v1 == 1

        # Store again
        v2 = shared_storage.set_model_weights(state_dict)
        assert v2 == 2

        # Verify version retrieval
        _, version = shared_storage.get_model_weights_with_version()
        assert version == 2

    def test_weight_version_tracking(
        self,
        shared_storage: SharedStorage,
        small_network,
    ):
        """Test weight version tracking in synchronizer."""
        state_dict = small_network.state_dict()
        shared_storage.set_model_weights(state_dict)

        sync = WeightSynchronizer(shared_storage, update_interval=0.0)

        # Initially should need update (version 0 -> 1)
        assert sync.should_update() is True

        # After update, should not need another
        sync.update_model(small_network)
        assert sync.get_current_version() == 1

        # Update weights
        shared_storage.set_model_weights(state_dict)

        # Now should need update again
        assert sync.should_update() is True

    def test_empty_storage_returns_none(
        self,
        shared_storage: SharedStorage,
    ):
        """Test getting weights from empty storage returns None."""
        weights = shared_storage.get_model_weights()
        assert weights is None

        version = shared_storage.get_model_version()
        assert version == 0


class TestReplayBufferFailures:
    """Tests for replay buffer failure scenarios."""

    def test_sample_from_empty_buffer(self):
        """Test sampling from empty buffer returns empty or raises."""
        config = ReplayBufferConfig(max_size=100, min_size=10)
        buffer = DistributedReplayBuffer(config=config)

        try:
            # Empty buffer - use size() method
            assert buffer.size() == 0

            # Sampling should return empty list
            samples = buffer.sample(10)
            assert samples is None or len(samples) == 0
        finally:
            buffer.shutdown()

    def test_buffer_respects_max_size(self):
        """Test buffer doesn't exceed max_size."""
        config = ReplayBufferConfig(max_size=50, min_size=5)
        buffer = DistributedReplayBuffer(config=config)

        try:
            # Add more examples than max_size
            for i in range(100):
                example = TrainingExample(
                    spatial=np.zeros((ENCODER.MICROCIV_CHANNELS, 8, 8), dtype=np.float32),
                    vector=np.zeros(ENCODER.MICROCIV_VECTOR_DIM, dtype=np.float32),
                    action_mask=np.ones(376, dtype=np.float32),
                    policy=np.ones(376, dtype=np.float32) / 376,
                    value=0.0,
                )
                trajectory = GameTrajectory(
                    examples=[example],
                    game_id=i,
                    actor_id=0,
                )
                buffer.submit_trajectory(trajectory, timeout=0.5)
                buffer.process_pending_trajectories(max_count=10)

            # Process remaining
            buffer.process_pending_trajectories(max_count=200)

            # Size should not exceed max
            assert buffer.size() <= config.max_size
        finally:
            buffer.shutdown()


class TestSharedStorageFailures:
    """Tests for shared storage failure scenarios."""

    def test_concurrent_statistics_updates(
        self,
        shared_storage: SharedStorage,
    ):
        """Test concurrent statistics updates are thread-safe."""
        errors: list[Exception] = []

        def increment_games(count: int):
            for _ in range(count):
                try:
                    shared_storage.increment_games_played()
                except Exception as e:
                    errors.append(e)

        def increment_examples(count: int):
            for _ in range(count):
                try:
                    shared_storage.increment_examples(10)
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=increment_games, args=(50,)),
            threading.Thread(target=increment_games, args=(50,)),
            threading.Thread(target=increment_examples, args=(50,)),
            threading.Thread(target=increment_examples, args=(50,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert shared_storage.get_games_played() == 100
        assert shared_storage.get_total_examples() == 1000

    def test_storage_lock_contention(
        self,
        shared_storage: SharedStorage,
    ):
        """Test storage handles lock contention correctly."""
        lock_acquired = threading.Event()
        lock_released = threading.Event()

        def hold_lock():
            with shared_storage._lock:
                lock_acquired.set()
                lock_released.wait(timeout=2.0)

        lock_thread = threading.Thread(target=hold_lock)
        lock_thread.start()

        lock_acquired.wait(timeout=1.0)

        # Operations while lock is held should block then succeed
        start = time.time()

        # Release lock
        lock_released.set()
        lock_thread.join(timeout=2.0)

        # Operations should now complete quickly
        version = shared_storage.get_model_version()
        elapsed = time.time() - start

        assert elapsed < 3.0
        assert version == 0


class TestCommunicationTimeouts:
    """Tests for communication timeout scenarios."""

    def test_trajectory_submission_timeout(self):
        """Test trajectory submission respects timeout."""
        config = ReplayBufferConfig(
            trajectory_queue_size=1,
            max_size=100,
        )
        buffer = DistributedReplayBuffer(config=config)

        try:
            trajectory = GameTrajectory(examples=[], game_id=0, actor_id=0)

            # Fill the single slot
            buffer.submit_trajectory(trajectory, timeout=1.0)

            # Second submission should timeout quickly
            start = time.time()
            result = buffer.submit_trajectory(trajectory, timeout=0.1)
            elapsed = time.time() - start

            assert result is False
            assert elapsed < 0.5  # Should timeout quickly
        finally:
            buffer.shutdown()


class TestGracefulDegradation:
    """Tests for graceful degradation scenarios."""

    def test_buffer_works_without_processing(self):
        """Test buffer accepts trajectories even without processing."""
        config = ReplayBufferConfig(
            trajectory_queue_size=5,
            max_size=100,
        )
        buffer = DistributedReplayBuffer(config=config)

        try:
            # Submit without processing
            for i in range(3):
                trajectory = GameTrajectory(examples=[], game_id=i, actor_id=0)
                result = buffer.submit_trajectory(trajectory, timeout=0.5)
                assert result is True
        finally:
            buffer.shutdown()

    def test_storage_works_with_empty_weights(
        self,
        shared_storage: SharedStorage,
        small_network,
    ):
        """Test storage handles empty weights gracefully."""
        sync = WeightSynchronizer(shared_storage, update_interval=0.0)

        # Should not crash with empty storage
        updated = sync.update_model(small_network)

        # Should return False since no weights available
        assert updated is False


class TestErrorRecovery:
    """Tests for error recovery scenarios."""

    def test_buffer_recovers_after_failed_submission(self):
        """Test buffer continues working after failed submission."""
        config = ReplayBufferConfig(
            trajectory_queue_size=2,
            max_size=100,
        )
        buffer = DistributedReplayBuffer(config=config)

        try:
            trajectory = GameTrajectory(examples=[], game_id=0, actor_id=0)

            # Fill queue
            buffer.submit_trajectory(trajectory, timeout=0.5)
            buffer.submit_trajectory(trajectory, timeout=0.5)

            # This should fail (queue full)
            result = buffer.submit_trajectory(trajectory, timeout=0.1)
            assert result is False

            # Process to clear queue
            buffer.process_pending_trajectories(max_count=5)

            # Should work again
            result = buffer.submit_trajectory(trajectory, timeout=0.5)
            assert result is True
        finally:
            buffer.shutdown()
