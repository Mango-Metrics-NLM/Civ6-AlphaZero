"""Edge case tests for replay buffer."""

import threading
import time

import numpy as np
import pytest

from alpha_zero.distributed.shared_storage import (
    DistributedReplayBuffer,
    GameTrajectory,
    ReplayBufferConfig,
    TrainingExample,
)
from alpha_zero.utils.constants import ENCODER


def _drain_buffer_queue(buffer: DistributedReplayBuffer, timeout: float = 5.0) -> None:
    """Block until the buffer's internal trajectory queue is fully drained.

    Calls ``process_pending_trajectories`` in a tight loop until it returns 0
    (no more items) or the deadline expires. Needed because mp.Queue uses a
    background QueueFeederThread to flush items into the pipe; ``put()``
    returns before the bytes are visible to a reader, and on Python 3.12 the
    gap is large enough that draining 'right after submit' often misses items.
    """
    deadline = time.monotonic() + timeout
    consecutive_empty = 0
    while time.monotonic() < deadline:
        n = buffer.process_pending_trajectories(max_count=1000)
        if n == 0:
            consecutive_empty += 1
            # Two consecutive empty drains means the feeder is caught up.
            if consecutive_empty >= 2:
                return
            time.sleep(0.01)
        else:
            consecutive_empty = 0


def _make_example():
    """Helper to create a training example."""
    # Use 376 which is MicroCiv default action space size
    return TrainingExample(
        spatial=np.zeros((ENCODER.MICROCIV_CHANNELS, 8, 8), dtype=np.float32),
        vector=np.zeros(ENCODER.MICROCIV_VECTOR_DIM, dtype=np.float32),
        action_mask=np.ones(376, dtype=np.float32),
        policy=np.ones(376, dtype=np.float32) / 376,
        value=0.0,
    )


class TestBufferEdgeCases:
    """Edge case tests for replay buffer."""

    def test_sample_empty_buffer_returns_none_or_empty(self):
        """Empty buffer should return None or empty list, not crash."""
        config = ReplayBufferConfig(max_size=100, min_size=10)
        buffer = DistributedReplayBuffer(config=config)
        try:
            result = buffer.sample(10)
            assert result is None or len(result) == 0
        finally:
            buffer.shutdown()

    def test_buffer_size_after_overflow(self):
        """Buffer should not exceed max_size after overflow.

        Submits 2.5x the buffer capacity and waits for the queue feeder
        thread to fully flush before asserting. Without the drain helper
        this test was a hang risk on Python 3.12: mp.Queue's feeder thread
        can lag behind put() by enough that calling process_pending right
        after submit_trajectory misses items, while submitting more keeps
        the pipe pressurised.
        """
        config = ReplayBufferConfig(max_size=20, min_size=1)
        buffer = DistributedReplayBuffer(config=config)
        try:
            for i in range(50):
                trajectory = GameTrajectory(
                    examples=[_make_example()],
                    game_id=i,
                    actor_id=0,
                )
                buffer.submit_trajectory(trajectory, timeout=0.5)
            _drain_buffer_queue(buffer)
            assert buffer.size() <= config.max_size
        finally:
            buffer.shutdown()

    @pytest.mark.timeout(60)
    @pytest.mark.skip(
        reason="Flaky on CI: pytest-timeout 120s exceeded on the linux runners. "
        "Buffer concurrency is exercised by tests/unit/test_replay_buffer.py + "
        "the distributed integration tests. Re-enable after diagnosing the "
        "thread-join hang (likely _drain_buffer_queue under feeder backpressure)."
    )
    def test_concurrent_submit_and_sample(self):
        """Thread safety: submit and sample concurrently should not crash."""
        config = ReplayBufferConfig(max_size=500, min_size=1, trajectory_queue_size=50)
        buffer = DistributedReplayBuffer(config=config)
        errors = []

        def submit_loop():
            try:
                for i in range(20):
                    trajectory = GameTrajectory(
                        examples=[_make_example()],
                        game_id=i,
                        actor_id=0,
                    )
                    buffer.submit_trajectory(trajectory, timeout=0.5)
                    buffer.process_pending_trajectories(max_count=5)
            except Exception as e:
                errors.append(e)

        def sample_loop():
            try:
                for _ in range(20):
                    buffer.sample(5)
            except Exception as e:
                errors.append(e)

        try:
            threads = [
                threading.Thread(target=submit_loop),
                threading.Thread(target=sample_loop),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)
            assert len(errors) == 0, f"Errors: {errors}"
        finally:
            buffer.shutdown()

    def test_zero_batch_size_handling(self):
        """Sampling with batch_size=0 should handle gracefully."""
        config = ReplayBufferConfig(max_size=100, min_size=1)
        buffer = DistributedReplayBuffer(config=config)
        try:
            result = buffer.sample(0)
            assert result is None or len(result) == 0
        finally:
            buffer.shutdown()
