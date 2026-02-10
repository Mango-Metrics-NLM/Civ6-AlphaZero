"""Edge case tests for replay buffer."""
import threading
import numpy as np
import pytest

from alpha_zero.distributed.shared_storage import (
    DistributedReplayBuffer,
    ReplayBufferConfig,
    GameTrajectory,
    TrainingExample,
)
from alpha_zero.utils.constants import ENCODER


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
        """Buffer should not exceed max_size after overflow."""
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
                buffer.process_pending_trajectories(max_count=10)
            buffer.process_pending_trajectories(max_count=100)
            assert buffer.size() <= config.max_size
        finally:
            buffer.shutdown()

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
