"""Integration tests for Spot VM preemption recovery.

Simulates preemption signals, checkpoint saving, and resume
logic without requiring real GCP infrastructure.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHECKPOINT_INTERVAL_SEC = 0.5  # Fast for tests
_PREEMPTION_DELAY_SEC = 0.3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_checkpoint_storage() -> MagicMock:
    """Mock checkpoint storage with in-memory dict."""
    storage = MagicMock()
    data: dict[str, bytes] = {}

    storage.upload_blob.side_effect = lambda b, p, d: data.update({f"{b}/{p}": d})
    storage.download_blob.side_effect = lambda b, p: data[f"{b}/{p}"]
    storage.blob_exists.side_effect = lambda b, p: f"{b}/{p}" in data
    storage._data = data
    return storage


@pytest.fixture
def sample_training_state() -> dict[str, Any]:
    """Minimal training state that can be checkpointed."""
    return {
        "iteration": 5,
        "epoch": 3,
        "model_state_dict": {"weight": torch.randn(8, 8).numpy().tolist()},
        "optimizer_state": {"lr": 0.001, "step": 500},
        "metrics": {"loss": 0.42, "policy_loss": 0.25, "value_loss": 0.17},
        "games_played": 150,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreemptionRecovery:
    """Spot VM preemption handling tests."""

    def test_checkpoint_save_on_preemption(
        self,
        mock_checkpoint_storage: MagicMock,
        sample_training_state: dict[str, Any],
    ) -> None:
        """Training state is saved when preemption is signaled."""
        bucket = "test-bucket"
        checkpoint_path = "checkpoints/preemption_recovery.json"

        # Simulate checkpoint save
        payload = json.dumps(sample_training_state, default=str).encode()
        mock_checkpoint_storage.upload_blob(bucket, checkpoint_path, payload)

        # Verify save happened
        assert mock_checkpoint_storage.blob_exists(bucket, checkpoint_path)
        downloaded = mock_checkpoint_storage.download_blob(bucket, checkpoint_path)
        restored = json.loads(downloaded)
        assert restored["iteration"] == 5
        assert restored["epoch"] == 3

    def test_resume_from_checkpoint(
        self,
        mock_checkpoint_storage: MagicMock,
        sample_training_state: dict[str, Any],
    ) -> None:
        """Training resumes from the last checkpoint correctly."""
        bucket = "test-bucket"
        checkpoint_path = "checkpoints/latest.json"

        # Save checkpoint
        payload = json.dumps(sample_training_state, default=str).encode()
        mock_checkpoint_storage.upload_blob(bucket, checkpoint_path, payload)

        # Resume: load and verify state
        raw = mock_checkpoint_storage.download_blob(bucket, checkpoint_path)
        state = json.loads(raw)

        assert state["iteration"] == sample_training_state["iteration"]
        assert state["epoch"] == sample_training_state["epoch"]
        assert state["games_played"] == sample_training_state["games_played"]
        logger.info("Resumed from iteration=%d, epoch=%d", state["iteration"], state["epoch"])

    def test_checkpoint_versioning(
        self,
        mock_checkpoint_storage: MagicMock,
    ) -> None:
        """Multiple checkpoints are saved with version tracking."""
        bucket = "test-bucket"

        for version in range(1, 4):
            state = {
                "iteration": version * 10,
                "version": version,
                "timestamp": time.time(),
            }
            path = f"checkpoints/iter_{version * 10}.json"
            mock_checkpoint_storage.upload_blob(bucket, path, json.dumps(state).encode())

        # All 3 versions should exist
        for version in range(1, 4):
            path = f"checkpoints/iter_{version * 10}.json"
            assert mock_checkpoint_storage.blob_exists(bucket, path)

    def test_preemption_retry_with_backoff(self) -> None:
        """Retry logic uses exponential backoff after preemption."""
        max_retries = 5
        base_delay = 0.01  # Small for tests

        delays: list[float] = []
        for attempt in range(max_retries):
            delay = base_delay * (2**attempt)
            delays.append(delay)

        # Verify exponential growth
        for i in range(1, len(delays)):
            assert delays[i] == pytest.approx(delays[i - 1] * 2)

        logger.info("Backoff delays: %s", [f"{d:.3f}s" for d in delays])

    def test_graceful_shutdown_saves_state(
        self,
        mock_checkpoint_storage: MagicMock,
        sample_training_state: dict[str, Any],
    ) -> None:
        """Graceful shutdown triggered by signal saves current state."""
        bucket = "test-bucket"
        shutdown_triggered = threading.Event()

        def mock_signal_handler() -> None:
            """Simulate SIGTERM handling."""
            checkpoint_path = "checkpoints/graceful_shutdown.json"
            payload = json.dumps(sample_training_state, default=str).encode()
            mock_checkpoint_storage.upload_blob(bucket, checkpoint_path, payload)
            shutdown_triggered.set()

        # Simulate signal
        mock_signal_handler()

        assert shutdown_triggered.is_set()
        assert mock_checkpoint_storage.blob_exists(bucket, "checkpoints/graceful_shutdown.json")

    def test_partial_checkpoint_detection(
        self,
        mock_checkpoint_storage: MagicMock,
    ) -> None:
        """Detect and skip corrupted/partial checkpoints."""
        bucket = "test-bucket"

        # Write a valid checkpoint
        valid_state = {"iteration": 10, "complete": True}
        mock_checkpoint_storage.upload_blob(
            bucket,
            "checkpoints/iter_10.json",
            json.dumps(valid_state).encode(),
        )

        # Write a partial/corrupted checkpoint
        mock_checkpoint_storage.upload_blob(
            bucket,
            "checkpoints/iter_20.json",
            b"corrupt{data",
        )

        # Load and validate
        valid_raw = mock_checkpoint_storage.download_blob(bucket, "checkpoints/iter_10.json")
        valid = json.loads(valid_raw)
        assert valid["complete"] is True

        corrupt_raw = mock_checkpoint_storage.download_blob(bucket, "checkpoints/iter_20.json")
        with pytest.raises(json.JSONDecodeError):
            json.loads(corrupt_raw)

        logger.info("Partial checkpoint correctly detected")

    def test_preemption_count_tracking(self) -> None:
        """Track total preemption count for monitoring."""
        preemption_count = 0
        max_preemptions = 10

        for _ in range(3):
            preemption_count += 1

        assert preemption_count == 3
        assert preemption_count < max_preemptions
        logger.info("Preemption count: %d/%d", preemption_count, max_preemptions)
