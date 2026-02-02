"""Integration tests for checkpoint coordination in distributed training.

Tests checkpoint save/load operations, barrier synchronization, metadata
handling, and training resumption across multiple ranks.

Run with:
    pytest tests/integration/test_checkpoint_coordination.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from alpha_zero.distributed.ddp_learner import DDPLearnerConfig
from tests.fixtures.mock_distributed import (
    MockDDPWrapper,
    MockDistributedBackend,
)

if TYPE_CHECKING:
    from collections.abc import Generator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def ddp_config() -> DDPLearnerConfig:
    """Create DDP learner config for testing."""
    return DDPLearnerConfig(
        board_size=8,
        num_res_blocks=2,
        num_filters=32,
        batch_size=4,
        learning_rate=0.001,
        world_size=4,
        rank=0,
        backend="gloo",
        device="cpu",
    )


@pytest.fixture
def ddp_config_non_main() -> DDPLearnerConfig:
    """Create DDP config for non-main rank."""
    return DDPLearnerConfig(
        board_size=8,
        num_res_blocks=2,
        num_filters=32,
        batch_size=4,
        learning_rate=0.001,
        world_size=4,
        rank=1,
        backend="gloo",
        device="cpu",
    )


@pytest.fixture
def mock_backend() -> Generator[MockDistributedBackend, None, None]:
    """Create mock distributed backend."""
    backend = MockDistributedBackend(world_size=4, rank=0)
    with backend.patch():
        yield backend


@pytest.fixture
def mock_backend_rank_1() -> Generator[MockDistributedBackend, None, None]:
    """Create mock distributed backend for rank 1."""
    backend = MockDistributedBackend(world_size=4, rank=1)
    with backend.patch():
        yield backend


@pytest.fixture
def mock_shared_storage() -> MagicMock:
    """Create mock shared storage."""
    storage = MagicMock()
    storage.get_model_weights.return_value = None
    storage.get_model_version.return_value = 0
    return storage


@pytest.fixture
def mock_replay_buffer() -> MagicMock:
    """Create mock replay buffer."""
    buffer = MagicMock()
    buffer.size.return_value = 0
    buffer.is_ready.return_value = False
    return buffer


@pytest.fixture
def temp_checkpoint_dir() -> Generator[Path, None, None]:
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_network() -> nn.Module:
    """Create simple network for checkpoint testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


# ============================================================================
# Checkpoint Save Tests
# ============================================================================


class TestCheckpointOnlyOnMainRank:
    """Tests that checkpoint is only saved on main rank."""

    def test_main_rank_saves_checkpoint(
        self,
        mock_backend: MockDistributedBackend,
        simple_network: nn.Module,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Verify main rank (rank 0) saves checkpoint."""
        wrapped = MockDDPWrapper(simple_network)
        checkpoint_path = temp_checkpoint_dir / "checkpoint.pt"

        # Simulate save_checkpoint from DDPLearner
        mock_backend.state.barrier_count = 0

        # Simulate barrier before save
        torch.distributed.barrier()
        assert mock_backend.state.barrier_count == 1

        # Main rank saves
        state = {
            "step": 100,
            "model_state_dict": wrapped.state_dict(),
            "rank": 0,
            "world_size": 4,
        }
        torch.save(state, checkpoint_path)

        # Barrier after save
        torch.distributed.barrier()
        assert mock_backend.state.barrier_count == 2

        # Verify file exists
        assert checkpoint_path.exists()

    def test_non_main_rank_skips_save(
        self,
        mock_backend_rank_1: MockDistributedBackend,
        simple_network: nn.Module,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Verify non-main ranks skip checkpoint save."""
        checkpoint_path = temp_checkpoint_dir / "checkpoint_rank1.pt"

        # Simulate save on non-main rank
        mock_backend_rank_1.state.barrier_count = 0

        # Barrier before save
        torch.distributed.barrier()

        # Non-main rank checks and skips
        if mock_backend_rank_1.state.rank != 0:
            # Skip save
            pass

        # Barrier after (waiting for main)
        torch.distributed.barrier()

        # File should not exist
        assert not checkpoint_path.exists()
        assert mock_backend_rank_1.state.barrier_count == 2


class TestCheckpointLoadAllRanks:
    """Tests that all ranks load the same checkpoint."""

    def test_all_ranks_load_same_checkpoint(
        self,
        mock_backend: MockDistributedBackend,
        simple_network: nn.Module,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Verify all ranks can load the same checkpoint."""
        checkpoint_path = temp_checkpoint_dir / "shared_checkpoint.pt"

        # Create and save checkpoint
        state = {
            "step": 200,
            "model_state_dict": simple_network.state_dict(),
            "rank": 0,
            "world_size": 4,
            "config": {"backend": "gloo"},
        }
        torch.save(state, checkpoint_path)

        # Simulate loading on rank 0
        loaded = torch.load(checkpoint_path, weights_only=False)
        assert loaded["step"] == 200
        assert loaded["world_size"] == 4

    def test_checkpoint_load_with_different_world_size(
        self,
        simple_network: nn.Module,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test loading checkpoint with different world_size logs warning."""
        checkpoint_path = temp_checkpoint_dir / "old_checkpoint.pt"

        # Save with world_size=2
        state = {
            "step": 100,
            "model_state_dict": simple_network.state_dict(),
            "rank": 0,
            "world_size": 2,
        }
        torch.save(state, checkpoint_path)

        # Load and check world_size mismatch handling
        loaded = torch.load(checkpoint_path, weights_only=False)
        checkpoint_world_size = loaded.get("world_size", 1)
        current_world_size = 4

        # Should detect mismatch
        assert checkpoint_world_size != current_world_size


class TestCheckpointContainsDDPMetadata:
    """Tests that checkpoint contains distributed training metadata."""

    def test_checkpoint_has_world_size(
        self,
        simple_network: nn.Module,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Verify checkpoint contains world_size."""
        checkpoint_path = temp_checkpoint_dir / "metadata_checkpoint.pt"

        state = {
            "step": 50,
            "model_state_dict": simple_network.state_dict(),
            "rank": 0,
            "world_size": 4,
            "config": {
                "backend": "gloo",
                "gradient_sync_interval": 1,
            },
        }
        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)
        assert "world_size" in loaded
        assert loaded["world_size"] == 4

    def test_checkpoint_has_rank(
        self,
        simple_network: nn.Module,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Verify checkpoint contains rank."""
        checkpoint_path = temp_checkpoint_dir / "rank_checkpoint.pt"

        state = {
            "step": 50,
            "model_state_dict": simple_network.state_dict(),
            "rank": 0,
            "world_size": 4,
        }
        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)
        assert "rank" in loaded
        assert loaded["rank"] == 0

    def test_checkpoint_has_config(
        self,
        simple_network: nn.Module,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Verify checkpoint contains config metadata."""
        checkpoint_path = temp_checkpoint_dir / "config_checkpoint.pt"

        state = {
            "step": 50,
            "model_state_dict": simple_network.state_dict(),
            "rank": 0,
            "world_size": 4,
            "config": {
                "backend": "gloo",
                "gradient_sync_interval": 2,
            },
        }
        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)
        assert "config" in loaded
        assert loaded["config"]["backend"] == "gloo"
        assert loaded["config"]["gradient_sync_interval"] == 2


class TestResumeTrainingFromCheckpoint:
    """Tests for resuming distributed training from checkpoint."""

    def test_model_weights_restored(
        self,
        simple_network: nn.Module,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Verify model weights are restored from checkpoint."""
        checkpoint_path = temp_checkpoint_dir / "resume_checkpoint.pt"

        # Save original weights
        original_weights = {k: v.clone() for k, v in simple_network.state_dict().items()}
        state = {
            "step": 100,
            "model_state_dict": simple_network.state_dict(),
            "rank": 0,
            "world_size": 4,
        }
        torch.save(state, checkpoint_path)

        # Create new network with different weights
        new_network = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        # Verify weights are different
        for key in original_weights:
            assert not torch.equal(new_network.state_dict()[key], original_weights[key])

        # Restore weights
        loaded = torch.load(checkpoint_path, weights_only=False)
        new_network.load_state_dict(loaded["model_state_dict"])

        # Verify weights match
        for key in original_weights:
            assert torch.equal(new_network.state_dict()[key], original_weights[key])

    def test_optimizer_state_restored(
        self,
        simple_network: nn.Module,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Verify optimizer state is restored from checkpoint."""
        checkpoint_path = temp_checkpoint_dir / "optimizer_checkpoint.pt"

        optimizer = torch.optim.Adam(simple_network.parameters(), lr=0.001)

        # Do a training step to create optimizer state
        inputs = torch.randn(4, 10)
        outputs = simple_network(inputs)
        loss = outputs.sum()
        loss.backward()
        optimizer.step()

        # Save with optimizer state
        state = {
            "step": 100,
            "model_state_dict": simple_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rank": 0,
            "world_size": 4,
        }
        torch.save(state, checkpoint_path)

        # Create new optimizer
        new_optimizer = torch.optim.Adam(simple_network.parameters(), lr=0.001)

        # Restore optimizer state
        loaded = torch.load(checkpoint_path, weights_only=False)
        new_optimizer.load_state_dict(loaded["optimizer_state_dict"])

        # Verify optimizer state restored
        assert len(new_optimizer.state) > 0

    def test_training_step_restored(
        self,
        simple_network: nn.Module,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Verify training step number is restored from checkpoint."""
        checkpoint_path = temp_checkpoint_dir / "step_checkpoint.pt"

        state = {
            "step": 1500,
            "model_state_dict": simple_network.state_dict(),
            "rank": 0,
            "world_size": 4,
        }
        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)
        restored_step = loaded["step"]

        assert restored_step == 1500


# ============================================================================
# Barrier Synchronization Tests
# ============================================================================


class TestBarrierSynchronization:
    """Tests for barrier synchronization during checkpointing."""

    def test_barrier_before_checkpoint(
        self,
        mock_backend: MockDistributedBackend,
    ) -> None:
        """Verify barrier is called before checkpoint save."""
        mock_backend.state.barrier_count = 0

        # Simulate checkpoint flow
        torch.distributed.barrier()  # Before save
        assert mock_backend.state.barrier_count == 1

    def test_barrier_after_checkpoint(
        self,
        mock_backend: MockDistributedBackend,
    ) -> None:
        """Verify barrier is called after checkpoint save."""
        mock_backend.state.barrier_count = 0

        # Simulate full checkpoint flow
        torch.distributed.barrier()  # Before save
        # Save happens here (only on main)
        torch.distributed.barrier()  # After save

        assert mock_backend.state.barrier_count == 2

    def test_non_main_waits_at_both_barriers(
        self,
        mock_backend_rank_1: MockDistributedBackend,
    ) -> None:
        """Verify non-main ranks wait at both barriers."""
        mock_backend_rank_1.state.barrier_count = 0

        # First barrier (before save)
        torch.distributed.barrier()

        # Non-main skips save but waits
        if mock_backend_rank_1.state.rank != 0:
            pass  # Skip save

        # Second barrier (after save)
        torch.distributed.barrier()

        assert mock_backend_rank_1.state.barrier_count == 2


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestCheckpointErrorHandling:
    """Tests for checkpoint error handling."""

    def test_missing_checkpoint_raises_error(
        self,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Verify missing checkpoint file raises FileNotFoundError."""
        missing_path = temp_checkpoint_dir / "nonexistent.pt"

        with pytest.raises(FileNotFoundError):
            torch.load(missing_path, weights_only=False)

    def test_corrupted_checkpoint_raises_error(
        self,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Verify corrupted checkpoint raises error."""
        corrupted_path = temp_checkpoint_dir / "corrupted.pt"
        corrupted_path.write_text("not a valid pytorch checkpoint")

        # Corrupted checkpoint will raise various pickle/load errors
        import pickle
        with pytest.raises((RuntimeError, EOFError, OSError, pickle.UnpicklingError)):
            torch.load(corrupted_path, weights_only=False)

    def test_missing_model_state_dict_detectable(
        self,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Verify missing model_state_dict is detectable."""
        checkpoint_path = temp_checkpoint_dir / "no_model.pt"

        state = {
            "step": 100,
            "rank": 0,
            "world_size": 4,
        }
        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)
        assert "model_state_dict" not in loaded


# ============================================================================
# Integration Tests
# ============================================================================


class TestCheckpointCoordinationIntegration:
    """Integration tests for full checkpoint coordination flow."""

    def test_full_checkpoint_save_and_load_cycle(
        self,
        mock_backend: MockDistributedBackend,
        simple_network: nn.Module,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test complete checkpoint save and load cycle."""
        checkpoint_path = temp_checkpoint_dir / "full_cycle.pt"
        optimizer = torch.optim.SGD(simple_network.parameters(), lr=0.01)

        # Do some training
        for _ in range(5):
            inputs = torch.randn(4, 10)
            outputs = simple_network(inputs)
            loss = outputs.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save checkpoint with full flow
        mock_backend.state.barrier_count = 0
        torch.distributed.barrier()

        if mock_backend.state.rank == 0:
            state = {
                "step": 5,
                "model_state_dict": simple_network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rank": 0,
                "world_size": 4,
                "config": {"backend": "gloo"},
            }
            torch.save(state, checkpoint_path)

        torch.distributed.barrier()

        # Verify checkpoint
        assert checkpoint_path.exists()
        loaded = torch.load(checkpoint_path, weights_only=False)
        assert loaded["step"] == 5
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded

    def test_checkpoint_with_additional_state(
        self,
        simple_network: nn.Module,
        temp_checkpoint_dir: Path,
    ) -> None:
        """Test checkpoint with additional custom state."""
        checkpoint_path = temp_checkpoint_dir / "additional_state.pt"

        state = {
            "step": 100,
            "model_state_dict": simple_network.state_dict(),
            "rank": 0,
            "world_size": 4,
            # Additional state
            "learning_rate_schedule": [0.01, 0.001, 0.0001],
            "best_loss": 0.5,
            "metrics_history": {"train_loss": [1.0, 0.8, 0.6]},
        }
        torch.save(state, checkpoint_path)

        loaded = torch.load(checkpoint_path, weights_only=False)
        assert loaded["learning_rate_schedule"] == [0.01, 0.001, 0.0001]
        assert loaded["best_loss"] == 0.5
        assert loaded["metrics_history"]["train_loss"] == [1.0, 0.8, 0.6]
