"""Integration tests for multi-node DDP simulation.

Tests simulate distributed training scenarios with mocked
process groups to verify gradient aggregation, weight
synchronization, and checkpoint coordination.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from alpha_zero.distributed.ddp_learner import DDPLearner, DDPLearnerConfig
from tests.fixtures.mock_distributed import (
    mock_ddp_context,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ToyModel(nn.Module):
    """Toy model for integration testing."""

    def __init__(self) -> None:
        """Initialize toy model."""
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass."""
        h = torch.relu(self.linear1(x))
        out = self.linear2(h)
        return {
            "policy": torch.softmax(out, dim=-1),
            "value": out.mean(dim=-1, keepdim=True),
        }


@pytest.fixture
def toy_model() -> ToyModel:
    """Create toy model for testing."""
    return ToyModel()


@pytest.fixture
def mock_storage() -> MagicMock:
    """Create mock shared storage."""
    storage = MagicMock()
    storage.get_weights.return_value = {}
    return storage


@pytest.fixture
def mock_buffer() -> MagicMock:
    """Create mock replay buffer."""
    buffer = MagicMock()
    buffer.sample.return_value = {
        "states": torch.randn(16, 10),
        "policies": torch.randn(16, 5),
        "values": torch.randn(16, 1),
    }
    return buffer


@pytest.mark.integration
class TestSimulatedGradientAggregation:
    """Tests for gradient aggregation across simulated ranks."""

    def test_gradient_all_reduce(
        self,
        toy_model: ToyModel,
        mock_storage: MagicMock,
        mock_buffer: MagicMock,
    ) -> None:
        """Test that gradients are aggregated via all_reduce."""
        config = DDPLearnerConfig(ddp_enabled=True, ddp_world_size=4, ddp_rank=0)

        with mock_ddp_context(
            world_size=4, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP"
        ) as mock_dist:
            learner = DDPLearner(
                config=config,
                shared_storage=mock_storage,
                replay_buffer=mock_buffer,
                network=toy_model,
            )

            # Simulate gradient tensor
            gradient = torch.ones(100)
            learner.all_reduce(gradient)

            # Mock multiplies by world_size to simulate sum
            assert mock_dist.state.all_reduce_count == 1
            assert torch.allclose(gradient, torch.full((100,), 4.0))

            learner.cleanup()

    def test_gradient_averaging(
        self,
        toy_model: ToyModel,
        mock_storage: MagicMock,
        mock_buffer: MagicMock,
    ) -> None:
        """Test manual gradient averaging after all_reduce."""
        config = DDPLearnerConfig(ddp_enabled=True, ddp_world_size=4, ddp_rank=0)

        with mock_ddp_context(
            world_size=4, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP"
        ):
            learner = DDPLearner(
                config=config,
                shared_storage=mock_storage,
                replay_buffer=mock_buffer,
                network=toy_model,
            )

            # Each rank has gradient value 2.0
            gradient = torch.full((50,), 2.0)

            # All-reduce sums (mock multiplies by world_size)
            learner.all_reduce(gradient)
            # After mock: 2.0 * 4 = 8.0

            # Average by dividing by world_size
            gradient.div_(learner.world_size)
            # After: 8.0 / 4 = 2.0

            assert torch.allclose(gradient, torch.full((50,), 2.0))
            learner.cleanup()


@pytest.mark.integration
class TestWeightSynchronization:
    """Tests for weight synchronization across ranks."""

    def test_broadcast_from_main(
        self,
        toy_model: ToyModel,
        mock_storage: MagicMock,
        mock_buffer: MagicMock,
    ) -> None:
        """Test weight broadcast from rank 0."""
        config = DDPLearnerConfig(ddp_enabled=True, ddp_world_size=4, ddp_rank=0)

        with mock_ddp_context(
            world_size=4, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP"
        ) as mock_dist:
            learner = DDPLearner(
                config=config,
                shared_storage=mock_storage,
                replay_buffer=mock_buffer,
                network=toy_model,
            )

            learner.broadcast_weights_from_main()

            # For world_size=1 broadcast is skipped, but mock tracks anyway
            assert mock_dist.state.broadcast_count > 0 or learner.world_size == 1
            learner.cleanup()



@pytest.mark.integration
class TestCheckpointCoordination:
    """Tests for checkpoint coordination across simulated ranks."""

    def test_only_main_rank_saves(
        self,
        toy_model: ToyModel,
        mock_storage: MagicMock,
        mock_buffer: MagicMock,
    ) -> None:
        """Test that only rank 0 (is_main=True) is intended to save checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Test with world_size=1 (main rank saves)
            config = DDPLearnerConfig(ddp_world_size=1, ddp_rank=0, checkpoint_dir=str(tmpdir))
            learner = DDPLearner(
                config=config,
                shared_storage=mock_storage,
                replay_buffer=mock_buffer,
                network=toy_model,
            )
            learner._save_checkpoint(100)
            
            # verify the fixed filename checkpoint_100.pt exists in tmpdir
            assert (Path(tmpdir) / "checkpoint_100.pt").exists()

            # Verify worker ranks have is_main=False
            for rank in [1, 2, 3]:
                config_worker = DDPLearnerConfig(ddp_enabled=True, ddp_world_size=4, ddp_rank=rank)
                with mock_ddp_context(world_size=4, rank=rank, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP"):
                    learner_worker = DDPLearner(
                        config=config_worker,
                        shared_storage=mock_storage,
                        replay_buffer=mock_buffer,
                        network=ToyModel(),
                    )
                assert learner_worker.is_main is False

    def test_all_ranks_load_same_checkpoint(
        self,
        toy_model: ToyModel,
        mock_storage: MagicMock,
        mock_buffer: MagicMock,
    ) -> None:
        """Test that all ranks can load the same checkpoint regardless of rank."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "shared_checkpoint.pt"

            # Save from main rank (world_size=1 to avoid distributed setup)
            config_main = DDPLearnerConfig(ddp_world_size=1, ddp_rank=0, checkpoint_dir=str(tmpdir))
            learner_main = DDPLearner(
                config=config_main,
                shared_storage=mock_storage,
                replay_buffer=mock_buffer,
                network=ToyModel(),
            )
            learner_main._save_checkpoint(42)

            # Load from a separate learner (simulating another rank but with world_size=1)
            learner_loader = DDPLearner(
                config=DDPLearnerConfig(ddp_world_size=1, ddp_rank=0),
                shared_storage=mock_storage,
                replay_buffer=mock_buffer,
                network=ToyModel(),
            )
            loaded_checkpoint = learner_loader.load_checkpoint(str(Path(tmpdir) / "checkpoint_42.pt"))
            assert loaded_checkpoint["step"] == 42


@pytest.mark.integration
class TestBarrierSynchronization:
    """Tests for barrier synchronization."""

    def test_barrier_blocks_all_ranks(
        self,
        toy_model: ToyModel,
        mock_storage: MagicMock,
        mock_buffer: MagicMock,
    ) -> None:
        """Test that barrier is called for synchronization."""
        config = DDPLearnerConfig(ddp_enabled=True, ddp_world_size=4, ddp_rank=0)

        with mock_ddp_context(
            world_size=4, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP"
        ) as mock_dist:
            learner = DDPLearner(
                config=config,
                shared_storage=mock_storage,
                replay_buffer=mock_buffer,
                network=toy_model,
            )

            # Simulate pre-checkpoint barrier
            learner.barrier()
            assert mock_dist.state.barrier_count == 1

            # Simulate post-checkpoint barrier
            learner.barrier()
            assert mock_dist.state.barrier_count == 2

            learner.cleanup()


@pytest.mark.integration
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_multi_rank_operations(
    world_size: int,
    toy_model: ToyModel,
    mock_storage: MagicMock,
    mock_buffer: MagicMock,
) -> None:
    """Test distributed operations with various world sizes."""
    config = DDPLearnerConfig(ddp_enabled=True, ddp_world_size=world_size, ddp_rank=0)

    with mock_ddp_context(
        world_size=world_size, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP"
    ) as mock_dist:
        learner = DDPLearner(
            config=config,
            shared_storage=mock_storage,
            replay_buffer=mock_buffer,
            network=toy_model,
        )

        # Test all operations
        tensor = torch.ones(10)
        learner.all_reduce(tensor)
        assert torch.allclose(tensor, torch.full((10,), float(world_size)))

        learner.barrier()
        assert mock_dist.state.barrier_count == 1

        result = learner.gather(torch.tensor([1.0]), dst=0)
        assert result is not None
        assert len(result) == world_size

        learner.cleanup()


@pytest.mark.integration
class TestMixedDeviceScenarios:
    """Tests for CPU/CUDA device handling."""

    def test_cpu_only_initialization(
        self,
        toy_model: ToyModel,
        mock_storage: MagicMock,
        mock_buffer: MagicMock,
    ) -> None:
        """Test initialization on CPU-only environment."""
        config = DDPLearnerConfig(ddp_enabled=True, ddp_world_size=2, ddp_rank=0, device="cpu")

        # Ensure we're testing CPU path
        with patch("torch.cuda.is_available", return_value=False):
            with mock_ddp_context(
                world_size=2, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP"
            ):
                from alpha_zero.distributed.ddp_config import DDPConfig
                learner = DDPLearner(
                    config=config,
                    shared_storage=mock_storage,
                    replay_buffer=mock_buffer,
                    ddp_config=DDPConfig(enabled=True, world_size=2, rank=0, use_cuda=False),
                    network=toy_model,
                )

                assert learner.config.device == "cpu"
                learner.cleanup()

    def test_cuda_device_assignment(
        self,
        toy_model: ToyModel,
        mock_storage: MagicMock,
        mock_buffer: MagicMock,
    ) -> None:
        """Test CUDA device assignment based on local_rank."""
        # Skip if not enough CUDA devices available
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Test requires at least 2 CUDA devices")

        config = DDPLearnerConfig(ddp_enabled=True, ddp_world_size=4, ddp_rank=2, ddp_local_rank=1)

        with mock_ddp_context(
            world_size=4, rank=2, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP"
        ):
            learner = DDPLearner(
                config=config,
                shared_storage=mock_storage,
                replay_buffer=mock_buffer,
                network=toy_model,
            )

            # Device should be cuda:1 based on local_rank
            assert "cuda:1" in str(learner.config.device)
            learner.cleanup()


@pytest.mark.integration
class TestProcessGroupLifecycle:
    """Tests for process group lifecycle management."""

    def test_init_destroy_cycle(
        self,
        toy_model: ToyModel,
        mock_storage: MagicMock,
        mock_buffer: MagicMock,
    ) -> None:
        """Test complete initialize/cleanup cycle."""
        config = DDPLearnerConfig(ddp_enabled=True, ddp_world_size=4, ddp_rank=0)

        with mock_ddp_context(
            world_size=4, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP"
        ) as mock_dist:
            learner = DDPLearner(
                config=config,
                shared_storage=mock_storage,
                replay_buffer=mock_buffer,
                network=toy_model,
            )

            # After init
            assert learner._process_group_initialized
            assert mock_dist.state.is_initialized

            # After cleanup
            learner.cleanup()
            assert not learner._process_group_initialized

    def test_multiple_init_cleanup_cycles(
        self,
        mock_storage: MagicMock,
        mock_buffer: MagicMock,
    ) -> None:
        """Test multiple initialization/cleanup cycles."""
        config = DDPLearnerConfig(ddp_enabled=True, ddp_world_size=2, ddp_rank=0)

        for cycle in range(3):
            model = ToyModel()
            with mock_ddp_context(
                world_size=2, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP"
            ):
                learner = DDPLearner(
                    config=config,
                    shared_storage=mock_storage,
                    replay_buffer=mock_buffer,
                    network=model,
                )
                assert learner._process_group_initialized
                learner.cleanup()
                assert not learner._process_group_initialized

            logger.debug("Completed cycle %d", cycle + 1)
