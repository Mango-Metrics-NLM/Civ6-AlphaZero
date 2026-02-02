"""Unit tests for DDPLearner with mocked distributed backend.

Tests cover configuration, initialization, training, communication
primitives, and checkpointing without requiring actual distributed setup.
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
    MockDistributedBackend,
    MockDDPWrapper,
    mock_ddp_context,
)

if TYPE_CHECKING:
    from typing import Generator

logger = logging.getLogger(__name__)


class SimpleNetwork(nn.Module):
    """Minimal network for testing DDPLearner."""

    def __init__(self, input_size: int = 10, output_size: int = 5) -> None:
        """Initialize simple network.

        Args:
            input_size: Input feature dimension.
            output_size: Output feature dimension.
        """
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass returning policy and value predictions."""
        out = self.linear(x)
        return {
            "policy": torch.softmax(out, dim=-1),
            "value": out.mean(dim=-1, keepdim=True),
        }


@pytest.fixture
def simple_network() -> SimpleNetwork:
    """Create simple network for testing."""
    return SimpleNetwork()


@pytest.fixture
def mock_shared_storage() -> MagicMock:
    """Create mock shared storage."""
    storage = MagicMock()
    storage.get_weights.return_value = {}
    storage.set_weights.return_value = None
    return storage


@pytest.fixture
def mock_replay_buffer() -> MagicMock:
    """Create mock replay buffer."""
    buffer = MagicMock()
    buffer.sample.return_value = {
        "states": torch.randn(8, 10),
        "policies": torch.randn(8, 5),
        "values": torch.randn(8, 1),
    }
    return buffer


class TestDDPLearnerConfig:
    """Tests for DDPLearnerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DDPLearnerConfig()

        assert config.backend == "nccl"
        assert config.init_method == "env://"
        assert config.world_size == 1
        assert config.rank == 0
        assert config.local_rank == 0
        assert config.master_port == "29500"
        assert config.find_unused_parameters is False
        assert config.broadcast_buffers is True
        assert config.gradient_as_bucket_view is True

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = DDPLearnerConfig(
            backend="gloo",
            world_size=4,
            rank=2,
            local_rank=2,
            master_addr="192.168.1.100",
            master_port="12345",
            find_unused_parameters=True,
        )

        assert config.backend == "gloo"
        assert config.world_size == 4
        assert config.rank == 2
        assert config.local_rank == 2
        assert config.master_addr == "192.168.1.100"
        assert config.master_port == "12345"
        assert config.find_unused_parameters is True

    def test_from_env_defaults(self) -> None:
        """Test from_env with no environment variables."""
        with patch.dict("os.environ", {}, clear=True):
            config = DDPLearnerConfig.from_env()

        assert config.world_size == 1
        assert config.rank == 0
        assert config.local_rank == 0
        assert config.master_addr == "localhost"
        assert config.master_port == "29500"

    def test_from_env_with_variables(self) -> None:
        """Test from_env with environment variables set."""
        env_vars = {
            "WORLD_SIZE": "8",
            "RANK": "3",
            "LOCAL_RANK": "3",
            "MASTER_ADDR": "10.0.0.1",
            "MASTER_PORT": "29501",
        }
        with patch.dict("os.environ", env_vars, clear=True):
            config = DDPLearnerConfig.from_env()

        assert config.world_size == 8
        assert config.rank == 3
        assert config.local_rank == 3
        assert config.master_addr == "10.0.0.1"
        assert config.master_port == "29501"

    def test_from_env_with_overrides(self) -> None:
        """Test from_env with kwarg overrides."""
        env_vars = {"WORLD_SIZE": "4", "RANK": "2"}
        with patch.dict("os.environ", env_vars, clear=True):
            config = DDPLearnerConfig.from_env(world_size=8, backend="gloo")

        assert config.world_size == 8  # Override
        assert config.rank == 2  # From env
        assert config.backend == "gloo"  # Override


class TestDDPLearnerInitialization:
    """Tests for DDPLearner initialization."""

    def test_init_single_node(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test initialization with world_size=1 (no DDP)."""
        config = DDPLearnerConfig(world_size=1, rank=0)
        learner = DDPLearner(
            config=config,
            shared_storage=mock_shared_storage,
            replay_buffer=mock_replay_buffer,
            network=simple_network,
        )

        assert learner.rank == 0
        assert learner.world_size == 1
        assert learner.is_main is True
        assert learner._is_initialized is False

    def test_init_multi_node_mocked(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test initialization with mocked distributed backend."""
        config = DDPLearnerConfig(
            world_size=4,
            rank=0,
            local_rank=0,
            backend="gloo",
        )

        with mock_ddp_context(world_size=4, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP") as mock_dist:
            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
                network=simple_network,
            )
            learner.initialize()

            assert learner._is_initialized is True
            assert mock_dist.state.is_initialized is True

            learner.cleanup()

    def test_init_non_main_rank(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test initialization on non-main rank."""
        config = DDPLearnerConfig(
            world_size=4,
            rank=2,
            local_rank=0,
        )

        learner = DDPLearner(
            config=config,
            shared_storage=mock_shared_storage,
            replay_buffer=mock_replay_buffer,
            network=simple_network,
        )

        assert learner.is_main is False
        assert learner.rank == 2

    def test_double_initialization_idempotent(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test that double initialization is safe."""
        config = DDPLearnerConfig(world_size=1)
        learner = DDPLearner(
            config=config,
            shared_storage=mock_shared_storage,
            replay_buffer=mock_replay_buffer,
            network=simple_network,
        )

        learner.initialize()
        learner.initialize()  # Should not raise

        assert learner._is_initialized is True


class TestDDPLearnerCommunication:
    """Tests for DDP communication primitives."""

    def test_barrier_single_node(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test barrier on single node (no-op)."""
        config = DDPLearnerConfig(world_size=1)
        learner = DDPLearner(
            config=config,
            shared_storage=mock_shared_storage,
            replay_buffer=mock_replay_buffer,
            network=simple_network,
        )
        learner.initialize()

        # Should not raise on single node
        learner.barrier()

    def test_barrier_multi_node(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test barrier with mocked distributed backend."""
        config = DDPLearnerConfig(world_size=4, rank=0)

        with mock_ddp_context(world_size=4, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP") as mock_dist:
            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
                network=simple_network,
            )
            learner.initialize()

            learner.barrier()
            assert mock_dist.state.barrier_count == 1

            learner.barrier()
            assert mock_dist.state.barrier_count == 2

            learner.cleanup()

    def test_all_reduce_multi_node(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test all_reduce with mocked distributed backend."""
        config = DDPLearnerConfig(world_size=4, rank=0)

        with mock_ddp_context(world_size=4, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP") as mock_dist:
            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
                network=simple_network,
            )
            learner.initialize()

            tensor = torch.ones(5)
            learner.all_reduce(tensor)

            assert mock_dist.state.all_reduce_count == 1
            # Mock multiplies by world_size
            assert torch.allclose(tensor, torch.full((5,), 4.0))

            learner.cleanup()

    def test_gather_on_main(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test gather to main rank."""
        config = DDPLearnerConfig(world_size=4, rank=0)

        with mock_ddp_context(world_size=4, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP") as mock_dist:
            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
                network=simple_network,
            )
            learner.initialize()

            tensor = torch.tensor([1.0, 2.0, 3.0])
            result = learner.gather(tensor, dst=0)

            assert mock_dist.state.gather_count == 1
            assert result is not None
            assert len(result) == 4  # world_size

            learner.cleanup()


class TestDDPLearnerCheckpointing:
    """Tests for DDP checkpoint save/load."""

    def test_save_checkpoint_main_rank(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test that main rank (rank=0) saves checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Main rank (0, world_size=1 to avoid distributed setup)
            config_main = DDPLearnerConfig(world_size=1, rank=0)
            learner_main = DDPLearner(
                config=config_main,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
                network=simple_network,
            )
            learner_main.initialize()
            learner_main.save_checkpoint(str(checkpoint_path), step=100)

            assert checkpoint_path.exists()

    def test_is_main_property_for_worker_rank(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test that is_main property is False for non-zero ranks."""
        # Worker rank (non-0), world_size can be anything
        config_worker = DDPLearnerConfig(world_size=4, rank=2)
        learner_worker = DDPLearner(
            config=config_worker,
            shared_storage=mock_shared_storage,
            replay_buffer=mock_replay_buffer,
            network=simple_network,
        )

        # Don't initialize (would require distributed), just check property
        assert learner_worker.is_main is False
        assert learner_worker.rank == 2

    def test_load_checkpoint(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test checkpoint load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            # Save a checkpoint
            config = DDPLearnerConfig(world_size=1, rank=0)
            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
                network=SimpleNetwork(),
            )
            learner.initialize()
            learner.save_checkpoint(str(checkpoint_path), step=50)

            # Load the checkpoint (same learner)
            checkpoint = learner.load_checkpoint(str(checkpoint_path))
            assert checkpoint["step"] == 50

    def test_checkpoint_additional_state(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test checkpoint with additional custom state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"

            config = DDPLearnerConfig(world_size=1, rank=0)
            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
                network=simple_network,
            )
            learner.initialize()

            additional = {"training_info": {"loss": 0.5}, "epoch": 10}
            learner.save_checkpoint(str(checkpoint_path), step=200, additional_state=additional)

            checkpoint = learner.load_checkpoint(str(checkpoint_path))
            assert checkpoint["step"] == 200
            assert checkpoint["training_info"]["loss"] == 0.5
            assert checkpoint["epoch"] == 10


class TestDDPLearnerModelAccess:
    """Tests for model access methods."""

    def test_get_model_single_node(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test get_model returns base model on single node."""
        config = DDPLearnerConfig(world_size=1)
        learner = DDPLearner(
            config=config,
            shared_storage=mock_shared_storage,
            replay_buffer=mock_replay_buffer,
            network=simple_network,
        )
        learner.initialize()

        model = learner.get_model()
        assert model is simple_network

    def test_get_base_model_unwraps_ddp(
        self,
        simple_network: SimpleNetwork,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test get_base_model returns unwrapped model."""
        config = DDPLearnerConfig(world_size=4, rank=0)

        with mock_ddp_context(world_size=4, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP"):
            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
                network=simple_network,
            )
            learner.initialize()

            # Both should return the base network since MockDDPWrapper stores it
            base = learner.get_base_model()
            assert base is simple_network

            learner.cleanup()


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_world_size_parametrized(
    world_size: int,
    simple_network: SimpleNetwork,
    mock_shared_storage: MagicMock,
    mock_replay_buffer: MagicMock,
) -> None:
    """Test DDPLearner with various world sizes."""
    config = DDPLearnerConfig(world_size=world_size, rank=0)

    if world_size > 1:
        with mock_ddp_context(world_size=world_size, rank=0, ddp_module_path="alpha_zero.distributed.ddp_learner.DDP") as mock_dist:
            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
                network=simple_network,
            )
            learner.initialize()

            assert learner.world_size == world_size
            assert mock_dist.state.world_size == world_size

            learner.cleanup()
    else:
        learner = DDPLearner(
            config=config,
            shared_storage=mock_shared_storage,
            replay_buffer=mock_replay_buffer,
            network=simple_network,
        )
        learner.initialize()
        assert learner.world_size == 1
