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
    mock_ddp_context,
)

if TYPE_CHECKING:
    pass

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
    storage.get_model_weights.return_value = {}
    storage.set_model_weights.return_value = 1
    storage.get_model_version.return_value = 1
    storage.should_stop.return_value = False
    storage.set_training.return_value = None
    storage.set_training_step.return_value = None
    storage.add_loss_entry.return_value = None
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
    buffer.sample_batch.return_value = None
    buffer.is_ready.return_value = False
    buffer.size.return_value = 0
    buffer.process_pending_trajectories.return_value = None
    return buffer


class TestDDPLearnerConfig:
    """Tests for DDPLearnerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DDPLearnerConfig()

        # Check DDP-prefixed attributes (our implementation)
        assert config.ddp_backend == "auto"
        assert config.init_method == "env://"
        assert config.ddp_world_size == 1
        assert config.ddp_rank == 0
        assert config.ddp_local_rank == 0
        assert config.master_port == "29500"
        assert config.find_unused_parameters is False
        assert config.broadcast_buffers is True
        assert config.gradient_as_bucket_view is True

    def test_custom_values(self) -> None:
        """Test configuration with custom values."""
        config = DDPLearnerConfig(
            ddp_backend="gloo",
            ddp_world_size=4,
            ddp_rank=2,
            ddp_local_rank=2,
            master_addr="192.168.1.100",
            master_port="12345",
            find_unused_parameters=True,
        )

        assert config.ddp_backend == "gloo"
        assert config.ddp_world_size == 4
        assert config.ddp_rank == 2
        assert config.ddp_local_rank == 2
        assert config.master_addr == "192.168.1.100"
        assert config.master_port == "12345"
        assert config.find_unused_parameters is True

    def test_from_env_defaults(self) -> None:
        """Test from_env with no environment variables."""
        with patch.dict("os.environ", {}, clear=True):
            config = DDPLearnerConfig.from_env()

        assert config.ddp_world_size == 1
        assert config.ddp_rank == 0
        assert config.ddp_local_rank == 0
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

        assert config.ddp_world_size == 8
        assert config.ddp_rank == 3
        assert config.ddp_local_rank == 3
        assert config.master_addr == "10.0.0.1"
        assert config.master_port == "29501"

    def test_from_env_with_overrides(self) -> None:
        """Test from_env with kwarg overrides."""
        env_vars = {"WORLD_SIZE": "4", "RANK": "2"}
        with patch.dict("os.environ", env_vars, clear=True):
            config = DDPLearnerConfig.from_env(ddp_world_size=8, ddp_backend="gloo")

        assert config.ddp_world_size == 8  # Override
        assert config.ddp_rank == 2  # From env
        assert config.ddp_backend == "gloo"  # Override

    def test_ddp_enabled_auto_detection(self) -> None:
        """Test that ddp_enabled is auto-set when world_size > 1."""
        env_vars = {"WORLD_SIZE": "4", "RANK": "0"}
        with patch.dict("os.environ", env_vars, clear=True):
            config = DDPLearnerConfig.from_env()

        assert config.ddp_enabled is True
        assert config.ddp_world_size == 4


class TestDDPLearnerInitialization:
    """Tests for DDPLearner initialization."""

    def test_init_single_node(
        self,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test initialization with world_size=1 (no DDP)."""
        config = DDPLearnerConfig(ddp_world_size=1, ddp_rank=0)

        # Mock the environment and MicroCivEnv
        with patch("alpha_zero.distributed.ddp_learner.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = 100
            mock_env_cls.return_value = mock_env

            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
            )

            assert learner.rank == 0
            assert learner.world_size == 1
            assert learner.is_main is True
            assert learner.is_main_process() is True

    def test_init_non_main_rank(
        self,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test initialization on non-main rank.

        Note: When ddp_enabled=False, is_main returns True because DDP is not active.
        This test verifies rank and local_rank are set correctly from config.
        """
        config = DDPLearnerConfig(
            ddp_world_size=4,
            ddp_rank=2,
            ddp_local_rank=0,
            ddp_enabled=False,  # Disable to avoid actual DDP init
        )

        with patch("alpha_zero.distributed.ddp_learner.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = 100
            mock_env_cls.return_value = mock_env

            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
            )

            # Rank is stored from config
            assert learner.rank == 2
            assert learner.local_rank == 0
            # When DDP is disabled, is_main returns True (single-process semantics)
            assert learner.is_main_process() is True


class TestDDPLearnerCommunication:
    """Tests for DDP communication primitives."""

    def test_barrier_single_node(
        self,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test barrier on single node (no-op)."""
        config = DDPLearnerConfig(ddp_world_size=1)

        with patch("alpha_zero.distributed.ddp_learner.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = 100
            mock_env_cls.return_value = mock_env

            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
            )

            # Should not raise on single node
            learner.barrier()

    def test_all_reduce_single_node(
        self,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test all_reduce on single node (no-op)."""
        config = DDPLearnerConfig(ddp_world_size=1)

        with patch("alpha_zero.distributed.ddp_learner.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = 100
            mock_env_cls.return_value = mock_env

            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
            )

            tensor = torch.ones(5)
            result = learner.all_reduce(tensor)

            # On single node, tensor should be unchanged
            assert torch.allclose(result, torch.ones(5))

    def test_gather_single_node(
        self,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test gather on single node."""
        config = DDPLearnerConfig(ddp_world_size=1)

        with patch("alpha_zero.distributed.ddp_learner.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = 100
            mock_env_cls.return_value = mock_env

            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
            )

            tensor = torch.tensor([1.0, 2.0, 3.0])
            result = learner.gather(tensor, dst=0)

            assert result is not None
            assert len(result) == 1  # world_size=1
            assert torch.allclose(result[0], tensor)


class TestDDPLearnerModelAccess:
    """Tests for model access methods."""

    def test_get_model_single_node(
        self,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test get_model returns the network on single node."""
        config = DDPLearnerConfig(ddp_world_size=1)

        with patch("alpha_zero.distributed.ddp_learner.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = 100
            mock_env_cls.return_value = mock_env

            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
            )

            model = learner.get_model()
            base_model = learner.get_base_model()

            # On single node, both should return the same network
            assert model is learner.network
            assert base_model is learner.network

    def test_get_base_model(
        self,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test get_base_model returns unwrapped model."""
        config = DDPLearnerConfig(ddp_world_size=1)

        with patch("alpha_zero.distributed.ddp_learner.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = 100
            mock_env_cls.return_value = mock_env

            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
            )

            base = learner.get_base_model()
            # Should be the same as _get_base_network
            assert base is learner._get_base_network()


class TestDDPLearnerStatistics:
    """Tests for learner statistics."""

    def test_get_statistics(
        self,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test get_statistics returns expected fields."""
        config = DDPLearnerConfig(ddp_world_size=1)

        with patch("alpha_zero.distributed.ddp_learner.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = 100
            mock_env_cls.return_value = mock_env

            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
            )

            stats = learner.get_statistics()

            assert "training_step" in stats
            assert "ddp_enabled" in stats
            assert "ddp_rank" in stats
            assert "ddp_world_size" in stats
            assert stats["ddp_world_size"] == 1
            assert stats["ddp_rank"] == 0


class TestDDPLearnerCleanup:
    """Tests for cleanup functionality."""

    def test_cleanup_single_node(
        self,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test cleanup on single node."""
        config = DDPLearnerConfig(ddp_world_size=1)

        with patch("alpha_zero.distributed.ddp_learner.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = 100
            mock_env_cls.return_value = mock_env

            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
            )

            # Should not raise
            learner.cleanup()


class TestDDPLearnerAllReduceLosses:
    """Tests for loss all-reduce functionality."""

    def test_all_reduce_losses_single_node(
        self,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test all_reduce_losses on single node (no-op)."""
        config = DDPLearnerConfig(ddp_world_size=1)

        with patch("alpha_zero.distributed.ddp_learner.MicroCivEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.action_space.n = 100
            mock_env_cls.return_value = mock_env

            learner = DDPLearner(
                config=config,
                shared_storage=mock_shared_storage,
                replay_buffer=mock_replay_buffer,
            )

            losses = {
                "total_loss": 1.5,
                "policy_loss": 0.5,
                "value_loss": 1.0,
                "learning_rate": 0.001,
            }

            result = learner.all_reduce_losses(losses)

            # On single node, losses should be unchanged
            assert result["total_loss"] == 1.5
            assert result["policy_loss"] == 0.5
            assert result["value_loss"] == 1.0


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_world_size_parametrized(
    world_size: int,
    mock_shared_storage: MagicMock,
    mock_replay_buffer: MagicMock,
) -> None:
    """Test DDPLearnerConfig with various world sizes."""
    config = DDPLearnerConfig(ddp_world_size=world_size, ddp_rank=0, ddp_enabled=False)

    with patch("alpha_zero.distributed.ddp_learner.MicroCivEnv") as mock_env_cls:
        mock_env = MagicMock()
        mock_env.action_space.n = 100
        mock_env_cls.return_value = mock_env

        learner = DDPLearner(
            config=config,
            shared_storage=mock_shared_storage,
            replay_buffer=mock_replay_buffer,
        )

        assert learner.world_size == world_size
        assert learner.rank == 0
