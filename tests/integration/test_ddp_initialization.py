"""Integration tests for DDP (DistributedDataParallel) initialization.

Tests DDP process group initialization, device selection, and cleanup
using MockDistributedBackend for isolated testing without actual
multi-process setup.

Run with:
    pytest tests/integration/test_ddp_initialization.py -v
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

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
        ddp_world_size=1,
        ddp_rank=0,
        ddp_backend="gloo",
        device="cpu",
    )


@pytest.fixture
def ddp_config_multi_process() -> DDPLearnerConfig:
    """Create DDP config for multi-process simulation."""
    return DDPLearnerConfig(
        board_size=8,
        num_res_blocks=2,
        num_filters=32,
        batch_size=4,
        learning_rate=0.001,
        ddp_world_size=4,
        ddp_rank=0,
        ddp_backend="gloo",
        device="cpu",
    )


@pytest.fixture
def mock_backend() -> Generator[MockDistributedBackend, None, None]:
    """Create mock distributed backend."""
    backend = MockDistributedBackend(world_size=1, rank=0)
    with backend.patch():
        yield backend


@pytest.fixture
def mock_backend_multi() -> Generator[MockDistributedBackend, None, None]:
    """Create mock distributed backend for multi-process."""
    backend = MockDistributedBackend(world_size=4, rank=0)
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


# ============================================================================
# DDP Config Tests
# ============================================================================


class TestDDPLearnerConfig:
    """Tests for DDPLearnerConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = DDPLearnerConfig()
        assert config.backend == "auto"
        assert config.world_size == 1
        assert config.rank == 0
        assert config.init_method == "env://"

    def test_from_env_reads_environment(self) -> None:
        """Test config loads from environment variables."""
        with patch.dict(
            os.environ,
            {
                "WORLD_SIZE": "4",
                "RANK": "2",
                "LOCAL_RANK": "1",
                "MASTER_ADDR": "node0",
                "MASTER_PORT": "29501",
            },
        ):
            config = DDPLearnerConfig.from_env()
            assert config.ddp_world_size == 4
            assert config.ddp_rank == 2
            assert config.ddp_local_rank == 1
            assert config.master_addr == "node0"
            assert config.master_port == "29501"

    def test_from_env_with_overrides(self) -> None:
        """Test config from env with explicit overrides."""
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            config = DDPLearnerConfig.from_env(ddp_world_size=8, ddp_rank=3)
            assert config.ddp_world_size == 8  # Override takes precedence
            assert config.ddp_rank == 3

    def test_gloo_backend_for_cpu(self) -> None:
        """Test gloo backend is used for CPU training."""
        config = DDPLearnerConfig(ddp_backend="gloo", device="cpu")
        assert config.ddp_backend == "gloo"
        assert config.device == "cpu"


# ============================================================================
# DDP Initialization Tests
# ============================================================================


class TestDDPInitSingleProcess:
    """Tests for DDP initialization with world_size=1."""

    def test_ddp_init_world_size_one(
        self,
        mock_backend: MockDistributedBackend,
        ddp_config: DDPLearnerConfig,
        mock_shared_storage: MagicMock,
        mock_replay_buffer: MagicMock,
    ) -> None:
        """Test DDP initialization with single process."""
        # Initialize process group
        torch.distributed.init_process_group(
            backend=ddp_config.ddp_backend,
            world_size=ddp_config.ddp_world_size,
            rank=ddp_config.ddp_rank,
        )

        assert mock_backend.state.is_initialized
        assert mock_backend.state.world_size == 1
        assert mock_backend.state.rank == 0

    def test_ddp_config_properties(
        self,
        ddp_config: DDPLearnerConfig,
    ) -> None:
        """Test DDPLearnerConfig properties."""
        assert ddp_config.ddp_world_size == 1
        assert ddp_config.ddp_rank == 0
        assert ddp_config.ddp_backend == "gloo"


class TestDDPInitMultiProcessMocked:
    """Tests for DDP initialization with mocked multi-process."""

    def test_ddp_init_multi_process(
        self,
        mock_backend_multi: MockDistributedBackend,
        ddp_config_multi_process: DDPLearnerConfig,
    ) -> None:
        """Test DDP initialization with world_size=4 (mocked)."""
        torch.distributed.init_process_group(
            backend=ddp_config_multi_process.ddp_backend,
            world_size=ddp_config_multi_process.ddp_world_size,
            rank=ddp_config_multi_process.ddp_rank,
        )

        assert mock_backend_multi.state.is_initialized
        assert mock_backend_multi.state.world_size == 4
        assert mock_backend_multi.state.rank == 0

    def test_ddp_barrier_synchronization(
        self,
        mock_backend_multi: MockDistributedBackend,
    ) -> None:
        """Test barrier synchronization in multi-process setup."""
        torch.distributed.init_process_group(backend="gloo", world_size=4, rank=0)

        # Barrier should be trackable
        torch.distributed.barrier()
        assert mock_backend_multi.state.barrier_count == 1

        torch.distributed.barrier()
        assert mock_backend_multi.state.barrier_count == 2

    def test_ddp_all_reduce_operation(
        self,
        mock_backend_multi: MockDistributedBackend,
    ) -> None:
        """Test all_reduce operation in multi-process setup."""
        torch.distributed.init_process_group(backend="gloo", world_size=4, rank=0)

        tensor = torch.ones(10)
        torch.distributed.all_reduce(tensor)

        # Mock multiplies by world_size to simulate sum
        assert torch.allclose(tensor, torch.full((10,), 4.0))
        assert mock_backend_multi.state.all_reduce_count == 1


# ============================================================================
# Device Selection Tests
# ============================================================================


class TestDDPDeviceSelection:
    """Tests for device selection in DDP."""

    def test_cpu_device_selection(self, ddp_config: DDPLearnerConfig) -> None:
        """Test CPU device is correctly selected."""
        assert ddp_config.device == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_selection(self) -> None:
        """Test CUDA device selection when available."""
        config = DDPLearnerConfig(device="cuda", ddp_local_rank=0)
        assert config.device == "cuda"
        assert config.ddp_local_rank == 0

    def test_device_from_local_rank(self) -> None:
        """Test device assignment from local_rank."""
        config = DDPLearnerConfig(ddp_local_rank=2, device="cpu")
        # With local_rank=2, in CUDA setup this would be cuda:2
        assert config.ddp_local_rank == 2


# ============================================================================
# Process Group Cleanup Tests
# ============================================================================


class TestDDPProcessGroupCleanup:
    """Tests for process group cleanup."""

    def test_destroy_process_group(
        self,
        mock_backend: MockDistributedBackend,
    ) -> None:
        """Test process group is properly destroyed."""
        torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)
        assert mock_backend.state.is_initialized

        torch.distributed.destroy_process_group()
        assert not mock_backend.state.is_initialized

    def test_cleanup_after_error(
        self,
        mock_backend: MockDistributedBackend,
    ) -> None:
        """Test cleanup happens even after errors."""
        torch.distributed.init_process_group(backend="gloo", world_size=1, rank=0)

        try:
            raise RuntimeError("Simulated error")
        except RuntimeError:
            pass
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

        assert not mock_backend.state.is_initialized


# ============================================================================
# MockDDPWrapper Tests
# ============================================================================


class TestMockDDPWrapper:
    """Tests for MockDDPWrapper functionality."""

    def test_wrapper_forward_pass(self) -> None:
        """Test DDP wrapper forward pass."""
        model = nn.Linear(10, 5)
        wrapped = MockDDPWrapper(model)

        inputs = torch.randn(4, 10)
        outputs = wrapped(inputs)

        assert outputs.shape == (4, 5)

    def test_wrapper_train_eval_modes(self) -> None:
        """Test DDP wrapper train/eval mode switching."""
        model = nn.Linear(10, 5)
        wrapped = MockDDPWrapper(model)

        wrapped.train()
        assert model.training

        wrapped.eval()
        assert not model.training

    def test_wrapper_state_dict(self) -> None:
        """Test DDP wrapper state dict access."""
        model = nn.Linear(10, 5)
        wrapped = MockDDPWrapper(model)

        state = wrapped.state_dict()
        assert "weight" in state
        assert "bias" in state

    def test_wrapper_load_state_dict(self) -> None:
        """Test DDP wrapper state dict loading."""
        model1 = nn.Linear(10, 5)
        model2 = nn.Linear(10, 5)
        wrapped = MockDDPWrapper(model2)

        # Load state from model1 into wrapped model2
        wrapped.load_state_dict(model1.state_dict())

        # States should match
        for p1, p2 in zip(model1.parameters(), wrapped.module.parameters(), strict=True):
            assert torch.allclose(p1, p2)

    def test_wrapper_parameters_iteration(self) -> None:
        """Test DDP wrapper parameter iteration."""
        model = nn.Linear(10, 5)
        wrapped = MockDDPWrapper(model)

        params = list(wrapped.parameters())
        assert len(params) == 2  # weight and bias


# ============================================================================
# Integration Tests
# ============================================================================


class TestDDPIntegration:
    """Integration tests combining multiple DDP components."""

    def test_full_ddp_setup_and_teardown(
        self,
        mock_backend: MockDistributedBackend,
        ddp_config: DDPLearnerConfig,
    ) -> None:
        """Test complete DDP setup and teardown cycle."""
        # Setup
        torch.distributed.init_process_group(
            backend=ddp_config.ddp_backend,
            world_size=ddp_config.ddp_world_size,
            rank=ddp_config.ddp_rank,
        )
        assert torch.distributed.is_initialized()

        # Create mock DDP-wrapped model
        model = nn.Linear(10, 5)
        wrapped = MockDDPWrapper(model)

        # Simulate training step
        inputs = torch.randn(4, 10)
        outputs = wrapped(inputs)
        loss = outputs.sum()
        loss.backward()

        # Gradient sync
        for param in wrapped.parameters():
            if param.grad is not None:
                torch.distributed.all_reduce(param.grad)

        # Teardown
        torch.distributed.destroy_process_group()
        assert not torch.distributed.is_initialized()

    def test_ddp_with_different_backends(
        self,
    ) -> None:
        """Test DDP initialization with different backends."""
        backends = ["gloo"]  # nccl requires CUDA

        for backend in backends:
            mock_backend = MockDistributedBackend(world_size=2, rank=0, backend=backend)
            with mock_backend.patch():
                torch.distributed.init_process_group(
                    backend=backend,
                    world_size=2,
                    rank=0,
                )
                assert mock_backend.state.backend == backend
                torch.distributed.destroy_process_group()

    def test_operation_counts_tracking(
        self,
        mock_backend_multi: MockDistributedBackend,
    ) -> None:
        """Test operation counts are properly tracked."""
        torch.distributed.init_process_group(backend="gloo", world_size=4, rank=0)

        # Perform various operations
        torch.distributed.barrier()
        torch.distributed.all_reduce(torch.ones(5))
        torch.distributed.broadcast(torch.ones(5), src=0)

        assert mock_backend_multi.state.barrier_count == 1
        assert mock_backend_multi.state.all_reduce_count == 1
        assert mock_backend_multi.state.broadcast_count == 1

        # Reset and verify
        mock_backend_multi.reset_counts()
        assert mock_backend_multi.state.barrier_count == 0
        assert mock_backend_multi.state.all_reduce_count == 0
        assert mock_backend_multi.state.broadcast_count == 0
