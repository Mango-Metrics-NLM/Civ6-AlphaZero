"""Unit tests for gradient synchronization in distributed training.

Tests gradient accumulation, all-reduce operations, weight broadcasting,
and gradient clipping with DDP wrapper using MockDistributedBackend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn

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
def mock_backend() -> Generator[MockDistributedBackend, None, None]:
    """Create mock distributed backend with world_size=4."""
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
def simple_network() -> nn.Module:
    """Create simple network for gradient testing."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )


@pytest.fixture
def ddp_network(simple_network: nn.Module) -> MockDDPWrapper:
    """Create DDP-wrapped network for testing."""
    return MockDDPWrapper(simple_network, device_ids=[0])


# ============================================================================
# All-Reduce Tests
# ============================================================================


class TestAllReduceGradientAccumulation:
    """Tests for all-reduce gradient accumulation."""

    def test_all_reduce_sums_gradients(self, mock_backend: MockDistributedBackend) -> None:
        """Verify gradients are correctly summed across processes."""
        # Create a tensor representing gradients
        gradients = torch.ones(10)

        # Perform all_reduce (simulates summing across world_size processes)
        torch.distributed.all_reduce(gradients)

        # With world_size=4 and initial value 1, result should be 4
        assert torch.allclose(gradients, torch.full((10,), 4.0))
        assert mock_backend.state.all_reduce_count == 1

    def test_all_reduce_called_for_each_parameter(
        self, mock_backend: MockDistributedBackend, simple_network: nn.Module
    ) -> None:
        """Verify all_reduce is called for each parameter's gradient."""
        # Set up mock gradients for all parameters
        for param in simple_network.parameters():
            param.grad = torch.ones_like(param)

        # Simulate gradient synchronization for all parameters
        for param in simple_network.parameters():
            if param.grad is not None:
                torch.distributed.all_reduce(param.grad)

        # Should have called all_reduce for each parameter (4 in simple network)
        num_params = sum(1 for _ in simple_network.parameters())
        assert mock_backend.state.all_reduce_count == num_params

    def test_all_reduce_handles_different_tensor_shapes(
        self, mock_backend: MockDistributedBackend
    ) -> None:
        """Verify all_reduce works with various tensor shapes."""
        shapes = [(10,), (5, 5), (2, 3, 4), (1, 1, 1, 1)]

        for shape in shapes:
            tensor = torch.ones(shape)
            initial_count = mock_backend.state.all_reduce_count
            torch.distributed.all_reduce(tensor)

            # All values should be multiplied by world_size
            expected = torch.full(shape, float(mock_backend.state.world_size))
            assert torch.allclose(tensor, expected)
            assert mock_backend.state.all_reduce_count == initial_count + 1


# ============================================================================
# Broadcast Tests
# ============================================================================


class TestWeightBroadcastFromMain:
    """Tests for weight broadcasting from main rank."""

    def test_broadcast_called_from_rank_0(self, mock_backend: MockDistributedBackend) -> None:
        """Verify weights broadcast correctly from rank 0."""
        weights = torch.randn(10, 20)

        torch.distributed.broadcast(weights, src=0)

        assert mock_backend.state.broadcast_count == 1

    def test_broadcast_all_parameters(
        self, mock_backend: MockDistributedBackend, simple_network: nn.Module
    ) -> None:
        """Verify all parameters are broadcast."""
        for param in simple_network.parameters():
            torch.distributed.broadcast(param.data, src=0)

        num_params = sum(1 for _ in simple_network.parameters())
        assert mock_backend.state.broadcast_count == num_params

    def test_broadcast_to_non_main_rank(self, mock_backend_rank_1: MockDistributedBackend) -> None:
        """Verify broadcast works when called from non-main rank."""
        weights = torch.randn(5, 5)

        # Broadcast from rank 0 (we're on rank 1)
        torch.distributed.broadcast(weights, src=0)

        assert mock_backend_rank_1.state.broadcast_count == 1


# ============================================================================
# Gradient Clipping with DDP Tests
# ============================================================================


class TestGradientClippingWithDDP:
    """Tests for gradient clipping with DDP wrapper."""

    def test_gradient_clipping_preserves_direction(self, ddp_network: MockDDPWrapper) -> None:
        """Verify gradient clipping preserves gradient direction."""
        # Set large gradients
        for param in ddp_network.parameters():
            param.grad = torch.randn_like(param) * 100  # Large gradients

        # Get gradient directions before clipping
        directions_before = []
        for param in ddp_network.parameters():
            if param.grad is not None:
                direction = param.grad / (param.grad.norm() + 1e-8)
                directions_before.append(direction.clone())

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(ddp_network.parameters(), max_norm=1.0)

        # Check directions are preserved (within tolerance)
        for i, param in enumerate(ddp_network.parameters()):
            if param.grad is not None:
                direction_after = param.grad / (param.grad.norm() + 1e-8)
                # Directions should be approximately the same
                cosine_sim = (directions_before[i] * direction_after).sum()
                assert cosine_sim > 0.99, f"Direction changed for param {i}"

    def test_gradient_clipping_limits_norm(self, ddp_network: MockDDPWrapper) -> None:
        """Verify gradient clipping limits total gradient norm."""
        # Set very large gradients
        for param in ddp_network.parameters():
            param.grad = torch.ones_like(param) * 1000

        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(ddp_network.parameters(), max_norm=max_norm)

        # Calculate clipped gradient norm
        clipped_norm = 0.0
        for param in ddp_network.parameters():
            if param.grad is not None:
                clipped_norm += param.grad.norm() ** 2
        clipped_norm = clipped_norm**0.5

        # Clipped norm should be at most max_norm (within tolerance)
        assert clipped_norm <= max_norm + 1e-5

    def test_gradient_clipping_no_effect_on_small_gradients(
        self, ddp_network: MockDDPWrapper
    ) -> None:
        """Verify small gradients are not affected by clipping."""
        # Set small gradients
        for param in ddp_network.parameters():
            param.grad = torch.ones_like(param) * 0.001

        # Store original gradients
        original_grads = [
            param.grad.clone() for param in ddp_network.parameters() if param.grad is not None
        ]

        max_norm = 10.0  # Much larger than actual gradient norm
        torch.nn.utils.clip_grad_norm_(ddp_network.parameters(), max_norm=max_norm)

        # Gradients should be unchanged
        for original, param in zip(original_grads, ddp_network.parameters(), strict=True):
            if param.grad is not None:
                assert torch.allclose(original, param.grad)


# ============================================================================
# Gradient Accumulation Steps Tests
# ============================================================================


class TestGradientAccumulationWithDDP:
    """Tests for gradient accumulation with DDP."""

    def test_gradient_accumulation_sums_gradients(self, ddp_network: MockDDPWrapper) -> None:
        """Verify gradients accumulate correctly over multiple steps."""
        accumulation_steps = 4
        batch_size = 8
        input_dim = 10

        # Initialize gradients
        for param in ddp_network.parameters():
            param.grad = None

        # Accumulate gradients over multiple steps
        for _step in range(accumulation_steps):
            # Create random input
            inputs = torch.randn(batch_size, input_dim)

            # Forward pass
            outputs = ddp_network(inputs)
            loss = outputs.sum()

            # Backward pass (gradients accumulate)
            loss.backward()

        # All parameters should have gradients
        for param in ddp_network.parameters():
            assert param.grad is not None
            # Gradient should be non-zero (accumulated)
            assert param.grad.abs().sum() > 0

    def test_gradient_zero_resets_accumulation(self, ddp_network: MockDDPWrapper) -> None:
        """Verify optimizer.zero_grad() resets accumulated gradients."""
        # Create optimizer
        optimizer = torch.optim.SGD(ddp_network.parameters(), lr=0.01)

        # Accumulate some gradients
        inputs = torch.randn(4, 10)
        outputs = ddp_network(inputs)
        outputs.sum().backward()

        # Verify gradients exist
        for param in ddp_network.parameters():
            assert param.grad is not None

        # Zero gradients
        optimizer.zero_grad()

        # Gradients should be zero (or None depending on set_to_none)
        for param in ddp_network.parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_gradient_accumulation_with_sync_interval(
        self, mock_backend: MockDistributedBackend, simple_network: nn.Module
    ) -> None:
        """Test gradient sync only happens at specified intervals."""
        sync_interval = 4
        total_steps = 10

        for step in range(total_steps):
            # Simulate forward and backward
            inputs = torch.randn(4, 10)
            outputs = simple_network(inputs)
            outputs.sum().backward()

            # Only sync at interval
            if (step + 1) % sync_interval == 0:
                for param in simple_network.parameters():
                    if param.grad is not None:
                        torch.distributed.all_reduce(param.grad)

        # Should have synced floor(total_steps / sync_interval) times
        expected_syncs = total_steps // sync_interval
        num_params = sum(1 for _ in simple_network.parameters())
        expected_all_reduce_calls = expected_syncs * num_params

        assert mock_backend.state.all_reduce_count == expected_all_reduce_calls


# ============================================================================
# Barrier Synchronization Tests
# ============================================================================


class TestBarrierSynchronization:
    """Tests for barrier synchronization in distributed training."""

    def test_barrier_increments_count(self, mock_backend: MockDistributedBackend) -> None:
        """Verify barrier call is tracked."""
        assert mock_backend.state.barrier_count == 0

        torch.distributed.barrier()

        assert mock_backend.state.barrier_count == 1

    def test_multiple_barriers(self, mock_backend: MockDistributedBackend) -> None:
        """Verify multiple barrier calls are tracked."""
        for i in range(5):
            torch.distributed.barrier()
            assert mock_backend.state.barrier_count == i + 1

    def test_barrier_before_checkpoint(self, mock_backend: MockDistributedBackend) -> None:
        """Simulate barrier usage before checkpoint save."""
        # This is the pattern used in distributed checkpointing
        torch.distributed.barrier()  # Wait for all workers

        # Only main rank saves (simulated)
        if mock_backend.state.rank == 0:
            # Simulated checkpoint save
            pass

        torch.distributed.barrier()  # Wait for save to complete

        assert mock_backend.state.barrier_count == 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestDistributedGradientSyncIntegration:
    """Integration tests for distributed gradient synchronization."""

    def test_full_training_step_with_sync(
        self, mock_backend: MockDistributedBackend, simple_network: nn.Module
    ) -> None:
        """Test complete training step with gradient sync."""
        optimizer = torch.optim.SGD(simple_network.parameters(), lr=0.01)

        # Forward pass
        inputs = torch.randn(8, 10)
        outputs = simple_network(inputs)
        loss = outputs.sum()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Sync gradients across workers
        for param in simple_network.parameters():
            if param.grad is not None:
                torch.distributed.all_reduce(param.grad)
                # Average gradients (divide by world_size)
                param.grad /= mock_backend.state.world_size

        # Optimizer step
        optimizer.step()

        # Verify all_reduce was called for each parameter
        num_params = sum(1 for _ in simple_network.parameters())
        assert mock_backend.state.all_reduce_count == num_params

    def test_ddp_wrapper_preserves_module_functionality(self, ddp_network: MockDDPWrapper) -> None:
        """Verify DDP wrapper doesn't break module functionality."""
        # Test forward pass
        inputs = torch.randn(4, 10)
        outputs = ddp_network(inputs)
        assert outputs.shape == (4, 5)  # 5 output features

        # Test backward pass
        outputs.sum().backward()
        for param in ddp_network.parameters():
            assert param.grad is not None

        # Test train/eval mode switching
        ddp_network.train()
        assert ddp_network.module.training

        ddp_network.eval()
        assert not ddp_network.module.training

        # Test state dict
        state_dict = ddp_network.state_dict()
        assert len(state_dict) > 0

    def test_reset_counts_clears_all(self, mock_backend: MockDistributedBackend) -> None:
        """Verify reset_counts clears all operation counters."""
        # Perform various operations
        torch.distributed.barrier()
        torch.distributed.all_reduce(torch.ones(5))
        torch.distributed.broadcast(torch.ones(5), src=0)

        # Verify counts are non-zero
        assert mock_backend.state.barrier_count > 0
        assert mock_backend.state.all_reduce_count > 0
        assert mock_backend.state.broadcast_count > 0

        # Reset
        mock_backend.reset_counts()

        # All counts should be zero
        assert mock_backend.state.barrier_count == 0
        assert mock_backend.state.all_reduce_count == 0
        assert mock_backend.state.broadcast_count == 0
        assert mock_backend.state.gather_count == 0
        assert mock_backend.state.reduce_count == 0
