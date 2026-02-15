"""Unit tests for CUDA graph caching optimization.

Tests the CUDAGraphConfig and CUDAGraphCache classes which provide
CUDA graph caching for faster inference on GPU.
"""

from __future__ import annotations

import pytest
import torch

from alpha_zero.optimizations.cuda_graph_cache import (
    CUDAGraphCache,
    CUDAGraphConfig,
)


class MockNetwork:
    """Mock network for testing CUDA graph cache."""

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mock prediction that returns valid policy and value tensors.

        Args:
            spatial: Spatial input tensor of shape (B, C, H, W)
            vector: Vector input tensor of shape (B, V)
            mask: Action mask of shape (B, A)

        Returns:
            Tuple of (policy, value) where:
                - policy has shape (B, A) with valid probabilities
                - value has shape (B, 1)
        """
        batch_size = spatial.shape[0]
        action_size = mask.shape[-1]

        # Create valid policy (masked and normalized)
        policy = (
            torch.ones(batch_size, action_size, device=spatial.device, dtype=spatial.dtype)
            / action_size
            * mask
        )
        policy = policy / (policy.sum(dim=-1, keepdim=True) + 1e-8)

        # Create value tensor
        value = torch.zeros(batch_size, 1, device=spatial.device, dtype=spatial.dtype)

        return policy, value

    def eval(self) -> None:
        """Set network to evaluation mode."""
        pass


class TestCUDAGraphConfig:
    """Test suite for CUDAGraphConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = CUDAGraphConfig()

        assert config.enabled is True
        assert config.warmup_steps == 3
        assert config.max_cached_sizes == 8
        assert config.allow_cpu_fallback is True

    def test_invalid_warmup_steps(self) -> None:
        """Test that warmup_steps < 1 raises ValueError."""
        with pytest.raises(ValueError, match="warmup_steps must be >= 1"):
            CUDAGraphConfig(warmup_steps=0)

        with pytest.raises(ValueError, match="warmup_steps must be >= 1"):
            CUDAGraphConfig(warmup_steps=-1)

    def test_invalid_max_cached_sizes(self) -> None:
        """Test that max_cached_sizes < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_cached_sizes must be >= 1"):
            CUDAGraphConfig(max_cached_sizes=0)

        with pytest.raises(ValueError, match="max_cached_sizes must be >= 1"):
            CUDAGraphConfig(max_cached_sizes=-5)

    def test_valid_custom_config(self) -> None:
        """Test creating config with valid custom values."""
        config = CUDAGraphConfig(
            enabled=False,
            warmup_steps=5,
            max_cached_sizes=16,
            allow_cpu_fallback=False,
        )

        assert config.enabled is False
        assert config.warmup_steps == 5
        assert config.max_cached_sizes == 16
        assert config.allow_cpu_fallback is False


class TestCUDAGraphCache:
    """Test suite for CUDAGraphCache class."""

    def test_cpu_fallback(self) -> None:
        """Test that cache works on CPU (using direct predict, no CUDA graphs)."""
        network = MockNetwork()
        config = CUDAGraphConfig(allow_cpu_fallback=True)
        cache = CUDAGraphCache(network, device="cpu", config=config)

        # Create CPU tensors
        spatial = torch.randn(2, 13, 8, 8)
        vector = torch.randn(2, 64)
        mask = torch.ones(2, 376)

        # Should work without error on CPU
        policy, value = cache.predict_with_cache(spatial, vector, mask)

        assert policy.shape == (2, 376)
        assert value.shape == (2, 1)
        assert policy.device.type == "cpu"
        assert value.device.type == "cpu"

    def test_predict_returns_correct_shapes(self) -> None:
        """Test that predict returns (B, A) policy and (B, 1) value."""
        network = MockNetwork()
        cache = CUDAGraphCache(network, device="cpu")

        # Test different batch sizes
        for batch_size in [1, 2, 4, 8]:
            spatial = torch.randn(batch_size, 13, 8, 8)
            vector = torch.randn(batch_size, 64)
            mask = torch.ones(batch_size, 376)

            policy, value = cache.predict_with_cache(spatial, vector, mask)

            assert policy.shape == (batch_size, 376)
            assert value.shape == (batch_size, 1)

    def test_clear_cache(self) -> None:
        """Test that cache is empty after clear."""
        network = MockNetwork()
        cache = CUDAGraphCache(network, device="cpu")

        # Run predictions to potentially populate cache
        spatial = torch.randn(2, 13, 8, 8)
        vector = torch.randn(2, 64)
        mask = torch.ones(2, 376)
        cache.predict_with_cache(spatial, vector, mask)

        # Clear cache
        cache.clear_cache()

        # Cache should be empty
        assert len(cache.cached_sizes) == 0

    def test_cached_sizes_empty_initially(self) -> None:
        """Test that cached_sizes is empty on initialization."""
        network = MockNetwork()
        cache = CUDAGraphCache(network, device="cpu")

        assert len(cache.cached_sizes) == 0
        assert cache.cached_sizes == []

    def test_disabled_config(self) -> None:
        """Test that enabled=False uses direct predict without caching."""
        network = MockNetwork()
        config = CUDAGraphConfig(enabled=False)
        cache = CUDAGraphCache(network, device="cpu", config=config)

        spatial = torch.randn(2, 13, 8, 8)
        vector = torch.randn(2, 64)
        mask = torch.ones(2, 376)

        # Should work without caching
        policy, value = cache.predict_with_cache(spatial, vector, mask)

        assert policy.shape == (2, 376)
        assert value.shape == (2, 1)
        # Cache should remain empty with disabled config
        assert len(cache.cached_sizes) == 0

    def test_policy_normalized(self) -> None:
        """Test that returned policy is properly normalized."""
        network = MockNetwork()
        cache = CUDAGraphCache(network, device="cpu")

        spatial = torch.randn(3, 13, 8, 8)
        vector = torch.randn(3, 64)
        mask = torch.ones(3, 376)

        policy, _ = cache.predict_with_cache(spatial, vector, mask)

        # Policy should sum to 1.0 per batch element
        policy_sums = policy.sum(dim=-1)
        assert torch.allclose(policy_sums, torch.ones(3), atol=1e-6)

    def test_mask_applied(self) -> None:
        """Test that action mask is properly applied to policy."""
        network = MockNetwork()
        cache = CUDAGraphCache(network, device="cpu")

        spatial = torch.randn(1, 13, 8, 8)
        vector = torch.randn(1, 64)

        # Create mask with some actions disabled
        mask = torch.ones(1, 376)
        mask[0, :10] = 0  # Disable first 10 actions

        policy, _ = cache.predict_with_cache(spatial, vector, mask)

        # Disabled actions should have zero probability
        assert torch.allclose(policy[0, :10], torch.zeros(10), atol=1e-6)
        # Enabled actions should have non-zero probability
        assert (policy[0, 10:] > 0).all()

    # GPU-specific tests
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_graph_caching(self) -> None:
        """Test CUDA graph caching on GPU."""
        network = MockNetwork()
        config = CUDAGraphConfig(warmup_steps=2, max_cached_sizes=4)
        cache = CUDAGraphCache(network, device="cuda", config=config)

        # Run predictions with same batch size multiple times
        batch_size = 4
        for _ in range(5):
            spatial = torch.randn(batch_size, 13, 8, 8, device="cuda")
            vector = torch.randn(batch_size, 64, device="cuda")
            mask = torch.ones(batch_size, 376, device="cuda")

            policy, value = cache.predict_with_cache(spatial, vector, mask)

            assert policy.shape == (batch_size, 376)
            assert value.shape == (batch_size, 1)
            assert policy.device.type == "cuda"
            assert value.device.type == "cuda"

        # Should have cached this batch size
        assert batch_size in cache.cached_sizes

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multiple_batch_sizes_cached(self) -> None:
        """Test caching of multiple different batch sizes."""
        network = MockNetwork()
        config = CUDAGraphConfig(warmup_steps=1, max_cached_sizes=4)
        cache = CUDAGraphCache(network, device="cuda", config=config)

        # Run predictions with different batch sizes
        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            # Run enough times to trigger caching
            for _ in range(2):
                spatial = torch.randn(batch_size, 13, 8, 8, device="cuda")
                vector = torch.randn(batch_size, 64, device="cuda")
                mask = torch.ones(batch_size, 376, device="cuda")

                cache.predict_with_cache(spatial, vector, mask)

        # All batch sizes should be cached
        for batch_size in batch_sizes:
            assert batch_size in cache.cached_sizes

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_max_cached_sizes_limit(self) -> None:
        """Test that cache respects max_cached_sizes limit."""
        network = MockNetwork()
        config = CUDAGraphConfig(warmup_steps=1, max_cached_sizes=2)
        cache = CUDAGraphCache(network, device="cuda", config=config)

        # Try to cache more batch sizes than max
        batch_sizes = [1, 2, 4, 8, 16]
        for batch_size in batch_sizes:
            for _ in range(2):
                spatial = torch.randn(batch_size, 13, 8, 8, device="cuda")
                vector = torch.randn(batch_size, 64, device="cuda")
                mask = torch.ones(batch_size, 376, device="cuda")

                cache.predict_with_cache(spatial, vector, mask)

        # Should not exceed max_cached_sizes
        assert len(cache.cached_sizes) <= config.max_cached_sizes
