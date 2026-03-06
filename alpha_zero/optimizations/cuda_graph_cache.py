"""CUDA graph caching for neural network inference.

This module implements CUDA graph caching to eliminate kernel launch overhead
for fixed batch sizes during MCTS inference. CUDA graphs capture the GPU
execution graph once and replay it for subsequent predictions with the same
batch size.

Example:
    >>> from alpha_zero.optimizations.cuda_graph_cache import CUDAGraphCache, CUDAGraphConfig
    >>> config = CUDAGraphConfig(enabled=True, warmup_steps=3)
    >>> cache = CUDAGraphCache(network, device="cuda:0", config=config)
    >>> policy, value = cache.predict_with_cache(spatial, vector, action_mask)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from alpha_zero.utils.wandb_logger import log_metrics

logger = logging.getLogger(__name__)


@dataclass
class CUDAGraphConfig:
    """Configuration for CUDA graph caching.

    Attributes:
        enabled: Whether to use CUDA graphs (default: True).
        warmup_steps: Number of warmup iterations before capturing graph (default: 3).
        max_cached_sizes: Maximum number of batch sizes to cache (default: 8).
        allow_cpu_fallback: Fall back to direct predict if CUDA unavailable (default: True).
    """

    enabled: bool = True
    warmup_steps: int = 3
    max_cached_sizes: int = 8
    allow_cpu_fallback: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.warmup_steps < 1:
            raise ValueError(f"warmup_steps must be >= 1, got {self.warmup_steps}")
        if self.max_cached_sizes < 1:
            raise ValueError(f"max_cached_sizes must be >= 1, got {self.max_cached_sizes}")


class _CachedGraph(NamedTuple):
    """Internal structure for a cached CUDA graph.

    Attributes:
        graph: The captured CUDA graph.
        static_spatial: Pre-allocated spatial input tensor.
        static_vector: Pre-allocated vector input tensor.
        static_mask: Pre-allocated action mask input tensor.
        static_policy: Pre-allocated policy output tensor.
        static_value: Pre-allocated value output tensor.
    """

    graph: torch.cuda.CUDAGraph
    static_spatial: torch.Tensor
    static_vector: torch.Tensor
    static_mask: torch.Tensor
    static_policy: torch.Tensor
    static_value: torch.Tensor


class CUDAGraphCache:
    """CUDA graph cache for neural network inference.

    This class wraps a neural network and caches CUDA graphs for different batch
    sizes to eliminate kernel launch overhead. For each unique batch size, it
    captures the GPU execution graph once and replays it for subsequent predictions.

    Attributes:
        network: The neural network following MCTSNetwork protocol.
        device: The device string (e.g., "cuda:0", "cpu").
        config: CUDA graph configuration.
    """

    def __init__(
        self,
        network: Any,
        device: str,
        config: CUDAGraphConfig | None = None,
    ) -> None:
        """Initialize CUDA graph cache.

        Args:
            network: Neural network with predict(spatial, vector, mask) method.
            device: Device string (e.g., "cuda:0", "cpu").
            config: CUDA graph configuration (default: CUDAGraphConfig()).
        """
        self.network = network
        self.device = device
        self.config = config if config is not None else CUDAGraphConfig()

        # Cache maps batch_size -> _CachedGraph
        self._graphs: dict[int, _CachedGraph] = {}

        # Check if CUDA is available
        self._use_cuda = torch.cuda.is_available() and "cuda" in device

        if not self._use_cuda:
            logger.info(
                "CUDA graphs disabled: CUDA not available or device is CPU (%s)",
                device,
            )
        elif not self.config.enabled:
            logger.info("CUDA graphs disabled via config")

    def predict_with_cache(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict with CUDA graph caching.

        Args:
            spatial: Spatial features tensor (B, C, H, W).
            vector: Vector features tensor (B, D).
            action_mask: Action mask tensor (B, A).

        Returns:
            Tuple of (policy_probs, value):
                - policy_probs: Policy probabilities (B, A).
                - value: Value estimates (B, 1).
        """
        batch_size = spatial.shape[0]

        # Fast path: no CUDA or disabled
        if not self._use_cuda or not self.config.enabled:
            result = self.network.predict(spatial, vector, action_mask)
            return (result[0], result[1])

        # Fast path: cached graph exists
        if batch_size in self._graphs:
            cached = self._graphs[batch_size]
            start_time = time.perf_counter()
            policy, value = self._replay_graph(cached, spatial, vector, action_mask)
            replay_time_ms = (time.perf_counter() - start_time) * 1000

            log_metrics(
                {
                    "cuda_graph/cache_hit": 1,
                    "cuda_graph/cached_sizes": len(self._graphs),
                    "cuda_graph/replay_time_ms": replay_time_ms,
                }
            )

            return policy, value

        # Slow path: cache miss
        log_metrics(
            {
                "cuda_graph/cache_hit": 0,
                "cuda_graph/cached_sizes": len(self._graphs),
            }
        )

        # Try to capture new graph if we have space
        if len(self._graphs) < self.config.max_cached_sizes:
            start_time = time.perf_counter()
            captured = self._capture_graph(batch_size, spatial, vector, action_mask)
            capture_time_ms = (time.perf_counter() - start_time) * 1000

            if captured is not None:
                cached = captured
                self._graphs[batch_size] = cached
                logger.info(
                    "Captured CUDA graph for batch_size=%d (%.2f ms)",
                    batch_size,
                    capture_time_ms,
                )
                log_metrics(
                    {
                        "cuda_graph/capture_time_ms": capture_time_ms,
                    }
                )

                # Use the newly captured graph
                return self._replay_graph(cached, spatial, vector, action_mask)
        else:
            logger.debug(
                "CUDA graph cache full (%d/%d), falling back to direct predict",
                len(self._graphs),
                self.config.max_cached_sizes,
            )

        # Fallback to direct predict
        result = self.network.predict(spatial, vector, action_mask)
        return (result[0], result[1])

    def _capture_graph(
        self,
        batch_size: int,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> _CachedGraph | None:
        """Capture a CUDA graph for the given batch size.

        Args:
            batch_size: The batch size to capture.
            spatial: Example spatial tensor (B, C, H, W).
            vector: Example vector tensor (B, D).
            action_mask: Example action mask tensor (B, A).

        Returns:
            Cached graph or None if capture failed.
        """
        try:
            # Create static input tensors with the same shape and device
            static_spatial = torch.zeros_like(spatial)
            static_vector = torch.zeros_like(vector)
            static_mask = torch.zeros_like(action_mask)

            # Warmup: run several iterations to stabilize CUDA kernels
            with torch.no_grad():
                for _ in range(self.config.warmup_steps):
                    _ = self.network.predict(static_spatial, static_vector, static_mask)

            # Capture the graph
            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph):
                static_policy, static_value = self.network.predict(
                    static_spatial, static_vector, static_mask
                )

            return _CachedGraph(
                graph=graph,
                static_spatial=static_spatial,
                static_vector=static_vector,
                static_mask=static_mask,
                static_policy=static_policy,
                static_value=static_value,
            )

        except Exception as e:
            logger.warning(
                "Failed to capture CUDA graph for batch_size=%d: %s",
                batch_size,
                e,
            )
            if not self.config.allow_cpu_fallback:
                raise
            return None

    def _replay_graph(
        self,
        cached: _CachedGraph,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay a cached CUDA graph with new inputs.

        Args:
            cached: The cached graph to replay.
            spatial: New spatial input tensor (B, C, H, W).
            vector: New vector input tensor (B, D).
            action_mask: New action mask input tensor (B, A).

        Returns:
            Tuple of (policy_probs, value).
        """
        # Copy new inputs into static tensors
        cached.static_spatial.copy_(spatial)
        cached.static_vector.copy_(vector)
        cached.static_mask.copy_(action_mask)

        # Replay the graph
        cached.graph.replay()

        # Return clones of static outputs
        return cached.static_policy.clone(), cached.static_value.clone()

    def clear_cache(self) -> None:
        """Clear all cached CUDA graphs."""
        num_cleared = len(self._graphs)
        self._graphs.clear()
        logger.info("Cleared %d cached CUDA graphs", num_cleared)

    @property
    def cached_sizes(self) -> list[int]:
        """Get list of cached batch sizes.

        Returns:
            List of batch sizes with cached graphs.
        """
        return sorted(self._graphs.keys())
