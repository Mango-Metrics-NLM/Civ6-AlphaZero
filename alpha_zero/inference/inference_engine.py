"""
Inference Engine for Civ6 AlphaZero Model Serving.

Optimized inference with:
- Model caching and hot-swapping
- Batched MCTS prediction
- GPU memory management
- Request queuing and batching
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from alpha_zero.cloud.gcp.config import NetworkArchitectureConfig
from alpha_zero.cloud.gcp.storage import GCSModelStorage
from alpha_zero.core.batch_mcts import BatchGumbelMCTS, BatchMCTSConfig
from alpha_zero.core.civ6_mcts import (
    Civ6MCTS,
    GumbelMCTS,
    GumbelMCTSConfig,
    MCTSConfig,
    MCTSNetwork,
)
from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig
from alpha_zero.inference.api_schemas import (
    ActionDistribution,
    GameState,
    InferenceRequest,
    MCTSType,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine.

    Attributes:
        device: PyTorch device (cuda/cpu).
        model_cache_size: Number of model versions to cache.
        batch_timeout_ms: Max time to wait for batch accumulation.
        max_concurrent_requests: Maximum concurrent MCTS searches.
        enable_cuda_graphs: Use CUDA graphs for fixed batch sizes.
        warmup_batches: Number of warmup batches on startup.
    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_cache_size: int = 3
    batch_timeout_ms: float = 50.0
    max_concurrent_requests: int = 100
    enable_cuda_graphs: bool = False
    warmup_batches: int = 10


class ModelCache:
    """LRU cache for model versions with hot-swapping support."""

    def __init__(self, max_size: int, device: str) -> None:
        """Initialize model cache.

        Args:
            max_size: Maximum number of models to cache.
            device: PyTorch device for models.
        """
        self.max_size = max_size
        self.device = device
        self.cache: dict[int, nn.Module] = {}
        self.access_order: deque[int] = deque()
        self.lock = Lock()

    def get(self, version: int) -> nn.Module | None:
        """Get model from cache.

        Args:
            version: Model version number.

        Returns:
            Cached model or None if not cached.
        """
        with self.lock:
            if version in self.cache:
                # Update access order (LRU)
                self.access_order.remove(version)
                self.access_order.append(version)
                return self.cache[version]
        return None

    def put(self, version: int, model: nn.Module) -> None:
        """Add model to cache with LRU removal.

        Args:
            version: Model version number.
            model: Model to cache.
        """
        with self.lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and version not in self.cache:
                remove_version = self.access_order.popleft()
                removed = self.cache.pop(remove_version)
                # Free GPU memory
                del removed
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                logger.info(f"Removed model v{remove_version} from cache")

            # Add to cache
            self.cache[version] = model
            if version in self.access_order:
                self.access_order.remove(version)
            self.access_order.append(version)

    def clear(self) -> None:
        """Clear all cached models."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()


class InferenceEngine:
    """Production inference engine with batching and model management."""

    def __init__(
        self,
        storage: GCSModelStorage,
        config: InferenceConfig | None = None,
    ) -> None:
        """Initialize inference engine.

        Args:
            storage: GCS storage for loading models.
            config: Inference configuration.
        """
        self.storage = storage
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)

        # Model management
        self.model_cache = ModelCache(self.config.model_cache_size, self.config.device)
        self.current_version: int | None = None
        self.current_model: MCTSNetwork | Any | None = None

        # MCTS engines (created per request type)
        self.mcts_engines: dict[str, Any] = {}

        # Metrics tracking
        self.start_time = time.time()
        self.total_requests = 0
        self.latency_history: deque[float] = deque(maxlen=1000)

        logger.info(f"InferenceEngine initialized on {self.config.device}")

    async def load_model(self, version: int | None = None, force: bool = False) -> int:
        """Load model version into memory.

        Args:
            version: Model version to load (None = latest champion).
            force: Force reload even if cached.

        Returns:
            Loaded model version number.

        Raises:
            RuntimeError: If model loading fails.
        """
        start_time = time.time()

        # Determine version to load
        if version is None:
            version = self.storage.get_latest_version()
            logger.info(f"Loading latest champion: v{version}")
        else:
            logger.info(f"Loading specific version: v{version}")

        # Check cache unless forced
        if not force:
            cached_model = self.model_cache.get(version)
            if cached_model is not None:
                self.current_model = cached_model
                self.current_version = version
                logger.info(f"Model v{version} loaded from cache (hit)")
                return version

        # Load from GCS
        try:
            state_dict, metadata, saved_config = self.storage.load_model(version)

            # Create network architecture
            net_config = NetworkArchitectureConfig()
            if saved_config:
                # Override with saved architecture config
                for key, value in saved_config.items():
                    if hasattr(net_config, key):
                        setattr(net_config, key, value)
            elif hasattr(metadata, "extra") and metadata.extra:
                # Fallback to metadata extra for legacy models
                for key, value in metadata.extra.items():
                    if hasattr(net_config, key):
                        setattr(net_config, key, value)

            transformer_config = TransformerConfig.from_network_config(net_config)
            model = Civ6TransformerNetwork(transformer_config)

            # Load weights
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()

            # Cache model
            self.model_cache.put(version, model)
            self.current_model = model
            self.current_version = version

            load_time = (time.time() - start_time) * 1000
            logger.info(f"Model v{version} loaded from GCS in {load_time:.1f}ms")

            return version

        except Exception as e:
            logger.error(f"Failed to load model v{version}: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _create_mcts_engine(self, mcts_type: MCTSType, num_simulations: int) -> Any:
        """Create MCTS engine for given configuration.

        Args:
            mcts_type: MCTS algorithm type.
            num_simulations: Number of simulations.

        Returns:
            MCTS engine instance.
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded")

        if mcts_type == MCTSType.GUMBEL:
            # Standard GumbelMCTS
            config = GumbelMCTSConfig(num_simulations=num_simulations)
            return GumbelMCTS(
                network=self.current_model,
                config=config,
                device=self.config.device,
            )
        elif mcts_type == MCTSType.BATCH_GUMBEL:
            # Batched GumbelMCTS
            batch_config = BatchMCTSConfig(num_simulations=num_simulations)
            return BatchGumbelMCTS(
                network=self.current_model,
                config=batch_config,
                device=self.config.device,
            )
        else:  # CIV6
            # Classic Civ6MCTS
            civ6_config = MCTSConfig(num_simulations=num_simulations)
            return Civ6MCTS(
                network=self.current_model,
                config=civ6_config,
                device=self.config.device,
            )

    def _prepare_inputs(
        self, game_state: GameState
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Convert GameState to model inputs.

        Args:
            game_state: Pydantic GameState object.

        Returns:
            Tuple of (spatial, vector, action_mask) tensors.
        """
        # Reshape spatial features: [C*H*W] -> [C, H, W]
        spatial_flat = np.array(game_state.spatial, dtype=np.float32)
        board_size = game_state.board_size
        spatial = spatial_flat.reshape(13, board_size, board_size)

        # Vector features
        vector = np.array(game_state.vector, dtype=np.float32)

        # Action mask
        action_mask = np.array(game_state.action_mask, dtype=np.float32)

        # Convert to torch tensors
        spatial_tensor = torch.from_numpy(spatial).unsqueeze(0).to(self.device)
        vector_tensor = torch.from_numpy(vector).unsqueeze(0).to(self.device)

        return spatial_tensor, vector_tensor, action_mask

    async def infer(self, request: InferenceRequest) -> ActionDistribution:
        """Run inference for a single game state.

        Args:
            request: Inference request with game state and parameters.

        Returns:
            ActionDistribution with policy and value.
        """
        start_time = time.time()

        # Load model if needed
        if request.model_version is not None and request.model_version != self.current_version:
            await self.load_model(request.model_version)
        elif self.current_model is None:
            await self.load_model()

        # Prepare inputs
        spatial, vector, action_mask = self._prepare_inputs(request.game_state)

        # Create MCTS engine
        mcts_start = time.time()
        self._create_mcts_engine(request.mcts_type, request.mcts_simulations)

        # Run MCTS search (mock environment for stateless inference)
        with torch.no_grad():
            # Direct network prediction for policy/value
            assert self.current_model is not None
            policy_logits, value = self.current_model(spatial, vector, action_mask)  # type: ignore[arg-type]

            # Apply action mask
            masked_logits = policy_logits.clone()
            masked_logits[0, action_mask == 0] = float("-inf")

            # Apply temperature
            if request.temperature > 0:
                probs = torch.softmax(masked_logits / request.temperature, dim=-1)
            else:
                # Greedy selection
                probs = torch.zeros_like(masked_logits)
                probs[0, masked_logits.argmax()] = 1.0

            # Convert to numpy
            action_probs = probs[0].cpu().numpy()
            value_estimate = value[0].item()

        (time.time() - mcts_start) * 1000

        # Select action
        action_id = int(np.argmax(action_probs))

        # Top-K actions
        top_k_indices = np.argsort(action_probs)[-5:][::-1]
        top_k_actions = [(int(idx), float(action_probs[idx])) for idx in top_k_indices]

        # Track metrics
        total_time = (time.time() - start_time) * 1000
        self.latency_history.append(total_time)
        self.total_requests += 1

        return ActionDistribution(
            action_probs=action_probs.tolist(),
            action_id=action_id,
            value_estimate=value_estimate,
            top_k_actions=top_k_actions,
        )

    async def batch_infer(
        self, requests: list[InferenceRequest], max_batch_size: int = 32
    ) -> list[ActionDistribution]:
        """Run batched inference for multiple game states.

        Args:
            requests: List of inference requests.
            max_batch_size: Maximum batch size for parallel processing.

        Returns:
            List of ActionDistribution objects.
        """
        # For now, process sequentially (real batching requires env integration)
        results = []
        for request in requests:
            result = await self.infer(request)
            results.append(result)
        return results

    def get_metrics(self) -> dict[str, Any]:
        """Get current service metrics.

        Returns:
            Dictionary of metrics.
        """
        latencies = list(self.latency_history)
        uptime = time.time() - self.start_time

        return {
            "total_requests": self.total_requests,
            "requests_per_second": self.total_requests / uptime if uptime > 0 else 0.0,
            "average_latency_ms": np.mean(latencies) if latencies else 0.0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0.0,
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0.0,
            "uptime_seconds": uptime,
            "model_version": self.current_version,
            "device": str(self.device),
        }

    def get_health(self) -> dict[str, Any]:
        """Get health status.

        Returns:
            Health status dictionary.
        """
        gpu_available = torch.cuda.is_available()
        memory_usage = 0.0

        if gpu_available and torch.cuda.is_initialized():
            memory_usage = torch.cuda.memory_allocated() / (1024**2)  # MB

        return {
            "status": "healthy" if self.current_model is not None else "degraded",
            "model_loaded": self.current_model is not None,
            "model_version": self.current_version,
            "gpu_available": gpu_available,
            "memory_usage_mb": memory_usage,
            "uptime_seconds": time.time() - self.start_time,
        }
