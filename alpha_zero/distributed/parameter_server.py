"""Parameter server for distributed AlphaZero training.

Provides centralized weight management for multi-node training:
- Weight broadcasting to all nodes
- Model version tracking
- Checkpoint coordination
- Gradient aggregation (optional)
"""

from __future__ import annotations

import io
import logging
import lzma
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class WeightUpdateMode(str, Enum):
    """Mode for weight updates from learners."""

    REPLACE = "replace"  # Learner sends full weights
    GRADIENT = "gradient"  # Learner sends gradients (for async SGD)
    DELTA = "delta"  # Learner sends weight deltas


@dataclass
class ParameterServerConfig:
    """Configuration for parameter server."""

    # Update settings
    update_mode: str = WeightUpdateMode.REPLACE.value

    # Compression
    compress_weights: bool = True
    compression_level: int = 3  # 1-9, higher = more compression, slower

    # Version tracking
    max_version_history: int = 5  # Keep N previous versions

    # Async gradient aggregation
    async_gradient_steps: int = 1  # Steps to average for async SGD
    gradient_staleness_threshold: int = 10  # Max version difference

    # Performance
    lock_timeout_seconds: float = 30.0

    # Callbacks
    on_weight_update: Callable[[int], None] | None = None


class ParameterServer:
    """Central parameter server for distributed training.

    Manages model weights and coordinates updates between:
    - Learner nodes (push weight updates)
    - Actor nodes (pull latest weights)

    Supports:
    - Thread-safe weight access
    - Version tracking
    - Weight compression
    - Optional gradient aggregation
    """

    def __init__(
        self,
        network: nn.Module | None = None,
        config: ParameterServerConfig | None = None,
    ) -> None:
        """Initialize parameter server.

        Args:
            network: Optional network for weight initialization.
            config: Server configuration.
        """
        self.config = config or ParameterServerConfig()

        # Weight storage
        self._weights: dict[str, torch.Tensor] | None = None
        self._serialized_weights: bytes | None = None
        self._version = 0

        # Version history for rollback
        self._weight_history: list[tuple[int, bytes]] = []

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "total_updates": 0,
            "total_fetches": 0,
            "compression_ratio": 0.0,
            "last_update_time": 0.0,
        }

        # Initialize from network if provided
        if network is not None:
            self.set_weights(network.state_dict())

        logger.info("ParameterServer initialized")

    def set_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        increment_version: bool = True,
    ) -> int:
        """Store new model weights.

        Args:
            state_dict: Model state dictionary.
            increment_version: Whether to increment version.

        Returns:
            New version number.
        """
        with self._lock:
            # Store raw weights
            self._weights = {k: v.clone() for k, v in state_dict.items()}

            # Serialize and optionally compress
            buffer = io.BytesIO()
            torch.save(state_dict, buffer)
            raw_bytes = buffer.getvalue()

            if self.config.compress_weights:
                self._serialized_weights = lzma.compress(
                    raw_bytes,
                    preset=self.config.compression_level,
                )
                self._stats["compression_ratio"] = len(self._serialized_weights) / len(raw_bytes)
            else:
                self._serialized_weights = raw_bytes
                self._stats["compression_ratio"] = 1.0

            if increment_version:
                self._version += 1

                # Store in history
                self._weight_history.append((self._version, self._serialized_weights))

                # Prune old versions
                if len(self._weight_history) > self.config.max_version_history:
                    self._weight_history = self._weight_history[-self.config.max_version_history :]

            self._stats["total_updates"] += 1
            self._stats["last_update_time"] = time.time()

            # Callback
            if self.config.on_weight_update is not None:
                try:
                    self.config.on_weight_update(self._version)
                except Exception as e:
                    logger.warning(f"Weight update callback failed: {e}")

            logger.debug(
                f"Stored weights v{self._version} "
                f"({len(self._serialized_weights)} bytes, "
                f"ratio={self._stats['compression_ratio']:.2f})"
            )

            return self._version

    def get_weights(self) -> dict[str, torch.Tensor] | None:
        """Get current model weights.

        Returns:
            Model state dictionary, or None if not set.
        """
        weights, _ = self.get_weights_with_version()
        return weights

    def get_weights_with_version(
        self,
    ) -> tuple[dict[str, torch.Tensor] | None, int]:
        """Get weights and version atomically.

        Returns:
            Tuple of (state_dict, version).
        """
        with self._lock:
            if self._weights is None:
                return None, self._version

            # Return cloned weights
            weights = {k: v.clone() for k, v in self._weights.items()}
            version = self._version

        self._stats["total_fetches"] += 1
        return weights, version

    def get_serialized_weights(self) -> tuple[bytes | None, int]:
        """Get serialized weights for network transfer.

        Returns:
            Tuple of (serialized bytes, version).
        """
        with self._lock:
            return self._serialized_weights, self._version

    def deserialize_weights(
        self,
        data: bytes,
        compressed: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        """Deserialize weights from bytes.

        Args:
            data: Serialized weight bytes.
            compressed: Whether data is compressed (None = auto-detect).

        Returns:
            State dictionary.
        """
        if compressed is None:
            compressed = self.config.compress_weights

        if compressed:
            data = lzma.decompress(data)

        buffer = io.BytesIO(data)
        return torch.load(buffer, map_location="cpu", weights_only=True)

    def get_version(self) -> int:
        """Get current version number."""
        with self._lock:
            return self._version

    def get_weights_at_version(
        self,
        version: int,
    ) -> dict[str, torch.Tensor] | None:
        """Get weights at a specific version.

        Args:
            version: Version to retrieve.

        Returns:
            State dictionary, or None if version not found.
        """
        with self._lock:
            for v, data in self._weight_history:
                if v == version:
                    return self.deserialize_weights(data)
            return None

    def update_model(
        self,
        model: nn.Module,
        min_version: int = 0,
    ) -> tuple[bool, int]:
        """Update a model with latest weights.

        Args:
            model: Model to update.
            min_version: Only update if version > min_version.

        Returns:
            Tuple of (was_updated, current_version).
        """
        with self._lock:
            if self._version <= min_version:
                return False, self._version

            if self._weights is None:
                return False, self._version

            try:
                model.load_state_dict(self._weights)
                return True, self._version
            except Exception as e:
                logger.error(f"Failed to update model: {e}")
                return False, self._version

    def apply_gradients(
        self,
        gradients: dict[str, torch.Tensor],
        learning_rate: float,
        source_version: int | None = None,
    ) -> int:
        """Apply gradients to weights (for async SGD).

        Args:
            gradients: Gradient dictionary.
            learning_rate: Learning rate for update.
            source_version: Version gradients were computed on.

        Returns:
            New version number.
        """
        with self._lock:
            if self._weights is None:
                raise RuntimeError("Cannot apply gradients: no weights set")

            # Check staleness
            if source_version is not None:
                staleness = self._version - source_version
                if staleness > self.config.gradient_staleness_threshold:
                    logger.warning(
                        f"Gradient staleness {staleness} exceeds threshold "
                        f"{self.config.gradient_staleness_threshold}"
                    )

            # Apply gradients with SGD update
            for name, grad in gradients.items():
                if name in self._weights:
                    self._weights[name] -= learning_rate * grad

            # Re-serialize
            buffer = io.BytesIO()
            torch.save(self._weights, buffer)
            raw_bytes = buffer.getvalue()

            if self.config.compress_weights:
                self._serialized_weights = lzma.compress(
                    raw_bytes,
                    preset=self.config.compression_level,
                )
            else:
                self._serialized_weights = raw_bytes

            self._version += 1
            self._stats["total_updates"] += 1

            logger.debug(f"Applied gradients, new version: {self._version}")
            return self._version

    def apply_delta(
        self,
        delta: dict[str, torch.Tensor],
        source_version: int,
    ) -> int | None:
        """Apply weight delta from a learner.

        Args:
            delta: Weight differences (new - old).
            source_version: Version delta was computed against.

        Returns:
            New version number, or None if version mismatch.
        """
        with self._lock:
            if self._weights is None:
                raise RuntimeError("Cannot apply delta: no weights set")

            # Only apply if source version matches current
            if source_version != self._version:
                logger.warning(
                    f"Version mismatch: delta from v{source_version}, current is v{self._version}"
                )
                return None

            # Apply delta
            for name, d in delta.items():
                if name in self._weights:
                    self._weights[name] += d

            # Re-serialize
            buffer = io.BytesIO()
            torch.save(self._weights, buffer)
            raw_bytes = buffer.getvalue()

            if self.config.compress_weights:
                self._serialized_weights = lzma.compress(
                    raw_bytes,
                    preset=self.config.compression_level,
                )
            else:
                self._serialized_weights = raw_bytes

            self._version += 1
            self._stats["total_updates"] += 1

            return self._version

    def get_statistics(self) -> dict[str, Any]:
        """Get server statistics.

        Returns:
            Dictionary with statistics.
        """
        with self._lock:
            return {
                "version": self._version,
                "total_updates": self._stats["total_updates"],
                "total_fetches": self._stats["total_fetches"],
                "compression_ratio": self._stats["compression_ratio"],
                "last_update_time": self._stats["last_update_time"],
                "history_size": len(self._weight_history),
                "weights_size_bytes": (
                    len(self._serialized_weights) if self._serialized_weights else 0
                ),
            }

    def reset(self) -> None:
        """Reset server state."""
        with self._lock:
            self._weights = None
            self._serialized_weights = None
            self._version = 0
            self._weight_history = []
            self._stats = {
                "total_updates": 0,
                "total_fetches": 0,
                "compression_ratio": 0.0,
                "last_update_time": 0.0,
            }
        logger.info("ParameterServer reset")


class MockParameterServer(ParameterServer):
    """Mock parameter server for testing.

    Provides additional test utilities:
    - Simulated network latency
    - Failure injection
    - Update tracking
    """

    def __init__(
        self,
        network: nn.Module | None = None,
        config: ParameterServerConfig | None = None,
    ) -> None:
        """Initialize mock parameter server."""
        super().__init__(network, config)

        # Test utilities
        self._simulate_latency: float = 0.0
        self._fail_next_update: bool = False
        self._fail_next_fetch: bool = False
        self._update_history: list[dict[str, Any]] = []

    def simulate_latency(self, seconds: float) -> None:
        """Set simulated network latency.

        Args:
            seconds: Latency in seconds.
        """
        self._simulate_latency = seconds

    def fail_next_update(self) -> None:
        """Make the next update operation fail."""
        self._fail_next_update = True

    def fail_next_fetch(self) -> None:
        """Make the next fetch operation fail."""
        self._fail_next_fetch = True

    def set_weights(
        self,
        state_dict: dict[str, torch.Tensor],
        increment_version: bool = True,
    ) -> int:
        """Store weights with optional simulated failure."""
        if self._simulate_latency > 0:
            time.sleep(self._simulate_latency)

        if self._fail_next_update:
            self._fail_next_update = False
            raise RuntimeError("Simulated update failure")

        version = super().set_weights(state_dict, increment_version)

        # Track update
        self._update_history.append(
            {
                "version": version,
                "time": time.time(),
                "param_count": len(state_dict),
            }
        )

        return version

    def get_weights(self) -> dict[str, torch.Tensor] | None:
        """Get weights with optional simulated failure."""
        if self._simulate_latency > 0:
            time.sleep(self._simulate_latency)

        if self._fail_next_fetch:
            self._fail_next_fetch = False
            raise RuntimeError("Simulated fetch failure")

        return super().get_weights()

    def get_update_history(self) -> list[dict[str, Any]]:
        """Get history of weight updates.

        Returns:
            List of update records.
        """
        return list(self._update_history)

    def clear_history(self) -> None:
        """Clear update history."""
        self._update_history = []


class DistributedParameterServer(ParameterServer):
    """Parameter server with network transport for multi-node deployments.

    Extends base ParameterServer with:
    - gRPC-style interface for remote access
    - Push/pull weight synchronization
    - Heartbeat monitoring

    Note: This is a base class. Actual network transport would be implemented
    via gRPC or similar in production.
    """

    def __init__(
        self,
        network: nn.Module | None = None,
        config: ParameterServerConfig | None = None,
        node_id: str = "ps-0",
    ) -> None:
        """Initialize distributed parameter server.

        Args:
            network: Optional network for weight initialization.
            config: Server configuration.
            node_id: Unique identifier for this server.
        """
        super().__init__(network, config)

        self.node_id = node_id

        # Connected clients
        self._clients: dict[str, dict[str, Any]] = {}
        self._client_lock = threading.Lock()

        # Last heartbeat times
        self._heartbeats: dict[str, float] = {}

        logger.info(f"DistributedParameterServer '{node_id}' initialized")

    def register_client(
        self,
        client_id: str,
        client_type: str = "learner",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Register a client node.

        Args:
            client_id: Unique client identifier.
            client_type: Type of client (learner, actor, etc).
            metadata: Optional client metadata.

        Returns:
            True if registration successful.
        """
        with self._client_lock:
            self._clients[client_id] = {
                "type": client_type,
                "registered_at": time.time(),
                "metadata": metadata or {},
            }
            self._heartbeats[client_id] = time.time()

        logger.info(f"Registered {client_type} client: {client_id}")
        return True

    def unregister_client(self, client_id: str) -> bool:
        """Unregister a client node.

        Args:
            client_id: Client to unregister.

        Returns:
            True if client was registered.
        """
        with self._client_lock:
            if client_id in self._clients:
                del self._clients[client_id]
                self._heartbeats.pop(client_id, None)
                logger.info(f"Unregistered client: {client_id}")
                return True
            return False

    def heartbeat(self, client_id: str) -> bool:
        """Record heartbeat from client.

        Args:
            client_id: Client sending heartbeat.

        Returns:
            True if client is registered.
        """
        with self._client_lock:
            if client_id in self._clients:
                self._heartbeats[client_id] = time.time()
                return True
            return False

    def get_stale_clients(self, timeout_seconds: float = 60.0) -> list[str]:
        """Get clients that haven't sent heartbeat recently.

        Args:
            timeout_seconds: Threshold for considering client stale.

        Returns:
            List of stale client IDs.
        """
        now = time.time()
        stale = []

        with self._client_lock:
            for client_id, last_hb in self._heartbeats.items():
                if now - last_hb > timeout_seconds:
                    stale.append(client_id)

        return stale

    def get_client_count(self, client_type: str | None = None) -> int:
        """Get count of connected clients.

        Args:
            client_type: Optional filter by type.

        Returns:
            Number of clients.
        """
        with self._client_lock:
            if client_type is None:
                return len(self._clients)
            return sum(1 for c in self._clients.values() if c["type"] == client_type)

    def broadcast_weights(self) -> dict[str, bool]:
        """Broadcast weights to all registered clients.

        Returns:
            Dictionary mapping client_id to success status.

        Note: In production, this would use actual network transport.
        """
        results = {}

        with self._client_lock:
            client_ids = list(self._clients.keys())

        for client_id in client_ids:
            # Placeholder for actual network broadcast
            results[client_id] = True
            logger.debug(f"Would broadcast to {client_id}")

        return results

    def get_cluster_info(self) -> dict[str, Any]:
        """Get information about the parameter server cluster.

        Returns:
            Dictionary with cluster information.
        """
        with self._client_lock:
            return {
                "node_id": self.node_id,
                "version": self._version,
                "clients": {
                    "total": len(self._clients),
                    "learners": self.get_client_count("learner"),
                    "actors": self.get_client_count("actor"),
                },
                "stale_clients": len(self.get_stale_clients()),
            }


def create_parameter_server(
    network: nn.Module | None = None,
    config: ParameterServerConfig | None = None,
    distributed: bool = False,
    node_id: str = "ps-0",
    force_mock: bool = False,
) -> ParameterServer:
    """Factory function to create a parameter server.

    Args:
        network: Optional network for weight initialization.
        config: Server configuration.
        distributed: Whether to use distributed server.
        node_id: Node ID for distributed server.
        force_mock: Force mock implementation.

    Returns:
        ParameterServer instance.
    """
    if force_mock:
        return MockParameterServer(network, config)
    elif distributed:
        return DistributedParameterServer(network, config, node_id)
    else:
        return ParameterServer(network, config)
