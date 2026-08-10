"""Mock implementations for SpacetimeDB testing.

Provides in-memory mock clients that simulate SpacetimeDB behavior
without requiring a running server or the SDK installed.

Follows the MockDistributedBackend pattern from tests/fixtures/mock_distributed.py.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class MockSpacetimeDBClient:
    """In-memory mock of SpacetimeDBClient for unit testing.

    Stores all data in Python dicts. Fires callbacks synchronously
    when data changes. Tracks all reducer calls for assertions.

    Usage:
        mock = MockSpacetimeDBClient()
        mock.connect_sync()

        # Publish weights
        mock.publish_weights_sync(iteration=1, version=1, weights_bytes=b"...", metadata={})

        # Verify
        assert mock.call_counts["publish_weights"] == 1
        assert mock.current_version == 1
    """

    def __init__(
        self,
        config: Any = None,
        *,
        simulate_failures: bool = False,
    ) -> None:
        self.config = config
        self._simulate_failures = simulate_failures

        # Data stores
        self._models: dict[int, tuple[bytes, dict[str, Any]]] = {}
        self._champion_version: int = 0
        self._current_version: int = 0
        self._trajectories: dict[str, tuple[bytes, dict[str, Any]]] = {}
        self._metrics: list[dict[str, Any]] = []
        self._evaluations: list[dict[str, Any]] = []
        self._checkpoints: dict[int, tuple[bytes, dict[str, Any]]] = {}
        self._configs: dict[int, str] = {}
        self._workers: dict[str, dict[str, Any]] = {}
        self._training_state: dict[str, Any] = {
            "is_training": False,
            "should_stop": False,
            "current_iteration": 0,
            "total_games_played": 0,
            "total_examples": 0,
        }

        # Callback registry
        self._callbacks: dict[str, list[Callable[..., Any]]] = defaultdict(list)

        # Call tracking
        self.call_counts: dict[str, int] = defaultdict(int)
        self.call_args: dict[str, list[tuple[Any, ...]]] = defaultdict(list)

        # Connection state
        self._connected: bool = False
        self._lock = threading.RLock()

    # --- Connection lifecycle ---

    @property
    def is_connected(self) -> bool:
        """Check if mock is connected."""
        return self._connected

    @property
    def current_version(self) -> int:
        """Get current model version."""
        return self._current_version

    @property
    def champion_version(self) -> int:
        """Get champion model version."""
        return self._champion_version

    async def connect(self) -> bool:
        """Simulate connecting to SpacetimeDB."""
        self._track("connect")
        if self._simulate_failures:
            return False
        self._connected = True
        return True

    def connect_sync(self) -> bool:
        """Synchronous connect."""
        self._track("connect")
        if self._simulate_failures:
            return False
        self._connected = True
        return True

    async def disconnect(self) -> None:
        """Simulate disconnecting."""
        self._track("disconnect")
        self._connected = False

    def disconnect_sync(self) -> None:
        """Synchronous disconnect."""
        self._track("disconnect")
        self._connected = False

    # --- Weight operations ---

    async def publish_weights(
        self,
        iteration: int,
        version: int,
        weights_bytes: bytes,
        metadata: dict[str, Any],
        is_champion: bool = False,
    ) -> bool:
        """Mock publish weights."""
        return self.publish_weights_sync(
            iteration=iteration,
            version=version,
            weights_bytes=weights_bytes,
            metadata=metadata,
            is_champion=is_champion,
        )

    def publish_weights_sync(
        self,
        iteration: int,
        version: int,
        weights_bytes: bytes,
        metadata: dict[str, Any],
        is_champion: bool = False,
    ) -> bool:
        """Synchronous publish weights."""
        self._track("publish_weights", iteration, version, len(weights_bytes), is_champion)
        if self._simulate_failures:
            return False

        with self._lock:
            self._models[version] = (weights_bytes, {**metadata, "iteration": iteration})
            self._current_version = max(self._current_version, version)
            if is_champion:
                self._champion_version = version

        self._fire_callbacks("on_model_version_changed", version)
        return True

    async def get_latest_weights(self) -> tuple[bytes, dict[str, Any]] | None:
        """Get latest model weights."""
        return self.get_latest_weights_sync()

    def get_latest_weights_sync(self) -> tuple[bytes, dict[str, Any]] | None:
        """Synchronous get latest weights."""
        self._track("get_latest_weights")
        with self._lock:
            if not self._models:
                return None
            return self._models.get(self._current_version)

    async def get_champion_weights(self) -> tuple[bytes, dict[str, Any]] | None:
        """Get champion model weights."""
        return self.get_champion_weights_sync()

    def get_champion_weights_sync(self) -> tuple[bytes, dict[str, Any]] | None:
        """Synchronous get champion weights."""
        self._track("get_champion_weights")
        with self._lock:
            if self._champion_version == 0:
                return None
            return self._models.get(self._champion_version)

    async def get_weights_by_version(
        self,
        version: int,
    ) -> tuple[bytes, dict[str, Any]] | None:
        """Get weights by version."""
        return self.get_weights_by_version_sync(version)

    def get_weights_by_version_sync(
        self,
        version: int,
    ) -> tuple[bytes, dict[str, Any]] | None:
        """Synchronous get weights by version."""
        self._track("get_weights_by_version", version)
        with self._lock:
            return self._models.get(version)

    async def get_model_by_iteration(
        self,
        iteration: int,
    ) -> tuple[bytes, dict[str, Any]] | None:
        """Get model weights by training iteration."""
        return self.get_model_by_iteration_sync(iteration)

    def get_model_by_iteration_sync(
        self,
        iteration: int,
    ) -> tuple[bytes, dict[str, Any]] | None:
        """Synchronous get model by iteration."""
        self._track("get_model_by_iteration", iteration)
        with self._lock:
            for _version, (weights, meta) in self._models.items():
                if meta.get("iteration") == iteration:
                    return weights, meta
            return None

    async def get_server_version(self) -> int:
        """Get the latest model version (server-side query)."""
        return self.get_server_version_sync()

    def get_server_version_sync(self) -> int:
        """Synchronous get server version."""
        self._track("get_server_version")
        with self._lock:
            return self._current_version

    # --- Trajectory operations ---

    async def submit_trajectory(
        self,
        iteration: int,
        worker_id: int,
        game_id: str,
        trajectory_bytes: bytes,
        num_examples: int,
    ) -> bool:
        """Mock submit trajectory."""
        return self.submit_trajectory_sync(
            iteration, worker_id, game_id, trajectory_bytes, num_examples
        )

    def submit_trajectory_sync(
        self,
        iteration: int,
        worker_id: int,
        game_id: str,
        trajectory_bytes: bytes,
        num_examples: int,
    ) -> bool:
        """Synchronous submit trajectory."""
        self._track("submit_trajectory", iteration, worker_id, game_id, num_examples)
        if self._simulate_failures:
            return False

        key = f"{iteration}_{worker_id}_{game_id}"
        with self._lock:
            self._trajectories[key] = (
                trajectory_bytes,
                {
                    "iteration": iteration,
                    "worker_id": worker_id,
                    "game_id": game_id,
                    "num_examples": num_examples,
                },
            )
            self._training_state["total_games_played"] += 1
            self._training_state["total_examples"] += num_examples

        self._fire_callbacks("on_trajectory_added", key)
        return True

    def get_trajectory(
        self,
        iteration: int,
        worker_id: int,
        game_id: str,
    ) -> tuple[bytes, dict[str, Any]] | None:
        """Get trajectory by key."""
        key = f"{iteration}_{worker_id}_{game_id}"
        with self._lock:
            return self._trajectories.get(key)

    def list_trajectories(
        self,
        iteration: int | None = None,
        worker_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """List trajectory metadata."""
        with self._lock:
            results = []
            for _, meta in self._trajectories.values():
                if iteration is not None and meta["iteration"] != iteration:
                    continue
                if worker_id is not None and meta["worker_id"] != worker_id:
                    continue
                results.append(meta)
            return results

    def count_trajectories(self, iteration: int | None = None) -> int:
        """Count trajectories."""
        return len(self.list_trajectories(iteration=iteration))

    # --- Metrics operations ---

    async def log_metrics(self, step: int, metrics: dict[str, float]) -> bool:
        """Mock log metrics."""
        return self.log_metrics_sync(step, metrics)

    def log_metrics_sync(self, step: int, metrics: dict[str, float]) -> bool:
        """Synchronous log metrics."""
        self._track("log_metrics", step, metrics)
        if self._simulate_failures:
            return False

        with self._lock:
            self._metrics.append({"step": step, **metrics})

        self._fire_callbacks("on_metrics_updated", step, metrics)
        return True

    # --- Evaluation operations ---

    def save_evaluation(self, result: dict[str, Any]) -> bool:
        """Store evaluation result."""
        self._track("save_evaluation", result)
        if self._simulate_failures:
            return False
        with self._lock:
            self._evaluations.append(result)
        return True

    def list_evaluations(self) -> list[dict[str, Any]]:
        """List all evaluations."""
        with self._lock:
            return list(self._evaluations)

    # --- Checkpoint operations ---

    def save_checkpoint(
        self,
        iteration: int,
        data: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store checkpoint."""
        self._track("save_checkpoint", iteration)
        if self._simulate_failures:
            return False
        with self._lock:
            self._checkpoints[iteration] = (data, metadata or {})
        return True

    def load_checkpoint(self, iteration: int) -> tuple[bytes, dict[str, Any]] | None:
        """Load checkpoint."""
        with self._lock:
            return self._checkpoints.get(iteration)

    def load_latest_checkpoint(self) -> tuple[int, bytes, dict[str, Any]] | None:
        """Load latest checkpoint."""
        with self._lock:
            if not self._checkpoints:
                return None
            latest_iter = max(self._checkpoints.keys())
            data, meta = self._checkpoints[latest_iter]
            return latest_iter, data, meta

    # --- Config operations ---

    def save_config(self, iteration: int, config_json: str) -> bool:
        """Store config."""
        self._track("save_config", iteration)
        with self._lock:
            self._configs[iteration] = config_json
        return True

    def load_config(self, iteration: int) -> str | None:
        """Load config."""
        with self._lock:
            return self._configs.get(iteration)

    # --- Worker operations ---

    async def register_worker(self, worker_type: str, worker_id: str) -> bool:
        """Mock register worker."""
        return self.register_worker_sync(worker_type, worker_id)

    def register_worker_sync(self, worker_type: str, worker_id: str) -> bool:
        """Synchronous register worker."""
        self._track("register_worker", worker_type, worker_id)
        with self._lock:
            self._workers[worker_id] = {
                "worker_type": worker_type,
                "status": "running",
                "games_played": 0,
            }
        return True

    async def heartbeat(self, worker_id: str, games_completed: int) -> bool:
        """Mock heartbeat."""
        return self.heartbeat_sync(worker_id, games_completed)

    def heartbeat_sync(self, worker_id: str, games_completed: int) -> bool:
        """Synchronous heartbeat."""
        self._track("heartbeat", worker_id, games_completed)
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id]["games_played"] = games_completed
        return True

    # --- Training state ---

    def get_training_state(self) -> dict[str, Any]:
        """Get current training state."""
        with self._lock:
            return dict(self._training_state)

    async def request_stop(self) -> bool:
        """Request training stop."""
        return self.request_stop_sync()

    def request_stop_sync(self) -> bool:
        """Synchronous request stop."""
        self._track("request_stop")
        with self._lock:
            self._training_state["should_stop"] = True
        self._fire_callbacks("on_training_state_changed", self._training_state)
        return True

    # --- Callback registration ---

    def on_model_version_changed(self, callback: Callable[..., Any]) -> None:
        """Register callback for model version changes."""
        self._callbacks["on_model_version_changed"].append(callback)

    def on_trajectory_added(self, callback: Callable[..., Any]) -> None:
        """Register callback for new trajectories."""
        self._callbacks["on_trajectory_added"].append(callback)

    def on_metrics_updated(self, callback: Callable[..., Any]) -> None:
        """Register callback for metrics updates."""
        self._callbacks["on_metrics_updated"].append(callback)

    def on_training_state_changed(self, callback: Callable[..., Any]) -> None:
        """Register callback for training state changes."""
        self._callbacks["on_training_state_changed"].append(callback)

    # --- Internal helpers ---

    def _track(self, method: str, *args: Any) -> None:
        """Track method calls for assertions."""
        self.call_counts[method] += 1
        self.call_args[method].append(args)

    def _fire_callbacks(self, event: str, *args: Any) -> None:
        """Fire registered callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args)
            except Exception:
                logger.exception("Callback error for event %s", event)

    def reset(self) -> None:
        """Reset all state. Useful between tests."""
        with self._lock:
            self._models.clear()
            self._trajectories.clear()
            self._metrics.clear()
            self._evaluations.clear()
            self._checkpoints.clear()
            self._configs.clear()
            self._workers.clear()
            self._champion_version = 0
            self._current_version = 0
            self._training_state = {
                "is_training": False,
                "should_stop": False,
                "current_iteration": 0,
                "total_games_played": 0,
                "total_examples": 0,
            }
            self.call_counts.clear()
            self.call_args.clear()


__all__ = ["MockSpacetimeDBClient"]
