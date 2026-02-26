"""SpacetimeDB client wrapper.

Provides a resilient async/sync client for SpacetimeDB operations,
wrapping the underlying SDK with retry logic, error handling,
and typed method signatures.

Follows the async-first + sync wrapper pattern from llm_client.py.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from alpha_zero.spacetimedb.connection import ConnectionManager
from alpha_zero.spacetimedb.events import (
    CONNECTION_STATE_CHANGED,
    METRICS_UPDATED,
    MODEL_VERSION_CHANGED,
    TRAINING_STATE_CHANGED,
    TRAJECTORY_ADDED,
    WORKER_STATUS_CHANGED,
    EventRegistry,
)
from alpha_zero.spacetimedb.exceptions import (
    SpacetimeDBConnectionError,
    SpacetimeDBReducerError,
    SpacetimeDBSerializationError,
)
from alpha_zero.spacetimedb.serialization import estimate_blob_size_mb

if TYPE_CHECKING:
    from alpha_zero.spacetimedb.config import SpacetimeDBConfig

logger = logging.getLogger(__name__)


class SpacetimeDBClient:
    """Resilient client for SpacetimeDB operations.

    Wraps the SpacetimeDB Python SDK with:
    - Connection lifecycle management (connect, disconnect, reconnect)
    - Typed reducer calls for all operations
    - Subscription management for real-time updates
    - Async-first design with synchronous wrappers
    - Graceful error handling and logging

    Each process should create its own client instance
    (not shared across multiprocessing boundaries).

    Usage:
        config = SpacetimeDBConfig.for_local("my-module")
        client = SpacetimeDBClient(config)
        await client.connect()

        # Publish weights
        await client.publish_weights(
            iteration=1, version=1,
            weights_bytes=b"...", metadata={"loss": 0.5}
        )

        # Subscribe to version changes
        await client.subscribe_model_versions(my_callback)

        await client.disconnect()
    """

    def __init__(self, config: SpacetimeDBConfig) -> None:
        self._config = config
        self._events = EventRegistry()
        self._connection = ConnectionManager(config, self._events)
        self._current_version: int = 0
        self._champion_version: int = 0
        self._worker_id: str | None = None

    # --- Properties ---

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to SpacetimeDB."""
        return self._connection.is_connected

    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return self._connection.is_healthy

    @property
    def current_version(self) -> int:
        """Get the current model version."""
        return self._current_version

    @property
    def champion_version(self) -> int:
        """Get the champion model version."""
        return self._champion_version

    @property
    def events(self) -> EventRegistry:
        """Access the event registry for custom handlers."""
        return self._events

    @property
    def config(self) -> SpacetimeDBConfig:
        """Access client configuration."""
        return self._config

    # --- Connection lifecycle ---

    async def connect(self) -> bool:
        """Establish connection to SpacetimeDB.

        Returns:
            True if connection successful.

        Raises:
            SpacetimeDBUnavailableError: If SDK not installed.
            SpacetimeDBConnectionError: If connection fails.
        """
        logger.info(
            "SpacetimeDB client connecting to %s (module: %s)",
            self._config.host,
            self._config.module_address,
        )
        result = await self._connection.connect()
        if result:
            logger.info("SpacetimeDB client connected successfully")
        return result

    def connect_with_mock(self, mock_client: Any) -> bool:
        """Connect using a mock client (for testing).

        Args:
            mock_client: Mock client instance to use.

        Returns:
            True if setup successful.
        """
        result = self._connection.connect_with_mock(mock_client)
        if result:
            logger.debug("SpacetimeDB client connected with mock")
        return result

    async def disconnect(self) -> None:
        """Gracefully disconnect from SpacetimeDB."""
        logger.info("SpacetimeDB client disconnecting")
        await self._connection.disconnect()
        self._events.clear()

    def disconnect_sync(self) -> None:
        """Synchronous disconnect."""
        self._connection.disconnect_sync()
        self._events.clear()

    # --- Weight operations ---

    async def publish_weights(
        self,
        iteration: int,
        version: int,
        weights_bytes: bytes,
        metadata: dict[str, Any],
        is_champion: bool = False,
    ) -> bool:
        """Publish model weights to SpacetimeDB.

        Args:
            iteration: Current training iteration.
            version: Model version number.
            weights_bytes: Serialized model weights.
            metadata: Additional metadata (loss, metrics, etc.).
            is_champion: Whether this is the new champion model.

        Returns:
            True if publish was successful.

        Raises:
            SpacetimeDBConnectionError: If not connected.
            SpacetimeDBReducerError: If reducer call fails.
            SpacetimeDBSerializationError: If weight blob exceeds size limit.
        """
        self._check_connected()

        blob_size_mb = estimate_blob_size_mb(weights_bytes)
        if blob_size_mb > self._config.max_weight_blob_mb:
            raise SpacetimeDBSerializationError(
                f"Weight blob size ({blob_size_mb:.1f} MB) exceeds "
                f"max_weight_blob_mb ({self._config.max_weight_blob_mb} MB)"
            )

        logger.info(
            "Publishing weights v%d (iteration=%d, size=%.1f MB, champion=%s)",
            version, iteration, blob_size_mb, is_champion,
        )

        try:
            sdk = self._connection.sdk_client
            if hasattr(sdk, "publish_weights"):
                result = await self._call_async_or_sync(
                    sdk.publish_weights,
                    iteration=iteration,
                    version=version,
                    weights_bytes=weights_bytes,
                    metadata=metadata,
                    is_champion=is_champion,
                )
            else:
                result = self._call_reducer(
                    "publish_model_weights",
                    version,
                    iteration,
                    weights_bytes,
                    is_champion,
                    self._serialize_metadata(metadata),
                )

            if result is not False:
                self._current_version = max(self._current_version, version)
                if is_champion:
                    self._champion_version = version
                self._events.fire(MODEL_VERSION_CHANGED, version)
                logger.info("Published weights v%d successfully", version)
                return True
            return False

        except SpacetimeDBReducerError:
            raise
        except Exception as e:
            raise SpacetimeDBReducerError(
                "publish_model_weights",
                str(e),
            ) from e

    def publish_weights_sync(
        self,
        iteration: int,
        version: int,
        weights_bytes: bytes,
        metadata: dict[str, Any],
        is_champion: bool = False,
    ) -> bool:
        """Synchronous version of publish_weights."""
        return self._run_sync(
            self.publish_weights(
                iteration=iteration,
                version=version,
                weights_bytes=weights_bytes,
                metadata=metadata,
                is_champion=is_champion,
            )
        )

    async def get_latest_weights(self) -> tuple[bytes, dict[str, Any]] | None:
        """Get the latest model weights.

        Returns:
            Tuple of (weights_bytes, metadata) or None if no weights available.

        Raises:
            SpacetimeDBConnectionError: If not connected.
        """
        self._check_connected()
        sdk = self._connection.sdk_client
        if hasattr(sdk, "get_latest_weights"):
            return await self._call_async_or_sync(sdk.get_latest_weights)
        return None

    def get_latest_weights_sync(self) -> tuple[bytes, dict[str, Any]] | None:
        """Synchronous version of get_latest_weights."""
        return self._run_sync(self.get_latest_weights())

    async def get_champion_weights(self) -> tuple[bytes, dict[str, Any]] | None:
        """Get the champion model weights.

        Returns:
            Tuple of (weights_bytes, metadata) or None if no champion set.

        Raises:
            SpacetimeDBConnectionError: If not connected.
        """
        self._check_connected()
        sdk = self._connection.sdk_client
        if hasattr(sdk, "get_champion_weights"):
            return await self._call_async_or_sync(sdk.get_champion_weights)
        return None

    def get_champion_weights_sync(self) -> tuple[bytes, dict[str, Any]] | None:
        """Synchronous version of get_champion_weights."""
        return self._run_sync(self.get_champion_weights())

    async def get_weights_by_version(
        self,
        version: int,
    ) -> tuple[bytes, dict[str, Any]] | None:
        """Get weights for a specific version.

        Args:
            version: Model version to retrieve.

        Returns:
            Tuple of (weights_bytes, metadata) or None if version not found.

        Raises:
            SpacetimeDBConnectionError: If not connected.
        """
        self._check_connected()
        sdk = self._connection.sdk_client
        if hasattr(sdk, "get_weights_by_version"):
            return await self._call_async_or_sync(sdk.get_weights_by_version, version)
        return None

    def get_weights_by_version_sync(
        self,
        version: int,
    ) -> tuple[bytes, dict[str, Any]] | None:
        """Synchronous version of get_weights_by_version."""
        return self._run_sync(self.get_weights_by_version(version))

    # --- Trajectory operations ---

    async def submit_trajectory(
        self,
        iteration: int,
        worker_id: int,
        game_id: str,
        trajectory_bytes: bytes,
        num_examples: int,
    ) -> bool:
        """Submit a game trajectory.

        Args:
            iteration: Training iteration.
            worker_id: ID of the worker that generated this trajectory.
            game_id: Unique game identifier.
            trajectory_bytes: Serialized trajectory data.
            num_examples: Number of training examples in the trajectory.

        Returns:
            True if submission was successful.

        Raises:
            SpacetimeDBConnectionError: If not connected.
            SpacetimeDBReducerError: If reducer call fails.
        """
        self._check_connected()

        trajectory_id = f"{iteration}_{worker_id}_{game_id}"
        logger.debug(
            "Submitting trajectory %s (%d examples, %.1f KB)",
            trajectory_id, num_examples,
            len(trajectory_bytes) / 1024,
        )

        try:
            sdk = self._connection.sdk_client
            if hasattr(sdk, "submit_trajectory"):
                result = await self._call_async_or_sync(
                    sdk.submit_trajectory,
                    iteration=iteration,
                    worker_id=worker_id,
                    game_id=game_id,
                    trajectory_bytes=trajectory_bytes,
                    num_examples=num_examples,
                )
            else:
                result = self._call_reducer(
                    "submit_trajectory",
                    trajectory_id,
                    iteration,
                    worker_id,
                    game_id,
                    trajectory_bytes,
                    num_examples,
                )

            if result is not False:
                self._events.fire(TRAJECTORY_ADDED, trajectory_id)
                return True
            return False

        except SpacetimeDBReducerError:
            raise
        except Exception as e:
            raise SpacetimeDBReducerError(
                "submit_trajectory",
                str(e),
            ) from e

    def submit_trajectory_sync(
        self,
        iteration: int,
        worker_id: int,
        game_id: str,
        trajectory_bytes: bytes,
        num_examples: int,
    ) -> bool:
        """Synchronous version of submit_trajectory."""
        return self._run_sync(
            self.submit_trajectory(
                iteration=iteration,
                worker_id=worker_id,
                game_id=game_id,
                trajectory_bytes=trajectory_bytes,
                num_examples=num_examples,
            )
        )

    # --- Metrics operations ---

    async def log_metrics(
        self,
        step: int,
        metrics: dict[str, float],
    ) -> bool:
        """Log training metrics.

        Args:
            step: Training step number.
            metrics: Dictionary of metric name -> value.

        Returns:
            True if logging was successful.

        Raises:
            SpacetimeDBConnectionError: If not connected.
            SpacetimeDBReducerError: If reducer call fails.
        """
        self._check_connected()
        logger.debug("Logging metrics at step %d: %s", step, metrics)

        try:
            sdk = self._connection.sdk_client
            if hasattr(sdk, "log_metrics"):
                result = await self._call_async_or_sync(
                    sdk.log_metrics, step, metrics,
                )
            else:
                result = self._call_reducer(
                    "record_training_metrics",
                    step,
                    metrics.get("total_loss", 0.0),
                    metrics.get("policy_loss", 0.0),
                    metrics.get("value_loss", 0.0),
                    metrics.get("learning_rate", 0.0),
                )

            if result is not False:
                self._events.fire(METRICS_UPDATED, step, metrics)
                return True
            return False

        except SpacetimeDBReducerError:
            raise
        except Exception as e:
            raise SpacetimeDBReducerError(
                "record_training_metrics",
                str(e),
            ) from e

    def log_metrics_sync(
        self,
        step: int,
        metrics: dict[str, float],
    ) -> bool:
        """Synchronous version of log_metrics."""
        return self._run_sync(self.log_metrics(step, metrics))

    # --- Evaluation operations ---

    async def save_evaluation(
        self,
        result: dict[str, Any],
    ) -> bool:
        """Save an evaluation result.

        Args:
            result: Evaluation result dictionary.

        Returns:
            True if save was successful.

        Raises:
            SpacetimeDBConnectionError: If not connected.
            SpacetimeDBReducerError: If reducer call fails.
        """
        self._check_connected()

        try:
            sdk = self._connection.sdk_client
            if hasattr(sdk, "save_evaluation"):
                return await self._call_async_or_sync(sdk.save_evaluation, result)

            evaluation_id = result.get(
                "evaluation_id",
                f"eval_{result.get('iteration', 0)}_{uuid.uuid4().hex[:8]}",
            )
            return self._call_reducer(
                "record_evaluation",
                evaluation_id,
                result.get("iteration", 0),
                result.get("win_rate", 0.0),
                result.get("games_played", 0),
                result.get("promoted", False),
            )
        except SpacetimeDBReducerError:
            raise
        except Exception as e:
            raise SpacetimeDBReducerError(
                "record_evaluation",
                str(e),
            ) from e

    def save_evaluation_sync(self, result: dict[str, Any]) -> bool:
        """Synchronous version of save_evaluation."""
        return self._run_sync(self.save_evaluation(result))

    # --- Checkpoint operations ---

    async def save_checkpoint(
        self,
        iteration: int,
        data: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Save a training checkpoint.

        Args:
            iteration: Training iteration.
            data: Serialized checkpoint data.
            metadata: Optional metadata.

        Returns:
            True if save was successful.

        Raises:
            SpacetimeDBConnectionError: If not connected.
            SpacetimeDBReducerError: If reducer call fails.
        """
        self._check_connected()
        logger.info(
            "Saving checkpoint for iteration %d (%.1f MB)",
            iteration, estimate_blob_size_mb(data),
        )

        try:
            sdk = self._connection.sdk_client
            if hasattr(sdk, "save_checkpoint"):
                return await self._call_async_or_sync(
                    sdk.save_checkpoint, iteration, data, metadata,
                )
            return self._call_reducer(
                "save_checkpoint",
                iteration,
                data,
                self._current_version,
            )
        except SpacetimeDBReducerError:
            raise
        except Exception as e:
            raise SpacetimeDBReducerError(
                "save_checkpoint",
                str(e),
            ) from e

    def save_checkpoint_sync(
        self,
        iteration: int,
        data: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Synchronous version of save_checkpoint."""
        return self._run_sync(self.save_checkpoint(iteration, data, metadata))

    # --- Config operations ---

    async def save_config(
        self,
        iteration: int,
        config_json: str,
    ) -> bool:
        """Save training configuration.

        Args:
            iteration: Training iteration.
            config_json: Serialized configuration string.

        Returns:
            True if save was successful.
        """
        self._check_connected()

        try:
            sdk = self._connection.sdk_client
            if hasattr(sdk, "save_config"):
                return await self._call_async_or_sync(
                    sdk.save_config, iteration, config_json,
                )
            return self._call_reducer(
                "save_config",
                iteration,
                config_json,
            )
        except SpacetimeDBReducerError:
            raise
        except Exception as e:
            raise SpacetimeDBReducerError(
                "save_config",
                str(e),
            ) from e

    def save_config_sync(self, iteration: int, config_json: str) -> bool:
        """Synchronous version of save_config."""
        return self._run_sync(self.save_config(iteration, config_json))

    # --- Worker operations ---

    async def register_worker(
        self,
        worker_type: str,
        worker_id: str,
    ) -> bool:
        """Register a worker with SpacetimeDB.

        Args:
            worker_type: Type of worker (e.g., "selfplay", "trainer", "evaluator").
            worker_id: Unique worker identifier.

        Returns:
            True if registration was successful.

        Raises:
            SpacetimeDBConnectionError: If not connected.
            SpacetimeDBReducerError: If reducer call fails.
        """
        self._check_connected()
        self._worker_id = worker_id
        logger.info("Registering worker %s (type=%s)", worker_id, worker_type)

        try:
            sdk = self._connection.sdk_client
            if hasattr(sdk, "register_worker"):
                result = await self._call_async_or_sync(
                    sdk.register_worker, worker_type, worker_id,
                )
            else:
                result = self._call_reducer(
                    "worker_heartbeat",
                    worker_id,
                    worker_type,
                    "running",
                    0,
                )

            if result is not False:
                self._events.fire(WORKER_STATUS_CHANGED, worker_id, "running")
                return True
            return False

        except SpacetimeDBReducerError:
            raise
        except Exception as e:
            raise SpacetimeDBReducerError(
                "register_worker",
                str(e),
            ) from e

    def register_worker_sync(self, worker_type: str, worker_id: str) -> bool:
        """Synchronous version of register_worker."""
        return self._run_sync(self.register_worker(worker_type, worker_id))

    async def heartbeat(
        self,
        worker_id: str,
        games_completed: int,
    ) -> bool:
        """Send a worker heartbeat.

        Args:
            worker_id: Worker identifier.
            games_completed: Total games completed by this worker.

        Returns:
            True if heartbeat was successful.
        """
        self._check_connected()

        try:
            sdk = self._connection.sdk_client
            if hasattr(sdk, "heartbeat"):
                result = await self._call_async_or_sync(
                    sdk.heartbeat, worker_id, games_completed,
                )
            else:
                result = self._call_reducer(
                    "worker_heartbeat",
                    worker_id,
                    "selfplay",
                    "running",
                    games_completed,
                )
            return result is not False

        except Exception as e:
            logger.warning("Heartbeat failed for %s: %s", worker_id, e)
            return False

    def heartbeat_sync(self, worker_id: str, games_completed: int) -> bool:
        """Synchronous version of heartbeat."""
        return self._run_sync(self.heartbeat(worker_id, games_completed))

    # --- Training state ---

    async def request_stop(self) -> bool:
        """Request training to stop.

        Returns:
            True if stop request was sent successfully.
        """
        self._check_connected()
        logger.info("Requesting training stop")

        try:
            sdk = self._connection.sdk_client
            if hasattr(sdk, "request_stop"):
                result = await self._call_async_or_sync(sdk.request_stop)
            else:
                result = self._call_reducer("request_stop")

            if result is not False:
                self._events.fire(TRAINING_STATE_CHANGED, {"should_stop": True})
                return True
            return False

        except Exception as e:
            logger.error("Failed to request stop: %s", e)
            return False

    def request_stop_sync(self) -> bool:
        """Synchronous version of request_stop."""
        return self._run_sync(self.request_stop())

    # --- Subscription management ---

    def subscribe_model_versions(
        self,
        callback: Callable[..., Any],
    ) -> None:
        """Subscribe to model version changes.

        Args:
            callback: Function called with (version: int) on version change.
        """
        self._events.on(MODEL_VERSION_CHANGED, callback)
        logger.debug("Registered model version subscription")

    def subscribe_trajectories(
        self,
        callback: Callable[..., Any],
    ) -> None:
        """Subscribe to new trajectory additions.

        Args:
            callback: Function called with (trajectory_id: str) on new trajectory.
        """
        self._events.on(TRAJECTORY_ADDED, callback)
        logger.debug("Registered trajectory subscription")

    def subscribe_metrics(
        self,
        callback: Callable[..., Any],
    ) -> None:
        """Subscribe to metrics updates.

        Args:
            callback: Function called with (step: int, metrics: dict) on update.
        """
        self._events.on(METRICS_UPDATED, callback)
        logger.debug("Registered metrics subscription")

    def subscribe_training_state(
        self,
        callback: Callable[..., Any],
    ) -> None:
        """Subscribe to training state changes.

        Args:
            callback: Function called with (state: dict) on state change.
        """
        self._events.on(TRAINING_STATE_CHANGED, callback)
        logger.debug("Registered training state subscription")

    def subscribe_worker_status(
        self,
        callback: Callable[..., Any],
    ) -> None:
        """Subscribe to worker status changes.

        Args:
            callback: Function called with (worker_id: str, status: str).
        """
        self._events.on(WORKER_STATUS_CHANGED, callback)
        logger.debug("Registered worker status subscription")

    def subscribe_connection_state(
        self,
        callback: Callable[..., Any],
    ) -> None:
        """Subscribe to connection state changes.

        Args:
            callback: Function called with (connected: bool).
        """
        self._events.on(CONNECTION_STATE_CHANGED, callback)
        logger.debug("Registered connection state subscription")

    # --- Internal helpers ---

    def _check_connected(self) -> None:
        """Verify client is connected.

        Raises:
            SpacetimeDBConnectionError: If not connected.
        """
        if not self.is_connected:
            raise SpacetimeDBConnectionError(
                "Not connected to SpacetimeDB. Call connect() first."
            )

    def _call_reducer(self, reducer_name: str, *args: Any) -> bool:
        """Call a SpacetimeDB reducer through the SDK.

        Args:
            reducer_name: Name of the reducer to call.
            *args: Arguments to pass to the reducer.

        Returns:
            True if the call succeeded.

        Raises:
            SpacetimeDBReducerError: If the call fails.
        """
        try:
            sdk = self._connection.sdk_client
            if sdk is None:
                raise SpacetimeDBConnectionError("SDK client not initialized")

            if hasattr(sdk, "call_reducer"):
                sdk.call_reducer(reducer_name, *args)
                return True

            logger.debug(
                "SDK does not support call_reducer, skipping %s",
                reducer_name,
            )
            return True

        except (SpacetimeDBConnectionError, SpacetimeDBReducerError):
            raise
        except Exception as e:
            raise SpacetimeDBReducerError(reducer_name, str(e)) from e

    async def _call_async_or_sync(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call a function that may be async or sync.

        Args:
            fn: Function to call.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            Result of the function call.
        """
        result = fn(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _run_sync(self, coro: Any) -> Any:
        """Run an async coroutine synchronously.

        Uses asyncio.run() for sync wrapper pattern (following llm_client.py).

        Args:
            coro: Coroutine to run.

        Returns:
            Result of the coroutine.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=self._config.connection_timeout)
        else:
            return asyncio.run(coro)

    @staticmethod
    def _serialize_metadata(metadata: dict[str, Any]) -> str:
        """Serialize metadata dict to JSON string.

        Args:
            metadata: Metadata dictionary.

        Returns:
            JSON string.
        """
        import json

        return json.dumps(metadata, default=str)


__all__ = ["SpacetimeDBClient"]
