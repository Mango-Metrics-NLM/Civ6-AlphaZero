"""Connection lifecycle manager for SpacetimeDB.

Handles connection establishment, health checks, and automatic
reconnection with exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from alpha_zero.spacetimedb.events import (
    CONNECTION_STATE_CHANGED,
    MODEL_VERSION_CHANGED,
    TRAINING_STATE_CHANGED,
    EventRegistry,
)
from alpha_zero.spacetimedb.exceptions import (
    SpacetimeDBConnectionError,
    SpacetimeDBError,
    SpacetimeDBTimeoutError,
    SpacetimeDBUnavailableError,
)

if TYPE_CHECKING:
    from alpha_zero.spacetimedb.config import SpacetimeDBConfig

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages SpacetimeDB WebSocket connection lifecycle.

    Handles:
    - Initial connection with configurable timeout
    - Table subscription callbacks for real-time updates
    - Automatic reconnection with exponential backoff
    - Health monitoring
    - Graceful shutdown

    Each process should create its own ConnectionManager instance
    (not shared across multiprocessing boundaries).
    """

    def __init__(
        self,
        config: SpacetimeDBConfig,
        events: EventRegistry | None = None,
    ) -> None:
        self._config = config
        self._events = events or EventRegistry()
        self._sdk_client: Any = None
        self._connected = False
        self._last_connected_at: float = 0.0

    @property
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._connected

    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return self._connected

    def _build_table_callbacks(self) -> list[tuple[str, Any]]:
        """Build table subscription callbacks based on config.

        Returns:
            List of (table_name, callback) tuples for SDK registration.
        """
        callbacks: list[tuple[str, Any]] = []

        if self._config.subscribe_model_versions:
            callbacks.append(
                ("ModelVersionInfo", self._on_model_version_update),
            )

        if self._config.subscribe_training_state:
            callbacks.append(
                ("TrainingState", self._on_training_state_update),
            )

        return callbacks

    def _on_model_version_update(
        self,
        op: str,
        old: Any,
        new: Any,
        _reducer_event: Any = None,
    ) -> None:
        """Callback when ModelVersionInfo table changes.

        Fires MODEL_VERSION_CHANGED event with the new version number.

        Args:
            op: Operation type ("insert", "update", "delete").
            old: Previous row value (None for inserts).
            new: New row value (None for deletes).
            _reducer_event: Optional reducer event context.
        """
        if op in ("insert", "update") and new is not None:
            # Support both dict-like and attribute access
            if isinstance(new, dict):
                new_version = new.get("current_version", 0)
            else:
                new_version = getattr(new, "current_version", 0)

            if new_version:
                self._events.fire(MODEL_VERSION_CHANGED, new_version)
                logger.debug(
                    "Model version updated to %d via subscription",
                    new_version,
                )

    def _on_training_state_update(
        self,
        op: str,
        old: Any,
        new: Any,
        _reducer_event: Any = None,
    ) -> None:
        """Callback when TrainingState table changes.

        Fires TRAINING_STATE_CHANGED event with the new state.

        Args:
            op: Operation type ("insert", "update", "delete").
            old: Previous row value (None for inserts).
            new: New row value (None for deletes).
            _reducer_event: Optional reducer event context.
        """
        if op in ("insert", "update") and new is not None:
            if isinstance(new, dict):
                state = new
            else:
                state = {
                    "is_training": getattr(new, "is_training", False),
                    "should_stop": getattr(new, "should_stop", False),
                    "current_iteration": getattr(new, "current_iteration", 0),
                }
            self._events.fire(TRAINING_STATE_CHANGED, state)
            logger.debug("Training state updated via subscription")

    async def connect(self) -> bool:
        """Establish connection to SpacetimeDB.

        Registers table subscription callbacks based on config settings
        so that push-based updates (model versions, training state) fire
        events through the EventRegistry.

        Returns:
            True if connection successful.

        Raises:
            SpacetimeDBUnavailableError: If SDK is not installed.
            SpacetimeDBConnectionError: If connection fails after retries.
        """
        try:
            from alpha_zero.spacetimedb import HAS_SPACETIMEDB

            has_sdk = HAS_SPACETIMEDB
        except ImportError:
            has_sdk = False

        if not has_sdk:
            raise SpacetimeDBUnavailableError(
                "SpacetimeDB SDK not installed. Install with: pip install spacetimedb-sdk"
            )

        if not self._config.module_address:
            raise SpacetimeDBConnectionError(
                "module_address is required. Set CIV6_SPACETIMEDB_MODULE environment variable."
            )

        try:
            from spacetimedb_sdk.spacetimedb_async_client import (
                SpacetimeDBAsyncClient,
            )

            self._sdk_client = SpacetimeDBAsyncClient(None)

            logger.info(
                "Connecting to SpacetimeDB at %s (module: %s)",
                self._config.host,
                self._config.module_address,
            )

            # Build table callbacks from config
            table_callbacks = self._build_table_callbacks()
            logger.debug(
                "Registering %d table callbacks: %s",
                len(table_callbacks),
                [name for name, _ in table_callbacks],
            )

            # Attempt connection with timeout
            await asyncio.wait_for(
                self._sdk_client.run(
                    self._config.auth_token or "",
                    self._config.host,
                    self._config.module_address,
                    self._on_connected,
                    table_callbacks,
                ),
                timeout=self._config.connection_timeout,
            )

            return self._connected

        except TimeoutError as e:
            raise SpacetimeDBTimeoutError(
                f"Connection timed out after {self._config.connection_timeout}s"
            ) from e
        except Exception as e:
            raise SpacetimeDBConnectionError(f"Failed to connect: {e}") from e

    def connect_with_mock(self, mock_client: Any) -> bool:
        """Connect using a mock client (for testing).

        Args:
            mock_client: Mock client instance to use.

        Returns:
            True if setup successful.
        """
        self._sdk_client = mock_client
        self._connected = True
        self._last_connected_at = time.monotonic()
        self._events.fire(CONNECTION_STATE_CHANGED, connected=True)
        return True

    async def disconnect(self) -> None:
        """Gracefully disconnect from SpacetimeDB."""
        self._connected = False
        self._sdk_client = None
        self._events.fire(CONNECTION_STATE_CHANGED, connected=False)
        logger.info("Disconnected from SpacetimeDB")

    def disconnect_sync(self) -> None:
        """Synchronous disconnect."""
        self._connected = False
        self._sdk_client = None
        self._events.fire(CONNECTION_STATE_CHANGED, connected=False)

    async def reconnect(self) -> bool:
        """Attempt reconnection with exponential backoff.

        Returns:
            True if reconnection succeeded.
        """
        attempt = 0
        while attempt < self._config.max_reconnect_attempts:
            delay = min(
                self._config.reconnect_interval * (2**attempt),
                60.0,  # Cap at 60 seconds
            )
            logger.info(
                "Reconnect attempt %d/%d in %.1fs",
                attempt + 1,
                self._config.max_reconnect_attempts,
                delay,
            )
            await asyncio.sleep(delay)

            try:
                await self.connect()
                if self._connected:
                    logger.info("Reconnected to SpacetimeDB")
                    return True
            except SpacetimeDBError:
                logger.warning("Reconnect attempt %d failed", attempt + 1)

            attempt += 1

        logger.error(
            "Failed to reconnect after %d attempts",
            self._config.max_reconnect_attempts,
        )
        return False

    def _on_connected(self) -> None:
        """Callback when SDK connection is established."""
        self._connected = True
        self._last_connected_at = time.monotonic()
        self._events.fire(CONNECTION_STATE_CHANGED, connected=True)
        logger.info("Connected to SpacetimeDB")

    @property
    def sdk_client(self) -> Any:
        """Get the underlying SDK client (or mock)."""
        return self._sdk_client


__all__ = ["ConnectionManager"]
