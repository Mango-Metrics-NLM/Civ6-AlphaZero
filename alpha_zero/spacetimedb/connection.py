"""Connection lifecycle manager for SpacetimeDB.

Handles connection establishment, health checks, and automatic
reconnection with exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from alpha_zero.spacetimedb.events import CONNECTION_STATE_CHANGED, EventRegistry
from alpha_zero.spacetimedb.exceptions import (
    SpacetimeDBConnectionError,
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
        self._reconnect_task: asyncio.Task[None] | None = None
        self._should_reconnect = True
        self._reconnect_count = 0
        self._last_connected_at: float = 0.0

    @property
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._connected

    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        return self._connected

    async def connect(self) -> bool:
        """Establish connection to SpacetimeDB.

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
                "SpacetimeDB SDK not installed. "
                "Install with: pip install spacetimedb-sdk"
            )

        if not self._config.module_address:
            raise SpacetimeDBConnectionError(
                "module_address is required. "
                "Set CIV6_SPACETIMEDB_MODULE environment variable."
            )

        try:
            from spacetimedb_sdk.spacetimedb_async_client import (  # type: ignore[import-untyped]
                SpacetimeDBAsyncClient,
            )

            self._sdk_client = SpacetimeDBAsyncClient(None)

            logger.info(
                "Connecting to SpacetimeDB at %s (module: %s)",
                self._config.host,
                self._config.module_address,
            )

            # Attempt connection with timeout
            await asyncio.wait_for(
                self._sdk_client.run(
                    self._config.auth_token or "",
                    self._config.host,
                    self._config.module_address,
                    self._on_connected,
                    [],
                ),
                timeout=self._config.connection_timeout,
            )

            return self._connected

        except TimeoutError as e:
            raise SpacetimeDBTimeoutError(
                f"Connection timed out after {self._config.connection_timeout}s"
            ) from e
        except Exception as e:
            raise SpacetimeDBConnectionError(
                f"Failed to connect: {e}"
            ) from e

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
        self._should_reconnect = False

        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        self._connected = False
        self._sdk_client = None
        self._events.fire(CONNECTION_STATE_CHANGED, connected=False)
        logger.info("Disconnected from SpacetimeDB")

    def disconnect_sync(self) -> None:
        """Synchronous disconnect."""
        self._should_reconnect = False
        self._connected = False
        self._sdk_client = None
        self._events.fire(CONNECTION_STATE_CHANGED, connected=False)

    async def _reconnect_loop(self) -> None:
        """Attempt reconnection with exponential backoff."""
        attempt = 0
        while (
            self._should_reconnect
            and attempt < self._config.max_reconnect_attempts
        ):
            delay = min(
                self._config.reconnect_interval * (2 ** attempt),
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
                    self._reconnect_count += 1
                    return
            except SpacetimeDBError:
                logger.warning("Reconnect attempt %d failed", attempt + 1)

            attempt += 1

        logger.error(
            "Failed to reconnect after %d attempts",
            self._config.max_reconnect_attempts,
        )

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


# Avoid circular import at module level
from alpha_zero.spacetimedb.exceptions import SpacetimeDBError  # noqa: E402

__all__ = ["ConnectionManager"]
