"""SpacetimeDB metrics sink for real-time metrics streaming.

Implements CloudMonitoringProtocol from alpha_zero/cloud/protocols.py,
publishing training metrics to SpacetimeDB for real-time monitoring
and subscription-based dashboards.

Supports batching for efficiency and periodic flushing.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any

from alpha_zero.spacetimedb.client import SpacetimeDBClient

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SpacetimeDBMetricsSink:
    """Real-time metrics publisher using SpacetimeDB.

    Implements the CloudMonitoringProtocol interface, allowing it to be
    used as a drop-in replacement for other monitoring backends.

    Supports:
    - Immediate or batched metric publishing
    - Configurable batch size and flush interval
    - Thread-safe operation
    - Subscription-based real-time updates for dashboards

    Usage:
        client = SpacetimeDBClient(config)
        client.connect_with_mock(mock)

        sink = SpacetimeDBMetricsSink(client)
        sink.log_metrics(step=100, metrics={"loss": 0.5})
        sink.flush()
    """

    def __init__(
        self,
        client: SpacetimeDBClient,
        batch_size: int = 10,
        flush_interval: float = 5.0,
    ) -> None:
        """Initialize the metrics sink.

        Args:
            client: Connected SpacetimeDB client.
            batch_size: Number of metrics to batch before auto-flushing.
            flush_interval: Maximum seconds between flushes.
        """
        self._client = client
        self._batch_size = batch_size
        self._flush_interval = flush_interval

        self._buffer: deque[tuple[int, dict[str, float], dict[str, str] | None]] = deque()
        self._event_buffer: deque[tuple[str, str, dict[str, str] | None, dict[str, Any] | None]] = deque()
        self._lock = threading.RLock()
        self._last_flush_time = time.monotonic()
        self._total_metrics_logged: int = 0
        self._total_events_logged: int = 0
        self._flush_count: int = 0

        logger.info(
            "SpacetimeDB metrics sink initialized (batch_size=%d, flush_interval=%.1fs)",
            batch_size, flush_interval,
        )

    # --- CloudMonitoringProtocol interface ---

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, float],
        labels: dict[str, str] | None = None,
    ) -> None:
        """Log training metrics.

        Buffers metrics and auto-flushes when batch_size is reached.

        Args:
            step: Training step number.
            metrics: Dictionary of metric names to values.
            labels: Optional labels for filtering.
        """
        with self._lock:
            self._buffer.append((step, metrics, labels))
            self._total_metrics_logged += 1

            if len(self._buffer) >= self._batch_size:
                self._flush_metrics()

    def log_event(
        self,
        severity: str,
        message: str,
        labels: dict[str, str] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Log an event with severity level.

        Args:
            severity: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            message: Event message.
            labels: Optional labels for filtering.
            payload: Optional structured data.
        """
        with self._lock:
            self._event_buffer.append((severity, message, labels, payload))
            self._total_events_logged += 1

        # Map to Python logging levels
        log_level = getattr(logging, severity.upper(), logging.INFO)
        logger.log(log_level, "SpacetimeDB event: %s", message)

    def flush(self) -> None:
        """Flush any pending metrics and events to SpacetimeDB."""
        with self._lock:
            self._flush_metrics()
            self._flush_events()
            self._last_flush_time = time.monotonic()
            self._flush_count += 1

    # --- Additional methods ---

    @property
    def total_metrics_logged(self) -> int:
        """Total number of metric entries logged."""
        return self._total_metrics_logged

    @property
    def total_events_logged(self) -> int:
        """Total number of events logged."""
        return self._total_events_logged

    @property
    def flush_count(self) -> int:
        """Number of times flush has been called."""
        return self._flush_count

    @property
    def pending_count(self) -> int:
        """Number of pending (unbuffered) metrics."""
        with self._lock:
            return len(self._buffer)

    @property
    def pending_event_count(self) -> int:
        """Number of pending (unbuffered) events."""
        with self._lock:
            return len(self._event_buffer)

    def should_flush(self) -> bool:
        """Check if buffer should be flushed based on time or size.

        Considers both metrics and event buffers to prevent unbounded
        growth of the event buffer when only events are being logged.

        Returns:
            True if a flush is recommended.
        """
        with self._lock:
            total_pending = len(self._buffer) + len(self._event_buffer)
            if total_pending == 0:
                return False
            if len(self._buffer) >= self._batch_size:
                return True
            if len(self._event_buffer) >= self._batch_size:
                return True
            elapsed = time.monotonic() - self._last_flush_time
            return elapsed >= self._flush_interval

    # --- Internal helpers ---

    def _flush_metrics(self) -> None:
        """Flush buffered metrics to SpacetimeDB.

        Must be called with self._lock held.
        """
        while self._buffer:
            step, metrics, labels = self._buffer.popleft()
            try:
                self._client.log_metrics_sync(step, metrics)
            except Exception:
                logger.exception(
                    "Failed to flush metrics for step %d", step,
                )
                # Re-add failed metric to buffer for retry
                self._buffer.appendleft((step, metrics, labels))
                break

    def _flush_events(self) -> None:
        """Flush buffered events to SpacetimeDB.

        Must be called with self._lock held.
        Events are logged locally and can be stored in SpacetimeDB
        for distributed event tracking.
        """
        while self._event_buffer:
            severity, message, _labels, _payload = self._event_buffer.popleft()
            # Events are already logged locally in log_event()
            # Additional SpacetimeDB storage can be added here
            logger.debug(
                "Flushed event: [%s] %s", severity, message,
            )


__all__ = ["SpacetimeDBMetricsSink"]
