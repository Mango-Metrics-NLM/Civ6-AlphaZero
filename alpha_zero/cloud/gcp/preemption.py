"""GCE preemption detection and handling.

Provides PreemptionHandler for detecting and handling preemption
events on GCE spot/preemptible instances. Uses the GCE metadata
service to detect upcoming termination.

GCE provides a 30-second warning before termination:
https://cloud.google.com/compute/docs/instances/spot#handle-preemption
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from types import FrameType
from typing import Any

from alpha_zero.cloud.exceptions import PreemptionError
from alpha_zero.cloud.protocols import PreemptionHandlerProtocol

logger = logging.getLogger(__name__)

# GCE metadata service URL for preemption check
METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
METADATA_HEADERS = {"Metadata-Flavor": "Google"}
# Default preemption warning time (GCE provides 30 seconds)
# Can be overridden via environment variable or constructor parameter
DEFAULT_PREEMPTION_WARNING_SECONDS = float(
    os.environ.get("CIV6AZ_PREEMPTION_WARNING_SECONDS", "30.0")
)


class PreemptionHandler(PreemptionHandlerProtocol):
    """Handler for GCE spot/preemptible instance preemption.

    Monitors the GCE metadata service for preemption notices and
    triggers callbacks for graceful shutdown.

    Example:
        >>> def save_checkpoint():
        ...     print("Saving checkpoint before termination")
        ...
        >>> handler = PreemptionHandler(
        ...     checkpoint_callback=save_checkpoint,
        ...     cleanup_callbacks=[lambda: print("Cleanup 1")]
        ... )
        >>> handler.start_monitoring()
        >>> # ... training loop ...
        >>> handler.stop_monitoring()
    """

    def __init__(
        self,
        checkpoint_callback: Callable[[], None] | None = None,
        cleanup_callbacks: list[Callable[[], None]] | None = None,
        check_interval: float = 5.0,
        force_mock: bool = False,
        warning_seconds: float | None = None,
    ) -> None:
        """Initialize preemption handler.

        Args:
            checkpoint_callback: Function to call for saving checkpoint.
            cleanup_callbacks: List of cleanup functions to call.
            check_interval: Seconds between preemption checks.
            force_mock: Force use of mock (no actual HTTP requests).
            warning_seconds: Seconds of warning before termination (default: 30.0).
        """
        self._checkpoint_callback = checkpoint_callback
        self._cleanup_callbacks = list(cleanup_callbacks or [])
        self._check_interval = check_interval
        self._use_mock = force_mock
        self._warning_seconds = warning_seconds or DEFAULT_PREEMPTION_WARNING_SECONDS

        self._monitoring = False
        self._preemption_pending = False
        self._preemption_time: float | None = None
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Install signal handlers
        self._original_sigterm: Callable[[int, FrameType | None], Any] | int | None = None
        self._original_sigint: Callable[[int, FrameType | None], Any] | int | None = None

    def start_monitoring(self) -> None:
        """Start monitoring for preemption signals.

        Starts a background thread that polls the metadata service
        and installs signal handlers for SIGTERM and SIGINT.
        """
        if self._monitoring:
            logger.warning("Preemption monitoring already started")
            return

        self._monitoring = True
        self._stop_event.clear()

        # Install signal handlers
        try:
            self._original_sigterm = signal.signal(
                signal.SIGTERM, self._signal_handler
            )
            self._original_sigint = signal.signal(
                signal.SIGINT, self._signal_handler
            )
        except ValueError:
            # Signal handlers can only be set from main thread
            logger.warning("Cannot set signal handlers from non-main thread")

        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="preemption-monitor",
        )
        self._monitor_thread.start()

        logger.info(
            "Preemption monitoring started (mock=%s, interval=%.1fs)",
            self._use_mock,
            self._check_interval,
        )

    def stop_monitoring(self) -> None:
        """Stop monitoring for preemption signals.

        Stops the background thread and restores original signal handlers.
        """
        if not self._monitoring:
            return

        self._monitoring = False
        self._stop_event.set()

        # Wait for thread to finish
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=self._check_interval + 1)

        # Restore signal handlers
        try:
            if self._original_sigterm is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            if self._original_sigint is not None:
                signal.signal(signal.SIGINT, self._original_sigint)
        except ValueError:
            pass

        logger.info("Preemption monitoring stopped")

    def is_preemption_pending(self) -> bool:
        """Check if preemption is pending.

        Returns:
            True if preemption notice has been received.
        """
        return self._preemption_pending

    def get_time_remaining(self) -> float | None:
        """Get estimated time remaining before termination.

        Returns:
            Seconds remaining, or None if not preempted.
        """
        if not self._preemption_pending or self._preemption_time is None:
            return None
        remaining = self._warning_seconds - (time.time() - self._preemption_time)
        return max(0.0, remaining)

    def register_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a cleanup callback for preemption.

        Args:
            callback: Function to call during graceful shutdown.
        """
        with self._lock:
            self._cleanup_callbacks.append(callback)

    def _signal_handler(self, signum: int, frame: object) -> None:
        """Handle SIGTERM/SIGINT signals."""
        logger.warning("Received signal %d, initiating graceful shutdown", signum)
        self._trigger_preemption()

    def _monitoring_loop(self) -> None:
        """Background loop to check for preemption."""
        while not self._stop_event.is_set():
            try:
                if self._check_preemption():
                    self._trigger_preemption()
                    break
            except Exception as e:
                logger.debug("Preemption check failed: %s", e)

            self._stop_event.wait(timeout=self._check_interval)

    def _check_preemption(self) -> bool:
        """Check metadata service for preemption notice.

        Returns:
            True if preemption is pending.
        """
        if self._use_mock:
            return False

        try:
            import urllib.request

            req = urllib.request.Request(METADATA_URL, headers=METADATA_HEADERS)
            with urllib.request.urlopen(req, timeout=2) as response:
                result = response.read().decode().strip().lower()
                return bool(result == "true")
        except Exception:
            # Connection errors are expected when not on GCE
            return False

    def _trigger_preemption(self) -> None:
        """Trigger preemption handling callbacks."""
        with self._lock:
            if self._preemption_pending:
                return

            self._preemption_pending = True
            self._preemption_time = time.time()

        logger.warning("Preemption detected! Initiating graceful shutdown...")

        # Save checkpoint first
        if self._checkpoint_callback:
            try:
                logger.info("Saving checkpoint...")
                self._checkpoint_callback()
                logger.info("Checkpoint saved successfully")
            except Exception as e:
                logger.error("Failed to save checkpoint: %s", e)
                raise PreemptionError(
                    "Checkpoint save failed during preemption",
                    checkpoint_saved=False,
                    cause=e,
                ) from e

        # Run cleanup callbacks
        for i, callback in enumerate(self._cleanup_callbacks):
            try:
                logger.debug("Running cleanup callback %d", i + 1)
                callback()
            except Exception as e:
                logger.warning("Cleanup callback %d failed: %s", i + 1, e)

        logger.info("Preemption handling complete")

    def simulate_preemption(self) -> None:
        """Simulate a preemption event for testing.

        This triggers the same callbacks as a real preemption.
        """
        logger.info("Simulating preemption event")
        self._trigger_preemption()


def register_preemption_handler(
    checkpoint_callback: Callable[[], None] | None = None,
    cleanup_callbacks: list[Callable[[], None]] | None = None,
    check_interval: float = 5.0,
    auto_start: bool = True,
) -> PreemptionHandler:
    """Convenience function to create and start a preemption handler.

    Args:
        checkpoint_callback: Function to call for saving checkpoint.
        cleanup_callbacks: List of cleanup functions to call.
        check_interval: Seconds between preemption checks.
        auto_start: Whether to start monitoring immediately.

    Returns:
        Configured PreemptionHandler instance.
    """
    handler = PreemptionHandler(
        checkpoint_callback=checkpoint_callback,
        cleanup_callbacks=cleanup_callbacks,
        check_interval=check_interval,
    )

    if auto_start:
        handler.start_monitoring()

    return handler
