"""Preemption handling for cloud training.

This module provides:
- PreemptionHandler: Graceful handling of VM preemption signals
- Signal-based detection for SIGTERM/SIGINT
- GCE metadata service monitoring for preemption notices
- Checkpoint callbacks for emergency saves

Google Cloud sends SIGTERM with a 30-second grace period before
terminating preemptible instances.

Usage:
    from alpha_zero.cloud.preemption import PreemptionHandler

    def save_checkpoint(iteration: int) -> str:
        # Save checkpoint logic
        return "/path/to/checkpoint.pt"

    handler = PreemptionHandler(
        checkpoint_callback=save_checkpoint,
    )
    handler.setup()

    while training:
        if handler.should_stop():
            break
        # Training logic...
        handler.update_iteration(current_iteration)

Simple usage with global handler:
    from alpha_zero.cloud.preemption import register_preemption_handler, is_preempted

    handler = register_preemption_handler()

    while not is_preempted():
        # Do work
        pass
"""

from __future__ import annotations

import atexit
import logging
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# GCE metadata service URL for preemption detection
GCE_METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
GCE_METADATA_HEADERS = {"Metadata-Flavor": "Google"}


class PreemptionHandlerProtocol(Protocol):
    """Protocol for preemption handlers."""

    def start_monitoring(self) -> None:
        """Start monitoring for preemption signals."""
        ...

    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        ...

    def is_preemption_pending(self) -> bool:
        """Check if preemption is pending."""
        ...


@dataclass
class PreemptionState:
    """State tracking for preemption handling.

    Attributes:
        preemption_requested: Whether preemption signal received.
        current_iteration: Current training iteration.
        checkpoint_saved: Whether emergency checkpoint was saved.
        checkpoint_path: Path to saved checkpoint.
        shutdown_requested: Whether shutdown was requested.
        preemption_time: Timestamp when preemption was signaled.
    """

    preemption_requested: bool = False
    current_iteration: int = 0
    checkpoint_saved: bool = False
    checkpoint_path: str = ""
    shutdown_requested: bool = False
    preemption_time: datetime | None = None


class PreemptionHandler:
    """Handles graceful shutdown on preemption signals.

    Features:
    - Signal handler installation (SIGTERM, SIGINT)
    - Optional GCE metadata service monitoring
    - Checkpoint callback on preemption
    - Cleanup callbacks for graceful shutdown
    - Thread-safe state management

    Attributes:
        checkpoint_callback: Function to call for emergency checkpoint.
        cleanup_callbacks: List of cleanup functions to call on shutdown.
        check_interval: Interval for metadata service checks.
        enable_metadata_check: Enable GCE metadata service monitoring.
        grace_period: Seconds to allow for cleanup (default: 25s).
    """

    def __init__(
        self,
        checkpoint_callback: Callable[[int], str] | None = None,
        cleanup_callbacks: list[Callable[[], None]] | None = None,
        check_interval: float = 5.0,
        enable_metadata_check: bool = True,
        force_mock: bool = False,
        grace_period: float = 25.0,
    ) -> None:
        """Initialize preemption handler.

        Args:
            checkpoint_callback: Function to save checkpoint, takes iteration,
                returns checkpoint path.
            cleanup_callbacks: List of cleanup functions to call on shutdown.
            check_interval: Seconds between metadata service checks.
            enable_metadata_check: Enable GCE metadata service monitoring.
            force_mock: Force mock mode (disable metadata checks).
            grace_period: Seconds to allow for cleanup.
        """
        self.checkpoint_callback = checkpoint_callback
        self.cleanup_callbacks = cleanup_callbacks or []
        self.check_interval = check_interval
        self.enable_metadata_check = enable_metadata_check and not force_mock
        self.grace_period = grace_period
        self._force_mock = force_mock

        # State
        self._state = PreemptionState()
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # Monitoring thread
        self._monitor_thread: threading.Thread | None = None
        self._monitoring = False

        # Original signal handlers
        self._original_sigterm: Any = None
        self._original_sigint: Any = None

    # Alias for compatibility with simpler API
    @property
    def is_preempted(self) -> bool:
        """Whether preemption signal was received."""
        return self.is_preemption_pending()

    def setup(self) -> None:
        """Set up preemption handling.

        Installs signal handlers and optionally starts metadata monitoring.
        """
        self._install_signal_handlers()
        self._register_atexit()

        if self.enable_metadata_check:
            self._start_metadata_monitoring()

        logger.info("Preemption handler initialized")

    def register(self) -> None:
        """Register signal handlers (alias for setup)."""
        self.setup()

    def unregister(self) -> None:
        """Restore original signal handlers (alias for shutdown)."""
        self.shutdown()

    def shutdown(self) -> None:
        """Shutdown preemption handler.

        Stops monitoring and restores original signal handlers.
        """
        self._stop_metadata_monitoring()
        self._restore_signal_handlers()
        logger.info("Preemption handler shutdown complete")

    def update_iteration(self, iteration: int) -> None:
        """Update current training iteration.

        Args:
            iteration: Current training iteration number.
        """
        with self._lock:
            self._state.current_iteration = iteration

    def should_stop(self) -> bool:
        """Check if training should stop.

        Returns:
            True if preemption or shutdown was requested.
        """
        with self._lock:
            return self._state.preemption_requested or self._state.shutdown_requested

    def is_preemption_pending(self) -> bool:
        """Check if preemption is pending.

        Returns:
            True if preemption signal was received.
        """
        with self._lock:
            return self._state.preemption_requested

    def check_preemption(self) -> bool:
        """Check if preemption was signaled (alias for is_preemption_pending).

        Returns:
            True if preemption signal was received.
        """
        return self.is_preemption_pending()

    def get_state(self) -> PreemptionState:
        """Get current preemption state.

        Returns:
            Copy of current state.
        """
        with self._lock:
            return PreemptionState(
                preemption_requested=self._state.preemption_requested,
                current_iteration=self._state.current_iteration,
                checkpoint_saved=self._state.checkpoint_saved,
                checkpoint_path=self._state.checkpoint_path,
                shutdown_requested=self._state.shutdown_requested,
                preemption_time=self._state.preemption_time,
            )

    def time_since_preemption(self) -> float | None:
        """Get time elapsed since preemption signal.

        Returns:
            Seconds since preemption, or None if not preempted.
        """
        with self._lock:
            if self._state.preemption_time is None:
                return None
            return (datetime.now(UTC) - self._state.preemption_time).total_seconds()

    def trigger_checkpoint(self) -> str | None:
        """Manually trigger checkpoint save.

        Returns:
            Checkpoint path if saved, None otherwise.
        """
        return self._save_emergency_checkpoint()

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback to run on preemption.

        Callbacks are executed in order of registration.
        Each callback should complete quickly (target <5s each).

        Args:
            callback: Function to call on preemption.
        """
        self.cleanup_callbacks.append(callback)
        logger.debug(f"Added cleanup callback (total: {len(self.cleanup_callbacks)})")

    def wait_for_shutdown(self, timeout: float | None = None) -> bool:
        """Wait for shutdown signal.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if shutdown was signaled, False if timeout occurred.
        """
        return self._shutdown_event.wait(timeout=timeout)

    def _install_signal_handlers(self) -> None:
        """Install signal handlers for SIGTERM and SIGINT."""
        # Only install handlers on Unix-like systems or when not in mock mode
        if sys.platform == "win32" and not self._force_mock:
            logger.info("Signal handlers limited on Windows")
            self._original_sigint = signal.signal(signal.SIGINT, self._handle_signal)
            return

        try:
            self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
            self._original_sigint = signal.signal(signal.SIGINT, self._handle_signal)
            logger.debug("Signal handlers installed (SIGTERM, SIGINT)")
        except ValueError as e:
            # Can happen if not main thread
            logger.warning(f"Could not install signal handlers: {e}")

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        try:
            if self._original_sigterm is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm)
            if self._original_sigint is not None:
                signal.signal(signal.SIGINT, self._original_sigint)
        except ValueError:
            pass  # Ignore if not main thread

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle preemption/shutdown signal.

        Args:
            signum: Signal number.
            frame: Current stack frame.
        """
        signal_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.warning(
            f"Received {signal_name} - initiating graceful shutdown "
            f"(grace_period={self.grace_period}s)"
        )

        with self._lock:
            self._state.preemption_time = datetime.now(UTC)
            if signum == signal.SIGTERM:
                self._state.preemption_requested = True
            else:
                self._state.shutdown_requested = True

        self._shutdown_event.set()

        # Run cleanup in a thread to avoid blocking signal handler
        cleanup_thread = threading.Thread(target=self._run_cleanup_with_timeout)
        cleanup_thread.start()
        cleanup_thread.join(timeout=self.grace_period)

        if cleanup_thread.is_alive():
            logger.error("Cleanup did not complete within grace period, forcing exit")

        # For SIGTERM (preemption), exit gracefully
        if signum == signal.SIGTERM:
            logger.info("Exiting due to preemption")
            sys.exit(0)

    def _run_cleanup_with_timeout(self) -> None:
        """Run cleanup callbacks with timeout monitoring."""
        deadline = time.time() + self.grace_period

        # First, save emergency checkpoint
        self._save_emergency_checkpoint()

        # Then run other cleanup callbacks
        for i, callback in enumerate(self.cleanup_callbacks):
            if time.time() >= deadline:
                logger.warning("Cleanup timeout - skipping remaining callbacks")
                break

            try:
                remaining = deadline - time.time()
                logger.info(
                    f"Running cleanup callback {i + 1}/{len(self.cleanup_callbacks)} "
                    f"({remaining:.1f}s remaining)"
                )
                callback()
            except Exception as e:
                logger.error(f"Error in cleanup callback {i + 1}: {e}")

        logger.info("Cleanup completed")

    def _save_emergency_checkpoint(self) -> str | None:
        """Save emergency checkpoint.

        Returns:
            Checkpoint path if saved, None otherwise.
        """
        if self.checkpoint_callback is None:
            logger.warning("No checkpoint callback configured")
            return None

        with self._lock:
            if self._state.checkpoint_saved:
                logger.info(f"Checkpoint already saved: {self._state.checkpoint_path}")
                return self._state.checkpoint_path

            iteration = self._state.current_iteration

        try:
            checkpoint_path = self.checkpoint_callback(iteration)
            with self._lock:
                self._state.checkpoint_saved = True
                self._state.checkpoint_path = checkpoint_path

            logger.info(f"Emergency checkpoint saved: {checkpoint_path}")
            return checkpoint_path

        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")
            return None

    def _register_atexit(self) -> None:
        """Register atexit handler for cleanup."""
        atexit.register(self._atexit_handler)

    def _atexit_handler(self) -> None:
        """Handle process exit."""
        self._stop_metadata_monitoring()
        # Run cleanup callbacks without timeout (process is already exiting)
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")

    def _start_metadata_monitoring(self) -> None:
        """Start GCE metadata service monitoring thread."""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._metadata_monitoring_loop,
            daemon=True,
            name="preemption-monitor",
        )
        self._monitor_thread.start()
        logger.debug("Started metadata monitoring thread")

    def _stop_metadata_monitoring(self) -> None:
        """Stop GCE metadata service monitoring."""
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=self.check_interval + 1)

    def _metadata_monitoring_loop(self) -> None:
        """Background thread to monitor GCE metadata service."""
        while self._monitoring:
            try:
                if self._check_gce_preemption():
                    logger.warning("GCE preemption detected via metadata service")
                    with self._lock:
                        self._state.preemption_requested = True
                        self._state.preemption_time = datetime.now(UTC)

                    self._shutdown_event.set()

                    # Save checkpoint and run cleanup
                    self._save_emergency_checkpoint()
                    for callback in self.cleanup_callbacks:
                        try:
                            callback()
                        except Exception as e:
                            logger.error(f"Cleanup callback failed: {e}")
                    break

            except Exception as e:
                # Log but continue monitoring
                logger.debug(f"Metadata check error (expected on non-GCE): {e}")

            time.sleep(self.check_interval)

    def _check_gce_preemption(self) -> bool:
        """Check GCE metadata service for preemption notice.

        Returns:
            True if preemption is pending.
        """
        if self._force_mock:
            return False

        try:
            import urllib.request

            req = urllib.request.Request(
                GCE_METADATA_URL,
                headers=GCE_METADATA_HEADERS,
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                result = response.read().decode().strip().lower()
                return bool(result == "true")

        except Exception:
            # Not on GCE or metadata service unavailable
            return False


class MockPreemptionHandler(PreemptionHandler):
    """Mock preemption handler for testing.

    Allows manual triggering of preemption events.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize mock handler.

        Args:
            **kwargs: Arguments passed to parent class.
        """
        kwargs["force_mock"] = True
        kwargs["enable_metadata_check"] = False
        super().__init__(**kwargs)

    def trigger_preemption(self) -> None:
        """Manually trigger a preemption event (for testing)."""
        logger.info("Mock preemption triggered")
        with self._lock:
            self._state.preemption_requested = True
            self._state.preemption_time = datetime.now(UTC)

        self._shutdown_event.set()
        self._save_emergency_checkpoint()

        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")

    def trigger_shutdown(self) -> None:
        """Manually trigger a shutdown request (for testing)."""
        logger.info("Mock shutdown triggered")
        with self._lock:
            self._state.shutdown_requested = True

        self._shutdown_event.set()

        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")

    def reset(self) -> None:
        """Reset state for testing."""
        with self._lock:
            self._state = PreemptionState()
        self._shutdown_event.clear()


def create_preemption_handler(
    checkpoint_callback: Callable[[int], str] | None = None,
    cleanup_callbacks: list[Callable[[], None]] | None = None,
    force_mock: bool = False,
    grace_period: float = 25.0,
) -> PreemptionHandler:
    """Create a preemption handler.

    Factory function that creates the appropriate handler type.

    Args:
        checkpoint_callback: Function to save checkpoint.
        cleanup_callbacks: Cleanup functions to call on shutdown.
        force_mock: Force use of mock handler.
        grace_period: Seconds to allow for cleanup.

    Returns:
        PreemptionHandler instance.
    """
    if force_mock:
        return MockPreemptionHandler(
            checkpoint_callback=checkpoint_callback,
            cleanup_callbacks=cleanup_callbacks,
            grace_period=grace_period,
        )

    return PreemptionHandler(
        checkpoint_callback=checkpoint_callback,
        cleanup_callbacks=cleanup_callbacks,
        grace_period=grace_period,
    )


# Global preemption handler instance
_global_handler: PreemptionHandler | None = None


def register_preemption_handler(
    grace_period: float = 25.0,
    cleanup_callbacks: list[Callable[[], None]] | None = None,
    checkpoint_callback: Callable[[int], str] | None = None,
) -> PreemptionHandler:
    """Register a global preemption handler.

    This is a convenience function for simple use cases where
    only one handler is needed.

    Args:
        grace_period: Seconds to allow for cleanup.
        cleanup_callbacks: Optional list of cleanup callbacks.
        checkpoint_callback: Optional checkpoint callback.

    Returns:
        The registered PreemptionHandler instance.
    """
    global _global_handler

    _global_handler = PreemptionHandler(
        grace_period=grace_period,
        cleanup_callbacks=cleanup_callbacks or [],
        checkpoint_callback=checkpoint_callback,
    )
    _global_handler.register()

    return _global_handler


def get_preemption_handler() -> PreemptionHandler | None:
    """Get the global preemption handler.

    Returns:
        The global handler, or None if not registered.
    """
    return _global_handler


def is_preempted() -> bool:
    """Check if the global handler was preempted.

    Returns:
        True if preemption signal was received.
    """
    if _global_handler is None:
        return False
    return _global_handler.is_preempted


__all__ = [
    # Mock for testing
    "MockPreemptionHandler",
    # Main classes (sorted)
    "PreemptionHandler",
    "PreemptionHandlerProtocol",
    "PreemptionState",
    # Factory function
    "create_preemption_handler",
    # Global handler functions
    "get_preemption_handler",
    "is_preempted",
    # Global handler registration
    "register_preemption_handler",
]
