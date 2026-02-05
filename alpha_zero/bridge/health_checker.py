"""Game bridge health checker with circuit breaker pattern.

This module provides resilient health checking for the Civ6 game bridge,
implementing circuit breaker patterns and exponential backoff retry logic
to handle transient failures gracefully.

Usage:
    from alpha_zero.bridge.health_checker import (
        BridgeHealthConfig,
        GameBridgeHealthChecker,
    )

    config = BridgeHealthConfig()
    checker = GameBridgeHealthChecker(bridge, config)

    # Check health
    health = await checker.check_health()

    # Execute operation with retry
    result = await checker.execute_with_retry(some_async_operation, arg1, arg2)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
<<<<<<< HEAD
from typing import Any, TypeVar, cast
=======
from typing import Any, Protocol, TypeVar, cast
>>>>>>> origin/HEAD

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


<<<<<<< HEAD
=======
class BridgeProtocol(Protocol):
    """Protocol for bridge objects that support health checking.

    Any bridge implementation must provide a check_heartbeat method
    for use with GameBridgeHealthChecker.
    """

    def check_heartbeat(self) -> bool:
        """Check if the bridge connection is alive.

        Returns:
            True if heartbeat succeeds, False otherwise.
        """
        ...


>>>>>>> origin/HEAD
class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class BridgeHealthConfig:
    """Configuration for game bridge health checker.

    Attributes:
        check_interval_seconds: Interval between automatic health checks.
        timeout_seconds: Timeout for individual health check operations.
        max_retries: Maximum retry attempts for operations.
        retry_delay_seconds: Initial delay between retries.
        backoff_multiplier: Multiplier for exponential backoff.
        max_backoff_seconds: Maximum backoff delay.
        circuit_breaker_threshold: Failures before opening circuit.
        circuit_breaker_timeout_seconds: Time before half-open retry.
        success_threshold: Successes in half-open to close circuit.
        health_window_size: Number of recent results to track.
        degraded_threshold: Failure ratio to mark as degraded.
    """

    check_interval_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_BRIDGE_CHECK_INTERVAL", "5.0")
        )
    )
    timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_BRIDGE_TIMEOUT", "10.0")
        )
    )
    max_retries: int = field(
        default_factory=lambda: int(
            os.environ.get("CIV6AZ_BRIDGE_MAX_RETRIES", "5")
        )
    )
    retry_delay_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_BRIDGE_RETRY_DELAY", "1.0")
        )
    )
    backoff_multiplier: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_BRIDGE_BACKOFF_MULTIPLIER", "2.0")
        )
    )
    max_backoff_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_BRIDGE_MAX_BACKOFF", "60.0")
        )
    )
    circuit_breaker_threshold: int = field(
        default_factory=lambda: int(
            os.environ.get("CIV6AZ_BRIDGE_CIRCUIT_THRESHOLD", "3")
        )
    )
    circuit_breaker_timeout_seconds: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_BRIDGE_CIRCUIT_TIMEOUT", "30.0")
        )
    )
    success_threshold: int = field(
        default_factory=lambda: int(
            os.environ.get("CIV6AZ_BRIDGE_SUCCESS_THRESHOLD", "2")
        )
    )
    health_window_size: int = field(
        default_factory=lambda: int(
            os.environ.get("CIV6AZ_BRIDGE_HEALTH_WINDOW", "10")
        )
    )
    degraded_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_BRIDGE_DEGRADED_THRESHOLD", "0.3")
        )
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_delay_seconds <= 0:
            raise ValueError("retry_delay_seconds must be positive")
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1.0")
        if self.max_backoff_seconds <= 0:
            raise ValueError("max_backoff_seconds must be positive")
        if self.circuit_breaker_threshold < 1:
            raise ValueError("circuit_breaker_threshold must be >= 1")
        if self.circuit_breaker_timeout_seconds <= 0:
            raise ValueError("circuit_breaker_timeout_seconds must be positive")
        if self.success_threshold < 1:
            raise ValueError("success_threshold must be >= 1")
        if self.health_window_size < 1:
            raise ValueError("health_window_size must be >= 1")
        if not 0 <= self.degraded_threshold <= 1:
            raise ValueError("degraded_threshold must be between 0 and 1")

    @classmethod
    def from_env(cls) -> BridgeHealthConfig:
        """Create configuration from environment variables.

        Returns:
            BridgeHealthConfig instance.
        """
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> BridgeHealthConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            BridgeHealthConfig instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If YAML is invalid.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError("YAML must contain a mapping")

        # Extract bridge_health section if present
        if "bridge_health" in data:
            data = data["bridge_health"]

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BridgeHealthConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            BridgeHealthConfig instance.
        """
        kwargs: dict[str, Any] = {}

        if "check_interval_seconds" in data:
            kwargs["check_interval_seconds"] = float(data["check_interval_seconds"])
        if "timeout_seconds" in data:
            kwargs["timeout_seconds"] = float(data["timeout_seconds"])
        if "max_retries" in data:
            kwargs["max_retries"] = int(data["max_retries"])
        if "retry_delay_seconds" in data:
            kwargs["retry_delay_seconds"] = float(data["retry_delay_seconds"])
        if "backoff_multiplier" in data:
            kwargs["backoff_multiplier"] = float(data["backoff_multiplier"])
        if "max_backoff_seconds" in data:
            kwargs["max_backoff_seconds"] = float(data["max_backoff_seconds"])
        if "circuit_breaker_threshold" in data:
            kwargs["circuit_breaker_threshold"] = int(data["circuit_breaker_threshold"])
        if "circuit_breaker_timeout_seconds" in data:
            kwargs["circuit_breaker_timeout_seconds"] = float(
                data["circuit_breaker_timeout_seconds"]
            )
        if "success_threshold" in data:
            kwargs["success_threshold"] = int(data["success_threshold"])
        if "health_window_size" in data:
            kwargs["health_window_size"] = int(data["health_window_size"])
        if "degraded_threshold" in data:
            kwargs["degraded_threshold"] = float(data["degraded_threshold"])

        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "check_interval_seconds": self.check_interval_seconds,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "backoff_multiplier": self.backoff_multiplier,
            "max_backoff_seconds": self.max_backoff_seconds,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_timeout_seconds": self.circuit_breaker_timeout_seconds,
            "success_threshold": self.success_threshold,
            "health_window_size": self.health_window_size,
            "degraded_threshold": self.degraded_threshold,
        }


@dataclass
class BridgeHealth:
    """Health status of the game bridge.

    Attributes:
        status: Overall health status.
        circuit_state: Current circuit breaker state.
        is_connected: Whether bridge is connected to game.
        last_check_time: Timestamp of last health check.
        consecutive_failures: Number of consecutive failures.
        consecutive_successes: Number of consecutive successes.
        failure_rate: Recent failure rate (0-1).
        latency_ms: Average operation latency in milliseconds.
        error_message: Most recent error message if any.
        details: Additional health details.
    """

    status: HealthStatus
    circuit_state: CircuitState
    is_connected: bool
    last_check_time: float
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    failure_rate: float = 0.0
    latency_ms: float = 0.0
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "status": self.status.value,
            "circuit_state": self.circuit_state.value,
            "is_connected": self.is_connected,
            "last_check_time": self.last_check_time,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "failure_rate": self.failure_rate,
            "latency_ms": self.latency_ms,
            "error_message": self.error_message,
            "details": self.details,
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, message: str = "Circuit breaker is open", retry_after: float = 0):
        super().__init__(message)
        self.retry_after = retry_after


class OperationTimeoutError(Exception):
    """Exception raised when operation times out."""

    def __init__(self, message: str = "Operation timed out", timeout: float = 0):
        super().__init__(message)
        self.timeout = timeout


class GameBridgeHealthChecker:
    """Health checker for the game bridge with circuit breaker pattern.

    This class monitors bridge health and provides resilient operation
    execution with retry logic and circuit breaker protection.

    Attributes:
        bridge: The file bridge to monitor.
        config: Health checker configuration.
    """

    def __init__(
        self,
<<<<<<< HEAD
        bridge: Any,  # FileBridge or protocol-compatible
=======
        bridge: BridgeProtocol,
>>>>>>> origin/HEAD
        config: BridgeHealthConfig | None = None,
    ):
        """Initialize health checker.

        Args:
<<<<<<< HEAD
            bridge: File bridge to monitor (must have check_heartbeat method).
=======
            bridge: File bridge to monitor (must implement BridgeProtocol).
>>>>>>> origin/HEAD
            config: Health checker configuration.
        """
        self.bridge = bridge
        self.config = config or BridgeHealthConfig()

        # Circuit breaker state
        self._circuit_state = CircuitState.CLOSED
        self._circuit_opened_at: float = 0.0
        self._consecutive_failures = 0
        self._consecutive_successes = 0

        # Health tracking
        self._health_results: deque[bool] = deque(maxlen=self.config.health_window_size)
        self._last_check_time: float = 0.0
        self._last_error: str | None = None
        self._latencies: deque[float] = deque(maxlen=self.config.health_window_size)

        # Monitoring state
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None

        # Callbacks
        self._on_circuit_open: list[Callable[[], None]] = []
        self._on_circuit_close: list[Callable[[], None]] = []
        self._on_health_change: list[Callable[[BridgeHealth], None]] = []

        logger.info(
            f"GameBridgeHealthChecker initialized: "
            f"circuit_threshold={self.config.circuit_breaker_threshold}, "
            f"timeout={self.config.timeout_seconds}s"
        )

    @property
    def circuit_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self._circuit_state

    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open.

        Returns:
            True if circuit is open and rejecting requests.
        """
        if self._circuit_state == CircuitState.CLOSED:
            return False

        if self._circuit_state == CircuitState.OPEN:
            # Check if timeout has passed
            elapsed = time.time() - self._circuit_opened_at
            if elapsed >= self.config.circuit_breaker_timeout_seconds:
                self._transition_to_half_open()
                return False
            return True

        # Half-open allows some requests through
        return False

    def _transition_to_half_open(self) -> None:
        """Transition circuit to half-open state."""
        logger.info("Circuit breaker transitioning to half-open state")
        self._circuit_state = CircuitState.HALF_OPEN
        self._consecutive_successes = 0

    def _open_circuit(self) -> None:
        """Open the circuit breaker."""
        if self._circuit_state != CircuitState.OPEN:
            logger.warning(
                f"Circuit breaker opening after {self._consecutive_failures} consecutive failures"
            )
            self._circuit_state = CircuitState.OPEN
            self._circuit_opened_at = time.time()
            for callback in self._on_circuit_open:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in circuit open callback: {e}")

    def _close_circuit(self) -> None:
        """Close the circuit breaker."""
        if self._circuit_state != CircuitState.CLOSED:
            logger.info("Circuit breaker closing - service recovered")
            self._circuit_state = CircuitState.CLOSED
            self._consecutive_failures = 0
            for callback in self._on_circuit_close:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in circuit close callback: {e}")

    def record_success(self) -> None:
        """Record a successful operation."""
        self._health_results.append(True)
        self._consecutive_failures = 0
        self._consecutive_successes += 1
        self._last_error = None

        if self._circuit_state == CircuitState.HALF_OPEN:
            if self._consecutive_successes >= self.config.success_threshold:
                self._close_circuit()

    def record_failure(self, error: str | None = None) -> None:
        """Record a failed operation.

        Args:
            error: Error message to record.
        """
        self._health_results.append(False)
        self._consecutive_failures += 1
        self._consecutive_successes = 0
        self._last_error = error

        if self._circuit_state == CircuitState.HALF_OPEN:
            # Single failure in half-open returns to open
            self._open_circuit()
        elif self._consecutive_failures >= self.config.circuit_breaker_threshold:
            self._open_circuit()

    def _calculate_failure_rate(self) -> float:
        """Calculate recent failure rate.

        Returns:
            Failure rate between 0 and 1.
        """
        if not self._health_results:
            return 0.0
        failures = sum(1 for r in self._health_results if not r)
        return failures / len(self._health_results)

    def _calculate_avg_latency(self) -> float:
        """Calculate average operation latency in milliseconds.

        Returns:
            Average latency in ms.
        """
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def _determine_health_status(self, is_connected: bool) -> HealthStatus:
        """Determine overall health status.

        Args:
            is_connected: Whether bridge is connected.

        Returns:
            Health status.
        """
        if self._circuit_state == CircuitState.OPEN:
            return HealthStatus.UNHEALTHY

        if not is_connected:
            return HealthStatus.UNHEALTHY

        failure_rate = self._calculate_failure_rate()
        if failure_rate >= self.config.degraded_threshold:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    async def check_health(self) -> BridgeHealth:
        """Check bridge health status.

        Returns:
            BridgeHealth with current status.
        """
        start_time = time.time()

        try:
            # Check basic connectivity
            is_connected = await asyncio.to_thread(self.bridge.check_heartbeat)
            latency_ms = (time.time() - start_time) * 1000
            self._latencies.append(latency_ms)

            if is_connected:
                self.record_success()
            else:
                self.record_failure("Heartbeat check failed")

        except Exception as e:
            is_connected = False
            latency_ms = (time.time() - start_time) * 1000
            self._latencies.append(latency_ms)
            self.record_failure(str(e))
            logger.warning(f"Health check failed: {e}")

        self._last_check_time = time.time()

        # Build health report
        health = BridgeHealth(
            status=self._determine_health_status(is_connected),
            circuit_state=self._circuit_state,
            is_connected=is_connected,
            last_check_time=self._last_check_time,
            consecutive_failures=self._consecutive_failures,
            consecutive_successes=self._consecutive_successes,
            failure_rate=self._calculate_failure_rate(),
            latency_ms=self._calculate_avg_latency(),
            error_message=self._last_error,
            details={
                "health_window_size": len(self._health_results),
                "circuit_timeout_remaining": max(
                    0,
                    self.config.circuit_breaker_timeout_seconds
                    - (time.time() - self._circuit_opened_at)
                    if self._circuit_state == CircuitState.OPEN
                    else 0,
                ),
            },
        )

        # Notify listeners
        for callback in self._on_health_change:
            try:
                callback(health)
            except Exception as e:
                logger.error(f"Error in health change callback: {e}")

        return health

    async def execute_with_retry(
        self,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute an async operation with retry and circuit breaker.

        Args:
            operation: Async callable to execute.
            *args: Positional arguments for operation.
            **kwargs: Keyword arguments for operation.

        Returns:
            Operation result.

        Raises:
            CircuitBreakerOpenError: If circuit is open.
            OperationTimeoutError: If all retries timeout.
            Exception: If operation fails after all retries.
        """
        # Check circuit breaker
        if self.is_circuit_open():
            retry_after = max(
                0,
                self.config.circuit_breaker_timeout_seconds
                - (time.time() - self._circuit_opened_at),
            )
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open, retry after {retry_after:.1f}s",
                retry_after=retry_after,
            )

        last_exception: Exception | None = None
        delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()

                # Execute with timeout
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=self.config.timeout_seconds,
                )

                # Record success
                latency_ms = (time.time() - start_time) * 1000
                self._latencies.append(latency_ms)
                self.record_success()

                logger.debug(
                    f"Operation succeeded on attempt {attempt + 1}, "
                    f"latency={latency_ms:.1f}ms"
                )

                return cast(T, result)

<<<<<<< HEAD
            except asyncio.TimeoutError:
=======
            except TimeoutError:
>>>>>>> origin/HEAD
                last_exception = OperationTimeoutError(
                    f"Operation timed out after {self.config.timeout_seconds}s",
                    timeout=self.config.timeout_seconds,
                )
                self.record_failure("Timeout")
                logger.warning(f"Operation timeout on attempt {attempt + 1}")

            except Exception as e:
                last_exception = e
                self.record_failure(str(e))
                logger.warning(f"Operation failed on attempt {attempt + 1}: {e}")

            # Check if we should retry
            if attempt < self.config.max_retries:
                if self.is_circuit_open():
                    raise CircuitBreakerOpenError(
                        "Circuit breaker opened during retries"
                    )

                logger.debug(f"Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
                delay = min(
                    delay * self.config.backoff_multiplier,
                    self.config.max_backoff_seconds,
                )

        # All retries exhausted
        if last_exception:
            raise last_exception
        raise RuntimeError("Operation failed with no exception captured")

    def register_circuit_open_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for circuit open events.

        Args:
            callback: Function to call when circuit opens.
        """
        self._on_circuit_open.append(callback)

    def register_circuit_close_callback(self, callback: Callable[[], None]) -> None:
        """Register callback for circuit close events.

        Args:
            callback: Function to call when circuit closes.
        """
        self._on_circuit_close.append(callback)

    def register_health_change_callback(
        self, callback: Callable[[BridgeHealth], None]
    ) -> None:
        """Register callback for health status changes.

        Args:
            callback: Function to call on health check completion.
        """
        self._on_health_change.append(callback)

    async def start_monitoring(self) -> None:
        """Start automatic health monitoring in background."""
        if self._monitoring:
            logger.warning("Health monitoring already running")
            return

        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info(
            f"Started health monitoring with {self.config.check_interval_seconds}s interval"
        )

    async def stop_monitoring(self) -> None:
        """Stop automatic health monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                logger.debug("Health monitoring task cancellation acknowledged during stop.")
            self._monitor_task = None
        logger.info("Stopped health monitoring")

    async def _monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while self._monitoring:
            try:
                await self.check_health()
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")

            await asyncio.sleep(self.config.check_interval_seconds)

    def reset(self) -> None:
        """Reset health checker state."""
        self._circuit_state = CircuitState.CLOSED
        self._circuit_opened_at = 0.0
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._health_results.clear()
        self._latencies.clear()
        self._last_error = None
        logger.info("Health checker state reset")

    def get_statistics(self) -> dict[str, Any]:
        """Get health checker statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "circuit_state": self._circuit_state.value,
            "consecutive_failures": self._consecutive_failures,
            "consecutive_successes": self._consecutive_successes,
            "failure_rate": self._calculate_failure_rate(),
            "avg_latency_ms": self._calculate_avg_latency(),
            "health_window_size": len(self._health_results),
            "last_check_time": self._last_check_time,
            "last_error": self._last_error,
            "monitoring_active": self._monitoring,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"GameBridgeHealthChecker("
            f"circuit={self._circuit_state.value}, "
            f"failures={self._consecutive_failures}, "
            f"failure_rate={self._calculate_failure_rate():.2%})"
        )
