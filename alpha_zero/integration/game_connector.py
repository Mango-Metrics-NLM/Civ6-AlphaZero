"""Game Integration Layer with Error Recovery.

This module provides a robust integration layer for connecting to various
game environments (MicroCiv, Unciv, Civ6) with comprehensive error handling,
retry logic, and latency compensation.

Features:
- Automatic retry with exponential backoff
- Connection health monitoring
- Latency measurement and compensation
- Graceful degradation on errors
- State checkpointing for recovery
- Configurable timeout policies

Example usage:
    >>> from alpha_zero.integration.game_connector import (
    ...     GameConnector,
    ...     ConnectionConfig,
    ...     GameType,
    ... )
    >>> config = ConnectionConfig.default()
    >>> connector = GameConnector(config, game_type=GameType.MICROCIV)
    >>> connector.connect()
    >>> state = connector.get_state()
    >>> connector.apply_action(action)
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol, TypeVar

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

T = TypeVar("T")
StateT = TypeVar("StateT")


class GameStateProtocol(Protocol):
    """Protocol for game state objects."""
    def to_dict(self) -> dict[str, Any]: ...


# =============================================================================
# Enums
# =============================================================================


class GameType(Enum):
    """Supported game types."""
    MICROCIV = auto()
    UNCIV = auto()
    CIV6 = auto()


class ConnectionStatus(Enum):
    """Connection status states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    ERROR = auto()
    TIMEOUT = auto()


class ErrorSeverity(Enum):
    """Error severity levels."""
    WARNING = auto()  # Non-critical, can continue
    RECOVERABLE = auto()  # Can recover with retry
    CRITICAL = auto()  # Fatal, need restart


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries (seconds).
        max_delay: Maximum delay between retries (seconds).
        exponential_base: Base for exponential backoff.
        jitter: Random jitter factor (0-1).
    """
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in seconds.
        """
        import random
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay,
        )
        jitter = delay * self.jitter * random.uniform(-1, 1)
        return max(0, delay + jitter)


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior.

    Attributes:
        connect_timeout: Timeout for initial connection (seconds).
        action_timeout: Timeout for action execution (seconds).
        state_timeout: Timeout for state retrieval (seconds).
        heartbeat_interval: Interval for health checks (seconds).
        idle_timeout: Timeout for idle connections (seconds).
    """
    connect_timeout: float = 30.0
    action_timeout: float = 10.0
    state_timeout: float = 5.0
    heartbeat_interval: float = 5.0
    idle_timeout: float = 300.0


@dataclass
class LatencyConfig:
    """Configuration for latency handling.

    Attributes:
        measurement_window: Number of samples for latency averaging.
        latency_threshold_ms: Threshold for latency warnings (ms).
        compensation_enabled: Whether to enable latency compensation.
        target_latency_ms: Target latency for compensation (ms).
    """
    measurement_window: int = 50
    latency_threshold_ms: float = 100.0
    compensation_enabled: bool = True
    target_latency_ms: float = 50.0


@dataclass
class RecoveryConfig:
    """Configuration for error recovery.

    Attributes:
        checkpoint_enabled: Whether to enable state checkpointing.
        checkpoint_interval: Turns between checkpoints.
        checkpoint_dir: Directory for checkpoint files.
        max_checkpoints: Maximum checkpoints to keep.
        auto_recover: Whether to auto-recover from errors.
    """
    checkpoint_enabled: bool = True
    checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints/game_state"
    max_checkpoints: int = 5
    auto_recover: bool = True


@dataclass
class ConnectionConfig:
    """Complete connection configuration.

    Attributes:
        retry: Retry configuration.
        timeout: Timeout configuration.
        latency: Latency configuration.
        recovery: Recovery configuration.
        log_level: Logging level for connection events.
        metrics_enabled: Whether to collect connection metrics.
    """
    retry: RetryConfig = field(default_factory=RetryConfig)
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    log_level: str = "INFO"
    metrics_enabled: bool = True

    @classmethod
    def default(cls) -> ConnectionConfig:
        """Create default configuration."""
        return cls()

    @classmethod
    def aggressive_retry(cls) -> ConnectionConfig:
        """Create configuration with aggressive retry settings."""
        return cls(
            retry=RetryConfig(
                max_retries=10,
                base_delay=0.5,
                max_delay=60.0,
            ),
        )

    @classmethod
    def low_latency(cls) -> ConnectionConfig:
        """Create configuration optimized for low latency."""
        return cls(
            timeout=TimeoutConfig(
                action_timeout=5.0,
                state_timeout=2.0,
                heartbeat_interval=2.0,
            ),
            latency=LatencyConfig(
                measurement_window=20,
                latency_threshold_ms=50.0,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "retry": {
                "max_retries": self.retry.max_retries,
                "base_delay": self.retry.base_delay,
                "max_delay": self.retry.max_delay,
                "exponential_base": self.retry.exponential_base,
                "jitter": self.retry.jitter,
            },
            "timeout": {
                "connect_timeout": self.timeout.connect_timeout,
                "action_timeout": self.timeout.action_timeout,
                "state_timeout": self.timeout.state_timeout,
                "heartbeat_interval": self.timeout.heartbeat_interval,
                "idle_timeout": self.timeout.idle_timeout,
            },
            "latency": {
                "measurement_window": self.latency.measurement_window,
                "latency_threshold_ms": self.latency.latency_threshold_ms,
                "compensation_enabled": self.latency.compensation_enabled,
                "target_latency_ms": self.latency.target_latency_ms,
            },
            "recovery": {
                "checkpoint_enabled": self.recovery.checkpoint_enabled,
                "checkpoint_interval": self.recovery.checkpoint_interval,
                "checkpoint_dir": self.recovery.checkpoint_dir,
                "max_checkpoints": self.recovery.max_checkpoints,
                "auto_recover": self.recovery.auto_recover,
            },
            "log_level": self.log_level,
            "metrics_enabled": self.metrics_enabled,
        }


# =============================================================================
# Error Handling
# =============================================================================


class GameConnectionError(Exception):
    """Base exception for game connection errors."""

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.RECOVERABLE,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.severity = severity
        self.original_error = original_error


class ConnectionTimeoutError(GameConnectionError):
    """Connection timeout error."""

    def __init__(self, message: str = "Connection timed out"):
        super().__init__(message, ErrorSeverity.RECOVERABLE)


class ActionExecutionError(GameConnectionError):
    """Error executing action."""

    def __init__(
        self,
        message: str,
        action: Any = None,
        original_error: Exception | None = None,
    ):
        super().__init__(message, ErrorSeverity.RECOVERABLE, original_error)
        self.action = action


class StateRetrievalError(GameConnectionError):
    """Error retrieving game state."""

    def __init__(self, message: str, original_error: Exception | None = None):
        super().__init__(message, ErrorSeverity.RECOVERABLE, original_error)


class InvalidStateError(GameConnectionError):
    """Invalid game state error."""

    def __init__(self, message: str):
        super().__init__(message, ErrorSeverity.CRITICAL)


# =============================================================================
# Latency Tracker
# =============================================================================


class LatencyTracker:
    """Tracks connection latency for monitoring and compensation."""

    def __init__(self, config: LatencyConfig):
        """Initialize latency tracker.

        Args:
            config: Latency configuration.
        """
        self.config = config
        self._samples: list[float] = []
        self._action_samples: list[float] = []
        self._logger = logging.getLogger(f"{__name__}.LatencyTracker")

    def record_sample(self, latency_ms: float, is_action: bool = False) -> None:
        """Record a latency sample.

        Args:
            latency_ms: Latency in milliseconds.
            is_action: Whether this is an action latency sample.
        """
        self._samples.append(latency_ms)
        if len(self._samples) > self.config.measurement_window:
            self._samples.pop(0)

        if is_action:
            self._action_samples.append(latency_ms)
            if len(self._action_samples) > self.config.measurement_window:
                self._action_samples.pop(0)

        if latency_ms > self.config.latency_threshold_ms:
            self._logger.warning(f"High latency detected: {latency_ms:.1f}ms")

    @property
    def average_latency(self) -> float:
        """Get average latency in milliseconds."""
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    @property
    def action_latency(self) -> float:
        """Get average action latency in milliseconds."""
        if not self._action_samples:
            return 0.0
        return sum(self._action_samples) / len(self._action_samples)

    @property
    def p95_latency(self) -> float:
        """Get 95th percentile latency."""
        if not self._samples:
            return 0.0
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * 0.95)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    @property
    def compensation_delay(self) -> float:
        """Get delay needed for latency compensation (seconds)."""
        if not self.config.compensation_enabled:
            return 0.0
        target_ms = self.config.target_latency_ms
        current_ms = self.average_latency
        if current_ms < target_ms:
            return 0.0
        return (current_ms - target_ms) / 1000.0

    def get_stats(self) -> dict[str, float]:
        """Get latency statistics.

        Returns:
            Dictionary with latency statistics.
        """
        return {
            "average_ms": self.average_latency,
            "action_ms": self.action_latency,
            "p95_ms": self.p95_latency,
            "samples": len(self._samples),
            "compensation_delay": self.compensation_delay,
        }

    def reset(self) -> None:
        """Reset latency tracking."""
        self._samples.clear()
        self._action_samples.clear()


# =============================================================================
# Connection Metrics
# =============================================================================


@dataclass
class ConnectionMetrics:
    """Metrics for connection monitoring."""

    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    retries: int = 0
    reconnections: int = 0
    total_games: int = 0
    checkpoints_created: int = 0
    recoveries: int = 0
    uptime_seconds: float = 0.0
    last_error: str | None = None
    last_error_time: float | None = None

    @property
    def success_rate(self) -> float:
        """Get action success rate."""
        if self.total_actions == 0:
            return 1.0
        return self.successful_actions / self.total_actions

    @property
    def retry_rate(self) -> float:
        """Get retry rate."""
        if self.total_actions == 0:
            return 0.0
        return self.retries / self.total_actions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_actions": self.total_actions,
            "successful_actions": self.successful_actions,
            "failed_actions": self.failed_actions,
            "success_rate": self.success_rate,
            "retries": self.retries,
            "retry_rate": self.retry_rate,
            "reconnections": self.reconnections,
            "total_games": self.total_games,
            "checkpoints_created": self.checkpoints_created,
            "recoveries": self.recoveries,
            "uptime_seconds": self.uptime_seconds,
            "last_error": self.last_error,
        }


# =============================================================================
# State Checkpoint Manager
# =============================================================================


class CheckpointManager:
    """Manages game state checkpoints for recovery."""

    def __init__(self, config: RecoveryConfig):
        """Initialize checkpoint manager.

        Args:
            config: Recovery configuration.
        """
        self.config = config
        self._checkpoint_dir = Path(config.checkpoint_dir)
        self._checkpoints: list[Path] = []
        self._logger = logging.getLogger(f"{__name__}.CheckpointManager")

        if config.checkpoint_enabled:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._load_existing_checkpoints()

    def _load_existing_checkpoints(self) -> None:
        """Load list of existing checkpoints."""
        if self._checkpoint_dir.exists():
            self._checkpoints = sorted(
                self._checkpoint_dir.glob("checkpoint_*.json"),
                key=lambda p: p.stat().st_mtime,
            )

    def save_checkpoint(
        self,
        state: dict[str, Any],
        game_id: str,
        turn: int,
    ) -> Path | None:
        """Save a state checkpoint.

        Args:
            state: Game state dictionary.
            game_id: Game identifier.
            turn: Current turn number.

        Returns:
            Path to checkpoint file, or None if disabled.
        """
        if not self.config.checkpoint_enabled:
            return None

        checkpoint_name = f"checkpoint_{game_id}_{turn:05d}.json"
        checkpoint_path = self._checkpoint_dir / checkpoint_name

        try:
            with open(checkpoint_path, "w") as f:
                json.dump({
                    "game_id": game_id,
                    "turn": turn,
                    "timestamp": time.time(),
                    "state": state,
                }, f, indent=2)

            self._checkpoints.append(checkpoint_path)
            self._cleanup_old_checkpoints()

            self._logger.debug(f"Saved checkpoint: {checkpoint_name}")
            return checkpoint_path

        except Exception as e:
            self._logger.error(f"Failed to save checkpoint: {e}")
            return None

    def load_latest_checkpoint(
        self,
        game_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Load the latest checkpoint.

        Args:
            game_id: Optional game ID to filter by.

        Returns:
            Checkpoint data dictionary, or None if not found.
        """
        if not self.config.checkpoint_enabled or not self._checkpoints:
            return None

        # Filter by game ID if provided
        candidates = self._checkpoints
        if game_id:
            candidates = [
                p for p in candidates
                if game_id in p.name
            ]

        if not candidates:
            return None

        # Load most recent
        latest = candidates[-1]
        try:
            with open(latest) as f:
                data: dict[str, Any] = json.load(f)
            self._logger.info(f"Loaded checkpoint: {latest.name}")
            return data
        except Exception as e:
            self._logger.error(f"Failed to load checkpoint: {e}")
            return None

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints exceeding max count."""
        while len(self._checkpoints) > self.config.max_checkpoints:
            oldest = self._checkpoints.pop(0)
            try:
                oldest.unlink()
                self._logger.debug(f"Removed old checkpoint: {oldest.name}")
            except Exception as e:
                self._logger.warning(f"Failed to remove checkpoint: {e}")

    def clear_checkpoints(self, game_id: str | None = None) -> int:
        """Clear checkpoints.

        Args:
            game_id: Optional game ID to filter by.

        Returns:
            Number of checkpoints removed.
        """
        removed = 0
        to_remove = []

        for checkpoint in self._checkpoints:
            if game_id is None or game_id in checkpoint.name:
                to_remove.append(checkpoint)

        for checkpoint in to_remove:
            try:
                checkpoint.unlink()
                self._checkpoints.remove(checkpoint)
                removed += 1
            except Exception as e:
                self._logger.warning(f"Failed to remove checkpoint: {e}")

        return removed


# =============================================================================
# Game Backend Interface
# =============================================================================


class GameBackend(ABC):
    """Abstract interface for game backends."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to game.

        Returns:
            True if connection successful.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from game."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected.

        Returns:
            True if connected.
        """

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get current game state.

        Returns:
            Game state dictionary.
        """

    @abstractmethod
    def apply_action(self, action: Any) -> dict[str, Any]:
        """Apply action to game.

        Args:
            action: Action to apply.

        Returns:
            Result of action application.
        """

    @abstractmethod
    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset game to initial state.

        Args:
            options: Optional reset options.

        Returns:
            Initial game state.
        """

    @abstractmethod
    def get_valid_actions(self) -> Any:
        """Get valid actions for current state.

        Returns:
            Valid actions (format depends on backend).
        """


class MicroCivBackend(GameBackend):
    """Backend for MicroCiv environment."""

    def __init__(self, **env_kwargs: Any):
        """Initialize MicroCiv backend.

        Args:
            **env_kwargs: Arguments for MicroCiv environment.
        """
        self._env: Any | None = None
        self._env_kwargs = env_kwargs
        self._connected = False

    def connect(self) -> bool:
        """Connect to MicroCiv environment."""
        try:
            from alpha_zero.envs.microciv import MicroCivEnv
            self._env = MicroCivEnv(**self._env_kwargs)
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to create MicroCiv environment: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from environment."""
        self._env = None
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self._env is not None

    def get_state(self) -> dict[str, Any]:
        """Get current game state."""
        if not self.is_connected():
            raise StateRetrievalError("Not connected")
        assert self._env is not None  # Type narrowing for mypy
        return self._env._get_game_state()  # type: ignore[no-any-return]

    def apply_action(self, action: Any) -> dict[str, Any]:
        """Apply action to game."""
        if not self.is_connected():
            raise ActionExecutionError("Not connected", action)
        assert self._env is not None  # Type narrowing for mypy

        obs, reward, terminated, truncated, info = self._env.step(action)
        return {
            "observation": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset game to initial state."""
        if not self.is_connected():
            raise StateRetrievalError("Not connected")
        assert self._env is not None  # Type narrowing for mypy

        obs, info = self._env.reset(**options or {})
        return {"observation": obs, "info": info}

    def get_valid_actions(self) -> Any:
        """Get valid action mask."""
        if not self.is_connected():
            raise StateRetrievalError("Not connected")
        assert self._env is not None  # Type narrowing for mypy
        return self._env.get_valid_actions()


class UncivBackend(GameBackend):
    """Backend for Unciv interface."""

    def __init__(self, interface: Any | None = None, **kwargs: Any):
        """Initialize Unciv backend.

        Args:
            interface: UncivInterface instance.
            **kwargs: Additional arguments.
        """
        self._interface = interface
        self._state: dict[str, Any] | None = None
        self._connected = False
        self._kwargs = kwargs

    def connect(self) -> bool:
        """Connect to Unciv."""
        try:
            if self._interface is None:
                from alpha_zero.interfaces.unciv_interface import UncivInterface
                self._interface = UncivInterface(**self._kwargs)
            self._connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to create Unciv interface: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from Unciv."""
        self._interface = None
        self._state = None
        self._connected = False

    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected and self._interface is not None

    def get_state(self) -> dict[str, Any]:
        """Get current game state."""
        if not self.is_connected() or self._state is None:
            raise StateRetrievalError("No state available")
        return self._state

    def apply_action(self, action: Any) -> dict[str, Any]:
        """Apply action to game."""
        if not self.is_connected() or self._state is None:
            raise ActionExecutionError("Not connected", action)
        assert self._interface is not None  # Type narrowing for mypy

        try:
            new_state = self._interface.apply_action(self._state, action)
            self._state = new_state
            return {"state": new_state}
        except Exception as e:
            raise ActionExecutionError(str(e), action, e) from e

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset/load new game."""
        if not self.is_connected():
            raise StateRetrievalError("Not connected")
        assert self._interface is not None  # Type narrowing for mypy

        save_path = options.get("save_path") if options else None
        if save_path:
            game_state = self._interface.load_game(save_path)
            self._state = self._interface.convert_to_internal(game_state)
        else:
            # Generate empty state
            self._state = {}

        return {"state": self._state}

    def get_valid_actions(self) -> Any:
        """Get valid action mask."""
        if not self.is_connected() or self._state is None:
            raise StateRetrievalError("No state available")
        assert self._interface is not None  # Type narrowing for mypy
        return self._interface.get_valid_actions(self._state)


# =============================================================================
# Main Game Connector
# =============================================================================


class GameConnector:
    """Main game connector with error recovery.

    Provides a robust interface for connecting to game environments
    with automatic retry, health monitoring, and error recovery.

    Example:
        >>> config = ConnectionConfig.default()
        >>> connector = GameConnector(config, game_type=GameType.MICROCIV)
        >>> connector.connect()
        >>> state = connector.get_state()
        >>> result = connector.apply_action(action)
    """

    def __init__(
        self,
        config: ConnectionConfig,
        game_type: GameType = GameType.MICROCIV,
        backend: GameBackend | None = None,
        **backend_kwargs: Any,
    ):
        """Initialize game connector.

        Args:
            config: Connection configuration.
            game_type: Type of game to connect to.
            backend: Optional custom backend (created if None).
            **backend_kwargs: Arguments for backend creation.
        """
        self.config = config
        self.game_type = game_type
        self._backend = backend or self._create_backend(game_type, backend_kwargs)
        self._status = ConnectionStatus.DISCONNECTED
        self._latency = LatencyTracker(config.latency)
        self._metrics = ConnectionMetrics()
        self._checkpoint_manager = CheckpointManager(config.recovery)
        self._logger = logging.getLogger(f"{__name__}.GameConnector")
        self._start_time: float | None = None
        self._game_id: str = ""
        self._turn: int = 0
        self._callbacks: dict[str, list[Callable]] = {
            "on_connect": [],
            "on_disconnect": [],
            "on_error": [],
            "on_action": [],
            "on_recovery": [],
        }

    def _create_backend(
        self,
        game_type: GameType,
        kwargs: dict[str, Any],
    ) -> GameBackend:
        """Create appropriate backend for game type."""
        if game_type == GameType.MICROCIV:
            return MicroCivBackend(**kwargs)
        elif game_type == GameType.UNCIV:
            return UncivBackend(**kwargs)
        else:
            raise ValueError(f"Unsupported game type: {game_type}")

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def connect(self) -> bool:
        """Establish connection to game.

        Returns:
            True if connection successful.

        Raises:
            ConnectionTimeoutError: If connection times out.
            GameConnectionError: If connection fails after retries.
        """
        if self._status == ConnectionStatus.CONNECTED:
            return True

        self._status = ConnectionStatus.CONNECTING
        start_time = time.time()

        for attempt in range(self.config.retry.max_retries):
            try:
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.config.timeout.connect_timeout:
                    raise ConnectionTimeoutError()

                self._logger.info(
                    f"Connection attempt {attempt + 1}/{self.config.retry.max_retries}",
                )

                if self._backend.connect():
                    self._status = ConnectionStatus.CONNECTED
                    self._start_time = time.time()
                    self._game_id = f"game_{int(time.time() * 1000)}"
                    self._metrics.total_games += 1

                    # Notify callbacks
                    for callback in self._callbacks["on_connect"]:
                        callback()

                    self._logger.info("Connection established")
                    return True

            except Exception as e:
                self._logger.warning(f"Connection attempt failed: {e}")
                self._metrics.last_error = str(e)
                self._metrics.last_error_time = time.time()

            # Wait before retry
            if attempt < self.config.retry.max_retries - 1:
                delay = self.config.retry.get_delay(attempt)
                self._logger.debug(f"Waiting {delay:.1f}s before retry")
                time.sleep(delay)
                self._metrics.retries += 1

        self._status = ConnectionStatus.ERROR
        raise GameConnectionError(
            f"Failed to connect after {self.config.retry.max_retries} attempts",
            ErrorSeverity.CRITICAL,
        )

    def disconnect(self) -> None:
        """Disconnect from game."""
        if self._status == ConnectionStatus.DISCONNECTED:
            return

        try:
            self._backend.disconnect()
        except Exception as e:
            self._logger.warning(f"Error during disconnect: {e}")

        self._status = ConnectionStatus.DISCONNECTED

        # Update uptime
        if self._start_time:
            self._metrics.uptime_seconds += time.time() - self._start_time
            self._start_time = None

        # Notify callbacks
        for callback in self._callbacks["on_disconnect"]:
            callback()

        self._logger.info("Disconnected")

    def reconnect(self) -> bool:
        """Reconnect to game.

        Returns:
            True if reconnection successful.
        """
        self._status = ConnectionStatus.RECONNECTING
        self._metrics.reconnections += 1

        self.disconnect()
        return self.connect()

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._status == ConnectionStatus.CONNECTED and self._backend.is_connected()

    @property
    def status(self) -> ConnectionStatus:
        """Get connection status."""
        return self._status

    # -------------------------------------------------------------------------
    # Game Operations
    # -------------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Get current game state with retry.

        Returns:
            Game state dictionary.

        Raises:
            StateRetrievalError: If state retrieval fails.
        """
        return self._with_retry(
            self._get_state_impl,
            timeout=self.config.timeout.state_timeout,
            error_class=StateRetrievalError,
        )

    def _get_state_impl(self) -> dict[str, Any]:
        """Internal state retrieval."""
        start = time.perf_counter()
        state = self._backend.get_state()
        latency_ms = (time.perf_counter() - start) * 1000
        self._latency.record_sample(latency_ms)
        return state

    def apply_action(self, action: Any) -> dict[str, Any]:
        """Apply action to game with retry.

        Args:
            action: Action to apply.

        Returns:
            Result of action application.

        Raises:
            ActionExecutionError: If action execution fails.
        """
        self._metrics.total_actions += 1

        try:
            result = self._with_retry(
                lambda: self._apply_action_impl(action),
                timeout=self.config.timeout.action_timeout,
                error_class=ActionExecutionError,
            )
            self._metrics.successful_actions += 1
            self._turn += 1

            # Checkpoint if enabled
            if (
                self.config.recovery.checkpoint_enabled
                and self._turn % self.config.recovery.checkpoint_interval == 0
            ):
                state = self.get_state()
                self._checkpoint_manager.save_checkpoint(
                    state, self._game_id, self._turn,
                )
                self._metrics.checkpoints_created += 1

            # Notify callbacks
            for callback in self._callbacks["on_action"]:
                callback(action, result)

            return result

        except Exception as e:
            self._metrics.failed_actions += 1
            self._metrics.last_error = str(e)
            self._metrics.last_error_time = time.time()

            # Notify error callbacks
            for callback in self._callbacks["on_error"]:
                callback(e)

            # Attempt recovery
            if self.config.recovery.auto_recover:
                if self._attempt_recovery():
                    return {"recovered": True}

            raise

    def _apply_action_impl(self, action: Any) -> dict[str, Any]:
        """Internal action application."""
        start = time.perf_counter()
        result = self._backend.apply_action(action)
        latency_ms = (time.perf_counter() - start) * 1000
        self._latency.record_sample(latency_ms, is_action=True)
        return result

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset game with retry.

        Args:
            options: Optional reset options.

        Returns:
            Initial game state.
        """
        # Clear old checkpoints
        self._checkpoint_manager.clear_checkpoints(self._game_id)

        # Reset state
        self._game_id = f"game_{int(time.time() * 1000)}"
        self._turn = 0
        self._metrics.total_games += 1

        return self._with_retry(
            lambda: self._backend.reset(options),
            timeout=self.config.timeout.state_timeout,
            error_class=StateRetrievalError,
        )

    def get_valid_actions(self) -> Any:
        """Get valid actions for current state.

        Returns:
            Valid actions.
        """
        return self._with_retry(
            self._backend.get_valid_actions,
            timeout=self.config.timeout.state_timeout,
            error_class=StateRetrievalError,
        )

    # -------------------------------------------------------------------------
    # Retry Logic
    # -------------------------------------------------------------------------

    def _with_retry(
        self,
        operation: Callable[[], T],
        timeout: float,
        error_class: type,
    ) -> T:
        """Execute operation with retry logic.

        Args:
            operation: Operation to execute.
            timeout: Timeout in seconds.
            error_class: Exception class for errors.

        Returns:
            Operation result.

        Raises:
            error_class: If all retries fail.
        """
        start_time = time.time()
        last_error: Exception | None = None

        for attempt in range(self.config.retry.max_retries):
            # Check timeout
            if time.time() - start_time > timeout:
                raise error_class(
                    f"Operation timed out after {timeout}s",
                    last_error,
                )

            try:
                return operation()
            except Exception as e:
                last_error = e
                self._logger.warning(
                    f"Operation failed (attempt {attempt + 1}): {e}",
                )

                # Check if we should retry
                if attempt < self.config.retry.max_retries - 1:
                    delay = self.config.retry.get_delay(attempt)
                    time.sleep(delay)
                    self._metrics.retries += 1

                    # Attempt reconnect if needed
                    if not self._backend.is_connected():
                        self.reconnect()

        raise error_class(
            f"Operation failed after {self.config.retry.max_retries} attempts",
            last_error,
        )

    # -------------------------------------------------------------------------
    # Recovery
    # -------------------------------------------------------------------------

    def _attempt_recovery(self) -> bool:
        """Attempt to recover from error.

        Returns:
            True if recovery successful.
        """
        self._logger.info("Attempting recovery...")

        # Load latest checkpoint
        checkpoint = self._checkpoint_manager.load_latest_checkpoint(self._game_id)
        if checkpoint is None:
            self._logger.warning("No checkpoint available for recovery")
            return False

        try:
            # Reconnect
            self.reconnect()

            # Restore state
            self._turn = checkpoint.get("turn", 0)
            self._metrics.recoveries += 1

            # Notify callbacks
            for callback in self._callbacks["on_recovery"]:
                callback(checkpoint)

            self._logger.info(f"Recovered to turn {self._turn}")
            return True

        except Exception as e:
            self._logger.error(f"Recovery failed: {e}")
            return False

    # -------------------------------------------------------------------------
    # Metrics and Monitoring
    # -------------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Get connection metrics.

        Returns:
            Metrics dictionary.
        """
        # Update uptime
        if self._start_time:
            self._metrics.uptime_seconds = time.time() - self._start_time

        return {
            **self._metrics.to_dict(),
            "latency": self._latency.get_stats(),
            "status": self._status.name,
            "game_id": self._game_id,
            "turn": self._turn,
        }

    def get_latency_stats(self) -> dict[str, float]:
        """Get latency statistics.

        Returns:
            Latency statistics dictionary.
        """
        return self._latency.get_stats()

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def on_connect(self, callback: Callable[[], None]) -> None:
        """Register connection callback."""
        self._callbacks["on_connect"].append(callback)

    def on_disconnect(self, callback: Callable[[], None]) -> None:
        """Register disconnection callback."""
        self._callbacks["on_disconnect"].append(callback)

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register error callback."""
        self._callbacks["on_error"].append(callback)

    def on_action(self, callback: Callable[[Any, dict], None]) -> None:
        """Register action callback."""
        self._callbacks["on_action"].append(callback)

    def on_recovery(self, callback: Callable[[dict], None]) -> None:
        """Register recovery callback."""
        self._callbacks["on_recovery"].append(callback)


# =============================================================================
# Factory Functions
# =============================================================================


def create_game_connector(
    game_type: GameType = GameType.MICROCIV,
    preset: str = "default",
    **kwargs: Any,
) -> GameConnector:
    """Factory function to create game connector.

    Args:
        game_type: Type of game to connect to.
        preset: Configuration preset ('default', 'aggressive', 'low_latency').
        **kwargs: Additional backend arguments.

    Returns:
        GameConnector instance.
    """
    if preset == "aggressive":
        config = ConnectionConfig.aggressive_retry()
    elif preset == "low_latency":
        config = ConnectionConfig.low_latency()
    else:
        config = ConnectionConfig.default()

    return GameConnector(config, game_type, **kwargs)


def create_microciv_connector(
    board_size: int = 16,
    max_turns: int = 200,
    **kwargs: Any,
) -> GameConnector:
    """Factory function for MicroCiv connector.

    Args:
        board_size: Board size.
        max_turns: Maximum turns.
        **kwargs: Additional arguments.

    Returns:
        GameConnector for MicroCiv.
    """
    return create_game_connector(
        game_type=GameType.MICROCIV,
        board_size=board_size,
        max_turns=max_turns,
        **kwargs,
    )


def create_unciv_connector(
    interface: Any | None = None,
    **kwargs: Any,
) -> GameConnector:
    """Factory function for Unciv connector.

    Args:
        interface: UncivInterface instance.
        **kwargs: Additional arguments.

    Returns:
        GameConnector for Unciv.
    """
    return create_game_connector(
        game_type=GameType.UNCIV,
        interface=interface,
        **kwargs,
    )
