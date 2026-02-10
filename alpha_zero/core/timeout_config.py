"""Centralized timeout configuration.

This module provides TimeoutConfig which centralizes all timeout values
that were previously hardcoded across the codebase. This allows for:
- Consistent timeout handling
- Environment variable overrides
- Easy tuning for different deployment scenarios

Example:
    >>> from alpha_zero.core.timeout_config import TimeoutConfig
    >>>
    >>> # Use defaults
    >>> timeouts = TimeoutConfig()
    >>>
    >>> # Use fast timeouts for testing
    >>> timeouts = TimeoutConfig.fast()
    >>>
    >>> # Override via environment
    >>> import os
    >>> os.environ["CIV6AZ_TIMEOUT_STATE_TIMEOUT"] = "120.0"
    >>> timeouts = TimeoutConfig().apply_env_overrides()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from alpha_zero.core.config import ConfigMixin


@dataclass
class TimeoutConfig(ConfigMixin):
    """Centralized timeout configuration for all components.

    This config consolidates timeout values from:
    - actor.py (trajectory submission, weight updates)
    - coordinator.py (actor pool management, monitoring)
    - learner.py (buffer polling, training steps)
    - real_civ6_env.py (state/action timeouts)
    - planning_agent.py (agent planning timeout)

    Environment variables use prefix: CIV6AZ_TIMEOUT_
    Example: CIV6AZ_TIMEOUT_STATE_TIMEOUT=120.0

    Attributes:
        trajectory_submission_timeout: Timeout for submitting trajectories to buffer.
        weight_update_interval: Interval between weight sync checks.
        actor_error_retry_delay: Delay before retrying after actor error.
        actor_pool_stop_timeout: Timeout for graceful actor pool shutdown.
        monitor_interval: Interval between coordinator monitoring cycles.
        buffer_wait_interval: Delay when waiting for buffer to fill.
        buffer_ready_poll_interval: Interval for checking if buffer is ready.
        training_step_retry_delay: Delay between training step retries.
        state_timeout: Timeout for reading game state from environment.
        action_timeout: Timeout for executing actions in environment.
        command_timeout_ms: Timeout for bridge commands in milliseconds.
        heartbeat_timeout_s: Timeout for heartbeat checks.
        planning_timeout: Timeout for agent planning operations.
        implementation_timeout: Timeout for agent implementation operations.
        testing_timeout: Timeout for agent testing operations.
        verification_timeout: Timeout for agent verification operations.
    """

    ENV_PREFIX: ClassVar[str] = "CIV6AZ_TIMEOUT_"

    # Actor timeouts (from actor.py)
    trajectory_submission_timeout: float = 5.0
    weight_update_interval: float = 1.0
    actor_error_retry_delay: float = 1.0

    # Coordinator timeouts (from coordinator.py)
    actor_pool_stop_timeout: float = 10.0
    monitor_interval: float = 10.0
    buffer_wait_interval: float = 1.0

    # Learner timeouts (from learner.py)
    buffer_ready_poll_interval: float = 1.0
    training_step_retry_delay: float = 0.1

    # Environment timeouts (from real_civ6_env.py)
    state_timeout: float = 60.0
    action_timeout: float = 30.0

    # Bridge timeouts (mixed units)
    command_timeout_ms: int = 5000
    heartbeat_timeout_s: float = 2.0

    # Agent timeouts (for future LLM-based agent operations)
    planning_timeout: float = 300.0
    implementation_timeout: float = 600.0
    testing_timeout: float = 600.0
    verification_timeout: float = 300.0

    @classmethod
    def defaults(cls) -> TimeoutConfig:
        """Return default timeout configuration.

        Returns:
            TimeoutConfig with default values.
        """
        return cls()

    @classmethod
    def fast(cls) -> TimeoutConfig:
        """Return fast timeouts for testing environments.

        Reduces all timeouts for faster test execution.

        Returns:
            TimeoutConfig with reduced timeout values.
        """
        return cls(
            trajectory_submission_timeout=1.0,
            weight_update_interval=0.5,
            actor_error_retry_delay=0.1,
            actor_pool_stop_timeout=2.0,
            monitor_interval=2.0,
            buffer_wait_interval=0.1,
            buffer_ready_poll_interval=0.1,
            training_step_retry_delay=0.01,
            state_timeout=10.0,
            action_timeout=5.0,
            command_timeout_ms=1000,
            heartbeat_timeout_s=0.5,
            planning_timeout=30.0,
            implementation_timeout=60.0,
            testing_timeout=60.0,
            verification_timeout=30.0,
        )

    @classmethod
    def production(cls) -> TimeoutConfig:
        """Return production timeouts with conservative values.

        Uses longer timeouts suitable for production deployments
        where stability is more important than speed.

        Returns:
            TimeoutConfig with production-safe timeout values.
        """
        return cls(
            trajectory_submission_timeout=10.0,
            weight_update_interval=2.0,
            actor_error_retry_delay=5.0,
            actor_pool_stop_timeout=30.0,
            monitor_interval=30.0,
            buffer_wait_interval=5.0,
            buffer_ready_poll_interval=5.0,
            training_step_retry_delay=1.0,
            state_timeout=120.0,
            action_timeout=60.0,
            command_timeout_ms=10000,
            heartbeat_timeout_s=5.0,
            planning_timeout=600.0,
            implementation_timeout=1200.0,
            testing_timeout=1200.0,
            verification_timeout=600.0,
        )


# Global default instance for convenience
_default_timeouts: TimeoutConfig | None = None


def get_timeout_config() -> TimeoutConfig:
    """Get the global timeout configuration.

    Creates a default instance on first call, then returns the same instance.
    The instance is initialized with environment variable overrides applied.

    Returns:
        The global TimeoutConfig instance.
    """
    global _default_timeouts
    if _default_timeouts is None:
        _default_timeouts = TimeoutConfig().apply_env_overrides()
    return _default_timeouts


def set_timeout_config(config: TimeoutConfig) -> None:
    """Set the global timeout configuration.

    Args:
        config: TimeoutConfig instance to use globally.
    """
    global _default_timeouts
    _default_timeouts = config
