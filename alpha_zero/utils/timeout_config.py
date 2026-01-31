"""Centralized timeout configuration for all Civ6-AlphaZero components.

This module provides a unified TimeoutConfig dataclass that centralizes
all timeout values used throughout the codebase, enabling consistent
configuration and environment variable overrides.

Usage:
    from alpha_zero.utils.timeout_config import TimeoutConfig

    # Create with defaults
    config = TimeoutConfig()

    # Create from environment variables
    config = TimeoutConfig.from_env()

    # Create from dictionary
    config = TimeoutConfig.from_dict({"trajectory_submission": 10.0})
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class TimeoutConfig:
    """Centralized timeout configuration for all components.

    This dataclass consolidates timeout values that were previously
    scattered across multiple files. All timeouts are in seconds unless
    otherwise noted.

    Attributes:
        trajectory_submission: Timeout for submitting trajectories to storage.
        process_join: Timeout for joining worker processes.
        actor_pool_stop: Timeout for stopping the actor pool.
        state_reading: Timeout for reading game state from Civ6.
        action_execution: Timeout for executing an action in the game.
        game_action: Timeout for a single game action API call.
        game_state: Timeout for a single game state API call.
        agent_planning: Timeout for agentic framework planning phase.
        agent_implementation: Timeout for agentic framework implementation phase.
        agent_testing: Timeout for agentic framework testing phase.
        agent_verification: Timeout for agentic framework verification phase.
        agent_documentation: Timeout for agentic framework documentation phase.
        exploration_timeout_minutes: Timeout for exploration phase (in minutes).
    """

    # Distributed training timeouts
    trajectory_submission: float = 5.0
    process_join: float = 1.0
    actor_pool_stop: float = 10.0

    # Environment interaction timeouts
    state_reading: float = 60.0
    action_execution: float = 30.0
    game_action: float = 5.0
    game_state: float = 2.0

    # Agentic framework timeouts
    agent_planning: float = 300.0
    agent_implementation: float = 600.0
    agent_testing: float = 600.0
    agent_verification: float = 300.0
    agent_documentation: float = 300.0
    exploration_timeout_minutes: float = 10.0

    @classmethod
    def from_env(cls, prefix: str = "CIV6AZ_TIMEOUT_") -> TimeoutConfig:
        """Create TimeoutConfig from environment variables.

        Environment variables are looked up with the given prefix followed
        by the uppercase field name. For example:
            CIV6AZ_TIMEOUT_TRAJECTORY_SUBMISSION=10.0

        Args:
            prefix: Environment variable prefix. Defaults to "CIV6AZ_TIMEOUT_".

        Returns:
            TimeoutConfig instance with values from environment or defaults.
        """
        config = cls()
        field_names = [
            "trajectory_submission",
            "process_join",
            "actor_pool_stop",
            "state_reading",
            "action_execution",
            "game_action",
            "game_state",
            "agent_planning",
            "agent_implementation",
            "agent_testing",
            "agent_verification",
            "agent_documentation",
            "exploration_timeout_minutes",
        ]

        for field_name in field_names:
            env_name = f"{prefix}{field_name.upper()}"
            env_value = os.getenv(env_name)
            if env_value is not None:
                try:
                    setattr(config, field_name, float(env_value))
                except ValueError:
                    pass  # Keep default if conversion fails

        return config

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimeoutConfig:
        """Create TimeoutConfig from a dictionary.

        Args:
            data: Dictionary with timeout values. Keys should match field names.

        Returns:
            TimeoutConfig instance with values from dictionary or defaults.
        """
        valid_fields = {
            "trajectory_submission",
            "process_join",
            "actor_pool_stop",
            "state_reading",
            "action_execution",
            "game_action",
            "game_state",
            "agent_planning",
            "agent_implementation",
            "agent_testing",
            "agent_verification",
            "agent_documentation",
            "exploration_timeout_minutes",
        }

        filtered = {k: float(v) for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> dict[str, float]:
        """Convert TimeoutConfig to a dictionary.

        Returns:
            Dictionary with all timeout values.
        """
        return {
            "trajectory_submission": self.trajectory_submission,
            "process_join": self.process_join,
            "actor_pool_stop": self.actor_pool_stop,
            "state_reading": self.state_reading,
            "action_execution": self.action_execution,
            "game_action": self.game_action,
            "game_state": self.game_state,
            "agent_planning": self.agent_planning,
            "agent_implementation": self.agent_implementation,
            "agent_testing": self.agent_testing,
            "agent_verification": self.agent_verification,
            "agent_documentation": self.agent_documentation,
            "exploration_timeout_minutes": self.exploration_timeout_minutes,
        }


# Default singleton instance
_default_config: TimeoutConfig | None = None


def get_timeout_config() -> TimeoutConfig:
    """Get the default TimeoutConfig instance.

    This function returns a singleton instance that is initialized from
    environment variables on first call.

    Returns:
        The default TimeoutConfig instance.
    """
    global _default_config
    if _default_config is None:
        _default_config = TimeoutConfig.from_env()
    return _default_config
