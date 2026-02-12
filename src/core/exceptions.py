"""Exception hierarchy for the Strategic Planning SDK.

All SDK exceptions inherit from :class:`SDKError`, enabling callers to catch
broad or narrow failure categories as needed::

    try:
        env.step(action)
    except InvalidActionError as exc:
        print(f"Bad action {exc.action}, pick from {exc.valid_actions}")
    except EnvironmentError:
        ...  # any env-level problem
    except SDKError:
        ...  # any SDK problem at all
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "CheckpointError",
    # Config
    "ConfigError",
    "ConfigFileNotFoundError",
    "ConfigValidationError",
    "DuplicateRegistrationError",
    "EnvironmentDeadlockError",
    # Environment
    "EnvironmentError",
    "EnvironmentNotRegisteredError",
    "InsufficientDataError",
    "InvalidActionError",
    "NetworkDivergenceError",
    "NoValidActionsError",
    # Plugin
    "PluginError",
    "ProtocolViolationError",
    "ReplayBufferError",
    # Base
    "SDKError",
    # Search
    "SearchError",
    "SearchTimeoutError",
    "SerializationError",
    # Training
    "TrainingError",
]


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class SDKError(Exception):
    """Base exception for all Strategic Planning SDK errors.

    Every exception raised by the SDK is a subclass of ``SDKError``, so
    callers can use a single ``except SDKError`` clause to handle any
    SDK-originated failure.

    Args:
        message: Human-readable description of the error.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class ConfigError(SDKError):
    """Raised when configuration validation fails.

    This covers any problem with loading, parsing, or validating
    configuration objects (YAML files, environment variables, dataclass
    defaults, etc.).

    Args:
        message: Human-readable description of the configuration problem.
    """


class ConfigFileNotFoundError(ConfigError):
    """Raised when a referenced YAML configuration file does not exist.

    Args:
        message: Human-readable description of the error.
        path: Filesystem path that was not found.
    """

    def __init__(self, message: str, path: str | Path) -> None:
        self.path = Path(path)
        super().__init__(message)


class ConfigValidationError(ConfigError):
    """Raised when a configuration object fails Pydantic validation.

    This typically wraps a ``pydantic.ValidationError`` with additional
    SDK context so that callers do not need to depend on Pydantic
    directly.

    Args:
        message: Human-readable description of the validation failure.
    """


# ---------------------------------------------------------------------------
# Environment errors
# ---------------------------------------------------------------------------


class EnvironmentError(SDKError):
    """Raised for any environment-related failure.

    Covers invalid actions, missing registrations, deadlocks, and
    serialization problems that occur when interacting with game
    environments.

    Args:
        message: Human-readable description of the environment problem.
    """


class InvalidActionError(EnvironmentError):
    """Raised when an action is not in the set of currently valid actions.

    Args:
        message: Human-readable description of the error.
        action: The invalid action that was attempted.
        valid_actions: The set of actions that would have been accepted.
    """

    def __init__(
        self,
        message: str,
        action: Any,
        valid_actions: Sequence[Any],
    ) -> None:
        self.action = action
        self.valid_actions = valid_actions
        super().__init__(message)


class EnvironmentNotRegisteredError(EnvironmentError):
    """Raised when an unknown environment name is requested.

    Args:
        message: Human-readable description of the error.
        name: The environment name that was not found in the registry.
        available: List of environment names that *are* registered.
    """

    def __init__(
        self,
        message: str,
        name: str,
        available: Sequence[str],
    ) -> None:
        self.name = name
        self.available = list(available)
        super().__init__(message)


class EnvironmentDeadlockError(EnvironmentError):
    """Raised when an environment step exceeds its timeout budget.

    This usually indicates an infinite loop or blocking call inside the
    environment's ``step`` implementation.

    Args:
        message: Human-readable description of the deadlock condition.
    """


class SerializationError(EnvironmentError):
    """Raised when environment state serialization or deserialization fails.

    Covers failures in ``serialize_state`` / ``deserialize_state`` such as
    incompatible formats, corrupt data, or schema mismatches.

    Args:
        message: Human-readable description of the serialization failure.
    """


# ---------------------------------------------------------------------------
# Search errors
# ---------------------------------------------------------------------------


class SearchError(SDKError):
    """Raised for failures during MCTS or other tree-search algorithms.

    Args:
        message: Human-readable description of the search failure.
    """


class NoValidActionsError(SearchError):
    """Raised when the search encounters a state with no valid actions.

    This can happen at a terminal state (which should not be expanded) or
    when the action-masking logic is broken.

    Args:
        message: Human-readable description of the error.
    """


class SearchTimeoutError(SearchError):
    """Raised when a search exceeds its allotted time budget.

    Args:
        message: Human-readable description of the timeout.
    """


# ---------------------------------------------------------------------------
# Training errors
# ---------------------------------------------------------------------------


class TrainingError(SDKError):
    """Raised for failures during the training loop.

    Covers replay-buffer issues, checkpoint I/O, divergence detection, and
    insufficient data conditions.

    Args:
        message: Human-readable description of the training failure.
    """


class ReplayBufferError(TrainingError):
    """Raised when the replay buffer is corrupted or overflows.

    Args:
        message: Human-readable description of the buffer problem.
    """


class CheckpointError(TrainingError):
    """Raised when saving or loading a model checkpoint fails.

    Args:
        message: Human-readable description of the checkpoint failure.
    """


class NetworkDivergenceError(TrainingError):
    """Raised when a loss spike is detected during training.

    Args:
        message: Human-readable description of the divergence event.
        loss_value: The observed loss value that triggered the error.
        threshold: The divergence threshold that was exceeded.
    """

    def __init__(
        self,
        message: str,
        loss_value: float,
        threshold: float,
    ) -> None:
        self.loss_value = loss_value
        self.threshold = threshold
        super().__init__(message)


class InsufficientDataError(TrainingError):
    """Raised when the replay buffer is too small to form a training batch.

    Args:
        message: Human-readable description of the data shortage.
    """


# ---------------------------------------------------------------------------
# Plugin errors
# ---------------------------------------------------------------------------


class PluginError(SDKError):
    """Raised for failures in the plugin / registration system.

    Covers protocol violations (an environment does not satisfy the
    required protocol) and duplicate registrations.

    Args:
        message: Human-readable description of the plugin failure.
    """


class ProtocolViolationError(PluginError):
    """Raised when a registered environment does not satisfy its protocol.

    Args:
        message: Human-readable description of the violation.
        env_class: The environment class that failed protocol checks.
        missing_methods: Names of required methods that are absent.
    """

    def __init__(
        self,
        message: str,
        env_class: type,
        missing_methods: Sequence[str],
    ) -> None:
        self.env_class = env_class
        self.missing_methods = list(missing_methods)
        super().__init__(message)


class DuplicateRegistrationError(PluginError):
    """Raised when a plugin name that is already registered is used again.

    Args:
        message: Human-readable description of the conflict.
    """
