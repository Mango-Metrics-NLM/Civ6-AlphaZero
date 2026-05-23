"""Custom exceptions for Civ6-AlphaZero.

This module provides a hierarchy of exceptions for better error handling
and more informative error messages throughout the codebase.

Exception Hierarchy:
    Civ6AlphaZeroError (base)
    ├── ConfigurationError
    ├── NetworkError
    │   ├── NetworkInferenceError
    │   └── NetworkLoadError
    ├── MCTSError
    │   ├── MCTSSearchError
    │   └── MCTSTimeoutError
    ├── EnvironmentError
    │   ├── InvalidActionError
    │   └── StateEncodingError
    ├── BridgeError
    │   ├── StateTimeoutError
    │   ├── ActionTimeoutError
    │   └── ConnectionError
    ├── TrainingError
    │   ├── ReplayBufferError
    │   └── GradientError
    └── DistributedError
        ├── ActorError
        ├── LearnerError
        └── WeightSyncError
"""

from __future__ import annotations

from typing import Any


class Civ6AlphaZeroError(Exception):
    """Base exception for all Civ6-AlphaZero errors.

    All custom exceptions in the project should inherit from this class.
    This allows catching all project-specific exceptions with a single
    except clause.

    Attributes:
        message: Human-readable error message.
        context: Optional dictionary with additional context information.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Initialize exception.

        Args:
            message: Human-readable error message.
            context: Optional dictionary with additional context information.
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{context_str}]"
        return self.message


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(Civ6AlphaZeroError):
    """Invalid configuration error.

    Raised when configuration values are invalid, missing, or incompatible.
    """

    pass


# =============================================================================
# Network Errors
# =============================================================================


class NetworkError(Civ6AlphaZeroError):
    """Neural network related error.

    Base class for all network-related errors.
    """

    pass


class NetworkInferenceError(NetworkError):
    """Error during network inference.

    Raised when forward pass fails, produces NaN/Inf, or has shape mismatches.
    """

    pass


class NetworkLoadError(NetworkError):
    """Error loading network weights.

    Raised when checkpoint loading fails or weights are incompatible.
    """

    pass


# =============================================================================
# MCTS Errors
# =============================================================================


class MCTSError(Civ6AlphaZeroError):
    """MCTS search error.

    Base class for all MCTS-related errors.
    """

    pass


class MCTSSearchError(MCTSError):
    """Error during MCTS search.

    Raised when search produces invalid results or encounters numerical issues.
    """

    pass


class MCTSTimeoutError(MCTSError):
    """MCTS search timeout.

    Raised when search exceeds time budget.
    """

    pass


# =============================================================================
# Environment Errors
# =============================================================================


class EnvironmentError(Civ6AlphaZeroError):
    """Environment-related error.

    Base class for all environment-related errors.
    Note: This shadows the built-in EnvironmentError, but we use a different
    namespace (alpha_zero.exceptions.EnvironmentError).
    """

    pass


class InvalidActionError(EnvironmentError):
    """Invalid action error.

    Raised when an action is invalid for the current state.
    """

    def __init__(
        self,
        message: str,
        action: int | None = None,
        valid_actions: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if action is not None:
            context["action"] = action
        if valid_actions is not None:
            context["valid_actions_count"] = len(valid_actions)
        super().__init__(message, context=context)


class StateEncodingError(EnvironmentError):
    """State encoding error.

    Raised when state cannot be encoded to tensor format.
    """

    pass


# =============================================================================
# Bridge Errors
# =============================================================================


class BridgeError(Civ6AlphaZeroError):
    """Game bridge communication error.

    Base class for all bridge-related errors.
    """

    pass


class StateTimeoutError(BridgeError):
    """Timeout waiting for game state.

    Raised when the game doesn't respond with state in time.
    """

    def __init__(
        self,
        message: str = "Timeout waiting for game state",
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if timeout is not None:
            context["timeout_seconds"] = timeout
        super().__init__(message, context=context)


class ActionTimeoutError(BridgeError):
    """Timeout waiting for action acknowledgment.

    Raised when the game doesn't acknowledge an action in time.
    """

    def __init__(
        self,
        message: str = "Timeout waiting for action acknowledgment",
        timeout: float | None = None,
        action: int | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if timeout is not None:
            context["timeout_seconds"] = timeout
        if action is not None:
            context["action"] = action
        super().__init__(message, context=context)


class BridgeConnectionError(BridgeError):
    """Bridge connection error.

    Raised when connection to the game is lost or cannot be established.
    """

    pass


# =============================================================================
# Training Errors
# =============================================================================


class TrainingError(Civ6AlphaZeroError):
    """Training-related error.

    Base class for all training-related errors.
    """

    pass


class ReplayBufferError(TrainingError):
    """Replay buffer error.

    Raised when buffer operations fail (e.g., sampling from empty buffer).
    """

    pass


class GradientError(TrainingError):
    """Gradient computation error.

    Raised when gradients are NaN/Inf or explode.
    """

    def __init__(
        self,
        message: str = "Gradient computation error",
        gradient_norm: float | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if gradient_norm is not None:
            context["gradient_norm"] = gradient_norm
        super().__init__(message, context=context)


# =============================================================================
# Distributed Training Errors
# =============================================================================


class DistributedError(Civ6AlphaZeroError):
    """Distributed training error.

    Base class for all distributed training errors.
    """

    pass


class ActorError(DistributedError):
    """Actor process error.

    Raised when an actor process fails or becomes unresponsive.
    """

    def __init__(
        self,
        message: str,
        actor_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if actor_id is not None:
            context["actor_id"] = actor_id
        super().__init__(message, context=context)


class LearnerError(DistributedError):
    """Learner process error.

    Raised when the learner process fails.
    """

    pass


class WeightSyncError(DistributedError):
    """Weight synchronization error.

    Raised when weight synchronization between processes fails.
    """

    def __init__(
        self,
        message: str = "Weight synchronization failed",
        version: int | None = None,
        **kwargs: Any,
    ) -> None:
        context = kwargs.pop("context", {})
        if version is not None:
            context["version"] = version
        super().__init__(message, context=context)


# =============================================================================
# Utility Functions
# =============================================================================


def format_exception_chain(exc: Exception) -> str:
    """Format exception with its cause chain.

    Args:
        exc: Exception to format.

    Returns:
        Formatted string showing exception chain.
    """
    messages = []
    current: Exception | None = exc

    while current is not None:
        if isinstance(current, Civ6AlphaZeroError):
            messages.append(str(current))
        else:
            messages.append(f"{type(current).__name__}: {current}")
        current = current.__cause__  # type: ignore[assignment]

    return " -> ".join(messages)
