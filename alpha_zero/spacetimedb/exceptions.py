"""SpacetimeDB exception hierarchy.

Provides a structured exception hierarchy for SpacetimeDB operations,
following the pattern established by the LLM client module.
"""

from __future__ import annotations


class SpacetimeDBError(Exception):
    """Base exception for all SpacetimeDB errors."""


class SpacetimeDBConnectionError(SpacetimeDBError):
    """Raised when connection to SpacetimeDB fails."""


class SpacetimeDBTimeoutError(SpacetimeDBError):
    """Raised when a SpacetimeDB operation times out."""


class SpacetimeDBReducerError(SpacetimeDBError):
    """Raised when a reducer call fails."""

    def __init__(self, reducer_name: str, message: str) -> None:
        self.reducer_name = reducer_name
        super().__init__(f"Reducer '{reducer_name}' failed: {message}")


class SpacetimeDBSubscriptionError(SpacetimeDBError):
    """Raised when a subscription operation fails."""


class SpacetimeDBSerializationError(SpacetimeDBError):
    """Raised when serialization/deserialization of data fails."""


class SpacetimeDBUnavailableError(SpacetimeDBError):
    """Raised when SpacetimeDB SDK is not installed or server unreachable."""
