"""Tests for alpha_zero.exceptions module.

Validates the exception hierarchy and the Civ6AlphaZeroError base class
context handling. All custom exceptions are subclasses of
Civ6AlphaZeroError so a single except clause can catch them all.
"""

from __future__ import annotations

import pytest

from alpha_zero.exceptions import (
    ActionTimeoutError,
    ActorError,
    BridgeConnectionError,
    BridgeError,
    Civ6AlphaZeroError,
    ConfigurationError,
    DistributedError,
    EnvironmentError,
    GradientError,
    InvalidActionError,
    LearnerError,
    MCTSError,
    MCTSSearchError,
    MCTSTimeoutError,
    NetworkError,
    NetworkInferenceError,
    NetworkLoadError,
    ReplayBufferError,
    StateEncodingError,
    StateTimeoutError,
    TrainingError,
    WeightSyncError,
)


class TestBaseException:
    """Tests for the Civ6AlphaZeroError base class."""

    def test_simple_message(self) -> None:
        err = Civ6AlphaZeroError("boom")
        assert err.message == "boom"
        assert err.context == {}
        assert str(err) == "boom"

    def test_with_context(self) -> None:
        err = Civ6AlphaZeroError("boom", context={"key": "value", "n": 7})
        assert err.context == {"key": "value", "n": 7}
        rendered = str(err)
        assert "boom" in rendered
        assert "key=value" in rendered
        assert "n=7" in rendered

    def test_none_context_normalized_to_empty_dict(self) -> None:
        err = Civ6AlphaZeroError("boom", context=None)
        assert err.context == {}

    def test_inherits_from_builtin_exception(self) -> None:
        assert issubclass(Civ6AlphaZeroError, Exception)

    def test_is_raisable(self) -> None:
        with pytest.raises(Civ6AlphaZeroError, match="boom"):
            raise Civ6AlphaZeroError("boom")


class TestExceptionHierarchy:
    """Every project exception inherits from Civ6AlphaZeroError so callers
    can catch them all with one except clause."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationError,
            NetworkError,
            NetworkInferenceError,
            NetworkLoadError,
            MCTSError,
            MCTSSearchError,
            MCTSTimeoutError,
            EnvironmentError,
            InvalidActionError,
            StateEncodingError,
            BridgeError,
            StateTimeoutError,
            ActionTimeoutError,
            BridgeConnectionError,
            TrainingError,
            ReplayBufferError,
            GradientError,
            DistributedError,
            ActorError,
            LearnerError,
            WeightSyncError,
        ],
    )
    def test_inherits_from_base(self, exc_cls: type) -> None:
        assert issubclass(exc_cls, Civ6AlphaZeroError)

    def test_network_hierarchy(self) -> None:
        assert issubclass(NetworkInferenceError, NetworkError)
        assert issubclass(NetworkLoadError, NetworkError)

    def test_mcts_hierarchy(self) -> None:
        assert issubclass(MCTSSearchError, MCTSError)
        assert issubclass(MCTSTimeoutError, MCTSError)

    def test_environment_hierarchy(self) -> None:
        assert issubclass(InvalidActionError, EnvironmentError)
        assert issubclass(StateEncodingError, EnvironmentError)

    def test_bridge_hierarchy(self) -> None:
        assert issubclass(StateTimeoutError, BridgeError)
        assert issubclass(ActionTimeoutError, BridgeError)
        assert issubclass(BridgeConnectionError, BridgeError)

    def test_training_hierarchy(self) -> None:
        assert issubclass(ReplayBufferError, TrainingError)
        assert issubclass(GradientError, TrainingError)

    def test_distributed_hierarchy(self) -> None:
        assert issubclass(ActorError, DistributedError)
        assert issubclass(LearnerError, DistributedError)
        assert issubclass(WeightSyncError, DistributedError)

    def test_catch_all_via_base(self) -> None:
        """Single except Civ6AlphaZeroError catches every subclass."""
        for cls in (
            NetworkInferenceError,
            MCTSTimeoutError,
            InvalidActionError,
            ActionTimeoutError,
            ReplayBufferError,
            ActorError,
        ):
            with pytest.raises(Civ6AlphaZeroError):
                raise cls("simulated")


class TestExceptionContext:
    """Subclasses inherit the context-handling protocol from the base."""

    def test_subclass_with_context(self) -> None:
        err = MCTSTimeoutError("search timed out", context={"simulations": 100})
        assert err.message == "search timed out"
        assert err.context["simulations"] == 100
        assert "simulations=100" in str(err)

    def test_environment_error_does_not_shadow_builtin(self) -> None:
        """alpha_zero.exceptions.EnvironmentError is its own type even though
        Python has a builtin EnvironmentError (alias for OSError)."""
        import builtins

        # Project-specific EnvironmentError is distinct from the builtin
        # (which on modern Pythons is just an alias for OSError).
        assert EnvironmentError is not builtins.OSError
        assert issubclass(EnvironmentError, Civ6AlphaZeroError)

    def test_chaining_with_from_clause(self) -> None:
        try:
            try:
                raise ValueError("root cause")
            except ValueError as e:
                raise NetworkLoadError("could not load checkpoint") from e
        except NetworkLoadError as e:
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "root cause"
