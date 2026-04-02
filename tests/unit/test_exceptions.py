"""Unit tests for alpha_zero.exceptions module.

Covers the full exception hierarchy, context handling, custom __init__ methods,
and the format_exception_chain utility function.
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
    format_exception_chain,
)


class TestCiv6AlphaZeroError:
    """Tests for the base Civ6AlphaZeroError exception."""

    def test_basic_message(self) -> None:
        exc = Civ6AlphaZeroError("something went wrong")
        assert exc.message == "something went wrong"
        assert str(exc) == "something went wrong"

    def test_context_is_empty_dict_by_default(self) -> None:
        exc = Civ6AlphaZeroError("msg")
        assert exc.context == {}

    def test_context_none_gives_empty_dict(self) -> None:
        exc = Civ6AlphaZeroError("msg", context=None)
        assert exc.context == {}

    def test_context_included_in_str(self) -> None:
        exc = Civ6AlphaZeroError("msg", context={"key": "val"})
        result = str(exc)
        assert "msg" in result
        assert "key=val" in result

    def test_multiple_context_keys(self) -> None:
        exc = Civ6AlphaZeroError("msg", context={"a": 1, "b": 2})
        result = str(exc)
        assert "a=1" in result
        assert "b=2" in result

    def test_is_subclass_of_exception(self) -> None:
        exc = Civ6AlphaZeroError("msg")
        assert isinstance(exc, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(Civ6AlphaZeroError, match="test error"):
            raise Civ6AlphaZeroError("test error")

    def test_can_be_caught_as_exception(self) -> None:
        with pytest.raises(Exception):
            raise Civ6AlphaZeroError("test error")


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_is_subclass_of_base(self) -> None:
        exc = ConfigurationError("bad config")
        assert isinstance(exc, Civ6AlphaZeroError)

    def test_message(self) -> None:
        exc = ConfigurationError("invalid value for batch_size")
        assert str(exc) == "invalid value for batch_size"

    def test_with_context(self) -> None:
        exc = ConfigurationError("bad param", context={"param": "lr", "value": -1})
        assert "bad param" in str(exc)
        assert "param=lr" in str(exc)

    def test_can_be_raised(self) -> None:
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("missing required field")

    def test_caught_as_base(self) -> None:
        with pytest.raises(Civ6AlphaZeroError):
            raise ConfigurationError("bad")


class TestNetworkErrors:
    """Tests for NetworkError hierarchy."""

    def test_network_error_is_base(self) -> None:
        exc = NetworkError("network problem")
        assert isinstance(exc, Civ6AlphaZeroError)

    def test_inference_error_is_network_error(self) -> None:
        exc = NetworkInferenceError("NaN in output")
        assert isinstance(exc, NetworkError)
        assert isinstance(exc, Civ6AlphaZeroError)

    def test_load_error_is_network_error(self) -> None:
        exc = NetworkLoadError("checkpoint not found")
        assert isinstance(exc, NetworkError)
        assert isinstance(exc, Civ6AlphaZeroError)

    def test_inference_error_message(self) -> None:
        exc = NetworkInferenceError("shape mismatch: expected (1,13,16,16)")
        assert "shape mismatch" in str(exc)

    def test_load_error_with_context(self) -> None:
        exc = NetworkLoadError("weights incompatible", context={"path": "/ckpt/v1.pt"})
        assert "path=/ckpt/v1.pt" in str(exc)

    def test_inference_error_caught_as_network_error(self) -> None:
        with pytest.raises(NetworkError):
            raise NetworkInferenceError("inf in logits")

    def test_load_error_caught_as_base(self) -> None:
        with pytest.raises(Civ6AlphaZeroError):
            raise NetworkLoadError("failed")


class TestMCTSErrors:
    """Tests for MCTSError hierarchy."""

    def test_mcts_error_is_base(self) -> None:
        exc = MCTSError("mcts problem")
        assert isinstance(exc, Civ6AlphaZeroError)

    def test_search_error_is_mcts_error(self) -> None:
        exc = MCTSSearchError("no valid children")
        assert isinstance(exc, MCTSError)

    def test_timeout_error_is_mcts_error(self) -> None:
        exc = MCTSTimeoutError("exceeded time budget")
        assert isinstance(exc, MCTSError)

    def test_search_error_message(self) -> None:
        exc = MCTSSearchError("numerical instability at depth 5")
        assert "numerical instability" in str(exc)

    def test_timeout_error_with_context(self) -> None:
        exc = MCTSTimeoutError("timed out", context={"budget_ms": 100, "elapsed_ms": 200})
        assert "budget_ms=100" in str(exc)

    def test_search_error_caught_as_mcts_error(self) -> None:
        with pytest.raises(MCTSError):
            raise MCTSSearchError("bad search")

    def test_timeout_caught_as_base(self) -> None:
        with pytest.raises(Civ6AlphaZeroError):
            raise MCTSTimeoutError("timeout")


class TestEnvironmentErrors:
    """Tests for EnvironmentError hierarchy."""

    def test_environment_error_is_base(self) -> None:
        exc = EnvironmentError("env problem")
        assert isinstance(exc, Civ6AlphaZeroError)

    def test_state_encoding_error_is_environment_error(self) -> None:
        exc = StateEncodingError("cannot encode state")
        assert isinstance(exc, EnvironmentError)

    def test_state_encoding_with_context(self) -> None:
        exc = StateEncodingError("bad state", context={"turn": 42})
        assert "turn=42" in str(exc)


class TestInvalidActionError:
    """Tests for InvalidActionError."""

    def test_is_environment_error(self) -> None:
        exc = InvalidActionError("action out of range")
        assert isinstance(exc, EnvironmentError)

    def test_message_only(self) -> None:
        exc = InvalidActionError("invalid action")
        assert exc.message == "invalid action"
        assert exc.context == {}

    def test_with_action(self) -> None:
        exc = InvalidActionError("invalid", action=42)
        assert exc.context["action"] == 42

    def test_with_valid_actions(self) -> None:
        exc = InvalidActionError("invalid", valid_actions=[0, 1, 2, 5])
        assert exc.context["valid_actions_count"] == 4

    def test_with_action_and_valid_actions(self) -> None:
        exc = InvalidActionError("invalid", action=99, valid_actions=[0, 1, 2])
        assert exc.context["action"] == 99
        assert exc.context["valid_actions_count"] == 3

    def test_empty_valid_actions(self) -> None:
        exc = InvalidActionError("no valid actions", valid_actions=[])
        assert exc.context["valid_actions_count"] == 0

    def test_with_extra_context(self) -> None:
        exc = InvalidActionError("invalid", action=5, context={"player": 0})
        assert exc.context["action"] == 5
        assert exc.context["player"] == 0

    def test_str_includes_action_context(self) -> None:
        exc = InvalidActionError("invalid", action=7)
        s = str(exc)
        assert "invalid" in s
        assert "action=7" in s

    def test_caught_as_environment_error(self) -> None:
        with pytest.raises(EnvironmentError):
            raise InvalidActionError("bad action", action=0)


class TestBridgeErrors:
    """Tests for BridgeError hierarchy."""

    def test_bridge_error_is_base(self) -> None:
        exc = BridgeError("bridge problem")
        assert isinstance(exc, Civ6AlphaZeroError)

    def test_connection_error_is_bridge_error(self) -> None:
        exc = BridgeConnectionError("connection lost")
        assert isinstance(exc, BridgeError)

    def test_connection_error_message(self) -> None:
        exc = BridgeConnectionError("cannot connect to Civ6")
        assert "cannot connect" in str(exc)


class TestStateTimeoutError:
    """Tests for StateTimeoutError."""

    def test_is_bridge_error(self) -> None:
        exc = StateTimeoutError()
        assert isinstance(exc, BridgeError)

    def test_default_message(self) -> None:
        exc = StateTimeoutError()
        assert "Timeout" in str(exc)

    def test_custom_message(self) -> None:
        exc = StateTimeoutError("game state never arrived")
        assert "game state never arrived" in str(exc)

    def test_with_timeout(self) -> None:
        exc = StateTimeoutError(timeout=5.0)
        assert exc.context["timeout_seconds"] == 5.0

    def test_without_timeout(self) -> None:
        exc = StateTimeoutError("timed out")
        assert "timeout_seconds" not in exc.context

    def test_str_includes_timeout_context(self) -> None:
        exc = StateTimeoutError("timed out", timeout=3.5)
        s = str(exc)
        assert "3.5" in s

    def test_with_extra_context(self) -> None:
        exc = StateTimeoutError(timeout=2.0, context={"game_id": "g001"})
        assert exc.context["timeout_seconds"] == 2.0
        assert exc.context["game_id"] == "g001"

    def test_caught_as_bridge_error(self) -> None:
        with pytest.raises(BridgeError):
            raise StateTimeoutError("timeout")


class TestActionTimeoutError:
    """Tests for ActionTimeoutError."""

    def test_is_bridge_error(self) -> None:
        exc = ActionTimeoutError()
        assert isinstance(exc, BridgeError)

    def test_default_message(self) -> None:
        exc = ActionTimeoutError()
        assert "Timeout" in str(exc)

    def test_with_timeout(self) -> None:
        exc = ActionTimeoutError(timeout=10.0)
        assert exc.context["timeout_seconds"] == 10.0

    def test_with_action(self) -> None:
        exc = ActionTimeoutError(action=42)
        assert exc.context["action"] == 42

    def test_with_both_timeout_and_action(self) -> None:
        exc = ActionTimeoutError(timeout=5.0, action=7)
        assert exc.context["timeout_seconds"] == 5.0
        assert exc.context["action"] == 7

    def test_without_optional_fields(self) -> None:
        exc = ActionTimeoutError("ack not received")
        assert exc.context == {}

    def test_str_includes_all_context(self) -> None:
        exc = ActionTimeoutError("ack timeout", timeout=2.0, action=3)
        s = str(exc)
        assert "timeout_seconds=2.0" in s
        assert "action=3" in s

    def test_with_extra_context(self) -> None:
        exc = ActionTimeoutError(timeout=1.0, context={"worker": 3})
        assert exc.context["timeout_seconds"] == 1.0
        assert exc.context["worker"] == 3

    def test_caught_as_base(self) -> None:
        with pytest.raises(Civ6AlphaZeroError):
            raise ActionTimeoutError("timeout")


class TestTrainingErrors:
    """Tests for TrainingError hierarchy."""

    def test_training_error_is_base(self) -> None:
        exc = TrainingError("training failed")
        assert isinstance(exc, Civ6AlphaZeroError)

    def test_replay_buffer_error_is_training_error(self) -> None:
        exc = ReplayBufferError("buffer empty")
        assert isinstance(exc, TrainingError)

    def test_replay_buffer_with_context(self) -> None:
        exc = ReplayBufferError("sample failed", context={"size": 0, "min_required": 1000})
        assert "size=0" in str(exc)

    def test_replay_buffer_caught_as_training(self) -> None:
        with pytest.raises(TrainingError):
            raise ReplayBufferError("empty buffer")


class TestGradientError:
    """Tests for GradientError."""

    def test_is_training_error(self) -> None:
        exc = GradientError()
        assert isinstance(exc, TrainingError)

    def test_default_message(self) -> None:
        exc = GradientError()
        assert "Gradient" in str(exc)

    def test_custom_message(self) -> None:
        exc = GradientError("NaN gradient detected")
        assert "NaN gradient" in str(exc)

    def test_with_gradient_norm(self) -> None:
        exc = GradientError(gradient_norm=1e9)
        assert exc.context["gradient_norm"] == 1e9

    def test_without_gradient_norm(self) -> None:
        exc = GradientError("exploded")
        assert "gradient_norm" not in exc.context

    def test_str_includes_norm(self) -> None:
        exc = GradientError("exploded", gradient_norm=float("inf"))
        s = str(exc)
        assert "gradient_norm=inf" in s

    def test_with_extra_context(self) -> None:
        exc = GradientError(gradient_norm=500.0, context={"step": 1234})
        assert exc.context["gradient_norm"] == 500.0
        assert exc.context["step"] == 1234

    def test_caught_as_base(self) -> None:
        with pytest.raises(Civ6AlphaZeroError):
            raise GradientError("bad grad")


class TestDistributedErrors:
    """Tests for DistributedError hierarchy."""

    def test_distributed_error_is_base(self) -> None:
        exc = DistributedError("distributed failure")
        assert isinstance(exc, Civ6AlphaZeroError)

    def test_learner_error_is_distributed(self) -> None:
        exc = LearnerError("learner crashed")
        assert isinstance(exc, DistributedError)

    def test_learner_with_context(self) -> None:
        exc = LearnerError("OOM", context={"step": 5000})
        assert "step=5000" in str(exc)

    def test_learner_caught_as_distributed(self) -> None:
        with pytest.raises(DistributedError):
            raise LearnerError("failed")


class TestActorError:
    """Tests for ActorError."""

    def test_is_distributed_error(self) -> None:
        exc = ActorError("actor failed")
        assert isinstance(exc, DistributedError)

    def test_message_only(self) -> None:
        exc = ActorError("actor timed out")
        assert exc.message == "actor timed out"
        assert exc.context == {}

    def test_with_actor_id(self) -> None:
        exc = ActorError("actor unresponsive", actor_id=3)
        assert exc.context["actor_id"] == 3

    def test_without_actor_id(self) -> None:
        exc = ActorError("actor failed")
        assert "actor_id" not in exc.context

    def test_str_includes_actor_id(self) -> None:
        exc = ActorError("crash", actor_id=7)
        assert "actor_id=7" in str(exc)

    def test_with_extra_context(self) -> None:
        exc = ActorError("failed", actor_id=2, context={"games_played": 50})
        assert exc.context["actor_id"] == 2
        assert exc.context["games_played"] == 50

    def test_caught_as_base(self) -> None:
        with pytest.raises(Civ6AlphaZeroError):
            raise ActorError("crash", actor_id=1)


class TestWeightSyncError:
    """Tests for WeightSyncError."""

    def test_is_distributed_error(self) -> None:
        exc = WeightSyncError()
        assert isinstance(exc, DistributedError)

    def test_default_message(self) -> None:
        exc = WeightSyncError()
        assert "synchronization" in str(exc).lower() or "sync" in str(exc).lower()

    def test_custom_message(self) -> None:
        exc = WeightSyncError("failed to push weights")
        assert "failed to push" in str(exc)

    def test_with_version(self) -> None:
        exc = WeightSyncError(version=42)
        assert exc.context["version"] == 42

    def test_without_version(self) -> None:
        exc = WeightSyncError("sync error")
        assert "version" not in exc.context

    def test_str_includes_version(self) -> None:
        exc = WeightSyncError("failed", version=10)
        assert "version=10" in str(exc)

    def test_with_extra_context(self) -> None:
        exc = WeightSyncError(version=5, context={"actor_count": 8})
        assert exc.context["version"] == 5
        assert exc.context["actor_count"] == 8

    def test_caught_as_base(self) -> None:
        with pytest.raises(Civ6AlphaZeroError):
            raise WeightSyncError()


class TestExceptionHierarchy:
    """Tests that verify the full inheritance hierarchy."""

    def test_all_exceptions_are_base(self) -> None:
        exceptions = [
            ConfigurationError("c"),
            NetworkError("n"),
            NetworkInferenceError("ni"),
            NetworkLoadError("nl"),
            MCTSError("m"),
            MCTSSearchError("ms"),
            MCTSTimeoutError("mt"),
            EnvironmentError("e"),
            InvalidActionError("ia"),
            StateEncodingError("se"),
            BridgeError("b"),
            StateTimeoutError(),
            ActionTimeoutError(),
            BridgeConnectionError("bc"),
            TrainingError("t"),
            ReplayBufferError("rb"),
            GradientError(),
            DistributedError("d"),
            ActorError("a"),
            LearnerError("l"),
            WeightSyncError(),
        ]
        for exc in exceptions:
            assert isinstance(exc, Civ6AlphaZeroError), f"{type(exc).__name__} not base"

    def test_catch_all_via_base_class(self) -> None:
        """Catching Civ6AlphaZeroError catches all subclasses."""
        caught: list[type] = []
        to_raise = [
            ConfigurationError("c"),
            NetworkInferenceError("ni"),
            MCTSTimeoutError("mt"),
            InvalidActionError("ia"),
            StateTimeoutError(),
            GradientError(),
            WeightSyncError(),
        ]
        for exc in to_raise:
            try:
                raise exc
            except Civ6AlphaZeroError as e:
                caught.append(type(e))
        assert len(caught) == len(to_raise)


class TestFormatExceptionChain:
    """Tests for format_exception_chain utility function."""

    def test_single_exception(self) -> None:
        exc = Civ6AlphaZeroError("root error")
        result = format_exception_chain(exc)
        assert "root error" in result

    def test_single_standard_exception(self) -> None:
        exc = ValueError("bad value")
        result = format_exception_chain(exc)
        assert "ValueError" in result
        assert "bad value" in result

    def test_chained_exceptions(self) -> None:
        try:
            try:
                raise ValueError("original cause")
            except ValueError as cause:
                raise Civ6AlphaZeroError("wrapped error") from cause
        except Civ6AlphaZeroError as exc:
            result = format_exception_chain(exc)
            assert "wrapped error" in result
            assert "ValueError" in result
            assert "original cause" in result
            assert "->" in result

    def test_chained_project_exceptions(self) -> None:
        try:
            try:
                raise NetworkLoadError("weights missing")
            except NetworkLoadError as cause:
                raise TrainingError("cannot start training") from cause
        except TrainingError as exc:
            result = format_exception_chain(exc)
            assert "cannot start training" in result
            assert "weights missing" in result
            assert "->" in result

    def test_no_cause(self) -> None:
        exc = ConfigurationError("standalone error")
        result = format_exception_chain(exc)
        assert "standalone error" in result
        assert "->" not in result

    def test_exception_with_context_in_chain(self) -> None:
        exc = ActorError("actor crash", actor_id=3)
        result = format_exception_chain(exc)
        assert "actor_id=3" in result

    def test_standard_exception_format(self) -> None:
        exc = RuntimeError("runtime issue")
        result = format_exception_chain(exc)
        assert "RuntimeError: runtime issue" == result

    def test_base_exception_format(self) -> None:
        exc = Civ6AlphaZeroError("just message")
        result = format_exception_chain(exc)
        assert result == "just message"
