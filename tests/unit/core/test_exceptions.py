"""Tests for the SDK exception hierarchy defined in src.core.exceptions.

Covers:
- SDKError base class behaviour (message attribute, str representation)
- Full inheritance chains (every exception caught by each ancestor)
- Context-rich exceptions store extra attributes correctly
- All names in __all__ are importable
- Simple (no-extra-attrs) exceptions can be raised and caught
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.core import exceptions as exceptions_module
from src.core.exceptions import (
    CheckpointError,
    ConfigError,
    ConfigFileNotFoundError,
    ConfigValidationError,
    DuplicateRegistrationError,
    EnvironmentDeadlockError,
    EnvironmentError,
    EnvironmentNotRegisteredError,
    InsufficientDataError,
    InvalidActionError,
    NetworkDivergenceError,
    NoValidActionsError,
    PluginError,
    ProtocolViolationError,
    ReplayBufferError,
    SDKError,
    SearchError,
    SearchTimeoutError,
    SerializationError,
    TrainingError,
)

# ---------------------------------------------------------------------------
# 1. SDKError base class
# ---------------------------------------------------------------------------


class TestSDKError:
    """Tests for the SDKError base class."""

    def test_is_exception(self) -> None:
        assert issubclass(SDKError, Exception)

    def test_message_attribute(self) -> None:
        exc = SDKError("something went wrong")
        assert exc.message == "something went wrong"

    def test_str_representation(self) -> None:
        exc = SDKError("descriptive text")
        assert str(exc) == "descriptive text"

    def test_args_tuple(self) -> None:
        exc = SDKError("oops")
        assert exc.args == ("oops",)

    def test_catchable_as_exception(self) -> None:
        with pytest.raises(SDKError):
            raise SDKError("caught broadly")

    def test_catchable_as_sdk_error(self) -> None:
        with pytest.raises(SDKError):
            raise SDKError("caught narrowly")


# ---------------------------------------------------------------------------
# 2. Inheritance hierarchy (parametrized)
# ---------------------------------------------------------------------------

# Each tuple: (exception_class, constructor_args, expected_parent_chain)
# The parent chain lists every class that should catch the exception,
# from most-specific to most-general (excluding Exception/BaseException).
HIERARCHY: list[tuple[type, tuple[Any, ...], list[type]]] = [
    # Config branch
    (ConfigError, ("cfg",), [SDKError]),
    (ConfigFileNotFoundError, ("missing", "/a/b"), [ConfigError, SDKError]),
    (ConfigValidationError, ("bad field",), [ConfigError, SDKError]),
    # Environment branch
    (EnvironmentError, ("env",), [SDKError]),
    (InvalidActionError, ("bad", 5, [1, 2, 3]), [EnvironmentError, SDKError]),
    (
        EnvironmentNotRegisteredError,
        ("not found", "FooEnv", ["BarEnv"]),
        [EnvironmentError, SDKError],
    ),
    (EnvironmentDeadlockError, ("stuck",), [EnvironmentError, SDKError]),
    (SerializationError, ("corrupt",), [EnvironmentError, SDKError]),
    # Search branch
    (SearchError, ("search",), [SDKError]),
    (NoValidActionsError, ("empty",), [SearchError, SDKError]),
    (SearchTimeoutError, ("too slow",), [SearchError, SDKError]),
    # Training branch
    (TrainingError, ("train",), [SDKError]),
    (ReplayBufferError, ("overflow",), [TrainingError, SDKError]),
    (CheckpointError, ("io fail",), [TrainingError, SDKError]),
    (
        NetworkDivergenceError,
        ("spike", 100.0, 10.0),
        [TrainingError, SDKError],
    ),
    (InsufficientDataError, ("need more",), [TrainingError, SDKError]),
    # Plugin branch
    (PluginError, ("plugin",), [SDKError]),
    (
        ProtocolViolationError,
        ("missing methods", int, ["step", "reset"]),
        [PluginError, SDKError],
    ),
    (DuplicateRegistrationError, ("already exists",), [PluginError, SDKError]),
]


def _make_id(entry: tuple[type, tuple[Any, ...], list[type]]) -> str:
    """Produce a readable pytest ID for each hierarchy entry."""
    return entry[0].__name__


@pytest.mark.parametrize(
    ("exc_class", "ctor_args", "parent_chain"),
    HIERARCHY,
    ids=[_make_id(e) for e in HIERARCHY],
)
class TestInheritanceHierarchy:
    """Every exception is catchable by each of its ancestors."""

    def test_isinstance_of_each_parent(
        self,
        exc_class: type,
        ctor_args: tuple[Any, ...],
        parent_chain: list[type],
    ) -> None:
        exc = exc_class(*ctor_args)
        for parent in parent_chain:
            assert isinstance(exc, parent), (
                f"{exc_class.__name__} should be an instance of {parent.__name__}"
            )

    def test_isinstance_of_exception(
        self,
        exc_class: type,
        ctor_args: tuple[Any, ...],
        parent_chain: list[type],
    ) -> None:
        exc = exc_class(*ctor_args)
        assert isinstance(exc, Exception)

    def test_raise_and_catch_by_parent(
        self,
        exc_class: type,
        ctor_args: tuple[Any, ...],
        parent_chain: list[type],
    ) -> None:
        """Actually raise and catch by each ancestor type."""
        for parent in parent_chain:
            with pytest.raises(parent):
                raise exc_class(*ctor_args)

    def test_message_propagates(
        self,
        exc_class: type,
        ctor_args: tuple[Any, ...],
        parent_chain: list[type],
    ) -> None:
        """The first arg is always the human-readable message."""
        exc = exc_class(*ctor_args)
        assert exc.message == ctor_args[0]


# ---------------------------------------------------------------------------
# 3. Context-rich exceptions
# ---------------------------------------------------------------------------


class TestConfigFileNotFoundError:
    """ConfigFileNotFoundError stores path as a Path object."""

    def test_path_from_string(self) -> None:
        exc = ConfigFileNotFoundError("missing", "/etc/config.yaml")
        assert exc.path == Path("/etc/config.yaml")
        assert isinstance(exc.path, Path)

    def test_path_from_path_object(self) -> None:
        p = Path("/var/data/config.yaml")
        exc = ConfigFileNotFoundError("missing", p)
        assert exc.path == p
        assert isinstance(exc.path, Path)

    def test_message_preserved(self) -> None:
        exc = ConfigFileNotFoundError("file missing", "x.yaml")
        assert exc.message == "file missing"

    def test_str_is_message(self) -> None:
        exc = ConfigFileNotFoundError("not found", "a.yaml")
        assert str(exc) == "not found"


class TestInvalidActionError:
    """InvalidActionError stores action and valid_actions."""

    def test_attributes(self) -> None:
        exc = InvalidActionError("bad action", action=42, valid_actions=[1, 2, 3])
        assert exc.action == 42
        assert exc.valid_actions == [1, 2, 3]

    def test_valid_actions_preserves_sequence_type(self) -> None:
        """valid_actions keeps whatever sequence type the caller provides."""
        actions_tuple = (10, 20, 30)
        exc = InvalidActionError("bad", action=0, valid_actions=actions_tuple)
        # The source stores it as-is (no list() conversion)
        assert exc.valid_actions == (10, 20, 30)

    def test_action_can_be_any_type(self) -> None:
        exc = InvalidActionError("bad", action="move_north", valid_actions=["move_south"])
        assert exc.action == "move_north"

    def test_message(self) -> None:
        exc = InvalidActionError("nope", action=0, valid_actions=[])
        assert exc.message == "nope"


class TestEnvironmentNotRegisteredError:
    """EnvironmentNotRegisteredError stores name and available as list."""

    def test_attributes(self) -> None:
        exc = EnvironmentNotRegisteredError(
            "unknown env", name="FooEnv", available=["BarEnv", "BazEnv"]
        )
        assert exc.name == "FooEnv"
        assert exc.available == ["BarEnv", "BazEnv"]

    def test_available_converted_to_list_from_tuple(self) -> None:
        exc = EnvironmentNotRegisteredError(
            "unknown", name="X", available=("A", "B")
        )
        assert exc.available == ["A", "B"]
        assert isinstance(exc.available, list)

    def test_available_converted_to_list_from_generator(self) -> None:
        exc = EnvironmentNotRegisteredError(
            "unknown", name="X", available=(s for s in ["A", "B", "C"])
        )
        assert exc.available == ["A", "B", "C"]
        assert isinstance(exc.available, list)

    def test_empty_available(self) -> None:
        exc = EnvironmentNotRegisteredError(
            "nothing registered", name="Foo", available=[]
        )
        assert exc.available == []


class TestNetworkDivergenceError:
    """NetworkDivergenceError stores loss_value and threshold."""

    def test_attributes(self) -> None:
        exc = NetworkDivergenceError("diverged", loss_value=999.0, threshold=100.0)
        assert exc.loss_value == 999.0
        assert exc.threshold == 100.0

    def test_message(self) -> None:
        exc = NetworkDivergenceError("loss spike", loss_value=50.0, threshold=10.0)
        assert exc.message == "loss spike"

    def test_values_are_float(self) -> None:
        exc = NetworkDivergenceError("spike", loss_value=42.5, threshold=10.0)
        assert isinstance(exc.loss_value, float)
        assert isinstance(exc.threshold, float)

    def test_integer_args_preserved(self) -> None:
        """Passing ints should still work (no explicit float conversion in source)."""
        exc = NetworkDivergenceError("spike", loss_value=100, threshold=50)
        assert exc.loss_value == 100
        assert exc.threshold == 50


class TestProtocolViolationError:
    """ProtocolViolationError stores env_class and missing_methods as list."""

    def test_attributes(self) -> None:
        exc = ProtocolViolationError(
            "bad env", env_class=int, missing_methods=["step", "reset"]
        )
        assert exc.env_class is int
        assert exc.missing_methods == ["step", "reset"]

    def test_missing_methods_converted_to_list_from_tuple(self) -> None:
        exc = ProtocolViolationError(
            "violation", env_class=str, missing_methods=("foo", "bar")
        )
        assert exc.missing_methods == ["foo", "bar"]
        assert isinstance(exc.missing_methods, list)

    def test_missing_methods_converted_to_list_from_generator(self) -> None:
        exc = ProtocolViolationError(
            "violation",
            env_class=dict,
            missing_methods=(m for m in ["a", "b"]),
        )
        assert exc.missing_methods == ["a", "b"]
        assert isinstance(exc.missing_methods, list)

    def test_env_class_is_type(self) -> None:
        class MyEnv:
            pass

        exc = ProtocolViolationError(
            "bad", env_class=MyEnv, missing_methods=["step"]
        )
        assert exc.env_class is MyEnv

    def test_empty_missing_methods(self) -> None:
        exc = ProtocolViolationError("ok?", env_class=object, missing_methods=[])
        assert exc.missing_methods == []


# ---------------------------------------------------------------------------
# 4. __all__ completeness
# ---------------------------------------------------------------------------


class TestAllExports:
    """Every name listed in __all__ is importable from the module."""

    def test_all_names_importable(self) -> None:
        for name in exceptions_module.__all__:
            obj = getattr(exceptions_module, name)
            assert isinstance(obj, type), f"{name} should be a class"
            assert issubclass(obj, SDKError), f"{name} should be a subclass of SDKError"

    def test_all_count(self) -> None:
        """There should be exactly 20 exception classes in __all__."""
        assert len(exceptions_module.__all__) == 20

    @pytest.mark.parametrize("name", exceptions_module.__all__)
    def test_each_name_is_class(self, name: str) -> None:
        cls = getattr(exceptions_module, name)
        assert isinstance(cls, type)


# ---------------------------------------------------------------------------
# 5. Simple exceptions (no extra attributes) raise and catch cleanly
# ---------------------------------------------------------------------------

SIMPLE_EXCEPTIONS: list[type] = [
    ConfigError,
    ConfigValidationError,
    EnvironmentError,
    EnvironmentDeadlockError,
    SerializationError,
    SearchError,
    NoValidActionsError,
    SearchTimeoutError,
    TrainingError,
    ReplayBufferError,
    CheckpointError,
    InsufficientDataError,
    PluginError,
    DuplicateRegistrationError,
]


@pytest.mark.parametrize("exc_class", SIMPLE_EXCEPTIONS, ids=lambda c: c.__name__)
class TestSimpleExceptions:
    """Simple exceptions (single message arg) raise and catch correctly."""

    def test_raise_and_catch(self, exc_class: type) -> None:
        with pytest.raises(exc_class):
            raise exc_class("test message")

    def test_message_attribute(self, exc_class: type) -> None:
        exc = exc_class("hello")
        assert exc.message == "hello"

    def test_str(self, exc_class: type) -> None:
        exc = exc_class("description")
        assert str(exc) == "description"

    def test_args_tuple(self, exc_class: type) -> None:
        exc = exc_class("msg")
        assert exc.args == ("msg",)

    def test_is_sdk_error(self, exc_class: type) -> None:
        exc = exc_class("msg")
        assert isinstance(exc, SDKError)


# ---------------------------------------------------------------------------
# 6. Cross-branch isolation: exceptions from different branches
#    should NOT catch each other
# ---------------------------------------------------------------------------


class TestCrossBranchIsolation:
    """Exceptions from different branches of the hierarchy do not overlap."""

    def test_config_not_caught_by_environment(self) -> None:
        with pytest.raises(ConfigError):
            try:
                raise ConfigError("cfg problem")
            except EnvironmentError:
                pytest.fail("ConfigError should not be caught by EnvironmentError")

    def test_search_not_caught_by_training(self) -> None:
        with pytest.raises(SearchError):
            try:
                raise SearchError("search fail")
            except TrainingError:
                pytest.fail("SearchError should not be caught by TrainingError")

    def test_plugin_not_caught_by_config(self) -> None:
        with pytest.raises(PluginError):
            try:
                raise PluginError("plugin fail")
            except ConfigError:
                pytest.fail("PluginError should not be caught by ConfigError")

    def test_training_not_caught_by_search(self) -> None:
        with pytest.raises(TrainingError):
            try:
                raise TrainingError("train fail")
            except SearchError:
                pytest.fail("TrainingError should not be caught by SearchError")

    def test_environment_not_caught_by_plugin(self) -> None:
        with pytest.raises(EnvironmentError):
            try:
                raise EnvironmentError("env fail")
            except PluginError:
                pytest.fail("EnvironmentError should not be caught by PluginError")


# ---------------------------------------------------------------------------
# 7. Edge cases and repr
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge-case coverage."""

    def test_empty_message(self) -> None:
        exc = SDKError("")
        assert exc.message == ""
        assert str(exc) == ""

    def test_multiline_message(self) -> None:
        msg = "line1\nline2\nline3"
        exc = SDKError(msg)
        assert exc.message == msg

    def test_unicode_message(self) -> None:
        msg = "Error: invalid config \u2014 check encoding \u00e9\u00e8\u00ea"
        exc = SDKError(msg)
        assert exc.message == msg

    def test_config_file_not_found_relative_path(self) -> None:
        exc = ConfigFileNotFoundError("missing", "relative/path.yaml")
        assert exc.path == Path("relative/path.yaml")

    def test_invalid_action_with_none_action(self) -> None:
        exc = InvalidActionError("null action", action=None, valid_actions=[1, 2])
        assert exc.action is None

    def test_network_divergence_zero_threshold(self) -> None:
        exc = NetworkDivergenceError("zero", loss_value=0.001, threshold=0.0)
        assert exc.threshold == 0.0

    def test_protocol_violation_preserves_class_identity(self) -> None:
        class SomeEnv:
            pass

        exc = ProtocolViolationError("bad", env_class=SomeEnv, missing_methods=["step"])
        assert exc.env_class.__name__ == "SomeEnv"

    def test_sdk_error_not_caught_by_children(self) -> None:
        """SDKError should not be caught by any of its sub-categories."""
        exc = SDKError("base")
        assert not isinstance(exc, ConfigError)
        assert not isinstance(exc, EnvironmentError)
        assert not isinstance(exc, SearchError)
        assert not isinstance(exc, TrainingError)
        assert not isinstance(exc, PluginError)
