"""Comprehensive tests for the SDK exception hierarchy in src.core.exceptions.

Covers:
1. SDKError base class (instantiation, message attribute, Exception subclass, str())
2. Full inheritance hierarchy (every subclass catchable by its ancestor chain)
3. ConfigFileNotFoundError (path attribute as Path, str-to-Path conversion)
4. InvalidActionError (action and valid_actions attributes)
5. EnvironmentNotRegisteredError (name and available attributes, list conversion)
6. NetworkDivergenceError (loss_value and threshold attributes)
7. ProtocolViolationError (env_class, missing_methods, Sequence-to-list conversion)
8. Catch patterns (broad SDKError catches all, mid-level catches own branch only)
9. All 20 exception classes instantiable with appropriate arguments
10. __all__ contains exactly 20 entries
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
    """SDKError base class behaviour."""

    def test_instantiation(self) -> None:
        exc = SDKError("boom")
        assert exc is not None

    def test_message_attribute(self) -> None:
        exc = SDKError("something went wrong")
        assert exc.message == "something went wrong"

    def test_is_subclass_of_exception(self) -> None:
        assert issubclass(SDKError, Exception)

    def test_is_instance_of_exception(self) -> None:
        exc = SDKError("test")
        assert isinstance(exc, Exception)

    def test_str_returns_message(self) -> None:
        exc = SDKError("human-readable text")
        assert str(exc) == "human-readable text"

    def test_args_tuple_contains_message(self) -> None:
        exc = SDKError("msg")
        assert exc.args == ("msg",)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(SDKError):
            raise SDKError("boom")

    def test_caught_as_exception(self) -> None:
        with pytest.raises(SDKError):
            raise SDKError("broad catch")

    def test_empty_message(self) -> None:
        exc = SDKError("")
        assert exc.message == ""
        assert str(exc) == ""

    def test_multiline_message(self) -> None:
        msg = "line one\nline two\nline three"
        exc = SDKError(msg)
        assert exc.message == msg
        assert str(exc) == msg

    def test_unicode_message(self) -> None:
        msg = "Error \u2014 invalid \u00e9ncoding"
        exc = SDKError(msg)
        assert exc.message == msg

    def test_not_instance_of_any_child(self) -> None:
        exc = SDKError("base only")
        assert not isinstance(exc, ConfigError)
        assert not isinstance(exc, EnvironmentError)
        assert not isinstance(exc, SearchError)
        assert not isinstance(exc, TrainingError)
        assert not isinstance(exc, PluginError)


# ---------------------------------------------------------------------------
# 2. Hierarchy tests -- every subclass catchable by its ancestors
# ---------------------------------------------------------------------------

# (exception_class, constructor_args, ancestor_chain from specific to general)
HIERARCHY_CASES: list[tuple[type, tuple[Any, ...], list[type]]] = [
    # Config branch
    (ConfigError, ("cfg",), [SDKError]),
    (ConfigFileNotFoundError, ("missing", "/a/b.yaml"), [ConfigError, SDKError]),
    (ConfigValidationError, ("invalid field",), [ConfigError, SDKError]),
    # Environment branch
    (EnvironmentError, ("env fail",), [SDKError]),
    (InvalidActionError, ("bad action", 5, [1, 2, 3]), [EnvironmentError, SDKError]),
    (
        EnvironmentNotRegisteredError,
        ("unknown", "FooEnv", ["BarEnv"]),
        [EnvironmentError, SDKError],
    ),
    (EnvironmentDeadlockError, ("deadlock",), [EnvironmentError, SDKError]),
    (SerializationError, ("corrupt data",), [EnvironmentError, SDKError]),
    # Search branch
    (SearchError, ("search fail",), [SDKError]),
    (NoValidActionsError, ("empty mask",), [SearchError, SDKError]),
    (SearchTimeoutError, ("too slow",), [SearchError, SDKError]),
    # Training branch
    (TrainingError, ("training fail",), [SDKError]),
    (ReplayBufferError, ("overflow",), [TrainingError, SDKError]),
    (CheckpointError, ("io error",), [TrainingError, SDKError]),
    (NetworkDivergenceError, ("loss spike", 100.0, 10.0), [TrainingError, SDKError]),
    (InsufficientDataError, ("need more data",), [TrainingError, SDKError]),
    # Plugin branch
    (PluginError, ("plugin fail",), [SDKError]),
    (
        ProtocolViolationError,
        ("missing methods", int, ["step", "reset"]),
        [PluginError, SDKError],
    ),
    (DuplicateRegistrationError, ("already registered",), [PluginError, SDKError]),
]


def _hierarchy_id(entry: tuple[type, tuple[Any, ...], list[type]]) -> str:
    return entry[0].__name__


@pytest.mark.parametrize(
    ("exc_class", "ctor_args", "ancestors"),
    HIERARCHY_CASES,
    ids=[_hierarchy_id(e) for e in HIERARCHY_CASES],
)
class TestHierarchy:
    """Each exception is an instance of every ancestor in its chain."""

    def test_isinstance_of_each_ancestor(
        self,
        exc_class: type,
        ctor_args: tuple[Any, ...],
        ancestors: list[type],
    ) -> None:
        exc = exc_class(*ctor_args)
        for ancestor in ancestors:
            assert isinstance(exc, ancestor), (
                f"{exc_class.__name__} should be instance of {ancestor.__name__}"
            )

    def test_isinstance_of_exception(
        self,
        exc_class: type,
        ctor_args: tuple[Any, ...],
        ancestors: list[type],
    ) -> None:
        exc = exc_class(*ctor_args)
        assert isinstance(exc, Exception)

    def test_raise_and_catch_by_each_ancestor(
        self,
        exc_class: type,
        ctor_args: tuple[Any, ...],
        ancestors: list[type],
    ) -> None:
        for ancestor in ancestors:
            with pytest.raises(ancestor):
                raise exc_class(*ctor_args)

    def test_message_propagated(
        self,
        exc_class: type,
        ctor_args: tuple[Any, ...],
        ancestors: list[type],
    ) -> None:
        exc = exc_class(*ctor_args)
        assert exc.message == ctor_args[0]


# ---------------------------------------------------------------------------
# 3. ConfigFileNotFoundError -- path attribute
# ---------------------------------------------------------------------------


class TestConfigFileNotFoundError:
    """ConfigFileNotFoundError stores path as a pathlib.Path."""

    def test_path_from_string(self) -> None:
        exc = ConfigFileNotFoundError("not found", "/etc/config.yaml")
        assert exc.path == Path("/etc/config.yaml")
        assert isinstance(exc.path, Path)

    def test_path_from_path_object(self) -> None:
        p = Path("/var/data/settings.yaml")
        exc = ConfigFileNotFoundError("missing file", p)
        assert exc.path == p
        assert isinstance(exc.path, Path)

    def test_relative_path(self) -> None:
        exc = ConfigFileNotFoundError("not found", "configs/local.yaml")
        assert exc.path == Path("configs/local.yaml")

    def test_message_set_correctly(self) -> None:
        exc = ConfigFileNotFoundError("config file not found", "/x.yaml")
        assert exc.message == "config file not found"

    def test_str_returns_message(self) -> None:
        exc = ConfigFileNotFoundError("could not load", "/a.yaml")
        assert str(exc) == "could not load"

    def test_catchable_as_config_error(self) -> None:
        with pytest.raises(ConfigError):
            raise ConfigFileNotFoundError("gone", "/b.yaml")

    def test_catchable_as_sdk_error(self) -> None:
        with pytest.raises(SDKError):
            raise ConfigFileNotFoundError("gone", "/c.yaml")


# ---------------------------------------------------------------------------
# 4. InvalidActionError -- action and valid_actions attributes
# ---------------------------------------------------------------------------


class TestInvalidActionError:
    """InvalidActionError stores action and valid_actions."""

    def test_integer_action_and_list(self) -> None:
        exc = InvalidActionError("bad action", action=42, valid_actions=[1, 2, 3])
        assert exc.action == 42
        assert exc.valid_actions == [1, 2, 3]

    def test_string_action(self) -> None:
        exc = InvalidActionError(
            "invalid", action="move_north", valid_actions=["move_south", "wait"]
        )
        assert exc.action == "move_north"
        assert exc.valid_actions == ["move_south", "wait"]

    def test_none_action(self) -> None:
        exc = InvalidActionError("null", action=None, valid_actions=[0, 1])
        assert exc.action is None

    def test_valid_actions_preserves_sequence_type(self) -> None:
        """The source stores valid_actions as-is, without list() conversion."""
        tup = (10, 20, 30)
        exc = InvalidActionError("bad", action=0, valid_actions=tup)
        assert exc.valid_actions == (10, 20, 30)

    def test_empty_valid_actions(self) -> None:
        exc = InvalidActionError("no choices", action=5, valid_actions=[])
        assert exc.valid_actions == []

    def test_message_set(self) -> None:
        exc = InvalidActionError("nope", action=0, valid_actions=[])
        assert exc.message == "nope"

    def test_caught_as_environment_error(self) -> None:
        with pytest.raises(EnvironmentError):
            raise InvalidActionError("bad", action=1, valid_actions=[2, 3])

    def test_caught_as_sdk_error(self) -> None:
        with pytest.raises(SDKError):
            raise InvalidActionError("bad", action=1, valid_actions=[2, 3])


# ---------------------------------------------------------------------------
# 5. EnvironmentNotRegisteredError -- name and available attributes
# ---------------------------------------------------------------------------


class TestEnvironmentNotRegisteredError:
    """EnvironmentNotRegisteredError stores name and converts available to list."""

    def test_attributes_with_list(self) -> None:
        exc = EnvironmentNotRegisteredError(
            "not found", name="FooEnv", available=["BarEnv", "BazEnv"]
        )
        assert exc.name == "FooEnv"
        assert exc.available == ["BarEnv", "BazEnv"]
        assert isinstance(exc.available, list)

    def test_available_converted_from_tuple(self) -> None:
        exc = EnvironmentNotRegisteredError(
            "unknown", name="X", available=("A", "B")
        )
        assert exc.available == ["A", "B"]
        assert isinstance(exc.available, list)

    def test_available_converted_from_generator(self) -> None:
        exc = EnvironmentNotRegisteredError(
            "unknown", name="Y", available=(s for s in ["C", "D", "E"])
        )
        assert exc.available == ["C", "D", "E"]
        assert isinstance(exc.available, list)

    def test_available_converted_from_set(self) -> None:
        exc = EnvironmentNotRegisteredError(
            "unknown", name="Z", available={"Alpha"}
        )
        assert isinstance(exc.available, list)
        assert exc.available == ["Alpha"]

    def test_empty_available(self) -> None:
        exc = EnvironmentNotRegisteredError(
            "nothing registered", name="Foo", available=[]
        )
        assert exc.available == []
        assert isinstance(exc.available, list)

    def test_message_set(self) -> None:
        exc = EnvironmentNotRegisteredError("nope", name="X", available=[])
        assert exc.message == "nope"


# ---------------------------------------------------------------------------
# 6. NetworkDivergenceError -- loss_value and threshold
# ---------------------------------------------------------------------------


class TestNetworkDivergenceError:
    """NetworkDivergenceError stores loss_value and threshold as floats."""

    def test_float_attributes(self) -> None:
        exc = NetworkDivergenceError("diverged", loss_value=999.5, threshold=100.0)
        assert exc.loss_value == 999.5
        assert exc.threshold == 100.0

    def test_values_are_float_type(self) -> None:
        exc = NetworkDivergenceError("spike", loss_value=42.5, threshold=10.0)
        assert isinstance(exc.loss_value, float)
        assert isinstance(exc.threshold, float)

    def test_integer_args_accepted(self) -> None:
        """Passing ints works (no explicit float cast in source)."""
        exc = NetworkDivergenceError("spike", loss_value=100, threshold=50)
        assert exc.loss_value == 100
        assert exc.threshold == 50

    def test_zero_threshold(self) -> None:
        exc = NetworkDivergenceError("zero", loss_value=0.001, threshold=0.0)
        assert exc.threshold == 0.0

    def test_message_set(self) -> None:
        exc = NetworkDivergenceError("loss spike", loss_value=50.0, threshold=10.0)
        assert exc.message == "loss spike"

    def test_caught_as_training_error(self) -> None:
        with pytest.raises(TrainingError):
            raise NetworkDivergenceError("div", loss_value=1.0, threshold=0.5)

    def test_caught_as_sdk_error(self) -> None:
        with pytest.raises(SDKError):
            raise NetworkDivergenceError("div", loss_value=1.0, threshold=0.5)


# ---------------------------------------------------------------------------
# 7. ProtocolViolationError -- env_class and missing_methods
# ---------------------------------------------------------------------------


class TestProtocolViolationError:
    """ProtocolViolationError stores env_class (type) and missing_methods (list[str])."""

    def test_attributes_with_list(self) -> None:
        exc = ProtocolViolationError(
            "bad env", env_class=int, missing_methods=["step", "reset"]
        )
        assert exc.env_class is int
        assert exc.missing_methods == ["step", "reset"]
        assert isinstance(exc.missing_methods, list)

    def test_missing_methods_converted_from_tuple(self) -> None:
        exc = ProtocolViolationError(
            "violation", env_class=str, missing_methods=("foo", "bar")
        )
        assert exc.missing_methods == ["foo", "bar"]
        assert isinstance(exc.missing_methods, list)

    def test_missing_methods_converted_from_generator(self) -> None:
        exc = ProtocolViolationError(
            "violation",
            env_class=dict,
            missing_methods=(m for m in ["a", "b", "c"]),
        )
        assert exc.missing_methods == ["a", "b", "c"]
        assert isinstance(exc.missing_methods, list)

    def test_env_class_is_a_type(self) -> None:
        class MyGameEnv:
            pass

        exc = ProtocolViolationError(
            "missing protocol", env_class=MyGameEnv, missing_methods=["step"]
        )
        assert exc.env_class is MyGameEnv
        assert isinstance(exc.env_class, type)

    def test_env_class_name_preserved(self) -> None:
        class CustomEnv:
            pass

        exc = ProtocolViolationError(
            "bad", env_class=CustomEnv, missing_methods=["reset"]
        )
        assert exc.env_class.__name__ == "CustomEnv"

    def test_empty_missing_methods(self) -> None:
        exc = ProtocolViolationError("ok?", env_class=object, missing_methods=[])
        assert exc.missing_methods == []
        assert isinstance(exc.missing_methods, list)

    def test_message_set(self) -> None:
        exc = ProtocolViolationError("protocol error", env_class=int, missing_methods=[])
        assert exc.message == "protocol error"

    def test_caught_as_plugin_error(self) -> None:
        with pytest.raises(PluginError):
            raise ProtocolViolationError("bad", env_class=int, missing_methods=[])

    def test_caught_as_sdk_error(self) -> None:
        with pytest.raises(SDKError):
            raise ProtocolViolationError("bad", env_class=int, missing_methods=[])


# ---------------------------------------------------------------------------
# 8. Catch pattern tests -- realistic raise/except scenarios
# ---------------------------------------------------------------------------


class TestCatchPatterns:
    """Realistic catch scenarios matching how callers would use the hierarchy."""

    def test_sdk_error_catches_all(self) -> None:
        """except SDKError catches every exception in the hierarchy."""
        all_exceptions: list[tuple[type, tuple[Any, ...]]] = [
            (SDKError, ("base",)),
            (ConfigError, ("cfg",)),
            (ConfigFileNotFoundError, ("gone", "/a.yaml")),
            (ConfigValidationError, ("invalid",)),
            (EnvironmentError, ("env",)),
            (InvalidActionError, ("bad", 1, [2])),
            (EnvironmentNotRegisteredError, ("missing", "X", ["Y"])),
            (EnvironmentDeadlockError, ("stuck",)),
            (SerializationError, ("corrupt",)),
            (SearchError, ("fail",)),
            (NoValidActionsError, ("empty",)),
            (SearchTimeoutError, ("timeout",)),
            (TrainingError, ("train",)),
            (ReplayBufferError, ("overflow",)),
            (CheckpointError, ("io",)),
            (NetworkDivergenceError, ("spike", 10.0, 1.0)),
            (InsufficientDataError, ("need more",)),
            (PluginError, ("plugin",)),
            (ProtocolViolationError, ("violation", int, ["step"])),
            (DuplicateRegistrationError, ("dup",)),
        ]
        for exc_class, args in all_exceptions:
            with pytest.raises(SDKError):
                raise exc_class(*args)

    def test_config_error_catches_config_variants(self) -> None:
        """except ConfigError catches ConfigFileNotFoundError and ConfigValidationError."""
        with pytest.raises(ConfigError):
            raise ConfigFileNotFoundError("missing", "/etc/app.yaml")
        with pytest.raises(ConfigError):
            raise ConfigValidationError("bad field")

    def test_config_error_does_not_catch_search_error(self) -> None:
        """except ConfigError does NOT catch SearchError."""
        with pytest.raises(SearchError):
            try:
                raise SearchError("search fail")
            except ConfigError:
                pytest.fail("SearchError should not be caught by ConfigError")

    def test_config_error_does_not_catch_training_error(self) -> None:
        with pytest.raises(TrainingError):
            try:
                raise CheckpointError("io fail")
            except ConfigError:
                pytest.fail("CheckpointError should not be caught by ConfigError")

    def test_environment_error_catches_environment_variants(self) -> None:
        """except EnvironmentError catches all environment-branch exceptions."""
        with pytest.raises(EnvironmentError):
            raise InvalidActionError("bad move", action=99, valid_actions=[0, 1, 2])
        with pytest.raises(EnvironmentError):
            raise EnvironmentNotRegisteredError("missing", name="X", available=[])
        with pytest.raises(EnvironmentError):
            raise EnvironmentDeadlockError("stuck")
        with pytest.raises(EnvironmentError):
            raise SerializationError("corrupt")

    def test_search_error_catches_search_variants(self) -> None:
        with pytest.raises(SearchError):
            raise NoValidActionsError("terminal state")
        with pytest.raises(SearchError):
            raise SearchTimeoutError("too slow")

    def test_training_error_catches_training_variants(self) -> None:
        with pytest.raises(TrainingError):
            raise ReplayBufferError("overflow")
        with pytest.raises(TrainingError):
            raise CheckpointError("could not save")
        with pytest.raises(TrainingError):
            raise NetworkDivergenceError("diverged", loss_value=100.0, threshold=10.0)
        with pytest.raises(TrainingError):
            raise InsufficientDataError("need more data")

    def test_plugin_error_catches_plugin_variants(self) -> None:
        with pytest.raises(PluginError):
            raise ProtocolViolationError("bad", env_class=int, missing_methods=[])
        with pytest.raises(PluginError):
            raise DuplicateRegistrationError("already registered")

    def test_specific_handler_takes_priority(self) -> None:
        """When both specific and general handlers exist, specific wins."""
        handler_used = None
        try:
            raise InvalidActionError("bad", action=5, valid_actions=[1, 2])
        except InvalidActionError:
            handler_used = "InvalidActionError"
        except EnvironmentError:
            handler_used = "EnvironmentError"
        except SDKError:
            handler_used = "SDKError"
        assert handler_used == "InvalidActionError"

    def test_mid_level_handler_catches_leaf(self) -> None:
        """A mid-level handler (e.g., TrainingError) catches leaf exceptions."""
        handler_used = None
        try:
            raise ReplayBufferError("overflow")
        except TrainingError:
            handler_used = "TrainingError"
        except SDKError:
            handler_used = "SDKError"
        assert handler_used == "TrainingError"


# ---------------------------------------------------------------------------
# 9. All 20 exception classes can be instantiated with appropriate args
# ---------------------------------------------------------------------------

ALL_TWENTY_CLASSES = [
    SDKError,
    ConfigError,
    ConfigFileNotFoundError,
    ConfigValidationError,
    EnvironmentError,
    InvalidActionError,
    EnvironmentNotRegisteredError,
    EnvironmentDeadlockError,
    SerializationError,
    SearchError,
    NoValidActionsError,
    SearchTimeoutError,
    TrainingError,
    ReplayBufferError,
    CheckpointError,
    NetworkDivergenceError,
    InsufficientDataError,
    PluginError,
    ProtocolViolationError,
    DuplicateRegistrationError,
]


class TestAllTwentyExist:
    """All 20 exception classes exist and are importable."""

    def test_count(self) -> None:
        assert len(ALL_TWENTY_CLASSES) == 20

    @pytest.mark.parametrize("cls", ALL_TWENTY_CLASSES, ids=lambda c: c.__name__)
    def test_is_type(self, cls: type) -> None:
        assert isinstance(cls, type)

    @pytest.mark.parametrize("cls", ALL_TWENTY_CLASSES, ids=lambda c: c.__name__)
    def test_is_subclass_of_sdk_error(self, cls: type) -> None:
        assert issubclass(cls, SDKError)


# Simple exceptions (message-only constructor)
SIMPLE_EXCEPTIONS: list[type] = [
    SDKError,
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
    """Simple exceptions (single message arg) instantiate and behave correctly."""

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


# Rich exceptions (extra constructor params)
class TestRichExceptionInstantiation:
    """Exceptions with extra constructor parameters can be instantiated correctly."""

    def test_config_file_not_found_error(self) -> None:
        exc = ConfigFileNotFoundError("missing file", "/etc/config.yaml")
        assert exc.message == "missing file"
        assert exc.path == Path("/etc/config.yaml")

    def test_invalid_action_error(self) -> None:
        exc = InvalidActionError("bad action", action=42, valid_actions=[1, 2, 3])
        assert exc.message == "bad action"
        assert exc.action == 42
        assert exc.valid_actions == [1, 2, 3]

    def test_environment_not_registered_error(self) -> None:
        exc = EnvironmentNotRegisteredError(
            "unknown env", name="FooEnv", available=["BarEnv"]
        )
        assert exc.message == "unknown env"
        assert exc.name == "FooEnv"
        assert exc.available == ["BarEnv"]

    def test_network_divergence_error(self) -> None:
        exc = NetworkDivergenceError("loss spike", loss_value=100.0, threshold=10.0)
        assert exc.message == "loss spike"
        assert exc.loss_value == 100.0
        assert exc.threshold == 10.0

    def test_protocol_violation_error(self) -> None:
        exc = ProtocolViolationError(
            "missing methods", env_class=int, missing_methods=["step", "reset"]
        )
        assert exc.message == "missing methods"
        assert exc.env_class is int
        assert exc.missing_methods == ["step", "reset"]


# ---------------------------------------------------------------------------
# 10. __all__ contains exactly 20 entries
# ---------------------------------------------------------------------------

EXPECTED_ALL_NAMES = [
    "SDKError",
    "ConfigError",
    "ConfigFileNotFoundError",
    "ConfigValidationError",
    "EnvironmentError",
    "InvalidActionError",
    "EnvironmentNotRegisteredError",
    "EnvironmentDeadlockError",
    "SerializationError",
    "SearchError",
    "NoValidActionsError",
    "SearchTimeoutError",
    "TrainingError",
    "ReplayBufferError",
    "CheckpointError",
    "NetworkDivergenceError",
    "InsufficientDataError",
    "PluginError",
    "ProtocolViolationError",
    "DuplicateRegistrationError",
]


class TestDunderAll:
    """The __all__ export list contains exactly the 20 exception classes."""

    def test_all_has_twenty_entries(self) -> None:
        assert len(exceptions_module.__all__) == 20

    def test_all_names_are_importable(self) -> None:
        for name in exceptions_module.__all__:
            obj = getattr(exceptions_module, name)
            assert isinstance(obj, type), f"{name} should be a class"

    def test_all_names_are_sdk_error_subclasses(self) -> None:
        for name in exceptions_module.__all__:
            cls = getattr(exceptions_module, name)
            assert issubclass(cls, SDKError), (
                f"{name} should be a subclass of SDKError"
            )

    @pytest.mark.parametrize("name", EXPECTED_ALL_NAMES)
    def test_expected_name_in_all(self, name: str) -> None:
        assert name in exceptions_module.__all__, (
            f"{name} should be in __all__"
        )

    def test_no_extra_names(self) -> None:
        """__all__ should not contain any unexpected names."""
        all_set = set(exceptions_module.__all__)
        expected_set = set(EXPECTED_ALL_NAMES)
        assert all_set == expected_set


# ---------------------------------------------------------------------------
# Cross-branch isolation -- siblings should not catch each other
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

    def test_config_validation_not_caught_by_training(self) -> None:
        with pytest.raises(ConfigError):
            try:
                raise ConfigValidationError("bad field")
            except TrainingError:
                pytest.fail(
                    "ConfigValidationError should not be caught by TrainingError"
                )

    def test_search_timeout_not_caught_by_plugin(self) -> None:
        with pytest.raises(SearchError):
            try:
                raise SearchTimeoutError("timeout")
            except PluginError:
                pytest.fail("SearchTimeoutError should not be caught by PluginError")
