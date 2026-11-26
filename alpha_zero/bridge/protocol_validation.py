"""Validation helpers, enums, constants, and error types for Civ6 bridge protocol.

Layer 0 of the split bridge protocol (stdlib only, no sibling imports):
- :mod:`alpha_zero.bridge.protocol_validation` (Layer 0 — this module)
- :mod:`alpha_zero.bridge.protocol_entities` (Layer 1 — entities & map primitives)
- :mod:`alpha_zero.bridge.protocol` (Layer 2 — top-level messages & backwards-compatible facade)
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

__all__ = [
    "END_TURN_BLOCKED",
    "ERA_NAME_TO_INDEX",
    "LEGACY_STATE_SCHEMA_VERSION",
    "ActionType",
    "ErrorCode",
    "MessageType",
    "ProtocolValidationError",
]


class ProtocolValidationError(ValueError):
    """Exception raised when protocol data validation fails.

    Attributes:
        field_name: Name of the field that failed validation.
        expected: Description of expected value/type.
        actual: The actual value received.
    """

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        expected: str | None = None,
        actual: Any = None,
    ) -> None:
        self.field_name = field_name
        self.expected = expected
        self.actual = actual

        parts = [message]
        if field_name:
            parts.append(f"field='{field_name}'")
        if expected:
            parts.append(f"expected={expected}")
        if actual is not None:
            actual_repr = repr(actual) if len(repr(actual)) < 100 else f"{type(actual).__name__}"
            parts.append(f"got={actual_repr}")

        super().__init__(" | ".join(parts))


def _validate_required(data: dict[str, Any], field: str, context: str = "") -> Any:
    """Validate that a required field exists."""
    if field not in data:
        ctx = f" in {context}" if context else ""
        raise ProtocolValidationError(
            f"Missing required field{ctx}",
            field_name=field,
            expected="present",
            actual="missing",
        )
    return data[field]


def _validate_type(
    value: Any,
    expected_types: type | tuple[type, ...],
    field: str,
    context: str = "",
    allow_none: bool = False,
) -> Any:
    """Validate that a value has the expected type."""
    if allow_none and value is None:
        return value

    if not isinstance(value, expected_types):
        type_names = (
            expected_types.__name__
            if isinstance(expected_types, type)
            else "|".join(t.__name__ for t in expected_types)
        )
        ctx = f" in {context}" if context else ""
        raise ProtocolValidationError(
            f"Invalid type{ctx}",
            field_name=field,
            expected=type_names,
            actual=value,
        )
    return value


def _validate_int(
    value: Any,
    field: str,
    context: str = "",
    default: int | None = None,
    min_val: int | None = None,
    max_val: int | None = None,
) -> int:
    """Validate and convert to integer."""
    if value is None:
        if default is not None:
            return default
        raise ProtocolValidationError(
            "Missing required integer",
            field_name=field,
            expected="int",
            actual=None,
        )

    try:
        int_val = int(value)
    except (TypeError, ValueError) as err:
        ctx = f" in {context}" if context else ""
        raise ProtocolValidationError(
            f"Cannot convert to integer{ctx}",
            field_name=field,
            expected="int-convertible",
            actual=value,
        ) from err

    if min_val is not None and int_val < min_val:
        raise ProtocolValidationError(
            "Value below minimum",
            field_name=field,
            expected=f">={min_val}",
            actual=int_val,
        )

    if max_val is not None and int_val > max_val:
        raise ProtocolValidationError(
            "Value above maximum",
            field_name=field,
            expected=f"<={max_val}",
            actual=int_val,
        )

    return int_val


def _validate_float(
    value: Any,
    field: str,
    context: str = "",
    default: float | None = None,
) -> float:
    """Validate and convert to float."""
    if value is None:
        if default is not None:
            return default
        raise ProtocolValidationError(
            "Missing required float",
            field_name=field,
            expected="float",
            actual=None,
        )

    try:
        return float(value)
    except (TypeError, ValueError) as err:
        ctx = f" in {context}" if context else ""
        raise ProtocolValidationError(
            f"Cannot convert to float{ctx}",
            field_name=field,
            expected="float-convertible",
            actual=value,
        ) from err


def _validate_string(
    value: Any,
    field: str,
    context: str = "",
    default: str | None = None,
    allowed_values: list[str] | None = None,
) -> str:
    """Validate string value."""
    if value is None:
        if default is not None:
            return default
        raise ProtocolValidationError(
            "Missing required string",
            field_name=field,
            expected="str",
            actual=None,
        )

    str_val = str(value)

    if allowed_values is not None and str_val not in allowed_values:
        raise ProtocolValidationError(
            "Value not in allowed set",
            field_name=field,
            expected=f"one of {allowed_values[:5]}{'...' if len(allowed_values) > 5 else ''}",
            actual=str_val,
        )

    return str_val


def _validate_list(
    value: Any,
    field: str,
    context: str = "",
    default_factory: Callable[[], list] = list,
) -> list[Any]:
    """Validate list value."""
    if value is None:
        return default_factory()

    if not isinstance(value, list):
        ctx = f" in {context}" if context else ""
        raise ProtocolValidationError(
            f"Expected list{ctx}",
            field_name=field,
            expected="list",
            actual=value,
        )

    return value


def _validate_dict(
    value: Any,
    field: str,
    context: str = "",
    default_factory: Callable[[], dict] = dict,
) -> dict[str, Any]:
    """Validate dict value."""
    if value is None:
        return default_factory()

    # In Lua, empty tables {} with no string keys are encoded as [] in JSON.
    if isinstance(value, list) and len(value) == 0:
        return default_factory()

    if not isinstance(value, dict):
        ctx = f" in {context}" if context else ""
        raise ProtocolValidationError(
            f"Expected dict{ctx}",
            field_name=field,
            expected="dict",
            actual=value,
        )

    return value


def _optional_text(value: Any) -> str | None:
    """A non-empty string, or None."""
    return value if isinstance(value, str) and value else None


def _optional_number(value: Any) -> float | None:
    """A JSON number as float; anything else (bool included) as None."""
    return None if isinstance(value, bool) or not isinstance(value, (int, float)) else float(value)


def _optional_int(value: Any) -> int | None:
    """A whole JSON number as int; anything else (bool included) as None."""
    number = _optional_number(value)
    if number is None or number != int(number):
        return None
    return int(number)


class MessageType(str, Enum):
    """Protocol message types."""

    STATE = "state"  # Game state from Lua
    COMMAND = "command"  # Action command to Lua
    ACK = "ack"  # Command acknowledgment from Lua
    ERROR = "error"  # Error report
    HEARTBEAT = "heartbeat"  # Liveness check


class ActionType(str, Enum):
    """Supported action types for Lua commands.

    Values MUST match the keys in ``mod/AlphaZeroAI/Scripts/ActionExecutor.lua``
    ``ACTION_HANDLERS`` table (and the ``ACTION_TYPES`` constants there).
    See ``tests/unit/test_bridge_action_invariant.py`` for the runtime check
    that the Python enum and the Lua dispatch table stay in sync (v1.8.0
    W2.1 step 6).
    """

    END_TURN = "end_turn"
    UNIT_MOVE = "unit_move"
    UNIT_ATTACK = "unit_attack"
    UNIT_RANGED_ATTACK = "unit_ranged_attack"
    UNIT_FORTIFY = "unit_fortify"
    UNIT_SKIP = "unit_skip"
    UNIT_SLEEP = "unit_sleep"
    UNIT_FOUND_CITY = "unit_found_city"
    UNIT_BUILD = "unit_build"
    UNIT_PROMOTE = "unit_promote"  # v1.8.0 W2.1
    UNIT_DELETE = "unit_delete"  # v1.8.0 W2.1
    CITY_PRODUCE = "city_produce"
    CITY_BUY = "city_buy"
    CITY_CITIZEN = "city_citizen"  # v1.8.0 W2.1
    RESEARCH_TECH = "research_tech"
    RESEARCH_CIVIC = "research_civic"
    DIPLOMACY = "diplomacy"  # v1.8.0 W2.1 (dispatcher; sub-action in metadata)


#: Error code the Lua executor gives ``end_turn`` while a notification blocks
#: ending the turn (``ERROR_CODES.END_TURN_BLOCKED`` in ``ActionExecutor.lua``).
#: It arrives per command, in ``AckMessage.executed``.
END_TURN_BLOCKED = "END_TURN_BLOCKED"

#: ``schema_version`` of a state payload that does not carry the key.
LEGACY_STATE_SCHEMA_VERSION = 1


class ErrorCode:
    """Standard error codes for the bridge protocol."""

    UNKNOWN = "E000"
    INVALID_ACTION = "E001"
    STATE_UNAVAILABLE = "E002"
    UNIT_NOT_FOUND = "E003"
    CITY_NOT_FOUND = "E004"
    FILE_IO_ERROR = "E005"
    SERIALIZATION_ERROR = "E006"
    TIMEOUT = "E007"
    CONNECTION_LOST = "E008"
    INVALID_STATE = "E009"
    GAME_OVER = "E010"


ERA_NAME_TO_INDEX: dict[str, int] = {
    "ERA_ANCIENT": 0,
    "ERA_CLASSICAL": 1,
    "ERA_MEDIEVAL": 2,
    "ERA_RENAISSANCE": 3,
    "ERA_INDUSTRIAL": 4,
    "ERA_MODERN": 5,
    "ERA_ATOMIC": 6,
    "ERA_INFORMATION": 7,
    "ERA_FUTURE": 8,
}


def _parse_era(era_value: int | str | None, default: int = 0) -> int:
    """Parse era value from int or string."""
    if era_value is None:
        return default
    if isinstance(era_value, int):
        return era_value
    if isinstance(era_value, str):
        return ERA_NAME_TO_INDEX.get(era_value, default)
    return default
