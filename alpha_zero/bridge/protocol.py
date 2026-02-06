"""Protocol definitions for Civ6 bridge communication.

This module defines message types, schemas, and serialization for the
file-based IPC protocol between Python and the Civ6 Lua mod.

Message Flow:
    1. Lua writes state.json when player turn starts
    2. Python reads state, processes with neural network
    3. Python writes commands.json with selected action
    4. Lua reads commands, executes action
    5. Lua writes ack.json to confirm execution

Usage:
    from alpha_zero.bridge.protocol import (
        MessageType,
        StateMessage,
        CommandMessage,
        AckMessage,
    )

    # Parse state from Lua
    state = StateMessage.from_json(json_str)

    # Create command for Lua
    cmd = CommandMessage(
        msg_id=1,
        commands=[LuaCommand(action_type="unit_move", ...)]
    )
    json_str = cmd.to_json()
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


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

        # Build detailed message
        parts = [message]
        if field_name:
            parts.append(f"field='{field_name}'")
        if expected:
            parts.append(f"expected={expected}")
        if actual is not None:
            actual_repr = repr(actual) if len(repr(actual)) < 100 else f"{type(actual).__name__}"
            parts.append(f"got={actual_repr}")

        super().__init__(" | ".join(parts))


def _validate_required(data: dict, field: str, context: str = "") -> Any:
    """Validate that a required field exists.

    Args:
        data: Dictionary to check.
        field: Field name to validate.
        context: Context string for error message.

    Returns:
        The field value.

    Raises:
        ProtocolValidationError: If field is missing.
    """
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
    """Validate that a value has the expected type.

    Args:
        value: Value to validate.
        expected_types: Expected type or tuple of types.
        field: Field name for error message.
        context: Context string for error message.
        allow_none: Whether None is acceptable.

    Returns:
        The value if valid.

    Raises:
        ProtocolValidationError: If type is invalid.
    """
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
    """Validate and convert to integer.

    Args:
        value: Value to validate.
        field: Field name for error message.
        context: Context string for error message.
        default: Default value if value is None.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        Integer value.

    Raises:
        ProtocolValidationError: If validation fails.
    """
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
    """Validate and convert to float.

    Args:
        value: Value to validate.
        field: Field name for error message.
        context: Context string for error message.
        default: Default value if value is None.

    Returns:
        Float value.

    Raises:
        ProtocolValidationError: If validation fails.
    """
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
    """Validate string value.

    Args:
        value: Value to validate.
        field: Field name for error message.
        context: Context string for error message.
        default: Default value if value is None.
        allowed_values: List of allowed string values.

    Returns:
        String value.

    Raises:
        ProtocolValidationError: If validation fails.
    """
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
) -> list:
    """Validate list value.

    Args:
        value: Value to validate.
        field: Field name for error message.
        context: Context string for error message.
        default_factory: Factory to create default list.

    Returns:
        List value.

    Raises:
        ProtocolValidationError: If validation fails.
    """
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
) -> dict:
    """Validate dict value.

    Args:
        value: Value to validate.
        field: Field name for error message.
        context: Context string for error message.
        default_factory: Factory to create default dict.

    Returns:
        Dict value.

    Raises:
        ProtocolValidationError: If validation fails.
    """
    if value is None:
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


class MessageType(str, Enum):
    """Protocol message types."""

    STATE = "state"  # Game state from Lua
    COMMAND = "command"  # Action command to Lua
    ACK = "ack"  # Command acknowledgment from Lua
    ERROR = "error"  # Error report
    HEARTBEAT = "heartbeat"  # Liveness check


class ActionType(str, Enum):
    """Supported action types for Lua commands."""

    END_TURN = "end_turn"
    UNIT_MOVE = "unit_move"
    UNIT_ATTACK = "unit_attack"
    UNIT_RANGED_ATTACK = "unit_ranged_attack"
    UNIT_FORTIFY = "unit_fortify"
    UNIT_SKIP = "unit_skip"
    UNIT_FOUND_CITY = "unit_found_city"
    UNIT_BUILD = "unit_build"
    CITY_PRODUCE = "city_produce"
    CITY_BUY = "city_buy"
    RESEARCH_TECH = "research_tech"
    RESEARCH_CIVIC = "research_civic"


@dataclass
class Position:
    """A position on the game map."""

    x: int
    y: int

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict) -> Position:
        """Create from dictionary.

        Args:
            data: Dictionary with 'x' and 'y' keys.

        Returns:
            Position instance.

        Raises:
            ProtocolValidationError: If required fields are missing or invalid.
        """
        _validate_type(data, dict, "data", context="Position.from_dict")
        x = _validate_int(data.get("x"), "x", context="Position")
        y = _validate_int(data.get("y"), "y", context="Position")
        return cls(x=x, y=y)


@dataclass
class LuaCommand:
    """A command to be executed by the Lua mod.

    Attributes:
        action_type: Type of action to execute.
        unit_id: Unit ID for unit actions.
        city_id: City ID for city actions.
        target: Target position for movement/attack.
        item_id: Item ID for production/research.
        item_type: Type of item (unit, building, wonder, district).
        metadata: Additional action-specific data.
    """

    action_type: ActionType
    unit_id: int | None = None
    city_id: int | None = None
    target: Position | None = None
    item_id: int | None = None
    item_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"action_type": self.action_type.value}

        if self.unit_id is not None:
            result["unit_id"] = self.unit_id
        if self.city_id is not None:
            result["city_id"] = self.city_id
        if self.target is not None:
            result["target"] = self.target.to_dict()
        if self.item_id is not None:
            result["item_id"] = self.item_id
        if self.item_type is not None:
            result["item_type"] = self.item_type
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LuaCommand:
        """Create from dictionary.

        Args:
            data: Dictionary with command data.

        Returns:
            LuaCommand instance.

        Raises:
            ProtocolValidationError: If required fields are missing or invalid.
        """
        _validate_type(data, dict, "data", context="LuaCommand.from_dict")

        # Validate and parse action_type
        action_type_raw = _validate_required(data, "action_type", context="LuaCommand")
        try:
            action_type = ActionType(action_type_raw)
        except ValueError as err:
            valid_actions = [a.value for a in ActionType]
            raise ProtocolValidationError(
                "Invalid action type",
                field_name="action_type",
                expected=f"one of {valid_actions}",
                actual=action_type_raw,
            ) from err

        # Parse optional target
        target = None
        if "target" in data and data["target"] is not None:
            target_data = _validate_type(
                data["target"], dict, "target", context="LuaCommand", allow_none=True
            )
            if target_data:
                target = Position.from_dict(target_data)

        # Validate optional IDs are integers if present
        unit_id = data.get("unit_id")
        if unit_id is not None:
            unit_id = _validate_int(unit_id, "unit_id", context="LuaCommand")

        city_id = data.get("city_id")
        if city_id is not None:
            city_id = _validate_int(city_id, "city_id", context="LuaCommand")

        item_id = data.get("item_id")
        if item_id is not None:
            item_id = _validate_int(item_id, "item_id", context="LuaCommand")

        metadata = _validate_dict(data.get("metadata"), "metadata", context="LuaCommand")

        return cls(
            action_type=action_type,
            unit_id=unit_id,
            city_id=city_id,
            target=target,
            item_id=item_id,
            item_type=data.get("item_type"),
            metadata=metadata,
        )


@dataclass
class TileData:
    """Data for a single map tile."""

    x: int
    y: int
    terrain: str
    features: list[str] = field(default_factory=list)
    resource: str | None = None
    improvement: str | None = None
    owner_id: int | None = None
    is_visible: bool = True
    is_revealed: bool = True
    yields: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TileData:
        """Create from dictionary.

        Args:
            data: Dictionary with tile data.

        Returns:
            TileData instance.

        Raises:
            ProtocolValidationError: If data is not a dictionary.
        """
        _validate_type(data, dict, "data", context="TileData.from_dict")

        # Parse coordinates with defaults
        x = _validate_int(data.get("x"), "x", context="TileData", default=0)
        y = _validate_int(data.get("y"), "y", context="TileData", default=0)

        # Handle both 'owner' and 'owner_id' keys (consistent with UnitData/CityData)
        raw_owner = data.get("owner", data.get("owner_id", -1))
        owner_id = None
        if raw_owner is not None and raw_owner != -1:
            owner_id = _validate_int(raw_owner, "owner_id", context="TileData", default=-1)
            if owner_id == -1:
                owner_id = None

        # Validate features list
        features = _validate_list(data.get("features"), "features", context="TileData")

        # Validate yields dict
        yields = _validate_dict(data.get("yields"), "yields", context="TileData")

        return cls(
            x=x,
            y=y,
            terrain=_validate_string(
                data.get("terrain"), "terrain", context="TileData", default="TERRAIN_PLAINS"
            ),
            features=features,
            resource=data.get("resource"),
            improvement=data.get("improvement"),
            owner_id=owner_id,
            is_visible=bool(data.get("is_visible", True)),
            is_revealed=bool(data.get("is_revealed", True)),
            yields=yields,
        )


@dataclass
class UnitData:
    """Data for a game unit."""

    id: int
    unit_type: str
    owner_id: int
    x: int
    y: int
    health: int = 100
    max_health: int = 100
    damage: int = 0
    moves_remaining: float = 0.0
    max_moves: float = 2.0
    combat_strength: int = 0
    ranged_strength: int = 0
    range: int = 0
    is_civilian: bool = False
    can_attack: bool = True
    promotions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnitData:
        """Create from dictionary.

        Args:
            data: Dictionary with unit data.

        Returns:
            UnitData instance.

        Raises:
            ProtocolValidationError: If data is not a dictionary.
        """
        _validate_type(data, dict, "data", context="UnitData.from_dict")

        # Parse required and optional fields with validation
        unit_id = _validate_int(data.get("id"), "id", context="UnitData", default=0)
        unit_type = _validate_string(
            data.get("type", data.get("type_name")),
            "unit_type",
            context="UnitData",
            default="UNIT_WARRIOR",
        )
        owner_id = _validate_int(
            data.get("owner", data.get("owner_id")),
            "owner_id",
            context="UnitData",
            default=0,
        )
        x = _validate_int(data.get("x"), "x", context="UnitData", default=0)
        y = _validate_int(data.get("y"), "y", context="UnitData", default=0)
        health = _validate_int(
            data.get("hp", data.get("health")),
            "health",
            context="UnitData",
            default=100,
            min_val=0,
        )
        max_health = _validate_int(
            data.get("max_hp", data.get("max_health")),
            "max_health",
            context="UnitData",
            default=100,
            min_val=1,
        )
        damage = _validate_int(
            data.get("damage"), "damage", context="UnitData", default=0, min_val=0
        )
        moves_remaining = _validate_float(
            data.get("movement", data.get("moves_remaining")),
            "moves_remaining",
            context="UnitData",
            default=0.0,
        )
        max_moves = _validate_float(
            data.get("max_movement", data.get("max_moves")),
            "max_moves",
            context="UnitData",
            default=2.0,
        )
        combat_strength = _validate_int(
            data.get("combat_strength"), "combat_strength", context="UnitData", default=0
        )
        ranged_strength = _validate_int(
            data.get("ranged_strength"), "ranged_strength", context="UnitData", default=0
        )
        range_val = _validate_int(data.get("range"), "range", context="UnitData", default=0)
        promotions = _validate_list(data.get("promotions"), "promotions", context="UnitData")

        return cls(
            id=unit_id,
            unit_type=unit_type,
            owner_id=owner_id,
            x=x,
            y=y,
            health=health,
            max_health=max_health,
            damage=damage,
            moves_remaining=moves_remaining,
            max_moves=max_moves,
            combat_strength=combat_strength,
            ranged_strength=ranged_strength,
            range=range_val,
            is_civilian=bool(data.get("is_civilian", False)),
            can_attack=bool(data.get("can_attack", True)),
            promotions=promotions,
        )

    @property
    def position(self) -> tuple[int, int]:
        """Get position as tuple."""
        return (self.x, self.y)


@dataclass
class CityData:
    """Data for a city."""

    id: int
    name: str
    owner_id: int
    x: int
    y: int
    population: int = 1
    is_capital: bool = False
    defense_strength: int = 0
    buildings: list[str] = field(default_factory=list)
    districts: list[str] = field(default_factory=list)
    current_production: str | None = None
    production_progress: int = 0
    available_production: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CityData:
        """Create from dictionary.

        Args:
            data: Dictionary with city data.

        Returns:
            CityData instance.

        Raises:
            ProtocolValidationError: If data is not a dictionary.
        """
        _validate_type(data, dict, "data", context="CityData.from_dict")

        city_id = _validate_int(data.get("id"), "id", context="CityData", default=0)
        name = _validate_string(data.get("name"), "name", context="CityData", default="Unknown")
        owner_id = _validate_int(
            data.get("owner", data.get("owner_id")),
            "owner_id",
            context="CityData",
            default=0,
        )
        x = _validate_int(data.get("x"), "x", context="CityData", default=0)
        y = _validate_int(data.get("y"), "y", context="CityData", default=0)
        population = _validate_int(
            data.get("population"), "population", context="CityData", default=1, min_val=1
        )
        defense_strength = _validate_int(
            data.get("defense_strength", data.get("defense")),
            "defense_strength",
            context="CityData",
            default=0,
        )
        production_progress = _validate_int(
            data.get("production_progress"),
            "production_progress",
            context="CityData",
            default=0,
        )
        buildings = _validate_list(data.get("buildings"), "buildings", context="CityData")
        districts = _validate_list(data.get("districts"), "districts", context="CityData")
        available_production = _validate_list(
            data.get("available_production"), "available_production", context="CityData"
        )

        return cls(
            id=city_id,
            name=name,
            owner_id=owner_id,
            x=x,
            y=y,
            population=population,
            is_capital=bool(data.get("is_capital", False)),
            defense_strength=defense_strength,
            buildings=buildings,
            districts=districts,
            current_production=data.get("current_production"),
            production_progress=production_progress,
            available_production=available_production,
        )

    @property
    def position(self) -> tuple[int, int]:
        """Get position as tuple."""
        return (self.x, self.y)


@dataclass
class PlayerData:
    """Data for a player."""

    id: int
    civilization: str
    gold: float = 0.0
    science_per_turn: float = 0.0
    culture_per_turn: float = 0.0
    faith: float = 0.0
    score: int = 0
    is_alive: bool = True
    is_human: bool = True
    techs_researched: list[str] = field(default_factory=list)
    current_research: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayerData:
        """Create from dictionary.

        Args:
            data: Dictionary with player data.

        Returns:
            PlayerData instance.

        Raises:
            ProtocolValidationError: If data is not a dictionary.
        """
        _validate_type(data, dict, "data", context="PlayerData.from_dict")

        player_id = _validate_int(data.get("id"), "id", context="PlayerData", default=0)
        civilization = _validate_string(
            data.get("civilization", data.get("civ")),
            "civilization",
            context="PlayerData",
            default="UNKNOWN",
        )
        gold = _validate_float(data.get("gold"), "gold", context="PlayerData", default=0.0)
        science_per_turn = _validate_float(
            data.get("science_per_turn", data.get("science")),
            "science_per_turn",
            context="PlayerData",
            default=0.0,
        )
        culture_per_turn = _validate_float(
            data.get("culture_per_turn", data.get("culture")),
            "culture_per_turn",
            context="PlayerData",
            default=0.0,
        )
        faith = _validate_float(data.get("faith"), "faith", context="PlayerData", default=0.0)
        score = _validate_int(data.get("score"), "score", context="PlayerData", default=0)
        techs_researched = _validate_list(
            data.get("techs_researched"), "techs_researched", context="PlayerData"
        )

        return cls(
            id=player_id,
            civilization=civilization,
            gold=gold,
            science_per_turn=science_per_turn,
            culture_per_turn=culture_per_turn,
            faith=faith,
            score=score,
            is_alive=bool(data.get("is_alive", True)),
            is_human=bool(data.get("is_human", True)),
            techs_researched=techs_researched,
            current_research=data.get("current_research"),
        )


@dataclass
class LegalAction:
    """A legal action from the Lua mod.

    Attributes:
        action_type: Type of action (e.g., "move", "attack", "produce").
        unit_id: Unit ID for unit-related actions.
        city_id: City ID for city-related actions.
        targets: Valid target positions as (x, y) tuples.
        items: Item identifiers (int for production IDs, str for tech/civic names).
        item_type: Type of item (e.g., "unit", "building", "tech").
    """

    action_type: str
    unit_id: int | None = None
    city_id: int | None = None
    targets: list[tuple[int, int]] = field(default_factory=list)
    items: list[int | str] = field(default_factory=list)
    item_type: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LegalAction:
        """Create from dictionary.

        Args:
            data: Dictionary with legal action data.

        Returns:
            LegalAction instance.

        Raises:
            ProtocolValidationError: If data is not a dictionary.
        """
        _validate_type(data, dict, "data", context="LegalAction.from_dict")

        # Parse targets
        targets = []
        raw_targets = data.get("targets", [])
        if raw_targets is not None:
            _validate_type(raw_targets, list, "targets", context="LegalAction")
            for i, t in enumerate(raw_targets):
                if isinstance(t, dict):
                    x = _validate_int(
                        t.get("x"), f"targets[{i}].x", context="LegalAction", default=0
                    )
                    y = _validate_int(
                        t.get("y"), f"targets[{i}].y", context="LegalAction", default=0
                    )
                    targets.append((x, y))
                elif isinstance(t, (list, tuple)) and len(t) >= 2:
                    x = _validate_int(t[0], f"targets[{i}][0]", context="LegalAction", default=0)
                    y = _validate_int(t[1], f"targets[{i}][1]", context="LegalAction", default=0)
                    targets.append((x, y))
                else:
                    logger.warning(f"Skipping invalid target format at index {i}: {t}")

        # Parse items - can be int IDs or string names (e.g., tech names)
        raw_items = data.get("items", data.get("item", []))
        if raw_items is None:
            raw_items = []
        elif not isinstance(raw_items, list):
            raw_items = [raw_items]  # Wrap single item in list

        # Convert numeric strings to int, keep non-numeric strings as-is
        items: list[int | str] = []
        for item in raw_items:
            if isinstance(item, int):
                items.append(item)
            elif isinstance(item, str):
                # Try to convert numeric strings to int
                if item.isdigit() or (item.startswith("-") and item[1:].isdigit()):
                    items.append(int(item))
                else:
                    items.append(item)  # Keep as string (e.g., tech name)
            else:
                logger.warning(f"Skipping invalid item type: {type(item).__name__}")

        # Validate unit_id and city_id if present
        unit_id = data.get("unit_id")
        if unit_id is not None:
            unit_id = _validate_int(unit_id, "unit_id", context="LegalAction")

        city_id = data.get("city_id")
        if city_id is not None:
            city_id = _validate_int(city_id, "city_id", context="LegalAction")

        action_type = _validate_string(
            data.get("action", data.get("type")),
            "action_type",
            context="LegalAction",
            default="end_turn",
        )

        return cls(
            action_type=action_type,
            unit_id=unit_id,
            city_id=city_id,
            targets=targets,
            items=items,
            item_type=data.get("item_type"),
        )


@dataclass
class MapData:
    """Data for the game map."""

    width: int
    height: int
    tiles: list[TileData] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MapData:
        """Create from dictionary.

        Args:
            data: Dictionary with map data.

        Returns:
            MapData instance.

        Raises:
            ProtocolValidationError: If data is not a dictionary.
        """
        _validate_type(data, dict, "data", context="MapData.from_dict")

        width = _validate_int(data.get("width"), "width", context="MapData", default=0, min_val=0)
        height = _validate_int(
            data.get("height"), "height", context="MapData", default=0, min_val=0
        )

        raw_tiles = _validate_list(data.get("tiles"), "tiles", context="MapData")
        tiles = []
        for i, t in enumerate(raw_tiles):
            try:
                tiles.append(TileData.from_dict(t))
            except ProtocolValidationError as e:
                logger.warning(f"Skipping invalid tile at index {i}: {e}")

        return cls(width=width, height=height, tiles=tiles)


@dataclass
class GameStateData:
    """Complete game state from Lua mod."""

    turn_id: int
    player_id: int
    map_data: MapData
    units: list[UnitData]
    cities: list[CityData]
    players: list[PlayerData]
    legal_actions: list[LegalAction]
    current_player_civ: str = ""
    era: int = 0
    game_speed: str = "GAMESPEED_STANDARD"
    is_game_over: bool = False
    winner_id: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameStateData:
        """Create from dictionary.

        Args:
            data: Dictionary with game state data.

        Returns:
            GameStateData instance.

        Raises:
            ProtocolValidationError: If data is not a dictionary.
        """
        _validate_type(data, dict, "data", context="GameStateData.from_dict")

        # Handle the 'player' field which contains summary data
        player_summary = _validate_dict(data.get("player"), "player", context="GameStateData")

        # Build player data from summary if no players array
        players_data = _validate_list(data.get("players"), "players", context="GameStateData")
        if not players_data and player_summary:
            players_data = [player_summary]

        # Parse era from top-level or player summary (handles both int and string)
        era_value = data.get("era", player_summary.get("era"))
        era = _parse_era(era_value, default=0)

        # Parse turn_id
        turn_id = _validate_int(
            data.get("turn_id", data.get("turn")),
            "turn_id",
            context="GameStateData",
            default=0,
        )

        # Parse player_id
        player_id = _validate_int(
            data.get("player_id", player_summary.get("id")),
            "player_id",
            context="GameStateData",
            default=0,
        )

        # Parse map with validation
        map_dict = _validate_dict(data.get("map"), "map", context="GameStateData")
        map_data = MapData.from_dict(map_dict)

        # Parse units with error handling
        raw_units = _validate_list(data.get("units"), "units", context="GameStateData")
        units = []
        for i, u in enumerate(raw_units):
            try:
                units.append(UnitData.from_dict(u))
            except ProtocolValidationError as e:
                logger.warning(f"Skipping invalid unit at index {i}: {e}")

        # Parse cities with error handling
        raw_cities = _validate_list(data.get("cities"), "cities", context="GameStateData")
        cities = []
        for i, c in enumerate(raw_cities):
            try:
                cities.append(CityData.from_dict(c))
            except ProtocolValidationError as e:
                logger.warning(f"Skipping invalid city at index {i}: {e}")

        # Parse players with error handling
        players = []
        for i, p in enumerate(players_data):
            try:
                players.append(PlayerData.from_dict(p))
            except ProtocolValidationError as e:
                logger.warning(f"Skipping invalid player at index {i}: {e}")

        # Parse legal actions with error handling
        raw_legal_actions = _validate_list(
            data.get("legal_actions"), "legal_actions", context="GameStateData"
        )
        legal_actions = []
        for i, a in enumerate(raw_legal_actions):
            try:
                legal_actions.append(LegalAction.from_dict(a))
            except ProtocolValidationError as e:
                logger.warning(f"Skipping invalid legal action at index {i}: {e}")

        # Parse winner_id if present
        winner_id = data.get("winner_id")
        if winner_id is not None:
            winner_id = _validate_int(winner_id, "winner_id", context="GameStateData")

        return cls(
            turn_id=turn_id,
            player_id=player_id,
            map_data=map_data,
            units=units,
            cities=cities,
            players=players,
            legal_actions=legal_actions,
            current_player_civ=_validate_string(
                data.get("current_player"), "current_player", context="GameStateData", default=""
            ),
            era=era,
            game_speed=_validate_string(
                data.get("game_speed"),
                "game_speed",
                context="GameStateData",
                default="GAMESPEED_STANDARD",
            ),
            is_game_over=bool(data.get("is_game_over", data.get("game_over", False))),
            winner_id=winner_id,
        )


@dataclass
class StateMessage:
    """State message from Lua mod.

    Attributes:
        msg_id: Unique message identifier.
        timestamp: Unix timestamp when message was created.
        state: Complete game state data.
    """

    msg_id: int
    timestamp: float
    state: GameStateData

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "msg_id": self.msg_id,
                "type": MessageType.STATE.value,
                "timestamp": self.timestamp,
                "state": self._state_to_dict(),
            }
        )

    def _state_to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary (simplified for serialization)."""
        return {
            "turn_id": self.state.turn_id,
            "player_id": self.state.player_id,
            "map": {
                "width": self.state.map_data.width,
                "height": self.state.map_data.height,
            },
            "units_count": len(self.state.units),
            "cities_count": len(self.state.cities),
            "legal_actions_count": len(self.state.legal_actions),
        }

    @classmethod
    def from_json(cls, json_str: str) -> StateMessage:
        """Parse from JSON string.

        Args:
            json_str: JSON string to parse.

        Returns:
            StateMessage instance.

        Raises:
            ProtocolValidationError: If JSON is invalid or data is malformed.
            json.JSONDecodeError: If JSON parsing fails.
        """
        _validate_type(json_str, str, "json_str", context="StateMessage.from_json")
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ProtocolValidationError(
                f"Invalid JSON: {e}",
                field_name="json_str",
                expected="valid JSON string",
            ) from e
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateMessage:
        """Create from dictionary.

        Args:
            data: Dictionary with state message data.

        Returns:
            StateMessage instance.

        Raises:
            ProtocolValidationError: If data is not a dictionary.
        """
        _validate_type(data, dict, "data", context="StateMessage.from_dict")

        # Handle both wrapped and unwrapped formats
        if "state" in data:
            state_data = _validate_type(data["state"], dict, "state", context="StateMessage")
        else:
            # The entire dict is the state (direct Lua output)
            state_data = data

        msg_id = _validate_int(
            data.get("msg_id"),
            "msg_id",
            context="StateMessage",
            default=int(time.time() * 1000),
        )
        timestamp = _validate_float(
            data.get("timestamp"),
            "timestamp",
            context="StateMessage",
            default=time.time(),
        )

        return cls(
            msg_id=msg_id,
            timestamp=timestamp,
            state=GameStateData.from_dict(state_data),
        )


@dataclass
class CommandMessage:
    """Command message to Lua mod.

    Attributes:
        msg_id: Unique message identifier (for ack correlation).
        timestamp: Unix timestamp when message was created.
        commands: List of commands to execute.
    """

    msg_id: int
    commands: list[LuaCommand]
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "msg_id": self.msg_id,
                "type": MessageType.COMMAND.value,
                "timestamp": self.timestamp,
                "commands": [cmd.to_dict() for cmd in self.commands],
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> CommandMessage:
        """Parse from JSON string.

        Args:
            json_str: JSON string to parse.

        Returns:
            CommandMessage instance.

        Raises:
            ProtocolValidationError: If JSON is invalid or data is malformed.
            json.JSONDecodeError: If JSON parsing fails.
        """
        _validate_type(json_str, str, "json_str", context="CommandMessage.from_json")
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ProtocolValidationError(
                f"Invalid JSON: {e}",
                field_name="json_str",
                expected="valid JSON string",
            ) from e

        _validate_type(data, dict, "data", context="CommandMessage.from_json")

        # msg_id is required for command messages
        msg_id = _validate_int(
            _validate_required(data, "msg_id", context="CommandMessage"),
            "msg_id",
            context="CommandMessage",
        )
        timestamp = _validate_float(
            data.get("timestamp"),
            "timestamp",
            context="CommandMessage",
            default=time.time(),
        )

        # Parse commands with error handling
        raw_commands = _validate_list(data.get("commands"), "commands", context="CommandMessage")
        commands = []
        for i, c in enumerate(raw_commands):
            try:
                commands.append(LuaCommand.from_dict(c))
            except ProtocolValidationError as e:
                logger.warning(f"Skipping invalid command at index {i}: {e}")

        return cls(msg_id=msg_id, timestamp=timestamp, commands=commands)


@dataclass
class AckMessage:
    """Acknowledgment message from Lua mod.

    Attributes:
        msg_id: ID of the command being acknowledged.
        timestamp: Unix timestamp when ack was created.
        success: Whether the command was executed successfully.
        error_code: Error code if not successful.
        error_message: Human-readable error message.
        executed: List of executed command results.
    """

    msg_id: int
    success: bool
    timestamp: float = field(default_factory=time.time)
    error_code: str | None = None
    error_message: str | None = None
    executed: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        result = {
            "msg_id": self.msg_id,
            "type": MessageType.ACK.value,
            "timestamp": self.timestamp,
            "success": self.success,
            "status": "success" if self.success else "error",
        }
        if self.error_code:
            result["error_code"] = self.error_code
        if self.error_message:
            result["error_message"] = self.error_message
        if self.executed:
            result["executed"] = self.executed
        return json.dumps(result)

    @classmethod
    def from_json(cls, json_str: str) -> AckMessage:
        """Parse from JSON string.

        Args:
            json_str: JSON string to parse.

        Returns:
            AckMessage instance.

        Raises:
            ProtocolValidationError: If JSON is invalid or data is malformed.
            json.JSONDecodeError: If JSON parsing fails.
        """
        _validate_type(json_str, str, "json_str", context="AckMessage.from_json")
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ProtocolValidationError(
                f"Invalid JSON: {e}",
                field_name="json_str",
                expected="valid JSON string",
            ) from e
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AckMessage:
        """Create from dictionary.

        Args:
            data: Dictionary with ack message data.

        Returns:
            AckMessage instance.

        Raises:
            ProtocolValidationError: If data is not a dictionary.
        """
        _validate_type(data, dict, "data", context="AckMessage.from_dict")

        # Parse success from 'success' field or 'status' field
        success_raw = data.get("success")
        if success_raw is not None:
            success = bool(success_raw)
        else:
            success = data.get("status") == "success"

        msg_id = _validate_int(data.get("msg_id"), "msg_id", context="AckMessage", default=0)
        timestamp = _validate_float(
            data.get("timestamp"),
            "timestamp",
            context="AckMessage",
            default=time.time(),
        )
        executed = _validate_list(data.get("executed"), "executed", context="AckMessage")

        return cls(
            msg_id=msg_id,
            timestamp=timestamp,
            success=success,
            error_code=data.get("error_code", data.get("code")),
            error_message=data.get("error_message", data.get("message")),
            executed=executed,
        )


@dataclass
class ErrorMessage:
    """Error message from Lua mod or Python.

    Attributes:
        msg_id: Related message ID (if applicable).
        timestamp: Unix timestamp when error occurred.
        error_code: Error code for programmatic handling.
        message: Human-readable error message.
        details: Additional error details.
    """

    error_code: str
    message: str
    msg_id: int | None = None
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "type": MessageType.ERROR.value,
                "msg_id": self.msg_id,
                "timestamp": self.timestamp,
                "error_code": self.error_code,
                "message": self.message,
                "details": self.details,
            }
        )

    @classmethod
    def from_json(cls, json_str: str) -> ErrorMessage:
        """Parse from JSON string.

        Args:
            json_str: JSON string to parse.

        Returns:
            ErrorMessage instance.

        Raises:
            ProtocolValidationError: If JSON is invalid or data is malformed.
            json.JSONDecodeError: If JSON parsing fails.
        """
        _validate_type(json_str, str, "json_str", context="ErrorMessage.from_json")
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ProtocolValidationError(
                f"Invalid JSON: {e}",
                field_name="json_str",
                expected="valid JSON string",
            ) from e

        _validate_type(data, dict, "data", context="ErrorMessage.from_json")

        msg_id = data.get("msg_id")
        if msg_id is not None:
            msg_id = _validate_int(msg_id, "msg_id", context="ErrorMessage")

        timestamp = _validate_float(
            data.get("timestamp"),
            "timestamp",
            context="ErrorMessage",
            default=time.time(),
        )
        error_code = _validate_string(
            data.get("error_code"),
            "error_code",
            context="ErrorMessage",
            default="E000",
        )
        message = _validate_string(
            data.get("message"),
            "message",
            context="ErrorMessage",
            default="Unknown error",
        )
        details = _validate_dict(data.get("details"), "details", context="ErrorMessage")

        return cls(
            msg_id=msg_id,
            timestamp=timestamp,
            error_code=error_code,
            message=message,
            details=details,
        )


# Error codes
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


# Era name to index mapping for Civ6
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
    """Parse era value from int or string.

    Args:
        era_value: Era as int index or string name (e.g., "ERA_ANCIENT").
        default: Default value if era_value is None or invalid.

    Returns:
        Era index as int.
    """
    if era_value is None:
        return default
    if isinstance(era_value, int):
        return era_value
    if isinstance(era_value, str):
        return ERA_NAME_TO_INDEX.get(era_value, default)
    return default


def validate_state_json(json_str: str) -> tuple[bool, str | None]:
    """Validate a state JSON string.

    Args:
        json_str: JSON string to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    try:
        data = json.loads(json_str)

        # Check required fields
        if "turn_id" not in data and "turn" not in data:
            return False, "Missing turn_id or turn field"

        if "map" not in data:
            return False, "Missing map field"

        map_data = data["map"]
        if "width" not in map_data or "height" not in map_data:
            return False, "Map missing width or height"

        return True, None

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    except Exception as e:
        return False, f"Validation error: {e}"


def create_end_turn_command(msg_id: int) -> CommandMessage:
    """Create an end turn command.

    Args:
        msg_id: Message ID for acknowledgment tracking.

    Returns:
        CommandMessage with end turn action.
    """
    return CommandMessage(
        msg_id=msg_id,
        commands=[LuaCommand(action_type=ActionType.END_TURN)],
    )


def create_unit_move_command(
    msg_id: int,
    unit_id: int,
    target_x: int,
    target_y: int,
) -> CommandMessage:
    """Create a unit move command.

    Args:
        msg_id: Message ID for acknowledgment tracking.
        unit_id: ID of the unit to move.
        target_x: Target X coordinate.
        target_y: Target Y coordinate.

    Returns:
        CommandMessage with unit move action.
    """
    return CommandMessage(
        msg_id=msg_id,
        commands=[
            LuaCommand(
                action_type=ActionType.UNIT_MOVE,
                unit_id=unit_id,
                target=Position(x=target_x, y=target_y),
            )
        ],
    )


def create_city_produce_command(
    msg_id: int,
    city_id: int,
    item_id: int,
    item_type: str = "unit",
) -> CommandMessage:
    """Create a city production command.

    Args:
        msg_id: Message ID for acknowledgment tracking.
        city_id: ID of the city.
        item_id: ID of the item to produce.
        item_type: Type of item (unit, building, wonder, district).

    Returns:
        CommandMessage with city production action.
    """
    return CommandMessage(
        msg_id=msg_id,
        commands=[
            LuaCommand(
                action_type=ActionType.CITY_PRODUCE,
                city_id=city_id,
                item_id=item_id,
                item_type=item_type,
            )
        ],
    )
