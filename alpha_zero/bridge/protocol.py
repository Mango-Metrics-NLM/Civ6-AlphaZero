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
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
        """Create from dictionary."""
        return cls(x=data["x"], y=data["y"])


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
        """Create from dictionary."""
        target = None
        if "target" in data:
            target = Position.from_dict(data["target"])

        return cls(
            action_type=ActionType(data["action_type"]),
            unit_id=data.get("unit_id"),
            city_id=data.get("city_id"),
            target=target,
            item_id=data.get("item_id"),
            item_type=data.get("item_type"),
            metadata=data.get("metadata", {}),
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
        """Create from dictionary."""
        return cls(
            x=data.get("x", 0),
            y=data.get("y", 0),
            terrain=data.get("terrain", "TERRAIN_PLAINS"),
            features=data.get("features", []),
            resource=data.get("resource"),
            improvement=data.get("improvement"),
            owner_id=data.get("owner_id") if data.get("owner_id", -1) != -1 else None,
            is_visible=data.get("is_visible", True),
            is_revealed=data.get("is_revealed", True),
            yields=data.get("yields", {}),
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
        """Create from dictionary."""
        return cls(
            id=data.get("id", 0),
            unit_type=data.get("type", data.get("type_name", "UNIT_WARRIOR")),
            owner_id=data.get("owner", data.get("owner_id", 0)),
            x=data.get("x", 0),
            y=data.get("y", 0),
            health=data.get("hp", data.get("health", 100)),
            max_health=data.get("max_hp", data.get("max_health", 100)),
            damage=data.get("damage", 0),
            moves_remaining=float(data.get("movement", data.get("moves_remaining", 0))),
            max_moves=float(data.get("max_movement", data.get("max_moves", 2))),
            combat_strength=data.get("combat_strength", 0),
            ranged_strength=data.get("ranged_strength", 0),
            range=data.get("range", 0),
            is_civilian=data.get("is_civilian", False),
            can_attack=data.get("can_attack", True),
            promotions=data.get("promotions", []),
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
        """Create from dictionary."""
        return cls(
            id=data.get("id", 0),
            name=data.get("name", "Unknown"),
            owner_id=data.get("owner", data.get("owner_id", 0)),
            x=data.get("x", 0),
            y=data.get("y", 0),
            population=data.get("population", 1),
            is_capital=data.get("is_capital", False),
            defense_strength=data.get("defense_strength", data.get("defense", 0)),
            buildings=data.get("buildings", []),
            districts=data.get("districts", []),
            current_production=data.get("current_production"),
            production_progress=data.get("production_progress", 0),
            available_production=data.get("available_production", []),
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
        """Create from dictionary."""
        return cls(
            id=data.get("id", 0),
            civilization=data.get("civilization", data.get("civ", "UNKNOWN")),
            gold=float(data.get("gold", 0)),
            science_per_turn=float(data.get("science_per_turn", data.get("science", 0))),
            culture_per_turn=float(data.get("culture_per_turn", data.get("culture", 0))),
            faith=float(data.get("faith", 0)),
            score=data.get("score", 0),
            is_alive=data.get("is_alive", True),
            is_human=data.get("is_human", True),
            techs_researched=data.get("techs_researched", []),
            current_research=data.get("current_research"),
        )


@dataclass
class LegalAction:
    """A legal action from the Lua mod."""

    action_type: str
    unit_id: int | None = None
    city_id: int | None = None
    targets: list[tuple[int, int]] = field(default_factory=list)
    items: list[int] = field(default_factory=list)
    item_type: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LegalAction:
        """Create from dictionary."""
        targets = []
        raw_targets = data.get("targets", [])
        for t in raw_targets:
            if isinstance(t, dict):
                targets.append((t.get("x", 0), t.get("y", 0)))
            elif isinstance(t, (list, tuple)) and len(t) >= 2:
                targets.append((t[0], t[1]))

        return cls(
            action_type=data.get("action", data.get("type", "end_turn")),
            unit_id=data.get("unit_id"),
            city_id=data.get("city_id"),
            targets=targets,
            items=data.get("items", data.get("item", [])),
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
        """Create from dictionary."""
        tiles = [TileData.from_dict(t) for t in data.get("tiles", [])]
        return cls(
            width=data.get("width", 0),
            height=data.get("height", 0),
            tiles=tiles,
        )


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
        """Create from dictionary."""
        # Handle the 'player' field which contains summary data
        player_summary = data.get("player", {})

        # Build player data from summary if no players array
        players_data = data.get("players", [])
        if not players_data and player_summary:
            players_data = [player_summary]

        return cls(
            turn_id=data.get("turn_id", data.get("turn", 0)),
            player_id=data.get("player_id", player_summary.get("id", 0)),
            map_data=MapData.from_dict(data.get("map", {})),
            units=[UnitData.from_dict(u) for u in data.get("units", [])],
            cities=[CityData.from_dict(c) for c in data.get("cities", [])],
            players=[PlayerData.from_dict(p) for p in players_data],
            legal_actions=[LegalAction.from_dict(a) for a in data.get("legal_actions", [])],
            current_player_civ=data.get("current_player", ""),
            era=data.get("era", 0),
            game_speed=data.get("game_speed", "GAMESPEED_STANDARD"),
            is_game_over=data.get("is_game_over", data.get("game_over", False)),
            winner_id=data.get("winner_id"),
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
        return json.dumps({
            "msg_id": self.msg_id,
            "type": MessageType.STATE.value,
            "timestamp": self.timestamp,
            "state": self._state_to_dict(),
        })

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
        """Parse from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateMessage:
        """Create from dictionary."""
        # Handle both wrapped and unwrapped formats
        if "state" in data:
            state_data = data["state"]
        else:
            # The entire dict is the state (direct Lua output)
            state_data = data

        return cls(
            msg_id=data.get("msg_id", int(time.time() * 1000)),
            timestamp=data.get("timestamp", time.time()),
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
        return json.dumps({
            "msg_id": self.msg_id,
            "type": MessageType.COMMAND.value,
            "timestamp": self.timestamp,
            "commands": [cmd.to_dict() for cmd in self.commands],
        })

    @classmethod
    def from_json(cls, json_str: str) -> CommandMessage:
        """Parse from JSON string."""
        data = json.loads(json_str)
        return cls(
            msg_id=data["msg_id"],
            timestamp=data.get("timestamp", time.time()),
            commands=[LuaCommand.from_dict(c) for c in data.get("commands", [])],
        )


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
        """Parse from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AckMessage:
        """Create from dictionary."""
        success = data.get("success", data.get("status") == "success")
        return cls(
            msg_id=data.get("msg_id", 0),
            timestamp=data.get("timestamp", time.time()),
            success=success,
            error_code=data.get("error_code", data.get("code")),
            error_message=data.get("error_message", data.get("message")),
            executed=data.get("executed", []),
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
        return json.dumps({
            "type": MessageType.ERROR.value,
            "msg_id": self.msg_id,
            "timestamp": self.timestamp,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        })

    @classmethod
    def from_json(cls, json_str: str) -> ErrorMessage:
        """Parse from JSON string."""
        data = json.loads(json_str)
        return cls(
            msg_id=data.get("msg_id"),
            timestamp=data.get("timestamp", time.time()),
            error_code=data.get("error_code", "E000"),
            message=data.get("message", "Unknown error"),
            details=data.get("details", {}),
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
        commands=[LuaCommand(
            action_type=ActionType.UNIT_MOVE,
            unit_id=unit_id,
            target=Position(x=target_x, y=target_y),
        )],
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
        commands=[LuaCommand(
            action_type=ActionType.CITY_PRODUCE,
            city_id=city_id,
            item_id=item_id,
            item_type=item_type,
        )],
    )
