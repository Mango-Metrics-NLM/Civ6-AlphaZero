"""Protocol definitions for Civ6 bridge communication.

Layer 2 of the split bridge protocol (top-level state/message envelopes, factory
helpers, and backwards-compatible facade re-exporting all Layer 0/1 symbols):
- :mod:`alpha_zero.bridge.protocol_validation` (Layer 0)
- :mod:`alpha_zero.bridge.protocol_entities` (Layer 1)
- :mod:`alpha_zero.bridge.protocol` (Layer 2 — this module)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from alpha_zero.bridge.other_players import dict_entries, optional_bool
from alpha_zero.bridge.protocol_entities import (
    _NAMED_ITEM_ACTIONS as _NAMED_ITEM_ACTIONS,
)
from alpha_zero.bridge.protocol_entities import (
    _TYPE_NAME as _TYPE_NAME,
)
from alpha_zero.bridge.protocol_entities import (
    CityData,
    LegalAction,
    LuaCommand,
    MapData,
    PlayerData,
    Position,
    ResearchStatus,
    TileData,
    UnitData,
)
from alpha_zero.bridge.protocol_entities import (
    _parse_research_status as _parse_research_status,
)
from alpha_zero.bridge.protocol_validation import (
    END_TURN_BLOCKED,
    ERA_NAME_TO_INDEX,
    LEGACY_STATE_SCHEMA_VERSION,
    ActionType,
    ErrorCode,
    MessageType,
    ProtocolValidationError,
)
from alpha_zero.bridge.protocol_validation import (
    _optional_int as _optional_int,
)
from alpha_zero.bridge.protocol_validation import (
    _optional_number as _optional_number,
)
from alpha_zero.bridge.protocol_validation import (
    _optional_text as _optional_text,
)
from alpha_zero.bridge.protocol_validation import (
    _parse_era as _parse_era,
)
from alpha_zero.bridge.protocol_validation import (
    _validate_dict as _validate_dict,
)
from alpha_zero.bridge.protocol_validation import (
    _validate_float as _validate_float,
)
from alpha_zero.bridge.protocol_validation import (
    _validate_int as _validate_int,
)
from alpha_zero.bridge.protocol_validation import (
    _validate_list as _validate_list,
)
from alpha_zero.bridge.protocol_validation import (
    _validate_required as _validate_required,
)
from alpha_zero.bridge.protocol_validation import (
    _validate_string as _validate_string,
)
from alpha_zero.bridge.protocol_validation import (
    _validate_type as _validate_type,
)

logger = logging.getLogger(__name__)

__all__ = [
    "END_TURN_BLOCKED",
    "ERA_NAME_TO_INDEX",
    "LEGACY_STATE_SCHEMA_VERSION",
    "AckMessage",
    "ActionType",
    "CityData",
    "CommandMessage",
    "ErrorCode",
    "ErrorMessage",
    "GameStateData",
    "LegalAction",
    "LuaCommand",
    "MapData",
    "MessageType",
    "PlayerData",
    "Position",
    "ProtocolValidationError",
    "ResearchStatus",
    "StateMessage",
    "TileData",
    "UnitData",
    "create_city_produce_command",
    "create_end_turn_command",
    "create_unit_move_command",
    "validate_state_json",
]


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
    #: Shape version the mod wrote; 1 for a payload without the key.
    schema_version: int = LEGACY_STATE_SCHEMA_VERSION
    end_turn_blocking: str | None = None  #: blocking type name, None when nothing blocks
    end_turn_blocking_message: str | None = None
    #: Other players exported? (None: older mod); what each is (``other_players.py``).
    observes_others: bool | None = None
    other_players: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameStateData:
        """Create from dictionary."""
        _validate_type(data, dict, "data", context="GameStateData.from_dict")

        player_summary = _validate_dict(data.get("player"), "player", context="GameStateData")

        players_data = _validate_list(data.get("players"), "players", context="GameStateData")
        if not players_data and player_summary:
            players_data = [player_summary]

        era_value = data.get("era", player_summary.get("era"))
        era = _parse_era(era_value, default=0)

        turn_id = _validate_int(
            data.get("turn_id", data.get("turn")),
            "turn_id",
            context="GameStateData",
            default=0,
        )
        player_id = _validate_int(
            data.get("player_id", player_summary.get("id")),
            "player_id",
            context="GameStateData",
            default=0,
        )

        map_dict = _validate_dict(data.get("map"), "map", context="GameStateData")
        map_data = MapData.from_dict(map_dict)

        raw_units = _validate_list(data.get("units"), "units", context="GameStateData")
        units = []
        for i, u in enumerate(raw_units):
            try:
                units.append(UnitData.from_dict(u))
            except ProtocolValidationError as e:
                logger.warning(f"Skipping invalid unit at index {i}: {e}")

        raw_cities = _validate_list(data.get("cities"), "cities", context="GameStateData")
        cities = []
        for i, c in enumerate(raw_cities):
            try:
                cities.append(CityData.from_dict(c))
            except ProtocolValidationError as e:
                logger.warning(f"Skipping invalid city at index {i}: {e}")

        players = []
        for i, p in enumerate(players_data):
            try:
                players.append(PlayerData.from_dict(p))
            except ProtocolValidationError as e:
                logger.warning(f"Skipping invalid player at index {i}: {e}")

        raw_legal_actions = _validate_list(
            data.get("legal_actions"), "legal_actions", context="GameStateData"
        )
        legal_actions = []
        for i, a in enumerate(raw_legal_actions):
            try:
                legal_actions.append(LegalAction.from_dict(a))
            except ProtocolValidationError as e:
                logger.warning(f"Skipping invalid legal action at index {i}: {e}")

        # Parse winner_id if present. StateExporter.lua always writes -1 as a
        # "no winner yet" sentinel (it never omits or nils the key), so that
        # sentinel must be mapped back to None here — otherwise every
        # downstream `winner_id is not None` check (real_civ6_env.py) would
        # treat "no winner" as "player -1 won" once is_game_over is true.
        winner_id = data.get("winner_id")
        if winner_id is not None:
            winner_id = _validate_int(winner_id, "winner_id", context="GameStateData")
            if winner_id == -1:
                winner_id = None

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
            schema_version=_optional_int(data.get("schema_version")) or LEGACY_STATE_SCHEMA_VERSION,
            end_turn_blocking=_optional_text(data.get("end_turn_blocking")),
            end_turn_blocking_message=_optional_text(data.get("end_turn_blocking_message")),
            observes_others=optional_bool(data.get("observes_others")),
            other_players=dict_entries(data.get("other_players")),
        )


@dataclass
class StateMessage:
    """State message from Lua mod."""

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
        """Parse from JSON string."""
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
        """Create from dictionary."""
        _validate_type(data, dict, "data", context="StateMessage.from_dict")

        if "state" in data:
            state_data = _validate_type(data["state"], dict, "state", context="StateMessage")
        else:
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
    """Command message to Lua mod."""

    msg_id: int
    commands: list[LuaCommand]
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def create_single(cls, msg_id: int, command: LuaCommand) -> CommandMessage:
        """Create a command message with a single command."""
        return cls(msg_id=msg_id, commands=[command])

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
        """Parse from JSON string."""
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
    """Acknowledgment message from Lua mod."""

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
        """Create from dictionary."""
        _validate_type(data, dict, "data", context="AckMessage.from_dict")

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
    """Error message from Lua mod or Python."""

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
        """Parse from JSON string."""
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


def validate_state_json(json_str: str) -> tuple[bool, str | None]:
    """Validate a state JSON string."""
    try:
        data = json.loads(json_str)

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
    """Create an end turn command."""
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
    """Create a unit move command."""
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
    """Create a city production command."""
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
