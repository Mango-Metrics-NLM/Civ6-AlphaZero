"""Tests for alpha_zero.bridge.protocol module.

Validates protocol message types, serialization, deserialization,
and validation logic for the Civ6 bridge communication protocol.
"""

from __future__ import annotations

import json

import pytest

from alpha_zero.bridge.protocol import (
    AckMessage,
    ActionType,
    CityData,
    CommandMessage,
    ErrorCode,
    ErrorMessage,
    GameStateData,
    LegalAction,
    LuaCommand,
    MapData,
    MessageType,
    PlayerData,
    Position,
    ProtocolValidationError,
    StateMessage,
    TileData,
    UnitData,
    _validate_required,
    _validate_type,
)


class TestProtocolValidationError:
    """Tests for ProtocolValidationError exception."""

    def test_basic_message(self) -> None:
        """Test basic error message."""
        err = ProtocolValidationError("Test error")
        assert "Test error" in str(err)

    def test_with_field_name(self) -> None:
        """Test error with field name."""
        err = ProtocolValidationError("Test error", field_name="test_field")
        assert "test_field" in str(err)

    def test_with_expected_and_actual(self) -> None:
        """Test error with expected and actual values."""
        err = ProtocolValidationError(
            "Type mismatch",
            field_name="value",
            expected="int",
            actual="string",
        )
        assert "int" in str(err)
        assert "string" in str(err)

    def test_attributes_stored(self) -> None:
        """Test that attributes are properly stored."""
        err = ProtocolValidationError(
            "Error",
            field_name="test",
            expected="expected",
            actual="actual",
        )
        assert err.field_name == "test"
        assert err.expected == "expected"
        assert err.actual == "actual"


class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_required_present(self) -> None:
        """Test validate_required with present field."""
        data = {"field": "value"}
        result = _validate_required(data, "field")
        assert result == "value"

    def test_validate_required_missing(self) -> None:
        """Test validate_required with missing field."""
        data = {}
        with pytest.raises(ProtocolValidationError) as exc_info:
            _validate_required(data, "field")
        assert "field" in str(exc_info.value)

    def test_validate_required_with_context(self) -> None:
        """Test validate_required with context string."""
        data = {}
        with pytest.raises(ProtocolValidationError) as exc_info:
            _validate_required(data, "field", context="test context")
        assert "test context" in str(exc_info.value)

    def test_validate_type_correct(self) -> None:
        """Test validate_type with correct type."""
        result = _validate_type("test", str, "field")
        assert result == "test"

    def test_validate_type_incorrect(self) -> None:
        """Test validate_type with incorrect type."""
        with pytest.raises(ProtocolValidationError) as exc_info:
            _validate_type(123, str, "field")
        assert "str" in str(exc_info.value)


class TestMessageType:
    """Tests for MessageType enum."""

    def test_enum_values(self) -> None:
        """Test MessageType enum values."""
        assert MessageType.STATE == "state"
        assert MessageType.COMMAND == "command"
        assert MessageType.ACK == "ack"
        assert MessageType.ERROR == "error"

    def test_string_representation(self) -> None:
        """Test string representation of MessageType."""
        # MessageType inherits from str, so .value gives the string value
        assert MessageType.STATE.value == "state"
        # Direct equality comparison also works due to str inheritance
        assert MessageType.STATE == "state"


class TestActionType:
    """Tests for ActionType enum."""

    def test_common_action_types(self) -> None:
        """Test common action types exist."""
        assert ActionType.UNIT_MOVE == "unit_move"
        assert ActionType.UNIT_ATTACK == "unit_attack"
        assert ActionType.CITY_PRODUCE == "city_produce"
        assert ActionType.END_TURN == "end_turn"

    def test_all_action_types_are_strings(self) -> None:
        """Test all action types are valid strings."""
        for action_type in ActionType:
            assert isinstance(action_type.value, str)
            assert len(action_type.value) > 0


class TestPosition:
    """Tests for Position dataclass."""

    def test_creation(self) -> None:
        """Test Position creation."""
        pos = Position(x=5, y=10)
        assert pos.x == 5
        assert pos.y == 10

    def test_from_dict(self) -> None:
        """Test Position.from_dict."""
        data = {"x": 3, "y": 7}
        pos = Position.from_dict(data)
        assert pos.x == 3
        assert pos.y == 7

    def test_to_dict(self) -> None:
        """Test Position.to_dict."""
        pos = Position(x=1, y=2)
        data = pos.to_dict()
        assert data == {"x": 1, "y": 2}

    def test_from_dict_missing_field(self) -> None:
        """Test Position.from_dict with missing field."""
        with pytest.raises(ProtocolValidationError):
            Position.from_dict({"x": 1})

    def test_equality(self) -> None:
        """Test Position equality."""
        pos1 = Position(x=1, y=2)
        pos2 = Position(x=1, y=2)
        pos3 = Position(x=1, y=3)
        assert pos1 == pos2
        assert pos1 != pos3


class TestLuaCommand:
    """Tests for LuaCommand dataclass."""

    def test_creation(self) -> None:
        """Test LuaCommand creation."""
        cmd = LuaCommand(
            action_type=ActionType.UNIT_MOVE,
            target=Position(x=5, y=5),
        )
        assert cmd.action_type == ActionType.UNIT_MOVE
        assert cmd.target is not None and cmd.target.x == 5

    def test_from_dict(self) -> None:
        """Test LuaCommand.from_dict."""
        data = {
            "action_type": "unit_move",
            "unit_id": 123,
            "target_x": 5,
            "target_y": 10,
        }
        cmd = LuaCommand.from_dict(data)
        assert cmd.action_type == ActionType.UNIT_MOVE
        assert cmd.unit_id == 123

    def test_to_dict(self) -> None:
        """Test LuaCommand.to_dict."""
        cmd = LuaCommand(
            action_type=ActionType.END_TURN,
        )
        data = cmd.to_dict()
        assert data["action_type"] == "end_turn"

    def test_end_turn_command(self) -> None:
        """Test end turn command creation."""
        cmd = LuaCommand(action_type=ActionType.END_TURN)
        assert cmd.action_type == ActionType.END_TURN
        assert cmd.unit_id is None
        assert cmd.city_id is None


class TestTileData:
    """Tests for TileData dataclass."""

    def test_creation(self) -> None:
        """Test TileData creation."""
        tile = TileData(
            x=0,
            y=0,
            terrain="plains",
        )
        assert tile.terrain == "plains"

    def test_from_dict(self) -> None:
        """Test TileData.from_dict."""
        data = {
            "x": 1,
            "y": 2,
            "terrain": "hills",
            "features": ["forest"],
            "resource": "iron",
        }
        tile = TileData.from_dict(data)
        assert tile.x == 1
        assert tile.terrain == "hills"
        assert "forest" in tile.features

    @pytest.mark.skip(reason="TileData.to_dict() not implemented")
    def test_to_dict(self) -> None:
        """Test TileData.to_dict."""
        tile = TileData(
            x=5,
            y=5,
            terrain="mountain",
        )
        data = tile.to_dict()
        assert data["terrain"] == "mountain"


class TestUnitData:
    """Tests for UnitData dataclass."""

    def test_creation(self) -> None:
        """Test UnitData creation."""
        unit = UnitData(
            id=1,
            x=0,
            y=0,
            unit_type="warrior",
            owner_id=0,
            health=100,
            moves_remaining=2.0,
        )
        assert unit.unit_type == "warrior"
        assert unit.health == 100

    def test_from_dict(self) -> None:
        """Test UnitData.from_dict."""
        data = {
            "id": 42,
            "x": 3,
            "y": 4,
            "unit_type": "settler",
            "owner_id": 1,
            "health": 100,
            "moves_remaining": 2.0,
            "max_moves": 2.0,
            "combat_strength": 0,
        }
        unit = UnitData.from_dict(data)
        assert unit.id == 42
        assert unit.unit_type == "settler"

    @pytest.mark.skip(reason="UnitData.to_dict() not implemented")
    def test_to_dict(self) -> None:
        """Test UnitData.to_dict."""
        unit = UnitData(
            id=1,
            x=0,
            y=0,
            unit_type="archer",
            owner_id=0,
            health=75,
            moves_remaining=1.5,
        )
        data = unit.to_dict()
        assert data["unit_type"] == "archer"
        assert data["health"] == 75


class TestCityData:
    """Tests for CityData dataclass."""

    def test_creation(self) -> None:
        """Test CityData creation."""
        city = CityData(
            id=1,
            x=5,
            y=5,
            name="Test City",
            owner_id=0,
            population=3,
        )
        assert city.name == "Test City"
        assert city.population == 3

    def test_from_dict(self) -> None:
        """Test CityData.from_dict."""
        data = {
            "id": 100,
            "x": 10,
            "y": 10,
            "name": "Capital",
            "owner_id": 0,
            "population": 5,
        }
        city = CityData.from_dict(data)
        assert city.id == 100
        assert city.name == "Capital"

    @pytest.mark.skip(reason="CityData.to_dict() not implemented")
    def test_to_dict(self) -> None:
        """Test CityData.to_dict."""
        city = CityData(
            id=1,
            x=0,
            y=0,
            name="City",
            owner_id=0,
            population=1,
        )
        data = city.to_dict()
        assert data["name"] == "City"


class TestPlayerData:
    """Tests for PlayerData dataclass."""

    def test_creation(self) -> None:
        """Test PlayerData creation."""
        player = PlayerData(
            id=0,
            civilization="Rome",
            is_human=True,
        )
        assert player.civilization == "Rome"
        assert player.is_human is True

    def test_from_dict(self) -> None:
        """Test PlayerData.from_dict."""
        data = {
            "id": 1,
            "civilization": "Greece",
            "is_human": False,
            "gold": 100.0,
            "science_per_turn": 50.0,
        }
        player = PlayerData.from_dict(data)
        assert player.id == 1
        assert player.civilization == "Greece"

    @pytest.mark.skip(reason="PlayerData.to_dict() not implemented")
    def test_to_dict(self) -> None:
        """Test PlayerData.to_dict."""
        player = PlayerData(
            id=0,
            civilization="Egypt",
            is_human=True,
        )
        data = player.to_dict()
        assert data["civilization"] == "Egypt"


class TestLegalAction:
    """Tests for LegalAction dataclass."""

    def test_creation(self) -> None:
        """Test LegalAction creation."""
        action = LegalAction(
            action_type="unit_move",
            unit_id=1,
            targets=[(5, 5)],
        )
        assert action.action_type == "unit_move"

    def test_from_dict(self) -> None:
        """Test LegalAction.from_dict."""
        data = {
            "action_type": "unit_attack",
            "unit_id": 10,
            "targets": [{"x": 3, "y": 4}],
        }
        action = LegalAction.from_dict(data)
        assert action.action_type == "unit_attack"

    @pytest.mark.skip(reason="LegalAction.to_dict() not implemented")
    def test_to_dict(self) -> None:
        """Test LegalAction.to_dict."""
        action = LegalAction(
            action_type="end_turn",
        )
        data = action.to_dict()
        assert data["action_type"] == "end_turn"


class TestStateMessage:
    """Tests for StateMessage dataclass."""

    def test_from_json(self) -> None:
        """Test StateMessage.from_json."""
        json_str = json.dumps(
            {
                "type": "state",
                "msg_id": 1,
                "timestamp": 1234567890.0,
                "state": {
                    "turn_id": 1,
                    "player_id": 0,
                    "map": {
                        "width": 10,
                        "height": 10,
                        "tiles": [],
                    },
                    "players": [],
                    "units": [],
                    "cities": [],
                    "legal_actions": [],
                },
            }
        )
        state = StateMessage.from_json(json_str)
        assert state.msg_id == 1
        assert state.state.turn_id == 1

    def test_to_json(self) -> None:
        """Test StateMessage.to_json."""
        state = StateMessage(
            msg_id=1,
            timestamp=1234567890.0,
            state=GameStateData(
                turn_id=1,
                player_id=0,
                map_data=MapData(width=10, height=10, tiles=[]),
                players=[],
                units=[],
                cities=[],
                legal_actions=[],
            ),
        )
        json_str = state.to_json()
        data = json.loads(json_str)
        assert data["msg_id"] == 1
        assert data["type"] == "state"

    def test_from_dict(self) -> None:
        """Test StateMessage.from_dict."""
        data = {
            "type": "state",
            "msg_id": 42,
            "timestamp": 0.0,
            "state": {
                "turn_id": 10,
                "player_id": 1,
                "map": {"width": 8, "height": 8, "tiles": []},
                "players": [],
                "units": [],
                "cities": [],
                "legal_actions": [],
            },
        }
        state = StateMessage.from_dict(data)
        assert state.msg_id == 42
        assert state.state.turn_id == 10


class TestCommandMessage:
    """Tests for CommandMessage dataclass."""

    def test_creation(self) -> None:
        """Test CommandMessage creation."""
        cmd = CommandMessage(
            msg_id=1,
            commands=[LuaCommand(action_type=ActionType.END_TURN)],
        )
        assert cmd.msg_id == 1
        assert len(cmd.commands) == 1

    def test_to_json(self) -> None:
        """Test CommandMessage.to_json."""
        cmd = CommandMessage(
            msg_id=1,
            commands=[LuaCommand(action_type=ActionType.END_TURN)],
        )
        json_str = cmd.to_json()
        data = json.loads(json_str)
        assert data["type"] == "command"
        assert data["msg_id"] == 1

    def test_from_json(self) -> None:
        """Test CommandMessage.from_json."""
        json_str = json.dumps(
            {
                "msg_type": "command",
                "msg_id": 5,
                "timestamp": 0.0,
                "commands": [{"action_type": "end_turn"}],
            }
        )
        cmd = CommandMessage.from_json(json_str)
        assert cmd.msg_id == 5
        assert len(cmd.commands) == 1

    @pytest.mark.skip(reason="CommandMessage.create_single() not implemented")
    def test_single_command(self) -> None:
        """Test single command message."""
        cmd = CommandMessage.create_single(
            msg_id=1,
            command=LuaCommand(
                action_type=ActionType.UNIT_MOVE,
                unit_id=10,
                target=Position(x=5, y=5),
            ),
        )
        assert len(cmd.commands) == 1
        assert cmd.commands[0].unit_id == 10


class TestAckMessage:
    """Tests for AckMessage dataclass."""

    def test_creation(self) -> None:
        """Test AckMessage creation."""
        ack = AckMessage(
            msg_id=1,
            success=True,
        )
        assert ack.success is True

    def test_from_json(self) -> None:
        """Test AckMessage.from_json."""
        json_str = json.dumps(
            {
                "type": "ack",
                "msg_id": 2,
                "timestamp": 0.0,
                "success": True,
            }
        )
        ack = AckMessage.from_json(json_str)
        assert ack.msg_id == 2
        assert ack.success is True

    def test_to_json(self) -> None:
        """Test AckMessage.to_json."""
        ack = AckMessage(
            msg_id=2,
            success=False,
            error_message="Test error",
        )
        json_str = ack.to_json()
        data = json.loads(json_str)
        assert data["success"] is False
        assert data.get("error_message") == "Test error"


class TestErrorMessage:
    """Tests for ErrorMessage dataclass."""

    def test_creation(self) -> None:
        """Test ErrorMessage creation."""
        err = ErrorMessage(
            msg_id=1,
            error_code=ErrorCode.INVALID_ACTION,
            message="Invalid action",
        )
        assert err.error_code == ErrorCode.INVALID_ACTION

    def test_from_json(self) -> None:
        """Test ErrorMessage.from_json."""
        json_str = json.dumps(
            {
                "type": "error",
                "msg_id": 1,
                "timestamp": 0.0,
                "error_code": "E001",
                "message": "Cannot move there",
            }
        )
        err = ErrorMessage.from_json(json_str)
        assert "Cannot move there" in err.message

    def test_to_json(self) -> None:
        """Test ErrorMessage.to_json."""
        err = ErrorMessage(
            msg_id=1,
            error_code=ErrorCode.CONNECTION_LOST,
            message="Lost connection",
        )
        json_str = err.to_json()
        data = json.loads(json_str)
        assert data["type"] == "error"


class TestErrorCode:
    """Tests for ErrorCode class."""

    def test_common_error_codes(self) -> None:
        """Test common error codes exist."""
        assert ErrorCode.INVALID_ACTION is not None
        assert ErrorCode.INVALID_STATE is not None
        assert ErrorCode.CONNECTION_LOST is not None

    def test_error_codes_are_strings(self) -> None:
        """Test error codes are valid strings."""
        # ErrorCode is a class with class attributes, not an Enum
        assert isinstance(ErrorCode.INVALID_ACTION, str)
        assert isinstance(ErrorCode.CONNECTION_LOST, str)


class TestSerializationRoundTrip:
    """Tests for serialization round-trip consistency."""

    def test_position_round_trip(self) -> None:
        """Test Position serialization round-trip."""
        original = Position(x=42, y=99)
        restored = Position.from_dict(original.to_dict())
        assert original == restored

    @pytest.mark.skip(reason="UnitData.to_dict() not implemented")
    def test_unit_data_round_trip(self) -> None:
        """Test UnitData serialization round-trip."""
        original = UnitData(
            id=1,
            x=5,
            y=5,
            unit_type="warrior",
            owner_id=0,
            health=75,
            moves_remaining=1.5,
        )
        data = original.to_dict()
        restored = UnitData.from_dict(data)
        assert restored.id == original.id
        assert restored.unit_type == original.unit_type

    def test_command_message_round_trip(self) -> None:
        """Test CommandMessage serialization round-trip."""
        original = CommandMessage(
            msg_id=42,
            commands=[
                LuaCommand(
                    action_type=ActionType.UNIT_MOVE,
                    unit_id=1,
                    target=Position(x=5, y=5),
                ),
            ],
        )
        json_str = original.to_json()
        restored = CommandMessage.from_json(json_str)
        assert restored.msg_id == original.msg_id
        assert len(restored.commands) == len(original.commands)
