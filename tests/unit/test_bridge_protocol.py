"""Tests for alpha_zero.bridge.protocol module.

Validates protocol message types, serialization, deserialization,
and validation logic for the Civ6 bridge communication protocol.
"""

from __future__ import annotations

import json

import pytest

from alpha_zero.bridge.protocol import (
    ERA_NAME_TO_INDEX,
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
    _parse_era,
    _validate_dict,
    _validate_float,
    _validate_int,
    _validate_list,
    _validate_required,
    _validate_string,
    _validate_type,
    create_city_produce_command,
    create_end_turn_command,
    create_unit_move_command,
    validate_state_json,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_game_state_dict() -> dict:
    """A minimal valid game state dictionary."""
    return {
        "turn_id": 10,
        "player_id": 0,
        "map": {"width": 16, "height": 16, "tiles": []},
        "players": [
            {"id": 0, "civilization": "Rome", "is_human": True},
        ],
        "units": [
            {
                "id": 1,
                "x": 3,
                "y": 4,
                "unit_type": "warrior",
                "owner_id": 0,
                "health": 100,
                "moves_remaining": 2.0,
            }
        ],
        "cities": [
            {
                "id": 1,
                "x": 5,
                "y": 5,
                "name": "Rome",
                "owner_id": 0,
                "population": 3,
            }
        ],
        "legal_actions": [
            {"action_type": "unit_move", "unit_id": 1, "targets": [{"x": 4, "y": 4}]}
        ],
    }


@pytest.fixture
def valid_state_message_dict(valid_game_state_dict: dict) -> dict:
    """A valid wrapped state message dictionary."""
    return {
        "type": "state",
        "msg_id": 42,
        "timestamp": 1000000.0,
        "state": valid_game_state_dict,
    }


# ---------------------------------------------------------------------------
# ProtocolValidationError
# ---------------------------------------------------------------------------

class TestProtocolValidationError:
    """Tests for ProtocolValidationError exception."""

    def test_basic_message(self) -> None:
        err = ProtocolValidationError("Test error")
        assert "Test error" in str(err)

    def test_with_field_name(self) -> None:
        err = ProtocolValidationError("Test error", field_name="test_field")
        assert "test_field" in str(err)

    def test_with_expected_and_actual(self) -> None:
        err = ProtocolValidationError(
            "Type mismatch", field_name="value", expected="int", actual="string"
        )
        assert "int" in str(err)
        assert "string" in str(err)

    def test_attributes_stored(self) -> None:
        err = ProtocolValidationError(
            "Error", field_name="test", expected="expected", actual="actual"
        )
        assert err.field_name == "test"
        assert err.expected == "expected"
        assert err.actual == "actual"

    def test_long_actual_repr_uses_type_name(self) -> None:
        """When repr(actual) >= 100 chars, should use type name instead."""
        long_value = "x" * 120
        err = ProtocolValidationError("Error", actual=long_value)
        assert "str" in str(err)

    def test_none_actual_omitted(self) -> None:
        """When actual is None, the 'got=' part should not appear."""
        err = ProtocolValidationError("Error", field_name="f", expected="int")
        msg = str(err)
        assert "got=" not in msg

    def test_no_optional_fields(self) -> None:
        """Just message, no field_name, expected, or actual."""
        err = ProtocolValidationError("Simple")
        assert str(err) == "Simple"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class TestValidationHelpers:
    """Tests for validation helper functions."""

    def test_validate_required_present(self) -> None:
        result = _validate_required({"field": "value"}, "field")
        assert result == "value"

    def test_validate_required_missing(self) -> None:
        with pytest.raises(ProtocolValidationError):
            _validate_required({}, "field")

    def test_validate_required_with_context(self) -> None:
        with pytest.raises(ProtocolValidationError, match="test context"):
            _validate_required({}, "field", context="test context")

    def test_validate_type_correct(self) -> None:
        assert _validate_type("test", str, "field") == "test"

    def test_validate_type_incorrect(self) -> None:
        with pytest.raises(ProtocolValidationError, match="str"):
            _validate_type(123, str, "field")

    def test_validate_type_allow_none_with_none(self) -> None:
        """allow_none=True should return None without error."""
        result = _validate_type(None, str, "field", allow_none=True)
        assert result is None

    def test_validate_type_allow_none_with_value(self) -> None:
        """allow_none=True with a valid value should return the value."""
        result = _validate_type("hello", str, "field", allow_none=True)
        assert result == "hello"

    def test_validate_type_tuple_of_types(self) -> None:
        """Passing a tuple of expected types should work."""
        assert _validate_type(42, (int, float), "field") == 42
        assert _validate_type(3.14, (int, float), "field") == 3.14

    def test_validate_type_tuple_mismatch(self) -> None:
        """Error message should contain pipe-separated type names."""
        with pytest.raises(ProtocolValidationError, match=r"int|float"):
            _validate_type("nope", (int, float), "field")

    def test_validate_type_with_context(self) -> None:
        """Context should appear in error message."""
        with pytest.raises(ProtocolValidationError, match="TestCtx"):
            _validate_type("bad", int, "field", context="TestCtx")


class TestValidateInt:
    """Tests for _validate_int."""

    def test_valid_int(self) -> None:
        assert _validate_int(42, "f") == 42

    def test_string_convertible(self) -> None:
        assert _validate_int("100", "f") == 100

    def test_none_with_default(self) -> None:
        assert _validate_int(None, "f", default=7) == 7

    def test_none_without_default(self) -> None:
        with pytest.raises(ProtocolValidationError, match="Missing required integer"):
            _validate_int(None, "f")

    def test_unconvertible(self) -> None:
        with pytest.raises(ProtocolValidationError, match="Cannot convert to integer"):
            _validate_int("abc", "f")

    def test_unconvertible_with_context(self) -> None:
        with pytest.raises(ProtocolValidationError, match="MyCtx"):
            _validate_int("abc", "f", context="MyCtx")

    def test_below_min(self) -> None:
        with pytest.raises(ProtocolValidationError, match="below minimum"):
            _validate_int(3, "f", min_val=5)

    def test_above_max(self) -> None:
        with pytest.raises(ProtocolValidationError, match="above maximum"):
            _validate_int(100, "f", max_val=50)

    def test_at_min_boundary(self) -> None:
        assert _validate_int(5, "f", min_val=5) == 5

    def test_at_max_boundary(self) -> None:
        assert _validate_int(50, "f", max_val=50) == 50

    def test_float_to_int(self) -> None:
        assert _validate_int(3.7, "f") == 3

    def test_unconvertible_object(self) -> None:
        with pytest.raises(ProtocolValidationError):
            _validate_int(object(), "f")


class TestValidateFloat:
    """Tests for _validate_float."""

    def test_valid_float(self) -> None:
        assert _validate_float(3.14, "f") == 3.14

    def test_int_to_float(self) -> None:
        assert _validate_float(5, "f") == 5.0

    def test_string_convertible(self) -> None:
        assert _validate_float("2.5", "f") == 2.5

    def test_none_with_default(self) -> None:
        assert _validate_float(None, "f", default=1.0) == 1.0

    def test_none_without_default(self) -> None:
        with pytest.raises(ProtocolValidationError, match="Missing required float"):
            _validate_float(None, "f")

    def test_unconvertible(self) -> None:
        with pytest.raises(ProtocolValidationError, match="Cannot convert to float"):
            _validate_float("not_a_number", "f")

    def test_unconvertible_with_context(self) -> None:
        with pytest.raises(ProtocolValidationError, match="MyCtx"):
            _validate_float("bad", "f", context="MyCtx")


class TestValidateString:
    """Tests for _validate_string."""

    def test_valid_string(self) -> None:
        assert _validate_string("hello", "f") == "hello"

    def test_int_converted_to_string(self) -> None:
        assert _validate_string(42, "f") == "42"

    def test_none_with_default(self) -> None:
        assert _validate_string(None, "f", default="fallback") == "fallback"

    def test_none_without_default(self) -> None:
        with pytest.raises(ProtocolValidationError, match="Missing required string"):
            _validate_string(None, "f")

    def test_allowed_values_valid(self) -> None:
        assert _validate_string("a", "f", allowed_values=["a", "b", "c"]) == "a"

    def test_allowed_values_invalid(self) -> None:
        with pytest.raises(ProtocolValidationError, match="not in allowed set"):
            _validate_string("z", "f", allowed_values=["a", "b", "c"])

    def test_allowed_values_truncated_in_error(self) -> None:
        """When allowed_values has more than 5 entries, error shows '...'."""
        many = ["a", "b", "c", "d", "e", "f", "g"]
        with pytest.raises(ProtocolValidationError, match=r"\.\.\."):
            _validate_string("z", "f", allowed_values=many)


class TestValidateList:
    """Tests for _validate_list."""

    def test_valid_list(self) -> None:
        assert _validate_list([1, 2], "f") == [1, 2]

    def test_none_returns_default(self) -> None:
        assert _validate_list(None, "f") == []

    def test_not_a_list(self) -> None:
        with pytest.raises(ProtocolValidationError, match="Expected list"):
            _validate_list("not_list", "f")

    def test_not_a_list_with_context(self) -> None:
        with pytest.raises(ProtocolValidationError, match="MyCtx"):
            _validate_list(42, "f", context="MyCtx")

    def test_custom_default_factory(self) -> None:
        result = _validate_list(None, "f", default_factory=lambda: [1, 2, 3])
        assert result == [1, 2, 3]


class TestValidateDict:
    """Tests for _validate_dict."""

    def test_valid_dict(self) -> None:
        assert _validate_dict({"a": 1}, "f") == {"a": 1}

    def test_none_returns_default(self) -> None:
        assert _validate_dict(None, "f") == {}

    def test_not_a_dict(self) -> None:
        with pytest.raises(ProtocolValidationError, match="Expected dict"):
            _validate_dict("not_dict", "f")

    def test_not_a_dict_with_context(self) -> None:
        with pytest.raises(ProtocolValidationError, match="MyCtx"):
            _validate_dict(42, "f", context="MyCtx")

    def test_custom_default_factory(self) -> None:
        result = _validate_dict(None, "f", default_factory=lambda: {"x": 1})
        assert result == {"x": 1}


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestMessageType:
    """Tests for MessageType enum."""

    def test_enum_values(self) -> None:
        assert MessageType.STATE == "state"
        assert MessageType.COMMAND == "command"
        assert MessageType.ACK == "ack"
        assert MessageType.ERROR == "error"
        assert MessageType.HEARTBEAT == "heartbeat"

    def test_string_representation(self) -> None:
        assert MessageType.STATE.value == "state"
        assert MessageType.STATE == "state"


class TestActionType:
    """Tests for ActionType enum."""

    def test_common_action_types(self) -> None:
        assert ActionType.UNIT_MOVE == "unit_move"
        assert ActionType.UNIT_ATTACK == "unit_attack"
        assert ActionType.CITY_PRODUCE == "city_produce"
        assert ActionType.END_TURN == "end_turn"

    def test_all_action_types_are_strings(self) -> None:
        for action_type in ActionType:
            assert isinstance(action_type.value, str)
            assert len(action_type.value) > 0

    def test_all_action_types_exist(self) -> None:
        expected = {
            "end_turn", "unit_move", "unit_attack", "unit_ranged_attack",
            "unit_fortify", "unit_skip", "unit_found_city", "unit_build",
            "city_produce", "city_buy", "research_tech", "research_civic",
        }
        actual = {a.value for a in ActionType}
        assert actual == expected


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class TestPosition:
    """Tests for Position dataclass."""

    def test_creation(self) -> None:
        pos = Position(x=5, y=10)
        assert pos.x == 5
        assert pos.y == 10

    def test_from_dict(self) -> None:
        pos = Position.from_dict({"x": 3, "y": 7})
        assert pos.x == 3
        assert pos.y == 7

    def test_to_dict(self) -> None:
        assert Position(x=1, y=2).to_dict() == {"x": 1, "y": 2}

    def test_from_dict_missing_field(self) -> None:
        with pytest.raises(ProtocolValidationError):
            Position.from_dict({"x": 1})

    def test_from_dict_not_a_dict(self) -> None:
        with pytest.raises(ProtocolValidationError):
            Position.from_dict("not_dict")  # type: ignore[arg-type]

    def test_equality(self) -> None:
        assert Position(1, 2) == Position(1, 2)
        assert Position(1, 2) != Position(1, 3)


# ---------------------------------------------------------------------------
# LuaCommand
# ---------------------------------------------------------------------------

class TestLuaCommand:
    """Tests for LuaCommand dataclass."""

    def test_creation(self) -> None:
        cmd = LuaCommand(action_type=ActionType.UNIT_MOVE, target=Position(5, 5))
        assert cmd.action_type == ActionType.UNIT_MOVE

    def test_from_dict(self) -> None:
        data = {"action_type": "unit_move", "unit_id": 123, "target_x": 5, "target_y": 10}
        cmd = LuaCommand.from_dict(data)
        assert cmd.action_type == ActionType.UNIT_MOVE
        assert cmd.unit_id == 123

    def test_to_dict_minimal(self) -> None:
        cmd = LuaCommand(action_type=ActionType.END_TURN)
        data = cmd.to_dict()
        assert data == {"action_type": "end_turn"}

    def test_to_dict_all_optional_fields(self) -> None:
        """Cover lines 448-459: all optional fields in to_dict."""
        cmd = LuaCommand(
            action_type=ActionType.CITY_PRODUCE,
            unit_id=10,
            city_id=20,
            target=Position(x=5, y=6),
            item_id=30,
            item_type="building",
            metadata={"priority": "high"},
        )
        data = cmd.to_dict()
        assert data["action_type"] == "city_produce"
        assert data["unit_id"] == 10
        assert data["city_id"] == 20
        assert data["target"] == {"x": 5, "y": 6}
        assert data["item_id"] == 30
        assert data["item_type"] == "building"
        assert data["metadata"] == {"priority": "high"}

    def test_from_dict_invalid_action_type(self) -> None:
        """Cover lines 482-489: invalid action type raises error."""
        with pytest.raises(ProtocolValidationError, match="Invalid action type"):
            LuaCommand.from_dict({"action_type": "invalid_action"})

    def test_from_dict_with_target_dict(self) -> None:
        """Cover lines 492-498: target parsed from dict."""
        cmd = LuaCommand.from_dict({
            "action_type": "unit_move",
            "target": {"x": 3, "y": 4},
        })
        assert cmd.target is not None
        assert cmd.target.x == 3
        assert cmd.target.y == 4

    def test_from_dict_with_city_id(self) -> None:
        """Cover line 507: city_id validation."""
        cmd = LuaCommand.from_dict({
            "action_type": "city_produce",
            "city_id": 42,
        })
        assert cmd.city_id == 42

    def test_from_dict_with_item_id(self) -> None:
        """Cover line 511: item_id validation."""
        cmd = LuaCommand.from_dict({
            "action_type": "city_produce",
            "item_id": 99,
        })
        assert cmd.item_id == 99

    def test_from_dict_with_metadata(self) -> None:
        """Cover line 513: metadata validation."""
        cmd = LuaCommand.from_dict({
            "action_type": "end_turn",
            "metadata": {"key": "val"},
        })
        assert cmd.metadata == {"key": "val"}

    def test_from_dict_target_none(self) -> None:
        """Target explicitly set to None should be ignored."""
        cmd = LuaCommand.from_dict({
            "action_type": "end_turn",
            "target": None,
        })
        assert cmd.target is None

    def test_from_dict_target_empty_dict(self) -> None:
        """Cover branch 497->501: empty dict target is falsy in Python, target stays None."""
        cmd = LuaCommand.from_dict({
            "action_type": "unit_move",
            "target": {},
        })
        # Empty dict {} is falsy in Python, so `if target_data:` is False -> target stays None
        assert cmd.target is None

    def test_end_turn_command(self) -> None:
        cmd = LuaCommand(action_type=ActionType.END_TURN)
        assert cmd.unit_id is None
        assert cmd.city_id is None


# ---------------------------------------------------------------------------
# TileData
# ---------------------------------------------------------------------------

class TestTileData:
    """Tests for TileData dataclass."""

    def test_creation(self) -> None:
        tile = TileData(x=0, y=0, terrain="plains")
        assert tile.terrain == "plains"

    def test_from_dict(self) -> None:
        tile = TileData.from_dict({
            "x": 1, "y": 2, "terrain": "hills",
            "features": ["forest"], "resource": "iron",
        })
        assert tile.terrain == "hills"
        assert "forest" in tile.features

    def test_from_dict_defaults(self) -> None:
        """Empty dict should use all defaults without error."""
        tile = TileData.from_dict({})
        assert tile.x == 0
        assert tile.y == 0
        assert tile.terrain == "TERRAIN_PLAINS"
        assert tile.features == []
        assert tile.owner_id is None

    def test_from_dict_owner_key(self) -> None:
        """'owner' key should work the same as 'owner_id'."""
        tile = TileData.from_dict({"owner": 2})
        assert tile.owner_id == 2

    def test_from_dict_owner_negative_one(self) -> None:
        """owner_id of -1 should be treated as None."""
        tile = TileData.from_dict({"owner_id": -1})
        assert tile.owner_id is None

    def test_from_dict_owner_none(self) -> None:
        """owner_id of None should stay None."""
        tile = TileData.from_dict({"owner_id": None})
        assert tile.owner_id is None

    def test_from_dict_owner_string_negative_one(self) -> None:
        """Cover line 591: owner as string '-1' converts to -1, then set to None."""
        tile = TileData.from_dict({"owner": "-1"})
        assert tile.owner_id is None

    def test_from_dict_with_yields(self) -> None:
        """Cover yields validation."""
        tile = TileData.from_dict({"yields": {"food": 2, "production": 1}})
        assert tile.yields == {"food": 2, "production": 1}

    def test_from_dict_visibility_flags(self) -> None:
        tile = TileData.from_dict({"is_visible": False, "is_revealed": False})
        assert tile.is_visible is False
        assert tile.is_revealed is False

    def test_to_dict(self) -> None:
        tile = TileData(x=5, y=5, terrain="mountain")
        data = tile.to_dict()
        assert data["terrain"] == "mountain"
        assert data["x"] == 5
        assert data["y"] == 5

    def test_to_dict_with_optional_fields(self) -> None:
        """Cover lines 554-563: optional fields in to_dict."""
        tile = TileData(
            x=1, y=2, terrain="plains",
            features=["forest"],
            resource="iron",
            improvement="mine",
            owner_id=1,
            yields={"food": 3},
        )
        data = tile.to_dict()
        assert data["features"] == ["forest"]
        assert data["resource"] == "iron"
        assert data["improvement"] == "mine"
        assert data["owner_id"] == 1
        assert data["yields"] == {"food": 3}

    def test_to_dict_omits_empty_optional(self) -> None:
        """Without optional fields set, they should be omitted."""
        tile = TileData(x=0, y=0, terrain="ocean")
        data = tile.to_dict()
        assert "features" not in data
        assert "resource" not in data
        assert "improvement" not in data
        assert "owner_id" not in data
        assert "yields" not in data


# ---------------------------------------------------------------------------
# UnitData
# ---------------------------------------------------------------------------

class TestUnitData:
    """Tests for UnitData dataclass."""

    def test_creation(self) -> None:
        unit = UnitData(id=1, x=0, y=0, unit_type="warrior", owner_id=0,
                        health=100, moves_remaining=2.0)
        assert unit.unit_type == "warrior"

    def test_from_dict(self) -> None:
        unit = UnitData.from_dict({
            "id": 42, "x": 3, "y": 4, "unit_type": "settler",
            "owner_id": 1, "health": 100, "moves_remaining": 2.0,
        })
        assert unit.id == 42
        assert unit.unit_type == "settler"

    def test_from_dict_legacy_type_key(self) -> None:
        """Cover legacy 'type' field priority."""
        unit = UnitData.from_dict({"type": "archer"})
        assert unit.unit_type == "archer"

    def test_from_dict_legacy_type_name_key(self) -> None:
        """Cover legacy 'type_name' field priority."""
        unit = UnitData.from_dict({"type_name": "scout"})
        assert unit.unit_type == "scout"

    def test_from_dict_legacy_owner_key(self) -> None:
        """Cover 'owner' field fallback."""
        unit = UnitData.from_dict({"owner": 3})
        assert unit.owner_id == 3

    def test_from_dict_legacy_hp_key(self) -> None:
        """Cover 'hp' field fallback."""
        unit = UnitData.from_dict({"hp": 75})
        assert unit.health == 75

    def test_from_dict_legacy_max_hp_key(self) -> None:
        """Cover 'max_hp' field fallback."""
        unit = UnitData.from_dict({"max_hp": 200})
        assert unit.max_health == 200

    def test_from_dict_legacy_movement_key(self) -> None:
        """Cover 'movement' field fallback."""
        unit = UnitData.from_dict({"movement": 3.0})
        assert unit.moves_remaining == 3.0

    def test_from_dict_legacy_max_movement_key(self) -> None:
        """Cover 'max_movement' field fallback."""
        unit = UnitData.from_dict({"max_movement": 4.0})
        assert unit.max_moves == 4.0

    def test_from_dict_defaults(self) -> None:
        """All defaults when empty dict is passed."""
        unit = UnitData.from_dict({})
        assert unit.id == 0
        assert unit.unit_type == "UNIT_WARRIOR"
        assert unit.owner_id == 0
        assert unit.health == 100
        assert unit.damage == 0
        assert unit.combat_strength == 0
        assert unit.ranged_strength == 0
        assert unit.range == 0
        assert unit.is_civilian is False
        assert unit.can_attack is True
        assert unit.promotions == []

    def test_from_dict_with_promotions(self) -> None:
        unit = UnitData.from_dict({"promotions": ["PROMO_A", "PROMO_B"]})
        assert unit.promotions == ["PROMO_A", "PROMO_B"]

    def test_from_dict_boolean_flags(self) -> None:
        unit = UnitData.from_dict({"is_civilian": True, "can_attack": False})
        assert unit.is_civilian is True
        assert unit.can_attack is False

    def test_to_dict(self) -> None:
        unit = UnitData(id=1, x=0, y=0, unit_type="archer", owner_id=0,
                        health=75, moves_remaining=1.5)
        data = unit.to_dict()
        assert data["unit_type"] == "archer"
        assert data["health"] == 75

    def test_to_dict_with_promotions(self) -> None:
        """Cover line 660: promotions included in to_dict."""
        unit = UnitData(id=1, x=0, y=0, unit_type="warrior", owner_id=0,
                        promotions=["PROMO_A"])
        data = unit.to_dict()
        assert data["promotions"] == ["PROMO_A"]

    def test_to_dict_without_promotions(self) -> None:
        """When no promotions, key should be absent."""
        unit = UnitData(id=1, x=0, y=0, unit_type="warrior", owner_id=0)
        data = unit.to_dict()
        assert "promotions" not in data

    def test_position_property(self) -> None:
        """Cover line 759: position property."""
        unit = UnitData(id=1, x=3, y=7, unit_type="warrior", owner_id=0)
        assert unit.position == (3, 7)


# ---------------------------------------------------------------------------
# CityData
# ---------------------------------------------------------------------------

class TestCityData:
    """Tests for CityData dataclass."""

    def test_creation(self) -> None:
        city = CityData(id=1, x=5, y=5, name="Test City", owner_id=0, population=3)
        assert city.name == "Test City"

    def test_from_dict(self) -> None:
        city = CityData.from_dict({
            "id": 100, "x": 10, "y": 10, "name": "Capital",
            "owner_id": 0, "population": 5,
        })
        assert city.name == "Capital"

    def test_from_dict_defaults(self) -> None:
        city = CityData.from_dict({})
        assert city.id == 0
        assert city.name == "Unknown"
        assert city.owner_id == 0
        assert city.population == 1
        assert city.is_capital is False
        assert city.buildings == []
        assert city.districts == []

    def test_from_dict_legacy_owner_key(self) -> None:
        city = CityData.from_dict({"owner": 2})
        assert city.owner_id == 2

    def test_from_dict_legacy_defense_key(self) -> None:
        city = CityData.from_dict({"defense": 30})
        assert city.defense_strength == 30

    def test_from_dict_with_production(self) -> None:
        city = CityData.from_dict({
            "current_production": "warrior",
            "production_progress": 15,
            "available_production": [1, 2, 3],
        })
        assert city.current_production == "warrior"
        assert city.production_progress == 15
        assert city.available_production == [1, 2, 3]

    def test_to_dict(self) -> None:
        city = CityData(id=1, x=0, y=0, name="City", owner_id=0, population=1)
        data = city.to_dict()
        assert data["name"] == "City"

    def test_to_dict_with_optional_fields(self) -> None:
        """Cover lines 798-804: optional fields in CityData.to_dict."""
        city = CityData(
            id=1, x=0, y=0, name="City", owner_id=0, population=2,
            buildings=["BUILDING_MONUMENT"],
            districts=["DISTRICT_CAMPUS"],
            current_production="warrior",
            available_production=[10, 20],
        )
        data = city.to_dict()
        assert data["buildings"] == ["BUILDING_MONUMENT"]
        assert data["districts"] == ["DISTRICT_CAMPUS"]
        assert data["current_production"] == "warrior"
        assert data["available_production"] == [10, 20]

    def test_to_dict_omits_empty_optional(self) -> None:
        city = CityData(id=1, x=0, y=0, name="City", owner_id=0, population=1)
        data = city.to_dict()
        assert "buildings" not in data
        assert "districts" not in data
        assert "current_production" not in data
        assert "available_production" not in data

    def test_position_property(self) -> None:
        """Cover line 872: position property."""
        city = CityData(id=1, x=5, y=9, name="C", owner_id=0)
        assert city.position == (5, 9)


# ---------------------------------------------------------------------------
# PlayerData
# ---------------------------------------------------------------------------

class TestPlayerData:
    """Tests for PlayerData dataclass."""

    def test_creation(self) -> None:
        player = PlayerData(id=0, civilization="Rome", is_human=True)
        assert player.civilization == "Rome"

    def test_from_dict(self) -> None:
        player = PlayerData.from_dict({
            "id": 1, "civilization": "Greece", "is_human": False,
            "gold": 100.0, "science_per_turn": 50.0,
        })
        assert player.civilization == "Greece"

    def test_from_dict_defaults(self) -> None:
        player = PlayerData.from_dict({})
        assert player.id == 0
        assert player.civilization == "UNKNOWN"
        assert player.gold == 0.0
        assert player.science_per_turn == 0.0
        assert player.culture_per_turn == 0.0
        assert player.faith == 0.0
        assert player.score == 0
        assert player.is_alive is True
        assert player.is_human is True

    def test_from_dict_legacy_civ_key(self) -> None:
        player = PlayerData.from_dict({"civ": "Egypt"})
        assert player.civilization == "Egypt"

    def test_from_dict_legacy_science_key(self) -> None:
        player = PlayerData.from_dict({"science": 42.0})
        assert player.science_per_turn == 42.0

    def test_from_dict_legacy_culture_key(self) -> None:
        player = PlayerData.from_dict({"culture": 10.0})
        assert player.culture_per_turn == 10.0

    def test_to_dict(self) -> None:
        player = PlayerData(id=0, civilization="Egypt", is_human=True)
        data = player.to_dict()
        assert data["civilization"] == "Egypt"

    def test_to_dict_with_optional_fields(self) -> None:
        """Cover lines 910-917: optional fields in PlayerData.to_dict."""
        player = PlayerData(
            id=0, civilization="Rome",
            techs_researched=["TECH_POTTERY"],
            current_research="TECH_MINING",
            civics_researched=["CIVIC_CODE_OF_LAWS"],
            current_civic="CIVIC_FOREIGN_TRADE",
        )
        data = player.to_dict()
        assert data["techs_researched"] == ["TECH_POTTERY"]
        assert data["current_research"] == "TECH_MINING"
        assert data["civics_researched"] == ["CIVIC_CODE_OF_LAWS"]
        assert data["current_civic"] == "CIVIC_FOREIGN_TRADE"

    def test_to_dict_omits_empty_optional(self) -> None:
        player = PlayerData(id=0, civilization="Rome")
        data = player.to_dict()
        assert "techs_researched" not in data
        assert "current_research" not in data
        assert "civics_researched" not in data
        assert "current_civic" not in data

    def test_civics_fields(self) -> None:
        player = PlayerData(
            id=0, civilization="Rome",
            civics_researched=["CIVIC_CODE_OF_LAWS"],
            current_civic="CIVIC_EARLY_EMPIRE",
        )
        assert player.civics_researched == ["CIVIC_CODE_OF_LAWS"]

    def test_civics_defaults(self) -> None:
        player = PlayerData(id=0, civilization="Rome")
        assert player.civics_researched == []
        assert player.current_civic is None


# ---------------------------------------------------------------------------
# LegalAction
# ---------------------------------------------------------------------------

class TestLegalAction:
    """Tests for LegalAction dataclass."""

    def test_creation(self) -> None:
        action = LegalAction(action_type="unit_move", unit_id=1, targets=[(5, 5)])
        assert action.action_type == "unit_move"

    def test_from_dict(self) -> None:
        action = LegalAction.from_dict({
            "action_type": "unit_attack",
            "unit_id": 10,
            "targets": [{"x": 3, "y": 4}],
        })
        assert action.action_type == "unit_attack"
        assert action.targets == [(3, 4)]

    def test_from_dict_targets_as_tuples(self) -> None:
        """Cover lines 1052-1055: targets as list/tuple format."""
        action = LegalAction.from_dict({
            "action_type": "move",
            "targets": [[1, 2], [3, 4]],
        })
        assert action.targets == [(1, 2), (3, 4)]

    def test_from_dict_targets_invalid_format(self) -> None:
        """Cover line 1057: invalid target format is skipped."""
        action = LegalAction.from_dict({
            "action_type": "move",
            "targets": [42, "bad", {"x": 1, "y": 2}],
        })
        # Only the dict target should be parsed
        assert action.targets == [(1, 2)]

    def test_from_dict_targets_none(self) -> None:
        """Cover line 1041: targets is None."""
        action = LegalAction.from_dict({
            "action_type": "end_turn",
            "targets": None,
        })
        assert action.targets == []

    def test_from_dict_items_as_ints(self) -> None:
        """Cover lines 1069-1070: integer items."""
        action = LegalAction.from_dict({
            "action_type": "produce",
            "items": [1, 2, 3],
        })
        assert action.items == [1, 2, 3]

    def test_from_dict_items_as_strings(self) -> None:
        """Cover lines 1071-1076: string items (tech names)."""
        action = LegalAction.from_dict({
            "action_type": "research",
            "items": ["TECH_POTTERY", "TECH_MINING"],
        })
        assert action.items == ["TECH_POTTERY", "TECH_MINING"]

    def test_from_dict_items_numeric_strings(self) -> None:
        """Cover lines 1073-1074: numeric strings converted to int."""
        action = LegalAction.from_dict({
            "action_type": "produce",
            "items": ["42", "-3"],
        })
        assert action.items == [42, -3]

    def test_from_dict_items_single_value(self) -> None:
        """Cover line 1064: single item wrapped in list."""
        action = LegalAction.from_dict({
            "action_type": "produce",
            "items": 42,
        })
        assert action.items == [42]

    def test_from_dict_items_none(self) -> None:
        """Cover line 1062: items is None."""
        action = LegalAction.from_dict({
            "action_type": "produce",
            "items": None,
        })
        assert action.items == []

    def test_from_dict_items_invalid_type(self) -> None:
        """Cover line 1078: invalid item type is skipped."""
        action = LegalAction.from_dict({
            "action_type": "produce",
            "items": [1, 3.14, "TECH"],
        })
        # int=1 kept, float=3.14 skipped, str="TECH" kept
        assert 1 in action.items
        assert "TECH" in action.items
        assert len(action.items) == 2

    def test_from_dict_legacy_item_key(self) -> None:
        """Cover 'item' key fallback."""
        action = LegalAction.from_dict({
            "action_type": "produce",
            "item": [10],
        })
        assert action.items == [10]

    def test_from_dict_with_city_id(self) -> None:
        """Cover lines 1085-1087: city_id validation."""
        action = LegalAction.from_dict({
            "action_type": "city_produce",
            "city_id": 55,
        })
        assert action.city_id == 55

    def test_from_dict_with_unit_id(self) -> None:
        """Cover lines 1082-1083: unit_id validation."""
        action = LegalAction.from_dict({
            "action_type": "unit_move",
            "unit_id": 77,
        })
        assert action.unit_id == 77

    def test_from_dict_legacy_action_key(self) -> None:
        """Cover legacy 'action' field priority."""
        action = LegalAction.from_dict({"action": "move"})
        assert action.action_type == "move"

    def test_from_dict_legacy_type_key(self) -> None:
        """Cover legacy 'type' field priority."""
        action = LegalAction.from_dict({"type": "attack"})
        assert action.action_type == "attack"

    def test_from_dict_no_action_type_defaults(self) -> None:
        """No action type falls back to 'end_turn'."""
        action = LegalAction.from_dict({})
        assert action.action_type == "end_turn"

    def test_to_dict(self) -> None:
        action = LegalAction(action_type="end_turn")
        data = action.to_dict()
        assert data["action_type"] == "end_turn"

    def test_to_dict_with_targets(self) -> None:
        action = LegalAction(action_type="move", unit_id=5, targets=[(1, 2), (3, 4)])
        data = action.to_dict()
        assert data["targets"][0] == {"x": 1, "y": 2}

    def test_to_dict_with_city_id(self) -> None:
        """Cover line 1011: city_id in to_dict."""
        action = LegalAction(action_type="produce", city_id=10)
        data = action.to_dict()
        assert data["city_id"] == 10

    def test_to_dict_with_items(self) -> None:
        """Cover lines 1014-1015: items in to_dict."""
        action = LegalAction(action_type="produce", items=[1, 2])
        data = action.to_dict()
        assert data["items"] == [1, 2]

    def test_to_dict_with_item_type(self) -> None:
        """Cover lines 1016-1017: item_type in to_dict."""
        action = LegalAction(action_type="produce", item_type="building")
        data = action.to_dict()
        assert data["item_type"] == "building"


# ---------------------------------------------------------------------------
# MapData
# ---------------------------------------------------------------------------

class TestMapData:
    """Tests for MapData dataclass."""

    def test_from_dict(self) -> None:
        md = MapData.from_dict({"width": 10, "height": 10, "tiles": []})
        assert md.width == 10
        assert md.height == 10

    def test_from_dict_with_tiles(self) -> None:
        md = MapData.from_dict({
            "width": 2, "height": 2,
            "tiles": [
                {"x": 0, "y": 0, "terrain": "plains"},
                {"x": 1, "y": 0, "terrain": "hills"},
            ],
        })
        assert len(md.tiles) == 2

    def test_from_dict_skips_invalid_tiles(self) -> None:
        """Cover lines 1139-1142: invalid tile data is skipped."""
        md = MapData.from_dict({
            "width": 2, "height": 2,
            "tiles": [
                {"x": 0, "y": 0, "terrain": "plains"},
                "not_a_dict_tile",  # invalid
            ],
        })
        assert len(md.tiles) == 1

    def test_from_dict_defaults(self) -> None:
        md = MapData.from_dict({})
        assert md.width == 0
        assert md.height == 0
        assert md.tiles == []


# ---------------------------------------------------------------------------
# GameStateData
# ---------------------------------------------------------------------------

class TestGameStateData:
    """Tests for GameStateData dataclass."""

    def test_from_dict_full(self, valid_game_state_dict: dict) -> None:
        gs = GameStateData.from_dict(valid_game_state_dict)
        assert gs.turn_id == 10
        assert gs.player_id == 0
        assert len(gs.players) == 1
        assert len(gs.units) == 1
        assert len(gs.cities) == 1
        assert len(gs.legal_actions) == 1

    def test_from_dict_defaults(self) -> None:
        gs = GameStateData.from_dict({"map": {"width": 4, "height": 4}})
        assert gs.turn_id == 0
        assert gs.player_id == 0
        assert gs.era == 0
        assert gs.game_speed == "GAMESPEED_STANDARD"
        assert gs.is_game_over is False
        assert gs.winner_id is None

    def test_from_dict_player_summary_fallback(self) -> None:
        """Cover line 1185: when players list is empty, use player summary."""
        gs = GameStateData.from_dict({
            "map": {"width": 4, "height": 4},
            "player": {"id": 1, "civilization": "Rome"},
            "players": [],
        })
        assert len(gs.players) == 1
        assert gs.players[0].civilization == "Rome"

    def test_from_dict_turn_legacy_key(self) -> None:
        gs = GameStateData.from_dict({
            "turn": 42,
            "map": {"width": 4, "height": 4},
        })
        assert gs.turn_id == 42

    def test_from_dict_era_string(self) -> None:
        """Cover era parsing from string."""
        gs = GameStateData.from_dict({
            "map": {"width": 4, "height": 4},
            "era": "ERA_MEDIEVAL",
        })
        assert gs.era == 2

    def test_from_dict_era_from_player_summary(self) -> None:
        """Cover era parsed from player summary."""
        gs = GameStateData.from_dict({
            "map": {"width": 4, "height": 4},
            "player": {"era": "ERA_RENAISSANCE"},
        })
        assert gs.era == 3

    def test_from_dict_skips_invalid_units(self) -> None:
        """Cover lines 1215-1218: invalid unit data is skipped."""
        gs = GameStateData.from_dict({
            "map": {"width": 4, "height": 4},
            "units": [
                {"id": 1, "unit_type": "warrior"},
                "not_a_dict",
            ],
        })
        assert len(gs.units) == 1

    def test_from_dict_skips_invalid_cities(self) -> None:
        """Cover lines 1224-1227: invalid city data is skipped."""
        gs = GameStateData.from_dict({
            "map": {"width": 4, "height": 4},
            "cities": [
                {"id": 1, "name": "City"},
                "not_a_dict",
            ],
        })
        assert len(gs.cities) == 1

    def test_from_dict_skips_invalid_players(self) -> None:
        """Cover lines 1232-1235: invalid player data is skipped."""
        gs = GameStateData.from_dict({
            "map": {"width": 4, "height": 4},
            "players": [
                {"id": 0, "civilization": "Rome"},
                "not_a_dict",
            ],
        })
        assert len(gs.players) == 1

    def test_from_dict_skips_invalid_legal_actions(self) -> None:
        """Cover lines 1243-1246: invalid legal action data is skipped."""
        gs = GameStateData.from_dict({
            "map": {"width": 4, "height": 4},
            "legal_actions": [
                {"action_type": "end_turn"},
                "not_a_dict",
            ],
        })
        assert len(gs.legal_actions) == 1

    def test_from_dict_with_winner_id(self) -> None:
        """Cover line 1251: winner_id parsing."""
        gs = GameStateData.from_dict({
            "map": {"width": 4, "height": 4},
            "winner_id": 1,
            "is_game_over": True,
        })
        assert gs.winner_id == 1
        assert gs.is_game_over is True

    def test_from_dict_game_over_legacy_key(self) -> None:
        """Cover 'game_over' legacy key."""
        gs = GameStateData.from_dict({
            "map": {"width": 4, "height": 4},
            "game_over": True,
        })
        assert gs.is_game_over is True

    def test_from_dict_current_player(self) -> None:
        gs = GameStateData.from_dict({
            "map": {"width": 4, "height": 4},
            "current_player": "CIVILIZATION_ROME",
        })
        assert gs.current_player_civ == "CIVILIZATION_ROME"


# ---------------------------------------------------------------------------
# StateMessage
# ---------------------------------------------------------------------------

class TestStateMessage:
    """Tests for StateMessage dataclass."""

    def test_from_json(self) -> None:
        json_str = json.dumps({
            "type": "state", "msg_id": 1, "timestamp": 1234567890.0,
            "state": {
                "turn_id": 1, "player_id": 0,
                "map": {"width": 10, "height": 10, "tiles": []},
                "players": [], "units": [], "cities": [], "legal_actions": [],
            },
        })
        state = StateMessage.from_json(json_str)
        assert state.msg_id == 1

    def test_from_json_invalid_json(self) -> None:
        """Cover lines 1332-1337: invalid JSON string."""
        with pytest.raises(ProtocolValidationError, match="Invalid JSON"):
            StateMessage.from_json("{bad json")

    def test_from_json_not_a_string(self) -> None:
        """Non-string input should raise validation error."""
        with pytest.raises(ProtocolValidationError):
            StateMessage.from_json(123)  # type: ignore[arg-type]

    def test_to_json(self) -> None:
        state = StateMessage(
            msg_id=1, timestamp=1234567890.0,
            state=GameStateData(
                turn_id=1, player_id=0,
                map_data=MapData(width=10, height=10, tiles=[]),
                players=[], units=[], cities=[], legal_actions=[],
            ),
        )
        data = json.loads(state.to_json())
        assert data["msg_id"] == 1
        assert data["type"] == "state"

    def test_from_dict(self, valid_state_message_dict: dict) -> None:
        state = StateMessage.from_dict(valid_state_message_dict)
        assert state.msg_id == 42

    def test_from_dict_unwrapped(self) -> None:
        """Cover line 1360: direct Lua output without 'state' wrapper."""
        data = {
            "turn_id": 5, "player_id": 0,
            "map": {"width": 8, "height": 8, "tiles": []},
            "players": [], "units": [], "cities": [], "legal_actions": [],
        }
        state = StateMessage.from_dict(data)
        assert state.state.turn_id == 5

    def test_from_dict_auto_msg_id(self) -> None:
        """When msg_id is missing, default is generated."""
        data = {
            "state": {
                "turn_id": 1, "player_id": 0,
                "map": {"width": 4, "height": 4},
            }
        }
        state = StateMessage.from_dict(data)
        assert isinstance(state.msg_id, int)
        assert isinstance(state.timestamp, float)

    def test_state_to_dict_content(self) -> None:
        """Verify _state_to_dict produces expected keys."""
        state = StateMessage(
            msg_id=1, timestamp=0.0,
            state=GameStateData(
                turn_id=5, player_id=1,
                map_data=MapData(width=8, height=8, tiles=[]),
                players=[], units=[
                    UnitData(id=1, x=0, y=0, unit_type="w", owner_id=0),
                ],
                cities=[
                    CityData(id=1, x=0, y=0, name="C", owner_id=0),
                ],
                legal_actions=[
                    LegalAction(action_type="end_turn"),
                ],
            ),
        )
        data = json.loads(state.to_json())
        assert data["state"]["units_count"] == 1
        assert data["state"]["cities_count"] == 1
        assert data["state"]["legal_actions_count"] == 1


# ---------------------------------------------------------------------------
# CommandMessage
# ---------------------------------------------------------------------------

class TestCommandMessage:
    """Tests for CommandMessage dataclass."""

    def test_creation(self) -> None:
        cmd = CommandMessage(msg_id=1, commands=[LuaCommand(action_type=ActionType.END_TURN)])
        assert cmd.msg_id == 1

    def test_to_json(self) -> None:
        cmd = CommandMessage(msg_id=1, commands=[LuaCommand(action_type=ActionType.END_TURN)])
        data = json.loads(cmd.to_json())
        assert data["type"] == "command"

    def test_from_json(self) -> None:
        json_str = json.dumps({
            "msg_type": "command", "msg_id": 5, "timestamp": 0.0,
            "commands": [{"action_type": "end_turn"}],
        })
        cmd = CommandMessage.from_json(json_str)
        assert cmd.msg_id == 5

    def test_from_json_invalid_json(self) -> None:
        """Cover lines 1439-1444: invalid JSON string."""
        with pytest.raises(ProtocolValidationError, match="Invalid JSON"):
            CommandMessage.from_json("{bad json")

    def test_from_json_skips_invalid_commands(self) -> None:
        """Cover lines 1467-1468: invalid command entries are skipped."""
        json_str = json.dumps({
            "msg_id": 1, "timestamp": 0.0,
            "commands": [
                {"action_type": "end_turn"},
                {"action_type": "completely_fake_action"},
            ],
        })
        cmd = CommandMessage.from_json(json_str)
        # First command valid, second invalid action type skipped
        assert len(cmd.commands) == 1

    def test_single_command(self) -> None:
        cmd = CommandMessage.create_single(
            msg_id=1,
            command=LuaCommand(action_type=ActionType.UNIT_MOVE, unit_id=10, target=Position(5, 5)),
        )
        assert len(cmd.commands) == 1
        assert cmd.commands[0].unit_id == 10

    def test_round_trip(self) -> None:
        original = CommandMessage(
            msg_id=42,
            commands=[LuaCommand(action_type=ActionType.UNIT_MOVE, unit_id=1, target=Position(5, 5))],
        )
        restored = CommandMessage.from_json(original.to_json())
        assert restored.msg_id == original.msg_id
        assert len(restored.commands) == len(original.commands)


# ---------------------------------------------------------------------------
# AckMessage
# ---------------------------------------------------------------------------

class TestAckMessage:
    """Tests for AckMessage dataclass."""

    def test_creation(self) -> None:
        ack = AckMessage(msg_id=1, success=True)
        assert ack.success is True

    def test_from_json(self) -> None:
        json_str = json.dumps({
            "type": "ack", "msg_id": 2, "timestamp": 0.0, "success": True,
        })
        ack = AckMessage.from_json(json_str)
        assert ack.msg_id == 2
        assert ack.success is True

    def test_from_json_invalid_json(self) -> None:
        """Cover lines 1527-1532: invalid JSON string."""
        with pytest.raises(ProtocolValidationError, match="Invalid JSON"):
            AckMessage.from_json("{bad json")

    def test_to_json(self) -> None:
        ack = AckMessage(msg_id=2, success=False, error_message="Test error")
        data = json.loads(ack.to_json())
        assert data["success"] is False
        assert data["error_message"] == "Test error"

    def test_to_json_with_error_code(self) -> None:
        """Cover line 1503: error_code in to_json."""
        ack = AckMessage(msg_id=1, success=False, error_code="E001")
        data = json.loads(ack.to_json())
        assert data["error_code"] == "E001"

    def test_to_json_with_executed(self) -> None:
        """Cover line 1507: executed list in to_json."""
        ack = AckMessage(msg_id=1, success=True, executed=[{"cmd": "move", "ok": True}])
        data = json.loads(ack.to_json())
        assert data["executed"] == [{"cmd": "move", "ok": True}]

    def test_to_json_status_field(self) -> None:
        """Verify 'status' field is 'success' or 'error'."""
        ack_ok = AckMessage(msg_id=1, success=True)
        assert json.loads(ack_ok.to_json())["status"] == "success"

        ack_err = AckMessage(msg_id=1, success=False)
        assert json.loads(ack_err.to_json())["status"] == "error"

    def test_from_dict_status_fallback(self) -> None:
        """Cover line 1555: parse success from 'status' field when 'success' is absent."""
        ack = AckMessage.from_dict({"msg_id": 1, "status": "success"})
        assert ack.success is True

        ack2 = AckMessage.from_dict({"msg_id": 1, "status": "error"})
        assert ack2.success is False

    def test_from_dict_error_code_fallback(self) -> None:
        """Cover 'code' key fallback for error_code."""
        ack = AckMessage.from_dict({"msg_id": 1, "success": False, "code": "E005"})
        assert ack.error_code == "E005"

    def test_from_dict_message_fallback(self) -> None:
        """Cover 'message' key fallback for error_message."""
        ack = AckMessage.from_dict({
            "msg_id": 1, "success": False, "message": "Something broke"
        })
        assert ack.error_message == "Something broke"

    def test_from_dict_executed_list(self) -> None:
        ack = AckMessage.from_dict({
            "msg_id": 1, "success": True,
            "executed": [{"cmd": "end_turn"}],
        })
        assert len(ack.executed) == 1


# ---------------------------------------------------------------------------
# ErrorMessage
# ---------------------------------------------------------------------------

class TestErrorMessage:
    """Tests for ErrorMessage dataclass."""

    def test_creation(self) -> None:
        err = ErrorMessage(msg_id=1, error_code=ErrorCode.INVALID_ACTION, message="Invalid action")
        assert err.error_code == ErrorCode.INVALID_ACTION

    def test_from_json(self) -> None:
        json_str = json.dumps({
            "type": "error", "msg_id": 1, "timestamp": 0.0,
            "error_code": "E001", "message": "Cannot move there",
        })
        err = ErrorMessage.from_json(json_str)
        assert "Cannot move there" in err.message

    def test_from_json_invalid_json(self) -> None:
        """Cover lines 1624-1629: invalid JSON string."""
        with pytest.raises(ProtocolValidationError, match="Invalid JSON"):
            ErrorMessage.from_json("{bad json")

    def test_from_json_with_msg_id(self) -> None:
        """Cover line 1634: msg_id present and validated."""
        json_str = json.dumps({
            "msg_id": 99, "error_code": "E002", "message": "Error",
        })
        err = ErrorMessage.from_json(json_str)
        assert err.msg_id == 99

    def test_from_json_without_msg_id(self) -> None:
        """Cover line 1634: msg_id is None."""
        json_str = json.dumps({
            "error_code": "E003", "message": "Error",
        })
        err = ErrorMessage.from_json(json_str)
        assert err.msg_id is None

    def test_from_json_defaults(self) -> None:
        """Cover defaults for error_code and message."""
        json_str = json.dumps({})
        err = ErrorMessage.from_json(json_str)
        assert err.error_code == "E000"
        assert err.message == "Unknown error"

    def test_from_json_with_details(self) -> None:
        json_str = json.dumps({
            "error_code": "E001", "message": "Error",
            "details": {"unit_id": 42},
        })
        err = ErrorMessage.from_json(json_str)
        assert err.details == {"unit_id": 42}

    def test_to_json(self) -> None:
        err = ErrorMessage(msg_id=1, error_code=ErrorCode.CONNECTION_LOST, message="Lost connection")
        data = json.loads(err.to_json())
        assert data["type"] == "error"
        assert data["error_code"] == ErrorCode.CONNECTION_LOST


# ---------------------------------------------------------------------------
# ErrorCode
# ---------------------------------------------------------------------------

class TestErrorCode:
    """Tests for ErrorCode class."""

    def test_common_error_codes(self) -> None:
        assert ErrorCode.INVALID_ACTION is not None
        assert ErrorCode.INVALID_STATE is not None
        assert ErrorCode.CONNECTION_LOST is not None

    def test_error_codes_are_strings(self) -> None:
        assert isinstance(ErrorCode.INVALID_ACTION, str)
        assert isinstance(ErrorCode.CONNECTION_LOST, str)

    def test_all_error_codes(self) -> None:
        assert ErrorCode.UNKNOWN == "E000"
        assert ErrorCode.INVALID_ACTION == "E001"
        assert ErrorCode.STATE_UNAVAILABLE == "E002"
        assert ErrorCode.UNIT_NOT_FOUND == "E003"
        assert ErrorCode.CITY_NOT_FOUND == "E004"
        assert ErrorCode.FILE_IO_ERROR == "E005"
        assert ErrorCode.SERIALIZATION_ERROR == "E006"
        assert ErrorCode.TIMEOUT == "E007"
        assert ErrorCode.CONNECTION_LOST == "E008"
        assert ErrorCode.INVALID_STATE == "E009"
        assert ErrorCode.GAME_OVER == "E010"


# ---------------------------------------------------------------------------
# _parse_era
# ---------------------------------------------------------------------------

class TestParseEra:
    """Tests for _parse_era helper."""

    def test_none_returns_default(self) -> None:
        assert _parse_era(None) == 0
        assert _parse_era(None, default=5) == 5

    def test_int_passthrough(self) -> None:
        assert _parse_era(3) == 3

    def test_string_known_era(self) -> None:
        assert _parse_era("ERA_ANCIENT") == 0
        assert _parse_era("ERA_CLASSICAL") == 1
        assert _parse_era("ERA_MEDIEVAL") == 2
        assert _parse_era("ERA_INFORMATION") == 7

    def test_string_unknown_era(self) -> None:
        assert _parse_era("ERA_UNKNOWN") == 0
        assert _parse_era("ERA_UNKNOWN", default=99) == 99

    def test_other_type_returns_default(self) -> None:
        """Cover line 1713: non-int, non-str type returns default."""
        assert _parse_era(3.14) == 0  # type: ignore[arg-type]
        assert _parse_era([1, 2]) == 0  # type: ignore[arg-type]


class TestEraNameMapping:
    """Tests for ERA_NAME_TO_INDEX."""

    def test_all_eras_present(self) -> None:
        expected_eras = [
            "ERA_ANCIENT", "ERA_CLASSICAL", "ERA_MEDIEVAL", "ERA_RENAISSANCE",
            "ERA_INDUSTRIAL", "ERA_MODERN", "ERA_ATOMIC", "ERA_INFORMATION", "ERA_FUTURE",
        ]
        for i, era in enumerate(expected_eras):
            assert ERA_NAME_TO_INDEX[era] == i


# ---------------------------------------------------------------------------
# validate_state_json
# ---------------------------------------------------------------------------

class TestValidateStateJson:
    """Tests for validate_state_json."""

    def test_valid_json(self) -> None:
        json_str = json.dumps({
            "turn_id": 1,
            "map": {"width": 10, "height": 10},
        })
        is_valid, error = validate_state_json(json_str)
        assert is_valid is True
        assert error is None

    def test_valid_with_turn_key(self) -> None:
        """'turn' key should also be accepted."""
        json_str = json.dumps({
            "turn": 1,
            "map": {"width": 10, "height": 10},
        })
        is_valid, _error = validate_state_json(json_str)
        assert is_valid is True

    def test_missing_turn(self) -> None:
        json_str = json.dumps({"map": {"width": 10, "height": 10}})
        is_valid, error = validate_state_json(json_str)
        assert is_valid is False
        assert "turn" in error.lower()

    def test_missing_map(self) -> None:
        json_str = json.dumps({"turn_id": 1})
        is_valid, error = validate_state_json(json_str)
        assert is_valid is False
        assert "map" in error.lower()

    def test_missing_map_dimensions(self) -> None:
        json_str = json.dumps({"turn_id": 1, "map": {}})
        is_valid, error = validate_state_json(json_str)
        assert is_valid is False
        assert "width" in error.lower() or "height" in error.lower()

    def test_invalid_json(self) -> None:
        is_valid, error = validate_state_json("{bad json")
        assert is_valid is False
        assert "Invalid JSON" in error

    def test_exception_handling_empty_string(self) -> None:
        """Empty string is invalid JSON."""
        is_valid, _error = validate_state_json("")
        assert is_valid is False

    def test_exception_handling_general(self) -> None:
        """Cover lines 1743-1744: general exception (not JSONDecodeError).

        When map data is an integer, `"width" not in map_data` raises TypeError
        ('argument of type int is not iterable'), caught by the generic handler.
        """
        json_str = json.dumps({"turn_id": 1, "map": 42})
        is_valid, error = validate_state_json(json_str)
        assert is_valid is False
        assert "Validation error" in error


# ---------------------------------------------------------------------------
# Convenience command factories
# ---------------------------------------------------------------------------

class TestCommandFactories:
    """Tests for create_*_command helper functions."""

    def test_create_end_turn_command(self) -> None:
        """Cover line 1756."""
        cmd = create_end_turn_command(msg_id=42)
        assert cmd.msg_id == 42
        assert len(cmd.commands) == 1
        assert cmd.commands[0].action_type == ActionType.END_TURN

    def test_create_unit_move_command(self) -> None:
        """Cover line 1779."""
        cmd = create_unit_move_command(msg_id=10, unit_id=5, target_x=3, target_y=7)
        assert cmd.msg_id == 10
        assert cmd.commands[0].action_type == ActionType.UNIT_MOVE
        assert cmd.commands[0].unit_id == 5
        assert cmd.commands[0].target == Position(x=3, y=7)

    def test_create_city_produce_command(self) -> None:
        """Cover line 1808."""
        cmd = create_city_produce_command(msg_id=20, city_id=1, item_id=42, item_type="building")
        assert cmd.msg_id == 20
        assert cmd.commands[0].action_type == ActionType.CITY_PRODUCE
        assert cmd.commands[0].city_id == 1
        assert cmd.commands[0].item_id == 42
        assert cmd.commands[0].item_type == "building"

    def test_create_city_produce_default_item_type(self) -> None:
        cmd = create_city_produce_command(msg_id=1, city_id=1, item_id=1)
        assert cmd.commands[0].item_type == "unit"


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------

class TestSerializationRoundTrip:
    """Tests for serialization round-trip consistency."""

    def test_position_round_trip(self) -> None:
        original = Position(x=42, y=99)
        restored = Position.from_dict(original.to_dict())
        assert original == restored

    def test_unit_data_round_trip(self) -> None:
        original = UnitData(id=1, x=5, y=5, unit_type="warrior", owner_id=0,
                            health=75, moves_remaining=1.5)
        restored = UnitData.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.unit_type == original.unit_type

    def test_tile_data_round_trip(self) -> None:
        original = TileData(x=3, y=7, terrain="TERRAIN_PLAINS",
                            features=["FEATURE_FOREST"], resource="RESOURCE_IRON", owner_id=1)
        restored = TileData.from_dict(original.to_dict())
        assert restored.terrain == original.terrain
        assert restored.resource == original.resource

    def test_city_data_round_trip(self) -> None:
        original = CityData(id=5, name="TestCity", owner_id=0, x=10, y=12,
                            population=7, is_capital=True, buildings=["BUILDING_MONUMENT"])
        restored = CityData.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.is_capital == original.is_capital

    def test_player_data_round_trip(self) -> None:
        original = PlayerData(
            id=0, civilization="CIVILIZATION_ROME", gold=250.5,
            science_per_turn=42.0, techs_researched=["TECH_POTTERY"],
            civics_researched=["CIVIC_CODE_OF_LAWS"], current_civic="CIVIC_FOREIGN_TRADE",
        )
        restored = PlayerData.from_dict(original.to_dict())
        assert restored.gold == original.gold
        assert restored.civics_researched == original.civics_researched

    def test_legal_action_round_trip(self) -> None:
        original = LegalAction(action_type="move", unit_id=42,
                               targets=[(5, 6), (7, 8)], items=[1, 2, 3], item_type="unit")
        restored = LegalAction.from_dict(original.to_dict())
        assert restored.action_type == original.action_type
        assert restored.targets == original.targets
        assert restored.items == original.items

    def test_command_message_round_trip(self) -> None:
        original = CommandMessage(
            msg_id=42,
            commands=[LuaCommand(action_type=ActionType.UNIT_MOVE, unit_id=1, target=Position(5, 5))],
        )
        restored = CommandMessage.from_json(original.to_json())
        assert restored.msg_id == original.msg_id

    def test_ack_message_round_trip(self) -> None:
        original = AckMessage(msg_id=99, success=False, error_code="E001",
                              error_message="Test", executed=[{"a": 1}])
        restored = AckMessage.from_json(original.to_json())
        assert restored.msg_id == original.msg_id
        assert restored.success == original.success
        assert restored.error_code == original.error_code
        assert restored.error_message == original.error_message

    def test_error_message_round_trip(self) -> None:
        original = ErrorMessage(
            msg_id=5, error_code="E003", message="Not found",
            details={"unit_id": 42},
        )
        restored = ErrorMessage.from_json(original.to_json())
        assert restored.error_code == original.error_code
        assert restored.message == original.message
        assert restored.details == original.details
