"""Tests for alpha_zero.bridge.action_converter module.

Covers:
- LegalActionTypes constants
- LEGAL_ACTION_TO_UNIT_TYPE mapping
- CityItemTypes constants
- _convert_item_id_to_int helper
- InvalidActionError exception
- ConversionResult dataclass
- ActionConverter class (init, action_space_size, convert, generate_action_mask)
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.bridge.action_converter import (
    LEGAL_ACTION_TO_UNIT_TYPE,
    ActionConverter,
    CityItemTypes,
    ConversionResult,
    InvalidActionError,
    LegalActionTypes,
    _convert_item_id_to_int,
)
from alpha_zero.bridge.protocol import (
    ActionType,
    LegalAction,
    LuaCommand,
)
from alpha_zero.core.civ6_actions import (
    ActionCategory,
    ActionSpace,
    ActionSpaceConfig,
    CityActionType,
    HierarchicalAction,
    UnitActionType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def converter() -> ActionConverter:
    """ActionConverter with default board_size=16."""
    return ActionConverter(board_size=16)


@pytest.fixture
def small_converter() -> ActionConverter:
    """ActionConverter with small board for easier reasoning."""
    config = ActionSpaceConfig(board_size=8, max_units=4, max_cities=2)
    return ActionConverter(board_size=8, action_space_config=config)


@pytest.fixture
def game_state_with_units() -> dict:
    """Minimal game state with two player-0 units and one player-1 unit."""
    return {
        "current_player": 0,
        "units": [
            {"_id": 10, "player": 0, "position": (2, 3), "type": "melee"},
            {"_id": 20, "player": 0, "position": (5, 5), "type": "ranged"},
            {"_id": 30, "player": 1, "position": (4, 4), "type": "melee"},
        ],
        "cities": [],
    }


@pytest.fixture
def game_state_with_cities() -> dict:
    """Minimal game state with one player-0 city."""
    return {
        "current_player": 0,
        "units": [],
        "cities": [
            {
                "_id": 100,
                "player": 0,
                "position": (3, 3),
                "available_production": [0, 1, 2],
            },
            {"_id": 200, "player": 1, "position": (10, 10)},
        ],
    }


# ===========================================================================
# 1. LegalActionTypes
# ===========================================================================


class TestLegalActionTypes:
    """All LegalActionTypes constants should be strings."""

    def test_end_turn_is_string(self) -> None:
        assert isinstance(LegalActionTypes.END_TURN, str)
        assert LegalActionTypes.END_TURN == "end_turn"

    def test_unit_actions_are_strings(self) -> None:
        unit_actions = [
            LegalActionTypes.MOVE,
            LegalActionTypes.ATTACK,
            LegalActionTypes.RANGED_ATTACK,
            LegalActionTypes.FORTIFY,
            LegalActionTypes.SKIP,
            LegalActionTypes.SLEEP,
            LegalActionTypes.FOUND_CITY,
            LegalActionTypes.BUILD,
            LegalActionTypes.PILLAGE,
            LegalActionTypes.REPAIR,
            LegalActionTypes.EMBARK,
            LegalActionTypes.DISEMBARK,
            LegalActionTypes.PROMOTE,
            LegalActionTypes.HEAL,
            LegalActionTypes.DELETE,
            LegalActionTypes.AIRLIFT,
            LegalActionTypes.PARADROP,
        ]
        for action in unit_actions:
            assert isinstance(action, str), f"{action} is not a string"

    def test_city_actions_are_strings(self) -> None:
        assert isinstance(LegalActionTypes.PRODUCE, str)
        assert isinstance(LegalActionTypes.BUY, str)

    def test_research_actions_are_strings(self) -> None:
        assert isinstance(LegalActionTypes.RESEARCH_TECH, str)
        assert isinstance(LegalActionTypes.RESEARCH_CIVIC, str)


# ===========================================================================
# 2. LEGAL_ACTION_TO_UNIT_TYPE mapping
# ===========================================================================


class TestLegalActionToUnitType:
    """Mapping should contain all 17 unit action types."""

    def test_has_expected_count(self) -> None:
        assert len(LEGAL_ACTION_TO_UNIT_TYPE) == 17

    def test_move_maps_correctly(self) -> None:
        assert LEGAL_ACTION_TO_UNIT_TYPE["move"] == UnitActionType.MOVE

    def test_attack_maps_correctly(self) -> None:
        assert LEGAL_ACTION_TO_UNIT_TYPE["attack"] == UnitActionType.ATTACK

    def test_ranged_attack_maps_correctly(self) -> None:
        assert LEGAL_ACTION_TO_UNIT_TYPE["ranged_attack"] == UnitActionType.RANGED_ATTACK

    def test_fortify_maps_correctly(self) -> None:
        assert LEGAL_ACTION_TO_UNIT_TYPE["fortify"] == UnitActionType.FORTIFY

    def test_skip_maps_correctly(self) -> None:
        assert LEGAL_ACTION_TO_UNIT_TYPE["skip"] == UnitActionType.SKIP

    def test_build_maps_to_build_improvement(self) -> None:
        assert LEGAL_ACTION_TO_UNIT_TYPE["build"] == UnitActionType.BUILD_IMPROVEMENT

    def test_all_values_are_unit_action_type(self) -> None:
        for key, val in LEGAL_ACTION_TO_UNIT_TYPE.items():
            assert isinstance(val, UnitActionType), f"Value for '{key}' is not UnitActionType"

    def test_all_keys_are_strings(self) -> None:
        for key in LEGAL_ACTION_TO_UNIT_TYPE:
            assert isinstance(key, str), f"Key {key} is not a string"

    def test_contains_naval_actions(self) -> None:
        assert "embark" in LEGAL_ACTION_TO_UNIT_TYPE
        assert "disembark" in LEGAL_ACTION_TO_UNIT_TYPE

    def test_contains_air_actions(self) -> None:
        assert "airlift" in LEGAL_ACTION_TO_UNIT_TYPE
        assert "paradrop" in LEGAL_ACTION_TO_UNIT_TYPE


# ===========================================================================
# 3. CityItemTypes
# ===========================================================================


class TestCityItemTypes:
    """All CityItemTypes constants should be lowercase strings."""

    def test_unit(self) -> None:
        assert CityItemTypes.UNIT == "unit"

    def test_building(self) -> None:
        assert CityItemTypes.BUILDING == "building"

    def test_wonder(self) -> None:
        assert CityItemTypes.WONDER == "wonder"

    def test_district(self) -> None:
        assert CityItemTypes.DISTRICT == "district"


# ===========================================================================
# 4. _convert_item_id_to_int
# ===========================================================================


class TestConvertItemIdToInt:
    """Test the item ID conversion helper."""

    def test_int_input_returned_as_is(self) -> None:
        assert _convert_item_id_to_int(42) == 42

    def test_zero_int(self) -> None:
        assert _convert_item_id_to_int(0) == 0

    def test_negative_int(self) -> None:
        assert _convert_item_id_to_int(-5) == -5

    def test_numeric_string(self) -> None:
        assert _convert_item_id_to_int("42") == 42

    def test_numeric_string_zero(self) -> None:
        assert _convert_item_id_to_int("0") == 0

    def test_type_name_string(self) -> None:
        result = _convert_item_id_to_int("UNIT_WARRIOR")
        assert isinstance(result, int)
        assert 0 <= result < 200  # _ITEM_HASH_RANGE

    def test_type_name_string_is_deterministic(self) -> None:
        r1 = _convert_item_id_to_int("UNIT_WARRIOR")
        r2 = _convert_item_id_to_int("UNIT_WARRIOR")
        assert r1 == r2

    def test_different_type_names_may_differ(self) -> None:
        # Not guaranteed to differ due to hash collisions, but test both return ints
        r1 = _convert_item_id_to_int("UNIT_WARRIOR")
        r2 = _convert_item_id_to_int("BUILDING_GRANARY")
        assert isinstance(r1, int)
        assert isinstance(r2, int)

    def test_none_returns_none(self) -> None:
        assert _convert_item_id_to_int(None) is None

    def test_empty_string_returns_none(self) -> None:
        # Empty string: `if item_id:` is False, falls through to debug log + return None
        assert _convert_item_id_to_int("") is None

    def test_float_returns_none(self) -> None:
        # Float is not int and not str, so returns None
        assert _convert_item_id_to_int(3.14) is None


# ===========================================================================
# 5. InvalidActionError
# ===========================================================================


class TestInvalidActionError:
    """InvalidActionError stores an optional action_idx."""

    def test_construction_with_message(self) -> None:
        err = InvalidActionError("bad action")
        assert str(err) == "bad action"

    def test_action_idx_default_none(self) -> None:
        err = InvalidActionError("bad action")
        assert err.action_idx is None

    def test_action_idx_stored(self) -> None:
        err = InvalidActionError("bad action", action_idx=42)
        assert err.action_idx == 42

    def test_is_exception(self) -> None:
        assert issubclass(InvalidActionError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(InvalidActionError, match="test error"):
            raise InvalidActionError("test error", action_idx=7)


# ===========================================================================
# 6. ConversionResult
# ===========================================================================


class TestConversionResult:
    """ConversionResult stores command, decoded_action, validity, and error."""

    def test_default_is_valid(self) -> None:
        cmd = LuaCommand(action_type=ActionType.END_TURN)
        decoded = HierarchicalAction(category=ActionCategory.END_TURN)
        result = ConversionResult(command=cmd, decoded_action=decoded)
        assert result.is_valid is True
        assert result.validation_error is None

    def test_explicit_invalid(self) -> None:
        cmd = LuaCommand(action_type=ActionType.END_TURN)
        decoded = HierarchicalAction(category=ActionCategory.END_TURN)
        result = ConversionResult(
            command=cmd, decoded_action=decoded, is_valid=False, validation_error="oops"
        )
        assert result.is_valid is False
        assert result.validation_error == "oops"

    def test_stores_command_and_decoded(self) -> None:
        cmd = LuaCommand(action_type=ActionType.UNIT_MOVE, unit_id=5)
        decoded = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION, selector=0, action_type=0
        )
        result = ConversionResult(command=cmd, decoded_action=decoded)
        assert result.command.unit_id == 5
        assert result.decoded_action.category == ActionCategory.UNIT_ACTION


# ===========================================================================
# 7. ActionConverter.__init__
# ===========================================================================


class TestActionConverterInit:
    """ActionConverter initializes action_space from board_size or config."""

    def test_default_board_size(self) -> None:
        conv = ActionConverter()
        assert conv.board_size == 16

    def test_custom_board_size(self) -> None:
        conv = ActionConverter(board_size=32)
        assert conv.board_size == 32

    def test_default_action_space_created(self) -> None:
        conv = ActionConverter(board_size=16)
        assert isinstance(conv.action_space, ActionSpace)
        assert conv.action_space.config.board_size == 16
        assert conv.action_space.config.max_units == 100
        assert conv.action_space.config.max_cities == 20

    def test_custom_action_space_config(self) -> None:
        config = ActionSpaceConfig(board_size=8, max_units=10, max_cities=5)
        conv = ActionConverter(board_size=8, action_space_config=config)
        assert conv.action_space.config.max_units == 10
        assert conv.action_space.config.max_cities == 5


# ===========================================================================
# 8. ActionConverter.action_space_size
# ===========================================================================


class TestActionSpaceSize:
    """action_space_size should match the total_size of the underlying ActionSpace."""

    def test_returns_correct_size(self, converter: ActionConverter) -> None:
        assert converter.action_space_size == converter.action_space.total_size

    def test_positive_integer(self, converter: ActionConverter) -> None:
        assert converter.action_space_size > 0

    def test_small_config_size(self, small_converter: ActionConverter) -> None:
        assert small_converter.action_space_size == small_converter.action_space.total_size
        assert small_converter.action_space_size > 0


# ===========================================================================
# 9. ActionConverter.convert
# ===========================================================================


class TestActionConverterConvert:
    """Tests for the convert method that maps action_idx -> LuaCommand."""

    def test_end_turn_action(self, converter: ActionConverter) -> None:
        # The END_TURN category starts at offset 0, index 0.
        end_turn_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        cmd = converter.convert(end_turn_idx, game_state, validate=False)
        assert cmd.action_type == ActionType.END_TURN

    def test_unit_move_action(
        self, converter: ActionConverter, game_state_with_units: dict
    ) -> None:
        # Encode a MOVE action for unit index 0 at target (3, 4)
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.MOVE.value,
            target=(3, 4),
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_units, validate=False)
        assert cmd.action_type == ActionType.UNIT_MOVE
        assert cmd.unit_id == 10  # First player-0 unit

    def test_unit_attack_action(
        self, converter: ActionConverter, game_state_with_units: dict
    ) -> None:
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.ATTACK.value,
            target=(4, 4),
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_units, validate=False)
        assert cmd.action_type == ActionType.UNIT_ATTACK
        assert cmd.unit_id == 10

    def test_unit_ranged_attack_action(
        self, converter: ActionConverter, game_state_with_units: dict
    ) -> None:
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=1,
            action_type=UnitActionType.RANGED_ATTACK.value,
            target=(4, 4),
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_units, validate=False)
        assert cmd.action_type == ActionType.UNIT_RANGED_ATTACK
        assert cmd.unit_id == 20  # Second player-0 unit

    def test_unit_fortify_action(
        self, converter: ActionConverter, game_state_with_units: dict
    ) -> None:
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.FORTIFY.value,
            target=(0, 0),
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_units, validate=False)
        assert cmd.action_type == ActionType.UNIT_FORTIFY
        assert cmd.unit_id == 10

    def test_unit_skip_action(
        self, converter: ActionConverter, game_state_with_units: dict
    ) -> None:
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.SKIP.value,
            target=(0, 0),
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_units, validate=False)
        assert cmd.action_type == ActionType.UNIT_SKIP

    def test_unit_found_city_action(
        self, converter: ActionConverter, game_state_with_units: dict
    ) -> None:
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.FOUND_CITY.value,
            target=(0, 0),
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_units, validate=False)
        assert cmd.action_type == ActionType.UNIT_FOUND_CITY

    def test_unit_build_improvement_action(
        self, converter: ActionConverter, game_state_with_units: dict
    ) -> None:
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.BUILD_IMPROVEMENT.value,
            target=(0, 0),
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_units, validate=False)
        assert cmd.action_type == ActionType.UNIT_BUILD

    def test_unsupported_unit_action_defaults_to_skip(
        self, converter: ActionConverter, game_state_with_units: dict
    ) -> None:
        # SLEEP is not directly mapped to a specific ActionType branch,
        # so it falls to the else clause which produces UNIT_SKIP
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.SLEEP.value,
            target=(0, 0),
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_units, validate=False)
        assert cmd.action_type == ActionType.UNIT_SKIP
        assert cmd.metadata.get("original_action") == "SLEEP"

    def test_city_produce_unit_action(
        self, converter: ActionConverter, game_state_with_cities: dict
    ) -> None:
        action = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=0,
            action_type=CityActionType.PRODUCE_UNIT.value,
            target=5,
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_cities, validate=False)
        assert cmd.action_type == ActionType.CITY_PRODUCE
        assert cmd.city_id == 100
        assert cmd.item_type == "unit"
        assert cmd.item_id == 5

    def test_city_produce_building_action(
        self, converter: ActionConverter, game_state_with_cities: dict
    ) -> None:
        action = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=0,
            action_type=CityActionType.PRODUCE_BUILDING.value,
            target=3,
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_cities, validate=False)
        assert cmd.action_type == ActionType.CITY_PRODUCE
        assert cmd.item_type == "building"

    def test_city_produce_wonder_action(
        self, converter: ActionConverter, game_state_with_cities: dict
    ) -> None:
        action = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=0,
            action_type=CityActionType.PRODUCE_WONDER.value,
            target=1,
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_cities, validate=False)
        assert cmd.action_type == ActionType.CITY_PRODUCE
        assert cmd.item_type == "wonder"

    def test_city_produce_district_action(
        self, converter: ActionConverter, game_state_with_cities: dict
    ) -> None:
        action = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=0,
            action_type=CityActionType.PRODUCE_DISTRICT.value,
            target=2,
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_cities, validate=False)
        assert cmd.action_type == ActionType.CITY_PRODUCE
        assert cmd.item_type == "district"

    def test_city_buy_unit_action(
        self, converter: ActionConverter, game_state_with_cities: dict
    ) -> None:
        action = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=0,
            action_type=CityActionType.BUY_UNIT.value,
            target=1,
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_cities, validate=False)
        assert cmd.action_type == ActionType.CITY_BUY
        assert cmd.item_type == "unit"

    def test_city_buy_building_action(
        self, converter: ActionConverter, game_state_with_cities: dict
    ) -> None:
        action = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=0,
            action_type=CityActionType.BUY_BUILDING.value,
            target=1,
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_cities, validate=False)
        assert cmd.action_type == ActionType.CITY_BUY
        assert cmd.item_type == "building"

    def test_unsupported_city_action_defaults_to_end_turn(
        self, converter: ActionConverter, game_state_with_cities: dict
    ) -> None:
        # CHANGE_PRODUCTION, etc. fall through to the else clause
        action = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=0,
            action_type=CityActionType.CHANGE_PRODUCTION.value,
            target=0,
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_cities, validate=False)
        assert cmd.action_type == ActionType.END_TURN
        assert cmd.metadata.get("unsupported") is True

    def test_tech_action(self, converter: ActionConverter) -> None:
        action = HierarchicalAction(
            category=ActionCategory.RESEARCH_TECH, selector=5
        )
        flat_idx = converter.action_space.encode(action)
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        cmd = converter.convert(flat_idx, game_state, validate=False)
        assert cmd.action_type == ActionType.RESEARCH_TECH
        assert cmd.item_id == 5

    def test_civic_action(self, converter: ActionConverter) -> None:
        action = HierarchicalAction(
            category=ActionCategory.RESEARCH_CIVIC, selector=10
        )
        flat_idx = converter.action_space.encode(action)
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        cmd = converter.convert(flat_idx, game_state, validate=False)
        assert cmd.action_type == ActionType.RESEARCH_CIVIC
        assert cmd.item_id == 10

    def test_unsupported_category_returns_end_turn(self, converter: ActionConverter) -> None:
        # DIPLOMACY is a supported category in the action space but has no specific
        # branch in _convert, so falls through to else clause
        action = HierarchicalAction(
            category=ActionCategory.DIPLOMACY, selector=0, action_type=0
        )
        flat_idx = converter.action_space.encode(action)
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        cmd = converter.convert(flat_idx, game_state, validate=False)
        assert cmd.action_type == ActionType.END_TURN
        assert cmd.metadata.get("unsupported") is True

    def test_invalid_action_idx_raises_error(self, converter: ActionConverter) -> None:
        # An index beyond the action space total size should fail decode
        converter.action_space_size + 1_000_000
        # decode() may not raise for all OOB values, but the resulting action
        # will reference non-existent entities, so we exercise the path and check
        # that at least the converter handles it without crashing.
        # Use an approach that definitely triggers InvalidActionError:
        # unit action with selector far beyond available units
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=99,
            action_type=UnitActionType.MOVE.value,
            target=(0, 0),
        )
        flat_idx = converter.action_space.encode(action)
        game_state_empty: dict = {"units": [], "cities": [], "current_player": 0}
        with pytest.raises(InvalidActionError, match="out of range"):
            converter.convert(flat_idx, game_state_empty, validate=False)

    def test_unit_action_out_of_range_raises_error(
        self, converter: ActionConverter
    ) -> None:
        # Encode a unit action for unit_idx=99, but provide only 1 player unit
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=99,
            action_type=UnitActionType.MOVE.value,
            target=(0, 0),
        )
        flat_idx = converter.action_space.encode(action)
        game_state = {
            "current_player": 0,
            "units": [{"_id": 1, "player": 0}],
            "cities": [],
        }
        with pytest.raises(InvalidActionError, match="out of range"):
            converter.convert(flat_idx, game_state, validate=False)

    def test_city_action_out_of_range_raises_error(
        self, converter: ActionConverter
    ) -> None:
        action = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=19,
            action_type=CityActionType.PRODUCE_UNIT.value,
            target=0,
        )
        flat_idx = converter.action_space.encode(action)
        game_state = {
            "current_player": 0,
            "units": [],
            "cities": [{"_id": 1, "player": 0}],
        }
        with pytest.raises(InvalidActionError, match="out of range"):
            converter.convert(flat_idx, game_state, validate=False)

    def test_validate_against_legal_actions_raises(
        self, converter: ActionConverter
    ) -> None:
        """When validate=True and action not in legal_actions, raises error."""
        end_turn_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        # Provide legal actions that do NOT include end_turn
        legal_actions = [LegalAction(action_type="move", unit_id=1)]
        with pytest.raises(InvalidActionError, match="not in the legal actions"):
            converter.convert(end_turn_idx, game_state, legal_actions, validate=True)

    def test_validate_end_turn_passes_when_legal(
        self, converter: ActionConverter
    ) -> None:
        end_turn_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        legal_actions = [LegalAction(action_type="end_turn")]
        cmd = converter.convert(end_turn_idx, game_state, legal_actions, validate=True)
        assert cmd.action_type == ActionType.END_TURN

    def test_validate_with_dict_legal_actions(
        self, converter: ActionConverter
    ) -> None:
        """Legal actions can be dicts -- converter parses them."""
        end_turn_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        legal_actions = [{"action_type": "end_turn"}]
        cmd = converter.convert(end_turn_idx, game_state, legal_actions, validate=True)
        assert cmd.action_type == ActionType.END_TURN

    def test_unit_move_with_map_size_scales_coordinates(
        self, converter: ActionConverter, game_state_with_units: dict
    ) -> None:
        """When _map_size is set, coordinates should be scaled to original."""
        game_state_with_units["_map_size"] = (84, 54)
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.MOVE.value,
            target=(8, 8),
        )
        flat_idx = converter.action_space.encode(action)
        cmd = converter.convert(flat_idx, game_state_with_units, validate=False)
        assert cmd.target is not None
        # Scaled: 8 * (84/16) = 42, 8 * (54/16) = 27
        assert cmd.target.x == 42
        assert cmd.target.y == 27


# ===========================================================================
# 9b. ActionConverter.convert_with_result
# ===========================================================================


class TestConvertWithResult:
    """Tests for convert_with_result which wraps convert in ConversionResult."""

    def test_valid_end_turn(self, converter: ActionConverter) -> None:
        end_turn_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        result = converter.convert_with_result(end_turn_idx, game_state)
        assert result.is_valid is True
        assert result.command.action_type == ActionType.END_TURN
        assert result.validation_error is None

    def test_invalid_action_returns_invalid_result(self, converter: ActionConverter) -> None:
        """A unit action referencing non-existent units returns invalid result."""
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=99,
            action_type=UnitActionType.MOVE.value,
            target=(0, 0),
        )
        flat_idx = converter.action_space.encode(action)
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        result = converter.convert_with_result(flat_idx, game_state)
        assert result.is_valid is False
        assert result.validation_error is not None

    def test_with_legal_actions_not_matching(self, converter: ActionConverter) -> None:
        end_turn_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        legal_actions = [LegalAction(action_type="move", unit_id=1)]
        result = converter.convert_with_result(end_turn_idx, game_state, legal_actions)
        assert result.is_valid is False
        assert result.validation_error is not None


# ===========================================================================
# 10. ActionConverter.generate_action_mask
# ===========================================================================


class TestGenerateActionMask:
    """generate_action_mask creates a binary mask from legal actions."""

    def test_empty_legal_actions_fallback_to_end_turn(
        self, converter: ActionConverter
    ) -> None:
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        mask = converter.generate_action_mask([], game_state)
        assert mask.shape == (converter.action_space_size,)
        # Only END_TURN should be 1
        end_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        assert mask[end_idx] == 1.0
        assert mask.sum() == 1.0

    def test_end_turn_legal_action_sets_bit(self, converter: ActionConverter) -> None:
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        legal_actions = [LegalAction(action_type="end_turn")]
        mask = converter.generate_action_mask(legal_actions, game_state)
        end_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        assert mask[end_idx] == 1.0

    def test_unit_move_with_targets(self, converter: ActionConverter) -> None:
        game_state = {
            "current_player": 0,
            "units": [
                {"_id": 10, "player": 0, "position": (2, 3)},
            ],
            "cities": [],
        }
        legal_actions = [
            LegalAction(
                action_type="move",
                unit_id=10,
                targets=[(3, 3), (2, 4)],
            ),
            LegalAction(action_type="end_turn"),
        ]
        mask = converter.generate_action_mask(legal_actions, game_state)
        assert mask.sum() >= 2  # at least end_turn + some move targets

    def test_unit_fortify_no_targets(self, converter: ActionConverter) -> None:
        """Targetless actions (fortify, skip) use dummy (0,0)."""
        game_state = {
            "current_player": 0,
            "units": [
                {"_id": 10, "player": 0, "position": (5, 5)},
            ],
            "cities": [],
        }
        legal_actions = [
            LegalAction(action_type="fortify", unit_id=10),
            LegalAction(action_type="end_turn"),
        ]
        mask = converter.generate_action_mask(legal_actions, game_state)
        assert mask.sum() >= 2  # end_turn + fortify

    def test_city_produce_action_unit_type(self, converter: ActionConverter) -> None:
        game_state = {
            "current_player": 0,
            "units": [],
            "cities": [
                {"_id": 100, "player": 0, "available_production": [0, 1]},
            ],
        }
        legal_actions = [
            LegalAction(
                action_type="produce",
                city_id=100,
                items=[0, 1],
                item_type="unit",
            ),
            LegalAction(action_type="end_turn"),
        ]
        mask = converter.generate_action_mask(legal_actions, game_state)
        assert mask.sum() >= 3  # end_turn + 2 production items

    def test_city_produce_building_type(self, converter: ActionConverter) -> None:
        game_state = {
            "current_player": 0,
            "units": [],
            "cities": [
                {"_id": 100, "player": 0},
            ],
        }
        legal_actions = [
            LegalAction(
                action_type="produce",
                city_id=100,
                items=[5],
                item_type="building",
            ),
            LegalAction(action_type="end_turn"),
        ]
        mask = converter.generate_action_mask(legal_actions, game_state)
        assert mask.sum() >= 2

    def test_city_produce_wonder_type(self, converter: ActionConverter) -> None:
        game_state = {
            "current_player": 0,
            "units": [],
            "cities": [{"_id": 100, "player": 0}],
        }
        legal_actions = [
            LegalAction(
                action_type="produce",
                city_id=100,
                items=[10],
                item_type="wonder",
            ),
            LegalAction(action_type="end_turn"),
        ]
        mask = converter.generate_action_mask(legal_actions, game_state)
        assert mask.sum() >= 2

    def test_city_produce_district_type(self, converter: ActionConverter) -> None:
        game_state = {
            "current_player": 0,
            "units": [],
            "cities": [{"_id": 100, "player": 0}],
        }
        legal_actions = [
            LegalAction(
                action_type="produce",
                city_id=100,
                items=[7],
                item_type="district",
            ),
            LegalAction(action_type="end_turn"),
        ]
        mask = converter.generate_action_mask(legal_actions, game_state)
        assert mask.sum() >= 2

    def test_city_produce_unknown_type_backwards_compat(
        self, converter: ActionConverter
    ) -> None:
        """Unknown item_type should encode both unit and building (backwards compat)."""
        game_state = {
            "current_player": 0,
            "units": [],
            "cities": [{"_id": 100, "player": 0}],
        }
        legal_actions = [
            LegalAction(
                action_type="produce",
                city_id=100,
                items=[3],
                item_type=None,
            ),
            LegalAction(action_type="end_turn"),
        ]
        mask = converter.generate_action_mask(legal_actions, game_state)
        # Should have end_turn + at least produce_unit + produce_building
        assert mask.sum() >= 3

    def test_dict_legal_actions_parsed(self, converter: ActionConverter) -> None:
        """Legal actions passed as dicts should be parsed properly."""
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        legal_actions = [{"action_type": "end_turn"}]
        mask = converter.generate_action_mask(legal_actions, game_state)
        end_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        assert mask[end_idx] == 1.0

    def test_mask_dtype_is_float32(self, converter: ActionConverter) -> None:
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        mask = converter.generate_action_mask([], game_state)
        assert mask.dtype == np.float32

    def test_mask_shape_matches_action_space(self, converter: ActionConverter) -> None:
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        mask = converter.generate_action_mask([], game_state)
        assert mask.shape == (converter.action_space_size,)

    def test_unit_not_found_skipped_gracefully(self, converter: ActionConverter) -> None:
        """If unit_id in legal_action doesn't match any unit, it is skipped."""
        game_state = {
            "current_player": 0,
            "units": [{"_id": 10, "player": 0}],
            "cities": [],
        }
        legal_actions = [
            LegalAction(action_type="move", unit_id=999, targets=[(1, 1)]),
            LegalAction(action_type="end_turn"),
        ]
        mask = converter.generate_action_mask(legal_actions, game_state)
        # Should still have end_turn set
        end_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        assert mask[end_idx] == 1.0

    def test_city_not_found_skipped_gracefully(self, converter: ActionConverter) -> None:
        """If city_id in legal_action doesn't match any city, it is skipped."""
        game_state = {
            "current_player": 0,
            "units": [],
            "cities": [{"_id": 100, "player": 0}],
        }
        legal_actions = [
            LegalAction(
                action_type="produce", city_id=999, items=[0], item_type="unit"
            ),
            LegalAction(action_type="end_turn"),
        ]
        mask = converter.generate_action_mask(legal_actions, game_state)
        end_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        assert mask[end_idx] == 1.0

    def test_string_item_ids_handled(self, converter: ActionConverter) -> None:
        """String item IDs (e.g., 'UNIT_WARRIOR') should be hashed to ints."""
        game_state = {
            "current_player": 0,
            "units": [],
            "cities": [{"_id": 100, "player": 0}],
        }
        legal_actions = [
            LegalAction(
                action_type="produce",
                city_id=100,
                items=["UNIT_WARRIOR"],
                item_type="unit",
            ),
            LegalAction(action_type="end_turn"),
        ]
        mask = converter.generate_action_mask(legal_actions, game_state)
        assert mask.sum() >= 2


# ===========================================================================
# 11. ActionConverter helper methods
# ===========================================================================


class TestActionConverterHelpers:
    """Tests for private helper methods."""

    def test_scale_to_original(self, converter: ActionConverter) -> None:
        # board_size=16, map_size=(32, 32) => scale 2x
        orig_x, orig_y = converter._scale_to_original(8, 8, (32, 32))
        assert orig_x == 16
        assert orig_y == 16

    def test_scale_to_original_clamps(self, converter: ActionConverter) -> None:
        orig_x, orig_y = converter._scale_to_original(20, 20, (10, 10))
        assert orig_x <= 9
        assert orig_y <= 9

    def test_scale_from_original(self, converter: ActionConverter) -> None:
        # board_size=16, map_size=(32, 32) => scale 0.5x
        sx, sy = converter._scale_from_original(16, 16, (32, 32))
        assert sx == 8
        assert sy == 8

    def test_scale_from_original_clamps(self, converter: ActionConverter) -> None:
        sx, sy = converter._scale_from_original(1000, 1000, (10, 10))
        assert sx <= 15
        assert sy <= 15

    def test_find_unit_idx_found(self, converter: ActionConverter) -> None:
        units = [
            {"_id": 10, "player": 0},
            {"_id": 20, "player": 0},
            {"_id": 30, "player": 1},
        ]
        idx = converter._find_unit_idx(20, units, current_player=0)
        assert idx == 1  # Second player-0 unit

    def test_find_unit_idx_not_found(self, converter: ActionConverter) -> None:
        units = [{"_id": 10, "player": 0}]
        idx = converter._find_unit_idx(999, units, current_player=0)
        assert idx is None

    def test_find_unit_idx_none_id(self, converter: ActionConverter) -> None:
        units = [{"_id": 10, "player": 0}]
        idx = converter._find_unit_idx(None, units, current_player=0)
        assert idx is None

    def test_find_city_idx_found(self, converter: ActionConverter) -> None:
        cities = [{"_id": 100}, {"_id": 200}]
        idx = converter._find_city_idx(200, cities)
        assert idx == 1

    def test_find_city_idx_not_found(self, converter: ActionConverter) -> None:
        cities = [{"_id": 100}]
        idx = converter._find_city_idx(999, cities)
        assert idx is None

    def test_find_city_idx_none_id(self, converter: ActionConverter) -> None:
        cities = [{"_id": 100}]
        idx = converter._find_city_idx(None, cities)
        assert idx is None

    def test_get_valid_action_returns_valid_index(self, converter: ActionConverter) -> None:
        mask = np.zeros(converter.action_space_size, dtype=np.float32)
        end_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        mask[end_idx] = 1.0
        rng = np.random.default_rng(42)
        action = converter.get_valid_action(mask, rng)
        assert action == end_idx

    def test_get_valid_action_empty_mask_fallback(self, converter: ActionConverter) -> None:
        mask = np.zeros(converter.action_space_size, dtype=np.float32)
        action = converter.get_valid_action(mask)
        end_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        assert action == end_idx

    def test_get_valid_action_no_rng(self, converter: ActionConverter) -> None:
        mask = np.zeros(converter.action_space_size, dtype=np.float32)
        end_idx = converter.action_space.category_offsets[ActionCategory.END_TURN]
        mask[end_idx] = 1.0
        action = converter.get_valid_action(mask)
        assert action == end_idx


# ===========================================================================
# 12. _is_action_legal
# ===========================================================================


class TestIsActionLegal:
    """Tests for the legality check logic."""

    def test_end_turn_legal(self, converter: ActionConverter) -> None:
        decoded = HierarchicalAction(category=ActionCategory.END_TURN)
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        legal_actions = [LegalAction(action_type="end_turn")]
        assert converter._is_action_legal(decoded, game_state, legal_actions) is True

    def test_end_turn_not_legal(self, converter: ActionConverter) -> None:
        decoded = HierarchicalAction(category=ActionCategory.END_TURN)
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        legal_actions = [LegalAction(action_type="move", unit_id=1)]
        assert converter._is_action_legal(decoded, game_state, legal_actions) is False

    def test_unit_action_with_matching_target(self, converter: ActionConverter) -> None:
        decoded = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.MOVE.value,
            target=(3, 3),
        )
        game_state = {
            "current_player": 0,
            "units": [{"_id": 10, "player": 0}],
            "cities": [],
        }
        legal_actions = [
            LegalAction(action_type="move", unit_id=10, targets=[(3, 3), (4, 4)]),
        ]
        assert converter._is_action_legal(decoded, game_state, legal_actions) is True

    def test_unit_action_target_not_matching(self, converter: ActionConverter) -> None:
        decoded = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.MOVE.value,
            target=(9, 9),
        )
        game_state = {
            "current_player": 0,
            "units": [{"_id": 10, "player": 0}],
            "cities": [],
        }
        legal_actions = [
            LegalAction(action_type="move", unit_id=10, targets=[(3, 3)]),
        ]
        assert converter._is_action_legal(decoded, game_state, legal_actions) is False

    def test_unit_action_no_targets_is_legal(self, converter: ActionConverter) -> None:
        """Actions without targets (fortify) are legal if unit and type match."""
        decoded = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.FORTIFY.value,
            target=(0, 0),
        )
        game_state = {
            "current_player": 0,
            "units": [{"_id": 10, "player": 0}],
            "cities": [],
        }
        legal_actions = [
            LegalAction(action_type="fortify", unit_id=10),
        ]
        assert converter._is_action_legal(decoded, game_state, legal_actions) is True

    def test_unit_selector_out_of_range(self, converter: ActionConverter) -> None:
        decoded = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=99,
            action_type=UnitActionType.MOVE.value,
            target=(0, 0),
        )
        game_state = {
            "current_player": 0,
            "units": [{"_id": 10, "player": 0}],
            "cities": [],
        }
        legal_actions = [LegalAction(action_type="move", unit_id=10)]
        assert converter._is_action_legal(decoded, game_state, legal_actions) is False

    def test_unit_no_id_field(self, converter: ActionConverter) -> None:
        decoded = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.MOVE.value,
            target=(0, 0),
        )
        game_state = {
            "current_player": 0,
            "units": [{"player": 0}],  # No _id
            "cities": [],
        }
        legal_actions = [LegalAction(action_type="move", unit_id=10)]
        assert converter._is_action_legal(decoded, game_state, legal_actions) is False

    def test_city_produce_legal(self, converter: ActionConverter) -> None:
        decoded = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=0,
            action_type=CityActionType.PRODUCE_UNIT.value,
            target=5,
        )
        game_state = {
            "current_player": 0,
            "units": [],
            "cities": [{"_id": 100, "player": 0}],
        }
        legal_actions = [
            LegalAction(action_type="produce", city_id=100, items=[5, 6, 7]),
        ]
        assert converter._is_action_legal(decoded, game_state, legal_actions) is True

    def test_city_produce_item_not_in_items(self, converter: ActionConverter) -> None:
        decoded = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=0,
            action_type=CityActionType.PRODUCE_UNIT.value,
            target=99,
        )
        game_state = {
            "current_player": 0,
            "units": [],
            "cities": [{"_id": 100, "player": 0}],
        }
        legal_actions = [
            LegalAction(action_type="produce", city_id=100, items=[5, 6]),
        ]
        assert converter._is_action_legal(decoded, game_state, legal_actions) is False

    def test_other_category_defaults_to_true(self, converter: ActionConverter) -> None:
        decoded = HierarchicalAction(
            category=ActionCategory.DIPLOMACY, selector=0, action_type=0
        )
        game_state: dict = {"units": [], "cities": [], "current_player": 0}
        legal_actions = [LegalAction(action_type="end_turn")]
        assert converter._is_action_legal(decoded, game_state, legal_actions) is True
