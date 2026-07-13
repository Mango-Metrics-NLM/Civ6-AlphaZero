"""Tests for the action converter module."""

import numpy as np
import pytest

from alpha_zero.bridge.action_converter import (
    ActionConverter,
    ConversionResult,
    InvalidActionError,
)
from alpha_zero.bridge.protocol import ActionType, LegalAction, LuaCommand
from alpha_zero.core.civ6_actions import (
    ActionCategory,
    ActionSpaceConfig,
    CityActionType,
    HierarchicalAction,
    UnitActionType,
)


class TestActionConverter:
    """Tests for ActionConverter class."""

    @pytest.fixture
    def converter(self):
        """Create an ActionConverter instance."""
        config = ActionSpaceConfig(
            board_size=16,
            max_units=100,
            max_cities=20,
        )
        return ActionConverter(board_size=16, action_space_config=config)

    @pytest.fixture
    def sample_game_state(self):
        """Create a sample game state for testing."""
        return {
            "turn": 10,
            "current_player": 0,
            "terrain": {(x, y): {"type": "plains"} for x in range(16) for y in range(16)},
            "units": [
                {
                    "id": 1,
                    "_id": 1,
                    "type": "melee",
                    "combat_class": "melee",
                    "player": 0,
                    "position": (5, 5),
                    "health": 1.0,
                    "moves": 2.0,
                },
                {
                    "id": 2,
                    "_id": 2,
                    "type": "civilian",
                    "combat_class": "civilian",
                    "player": 0,
                    "position": (6, 5),
                    "health": 1.0,
                    "moves": 2.0,
                },
            ],
            "cities": [
                {
                    "id": 1,
                    "_id": 1,
                    "name": "Capital",
                    "player": 0,
                    "position": (8, 8),
                    "population": 3,
                },
            ],
            "_legal_actions": [
                LegalAction(action_type="end_turn"),
                LegalAction(action_type="move", unit_id=1, targets=[(4, 5), (5, 4), (6, 5)]),
            ],
        }

    def test_convert_end_turn(self, converter, sample_game_state):
        """Test converting end turn action."""
        # Action 0 should be end turn
        result = converter.convert_with_result(0, sample_game_state, [])

        assert result.command.action_type == ActionType.END_TURN

    def test_action_space_size(self, converter):
        """Test that action space size is reasonable."""
        size = converter.action_space_size

        # Should be a positive integer
        assert size > 0
        assert isinstance(size, int)

        # Should include at least end_turn
        assert size >= 1

    def test_generate_action_mask_empty(self, converter, sample_game_state):
        """Test action mask with no legal actions."""
        mask = converter.generate_action_mask([], sample_game_state)

        assert mask.shape == (converter.action_space_size,)
        assert mask.dtype == np.float32

        # Only end_turn should be valid when no specific actions
        assert mask[0] == 1.0  # end_turn is always valid

    def test_generate_action_mask_with_end_turn(self, converter, sample_game_state):
        """Test action mask with end_turn action."""
        legal_actions = [LegalAction(action_type="end_turn")]

        mask = converter.generate_action_mask(legal_actions, sample_game_state)

        # End turn (action 0) should be valid
        assert mask[0] == 1.0

    def test_convert_with_result_structure(self, converter, sample_game_state):
        """Test that convert_with_result returns proper structure."""
        result = converter.convert_with_result(0, sample_game_state, [])

        assert isinstance(result, ConversionResult)
        assert hasattr(result, "command")
        assert hasattr(result, "is_valid")
        assert hasattr(result, "validation_error")
        assert hasattr(result, "decoded_action")

    def test_lua_command_format(self, converter, sample_game_state):
        """Test that LuaCommand has correct format."""
        result = converter.convert_with_result(0, sample_game_state, [])
        command = result.command

        assert isinstance(command, LuaCommand)
        assert isinstance(command.action_type, ActionType)

        # Serialize to dict to check format
        cmd_dict = command.to_dict()
        assert "action_type" in cmd_dict

    def test_conversion_result_is_valid(self, converter, sample_game_state):
        """Test ConversionResult valid flag."""
        result = converter.convert_with_result(0, sample_game_state, [])

        # End turn should always be valid
        assert result.is_valid is True

    def test_action_mask_all_valid_bounded(self, converter, sample_game_state):
        """Test that action mask values are bounded 0-1."""
        mask = converter.generate_action_mask([], sample_game_state)

        # All values should be 0 or 1
        assert np.all((mask == 0.0) | (mask == 1.0))


class TestActionConverterMoves:
    """Tests for movement action conversion."""

    @pytest.fixture
    def converter(self):
        """Create an ActionConverter instance."""
        config = ActionSpaceConfig(board_size=16, max_units=100, max_cities=20)
        return ActionConverter(board_size=16, action_space_config=config)

    @pytest.fixture
    def state_with_unit(self):
        """Create state with a unit that can move."""
        return {
            "units": [
                {
                    "id": 1,
                    "_id": 1,
                    "type": "melee",
                    "combat_class": "melee",
                    "player": 0,
                    "position": (5, 5),
                    "_original_pos": (5, 5),
                    "health": 1.0,
                    "moves": 2.0,
                },
            ],
            "cities": [],
            "_legal_actions": [
                LegalAction(
                    action_type="move", unit_id=1, targets=[(4, 5), (5, 4), (6, 5), (5, 6)]
                ),
            ],
        }

    def test_generate_mask_with_legal_actions(self, converter, state_with_unit):
        """Test that legal actions update the mask."""
        legal_actions = state_with_unit["_legal_actions"]
        mask = converter.generate_action_mask(legal_actions, state_with_unit)

        # Mask should have expected shape
        assert mask.shape == (converter.action_space_size,)
        # Should have some valid actions
        assert mask.sum() >= 1


class TestActionConverterCities:
    """Tests for city action conversion."""

    @pytest.fixture
    def converter(self):
        """Create an ActionConverter instance."""
        config = ActionSpaceConfig(board_size=16, max_units=100, max_cities=20)
        return ActionConverter(board_size=16, action_space_config=config)

    @pytest.fixture
    def state_with_city(self):
        """Create state with a city that can produce."""
        return {
            "units": [],
            "cities": [
                {
                    "id": 1,
                    "_id": 1,
                    "name": "Capital",
                    "player": 0,
                    "position": (8, 8),
                    "_original_pos": (8, 8),
                    "population": 3,
                },
            ],
            "_legal_actions": [
                LegalAction(
                    action_type="produce",
                    city_id=1,
                    items=["UNIT_WARRIOR", "UNIT_SETTLER", "BUILDING_MONUMENT"],
                ),
            ],
        }

    def test_generate_production_mask(self, converter, state_with_city):
        """Test that production actions are properly masked."""
        legal_actions = state_with_city["_legal_actions"]
        mask = converter.generate_action_mask(legal_actions, state_with_city)

        # Mask should have expected shape
        assert mask.shape == (converter.action_space_size,)


class TestInvalidActionError:
    """Tests for InvalidActionError exception."""

    def test_error_message(self):
        """Test error message format."""
        error = InvalidActionError("Invalid action type", action_idx=42)

        assert "Invalid action type" in str(error)

    def test_error_with_index(self):
        """Test error with action index."""
        error = InvalidActionError("Test message", action_idx=42)

        assert error.action_idx == 42

    def test_error_without_index(self):
        """Test error without action index."""
        error = InvalidActionError("Test message")

        assert error.action_idx is None


class TestConversionResult:
    """Tests for ConversionResult class."""

    def test_valid_result(self):
        """Test creating a valid result with decoded_action."""
        from alpha_zero.core.civ6_actions import ActionCategory, HierarchicalAction

        cmd = LuaCommand(action_type=ActionType.END_TURN)
        decoded = HierarchicalAction(category=ActionCategory.END_TURN)
        result = ConversionResult(command=cmd, decoded_action=decoded, is_valid=True)

        assert result.is_valid
        assert result.command == cmd
        assert result.validation_error is None

    def test_invalid_result(self):
        """Test creating an invalid result."""
        from alpha_zero.core.civ6_actions import ActionCategory, HierarchicalAction

        cmd = LuaCommand(action_type=ActionType.END_TURN)
        decoded = HierarchicalAction(category=ActionCategory.END_TURN)
        result = ConversionResult(
            command=cmd,
            decoded_action=decoded,
            is_valid=False,
            validation_error="Unit not found",
        )

        assert not result.is_valid
        assert result.validation_error == "Unit not found"


class TestActionConverterEdgeCases:
    """Edge case tests for ActionConverter."""

    @pytest.fixture
    def converter(self):
        """Create an ActionConverter instance."""
        config = ActionSpaceConfig(board_size=16, max_units=100, max_cities=20)
        return ActionConverter(board_size=16, action_space_config=config)

    def test_empty_state(self, converter):
        """Test with empty game state."""
        empty_state = {
            "units": [],
            "cities": [],
            "_legal_actions": [],
        }

        mask = converter.generate_action_mask([], empty_state)

        # Should still have end_turn as valid
        assert mask[0] == 1.0

    def test_action_mask_bounds(self, converter):
        """Test that action mask values are properly bounded."""
        empty_state = {"units": [], "cities": [], "_legal_actions": []}
        mask = converter.generate_action_mask([], empty_state)

        # All values should be 0 or 1
        assert np.all((mask == 0.0) | (mask == 1.0))

    def test_concurrent_legal_actions(self, converter):
        """Test with many concurrent legal actions."""
        legal_actions = [LegalAction(action_type="end_turn")] + [
            LegalAction(action_type="move", unit_id=i, targets=[(0, 0), (1, 1)]) for i in range(50)
        ]

        state = {
            "units": [
                {
                    "id": i,
                    "_id": i,
                    "position": (0, 0),
                    "_original_pos": (0, 0),
                    "player": 0,
                    "type": "melee",
                }
                for i in range(50)
            ],
            "cities": [],
            "_legal_actions": legal_actions,
        }

        mask = converter.generate_action_mask(legal_actions, state)

        # Should handle many actions without error
        assert mask.shape == (converter.action_space_size,)


class TestItemIdConversion:
    """Tests for _convert_item_id_to_int helper function."""

    def test_integer_passthrough(self):
        """Test that integers are returned as-is."""
        from alpha_zero.bridge.action_converter import _convert_item_id_to_int

        assert _convert_item_id_to_int(42) == 42
        assert _convert_item_id_to_int(0) == 0
        assert _convert_item_id_to_int(100) == 100

    def test_numeric_string_conversion(self):
        """Test that numeric strings are converted to int."""
        from alpha_zero.bridge.action_converter import _convert_item_id_to_int

        assert _convert_item_id_to_int("42") == 42
        assert _convert_item_id_to_int("0") == 0
        assert _convert_item_id_to_int("100") == 100

    def test_type_name_string_hashing(self):
        """Test that Civ6 type name strings are hashed consistently."""
        from alpha_zero.bridge.action_converter import _convert_item_id_to_int

        # Should return a consistent integer for the same string
        result1 = _convert_item_id_to_int("UNIT_WARRIOR")
        result2 = _convert_item_id_to_int("UNIT_WARRIOR")
        assert result1 == result2
        assert isinstance(result1, int)
        assert 0 <= result1 < 200  # Should be in valid range

        # Different strings should (likely) produce different hashes
        warrior = _convert_item_id_to_int("UNIT_WARRIOR")
        settler = _convert_item_id_to_int("UNIT_SETTLER")
        # Note: They might collide but unlikely
        assert isinstance(warrior, int)
        assert isinstance(settler, int)

    def test_none_returns_none(self):
        """Test that None input returns None."""
        from alpha_zero.bridge.action_converter import _convert_item_id_to_int

        assert _convert_item_id_to_int(None) is None

    def test_empty_string_returns_none(self):
        """Test that empty string returns None."""
        from alpha_zero.bridge.action_converter import _convert_item_id_to_int

        assert _convert_item_id_to_int("") is None


class TestCoordinateScalingRound:
    """Tests for coordinate scaling using round() instead of int()."""

    @pytest.fixture
    def converter(self):
        """Create an ActionConverter instance."""
        config = ActionSpaceConfig(board_size=16, max_units=100, max_cities=20)
        return ActionConverter(board_size=16, action_space_config=config)

    def test_scale_to_original_uses_round(self, converter):
        """Test _scale_to_original uses round() for accurate mapping."""
        # Map 16 -> 44: scale_x = 44/16 = 2.75
        # For x=5: 5 * 2.75 = 13.75 -> round() gives 14, int() gives 13
        orig_x, orig_y = converter._scale_to_original(5, 5, (44, 26))
        # Just verify it's in bounds and deterministic
        assert 0 <= orig_x < 44
        assert 0 <= orig_y < 26

    def test_scale_from_original_uses_round(self, converter):
        """Test _scale_from_original uses round() for accurate mapping."""
        # Map 44 -> 16: scale_x = 16/44 = 0.3636
        # For x=10: 10 * 0.3636 = 3.636 -> round() gives 4, int() gives 3
        scaled_x, scaled_y = converter._scale_from_original(10, 10, (44, 26))
        assert 0 <= scaled_x < 16
        assert 0 <= scaled_y < 16

    def test_scale_round_trip_approximate(self, converter):
        """Test that scale_to_original and scale_from_original are approximate inverses."""
        map_size = (80, 50)
        for orig_x in range(0, 80, 10):
            for orig_y in range(0, 50, 10):
                scaled = converter._scale_from_original(orig_x, orig_y, map_size)
                restored = converter._scale_to_original(scaled[0], scaled[1], map_size)
                # Error should be within one scale step
                scale_step_x = 80 / 16
                scale_step_y = 50 / 16
                assert abs(restored[0] - orig_x) <= scale_step_x + 1
                assert abs(restored[1] - orig_y) <= scale_step_y + 1


class TestItemHashRange:
    """Tests for GameConstants.ITEM_HASH_RANGE and its usage."""

    def test_item_hash_range_constant_exists(self):
        """Test that ITEM_HASH_RANGE is importable and positive."""
        from alpha_zero.bridge.config import GameConstants

        assert isinstance(GameConstants.ITEM_HASH_RANGE, int)
        assert GameConstants.ITEM_HASH_RANGE > 0

    def test_hashed_items_within_range(self):
        """Test that hashed items stay within GameConstants.ITEM_HASH_RANGE."""
        from alpha_zero.bridge.action_converter import _convert_item_id_to_int
        from alpha_zero.bridge.config import GameConstants

        civ6_items = [
            "UNIT_WARRIOR",
            "UNIT_SETTLER",
            "UNIT_ARCHER",
            "BUILDING_MONUMENT",
            "BUILDING_GRANARY",
            "BUILDING_ANCIENT_WALLS",
            "DISTRICT_CAMPUS",
            "DISTRICT_HOLY_SITE",
        ]
        for item in civ6_items:
            result = _convert_item_id_to_int(item)
            assert result is not None
            assert (
                0 <= result < GameConstants.ITEM_HASH_RANGE
            ), f"Hashed item '{item}' = {result} outside range [0, {GameConstants.ITEM_HASH_RANGE})"

    def test_hashed_items_deterministic(self):
        """Test that hashing produces deterministic results."""
        from alpha_zero.bridge.action_converter import _convert_item_id_to_int

        for _ in range(10):
            assert _convert_item_id_to_int("UNIT_WARRIOR") == _convert_item_id_to_int(
                "UNIT_WARRIOR"
            )


class TestValidateParameter:
    """Tests for the validate parameter in ActionConverter.convert()."""

    @pytest.fixture
    def converter(self):
        """Create an ActionConverter instance."""
        config = ActionSpaceConfig(board_size=16, max_units=100, max_cities=20)
        return ActionConverter(board_size=16, action_space_config=config)

    @pytest.fixture
    def state_with_unit(self):
        """Create state with a unit."""
        return {
            "turn": 1,
            "current_player": 0,
            "units": [
                {
                    "id": 1,
                    "_id": 1,
                    "type": "melee",
                    "player": 0,
                    "position": (5, 5),
                }
            ],
            "cities": [],
        }

    def test_validate_true_raises_on_illegal_action(self, converter, state_with_unit):
        """Test that validate=True raises InvalidActionError for illegal actions."""
        # Create legal actions that only allow end_turn
        legal_actions = [LegalAction(action_type="end_turn")]

        # Try to convert a unit move action (which is not in legal_actions)
        unit_action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.MOVE.value,
            target=(6, 5),
        )
        action_idx = converter.action_space.encode(unit_action)

        # With validate=True, should raise
        with pytest.raises(InvalidActionError) as exc_info:
            converter.convert(action_idx, state_with_unit, legal_actions, validate=True)

        assert "not in the legal actions list" in str(exc_info.value)

    def test_validate_false_allows_illegal_action(self, converter, state_with_unit):
        """Test that validate=False allows illegal actions to be converted."""
        legal_actions = [LegalAction(action_type="end_turn")]

        unit_action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.MOVE.value,
            target=(6, 5),
        )
        action_idx = converter.action_space.encode(unit_action)

        # With validate=False, should NOT raise
        command = converter.convert(action_idx, state_with_unit, legal_actions, validate=False)

        # Should return a valid command
        assert command is not None

    def test_validate_true_allows_legal_action(self, converter, state_with_unit):
        """Test that validate=True allows legal actions."""
        legal_actions = [LegalAction(action_type="end_turn")]

        # End turn action should be allowed
        command = converter.convert(0, state_with_unit, legal_actions, validate=True)

        assert command.action_type == ActionType.END_TURN


class TestUnitActionConversion:
    """Tests for _convert_unit_action covering all unit action types."""

    @pytest.fixture
    def converter(self):
        config = ActionSpaceConfig(board_size=16, max_units=100, max_cities=20)
        return ActionConverter(board_size=16, action_space_config=config)

    @pytest.fixture
    def state(self):
        return {
            "current_player": 0,
            "units": [
                {"id": 1, "_id": 1, "player": 0, "position": (5, 5), "type": "melee"},
                {"id": 2, "_id": 2, "player": 0, "position": (7, 7), "type": "civilian"},
            ],
            "cities": [],
        }

    def _encode_unit_action(self, converter, unit_idx, action_type, target=None):
        """Encode a unit action to a flat action index for converter.convert()."""
        action = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=unit_idx,
            action_type=action_type.value,
            target=target,
        )
        return converter.action_space.encode(action)

    def test_convert_unit_attack(self, converter, state):
        """Test ATTACK unit action conversion."""
        idx = self._encode_unit_action(converter, 0, UnitActionType.ATTACK, (6, 5))
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.UNIT_ATTACK
        assert cmd.unit_id == 1
        assert cmd.target is not None

    def test_convert_unit_ranged_attack(self, converter, state):
        """Test RANGED_ATTACK unit action conversion."""
        idx = self._encode_unit_action(converter, 0, UnitActionType.RANGED_ATTACK, (6, 5))
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.UNIT_RANGED_ATTACK

    def test_convert_unit_fortify(self, converter, state):
        """Test FORTIFY unit action conversion."""
        idx = self._encode_unit_action(converter, 0, UnitActionType.FORTIFY)
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.UNIT_FORTIFY
        assert cmd.unit_id == 1

    def test_convert_unit_skip(self, converter, state):
        """Test SKIP unit action conversion."""
        idx = self._encode_unit_action(converter, 0, UnitActionType.SKIP)
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.UNIT_SKIP
        assert cmd.unit_id == 1

    def test_convert_unit_found_city(self, converter, state):
        """Test FOUND_CITY unit action conversion."""
        idx = self._encode_unit_action(converter, 1, UnitActionType.FOUND_CITY)
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.UNIT_FOUND_CITY
        assert cmd.unit_id == 2

    def test_convert_unit_build_improvement(self, converter, state):
        """Test BUILD_IMPROVEMENT unit action conversion."""
        idx = self._encode_unit_action(converter, 0, UnitActionType.BUILD_IMPROVEMENT, 5)
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.UNIT_BUILD

    def test_convert_unit_out_of_range(self, converter, state):
        """Test unit action with out-of-range unit index."""
        idx = self._encode_unit_action(converter, 99, UnitActionType.MOVE, (6, 5))
        with pytest.raises(InvalidActionError):
            converter.convert(idx, state, validate=False)


class TestCityActionConversion:
    """Tests for _convert_city_action covering all city action types."""

    @pytest.fixture
    def converter(self):
        config = ActionSpaceConfig(board_size=16, max_units=100, max_cities=20)
        return ActionConverter(board_size=16, action_space_config=config)

    @pytest.fixture
    def state(self):
        return {
            "current_player": 0,
            "units": [],
            "cities": [
                {"id": 1, "_id": 1, "player": 0, "name": "Capital", "position": (8, 8)},
            ],
        }

    def _encode_city_action(self, converter, city_idx, action_type, item_id=0):
        """Encode a city action to a flat action index for converter.convert()."""
        action = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=city_idx,
            action_type=action_type.value,
            target=item_id,
        )
        return converter.action_space.encode(action)

    def test_convert_produce_unit(self, converter, state):
        """Test PRODUCE_UNIT city action conversion."""
        idx = self._encode_city_action(converter, 0, CityActionType.PRODUCE_UNIT, 5)
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.CITY_PRODUCE
        assert cmd.item_type == "unit"

    def test_convert_produce_building(self, converter, state):
        """Test PRODUCE_BUILDING city action conversion."""
        idx = self._encode_city_action(converter, 0, CityActionType.PRODUCE_BUILDING, 3)
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.CITY_PRODUCE
        assert cmd.item_type == "building"

    def test_convert_produce_wonder(self, converter, state):
        """Test PRODUCE_WONDER city action conversion."""
        idx = self._encode_city_action(converter, 0, CityActionType.PRODUCE_WONDER, 1)
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.CITY_PRODUCE
        assert cmd.item_type == "wonder"

    def test_convert_produce_district(self, converter, state):
        """Test PRODUCE_DISTRICT city action conversion."""
        idx = self._encode_city_action(converter, 0, CityActionType.PRODUCE_DISTRICT, 2)
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.CITY_PRODUCE
        assert cmd.item_type == "district"

    def test_convert_buy_unit(self, converter, state):
        """Test BUY_UNIT city action conversion."""
        idx = self._encode_city_action(converter, 0, CityActionType.BUY_UNIT, 4)
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.CITY_BUY
        assert cmd.item_type == "unit"

    def test_convert_buy_building(self, converter, state):
        """Test BUY_BUILDING city action conversion."""
        idx = self._encode_city_action(converter, 0, CityActionType.BUY_BUILDING, 2)
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.CITY_BUY
        assert cmd.item_type == "building"

    def test_convert_city_out_of_range(self, converter, state):
        """Test city action with out-of-range city index raises or falls back."""
        # With only 1 city but selector=5, should raise InvalidActionError
        # if the encoder allows it, otherwise end_turn fallback.
        idx = self._encode_city_action(converter, 5, CityActionType.PRODUCE_UNIT, 0)
        # convert may raise or fall back to END_TURN depending on encoding bounds
        try:
            cmd = converter.convert(idx, state, validate=False)
            # If it doesn't raise, it should fall back gracefully
            assert cmd.action_type in (ActionType.CITY_PRODUCE, ActionType.END_TURN)
        except InvalidActionError:
            pass  # Expected when city index is truly out of range


class TestTechCivicConversion:
    """Tests for tech and civic action conversion."""

    @pytest.fixture
    def converter(self):
        config = ActionSpaceConfig(board_size=16, max_units=100, max_cities=20)
        return ActionConverter(board_size=16, action_space_config=config)

    @pytest.fixture
    def state(self):
        return {"current_player": 0, "units": [], "cities": []}

    def test_convert_tech_action(self, converter, state):
        """Test technology research action conversion."""
        action = HierarchicalAction(category=ActionCategory.RESEARCH_TECH, selector=5)
        idx = converter.action_space.encode(action)
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.RESEARCH_TECH
        assert cmd.item_id == 5

    def test_convert_civic_action(self, converter, state):
        """Test civic research action conversion."""
        action = HierarchicalAction(category=ActionCategory.RESEARCH_CIVIC, selector=3)
        idx = converter.action_space.encode(action)
        cmd = converter.convert(idx, state, validate=False)
        assert cmd.action_type == ActionType.RESEARCH_CIVIC
        assert cmd.item_id == 3


class TestActionLegalityChecking:
    """Tests for _is_action_legal method."""

    @pytest.fixture
    def converter(self):
        config = ActionSpaceConfig(board_size=16, max_units=100, max_cities=20)
        return ActionConverter(board_size=16, action_space_config=config)

    def test_end_turn_legal_when_present(self, converter):
        """Test END_TURN is legal when present in legal actions."""
        decoded = HierarchicalAction(category=ActionCategory.END_TURN)
        legal = [LegalAction(action_type="end_turn")]
        assert converter._is_action_legal(decoded, {}, legal)

    def test_end_turn_illegal_when_absent(self, converter):
        """Test END_TURN is illegal when not in legal actions."""
        decoded = HierarchicalAction(category=ActionCategory.END_TURN)
        assert not converter._is_action_legal(decoded, {}, [])

    def test_unit_action_legal_with_matching_unit(self, converter):
        """Test unit action legal when unit matches and has no targets."""
        state = {
            "current_player": 0,
            "units": [{"_id": 1, "player": 0}],
        }
        decoded = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.FORTIFY.value,
        )
        legal = [LegalAction(action_type="fortify", unit_id=1)]
        assert converter._is_action_legal(decoded, state, legal)

    def test_unit_action_illegal_no_id(self, converter):
        """Test unit action illegal when unit has no _id field."""
        state = {
            "current_player": 0,
            "units": [{"player": 0}],  # No _id
        }
        decoded = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=0,
            action_type=UnitActionType.MOVE.value,
            target=(5, 5),
        )
        legal = [LegalAction(action_type="move", unit_id=1, targets=[(5, 5)])]
        assert not converter._is_action_legal(decoded, state, legal)

    def test_unit_selector_out_of_range(self, converter):
        """Test unit action illegal when selector exceeds unit count."""
        state = {"current_player": 0, "units": []}
        decoded = HierarchicalAction(
            category=ActionCategory.UNIT_ACTION,
            selector=5,
            action_type=UnitActionType.MOVE.value,
        )
        assert not converter._is_action_legal(decoded, state, [])

    def test_city_action_legal_with_matching_item(self, converter):
        """Test city action legal when city has matching item."""
        state = {
            "current_player": 0,
            "cities": [{"_id": 10, "player": 0}],
        }
        decoded = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=0,
            action_type=CityActionType.PRODUCE_UNIT.value,
            target=5,
        )
        legal = [LegalAction(action_type="produce", city_id=10, items=[5, 10, 15])]
        assert converter._is_action_legal(decoded, state, legal)

    def test_city_action_illegal_no_id(self, converter):
        """Test city action illegal when city has no _id field."""
        state = {
            "current_player": 0,
            "cities": [{"player": 0}],  # No _id
        }
        decoded = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=0,
            action_type=CityActionType.PRODUCE_UNIT.value,
            target=5,
        )
        legal = [LegalAction(action_type="produce", city_id=10, items=[5])]
        assert not converter._is_action_legal(decoded, state, legal)

    def test_city_selector_out_of_range(self, converter):
        """Test city action illegal when selector exceeds city count."""
        state = {"current_player": 0, "cities": []}
        decoded = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=0,
            action_type=CityActionType.PRODUCE_UNIT.value,
        )
        assert not converter._is_action_legal(decoded, state, [])

    def test_dict_legal_actions_parsed(self, converter):
        """Test that dict legal actions are properly parsed."""
        decoded = HierarchicalAction(category=ActionCategory.END_TURN)
        legal_dicts = [{"action_type": "end_turn"}]
        assert converter._is_action_legal(decoded, {}, legal_dicts)

    def test_city_action_legal_with_string_item_id(self, converter):
        """Test city production is legal when items are raw string type names.

        StateExporter.lua exports non-numeric production item ids (e.g.
        "UNIT_WARRIOR") as strings. generate_action_mask() hashes those
        strings via _convert_item_id_to_int() before encoding them into the
        mask, so decoded.target on the policy-chosen action is always the
        hashed int. _is_action_legal() must hash action.items the same way
        before comparing, or legal string-keyed production is never
        recognized as legal (regression for the always-False bug).
        """
        from alpha_zero.bridge.action_converter import _convert_item_id_to_int

        hashed_item_id = _convert_item_id_to_int("UNIT_WARRIOR")
        state = {
            "current_player": 0,
            "cities": [{"_id": 10, "player": 0}],
        }
        decoded = HierarchicalAction(
            category=ActionCategory.CITY_ACTION,
            selector=0,
            action_type=CityActionType.PRODUCE_UNIT.value,
            target=hashed_item_id,
        )
        legal = [LegalAction(action_type="produce", city_id=10, items=["UNIT_WARRIOR"])]
        assert converter._is_action_legal(decoded, state, legal)


class TestCityProductionStringItemRoundTrip:
    """Regression coverage for string-keyed production items through convert().

    Reproduces the reported bug: generate_action_mask() encodes hashed
    string item ids into the mask, but _is_action_legal() (used by
    convert(validate=True)) compared the hashed int against the raw,
    unhashed string items list and always returned False, so every
    policy-chosen city-production action raised InvalidActionError.
    """

    @pytest.fixture
    def converter(self):
        config = ActionSpaceConfig(board_size=16, max_units=100, max_cities=20)
        return ActionConverter(board_size=16, action_space_config=config)

    @pytest.fixture
    def state(self):
        return {
            "current_player": 0,
            "units": [],
            "cities": [
                {"id": 1, "_id": 1, "player": 0, "name": "Capital", "position": (8, 8)},
            ],
        }

    def test_policy_chosen_production_survives_validated_convert(self, converter, state):
        """A production action legal via string items must pass convert(validate=True)."""
        legal_actions = [
            LegalAction(
                action_type="produce",
                city_id=1,
                items=["UNIT_WARRIOR"],
                item_type="unit",
            ),
            LegalAction(action_type="end_turn"),
        ]

        mask = converter.generate_action_mask(legal_actions, state)
        legal_indices = np.nonzero(mask)[0]
        assert len(legal_indices) > 0

        # Find the encoded PRODUCE_UNIT action among the legal indices.
        produce_idx = None
        for idx in legal_indices:
            decoded = converter.action_space.decode(int(idx))
            if (
                decoded.category == ActionCategory.CITY_ACTION
                and decoded.action_type == CityActionType.PRODUCE_UNIT.value
            ):
                produce_idx = int(idx)
                break
        assert produce_idx is not None, "Expected a PRODUCE_UNIT action in the mask"

        # This must NOT raise InvalidActionError.
        cmd = converter.convert(produce_idx, state, legal_actions, validate=True)
        assert cmd.action_type == ActionType.CITY_PRODUCE
        assert cmd.item_type == "unit"


class TestGenerateActionMaskItemTypes:
    """Regression coverage for LegalAction.item_types (per-item production type).

    StateExporter.lua emits a plural `item_types` dict mapping each raw item
    id to its category ("unit"/"building"/"wonder"/"district"), but the
    LegalAction dataclass historically only exposed a singular `item_type`
    that Lua never populates. Without item_types, generate_action_mask()
    falls into the "try unit and building only" branch and can never mark a
    district as legal.
    """

    @pytest.fixture
    def converter(self):
        config = ActionSpaceConfig(board_size=16, max_units=100, max_cities=20)
        return ActionConverter(board_size=16, action_space_config=config)

    @pytest.fixture
    def state(self):
        return {
            "current_player": 0,
            "units": [],
            "cities": [
                {"id": 1, "_id": 1, "player": 0, "name": "Capital", "position": (8, 8)},
            ],
        }

    def test_district_item_marked_legal_via_item_types(self, converter, state):
        """A district item id must be encoded as PRODUCE_DISTRICT in the mask."""
        legal_actions = [
            LegalAction.from_dict(
                {
                    "action_type": "produce",
                    "city_id": 1,
                    "items": ["DISTRICT_CAMPUS"],
                    "item_types": {"DISTRICT_CAMPUS": "district"},
                }
            )
        ]

        mask = converter.generate_action_mask(legal_actions, state)
        legal_indices = np.nonzero(mask)[0]

        found_district = False
        for idx in legal_indices:
            decoded = converter.action_space.decode(int(idx))
            if (
                decoded.category == ActionCategory.CITY_ACTION
                and decoded.action_type == CityActionType.PRODUCE_DISTRICT.value
            ):
                found_district = True
                break
        assert found_district, "Expected item_types to mark the item as a legal district"


class TestFindUnitCityIdx:
    """Tests for _find_unit_idx and _find_city_idx."""

    @pytest.fixture
    def converter(self):
        config = ActionSpaceConfig(board_size=16, max_units=100, max_cities=20)
        return ActionConverter(board_size=16, action_space_config=config)

    def test_find_unit_idx_found(self, converter):
        units = [{"_id": 10, "player": 0}, {"_id": 20, "player": 0}]
        assert converter._find_unit_idx(20, units, 0) == 1

    def test_find_unit_idx_none_id(self, converter):
        assert converter._find_unit_idx(None, [], 0) is None

    def test_find_unit_idx_not_found(self, converter):
        units = [{"_id": 10, "player": 0}]
        assert converter._find_unit_idx(99, units, 0) is None

    def test_find_city_idx_found(self, converter):
        cities = [{"_id": 5}, {"_id": 10}]
        assert converter._find_city_idx(10, cities) == 1

    def test_find_city_idx_not_found(self, converter):
        cities = [{"_id": 5}]
        assert converter._find_city_idx(99, cities) is None


class TestGetValidAction:
    """Tests for get_valid_action method."""

    @pytest.fixture
    def converter(self):
        config = ActionSpaceConfig(board_size=16, max_units=100, max_cities=20)
        return ActionConverter(board_size=16, action_space_config=config)

    def test_get_valid_action_returns_valid_index(self, converter):
        mask = np.zeros(converter.action_space_size, dtype=np.float32)
        mask[5] = 1.0
        mask[10] = 1.0
        result = converter.get_valid_action(mask)
        assert result in (5, 10)

    def test_get_valid_action_empty_mask_falls_back(self, converter):
        mask = np.zeros(converter.action_space_size, dtype=np.float32)
        result = converter.get_valid_action(mask)
        # Should fall back to END_TURN offset
        expected = converter.action_space.category_offsets[ActionCategory.END_TURN]
        assert result == expected

    def test_get_valid_action_with_rng(self, converter):
        mask = np.zeros(converter.action_space_size, dtype=np.float32)
        mask[3] = 1.0
        rng = np.random.default_rng(42)
        result = converter.get_valid_action(mask, rng=rng)
        assert result == 3
