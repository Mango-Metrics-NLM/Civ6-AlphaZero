"""Tests for the action converter module."""

import numpy as np
import pytest

from alpha_zero.bridge.action_converter import (
    ActionConverter,
    ConversionResult,
    InvalidActionError,
)
from alpha_zero.bridge.protocol import ActionType, LegalAction, LuaCommand
from alpha_zero.core.civ6_actions import ActionSpaceConfig


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
            "terrain": {
                (x, y): {"type": "plains"}
                for x in range(16) for y in range(16)
            },
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
                LegalAction(action_type="move", unit_id=1, targets=[(4, 5), (5, 4), (6, 5), (5, 6)]),
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
            LegalAction(action_type="move", unit_id=i, targets=[(0, 0), (1, 1)])
            for i in range(50)
        ]

        state = {
            "units": [
                {"id": i, "_id": i, "position": (0, 0), "_original_pos": (0, 0), "player": 0, "type": "melee"}
                for i in range(50)
            ],
            "cities": [],
            "_legal_actions": legal_actions,
        }

        mask = converter.generate_action_mask(legal_actions, state)

        # Should handle many actions without error
        assert mask.shape == (converter.action_space_size,)
