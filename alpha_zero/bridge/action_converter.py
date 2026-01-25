"""Action converter for Civ6 bridge.

Converts neural network action indices to Lua mod commands, and
generates action masks from the legal actions provided by the game.

The converter bridges between:
- ActionSpace (hierarchical action encoding from civ6_actions.py)
- LuaCommand (commands to be executed by the Lua mod)
- Legal actions list from Civ6 state

Usage:
    from alpha_zero.bridge.action_converter import ActionConverter

    converter = ActionConverter(board_size=16)

    # Convert action index to Lua command
    command = converter.convert(action_idx, game_state, legal_actions)

    # Generate action mask from legal actions
    mask = converter.generate_action_mask(legal_actions, game_state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from alpha_zero.bridge.protocol import (
    ActionType,
    LegalAction,
    LuaCommand,
    Position,
)
from alpha_zero.core.civ6_actions import (
    ActionCategory,
    ActionSpace,
    ActionSpaceConfig,
    CityActionType,
    HierarchicalAction,
    UnitActionType,
)


class InvalidActionError(Exception):
    """Raised when an action cannot be converted or is invalid."""

    def __init__(self, message: str, action_idx: int | None = None):
        super().__init__(message)
        self.action_idx = action_idx


@dataclass
class ConversionResult:
    """Result of action conversion.

    Attributes:
        command: The Lua command to execute.
        decoded_action: The decoded hierarchical action.
        is_valid: Whether the action was valid for the current state.
        validation_error: Error message if not valid.
    """

    command: LuaCommand
    decoded_action: HierarchicalAction
    is_valid: bool = True
    validation_error: str | None = None


class ActionConverter:
    """Converts action indices to Lua commands.

    This class handles the bidirectional mapping between:
    - Flat action indices (from neural network policy)
    - Hierarchical actions (for understanding)
    - Lua commands (for game execution)

    Attributes:
        action_space: The action space for encoding/decoding.
        board_size: Board size for coordinate calculations.
    """

    def __init__(
        self,
        board_size: int = 16,
        action_space_config: ActionSpaceConfig | None = None,
    ):
        """Initialize the action converter.

        Args:
            board_size: Board size for coordinate mapping.
            action_space_config: Custom action space config (uses defaults if None).
        """
        self.board_size = board_size

        if action_space_config is None:
            action_space_config = ActionSpaceConfig(
                board_size=board_size,
                max_units=100,
                max_cities=20,
            )

        self.action_space = ActionSpace(action_space_config)

    @property
    def action_space_size(self) -> int:
        """Get total action space size."""
        return self.action_space.total_size

    def convert(
        self,
        action_idx: int,
        game_state: dict[str, Any],
        legal_actions: list[LegalAction | dict] | None = None,
        validate: bool = True,
    ) -> LuaCommand:
        """Convert action index to Lua command.

        Args:
            action_idx: Flat action index from policy network.
            game_state: Current game state (converted format).
            legal_actions: Legal actions from Lua (for validation).
            validate: Whether to validate against legal actions.

        Returns:
            LuaCommand ready for execution.

        Raises:
            InvalidActionError: If action cannot be converted or is invalid.
        """
        # Decode the hierarchical action
        try:
            decoded = self.action_space.decode(action_idx)
        except Exception as e:
            raise InvalidActionError(f"Failed to decode action {action_idx}: {e}", action_idx) from e

        # Convert to Lua command based on category
        if decoded.category == ActionCategory.END_TURN:
            return LuaCommand(action_type=ActionType.END_TURN)

        elif decoded.category == ActionCategory.UNIT_ACTION:
            return self._convert_unit_action(decoded, game_state, legal_actions, validate)

        elif decoded.category == ActionCategory.CITY_ACTION:
            return self._convert_city_action(decoded, game_state, legal_actions, validate)

        elif decoded.category == ActionCategory.RESEARCH_TECH:
            return self._convert_tech_action(decoded, game_state)

        elif decoded.category == ActionCategory.RESEARCH_CIVIC:
            return self._convert_civic_action(decoded, game_state)

        else:
            # For other categories, create a generic command
            return LuaCommand(
                action_type=ActionType.END_TURN,
                metadata={"category": decoded.category.name, "unsupported": True},
            )

    def convert_with_result(
        self,
        action_idx: int,
        game_state: dict[str, Any],
        legal_actions: list[LegalAction | dict] | None = None,
    ) -> ConversionResult:
        """Convert action with detailed result.

        Args:
            action_idx: Flat action index.
            game_state: Current game state.
            legal_actions: Legal actions from Lua.

        Returns:
            ConversionResult with command and validation info.
        """
        try:
            decoded = self.action_space.decode(action_idx)
            command = self.convert(action_idx, game_state, legal_actions, validate=False)

            # Check if valid
            is_valid = True
            error = None
            if legal_actions:
                is_valid = self._is_action_legal(decoded, game_state, legal_actions)
                if not is_valid:
                    error = "Action not in legal_actions list"

            return ConversionResult(
                command=command,
                decoded_action=decoded,
                is_valid=is_valid,
                validation_error=error,
            )

        except InvalidActionError as e:
            # Return invalid result with error
            return ConversionResult(
                command=LuaCommand(action_type=ActionType.END_TURN),
                decoded_action=HierarchicalAction(category=ActionCategory.END_TURN),
                is_valid=False,
                validation_error=str(e),
            )

    def _convert_unit_action(
        self,
        decoded: HierarchicalAction,
        game_state: dict[str, Any],
        legal_actions: list | None,
        validate: bool,
    ) -> LuaCommand:
        """Convert unit action to Lua command.

        Args:
            decoded: Decoded hierarchical action.
            game_state: Current game state.
            legal_actions: Legal actions for validation.
            validate: Whether to validate.

        Returns:
            LuaCommand for unit action.

        Raises:
            InvalidActionError: If unit not found or action invalid.
        """
        units = game_state.get("units", [])
        current_player = game_state.get("current_player", 0)

        # Filter to current player's units to match action encoding
        # (action mask encodes with player-filtered indices)
        player_units = [u for u in units if u.get("player") == current_player]
        unit_idx = decoded.selector

        if unit_idx >= len(player_units):
            raise InvalidActionError(
                f"Unit index {unit_idx} out of range (have {len(player_units)} player units)",
                decoded.to_flat_index(self.action_space),
            )

        unit = player_units[unit_idx]
        unit_id = unit.get("_id", unit_idx)

        # Get target position
        target = decoded.target
        if isinstance(target, tuple):
            target_x, target_y = target
        else:
            target_x, target_y = 0, 0

        # Convert to original coordinates if available
        map_size = game_state.get("_map_size")
        if map_size:
            orig_x, orig_y = self._scale_to_original(target_x, target_y, map_size)
        else:
            orig_x, orig_y = target_x, target_y

        action_type = UnitActionType(decoded.action_type)

        if action_type == UnitActionType.MOVE:
            return LuaCommand(
                action_type=ActionType.UNIT_MOVE,
                unit_id=unit_id,
                target=Position(x=orig_x, y=orig_y),
            )

        elif action_type == UnitActionType.ATTACK:
            return LuaCommand(
                action_type=ActionType.UNIT_ATTACK,
                unit_id=unit_id,
                target=Position(x=orig_x, y=orig_y),
            )

        elif action_type == UnitActionType.RANGED_ATTACK:
            return LuaCommand(
                action_type=ActionType.UNIT_RANGED_ATTACK,
                unit_id=unit_id,
                target=Position(x=orig_x, y=orig_y),
            )

        elif action_type == UnitActionType.FORTIFY:
            return LuaCommand(
                action_type=ActionType.UNIT_FORTIFY,
                unit_id=unit_id,
            )

        elif action_type == UnitActionType.SKIP:
            return LuaCommand(
                action_type=ActionType.UNIT_SKIP,
                unit_id=unit_id,
            )

        elif action_type == UnitActionType.FOUND_CITY:
            return LuaCommand(
                action_type=ActionType.UNIT_FOUND_CITY,
                unit_id=unit_id,
            )

        elif action_type == UnitActionType.BUILD_IMPROVEMENT:
            return LuaCommand(
                action_type=ActionType.UNIT_BUILD,
                unit_id=unit_id,
                item_id=target if isinstance(target, int) else 0,
            )

        else:
            # Default to skip for unsupported action types
            return LuaCommand(
                action_type=ActionType.UNIT_SKIP,
                unit_id=unit_id,
                metadata={"original_action": action_type.name},
            )

    def _convert_city_action(
        self,
        decoded: HierarchicalAction,
        game_state: dict[str, Any],
        legal_actions: list | None,
        validate: bool,
    ) -> LuaCommand:
        """Convert city action to Lua command.

        Args:
            decoded: Decoded hierarchical action.
            game_state: Current game state.
            legal_actions: Legal actions for validation.
            validate: Whether to validate.

        Returns:
            LuaCommand for city action.

        Raises:
            InvalidActionError: If city not found or action invalid.
        """
        cities = game_state.get("cities", [])
        current_player = game_state.get("current_player", 0)

        # Filter to player's cities
        player_cities = [c for c in cities if c.get("player") == current_player]
        city_idx = decoded.selector

        if city_idx >= len(player_cities):
            raise InvalidActionError(
                f"City index {city_idx} out of range (have {len(player_cities)} cities)",
                decoded.to_flat_index(self.action_space),
            )

        city = player_cities[city_idx]
        city_id = city.get("_id", city_idx)

        action_type = CityActionType(decoded.action_type)
        item_id = decoded.target if isinstance(decoded.target, int) else 0

        if action_type == CityActionType.PRODUCE_UNIT:
            return LuaCommand(
                action_type=ActionType.CITY_PRODUCE,
                city_id=city_id,
                item_id=item_id,
                item_type="unit",
            )

        elif action_type == CityActionType.PRODUCE_BUILDING:
            return LuaCommand(
                action_type=ActionType.CITY_PRODUCE,
                city_id=city_id,
                item_id=item_id,
                item_type="building",
            )

        elif action_type == CityActionType.PRODUCE_WONDER:
            return LuaCommand(
                action_type=ActionType.CITY_PRODUCE,
                city_id=city_id,
                item_id=item_id,
                item_type="wonder",
            )

        elif action_type == CityActionType.PRODUCE_DISTRICT:
            return LuaCommand(
                action_type=ActionType.CITY_PRODUCE,
                city_id=city_id,
                item_id=item_id,
                item_type="district",
            )

        elif action_type == CityActionType.BUY_UNIT:
            return LuaCommand(
                action_type=ActionType.CITY_BUY,
                city_id=city_id,
                item_id=item_id,
                item_type="unit",
            )

        elif action_type == CityActionType.BUY_BUILDING:
            return LuaCommand(
                action_type=ActionType.CITY_BUY,
                city_id=city_id,
                item_id=item_id,
                item_type="building",
            )

        else:
            # Default to no-op for unsupported actions
            return LuaCommand(
                action_type=ActionType.END_TURN,
                metadata={"city_action": action_type.name, "unsupported": True},
            )

    def _convert_tech_action(
        self,
        decoded: HierarchicalAction,
        game_state: dict[str, Any],
    ) -> LuaCommand:
        """Convert technology research action.

        Args:
            decoded: Decoded hierarchical action.
            game_state: Current game state.

        Returns:
            LuaCommand for tech research.
        """
        tech_id = decoded.selector
        return LuaCommand(
            action_type=ActionType.RESEARCH_TECH,
            item_id=tech_id,
        )

    def _convert_civic_action(
        self,
        decoded: HierarchicalAction,
        game_state: dict[str, Any],
    ) -> LuaCommand:
        """Convert civic research action.

        Args:
            decoded: Decoded hierarchical action.
            game_state: Current game state.

        Returns:
            LuaCommand for civic research.
        """
        civic_id = decoded.selector
        return LuaCommand(
            action_type=ActionType.RESEARCH_CIVIC,
            item_id=civic_id,
        )

    def _scale_to_original(
        self,
        x: int,
        y: int,
        map_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Scale coordinates from board_size to original map size.

        Args:
            x: Scaled X coordinate.
            y: Scaled Y coordinate.
            map_size: Original (width, height).

        Returns:
            Original (x, y) coordinates.
        """
        width, height = map_size
        scale_x = width / self.board_size
        scale_y = height / self.board_size

        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)

        # Clamp to bounds
        orig_x = max(0, min(width - 1, orig_x))
        orig_y = max(0, min(height - 1, orig_y))

        return (orig_x, orig_y)

    def _is_action_legal(
        self,
        decoded: HierarchicalAction,
        game_state: dict[str, Any],
        legal_actions: list,
    ) -> bool:
        """Check if action is in the legal actions list.

        Args:
            decoded: Decoded hierarchical action.
            game_state: Current game state.
            legal_actions: Legal actions from Lua.

        Returns:
            True if action is legal.
        """
        # Parse legal actions if they're dicts
        parsed_actions = []
        for action in legal_actions:
            if isinstance(action, dict):
                parsed_actions.append(LegalAction.from_dict(action))
            else:
                parsed_actions.append(action)

        # END_TURN is always legal if in list
        if decoded.category == ActionCategory.END_TURN:
            return any(a.action_type == "end_turn" for a in parsed_actions)

        # Check unit actions
        if decoded.category == ActionCategory.UNIT_ACTION:
            units = game_state.get("units", [])
            current_player = game_state.get("current_player", 0)
            # Filter to player units to match action encoding
            player_units = [u for u in units if u.get("player") == current_player]

            if decoded.selector >= len(player_units):
                return False

            unit = player_units[decoded.selector]
            unit_id = unit.get("_id")

            for action in parsed_actions:
                if action.unit_id == unit_id:
                    action_type = UnitActionType(decoded.action_type)
                    if action_type == UnitActionType.MOVE and action.action_type == "move":
                        # Check if target is in valid targets
                        # Scale decoded target back to original coordinates for comparison
                        target = decoded.target
                        if isinstance(target, tuple):
                            map_size = game_state.get("_map_size")
                            if map_size:
                                orig_target = self._scale_to_original(target[0], target[1], map_size)
                            else:
                                orig_target = target
                            if orig_target in action.targets:
                                return True
                    elif action_type == UnitActionType.ATTACK and action.action_type == "attack":
                        target = decoded.target
                        if isinstance(target, tuple):
                            map_size = game_state.get("_map_size")
                            if map_size:
                                orig_target = self._scale_to_original(target[0], target[1], map_size)
                            else:
                                orig_target = target
                            if orig_target in action.targets:
                                return True
            return False

        # Check city actions
        if decoded.category == ActionCategory.CITY_ACTION:
            cities = game_state.get("cities", [])
            current_player = game_state.get("current_player", 0)
            player_cities = [c for c in cities if c.get("player") == current_player]

            if decoded.selector >= len(player_cities):
                return False

            city = player_cities[decoded.selector]
            city_id = city.get("_id")

            for action in parsed_actions:
                if action.city_id == city_id and action.action_type == "produce":
                    item_id = decoded.target
                    if isinstance(item_id, int) and item_id in action.items:
                        return True
            return False

        # Default: assume valid for other categories
        return True

    def generate_action_mask(
        self,
        legal_actions: list[LegalAction | dict],
        game_state: dict[str, Any],
    ) -> np.ndarray:
        """Generate action mask from legal actions.

        Creates a boolean mask where valid actions are 1.0 and invalid
        actions are 0.0.

        Args:
            legal_actions: Legal actions from Lua mod.
            game_state: Current game state.

        Returns:
            Action mask of shape (action_space_size,).
        """
        mask = np.zeros(self.action_space_size, dtype=np.float32)

        # Parse legal actions
        parsed_actions = []
        for action in legal_actions:
            if isinstance(action, dict):
                parsed_actions.append(LegalAction.from_dict(action))
            else:
                parsed_actions.append(action)

        # Always check for END_TURN
        for action in parsed_actions:
            if action.action_type == "end_turn":
                end_turn_idx = self.action_space.category_offsets[ActionCategory.END_TURN]
                mask[end_turn_idx] = 1.0
                break

        # Process unit actions
        units = game_state.get("units", [])
        current_player = game_state.get("current_player", 0)

        for action in parsed_actions:
            if action.action_type in ("move", "attack"):
                # Find unit index
                unit_idx = self._find_unit_idx(action.unit_id, units, current_player)
                if unit_idx is None:
                    continue

                # Add valid targets
                action_type = (
                    UnitActionType.MOVE if action.action_type == "move"
                    else UnitActionType.ATTACK
                )

                for target in action.targets:
                    # Scale target to board coordinates
                    map_size = game_state.get("_map_size", (self.board_size, self.board_size))
                    scaled_target = self._scale_from_original(target[0], target[1], map_size)

                    try:
                        flat_idx = self.action_space.encode(HierarchicalAction(
                            category=ActionCategory.UNIT_ACTION,
                            selector=unit_idx,
                            action_type=action_type.value,
                            target=scaled_target,
                        ))
                        if flat_idx < self.action_space_size:
                            mask[flat_idx] = 1.0
                    except Exception as e:
                        logger.warning(
                            f"Skipping invalid unit action encoding: unit_idx={unit_idx}, "
                            f"action_type={action_type.name}, target={scaled_target}: {e}"
                        )

            elif action.action_type == "produce":
                # Find city index
                cities = game_state.get("cities", [])
                player_cities = [c for c in cities if c.get("player") == current_player]
                city_idx = self._find_city_idx(action.city_id, player_cities)

                if city_idx is not None:
                    for item_id in action.items:
                        try:
                            # Try both unit and building production
                            for action_type in [CityActionType.PRODUCE_UNIT, CityActionType.PRODUCE_BUILDING]:
                                flat_idx = self.action_space.encode(HierarchicalAction(
                                    category=ActionCategory.CITY_ACTION,
                                    selector=city_idx,
                                    action_type=action_type.value,
                                    target=item_id,
                                ))
                                if flat_idx < self.action_space_size:
                                    mask[flat_idx] = 1.0
                        except Exception as e:
                            logger.warning(
                                f"Skipping invalid city action encoding: city_idx={city_idx}, "
                                f"action_type={action_type.name}, item_id={item_id}: {e}"
                            )

        # Ensure at least END_TURN is valid
        if mask.sum() == 0:
            end_turn_idx = self.action_space.category_offsets[ActionCategory.END_TURN]
            mask[end_turn_idx] = 1.0

        return mask

    def _find_unit_idx(
        self,
        unit_id: int | None,
        units: list[dict],
        current_player: int,
    ) -> int | None:
        """Find unit index by ID.

        Args:
            unit_id: Unit ID to find.
            units: List of units.
            current_player: Current player index.

        Returns:
            Unit index or None if not found.
        """
        if unit_id is None:
            return None

        player_units = [u for u in units if u.get("player") == current_player]
        for idx, unit in enumerate(player_units):
            if unit.get("_id") == unit_id:
                return idx
        return None

    def _find_city_idx(
        self,
        city_id: int | None,
        cities: list[dict],
    ) -> int | None:
        """Find city index by ID.

        Args:
            city_id: City ID to find.
            cities: List of cities.

        Returns:
            City index or None if not found.
        """
        if city_id is None:
            return None

        for idx, city in enumerate(cities):
            if city.get("_id") == city_id:
                return idx
        return None

    def _scale_from_original(
        self,
        x: int,
        y: int,
        map_size: tuple[int, int],
    ) -> tuple[int, int]:
        """Scale coordinates from original map to board_size.

        Args:
            x: Original X coordinate.
            y: Original Y coordinate.
            map_size: Original (width, height).

        Returns:
            Scaled (x, y) coordinates.
        """
        width, height = map_size
        scale_x = self.board_size / width
        scale_y = self.board_size / height

        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)

        # Clamp to bounds
        scaled_x = max(0, min(self.board_size - 1, scaled_x))
        scaled_y = max(0, min(self.board_size - 1, scaled_y))

        return (scaled_x, scaled_y)

    def get_valid_action(
        self,
        mask: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Get a random valid action from mask.

        Args:
            mask: Action mask.
            rng: Random number generator.

        Returns:
            Random valid action index.
        """
        if rng is None:
            rng = np.random.default_rng()

        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) == 0:
            # Fallback to END_TURN
            return self.action_space.category_offsets[ActionCategory.END_TURN]

        return int(rng.choice(valid_indices))
