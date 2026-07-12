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

__all__ = ["ActionConverter", "ConversionResult", "InvalidActionError", "NoLegalActionsError"]

import numpy as np

from alpha_zero.bridge.config import GameConstants
from alpha_zero.bridge.protocol import (
    ActionType,
    LegalAction,
    LuaCommand,
    Position,
)
from alpha_zero.core.civ6_actions import (
    BRIDGE_IMPLEMENTED_DIPLOMACY,
    ActionCategory,
    ActionSpace,
    ActionSpaceConfig,
    CityActionType,
    DiplomacyActionType,
    HierarchicalAction,
    UnitActionType,
)

logger = logging.getLogger(__name__)


# Legal action type strings from Lua mod (centralized for backwards compatibility)
class LegalActionTypes:
    """Constants for legal action type strings from Lua mod.

    These map to the action types sent by the Civ6 Lua mod in legal_actions.
    Keeping them centralized ensures consistency and easy updates if protocol changes.

    Unit actions include all 17 UnitActionType values for completeness.
    """

    # Turn management
    END_TURN = "end_turn"

    # Basic unit actions
    MOVE = "move"
    ATTACK = "attack"
    RANGED_ATTACK = "ranged_attack"
    FORTIFY = "fortify"
    SKIP = "skip"
    SLEEP = "sleep"

    # Special unit actions
    FOUND_CITY = "found_city"
    BUILD = "build"
    PILLAGE = "pillage"
    REPAIR = "repair"

    # Naval/transport actions
    EMBARK = "embark"
    DISEMBARK = "disembark"

    # Unit management
    PROMOTE = "promote"
    HEAL = "heal"
    DELETE = "delete"

    # Air unit actions
    AIRLIFT = "airlift"
    PARADROP = "paradrop"

    # City actions
    PRODUCE = "produce"
    BUY = "buy"

    # Research actions
    RESEARCH_TECH = "research_tech"
    RESEARCH_CIVIC = "research_civic"


# Mapping from legal action strings to UnitActionType enum
# Maps all 17 UnitActionType values for comprehensive coverage
LEGAL_ACTION_TO_UNIT_TYPE: dict[str, UnitActionType] = {
    # Basic movement and combat
    LegalActionTypes.MOVE: UnitActionType.MOVE,
    LegalActionTypes.ATTACK: UnitActionType.ATTACK,
    LegalActionTypes.RANGED_ATTACK: UnitActionType.RANGED_ATTACK,
    # Defensive/wait actions
    LegalActionTypes.FORTIFY: UnitActionType.FORTIFY,
    LegalActionTypes.SKIP: UnitActionType.SKIP,
    LegalActionTypes.SLEEP: UnitActionType.SLEEP,
    # Special actions
    LegalActionTypes.FOUND_CITY: UnitActionType.FOUND_CITY,
    LegalActionTypes.BUILD: UnitActionType.BUILD_IMPROVEMENT,
    LegalActionTypes.PILLAGE: UnitActionType.PILLAGE,
    LegalActionTypes.REPAIR: UnitActionType.REPAIR,
    # Naval actions
    LegalActionTypes.EMBARK: UnitActionType.EMBARK,
    LegalActionTypes.DISEMBARK: UnitActionType.DISEMBARK,
    # Unit management
    LegalActionTypes.PROMOTE: UnitActionType.PROMOTE,
    LegalActionTypes.HEAL: UnitActionType.HEAL,
    LegalActionTypes.DELETE: UnitActionType.DELETE,
    # Air unit actions
    LegalActionTypes.AIRLIFT: UnitActionType.AIRLIFT,
    LegalActionTypes.PARADROP: UnitActionType.PARADROP,
}


# City item type strings (centralized for protocol consistency)
class CityItemTypes:
    """Constants for city production item types."""

    UNIT = "unit"
    BUILDING = "building"
    WONDER = "wonder"
    DISTRICT = "district"


# Maximum range for hashed item IDs. Must match ActionSpaceConfig max production
# items. Civ6 base game has ~170 distinct producible items across all categories.
# Defined in GameConstants.ITEM_HASH_RANGE


def _convert_item_id_to_int(item_id: Any) -> int | None:
    """Convert an item ID to integer, handling various input formats.

    This function handles backwards compatibility with different formats
    that the Lua mod might send:
    - Integer: returned as-is
    - Numeric string (e.g., "42"): converted to int
    - Type name string (e.g., "UNIT_WARRIOR"): hashed to consistent int

    Args:
        item_id: The item ID in various formats.

    Returns:
        Integer item ID, or None if conversion fails.
    """
    if item_id is None:
        return None

    if isinstance(item_id, int):
        return item_id

    if isinstance(item_id, str):
        # Try numeric string first
        try:
            return int(item_id)
        except ValueError:
            pass

        # Handle Civ6 type name strings by hashing to consistent int
        # Use modulo to keep in reasonable range for action space
        if item_id:
            return abs(hash(item_id)) % GameConstants.ITEM_HASH_RANGE

    # Log unexpected types at debug level
    logger.debug(f"Could not convert item_id '{item_id}' (type: {type(item_id).__name__}) to int")
    return None


class InvalidActionError(Exception):
    """Raised when an action cannot be converted or is invalid."""

    def __init__(self, message: str, action_idx: int | None = None) -> None:
        super().__init__(message)
        self.action_idx = action_idx


class NoLegalActionsError(InvalidActionError):
    """Raised when there are no legal actions available to choose from."""
    pass


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
    ) -> None:
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
        strict: bool = True,
    ) -> LuaCommand:
        """Convert action index to Lua command.

        Args:
            action_idx: Flat action index from policy network.
            game_state: Current game state (converted format).
            legal_actions: Legal actions from Lua (for validation).
            validate: Whether to validate against legal actions. If True and
                the action is not in legal_actions, raises InvalidActionError.
            strict: Default True as of v1.8.0 W2.1 step 3. When True, an
                action category that has no implemented branch (i.e. one of
                DIPLOMACY, ESPIONAGE, GOVERNMENT, GREAT_PERSON not yet
                handled here) raises :class:`InvalidActionError` instead of
                silently returning an END_TURN placeholder. Set to False to
                opt into the legacy silent-sink behaviour for backward
                compatibility with callers that haven't been audited yet
                (a warning is logged whenever the legacy branch fires).

        Returns:
            LuaCommand ready for execution.

        Raises:
            InvalidActionError: If action cannot be decoded, or if
                validate=True and the action is not in legal_actions, or
                if strict=True and the action category has no implemented
                branch.
        """
        # Decode the hierarchical action
        try:
            decoded = self.action_space.decode(action_idx)
        except Exception as e:
            raise InvalidActionError(
                f"Failed to decode action {action_idx}: {e}", action_idx
            ) from e

        # Validate against legal actions if requested
        if validate and legal_actions:
            if not self._is_action_legal(decoded, game_state, legal_actions):
                raise InvalidActionError(
                    f"Action {action_idx} (category={decoded.category.name}) "
                    f"is not in the legal actions list",
                    action_idx,
                )

        # Convert to Lua command based on category
        if decoded.category == ActionCategory.END_TURN:
            return LuaCommand(action_type=ActionType.END_TURN)

        elif decoded.category == ActionCategory.UNIT_ACTION:
            return self._convert_unit_action(decoded, game_state, legal_actions)

        elif decoded.category == ActionCategory.CITY_ACTION:
            return self._convert_city_action(decoded, game_state, legal_actions)

        elif decoded.category == ActionCategory.RESEARCH_TECH:
            return self._convert_tech_action(decoded, game_state)

        elif decoded.category == ActionCategory.RESEARCH_CIVIC:
            return self._convert_civic_action(decoded, game_state)

        elif decoded.category == ActionCategory.DIPLOMACY:
            return self._convert_diplomacy_action(decoded, game_state)

        else:
            # Unhandled category. Pre-v1.8.0 this silently returned END_TURN
            # with metadata.unsupported=True, masking real protocol gaps
            # (silent-sink fallthrough). v1.8.0 W2.1 step 3 made strict=True
            # the default; pass strict=False to opt into the legacy path.
            if strict:
                raise InvalidActionError(
                    f"Action {action_idx} (category={decoded.category.name}) "
                    f"has no implemented converter branch. Pass strict=False to "
                    f"fall back to the legacy END_TURN placeholder.",
                    action_idx,
                )
            logger.warning(
                "ActionConverter.convert(strict=False) silent fallback fired "
                "for action_idx=%d, category=%s — caller should be updated to "
                "handle the unsupported category explicitly.",
                action_idx,
                decoded.category.name,
            )
            return LuaCommand(
                action_type=ActionType.END_TURN,
                metadata={"category": decoded.category.name, "unsupported": True},
            )

    def _convert_diplomacy_action(
        self,
        decoded: HierarchicalAction,
        _game_state: dict[str, Any],
    ) -> LuaCommand:
        """Map a hierarchical diplomacy action to a Lua DIPLOMACY command.

        v1.8.0 W2.1 step 3 (hardened in step 3.5): only the 3 sub-actions in
        :data:`alpha_zero.core.civ6_actions.BRIDGE_IMPLEMENTED_DIPLOMACY`
        (DECLARE_WAR / MAKE_PEACE / DENOUNCE) get an ``implemented_this_sprint=True``
        marker. Other sub-actions still emit a DIPLOMACY command so the Lua
        side can log the deferred sub-action in one place — ``executeDiplomacy``
        returns ``NOT_IMPLEMENTED`` for them.

        Selector convention: ``decoded.selector`` holds the target player id
        for diplomacy actions (Level 1 of the action hierarchy). ``decoded.target``
        is reserved for spatial coordinates / item indices and is NOT a valid
        player-id field for diplomacy — earlier draft used it by mistake.
        """
        # The decoded.action_type for DIPLOMACY is the IntEnum int value.
        sub_action_int = int(decoded.action_type)
        try:
            sub_action = DiplomacyActionType(sub_action_int)
        except ValueError:
            # Decoder produced an out-of-range diplomacy id; record metadata
            # for the Lua side to log but mark unsupported.
            return LuaCommand(
                action_type=ActionType.DIPLOMACY,
                metadata={
                    "sub_action": sub_action_int,
                    "unsupported_sub_action": True,
                },
            )

        # Level 1 selector encodes the opposing player id for diplomacy.
        target_player = int(decoded.selector or 0)
        return LuaCommand(
            action_type=ActionType.DIPLOMACY,
            metadata={
                "sub_action": int(sub_action),
                "sub_action_name": sub_action.name,
                "target_player_id": target_player,
                "implemented_this_sprint": sub_action in BRIDGE_IMPLEMENTED_DIPLOMACY,
            },
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
            # strict=False preserves the legacy silent-sink contract that
            # `convert_with_result` callers (notably
            # `alpha_zero.envs.real_civ6_env`) depend on: unsupported
            # categories return an END_TURN placeholder rather than raising.
            # The outer wrapper distinguishes via `is_valid` / `validation_error`.
            command = self.convert(
                action_idx, game_state, legal_actions, validate=False, strict=False
            )

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
        legal_actions: list[LegalAction | dict] | None = None,
    ) -> LuaCommand:
        """Convert unit action to Lua command.

        Args:
            decoded: Decoded hierarchical action.
            game_state: Current game state.
            legal_actions: Legal actions (for future use, validation done in convert()).

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

        elif action_type == UnitActionType.SLEEP:
            # v1.8.0 W2.1 — surface SLEEP as its own command type instead of
            # silently degrading to SKIP. Lua handler at ActionExecutor.lua:233.
            return LuaCommand(
                action_type=ActionType.UNIT_SLEEP,
                unit_id=unit_id,
            )

        elif action_type == UnitActionType.PROMOTE:
            # v1.8.0 W2.1 — Lua handler implemented in step 4. Promotion id
            # carried in `item_id` so the Lua side can pick the promotion to
            # apply; defaults to 0 (mod selects the first available promotion).
            return LuaCommand(
                action_type=ActionType.UNIT_PROMOTE,
                unit_id=unit_id,
                item_id=target if isinstance(target, int) else 0,
            )

        elif action_type == UnitActionType.DELETE:
            # v1.8.0 W2.1 — Lua handler implemented in step 4.
            return LuaCommand(
                action_type=ActionType.UNIT_DELETE,
                unit_id=unit_id,
            )

        else:
            # Unit action types still without a dedicated converter branch
            # (ATTACK already handled above as the explicit case; remaining
            # below-this-line members are RANGED_ATTACK, PILLAGE, REPAIR,
            # EMBARK, DISEMBARK, HEAL, AIRLIFT, PARADROP — deferred to a
            # follow-up sprint). Fall back to SKIP and record metadata so
            # the action-mask invariant test can identify gaps; the agent
            # loop will report `invalid_actions += 1` when the Lua side
            # acks NOT_IMPLEMENTED.
            return LuaCommand(
                action_type=ActionType.UNIT_SKIP,
                unit_id=unit_id,
                metadata={"original_action": action_type.name, "unsupported": True},
            )

    def _convert_city_action(
        self,
        decoded: HierarchicalAction,
        game_state: dict[str, Any],
        legal_actions: list[LegalAction | dict] | None = None,
    ) -> LuaCommand:
        """Convert city action to Lua command.

        Args:
            decoded: Decoded hierarchical action.
            game_state: Current game state.
            legal_actions: Legal actions (for future use, validation done in convert()).

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
                item_type=CityItemTypes.UNIT,
            )

        elif action_type == CityActionType.PRODUCE_BUILDING:
            return LuaCommand(
                action_type=ActionType.CITY_PRODUCE,
                city_id=city_id,
                item_id=item_id,
                item_type=CityItemTypes.BUILDING,
            )

        elif action_type == CityActionType.PRODUCE_WONDER:
            return LuaCommand(
                action_type=ActionType.CITY_PRODUCE,
                city_id=city_id,
                item_id=item_id,
                item_type=CityItemTypes.WONDER,
            )

        elif action_type == CityActionType.PRODUCE_DISTRICT:
            return LuaCommand(
                action_type=ActionType.CITY_PRODUCE,
                city_id=city_id,
                item_id=item_id,
                item_type=CityItemTypes.DISTRICT,
            )

        elif action_type == CityActionType.BUY_UNIT:
            return LuaCommand(
                action_type=ActionType.CITY_BUY,
                city_id=city_id,
                item_id=item_id,
                item_type=CityItemTypes.UNIT,
            )

        elif action_type == CityActionType.BUY_BUILDING:
            return LuaCommand(
                action_type=ActionType.CITY_BUY,
                city_id=city_id,
                item_id=item_id,
                item_type=CityItemTypes.BUILDING,
            )

        elif action_type == CityActionType.MANAGE_CITIZENS:
            # v1.8.0 W2.1 — Lua handler implemented in step 4. `item_id`
            # carries the plot index to toggle; `metadata["lock"]` carries
            # the desired lock state (True/False) for `SetCitizenLocked`.
            lock_state = bool(getattr(decoded, "lock_state", False))
            return LuaCommand(
                action_type=ActionType.CITY_CITIZEN,
                city_id=city_id,
                item_id=item_id,
                metadata={"lock": lock_state},
            )

        else:
            # City action types still without a dedicated converter branch
            # (CHANGE_PRODUCTION, SELL_BUILDING, SET_FOCUS, RANGED_ATTACK,
            # PURCHASE_TILE). Fall back to END_TURN and record metadata so
            # the action-mask invariant test can identify gaps.
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

        orig_x = round(x * scale_x)
        orig_y = round(y * scale_y)

        # Clamp to bounds
        orig_x = max(0, min(width - 1, orig_x))
        orig_y = max(0, min(height - 1, orig_y))

        return (orig_x, orig_y)

    def _parse_legal_actions(
        self, legal_actions: list[LegalAction | dict]
    ) -> list[LegalAction]:
        """Normalize a legal_actions list to LegalAction instances.

        Args:
            legal_actions: Legal actions from Lua, as LegalAction instances
                and/or raw dicts (both forms may be mixed in one call).

        Returns:
            List of LegalAction instances.
        """
        parsed_actions = []
        for action in legal_actions:
            if isinstance(action, dict):
                parsed_actions.append(LegalAction.from_dict(action))
            else:
                parsed_actions.append(action)
        return parsed_actions

    def _is_action_legal(
        self,
        decoded: HierarchicalAction,
        game_state: dict[str, Any],
        legal_actions: list[LegalAction | dict],
    ) -> bool:
        """Check if action is in the legal actions list.

        Args:
            decoded: Decoded hierarchical action.
            game_state: Current game state.
            legal_actions: Legal actions from Lua.

        Returns:
            True if action is legal.
        """
        parsed_actions = self._parse_legal_actions(legal_actions)

        # END_TURN is always legal if in list
        if decoded.category == ActionCategory.END_TURN:
            return any(a.action_type == LegalActionTypes.END_TURN for a in parsed_actions)

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
            if unit_id is None:
                logger.warning(f"Unit at index {decoded.selector} has no _id field")
                return False

            action_type = UnitActionType(decoded.action_type)
            map_size = game_state.get("_map_size")

            for action in parsed_actions:
                if action.unit_id != unit_id:
                    continue

                # Get expected legal action string for this unit action type
                expected_legal_type = None
                for legal_str, unit_type in LEGAL_ACTION_TO_UNIT_TYPE.items():
                    if unit_type == action_type:
                        expected_legal_type = legal_str
                        break

                if expected_legal_type is None:
                    continue

                if action.action_type != expected_legal_type:
                    continue

                # For actions with targets, validate target is in legal targets
                if action.targets:
                    target = decoded.target
                    if isinstance(target, tuple):
                        if map_size:
                            orig_target = self._scale_to_original(target[0], target[1], map_size)
                        else:
                            orig_target = target
                        if orig_target in action.targets:
                            return True
                else:
                    # Actions without targets (fortify, skip, found_city)
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
            if city_id is None:
                logger.warning(f"City at index {decoded.selector} has no _id field")
                return False

            for action in parsed_actions:
                if action.city_id == city_id and action.action_type == LegalActionTypes.PRODUCE:
                    item_id = decoded.target
                    if not isinstance(item_id, int):
                        continue
                    # decoded.target is always the hashed/converted int form
                    # produced by generate_action_mask() via
                    # _convert_item_id_to_int(). action.items holds the raw,
                    # unhashed values Lua sent (which may be non-numeric
                    # strings like "UNIT_WARRIOR"), so it must be hashed the
                    # same way before comparing — comparing against the raw
                    # strings directly always fails.
                    legal_item_ids = {
                        converted
                        for raw_item in action.items
                        if (converted := _convert_item_id_to_int(raw_item)) is not None
                    }
                    if item_id in legal_item_ids:
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

        parsed_actions = self._parse_legal_actions(legal_actions)

        # Always check for END_TURN
        for action in parsed_actions:
            if action.action_type == LegalActionTypes.END_TURN:
                end_turn_idx = self.action_space.category_offsets[ActionCategory.END_TURN]
                mask[end_turn_idx] = 1.0
                break

        # Process unit actions
        units = game_state.get("units", [])
        current_player = game_state.get("current_player", 0)
        map_size = game_state.get("_map_size", (self.board_size, self.board_size))

        for action in parsed_actions:
            # Handle unit actions with targets (move, attack, ranged_attack)
            if action.action_type in LEGAL_ACTION_TO_UNIT_TYPE:
                unit_action_type = LEGAL_ACTION_TO_UNIT_TYPE[action.action_type]

                # Find unit index
                unit_idx = self._find_unit_idx(action.unit_id, units, current_player)
                if unit_idx is None:
                    logger.debug(f"Unit {action.unit_id} not found for action {action.action_type}")
                    continue

                # Actions with targets (move, attack, ranged_attack, build)
                if action.targets:
                    for target in action.targets:
                        scaled_target = self._scale_from_original(target[0], target[1], map_size)
                        self._try_encode_unit_action(
                            mask, unit_idx, unit_action_type, scaled_target
                        )
                else:
                    # Actions without targets (fortify, skip, found_city)
                    # Use (0, 0) as dummy target for targetless actions
                    self._try_encode_unit_action(mask, unit_idx, unit_action_type, (0, 0))

            elif action.action_type == LegalActionTypes.PRODUCE:
                # Find city index
                cities = game_state.get("cities", [])
                player_cities = [c for c in cities if c.get("player") == current_player]
                city_idx = self._find_city_idx(action.city_id, player_cities)

                if city_idx is not None:
                    for raw_item_id in action.items:
                        # Convert item_id to int using robust conversion
                        converted_id = _convert_item_id_to_int(raw_item_id)
                        if converted_id is None:
                            logger.debug(
                                f"Skipping unconvertible item_id '{raw_item_id}' "
                                f"for city {action.city_id} production"
                            )
                            continue
                        item_id = converted_id

                        # Determine item type per-item from action.item_types
                        # (StateExporter.lua's per-item "item_types" map,
                        # keyed by the raw/unhashed item id), falling back
                        # to the legacy singular item_type field for
                        # backwards compatibility with older payloads.
                        item_type = action.item_types.get(
                            str(raw_item_id), action.item_type
                        )
                        if item_type == CityItemTypes.UNIT:
                            self._try_encode_city_action(
                                mask, city_idx, CityActionType.PRODUCE_UNIT, item_id
                            )
                        elif item_type == CityItemTypes.BUILDING:
                            self._try_encode_city_action(
                                mask, city_idx, CityActionType.PRODUCE_BUILDING, item_id
                            )
                        elif item_type == CityItemTypes.WONDER:
                            self._try_encode_city_action(
                                mask, city_idx, CityActionType.PRODUCE_WONDER, item_id
                            )
                        elif item_type == CityItemTypes.DISTRICT:
                            self._try_encode_city_action(
                                mask, city_idx, CityActionType.PRODUCE_DISTRICT, item_id
                            )
                        else:
                            # Unknown type - try both unit and building (backwards compat)
                            self._try_encode_city_action(
                                mask, city_idx, CityActionType.PRODUCE_UNIT, item_id
                            )
                            self._try_encode_city_action(
                                mask, city_idx, CityActionType.PRODUCE_BUILDING, item_id
                            )

        # Ensure at least END_TURN is valid
        if mask.sum() == 0:
            logger.warning(
                "No legal actions could be encoded in action mask. "
                "Falling back to END_TURN. This may indicate a mismatch between "
                "legal_actions format and ActionSpace configuration."
            )
            end_turn_idx = self.action_space.category_offsets[ActionCategory.END_TURN]
            mask[end_turn_idx] = 1.0

        return mask

    def _try_encode_unit_action(
        self,
        mask: np.ndarray,
        unit_idx: int,
        action_type: UnitActionType,
        target: tuple[int, int],
    ) -> None:
        """Try to encode a unit action into the mask.

        Args:
            mask: Action mask to update.
            unit_idx: Unit index.
            action_type: Type of unit action.
            target: Target coordinates (scaled).
        """
        try:
            flat_idx = self.action_space.encode(
                HierarchicalAction(
                    category=ActionCategory.UNIT_ACTION,
                    selector=unit_idx,
                    action_type=action_type.value,
                    target=target,
                )
            )
            if flat_idx < self.action_space_size:
                mask[flat_idx] = 1.0
        except (IndexError, ValueError) as e:
            logger.debug(
                f"Could not encode unit action: unit_idx={unit_idx}, "
                f"action_type={action_type.name}, target={target}: {e}"
            )
        except Exception as e:
            logger.warning(
                f"Unexpected error encoding unit action: unit_idx={unit_idx}, "
                f"action_type={action_type.name}, target={target}: {e}"
            )

    def _try_encode_city_action(
        self,
        mask: np.ndarray,
        city_idx: int,
        action_type: CityActionType,
        item_id: int,
    ) -> None:
        """Try to encode a city action into the mask.

        Args:
            mask: Action mask to update.
            city_idx: City index.
            action_type: Type of city action.
            item_id: Production item ID.
        """
        try:
            flat_idx = self.action_space.encode(
                HierarchicalAction(
                    category=ActionCategory.CITY_ACTION,
                    selector=city_idx,
                    action_type=action_type.value,
                    target=item_id,
                )
            )
            if flat_idx < self.action_space_size:
                mask[flat_idx] = 1.0
        except (IndexError, ValueError) as e:
            logger.debug(
                f"Could not encode city action: city_idx={city_idx}, "
                f"action_type={action_type.name}, item_id={item_id}: {e}"
            )
        except Exception as e:
            logger.warning(
                f"Unexpected error encoding city action: city_idx={city_idx}, "
                f"action_type={action_type.name}, item_id={item_id}: {e}"
            )

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

        scaled_x = round(x * scale_x)
        scaled_y = round(y * scale_y)

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
