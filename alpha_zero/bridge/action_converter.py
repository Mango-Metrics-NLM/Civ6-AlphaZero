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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["ActionConverter", "ConversionResult", "InvalidActionError", "NoLegalActionsError"]

import numpy as np

from alpha_zero.bridge import production_slots
from alpha_zero.bridge.protocol import (
    ActionType,
    LegalAction,
    LuaCommand,
    Position,
)
from alpha_zero.bridge.tech_registry import (
    GAMEINFO_ENV,
    ResearchRegistry,
    ResearchRegistryError,
    choose_registry,
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


#: The production kind each city action type stands for (and the reverse).
_PRODUCE_KIND: dict[int, str] = {
    int(CityActionType.PRODUCE_UNIT): CityItemTypes.UNIT,
    int(CityActionType.PRODUCE_BUILDING): CityItemTypes.BUILDING,
    int(CityActionType.PRODUCE_WONDER): CityItemTypes.WONDER,
    int(CityActionType.PRODUCE_DISTRICT): CityItemTypes.DISTRICT,
}
_PRODUCE_ACTION: dict[str, CityActionType] = {
    kind: CityActionType(action) for action, kind in _PRODUCE_KIND.items()
}


def _convert_item_id_to_int(item_id: Any) -> int | None:
    """An item id as a number: an int as it is, a numeric string, a name by a stable bucket.

    Production no longer uses this (see ``production_slots``); kept for purchases.
    """
    return production_slots.legacy_slot(item_id)


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
        research_registry: ResearchRegistry | Literal[False] | None = None,
    ) -> None:
        """Initialize the action converter.

        Args:
            board_size: Board size for coordinate mapping.
            action_space_config: Custom action space config (uses defaults if None).
            research_registry: Which research action each technology and civic
                is. None asks the environment (``CIV6AZ_BRIDGE_RULESET``);
                False means none, without asking. With no registry a research
                action's selector is the game's own row index.

        Raises:
            ResearchRegistryError: If the environment names a registry that
                cannot be loaded.
        """
        self.board_size = board_size
        self.research_registry = choose_registry(research_registry)
        self._research_stats = {"unmapped_research_items": 0, "research_out_of_range": 0}
        self._warned_lost_research = False
        # The research offers the counters last counted, so that building the
        # mask twice for one state counts once.
        self._last_counted_offers: tuple[Any, ...] | None = None
        if self.research_registry is None:
            logger.info(
                "ActionConverter: no research registry; a research action is the game's row index"
            )
        else:
            logger.info("ActionConverter research actions: %s", self.research_registry.describe())
            if not self.research_registry.civic_ids:
                logger.warning(
                    "The research registry has no civic slots, so no civic can be chosen and "
                    "an end of turn blocked on a civic choice cannot be cleared. "
                    "Set %s to a GameInfo export for the civic order.",
                    GAMEINFO_ENV,
                )

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

    @property
    def research_stats(self) -> dict[str, int]:
        """Counters of research choices the mask could not offer, since construction.

        ``unmapped_research_items``: choices with no selector (no slot in the
        registry; without a registry, an item that is not a row index).
        ``research_out_of_range``: choices whose selector the action space has
        no research action for. The same offers seen again straight after
        themselves are counted once.
        """
        return dict(self._research_stats)

    def _check_same_registry(self, game_state: dict[str, Any]) -> None:
        """Refuse a state whose research was numbered by another registry.

        The state converter stamps its registry's fingerprint on the state. If
        it differs from this converter's, the encoder input and the research
        actions would mean different things. A state without the stamp (built
        by hand) is not checked.

        Raises:
            ResearchRegistryError: If the two converters use different registries.
        """
        if "_research_fingerprint" not in game_state:
            return
        theirs = game_state["_research_fingerprint"]
        ours = self.research_registry.fingerprint if self.research_registry else None
        if theirs != ours:
            raise ResearchRegistryError(
                "the state converter and the action converter use different research "
                f"registries (state: {theirs or 'none'}, actions: {ours or 'none'}); "
                "build both with the same registry"
            )

    def _research_selectors(self, action: LegalAction) -> tuple[list[int], list[str], list[str]]:
        """What one ``research_tech`` / ``research_civic`` legal action allows.

        Returns the selectors it allows, and the choices that got none: those
        with no selector at all, and those whose selector the action space has
        no research action for. With a registry the selector is the slot of
        each offered type name; without one it is the game's own row index.
        """
        is_tech = action.action_type == LegalActionTypes.RESEARCH_TECH
        config = self.action_space.config
        limit = config.num_techs if is_tech else config.num_civics
        registry = self.research_registry
        selectors: list[int] = []
        unmapped: list[str] = []
        out_of_range: list[str] = []

        if registry is None:
            for item in action.items:
                if not isinstance(item, int) or isinstance(item, bool):
                    unmapped.append(str(item))
                elif 0 <= item < limit:
                    selectors.append(item)
                else:
                    out_of_range.append(str(item))
        elif not action.item_names:
            # No names to look up (a mod older than state schema 2).
            unmapped.extend(str(item) for item in action.items)
        else:
            slot_of = registry.tech_slot if is_tech else registry.civic_slot
            for name in action.item_names:
                slot = slot_of(name)
                if slot is None:
                    unmapped.append(str(name))
                elif slot >= limit:
                    out_of_range.append(str(name))
                else:
                    selectors.append(slot)

        return selectors, unmapped, out_of_range

    def convert(
        self,
        action_idx: int,
        game_state: dict[str, Any],
        legal_actions: Sequence[LegalAction | dict] | None = None,
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
        self._check_same_registry(game_state)

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
        legal_actions: Sequence[LegalAction | dict] | None = None,
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
        legal_actions: Sequence[LegalAction | dict] | None = None,
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
        legal_actions: Sequence[LegalAction | dict] | None = None,
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

        kind = _PRODUCE_KIND.get(int(action_type))
        if kind is not None:
            # The slot is the item's position among what the game offered this
            # city; the command carries the item itself (its type name).
            item: int | str = item_id
            for action in self._parse_legal_actions(legal_actions or ()):
                if action.city_id == city_id and action.action_type == LegalActionTypes.PRODUCE:
                    offered = production_slots.item_at(
                        kind, item_id, action.items, action.item_types, action.item_type
                    )
                    if offered is not None:
                        item = offered
            return LuaCommand(
                action_type=ActionType.CITY_PRODUCE, city_id=city_id, item_id=item, item_type=kind
            )

        if action_type == CityActionType.BUY_UNIT:
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
        return self._research_command(ActionType.RESEARCH_TECH, decoded.selector, is_tech=True)

    def _research_command(
        self, action_type: ActionType, selector: int, *, is_tech: bool
    ) -> LuaCommand:
        """The research command for a selector.

        Without a registry the selector is the game's row index and is sent as
        it is. With one it is a slot, and the command names the technology or
        civic by the game's type name, which the mod resolves to its own row.

        Raises:
            InvalidActionError: If the registry has nothing at that slot.
        """
        registry = self.research_registry
        if registry is None:
            return LuaCommand(action_type=action_type, item_id=selector)

        type_name = registry.tech_type(selector) if is_tech else registry.civic_type(selector)
        if type_name is None:
            kind = "technology" if is_tech else "civic"
            raise InvalidActionError(
                f"No {kind} has research slot {selector} in rule set {registry.ruleset_id!r}"
            )
        return LuaCommand(action_type=action_type, item_id=type_name)

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
        return self._research_command(ActionType.RESEARCH_CIVIC, decoded.selector, is_tech=False)

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
        self, legal_actions: Sequence[LegalAction | dict]
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
        legal_actions: Sequence[LegalAction | dict],
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
                    kind = _PRODUCE_KIND.get(int(decoded.action_type))
                    slot = decoded.target
                    if kind is None or not isinstance(slot, int):
                        continue
                    if (
                        production_slots.item_at(
                            kind, slot, action.items, action.item_types, action.item_type
                        )
                        is not None
                    ):
                        return True
            return False

        # Research: legal when one of the offered choices has this selector.
        if decoded.category in (ActionCategory.RESEARCH_TECH, ActionCategory.RESEARCH_CIVIC):
            wanted = (
                LegalActionTypes.RESEARCH_TECH
                if decoded.category == ActionCategory.RESEARCH_TECH
                else LegalActionTypes.RESEARCH_CIVIC
            )
            return any(
                decoded.selector in self._research_selectors(action)[0]
                for action in parsed_actions
                if action.action_type == wanted
            )

        # Default: assume valid for other categories
        return True

    def generate_action_mask(
        self,
        legal_actions: Sequence[LegalAction | dict],
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
        self._check_same_registry(game_state)
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
                    # One slot per offered item, by its position among the
                    # offered items of its kind (production_slots).
                    groups = production_slots.offered(
                        action.items, action.item_types, action.item_type
                    )
                    for kind, group in groups.items():
                        city_action = _PRODUCE_ACTION.get(kind)
                        if city_action is None:
                            continue
                        # Past the action space's bound a slot would spill into
                        # the next action type's range: it cannot be chosen.
                        for slot in range(
                            min(len(group), self.action_space.config.num_production_items)
                        ):
                            self._try_encode_city_action(mask, city_idx, city_action, slot)

        # Research choices
        research_actions = [
            action
            for action in parsed_actions
            if action.action_type
            in (LegalActionTypes.RESEARCH_TECH, LegalActionTypes.RESEARCH_CIVIC)
        ]
        offers = tuple(
            (action.action_type, tuple(action.items), tuple(action.item_names))
            for action in research_actions
        )
        count = offers != self._last_counted_offers
        self._last_counted_offers = offers
        for action in research_actions:
            category = (
                ActionCategory.RESEARCH_TECH
                if action.action_type == LegalActionTypes.RESEARCH_TECH
                else ActionCategory.RESEARCH_CIVIC
            )
            offset = self.action_space.category_offsets[category]
            selectors, unmapped, out_of_range = self._research_selectors(action)
            for selector in selectors:
                mask[offset + selector] = 1.0
            if count:
                self._research_stats["unmapped_research_items"] += len(unmapped)
                self._research_stats["research_out_of_range"] += len(out_of_range)
            if (unmapped or out_of_range) and not self._warned_lost_research:
                self._warned_lost_research = True
                lost = (unmapped or out_of_range)[0]
                reason = (
                    "has no research action here"
                    if unmapped
                    else "is beyond the action space's research actions"
                )
                logger.warning(
                    "The game offers %s as %s, which %s, so it cannot be chosen "
                    "(%d of this offer's choices can). Warned once; see "
                    "ActionConverter.research_stats for the counts.",
                    lost,
                    action.action_type,
                    reason,
                    len(selectors),
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
