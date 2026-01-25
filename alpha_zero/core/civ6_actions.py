"""Hierarchical action space for Civilization VI.

This module defines the action space structure using a hierarchical approach
to reduce the effective branching factor from ~10^6 to ~10^3.

Action hierarchy:
- Level 0: Action Category (9 options)
- Level 1: Selector (unit, city, tech, etc.)
- Level 2: Action Type (move, attack, build, etc.)
- Level 3: Target (position on map or item)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import numpy as np


class ActionCategory(IntEnum):
    """Top-level action categories."""

    END_TURN = 0
    UNIT_ACTION = 1
    CITY_ACTION = 2
    RESEARCH_TECH = 3
    RESEARCH_CIVIC = 4
    DIPLOMACY = 5
    ESPIONAGE = 6
    GOVERNMENT = 7
    GREAT_PERSON = 8


class UnitActionType(IntEnum):
    """Unit action types for Level 2 selection."""

    MOVE = 0
    ATTACK = 1
    RANGED_ATTACK = 2
    FORTIFY = 3
    SKIP = 4
    SLEEP = 5
    PILLAGE = 6
    FOUND_CITY = 7
    BUILD_IMPROVEMENT = 8
    REPAIR = 9
    EMBARK = 10
    DISEMBARK = 11
    PROMOTE = 12
    HEAL = 13
    DELETE = 14
    AIRLIFT = 15
    PARADROP = 16


class CityActionType(IntEnum):
    """City action types for Level 2 selection."""

    PRODUCE_UNIT = 0
    PRODUCE_BUILDING = 1
    PRODUCE_WONDER = 2
    PRODUCE_DISTRICT = 3
    BUY_UNIT = 4
    BUY_BUILDING = 5
    CHANGE_PRODUCTION = 6
    SELL_BUILDING = 7
    MANAGE_CITIZENS = 8
    SET_FOCUS = 9
    RANGED_ATTACK = 10
    PURCHASE_TILE = 11


class DiplomacyActionType(IntEnum):
    """Diplomacy action types."""

    DECLARE_WAR = 0
    MAKE_PEACE = 1
    OPEN_BORDERS = 2
    ALLIANCE = 3
    TRADE_DEAL = 4
    DENOUNCE = 5
    DELEGATE = 6
    EMBASSY = 7
    JOINT_WAR = 8


@dataclass
class HierarchicalAction:
    """Represents a fully-specified hierarchical action.

    Attributes:
        category: The action category (Level 0)
        selector: Index selecting unit/city/tech (Level 1)
        action_type: Specific action type within category (Level 2)
        target: Target position (x, y) or item index (Level 3)
    """

    category: ActionCategory
    selector: int = 0
    action_type: int = 0
    target: tuple[int, int] | int = 0

    def to_flat_index(self, action_space: ActionSpace) -> int:
        """Convert hierarchical action to flat index."""
        return action_space.encode(self)

    @classmethod
    def from_flat_index(
        cls, flat_index: int, action_space: ActionSpace,
    ) -> HierarchicalAction:
        """Create hierarchical action from flat index."""
        return action_space.decode(flat_index)


class ActionSpaceConfig(NamedTuple):
    """Configuration for action space dimensions."""

    board_size: int = 16
    max_units: int = 100
    max_cities: int = 20
    num_techs: int = 73
    num_civics: int = 56
    num_players: int = 8
    num_production_items: int = 200
    num_unit_action_types: int = 17
    num_city_action_types: int = 12
    num_diplomacy_actions: int = 9


class ActionSpace:
    """Hierarchical action space for Civ 6.

    Manages encoding/decoding between hierarchical and flat action representations.
    Supports action masking for valid action selection.
    """

    def __init__(self, config: ActionSpaceConfig | None = None):
        """Initialize action space.

        Args:
            config: Action space configuration. Uses defaults if None.
        """
        self.config = config or ActionSpaceConfig()
        self._compute_sizes()

    def _compute_sizes(self) -> None:
        """Pre-compute action space sizes and offsets."""
        cfg = self.config
        board_tiles = cfg.board_size * cfg.board_size

        # Category sizes (flat representation)
        self.category_sizes = {
            ActionCategory.END_TURN: 1,
            ActionCategory.UNIT_ACTION: cfg.max_units * cfg.num_unit_action_types * board_tiles,
            ActionCategory.CITY_ACTION: cfg.max_cities
            * cfg.num_city_action_types
            * cfg.num_production_items,
            ActionCategory.RESEARCH_TECH: cfg.num_techs,
            ActionCategory.RESEARCH_CIVIC: cfg.num_civics,
            ActionCategory.DIPLOMACY: cfg.num_players * cfg.num_diplomacy_actions,
            ActionCategory.ESPIONAGE: 50,  # Simplified
            ActionCategory.GOVERNMENT: 50,  # Simplified
            ActionCategory.GREAT_PERSON: 20,  # Simplified
        }

        # Compute offsets
        self.category_offsets = {}
        offset = 0
        for cat in ActionCategory:
            self.category_offsets[cat] = offset
            offset += self.category_sizes[cat]

        self.total_size = offset

    @property
    def num_categories(self) -> int:
        """Number of action categories."""
        return len(ActionCategory)

    def get_category_mask(self, game_state: dict) -> np.ndarray:
        """Get mask for valid action categories.

        Args:
            game_state: Current game state dictionary.

        Returns:
            Boolean mask of shape (num_categories,).
        """
        mask = np.zeros(self.num_categories, dtype=np.float32)

        # END_TURN is always valid
        mask[ActionCategory.END_TURN] = 1.0

        # Check for units with moves remaining
        units = game_state.get("units", [])
        current_player = game_state.get("current_player", 0)
        has_movable_units = any(
            u.get("player") == current_player and u.get("moves_remaining", 0) > 0
            for u in units
        )
        if has_movable_units:
            mask[ActionCategory.UNIT_ACTION] = 1.0

        # Check for cities that can produce
        cities = game_state.get("cities", [])
        has_cities = any(c.get("player") == current_player for c in cities)
        if has_cities:
            mask[ActionCategory.CITY_ACTION] = 1.0

        # Tech research (if no current research)
        if game_state.get("current_tech") is None:
            mask[ActionCategory.RESEARCH_TECH] = 1.0

        # Civic research (if no current civic)
        if game_state.get("current_civic") is None:
            mask[ActionCategory.RESEARCH_CIVIC] = 1.0

        # Diplomacy (if other players exist)
        if game_state.get("num_players", 2) > 1:
            mask[ActionCategory.DIPLOMACY] = 1.0

        return mask

    def get_full_mask(self, game_state: dict) -> np.ndarray:
        """Get full action mask for all flat actions.

        Args:
            game_state: Current game state dictionary.

        Returns:
            Boolean mask of shape (total_size,).
        """
        mask = np.zeros(self.total_size, dtype=np.float32)

        cfg = self.config
        current_player = game_state.get("current_player", 0)
        board_size = cfg.board_size

        # END_TURN always valid
        mask[self.category_offsets[ActionCategory.END_TURN]] = 1.0

        # Unit actions
        units = game_state.get("units", [])
        unit_offset = self.category_offsets[ActionCategory.UNIT_ACTION]
        for i, unit in enumerate(units):
            if i >= cfg.max_units:
                break
            if unit.get("player") != current_player:
                continue
            if unit.get("moves_remaining", 0) <= 0:
                continue

            # For each valid action type for this unit
            valid_actions = self._get_unit_valid_actions(unit, game_state)
            for action_type, valid_targets in valid_actions.items():
                for tx, ty in valid_targets:
                    flat_idx = (
                        unit_offset
                        + i * cfg.num_unit_action_types * board_size * board_size
                        + action_type * board_size * board_size
                        + ty * board_size
                        + tx
                    )
                    if flat_idx < self.total_size:
                        mask[flat_idx] = 1.0

        # City actions
        cities = game_state.get("cities", [])
        city_offset = self.category_offsets[ActionCategory.CITY_ACTION]
        for i, city in enumerate(cities):
            if i >= cfg.max_cities:
                break
            if city.get("player") != current_player:
                continue

            # Production items this city can build
            available_items = city.get("available_production", list(range(10)))
            for action_type in [CityActionType.PRODUCE_UNIT, CityActionType.PRODUCE_BUILDING]:
                for item_id in available_items:
                    if item_id >= cfg.num_production_items:
                        continue
                    flat_idx = (
                        city_offset
                        + i * cfg.num_city_action_types * cfg.num_production_items
                        + action_type * cfg.num_production_items
                        + item_id
                    )
                    if flat_idx < self.total_size:
                        mask[flat_idx] = 1.0

        # Tech research
        tech_offset = self.category_offsets[ActionCategory.RESEARCH_TECH]
        available_techs = game_state.get("available_techs", list(range(cfg.num_techs)))
        for tech_id in available_techs:
            if tech_id < cfg.num_techs:
                mask[tech_offset + tech_id] = 1.0

        # Civic research
        civic_offset = self.category_offsets[ActionCategory.RESEARCH_CIVIC]
        available_civics = game_state.get("available_civics", list(range(cfg.num_civics)))
        for civic_id in available_civics:
            if civic_id < cfg.num_civics:
                mask[civic_offset + civic_id] = 1.0

        return mask

    def _get_unit_valid_actions(
        self, unit: dict, game_state: dict,
    ) -> dict[int, list[tuple[int, int]]]:
        """Get valid actions and targets for a unit.

        Args:
            unit: Unit dictionary.
            game_state: Current game state.

        Returns:
            Dict mapping action_type to list of valid (x, y) targets.
        """
        valid_actions: dict[int, list[tuple[int, int]]] = {}
        pos = unit.get("position", (0, 0))
        board_size = self.config.board_size

        # Movement targets (adjacent tiles)
        move_targets = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < board_size and 0 <= ny < board_size:
                    move_targets.append((nx, ny))

        if move_targets:
            valid_actions[UnitActionType.MOVE] = move_targets

        # Skip/Fortify at current position
        valid_actions[UnitActionType.SKIP] = [pos]
        valid_actions[UnitActionType.FORTIFY] = [pos]

        # Attack targets (adjacent enemy units)
        attack_targets = []
        enemy_units = [
            u
            for u in game_state.get("units", [])
            if u.get("player") != unit.get("player")
        ]
        for enemy in enemy_units:
            enemy_pos = enemy.get("position", (-100, -100))
            dx = abs(enemy_pos[0] - pos[0])
            dy = abs(enemy_pos[1] - pos[1])
            if dx <= 1 and dy <= 1 and (dx + dy) > 0:
                attack_targets.append(enemy_pos)

        if attack_targets:
            valid_actions[UnitActionType.ATTACK] = attack_targets

        # Settler can found city
        if unit.get("type") == "settler":
            valid_actions[UnitActionType.FOUND_CITY] = [pos]

        return valid_actions

    def encode(self, action: HierarchicalAction) -> int:
        """Encode hierarchical action to flat index.

        Args:
            action: Hierarchical action to encode.

        Returns:
            Flat action index.
        """
        cfg = self.config
        base_offset = self.category_offsets[action.category]

        if action.category == ActionCategory.END_TURN:
            return base_offset

        if action.category == ActionCategory.UNIT_ACTION:
            board_size = cfg.board_size
            target = action.target if isinstance(action.target, tuple) else (0, 0)
            return (
                base_offset
                + action.selector * cfg.num_unit_action_types * board_size * board_size
                + action.action_type * board_size * board_size
                + target[1] * board_size
                + target[0]
            )

        if action.category == ActionCategory.CITY_ACTION:
            item_id = action.target if isinstance(action.target, int) else 0
            return (
                base_offset
                + action.selector * cfg.num_city_action_types * cfg.num_production_items
                + action.action_type * cfg.num_production_items
                + item_id
            )

        if action.category == ActionCategory.RESEARCH_TECH:
            return base_offset + action.selector

        if action.category == ActionCategory.RESEARCH_CIVIC:
            return base_offset + action.selector

        if action.category == ActionCategory.DIPLOMACY:
            return (
                base_offset + action.selector * cfg.num_diplomacy_actions + action.action_type
            )

        # Default for other categories
        return base_offset + action.selector

    def decode(self, flat_index: int) -> HierarchicalAction:
        """Decode flat index to hierarchical action.

        Args:
            flat_index: Flat action index.

        Returns:
            Decoded hierarchical action.
        """
        cfg = self.config

        # Find category
        category = ActionCategory.END_TURN
        for cat in ActionCategory:
            if flat_index < self.category_offsets[cat] + self.category_sizes[cat]:
                category = cat
                break

        local_idx = flat_index - self.category_offsets[category]

        if category == ActionCategory.END_TURN:
            return HierarchicalAction(category=category)

        if category == ActionCategory.UNIT_ACTION:
            board_size = cfg.board_size
            tiles = board_size * board_size
            unit_idx = local_idx // (cfg.num_unit_action_types * tiles)
            remainder = local_idx % (cfg.num_unit_action_types * tiles)
            action_type = remainder // tiles
            tile_idx = remainder % tiles
            tx = tile_idx % board_size
            ty = tile_idx // board_size
            return HierarchicalAction(
                category=category,
                selector=unit_idx,
                action_type=action_type,
                target=(tx, ty),
            )

        if category == ActionCategory.CITY_ACTION:
            items = cfg.num_production_items
            city_idx = local_idx // (cfg.num_city_action_types * items)
            remainder = local_idx % (cfg.num_city_action_types * items)
            action_type = remainder // items
            item_id = remainder % items
            return HierarchicalAction(
                category=category,
                selector=city_idx,
                action_type=action_type,
                target=item_id,
            )

        if category == ActionCategory.RESEARCH_TECH:
            return HierarchicalAction(category=category, selector=local_idx)

        if category == ActionCategory.RESEARCH_CIVIC:
            return HierarchicalAction(category=category, selector=local_idx)

        if category == ActionCategory.DIPLOMACY:
            player_idx = local_idx // cfg.num_diplomacy_actions
            action_type = local_idx % cfg.num_diplomacy_actions
            return HierarchicalAction(
                category=category,
                selector=player_idx,
                action_type=action_type,
            )

        # Default
        return HierarchicalAction(category=category, selector=local_idx)

    def sample_valid_action(self, mask: np.ndarray, rng: np.random.Generator) -> int:
        """Sample a random valid action from mask.

        Args:
            mask: Boolean action mask.
            rng: Random number generator.

        Returns:
            Sampled action index.
        """
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) == 0:
            # Fallback to END_TURN
            return self.category_offsets[ActionCategory.END_TURN]
        return int(rng.choice(valid_indices))


# Simplified action space for MicroCiv (Phase 1)
class MicroCivActionSpace:
    """Simplified action space for MicroCiv training environment.

    Actions:
    - END_TURN: 1
    - UNIT_MOVE: max_units * 8 (8 directions)
    - UNIT_ATTACK: max_units * 8 (8 directions)
    - UNIT_SKIP: max_units
    - CITY_PRODUCE: max_cities * num_unit_types
    """

    def __init__(self, board_size: int = 16, max_units: int = 20, max_cities: int = 5):
        """Initialize MicroCiv action space.

        Args:
            board_size: Size of the game board.
            max_units: Maximum number of units per player.
            max_cities: Maximum number of cities per player.
        """
        self.board_size = board_size
        self.max_units = max_units
        self.max_cities = max_cities
        self.num_unit_types = 3  # warrior, settler, archer

        # Action offsets
        self.end_turn_offset = 0
        self.move_offset = 1
        self.attack_offset = self.move_offset + max_units * 8
        self.skip_offset = self.attack_offset + max_units * 8
        self.produce_offset = self.skip_offset + max_units
        self.total_size = self.produce_offset + max_cities * self.num_unit_types

        # Direction deltas (8 directions)
        self.directions = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]

    def get_valid_mask(self, game_state: dict) -> np.ndarray:
        """Get valid action mask for current state.

        Args:
            game_state: Current game state.

        Returns:
            Boolean mask of shape (total_size,).
        """
        mask = np.zeros(self.total_size, dtype=np.float32)
        current_player = game_state.get("current_player", 0)

        # END_TURN always valid
        mask[self.end_turn_offset] = 1.0

        units = game_state.get("units", [])
        cities = game_state.get("cities", [])
        terrain = game_state.get("terrain", {})

        # Get enemy positions for attack checking
        enemy_positions = {
            tuple(u.get("position", (-1, -1)))
            for u in units
            if u.get("player") != current_player
        }

        # Unit movement and attacks - use own_unit_idx for current player's units
        own_unit_idx = 0
        for unit in units:
            if own_unit_idx >= self.max_units:
                break
            if unit.get("player") != current_player:
                continue

            # Now own_unit_idx matches the index in _get_player_units()
            if unit.get("moves_remaining", 0) > 0:
                pos = unit.get("position", (0, 0))

                # Movement
                for dir_idx, (dx, dy) in enumerate(self.directions):
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if 0 <= nx < self.board_size and 0 <= ny < self.board_size:
                        # Check terrain is passable
                        tile = terrain.get((nx, ny), {})
                        if tile.get("type") != "mountain":
                            mask[self.move_offset + own_unit_idx * 8 + dir_idx] = 1.0

                # Attacks
                for dir_idx, (dx, dy) in enumerate(self.directions):
                    nx, ny = pos[0] + dx, pos[1] + dy
                    if (nx, ny) in enemy_positions:
                        mask[self.attack_offset + own_unit_idx * 8 + dir_idx] = 1.0

                # Skip
                mask[self.skip_offset + own_unit_idx] = 1.0

            own_unit_idx += 1

        # City production - use own_city_idx for current player's cities
        own_city_idx = 0
        for city in cities:
            if own_city_idx >= self.max_cities:
                break
            if city.get("player") != current_player:
                continue

            # Now own_city_idx matches the index in _get_player_cities()
            if city.get("producing") is None:
                for unit_type in range(self.num_unit_types):
                    mask[self.produce_offset + own_city_idx * self.num_unit_types + unit_type] = 1.0

            own_city_idx += 1

        return mask

    def decode_action(self, action_idx: int) -> dict:
        """Decode flat action index to action dict.

        Args:
            action_idx: Flat action index.

        Returns:
            Action dictionary with type and parameters.
        """
        if action_idx == self.end_turn_offset:
            return {"type": "end_turn"}

        if action_idx < self.attack_offset:
            # Movement
            local_idx = action_idx - self.move_offset
            unit_idx = local_idx // 8
            dir_idx = local_idx % 8
            dx, dy = self.directions[dir_idx]
            return {"type": "move", "unit_idx": unit_idx, "direction": (dx, dy)}

        if action_idx < self.skip_offset:
            # Attack
            local_idx = action_idx - self.attack_offset
            unit_idx = local_idx // 8
            dir_idx = local_idx % 8
            dx, dy = self.directions[dir_idx]
            return {"type": "attack", "unit_idx": unit_idx, "direction": (dx, dy)}

        if action_idx < self.produce_offset:
            # Skip
            unit_idx = action_idx - self.skip_offset
            return {"type": "skip", "unit_idx": unit_idx}

        # City production
        local_idx = action_idx - self.produce_offset
        city_idx = local_idx // self.num_unit_types
        unit_type = local_idx % self.num_unit_types
        unit_names = ["warrior", "settler", "archer"]
        return {"type": "produce", "city_idx": city_idx, "unit_type": unit_names[unit_type]}

    def encode_action(self, action: dict) -> int:
        """Encode action dict to flat index.

        Args:
            action: Action dictionary.

        Returns:
            Flat action index.
        """
        action_type = action.get("type")

        if action_type == "end_turn":
            return self.end_turn_offset

        if action_type == "move":
            unit_idx = int(action.get("unit_idx", 0))
            direction = action.get("direction", (0, 0))
            dir_idx = self.directions.index(direction) if direction in self.directions else 0
            return self.move_offset + unit_idx * 8 + dir_idx

        if action_type == "attack":
            unit_idx = int(action.get("unit_idx", 0))
            direction = action.get("direction", (0, 0))
            dir_idx = self.directions.index(direction) if direction in self.directions else 0
            return self.attack_offset + unit_idx * 8 + dir_idx

        if action_type == "skip":
            unit_idx = int(action.get("unit_idx", 0))
            return self.skip_offset + unit_idx

        if action_type == "produce":
            city_idx = int(action.get("city_idx", 0))
            unit_names = ["warrior", "settler", "archer"]
            unit_type_name = action.get("unit_type", "warrior")
            unit_type = (
                unit_names.index(unit_type_name) if unit_type_name in unit_names else 0
            )
            return self.produce_offset + city_idx * self.num_unit_types + unit_type

        return self.end_turn_offset

    def sample_valid_action(self, mask: np.ndarray, rng: np.random.Generator) -> int:
        """Sample a random valid action.

        Args:
            mask: Valid action mask.
            rng: Random number generator.

        Returns:
            Sampled action index.
        """
        valid_indices = np.where(mask > 0)[0]
        if len(valid_indices) == 0:
            return self.end_turn_offset
        return int(rng.choice(valid_indices))
