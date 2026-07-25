"""CivRealm environment wrapper for AlphaZero training.

This module provides a Gymnasium-compatible wrapper for CivRealm, the open-source
Freeciv-based reinforcement learning environment. CivRealm provides various
mini-games for training RL agents on Civilization-style strategy tasks.

Features:
- Gymnasium-compatible interface (reset, step, render, close)
- Flexible observation encoding (spatial, vector, action mask)
- Hierarchical action handling for CivRealm's action space
- Curriculum learning through mini-game progression
- Graceful fallback when CivRealm is not installed

Mini-games supported:
- development_build_city: Build and develop cities
- battle_ancient: Ancient era combat scenarios
- battle_medieval: Medieval era combat scenarios
- battle_modern: Modern era combat scenarios
- battle_info_era: Information era combat scenarios
- exploration: Map exploration tasks
- trade: Trade route management
- diplomacy: Diplomatic interaction scenarios

Example usage:
    >>> from alpha_zero.envs.civrealm_wrapper import (
    ...     CivRealmConfig,
    ...     CivRealmWrapper,
    ... )
    >>> config = CivRealmConfig(task="development_build_city")
    >>> env = CivRealmWrapper(config)
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
    >>> env.close()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, ClassVar, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Set up module logger
logger = logging.getLogger(__name__)

# Try to import CivRealm - graceful fallback if not installed or server unavailable
CIVREALM_AVAILABLE = False
CIVREALM_IMPORT_ERROR: str | None = None

try:
    import civrealm

    CIVREALM_AVAILABLE = True
    logger.info("CivRealm successfully imported")
except ImportError as e:
    CIVREALM_IMPORT_ERROR = f"CivRealm not installed: {e}"
    logger.warning(f"{CIVREALM_IMPORT_ERROR}. Using mock environment.")
except Exception as e:
    # CivRealm may fail to import if the server is not running
    # (it tries to connect during module initialization)
    CIVREALM_IMPORT_ERROR = f"CivRealm import failed (server may not be running): {e}"
    logger.warning(f"{CIVREALM_IMPORT_ERROR}. Using mock environment.")


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class CivRealmConfig:
    """CivRealm environment configuration.

    Attributes:
        task: Mini-game task name (e.g., 'development_build_city', 'battle_ancient').
        difficulty: Game difficulty level (1-10).
        spatial_size: Size of spatial observation tensor (H x W).
        max_units: Maximum number of units to encode.
        max_cities: Maximum number of cities to encode.
        max_actions_per_turn: Maximum discrete actions per turn.
        host: CivRealm server hostname.
        port: CivRealm server port.
        client_port: Client connection port.
        username: Player username for multiplayer.
        max_turns: Maximum game turns before truncation.
        render_mode: Rendering mode ('human', 'rgb_array', 'ansi', or None).
        use_mock: Force use of mock environment (for testing).
    """

    # Mini-game selection
    task: str = "development_build_city"
    difficulty: int = 1

    # Observation processing
    spatial_size: int = 32
    max_units: int = 100
    max_cities: int = 50

    # Action space
    max_actions_per_turn: int = 100

    # Docker/server configuration
    host: str = "localhost"
    port: int = 6300
    client_port: int = 6001
    username: str = "alphazero_agent"

    # Game settings
    max_turns: int = 500

    # Rendering
    render_mode: str | None = None

    # Testing/development
    use_mock: bool = False

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid.
        """
        valid_tasks = [
            "development_build_city",
            "battle_ancient",
            "battle_medieval",
            "battle_modern",
            "battle_info_era",
            "exploration",
            "trade",
            "diplomacy",
            "fullgame",
        ]

        if self.task not in valid_tasks:
            logger.warning(
                f"Unknown task '{self.task}'. Valid tasks: {valid_tasks}. "
                f"Proceeding anyway as CivRealm may support additional tasks."
            )

        if not 1 <= self.difficulty <= 10:
            raise ValueError(f"Difficulty must be 1-10, got {self.difficulty}")

        if self.spatial_size < 8:
            raise ValueError(f"Spatial size must be >= 8, got {self.spatial_size}")

        if self.max_units < 1:
            raise ValueError(f"max_units must be >= 1, got {self.max_units}")

        if self.max_cities < 1:
            raise ValueError(f"max_cities must be >= 1, got {self.max_cities}")

        if self.port < 1 or self.port > 65535:
            raise ValueError(f"Port must be 1-65535, got {self.port}")

        logger.debug(f"CivRealmConfig validated: task={self.task}, difficulty={self.difficulty}")


# ============================================================================
# Encoder
# ============================================================================


class TerrainType(IntEnum):
    """CivRealm terrain types for encoding."""

    UNKNOWN = 0
    OCEAN = 1
    DEEP_OCEAN = 2
    COAST = 3
    LAKE = 4
    PLAINS = 5
    GRASSLAND = 6
    DESERT = 7
    TUNDRA = 8
    ARCTIC = 9
    SWAMP = 10
    JUNGLE = 11
    FOREST = 12
    HILLS = 13
    MOUNTAINS = 14


class UnitCategory(IntEnum):
    """Unit categories for encoding."""

    LAND_MILITARY = 0
    NAVAL_MILITARY = 1
    AIR_MILITARY = 2
    WORKER = 3
    SETTLER = 4
    CARAVAN = 5
    DIPLOMAT = 6
    SPY = 7
    OTHER = 8


class CivRealmEncoder:
    """Encodes CivRealm observations into tensor format.

    Converts raw CivRealm game state into:
    - Spatial tensor: (C, H, W) encoding terrain, units, cities, fog
    - Vector tensor: (D,) encoding global game state
    - Action mask: (A,) boolean mask for valid actions

    Channel allocation:
    - 0-14: Terrain types (one-hot)
    - 15-23: Unit categories (own, by type)
    - 24-32: Unit categories (enemy, by type)
    - 33-34: Own cities (presence, population)
    - 35-36: Enemy cities (presence, population)
    - 37: Fog of war (visible=1)
    - 38: Previously seen (seen=1)
    - 39-43: Resources (food, production, trade, gold, science)
    - 44-47: Improvements (road, irrigation, mine, fortress)
    - 48: Unit health (own)
    - 49: Unit moves remaining (own)
    - 50: Constant bias term

    Attributes:
        config: CivRealmConfig with encoding parameters.
        spatial_channels: Number of spatial channels.
        vector_dim: Dimension of vector encoding.
    """

    # Channel constants
    TERRAIN_START = 0
    TERRAIN_END = 15
    OWN_UNITS_START = 15
    OWN_UNITS_END = 24
    ENEMY_UNITS_START = 24
    ENEMY_UNITS_END = 33
    OWN_CITIES_START = 33
    OWN_CITIES_END = 35
    ENEMY_CITIES_START = 35
    ENEMY_CITIES_END = 37
    FOG_START = 37
    FOG_END = 39
    RESOURCES_START = 39
    RESOURCES_END = 44
    IMPROVEMENTS_START = 44
    IMPROVEMENTS_END = 48
    UNIT_STATS_START = 48
    UNIT_STATS_END = 50
    BIAS_CHANNEL = 50

    TOTAL_CHANNELS = 51
    VECTOR_DIM = 256

    def __init__(self, config: CivRealmConfig) -> None:
        """Initialize encoder.

        Args:
            config: CivRealm configuration.
        """
        self.config = config
        self.spatial_channels = self.TOTAL_CHANNELS
        self.vector_dim = self.VECTOR_DIM

        logger.debug(
            f"CivRealmEncoder initialized: spatial={self.spatial_channels}x"
            f"{config.spatial_size}x{config.spatial_size}, vector={self.vector_dim}"
        )

    def encode_observation(self, raw_obs: dict) -> dict[str, np.ndarray]:
        """Encode raw CivRealm observation to tensor format.

        Args:
            raw_obs: Raw observation dictionary from CivRealm.

        Returns:
            Dictionary with 'spatial', 'vector', and 'action_mask' arrays.
        """
        spatial = self._encode_spatial(raw_obs)
        vector = self._encode_vector(raw_obs)
        action_mask = self._create_action_mask(raw_obs)

        return {
            "spatial": spatial,
            "vector": vector,
            "action_mask": action_mask,
        }

    def _encode_spatial(self, obs: dict) -> np.ndarray:
        """Encode spatial features.

        Args:
            obs: Raw observation.

        Returns:
            Spatial tensor of shape (C, H, W).
        """
        size = self.config.spatial_size
        spatial = np.zeros((self.TOTAL_CHANNELS, size, size), dtype=np.float32)

        # Get map data
        map_data = obs.get("map", {})
        player_id = obs.get("player_id", 0)

        # Encode terrain
        terrain = map_data.get("terrain", {})
        for pos_key, tile_data in terrain.items():
            try:
                x, y = self._parse_position(pos_key)
                if not self._in_bounds(x, y, size):
                    continue

                terrain_type = tile_data.get("terrain", 0)
                if 0 <= terrain_type < 15:
                    spatial[self.TERRAIN_START + terrain_type, y, x] = 1.0

                # Terrain features
                if tile_data.get("has_road"):
                    spatial[self.IMPROVEMENTS_START, y, x] = 1.0
                if tile_data.get("has_irrigation"):
                    spatial[self.IMPROVEMENTS_START + 1, y, x] = 1.0
                if tile_data.get("has_mine"):
                    spatial[self.IMPROVEMENTS_START + 2, y, x] = 1.0
                if tile_data.get("has_fortress"):
                    spatial[self.IMPROVEMENTS_START + 3, y, x] = 1.0

                # Resources/yields
                spatial[self.RESOURCES_START, y, x] = tile_data.get("food", 0) / 10.0
                spatial[self.RESOURCES_START + 1, y, x] = tile_data.get("production", 0) / 10.0
                spatial[self.RESOURCES_START + 2, y, x] = tile_data.get("trade", 0) / 10.0

            except (ValueError, TypeError):
                continue

        # Encode visibility/fog of war
        visibility = map_data.get("visibility", {})
        for pos_key, vis_state in visibility.items():
            try:
                x, y = self._parse_position(pos_key)
                if not self._in_bounds(x, y, size):
                    continue

                if vis_state == "visible":
                    spatial[self.FOG_START, y, x] = 1.0
                elif vis_state == "seen":
                    spatial[self.FOG_START + 1, y, x] = 1.0
                # Unexplored tiles remain 0
            except (ValueError, TypeError):
                continue

        # Encode units
        units = obs.get("units", {})
        for _unit_id, unit_data in units.items():
            try:
                pos = unit_data.get("position", {})
                x = pos.get("x", 0)
                y = pos.get("y", 0)
                if not self._in_bounds(x, y, size):
                    continue

                owner = unit_data.get("owner", -1)
                unit_type = unit_data.get("type_name", "")
                category = self._get_unit_category(unit_type)
                health = unit_data.get("hp", 100) / 100.0
                moves = unit_data.get("moves_left", 0) / 10.0

                if owner == player_id:
                    # Own unit
                    channel = self.OWN_UNITS_START + category
                    spatial[channel, y, x] = max(spatial[channel, y, x], health)
                    spatial[self.UNIT_STATS_START, y, x] = health
                    spatial[self.UNIT_STATS_START + 1, y, x] = moves
                else:
                    # Enemy unit
                    channel = self.ENEMY_UNITS_START + category
                    spatial[channel, y, x] = max(spatial[channel, y, x], health)
            except (ValueError, TypeError, KeyError):
                continue

        # Encode cities
        cities = obs.get("cities", {})
        for _city_id, city_data in cities.items():
            try:
                pos = city_data.get("position", {})
                x = pos.get("x", 0)
                y = pos.get("y", 0)
                if not self._in_bounds(x, y, size):
                    continue

                owner = city_data.get("owner", -1)
                population = city_data.get("population", 1) / 50.0

                if owner == player_id:
                    spatial[self.OWN_CITIES_START, y, x] = 1.0
                    spatial[self.OWN_CITIES_START + 1, y, x] = population
                else:
                    spatial[self.ENEMY_CITIES_START, y, x] = 1.0
                    spatial[self.ENEMY_CITIES_START + 1, y, x] = population
            except (ValueError, TypeError, KeyError):
                continue

        # Bias term
        spatial[self.BIAS_CHANNEL, :, :] = 1.0

        return spatial

    def _encode_vector(self, obs: dict) -> np.ndarray:
        """Encode non-spatial (vector) features.

        Args:
            obs: Raw observation.

        Returns:
            Vector tensor of shape (D,).
        """
        vector = np.zeros(self.VECTOR_DIM, dtype=np.float32)

        player_id = obs.get("player_id", 0)

        # Game progress (0-19)
        vector[0] = obs.get("turn", 0) / 500.0
        vector[1] = obs.get("year", 0) / 5000.0  # Years can be negative (BC)

        # Player info (20-49)
        player_info = obs.get("player", {})
        vector[20] = player_info.get("gold", 0) / 10000.0
        vector[21] = player_info.get("gold_income", 0) / 100.0
        vector[22] = player_info.get("science", 0) / 100.0
        vector[23] = player_info.get("luxury", 0) / 100.0
        vector[24] = player_info.get("tax", 50) / 100.0

        # Research
        current_tech = player_info.get("current_research")
        vector[25] = 1.0 if current_tech else 0.0
        vector[26] = player_info.get("research_progress", 0) / 100.0

        # Unit counts (50-79)
        units = obs.get("units", {})
        own_units = [u for u in units.values() if u.get("owner") == player_id]
        enemy_units = [u for u in units.values() if u.get("owner") != player_id]

        vector[50] = len(own_units) / 100.0
        vector[51] = len(enemy_units) / 100.0

        # Count by category
        for category in UnitCategory:
            own_count = sum(
                1
                for u in own_units
                if self._get_unit_category(u.get("type_name", "")) == category.value
            )
            enemy_count = sum(
                1
                for u in enemy_units
                if self._get_unit_category(u.get("type_name", "")) == category.value
            )
            vector[52 + category.value] = own_count / 50.0
            vector[61 + category.value] = enemy_count / 50.0

        # City counts and stats (80-119)
        cities = obs.get("cities", {})
        own_cities = [c for c in cities.values() if c.get("owner") == player_id]
        enemy_cities = [c for c in cities.values() if c.get("owner") != player_id]

        vector[80] = len(own_cities) / 50.0
        vector[81] = len(enemy_cities) / 50.0

        if own_cities:
            total_pop = sum(c.get("population", 1) for c in own_cities)
            vector[82] = total_pop / 500.0
            vector[83] = total_pop / len(own_cities) / 30.0  # Avg population

            total_food = sum(c.get("food_surplus", 0) for c in own_cities)
            vector[84] = total_food / 100.0

            total_production = sum(c.get("production", 0) for c in own_cities)
            vector[85] = total_production / 100.0

        # Diplomatic state (120-159)
        diplomacy = obs.get("diplomacy", {})
        num_allies = 0
        num_enemies = 0
        num_peace = 0

        for _player, state in diplomacy.items():
            if state == "ally":
                num_allies += 1
            elif state == "war":
                num_enemies += 1
            elif state == "peace":
                num_peace += 1

        vector[120] = num_allies / 10.0
        vector[121] = num_enemies / 10.0
        vector[122] = num_peace / 10.0

        # Score components (160-199)
        score = obs.get("score", {})
        vector[160] = score.get("total", 0) / 10000.0
        vector[161] = score.get("cities", 0) / 1000.0
        vector[162] = score.get("units", 0) / 1000.0
        vector[163] = score.get("techs", 0) / 1000.0
        vector[164] = score.get("wonders", 0) / 500.0

        # Game state flags (200-219)
        vector[200] = 1.0 if obs.get("is_phase_done", False) else 0.0
        vector[201] = 1.0 if obs.get("can_end_turn", False) else 0.0

        # Available actions count
        available_actions = obs.get("available_actions", {})
        vector[202] = len(available_actions) / 1000.0

        return vector

    def _create_action_mask(self, obs: dict) -> np.ndarray:
        """Create action mask from available actions.

        Args:
            obs: Raw observation.

        Returns:
            Boolean action mask of shape (max_actions,).
        """
        max_actions = self.config.max_actions_per_turn
        mask = np.zeros(max_actions, dtype=np.float32)

        available_actions = obs.get("available_actions", {})

        # Map available actions to mask indices
        for i, (_action_key, _action_info) in enumerate(available_actions.items()):
            if i >= max_actions:
                break
            mask[i] = 1.0

        # Always allow "end_turn" as last action
        mask[-1] = 1.0

        # Ensure at least one action is valid
        if mask.sum() == 0:
            mask[0] = 1.0

        return mask

    def _parse_position(self, pos_key: Any) -> tuple[int, int]:
        """Parse position from key.

        Args:
            pos_key: Position key (tuple, string, or dict).

        Returns:
            Tuple of (x, y) coordinates.
        """
        if isinstance(pos_key, tuple):
            return int(pos_key[0]), int(pos_key[1])
        elif isinstance(pos_key, str):
            parts = pos_key.strip("()").split(",")
            return int(parts[0].strip()), int(parts[1].strip())
        elif isinstance(pos_key, dict):
            return int(pos_key.get("x", 0)), int(pos_key.get("y", 0))
        else:
            return 0, 0

    def _in_bounds(self, x: int, y: int, size: int) -> bool:
        """Check if position is within bounds."""
        return 0 <= x < size and 0 <= y < size

    def _get_unit_category(self, unit_type: str) -> int:
        """Map unit type name to category index.

        Args:
            unit_type: Unit type name string.

        Returns:
            Category index.
        """
        unit_type_lower = unit_type.lower()

        # Settlers
        if "settler" in unit_type_lower or "pioneer" in unit_type_lower:
            return UnitCategory.SETTLER.value

        # Workers/Engineers
        if "worker" in unit_type_lower or "engineer" in unit_type_lower:
            return UnitCategory.WORKER.value

        # Caravans/Traders
        if (
            "caravan" in unit_type_lower
            or "freight" in unit_type_lower
            or "trader" in unit_type_lower
        ):
            return UnitCategory.CARAVAN.value

        # Diplomats/Spies
        if "diplomat" in unit_type_lower:
            return UnitCategory.DIPLOMAT.value
        if "spy" in unit_type_lower:
            return UnitCategory.SPY.value

        # Naval units
        naval_keywords = [
            "trireme",
            "caravel",
            "galleon",
            "frigate",
            "ironclad",
            "destroyer",
            "cruiser",
            "battleship",
            "carrier",
            "submarine",
            "transport",
            "ship",
        ]
        if any(kw in unit_type_lower for kw in naval_keywords):
            return UnitCategory.NAVAL_MILITARY.value

        # Air units
        air_keywords = ["fighter", "bomber", "helicopter", "stealth", "jet", "aircraft"]
        if any(kw in unit_type_lower for kw in air_keywords):
            return UnitCategory.AIR_MILITARY.value

        # Land military (default for military)
        return UnitCategory.LAND_MILITARY.value

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        """Shape of spatial tensor (C, H, W)."""
        return (self.TOTAL_CHANNELS, self.config.spatial_size, self.config.spatial_size)

    @property
    def observation_space(self) -> spaces.Dict:
        """Get Gymnasium observation space."""
        return spaces.Dict(
            {
                "spatial": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=self.spatial_shape,
                    dtype=np.float32,
                ),
                "vector": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.VECTOR_DIM,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.config.max_actions_per_turn,),
                    dtype=np.float32,
                ),
            }
        )


# ============================================================================
# Action Handler
# ============================================================================


class ActionType(IntEnum):
    """High-level action types in CivRealm."""

    END_TURN = 0
    UNIT_MOVE = 1
    UNIT_ATTACK = 2
    UNIT_BUILD = 3
    UNIT_FORTIFY = 4
    UNIT_SPECIAL = 5
    CITY_PRODUCE = 6
    CITY_BUY = 7
    CITY_MANAGE = 8
    TECH_RESEARCH = 9
    DIPLOMACY = 10
    GOVERNMENT = 11


class CivRealmActionHandler:
    """Handles action encoding/decoding for CivRealm.

    Maps between discrete action indices and CivRealm action dictionaries.
    Supports hierarchical action structure:
    - Level 0: Action type (unit, city, tech, diplomacy, etc.)
    - Level 1: Entity selection (which unit/city)
    - Level 2: Specific action (move direction, build type, etc.)
    - Level 3: Target (position, item, etc.)

    Attributes:
        config: CivRealm configuration.
        action_size: Total number of discrete actions.
    """

    def __init__(self, config: CivRealmConfig) -> None:
        """Initialize action handler.

        Args:
            config: CivRealm configuration.
        """
        self.config = config
        self.action_size = config.max_actions_per_turn
        self._action_cache: dict[int, dict] = {}
        self._reverse_cache: dict[str, int] = {}

        logger.debug(f"CivRealmActionHandler initialized with {self.action_size} actions")

    def build_action_mapping(self, available_actions: dict) -> None:
        """Build mapping from action indices to CivRealm actions.

        Args:
            available_actions: Dictionary of available actions from CivRealm.
        """
        self._action_cache.clear()
        self._reverse_cache.clear()

        # Map each available action to an index
        for i, (action_key, action_info) in enumerate(available_actions.items()):
            if i >= self.action_size - 1:  # Reserve last index for end_turn
                logger.warning(
                    f"Too many actions ({len(available_actions)}), "
                    f"truncating to {self.action_size - 1}"
                )
                break

            self._action_cache[i] = {
                "key": action_key,
                "info": action_info,
            }
            self._reverse_cache[str(action_key)] = i

        # Always add end_turn as last action
        self._action_cache[self.action_size - 1] = {
            "key": "end_turn",
            "info": {"type": "end_turn"},
        }
        self._reverse_cache["end_turn"] = self.action_size - 1

    def decode_action(self, action_idx: int, available_actions: dict) -> dict:
        """Decode discrete action index to CivRealm action.

        Args:
            action_idx: Discrete action index.
            available_actions: Dictionary of currently available actions.

        Returns:
            CivRealm action dictionary.
        """
        # Rebuild mapping for current available actions
        self.build_action_mapping(available_actions)

        # Handle end_turn
        if action_idx == self.action_size - 1:
            return {"type": "end_turn"}

        # Look up cached action
        if action_idx in self._action_cache:
            cached = self._action_cache[action_idx]
            return {
                "action_key": cached["key"],
                **cached["info"],
            }

        # Fallback to end_turn if action not found
        logger.warning(f"Action index {action_idx} not found, falling back to end_turn")
        return {"type": "end_turn"}

    def encode_action(self, action: dict, available_actions: dict) -> int:
        """Encode CivRealm action to discrete index.

        Args:
            action: CivRealm action dictionary.
            available_actions: Dictionary of available actions.

        Returns:
            Discrete action index.
        """
        # Rebuild mapping
        self.build_action_mapping(available_actions)

        action_key = action.get("action_key", action.get("type", "end_turn"))

        if str(action_key) in self._reverse_cache:
            return self._reverse_cache[str(action_key)]

        # Default to end_turn
        return self.action_size - 1

    def get_action_mask(self, available_actions: dict) -> np.ndarray:
        """Generate action mask from available actions.

        Args:
            available_actions: Dictionary of available actions.

        Returns:
            Boolean mask of shape (action_size,).
        """
        mask = np.zeros(self.action_size, dtype=np.float32)

        self.build_action_mapping(available_actions)

        for idx in self._action_cache.keys():
            mask[idx] = 1.0

        # Ensure at least end_turn is valid
        mask[self.action_size - 1] = 1.0

        return mask

    def sample_valid_action(self, mask: np.ndarray, rng: np.random.Generator | None = None) -> int:
        """Sample a random valid action.

        Args:
            mask: Action validity mask.
            rng: Random number generator.

        Returns:
            Sampled action index.
        """
        if rng is None:
            rng = np.random.default_rng()

        valid_indices = np.where(mask > 0)[0]

        if len(valid_indices) == 0:
            return self.action_size - 1  # end_turn

        return int(rng.choice(valid_indices))

    @property
    def action_space(self) -> spaces.Discrete:
        """Get Gymnasium action space."""
        return spaces.Discrete(self.action_size)


# ============================================================================
# Curriculum for Mini-Games
# ============================================================================


@dataclass
class MiniGameTask:
    """Configuration for a CivRealm mini-game task.

    Attributes:
        name: Task identifier.
        display_name: Human-readable name.
        description: Task description.
        difficulty: Relative difficulty (0.0-1.0).
        max_turns: Maximum turns for this task.
        success_threshold: Win rate threshold to master this task.
        prerequisites: List of task names that must be mastered first.
    """

    name: str
    display_name: str
    description: str
    difficulty: float = 0.5
    max_turns: int = 100
    success_threshold: float = 0.6
    prerequisites: list[str] = field(default_factory=list)


class CivRealmCurriculum:
    """Manages curriculum progression through CivRealm mini-games.

    Provides structured progression from simple tasks to complex scenarios,
    tracking agent performance and determining when to advance.

    Attributes:
        tasks: Dictionary of available mini-game tasks.
        task_order: Ordered list of task names by difficulty.
        success_rates: Tracked success rates per task.
    """

    # Default mini-game tasks in order of difficulty
    DEFAULT_TASKS: ClassVar[list[MiniGameTask]] = [
        MiniGameTask(
            name="development_build_city",
            display_name="Build Your First City",
            description="Learn to found and develop a city",
            difficulty=0.1,
            max_turns=50,
            success_threshold=0.7,
        ),
        MiniGameTask(
            name="exploration",
            display_name="Explore the Map",
            description="Scout and reveal the map using units",
            difficulty=0.2,
            max_turns=100,
            success_threshold=0.65,
            prerequisites=["development_build_city"],
        ),
        MiniGameTask(
            name="battle_ancient",
            display_name="Ancient Combat",
            description="Win battles using ancient era units",
            difficulty=0.3,
            max_turns=50,
            success_threshold=0.6,
            prerequisites=["development_build_city"],
        ),
        MiniGameTask(
            name="trade",
            display_name="Establish Trade",
            description="Create profitable trade routes",
            difficulty=0.4,
            max_turns=150,
            success_threshold=0.55,
            prerequisites=["development_build_city", "exploration"],
        ),
        MiniGameTask(
            name="battle_medieval",
            display_name="Medieval Combat",
            description="Win battles using medieval era units",
            difficulty=0.5,
            max_turns=75,
            success_threshold=0.55,
            prerequisites=["battle_ancient"],
        ),
        MiniGameTask(
            name="diplomacy",
            display_name="Diplomatic Relations",
            description="Navigate diplomatic interactions",
            difficulty=0.6,
            max_turns=200,
            success_threshold=0.5,
            prerequisites=["trade"],
        ),
        MiniGameTask(
            name="battle_modern",
            display_name="Modern Combat",
            description="Win battles using modern era units",
            difficulty=0.7,
            max_turns=100,
            success_threshold=0.5,
            prerequisites=["battle_medieval"],
        ),
        MiniGameTask(
            name="battle_info_era",
            display_name="Information Age Combat",
            description="Win battles using information era units",
            difficulty=0.8,
            max_turns=100,
            success_threshold=0.45,
            prerequisites=["battle_modern"],
        ),
        MiniGameTask(
            name="fullgame",
            display_name="Full Game",
            description="Play a complete Freeciv game",
            difficulty=1.0,
            max_turns=500,
            success_threshold=0.4,
            prerequisites=["diplomacy", "battle_info_era"],
        ),
    ]

    def __init__(self, custom_tasks: list[MiniGameTask] | None = None) -> None:
        """Initialize curriculum.

        Args:
            custom_tasks: Optional custom task list. Uses defaults if None.
        """
        self.tasks: dict[str, MiniGameTask] = {}
        self.task_order: list[str] = []
        self.success_rates: dict[str, list[bool]] = {}
        self.games_played: dict[str, int] = {}
        self.current_task_idx: int = 0

        # Initialize with default or custom tasks
        task_list = custom_tasks or self.DEFAULT_TASKS

        for task in task_list:
            self.tasks[task.name] = task
            self.success_rates[task.name] = []
            self.games_played[task.name] = 0

        # Sort tasks by difficulty
        self.task_order = sorted(self.tasks.keys(), key=lambda t: self.tasks[t].difficulty)

        logger.info(f"CivRealmCurriculum initialized with {len(self.tasks)} tasks")
        logger.debug(f"Task order: {self.task_order}")

    def get_current_task(self) -> MiniGameTask:
        """Get current curriculum task.

        Returns:
            Current MiniGameTask.
        """
        if self.current_task_idx >= len(self.task_order):
            # At final task
            return self.tasks[self.task_order[-1]]

        return self.tasks[self.task_order[self.current_task_idx]]

    def get_next_task(self, agent_skill_level: float = 0.5) -> str:
        """Get appropriate next task based on agent skill.

        Args:
            agent_skill_level: Agent's estimated skill level (0.0-1.0).

        Returns:
            Name of recommended task.
        """
        # Check prerequisites and mastery for each task
        available_tasks = []

        for task_name in self.task_order:
            task = self.tasks[task_name]

            # Check prerequisites
            prereqs_met = all(self._is_task_mastered(prereq) for prereq in task.prerequisites)

            if prereqs_met:
                # Check if appropriate difficulty for skill level
                if task.difficulty <= agent_skill_level + 0.2:
                    available_tasks.append(task_name)

        if not available_tasks:
            # Return first task if nothing available
            return self.task_order[0]

        # Prefer unmastered tasks at appropriate difficulty
        for task_name in available_tasks:
            if not self._is_task_mastered(task_name):
                return task_name

        # All available tasks mastered, try next difficulty level
        for task_name in self.task_order:
            if task_name not in available_tasks:
                task = self.tasks[task_name]
                prereqs_met = all(self._is_task_mastered(prereq) for prereq in task.prerequisites)
                if prereqs_met:
                    return task_name

        # Default to highest difficulty available task
        return available_tasks[-1]

    def _is_task_mastered(self, task_name: str) -> bool:
        """Check if a task has been mastered.

        Args:
            task_name: Name of task to check.

        Returns:
            True if task is mastered.
        """
        if task_name not in self.tasks:
            return False

        task = self.tasks[task_name]
        results = self.success_rates.get(task_name, [])

        # Need minimum games played
        if len(results) < 10:
            return False

        # Check recent success rate
        recent_results = results[-50:]  # Last 50 games
        win_rate = sum(recent_results) / len(recent_results)

        return win_rate >= task.success_threshold

    def record_result(self, task_name: str, success: bool) -> dict[str, Any]:
        """Record result of a task attempt.

        Args:
            task_name: Name of completed task.
            success: Whether the task was completed successfully.

        Returns:
            Dictionary with progress information.
        """
        if task_name not in self.tasks:
            logger.warning(f"Unknown task: {task_name}")
            return {"error": f"Unknown task: {task_name}"}

        self.success_rates[task_name].append(success)
        self.games_played[task_name] = self.games_played.get(task_name, 0) + 1

        # Trim history to last 100 games
        if len(self.success_rates[task_name]) > 100:
            self.success_rates[task_name] = self.success_rates[task_name][-100:]

        # Check for progression
        was_mastered = self._is_task_mastered(task_name)

        task = self.tasks[task_name]
        recent = self.success_rates[task_name][-50:] if self.success_rates[task_name] else []
        win_rate = sum(recent) / len(recent) if recent else 0.0

        result = {
            "task": task_name,
            "success": success,
            "games_played": self.games_played[task_name],
            "win_rate": win_rate,
            "threshold": task.success_threshold,
            "mastered": was_mastered,
            "to_mastery": max(0, task.success_threshold - win_rate),
        }

        if was_mastered:
            next_task = self.get_next_task(task.difficulty + 0.1)
            result["suggested_next"] = next_task

        return result

    def get_progress(self) -> dict[str, Any]:
        """Get overall curriculum progress.

        Returns:
            Progress summary dictionary.
        """
        mastered_tasks = [t for t in self.task_order if self._is_task_mastered(t)]
        total_games = sum(self.games_played.values())

        task_details = {}
        for task_name in self.task_order:
            task = self.tasks[task_name]
            recent = self.success_rates.get(task_name, [])[-50:]
            win_rate = sum(recent) / len(recent) if recent else 0.0

            task_details[task_name] = {
                "difficulty": task.difficulty,
                "games_played": self.games_played.get(task_name, 0),
                "win_rate": win_rate,
                "mastered": self._is_task_mastered(task_name),
            }

        return {
            "total_tasks": len(self.tasks),
            "mastered_tasks": len(mastered_tasks),
            "mastered_names": mastered_tasks,
            "total_games": total_games,
            "current_task": self.get_current_task().name,
            "task_details": task_details,
        }

    def reset(self) -> None:
        """Reset curriculum progress."""
        for task_name in self.tasks:
            self.success_rates[task_name] = []
            self.games_played[task_name] = 0

        self.current_task_idx = 0
        logger.info("Curriculum progress reset")


# ============================================================================
# Mock Environment (when CivRealm not installed)
# ============================================================================


class MockCivRealmEnv:
    """Mock CivRealm environment for testing when CivRealm is not installed.

    Provides a simplified simulation that mimics CivRealm's interface.
    """

    def __init__(self, config: CivRealmConfig) -> None:
        """Initialize mock environment.

        Args:
            config: CivRealm configuration.
        """
        self.config = config
        self._rng = np.random.default_rng()
        self._turn = 0
        self._done = False
        self._player_id = 0

        logger.info("Using MockCivRealmEnv (CivRealm not installed)")

    def reset(self) -> dict:
        """Reset mock environment.

        Returns:
            Mock observation dictionary.
        """
        self._turn = 0
        self._done = False
        return self._generate_mock_observation()

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """Execute mock action.

        Args:
            action: Action dictionary.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        self._turn += 1

        # Random reward
        reward = self._rng.uniform(-0.1, 0.1)

        # Check for termination
        if self._turn >= self.config.max_turns:
            self._done = True
            reward += self._rng.uniform(-0.5, 0.5)  # Final reward

        obs = self._generate_mock_observation()
        info = {
            "turn": self._turn,
            "task": self.config.task,
        }

        return obs, reward, self._done, info

    def close(self) -> None:
        """Close mock environment."""
        pass

    def _generate_mock_observation(self) -> dict:
        """Generate mock observation.

        Returns:
            Mock observation dictionary.
        """
        size = self.config.spatial_size

        # Generate mock map
        terrain = {}
        visibility = {}
        for x in range(size):
            for y in range(size):
                terrain[(x, y)] = {
                    "terrain": self._rng.integers(0, 15),
                    "food": self._rng.integers(0, 5),
                    "production": self._rng.integers(0, 5),
                    "trade": self._rng.integers(0, 3),
                }
                visibility[(x, y)] = self._rng.choice(["visible", "seen", "unexplored"])

        # Generate mock units
        num_units = self._rng.integers(2, 10)
        units = {}
        for i in range(num_units):
            units[f"unit_{i}"] = {
                "owner": 0 if i < num_units // 2 else 1,
                "position": {"x": self._rng.integers(0, size), "y": self._rng.integers(0, size)},
                "type_name": self._rng.choice(["Warrior", "Settler", "Archer", "Horseman"]),
                "hp": self._rng.integers(50, 100),
                "moves_left": self._rng.integers(0, 3),
            }

        # Generate mock cities
        num_cities = self._rng.integers(1, 5)
        cities = {}
        for i in range(num_cities):
            cities[f"city_{i}"] = {
                "owner": 0 if i < num_cities // 2 else 1,
                "position": {"x": self._rng.integers(0, size), "y": self._rng.integers(0, size)},
                "population": self._rng.integers(1, 20),
                "food_surplus": self._rng.integers(-5, 10),
                "production": self._rng.integers(0, 50),
            }

        # Generate mock available actions
        available_actions = {}
        num_actions = self._rng.integers(5, min(50, self.config.max_actions_per_turn))
        for i in range(num_actions):
            available_actions[f"action_{i}"] = {
                "type": self._rng.choice(["unit_move", "unit_build", "city_produce"]),
                "unit_id": f"unit_{self._rng.integers(0, max(1, num_units // 2))}",  # type: ignore[call-overload]
            }

        return {
            "turn": self._turn,
            "year": -4000 + self._turn * 50,
            "player_id": self._player_id,
            "map": {
                "terrain": terrain,
                "visibility": visibility,
            },
            "units": units,
            "cities": cities,
            "player": {
                "gold": self._rng.integers(0, 1000),
                "gold_income": self._rng.integers(-10, 50),
                "science": self._rng.integers(0, 100),
            },
            "diplomacy": {},
            "score": {
                "total": self._rng.integers(0, 500),
            },
            "available_actions": available_actions,
            "can_end_turn": True,
        }


# ============================================================================
# Main Wrapper
# ============================================================================


class CivRealmWrapper(gym.Env):
    """Gymnasium-compatible wrapper for CivRealm environment.

    Provides a standard Gymnasium interface for interacting with CivRealm,
    including observation encoding, action handling, and curriculum support.

    Attributes:
        config: CivRealm configuration.
        encoder: Observation encoder.
        action_handler: Action encoder/decoder.
        curriculum: Optional curriculum manager.
        observation_space: Gymnasium observation space.
        action_space: Gymnasium action space.

    Example:
        >>> config = CivRealmConfig(task="development_build_city")
        >>> env = CivRealmWrapper(config)
        >>> obs, info = env.reset()
        >>> print(obs["spatial"].shape)  # (51, 32, 32)
        >>> print(obs["vector"].shape)   # (256,)
        >>> action = 0  # First valid action
        >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata: ClassVar[dict[str, list[str | None]]] = {"render_modes": ["human", "ansi", None]}  # type: ignore[misc]

    def __init__(
        self,
        config: CivRealmConfig | None = None,
        curriculum: CivRealmCurriculum | None = None,
    ) -> None:
        """Initialize CivRealm wrapper.

        Args:
            config: CivRealm configuration. Uses defaults if None.
            curriculum: Optional curriculum manager for task progression.
        """
        super().__init__()

        self.config = config or CivRealmConfig()
        self.config.validate()

        # Initialize components
        self.encoder = CivRealmEncoder(self.config)
        self.action_handler = CivRealmActionHandler(self.config)
        self.curriculum = curriculum

        # Gymnasium spaces
        self.observation_space = self.encoder.observation_space
        self.action_space = self.action_handler.action_space

        # Internal state
        self._env: Any = None
        self._current_obs: dict = {}
        self._turn: int = 0
        self._done: bool = False
        self._truncated: bool = False
        self._episode_reward: float = 0.0

        # Initialize underlying environment
        self._initialize_env()

        logger.info(
            f"CivRealmWrapper initialized: task={self.config.task}, "
            f"spatial_size={self.config.spatial_size}, "
            f"civrealm_available={CIVREALM_AVAILABLE}"
        )

    def _initialize_env(self) -> None:
        """Initialize the underlying CivRealm or mock environment."""
        if self.config.use_mock or not CIVREALM_AVAILABLE:
            self._env = MockCivRealmEnv(self.config)
            logger.info("Initialized mock CivRealm environment")
        else:
            try:
                # Initialize real CivRealm environment
                self._env = civrealm.make(
                    self.config.task,
                    client_port=self.config.client_port,
                )
                logger.info(f"Initialized CivRealm environment: {self.config.task}")
            except Exception as e:
                logger.warning(f"Failed to initialize CivRealm: {e}. Using mock.")
                self._env = MockCivRealmEnv(self.config)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (encoded_observation, info_dict).
        """
        # Handle curriculum task selection
        if self.curriculum and options and options.get("use_curriculum", True):
            task = self.curriculum.get_next_task()
            if task != self.config.task:
                self.config.task = task
                self._initialize_env()

        # Reset underlying environment
        raw_obs = self._env.reset()

        # Reset internal state
        self._turn = 0
        self._done = False
        self._truncated = False
        self._episode_reward = 0.0
        self._current_obs = raw_obs

        # Encode observation
        encoded_obs = self.encoder.encode_observation(raw_obs)

        # Build action mapping for current available actions
        available_actions = raw_obs.get("available_actions", {})
        self.action_handler.build_action_mapping(available_actions)

        info = {
            "turn": self._turn,
            "task": self.config.task,
            "raw_obs": raw_obs if options and options.get("include_raw", False) else None,
        }

        logger.debug(f"Environment reset: task={self.config.task}")

        return encoded_obs, info

    def step(
        self,
        action: int | np.ndarray,
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute action and return results.

        Args:
            action: Discrete action index or array.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self._done or self._truncated:
            logger.warning("Episode already done, returning current state")
            encoded_obs = self.encoder.encode_observation(self._current_obs)
            return encoded_obs, 0.0, self._done, self._truncated, {"turn": self._turn}

        # Convert action to int if needed
        action_idx = int(action) if isinstance(action, np.ndarray) else action

        # Decode action to CivRealm format
        available_actions = self._current_obs.get("available_actions", {})
        civrealm_action = self.action_handler.decode_action(action_idx, available_actions)

        # Execute action
        try:
            raw_obs, reward, done, info = self._env.step(civrealm_action)
        except Exception as e:
            logger.error(f"Error executing action: {e}")
            # Return current state with penalty
            encoded_obs = self.encoder.encode_observation(self._current_obs)
            return encoded_obs, -0.1, False, False, {"error": str(e)}

        # Update internal state
        self._current_obs = raw_obs
        self._turn = raw_obs.get("turn", self._turn + 1)
        self._episode_reward += reward

        # Check termination conditions
        self._done = done
        self._truncated = self._turn >= self.config.max_turns

        # Encode observation
        encoded_obs = self.encoder.encode_observation(raw_obs)

        # Update action mapping
        available_actions = raw_obs.get("available_actions", {})
        self.action_handler.build_action_mapping(available_actions)

        # Record curriculum result if episode done
        if self._done or self._truncated:
            if self.curriculum:
                success = reward > 0 or info.get("win", False)
                self.curriculum.record_result(self.config.task, success)

        step_info = {
            "turn": self._turn,
            "task": self.config.task,
            "episode_reward": self._episode_reward,
            **info,
        }

        logger.debug(
            f"Step {self._turn}: action={action_idx}, reward={reward:.3f}, "
            f"done={self._done}, truncated={self._truncated}"
        )

        return encoded_obs, float(reward), self._done, self._truncated, step_info

    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions.

        Returns:
            Boolean array of shape (action_space_size,).
        """
        available_actions = self._current_obs.get("available_actions", {})
        return self.action_handler.get_action_mask(available_actions)

    def render(self) -> str | None:
        """Render the environment.

        Returns:
            String representation if render_mode is 'ansi', else None.
        """
        if self.config.render_mode == "ansi":
            return self._render_ansi()
        elif self.config.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _render_ansi(self) -> str:
        """Render environment as ASCII string.

        Returns:
            ASCII representation of game state.
        """
        lines = []
        lines.append(f"=== CivRealm: {self.config.task} ===")
        lines.append(f"Turn: {self._turn} / {self.config.max_turns}")

        # Unit summary
        units = self._current_obs.get("units", {})
        player_id = self._current_obs.get("player_id", 0)
        own_units = sum(1 for u in units.values() if u.get("owner") == player_id)
        enemy_units = len(units) - own_units
        lines.append(f"Units: {own_units} own, {enemy_units} enemy")

        # City summary
        cities = self._current_obs.get("cities", {})
        own_cities = sum(1 for c in cities.values() if c.get("owner") == player_id)
        enemy_cities = len(cities) - own_cities
        lines.append(f"Cities: {own_cities} own, {enemy_cities} enemy")

        # Resources
        player = self._current_obs.get("player", {})
        lines.append(f"Gold: {player.get('gold', 0)}")

        # Available actions
        available = self._current_obs.get("available_actions", {})
        lines.append(f"Available actions: {len(available)}")

        lines.append(f"Episode reward: {self._episode_reward:.3f}")
        lines.append("=" * 30)

        return "\n".join(lines)

    def close(self) -> None:
        """Clean up environment resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception as e:
                logger.warning(f"Error closing CivRealm environment: {e}")

        self._env = None
        logger.info("CivRealmWrapper closed")

    def clone(self) -> CivRealmWrapper:
        """Create a new environment instance with the same configuration.

        IMPORTANT LIMITATION:
            CivRealm runs on a Freeciv server, so true state cloning is not possible.
            This method creates a FRESH environment instance, NOT a clone of the
            current game state. The new instance will:
            - Start a new game with the same configuration
            - Have its own independent game state
            - NOT preserve any game progress from the original

        This limitation affects MCTS algorithms that rely on environment cloning
        for tree search. Use hidden-state MCTS (MuZero-style) instead, which uses
        a learned dynamics model rather than environment cloning.

        For algorithms requiring true state cloning, consider:
        - Using MicroCiv environment (supports full cloning)
        - Using MuZero with learned dynamics (no cloning needed)

        Returns:
            New CivRealmWrapper instance with same config (fresh game state).
        """
        logger.warning(
            "CivRealm does not support full environment cloning due to server-based "
            "architecture. Creating new instance with same config - game state will "
            "NOT be preserved. Use MuZero-style MCTS with learned dynamics instead."
        )
        return CivRealmWrapper(self.config, self.curriculum)

    @staticmethod
    def is_available() -> bool:
        """Check if CivRealm is installed and available.

        Returns:
            True if CivRealm can be imported.
        """
        return CIVREALM_AVAILABLE


# ============================================================================
# Factory Functions
# ============================================================================


def create_civrealm_env(
    task: str = "development_build_city",
    difficulty: int = 1,
    use_curriculum: bool = False,
    **kwargs: Any,
) -> CivRealmWrapper:
    """Factory function to create CivRealm environment.

    Args:
        task: Mini-game task name.
        difficulty: Game difficulty (1-10).
        use_curriculum: Whether to enable curriculum learning.
        **kwargs: Additional configuration parameters.

    Returns:
        Configured CivRealmWrapper instance.

    Example:
        >>> env = create_civrealm_env(task="battle_ancient", difficulty=2)
        >>> obs, info = env.reset()
    """
    config = CivRealmConfig(
        task=task,
        difficulty=difficulty,
        **kwargs,
    )

    curriculum = CivRealmCurriculum() if use_curriculum else None

    return CivRealmWrapper(config, curriculum)


def make(
    task: str = "development_build_city",
    **kwargs: Any,
) -> CivRealmWrapper:
    """Create CivRealm environment (Gymnasium-style make function).

    Args:
        task: Mini-game task name.
        **kwargs: Configuration parameters.

    Returns:
        CivRealmWrapper instance.
    """
    return create_civrealm_env(task=task, **kwargs)


# Module exports
__all__ = [
    "CIVREALM_AVAILABLE",
    "ActionType",
    "CivRealmActionHandler",
    "CivRealmConfig",
    "CivRealmCurriculum",
    "CivRealmEncoder",
    "CivRealmWrapper",
    "MiniGameTask",
    "TerrainType",
    "UnitCategory",
    "create_civrealm_env",
    "make",
]
