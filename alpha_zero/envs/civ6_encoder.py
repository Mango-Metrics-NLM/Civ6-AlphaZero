"""State encoder for Civilization VI.

Converts raw game state into tensor representation for neural network input.

Spatial tensor: (84, H, W) - encodes terrain, units, cities, fog, etc.
Vector tensor: (512,) - encodes turn, resources, techs, diplomacy, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class EncoderConfig:
    """Configuration for state encoder."""

    board_size: int = 16
    spatial_channels: int = 84
    vector_dim: int = 512
    num_players: int = 2
    max_units: int = 100
    max_cities: int = 20


class Civ6StateEncoder:
    """Encodes Civ 6 game state into tensors.

    Full encoding uses 84 spatial channels as specified:
    - Channels 0-6: Terrain types (one-hot)
    - Channels 7-14: Terrain features
    - Channels 15-26: Resources
    - Channels 27-38: Own units
    - Channels 39-50: Enemy units
    - Channels 51-54: Own cities
    - Channels 55-58: Enemy cities
    - Channels 59-62: Borders
    - Channels 63-65: Fog of war
    - Channels 66-70: Yields
    - Channels 71-78: Improvements
    - Channels 79-83: Movement/selection
    """

    # Channel allocation constants
    TERRAIN_START = 0
    TERRAIN_END = 7
    FEATURES_START = 7
    FEATURES_END = 15
    RESOURCES_START = 15
    RESOURCES_END = 27
    OWN_UNITS_START = 27
    OWN_UNITS_END = 39
    ENEMY_UNITS_START = 39
    ENEMY_UNITS_END = 51
    OWN_CITIES_START = 51
    OWN_CITIES_END = 55
    ENEMY_CITIES_START = 55
    ENEMY_CITIES_END = 59
    BORDERS_START = 59
    BORDERS_END = 63
    FOG_START = 63
    FOG_END = 66
    YIELDS_START = 66
    YIELDS_END = 71
    IMPROVEMENTS_START = 71
    IMPROVEMENTS_END = 79
    MOVEMENT_START = 79
    MOVEMENT_END = 84

    TOTAL_CHANNELS = 84

    # Terrain type mapping
    TERRAIN_TYPES = {
        "plains": 0,
        "grassland": 1,
        "desert": 2,
        "tundra": 3,
        "snow": 4,
        "coast": 5,
        "ocean": 6,
    }

    # Feature mapping
    FEATURE_TYPES = {
        "forest": 0,
        "jungle": 1,
        "marsh": 2,
        "floodplains": 3,
        "hills": 4,
        "mountains": 5,
        "river": 6,
        "ice": 7,
    }

    # Unit type mapping
    UNIT_TYPES = {
        "melee": 0,
        "warrior": 0,
        "ranged": 1,
        "archer": 1,
        "cavalry": 2,
        "siege": 3,
        "naval_melee": 4,
        "naval_ranged": 5,
        "air": 6,
        "support": 7,
        "civilian": 8,
        "settler": 8,
        "religious": 9,
        "great_person": 10,
        "spy": 11,
    }

    def __init__(self, config: EncoderConfig | None = None):
        """Initialize encoder.

        Args:
            config: Encoder configuration. Uses defaults if None.
        """
        self.config = config or EncoderConfig()

    def encode(self, game_state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode game state to tensors.

        Args:
            game_state: Raw game state dictionary.

        Returns:
            Tuple of (spatial_tensor, vector_tensor).
        """
        spatial = self._encode_spatial(game_state)
        vector = self._encode_vector(game_state)
        return torch.from_numpy(spatial), torch.from_numpy(vector)

    def _encode_spatial(self, game_state: dict) -> np.ndarray:
        """Encode spatial features.

        Args:
            game_state: Game state dictionary.

        Returns:
            Spatial tensor of shape (C, H, W).
        """
        size = self.config.board_size
        spatial = np.zeros((self.TOTAL_CHANNELS, size, size), dtype=np.float32)

        current_player = game_state.get("current_player", 0)
        terrain = game_state.get("terrain", {})
        units = game_state.get("units", [])
        cities = game_state.get("cities", [])
        visibility = game_state.get("visibility", {})

        # Encode terrain
        for y in range(size):
            for x in range(size):
                tile = terrain.get((x, y), {})
                terrain_type = tile.get("type", "plains")
                if terrain_type in self.TERRAIN_TYPES:
                    channel = self.TERRAIN_START + self.TERRAIN_TYPES[terrain_type]
                    spatial[channel, y, x] = 1.0

                # Features
                features = tile.get("features", [])
                for feature in features:
                    if feature in self.FEATURE_TYPES:
                        channel = self.FEATURES_START + self.FEATURE_TYPES[feature]
                        spatial[channel, y, x] = 1.0

                # Yields
                yields = tile.get("yields", {})
                spatial[self.YIELDS_START, y, x] = yields.get("food", 0) / 10.0
                spatial[self.YIELDS_START + 1, y, x] = yields.get("production", 0) / 10.0
                spatial[self.YIELDS_START + 2, y, x] = yields.get("gold", 0) / 10.0
                spatial[self.YIELDS_START + 3, y, x] = yields.get("science", 0) / 10.0
                spatial[self.YIELDS_START + 4, y, x] = yields.get("culture", 0) / 10.0

        # Encode units
        for unit in units:
            pos = unit.get("position", (0, 0))
            x, y = pos
            if not (0 <= x < size and 0 <= y < size):
                continue

            unit_type = unit.get("type", "melee")
            type_idx = self.UNIT_TYPES.get(unit_type, 0)
            health = unit.get("health", 100) / 100.0
            strength = unit.get("strength", 20) / 100.0
            value = health * max(strength, 0.1)

            player = unit.get("player", 0)
            if player == current_player:
                channel = self.OWN_UNITS_START + type_idx
            else:
                channel = self.ENEMY_UNITS_START + type_idx

            if channel < self.TOTAL_CHANNELS:
                spatial[channel, y, x] = max(spatial[channel, y, x], value)

        # Encode cities
        for city in cities:
            pos = city.get("position", (0, 0))
            x, y = pos
            if not (0 <= x < size and 0 <= y < size):
                continue

            player = city.get("player", 0)
            if player == current_player:
                base = self.OWN_CITIES_START
            else:
                base = self.ENEMY_CITIES_START

            spatial[base, y, x] = 1.0  # Presence
            spatial[base + 1, y, x] = city.get("population", 1) / 30.0
            spatial[base + 2, y, x] = city.get("production", 0) / 100.0
            spatial[base + 3, y, x] = city.get("defense", 0) / 100.0

        # Fog of war
        for y in range(size):
            for x in range(size):
                vis_state = visibility.get((x, y), "visible")
                if vis_state == "visible":
                    spatial[self.FOG_START, y, x] = 1.0
                elif vis_state == "seen":
                    spatial[self.FOG_START + 1, y, x] = 1.0
                else:
                    spatial[self.FOG_START + 2, y, x] = 1.0

        # Constant channel (bias term)
        spatial[self.MOVEMENT_END - 1, :, :] = 1.0

        return spatial

    def _encode_vector(self, game_state: dict) -> np.ndarray:
        """Encode non-spatial features.

        Args:
            game_state: Game state dictionary.

        Returns:
            Vector tensor of shape (vector_dim,).
        """
        vector = np.zeros(self.config.vector_dim, dtype=np.float32)

        # Turn/Phase (0-9)
        turn = game_state.get("turn", 0)
        vector[0] = turn / 500.0

        # Era (10-17)
        era = game_state.get("era", 0)
        if 0 <= era < 8:
            vector[10 + era] = 1.0

        # Resources (20-69)
        resources = game_state.get("resources", {})
        for i, (name, count) in enumerate(resources.items()):
            if 20 + i < 70:
                vector[20 + i] = count / 20.0

        # Technologies (70-169)
        techs = game_state.get("technologies", [])
        for tech_id in techs:
            if 0 <= tech_id < 100:
                vector[70 + tech_id] = 1.0

        # Civics (180-279)
        civics = game_state.get("civics", [])
        for civic_id in civics:
            if 0 <= civic_id < 100:
                vector[180 + civic_id] = 1.0

        # Economy (300-319)
        vector[300] = game_state.get("gold", 0) / 1000.0
        vector[301] = game_state.get("gold_per_turn", 0) / 100.0
        vector[302] = game_state.get("science_per_turn", 0) / 100.0
        vector[303] = game_state.get("culture_per_turn", 0) / 100.0
        vector[304] = game_state.get("faith_per_turn", 0) / 100.0

        # Military (320-349)
        vector[320] = game_state.get("military_strength", 0) / 500.0
        vector[321] = len(game_state.get("units", [])) / 50.0

        # Cities (350-399)
        cities = game_state.get("cities", [])
        current_player = game_state.get("current_player", 0)
        own_cities = [c for c in cities if c.get("player") == current_player]
        vector[350] = len(own_cities) / 20.0
        if own_cities:
            vector[351] = sum(c.get("population", 0) for c in own_cities) / len(own_cities) / 30.0

        # Victory progress (500-511)
        victory = game_state.get("victory_progress", {})
        vector[500] = victory.get("domination", 0)
        vector[501] = victory.get("science", 0)
        vector[502] = victory.get("culture", 0)
        vector[503] = victory.get("religious", 0)
        vector[504] = victory.get("diplomatic", 0)

        return vector


class MicroCivEncoder:
    """Simplified encoder for MicroCiv training environment.

    Uses fewer channels for faster training:
    - Channels 0-2: Terrain (plains, hills, mountain)
    - Channels 3-5: Own units (warrior, settler, archer)
    - Channels 6-8: Enemy units
    - Channel 9: Own cities
    - Channel 10: Enemy cities
    - Channel 11: Fog of war (visible)
    - Channel 12: Constant (bias)
    """

    TOTAL_CHANNELS = 13

    def __init__(self, board_size: int = 16):
        """Initialize encoder.

        Args:
            board_size: Size of the game board.
        """
        self.board_size = board_size

    def encode(self, game_state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode game state.

        Args:
            game_state: Game state dictionary.

        Returns:
            Tuple of (spatial_tensor, vector_tensor).
        """
        spatial = self._encode_spatial(game_state)
        vector = self._encode_vector(game_state)
        return torch.from_numpy(spatial), torch.from_numpy(vector)

    def _encode_spatial(self, game_state: dict) -> np.ndarray:
        """Encode spatial features."""
        size = self.board_size
        spatial = np.zeros((self.TOTAL_CHANNELS, size, size), dtype=np.float32)

        current_player = game_state.get("current_player", 0)
        terrain = game_state.get("terrain", {})
        units = game_state.get("units", [])
        cities = game_state.get("cities", [])

        # Terrain
        terrain_map = {"plains": 0, "hills": 1, "mountain": 2}
        for y in range(size):
            for x in range(size):
                tile = terrain.get((x, y), {})
                terrain_type = tile.get("type", "plains")
                if terrain_type in terrain_map:
                    spatial[terrain_map[terrain_type], y, x] = 1.0

        # Units
        unit_map = {"warrior": 0, "settler": 1, "archer": 2}
        for unit in units:
            pos = unit.get("position", (0, 0))
            x, y = pos
            if not (0 <= x < size and 0 <= y < size):
                continue

            unit_type = unit.get("type", "warrior")
            type_idx = unit_map.get(unit_type, 0)
            health = unit.get("health", 100) / 100.0

            player = unit.get("player", 0)
            if player == current_player:
                spatial[3 + type_idx, y, x] = health
            else:
                spatial[6 + type_idx, y, x] = health

        # Cities
        for city in cities:
            pos = city.get("position", (0, 0))
            x, y = pos
            if not (0 <= x < size and 0 <= y < size):
                continue

            player = city.get("player", 0)
            if player == current_player:
                spatial[9, y, x] = 1.0
            else:
                spatial[10, y, x] = 1.0

        # Fog of war (all visible for MicroCiv)
        spatial[11, :, :] = 1.0

        # Constant
        spatial[12, :, :] = 1.0

        return spatial

    def _encode_vector(self, game_state: dict) -> np.ndarray:
        """Encode vector features."""
        vector = np.zeros(64, dtype=np.float32)

        current_player = game_state.get("current_player", 0)
        units = game_state.get("units", [])
        cities = game_state.get("cities", [])

        # Turn
        vector[0] = game_state.get("turn", 0) / 200.0

        # Current player
        vector[1] = current_player

        # Unit counts
        own_units = [u for u in units if u.get("player") == current_player]
        enemy_units = [u for u in units if u.get("player") != current_player]
        vector[2] = len(own_units) / 20.0
        vector[3] = len(enemy_units) / 20.0

        # City counts
        own_cities = [c for c in cities if c.get("player") == current_player]
        enemy_cities = [c for c in cities if c.get("player") != current_player]
        vector[4] = len(own_cities) / 5.0
        vector[5] = len(enemy_cities) / 5.0

        # Total health
        if own_units:
            vector[6] = sum(u.get("health", 0) for u in own_units) / (len(own_units) * 100.0)
        if enemy_units:
            vector[7] = sum(u.get("health", 0) for u in enemy_units) / (len(enemy_units) * 100.0)

        return vector

    @property
    def spatial_shape(self) -> tuple[int, int, int]:
        """Shape of spatial tensor."""
        return (self.TOTAL_CHANNELS, self.board_size, self.board_size)

    @property
    def vector_dim(self) -> int:
        """Dimension of vector tensor."""
        return 64
