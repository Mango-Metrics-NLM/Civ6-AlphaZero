"""
State Encoders for Hierarchical Reasoning.

This module provides state encoders that transform game state into tensor
representations at different abstraction levels. Strategic encoders focus
on high-level features, while tactical and operational encoders capture
more detailed information.

Example:
    >>> from alpha_zero.hierarchical.encoders import StrategicStateEncoder
    >>> encoder = StrategicStateEncoder(embed_dim=256)
    >>> strategic_state = encoder.encode(game_state)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    import torch.nn.functional as F  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from alpha_zero.hierarchical.config import LevelConfig
from alpha_zero.hierarchical.goals.goal_types import HierarchyLevel

logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    """Configuration for state encoders.

    Attributes:
        embed_dim: Output embedding dimension.
        board_size: Game board size (height, width).
        use_attention: Whether to use attention mechanisms.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
    """

    embed_dim: int = 128
    board_size: tuple[int, int] = (16, 16)
    use_attention: bool = True
    num_heads: int = 4
    dropout: float = 0.1

    @classmethod
    def for_level(cls, level: HierarchyLevel) -> EncoderConfig:
        """Create level-specific encoder configuration.

        Args:
            level: Hierarchy level.

        Returns:
            EncoderConfig with appropriate settings.
        """
        configs = {
            HierarchyLevel.STRATEGIC: cls(
                embed_dim=256,
                use_attention=True,
                num_heads=8,
            ),
            HierarchyLevel.TACTICAL: cls(
                embed_dim=128,
                use_attention=True,
                num_heads=4,
            ),
            HierarchyLevel.OPERATIONAL: cls(
                embed_dim=64,
                use_attention=False,
                num_heads=2,
            ),
        }
        return configs.get(level, cls())


class StateEncoder(ABC):
    """Abstract base class for state encoders.

    Subclasses implement level-specific encoding logic that transforms
    raw game state into tensor representations.
    """

    def __init__(self, config: EncoderConfig) -> None:
        """Initialize the encoder.

        Args:
            config: Encoder configuration.
        """
        self.config = config
        self._initialized = False

    @abstractmethod
    def encode(self, game_state: dict[str, Any]) -> np.ndarray:
        """Encode game state into tensor representation.

        Args:
            game_state: Raw game state dictionary.

        Returns:
            Encoded state as numpy array.
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Get output dimension of encoded state."""
        pass

    @property
    @abstractmethod
    def level(self) -> HierarchyLevel:
        """Get hierarchy level this encoder is designed for."""
        pass

    def _extract_common_features(
        self,
        game_state: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Extract common features from game state.

        Args:
            game_state: Raw game state.

        Returns:
            Dictionary of extracted features.
        """
        features = {}

        # Turn information
        features["turn"] = np.array([game_state.get("turn", 0)], dtype=np.float32)

        # Resources (gold, science, culture, faith, etc.)
        resources = game_state.get("resources", {})
        features["resources"] = np.array(
            [
                resources.get("gold", 0),
                resources.get("science", 0),
                resources.get("culture", 0),
                resources.get("faith", 0),
                resources.get("production", 0),
                resources.get("food", 0),
            ],
            dtype=np.float32,
        )

        # City count
        cities = game_state.get("cities", [])
        features["city_count"] = np.array([len(cities)], dtype=np.float32)

        # Unit count by type
        units = game_state.get("units", [])
        unit_counts: dict[str, int] = {}
        for unit in units:
            unit_type = unit.get("type", "unknown")
            unit_counts[unit_type] = unit_counts.get(unit_type, 0) + 1
        features["unit_counts"] = np.array(list(unit_counts.values()) or [0], dtype=np.float32)

        return features


class StrategicStateEncoder(StateEncoder):
    """State encoder for strategic level.

    Focuses on high-level features like:
    - Victory progress indicators
    - Diplomatic relationships
    - Long-term resource trends
    - Technology/civic progress
    - Overall military strength

    Uses heavy abstraction to create compact representations suitable
    for long-horizon planning.
    """

    # Feature dimensions
    VICTORY_FEATURES = 6  # One per victory type + general progress
    DIPLOMATIC_FEATURES = 8  # Relationship scores with up to 8 civs
    RESOURCE_FEATURES = 12  # Resource levels and trends
    MILITARY_FEATURES = 8  # Military strength metrics
    TERRITORY_FEATURES = 4  # Territory control metrics
    TECH_FEATURES = 8  # Technology/civic progress

    def __init__(
        self,
        config: EncoderConfig | None = None,
        level_config: LevelConfig | None = None,
    ) -> None:
        """Initialize strategic encoder.

        Args:
            config: Encoder configuration.
            level_config: Level-specific configuration.
        """
        config = config or EncoderConfig.for_level(HierarchyLevel.STRATEGIC)
        super().__init__(config)
        self.level_config = level_config

        # Calculate total input features
        self._input_features = (
            self.VICTORY_FEATURES
            + self.DIPLOMATIC_FEATURES
            + self.RESOURCE_FEATURES
            + self.MILITARY_FEATURES
            + self.TERRITORY_FEATURES
            + self.TECH_FEATURES
        )

        logger.debug(
            "StrategicStateEncoder initialized with %d input features, %d output dim",
            self._input_features,
            self.config.embed_dim,
        )

    @property
    def output_dim(self) -> int:
        """Output embedding dimension."""
        return self.config.embed_dim

    @property
    def level(self) -> HierarchyLevel:
        """Strategic hierarchy level."""
        return HierarchyLevel.STRATEGIC

    def encode(self, game_state: dict[str, Any]) -> np.ndarray:
        """Encode game state for strategic reasoning.

        Args:
            game_state: Raw game state dictionary.

        Returns:
            Strategic state representation.
        """
        features = []

        # Victory progress
        victory = self._encode_victory_progress(game_state)
        features.append(victory)

        # Diplomatic state
        diplomatic = self._encode_diplomatic_state(game_state)
        features.append(diplomatic)

        # Resource metrics
        resources = self._encode_resource_metrics(game_state)
        features.append(resources)

        # Military strength
        military = self._encode_military_strength(game_state)
        features.append(military)

        # Territory control
        territory = self._encode_territory_control(game_state)
        features.append(territory)

        # Technology progress
        tech = self._encode_tech_progress(game_state)
        features.append(tech)

        # Concatenate all features
        combined = np.concatenate(features)

        # Pad or truncate to embed_dim
        if len(combined) < self.config.embed_dim:
            padded = np.zeros(self.config.embed_dim, dtype=np.float32)
            padded[: len(combined)] = combined
            return padded
        else:
            return combined[: self.config.embed_dim].astype(np.float32)

    def _encode_victory_progress(self, game_state: dict[str, Any]) -> np.ndarray:
        """Encode victory progress indicators."""
        victory = game_state.get("victory_progress", {})
        return np.array(
            [
                victory.get("domination", 0.0),
                victory.get("science", 0.0),
                victory.get("culture", 0.0),
                victory.get("religious", 0.0),
                victory.get("diplomatic", 0.0),
                victory.get("score", 0.0),
            ],
            dtype=np.float32,
        )

    def _encode_diplomatic_state(self, game_state: dict[str, Any]) -> np.ndarray:
        """Encode diplomatic relationships."""
        diplomacy = game_state.get("diplomacy", {})
        relationships = diplomacy.get("relationships", {})

        # Encode relationship scores (normalized -1 to 1)
        scores = []
        for i in range(self.DIPLOMATIC_FEATURES):
            civ_id = f"civ_{i}"
            rel = relationships.get(civ_id, {})
            score = rel.get("score", 0) / 100.0  # Normalize
            scores.append(score)

        return np.array(scores, dtype=np.float32)

    def _encode_resource_metrics(self, game_state: dict[str, Any]) -> np.ndarray:
        """Encode resource levels and trends."""
        resources = game_state.get("resources", {})
        yields = game_state.get("yields", {})
        trends = game_state.get("resource_trends", {})

        return np.array(
            [
                # Current levels (normalized)
                min(resources.get("gold", 0) / 1000.0, 1.0),
                min(resources.get("science", 0) / 500.0, 1.0),
                min(resources.get("culture", 0) / 500.0, 1.0),
                min(resources.get("faith", 0) / 500.0, 1.0),
                # Per-turn yields (normalized)
                min(yields.get("gold_per_turn", 0) / 100.0, 1.0),
                min(yields.get("science_per_turn", 0) / 100.0, 1.0),
                min(yields.get("culture_per_turn", 0) / 100.0, 1.0),
                min(yields.get("faith_per_turn", 0) / 100.0, 1.0),
                # Trends (positive/negative)
                np.sign(trends.get("gold", 0)),
                np.sign(trends.get("science", 0)),
                np.sign(trends.get("culture", 0)),
                np.sign(trends.get("faith", 0)),
            ],
            dtype=np.float32,
        )

    def _encode_military_strength(self, game_state: dict[str, Any]) -> np.ndarray:
        """Encode military strength metrics."""
        military = game_state.get("military", {})
        units = game_state.get("units", [])

        # Count unit types
        melee = sum(1 for u in units if u.get("class") == "melee")
        ranged = sum(1 for u in units if u.get("class") == "ranged")
        cavalry = sum(1 for u in units if u.get("class") == "cavalry")
        naval = sum(1 for u in units if u.get("class") == "naval")

        return np.array(
            [
                min(military.get("strength", 0) / 1000.0, 1.0),
                min(melee / 20.0, 1.0),
                min(ranged / 20.0, 1.0),
                min(cavalry / 10.0, 1.0),
                min(naval / 10.0, 1.0),
                min(len(units) / 50.0, 1.0),
                military.get("at_war", 0),
                military.get("defensive_pacts", 0) / 5.0,
            ],
            dtype=np.float32,
        )

    def _encode_territory_control(self, game_state: dict[str, Any]) -> np.ndarray:
        """Encode territory control metrics."""
        territory = game_state.get("territory", {})
        cities = game_state.get("cities", [])

        return np.array(
            [
                min(len(cities) / 20.0, 1.0),
                min(territory.get("tiles_owned", 0) / 500.0, 1.0),
                territory.get("continent_control", 0),
                territory.get("strategic_resources", 0) / 10.0,
            ],
            dtype=np.float32,
        )

    def _encode_tech_progress(self, game_state: dict[str, Any]) -> np.ndarray:
        """Encode technology and civic progress."""
        tech = game_state.get("technology", {})
        civics = game_state.get("civics", {})

        return np.array(
            [
                tech.get("era", 0) / 8.0,  # Normalize era (0-8)
                tech.get("techs_researched", 0) / 70.0,  # Total techs
                tech.get("current_progress", 0),  # 0-1
                tech.get("turns_to_complete", 10) / 50.0,
                civics.get("civics_completed", 0) / 50.0,
                civics.get("current_progress", 0),
                civics.get("government_tier", 0) / 4.0,
                civics.get("policy_slots", 4) / 10.0,
            ],
            dtype=np.float32,
        )


class TacticalStateEncoder(StateEncoder):
    """State encoder for tactical level.

    Focuses on medium-term features like:
    - Regional military disposition
    - City development status
    - Ongoing operations
    - Resource availability
    - Nearby threats and opportunities
    """

    # Feature dimensions for tactical encoding
    REGION_SIZE = 8  # 8x8 region around focus point
    REGION_CHANNELS = 16  # Channels per region
    CITY_FEATURES = 20  # Features per city
    MAX_CITIES = 10  # Maximum cities to encode
    UNIT_FEATURES = 12  # Features per unit group
    MAX_UNIT_GROUPS = 8  # Maximum unit groups

    def __init__(
        self,
        config: EncoderConfig | None = None,
        level_config: LevelConfig | None = None,
    ) -> None:
        """Initialize tactical encoder.

        Args:
            config: Encoder configuration.
            level_config: Level-specific configuration.
        """
        config = config or EncoderConfig.for_level(HierarchyLevel.TACTICAL)
        super().__init__(config)
        self.level_config = level_config

        logger.debug(
            "TacticalStateEncoder initialized with %d output dim",
            self.config.embed_dim,
        )

    @property
    def output_dim(self) -> int:
        """Output embedding dimension."""
        return self.config.embed_dim

    @property
    def level(self) -> HierarchyLevel:
        """Tactical hierarchy level."""
        return HierarchyLevel.TACTICAL

    def encode(
        self,
        game_state: dict[str, Any],
        focus_location: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Encode game state for tactical reasoning.

        Args:
            game_state: Raw game state dictionary.
            focus_location: Optional (x, y) focus point for local encoding.

        Returns:
            Tactical state representation.
        """
        features = []

        # Regional spatial features
        region = self._encode_region(game_state, focus_location)
        features.append(region.flatten())

        # City development
        cities = self._encode_cities(game_state)
        features.append(cities.flatten())

        # Unit disposition
        units = self._encode_unit_groups(game_state)
        features.append(units.flatten())

        # Resource availability
        resources = self._encode_tactical_resources(game_state)
        features.append(resources)

        # Concatenate and project to embed_dim
        combined = np.concatenate(features)

        # Truncate or pad to embed_dim
        result = np.zeros(self.config.embed_dim, dtype=np.float32)
        result[: min(len(combined), self.config.embed_dim)] = combined[: self.config.embed_dim]

        return result

    def _encode_region(
        self,
        game_state: dict[str, Any],
        focus: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Encode regional spatial information."""
        board_size = self.config.board_size
        region = np.zeros(
            (self.REGION_CHANNELS, self.REGION_SIZE, self.REGION_SIZE),
            dtype=np.float32,
        )

        # Default focus to center of map
        if focus is None:
            focus = (board_size[0] // 2, board_size[1] // 2)

        # Extract region bounds
        half = self.REGION_SIZE // 2
        x_start = max(0, focus[0] - half)
        y_start = max(0, focus[1] - half)

        # Encode terrain in region (channel 0-3)
        terrain = game_state.get("terrain", {})
        for dx in range(self.REGION_SIZE):
            for dy in range(self.REGION_SIZE):
                x = x_start + dx
                y = y_start + dy
                tile = terrain.get(f"{x},{y}", {})
                region[0, dx, dy] = tile.get("plains", 0)
                region[1, dx, dy] = tile.get("hills", 0)
                region[2, dx, dy] = tile.get("mountains", 0)
                region[3, dx, dy] = tile.get("water", 0)

        # Encode own units (channels 4-7)
        units = game_state.get("units", [])
        own_player = game_state.get("current_player", 0)
        for unit in units:
            if unit.get("player") != own_player:
                continue
            ux, uy = unit.get("x", 0), unit.get("y", 0)
            dx, dy = ux - x_start, uy - y_start
            if 0 <= dx < self.REGION_SIZE and 0 <= dy < self.REGION_SIZE:
                unit_class = unit.get("class", "melee")
                channel = {"melee": 4, "ranged": 5, "cavalry": 6, "naval": 7}.get(unit_class, 4)
                region[channel, dx, dy] += 1

        # Encode enemy units (channels 8-11)
        for unit in units:
            if unit.get("player") == own_player:
                continue
            ux, uy = unit.get("x", 0), unit.get("y", 0)
            dx, dy = ux - x_start, uy - y_start
            if 0 <= dx < self.REGION_SIZE and 0 <= dy < self.REGION_SIZE:
                unit_class = unit.get("class", "melee")
                channel = {"melee": 8, "ranged": 9, "cavalry": 10, "naval": 11}.get(unit_class, 8)
                region[channel, dx, dy] += 1

        # Encode cities (channels 12-13)
        cities = game_state.get("cities", [])
        for city in cities:
            cx, cy = city.get("x", 0), city.get("y", 0)
            dx, dy = cx - x_start, cy - y_start
            if 0 <= dx < self.REGION_SIZE and 0 <= dy < self.REGION_SIZE:
                if city.get("player") == own_player:
                    region[12, dx, dy] = 1
                else:
                    region[13, dx, dy] = 1

        # Encode improvements and resources (channels 14-15)
        improvements = game_state.get("improvements", {})
        for dx in range(self.REGION_SIZE):
            for dy in range(self.REGION_SIZE):
                x = x_start + dx
                y = y_start + dy
                imp = improvements.get(f"{x},{y}", {})
                region[14, dx, dy] = 1 if imp.get("improvement") else 0
                region[15, dx, dy] = 1 if imp.get("resource") else 0

        return region

    def _encode_cities(self, game_state: dict[str, Any]) -> np.ndarray:
        """Encode city development information."""
        cities = game_state.get("cities", [])
        own_player = game_state.get("current_player", 0)

        encoded = np.zeros((self.MAX_CITIES, self.CITY_FEATURES), dtype=np.float32)

        own_cities = [c for c in cities if c.get("player") == own_player]
        for i, city in enumerate(own_cities[: self.MAX_CITIES]):
            encoded[i, 0] = city.get("population", 0) / 30.0
            encoded[i, 1] = city.get("production", 0) / 100.0
            encoded[i, 2] = city.get("food", 0) / 50.0
            encoded[i, 3] = city.get("gold", 0) / 50.0
            encoded[i, 4] = city.get("science", 0) / 50.0
            encoded[i, 5] = city.get("culture", 0) / 50.0
            encoded[i, 6] = city.get("housing", 0) / 20.0
            encoded[i, 7] = city.get("amenities", 0) / 10.0
            encoded[i, 8] = city.get("districts", 0) / 10.0
            encoded[i, 9] = city.get("buildings", 0) / 20.0
            # Defensive info
            encoded[i, 10] = city.get("walls_level", 0) / 3.0
            encoded[i, 11] = city.get("garrison_strength", 0) / 100.0
            encoded[i, 12] = 1 if city.get("under_siege") else 0
            # Production queue
            encoded[i, 13] = 1 if city.get("producing_unit") else 0
            encoded[i, 14] = 1 if city.get("producing_building") else 0
            encoded[i, 15] = 1 if city.get("producing_district") else 0
            # Location (normalized)
            encoded[i, 16] = city.get("x", 0) / self.config.board_size[0]
            encoded[i, 17] = city.get("y", 0) / self.config.board_size[1]
            encoded[i, 18] = 1 if city.get("is_capital") else 0
            encoded[i, 19] = city.get("turns_since_founded", 0) / 200.0

        return encoded

    def _encode_unit_groups(self, game_state: dict[str, Any]) -> np.ndarray:
        """Encode unit groups for tactical planning."""
        units = game_state.get("units", [])
        own_player = game_state.get("current_player", 0)

        encoded = np.zeros((self.MAX_UNIT_GROUPS, self.UNIT_FEATURES), dtype=np.float32)

        # Group own units by location proximity
        own_units = [u for u in units if u.get("player") == own_player]
        groups = self._cluster_units(own_units, cluster_radius=3)

        for i, group in enumerate(groups[: self.MAX_UNIT_GROUPS]):
            # Group center
            cx = np.mean([u.get("x", 0) for u in group])
            cy = np.mean([u.get("y", 0) for u in group])
            encoded[i, 0] = cx / self.config.board_size[0]
            encoded[i, 1] = cy / self.config.board_size[1]

            # Group composition
            encoded[i, 2] = len(group) / 10.0
            encoded[i, 3] = sum(1 for u in group if u.get("class") == "melee") / 5.0
            encoded[i, 4] = sum(1 for u in group if u.get("class") == "ranged") / 5.0
            encoded[i, 5] = sum(1 for u in group if u.get("class") == "cavalry") / 3.0

            # Group strength
            encoded[i, 6] = sum(u.get("strength", 0) for u in group) / 500.0
            encoded[i, 7] = np.mean([u.get("health", 100) / 100.0 for u in group])
            encoded[i, 8] = np.mean([u.get("movement", 2) / 5.0 for u in group])

            # Group status
            encoded[i, 9] = sum(1 for u in group if u.get("fortified")) / len(group)
            encoded[i, 10] = sum(1 for u in group if u.get("has_moved")) / len(group)
            encoded[i, 11] = sum(1 for u in group if u.get("can_attack")) / len(group)

        return encoded

    def _cluster_units(
        self,
        units: list[dict[str, Any]],
        cluster_radius: int = 3,
    ) -> list[list[dict[str, Any]]]:
        """Cluster units by proximity."""
        if not units:
            return []

        # Simple clustering: group units within radius of each other
        clusters = []
        used = set()

        for i, unit in enumerate(units):
            if i in used:
                continue

            cluster = [unit]
            used.add(i)

            ux, uy = unit.get("x", 0), unit.get("y", 0)

            for j, other in enumerate(units):
                if j in used:
                    continue
                ox, oy = other.get("x", 0), other.get("y", 0)
                if abs(ox - ux) <= cluster_radius and abs(oy - uy) <= cluster_radius:
                    cluster.append(other)
                    used.add(j)

            clusters.append(cluster)

        return clusters

    def _encode_tactical_resources(self, game_state: dict[str, Any]) -> np.ndarray:
        """Encode tactical resource availability."""
        resources = game_state.get("resources", {})
        yields = game_state.get("yields", {})

        return np.array(
            [
                min(resources.get("gold", 0) / 500.0, 1.0),
                min(yields.get("gold_per_turn", 0) / 50.0, 1.0),
                min(resources.get("strategic_resources", 0) / 20.0, 1.0),
                min(resources.get("luxury_resources", 0) / 10.0, 1.0),
            ],
            dtype=np.float32,
        )


class OperationalStateEncoder(StateEncoder):
    """State encoder for operational level.

    Focuses on immediate features like:
    - Individual unit positions and stats
    - Immediate threats and targets
    - Available actions this turn
    - Local terrain details
    """

    # Operational encoding dimensions
    LOCAL_SIZE = 5  # 5x5 local view
    LOCAL_CHANNELS = 20  # Channels for local view
    UNIT_FEATURES = 16  # Features per unit
    MAX_UNITS = 20  # Maximum units to encode

    def __init__(
        self,
        config: EncoderConfig | None = None,
        level_config: LevelConfig | None = None,
    ) -> None:
        """Initialize operational encoder.

        Args:
            config: Encoder configuration.
            level_config: Level-specific configuration.
        """
        config = config or EncoderConfig.for_level(HierarchyLevel.OPERATIONAL)
        super().__init__(config)
        self.level_config = level_config

        logger.debug(
            "OperationalStateEncoder initialized with %d output dim",
            self.config.embed_dim,
        )

    @property
    def output_dim(self) -> int:
        """Output embedding dimension."""
        return self.config.embed_dim

    @property
    def level(self) -> HierarchyLevel:
        """Operational hierarchy level."""
        return HierarchyLevel.OPERATIONAL

    def encode(
        self,
        game_state: dict[str, Any],
        active_unit: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Encode game state for operational reasoning.

        Args:
            game_state: Raw game state dictionary.
            active_unit: Optional specific unit to focus on.

        Returns:
            Operational state representation.
        """
        features = []

        # Local view around active unit or first own unit
        local = self._encode_local_view(game_state, active_unit)
        features.append(local.flatten())

        # Individual unit status
        units = self._encode_individual_units(game_state)
        features.append(units.flatten())

        # Immediate threats
        threats = self._encode_threats(game_state, active_unit)
        features.append(threats)

        # Available actions
        actions = self._encode_available_actions(game_state, active_unit)
        features.append(actions)

        # Concatenate and project
        combined = np.concatenate(features)
        result = np.zeros(self.config.embed_dim, dtype=np.float32)
        result[: min(len(combined), self.config.embed_dim)] = combined[: self.config.embed_dim]

        return result

    def _encode_local_view(
        self,
        game_state: dict[str, Any],
        active_unit: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Encode local view around unit."""
        local = np.zeros(
            (self.LOCAL_CHANNELS, self.LOCAL_SIZE, self.LOCAL_SIZE),
            dtype=np.float32,
        )

        # Get focus point
        if active_unit:
            fx, fy = active_unit.get("x", 0), active_unit.get("y", 0)
        else:
            units = game_state.get("units", [])
            own_player = game_state.get("current_player", 0)
            own_units = [u for u in units if u.get("player") == own_player]
            if own_units:
                fx, fy = own_units[0].get("x", 0), own_units[0].get("y", 0)
            else:
                fx, fy = self.config.board_size[0] // 2, self.config.board_size[1] // 2

        half = self.LOCAL_SIZE // 2

        # Terrain (channels 0-4)
        terrain = game_state.get("terrain", {})
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                x, y = fx + dx, fy + dy
                tile = terrain.get(f"{x},{y}", {})
                lx, ly = dx + half, dy + half

                local[0, lx, ly] = tile.get("movement_cost", 1) / 3.0
                local[1, lx, ly] = tile.get("defense_bonus", 0) / 100.0
                local[2, lx, ly] = 1 if tile.get("impassable") else 0
                local[3, lx, ly] = 1 if tile.get("water") else 0
                local[4, lx, ly] = 1 if tile.get("has_road") else 0

        # Units (channels 5-14)
        units = game_state.get("units", [])
        own_player = game_state.get("current_player", 0)

        for unit in units:
            ux, uy = unit.get("x", 0), unit.get("y", 0)
            dx, dy = ux - fx + half, uy - fy + half

            if 0 <= dx < self.LOCAL_SIZE and 0 <= dy < self.LOCAL_SIZE:
                is_own = unit.get("player") == own_player
                base = 5 if is_own else 10

                local[base, dx, dy] = unit.get("strength", 0) / 100.0
                local[base + 1, dx, dy] = unit.get("health", 100) / 100.0
                local[base + 2, dx, dy] = unit.get("movement", 0) / 5.0
                local[base + 3, dx, dy] = 1 if unit.get("can_attack") else 0
                local[base + 4, dx, dy] = 1 if unit.get("fortified") else 0

        # Cities (channels 15-17)
        cities = game_state.get("cities", [])
        for city in cities:
            cx, cy = city.get("x", 0), city.get("y", 0)
            dx, dy = cx - fx + half, cy - fy + half

            if 0 <= dx < self.LOCAL_SIZE and 0 <= dy < self.LOCAL_SIZE:
                is_own = city.get("player") == own_player
                channel = 15 if is_own else 16
                local[channel, dx, dy] = 1
                local[17, dx, dy] = city.get("defense", 0) / 100.0

        # Improvements/resources (channels 18-19)
        improvements = game_state.get("improvements", {})
        for dx in range(-half, half + 1):
            for dy in range(-half, half + 1):
                x, y = fx + dx, fy + dy
                imp = improvements.get(f"{x},{y}", {})
                lx, ly = dx + half, dy + half
                local[18, lx, ly] = 1 if imp.get("improvement") else 0
                local[19, lx, ly] = 1 if imp.get("resource") else 0

        return local

    def _encode_individual_units(self, game_state: dict[str, Any]) -> np.ndarray:
        """Encode individual unit information."""
        units = game_state.get("units", [])
        own_player = game_state.get("current_player", 0)

        encoded = np.zeros((self.MAX_UNITS, self.UNIT_FEATURES), dtype=np.float32)

        own_units = [u for u in units if u.get("player") == own_player]
        for i, unit in enumerate(own_units[: self.MAX_UNITS]):
            encoded[i, 0] = unit.get("x", 0) / self.config.board_size[0]
            encoded[i, 1] = unit.get("y", 0) / self.config.board_size[1]
            encoded[i, 2] = unit.get("strength", 0) / 100.0
            encoded[i, 3] = unit.get("ranged_strength", 0) / 100.0
            encoded[i, 4] = unit.get("health", 100) / 100.0
            encoded[i, 5] = unit.get("movement", 0) / 5.0
            encoded[i, 6] = unit.get("max_movement", 2) / 5.0
            encoded[i, 7] = 1 if unit.get("can_attack") else 0
            encoded[i, 8] = 1 if unit.get("has_moved") else 0
            encoded[i, 9] = 1 if unit.get("fortified") else 0
            encoded[i, 10] = unit.get("experience", 0) / 100.0
            encoded[i, 11] = unit.get("level", 1) / 10.0
            # Unit type one-hot
            unit_class = unit.get("class", "melee")
            encoded[i, 12] = 1 if unit_class == "melee" else 0
            encoded[i, 13] = 1 if unit_class == "ranged" else 0
            encoded[i, 14] = 1 if unit_class == "cavalry" else 0
            encoded[i, 15] = 1 if unit_class == "naval" else 0

        return encoded

    def _encode_threats(
        self,
        game_state: dict[str, Any],
        active_unit: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Encode immediate threats."""
        units = game_state.get("units", [])
        own_player = game_state.get("current_player", 0)

        # Count nearby enemy units
        enemies = [u for u in units if u.get("player") != own_player]

        if active_unit:
            fx, fy = active_unit.get("x", 0), active_unit.get("y", 0)
        else:
            fx, fy = self.config.board_size[0] // 2, self.config.board_size[1] // 2

        threats_close = 0  # Within 2 tiles
        threats_medium = 0  # Within 5 tiles
        total_threat_strength = 0

        for enemy in enemies:
            ex, ey = enemy.get("x", 0), enemy.get("y", 0)
            dist = abs(ex - fx) + abs(ey - fy)

            if dist <= 2:
                threats_close += 1
                total_threat_strength += enemy.get("strength", 0)
            elif dist <= 5:
                threats_medium += 1

        return np.array(
            [
                min(threats_close / 5.0, 1.0),
                min(threats_medium / 10.0, 1.0),
                min(total_threat_strength / 500.0, 1.0),
                1 if threats_close > 0 else 0,
            ],
            dtype=np.float32,
        )

    def _encode_available_actions(
        self,
        game_state: dict[str, Any],
        active_unit: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Encode available actions this turn."""
        if not active_unit:
            return np.zeros(8, dtype=np.float32)

        return np.array(
            [
                1 if active_unit.get("can_move") else 0,
                1 if active_unit.get("can_attack") else 0,
                1 if active_unit.get("can_ranged_attack") else 0,
                1 if active_unit.get("can_fortify") else 0,
                1 if active_unit.get("can_heal") else 0,
                1 if active_unit.get("can_pillage") else 0,
                1 if active_unit.get("can_found_city") else 0,
                1 if active_unit.get("can_build") else 0,
            ],
            dtype=np.float32,
        )


class HierarchicalStateEncoder:
    """Combined encoder that produces state representations for all hierarchy levels.

    This class manages multiple level-specific encoders and provides
    a unified interface for encoding game state across all abstraction levels.
    """

    def __init__(
        self,
        strategic_config: EncoderConfig | None = None,
        tactical_config: EncoderConfig | None = None,
        operational_config: EncoderConfig | None = None,
    ) -> None:
        """Initialize hierarchical state encoder.

        Args:
            strategic_config: Configuration for strategic encoder.
            tactical_config: Configuration for tactical encoder.
            operational_config: Configuration for operational encoder.
        """
        self.strategic = StrategicStateEncoder(strategic_config)
        self.tactical = TacticalStateEncoder(tactical_config)
        self.operational = OperationalStateEncoder(operational_config)

        self._encoders = {
            HierarchyLevel.STRATEGIC: self.strategic,
            HierarchyLevel.TACTICAL: self.tactical,
            HierarchyLevel.OPERATIONAL: self.operational,
        }

        logger.info(
            "HierarchicalStateEncoder initialized with dims: "
            "strategic=%d, tactical=%d, operational=%d",
            self.strategic.output_dim,
            self.tactical.output_dim,
            self.operational.output_dim,
        )

    def encode(
        self,
        game_state: dict[str, Any],
        level: HierarchyLevel,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode game state for a specific hierarchy level.

        Args:
            game_state: Raw game state dictionary.
            level: Target hierarchy level.
            **kwargs: Additional arguments passed to level-specific encoder.

        Returns:
            Encoded state representation.
        """
        encoder = self._encoders.get(level)
        if not encoder:
            raise ValueError(f"Unknown hierarchy level: {level}")

        return encoder.encode(game_state, **kwargs)

    def encode_all(
        self,
        game_state: dict[str, Any],
        **kwargs: Any,
    ) -> dict[HierarchyLevel, np.ndarray]:
        """Encode game state for all hierarchy levels.

        Args:
            game_state: Raw game state dictionary.
            **kwargs: Additional arguments passed to encoders.

        Returns:
            Dictionary mapping levels to encoded states.
        """
        return {
            level: encoder.encode(game_state, **kwargs) for level, encoder in self._encoders.items()
        }

    def get_encoder(self, level: HierarchyLevel) -> StateEncoder:
        """Get encoder for a specific level.

        Args:
            level: Hierarchy level.

        Returns:
            Level-specific encoder.
        """
        return self._encoders[level]

    def get_output_dims(self) -> dict[HierarchyLevel, int]:
        """Get output dimensions for all encoders.

        Returns:
            Dictionary mapping levels to output dimensions.
        """
        return {level: encoder.output_dim for level, encoder in self._encoders.items()}
