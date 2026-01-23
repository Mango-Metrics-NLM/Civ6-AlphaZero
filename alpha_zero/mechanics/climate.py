"""Climate and Disasters System for Civ6-AlphaZero.

This module implements climate change and natural disaster mechanics including:
- CO2 emissions and climate phases
- Sea level rise and coastal flooding
- Natural disasters (volcanoes, floods, hurricanes, etc.)
- Mitigation and adaptation options

All mechanics are designed to be:
- Fully configurable via dataclasses
- Testable in isolation
- Compatible with the existing encoder system
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ClimatePhase(Enum):
    """Climate change phases."""

    NORMAL = auto()  # No climate effects
    WARMING_1 = auto()  # Mild effects
    WARMING_2 = auto()  # Moderate effects
    WARMING_3 = auto()  # Severe effects
    WARMING_4 = auto()  # Extreme effects
    WARMING_5 = auto()  # Catastrophic effects
    WARMING_6 = auto()  # Apocalyptic effects
    WARMING_7 = auto()  # Maximum warming


class DisasterType(Enum):
    """Types of natural disasters."""

    FLOOD = auto()
    VOLCANIC_ERUPTION = auto()
    HURRICANE = auto()
    TORNADO = auto()
    DROUGHT = auto()
    BLIZZARD = auto()
    DUST_STORM = auto()
    FOREST_FIRE = auto()
    METEOR_STRIKE = auto()  # Rare, catastrophic


class DisasterSeverity(Enum):
    """Severity levels for disasters."""

    MINOR = auto()  # Minimal damage, some benefits
    MODERATE = auto()  # Significant damage
    MAJOR = auto()  # Severe damage
    CATASTROPHIC = auto()  # Devastating damage


class TerrainFeature(Enum):
    """Terrain features relevant to disasters."""

    FLOODPLAINS = auto()
    VOLCANO = auto()
    COAST = auto()
    FOREST = auto()
    GRASSLAND = auto()
    DESERT = auto()
    TUNDRA = auto()


@dataclass
class ClimateConfig:
    """Configuration for climate mechanics.

    Attributes:
        co2_per_power_plant: CO2 generated per coal/oil power plant
        co2_per_industrial_zone: CO2 from industrial zones
        co2_threshold_per_phase: CO2 needed to advance climate phase
        sea_level_per_phase: Tiles flooded per climate phase
        disaster_frequency_base: Base chance of disaster per turn (0-1)
        disaster_frequency_per_phase: Additional chance per climate phase
        fertility_bonus_on_disaster: Food yield increase after disaster
        deforestation_co2: CO2 from removing forest
        nuclear_co2: CO2 from nuclear meltdown
    """

    co2_per_power_plant: float = 10.0
    co2_per_industrial_zone: float = 2.0
    co2_threshold_per_phase: float = 100.0
    sea_level_per_phase: int = 3
    disaster_frequency_base: float = 0.05
    disaster_frequency_per_phase: float = 0.02
    fertility_bonus_on_disaster: float = 1.0
    deforestation_co2: float = 5.0
    nuclear_co2: float = 50.0


@dataclass
class DisasterConfig:
    """Configuration for a disaster type.

    Attributes:
        disaster_type: Type of disaster
        base_frequency: Base occurrence frequency
        terrain_requirements: Required terrain features
        min_severity: Minimum severity
        max_severity: Maximum severity
        damage_buildings: Whether can damage buildings
        damage_improvements: Whether can damage improvements
        damage_units: Whether can damage units
        damage_population: Whether can reduce population
        yield_bonus_after: Bonus yields after disaster
        phase_frequency_multiplier: Frequency multiplier per climate phase
    """

    disaster_type: DisasterType
    base_frequency: float = 0.1
    terrain_requirements: list[TerrainFeature] = field(default_factory=list)
    min_severity: DisasterSeverity = DisasterSeverity.MINOR
    max_severity: DisasterSeverity = DisasterSeverity.MAJOR
    damage_buildings: bool = False
    damage_improvements: bool = True
    damage_units: bool = False
    damage_population: bool = False
    yield_bonus_after: dict[str, float] = field(default_factory=dict)
    phase_frequency_multiplier: float = 1.2


DEFAULT_DISASTER_CONFIGS: dict[DisasterType, DisasterConfig] = {
    DisasterType.FLOOD: DisasterConfig(
        disaster_type=DisasterType.FLOOD,
        base_frequency=0.15,
        terrain_requirements=[TerrainFeature.FLOODPLAINS],
        max_severity=DisasterSeverity.MAJOR,
        damage_improvements=True,
        damage_population=True,
        yield_bonus_after={"food": 2.0},
    ),
    DisasterType.VOLCANIC_ERUPTION: DisasterConfig(
        disaster_type=DisasterType.VOLCANIC_ERUPTION,
        base_frequency=0.05,
        terrain_requirements=[TerrainFeature.VOLCANO],
        max_severity=DisasterSeverity.CATASTROPHIC,
        damage_buildings=True,
        damage_improvements=True,
        damage_units=True,
        damage_population=True,
        yield_bonus_after={"production": 2.0, "science": 1.0},
    ),
    DisasterType.HURRICANE: DisasterConfig(
        disaster_type=DisasterType.HURRICANE,
        base_frequency=0.08,
        terrain_requirements=[TerrainFeature.COAST],
        max_severity=DisasterSeverity.MAJOR,
        damage_buildings=True,
        damage_improvements=True,
        yield_bonus_after={"food": 1.0},
        phase_frequency_multiplier=1.5,
    ),
    DisasterType.DROUGHT: DisasterConfig(
        disaster_type=DisasterType.DROUGHT,
        base_frequency=0.10,
        terrain_requirements=[TerrainFeature.GRASSLAND, TerrainFeature.DESERT],
        max_severity=DisasterSeverity.MODERATE,
        damage_improvements=True,
        yield_bonus_after={},
        phase_frequency_multiplier=1.4,
    ),
    DisasterType.BLIZZARD: DisasterConfig(
        disaster_type=DisasterType.BLIZZARD,
        base_frequency=0.08,
        terrain_requirements=[TerrainFeature.TUNDRA],
        max_severity=DisasterSeverity.MODERATE,
        damage_units=True,
        yield_bonus_after={"food": 1.0},
    ),
    DisasterType.FOREST_FIRE: DisasterConfig(
        disaster_type=DisasterType.FOREST_FIRE,
        base_frequency=0.12,
        terrain_requirements=[TerrainFeature.FOREST],
        max_severity=DisasterSeverity.MAJOR,
        damage_improvements=True,
        yield_bonus_after={"production": 1.0},
        phase_frequency_multiplier=1.6,
    ),
}


@dataclass
class DisasterEvent:
    """A disaster event that occurred.

    Attributes:
        event_id: Unique identifier
        disaster_type: Type of disaster
        severity: Severity level
        affected_tiles: List of affected tile positions
        affected_cities: List of affected city names
        damage: Dictionary of damage effects
        yield_bonuses: Dictionary of yield bonuses applied
        turn: Turn when disaster occurred
        phase: Climate phase when disaster occurred
    """

    event_id: str
    disaster_type: DisasterType
    severity: DisasterSeverity
    affected_tiles: list[tuple[int, int]]
    affected_cities: list[str]
    damage: dict[str, Any]
    yield_bonuses: dict[tuple[int, int], dict[str, float]]
    turn: int
    phase: ClimatePhase


@dataclass
class FloodedTile:
    """A tile that has been submerged.

    Attributes:
        position: Tile position
        turn_flooded: Turn when flooding occurred
        was_coastal: Whether tile was already coastal
        phase_flooded: Climate phase when flooded
    """

    position: tuple[int, int]
    turn_flooded: int
    was_coastal: bool
    phase_flooded: ClimatePhase


class ClimateManager:
    """Manages climate and disaster mechanics.

    This class handles:
    - CO2 tracking and climate phases
    - Sea level rise
    - Natural disaster generation
    - Yield bonuses from disasters

    Example:
        >>> config = ClimateConfig(co2_per_power_plant=10.0)
        >>> manager = ClimateManager(config)
        >>> manager.add_co2(50.0)
        >>> disasters = manager.process_turn(turn=100, map_features={})
    """

    def __init__(
        self,
        config: ClimateConfig | None = None,
        disaster_configs: dict[DisasterType, DisasterConfig] | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Initialize climate manager.

        Args:
            config: Climate configuration
            disaster_configs: Disaster configurations
            random_seed: Seed for reproducibility
        """
        self.config = config or ClimateConfig()
        self._disaster_configs = disaster_configs or DEFAULT_DISASTER_CONFIGS
        self._rng = random.Random(random_seed)

        self._total_co2: float = 0.0
        self._current_phase: ClimatePhase = ClimatePhase.NORMAL
        self._flooded_tiles: dict[tuple[int, int], FloodedTile] = {}
        self._disaster_history: list[DisasterEvent] = []
        self._tile_yield_bonuses: dict[tuple[int, int], dict[str, float]] = {}
        self._event_counter = 0
        self._turn = 0

        logger.debug(f"ClimateManager initialized with config: {self.config}")

    def add_co2(self, amount: float, source: str = "unknown") -> float:
        """Add CO2 emissions.

        Args:
            amount: Amount of CO2 to add
            source: Source description

        Returns:
            New total CO2
        """
        self._total_co2 += amount
        logger.debug(f"Added {amount:.1f} CO2 from {source} (total: {self._total_co2:.1f})")

        # Check for phase transition
        self._check_phase_transition()

        return self._total_co2

    def _check_phase_transition(self) -> bool:
        """Check if climate phase should advance.

        Returns:
            True if phase advanced
        """
        threshold = self.config.co2_threshold_per_phase
        phases = list(ClimatePhase)
        current_idx = phases.index(self._current_phase)

        # Calculate expected phase based on CO2
        expected_idx = min(int(self._total_co2 / threshold), len(phases) - 1)

        if expected_idx > current_idx:
            old_phase = self._current_phase
            self._current_phase = phases[expected_idx]
            logger.warning(
                f"Climate phase advanced: {old_phase.name} -> {self._current_phase.name} "
                f"(CO2: {self._total_co2:.0f})"
            )
            return True

        return False

    def process_turn(
        self,
        turn: int,
        map_features: dict[tuple[int, int], set[TerrainFeature]],
        coastal_tiles: set[tuple[int, int]] | None = None,
    ) -> list[DisasterEvent]:
        """Process turn for climate mechanics.

        Args:
            turn: Current turn number
            map_features: Dictionary of tile positions to terrain features
            coastal_tiles: Set of coastal tile positions

        Returns:
            List of disaster events that occurred
        """
        self._turn = turn
        disasters: list[DisasterEvent] = []

        # Process sea level rise
        if coastal_tiles:
            self._process_sea_level_rise(turn, coastal_tiles)

        # Generate potential disasters
        for position, features in map_features.items():
            for disaster_type, config in self._disaster_configs.items():
                # Check terrain requirements
                if config.terrain_requirements:
                    if not any(f in features for f in config.terrain_requirements):
                        continue

                # Calculate frequency
                frequency = self._calculate_disaster_frequency(config)

                if self._rng.random() < frequency:
                    event = self._create_disaster(
                        disaster_type=disaster_type,
                        config=config,
                        epicenter=position,
                        turn=turn,
                    )
                    disasters.append(event)
                    self._disaster_history.append(event)

        return disasters

    def _calculate_disaster_frequency(self, config: DisasterConfig) -> float:
        """Calculate disaster frequency based on climate phase.

        Args:
            config: Disaster configuration

        Returns:
            Adjusted frequency
        """
        base = config.base_frequency
        phase_idx = self._current_phase.value - 1

        # Apply phase multiplier
        multiplier = config.phase_frequency_multiplier ** phase_idx

        # Apply global frequency modifiers
        global_mod = (
            self.config.disaster_frequency_base +
            self.config.disaster_frequency_per_phase * phase_idx
        )

        return base * multiplier * (1 + global_mod)

    def _create_disaster(
        self,
        disaster_type: DisasterType,
        config: DisasterConfig,
        epicenter: tuple[int, int],
        turn: int,
    ) -> DisasterEvent:
        """Create a disaster event.

        Args:
            disaster_type: Type of disaster
            config: Disaster configuration
            epicenter: Center tile of disaster
            turn: Current turn

        Returns:
            Created disaster event
        """
        self._event_counter += 1
        event_id = f"disaster_{self._event_counter}"

        # Determine severity
        severities = list(DisasterSeverity)
        min_idx = severities.index(config.min_severity)
        max_idx = severities.index(config.max_severity)

        # Higher climate phase = higher chance of severe
        phase_bonus = (self._current_phase.value - 1) * 0.1
        roll = self._rng.random() + phase_bonus

        severity_idx = min(max_idx, max(min_idx, int(roll * (max_idx - min_idx + 1)) + min_idx))
        severity = severities[severity_idx]

        # Determine affected area (based on severity)
        radius = severity.value
        affected_tiles = self._get_tiles_in_radius(epicenter, radius)

        # Calculate damage
        damage: dict[str, Any] = {
            "buildings_damaged": 0,
            "improvements_destroyed": 0,
            "units_damaged": 0,
            "population_lost": 0,
            "tile_count": len(affected_tiles),
        }

        severity_multiplier = severity.value
        if config.damage_buildings:
            damage["buildings_damaged"] = severity_multiplier
        if config.damage_improvements:
            damage["improvements_destroyed"] = len(affected_tiles) * severity_multiplier
        if config.damage_units:
            damage["units_damaged"] = severity_multiplier * 10  # HP damage
        if config.damage_population:
            damage["population_lost"] = max(0, severity_multiplier - 1)

        # Apply yield bonuses
        yield_bonuses: dict[tuple[int, int], dict[str, float]] = {}
        for tile in affected_tiles:
            bonuses: dict[str, float] = {}
            for yield_type, bonus in config.yield_bonus_after.items():
                bonuses[yield_type] = bonus * self.config.fertility_bonus_on_disaster
            if bonuses:
                yield_bonuses[tile] = bonuses
                # Store for future reference
                if tile not in self._tile_yield_bonuses:
                    self._tile_yield_bonuses[tile] = {}
                for y_type, value in bonuses.items():
                    self._tile_yield_bonuses[tile][y_type] = (
                        self._tile_yield_bonuses[tile].get(y_type, 0.0) + value
                    )

        event = DisasterEvent(
            event_id=event_id,
            disaster_type=disaster_type,
            severity=severity,
            affected_tiles=affected_tiles,
            affected_cities=[],  # Would be populated by caller
            damage=damage,
            yield_bonuses=yield_bonuses,
            turn=turn,
            phase=self._current_phase,
        )

        logger.info(
            f"{disaster_type.name} ({severity.name}) at {epicenter}: "
            f"{len(affected_tiles)} tiles affected"
        )

        return event

    def _get_tiles_in_radius(
        self,
        center: tuple[int, int],
        radius: int,
    ) -> list[tuple[int, int]]:
        """Get all tiles within radius of center.

        Args:
            center: Center position
            radius: Radius in tiles

        Returns:
            List of tile positions
        """
        tiles: list[tuple[int, int]] = []
        cx, cy = center

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) + abs(dy) <= radius:  # Manhattan distance
                    tiles.append((cx + dx, cy + dy))

        return tiles

    def _process_sea_level_rise(
        self,
        turn: int,
        coastal_tiles: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        """Process sea level rise for current phase.

        Args:
            turn: Current turn
            coastal_tiles: Set of coastal tile positions

        Returns:
            List of newly flooded tiles
        """
        newly_flooded: list[tuple[int, int]] = []

        # Number of tiles to flood based on phase
        phase_idx = self._current_phase.value - 1
        tiles_to_flood = phase_idx * self.config.sea_level_per_phase

        # Get eligible tiles (coastal, not already flooded)
        eligible = [
            t for t in coastal_tiles
            if t not in self._flooded_tiles
        ]

        if not eligible:
            return newly_flooded

        # Select random tiles to flood
        num_to_flood = min(len(eligible), max(0, tiles_to_flood - len(self._flooded_tiles)))

        if num_to_flood > 0:
            selected = self._rng.sample(eligible, num_to_flood)
            for tile in selected:
                self._flooded_tiles[tile] = FloodedTile(
                    position=tile,
                    turn_flooded=turn,
                    was_coastal=True,
                    phase_flooded=self._current_phase,
                )
                newly_flooded.append(tile)

            logger.warning(
                f"Sea level rise: {len(newly_flooded)} tiles flooded "
                f"(total flooded: {len(self._flooded_tiles)})"
            )

        return newly_flooded

    def get_tile_yield_bonus(self, position: tuple[int, int]) -> dict[str, float]:
        """Get yield bonuses for a tile from disasters.

        Args:
            position: Tile position

        Returns:
            Dictionary of yield bonuses
        """
        return self._tile_yield_bonuses.get(position, {})

    def is_tile_flooded(self, position: tuple[int, int]) -> bool:
        """Check if a tile is flooded.

        Args:
            position: Tile position

        Returns:
            True if flooded
        """
        return position in self._flooded_tiles

    def get_climate_status(self) -> dict[str, Any]:
        """Get current climate status.

        Returns:
            Dictionary of climate information
        """
        return {
            "phase": self._current_phase.name,
            "phase_number": self._current_phase.value,
            "total_co2": self._total_co2,
            "next_phase_threshold": self.config.co2_threshold_per_phase * self._current_phase.value,
            "flooded_tiles": len(self._flooded_tiles),
            "disaster_count": len(self._disaster_history),
        }

    def get_disaster_forecast(
        self,
        map_features: dict[tuple[int, int], set[TerrainFeature]],
    ) -> dict[DisasterType, float]:
        """Get probability forecast for disasters.

        Args:
            map_features: Map terrain features

        Returns:
            Dictionary of disaster type to probability
        """
        forecast: dict[DisasterType, float] = {}

        for disaster_type, config in self._disaster_configs.items():
            # Count eligible tiles
            eligible_count = 0
            for features in map_features.values():
                if config.terrain_requirements:
                    if any(f in features for f in config.terrain_requirements):
                        eligible_count += 1
                else:
                    eligible_count += 1

            if eligible_count > 0:
                frequency = self._calculate_disaster_frequency(config)
                # Probability of at least one occurrence
                prob = 1 - (1 - frequency) ** eligible_count
                forecast[disaster_type] = min(1.0, prob)

        return forecast

    def get_mitigation_options(self) -> list[dict[str, Any]]:
        """Get available climate mitigation options.

        Returns:
            List of mitigation options with effects
        """
        return [
            {
                "name": "Carbon Capture Project",
                "cost": 500,
                "co2_reduction": 25.0,
                "description": "Removes CO2 from atmosphere",
            },
            {
                "name": "Renewable Energy Transition",
                "cost": 1000,
                "ongoing_reduction": 5.0,
                "description": "Reduces ongoing emissions",
            },
            {
                "name": "Seawall Construction",
                "cost": 200,
                "protection": "coastal",
                "description": "Protects coastal tile from flooding",
            },
            {
                "name": "Flood Barrier",
                "cost": 150,
                "protection": "floodplains",
                "description": "Reduces flood damage",
            },
        ]

    def encode_for_state(self, player_id: int = 0) -> dict[str, Any]:
        """Encode climate state for neural network input.

        Args:
            player_id: Player perspective (unused, global state)

        Returns:
            Dictionary of encoded features
        """
        phase_encoding = [0.0] * len(ClimatePhase)
        phase_encoding[self._current_phase.value - 1] = 1.0

        recent_disasters = [
            d for d in self._disaster_history
            if self._turn - d.turn <= 10
        ]

        return {
            "climate_phase": self._current_phase.value,
            "phase_encoding": phase_encoding,
            "total_co2": self._total_co2,
            "co2_normalized": min(1.0, self._total_co2 / (self.config.co2_threshold_per_phase * 7)),
            "flooded_tiles": len(self._flooded_tiles),
            "recent_disasters": len(recent_disasters),
            "disaster_total": len(self._disaster_history),
        }

    def reset(self) -> None:
        """Reset all climate state."""
        self._total_co2 = 0.0
        self._current_phase = ClimatePhase.NORMAL
        self._flooded_tiles.clear()
        self._disaster_history.clear()
        self._tile_yield_bonuses.clear()
        self._event_counter = 0
        self._turn = 0
        logger.debug("ClimateManager reset")
