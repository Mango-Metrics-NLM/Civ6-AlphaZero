"""Comprehensive gameplay configuration for Civ6 AlphaZero.

This module defines all gameplay parameters as nested dataclasses with no
hardcoded values. Every constant used by gameplay systems is configurable
through these structures. The master GameplayConfig class provides factory
methods for common presets and serialization to/from dictionaries.

Typical usage:

    # Backward-compatible defaults (all features disabled)
    config = GameplayConfig.default()

    # All features enabled
    config = GameplayConfig.enhanced()

    # From a saved dictionary
    config = GameplayConfig.from_dict(saved_data)

    # Serialize for checkpointing
    data = config.to_dict()
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FogOfWarConfig:
    """Configuration for fog of war visibility system.

    Controls how far each unit type and city can see, and whether
    previously explored tiles remain visible (remembered) after
    units move away.

    Attributes:
        sight_range_warrior: Tile radius a warrior can reveal.
        sight_range_settler: Tile radius a settler can reveal.
        sight_range_archer: Tile radius an archer can reveal.
        sight_range_city: Tile radius a city can reveal.
        remember_seen_tiles: If True, tiles once seen stay visible
            as last-known state even when no longer in sight range.
    """

    sight_range_warrior: int = 2
    sight_range_settler: int = 2
    sight_range_archer: int = 3
    sight_range_city: int = 3
    remember_seen_tiles: bool = True


@dataclass(frozen=True)
class CombatConfig:
    """Configuration for the combat system.

    Covers base damage, retaliation, terrain defense bonuses, flanking,
    ranged combat rules, and city combat parameters.

    Attributes:
        base_damage: Base damage dealt by a melee attack.
        base_retaliation: Base damage dealt by defender in retaliation.
        terrain_defense_bonus_hills: Fractional defense bonus on hills.
        terrain_defense_bonus_forest: Fractional defense bonus in forests.
        flanking_bonus_per_unit: Fractional attack bonus per flanking unit.
        max_flanking_bonus: Cap on total flanking bonus.
        ranged_attack_range: Maximum tile distance for ranged attacks.
        ranged_no_retaliation: If True, ranged attacks do not trigger
            retaliation from the defender.
        city_defense_strength: Base defensive strength of a city.
        city_ranged_attack_range: Range of a city's ranged counterattack.
        health_damage_scaling: If True, damage output scales with
            current health (injured units deal less damage).
    """

    base_damage: int = 30
    base_retaliation: int = 20
    terrain_defense_bonus_hills: float = 0.25
    terrain_defense_bonus_forest: float = 0.15
    flanking_bonus_per_unit: float = 0.1
    max_flanking_bonus: float = 0.3
    ranged_attack_range: int = 2
    ranged_no_retaliation: bool = True
    city_defense_strength: int = 25
    city_ranged_attack_range: int = 2
    health_damage_scaling: bool = True


@dataclass(frozen=True)
class EconomyConfig:
    """Configuration for the economy system.

    Defines gold income, unit maintenance costs, and purchasing rules.

    Attributes:
        starting_gold: Gold available at the start of the game.
        gold_per_city_base: Base gold income per city per turn.
        gold_per_population: Additional gold income per population point.
        unit_maintenance_warrior: Gold upkeep per warrior per turn.
        unit_maintenance_archer: Gold upkeep per archer per turn.
        unit_maintenance_settler: Gold upkeep per settler per turn.
        city_defense_maintenance: Gold upkeep for city defenses per turn.
        purchase_multiplier: Multiplier applied to production cost when
            buying a unit or building with gold.
    """

    starting_gold: int = 0
    gold_per_city_base: int = 3
    gold_per_population: int = 1
    unit_maintenance_warrior: int = 1
    unit_maintenance_archer: int = 1
    unit_maintenance_settler: int = 0
    city_defense_maintenance: int = 0
    purchase_multiplier: float = 4.0


@dataclass(frozen=True)
class CityGrowthConfig:
    """Configuration for the city growth system.

    Controls food consumption, surplus accumulation, housing limits,
    and population growth thresholds.

    Attributes:
        food_per_population: Food consumed per population point per turn.
        food_surplus_growth_rate: Fraction of surplus food applied to
            the growth bucket each turn.
        base_housing: Starting housing capacity for a new city.
        growth_threshold_base: Base food required for first population growth.
        growth_threshold_scaling: Additional food required per existing
            population point for the next growth.
        max_population: Hard cap on city population.
        food_per_turn_base: Base food generated by a city per turn
            before terrain yields.
        production_per_population: Production points generated per
            population point per turn.
    """

    food_per_population: int = 2
    food_surplus_growth_rate: float = 1.0
    base_housing: int = 3
    growth_threshold_base: int = 15
    growth_threshold_scaling: int = 8
    max_population: int = 15
    food_per_turn_base: int = 2
    production_per_population: int = 1


def _default_terrain_yields() -> dict[str, dict[str, int]]:
    """Return default terrain yield mapping.

    Each terrain type maps to a dictionary of resource yields (food,
    production, gold). These values mirror the simplified MicroCiv
    terrain model.

    Returns:
        Dictionary mapping terrain names to yield dictionaries.
    """
    return {
        "plains": {"food": 2, "production": 1, "gold": 0},
        "hills": {"food": 1, "production": 2, "gold": 0},
        "forest": {"food": 1, "production": 1, "gold": 0},
        "mountain": {"food": 0, "production": 0, "gold": 0},
        "coast": {"food": 1, "production": 0, "gold": 1},
        "desert": {"food": 0, "production": 0, "gold": 0},
        "grassland": {"food": 3, "production": 0, "gold": 0},
        "tundra": {"food": 1, "production": 0, "gold": 0},
    }


@dataclass(frozen=True)
class TerrainConfig:
    """Configuration for terrain movement and yield effects.

    Movement costs are in movement points. A cost of -1 indicates
    impassable terrain. Yields define per-tile resource generation.

    Attributes:
        movement_cost_plains: Movement cost to enter a plains tile.
        movement_cost_hills: Movement cost to enter a hills tile.
        movement_cost_forest: Movement cost to enter a forest tile.
        movement_cost_mountain: Movement cost to enter a mountain tile.
            A value of -1 means the tile is impassable.
        yields: Mapping of terrain type to resource yields. Each
            terrain type maps to a dict with keys 'food', 'production',
            and 'gold'.
    """

    movement_cost_plains: int = 1
    movement_cost_hills: int = 2
    movement_cost_forest: int = 2
    movement_cost_mountain: int = -1
    yields: dict[str, dict[str, int]] = field(default_factory=_default_terrain_yields)


@dataclass(frozen=True)
class VictoryConfig:
    """Configuration for victory conditions and scoring.

    Controls which victory types are active and how score is calculated
    for the score-based victory condition.

    Attributes:
        enable_elimination: If True, eliminating all enemy units and
            cities triggers a victory.
        enable_score: If True, the game can end by score comparison
            at the turn limit.
        score_unit_value: Score points awarded per surviving unit.
        score_city_value: Score points awarded per owned city.
        score_population_value: Score points awarded per population point.
        score_gold_value_divisor: Gold is divided by this value to
            produce score points (e.g., 50 gold / 50 = 1 point).
        score_tech_value: Score points awarded per researched technology.
    """

    enable_elimination: bool = True
    enable_score: bool = True
    score_unit_value: int = 1
    score_city_value: int = 5
    score_population_value: int = 2
    score_gold_value_divisor: int = 50
    score_tech_value: int = 3


@dataclass(frozen=True)
class GameplayConfig:
    """Master gameplay configuration containing all subsystem configs.

    This is the single source of truth for every gameplay parameter.
    Feature flags control which subsystems are active. When a feature
    is disabled, the corresponding config is still present but ignored
    by the game engine. This allows toggling features without losing
    parameter values.

    Attributes:
        enable_fog_of_war: Activate the fog of war visibility system.
            Defaults to False for backward compatibility.
        enable_city_growth: Activate the city population growth system.
            Defaults to False for backward compatibility.
        enable_economy: Activate the gold economy system.
            Defaults to False for backward compatibility.
        enable_enhanced_combat: Activate terrain bonuses, flanking,
            and ranged combat rules. Defaults to False for backward
            compatibility.
        enable_terrain_effects: Activate terrain movement costs and
            yields. Defaults to False for backward compatibility.
        num_players: Number of players in the game.
        fog_of_war: Fog of war visibility parameters.
        combat: Combat system parameters.
        economy: Economy system parameters.
        city_growth: City growth system parameters.
        terrain: Terrain movement and yield parameters.
        victory: Victory condition and scoring parameters.
    """

    enable_fog_of_war: bool = False
    enable_city_growth: bool = False
    enable_economy: bool = False
    enable_enhanced_combat: bool = False
    enable_terrain_effects: bool = False
    num_players: int = 2
    fog_of_war: FogOfWarConfig = field(default_factory=FogOfWarConfig)
    combat: CombatConfig = field(default_factory=CombatConfig)
    economy: EconomyConfig = field(default_factory=EconomyConfig)
    city_growth: CityGrowthConfig = field(default_factory=CityGrowthConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    victory: VictoryConfig = field(default_factory=VictoryConfig)

    @classmethod
    def default(cls) -> GameplayConfig:
        """Create a default configuration with all features disabled.

        This produces a configuration that is fully backward compatible
        with the original MicroCiv environment. All feature flags are
        set to False, and all subsystem configs use their defaults.

        Returns:
            A GameplayConfig with all features disabled.
        """
        logger.debug("Creating default GameplayConfig (all features disabled)")
        return cls()

    @classmethod
    def enhanced(cls) -> GameplayConfig:
        """Create a configuration with all gameplay features enabled.

        Suitable for training with the full gameplay system active.
        All subsystem configs use their defaults but every feature
        flag is turned on.

        Returns:
            A GameplayConfig with all features enabled.
        """
        logger.debug("Creating enhanced GameplayConfig (all features enabled)")
        return cls(
            enable_fog_of_war=True,
            enable_city_growth=True,
            enable_economy=True,
            enable_enhanced_combat=True,
            enable_terrain_effects=True,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameplayConfig:
        """Create a GameplayConfig from a dictionary.

        Nested dictionaries are automatically converted to their
        corresponding dataclass types. Unknown keys at any level
        are logged as warnings and ignored.

        Args:
            data: Dictionary of configuration values. May contain
                nested dictionaries for subsystem configs.

        Returns:
            A fully populated GameplayConfig.
        """
        logger.debug("Creating GameplayConfig from dict with %d keys", len(data))

        nested_mapping: dict[str, type] = {
            "fog_of_war": FogOfWarConfig,
            "combat": CombatConfig,
            "economy": EconomyConfig,
            "city_growth": CityGrowthConfig,
            "terrain": TerrainConfig,
            "victory": VictoryConfig,
        }

        # Collect valid top-level field names
        valid_top_level = {f.name for f in fields(cls)}

        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key not in valid_top_level:
                logger.warning("Unknown GameplayConfig key ignored: %s", key)
                continue

            if key in nested_mapping and isinstance(value, dict):
                nested_cls = nested_mapping[key]
                nested_valid = {f.name for f in fields(nested_cls)}
                filtered = {}
                for nested_key, nested_value in value.items():
                    if nested_key not in nested_valid:
                        logger.warning(
                            "Unknown %s key ignored: %s", nested_cls.__name__, nested_key
                        )
                        continue
                    filtered[nested_key] = nested_value
                kwargs[key] = nested_cls(**filtered)
            else:
                kwargs[key] = value

        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this configuration to a plain dictionary.

        All nested dataclasses are converted to dictionaries
        recursively. The result is suitable for JSON or YAML
        serialization and can be restored with ``from_dict``.

        Returns:
            A nested dictionary representation of this config.
        """
        return asdict(self)
