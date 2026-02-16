"""Gameplay systems for Civ6 AlphaZero.

Composable, configurable gameplay modules that can be integrated into
MicroCiv or full Civ6 environments. Each system operates independently
and communicates through dict-based game state interfaces.

All parameters are configurable via dataclasses with sensible defaults.
Use ``GameplayConfig.default()`` for backward-compatible mode (all
features disabled) or ``GameplayConfig.enhanced()`` for full gameplay.
"""

from alpha_zero.gameplay.city_growth import CityGrowthConfig, CityGrowthSystem
from alpha_zero.gameplay.combat import CombatConfig, CombatResult, CombatSystem
from alpha_zero.gameplay.config import (
    CityGrowthConfig as CityGrowthMasterConfig,
    CombatConfig as CombatMasterConfig,
    EconomyConfig as EconomyMasterConfig,
    FogOfWarConfig,
    GameplayConfig,
    TerrainConfig as TerrainMasterConfig,
    VictoryConfig as VictoryMasterConfig,
)
from alpha_zero.gameplay.economy import EconomyConfig, EconomySystem
from alpha_zero.gameplay.fog_of_war import FogOfWarSystem
from alpha_zero.gameplay.fog_of_war import FogOfWarConfig as FogOfWarDetailConfig
from alpha_zero.gameplay.terrain import TerrainConfig, TerrainSystem
from alpha_zero.gameplay.victory import (
    VictoryCondition,
    VictoryConfig,
    VictorySystem,
    VictoryType,
)

__all__ = [
    # Master config
    "GameplayConfig",
    "FogOfWarConfig",
    # System configs
    "CityGrowthConfig",
    "CombatConfig",
    "EconomyConfig",
    "FogOfWarDetailConfig",
    "TerrainConfig",
    "VictoryConfig",
    # Systems
    "CityGrowthSystem",
    "CombatSystem",
    "CombatResult",
    "EconomySystem",
    "FogOfWarSystem",
    "TerrainSystem",
    "VictorySystem",
    "VictoryCondition",
    "VictoryType",
]
