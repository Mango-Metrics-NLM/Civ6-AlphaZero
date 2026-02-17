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
)
from alpha_zero.gameplay.config import (
    CombatConfig as CombatMasterConfig,
)
from alpha_zero.gameplay.config import (
    EconomyConfig as EconomyMasterConfig,
)
from alpha_zero.gameplay.config import (
    FogOfWarConfig,
    GameplayConfig,
)
from alpha_zero.gameplay.config import (
    TerrainConfig as TerrainMasterConfig,
)
from alpha_zero.gameplay.config import (
    VictoryConfig as VictoryMasterConfig,
)
from alpha_zero.gameplay.economy import EconomyConfig, EconomySystem
from alpha_zero.gameplay.fog_of_war import FogOfWarConfig as FogOfWarDetailConfig
from alpha_zero.gameplay.fog_of_war import FogOfWarSystem
from alpha_zero.gameplay.terrain import TerrainConfig, TerrainSystem
from alpha_zero.gameplay.victory import (
    VictoryCondition,
    VictoryConfig,
    VictorySystem,
    VictoryType,
)

__all__ = [
    # System configs
    "CityGrowthConfig",
    # Master config aliases (from config.py — avoid name collisions with subsystem configs)
    "CityGrowthMasterConfig",
    # Systems
    "CityGrowthSystem",
    "CombatConfig",
    "CombatMasterConfig",
    # Data types
    "CombatResult",
    "CombatSystem",
    "EconomyConfig",
    "EconomyMasterConfig",
    "EconomySystem",
    "FogOfWarConfig",
    "FogOfWarDetailConfig",
    "FogOfWarSystem",
    # Master config
    "GameplayConfig",
    "TerrainConfig",
    "TerrainMasterConfig",
    "TerrainSystem",
    "VictoryCondition",
    "VictoryConfig",
    "VictoryMasterConfig",
    "VictorySystem",
    "VictoryType",
]
