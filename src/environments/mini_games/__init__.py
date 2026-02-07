"""Built-in mini-game environments for SDK testing and curriculum learning."""
from src.environments.mini_games.mini_resource import MiniResourceEnv
from src.environments.mini_games.tactical_combat import TacticalCombatEnv
from src.environments.mini_games.territory_control import TerritoryControlEnv

__all__ = ["MiniResourceEnv", "TacticalCombatEnv", "TerritoryControlEnv"]
