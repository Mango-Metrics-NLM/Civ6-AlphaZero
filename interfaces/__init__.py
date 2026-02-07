"""Game interface implementations for Unciv and Civ 6.

This module provides bridges between the AlphaZero implementation and
actual game instances, supporting multiple integration paths:

- Civ6Bridge: File-based IPC with Civ6 Lua mod
- UncivInterface: JSON save file manipulation for Unciv
- Civ6LogParser: Passive observation via game log files
- MiniGameGenerator: Curriculum training scenarios

Example:
    from interfaces import Civ6Bridge, UncivInterface

    # For Civ6 with Lua mod
    bridge = Civ6Bridge(state_dir=Path("/path/to/shared"))
    state = bridge.read_state()
    bridge.send_action(action)

    # For Unciv
    unciv = UncivInterface(save_path=Path("/path/to/save.json"))
    state = unciv.load_game_state()
    unciv.apply_action(action)
"""

from .base import (
    ActionSendError,
    GameAction,
    GameInterface,
    GameState,
    InterfaceConfig,
    InterfaceError,
    StateReadError,
)
from .civ6_bridge import (
    Civ6Bridge,
    Civ6LuaExporter,
    LuaStateParser,
)
from .log_parser import (
    Civ6LogParser,
    CombatLog,
    LogEntry,
    MovementLog,
)
from .mini_games import (
    CityDevelopmentMiniGame,
    CombatMiniGame,
    ExpansionMiniGame,
    MiniGameConfig,
    MiniGameGenerator,
)
from .unciv_interface import (
    UncivActionSerializer,
    UncivInterface,
    UncivStateParser,
)

__all__ = [
    "ActionSendError",
    "CityDevelopmentMiniGame",
    # Civ6 Bridge
    "Civ6Bridge",
    # Log Parser
    "Civ6LogParser",
    "Civ6LuaExporter",
    "CombatLog",
    "CombatMiniGame",
    "ExpansionMiniGame",
    "GameAction",
    # Base
    "GameInterface",
    "GameState",
    "InterfaceConfig",
    "InterfaceError",
    "LogEntry",
    "LuaStateParser",
    "MiniGameConfig",
    # Mini Games
    "MiniGameGenerator",
    "MovementLog",
    "StateReadError",
    "UncivActionSerializer",
    # Unciv
    "UncivInterface",
    "UncivStateParser",
]
