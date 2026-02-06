"""Human vs AI interface module for Civ6 AlphaZero.

This module provides an interactive interface for humans to play
against trained AI agents.

Components:
- config: Configuration for human interface
- game_controller: Game flow and turn management
- visualizer: ASCII visualization of game state
- input_handler: Human input parsing and validation
- replay: Game save/load and replay functionality
"""

from alpha_zero.human_interface.config import (
    HumanInterfaceConfig,
    InputMode,
)
from alpha_zero.human_interface.game_controller import (
    GameController,
    GamePhase,
    TurnResult,
)
from alpha_zero.human_interface.input_handler import (
    ActionParser,
    HumanInputHandler,
    ParsedAction,
)
from alpha_zero.human_interface.replay import (
    GameRecorder,
    GameReplay,
    ReplayManager,
)
from alpha_zero.human_interface.visualizer import (
    BoardTheme,
    StateVisualizer,
)

__all__ = [
    # Input
    "ActionParser",
    # Visualizer
    "BoardTheme",
    # Controller
    "GameController",
    "GamePhase",
    # Replay
    "GameRecorder",
    "GameReplay",
    "HumanInputHandler",
    # Config
    "HumanInterfaceConfig",
    "InputMode",
    "ParsedAction",
    "ReplayManager",
    "StateVisualizer",
    "TurnResult",
]
