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

import warnings

warnings.warn(
    "alpha_zero.human_interface is not yet active and will be replaced by "
    "alpha_zero.web for the web-based interface. This module is a placeholder.",
    DeprecationWarning,
    stacklevel=2,
)

# Activated in Phase 19: Web-based Human vs AI interface
# See alpha_zero.web for the web UI layer that uses these components.

from alpha_zero.human_interface.config import (
    HumanInterfaceConfig,
    InputMode,
)
from alpha_zero.human_interface.game_controller import (
    GameController,
    GamePhase,
    TurnResult,
)
from alpha_zero.human_interface.replay import (
    GameRecorder,
    GameReplay,
    ReplayManager,
)

__all__ = [
    # Input
    # Visualizer
    # Controller
    "GameController",
    "GamePhase",
    # Replay
    "GameRecorder",
    "GameReplay",
    # Config
    "HumanInterfaceConfig",
    "InputMode",
    "ReplayManager",
    "TurnResult",
]
