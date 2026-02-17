"""Multiplayer support for Civ6 AlphaZero.

This module extends the core AlphaZero implementation with N-player
multiplayer capabilities including:

- N-player MCTS with Max-N and Paranoid search variants
- Multi-player value heads for neural networks
- Environment adapters for N-player games
- Turn management (sequential, simultaneous, dynamic)
- Opponent modeling and scoring utilities
- Game connectors for Civ6/Unciv multiplayer integration

Architecture:
    All components are backward-compatible with existing 2-player code.
    The module uses Protocol-based interfaces and configurable dataclasses
    to ensure composability and testability.

Example:
    >>> from alpha_zero.multiplayer import (
    ...     MultiplayerConfig,
    ...     NPlayerMCTS,
    ...     NPlayerValueHead,
    ...     TurnManager,
    ... )
    >>> config = MultiplayerConfig(num_players=4)
    >>> mcts = NPlayerMCTS(network, config.mcts)
"""

from alpha_zero.multiplayer.config import (
    MultiplayerConfig,
    NPlayerMCTSConfig,
    OpponentModelConfig,
    ScoringConfig,
    SearchStrategy,
    TurnManagerConfig,
    TurnMode,
)
from alpha_zero.multiplayer.env_adapter import NPlayerEnvAdapter
from alpha_zero.multiplayer.game_connector import (
    GameConnector,
    GameConnectorConfig,
    MockGameConnector,
    UncivMultiplayerConnector,
    create_game_connector,
)
from alpha_zero.multiplayer.mcts import NPlayerMCTS
from alpha_zero.multiplayer.mcts_node import NPlayerMCTSNode
from alpha_zero.multiplayer.network_heads import (
    NPlayerNetworkAdapter,
    NPlayerNetworkConfig,
    NPlayerValueHead,
    compute_nplayer_value_loss,
)
from alpha_zero.multiplayer.opponent_model import OpponentModel
from alpha_zero.multiplayer.protocols import (
    MultiplayerEnvironment,
    MultiplayerNetwork,
)
from alpha_zero.multiplayer.scoring import (
    ScoringMethod,
    compute_player_values,
    compute_rankings,
    extract_game_result,
)
from alpha_zero.multiplayer.turn_manager import TurnManager

__all__ = [
    "GameConnector",
    "GameConnectorConfig",
    "MockGameConnector",
    "MultiplayerConfig",
    "MultiplayerEnvironment",
    "MultiplayerNetwork",
    "NPlayerEnvAdapter",
    "NPlayerMCTS",
    "NPlayerMCTSConfig",
    "NPlayerMCTSNode",
    "NPlayerNetworkAdapter",
    "NPlayerNetworkConfig",
    "NPlayerValueHead",
    "OpponentModel",
    "OpponentModelConfig",
    "ScoringConfig",
    "ScoringMethod",
    "SearchStrategy",
    "TurnManager",
    "TurnManagerConfig",
    "TurnMode",
    "UncivMultiplayerConnector",
    "compute_nplayer_value_loss",
    "compute_player_values",
    "compute_rankings",
    "create_game_connector",
    "extract_game_result",
]
