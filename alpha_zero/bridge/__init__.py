"""Civ6 Bridge - File-based IPC for real game communication.

This package provides the bridge system for communication between
the Python AlphaZero agent and the Civilization VI game via a Lua mod.

The bridge uses file-based IPC (polling) instead of TCP sockets because
Civ6's Lua runtime is sandboxed and blocks external libraries.

Modules:
    config: Configuration dataclasses for the bridge.
    protocol: Message types and JSON schemas.
    file_bridge: File-based IPC implementation.
    state_converter: Civ6 JSON to neural network tensor conversion.
    action_converter: Action index to Lua command conversion.

Usage:
    from alpha_zero.bridge import FileBridge, BridgeConfig

    config = BridgeConfig()
    bridge = FileBridge(config)

    # Wait for game state
    state = await bridge.poll_state()

    # Send action command
    await bridge.send_command(command)

    # Check if game is alive
    if bridge.check_heartbeat():
        logger.info("Game is connected")
"""

from alpha_zero.bridge.config import (
    MAP_SIZES,
    BridgeConfig,
    MapSizeConfig,
    get_map_config,
)
from alpha_zero.bridge.protocol import (
    AckMessage,
    ActionType,
    CityData,
    CommandMessage,
    ErrorCode,
    ErrorMessage,
    GameStateData,
    LegalAction,
    LuaCommand,
    MapData,
    MessageType,
    PlayerData,
    Position,
    StateMessage,
    TileData,
    UnitData,
    create_city_produce_command,
    create_end_turn_command,
    create_unit_move_command,
    validate_state_json,
)

__all__ = [
    "MAP_SIZES",
    "AckMessage",
    "ActionType",
    # Config
    "BridgeConfig",
    "CityData",
    "CommandMessage",
    "ErrorCode",
    "ErrorMessage",
    "GameStateData",
    "LegalAction",
    "LuaCommand",
    "MapData",
    "MapSizeConfig",
    # Protocol - Enums
    "MessageType",
    "PlayerData",
    # Protocol - Data classes
    "Position",
    # Protocol - Messages
    "StateMessage",
    "TileData",
    "UnitData",
    "create_city_produce_command",
    "create_end_turn_command",
    "create_unit_move_command",
    "get_map_config",
    # Protocol - Utilities
    "validate_state_json",
]


# Lazy imports for heavier modules
# NOTE: We use module-level __getattr__ for lazy loading to improve import times.
# FileBridge, StateConverter, and ActionConverter depend on heavier dependencies
# (asyncio, torch, numpy) that are not needed for basic protocol definitions.
# This pattern allows users who only need protocol types to import faster.
# IDE auto-completion may not recognize these lazy-loaded attributes.
def __getattr__(name: str):
    """Lazy import for FileBridge and converters.

    This function enables lazy loading of heavier modules to optimize import times.
    FileBridge, StateConverter, and ActionConverter are imported on first access
    rather than at module import time.

    Args:
        name: The attribute name being accessed.

    Returns:
        The requested class.

    Raises:
        AttributeError: If the attribute doesn't exist.
    """
    if name == "FileBridge":
        from alpha_zero.bridge.file_bridge import FileBridge

        return FileBridge
    if name == "StateConverter":
        from alpha_zero.bridge.state_converter import StateConverter

        return StateConverter
    if name == "ActionConverter":
        from alpha_zero.bridge.action_converter import ActionConverter

        return ActionConverter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
