"""Configuration for Civ6 bridge system.

This module defines configuration dataclasses for the file-based IPC
bridge between Python and the Civ6 Lua mod.

Usage:
    from alpha_zero.bridge.config import BridgeConfig

    config = BridgeConfig(poll_interval_ms=20)
    # Or load from environment:
    config = BridgeConfig.from_env()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _get_default_comm_dir() -> Path:
    """Get the default communication directory for Civ6.

    On Windows, this is typically:
    Documents/My Games/Sid Meier's Civilization VI/AlphaZeroAI/

    Returns:
        Path to the communication directory.
    """
    # Check for Windows Documents folder
    if os.name == "nt":
        documents = Path.home() / "Documents"
        # Also check OneDrive Documents
        onedrive_docs = Path.home() / "OneDrive" / "Documents"

        # Prefer the one that has the Civ6 folder
        civ6_path = "My Games/Sid Meier's Civilization VI"
        if (onedrive_docs / civ6_path).exists():
            base = onedrive_docs
        elif (documents / civ6_path).exists():
            base = documents
        else:
            # Default to regular Documents, Civ6 may not be installed yet
            base = documents

        return base / civ6_path / "AlphaZeroAI"
    else:
        # Linux/Mac - Civ6 typically uses ~/.local/share or similar
        # For now, use a generic path that can be overridden
        return Path.home() / ".civ6_alphazero"


@dataclass
class BridgeConfig:
    """Configuration for the Civ6 file-based IPC bridge.

    Attributes:
        comm_dir: Directory for IPC files (state.json, commands.json, etc.)
        poll_interval_ms: How often to poll for new state files (milliseconds)
        command_timeout_ms: How long to wait for command acknowledgment
        heartbeat_timeout_s: Consider game disconnected after this many seconds
        max_retries: Maximum reconnection attempts on error (reserved for future use)
        retry_delay_s: Base delay between retries (reserved for future use)
        board_size: Internal board size for coordinate scaling
        log_level: Logging verbosity level
        debug_mode: Enable debug features (extra logging, state dumps)

    Note:
        The max_retries and retry_delay_s fields are defined for future
        automatic reconnection logic but are not currently used by FileBridge.
        Implement retry logic in the calling code if needed.
    """

    comm_dir: Path = field(default_factory=_get_default_comm_dir)
    poll_interval_ms: int = 10
    command_timeout_ms: int = 5000
    heartbeat_timeout_s: float = 2.0
    max_retries: int = 5
    retry_delay_s: float = 1.0
    board_size: int = 16
    log_level: str = "INFO"
    debug_mode: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Ensure comm_dir is a Path object
        if isinstance(self.comm_dir, str):
            self.comm_dir = Path(self.comm_dir)

        # Validate ranges
        if self.poll_interval_ms < 1:
            raise ValueError("poll_interval_ms must be at least 1")
        if self.command_timeout_ms < 100:
            raise ValueError("command_timeout_ms must be at least 100")
        if self.heartbeat_timeout_s < 0.5:
            raise ValueError("heartbeat_timeout_s must be at least 0.5")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.board_size < 8 or self.board_size > 128:
            raise ValueError("board_size must be between 8 and 128")
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ValueError("log_level must be DEBUG, INFO, WARNING, or ERROR")

    @classmethod
    def from_env(cls) -> BridgeConfig:
        """Create configuration from environment variables.

        Environment variables:
            CIV6_BRIDGE_COMM_DIR: Communication directory path
            CIV6_BRIDGE_POLL_MS: Polling interval in milliseconds
            CIV6_BRIDGE_COMMAND_TIMEOUT_MS: Command acknowledgment timeout
            CIV6_BRIDGE_HEARTBEAT_TIMEOUT_S: Heartbeat timeout in seconds
            CIV6_BRIDGE_MAX_RETRIES: Maximum retry attempts
            CIV6_BRIDGE_RETRY_DELAY_S: Base retry delay in seconds
            CIV6_BOARD_SIZE: Internal board size
            CIV6_LOG_LEVEL: Logging level
            CIV6_DEBUG_MODE: Enable debug mode (1/true/yes)

        Returns:
            BridgeConfig instance with values from environment.
        """
        def get_bool(key: str, default: bool) -> bool:
            val = os.environ.get(key, "").lower()
            if val in ("1", "true", "yes"):
                return True
            if val in ("0", "false", "no"):
                return False
            return default

        kwargs = {}

        if comm_dir := os.environ.get("CIV6_BRIDGE_COMM_DIR"):
            kwargs["comm_dir"] = Path(comm_dir)

        if poll_ms := os.environ.get("CIV6_BRIDGE_POLL_MS"):
            kwargs["poll_interval_ms"] = int(poll_ms)

        if cmd_timeout := os.environ.get("CIV6_BRIDGE_COMMAND_TIMEOUT_MS"):
            kwargs["command_timeout_ms"] = int(cmd_timeout)

        if hb_timeout := os.environ.get("CIV6_BRIDGE_HEARTBEAT_TIMEOUT_S"):
            kwargs["heartbeat_timeout_s"] = float(hb_timeout)

        if max_retries := os.environ.get("CIV6_BRIDGE_MAX_RETRIES"):
            kwargs["max_retries"] = int(max_retries)

        if retry_delay := os.environ.get("CIV6_BRIDGE_RETRY_DELAY_S"):
            kwargs["retry_delay_s"] = float(retry_delay)

        if board_size := os.environ.get("CIV6_BOARD_SIZE"):
            kwargs["board_size"] = int(board_size)

        if log_level := os.environ.get("CIV6_LOG_LEVEL"):
            kwargs["log_level"] = log_level.upper()

        kwargs["debug_mode"] = get_bool("CIV6_DEBUG_MODE", False)

        return cls(**kwargs)

    @property
    def state_file(self) -> Path:
        """Path to the state.json file."""
        return self.comm_dir / "state.json"

    @property
    def commands_file(self) -> Path:
        """Path to the commands.json file."""
        return self.comm_dir / "commands.json"

    @property
    def ack_file(self) -> Path:
        """Path to the ack.json file."""
        return self.comm_dir / "ack.json"

    @property
    def errors_file(self) -> Path:
        """Path to the errors.json file."""
        return self.comm_dir / "errors.json"

    @property
    def heartbeat_file(self) -> Path:
        """Path to the heartbeat.txt file."""
        return self.comm_dir / "heartbeat.txt"

    def ensure_comm_dir(self) -> None:
        """Create the communication directory if it doesn't exist."""
        self.comm_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "comm_dir": str(self.comm_dir),
            "poll_interval_ms": self.poll_interval_ms,
            "command_timeout_ms": self.command_timeout_ms,
            "heartbeat_timeout_s": self.heartbeat_timeout_s,
            "max_retries": self.max_retries,
            "retry_delay_s": self.retry_delay_s,
            "board_size": self.board_size,
            "log_level": self.log_level,
            "debug_mode": self.debug_mode,
        }


@dataclass
class MapSizeConfig:
    """Configuration for handling different Civ6 map sizes.

    Civ6 has various map sizes that need to be scaled to the internal
    board_size for the neural network.

    Attributes:
        name: Map size name (Duel, Tiny, Small, Standard, Large, Huge)
        width: Map width in tiles
        height: Map height in tiles
    """

    name: str
    width: int
    height: int

    @property
    def total_tiles(self) -> int:
        """Total number of tiles on the map."""
        return self.width * self.height

    def get_scale_factor(self, board_size: int) -> tuple[float, float]:
        """Get scale factors for converting coordinates.

        Args:
            board_size: Target board size for the neural network.

        Returns:
            Tuple of (x_scale, y_scale) factors.
        """
        return (board_size / self.width, board_size / self.height)


# Standard Civ6 map sizes
MAP_SIZES = {
    "duel": MapSizeConfig("Duel", 44, 26),
    "tiny": MapSizeConfig("Tiny", 60, 38),
    "small": MapSizeConfig("Small", 74, 46),
    "standard": MapSizeConfig("Standard", 84, 54),
    "large": MapSizeConfig("Large", 96, 60),
    "huge": MapSizeConfig("Huge", 106, 66),
}


def get_map_config(width: int, height: int) -> MapSizeConfig:
    """Get or create a MapSizeConfig for the given dimensions.

    Args:
        width: Map width in tiles.
        height: Map height in tiles.

    Returns:
        MapSizeConfig matching the dimensions, or a custom one.
    """
    # Check standard sizes
    for config in MAP_SIZES.values():
        if config.width == width and config.height == height:
            return config

    # Create custom config
    return MapSizeConfig(f"Custom_{width}x{height}", width, height)
