"""Game connectors for Civ6/Unciv multiplayer integration.

Provides abstract and concrete connectors for interfacing AlphaZero
agents with actual game instances in multiplayer settings.

Supported integration paths:
1. Unciv JSON save files (highest reliability, async turns)
2. Civ6 Lua mod file-based IPC (moderate reliability, needs mod)
3. Abstract connector for future direct API integration

Example:
    >>> config = GameConnectorConfig(
    ...     connector_type="unciv",
    ...     save_dir="/path/to/unciv/saves",
    ...     player_id=0,
    ... )
    >>> connector = create_game_connector(config)
    >>> state = connector.read_game_state()
    >>> connector.submit_action(state, action=42)
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GameConnectorConfig:
    """Configuration for game connectors.

    Attributes:
        connector_type: Type of connector ('unciv', 'civ6_lua', 'mock').
        save_dir: Directory for save files / IPC files.
        player_id: This agent's player ID in the multiplayer game.
        poll_interval_seconds: How often to check for new game state.
        timeout_seconds: Maximum wait time for game state updates.
        num_players: Number of players in the multiplayer game.
        game_id: Unique identifier for the multiplayer game session.
    """

    connector_type: str = "unciv"
    save_dir: str = ""
    player_id: int = 0
    poll_interval_seconds: float = 1.0
    timeout_seconds: float = 300.0
    num_players: int = 2
    game_id: str = ""


class GameConnector(ABC):
    """Abstract base class for multiplayer game connectors.

    Defines the interface for reading game state and submitting
    actions to an external game instance (Civ6, Unciv, etc.).
    """

    def __init__(self, config: GameConnectorConfig) -> None:
        """Initialize connector.

        Args:
            config: Connector configuration.
        """
        self.config = config
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Whether the connector is actively connected."""
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the game.

        Returns:
            True if connection was successful.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the game."""
        ...

    @abstractmethod
    def read_game_state(self) -> dict[str, Any] | None:
        """Read current game state.

        Returns:
            Game state dictionary, or None if not available.
        """
        ...

    @abstractmethod
    def submit_action(
        self,
        game_state: dict[str, Any],
        action: int,
    ) -> bool:
        """Submit an action to the game.

        Args:
            game_state: Current game state for context.
            action: Action index to execute.

        Returns:
            True if action was accepted.
        """
        ...

    @abstractmethod
    def wait_for_turn(self) -> dict[str, Any] | None:
        """Block until it's this agent's turn.

        Returns:
            Game state when it's our turn, or None on timeout.
        """
        ...

    @abstractmethod
    def is_game_over(self) -> bool:
        """Check if the game has ended.

        Returns:
            True if the game is over.
        """
        ...

    def __enter__(self) -> GameConnector:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.disconnect()


class UncivMultiplayerConnector(GameConnector):
    """Connector for Unciv multiplayer via JSON save files.

    Unciv stores game state as JSON, making it ideal for programmatic
    interaction. This connector monitors save files for updates and
    modifies them to submit actions.

    Integration flow:
    1. Monitor save directory for updated .json files
    2. Parse JSON to extract game state
    3. Convert to internal representation
    4. Submit actions by modifying the save file
    5. Signal the game to reload
    """

    def __init__(self, config: GameConnectorConfig) -> None:
        super().__init__(config)
        self._save_path = Path(config.save_dir)
        self._last_state_hash: str = ""
        self._game_over = False

    def connect(self) -> bool:
        """Verify save directory exists and is accessible.

        Returns:
            True if directory is accessible.
        """
        if not self._save_path.exists():
            logger.warning("Save directory does not exist: %s", self._save_path)
            self._connected = False
            return False

        self._connected = True
        logger.info(
            "Connected to Unciv multiplayer",
            extra={
                "save_dir": str(self._save_path),
                "player_id": self.config.player_id,
                "game_id": self.config.game_id,
            },
        )
        return True

    def disconnect(self) -> None:
        """Disconnect from the game."""
        self._connected = False
        logger.info("Disconnected from Unciv multiplayer")

    def read_game_state(self) -> dict[str, Any] | None:
        """Read the latest game state from save files.

        Looks for the most recent .json file matching the game ID.

        Returns:
            Parsed game state dict, or None if not found.
        """
        if not self._connected:
            logger.warning("Not connected; call connect() first")
            return None

        save_file = self._find_save_file()
        if save_file is None:
            return None

        try:
            with open(save_file, encoding="utf-8") as f:
                raw_state = json.load(f)

            # Extract relevant state
            state = self._parse_unciv_state(raw_state)
            state["_raw"] = raw_state
            state["_save_file"] = str(save_file)

            return state

        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read save file %s: %s", save_file, e)
            return None

    def submit_action(
        self,
        game_state: dict[str, Any],
        action: int,
    ) -> bool:
        """Submit an action by modifying the save file.

        Args:
            game_state: Current game state (must contain _raw and _save_file).
            action: Action index to execute.

        Returns:
            True if action was written successfully.
        """
        if not self._connected:
            return False

        raw_state = game_state.get("_raw")
        save_file = game_state.get("_save_file")
        if raw_state is None or save_file is None:
            logger.error("Game state missing _raw or _save_file")
            return False

        try:
            # Apply action to raw state
            modified = self._apply_action_to_state(raw_state, action)

            # Write modified state
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(modified, f, indent=2)

            logger.info("Action %d submitted to %s", action, save_file)
            return True

        except (OSError, TypeError) as e:
            logger.error("Failed to submit action: %s", e)
            return False

    def wait_for_turn(self) -> dict[str, Any] | None:
        """Poll for game state changes until it's our turn.

        Returns:
            Game state when it's our turn, or None on timeout.
        """
        start = time.monotonic()
        while time.monotonic() - start < self.config.timeout_seconds:
            state = self.read_game_state()
            if state is not None:
                current_player = state.get("current_player", -1)
                if current_player == self.config.player_id:
                    return state

                # Check if game is over
                if state.get("game_over", False):
                    self._game_over = True
                    return state

            time.sleep(self.config.poll_interval_seconds)

        logger.warning("Timed out waiting for turn")
        return None

    def is_game_over(self) -> bool:
        """Check if the game has ended."""
        return self._game_over

    def _find_save_file(self) -> Path | None:
        """Find the most recent save file for this game.

        Returns:
            Path to the save file, or None.
        """
        pattern = f"*{self.config.game_id}*" if self.config.game_id else "*.json"
        files = sorted(
            self._save_path.glob(pattern),
            key=lambda f: f.stat().st_mtime if f.exists() else 0,
            reverse=True,
        )
        return files[0] if files else None

    def _parse_unciv_state(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Parse Unciv JSON into internal state format.

        Args:
            raw: Raw Unciv save JSON.

        Returns:
            Parsed state dictionary.
        """
        state: dict[str, Any] = {
            "current_player": self._extract_current_player(raw),
            "turn_number": raw.get("turns", 0),
            "num_players": len(raw.get("civilizations", [])),
            "game_over": raw.get("gameOver", False),
            "winner": raw.get("winner"),
        }

        # Extract civilization data
        civs = raw.get("civilizations", [])
        state["players"] = []
        for i, civ in enumerate(civs):
            state["players"].append({
                "id": i,
                "name": civ.get("civName", f"Player_{i}"),
                "is_alive": not civ.get("isDefeated", False),
                "gold": civ.get("gold", 0),
                "cities": len(civ.get("cities", [])),
            })

        return state

    def _extract_current_player(self, raw: dict[str, Any]) -> int:
        """Extract current player index from Unciv state.

        Args:
            raw: Raw Unciv save JSON.

        Returns:
            Current player index.
        """
        current_civ = raw.get("currentPlayer", "")
        civs = raw.get("civilizations", [])
        for i, civ in enumerate(civs):
            if civ.get("civName") == current_civ:
                return i
        return 0

    def _apply_action_to_state(
        self,
        raw_state: dict[str, Any],
        action: int,
    ) -> dict[str, Any]:
        """Apply an action to the raw game state.

        This is a stub - actual implementation requires the action
        decoder to translate action indices to Unciv-compatible
        modifications (unit movements, city production, etc.).

        Args:
            raw_state: Raw Unciv save JSON.
            action: Action index.

        Returns:
            Modified raw state.
        """
        # TODO: Integrate with action decoder for full implementation
        logger.debug("Applying action %d to Unciv state (stub)", action)
        return raw_state


class MockGameConnector(GameConnector):
    """Mock connector for testing multiplayer integration.

    Simulates game state updates and action submission without
    connecting to an actual game instance.
    """

    def __init__(
        self,
        config: GameConnectorConfig,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self._state: dict[str, Any] = initial_state or {
            "current_player": 0,
            "turn_number": 0,
            "num_players": config.num_players,
            "game_over": False,
            "winner": None,
        }
        self._actions: list[tuple[int, int]] = []  # (turn, action) pairs

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def read_game_state(self) -> dict[str, Any] | None:
        if not self._connected:
            return None
        return self._state.copy()

    def submit_action(
        self,
        game_state: dict[str, Any],
        action: int,
    ) -> bool:
        if not self._connected:
            return False
        turn = self._state.get("turn_number", 0)
        self._actions.append((turn, action))

        # Advance to next player
        current = self._state.get("current_player", 0)
        num_players = self._state.get("num_players", 2)
        self._state["current_player"] = (current + 1) % num_players

        # Advance turn when wrapping
        if self._state["current_player"] == 0:
            self._state["turn_number"] = turn + 1

        return True

    def wait_for_turn(self) -> dict[str, Any] | None:
        if self._state.get("current_player") == self.config.player_id:
            return self._state.copy()
        return None

    def is_game_over(self) -> bool:
        return self._state.get("game_over", False)


def create_game_connector(config: GameConnectorConfig) -> GameConnector:
    """Factory function to create the appropriate game connector.

    Args:
        config: Connector configuration.

    Returns:
        Configured GameConnector instance.

    Raises:
        ValueError: If connector_type is unknown.
    """
    connectors: dict[str, type[GameConnector]] = {
        "unciv": UncivMultiplayerConnector,
        "mock": MockGameConnector,
    }

    connector_cls = connectors.get(config.connector_type)
    if connector_cls is None:
        available = ", ".join(sorted(connectors.keys()))
        msg = f"Unknown connector_type '{config.connector_type}'. Available: {available}"
        raise ValueError(msg)

    return connector_cls(config)
