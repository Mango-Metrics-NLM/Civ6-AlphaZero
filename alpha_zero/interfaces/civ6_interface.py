"""Civ6 game interface implementation.

This module provides a BaseGameInterface implementation for real
Civilization VI gameplay through the bridge system.

The interface wraps RealCiv6Env to provide compatibility with the
distributed training system and other components that use BaseGameInterface.

Usage:
    from alpha_zero.interfaces.civ6_interface import Civ6Interface
    from alpha_zero.interfaces.base_interface import GameConfig

    config = GameConfig(board_size=16)
    interface = Civ6Interface(config)

    if interface.connect():
        obs = interface.reset()
        result = interface.step(action)
        interface.disconnect()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.envs.real_civ6_env import MockCiv6Env, RealCiv6Env
from alpha_zero.interfaces.base_interface import (
    BaseGameInterface,
    GameConfig,
    GameObservation,
    GameStatus,
    StepResult,
)

logger = logging.getLogger(__name__)


class Civ6Interface(BaseGameInterface):
    """Game interface for real Civilization VI.

    This class implements the BaseGameInterface contract for compatibility
    with the distributed training system. It wraps RealCiv6Env to provide
    the required interface methods.

    Attributes:
        config: Game configuration.
        bridge_config: Bridge-specific configuration.

    Example:
        >>> config = GameConfig(board_size=16)
        >>> interface = Civ6Interface(config)
        >>> interface.connect()
        True
        >>> obs = interface.reset()
        >>> result = interface.step(action)
        >>> interface.disconnect()
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        bridge_config: BridgeConfig | None = None,
        use_mock: bool = False,
    ):
        """Initialize Civ6 interface.

        Args:
            config: Game configuration.
            bridge_config: Bridge configuration.
            use_mock: If True, use MockCiv6Env for testing.
        """
        # Set appropriate defaults for Civ6
        if config is None:
            config = GameConfig(
                board_size=16,
                max_players=2,
                max_turns=500,
                observation_channels=84,
                vector_dim=512,
                action_space_size=None,  # Will be set from env
            )

        super().__init__(config)

        self.bridge_config = bridge_config or BridgeConfig()
        self._use_mock = use_mock
        self._env: RealCiv6Env | MockCiv6Env | None = None
        self._last_obs: GameObservation | None = None
        self._game_status = GameStatus.NOT_STARTED

    def connect(self) -> bool:
        """Establish connection to Civ6 game.

        Creates the environment and initializes the bridge.
        The actual game must be running with the Lua mod installed.

        Returns:
            True if connection successful.
        """
        try:
            logger.info("Connecting to Civ6 game...")

            # Create environment
            env_class = MockCiv6Env if self._use_mock else RealCiv6Env
            self._env = env_class(
                bridge_config=self.bridge_config,
                board_size=self.config.board_size,
            )

            # Update action space size from environment
            self.config.action_space_size = self._env.action_space.n

            self._connected = True
            self._game_status = GameStatus.NOT_STARTED
            logger.info("Connected to Civ6 successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Civ6: {e}")
            self._connected = False
            self._game_status = GameStatus.ERROR
            return False

    def disconnect(self) -> None:
        """Close connection to Civ6 game."""
        logger.info("Disconnecting from Civ6...")

        if self._env is not None:
            self._env.close()
            self._env = None

        self._connected = False
        self._last_obs = None
        logger.info("Disconnected from Civ6")

    def reset(self, seed: int | None = None) -> GameObservation:
        """Reset game to initial state.

        Waits for the game to provide initial state. The actual
        game reset must be triggered externally in Civ6.

        Args:
            seed: Random seed (not used for real game).

        Returns:
            Initial GameObservation.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or self._env is None:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            obs_dict, info = self._env.reset(seed=seed)
            self._game_status = GameStatus.IN_PROGRESS
            self._last_obs = self._dict_to_observation(obs_dict, info)
            return self._last_obs

        except Exception as e:
            logger.error(f"Reset failed: {e}")
            self._game_status = GameStatus.ERROR
            raise

    def step(self, action: int) -> StepResult:
        """Take an action in the game.

        Args:
            action: Action index to execute.

        Returns:
            StepResult with observation, reward, and done flag.

        Raises:
            RuntimeError: If not connected.
        """
        if not self._connected or self._env is None:
            raise RuntimeError("Not connected. Call connect() first.")

        try:
            obs_dict, reward, terminated, truncated, info = self._env.step(action)

            # Update game status
            if terminated:
                # Determine winner (player 0 is always the AI)
                if info.get("winner_id") == 0:
                    self._game_status = GameStatus.PLAYER1_WIN
                else:
                    self._game_status = GameStatus.PLAYER2_WIN
            elif truncated:
                self._game_status = GameStatus.DRAW

            self._last_obs = self._dict_to_observation(obs_dict, info)

            return StepResult(
                observation=self._last_obs,
                reward=float(reward),
                done=terminated,
                truncated=truncated,
                info=info,
            )

        except Exception as e:
            logger.error(f"Step failed: {e}")
            self._game_status = GameStatus.ERROR
            raise

    def get_observation(self) -> GameObservation:
        """Get current game observation.

        Returns:
            Current GameObservation.
        """
        if self._last_obs is None:
            # Return empty observation
            return GameObservation(
                spatial=np.zeros(
                    (self.config.observation_channels,
                     self.config.board_size,
                     self.config.board_size),
                    dtype=np.float32,
                ),
                vector=np.zeros(self.config.vector_dim, dtype=np.float32),
                action_mask=np.zeros(
                    self.config.action_space_size or 1000, dtype=np.float32
                ),
            )
        return self._last_obs

    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions.

        Returns:
            Boolean array where True indicates valid action.
        """
        if self._last_obs is None:
            return np.zeros(self.config.action_space_size or 1000, dtype=bool)
        return self._last_obs.action_mask > 0

    def get_game_status(self) -> GameStatus:
        """Get current game status.

        Returns:
            GameStatus enum value.
        """
        return self._game_status

    def get_current_player(self) -> int:
        """Get index of current player.

        Returns:
            Player index (always 0 for single-player AI).
        """
        if self._last_obs is not None:
            return self._last_obs.current_player
        return 0

    def save_state(self, path: str | Path) -> None:
        """Save current game state to file.

        Note: This saves the Python-side state. The actual Civ6
        save must be triggered separately in the game.

        Args:
            path: Path to save state.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._env is None:
            raise RuntimeError("Not connected. Cannot save state.")

        state = {
            "turn": self._last_obs.turn if self._last_obs else 0,
            "status": self._game_status.value,
            "config": self.config.to_dict(),
        }

        with open(path, "w") as f:
            json.dump(state, f)

        logger.info(f"State saved to {path}")

    def load_state(self, path: str | Path) -> GameObservation:
        """Load game state from file.

        Note: This only restores Python-side state. The Civ6 game
        must be loaded separately.

        Args:
            path: Path to load from.

        Returns:
            Observation after loading.
        """
        path = Path(path)

        with open(path) as f:
            state = json.load(f)

        self._game_status = GameStatus(state.get("status", GameStatus.NOT_STARTED.value))

        # The actual game state comes from the Civ6 game
        # Just wait for next state update
        return self.get_observation()

    def render(self, mode: str = "human") -> np.ndarray | None:
        """Render the game state.

        Args:
            mode: Render mode ("human" prints to console).

        Returns:
            None for "human" mode.
        """
        if self._env is not None:
            return self._env.render()
        return None

    def close(self) -> None:
        """Clean up resources."""
        self.disconnect()

    def _dict_to_observation(
        self,
        obs_dict: dict[str, np.ndarray],
        info: dict[str, Any],
    ) -> GameObservation:
        """Convert observation dictionary to GameObservation.

        Args:
            obs_dict: Observation dictionary from environment.
            info: Info dictionary from environment.

        Returns:
            GameObservation instance.
        """
        return GameObservation(
            spatial=obs_dict["spatial"],
            vector=obs_dict["vector"],
            action_mask=obs_dict["action_mask"],
            current_player=0,  # Always player 0 for AI
            turn=info.get("turn", 0),
            done=self._game_status in (
                GameStatus.PLAYER1_WIN,
                GameStatus.PLAYER2_WIN,
                GameStatus.DRAW,
            ),
            info=info,
        )


def _create_mock_civ6_interface(config: GameConfig | None = None) -> Civ6Interface:
    """Factory function for creating mock Civ6 interface.

    Args:
        config: Game configuration.

    Returns:
        Civ6Interface with use_mock=True.
    """
    return Civ6Interface(config, use_mock=True)


# Register with GameAdapter
def register_civ6_backend():
    """Register Civ6 backend with GameAdapter.

    Call this to make "civ6" available as a backend option.
    """
    from alpha_zero.interfaces.base_interface import GameAdapter
    GameAdapter.register_backend("civ6", Civ6Interface)
    GameAdapter.register_backend("civ6_mock", _create_mock_civ6_interface)
