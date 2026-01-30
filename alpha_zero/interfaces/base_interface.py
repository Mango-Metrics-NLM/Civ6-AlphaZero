"""Abstract base interface for game interactions.

This module defines the abstract interface that all game backends
(MicroCiv, Unciv, Civ6 mod) must implement.

Usage:
    from alpha_zero.interfaces.base_interface import BaseGameInterface

    class MyGameInterface(BaseGameInterface):
        def connect(self) -> bool:
            # Implementation
            pass
        ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class GameStatus(str, Enum):
    """Status of the game."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PLAYER1_WIN = "player1_win"
    PLAYER2_WIN = "player2_win"
    DRAW = "draw"
    ERROR = "error"


@dataclass
class GameObservation:
    """Observation from the game environment.

    Attributes:
        spatial: Spatial observation tensor (C, H, W)
        vector: Vector observation (global features)
        action_mask: Boolean mask of valid actions
        current_player: Index of current player
        turn: Current turn number
        done: Whether game is over
        info: Additional game information
    """

    spatial: np.ndarray
    vector: np.ndarray
    action_mask: np.ndarray
    current_player: int = 0
    turn: int = 0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (for serialization)."""
        return {
            "spatial": self.spatial.tolist(),
            "vector": self.vector.tolist(),
            "action_mask": self.action_mask.tolist(),
            "current_player": self.current_player,
            "turn": self.turn,
            "done": self.done,
            "info": self.info,
        }


@dataclass
class StepResult:
    """Result of taking an action in the game.

    Attributes:
        observation: New observation after action
        reward: Reward for the action
        done: Whether game is over
        truncated: Whether episode was truncated
        info: Additional information
    """

    observation: GameObservation
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameConfig:
    """Configuration for game interfaces.

    Attributes:
        board_size: Size of the game board
        max_players: Maximum number of players
        max_turns: Maximum turns before game ends
        observation_channels: Number of observation channels
        vector_dim: Dimension of vector observations
        action_space_size: Total number of possible actions
        seed: Random seed for reproducibility
    """

    board_size: int = 16
    max_players: int = 2
    max_turns: int = 200
    observation_channels: int = 13
    vector_dim: int = 64
    action_space_size: int = 1000
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "board_size": self.board_size,
            "max_players": self.max_players,
            "max_turns": self.max_turns,
            "observation_channels": self.observation_channels,
            "vector_dim": self.vector_dim,
            "action_space_size": self.action_space_size,
            "seed": self.seed,
        }


class BaseGameInterface(ABC):
    """Abstract base class for game interfaces.

    All game backends (MicroCiv, Unciv, Civ6 mod) must implement
    this interface to be compatible with the training system.

    Example:
        >>> class MyCivInterface(BaseGameInterface):
        ...     def connect(self) -> bool:
        ...         return True
        ...     # ... implement other methods
        >>>
        >>> interface = MyCivInterface(GameConfig())
        >>> interface.connect()
        >>> obs = interface.reset()
        >>> result = interface.step(action)
    """

    def __init__(self, config: GameConfig | None = None):
        """Initialize the game interface.

        Args:
            config: Game configuration
        """
        self.config = config or GameConfig()
        self._connected = False
        self._current_state: dict[str, Any] | None = None

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the game.

        Returns:
            True if connection successful.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the game."""
        pass

    @abstractmethod
    def reset(self, seed: int | None = None) -> GameObservation:
        """Reset game to initial state.

        Args:
            seed: Optional random seed

        Returns:
            Initial observation.
        """
        pass

    @abstractmethod
    def step(self, action: int) -> StepResult:
        """Take an action in the game.

        Args:
            action: Action index to take

        Returns:
            StepResult with observation, reward, and done flag.
        """
        pass

    @abstractmethod
    def get_observation(self) -> GameObservation:
        """Get current game observation.

        Returns:
            Current GameObservation.
        """
        pass

    @abstractmethod
    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions.

        Returns:
            Boolean array where True indicates valid action.
        """
        pass

    @abstractmethod
    def get_game_status(self) -> GameStatus:
        """Get current game status.

        Returns:
            GameStatus enum value.
        """
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        """Get index of current player.

        Returns:
            Player index (0-indexed).
        """
        pass

    @abstractmethod
    def save_state(self, path: str | Path) -> None:
        """Save current game state to file.

        Args:
            path: Path to save state
        """
        pass

    @abstractmethod
    def load_state(self, path: str | Path) -> GameObservation:
        """Load game state from file.

        Args:
            path: Path to load from

        Returns:
            Observation after loading state.
        """
        pass

    def render(self, mode: str = "human") -> np.ndarray | None:
        """Render the game state.

        Args:
            mode: Render mode ("human", "rgb_array", "ascii")

        Returns:
            Optional array for "rgb_array" mode.
        """
        # Default implementation does nothing
        return None

    def close(self) -> None:
        """Clean up resources."""
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Check if connected to game."""
        return self._connected

    @property
    def board_size(self) -> int:
        """Get board size."""
        return self.config.board_size

    @property
    def action_space_size(self) -> int:
        """Get action space size."""
        return self.config.action_space_size


class DummyGameInterface(BaseGameInterface):
    """Dummy implementation for testing.

    This provides a simple implementation that can be used
    for testing without a real game backend.
    """

    def __init__(self, config: GameConfig | None = None):
        """Initialize dummy interface."""
        super().__init__(config)
        self._turn = 0
        self._done = False
        self._rng = np.random.default_rng(config.seed if config else None)

    def connect(self) -> bool:
        """Connect (always succeeds)."""
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect."""
        self._connected = False

    def reset(self, seed: int | None = None) -> GameObservation:
        """Reset game."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._turn = 0
        self._done = False

        return self.get_observation()

    def step(self, action: int) -> StepResult:
        """Take an action."""
        self._turn += 1

        # Random game end
        if self._turn >= self.config.max_turns or self._rng.random() < 0.01:
            self._done = True

        obs = self.get_observation()
        reward = (
            self._rng.random() - 0.5
            if not self._done
            else (1.0 if self._rng.random() > 0.5 else -1.0)
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={"turn": self._turn},
        )

    def get_observation(self) -> GameObservation:
        """Get current observation."""
        spatial = self._rng.random(
            (self.config.observation_channels, self.config.board_size, self.config.board_size)
        ).astype(np.float32)

        vector = self._rng.random(self.config.vector_dim).astype(np.float32)

        action_mask = self._rng.random(self.config.action_space_size) > 0.5

        return GameObservation(
            spatial=spatial,
            vector=vector,
            action_mask=action_mask,
            current_player=self._turn % self.config.max_players,
            turn=self._turn,
            done=self._done,
        )

    def get_valid_actions(self) -> np.ndarray:
        """Get valid actions."""
        return self._rng.random(self.config.action_space_size) > 0.5

    def get_game_status(self) -> GameStatus:
        """Get game status."""
        if not self._done:
            return GameStatus.IN_PROGRESS
        return GameStatus.PLAYER1_WIN if self._rng.random() > 0.5 else GameStatus.PLAYER2_WIN

    def get_current_player(self) -> int:
        """Get current player."""
        return self._turn % self.config.max_players

    def save_state(self, path: str | Path) -> None:
        """Save state."""
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "turn": self._turn,
                    "done": self._done,
                    "config": self.config.to_dict(),
                },
                f,
            )

    def load_state(self, path: str | Path) -> GameObservation:
        """Load state."""
        import json

        with open(path) as f:
            data = json.load(f)
        self._turn = data["turn"]
        self._done = data["done"]
        return self.get_observation()


class GameAdapter:
    """Unified adapter for different game backends.

    This class provides a consistent interface regardless
    of the underlying game backend.

    Example:
        >>> adapter = GameAdapter("microciv", config)
        >>> adapter.connect()
        >>> obs = adapter.reset()
    """

    BACKENDS = {
        "dummy": DummyGameInterface,
        # "microciv": MicroCivInterface,  # To be implemented
        # "unciv": UncivInterface,  # Existing implementation
    }

    def __init__(
        self,
        backend: str = "dummy",
        config: GameConfig | None = None,
        **kwargs,
    ):
        """Initialize game adapter.

        Args:
            backend: Backend name
            config: Game configuration
            **kwargs: Additional backend-specific arguments
        """
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Available: {list(self.BACKENDS.keys())}")

        self.backend_name = backend
        self.config = config or GameConfig()

        interface_class = self.BACKENDS[backend]
        self._interface = (
            interface_class(self.config, **kwargs) if kwargs else interface_class(self.config)
        )

    def connect(self) -> bool:
        """Connect to backend."""
        return self._interface.connect()

    def disconnect(self) -> None:
        """Disconnect from backend."""
        self._interface.disconnect()

    def reset(self, seed: int | None = None) -> GameObservation:
        """Reset game."""
        return self._interface.reset(seed)

    def step(self, action: int) -> StepResult:
        """Take action."""
        return self._interface.step(action)

    def get_observation(self) -> GameObservation:
        """Get current observation."""
        return self._interface.get_observation()

    def get_valid_actions(self) -> np.ndarray:
        """Get valid actions mask."""
        return self._interface.get_valid_actions()

    def get_game_status(self) -> GameStatus:
        """Get game status."""
        return self._interface.get_game_status()

    def save_state(self, path: str | Path) -> None:
        """Save state."""
        self._interface.save_state(path)

    def load_state(self, path: str | Path) -> GameObservation:
        """Load state."""
        return self._interface.load_state(path)

    def close(self) -> None:
        """Close adapter."""
        self._interface.close()

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._interface.is_connected

    @classmethod
    def register_backend(cls, name: str, interface_class: type) -> None:
        """Register a new backend.

        Args:
            name: Backend name
            interface_class: Interface class (must inherit BaseGameInterface)
        """
        if not issubclass(interface_class, BaseGameInterface):
            raise TypeError("Interface must inherit from BaseGameInterface")
        cls.BACKENDS[name] = interface_class

    @classmethod
    def available_backends(cls) -> list[str]:
        """Get list of available backends."""
        return list(cls.BACKENDS.keys())
