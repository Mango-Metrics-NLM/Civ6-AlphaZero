"""Replay system for game recording and playback.

This module provides save/load functionality and game replay.

Usage:
    from alpha_zero.human_interface.replay import ReplayManager

    manager = ReplayManager(config)

    # Record during game
    manager.start_recording(game_id)
    manager.record_turn(turn_result)
    manager.save_recording()

    # Replay later
    replay = manager.load_replay(game_id)
    for turn in replay.iterate_turns():
        print(f"Turn {turn.turn_number}: {turn.action}")
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from alpha_zero.human_interface.config import HumanInterfaceConfig
from alpha_zero.human_interface.game_controller import TurnResult
from alpha_zero.utils.compat import UTC

logger = logging.getLogger(__name__)


def _serialize_array(arr: np.ndarray | None) -> list | None:
    """Convert numpy array to list for JSON serialization."""
    if arr is None:
        return None
    return arr.tolist()  # type: ignore[no-any-return]


def _deserialize_array(data: list | None) -> np.ndarray | None:
    """Convert list back to numpy array."""
    if data is None:
        return None
    return np.array(data)


@dataclass
class GameRecording:
    """A complete recording of a game.

    Attributes:
        game_id: Unique game identifier
        version: Recording format version
        created_at: When recording was created
        game_config: Game configuration
        human_player: Which player was human
        initial_state: Initial game state
        turns: List of turn results
        final_result: Final game outcome
        metadata: Additional recording data
    """

    game_id: str
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    game_config: dict[str, Any] = field(default_factory=dict)
    human_player: int = 0
    initial_state: list | None = None
    turns: list[dict[str, Any]] = field(default_factory=list)
    final_result: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "game_id": self.game_id,
            "version": self.version,
            "created_at": self.created_at,
            "game_config": self.game_config,
            "human_player": self.human_player,
            "initial_state": self.initial_state,
            "turns": self.turns,
            "final_result": self.final_result,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameRecording:
        """Create from dictionary."""
        return cls(
            game_id=data["game_id"],
            version=data.get("version", "1.0"),
            created_at=data.get("created_at", ""),
            game_config=data.get("game_config", {}),
            human_player=data.get("human_player", 0),
            initial_state=data.get("initial_state"),
            turns=data.get("turns", []),
            final_result=data.get("final_result"),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        """Save recording to file.

        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Saved recording to {path}")

    @classmethod
    def load(cls, path: str | Path) -> GameRecording:
        """Load recording from file.

        Args:
            path: Path to load from

        Returns:
            GameRecording instance.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class ReplayState:
    """Current state during replay playback.

    Attributes:
        current_turn: Current turn index
        total_turns: Total number of turns
        is_playing: Whether replay is playing
        playback_speed: Speed multiplier
        current_state: Current game state array
    """

    current_turn: int = 0
    total_turns: int = 0
    is_playing: bool = False
    playback_speed: float = 1.0
    current_state: np.ndarray | None = None


class GameReplay:
    """Playback controller for game recordings.

    This class manages replay playback with navigation controls.

    Example:
        >>> replay = GameReplay.from_file("game.replay")
        >>>
        >>> # Step through replay
        >>> while not replay.is_at_end:
        ...     turn = replay.step_forward()
        ...     print(f"Turn {turn['action']}")
        >>>
        >>> # Jump to specific turn
        >>> replay.jump_to_turn(5)
    """

    def __init__(self, recording: GameRecording) -> None:
        """Initialize replay.

        Args:
            recording: Game recording to replay
        """
        self.recording = recording
        self._state = ReplayState(
            total_turns=len(recording.turns),
        )

        if recording.initial_state:
            self._state.current_state = np.array(recording.initial_state)

    @classmethod
    def from_file(cls, path: str | Path) -> GameReplay:
        """Create replay from file.

        Args:
            path: Path to recording file

        Returns:
            GameReplay instance.
        """
        recording = GameRecording.load(path)
        return cls(recording)

    @property
    def current_turn(self) -> int:
        """Get current turn index."""
        return self._state.current_turn

    @property
    def total_turns(self) -> int:
        """Get total number of turns."""
        return self._state.total_turns

    @property
    def is_at_start(self) -> bool:
        """Check if at start of replay."""
        return self._state.current_turn == 0

    @property
    def is_at_end(self) -> bool:
        """Check if at end of replay."""
        return self._state.current_turn >= self._state.total_turns

    @property
    def current_state(self) -> np.ndarray | None:
        """Get current game state."""
        return self._state.current_state

    def step_forward(self) -> dict[str, Any] | None:
        """Advance one turn forward.

        Returns:
            Turn data or None if at end.
        """
        if self.is_at_end:
            return None

        turn_data = self.recording.turns[self._state.current_turn]
        self._state.current_turn += 1

        # Update state if available
        if turn_data.get("state_after"):
            self._state.current_state = np.array(turn_data["state_after"])

        return turn_data

    def step_backward(self) -> dict[str, Any] | None:
        """Go back one turn.

        Returns:
            Turn data or None if at start.
        """
        if self.is_at_start:
            return None

        self._state.current_turn -= 1
        turn_data = self.recording.turns[self._state.current_turn]

        # Update state if available
        if turn_data.get("state_before"):
            self._state.current_state = np.array(turn_data["state_before"])

        return turn_data

    def jump_to_turn(self, turn: int) -> dict[str, Any] | None:
        """Jump to specific turn.

        Args:
            turn: Turn number to jump to

        Returns:
            Turn data or None if invalid.
        """
        if not (0 <= turn <= self._state.total_turns):
            return None

        self._state.current_turn = turn

        if turn == 0:
            if self.recording.initial_state:
                self._state.current_state = np.array(self.recording.initial_state)
            return None

        turn_data = self.recording.turns[turn - 1]
        if turn_data.get("state_after"):
            self._state.current_state = np.array(turn_data["state_after"])

        return turn_data

    def reset(self) -> None:
        """Reset to beginning of replay."""
        self._state.current_turn = 0
        if self.recording.initial_state:
            self._state.current_state = np.array(self.recording.initial_state)

    def get_turn(self, turn: int) -> dict[str, Any] | None:
        """Get data for specific turn.

        Args:
            turn: Turn number (0-indexed)

        Returns:
            Turn data or None if invalid.
        """
        if not (0 <= turn < self._state.total_turns):
            return None
        return self.recording.turns[turn]

    def iterate_turns(self) -> Iterator[dict[str, Any]]:
        """Iterate through all turns.

        Yields:
            Turn data dictionaries.
        """
        yield from self.recording.turns

    def get_summary(self) -> dict[str, Any]:
        """Get replay summary.

        Returns:
            Summary dictionary.
        """
        return {
            "game_id": self.recording.game_id,
            "total_turns": self._state.total_turns,
            "human_player": self.recording.human_player,
            "final_result": self.recording.final_result,
            "created_at": self.recording.created_at,
        }


class GameRecorder:
    """Records game turns for later replay.

    This class captures game events during play.

    Example:
        >>> recorder = GameRecorder()
        >>> recorder.start(game_id="game123", human_player=0)
        >>>
        >>> # During game
        >>> recorder.record_initial_state(initial_obs)
        >>> recorder.record_turn(turn_result)
        >>>
        >>> # Save when done
        >>> recording = recorder.stop()
        >>> recording.save("game.replay")
    """

    def __init__(self, config: HumanInterfaceConfig | None = None) -> None:
        """Initialize recorder.

        Args:
            config: Human interface configuration
        """
        self.config = config or HumanInterfaceConfig()
        self._recording: GameRecording | None = None
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording

    def start(
        self,
        game_id: str | None = None,
        human_player: int = 0,
        game_config: dict[str, Any] | None = None,
    ) -> str:
        """Start a new recording.

        Args:
            game_id: Unique game identifier (auto-generated if not provided)
            human_player: Which player is human
            game_config: Game configuration

        Returns:
            Game ID.
        """
        game_id = game_id or str(uuid.uuid4())[:8]

        self._recording = GameRecording(
            game_id=game_id,
            human_player=human_player,
            game_config=game_config or {},
        )
        self._is_recording = True

        logger.info(f"Started recording game {game_id}")
        return game_id

    def record_initial_state(self, state: np.ndarray) -> None:
        """Record the initial game state.

        Args:
            state: Initial observation
        """
        if not self._is_recording or self._recording is None:
            return

        self._recording.initial_state = _serialize_array(state)

    def record_turn(self, turn_result: TurnResult) -> None:
        """Record a turn.

        Args:
            turn_result: Result of the turn
        """
        if not self._is_recording or self._recording is None:
            return

        turn_data: dict[str, Any] = {
            "action": turn_result.action,
            "player": turn_result.player,
            "is_human": turn_result.is_human,
            "reward": turn_result.reward,
            "game_over": turn_result.game_over,
            "winner": turn_result.winner,
            "ai_evaluation": turn_result.ai_evaluation,
            "thinking_time": turn_result.thinking_time,
            "timestamp": turn_result.timestamp,
        }

        # Optionally include states (can be large)
        if turn_result.state_before is not None:
            turn_data["state_before"] = _serialize_array(turn_result.state_before)
        if turn_result.state_after is not None:
            turn_data["state_after"] = _serialize_array(turn_result.state_after)

        self._recording.turns.append(turn_data)

        # Check for game over
        if turn_result.game_over:
            self._recording.final_result = {
                "winner": turn_result.winner,
                "total_turns": len(self._recording.turns),
            }

    def record_metadata(self, key: str, value: Any) -> None:
        """Add metadata to recording.

        Args:
            key: Metadata key
            value: Metadata value
        """
        if self._recording:
            self._recording.metadata[key] = value

    def stop(self) -> GameRecording | None:
        """Stop recording and return the recording.

        Returns:
            GameRecording or None if not recording.
        """
        if not self._is_recording:
            return None

        recording = self._recording
        self._is_recording = False
        self._recording = None

        assert recording is not None
        logger.info(f"Stopped recording. {len(recording.turns)} turns captured.")
        return recording

    def get_current_recording(self) -> GameRecording | None:
        """Get the current recording without stopping.

        Returns:
            Current GameRecording or None.
        """
        return self._recording


class ReplayManager:
    """Manager for game recordings and replays.

    This class handles saving, loading, and listing game recordings.

    Example:
        >>> manager = ReplayManager(config)
        >>>
        >>> # List available replays
        >>> replays = manager.list_replays()
        >>>
        >>> # Load a replay
        >>> replay = manager.load_replay("game123")
        >>>
        >>> # Save current recording
        >>> manager.save_recording(recording)
    """

    def __init__(self, config: HumanInterfaceConfig | None = None) -> None:
        """Initialize replay manager.

        Args:
            config: Human interface configuration
        """
        self.config = config or HumanInterfaceConfig()
        self._replay_dir = Path(self.config.replay_dir)
        self._save_dir = Path(self.config.save_dir)

        # Ensure directories exist
        self._replay_dir.mkdir(parents=True, exist_ok=True)
        self._save_dir.mkdir(parents=True, exist_ok=True)

    def save_recording(
        self,
        recording: GameRecording,
        as_save: bool = False,
    ) -> Path:
        """Save a recording to disk.

        Args:
            recording: Recording to save
            as_save: Save as game save (vs replay)

        Returns:
            Path where recording was saved.
        """
        directory = self._save_dir if as_save else self._replay_dir
        filename = f"{recording.game_id}.json"
        path = directory / filename

        recording.save(path)
        return path

    def load_replay(self, game_id: str) -> GameReplay:
        """Load a replay by game ID.

        Args:
            game_id: Game identifier

        Returns:
            GameReplay instance.

        Raises:
            FileNotFoundError: If replay not found.
        """
        # Try replay directory first
        path = self._replay_dir / f"{game_id}.json"
        if not path.exists():
            # Try save directory
            path = self._save_dir / f"{game_id}.json"

        if not path.exists():
            raise FileNotFoundError(f"Replay not found: {game_id}")

        return GameReplay.from_file(path)

    def load_recording(self, game_id: str) -> GameRecording:
        """Load a recording by game ID.

        Args:
            game_id: Game identifier

        Returns:
            GameRecording instance.
        """
        path = self._save_dir / f"{game_id}.json"
        if not path.exists():
            path = self._replay_dir / f"{game_id}.json"

        if not path.exists():
            raise FileNotFoundError(f"Recording not found: {game_id}")

        return GameRecording.load(path)

    def list_replays(self) -> list[dict[str, Any]]:
        """List all available replays.

        Returns:
            List of replay summaries.
        """
        replays = []

        for path in self._replay_dir.glob("*.json"):
            try:
                recording = GameRecording.load(path)
                replays.append(
                    {
                        "game_id": recording.game_id,
                        "created_at": recording.created_at,
                        "turns": len(recording.turns),
                        "final_result": recording.final_result,
                        "path": str(path),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load replay {path}: {e}")

        # Sort by creation time (newest first)
        replays.sort(key=lambda x: str(x["created_at"]), reverse=True)
        return replays

    def list_saves(self) -> list[dict[str, Any]]:
        """List all saved games.

        Returns:
            List of save summaries.
        """
        saves = []

        for path in self._save_dir.glob("*.json"):
            try:
                recording = GameRecording.load(path)
                saves.append(
                    {
                        "game_id": recording.game_id,
                        "created_at": recording.created_at,
                        "turns": len(recording.turns),
                        "human_player": recording.human_player,
                        "path": str(path),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to load save {path}: {e}")

        saves.sort(key=lambda x: str(x["created_at"]), reverse=True)
        return saves

    def delete_replay(self, game_id: str) -> bool:
        """Delete a replay.

        Args:
            game_id: Game identifier

        Returns:
            True if deleted.
        """
        path = self._replay_dir / f"{game_id}.json"
        if path.exists():
            path.unlink()
            logger.info(f"Deleted replay {game_id}")
            return True
        return False

    def delete_save(self, game_id: str) -> bool:
        """Delete a saved game.

        Args:
            game_id: Game identifier

        Returns:
            True if deleted.
        """
        path = self._save_dir / f"{game_id}.json"
        if path.exists():
            path.unlink()
            logger.info(f"Deleted save {game_id}")
            return True
        return False
