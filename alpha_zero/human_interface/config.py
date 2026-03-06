"""Configuration for Human vs AI interface.

This module provides configuration dataclasses for the human interface.

Usage:
    from alpha_zero.human_interface.config import HumanInterfaceConfig

    config = HumanInterfaceConfig()
    config = HumanInterfaceConfig.from_yaml("config.yaml")
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class InputMode(str, Enum):
    """Input mode for human player."""

    COORDINATE = "coordinate"  # e.g., "A1", "B3"
    INDEX = "index"  # e.g., "0", "5"
    INTERACTIVE = "interactive"  # Arrow keys + enter


class BoardTheme(str, Enum):
    """Visual theme for board display."""

    CLASSIC = "classic"
    DARK = "dark"
    COLORBLIND = "colorblind"
    MINIMAL = "minimal"


@dataclass
class HumanInterfaceConfig:
    """Configuration for Human vs AI interface.

    Attributes:
        thinking_time_seconds: Default AI thinking time
        max_thinking_time: Maximum AI thinking time allowed
        show_ai_evaluation: Show AI's position evaluation
        show_move_probabilities: Show move probability distribution
        show_mcts_stats: Show MCTS search statistics
        board_theme: Visual theme for board display
        animation_speed: Animation speed multiplier
        highlight_valid_moves: Highlight available moves
        input_mode: How human enters moves
        input_timeout_seconds: Timeout for human input
        confirm_moves: Require move confirmation
        auto_save: Auto-save game progress
        auto_save_interval: Turns between auto-saves
        save_dir: Directory for game saves
        enable_replay: Enable replay functionality
        replay_dir: Directory for replays
        enable_undo: Allow undoing moves
        max_undo_depth: Maximum moves to undo
        show_coordinates: Show board coordinates
        show_turn_number: Show current turn number
        verbose_logging: Enable verbose logging
    """

    # AI settings
    thinking_time_seconds: float = 10.0
    max_thinking_time: float = 60.0
    show_ai_evaluation: bool = True
    show_move_probabilities: bool = True
    show_mcts_stats: bool = True

    # Visualization settings
    board_theme: str = "classic"
    animation_speed: float = 1.0
    highlight_valid_moves: bool = True
    show_coordinates: bool = True
    show_turn_number: bool = True

    # Input settings
    input_mode: str = "coordinate"
    input_timeout_seconds: float = 300.0
    confirm_moves: bool = True

    # Save/Load settings
    auto_save: bool = True
    auto_save_interval: int = 10
    save_dir: str = "saves/human_games"

    # Replay settings
    enable_replay: bool = True
    replay_dir: str = "replays"

    # Undo settings
    enable_undo: bool = True
    max_undo_depth: int = 5

    # Logging
    verbose_logging: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values."""
        if self.thinking_time_seconds <= 0:
            raise ValueError("thinking_time_seconds must be positive")

        if self.max_thinking_time < self.thinking_time_seconds:
            raise ValueError("max_thinking_time must be >= thinking_time_seconds")

        if self.input_timeout_seconds <= 0:
            raise ValueError("input_timeout_seconds must be positive")

        if self.auto_save_interval <= 0:
            raise ValueError("auto_save_interval must be positive")

        if self.max_undo_depth < 0:
            raise ValueError("max_undo_depth must be non-negative")

        # Validate enums
        try:
            BoardTheme(self.board_theme)
        except ValueError as err:
            valid = [t.value for t in BoardTheme]
            raise ValueError(
                f"Invalid board_theme: {self.board_theme}. Must be one of {valid}"
            ) from err

        try:
            InputMode(self.input_mode)
        except ValueError as err:
            valid = [m.value for m in InputMode]
            raise ValueError(
                f"Invalid input_mode: {self.input_mode}. Must be one of {valid}"
            ) from err

    @classmethod
    def from_yaml(cls, path: str | Path) -> HumanInterfaceConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            HumanInterfaceConfig instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Extract human_interface section if present
        if "human_interface" in data:
            data = data["human_interface"]

        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    @classmethod
    def from_env(cls, base_config: HumanInterfaceConfig | None = None) -> HumanInterfaceConfig:
        """Create configuration with environment variable overrides.

        Environment variables use prefix CIV6AZ_HUMAN_ followed by
        uppercase field name.

        Args:
            base_config: Base configuration to override

        Returns:
            Configuration with environment overrides.
        """
        config_dict = (base_config or cls()).to_dict()

        env_mapping: dict[str, tuple[str, Callable[[str], Any]]] = {
            "CIV6AZ_HUMAN_THINKING_TIME_SECONDS": ("thinking_time_seconds", float),
            "CIV6AZ_HUMAN_MAX_THINKING_TIME": ("max_thinking_time", float),
            "CIV6AZ_HUMAN_SHOW_AI_EVALUATION": (
                "show_ai_evaluation",
                lambda x: x.lower() == "true",
            ),
            "CIV6AZ_HUMAN_SHOW_MOVE_PROBABILITIES": (
                "show_move_probabilities",
                lambda x: x.lower() == "true",
            ),
            "CIV6AZ_HUMAN_BOARD_THEME": ("board_theme", str),
            "CIV6AZ_HUMAN_INPUT_MODE": ("input_mode", str),
            "CIV6AZ_HUMAN_INPUT_TIMEOUT_SECONDS": ("input_timeout_seconds", float),
            "CIV6AZ_HUMAN_CONFIRM_MOVES": ("confirm_moves", lambda x: x.lower() == "true"),
            "CIV6AZ_HUMAN_AUTO_SAVE": ("auto_save", lambda x: x.lower() == "true"),
            "CIV6AZ_HUMAN_SAVE_DIR": ("save_dir", str),
            "CIV6AZ_HUMAN_ENABLE_REPLAY": ("enable_replay", lambda x: x.lower() == "true"),
            "CIV6AZ_HUMAN_ENABLE_UNDO": ("enable_undo", lambda x: x.lower() == "true"),
            "CIV6AZ_HUMAN_MAX_UNDO_DEPTH": ("max_undo_depth", int),
        }

        for env_var, (field_name, converter) in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    config_dict[field_name] = converter(value)
                except (ValueError, TypeError):
                    pass

        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "thinking_time_seconds": self.thinking_time_seconds,
            "max_thinking_time": self.max_thinking_time,
            "show_ai_evaluation": self.show_ai_evaluation,
            "show_move_probabilities": self.show_move_probabilities,
            "show_mcts_stats": self.show_mcts_stats,
            "board_theme": self.board_theme,
            "animation_speed": self.animation_speed,
            "highlight_valid_moves": self.highlight_valid_moves,
            "show_coordinates": self.show_coordinates,
            "show_turn_number": self.show_turn_number,
            "input_mode": self.input_mode,
            "input_timeout_seconds": self.input_timeout_seconds,
            "confirm_moves": self.confirm_moves,
            "auto_save": self.auto_save,
            "auto_save_interval": self.auto_save_interval,
            "save_dir": self.save_dir,
            "enable_replay": self.enable_replay,
            "replay_dir": self.replay_dir,
            "enable_undo": self.enable_undo,
            "max_undo_depth": self.max_undo_depth,
            "verbose_logging": self.verbose_logging,
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def get_board_theme_enum(self) -> BoardTheme:
        """Get board theme as enum."""
        return BoardTheme(self.board_theme)

    def get_input_mode_enum(self) -> InputMode:
        """Get input mode as enum."""
        return InputMode(self.input_mode)

    def get_save_path(self, game_id: str) -> Path:
        """Get path for saving a game.

        Args:
            game_id: Unique game identifier

        Returns:
            Path for save file.
        """
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"{game_id}.json"

    def get_replay_path(self, game_id: str) -> Path:
        """Get path for saving a replay.

        Args:
            game_id: Unique game identifier

        Returns:
            Path for replay file.
        """
        replay_dir = Path(self.replay_dir)
        replay_dir.mkdir(parents=True, exist_ok=True)
        return replay_dir / f"{game_id}.replay"
