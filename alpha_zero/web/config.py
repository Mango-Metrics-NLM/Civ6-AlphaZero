"""Configuration for the web-based Human vs AI interface.

Provides WebConfig dataclass with YAML loading, environment variable
overrides (CIV6AZ_WEB_ prefix), and preset support.

Usage:
    from alpha_zero.web.config import WebConfig

    config = WebConfig()
    config = WebConfig.from_yaml("configs/web.yaml")
    config = WebConfig.from_preset("dev")
    config = WebConfig.from_env()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from alpha_zero.human_interface.config import BoardTheme, HumanInterfaceConfig
from alpha_zero.utils.device import resolve_device_str
from alpha_zero.utils.network import LOOPBACK_HOST, loopback_origins, warn_if_non_loopback

logger = logging.getLogger(__name__)

# Valid opponent types
VALID_OPPONENT_TYPES = ("random", "heuristic", "neural")

# Valid log levels
DEFAULT_PORT = 8080
VALID_LOG_LEVELS = ("debug", "info", "warning", "error", "critical")


@dataclass
class WebConfig:
    """Configuration for the web-based Human vs AI interface.

    Attributes:
        host: Server bind address
        port: Server port
        debug: Enable debug mode
        log_level: Logging level
        workers: Number of uvicorn workers
        board_size: Default board size for new games
        max_turns: Maximum turns per game
        default_thinking_time: Default AI thinking time in seconds
        max_concurrent_sessions: Maximum simultaneous game sessions
        session_timeout_seconds: Session expiry time in seconds
        checkpoint_path: Path to trained model checkpoint (None for baselines)
        use_mcts: Use MCTS for neural network agent
        mcts_simulations: Number of MCTS simulations per move
        ai_temperature: Temperature for action selection (0=greedy)
        device: PyTorch device for inference
        default_opponent: Default opponent type for new games
        board_theme: Default board theme
        show_ai_evaluation: Show AI position evaluation
        show_move_probabilities: Show move probability distribution
        show_mcts_stats: Show MCTS search statistics
        enable_replay: Enable game recording
        replay_dir: Directory for replay files
        save_dir: Directory for game saves
        cors_origins: Allowed CORS origins
    """

    # Server settings
    host: str = LOOPBACK_HOST
    port: int = DEFAULT_PORT
    debug: bool = False
    log_level: str = "info"
    workers: int = 1

    # Game defaults
    board_size: int = 16
    max_turns: int = 200
    default_thinking_time: float = 5.0
    max_concurrent_sessions: int = 10
    session_timeout_seconds: float = 3600.0

    # AI defaults
    checkpoint_path: str | None = None
    use_mcts: bool = True
    mcts_simulations: int = 50
    ai_temperature: float = 0.0
    device: str = "cpu"
    default_opponent: str = "random"

    # Display defaults
    board_theme: str = "classic"
    show_ai_evaluation: bool = True
    show_move_probabilities: bool = True
    show_mcts_stats: bool = True

    # Replay/save
    enable_replay: bool = True
    replay_dir: str = "replays"
    save_dir: str = "saves/web_games"

    # Security
    cors_origins: list[str] = field(default_factory=lambda: loopback_origins(DEFAULT_PORT))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Default CORS origins follow the port; an explicit list is left alone.
        if self.cors_origins == loopback_origins(DEFAULT_PORT):
            self.cors_origins = loopback_origins(self.port)
        self._validate()
        warn_if_non_loopback(self.host, logger, "Web interface")

    def _validate(self) -> None:
        """Validate configuration values."""
        if self.port < 1 or self.port > 65535:
            msg = f"port must be between 1 and 65535, got {self.port}"
            raise ValueError(msg)

        if self.log_level.lower() not in VALID_LOG_LEVELS:
            msg = f"Invalid log_level: {self.log_level}. Must be one of {VALID_LOG_LEVELS}"
            raise ValueError(msg)

        if self.workers < 1:
            msg = f"workers must be >= 1, got {self.workers}"
            raise ValueError(msg)

        if self.board_size < 4:
            msg = f"board_size must be >= 4, got {self.board_size}"
            raise ValueError(msg)

        if self.max_turns < 1:
            msg = f"max_turns must be >= 1, got {self.max_turns}"
            raise ValueError(msg)

        if self.default_thinking_time <= 0:
            msg = "default_thinking_time must be positive"
            raise ValueError(msg)

        if self.max_concurrent_sessions < 1:
            msg = f"max_concurrent_sessions must be >= 1, got {self.max_concurrent_sessions}"
            raise ValueError(msg)

        if self.session_timeout_seconds <= 0:
            msg = "session_timeout_seconds must be positive"
            raise ValueError(msg)

        if self.mcts_simulations < 1:
            msg = f"mcts_simulations must be >= 1, got {self.mcts_simulations}"
            raise ValueError(msg)

        if self.ai_temperature < 0:
            msg = f"ai_temperature must be >= 0, got {self.ai_temperature}"
            raise ValueError(msg)

        if self.default_opponent not in VALID_OPPONENT_TYPES:
            msg = (
                f"Invalid default_opponent: {self.default_opponent}. "
                f"Must be one of {VALID_OPPONENT_TYPES}"
            )
            raise ValueError(msg)

        # Validate board theme via existing enum
        try:
            BoardTheme(self.board_theme)
        except ValueError as err:
            valid = [t.value for t in BoardTheme]
            msg = f"Invalid board_theme: {self.board_theme}. Must be one of {valid}"
            raise ValueError(msg) from err

        # Resolve the device rather than only validating it. Leaving an
        # unresolved "cuda" here would surface as a RuntimeError deep inside
        # the first neural game instead of at startup.
        self.device = resolve_device_str(self.device)

        if self.checkpoint_path is not None:
            path = Path(self.checkpoint_path)
            if not path.exists():
                logger.warning("Checkpoint path does not exist: %s", self.checkpoint_path)

    @classmethod
    def from_yaml(cls, path: str | Path) -> WebConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            WebConfig instance.

        Raises:
            FileNotFoundError: If config file does not exist.
        """
        path = Path(path)
        if not path.exists():
            msg = f"Config file not found: {path}"
            raise FileNotFoundError(msg)

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        # Extract 'web' section if present
        if "web" in data:
            data = data["web"]

        return cls._from_flat_dict(cls._flatten_dict(data))

    @classmethod
    def from_env(cls, base_config: WebConfig | None = None) -> WebConfig:
        """Create configuration with environment variable overrides.

        Environment variables use prefix CIV6AZ_WEB_ followed by
        uppercase field name. For example: CIV6AZ_WEB_PORT=3000

        Args:
            base_config: Base configuration to override.

        Returns:
            Configuration with environment overrides applied.
        """
        config_dict: dict[str, Any] = {}
        if base_config is not None:
            config_dict = base_config.to_dict()

        # Build from defaults if no base
        if not config_dict:
            try:
                config_dict = cls().to_dict()
            except Exception:
                config_dict = {}

        env_mapping: dict[str, tuple[str, type | Any]] = {
            "CIV6AZ_WEB_HOST": ("host", str),
            "CIV6AZ_WEB_PORT": ("port", int),
            "CIV6AZ_WEB_DEBUG": ("debug", lambda x: x.lower() == "true"),
            "CIV6AZ_WEB_LOG_LEVEL": ("log_level", str),
            "CIV6AZ_WEB_WORKERS": ("workers", int),
            "CIV6AZ_WEB_BOARD_SIZE": ("board_size", int),
            "CIV6AZ_WEB_MAX_TURNS": ("max_turns", int),
            "CIV6AZ_WEB_THINKING_TIME": ("default_thinking_time", float),
            "CIV6AZ_WEB_MAX_SESSIONS": ("max_concurrent_sessions", int),
            "CIV6AZ_WEB_SESSION_TIMEOUT": ("session_timeout_seconds", float),
            "CIV6AZ_WEB_CHECKPOINT": ("checkpoint_path", str),
            "CIV6AZ_WEB_USE_MCTS": ("use_mcts", lambda x: x.lower() == "true"),
            "CIV6AZ_WEB_MCTS_SIMULATIONS": ("mcts_simulations", int),
            "CIV6AZ_WEB_AI_TEMPERATURE": ("ai_temperature", float),
            "CIV6AZ_WEB_DEVICE": ("device", str),
            "CIV6AZ_WEB_DEFAULT_OPPONENT": ("default_opponent", str),
            "CIV6AZ_WEB_BOARD_THEME": ("board_theme", str),
            "CIV6AZ_WEB_SHOW_EVALUATION": ("show_ai_evaluation", lambda x: x.lower() == "true"),
            "CIV6AZ_WEB_SHOW_PROBABILITIES": (
                "show_move_probabilities",
                lambda x: x.lower() == "true",
            ),
            "CIV6AZ_WEB_SHOW_MCTS_STATS": (
                "show_mcts_stats",
                lambda x: x.lower() == "true",
            ),
            "CIV6AZ_WEB_ENABLE_REPLAY": ("enable_replay", lambda x: x.lower() == "true"),
            "CIV6AZ_WEB_REPLAY_DIR": ("replay_dir", str),
            "CIV6AZ_WEB_SAVE_DIR": ("save_dir", str),
        }

        for env_var, (field_name, converter) in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    config_dict[field_name] = converter(value)
                except (ValueError, TypeError):
                    logger.warning("Invalid value for %s: %s", env_var, value)

        return cls(**{k: v for k, v in config_dict.items() if k in cls._field_names()})

    @classmethod
    def from_preset(cls, preset: str) -> WebConfig:
        """Create configuration from a named preset.

        Available presets:
        - "dev": Small board, fast AI, debug enabled
        - "production": Full board, neural AI, production settings

        Args:
            preset: Preset name.

        Returns:
            WebConfig instance.

        Raises:
            ValueError: If preset name is unknown.
        """
        presets: dict[str, dict[str, Any]] = {
            "dev": {
                "host": "127.0.0.1",
                "port": 8080,
                "debug": True,
                "log_level": "debug",
                "board_size": 8,
                "max_turns": 50,
                "default_thinking_time": 1.0,
                "max_concurrent_sessions": 3,
                "session_timeout_seconds": 600.0,
                "use_mcts": False,
                "mcts_simulations": 10,
                "default_opponent": "random",
            },
            "production": {
                "host": "0.0.0.0",  # noqa: S104
                "port": 8080,
                "debug": False,
                "log_level": "info",
                "board_size": 16,
                "max_turns": 200,
                "default_thinking_time": 5.0,
                "max_concurrent_sessions": 20,
                "session_timeout_seconds": 7200.0,
                "use_mcts": True,
                "mcts_simulations": 100,
                "default_opponent": "heuristic",
            },
        }

        if preset not in presets:
            valid = list(presets.keys())
            msg = f"Unknown preset: {preset}. Must be one of {valid}"
            raise ValueError(msg)

        return cls(**presets[preset])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "log_level": self.log_level,
            "workers": self.workers,
            "board_size": self.board_size,
            "max_turns": self.max_turns,
            "default_thinking_time": self.default_thinking_time,
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "session_timeout_seconds": self.session_timeout_seconds,
            "checkpoint_path": self.checkpoint_path,
            "use_mcts": self.use_mcts,
            "mcts_simulations": self.mcts_simulations,
            "ai_temperature": self.ai_temperature,
            "device": self.device,
            "default_opponent": self.default_opponent,
            "board_theme": self.board_theme,
            "show_ai_evaluation": self.show_ai_evaluation,
            "show_move_probabilities": self.show_move_probabilities,
            "show_mcts_stats": self.show_mcts_stats,
            "enable_replay": self.enable_replay,
            "replay_dir": self.replay_dir,
            "save_dir": self.save_dir,
            "cors_origins": self.cors_origins,
        }

    def to_human_interface_config(self) -> HumanInterfaceConfig:
        """Convert to HumanInterfaceConfig for GameController compatibility.

        Returns:
            HumanInterfaceConfig instance with mapped settings.
        """
        return HumanInterfaceConfig(
            thinking_time_seconds=self.default_thinking_time,
            max_thinking_time=max(self.default_thinking_time * 2, 60.0),
            show_ai_evaluation=self.show_ai_evaluation,
            show_move_probabilities=self.show_move_probabilities,
            show_mcts_stats=self.show_mcts_stats,
            board_theme=self.board_theme,
            enable_replay=self.enable_replay,
            replay_dir=self.replay_dir,
            save_dir=self.save_dir,
            enable_undo=True,
            max_undo_depth=10,
        )

    @classmethod
    def _field_names(cls) -> set[str]:
        """Get the set of valid field names."""
        return set(cls.__dataclass_fields__.keys())

    @staticmethod
    def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested dictionary for dataclass construction.

        Handles YAML structure like:
            game:
                board_size: 8
        → {"board_size": 8}
        """
        result: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                result.update(WebConfig._flatten_dict(value, prefix=f"{key}_"))
            else:
                result[
                    f"{prefix}{key}" if prefix and key not in WebConfig._field_names() else key
                ] = value
        return result

    @classmethod
    def _from_flat_dict(cls, data: dict[str, Any]) -> WebConfig:
        """Create from a flat dictionary, ignoring unknown keys."""
        valid_fields = cls._field_names()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
