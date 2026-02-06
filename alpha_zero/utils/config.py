"""Configuration loading and management for Civ6 AlphaZero.

This module provides utilities for loading, validating, and managing
configuration from YAML files and environment variables.

Usage:
    from alpha_zero.utils.config import load_config, ConfigManager

    # Load from file
    config = load_config("configs/civ6_config.yaml")

    # Or use ConfigManager for dynamic updates
    manager = ConfigManager("configs/civ6_config.yaml")
    network_config = manager.get_network_config()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    type: str = "microciv"
    board_size: int = 16
    num_players: int = 2
    max_turns: int = 200


@dataclass
class NetworkConfig:
    """Neural network configuration."""

    spatial_channels: int = 84
    vector_dim: int = 512
    num_res_blocks: int = 10
    num_filters: int = 128
    num_heads: int = 4
    value_head_hidden: int = 256
    dropout: float = 0.0


@dataclass
class MCTSConfig:
    """MCTS configuration."""

    num_simulations: int = 200
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_threshold: int = 30


@dataclass
class TrainingConfig:
    """Training configuration."""

    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    lr_decay_steps: list[int] = field(default_factory=lambda: [100000, 300000, 500000])
    lr_decay_rate: float = 0.1
    replay_buffer_size: int = 500000
    min_replay_size: int = 10000
    gradient_clip: float = 1.0


@dataclass
class DistributedConfig:
    """Distributed training configuration."""

    enabled: bool = True
    num_actors: int = 64
    actor_batch_size: int = 8
    learner_batch_size: int = 256
    weight_sync_interval: int = 100
    max_queue_size: int = 10000
    compression_enabled: bool = True
    evaluation_interval: int = 1000
    checkpoint_interval: int = 5000
    max_training_steps: int = 1000000


@dataclass
class UncivConfig:
    """Unciv interface configuration."""

    save_dir: str = "saves/unciv"
    auto_save: bool = True
    auto_save_interval: int = 10
    board_size: int = 80
    max_players: int = 8


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    log_dir: str = "logs"
    game_logging: bool = True
    training_metrics: bool = True
    metrics_window: int = 100
    tensorboard: bool = False
    profile_enabled: bool = False


@dataclass
class ActionSpaceConfig:
    """Action space configuration."""

    hierarchical: bool = True
    max_units: int = 100
    max_cities: int = 20
    num_directions: int = 8
    tech_count: int = 73
    civic_count: int = 56


@dataclass
class RewardsConfig:
    """Reward shaping configuration."""

    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    city_founded: float = 0.1
    tech_researched: float = 0.05
    unit_killed: float = 0.02
    unit_lost: float = -0.02
    wonder_built: float = 0.15


@dataclass
class Civ6Config:
    """Complete Civ6 AlphaZero configuration."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    unciv: UncivConfig = field(default_factory=UncivConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Civ6Config:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            Civ6Config instance.
        """
        return cls(
            environment=_dict_to_dataclass(
                EnvironmentConfig,
                data.get("environment", {}),
            ),
            network=_dict_to_dataclass(NetworkConfig, data.get("network", {})),
            mcts=_dict_to_dataclass(MCTSConfig, data.get("mcts", {})),
            training=_dict_to_dataclass(TrainingConfig, data.get("training", {})),
            distributed=_dict_to_dataclass(
                DistributedConfig,
                data.get("distributed", {}),
            ),
            unciv=_dict_to_dataclass(UncivConfig, data.get("unciv", {})),
            logging=_dict_to_dataclass(LoggingConfig, data.get("logging", {})),
            action_space=_dict_to_dataclass(
                ActionSpaceConfig,
                data.get("action_space", {}),
            ),
            rewards=_dict_to_dataclass(RewardsConfig, data.get("rewards", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary.
        """
        return {
            "environment": _dataclass_to_dict(self.environment),
            "network": _dataclass_to_dict(self.network),
            "mcts": _dataclass_to_dict(self.mcts),
            "training": _dataclass_to_dict(self.training),
            "distributed": _dataclass_to_dict(self.distributed),
            "unciv": _dataclass_to_dict(self.unciv),
            "logging": _dataclass_to_dict(self.logging),
            "action_space": _dataclass_to_dict(self.action_space),
            "rewards": _dataclass_to_dict(self.rewards),
        }


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Convert dictionary to dataclass, ignoring unknown fields.

    Args:
        cls: Dataclass type.
        data: Dictionary with field values.

    Returns:
        Dataclass instance.
    """
    if not data:
        return cls()

    # Get valid field names for the dataclass
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]

    # Filter out unknown fields
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}

    return cls(**filtered_data)


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Convert dataclass to dictionary.

    Args:
        obj: Dataclass instance.

    Returns:
        Dictionary representation.
    """
    result = {}
    for f in obj.__dataclass_fields__:
        value = getattr(obj, f)
        if isinstance(value, list):
            result[f] = list(value)
        else:
            result[f] = value
    return result


def load_config(path: str | Path) -> Civ6Config:
    """Load configuration from YAML file.

    Args:
        path: Path to configuration file.

    Returns:
        Civ6Config instance.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return Civ6Config.from_dict(data or {})


def save_config(config: Civ6Config, path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration to save.
        path: Path to save configuration to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


class ConfigManager:
    """Dynamic configuration manager with environment variable overrides.

    Supports loading configuration from file and overriding with
    environment variables prefixed with CIV6AZ_.

    Example:
        Set CIV6AZ_TRAINING_BATCH_SIZE=512 to override training.batch_size

    Usage:
        manager = ConfigManager("configs/civ6_config.yaml")
        config = manager.config
        network_config = manager.get_network_config()
    """

    ENV_PREFIX = "CIV6AZ_"

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file. If None, uses defaults.
        """
        if config_path is not None:
            self._config = load_config(config_path)
        else:
            self._config = Civ6Config()

        self._apply_env_overrides()

    @property
    def config(self) -> Civ6Config:
        """Get the full configuration."""
        return self._config

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        for key, value in os.environ.items():
            if not key.startswith(self.ENV_PREFIX):
                continue

            # Parse key like CIV6AZ_TRAINING_BATCH_SIZE
            parts = key[len(self.ENV_PREFIX) :].lower().split("_", 1)
            if len(parts) != 2:
                continue

            section, param = parts

            # Get the section config
            section_config = getattr(self._config, section, None)
            if section_config is None:
                continue

            # Check if parameter exists
            if not hasattr(section_config, param):
                continue

            # Convert value to appropriate type
            current_value = getattr(section_config, param)
            try:
                typed_value: Any
                if isinstance(current_value, bool):
                    typed_value = value.lower() in ("true", "1", "yes")
                elif isinstance(current_value, int):
                    typed_value = int(value)
                elif isinstance(current_value, float):
                    typed_value = float(value)
                elif isinstance(current_value, list):
                    typed_value = yaml.safe_load(value)
                else:
                    typed_value = value

                setattr(section_config, param, typed_value)
            except (ValueError, yaml.YAMLError) as e:
                logger.warning(f"Failed to parse environment override {key}={value!r}: {e}")

    def get_environment_config(self) -> EnvironmentConfig:
        """Get environment configuration."""
        return self._config.environment

    def get_network_config(self) -> NetworkConfig:
        """Get network configuration."""
        return self._config.network

    def get_mcts_config(self) -> MCTSConfig:
        """Get MCTS configuration."""
        return self._config.mcts

    def get_training_config(self) -> TrainingConfig:
        """Get training configuration."""
        return self._config.training

    def get_distributed_config(self) -> DistributedConfig:
        """Get distributed training configuration."""
        return self._config.distributed

    def get_unciv_config(self) -> UncivConfig:
        """Get Unciv interface configuration."""
        return self._config.unciv

    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self._config.logging

    def get_action_space_config(self) -> ActionSpaceConfig:
        """Get action space configuration."""
        return self._config.action_space

    def get_rewards_config(self) -> RewardsConfig:
        """Get rewards configuration."""
        return self._config.rewards

    def get_microciv_overrides(self) -> dict[str, Any]:
        """Get MicroCiv-specific overrides.

        Returns:
            Dictionary of MicroCiv override values.
        """
        # Load raw YAML to get microciv section
        # This is a convenience for Phase 1 development
        return {
            "board_size": 16,
            "num_res_blocks": 6,
            "num_filters": 64,
            "num_simulations": 50,
            "batch_size": 128,
        }

    def save(self, path: str | Path) -> None:
        """Save current configuration to file.

        Args:
            path: Path to save configuration to.
        """
        save_config(self._config, path)


def get_default_config() -> Civ6Config:
    """Get default configuration.

    Returns:
        Default Civ6Config instance.
    """
    return Civ6Config()


def merge_configs(base: Civ6Config, override: dict[str, Any]) -> Civ6Config:
    """Merge override dictionary into base configuration.

    Args:
        base: Base configuration.
        override: Dictionary of overrides.

    Returns:
        New merged configuration.
    """
    base_dict = base.to_dict()

    for section, values in override.items():
        if section in base_dict and isinstance(values, dict):
            base_dict[section].update(values)
        else:
            base_dict[section] = values

    return Civ6Config.from_dict(base_dict)
