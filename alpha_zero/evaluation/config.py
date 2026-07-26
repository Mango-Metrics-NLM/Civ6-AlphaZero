"""Configuration dataclasses for Model Evaluation.

This module defines the configuration structures for model evaluation including
baseline configurations, diversity tracking, and evaluation parameters.

All values are configurable via dataclasses, YAML files, or environment
variables with the prefix CIV6AZ_EVAL_.

Usage:
    from alpha_zero.evaluation.config import EvaluationConfig, BaselineConfig

    # Default configuration
    config = EvaluationConfig()

    # From YAML file
    config = EvaluationConfig.from_yaml("configs/evaluation.yaml")

    # With environment variable overrides
    config = EvaluationConfig.from_env()
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from alpha_zero.utils.device import resolve_device_str


class BaselineType(str, Enum):
    """Types of baseline opponents for evaluation."""

    RANDOM = "random"
    HEURISTIC = "heuristic"
    PREVIOUS_MODEL = "previous_model"
    SPECIFIC_CHECKPOINT = "specific_checkpoint"


@dataclass
class HeuristicWeights:
    """Configurable weights for heuristic player evaluation.

    These weights control how the heuristic player prioritizes different
    strategic considerations when selecting actions.

    Attributes:
        attack_weight: Weight for attacking enemy units
        defense_weight: Weight for defensive positioning
        expansion_weight: Weight for expanding territory/cities
        production_weight: Weight for production actions
        exploration_weight: Weight for exploring unknown territory
        city_proximity_weight: Weight for actions near cities
    """

    attack_weight: float = 1.0
    defense_weight: float = 0.8
    expansion_weight: float = 1.2
    production_weight: float = 0.5
    exploration_weight: float = 0.3
    city_proximity_weight: float = 0.7

    def __post_init__(self) -> None:
        """Validate weights are non-negative."""
        for field_name in [
            "attack_weight",
            "defense_weight",
            "expansion_weight",
            "production_weight",
            "exploration_weight",
            "city_proximity_weight",
        ]:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeuristicWeights:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BaselineConfig:
    """Configuration for a single baseline opponent.

    Attributes:
        type: Type of baseline (random, heuristic, previous_model, specific_checkpoint)
        name: Display name for this baseline in reports
        num_games: Number of games to play against this baseline
        enabled: Whether this baseline is active for evaluation
        checkpoint_path: Path to model checkpoint (for model-based baselines)
        heuristic_weights: Weights for heuristic baseline behavior
        use_mcts: Whether to use MCTS for model-based baselines
        mcts_simulations: Number of MCTS simulations if using MCTS
    """

    type: BaselineType = BaselineType.RANDOM
    name: str = "random"
    num_games: int = 100
    enabled: bool = True
    checkpoint_path: str | None = None
    heuristic_weights: HeuristicWeights = field(default_factory=HeuristicWeights)
    use_mcts: bool = True
    mcts_simulations: int = 50

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if isinstance(self.type, str):
            self.type = BaselineType(self.type)

        if self.num_games < 1:
            raise ValueError(f"num_games must be at least 1, got {self.num_games}")

        if self.mcts_simulations < 1:
            raise ValueError(f"mcts_simulations must be at least 1, got {self.mcts_simulations}")

        if self.type in (
            BaselineType.PREVIOUS_MODEL,
            BaselineType.SPECIFIC_CHECKPOINT,
        ):
            if self.checkpoint_path is None:
                raise ValueError(f"checkpoint_path required for baseline type {self.type}")

        if isinstance(self.heuristic_weights, dict):
            self.heuristic_weights = HeuristicWeights.from_dict(self.heuristic_weights)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineConfig:
        """Create from dictionary."""
        data = data.copy()
        if "type" in data and isinstance(data["type"], str):
            data["type"] = BaselineType(data["type"])
        if "heuristic_weights" in data and isinstance(data["heuristic_weights"], dict):
            data["heuristic_weights"] = HeuristicWeights.from_dict(data["heuristic_weights"])
        if "heuristic_params" in data:
            # Support legacy key name
            data["heuristic_weights"] = HeuristicWeights.from_dict(data.pop("heuristic_params"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "name": self.name,
            "num_games": self.num_games,
            "enabled": self.enabled,
            "checkpoint_path": self.checkpoint_path,
            "heuristic_weights": self.heuristic_weights.to_dict(),
            "use_mcts": self.use_mcts,
            "mcts_simulations": self.mcts_simulations,
        }


@dataclass
class DiversityConfig:
    """Configuration for game diversity analysis.

    Tracks various metrics to measure how diverse the model's gameplay is,
    including opening move variety, game length distribution, and strategy patterns.

    Attributes:
        enabled: Whether to track diversity metrics
        track_opening_moves: Track first N moves distribution
        opening_depth: Number of opening moves to track
        track_game_lengths: Track distribution of game lengths
        track_victory_types: Track types of victories (if applicable)
        track_action_entropy: Track action selection entropy over time
        track_position_variety: Track variety of board positions visited
    """

    enabled: bool = True
    track_opening_moves: bool = True
    opening_depth: int = 10
    track_game_lengths: bool = True
    track_victory_types: bool = True
    track_action_entropy: bool = True
    track_position_variety: bool = False  # Can be expensive

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.opening_depth < 1:
            raise ValueError(f"opening_depth must be at least 1, got {self.opening_depth}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiversityConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvaluationConfig:
    """Master configuration for model evaluation.

    Provides a unified interface for configuring all aspects of model evaluation
    including baselines, diversity tracking, ELO ratings, and output settings.

    Attributes:
        experiment_name: Name for this evaluation run
        model_checkpoint_path: Path to model checkpoint to evaluate
        baselines: List of baseline configurations
        diversity: Diversity analysis configuration
        elo_enabled: Whether to track ELO ratings
        elo_initial_rating: Initial ELO rating for new models
        parallel_games: Number of games to run in parallel
        device: Device for neural network inference (cpu, cuda, cuda:0, etc.)
        random_seed: Seed for reproducibility (None for random)
        output_dir: Directory for evaluation outputs
        save_game_logs: Whether to save detailed game logs
        max_game_logs: Maximum number of game logs to save
        tensorboard_enabled: Enable TensorBoard logging
        wandb_enabled: Enable Weights & Biases logging
        env_board_size: Board size for evaluation environment
        env_max_turns: Maximum turns for evaluation games
    """

    experiment_name: str = "model_evaluation"
    model_checkpoint_path: str = "checkpoints/latest.pt"
    baselines: list[BaselineConfig] = field(
        default_factory=lambda: [
            BaselineConfig(type=BaselineType.RANDOM, name="random_baseline"),
            BaselineConfig(type=BaselineType.HEURISTIC, name="heuristic_basic"),
        ]
    )
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    elo_enabled: bool = True
    elo_initial_rating: float = 1500.0
    parallel_games: int = 1
    device: str = "cpu"
    random_seed: int | None = None
    output_dir: str = "logs/evaluation"
    save_game_logs: bool = True
    max_game_logs: int = 50
    tensorboard_enabled: bool = True
    wandb_enabled: bool = False
    env_board_size: int = 16
    env_max_turns: int = 100

    ENV_PREFIX: str = field(default="CIV6AZ_EVAL_", repr=False)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.elo_initial_rating < 100 or self.elo_initial_rating > 3000:
            raise ValueError(
                f"elo_initial_rating must be between 100 and 3000, got {self.elo_initial_rating}"
            )

        if self.parallel_games < 1:
            raise ValueError(f"parallel_games must be at least 1, got {self.parallel_games}")

        if self.max_game_logs < 0:
            raise ValueError(f"max_game_logs must be non-negative, got {self.max_game_logs}")

        if self.env_board_size < 4:
            raise ValueError(f"env_board_size must be at least 4, got {self.env_board_size}")

        if self.env_max_turns < 1:
            raise ValueError(f"env_max_turns must be at least 1, got {self.env_max_turns}")

        # Resolve rather than warn, so an unavailable "cuda" cannot reach .to().
        self.device = resolve_device_str(self.device)

        # Convert diversity dict to DiversityConfig if needed
        if isinstance(self.diversity, dict):
            self.diversity = DiversityConfig.from_dict(self.diversity)

        # Convert baseline dicts to BaselineConfig if needed
        converted_baselines = []
        for baseline in self.baselines:
            if isinstance(baseline, dict):
                converted_baselines.append(BaselineConfig.from_dict(baseline))
            else:
                converted_baselines.append(baseline)
        self.baselines = converted_baselines

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationConfig:
        """Create from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            EvaluationConfig instance
        """
        data = data.copy()

        # Handle nested configs
        if "diversity" in data and isinstance(data["diversity"], dict):
            data["diversity"] = DiversityConfig.from_dict(data["diversity"])

        if "baselines" in data:
            data["baselines"] = [
                BaselineConfig.from_dict(b) if isinstance(b, dict) else b for b in data["baselines"]
            ]

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvaluationConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            EvaluationConfig instance
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data or {})

    @classmethod
    def from_env(cls, base_config: EvaluationConfig | None = None) -> EvaluationConfig:
        """Create configuration with environment variable overrides.

        Environment variables should be prefixed with CIV6AZ_EVAL_.
        For example: CIV6AZ_EVAL_ELO_ENABLED=true

        Args:
            base_config: Base configuration to override. If None, uses defaults.

        Returns:
            EvaluationConfig with environment overrides applied
        """
        config = base_config or cls()
        prefix = config.ENV_PREFIX

        # Map of environment variable suffixes to config attributes
        env_mappings: dict[str, tuple[str, Callable[[str], Any]]] = {
            "EXPERIMENT_NAME": ("experiment_name", str),
            "MODEL_CHECKPOINT_PATH": ("model_checkpoint_path", str),
            "ELO_ENABLED": ("elo_enabled", lambda x: x.lower() == "true"),
            "ELO_INITIAL_RATING": ("elo_initial_rating", float),
            "PARALLEL_GAMES": ("parallel_games", int),
            "DEVICE": ("device", str),
            "RANDOM_SEED": ("random_seed", lambda x: int(x) if x else None),
            "OUTPUT_DIR": ("output_dir", str),
            "SAVE_GAME_LOGS": ("save_game_logs", lambda x: x.lower() == "true"),
            "MAX_GAME_LOGS": ("max_game_logs", int),
            "TENSORBOARD_ENABLED": ("tensorboard_enabled", lambda x: x.lower() == "true"),
            "WANDB_ENABLED": ("wandb_enabled", lambda x: x.lower() == "true"),
            "ENV_BOARD_SIZE": ("env_board_size", int),
            "ENV_MAX_TURNS": ("env_max_turns", int),
        }

        for env_suffix, (attr, converter) in env_mappings.items():
            env_key = f"{prefix}{env_suffix}"
            if env_key in os.environ:
                try:
                    value = converter(os.environ[env_key])
                    setattr(config, attr, value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid value for {env_key}: {os.environ[env_key]}") from e

        return config

    @classmethod
    def from_preset(cls, preset: str) -> EvaluationConfig:
        """Load from predefined preset.

        Available presets:
        - quick: Minimal games for fast testing
        - standard: Balanced evaluation (default-like)
        - comprehensive: Thorough evaluation with more games

        Args:
            preset: Preset name

        Returns:
            EvaluationConfig for the preset
        """
        presets = {
            "quick": cls(
                experiment_name="quick_eval",
                baselines=[
                    BaselineConfig(
                        type=BaselineType.RANDOM,
                        name="random",
                        num_games=10,
                    ),
                ],
                diversity=DiversityConfig(enabled=False),
                save_game_logs=False,
            ),
            "standard": cls(
                experiment_name="standard_eval",
                baselines=[
                    BaselineConfig(
                        type=BaselineType.RANDOM,
                        name="random_baseline",
                        num_games=100,
                    ),
                    BaselineConfig(
                        type=BaselineType.HEURISTIC,
                        name="heuristic_basic",
                        num_games=100,
                    ),
                ],
                diversity=DiversityConfig(enabled=True),
            ),
            "comprehensive": cls(
                experiment_name="comprehensive_eval",
                baselines=[
                    BaselineConfig(
                        type=BaselineType.RANDOM,
                        name="random_baseline",
                        num_games=200,
                    ),
                    BaselineConfig(
                        type=BaselineType.HEURISTIC,
                        name="heuristic_defensive",
                        num_games=100,
                        heuristic_weights=HeuristicWeights(
                            attack_weight=0.5,
                            defense_weight=1.5,
                        ),
                    ),
                    BaselineConfig(
                        type=BaselineType.HEURISTIC,
                        name="heuristic_aggressive",
                        num_games=100,
                        heuristic_weights=HeuristicWeights(
                            attack_weight=2.0,
                            defense_weight=0.5,
                        ),
                    ),
                    BaselineConfig(
                        type=BaselineType.HEURISTIC,
                        name="heuristic_expansionist",
                        num_games=100,
                        heuristic_weights=HeuristicWeights(
                            expansion_weight=2.0,
                            attack_weight=0.5,
                        ),
                    ),
                ],
                diversity=DiversityConfig(
                    enabled=True,
                    track_position_variety=True,
                ),
                max_game_logs=200,
            ),
        }

        if preset not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

        return presets[preset]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of config
        """
        return {
            "experiment_name": self.experiment_name,
            "model_checkpoint_path": self.model_checkpoint_path,
            "baselines": [b.to_dict() for b in self.baselines],
            "diversity": self.diversity.to_dict(),
            "elo_enabled": self.elo_enabled,
            "elo_initial_rating": self.elo_initial_rating,
            "parallel_games": self.parallel_games,
            "device": self.device,
            "random_seed": self.random_seed,
            "output_dir": self.output_dir,
            "save_game_logs": self.save_game_logs,
            "max_game_logs": self.max_game_logs,
            "tensorboard_enabled": self.tensorboard_enabled,
            "wandb_enabled": self.wandb_enabled,
            "env_board_size": self.env_board_size,
            "env_max_turns": self.env_max_turns,
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_enabled_baselines(self) -> list[BaselineConfig]:
        """Get list of enabled baseline configurations.

        Returns:
            List of enabled BaselineConfig instances
        """
        return [b for b in self.baselines if b.enabled]

    def validate(self) -> list[str]:
        """Validate configuration and return any warnings.

        Returns:
            List of warning messages (empty if no warnings)
        """
        warnings = []

        enabled_baselines = self.get_enabled_baselines()
        if not enabled_baselines:
            warnings.append("No baselines are enabled for evaluation")

        total_games = sum(b.num_games for b in enabled_baselines)
        if total_games < 50:
            warnings.append(
                f"Total games ({total_games}) is low; consider increasing for "
                "statistically significant results"
            )

        if self.model_checkpoint_path and not Path(self.model_checkpoint_path).exists():
            warnings.append(f"Model checkpoint not found: {self.model_checkpoint_path}")

        return warnings
