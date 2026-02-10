"""Configuration dataclasses for Population-Based Training (PBT).

This module defines the configuration structures for PBT including
hyperparameter search spaces, agent states, and training parameters.

All values are configurable via dataclasses, YAML files, or environment
variables with the prefix CIV6AZ_PBT_.

Usage:
    from alpha_zero.pbt.config import PBTConfig, HyperparameterSpace

    # Default configuration
    config = PBTConfig()

    # From YAML file
    config = PBTConfig.from_yaml("configs/pbt_config.yaml")

    # With environment variable overrides
    config = PBTConfig.from_env()
"""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class SelectionMethod(str, Enum):
    """Selection methods for PBT."""

    TRUNCATION = "truncation"
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"


class HyperparameterScale(str, Enum):
    """Scale types for hyperparameter sampling."""

    LINEAR = "linear"
    LOG = "log"
    CATEGORICAL = "categorical"


@dataclass
class HyperparameterRange:
    """Defines valid range for a hyperparameter.

    Attributes:
        name: Name of the hyperparameter
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)
        scale: Scale type for sampling ("linear", "log", "categorical")
        discrete: Whether values should be integers
        categories: List of categorical values (for categorical scale)
    """

    name: str
    min_value: float
    max_value: float
    scale: str = "linear"
    discrete: bool = False
    categories: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.scale == "categorical" and not self.categories:
            raise ValueError(f"Categorical hyperparameter '{self.name}' requires categories")
        if self.scale != "categorical" and self.min_value >= self.max_value:
            raise ValueError(f"min_value must be less than max_value for '{self.name}'")
        if self.scale == "log" and self.min_value <= 0:
            raise ValueError(f"min_value must be positive for log scale '{self.name}'")

    def sample(self, rng: np.random.Generator | None = None) -> float | int | Any:
        """Sample a value from this range.

        Args:
            rng: Random number generator. If None, uses default numpy rng.

        Returns:
            Sampled value of appropriate type.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.scale == "categorical":
            return rng.choice(self.categories)

        if self.scale == "log":
            log_min = math.log(self.min_value)
            log_max = math.log(self.max_value)
            value = math.exp(rng.uniform(log_min, log_max))
        else:  # linear
            value = rng.uniform(self.min_value, self.max_value)

        if self.discrete:
            value = round(value)

        return value

    def mutate(
        self,
        value: float | int | Any,
        rng: np.random.Generator | None = None,
        mutation_strength: float = 0.2,
    ) -> float | int | Any:
        """Mutate a value within valid bounds.

        Args:
            value: Current value to mutate
            rng: Random number generator
            mutation_strength: Relative strength of mutation (0-1)

        Returns:
            Mutated value within bounds.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.scale == "categorical":
            # Small chance to switch to different category
            if rng.random() < mutation_strength:
                return rng.choice(self.categories)
            return value

        # Compute mutation range
        if self.scale == "log":
            log_value = math.log(max(value, 1e-10))
            log_min = math.log(self.min_value)
            log_max = math.log(self.max_value)
            log_range = log_max - log_min

            # Mutate in log space
            perturbation = rng.normal(0, mutation_strength * log_range)
            new_log_value = log_value + perturbation
            new_value = math.exp(np.clip(new_log_value, log_min, log_max))
        else:
            value_range = self.max_value - self.min_value
            perturbation = rng.normal(0, mutation_strength * value_range)
            new_value = np.clip(value + perturbation, self.min_value, self.max_value)

        if self.discrete:
            new_value = round(new_value)

        return new_value

    def clip(self, value: float | int | Any) -> float | int | Any:
        """Clip value to valid bounds.

        Args:
            value: Value to clip

        Returns:
            Clipped value within bounds.
        """
        if self.scale == "categorical":
            if value not in self.categories:
                return self.categories[0]
            return value

        clipped = np.clip(value, self.min_value, self.max_value)
        if self.discrete:
            clipped = round(clipped)
        return clipped

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "scale": self.scale,
            "discrete": self.discrete,
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HyperparameterRange:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            min_value=data.get("min_value", 0.0),
            max_value=data.get("max_value", 1.0),
            scale=data.get("scale", "linear"),
            discrete=data.get("discrete", False),
            categories=data.get("categories", []),
        )


@dataclass
class HyperparameterSpace:
    """Defines the complete search space for PBT.

    Attributes:
        learning_rate: Learning rate range
        weight_decay: Weight decay range
        c_puct: MCTS exploration constant range
        dirichlet_alpha: Dirichlet noise alpha range
        temperature: MCTS temperature range
        temperature_initial: Initial temperature for schedule
        temperature_final: Final temperature for schedule
        temperature_decay_steps: Steps over which temperature decays
        batch_size: Training batch size range
        num_simulations: MCTS simulations per move range
        replay_buffer_size: Replay buffer size range
        custom_params: Additional custom hyperparameters
    """

    learning_rate: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(
            name="learning_rate",
            min_value=1e-5,
            max_value=1e-2,
            scale="log",
        )
    )
    weight_decay: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(
            name="weight_decay",
            min_value=1e-6,
            max_value=1e-3,
            scale="log",
        )
    )
    c_puct: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(
            name="c_puct",
            min_value=0.5,
            max_value=3.0,
            scale="linear",
        )
    )
    dirichlet_alpha: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(
            name="dirichlet_alpha",
            min_value=0.1,
            max_value=1.0,
            scale="linear",
        )
    )
    temperature: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(
            name="temperature",
            min_value=0.5,
            max_value=2.0,
            scale="linear",
        )
    )
    # Temperature schedule parameters
    temperature_initial: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(
            name="temperature_initial",
            min_value=1.0,
            max_value=2.5,
            scale="linear",
        )
    )
    temperature_final: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(
            name="temperature_final",
            min_value=0.1,
            max_value=1.0,
            scale="linear",
        )
    )
    temperature_decay_steps: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(
            name="temperature_decay_steps",
            min_value=10,
            max_value=50,
            scale="linear",
            discrete=True,
        )
    )
    batch_size: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(
            name="batch_size",
            min_value=32,
            max_value=512,
            scale="linear",
            discrete=True,
        )
    )
    num_simulations: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(
            name="num_simulations",
            min_value=25,
            max_value=400,
            scale="linear",
            discrete=True,
        )
    )
    replay_buffer_size: HyperparameterRange = field(
        default_factory=lambda: HyperparameterRange(
            name="replay_buffer_size",
            min_value=10000,
            max_value=200000,
            scale="linear",
            discrete=True,
        )
    )
    custom_params: dict[str, HyperparameterRange] = field(default_factory=dict)

    def get_all_params(self) -> dict[str, HyperparameterRange]:
        """Get all hyperparameter ranges as a dictionary.

        Returns:
            Dictionary mapping parameter names to their ranges.
        """
        params = {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "c_puct": self.c_puct,
            "dirichlet_alpha": self.dirichlet_alpha,
            "temperature": self.temperature,
            "temperature_initial": self.temperature_initial,
            "temperature_final": self.temperature_final,
            "temperature_decay_steps": self.temperature_decay_steps,
            "batch_size": self.batch_size,
            "num_simulations": self.num_simulations,
            "replay_buffer_size": self.replay_buffer_size,
        }
        params.update(self.custom_params)
        return params

    def sample_all(self, rng: np.random.Generator | None = None) -> dict[str, float | int | Any]:
        """Sample values for all hyperparameters.

        Args:
            rng: Random number generator

        Returns:
            Dictionary mapping parameter names to sampled values.
        """
        return {name: param.sample(rng) for name, param in self.get_all_params().items()}

    def mutate_all(
        self,
        values: dict[str, float | int | Any],
        rng: np.random.Generator | None = None,
        mutation_strength: float = 0.2,
        mutation_probability: float = 0.8,
    ) -> dict[str, float | int | Any]:
        """Mutate all hyperparameter values.

        Args:
            values: Current hyperparameter values
            rng: Random number generator
            mutation_strength: Relative strength of mutation
            mutation_probability: Probability of mutating each parameter

        Returns:
            Dictionary of mutated hyperparameter values.
        """
        if rng is None:
            rng = np.random.default_rng()

        result = {}
        for name, param in self.get_all_params().items():
            current_value = values.get(name, param.sample(rng))
            if rng.random() < mutation_probability:
                result[name] = param.mutate(current_value, rng, mutation_strength)
            else:
                result[name] = current_value
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "learning_rate": self.learning_rate.to_dict(),
            "weight_decay": self.weight_decay.to_dict(),
            "c_puct": self.c_puct.to_dict(),
            "dirichlet_alpha": self.dirichlet_alpha.to_dict(),
            "temperature": self.temperature.to_dict(),
            "temperature_initial": self.temperature_initial.to_dict(),
            "temperature_final": self.temperature_final.to_dict(),
            "temperature_decay_steps": self.temperature_decay_steps.to_dict(),
            "batch_size": self.batch_size.to_dict(),
            "num_simulations": self.num_simulations.to_dict(),
            "replay_buffer_size": self.replay_buffer_size.to_dict(),
            "custom_params": {k: v.to_dict() for k, v in self.custom_params.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HyperparameterSpace:
        """Create from dictionary."""
        custom_params = {}
        if "custom_params" in data:
            for k, v in data["custom_params"].items():
                custom_params[k] = HyperparameterRange.from_dict(v)

        return cls(
            learning_rate=HyperparameterRange.from_dict(data.get("learning_rate", {}))
            if "learning_rate" in data
            else HyperparameterRange("learning_rate", 1e-5, 1e-2, "log"),
            weight_decay=HyperparameterRange.from_dict(data.get("weight_decay", {}))
            if "weight_decay" in data
            else HyperparameterRange("weight_decay", 1e-6, 1e-3, "log"),
            c_puct=HyperparameterRange.from_dict(data.get("c_puct", {}))
            if "c_puct" in data
            else HyperparameterRange("c_puct", 0.5, 3.0),
            dirichlet_alpha=HyperparameterRange.from_dict(data.get("dirichlet_alpha", {}))
            if "dirichlet_alpha" in data
            else HyperparameterRange("dirichlet_alpha", 0.1, 1.0),
            temperature=HyperparameterRange.from_dict(data.get("temperature", {}))
            if "temperature" in data
            else HyperparameterRange("temperature", 0.5, 2.0),
            temperature_initial=HyperparameterRange.from_dict(data.get("temperature_initial", {}))
            if "temperature_initial" in data
            else HyperparameterRange("temperature_initial", 1.0, 2.5),
            temperature_final=HyperparameterRange.from_dict(data.get("temperature_final", {}))
            if "temperature_final" in data
            else HyperparameterRange("temperature_final", 0.1, 1.0),
            temperature_decay_steps=HyperparameterRange.from_dict(
                data.get("temperature_decay_steps", {})
            )
            if "temperature_decay_steps" in data
            else HyperparameterRange("temperature_decay_steps", 10, 50, discrete=True),
            batch_size=HyperparameterRange.from_dict(data.get("batch_size", {}))
            if "batch_size" in data
            else HyperparameterRange("batch_size", 32, 512, discrete=True),
            num_simulations=HyperparameterRange.from_dict(data.get("num_simulations", {}))
            if "num_simulations" in data
            else HyperparameterRange("num_simulations", 25, 400, discrete=True),
            replay_buffer_size=HyperparameterRange.from_dict(data.get("replay_buffer_size", {}))
            if "replay_buffer_size" in data
            else HyperparameterRange("replay_buffer_size", 10000, 200000, discrete=True),
            custom_params=custom_params,
        )


@dataclass
class AgentState:
    """State of an individual agent in the population.

    Attributes:
        agent_id: Unique identifier for the agent
        hyperparameters: Current hyperparameter values
        fitness_score: Current fitness score (e.g., win rate)
        fitness_history: History of fitness scores
        games_played: Total games played
        training_steps: Total training steps completed
        generation: Generation number (increments on exploitation)
        parent_id: ID of parent agent (if created via exploitation)
        model_path: Path to saved model weights
        elo_rating: ELO rating for tournament-based fitness evaluation
        created_at: Timestamp of agent creation
        last_updated: Timestamp of last update
    """

    agent_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    hyperparameters: dict[str, float | int | Any] = field(default_factory=dict)
    fitness_score: float = 0.0
    fitness_history: list[float] = field(default_factory=list)
    games_played: int = 0
    training_steps: int = 0
    generation: int = 0
    parent_id: str | None = None
    model_path: str | None = None
    elo_rating: float = 1200.0
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def update_fitness(self, new_fitness: float, max_history: int = 100) -> None:
        """Update fitness score and history.

        Args:
            new_fitness: New fitness score
            max_history: Maximum length of fitness history
        """
        self.fitness_score = new_fitness
        self.fitness_history.append(new_fitness)
        if len(self.fitness_history) > max_history:
            self.fitness_history = self.fitness_history[-max_history:]
        self.last_updated = datetime.now(UTC).isoformat()

    def get_smoothed_fitness(self, window: int = 10) -> float:
        """Get smoothed fitness score using recent history.

        Args:
            window: Number of recent scores to average

        Returns:
            Smoothed fitness score.
        """
        if not self.fitness_history:
            return self.fitness_score
        recent = self.fitness_history[-window:]
        return sum(recent) / len(recent)

    def clone_with_mutations(
        self,
        hyperparameter_space: HyperparameterSpace,
        rng: np.random.Generator | None = None,
        mutation_strength: float = 0.2,
        mutation_probability: float = 0.8,
    ) -> AgentState:
        """Create a mutated clone of this agent.

        Args:
            hyperparameter_space: Space defining valid mutations
            rng: Random number generator
            mutation_strength: Mutation strength
            mutation_probability: Probability of mutating each parameter

        Returns:
            New AgentState with mutated hyperparameters.
        """
        mutated_params = hyperparameter_space.mutate_all(
            self.hyperparameters,
            rng=rng,
            mutation_strength=mutation_strength,
            mutation_probability=mutation_probability,
        )

        return AgentState(
            hyperparameters=mutated_params,
            generation=self.generation + 1,
            parent_id=self.agent_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "hyperparameters": self.hyperparameters,
            "fitness_score": self.fitness_score,
            "fitness_history": self.fitness_history,
            "games_played": self.games_played,
            "training_steps": self.training_steps,
            "generation": self.generation,
            "parent_id": self.parent_id,
            "model_path": self.model_path,
            "elo_rating": self.elo_rating,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        """Create from dictionary."""
        return cls(
            agent_id=data.get("agent_id", str(uuid.uuid4())[:8]),
            hyperparameters=data.get("hyperparameters", {}),
            fitness_score=data.get("fitness_score", 0.0),
            fitness_history=data.get("fitness_history", []),
            games_played=data.get("games_played", 0),
            training_steps=data.get("training_steps", 0),
            generation=data.get("generation", 0),
            parent_id=data.get("parent_id"),
            model_path=data.get("model_path"),
            elo_rating=data.get("elo_rating", 1200.0),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            last_updated=data.get("last_updated", datetime.now(UTC).isoformat()),
        )

    def save(self, path: str | Path) -> None:
        """Save agent state to JSON file.

        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> AgentState:
        """Load agent state from JSON file.

        Args:
            path: Path to load from

        Returns:
            Loaded AgentState.
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class PBTConfig:
    """Configuration for Population-Based Training.

    All parameters can be overridden via environment variables with
    the prefix CIV6AZ_PBT_.

    Attributes:
        population_size: Number of agents in the population
        min_population_size: Minimum population size before warning
        evaluation_interval: Training steps between fitness evaluations
        evaluation_games: Number of games per fitness evaluation
        selection_method: Selection method for exploitation
        truncation_fraction: Fraction of population to replace
        mutation_probability: Probability of mutating each hyperparameter
        mutation_strength: Relative strength of mutations
        exploit_threshold: Fitness percentile below which agents are replaced
        copy_weights: Whether to copy weights during exploitation
        resample_probability: Probability of resampling instead of mutating
        hyperparameter_space: Search space for hyperparameters
        checkpoint_dir: Directory for saving checkpoints
        checkpoint_interval: Steps between checkpoints
        log_dir: Directory for logs
        wandb_enabled: Whether to enable W&B logging
        wandb_project: W&B project name
        wandb_tags: Tags for W&B runs
    """

    # Population settings
    population_size: int = 16
    min_population_size: int = 4

    # Evaluation settings
    evaluation_interval: int = 1000
    evaluation_games: int = 20
    fitness_smoothing_window: int = 10

    # Selection settings
    selection_method: str = "truncation"
    truncation_fraction: float = 0.2
    tournament_size: int = 3

    # Mutation settings
    mutation_probability: float = 0.8
    mutation_strength: float = 0.2
    resample_probability: float = 0.05

    # Exploitation settings
    exploit_threshold: float = 0.2
    copy_weights: bool = True

    # Search space
    hyperparameter_space: HyperparameterSpace = field(default_factory=HyperparameterSpace)

    # Checkpointing
    checkpoint_dir: str = "checkpoints/pbt"
    checkpoint_interval: int = 5000
    max_checkpoints: int = 10

    # Logging
    log_dir: str = "logs/pbt"
    wandb_enabled: bool = True
    wandb_project: str = "civ6-alphazero-pbt"
    wandb_tags: list[str] = field(default_factory=list)

    # Convergence detection
    convergence_threshold: float = 0.01
    convergence_window: int = 50

    # Runtime
    num_generations: int = 50

    # Cross-agent evaluation settings
    evaluation_strategy: str = "round_robin"  # "round_robin" or "top_k_vs_best"
    evaluation_games_per_match: int = 10
    elo_initial_rating: float = 1200.0
    elo_k_factor: float = 32.0

    # Baseline benchmarking
    baseline_interval: int = 5  # Run baselines every N generations (0=never)
    baseline_agents: list[str] = field(default_factory=lambda: ["random", "heuristic"])

    # Checkpoint retention
    checkpoint_retention_generations: int = 5  # Keep last N gens (0=unlimited)

    # Dynamic strategy switching
    evaluation_strategy_switch_gen: int = 0  # Switch to top_k_vs_best at gen N (0=never)

    ENV_PREFIX: str = field(default="CIV6AZ_PBT_", repr=False)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.population_size < self.min_population_size:
            raise ValueError(
                f"population_size ({self.population_size}) must be >= "
                f"min_population_size ({self.min_population_size})"
            )
        if not 0 < self.truncation_fraction < 1:
            raise ValueError("truncation_fraction must be between 0 and 1")
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError("mutation_probability must be between 0 and 1")
        if not 0 < self.mutation_strength <= 1:
            raise ValueError("mutation_strength must be between 0 and 1")
        _valid_strategies = ("round_robin", "top_k_vs_best")
        if self.evaluation_strategy not in _valid_strategies:
            raise ValueError(
                f"evaluation_strategy must be one of {sorted(_valid_strategies)}, "
                f"got '{self.evaluation_strategy}'"
            )
        if self.evaluation_games_per_match < 1:
            raise ValueError("evaluation_games_per_match must be >= 1")
        if self.elo_initial_rating <= 0:
            raise ValueError("elo_initial_rating must be > 0")
        if self.elo_k_factor <= 0:
            raise ValueError("elo_k_factor must be > 0")
        if self.baseline_interval < 0:
            raise ValueError("baseline_interval must be >= 0")
        if self.checkpoint_retention_generations < 0:
            raise ValueError("checkpoint_retention_generations must be >= 0")
        if self.evaluation_strategy_switch_gen < 0:
            raise ValueError("evaluation_strategy_switch_gen must be >= 0")

    @classmethod
    def from_yaml(cls, path: str | Path) -> PBTConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            PBTConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls._from_dict(data)

    @classmethod
    def from_env(cls, base_config: PBTConfig | None = None) -> PBTConfig:
        """Create configuration with environment variable overrides.

        Args:
            base_config: Base configuration to override

        Returns:
            PBTConfig with environment overrides applied
        """
        config = base_config or cls()
        prefix = cls.ENV_PREFIX if hasattr(cls, "ENV_PREFIX") else "CIV6AZ_PBT_"

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            param_name = key[len(prefix) :].lower()

            if not hasattr(config, param_name):
                continue

            current_value = getattr(config, param_name)

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

                setattr(config, param_name, typed_value)
            except (ValueError, yaml.YAMLError):
                pass

        # Re-validate after overrides
        config.__post_init__()
        return config

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> PBTConfig:
        """Create from dictionary, handling nested objects."""
        # Handle hyperparameter_space separately
        hp_space_data = data.pop("hyperparameter_space", None)

        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values() if f.name != "ENV_PREFIX"}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        config = cls(**filtered_data)

        if hp_space_data:
            config.hyperparameter_space = HyperparameterSpace.from_dict(hp_space_data)

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "population_size": self.population_size,
            "min_population_size": self.min_population_size,
            "evaluation_interval": self.evaluation_interval,
            "evaluation_games": self.evaluation_games,
            "fitness_smoothing_window": self.fitness_smoothing_window,
            "selection_method": self.selection_method,
            "truncation_fraction": self.truncation_fraction,
            "tournament_size": self.tournament_size,
            "mutation_probability": self.mutation_probability,
            "mutation_strength": self.mutation_strength,
            "resample_probability": self.resample_probability,
            "exploit_threshold": self.exploit_threshold,
            "copy_weights": self.copy_weights,
            "hyperparameter_space": self.hyperparameter_space.to_dict(),
            "checkpoint_dir": self.checkpoint_dir,
            "checkpoint_interval": self.checkpoint_interval,
            "max_checkpoints": self.max_checkpoints,
            "log_dir": self.log_dir,
            "wandb_enabled": self.wandb_enabled,
            "wandb_project": self.wandb_project,
            "wandb_tags": self.wandb_tags,
            "convergence_threshold": self.convergence_threshold,
            "convergence_window": self.convergence_window,
            "evaluation_strategy": self.evaluation_strategy,
            "evaluation_games_per_match": self.evaluation_games_per_match,
            "elo_initial_rating": self.elo_initial_rating,
            "elo_k_factor": self.elo_k_factor,
            "baseline_interval": self.baseline_interval,
            "baseline_agents": self.baseline_agents,
            "checkpoint_retention_generations": self.checkpoint_retention_generations,
            "evaluation_strategy_switch_gen": self.evaluation_strategy_switch_gen,
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def get_num_to_replace(self) -> int:
        """Get number of agents to replace during exploitation.

        Returns:
            Number of agents to replace.
        """
        return max(1, int(self.population_size * self.truncation_fraction))

    def get_selection_method_enum(self) -> SelectionMethod:
        """Get selection method as enum.

        Returns:
            SelectionMethod enum value.
        """
        return SelectionMethod(self.selection_method)
