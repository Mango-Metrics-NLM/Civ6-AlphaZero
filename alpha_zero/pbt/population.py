"""Population management for Population-Based Training.

This module provides the PopulationManager class for managing a population
of agents with different hyperparameters during training.

Usage:
    from alpha_zero.pbt.population import PopulationManager
    from alpha_zero.pbt.config import PBTConfig

    config = PBTConfig(population_size=16)
    manager = PopulationManager(config)

    # Initialize population
    manager.initialize()

    # Get agent for training
    agent = manager.get_agent("agent_id")

    # Update fitness after evaluation
    manager.update_fitness("agent_id", win_rate)
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from alpha_zero.pbt.config import (
    AgentState,
    PBTConfig,
    SelectionMethod,
)

logger = logging.getLogger(__name__)


@dataclass
class PopulationStats:
    """Statistics about the current population.

    Attributes:
        generation: Current generation number
        mean_fitness: Mean fitness across population
        max_fitness: Maximum fitness in population
        min_fitness: Minimum fitness in population
        std_fitness: Standard deviation of fitness
        hyperparameter_stats: Statistics for each hyperparameter
        num_active: Number of active agents
        total_games: Total games played by all agents
        total_steps: Total training steps by all agents
    """

    generation: int = 0
    mean_fitness: float = 0.0
    max_fitness: float = 0.0
    min_fitness: float = 0.0
    std_fitness: float = 0.0
    hyperparameter_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    num_active: int = 0
    total_games: int = 0
    total_steps: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "generation": self.generation,
            "mean_fitness": self.mean_fitness,
            "max_fitness": self.max_fitness,
            "min_fitness": self.min_fitness,
            "std_fitness": self.std_fitness,
            "hyperparameter_stats": self.hyperparameter_stats,
            "num_active": self.num_active,
            "total_games": self.total_games,
            "total_steps": self.total_steps,
            "timestamp": self.timestamp,
        }


class PopulationManager:
    """Manages a population of agents for Population-Based Training.

    This class handles:
    - Population initialization with random hyperparameters
    - Agent state tracking (fitness, hyperparameters, weights)
    - Selection for exploitation (identifying weak/strong agents)
    - Exploitation (copying weights from strong to weak agents)
    - Exploration (mutating hyperparameters)
    - Checkpointing and persistence

    Example:
        >>> config = PBTConfig(population_size=8)
        >>> manager = PopulationManager(config)
        >>> manager.initialize()
        >>> for agent_id in manager.get_agent_ids():
        ...     params = manager.get_hyperparameters(agent_id)
        ...     # Train agent with params...
        ...     manager.update_fitness(agent_id, win_rate)
        >>> manager.exploit_and_explore()
    """

    def __init__(
        self,
        config: PBTConfig,
        rng: np.random.Generator | None = None,
    ):
        """Initialize population manager.

        Args:
            config: PBT configuration
            rng: Random number generator for reproducibility
        """
        self.config = config
        self.rng = rng or np.random.default_rng()

        self._agents: dict[str, AgentState] = {}
        self._generation: int = 0
        self._stats_history: list[PopulationStats] = []

        # Callbacks for weight management
        self._save_weights_fn: Callable[[str, str], None] | None = None
        self._load_weights_fn: Callable[[str, str], None] | None = None
        self._copy_weights_fn: Callable[[str, str], None] | None = None

        logger.info(
            f"PopulationManager initialized with population_size={config.population_size}"
        )

    def initialize(self, existing_agents: list[AgentState] | None = None) -> None:
        """Initialize the population with random hyperparameters.

        Args:
            existing_agents: Optional list of existing agents to use
        """
        self._agents.clear()

        if existing_agents:
            for agent in existing_agents:
                self._agents[agent.agent_id] = agent
            logger.info(f"Initialized population with {len(existing_agents)} existing agents")
        else:
            for _i in range(self.config.population_size):
                hyperparameters = self.config.hyperparameter_space.sample_all(self.rng)
                agent = AgentState(hyperparameters=hyperparameters)
                self._agents[agent.agent_id] = agent

            logger.info(
                f"Initialized population with {self.config.population_size} random agents"
            )

        self._generation = 0

    def get_agent_ids(self) -> list[str]:
        """Get list of all agent IDs.

        Returns:
            List of agent IDs.
        """
        return list(self._agents.keys())

    def get_agent(self, agent_id: str) -> AgentState | None:
        """Get agent state by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentState or None if not found.
        """
        return self._agents.get(agent_id)

    def get_all_agents(self) -> list[AgentState]:
        """Get all agents in the population.

        Returns:
            List of all AgentState objects.
        """
        return list(self._agents.values())

    def get_hyperparameters(self, agent_id: str) -> dict[str, Any]:
        """Get hyperparameters for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Dictionary of hyperparameter values.

        Raises:
            KeyError: If agent not found.
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            raise KeyError(f"Agent not found: {agent_id}")
        return agent.hyperparameters.copy()

    def update_fitness(
        self,
        agent_id: str,
        fitness: float,
        games_played: int = 0,
        training_steps: int = 0,
    ) -> None:
        """Update fitness score for an agent.

        Args:
            agent_id: Agent identifier
            fitness: New fitness score
            games_played: Additional games played
            training_steps: Additional training steps
        """
        agent = self._agents.get(agent_id)
        if agent is None:
            logger.warning(f"Agent not found for fitness update: {agent_id}")
            return

        agent.update_fitness(
            fitness, max_history=self.config.fitness_smoothing_window * 2
        )
        agent.games_played += games_played
        agent.training_steps += training_steps

        logger.debug(
            f"Updated fitness for {agent_id}: {fitness:.4f} "
            f"(smoothed: {agent.get_smoothed_fitness():.4f})"
        )

    def get_fitness_ranking(self) -> list[tuple[str, float]]:
        """Get agents sorted by fitness (highest first).

        Returns:
            List of (agent_id, fitness) tuples sorted by fitness.
        """
        agents_with_fitness = [
            (agent_id, agent.get_smoothed_fitness(self.config.fitness_smoothing_window))
            for agent_id, agent in self._agents.items()
        ]
        agents_with_fitness.sort(key=lambda x: x[1], reverse=True)
        return agents_with_fitness

    def get_agents_to_replace(self) -> list[str]:
        """Get IDs of agents that should be replaced.

        Returns:
            List of agent IDs in the bottom percentile.
        """
        ranking = self.get_fitness_ranking()
        num_to_replace = self.config.get_num_to_replace()

        # Return the bottom agents
        return [agent_id for agent_id, _ in ranking[-num_to_replace:]]

    def get_agents_to_copy_from(self) -> list[str]:
        """Get IDs of top-performing agents to copy from.

        Returns:
            List of agent IDs in the top percentile.
        """
        ranking = self.get_fitness_ranking()
        num_top = self.config.get_num_to_replace()

        # Return the top agents
        return [agent_id for agent_id, _ in ranking[:num_top]]

    def exploit_and_explore(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Perform exploitation and exploration step.

        This identifies weak agents, copies weights from strong agents,
        and mutates hyperparameters.

        Returns:
            List of (weak_agent_id, strong_agent_id, new_hyperparameters) tuples.
        """
        weak_agents = self.get_agents_to_replace()
        strong_agents = self.get_agents_to_copy_from()

        if not weak_agents or not strong_agents:
            logger.info("No agents to replace")
            return []

        changes = []

        for weak_id in weak_agents:
            # Select a strong agent to copy from
            strong_id = self._select_strong_agent(strong_agents)
            strong_agent = self._agents[strong_id]
            weak_agent = self._agents[weak_id]

            # Copy weights if enabled
            if self.config.copy_weights and self._copy_weights_fn:
                try:
                    self._copy_weights_fn(strong_id, weak_id)
                    logger.debug(f"Copied weights from {strong_id} to {weak_id}")
                except Exception as e:
                    logger.warning(f"Failed to copy weights: {e}")

            # Decide whether to resample or mutate
            if self.rng.random() < self.config.resample_probability:
                # Resample from scratch
                new_params = self.config.hyperparameter_space.sample_all(self.rng)
                logger.debug(f"Resampled hyperparameters for {weak_id}")
            else:
                # Mutate from strong agent
                new_params = self.config.hyperparameter_space.mutate_all(
                    strong_agent.hyperparameters,
                    rng=self.rng,
                    mutation_strength=self.config.mutation_strength,
                    mutation_probability=self.config.mutation_probability,
                )
                logger.debug(f"Mutated hyperparameters for {weak_id} from {strong_id}")

            # Update weak agent
            weak_agent.hyperparameters = new_params
            weak_agent.parent_id = strong_id
            weak_agent.generation = self._generation + 1
            weak_agent.fitness_history.clear()  # Reset fitness history
            weak_agent.last_updated = datetime.utcnow().isoformat()

            changes.append((weak_id, strong_id, new_params))

        self._generation += 1
        logger.info(
            f"Generation {self._generation}: replaced {len(changes)} agents"
        )

        return changes

    def _select_strong_agent(self, strong_agents: list[str]) -> str:
        """Select a strong agent to copy from based on selection method.

        Args:
            strong_agents: List of strong agent IDs

        Returns:
            Selected agent ID.
        """
        method = self.config.get_selection_method_enum()

        if method == SelectionMethod.TRUNCATION:
            # Uniform random from top agents
            return self.rng.choice(strong_agents)

        elif method == SelectionMethod.TOURNAMENT:
            # Tournament selection
            tournament = self.rng.choice(
                strong_agents,
                size=min(self.config.tournament_size, len(strong_agents)),
                replace=False,
            )
            best_id = None
            best_fitness = float("-inf")
            for agent_id in tournament:
                fitness = self._agents[agent_id].get_smoothed_fitness()
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_id = agent_id
            return best_id or strong_agents[0]

        elif method == SelectionMethod.ROULETTE:
            # Roulette wheel selection based on fitness
            fitnesses = np.array([
                max(0.01, self._agents[aid].get_smoothed_fitness())
                for aid in strong_agents
            ])
            probabilities = fitnesses / fitnesses.sum()
            return self.rng.choice(strong_agents, p=probabilities)

        elif method == SelectionMethod.RANK:
            # Rank-based selection
            ranks = np.arange(len(strong_agents), 0, -1)  # Highest rank for best
            probabilities = ranks / ranks.sum()
            return self.rng.choice(strong_agents, p=probabilities)

        else:
            return self.rng.choice(strong_agents)

    def compute_stats(self) -> PopulationStats:
        """Compute current population statistics.

        Returns:
            PopulationStats object with current statistics.
        """
        fitnesses = [
            agent.get_smoothed_fitness(self.config.fitness_smoothing_window)
            for agent in self._agents.values()
        ]

        if not fitnesses:
            return PopulationStats()

        # Hyperparameter statistics
        hp_stats = {}
        for param_name in self.config.hyperparameter_space.get_all_params().keys():
            values = [
                agent.hyperparameters.get(param_name, 0)
                for agent in self._agents.values()
                if param_name in agent.hyperparameters
            ]
            if values:
                hp_stats[param_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }

        stats = PopulationStats(
            generation=self._generation,
            mean_fitness=float(np.mean(fitnesses)),
            max_fitness=float(np.max(fitnesses)),
            min_fitness=float(np.min(fitnesses)),
            std_fitness=float(np.std(fitnesses)),
            hyperparameter_stats=hp_stats,
            num_active=len(self._agents),
            total_games=sum(a.games_played for a in self._agents.values()),
            total_steps=sum(a.training_steps for a in self._agents.values()),
        )

        self._stats_history.append(stats)
        return stats

    def is_converged(self) -> bool:
        """Check if population has converged.

        Returns:
            True if population fitness variance is below threshold.
        """
        if len(self._stats_history) < self.config.convergence_window:
            return False

        recent_stats = self._stats_history[-self.config.convergence_window:]
        fitness_std = np.mean([s.std_fitness for s in recent_stats])

        return fitness_std < self.config.convergence_threshold

    def get_best_agent(self) -> AgentState | None:
        """Get the best performing agent.

        Returns:
            AgentState with highest fitness, or None if empty.
        """
        if not self._agents:
            return None

        ranking = self.get_fitness_ranking()
        best_id = ranking[0][0]
        return self._agents[best_id]

    def set_weight_callbacks(
        self,
        save_fn: Callable[[str, str], None] | None = None,
        load_fn: Callable[[str, str], None] | None = None,
        copy_fn: Callable[[str, str], None] | None = None,
    ) -> None:
        """Set callbacks for weight management.

        Args:
            save_fn: Function to save weights (agent_id, path)
            load_fn: Function to load weights (agent_id, path)
            copy_fn: Function to copy weights (from_id, to_id)
        """
        self._save_weights_fn = save_fn
        self._load_weights_fn = load_fn
        self._copy_weights_fn = copy_fn

    def save_checkpoint(self, path: str | Path) -> None:
        """Save population state to checkpoint.

        Args:
            path: Directory to save checkpoint
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save population metadata
        metadata = {
            "generation": self._generation,
            "num_agents": len(self._agents),
            "config": self.config.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save each agent
        agents_dir = path / "agents"
        agents_dir.mkdir(exist_ok=True)
        for agent_id, agent in self._agents.items():
            agent.save(agents_dir / f"{agent_id}.json")

        # Save weights if callback is set
        if self._save_weights_fn:
            weights_dir = path / "weights"
            weights_dir.mkdir(exist_ok=True)
            for agent_id in self._agents:
                try:
                    self._save_weights_fn(agent_id, str(weights_dir / f"{agent_id}.pt"))
                except Exception as e:
                    logger.warning(f"Failed to save weights for {agent_id}: {e}")

        # Save stats history
        stats_data = [s.to_dict() for s in self._stats_history[-100:]]
        with open(path / "stats_history.json", "w") as f:
            json.dump(stats_data, f, indent=2)

        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load population state from checkpoint.

        Args:
            path: Directory containing checkpoint
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load metadata
        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        self._generation = metadata.get("generation", 0)

        # Load agents
        self._agents.clear()
        agents_dir = path / "agents"
        if agents_dir.exists():
            for agent_file in agents_dir.glob("*.json"):
                agent = AgentState.load(agent_file)
                self._agents[agent.agent_id] = agent

        # Load weights if callback is set
        if self._load_weights_fn:
            weights_dir = path / "weights"
            if weights_dir.exists():
                for agent_id in self._agents:
                    weight_path = weights_dir / f"{agent_id}.pt"
                    if weight_path.exists():
                        try:
                            self._load_weights_fn(agent_id, str(weight_path))
                        except Exception as e:
                            logger.warning(f"Failed to load weights for {agent_id}: {e}")

        # Load stats history
        stats_file = path / "stats_history.json"
        if stats_file.exists():
            with open(stats_file) as f:
                json.load(f)
            # Note: We don't fully restore stats objects, just keep the data
            self._stats_history = []

        logger.info(
            f"Loaded checkpoint from {path}: "
            f"generation={self._generation}, agents={len(self._agents)}"
        )

    def cleanup_old_checkpoints(self, checkpoint_dir: str | Path) -> None:
        """Remove old checkpoints, keeping only the most recent.

        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return

        # Find all checkpoint directories
        checkpoints = sorted(
            [d for d in checkpoint_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        # Remove old checkpoints
        for old_checkpoint in checkpoints[self.config.max_checkpoints:]:
            try:
                shutil.rmtree(old_checkpoint)
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")

    @property
    def generation(self) -> int:
        """Get current generation number."""
        return self._generation

    @property
    def population_size(self) -> int:
        """Get current population size."""
        return len(self._agents)

    def __len__(self) -> int:
        """Get population size."""
        return len(self._agents)

    def __iter__(self):
        """Iterate over agents."""
        return iter(self._agents.values())
