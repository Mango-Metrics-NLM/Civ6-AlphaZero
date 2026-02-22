from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
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
from alpha_zero.utils.compat import UTC

logger = logging.getLogger(__name__)


@dataclass
class PopulationStats:
    """Statistics about the population state."""

    generation: int = 0
    mean_fitness: float = 0.0
    max_fitness: float = 0.0
    min_fitness: float = 0.0
    std_fitness: float = 0.0
    hyperparameter_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    num_active: int = 0
    total_games: int = 0
    total_steps: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
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
    """Manages the lifecycle of PBT agents."""

    def __init__(
        self,
        config: PBTConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize population manager.

        Args:
            config: PBT configuration
            rng: Random number generator for reproducibility
        """
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.agents: dict[str, AgentState] = {}
        self.generation = 0
        self.stats_history: list[PopulationStats] = []

        # Weight operation callbacks
        self._save_weights_fn: Callable[[str, str], None] | None = None
        self._load_weights_fn: Callable[[str, str], None] | None = None
        self._copy_weights_fn: Callable[[str, str], None] | None = None

    @property
    def population_size(self) -> int:
        return len(self.agents)

    def __len__(self) -> int:
        return len(self.agents)

    def __iter__(self) -> Iterator[AgentState]:
        return iter(self.agents.values())

    def initialize(self, existing_agents: list[AgentState] | None = None) -> None:
        """Initialize population."""
        self.agents = {}
        self.generation = 0

        if existing_agents:
            for agent in existing_agents:
                self.agents[agent.agent_id] = agent
                # Check for max generation
                self.generation = max(self.generation, agent.generation)
        else:
            for _ in range(self.config.population_size):
                params = self.config.hyperparameter_space.sample_all(self.rng)
                agent = AgentState(hyperparameters=params)
                self.agents[agent.agent_id] = agent

        logger.info(
            f"Initialized population with {len(self.agents)} agents (Gen {self.generation})"
        )

    def get_agent(self, agent_id: str) -> AgentState | None:
        return self.agents.get(agent_id)

    def get_agent_ids(self) -> list[str]:
        return list(self.agents.keys())

    def get_all_agents(self) -> list[AgentState]:
        return list(self.agents.values())

    def get_hyperparameters(self, agent_id: str) -> dict[str, Any]:
        return self.agents[agent_id].hyperparameters

    def update_fitness(
        self,
        agent_id: str,
        fitness: float,
        games_played: int = 0,
        training_steps: int = 0,
    ) -> None:
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found update_fitness")
            return

        agent = self.agents[agent_id]
        agent.update_fitness(fitness, self.config.fitness_smoothing_window)
        if games_played:
            agent.games_played += games_played
        if training_steps:
            agent.training_steps += training_steps

    def get_fitness_ranking(self) -> list[tuple[str, float]]:
        """Return list of (agent_id, score) sorted descending."""
        scores = [
            (aid, agent.get_smoothed_fitness(self.config.fitness_smoothing_window))
            for aid, agent in self.agents.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def get_best_agent(self) -> AgentState | None:
        if not self.agents:
            return None
        ranking = self.get_fitness_ranking()
        best_id = ranking[0][0]
        return self.agents[best_id]

    def get_agents_to_replace(self) -> list[str]:
        """Identify bottom performing agents."""
        num_replace = self.config.get_num_to_replace()
        ranking = self.get_fitness_ranking()
        # Bottom N agents
        to_replace = [aid for aid, _ in ranking[-num_replace:]]
        return to_replace

    def get_agents_to_copy_from(self) -> list[str]:
        """Identify top performing agents."""
        # Usually fraction logic implies top X%?
        num_replace = self.config.get_num_to_replace()
        # Standard PBT: Top 20% provides weights for Bottom 20%.
        # Here we use config.truncation_fraction for both.
        ranking = self.get_fitness_ranking()
        top_agents = [aid for aid, _ in ranking[:num_replace]]
        return top_agents

    def _select_strong_agent(self, candidates: list[str]) -> str:
        """Select a strong agent based on the configured method."""
        method = self.config.get_selection_method_enum()

        if not candidates:
            return next(iter(self.agents.values())).agent_id

        if method == SelectionMethod.TRUNCATION:
            # Uniformly sample from top candidates
            return self.rng.choice(candidates)

        elif method == SelectionMethod.TOURNAMENT:
            # Pick k random agents from candidates and return best
            k = min(len(candidates), self.config.tournament_size)
            pool = self.rng.choice(candidates, size=k, replace=False)
            window = self.config.fitness_smoothing_window
            best_id = max(pool, key=lambda aid: self.agents[aid].get_smoothed_fitness(window))
            return best_id

        elif method == SelectionMethod.RANK:
            # Probability proportional to rank (ascending: worst=rank 1, best=rank N)
            window = self.config.fitness_smoothing_window
            candidates = sorted(
                candidates,
                key=lambda aid: self.agents[aid].get_smoothed_fitness(window),
            )
            ranks = np.arange(1, len(candidates) + 1)
            probs = ranks / ranks.sum()
            return self.rng.choice(candidates, p=probs)

        elif method == SelectionMethod.ROULETTE:
            # Proportional to fitness
            window = self.config.fitness_smoothing_window
            fitnesses = np.array(
                [self.agents[aid].get_smoothed_fitness(window) for aid in candidates]
            )
            # If negative fitness possible, shift (softmax is safer)
            # Basic roulette:
            min_fit = fitnesses.min()
            if min_fit < 0:
                fitnesses -= min_fit
            total = fitnesses.sum()
            if total <= 1e-9:
                probs = np.ones_like(fitnesses) / len(fitnesses)
            else:
                probs = fitnesses / total
            return self.rng.choice(candidates, p=probs)

        # Fallback
        return self.rng.choice(candidates)

    def exploit_and_explore(self) -> list[tuple[str, str, dict[str, Any]]]:
        """Perform PBT step.

        Returns:
            List of (replaced_agent_id, source_agent_id, new_hyperparameters)
        """
        if not self.agents:
            return []

        to_replace = self.get_agents_to_replace()
        candidates = self.get_agents_to_copy_from()

        changes = []
        new_agents = {}

        # Keep non-replaced agents
        for aid, agent in self.agents.items():
            if aid not in to_replace:
                # Increment generation for survivors? PBT usually implies gen update on training steps.
                # But here we orchestrate gens.
                # Standard PBT: Only check exploit every X steps.
                # If we consider this function call as "Start of New Gen check"
                # Survivors move to next gen?
                # The prompt implies survivors just continue.
                # Let's increment survivor generation to keep in sync.
                # agent.generation += 1 # Optional
                new_agents[aid] = agent

        # Replace weak agents
        for weak_id in to_replace:
            # Exploit: Copy weights from strong agent
            strong_id = self._select_strong_agent(candidates)
            strong_agent = self.agents[strong_id]

            # Explore: Mutate
            # Use clone_with_mutations logic from AgentState but maintain original ID?
            # Or replace ID? Standard PBT often keeps ID to track slot, but effectively it's a new agent.
            # test_pbt_population expects (weak_id, strong_id, new_params) return.
            # And expects update to weak_id.

            # Create new agent state (logic inside AgentState mostly)
            # But we want to preserve weak_id?
            # Or we replace weak_agent object entirely but keep same ID?
            # "weak_before.hyperparameters" != "weak_after.hyperparameters" test constraint.

            if self.rng.random() < self.config.resample_probability:
                # Resample completely
                new_params = self.config.hyperparameter_space.sample_all(self.rng)
                child = AgentState(
                    agent_id=weak_id,  # Reuse ID
                    hyperparameters=new_params,
                    generation=self.generation + 1,
                    parent_id=None,  # Reset lineage for resample?
                )
            else:
                # Mutate from strong
                child = strong_agent.clone_with_mutations(
                    self.config.hyperparameter_space,
                    rng=self.rng,
                    mutation_strength=self.config.mutation_strength,
                    mutation_probability=self.config.mutation_probability,
                )
                child.agent_id = weak_id  # Enforce ID reuse

            # Copy weights if enabled
            if self.config.copy_weights and self._copy_weights_fn:
                try:
                    self._copy_weights_fn(strong_id, weak_id)
                except Exception:
                    logger.warning(
                        "Weight copy failed for %s -> %s; continuing with mutated hyperparameters only",
                        strong_id,
                        weak_id,
                        exc_info=True,
                    )

            new_agents[weak_id] = child
            changes.append((weak_id, strong_id, child.hyperparameters))

        self.agents = new_agents
        self.generation += 1
        return changes

    def compute_stats(self) -> PopulationStats:
        """Compute and return current stats."""
        fitnesses = [a.get_smoothed_fitness() for a in self.agents.values()]
        if not fitnesses:
            return PopulationStats()

        stats = PopulationStats(
            generation=self.generation,
            mean_fitness=float(np.mean(fitnesses)),
            max_fitness=float(np.max(fitnesses)),
            min_fitness=float(np.min(fitnesses)),
            std_fitness=float(np.std(fitnesses)),
            num_active=len(self.agents),
            total_games=sum(a.games_played for a in self.agents.values()),
            total_steps=sum(a.training_steps for a in self.agents.values()),
        )

        # Hyperparameter stats
        hp_stats = {}
        # Assume all agents have same keys
        if self.agents:
            keys = next(iter(self.agents.values())).hyperparameters.keys()
            for k in keys:
                vals = [
                    a.hyperparameters.get(k, 0)
                    for a in self.agents.values()
                    if isinstance(a.hyperparameters.get(k), (int, float))
                ]
                if vals:
                    hp_stats[k] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals)),
                    }
        stats.hyperparameter_stats = hp_stats
        self.stats_history.append(stats)
        return stats

    def is_converged(self) -> bool:
        """Check if population fitness has converged."""
        if len(self.stats_history) < self.config.convergence_window:
            return False

        recent = self.stats_history[-self.config.convergence_window :]
        # Simple check: Average std_fitness over window < threshold
        avg_std = np.mean([s.std_fitness for s in recent])
        return avg_std < self.config.convergence_threshold

    def set_weight_callbacks(
        self,
        save_fn: Callable[[str, str], None] | None = None,
        load_fn: Callable[[str, str], None] | None = None,
        copy_fn: Callable[[str, str], None] | None = None,
    ) -> None:
        self._save_weights_fn = save_fn
        self._load_weights_fn = load_fn
        self._copy_weights_fn = copy_fn

    def save_checkpoint(self, path: Path | str) -> None:
        """Save population state and weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Metadata (Gen, Stats)
        metadata = {
            "generation": self.generation,
            "timestamp": datetime.now(UTC).isoformat(),
            "config": self.config.to_dict(),
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save Stats History
        stats_data = [s.to_dict() for s in self.stats_history]
        with open(path / "stats_history.json", "w") as f:
            json.dump(stats_data, f, indent=2)

        # Save Agents
        agents_dir = path / "agents"
        agents_dir.mkdir(exist_ok=True)
        for aid, agent in self.agents.items():
            agent.save(agents_dir / f"{aid}.json")
            if self._save_weights_fn:
                # Save weights to separate dir or same
                weight_path = agents_dir / f"{aid}.weights"
                self._save_weights_fn(aid, str(weight_path))

    def load_checkpoint(self, path: Path | str) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path / "metadata.json") as f:
            metadata = json.load(f)
            self.generation = metadata["generation"]
            # Reload config? Maybe, or keep existing.

        # Load agents
        agents_dir = path / "agents"
        self.agents = {}
        for agent_file in agents_dir.glob("*.json"):
            agent = AgentState.load(agent_file)
            self.agents[agent.agent_id] = agent

            # Load weights callback
            if self._load_weights_fn:
                weight_path = agents_dir / f"{agent.agent_id}.weights"
                self._load_weights_fn(agent.agent_id, str(weight_path))

        logger.info(f"Loaded checkpoint from {path} (Gen {self.generation})")
