"""PBT Coordinator for integrating Population-Based Training with distributed training.

This module provides the PBTCoordinator class that orchestrates PBT
alongside the distributed training system.

Usage:
    from alpha_zero.pbt.coordinator import PBTCoordinator
    from alpha_zero.pbt.config import PBTConfig

    config = PBTConfig()
    coordinator = PBTCoordinator(config)

    # Run PBT training loop
    coordinator.run()
"""

from __future__ import annotations

import json
import logging
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from alpha_zero.pbt.config import PBTConfig
from alpha_zero.pbt.mutations import MutationConfig, MutationOperator
from alpha_zero.pbt.population import PopulationManager, PopulationStats
from alpha_zero.pbt.scheduler import (
    EvaluationResult,
    PBTScheduler,
)

logger = logging.getLogger(__name__)


@dataclass
class PBTProgress:
    """Progress tracking for PBT training.

    Attributes:
        current_step: Current training step
        generation: Current generation
        best_fitness: Best fitness observed
        mean_fitness: Current mean population fitness
        population_size: Current population size
        evaluations_completed: Number of evaluations
        exploit_explore_cycles: Number of exploit/explore cycles
        elapsed_time: Total elapsed time in seconds
        timestamp: Current timestamp
    """

    current_step: int = 0
    generation: int = 0
    best_fitness: float = 0.0
    mean_fitness: float = 0.0
    population_size: int = 0
    evaluations_completed: int = 0
    exploit_explore_cycles: int = 0
    elapsed_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_step": self.current_step,
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "mean_fitness": self.mean_fitness,
            "population_size": self.population_size,
            "evaluations_completed": self.evaluations_completed,
            "exploit_explore_cycles": self.exploit_explore_cycles,
            "elapsed_time": self.elapsed_time,
            "timestamp": self.timestamp,
        }


class PBTCoordinator:
    """Coordinates Population-Based Training with the training system.

    This class integrates PBT with the existing distributed training
    infrastructure, managing:
    - Population initialization and management
    - Scheduling of evaluations and exploit/explore cycles
    - Weight synchronization between agents
    - Checkpointing and recovery
    - Logging and metrics

    Example:
        >>> config = PBTConfig(population_size=16)
        >>> coordinator = PBTCoordinator(config)
        >>>
        >>> # Set up training callbacks
        >>> coordinator.set_training_callbacks(
        ...     train_fn=my_train_function,
        ...     evaluate_fn=my_evaluate_function,
        ... )
        >>>
        >>> # Run training
        >>> coordinator.run(max_steps=100000)
    """

    def __init__(
        self,
        config: PBTConfig,
        rng: np.random.Generator | None = None,
        wandb_logger: Any | None = None,
    ) -> None:
        """Initialize PBT coordinator.

        Args:
            config: PBT configuration
            rng: Random number generator
            wandb_logger: Optional W&B logger instance
        """
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.wandb_logger = wandb_logger

        # Initialize components
        self.population = PopulationManager(config, rng=self.rng)
        self.scheduler = PBTScheduler(config, rng=self.rng)
        self.mutation_operator = MutationOperator(
            MutationConfig(
                strategy="gaussian",
                strength=config.mutation_strength,
            ),
            rng=self.rng,
        )

        # State tracking
        self._current_step = 0
        self._start_time: float | None = None
        self._last_checkpoint_step = 0
        self._running = False
        self._shutdown_requested = False

        # Training callbacks
        self._train_step_fn: Callable[[str, dict], dict] | None = None
        self._evaluate_fn: Callable[[str], EvaluationResult] | None = None
        self._save_weights_fn: Callable[[str, str], None] | None = None
        self._load_weights_fn: Callable[[str, str], None] | None = None
        self._copy_weights_fn: Callable[[str, str], None] | None = None

        # Progress history
        self._progress_history: list[PBTProgress] = []
        self._stats_history: list[PopulationStats] = []

        logger.info(f"PBTCoordinator initialized with population_size={config.population_size}")

    def set_training_callbacks(
        self,
        train_step_fn: Callable[[str, dict], dict] | None = None,
        evaluate_fn: Callable[[str], EvaluationResult] | None = None,
        save_weights_fn: Callable[[str, str], None] | None = None,
        load_weights_fn: Callable[[str, str], None] | None = None,
        copy_weights_fn: Callable[[str, str], None] | None = None,
    ) -> None:
        """Set callbacks for training operations.

        Args:
            train_step_fn: Function to train agent (agent_id, hyperparams) -> metrics
            evaluate_fn: Function to evaluate agent (agent_id) -> EvaluationResult
            save_weights_fn: Function to save weights (agent_id, path) -> None
            load_weights_fn: Function to load weights (agent_id, path) -> None
            copy_weights_fn: Function to copy weights (from_id, to_id) -> None
        """
        self._train_step_fn = train_step_fn
        self._evaluate_fn = evaluate_fn
        self._save_weights_fn = save_weights_fn
        self._load_weights_fn = load_weights_fn
        self._copy_weights_fn = copy_weights_fn

        # Set callbacks on population manager
        self.population.set_weight_callbacks(
            save_fn=save_weights_fn,
            load_fn=load_weights_fn,
            copy_fn=copy_weights_fn,
        )

    def initialize(
        self,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        """Initialize the coordinator.

        Args:
            checkpoint_path: Optional path to resume from checkpoint
        """
        if checkpoint_path and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
            logger.info(f"Resumed from checkpoint: {checkpoint_path}")
        else:
            self.population.initialize()
            logger.info("Initialized fresh population")

        self._start_time = time.time()

    def run(
        self,
        max_steps: int = 1000000,
        max_generations: int | None = None,
    ) -> dict[str, Any]:
        """Run the PBT training loop.

        Args:
            max_steps: Maximum training steps
            max_generations: Maximum generations (optional)

        Returns:
            Dictionary with training results.
        """
        if not self.population.population_size:
            self.initialize()

        self._running = True
        self._setup_signal_handlers()

        logger.info(
            f"Starting PBT training: max_steps={max_steps}, max_generations={max_generations}"
        )

        try:
            while self._running and not self._shutdown_requested:
                # Check termination conditions
                if self._current_step >= max_steps:
                    logger.info(f"Reached max_steps={max_steps}")
                    break

                if max_generations and self.population.generation >= max_generations:
                    logger.info(f"Reached max_generations={max_generations}")
                    break

                if self.population.is_converged():
                    logger.info("Population converged, stopping")
                    break

                # Run training step for all agents
                self._run_training_step()

                # Check if evaluation is needed
                if self.scheduler.should_evaluate(self._current_step):
                    self._run_evaluation()

                # Check if exploit/explore is needed
                if self.scheduler.should_exploit_explore(self._current_step):
                    self._run_exploit_explore()

                # Check if checkpoint is needed
                if self.scheduler.should_checkpoint(self._current_step, self._last_checkpoint_step):
                    self._save_checkpoint()

                self._current_step += 1

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            self._running = False

        # Final checkpoint
        self._save_checkpoint()

        # Compute final results
        results = self._compute_final_results()
        logger.info(f"PBT training completed: {results}")

        return results

    def _run_training_step(self) -> None:
        """Run one training step for all agents."""
        if self._train_step_fn is None:
            return

        for agent_id in self.population.get_agent_ids():
            agent = self.population.get_agent(agent_id)
            if agent is None:
                continue

            try:
                metrics = self._train_step_fn(agent_id, agent.hyperparameters)
                agent.training_steps += 1

                # Log metrics
                if self.wandb_logger and self._current_step % 100 == 0:
                    self._log_training_metrics(agent_id, metrics)

            except Exception as e:
                logger.warning(f"Training step failed for {agent_id}: {e}")

    def _run_evaluation(self) -> None:
        """Run fitness evaluation for all agents."""
        logger.info(f"Running evaluation at step {self._current_step}")

        results = []
        for agent_id in self.population.get_agent_ids():
            if self._evaluate_fn:
                try:
                    result = self._evaluate_fn(agent_id)
                    results.append(result)

                    # Update population fitness
                    self.population.update_fitness(
                        agent_id,
                        result.fitness,
                        games_played=result.games_played,
                    )
                except Exception as e:
                    logger.warning(f"Evaluation failed for {agent_id}: {e}")
            else:
                # Use scheduler's evaluator
                agent = self.population.get_agent(agent_id)
                if agent:
                    result = self.scheduler.get_evaluator().evaluate_agent(agent)
                    results.append(result)
                    self.population.update_fitness(
                        agent_id,
                        result.fitness,
                        games_played=result.games_played,
                    )

        # Record evaluation
        self.scheduler.record_evaluation(self._current_step, results)

        # Compute and log stats
        stats = self.population.compute_stats()
        self._stats_history.append(stats)
        self._log_evaluation_results(results, stats)

        logger.info(
            f"Evaluation complete: mean_fitness={stats.mean_fitness:.4f}, "
            f"max_fitness={stats.max_fitness:.4f}"
        )

    def _run_exploit_explore(self) -> None:
        """Run exploit/explore cycle."""
        logger.info(f"Running exploit/explore at step {self._current_step}")

        changes = self.population.exploit_and_explore()
        self.scheduler.record_exploit_explore(self._current_step)
        self.mutation_operator.increment_generation()

        # Log changes
        for weak_id, strong_id, _new_params in changes:
            logger.debug(f"Replaced {weak_id} with mutated copy of {strong_id}")

        if self.wandb_logger:
            self._log_exploit_explore(changes)

        logger.info(
            f"Exploit/explore complete: replaced {len(changes)} agents, "
            f"generation={self.population.generation}"
        )

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_path = checkpoint_dir / f"checkpoint_{self._current_step}"

        self.population.save_checkpoint(checkpoint_path)

        # Save coordinator state
        state = {
            "current_step": self._current_step,
            "last_checkpoint_step": self._last_checkpoint_step,
            "start_time": self._start_time,
            "config": self.config.to_dict(),
        }
        with open(checkpoint_path / "coordinator_state.json", "w") as f:
            json.dump(state, f, indent=2)

        self._last_checkpoint_step = self._current_step

        # Cleanup old checkpoints
        self.population.cleanup_old_checkpoints(checkpoint_dir)

        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """Load from checkpoint.

        Args:
            path: Checkpoint directory path
        """
        path = Path(path)

        # Load population
        self.population.load_checkpoint(path)

        # Load coordinator state
        state_file = path / "coordinator_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            self._current_step = state.get("current_step", 0)
            self._last_checkpoint_step = state.get("last_checkpoint_step", 0)

        logger.info(f"Loaded checkpoint from {path}")

    def _compute_final_results(self) -> dict[str, Any]:
        """Compute final training results."""
        elapsed_time = time.time() - (self._start_time or 0)

        best_agent = self.population.get_best_agent()
        final_stats = self.population.compute_stats()

        return {
            "total_steps": self._current_step,
            "total_generations": self.population.generation,
            "elapsed_time": elapsed_time,
            "best_agent_id": best_agent.agent_id if best_agent else None,
            "best_fitness": final_stats.max_fitness,
            "mean_fitness": final_stats.mean_fitness,
            "best_hyperparameters": best_agent.hyperparameters if best_agent else {},
            "final_population_size": self.population.population_size,
        }

    def _log_training_metrics(self, agent_id: str, metrics: dict[str, Any]) -> None:
        """Log training metrics to W&B."""
        if self.wandb_logger is None:
            return

        try:
            import wandb

            wandb.log(
                {
                    f"pbt/agent/{agent_id}/loss": metrics.get("loss", 0),
                    f"pbt/agent/{agent_id}/policy_loss": metrics.get("policy_loss", 0),
                    f"pbt/agent/{agent_id}/value_loss": metrics.get("value_loss", 0),
                    "pbt/step": self._current_step,
                }
            )
        except Exception as e:
            logger.debug(f"Failed to log training metrics: {e}")

    def _log_evaluation_results(
        self,
        results: list[EvaluationResult],
        stats: PopulationStats,
    ) -> None:
        """Log evaluation results to W&B."""
        if self.wandb_logger is None:
            return

        try:
            import wandb

            # Population stats
            wandb.log(
                {
                    "pbt/population/mean_fitness": stats.mean_fitness,
                    "pbt/population/max_fitness": stats.max_fitness,
                    "pbt/population/min_fitness": stats.min_fitness,
                    "pbt/population/std_fitness": stats.std_fitness,
                    "pbt/population/generation": stats.generation,
                    "pbt/step": self._current_step,
                }
            )

            # Hyperparameter distributions
            for param_name, param_stats in stats.hyperparameter_stats.items():
                wandb.log(
                    {
                        f"pbt/hyperparams/{param_name}/mean": param_stats["mean"],
                        f"pbt/hyperparams/{param_name}/std": param_stats["std"],
                        f"pbt/hyperparams/{param_name}/min": param_stats["min"],
                        f"pbt/hyperparams/{param_name}/max": param_stats["max"],
                    }
                )

        except Exception as e:
            logger.debug(f"Failed to log evaluation results: {e}")

    def _log_exploit_explore(
        self,
        changes: list[tuple[str, str, dict[str, Any]]],
    ) -> None:
        """Log exploit/explore changes to W&B."""
        if self.wandb_logger is None:
            return

        try:
            import wandb

            wandb.log(
                {
                    "pbt/exploit_explore/agents_replaced": len(changes),
                    "pbt/exploit_explore/generation": self.population.generation,
                    "pbt/step": self._current_step,
                }
            )
        except Exception as e:
            logger.debug(f"Failed to log exploit/explore: {e}")

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def handle_signal(signum: int, frame: object | None) -> None:
            logger.info(f"Received signal {signum}, requesting shutdown")
            self._shutdown_requested = True

        try:
            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)
        except Exception:
            # Signal handling may not be available on all platforms
            logger.debug("Signal handling setup failed", exc_info=True)

    def get_progress(self) -> PBTProgress:
        """Get current training progress.

        Returns:
            PBTProgress object with current state.
        """
        stats = self.population.compute_stats() if self.population.population_size else None
        elapsed = time.time() - (self._start_time or time.time())

        return PBTProgress(
            current_step=self._current_step,
            generation=self.population.generation,
            best_fitness=stats.max_fitness if stats else 0.0,
            mean_fitness=stats.mean_fitness if stats else 0.0,
            population_size=self.population.population_size,
            evaluations_completed=self.scheduler.get_state().evaluation_count,
            exploit_explore_cycles=self.scheduler.get_state().exploit_explore_count,
            elapsed_time=elapsed,
        )

    def stop(self) -> None:
        """Request training to stop."""
        self._shutdown_requested = True
        logger.info("Stop requested")

    @property
    def is_running(self) -> bool:
        """Check if training is running."""
        return self._running

    @property
    def current_step(self) -> int:
        """Get current training step."""
        return self._current_step


def create_pbt_coordinator(
    population_size: int = 16,
    evaluation_interval: int = 1000,
    evaluation_games: int = 20,
    mutation_strength: float = 0.2,
    **kwargs: Any,
) -> PBTCoordinator:
    """Factory function to create a PBT coordinator.

    Args:
        population_size: Number of agents in population
        evaluation_interval: Steps between evaluations
        evaluation_games: Games per evaluation
        mutation_strength: Mutation strength
        **kwargs: Additional configuration parameters

    Returns:
        Configured PBTCoordinator.
    """
    config = PBTConfig(
        population_size=population_size,
        evaluation_interval=evaluation_interval,
        evaluation_games=evaluation_games,
        mutation_strength=mutation_strength,
        **kwargs,
    )
    return PBTCoordinator(config)
