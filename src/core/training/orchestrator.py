"""Training orchestrator that wires all components into an end-to-end loop.

The orchestrator manages the full AlphaZero-style training cycle:
self-play data generation → network training → evaluation → checkpoint.
It supports curriculum learning, configurable components, and
comprehensive logging.

Layer 4 -- depends on all lower layers.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.core.config import SDKConfig
from src.core.protocols import StrategicEnvironment
from src.core.training.checkpoint import CheckpointManager, CheckpointMetadata
from src.core.training.curriculum import CurriculumScheduler
from src.core.training.evaluator import EvalResult, Evaluator
from src.core.training.learner import Learner
from src.core.training.replay_buffer import ReplayBuffer
from src.core.training.self_play import SelfPlayCollector

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IterationMetrics:
    """Metrics collected during a single training iteration."""

    iteration: int
    self_play_games: int = 0
    self_play_examples: int = 0
    self_play_duration_s: float = 0.0
    training_steps: int = 0
    training_loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    training_duration_s: float = 0.0
    eval_elo: float = 0.0
    eval_win_rate: float = 0.0
    eval_duration_s: float = 0.0
    is_best: bool = False
    curriculum_stage: str = ""
    total_duration_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float | int | str | bool]:
        """Convert to a flat dictionary for logging."""
        return {
            "iteration": self.iteration,
            "self_play_games": self.self_play_games,
            "self_play_examples": self.self_play_examples,
            "self_play_duration_s": self.self_play_duration_s,
            "training_steps": self.training_steps,
            "training_loss": self.training_loss,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "training_duration_s": self.training_duration_s,
            "eval_elo": self.eval_elo,
            "eval_win_rate": self.eval_win_rate,
            "eval_duration_s": self.eval_duration_s,
            "is_best": self.is_best,
            "curriculum_stage": self.curriculum_stage,
            "total_duration_s": self.total_duration_s,
        }


class TrainingOrchestrator:
    """Orchestrates the full AlphaZero-style training loop.

    Manages the cycle of self-play → training → evaluation → checkpoint
    with support for curriculum learning and configurable components.

    Args:
        config: SDK configuration.
        env_factory: Callable returning a new environment instance.
        learner: The learner managing network updates.
            If None, creates a dry-run learner.
        replay_buffer: The replay buffer for storing training data.
            If None, creates one from config.
        evaluator: The evaluator for computing Elo ratings.
            If None, creates one from config.
        checkpoint_manager: The checkpoint manager.
            If None, creates one from config.
        curriculum: Optional curriculum scheduler.
            If None and config has stages, creates one.
    """

    def __init__(
        self,
        config: SDKConfig,
        env_factory: Callable[[], StrategicEnvironment],
        learner: Learner | None = None,
        replay_buffer: ReplayBuffer | None = None,
        evaluator: Evaluator | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        curriculum: CurriculumScheduler | None = None,
    ) -> None:
        self._config = config
        self._env_factory = env_factory

        # Initialize components with sensible defaults
        self._learner = learner or Learner(config=config.training)
        self._buffer = replay_buffer or ReplayBuffer(
            capacity=config.training.replay_buffer_size,
            seed=config.seed,
        )
        self._evaluator = evaluator or Evaluator(
            mcts_config=config.mcts,
            num_eval_games=config.training.eval_games,
            seed=config.seed,
        )
        self._checkpoint_mgr = checkpoint_manager or CheckpointManager.from_config(config)

        # Curriculum is optional
        self._curriculum: CurriculumScheduler | None
        if curriculum is not None:
            self._curriculum = curriculum
        elif config.curriculum.stages:
            self._curriculum = CurriculumScheduler(config.curriculum)
        else:
            self._curriculum = None

        # Self-play collector
        self._collector = SelfPlayCollector(
            eval_fn=self._learner.get_eval_fn(),
            mcts_config=config.mcts,
            max_steps=config.training.max_steps_per_game,
        )

        # Tracking
        self._iteration = 0
        self._best_elo = -float("inf")
        self._history: list[IterationMetrics] = []
        self._running = False

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def best_elo(self) -> float:
        return self._best_elo

    @property
    def history(self) -> list[IterationMetrics]:
        return list(self._history)

    @property
    def is_running(self) -> bool:
        return self._running

    def run(
        self,
        max_iterations: int | None = None,
        start_iteration: int = 0,
        dry_run: bool = False,
    ) -> list[IterationMetrics]:
        """Execute the training loop.

        Args:
            max_iterations: Override config max_iterations.
                If None, uses config value.
            start_iteration: Iteration to start from (for resume).
            dry_run: If True, run one iteration for validation.

        Returns:
            List of metrics for each completed iteration.
        """
        max_iter = max_iterations or self._config.training.max_iterations
        if dry_run:
            max_iter = min(max_iter, 1)

        self._iteration = start_iteration
        self._running = True

        logger.info(
            "Starting training: iterations=%d, games_per_iter=%d, "
            "batch_size=%d, buffer_capacity=%d",
            max_iter,
            self._config.training.games_per_iteration,
            self._config.training.batch_size,
            self._config.training.replay_buffer_size,
        )

        try:
            for _ in range(max_iter):
                metrics = self._run_iteration()
                self._history.append(metrics)
                self._iteration += 1

                if dry_run:
                    logger.info("Dry run complete after 1 iteration")
                    break
        finally:
            self._running = False

        logger.info(
            "Training complete: %d iterations, best Elo=%.1f",
            self._iteration - start_iteration,
            self._best_elo,
        )
        return self._history

    def _run_iteration(self) -> IterationMetrics:
        """Execute a single training iteration."""
        iter_start = time.monotonic()
        metrics = IterationMetrics(iteration=self._iteration)

        # Set curriculum stage
        if self._curriculum is not None:
            metrics.curriculum_stage = self._curriculum.current_stage_name

        # Phase 1: Self-play
        sp_metrics = self._self_play_phase()
        metrics.self_play_games = sp_metrics["games"]
        metrics.self_play_examples = sp_metrics["examples"]
        metrics.self_play_duration_s = sp_metrics["duration_s"]

        # Phase 2: Training
        train_metrics = self._training_phase()
        metrics.training_steps = train_metrics["steps"]
        metrics.training_loss = train_metrics["total_loss"]
        metrics.policy_loss = train_metrics["policy_loss"]
        metrics.value_loss = train_metrics["value_loss"]
        metrics.training_duration_s = train_metrics["duration_s"]

        # Phase 3: Evaluation (at configured interval)
        if self._should_evaluate():
            eval_result = self._evaluation_phase()
            metrics.eval_elo = eval_result.elo
            metrics.eval_win_rate = eval_result.win_rate
            metrics.eval_duration_s = eval_result.details.get("duration_s", 0.0)

            # Check if this is the best model
            if eval_result.elo > self._best_elo:
                self._best_elo = eval_result.elo
                metrics.is_best = True

            # Update curriculum
            if self._curriculum is not None:
                new_stage = self._curriculum.record_iteration(
                    elo=eval_result.elo,
                    games_played=metrics.self_play_games,
                )
                if new_stage is not None:
                    # Update self-play collector for new environment
                    self._collector = SelfPlayCollector(
                        eval_fn=self._learner.get_eval_fn(),
                        mcts_config=self._config.mcts,
                        max_steps=self._config.training.max_steps_per_game,
                    )

        # Phase 4: Checkpoint
        if self._checkpoint_mgr.should_save(self._iteration):
            self._checkpoint_phase(metrics)

        metrics.total_duration_s = time.monotonic() - iter_start

        logger.info(
            "Iteration %d: loss=%.4f, elo=%.1f, games=%d, examples=%d (%.1fs)",
            self._iteration,
            metrics.training_loss,
            metrics.eval_elo,
            metrics.self_play_games,
            metrics.self_play_examples,
            metrics.total_duration_s,
        )
        return metrics

    def _self_play_phase(self) -> dict[str, Any]:
        """Run self-play games and add examples to the replay buffer."""
        start = time.monotonic()
        num_games = self._config.training.games_per_iteration

        # Update eval function with latest network weights
        self._collector._eval_fn = self._learner.get_eval_fn()

        trajectories = self._collector.play_games(
            env_factory=self._env_factory,
            num_games=num_games,
            seed_offset=self._iteration * num_games,
        )

        total_examples = 0
        for trajectory in trajectories:
            self._buffer.add_trajectory(trajectory)
            total_examples += len(trajectory.examples)

        duration = time.monotonic() - start
        logger.debug(
            "Self-play: %d games, %d examples in %.1fs",
            num_games,
            total_examples,
            duration,
        )
        return {
            "games": num_games,
            "examples": total_examples,
            "duration_s": duration,
        }

    def _training_phase(self) -> dict[str, Any]:
        """Train the network on replay buffer data."""
        start = time.monotonic()
        batch_size = self._config.training.batch_size

        # Skip training if not enough data
        if self._buffer.size < batch_size:
            logger.debug(
                "Skipping training: buffer size %d < batch size %d",
                self._buffer.size,
                batch_size,
            )
            return {
                "steps": 0,
                "total_loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "duration_s": 0.0,
            }

        # Run training steps (one step per iteration by default)
        batch = self._buffer.sample_batch(batch_size)
        step_metrics = self._learner.train_step(batch)

        duration = time.monotonic() - start
        return {
            "steps": 1,
            "total_loss": step_metrics.get("total_loss", 0.0),
            "policy_loss": step_metrics.get("policy_loss", 0.0),
            "value_loss": step_metrics.get("value_loss", 0.0),
            "duration_s": duration,
        }

    def _evaluation_phase(self) -> EvalResult:
        """Evaluate the current model against a baseline."""
        start = time.monotonic()

        result = self._evaluator.evaluate(
            candidate_fn=self._learner.get_eval_fn(),
            env_factory=self._env_factory,
        )
        result.details["duration_s"] = time.monotonic() - start
        return result

    def _checkpoint_phase(self, metrics: IterationMetrics) -> None:
        """Save a checkpoint of the current training state."""
        meta = CheckpointMetadata(
            iteration=self._iteration,
            metrics={"elo": metrics.eval_elo, "loss": metrics.training_loss},
            is_best=metrics.is_best,
        )

        self._checkpoint_mgr.save(
            iteration=self._iteration,
            save_fn=self._learner.save_state,
            metadata=meta,
            is_best=metrics.is_best,
        )

    def _should_evaluate(self) -> bool:
        """Check if evaluation should run at the current iteration."""
        interval = self._config.training.eval_interval
        return interval > 0 and self._iteration % interval == 0

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of the current training state."""
        summary: dict[str, Any] = {
            "iteration": self._iteration,
            "best_elo": self._best_elo,
            "buffer_size": self._buffer.size,
            "buffer_capacity": self._buffer.capacity,
            "total_examples_added": self._buffer.total_added,
            "learner_steps": self._learner.step_count,
        }
        if self._curriculum is not None:
            summary["curriculum"] = self._curriculum.get_summary()
        return summary
