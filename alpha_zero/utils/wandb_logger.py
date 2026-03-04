"""Weights & Biases Integration for Civ6 AlphaZero.

This module provides comprehensive W&B integration for experiment tracking,
model logging, and visualization of training metrics.

Usage:
    from alpha_zero.utils.wandb_logger import WandbLogger, WandbConfig

    # Initialize from environment
    logger = WandbLogger.from_env()

    # Log training metrics
    logger.log_training_step(
        step=100,
        loss=0.5,
        policy_loss=0.3,
        value_loss=0.2,
        learning_rate=0.001,
    )

    # Log game results
    logger.log_game(
        game_id="game_001",
        winner=0,
        total_turns=150,
        reward=1.0,
    )

    # Finish run
    logger.finish()
"""

from __future__ import annotations

import atexit
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from alpha_zero.evaluation.evaluation_pipeline import EvaluationResult

# Optional wandb import — single import avoids isort ambiguity across ruff versions
try:
    import wandb

    Run = wandb.sdk.wandb_run.Run
    WANDB_AVAILABLE = True
except (ImportError, AttributeError):
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]
    Run = None  # type: ignore[assignment,misc]

# Load environment variables
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases integration."""

    # Enable/disable
    enabled: bool = False

    # Project settings
    project: str = "civ6-alphazero"
    entity: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    group: str | None = None

    # Run settings
    mode: str = "online"  # "online", "offline", "disabled"
    resume: str | None = None  # "allow", "must", "never", run_id
    reinit: bool = False

    # Logging settings
    log_frequency: int = 10  # Log every N steps
    log_model: bool = True  # Log model checkpoints
    log_code: bool = True  # Log source code
    log_gradients: bool = False  # Log gradient histograms
    log_parameters: bool = False  # Log parameter histograms

    # Sweep settings (for hyperparameter tuning)
    sweep_id: str | None = None

    @classmethod
    def from_env(cls) -> WandbConfig:
        """Create configuration from environment variables."""
        enabled = os.getenv("WANDB_ENABLED", "false").lower() == "true"

        # If API key is set, consider it enabled unless explicitly disabled
        if os.getenv("WANDB_API_KEY") and os.getenv("WANDB_ENABLED") is None:
            enabled = True

        return cls(
            enabled=enabled,
            project=os.getenv("WANDB_PROJECT", "civ6-alphazero"),
            entity=os.getenv("WANDB_ENTITY") or None,
            mode=os.getenv("WANDB_MODE", "online"),
            name=os.getenv("WANDB_RUN_NAME"),
            tags=os.getenv("WANDB_TAGS", "").split(",") if os.getenv("WANDB_TAGS") else [],
            notes=os.getenv("WANDB_NOTES", ""),
            group=os.getenv("WANDB_GROUP"),
            resume=os.getenv("WANDB_RESUME"),
            log_frequency=int(os.getenv("WANDB_LOG_FREQUENCY", "10")),
            log_model=os.getenv("WANDB_LOG_MODEL", "true").lower() == "true",
            log_code=os.getenv("WANDB_LOG_CODE", "true").lower() == "true",
        )


class WandbLogger:
    """Weights & Biases logger for experiment tracking.

    This class provides a comprehensive interface for logging training metrics,
    game results, model checkpoints, and visualizations to W&B.

    Attributes:
        config: WandbConfig instance with settings.
        run: Active W&B run instance (if initialized).
        step: Current global step counter.
    """

    def __init__(self, config: WandbConfig | None = None) -> None:
        """Initialize W&B logger.

        Args:
            config: Configuration for W&B. If None, loads from environment.
        """
        self.config = config or WandbConfig.from_env()
        self._run: Run | None = None
        self._step: int = 0
        self._initialized: bool = False

        if not WANDB_AVAILABLE and self.config.enabled:
            logger.warning("W&B package not installed. Install with: pip install wandb")
            self.config.enabled = False

    @classmethod
    def from_env(cls) -> WandbLogger:
        """Create logger from environment variables."""
        return cls(WandbConfig.from_env())

    @property
    def run(self) -> Run | None:
        """Get the active W&B run."""
        return self._run

    @property
    def step(self) -> int:
        """Get current step."""
        return self._step

    @property
    def is_enabled(self) -> bool:
        """Check if logging is enabled."""
        return self.config.enabled and WANDB_AVAILABLE

    @property
    def is_initialized(self) -> bool:
        """Check if run is initialized."""
        return self._initialized and self._run is not None

    def init(
        self,
        config_dict: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Run | None:
        """Initialize a W&B run.

        Args:
            config_dict: Dictionary of hyperparameters to log.
            **kwargs: Additional arguments passed to wandb.init().

        Returns:
            The W&B run instance, or None if disabled.
        """
        if not self.is_enabled:
            logger.info("W&B logging disabled")
            return None

        if self._initialized:
            logger.warning("W&B already initialized")
            return self._run

        try:
            # Set API key if provided in environment
            api_key = os.getenv("WANDB_API_KEY")
            if api_key:
                wandb.login(key=api_key, relogin=True)

            # Prepare init arguments
            init_kwargs = {
                "project": self.config.project,
                "entity": self.config.entity,
                "name": self.config.name,
                "tags": self.config.tags if self.config.tags else None,
                "notes": self.config.notes if self.config.notes else None,
                "group": self.config.group,
                "mode": self.config.mode,
                "reinit": self.config.reinit,
                "config": config_dict,
            }

            if self.config.resume:
                init_kwargs["resume"] = self.config.resume

            # Override with any provided kwargs
            init_kwargs.update(kwargs)

            # Initialize run
            self._run = wandb.init(**init_kwargs)  # type: ignore[arg-type]
            self._initialized = True

            # Log code if enabled
            if self.config.log_code:
                try:
                    wandb.run.log_code(".")  # type: ignore[union-attr]
                except Exception as e:
                    logger.warning(f"Failed to log code: {e}")

            logger.info(f"W&B run initialized: {self._run.name} ({self._run.id})")

            # Register cleanup
            atexit.register(self.finish)

            return self._run

        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.config.enabled = False
            return None

    def log(
        self,
        data: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
    ) -> None:
        """Log metrics to W&B.

        Args:
            data: Dictionary of metrics to log.
            step: Step number. If None, uses internal counter.
            commit: Whether to commit the log immediately.
        """
        if not self.is_initialized:
            return

        if step is not None:
            self._step = step
        else:
            self._step += 1

        # Convert numpy types to Python types
        cleaned_data = self._clean_data(data)

        try:
            wandb.log(cleaned_data, step=self._step, commit=commit)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def log_training_step(
        self,
        step: int,
        loss: float,
        policy_loss: float,
        value_loss: float,
        learning_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log training step metrics.

        Args:
            step: Training step number.
            loss: Total loss.
            policy_loss: Policy head loss.
            value_loss: Value head loss.
            learning_rate: Current learning rate.
            **kwargs: Additional metrics to log.
        """
        data = {
            "train/loss": loss,
            "train/policy_loss": policy_loss,
            "train/value_loss": value_loss,
        }

        if learning_rate is not None:
            data["train/learning_rate"] = learning_rate

        # Add any additional metrics
        for key, value in kwargs.items():
            if not key.startswith("train/"):
                key = f"train/{key}"
            data[key] = value

        self.log(data, step=step)

    def log_game(
        self,
        game_id: str,
        winner: int,
        total_turns: int,
        reward: float,
        game_length_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log game results.

        Args:
            game_id: Unique game identifier.
            winner: Winner player index (-1 for draw).
            total_turns: Total number of turns played.
            reward: Final reward.
            game_length_seconds: Game duration in seconds.
            **kwargs: Additional game metrics.
        """
        data = {
            "game/winner": winner,
            "game/total_turns": total_turns,
            "game/reward": reward,
        }

        if game_length_seconds is not None:
            data["game/duration_seconds"] = game_length_seconds

        for key, value in kwargs.items():
            if not key.startswith("game/"):
                key = f"game/{key}"
            data[key] = value

        self.log(data)

    def log_evaluation(
        self,
        win_rate: float,
        games_played: int,
        average_reward: float,
        average_turns: float,
        **kwargs: Any,
    ) -> None:
        """Log evaluation metrics.

        Args:
            win_rate: Win rate against opponent.
            games_played: Number of evaluation games.
            average_reward: Average reward per game.
            average_turns: Average turns per game.
            **kwargs: Additional evaluation metrics.
        """
        data = {
            "eval/win_rate": win_rate,
            "eval/games_played": games_played,
            "eval/average_reward": average_reward,
            "eval/average_turns": average_turns,
        }

        for key, value in kwargs.items():
            if not key.startswith("eval/"):
                key = f"eval/{key}"
            data[key] = value

        self.log(data)

    def log_baseline_evaluation(
        self,
        results: dict[str, EvaluationResult],
        iteration: int | None = None,
    ) -> None:
        """Log baseline evaluation results from EvaluationPipeline.

        Logs detailed win rate metrics for each opponent type (random, heuristic,
        checkpoint) with metrics namespaced as eval/{opponent}/{metric}.

        Args:
            results: Dictionary mapping opponent names to EvaluationResult.
                     Keys should be "random", "heuristic", or "checkpoint".
            iteration: Training iteration. If None, uses internal step counter.

        Example:
            >>> results = {
            ...     "random": pipeline.evaluate_vs_random(agent),
            ...     "heuristic": pipeline.evaluate_vs_heuristic(agent),
            ... }
            >>> logger.log_baseline_evaluation(results, iteration=100)
        """
        if not self.is_initialized:
            return

        data: dict[str, Any] = {}

        for opponent_name, result in results.items():
            prefix = f"eval/{opponent_name}"

            # Win rate metrics
            data[f"{prefix}/win_rate"] = result.win_rate.win_rate
            data[f"{prefix}/loss_rate"] = result.win_rate.loss_rate
            data[f"{prefix}/draw_rate"] = result.win_rate.draw_rate
            data[f"{prefix}/wins"] = result.win_rate.wins
            data[f"{prefix}/losses"] = result.win_rate.losses
            data[f"{prefix}/draws"] = result.win_rate.draws
            data[f"{prefix}/games_played"] = result.win_rate.total_games

            # Game statistics
            data[f"{prefix}/avg_game_length"] = result.avg_game_length
            data[f"{prefix}/avg_cities_won"] = result.avg_cities_won
            data[f"{prefix}/avg_cities_lost"] = result.avg_cities_lost
            data[f"{prefix}/avg_units_won"] = result.avg_units_won
            data[f"{prefix}/avg_units_lost"] = result.avg_units_lost

            # Value head diagnostics
            data[f"{prefix}/value_correlation"] = result.value_correlation
            data[f"{prefix}/evaluation_time"] = result.total_evaluation_time

        self.log(data, step=iteration)

    def log_reflection_stats(
        self,
        total_entries: int,
        unique_patterns: int | None = None,
        recent_additions: int | None = None,
        avg_critique_length: float | None = None,
        iteration: int | None = None,
    ) -> None:
        """Log reflection memory statistics.

        Tracks the size and characteristics of the LATS-style reflection memory
        system that learns from failed games.

        Args:
            total_entries: Total number of stored reflections.
            unique_patterns: Number of unique failure patterns identified.
            recent_additions: Reflections added since last log.
            avg_critique_length: Average length of stored critiques.
            iteration: Training iteration. If None, uses internal step counter.
        """
        if not self.is_initialized:
            return

        data: dict[str, Any] = {
            "reflection/total_entries": total_entries,
        }

        if unique_patterns is not None:
            data["reflection/unique_patterns"] = unique_patterns

        if recent_additions is not None:
            data["reflection/recent_additions"] = recent_additions

        if avg_critique_length is not None:
            data["reflection/avg_critique_length"] = avg_critique_length

        self.log(data, step=iteration)

    def log_imitation_training(
        self,
        epoch: int,
        loss: float,
        policy_loss: float,
        value_loss: float,
        policy_accuracy: float | None = None,
        value_mse: float | None = None,
        games_used: int | None = None,
        samples_per_epoch: int | None = None,
    ) -> None:
        """Log imitation pre-training progress.

        Tracks the supervised learning phase where the network learns from
        heuristic agent demonstrations before self-play begins.

        Args:
            epoch: Current training epoch.
            loss: Total combined loss.
            policy_loss: Policy head cross-entropy loss.
            value_loss: Value head MSE loss.
            policy_accuracy: Optional top-1 policy prediction accuracy.
            value_mse: Optional raw value mean squared error.
            games_used: Number of expert games in training set.
            samples_per_epoch: Training samples seen per epoch.
        """
        if not self.is_initialized:
            return

        data: dict[str, Any] = {
            "imitation/epoch": epoch,
            "imitation/loss": loss,
            "imitation/policy_loss": policy_loss,
            "imitation/value_loss": value_loss,
        }

        if policy_accuracy is not None:
            data["imitation/policy_accuracy"] = policy_accuracy

        if value_mse is not None:
            data["imitation/value_mse"] = value_mse

        if games_used is not None:
            data["imitation/games_used"] = games_used

        if samples_per_epoch is not None:
            data["imitation/samples_per_epoch"] = samples_per_epoch

        self.log(data)

    def define_baseline_charts(self) -> None:
        """Define custom WandB charts for baseline evaluation tracking.

        Sets up x-axis metrics and chart configurations for win rate tracking
        over training iterations. Should be called after init().

        Creates:
            - Win rate vs iteration charts for each opponent type
            - Value correlation tracking
            - Reflection memory growth charts
        """
        if not self.is_initialized:
            return

        try:
            # Define iteration as the step metric for evaluation charts
            wandb.define_metric("iteration")

            # Win rate charts for each opponent type
            for opponent in ["random", "heuristic", "checkpoint"]:
                wandb.define_metric(
                    f"eval/{opponent}/win_rate",
                    step_metric="iteration",
                    summary="max",
                )
                wandb.define_metric(
                    f"eval/{opponent}/value_correlation",
                    step_metric="iteration",
                    summary="max",
                )
                wandb.define_metric(
                    f"eval/{opponent}/avg_game_length",
                    step_metric="iteration",
                    summary="mean",
                )

            # Reflection memory metrics
            wandb.define_metric(
                "reflection/total_entries",
                step_metric="iteration",
                summary="last",
            )

            # Imitation training metrics (use epoch as x-axis)
            wandb.define_metric("imitation/epoch")
            wandb.define_metric(
                "imitation/loss",
                step_metric="imitation/epoch",
                summary="min",
            )
            wandb.define_metric(
                "imitation/policy_accuracy",
                step_metric="imitation/epoch",
                summary="max",
            )

            logger.info("Defined baseline evaluation WandB charts")

        except Exception as e:
            logger.warning(f"Failed to define baseline charts: {e}")

    def log_population_stats(
        self,
        generation: int,
        mean_fitness: float,
        max_fitness: float,
        min_fitness: float,
        mean_elo: float,
        max_elo: float,
        min_elo: float,
        population_size: int,
        mutations_count: int = 0,
        **kwargs: Any,
    ) -> None:
        """Log PBT population-level metrics.

        All metrics are namespaced under ``pbt/`` for W&B dashboard grouping.

        Args:
            generation: Current PBT generation.
            mean_fitness: Mean fitness across the population.
            max_fitness: Max fitness.
            min_fitness: Min fitness.
            mean_elo: Mean ELO rating.
            max_elo: Max ELO rating.
            min_elo: Min ELO rating.
            population_size: Number of agents.
            mutations_count: Number of mutations this generation.
            **kwargs: Additional metrics (auto-prefixed with ``pbt/``).
        """
        if not self.is_initialized:
            return

        data: dict[str, Any] = {
            "pbt/generation": generation,
            "pbt/fitness_mean": mean_fitness,
            "pbt/fitness_max": max_fitness,
            "pbt/fitness_min": min_fitness,
            "pbt/elo_mean": mean_elo,
            "pbt/elo_max": max_elo,
            "pbt/elo_min": min_elo,
            "pbt/population_size": population_size,
            "pbt/mutations_count": mutations_count,
        }

        for key, value in kwargs.items():
            if not key.startswith("pbt/"):
                key = f"pbt/{key}"
            data[key] = value

        self.log(data, step=generation)

    def define_pbt_charts(self) -> None:
        """Define W&B metric axes for PBT population dashboards.

        Registers ``pbt/generation`` as the step metric and sets up
        summary types for ELO, fitness, and mutation metrics. Should be
        called after ``init()``.
        """
        if not self.is_initialized:
            return

        try:
            wandb.define_metric("pbt/generation")

            for metric, summary in [
                ("pbt/elo_max", "max"),
                ("pbt/elo_mean", "last"),
                ("pbt/elo_min", "min"),
                ("pbt/fitness_max", "max"),
                ("pbt/fitness_mean", "last"),
                ("pbt/fitness_min", "min"),
                ("pbt/population_size", "last"),
                ("pbt/mutations_count", "sum"),
            ]:
                wandb.define_metric(
                    metric,
                    step_metric="pbt/generation",
                    summary=summary,
                )

            logger.info("Defined PBT population WandB charts")

        except Exception as e:
            logger.warning(f"Failed to define PBT charts: {e}")

    def log_selfplay(
        self,
        games_completed: int,
        buffer_size: int,
        samples_generated: int,
        **kwargs: Any,
    ) -> None:
        """Log self-play metrics.

        Args:
            games_completed: Total games completed.
            buffer_size: Current replay buffer size.
            samples_generated: Total training samples generated.
            **kwargs: Additional self-play metrics.
        """
        data = {
            "selfplay/games_completed": games_completed,
            "selfplay/buffer_size": buffer_size,
            "selfplay/samples_generated": samples_generated,
        }

        for key, value in kwargs.items():
            if not key.startswith("selfplay/"):
                key = f"selfplay/{key}"
            data[key] = value

        self.log(data)

    def log_mcts(
        self,
        average_depth: float,
        average_visits: float,
        tree_reuse_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Log MCTS statistics.

        Args:
            average_depth: Average search depth.
            average_visits: Average node visits.
            tree_reuse_rate: Rate of tree reuse between moves.
            **kwargs: Additional MCTS metrics.
        """
        data = {
            "mcts/average_depth": average_depth,
            "mcts/average_visits": average_visits,
        }

        if tree_reuse_rate is not None:
            data["mcts/tree_reuse_rate"] = tree_reuse_rate

        for key, value in kwargs.items():
            if not key.startswith("mcts/"):
                key = f"mcts/{key}"
            data[key] = value

        self.log(data)

    def log_model(
        self,
        model_path: str | Path,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Log model checkpoint as artifact.

        Args:
            model_path: Path to model checkpoint.
            name: Artifact name. Defaults to "model".
            metadata: Additional metadata for the artifact.
        """
        if not self.is_initialized or not self.config.log_model:
            return

        try:
            artifact_name = name or "model"
            artifact = wandb.Artifact(
                artifact_name,
                type="model",
                metadata=metadata or {},
            )
            artifact.add_file(str(model_path))
            wandb.log_artifact(artifact)
            logger.info(f"Logged model artifact: {artifact_name}")
        except Exception as e:
            logger.warning(f"Failed to log model artifact: {e}")

    def log_table(
        self,
        name: str,
        columns: list[str],
        data: list[list[Any]],
    ) -> None:
        """Log a table to W&B.

        Args:
            name: Table name.
            columns: Column names.
            data: Table data as list of rows.
        """
        if not self.is_initialized:
            return

        try:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({name: table})
        except Exception as e:
            logger.warning(f"Failed to log table: {e}")

    def log_histogram(
        self,
        name: str,
        values: np.ndarray | list,
        num_bins: int = 64,
    ) -> None:
        """Log a histogram to W&B.

        Args:
            name: Histogram name.
            values: Values to histogram.
            num_bins: Number of bins.
        """
        if not self.is_initialized:
            return

        try:
            wandb.log({name: wandb.Histogram(values, num_bins=num_bins)})  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(f"Failed to log histogram: {e}")

    def log_image(
        self,
        name: str,
        image: np.ndarray,
        caption: str | None = None,
    ) -> None:
        """Log an image to W&B.

        Args:
            name: Image name.
            image: Image array (HWC or HW format).
            caption: Optional caption.
        """
        if not self.is_initialized:
            return

        try:
            wandb.log({name: wandb.Image(image, caption=caption)})
        except Exception as e:
            logger.warning(f"Failed to log image: {e}")

    def watch_model(
        self,
        model: Any,
        log: str = "gradients",
        log_freq: int = 100,
    ) -> None:
        """Watch a PyTorch model for gradient/parameter logging.

        Args:
            model: PyTorch model to watch.
            log: What to log ("gradients", "parameters", "all").
            log_freq: Logging frequency.
        """
        if not self.is_initialized:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq)  # type: ignore[arg-type]
            logger.info(f"Watching model for {log}")
        except Exception as e:
            logger.warning(f"Failed to watch model: {e}")

    def define_metric(
        self,
        name: str,
        step_metric: str = "train/step",
        summary: str = "last",
    ) -> None:
        """Define a custom metric with specific settings.

        Args:
            name: Metric name.
            step_metric: Step metric to use as x-axis.
            summary: Summary type ("min", "max", "mean", "last", "none").
        """
        if not self.is_initialized:
            return

        try:
            wandb.define_metric(name, step_metric=step_metric, summary=summary)
        except Exception as e:
            logger.warning(f"Failed to define metric: {e}")

    def alert(
        self,
        title: str,
        text: str,
        level: str = "INFO",
    ) -> None:
        """Send an alert to W&B.

        Args:
            title: Alert title.
            text: Alert text.
            level: Alert level ("INFO", "WARN", "ERROR").
        """
        if not self.is_initialized:
            return

        try:
            wandb.alert(title=title, text=text, level=level)  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"Failed to send alert: {e}")

    def summary(self, data: dict[str, Any]) -> None:
        """Update run summary.

        Args:
            data: Dictionary of summary values.
        """
        if not self.is_initialized:
            return

        try:
            for key, value in data.items():
                wandb.run.summary[key] = value  # type: ignore[union-attr]
        except Exception as e:
            logger.warning(f"Failed to update summary: {e}")

    def finish(self, exit_code: int | None = None) -> None:
        """Finish the W&B run.

        Args:
            exit_code: Optional exit code for the run.
        """
        if not self._initialized or self._run is None or wandb is None:
            return

        try:
            wandb.finish(exit_code=exit_code)
            logger.info("W&B run finished")
        except TypeError:
            # Fallback if wandb version doesn't support exit_code parameter
            wandb.finish()
            logger.info("W&B run finished")
        except Exception as e:
            logger.warning(f"Error finishing W&B run: {e}")
        finally:
            self._run = None
            self._initialized = False

    def _clean_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert numpy types to Python types for logging.

        Args:
            data: Data dictionary.

        Returns:
            Cleaned dictionary with Python native types.
        """
        cleaned: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    cleaned[key] = float(value.item())
                else:
                    cleaned[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                cleaned[key] = float(value)
            elif isinstance(value, dict):
                cleaned[key] = self._clean_data(value)
            else:
                cleaned[key] = value
        return cleaned


# Singleton instance for convenience
_global_logger: WandbLogger | None = None


def get_wandb_logger() -> WandbLogger:
    """Get or create the global W&B logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = WandbLogger.from_env()
    return _global_logger


def init_wandb(
    config_dict: dict[str, Any] | None = None,
    **kwargs: Any,
) -> WandbLogger:
    """Initialize W&B with the global logger.

    Args:
        config_dict: Configuration dictionary to log.
        **kwargs: Additional arguments for wandb.init().

    Returns:
        The initialized WandbLogger instance.
    """
    logger = get_wandb_logger()
    logger.init(config_dict=config_dict, **kwargs)
    return logger


def log_metrics(data: dict[str, Any], step: int | None = None) -> None:
    """Log metrics using the global logger.

    Args:
        data: Metrics to log.
        step: Optional step number.
    """
    logger = get_wandb_logger()
    if logger.is_initialized:
        logger.log(data, step=step)
