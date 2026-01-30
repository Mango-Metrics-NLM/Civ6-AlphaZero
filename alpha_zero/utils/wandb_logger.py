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
from typing import Any

import numpy as np

# Optional wandb import
try:
    from wandb.sdk.wandb_run import Run

    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    Run = None

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

    def __init__(self, config: WandbConfig | None = None):
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
            self._run = wandb.init(**init_kwargs)
            self._initialized = True

            # Log code if enabled
            if self.config.log_code:
                try:
                    wandb.run.log_code(".")
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
            wandb.log({name: wandb.Histogram(values, num_bins=num_bins)})
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
            wandb.watch(model, log=log, log_freq=log_freq)
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
            wandb.alert(title=title, text=text, level=level)
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
                wandb.run.summary[key] = value
        except Exception as e:
            logger.warning(f"Failed to update summary: {e}")

    def finish(self, exit_code: int | None = None) -> None:
        """Finish the W&B run.

        Args:
            exit_code: Optional exit code for the run.
        """
        if not self._initialized or self._run is None:
            return

        try:
            wandb.finish(exit_code=exit_code)
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
