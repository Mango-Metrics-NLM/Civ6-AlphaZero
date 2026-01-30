"""Hyperparameter scheduling for dynamic training configuration.

This module provides unified scheduling for all training hyperparameters:
- Learning rate
- Batch size
- MCTS temperature
- Exploration epsilon

Each parameter can follow its own schedule (constant, linear, cosine, step,
warmup_cosine, exponential) independently.

Usage:
    from alpha_zero.orchestrator.scheduler import HyperparameterScheduler
    from alpha_zero.orchestrator.config import ScheduleConfig

    scheduler = HyperparameterScheduler(
        lr_schedule=ScheduleConfig(schedule_type="warmup_cosine", ...),
        batch_size_schedule=ScheduleConfig(schedule_type="step", ...),
        temperature_schedule=ScheduleConfig(schedule_type="linear", ...),
        exploration_schedule=ScheduleConfig(schedule_type="exponential", ...),
    )

    # Get values at step 1000
    lr = scheduler.get_learning_rate(1000)
    batch_size = scheduler.get_batch_size(1000)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alpha_zero.orchestrator.config import ScheduleConfig


@dataclass
class SchedulerState:
    """Current state of the hyperparameter scheduler."""

    current_step: int = 0
    learning_rate: float = 0.001
    batch_size: int = 64
    temperature: float = 1.0
    exploration: float = 0.25


class HyperparameterScheduler:
    """Scheduler for all training hyperparameters.

    Manages multiple schedule configurations and provides a unified interface
    for retrieving hyperparameter values at any training step.

    Supports multiple schedule types:
    - constant: Fixed value throughout training
    - linear: Linear interpolation from initial to final
    - cosine: Cosine annealing
    - step: Step-wise changes at specific iterations
    - warmup_cosine: Linear warmup followed by cosine decay
    - exponential: Exponential decay

    Attributes:
        lr_schedule: Schedule configuration for learning rate.
        batch_size_schedule: Schedule configuration for batch size.
        temperature_schedule: Schedule configuration for MCTS temperature.
        exploration_schedule: Schedule configuration for exploration epsilon.
    """

    def __init__(
        self,
        lr_schedule: ScheduleConfig | None = None,
        batch_size_schedule: ScheduleConfig | None = None,
        temperature_schedule: ScheduleConfig | None = None,
        exploration_schedule: ScheduleConfig | None = None,
    ):
        """Initialize scheduler with schedule configurations.

        Args:
            lr_schedule: Schedule for learning rate. Defaults to constant 0.001.
            batch_size_schedule: Schedule for batch size. Defaults to constant 64.
            temperature_schedule: Schedule for MCTS temperature. Defaults to constant 1.0.
            exploration_schedule: Schedule for exploration. Defaults to constant 0.25.
        """
        self.lr_schedule = lr_schedule or ScheduleConfig(
            schedule_type="constant",
            initial_value=0.001,
        )
        self.batch_size_schedule = batch_size_schedule or ScheduleConfig(
            schedule_type="constant",
            initial_value=64,
        )
        self.temperature_schedule = temperature_schedule or ScheduleConfig(
            schedule_type="constant",
            initial_value=1.0,
        )
        self.exploration_schedule = exploration_schedule or ScheduleConfig(
            schedule_type="constant",
            initial_value=0.25,
        )

        self._current_step = 0
        self._state = SchedulerState()

    @property
    def current_step(self) -> int:
        """Get current training step."""
        return self._current_step

    def step(self, n: int = 1) -> None:
        """Advance the scheduler by n steps.

        Args:
            n: Number of steps to advance.
        """
        self._current_step += n
        self._update_state()

    def set_step(self, step: int) -> None:
        """Set the current step directly.

        Args:
            step: Step number to set.
        """
        self._current_step = step
        self._update_state()

    def _update_state(self) -> None:
        """Update cached state values."""
        self._state.current_step = self._current_step
        self._state.learning_rate = self.get_learning_rate()
        self._state.batch_size = self.get_batch_size()
        self._state.temperature = self.get_temperature()
        self._state.exploration = self.get_exploration()

    def get_learning_rate(self, step: int | None = None) -> float:
        """Get learning rate at given step.

        Args:
            step: Training step. Defaults to current step.

        Returns:
            Learning rate value at the specified step.
        """
        s = step if step is not None else self._current_step
        return self.lr_schedule.get_value(s)

    def get_batch_size(self, step: int | None = None) -> int:
        """Get batch size at given step.

        Args:
            step: Training step. Defaults to current step.

        Returns:
            Batch size value at the specified step.
        """
        s = step if step is not None else self._current_step
        return int(self.batch_size_schedule.get_value(s))

    def get_temperature(self, step: int | None = None) -> float:
        """Get MCTS temperature at given step.

        Args:
            step: Training step. Defaults to current step.

        Returns:
            Temperature value at the specified step.
        """
        s = step if step is not None else self._current_step
        return self.temperature_schedule.get_value(s)

    def get_exploration(self, step: int | None = None) -> float:
        """Get exploration epsilon at given step.

        Args:
            step: Training step. Defaults to current step.

        Returns:
            Exploration value at the specified step.
        """
        s = step if step is not None else self._current_step
        return self.exploration_schedule.get_value(s)

    def get_all_values(self, step: int | None = None) -> dict[str, Any]:
        """Get all hyperparameter values at given step.

        Args:
            step: Training step. Defaults to current step.

        Returns:
            Dictionary containing all hyperparameter values.
        """
        s = step if step is not None else self._current_step
        return {
            "learning_rate": self.get_learning_rate(s),
            "batch_size": self.get_batch_size(s),
            "temperature": self.get_temperature(s),
            "exploration": self.get_exploration(s),
            "step": s,
        }

    def get_state(self) -> SchedulerState:
        """Get current scheduler state.

        Returns:
            Current state with all hyperparameter values.
        """
        self._update_state()
        return self._state

    def get_schedule_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all schedule configurations.

        Returns:
            Dictionary with schedule configuration details.
        """
        return {
            "learning_rate": {
                "type": self.lr_schedule.schedule_type,
                "initial": self.lr_schedule.initial_value,
                "final": self.lr_schedule.final_value,
                "decay_steps": self.lr_schedule.decay_steps,
                "warmup_steps": self.lr_schedule.warmup_steps,
            },
            "batch_size": {
                "type": self.batch_size_schedule.schedule_type,
                "initial": self.batch_size_schedule.initial_value,
                "final": self.batch_size_schedule.final_value,
                "decay_steps": self.batch_size_schedule.decay_steps,
            },
            "temperature": {
                "type": self.temperature_schedule.schedule_type,
                "initial": self.temperature_schedule.initial_value,
                "final": self.temperature_schedule.final_value,
                "decay_steps": self.temperature_schedule.decay_steps,
            },
            "exploration": {
                "type": self.exploration_schedule.schedule_type,
                "initial": self.exploration_schedule.initial_value,
                "final": self.exploration_schedule.final_value,
                "decay_steps": self.exploration_schedule.decay_steps,
            },
        }

    def preview_schedule(
        self,
        schedule_name: str,
        steps: list[int] | None = None,
        num_points: int = 10,
    ) -> list[tuple[int, float]]:
        """Preview schedule values at specified steps.

        Useful for debugging and visualization.

        Args:
            schedule_name: One of "learning_rate", "batch_size", "temperature", "exploration".
            steps: Specific steps to preview. If None, generates evenly spaced points.
            num_points: Number of points to generate if steps is None.

        Returns:
            List of (step, value) tuples.

        Raises:
            ValueError: If schedule_name is invalid.
        """
        schedule_map = {
            "learning_rate": self.lr_schedule,
            "batch_size": self.batch_size_schedule,
            "temperature": self.temperature_schedule,
            "exploration": self.exploration_schedule,
        }

        if schedule_name not in schedule_map:
            raise ValueError(
                f"Unknown schedule: {schedule_name}. "
                f"Choose from: {list(schedule_map.keys())}",
            )

        schedule = schedule_map[schedule_name]

        if steps is None:
            max_step = max(schedule.decay_steps, 1)
            step_size = max_step // max(num_points - 1, 1)
            steps = [i * step_size for i in range(num_points)]

        return [(s, schedule.get_value(s)) for s in steps]

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self._current_step = 0
        self._update_state()

    def __repr__(self) -> str:
        """String representation of scheduler."""
        return (
            f"HyperparameterScheduler("
            f"step={self._current_step}, "
            f"lr={self.get_learning_rate():.6f}, "
            f"batch_size={self.get_batch_size()}, "
            f"temp={self.get_temperature():.4f}, "
            f"explore={self.get_exploration():.4f})"
        )
