"""Cyclic learning rate schedules for training.

Implements various cyclic learning rate strategies that have shown
improvements over monotonic decay in AlphaZero-style training.

Available schedule types:
- triangular: Linear ramp up and down
- triangular2: Triangular with halving amplitude each cycle
- exp_range: Exponential decay of max LR within cycle
- cosine_annealing: Cosine decay within each cycle
- warm_restarts: SGDR-style warm restarts with increasing cycle length

Usage:
    from alpha_zero.optimizations.cyclic_lr import CyclicLRScheduler, CyclicLRConfig

    config = CyclicLRConfig(
        schedule_type="cosine_annealing",
        base_lr=1e-5,
        max_lr=1e-3,
        step_size_up=5000,
    )
    scheduler = CyclicLRScheduler(config)

    for step in range(num_steps):
        lr = scheduler.get_lr(step)
        optimizer.param_groups[0]['lr'] = lr
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CyclicScheduleType(str, Enum):
    """Types of cyclic learning rate schedules."""

    TRIANGULAR = "triangular"
    TRIANGULAR2 = "triangular2"
    EXP_RANGE = "exp_range"
    COSINE_ANNEALING = "cosine_annealing"
    WARM_RESTARTS = "warm_restarts"


@dataclass
class CyclicLRConfig:
    """Configuration for cyclic learning rate.

    Attributes:
        schedule_type: Type of cyclic schedule
        base_lr: Minimum learning rate (valley of cycle)
        max_lr: Maximum learning rate (peak of cycle)
        step_size_up: Steps to ramp up to max_lr
        step_size_down: Steps to ramp down (defaults to step_size_up)
        gamma: Decay factor per cycle (for triangular2/exp_range)
        warmup_steps: Initial warmup steps before cycling begins
        t_mult: Multiplier for cycle length after restart (warm_restarts)
        cycle_momentum: Whether to cycle momentum inversely to LR
        base_momentum: Base momentum value (if cycling)
        max_momentum: Max momentum value (if cycling)
    """

    schedule_type: str = "cosine_annealing"
    base_lr: float = 1e-5
    max_lr: float = 1e-3
    step_size_up: int = 5000
    step_size_down: int | None = None
    gamma: float = 0.99
    warmup_steps: int = 1000
    t_mult: float = 2.0
    cycle_momentum: bool = False
    base_momentum: float = 0.85
    max_momentum: float = 0.95

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.base_lr <= 0:
            raise ValueError(f"base_lr must be > 0, got {self.base_lr}")
        if self.max_lr <= 0:
            raise ValueError(f"max_lr must be > 0, got {self.max_lr}")
        if self.max_lr < self.base_lr:
            raise ValueError(f"max_lr ({self.max_lr}) must be >= base_lr ({self.base_lr})")
        if self.step_size_up < 1:
            raise ValueError(f"step_size_up must be >= 1, got {self.step_size_up}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.t_mult < 1.0:
            raise ValueError(f"t_mult must be >= 1.0, got {self.t_mult}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CyclicLRConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schedule_type": self.schedule_type,
            "base_lr": self.base_lr,
            "max_lr": self.max_lr,
            "step_size_up": self.step_size_up,
            "step_size_down": self.step_size_down,
            "gamma": self.gamma,
            "warmup_steps": self.warmup_steps,
            "t_mult": self.t_mult,
            "cycle_momentum": self.cycle_momentum,
            "base_momentum": self.base_momentum,
            "max_momentum": self.max_momentum,
        }


@dataclass
class CycleInfo:
    """Information about the current learning rate cycle."""

    cycle_number: int
    position_in_cycle: int
    cycle_length: int
    current_lr: float
    current_momentum: float | None
    in_warmup: bool
    phase: str  # "warmup", "ascending", "descending"


class CyclicLRScheduler:
    """Cyclic learning rate scheduler.

    Implements multiple cyclic learning rate strategies with optional
    warmup and momentum cycling.

    Example:
        >>> config = CyclicLRConfig(
        ...     schedule_type="cosine_annealing",
        ...     base_lr=1e-5,
        ...     max_lr=1e-3,
        ...     step_size_up=5000,
        ... )
        >>> scheduler = CyclicLRScheduler(config)
        >>>
        >>> for step in range(100000):
        ...     lr = scheduler.get_lr(step)
        ...     optimizer.param_groups[0]['lr'] = lr
    """

    def __init__(self, config: CyclicLRConfig) -> None:
        """Initialize cyclic LR scheduler.

        Args:
            config: Cyclic LR configuration
        """
        self.config = config
        self._step_size_down = config.step_size_down or config.step_size_up
        self._cycle_length = config.step_size_up + self._step_size_down

    def get_lr(self, step: int) -> float:
        """Get learning rate at given step.

        Args:
            step: Current training step

        Returns:
            Learning rate value
        """
        # Handle warmup
        if step < self.config.warmup_steps:
            return self._linear_warmup(step)

        # Adjust step for warmup
        adjusted_step = step - self.config.warmup_steps

        schedule_type = CyclicScheduleType(self.config.schedule_type)

        if schedule_type == CyclicScheduleType.TRIANGULAR:
            return self._triangular(adjusted_step)
        elif schedule_type == CyclicScheduleType.TRIANGULAR2:
            return self._triangular2(adjusted_step)
        elif schedule_type == CyclicScheduleType.EXP_RANGE:
            return self._exp_range(adjusted_step)
        elif schedule_type == CyclicScheduleType.COSINE_ANNEALING:
            return self._cosine_annealing(adjusted_step)
        elif schedule_type == CyclicScheduleType.WARM_RESTARTS:
            return self._warm_restarts(adjusted_step)

        return self.config.base_lr

    def get_momentum(self, step: int) -> float | None:
        """Get momentum at given step (if momentum cycling enabled).

        Args:
            step: Current training step

        Returns:
            Momentum value or None if not cycling momentum
        """
        if not self.config.cycle_momentum:
            return None

        # Momentum cycles inversely to LR
        lr = self.get_lr(step)
        lr_range = self.config.max_lr - self.config.base_lr
        if lr_range == 0:
            return self.config.base_momentum

        lr_position = (lr - self.config.base_lr) / lr_range
        momentum_range = self.config.max_momentum - self.config.base_momentum

        # Higher LR -> lower momentum
        return self.config.max_momentum - lr_position * momentum_range

    def _linear_warmup(self, step: int) -> float:
        """Linear warmup from 0 to max_lr."""
        if self.config.warmup_steps == 0:
            return self.config.max_lr
        return self.config.max_lr * (step / self.config.warmup_steps)

    def _triangular(self, step: int) -> float:
        """Triangular cycle (linear ramp up and down)."""
        cycle = step // self._cycle_length
        x = step - cycle * self._cycle_length

        if x < self.config.step_size_up:
            # Ascending phase
            scale = x / self.config.step_size_up
        else:
            # Descending phase
            scale = 1.0 - (x - self.config.step_size_up) / self._step_size_down

        return self.config.base_lr + (self.config.max_lr - self.config.base_lr) * scale

    def _triangular2(self, step: int) -> float:
        """Triangular with halving amplitude each cycle."""
        cycle = step // self._cycle_length
        x = step - cycle * self._cycle_length

        # Halve amplitude each cycle
        amplitude = (self.config.max_lr - self.config.base_lr) / (2**cycle)

        if x < self.config.step_size_up:
            scale = x / self.config.step_size_up
        else:
            scale = 1.0 - (x - self.config.step_size_up) / self._step_size_down

        return float(self.config.base_lr + amplitude * scale)

    def _exp_range(self, step: int) -> float:
        """Exponential decay of max_lr within each cycle."""
        cycle = step // self._cycle_length
        x = step - cycle * self._cycle_length

        # Decay max_lr exponentially with each cycle
        max_lr = self.config.max_lr * (self.config.gamma**cycle)

        if x < self.config.step_size_up:
            scale = x / self.config.step_size_up
        else:
            scale = 1.0 - (x - self.config.step_size_up) / self._step_size_down

        return self.config.base_lr + (max_lr - self.config.base_lr) * scale

    def _cosine_annealing(self, step: int) -> float:
        """Cosine annealing within each cycle."""
        x = step % self._cycle_length

        # Cosine from max to base over cycle
        cosine_value = math.cos(math.pi * x / self._cycle_length)
        return self.config.base_lr + 0.5 * (self.config.max_lr - self.config.base_lr) * (
            1 + cosine_value
        )

    def _warm_restarts(self, step: int) -> float:
        """SGDR-style warm restarts with increasing cycle length."""
        # Find which cycle we're in and position within cycle
        t_cur = step
        t_i = self.config.step_size_up

        cycle = 0
        while t_cur >= t_i:
            t_cur -= t_i
            t_i = int(t_i * self.config.t_mult)
            cycle += 1

        # Cosine decay within current cycle
        cosine_value = math.cos(math.pi * t_cur / t_i)
        return self.config.base_lr + 0.5 * (self.config.max_lr - self.config.base_lr) * (
            1 + cosine_value
        )

    def get_cycle_info(self, step: int) -> CycleInfo:
        """Get detailed information about current cycle.

        Args:
            step: Current training step

        Returns:
            CycleInfo object with cycle details
        """
        in_warmup = step < self.config.warmup_steps
        adjusted_step = max(0, step - self.config.warmup_steps)

        # Determine cycle info based on schedule type
        schedule_type = CyclicScheduleType(self.config.schedule_type)

        if schedule_type == CyclicScheduleType.WARM_RESTARTS:
            # Find cycle for warm restarts
            t_cur = adjusted_step
            t_i = self.config.step_size_up
            cycle = 0
            while t_cur >= t_i:
                t_cur -= t_i
                t_i = int(t_i * self.config.t_mult)
                cycle += 1
            cycle_length = t_i
            position = t_cur
        else:
            cycle = adjusted_step // self._cycle_length
            position = adjusted_step % self._cycle_length
            cycle_length = self._cycle_length

        # Determine phase
        if in_warmup:
            phase = "warmup"
        elif position < self.config.step_size_up:
            phase = "ascending"
        else:
            phase = "descending"

        return CycleInfo(
            cycle_number=cycle,
            position_in_cycle=position,
            cycle_length=cycle_length,
            current_lr=self.get_lr(step),
            current_momentum=self.get_momentum(step),
            in_warmup=in_warmup,
            phase=phase,
        )

    def preview_schedule(
        self,
        num_steps: int,
        sample_interval: int = 100,
    ) -> list[tuple[int, float]]:
        """Preview learning rate schedule.

        Args:
            num_steps: Total steps to preview
            sample_interval: Sample every N steps

        Returns:
            List of (step, lr) tuples
        """
        samples = []
        for step in range(0, num_steps, sample_interval):
            lr = self.get_lr(step)
            samples.append((step, lr))
        return samples

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CyclicLRScheduler(type={self.config.schedule_type}, "
            f"base_lr={self.config.base_lr}, max_lr={self.config.max_lr})"
        )
