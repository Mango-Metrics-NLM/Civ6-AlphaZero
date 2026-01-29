"""Enhanced training configuration with schedules and resource management.

This module provides extended training configuration capabilities including
parameter schedules, gradient accumulation, and resource management for
scalable training.

Features:
- Batch size and learning rate schedules
- Gradient accumulation for larger effective batch sizes
- Resource management for memory-efficient training
- Configuration validation and serialization

Example usage:
    >>> from alpha_zero.utils.training_config_enhanced import (
    ...     EnhancedTrainingConfig,
    ...     create_training_config,
    ... )
    >>> config = create_training_config(
    ...     batch_size=256,
    ...     num_iterations=100000,
    ...     use_schedules=True,
    ... )
    >>> current_lr = config.get_learning_rate(iteration=5000)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from alpha_zero.utils.logging import get_logger

logger = get_logger(__name__)


class ScheduleType(Enum):
    """Types of parameter schedules."""

    CONSTANT = "constant"
    STEP = "step"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    WARMUP_COSINE = "warmup_cosine"


@dataclass
class SchedulePoint:
    """A point in a parameter schedule.

    Attributes:
        iteration: Iteration at which this value takes effect.
        value: Parameter value at this iteration.
    """

    iteration: int
    value: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {"iteration": self.iteration, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchedulePoint:
        """Create from dictionary.

        Args:
            data: Dictionary with iteration and value.

        Returns:
            SchedulePoint instance.
        """
        return cls(
            iteration=data.get("iteration", 0),
            value=data.get("value", 0.0),
        )


@dataclass
class ParameterSchedule:
    """Configuration for a parameter schedule.

    Attributes:
        schedule_type: Type of schedule.
        initial_value: Starting value.
        final_value: Ending value (for decay schedules).
        decay_steps: Steps over which to decay.
        decay_rate: Decay rate per step interval.
        warmup_steps: Number of warmup steps.
        points: Explicit schedule points (for step schedule).
    """

    schedule_type: ScheduleType = ScheduleType.CONSTANT
    initial_value: float = 0.001
    final_value: float = 0.0001
    decay_steps: int = 100000
    decay_rate: float = 0.1
    warmup_steps: int = 0
    points: list[SchedulePoint] = field(default_factory=list)

    def get_value(self, iteration: int) -> float:
        """Get parameter value at given iteration.

        Args:
            iteration: Current iteration.

        Returns:
            Parameter value.
        """
        if self.schedule_type == ScheduleType.CONSTANT:
            return self.initial_value

        elif self.schedule_type == ScheduleType.STEP:
            return self._step_schedule(iteration)

        elif self.schedule_type == ScheduleType.LINEAR:
            return self._linear_schedule(iteration)

        elif self.schedule_type == ScheduleType.EXPONENTIAL:
            return self._exponential_schedule(iteration)

        elif self.schedule_type == ScheduleType.COSINE:
            return self._cosine_schedule(iteration)

        elif self.schedule_type == ScheduleType.WARMUP_COSINE:
            return self._warmup_cosine_schedule(iteration)

        return self.initial_value

    def _step_schedule(self, iteration: int) -> float:
        """Step decay schedule.

        Args:
            iteration: Current iteration.

        Returns:
            Parameter value.
        """
        if not self.points:
            return self.initial_value

        # Sort points by iteration
        sorted_points = sorted(self.points, key=lambda p: p.iteration)

        value = self.initial_value
        for point in sorted_points:
            if iteration >= point.iteration:
                value = point.value
            else:
                break

        return value

    def _linear_schedule(self, iteration: int) -> float:
        """Linear decay schedule.

        Args:
            iteration: Current iteration.

        Returns:
            Parameter value.
        """
        if iteration >= self.decay_steps:
            return self.final_value

        progress = iteration / self.decay_steps
        return self.initial_value + (self.final_value - self.initial_value) * progress

    def _exponential_schedule(self, iteration: int) -> float:
        """Exponential decay schedule.

        Args:
            iteration: Current iteration.

        Returns:
            Parameter value.
        """
        # Decay by decay_rate every decay_steps iterations
        num_decays = iteration // self.decay_steps
        return self.initial_value * (self.decay_rate ** num_decays)

    def _cosine_schedule(self, iteration: int) -> float:
        """Cosine annealing schedule.

        Args:
            iteration: Current iteration.

        Returns:
            Parameter value.
        """
        if iteration >= self.decay_steps:
            return self.final_value

        progress = iteration / self.decay_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.final_value + (self.initial_value - self.final_value) * cosine_decay

    def _warmup_cosine_schedule(self, iteration: int) -> float:
        """Warmup followed by cosine annealing.

        Args:
            iteration: Current iteration.

        Returns:
            Parameter value.
        """
        if iteration < self.warmup_steps:
            # Linear warmup
            return self.initial_value * (iteration / self.warmup_steps)

        # Cosine decay after warmup
        adjusted_iteration = iteration - self.warmup_steps
        adjusted_decay_steps = self.decay_steps - self.warmup_steps

        if adjusted_iteration >= adjusted_decay_steps:
            return self.final_value

        progress = adjusted_iteration / adjusted_decay_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.final_value + (self.initial_value - self.final_value) * cosine_decay

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "schedule_type": self.schedule_type.value,
            "initial_value": self.initial_value,
            "final_value": self.final_value,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "warmup_steps": self.warmup_steps,
            "points": [p.to_dict() for p in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterSchedule:
        """Create from dictionary.

        Args:
            data: Dictionary with schedule parameters.

        Returns:
            ParameterSchedule instance.
        """
        schedule_type = ScheduleType(data.get("schedule_type", "constant"))
        points_data = data.get("points", [])
        points = [SchedulePoint.from_dict(p) for p in points_data]

        return cls(
            schedule_type=schedule_type,
            initial_value=data.get("initial_value", 0.001),
            final_value=data.get("final_value", 0.0001),
            decay_steps=data.get("decay_steps", 100000),
            decay_rate=data.get("decay_rate", 0.1),
            warmup_steps=data.get("warmup_steps", 0),
            points=points,
        )


@dataclass
class ResourceConfig:
    """Configuration for resource management.

    Attributes:
        max_memory_gb: Maximum GPU memory to use.
        gradient_checkpointing: Enable gradient checkpointing.
        mixed_precision: Enable mixed precision training.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        prefetch_factor: Prefetch batches per worker.
    """

    max_memory_gb: float = 8.0
    gradient_checkpointing: bool = False
    mixed_precision: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "max_memory_gb": self.max_memory_gb,
            "gradient_checkpointing": self.gradient_checkpointing,
            "mixed_precision": self.mixed_precision,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with resource parameters.

        Returns:
            ResourceConfig instance.
        """
        return cls(
            max_memory_gb=data.get("max_memory_gb", 8.0),
            gradient_checkpointing=data.get("gradient_checkpointing", False),
            mixed_precision=data.get("mixed_precision", False),
            num_workers=data.get("num_workers", 4),
            pin_memory=data.get("pin_memory", True),
            prefetch_factor=data.get("prefetch_factor", 2),
        )


@dataclass
class EnhancedTrainingConfig:
    """Enhanced training configuration with schedules.

    Attributes:
        batch_size: Base batch size.
        batch_size_schedule: Optional batch size schedule.
        learning_rate: Base learning rate.
        learning_rate_schedule: Learning rate schedule.
        weight_decay: Weight decay coefficient.
        gradient_clip: Gradient clipping threshold.
        gradient_accumulation_steps: Steps to accumulate gradients.
        num_iterations: Total training iterations.
        games_per_iteration: Self-play games per iteration.
        replay_buffer_size: Maximum replay buffer capacity.
        min_replay_size: Minimum examples before training.
        num_epochs: Training epochs per iteration.
        evaluation_interval: Iterations between evaluations.
        checkpoint_interval: Iterations between checkpoints.
        resources: Resource management configuration.
        seed: Random seed for reproducibility.
    """

    # Batch size
    batch_size: int = 256
    batch_size_schedule: ParameterSchedule | None = None

    # Learning rate
    learning_rate: float = 0.001
    learning_rate_schedule: ParameterSchedule | None = None

    # Optimizer
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1

    # Training loop
    num_iterations: int = 100000
    games_per_iteration: int = 100
    replay_buffer_size: int = 500000
    min_replay_size: int = 10000
    num_epochs: int = 10

    # Evaluation and checkpointing
    evaluation_interval: int = 500
    checkpoint_interval: int = 1000

    # Resources
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    # Reproducibility
    seed: int | None = None

    def get_batch_size(self, iteration: int) -> int:
        """Get batch size at given iteration.

        Args:
            iteration: Current iteration.

        Returns:
            Batch size value.
        """
        if self.batch_size_schedule is None:
            return self.batch_size
        return int(self.batch_size_schedule.get_value(iteration))

    def get_learning_rate(self, iteration: int) -> float:
        """Get learning rate at given iteration.

        Args:
            iteration: Current iteration.

        Returns:
            Learning rate value.
        """
        if self.learning_rate_schedule is None:
            return self.learning_rate
        return self.learning_rate_schedule.get_value(iteration)

    def get_effective_batch_size(self, iteration: int) -> int:
        """Get effective batch size including gradient accumulation.

        Args:
            iteration: Current iteration.

        Returns:
            Effective batch size.
        """
        return self.get_batch_size(iteration) * self.gradient_accumulation_steps

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation error messages.
        """
        errors = []

        if self.batch_size < 1:
            errors.append("batch_size must be positive")

        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")

        if self.gradient_clip <= 0:
            errors.append("gradient_clip must be positive")

        if self.gradient_accumulation_steps < 1:
            errors.append("gradient_accumulation_steps must be at least 1")

        if self.num_iterations < 1:
            errors.append("num_iterations must be positive")

        if self.replay_buffer_size < self.min_replay_size:
            errors.append("replay_buffer_size must be >= min_replay_size")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "batch_size": self.batch_size,
            "batch_size_schedule": (
                self.batch_size_schedule.to_dict()
                if self.batch_size_schedule else None
            ),
            "learning_rate": self.learning_rate,
            "learning_rate_schedule": (
                self.learning_rate_schedule.to_dict()
                if self.learning_rate_schedule else None
            ),
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_iterations": self.num_iterations,
            "games_per_iteration": self.games_per_iteration,
            "replay_buffer_size": self.replay_buffer_size,
            "min_replay_size": self.min_replay_size,
            "num_epochs": self.num_epochs,
            "evaluation_interval": self.evaluation_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "resources": self.resources.to_dict(),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnhancedTrainingConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with configuration.

        Returns:
            EnhancedTrainingConfig instance.
        """
        batch_schedule_data = data.get("batch_size_schedule")
        lr_schedule_data = data.get("learning_rate_schedule")
        resources_data = data.get("resources", {})

        return cls(
            batch_size=data.get("batch_size", 256),
            batch_size_schedule=(
                ParameterSchedule.from_dict(batch_schedule_data)
                if batch_schedule_data else None
            ),
            learning_rate=data.get("learning_rate", 0.001),
            learning_rate_schedule=(
                ParameterSchedule.from_dict(lr_schedule_data)
                if lr_schedule_data else None
            ),
            weight_decay=data.get("weight_decay", 0.0001),
            gradient_clip=data.get("gradient_clip", 1.0),
            gradient_accumulation_steps=data.get("gradient_accumulation_steps", 1),
            num_iterations=data.get("num_iterations", 100000),
            games_per_iteration=data.get("games_per_iteration", 100),
            replay_buffer_size=data.get("replay_buffer_size", 500000),
            min_replay_size=data.get("min_replay_size", 10000),
            num_epochs=data.get("num_epochs", 10),
            evaluation_interval=data.get("evaluation_interval", 500),
            checkpoint_interval=data.get("checkpoint_interval", 1000),
            resources=ResourceConfig.from_dict(resources_data),
            seed=data.get("seed"),
        )


class ResourceManager:
    """Manages compute resources during training."""

    def __init__(self, config: ResourceConfig):
        """Initialize resource manager.

        Args:
            config: Resource configuration.
        """
        self.config = config
        self._logger = get_logger(f"{__name__}.ResourceManager")
        self._memory_usage: list[float] = []

    def estimate_memory_usage(
        self,
        batch_size: int,
        spatial_size: tuple[int, int, int],
        vector_size: int,
        model_params: int,
    ) -> float:
        """Estimate memory usage for training.

        Args:
            batch_size: Training batch size.
            spatial_size: Spatial tensor shape (C, H, W).
            vector_size: Vector tensor dimension.
            model_params: Number of model parameters.

        Returns:
            Estimated memory usage in GB.
        """
        # Estimate tensor sizes (assuming float32 = 4 bytes)
        bytes_per_float = 4

        # Input tensors
        spatial_bytes = batch_size * np.prod(spatial_size) * bytes_per_float
        vector_bytes = batch_size * vector_size * bytes_per_float

        # Model parameters (weights + gradients + optimizer state)
        model_bytes = model_params * bytes_per_float * 4  # ~4x for Adam

        # Activations (rough estimate: 2x input size per layer)
        activation_bytes = (spatial_bytes + vector_bytes) * 10  # Assume 10 layers

        total_bytes = spatial_bytes + vector_bytes + model_bytes + activation_bytes
        total_gb = total_bytes / (1024 ** 3)

        return float(total_gb)

    def recommend_batch_size(
        self,
        spatial_size: tuple[int, int, int],
        vector_size: int,
        model_params: int,
        target_effective_batch: int = 256,
    ) -> tuple[int, int]:
        """Recommend batch size and gradient accumulation steps.

        Args:
            spatial_size: Spatial tensor shape.
            vector_size: Vector tensor dimension.
            model_params: Number of model parameters.
            target_effective_batch: Target effective batch size.

        Returns:
            Tuple of (batch_size, accumulation_steps).
        """
        # Binary search for max batch size that fits in memory
        low, high = 1, target_effective_batch
        max_batch = 1

        while low <= high:
            mid = (low + high) // 2
            mem_usage = self.estimate_memory_usage(
                mid, spatial_size, vector_size, model_params,
            )

            if mem_usage <= self.config.max_memory_gb * 0.8:  # 80% safety margin
                max_batch = mid
                low = mid + 1
            else:
                high = mid - 1

        # Calculate accumulation steps needed
        accumulation_steps = max(1, target_effective_batch // max_batch)
        actual_batch = target_effective_batch // accumulation_steps

        self._logger.info(
            f"Recommended batch_size={actual_batch}, "
            f"accumulation_steps={accumulation_steps} "
            f"(effective={actual_batch * accumulation_steps})",
        )

        return actual_batch, accumulation_steps

    def track_memory_usage(self, usage_gb: float) -> None:
        """Track current memory usage.

        Args:
            usage_gb: Current memory usage in GB.
        """
        self._memory_usage.append(usage_gb)

        if usage_gb > self.config.max_memory_gb * 0.9:
            self._logger.warning(
                f"High memory usage: {usage_gb:.2f}GB "
                f"(limit: {self.config.max_memory_gb}GB)",
            )

    def get_memory_stats(self) -> dict[str, float]:
        """Get memory usage statistics.

        Returns:
            Dictionary with memory statistics.
        """
        if not self._memory_usage:
            return {
                "mean": 0.0,
                "max": 0.0,
                "current": 0.0,
            }

        return {
            "mean": np.mean(self._memory_usage),
            "max": np.max(self._memory_usage),
            "current": self._memory_usage[-1],
        }


def create_training_config(
    batch_size: int = 256,
    learning_rate: float = 0.001,
    num_iterations: int = 100000,
    use_schedules: bool = True,
    **kwargs: Any,
) -> EnhancedTrainingConfig:
    """Factory function to create training configuration.

    Args:
        batch_size: Base batch size.
        learning_rate: Base learning rate.
        num_iterations: Total iterations.
        use_schedules: Whether to use parameter schedules.
        **kwargs: Additional configuration parameters.

    Returns:
        EnhancedTrainingConfig instance.
    """
    config = EnhancedTrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        **kwargs,
    )

    if use_schedules:
        # Default learning rate schedule: warmup + cosine decay
        config.learning_rate_schedule = ParameterSchedule(
            schedule_type=ScheduleType.WARMUP_COSINE,
            initial_value=learning_rate,
            final_value=learning_rate * 0.01,
            decay_steps=num_iterations,
            warmup_steps=num_iterations // 20,
        )

        # Default batch size schedule: ramp up
        config.batch_size_schedule = ParameterSchedule(
            schedule_type=ScheduleType.STEP,
            initial_value=batch_size // 4,
            points=[
                SchedulePoint(iteration=num_iterations // 10, value=batch_size // 2),
                SchedulePoint(iteration=num_iterations // 5, value=batch_size),
            ],
        )

    # Validate
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {', '.join(errors)}")

    return config


def get_fast_training_config() -> EnhancedTrainingConfig:
    """Get configuration optimized for fast development/testing.

    Returns:
        EnhancedTrainingConfig for fast training.
    """
    return EnhancedTrainingConfig(
        batch_size=64,
        learning_rate=0.002,
        num_iterations=1000,
        games_per_iteration=20,
        replay_buffer_size=10000,
        min_replay_size=500,
        num_epochs=5,
        evaluation_interval=100,
        checkpoint_interval=200,
    )


def get_production_training_config() -> EnhancedTrainingConfig:
    """Get configuration for production training.

    Returns:
        EnhancedTrainingConfig for production.
    """
    return create_training_config(
        batch_size=512,
        learning_rate=0.001,
        num_iterations=500000,
        games_per_iteration=200,
        replay_buffer_size=1000000,
        min_replay_size=50000,
        num_epochs=10,
        evaluation_interval=1000,
        checkpoint_interval=5000,
        use_schedules=True,
        gradient_accumulation_steps=2,
    )
