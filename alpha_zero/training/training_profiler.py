"""
Training Profiler for Civ6 AlphaZero.

Profiles training to detect diminishing returns and recommends optimal
training steps per iteration. Enables early stopping when additional
training steps provide diminishing returns.

Features:
- Loss curve analysis
- Diminishing returns detection
- Training step recommendations
- Early stopping support
- Cost-per-improvement metrics

All values configurable via environment variables with CIV6AZ_PROFILE_ prefix.
"""

from __future__ import annotations

import logging
import os
import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class DiminishingReturnsLevel(str, Enum):
    """Level of diminishing returns detected."""

    NONE = "none"  # Training still improving well
    MILD = "mild"  # Slight diminishing returns
    MODERATE = "moderate"  # Noticeable diminishing returns
    SEVERE = "severe"  # Strong diminishing returns, consider stopping


@dataclass
class TrainingStepMetrics:
    """Metrics for a single training step.

    Attributes:
        step: Training step number within iteration.
        iteration: Training iteration.
        loss: Total training loss.
        policy_loss: Policy head loss.
        value_loss: Value head loss.
        grad_norm: Gradient norm.
        learning_rate: Current learning rate.
        wall_time_seconds: Wall clock time for this step.
    """

    step: int
    iteration: int
    loss: float
    policy_loss: float | None = None
    value_loss: float | None = None
    grad_norm: float | None = None
    learning_rate: float | None = None
    wall_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "iteration": self.iteration,
            "loss": self.loss,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
            "wall_time_seconds": self.wall_time_seconds,
        }


@dataclass
class DiminishingReturnsAnalysis:
    """Analysis of diminishing returns in training.

    Attributes:
        level: Detected level of diminishing returns.
        improvement_rate: Rate of improvement (loss decrease per step).
        cost_per_improvement: Estimated cost per unit improvement.
        recommended_steps: Recommended steps for this iteration.
        reason: Human-readable reason for recommendation.
        confidence: Confidence in analysis (0-1).
    """

    level: DiminishingReturnsLevel
    improvement_rate: float
    cost_per_improvement: float
    recommended_steps: int
    reason: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "improvement_rate": self.improvement_rate,
            "cost_per_improvement": self.cost_per_improvement,
            "recommended_steps": self.recommended_steps,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass
class TrainingProfilerConfig:
    """Configuration for training profiler.

    Environment Variables:
        CIV6AZ_PROFILE_WINDOW_SIZE: Steps to analyze (default: 100)
        CIV6AZ_PROFILE_MIN_SAMPLES: Min samples before analysis (default: 50)
        CIV6AZ_PROFILE_SEVERE_THRESHOLD: Improvement rate for severe (default: 0.001)
        CIV6AZ_PROFILE_MODERATE_THRESHOLD: Improvement rate for moderate (default: 0.005)
        CIV6AZ_PROFILE_MILD_THRESHOLD: Improvement rate for mild (default: 0.01)
        CIV6AZ_PROFILE_COST_PER_STEP: Estimated cost per step USD (default: 0.001)
        CIV6AZ_PROFILE_EARLY_STOP_ENABLED: Enable early stop rec (default: true)
    """

    window_size: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_PROFILE_WINDOW_SIZE", "100"))
    )
    min_samples: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_PROFILE_MIN_SAMPLES", "50"))
    )
    severe_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_PROFILE_SEVERE_THRESHOLD", "0.001"))
    )
    moderate_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_PROFILE_MODERATE_THRESHOLD", "0.005"))
    )
    mild_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_PROFILE_MILD_THRESHOLD", "0.01"))
    )
    cost_per_step_usd: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_PROFILE_COST_PER_STEP", "0.001"))
    )
    early_stop_enabled: bool = field(
        default_factory=lambda: os.environ.get("CIV6AZ_PROFILE_EARLY_STOP_ENABLED", "true").lower()
        == "true"
    )

    # Recommended steps configuration
    min_steps_per_iteration: int = 100
    max_steps_per_iteration: int = 10000
    default_steps_per_iteration: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values."""
        errors: list[str] = []

        if self.window_size < 10:
            errors.append("window_size must be at least 10")

        if self.min_samples < 10:
            errors.append("min_samples must be at least 10")

        if self.severe_threshold >= self.moderate_threshold:
            errors.append("severe_threshold must be < moderate_threshold")

        if self.moderate_threshold >= self.mild_threshold:
            errors.append("moderate_threshold must be < mild_threshold")

        if self.cost_per_step_usd < 0:
            errors.append("cost_per_step_usd cannot be negative")

        if errors:
            from alpha_zero.cloud.gcp.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid profiler configuration: {'; '.join(errors)}",
                details={"errors": errors},
            )

    @classmethod
    def from_env(cls) -> TrainingProfilerConfig:
        """Create from environment variables."""
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingProfilerConfig:
        """Load from YAML file."""
        path = Path(path)
        if not path.exists():
            from alpha_zero.cloud.gcp.exceptions import ConfigurationError

            raise ConfigurationError(f"Profiler config file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            from alpha_zero.cloud.gcp.exceptions import ConfigurationError

            raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingProfilerConfig:
        """Create from dictionary."""
        return cls(
            window_size=data.get("window_size", 100),
            min_samples=data.get("min_samples", 50),
            severe_threshold=data.get("severe_threshold", 0.001),
            moderate_threshold=data.get("moderate_threshold", 0.005),
            mild_threshold=data.get("mild_threshold", 0.01),
            cost_per_step_usd=data.get("cost_per_step_usd", 0.001),
            early_stop_enabled=data.get("early_stop_enabled", True),
            min_steps_per_iteration=data.get("min_steps_per_iteration", 100),
            max_steps_per_iteration=data.get("max_steps_per_iteration", 10000),
            default_steps_per_iteration=data.get("default_steps_per_iteration", 1000),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_size": self.window_size,
            "min_samples": self.min_samples,
            "severe_threshold": self.severe_threshold,
            "moderate_threshold": self.moderate_threshold,
            "mild_threshold": self.mild_threshold,
            "cost_per_step_usd": self.cost_per_step_usd,
            "early_stop_enabled": self.early_stop_enabled,
            "min_steps_per_iteration": self.min_steps_per_iteration,
            "max_steps_per_iteration": self.max_steps_per_iteration,
            "default_steps_per_iteration": self.default_steps_per_iteration,
        }


class TrainingProfiler:
    """Profiles training to detect diminishing returns.

    Analyzes training loss curves to detect when additional training steps
    provide diminishing returns, allowing dynamic adjustment of training
    steps per iteration.

    Usage:
        config = TrainingProfilerConfig()
        profiler = TrainingProfiler(config)

        for step in range(max_steps):
            loss = train_step()
            profiler.record_step(step, iteration, loss)

            analysis = profiler.analyze()
            if analysis.level == DiminishingReturnsLevel.SEVERE:
                break
    """

    def __init__(self, config: TrainingProfilerConfig | None = None) -> None:
        """Initialize profiler.

        Args:
            config: Configuration (defaults to from env).
        """
        self.config = config or TrainingProfilerConfig.from_env()
        self._current_iteration: int = -1
        self._step_history: deque[TrainingStepMetrics] = deque(maxlen=self.config.window_size * 2)
        self._iteration_history: dict[int, list[TrainingStepMetrics]] = {}

        logger.info(
            f"TrainingProfiler initialized: window={self.config.window_size}, "
            f"early_stop={self.config.early_stop_enabled}"
        )

    def record_step(
        self,
        step: int,
        iteration: int,
        loss: float,
        policy_loss: float | None = None,
        value_loss: float | None = None,
        grad_norm: float | None = None,
        learning_rate: float | None = None,
        wall_time_seconds: float = 0.0,
    ) -> None:
        """Record metrics for a training step.

        Args:
            step: Training step number within iteration.
            iteration: Training iteration number.
            loss: Total training loss.
            policy_loss: Policy head loss.
            value_loss: Value head loss.
            grad_norm: Gradient norm.
            learning_rate: Current learning rate.
            wall_time_seconds: Wall clock time for this step.
        """
        metrics = TrainingStepMetrics(
            step=step,
            iteration=iteration,
            loss=loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            grad_norm=grad_norm,
            learning_rate=learning_rate,
            wall_time_seconds=wall_time_seconds,
        )

        # Handle iteration change
        if iteration != self._current_iteration:
            self._current_iteration = iteration
            self._iteration_history[iteration] = []

        self._step_history.append(metrics)
        self._iteration_history[iteration].append(metrics)

    def analyze(self, iteration: int | None = None) -> DiminishingReturnsAnalysis:
        """Analyze training for diminishing returns.

        Args:
            iteration: Iteration to analyze (default: current).

        Returns:
            DiminishingReturnsAnalysis with results.
        """
        if iteration is None:
            iteration = self._current_iteration

        # Get metrics for this iteration
        metrics = self._iteration_history.get(iteration, [])
        if len(metrics) < self.config.min_samples:
            return DiminishingReturnsAnalysis(
                level=DiminishingReturnsLevel.NONE,
                improvement_rate=0.0,
                cost_per_improvement=0.0,
                recommended_steps=self.config.default_steps_per_iteration,
                reason=f"Insufficient samples ({len(metrics)} < {self.config.min_samples})",
                confidence=0.0,
            )

        # Calculate improvement rate
        improvement_rate = self.get_improvement_rate(window=self.config.window_size)

        # Determine diminishing returns level
        if improvement_rate < self.config.severe_threshold:
            level = DiminishingReturnsLevel.SEVERE
        elif improvement_rate < self.config.moderate_threshold:
            level = DiminishingReturnsLevel.MODERATE
        elif improvement_rate < self.config.mild_threshold:
            level = DiminishingReturnsLevel.MILD
        else:
            level = DiminishingReturnsLevel.NONE

        # Calculate cost per improvement
        if improvement_rate > 0:
            cost_per_improvement = self.config.cost_per_step_usd / improvement_rate
        else:
            cost_per_improvement = float("inf")

        # Get recommended steps
        recommended_steps = self._calculate_recommended_steps(len(metrics), improvement_rate, level)

        # Build reason
        reason = self._build_reason(level, improvement_rate, len(metrics))

        # Calculate confidence
        confidence = self._calculate_confidence(metrics, improvement_rate)

        return DiminishingReturnsAnalysis(
            level=level,
            improvement_rate=improvement_rate,
            cost_per_improvement=cost_per_improvement,
            recommended_steps=recommended_steps,
            reason=reason,
            confidence=confidence,
        )

    def get_improvement_rate(self, window: int | None = None) -> float:
        """Get recent improvement rate (loss decrease per step).

        Uses linear regression to calculate the slope of loss decrease.

        Args:
            window: Number of recent steps to analyze.

        Returns:
            Improvement rate (positive = improving).
        """
        if window is None:
            window = self.config.window_size

        # Get recent losses for current iteration
        metrics = self._iteration_history.get(self._current_iteration, [])
        recent = metrics[-window:] if len(metrics) > window else metrics

        if len(recent) < 2:
            return 0.0

        losses = [m.loss for m in recent]

        # Calculate improvement rate using linear regression slope
        n = len(losses)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(losses)

        numerator = sum((i - x_mean) * (loss - y_mean) for i, loss in enumerate(losses))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator

        # Negative slope means loss is decreasing (improvement)
        # Return positive value for improvement rate
        return -slope

    def should_stop_early(self) -> tuple[bool, str]:
        """Check if training should stop early this iteration.

        Returns:
            Tuple of (should_stop, reason).
        """
        if not self.config.early_stop_enabled:
            return False, "Early stopping disabled"

        analysis = self.analyze()

        if analysis.level == DiminishingReturnsLevel.SEVERE:
            return True, f"Severe diminishing returns: {analysis.reason}"

        if analysis.level == DiminishingReturnsLevel.MODERATE:
            metrics = self._iteration_history.get(self._current_iteration, [])
            if len(metrics) >= self.config.min_steps_per_iteration * 2:
                return True, f"Moderate diminishing returns after {len(metrics)} steps"

        return False, "Training progressing normally"

    def get_recommended_steps(
        self,
        iteration: int,
        total_iterations: int,
        default_steps: int | None = None,
    ) -> int:
        """Get recommended training steps for iteration.

        Args:
            iteration: Current iteration number.
            total_iterations: Total iterations planned.
            default_steps: Default steps if no recommendation.

        Returns:
            Recommended number of training steps.
        """
        if default_steps is None:
            default_steps = self.config.default_steps_per_iteration

        # If we have history from previous iterations, use that
        if iteration > 0 and (iteration - 1) in self._iteration_history:
            prev_metrics = self._iteration_history[iteration - 1]
            prev_steps = len(prev_metrics)

            # Analyze previous iteration's improvement rate
            if prev_metrics:
                prev_losses = [m.loss for m in prev_metrics]
                if len(prev_losses) >= 2:
                    # Check if previous iteration had diminishing returns
                    late_improvement = (
                        prev_losses[-len(prev_losses) // 2] - prev_losses[-1]
                    ) / max(len(prev_losses) // 2, 1)

                    if late_improvement < self.config.severe_threshold:
                        # Reduce steps if previous iteration had severe diminishing returns
                        return max(self.config.min_steps_per_iteration, int(prev_steps * 0.8))
                    elif late_improvement > self.config.mild_threshold * 2:
                        # Increase steps if previous iteration still improving
                        return min(self.config.max_steps_per_iteration, int(prev_steps * 1.2))

        return default_steps

    def _calculate_recommended_steps(
        self,
        current_steps: int,
        improvement_rate: float,
        level: DiminishingReturnsLevel,
    ) -> int:
        """Calculate recommended steps based on current analysis."""
        if level == DiminishingReturnsLevel.NONE:
            # Keep going
            return max(current_steps, self.config.default_steps_per_iteration)

        elif level == DiminishingReturnsLevel.MILD:
            # Continue a bit more
            return min(
                current_steps + self.config.window_size,
                self.config.max_steps_per_iteration,
            )

        elif level == DiminishingReturnsLevel.MODERATE:
            # Stop soon
            return current_steps + self.config.window_size // 2

        else:  # SEVERE
            # Stop now
            return current_steps

    def _build_reason(
        self,
        level: DiminishingReturnsLevel,
        improvement_rate: float,
        step_count: int,
    ) -> str:
        """Build human-readable reason string."""
        if level == DiminishingReturnsLevel.NONE:
            return f"Training improving well at rate {improvement_rate:.4f}/step"
        elif level == DiminishingReturnsLevel.MILD:
            return f"Mild diminishing returns at step {step_count} (rate={improvement_rate:.4f})"
        elif level == DiminishingReturnsLevel.MODERATE:
            return (
                f"Moderate diminishing returns at step {step_count} (rate={improvement_rate:.4f})"
            )
        else:
            return f"Severe diminishing returns at step {step_count} (rate={improvement_rate:.6f})"

    def _calculate_confidence(
        self,
        metrics: list[TrainingStepMetrics],
        improvement_rate: float,
    ) -> float:
        """Calculate confidence in analysis."""
        # More samples = higher confidence
        sample_confidence = min(1.0, len(metrics) / (self.config.window_size * 2))

        # More consistent losses = higher confidence
        losses = [m.loss for m in metrics[-self.config.window_size :]]
        if len(losses) > 1:
            cv = statistics.stdev(losses) / max(statistics.mean(losses), 1e-8)
            consistency_confidence = max(0.5, 1.0 - cv)
        else:
            consistency_confidence = 0.5

        return round((sample_confidence + consistency_confidence) / 2, 2)

    def reset_iteration(self) -> None:
        """Reset profiler for new iteration."""
        self._step_history.clear()
        logger.debug("TrainingProfiler reset for new iteration")

    def reset(self) -> None:
        """Fully reset profiler state."""
        self._current_iteration = -1
        self._step_history.clear()
        self._iteration_history.clear()
        logger.info("TrainingProfiler fully reset")

    def get_statistics(self) -> dict[str, Any]:
        """Get profiler statistics.

        Returns:
            Dictionary with profiler statistics.
        """
        current_metrics = self._iteration_history.get(self._current_iteration, [])

        return {
            "current_iteration": self._current_iteration,
            "current_step_count": len(current_metrics),
            "total_iterations_tracked": len(self._iteration_history),
            "window_size": self.config.window_size,
            "early_stop_enabled": self.config.early_stop_enabled,
            "current_improvement_rate": self.get_improvement_rate() if current_metrics else None,
        }


def get_training_profiler() -> TrainingProfiler:
    """Get a training profiler instance.

    Convenience function for getting a profiler.

    Returns:
        TrainingProfiler instance.
    """
    return TrainingProfiler(TrainingProfilerConfig.from_env())
