"""Training convergence monitoring and intervention.

This module monitors training progress, detects convergence stalls,
and provides recommendations for Population Based Training (PBT)
mutations and curriculum adjustments.

Usage:
    from alpha_zero.training.convergence_monitor import (
        ConvergenceMonitorConfig,
        ConvergenceMonitor,
    )

    config = ConvergenceMonitorConfig()
    monitor = ConvergenceMonitor(config)

    # Record training metrics
    monitor.record_metrics(
        iteration=1000,
        policy_loss=0.5,
        value_loss=0.3,
        win_rate=0.55,
    )

    # Check for stalls
    if monitor.is_stalled():
        recommendations = monitor.get_recommendations()
"""

from __future__ import annotations

import logging
import os
import statistics
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ConvergenceStatus(str, Enum):
    """Convergence status levels."""

    IMPROVING = "improving"  # Metrics are improving
    STABLE = "stable"  # Metrics are stable but acceptable
    SLOWING = "slowing"  # Improvement rate is decreasing
    STALLED = "stalled"  # No improvement detected
    DIVERGING = "diverging"  # Metrics are getting worse


class RecommendationType(str, Enum):
    """Types of recommendations."""

    CONTINUE = "continue"  # Continue training as is
    REDUCE_LR = "reduce_lr"  # Reduce learning rate
    INCREASE_EXPLORATION = "increase_exploration"  # Increase MCTS exploration
    CURRICULUM_ADVANCE = "curriculum_advance"  # Advance curriculum stage
    CURRICULUM_RETREAT = "curriculum_retreat"  # Go back in curriculum
    PBT_MUTATE = "pbt_mutate"  # Trigger PBT mutation
    EARLY_STOP = "early_stop"  # Consider early stopping
    INCREASE_SIMULATIONS = "increase_simulations"  # More MCTS simulations


@dataclass
class MetricSnapshot:
    """Snapshot of training metrics at a point in time.

    Attributes:
        iteration: Training iteration number.
        timestamp: Unix timestamp.
        policy_loss: Policy network loss.
        value_loss: Value network loss.
        total_loss: Total combined loss.
        win_rate: Win rate against reference.
        elo_rating: ELO rating estimate.
        exploration_rate: Current exploration rate.
        learning_rate: Current learning rate.
        extra_metrics: Additional custom metrics.
    """

    iteration: int
    timestamp: float = field(default_factory=time.time)
    policy_loss: float | None = None
    value_loss: float | None = None
    total_loss: float | None = None
    win_rate: float | None = None
    elo_rating: float | None = None
    exploration_rate: float | None = None
    learning_rate: float | None = None
    extra_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "total_loss": self.total_loss,
            "win_rate": self.win_rate,
            "elo_rating": self.elo_rating,
            "exploration_rate": self.exploration_rate,
            "learning_rate": self.learning_rate,
            "extra_metrics": self.extra_metrics,
        }


@dataclass
class Recommendation:
    """A recommendation for improving training.

    Attributes:
        type: Type of recommendation.
        priority: Priority level (higher = more urgent).
        reason: Human-readable reason.
        suggested_value: Suggested new value if applicable.
        confidence: Confidence in recommendation (0-1).
    """

    type: RecommendationType
    priority: int
    reason: str
    suggested_value: float | None = None
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type.value,
            "priority": self.priority,
            "reason": self.reason,
            "suggested_value": self.suggested_value,
            "confidence": self.confidence,
        }


@dataclass
class ConvergenceMonitorConfig:
    """Configuration for convergence monitor.

    Attributes:
        window_size: Number of iterations for trend analysis.
        stall_threshold_iterations: Iterations without improvement = stall.
        improvement_threshold: Minimum improvement to count as progress.
        divergence_threshold: Degradation threshold for divergence.
        min_win_rate: Minimum acceptable win rate.
        target_win_rate: Target win rate.
        lr_reduction_factor: Factor for learning rate reduction.
        exploration_boost_factor: Factor for exploration increase.
        enable_pbt_recommendations: Enable PBT mutation recommendations.
        enable_curriculum_recommendations: Enable curriculum recommendations.
    """

    window_size: int = field(
        default_factory=lambda: int(
            os.environ.get("CIV6AZ_CONVERGENCE_WINDOW", "50")
        )
    )
    stall_threshold_iterations: int = field(
        default_factory=lambda: int(
            os.environ.get("CIV6AZ_CONVERGENCE_STALL_THRESHOLD", "100")
        )
    )
    improvement_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_CONVERGENCE_IMPROVEMENT_THRESHOLD", "0.01")
        )
    )
    divergence_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_CONVERGENCE_DIVERGENCE_THRESHOLD", "0.1")
        )
    )
    min_win_rate: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_CONVERGENCE_MIN_WIN_RATE", "0.5")
        )
    )
    target_win_rate: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_CONVERGENCE_TARGET_WIN_RATE", "0.7")
        )
    )
    lr_reduction_factor: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_CONVERGENCE_LR_REDUCTION", "0.5")
        )
    )
    exploration_boost_factor: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_CONVERGENCE_EXPLORATION_BOOST", "1.5")
        )
    )
    enable_pbt_recommendations: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_CONVERGENCE_PBT_ENABLED", "true"
        ).lower() == "true"
    )
    enable_curriculum_recommendations: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_CONVERGENCE_CURRICULUM_ENABLED", "true"
        ).lower() == "true"
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values."""
        if self.window_size < 5:
            raise ValueError("window_size must be >= 5")
        if self.stall_threshold_iterations < 10:
            raise ValueError("stall_threshold_iterations must be >= 10")
        if self.improvement_threshold <= 0:
            raise ValueError("improvement_threshold must be positive")
        if self.divergence_threshold <= 0:
            raise ValueError("divergence_threshold must be positive")
        if not 0 <= self.min_win_rate <= 1:
            raise ValueError("min_win_rate must be between 0 and 1")
        if not 0 <= self.target_win_rate <= 1:
            raise ValueError("target_win_rate must be between 0 and 1")
        if not 0 < self.lr_reduction_factor < 1:
            raise ValueError("lr_reduction_factor must be between 0 and 1")
        if self.exploration_boost_factor <= 1:
            raise ValueError("exploration_boost_factor must be > 1")

    @classmethod
    def from_env(cls) -> ConvergenceMonitorConfig:
        """Create configuration from environment variables."""
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> ConvergenceMonitorConfig:
        """Load configuration from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if "convergence_monitor" in data:
            data = data["convergence_monitor"]

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvergenceMonitorConfig:
        """Create configuration from dictionary."""
        kwargs: dict[str, Any] = {}

        field_types = {
            "window_size": int,
            "stall_threshold_iterations": int,
            "improvement_threshold": float,
            "divergence_threshold": float,
            "min_win_rate": float,
            "target_win_rate": float,
            "lr_reduction_factor": float,
            "exploration_boost_factor": float,
            "enable_pbt_recommendations": bool,
            "enable_curriculum_recommendations": bool,
        }

        for field_name, field_type in field_types.items():
            if field_name in data:
                kwargs[field_name] = field_type(data[field_name])

        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "window_size": self.window_size,
            "stall_threshold_iterations": self.stall_threshold_iterations,
            "improvement_threshold": self.improvement_threshold,
            "divergence_threshold": self.divergence_threshold,
            "min_win_rate": self.min_win_rate,
            "target_win_rate": self.target_win_rate,
            "lr_reduction_factor": self.lr_reduction_factor,
            "exploration_boost_factor": self.exploration_boost_factor,
            "enable_pbt_recommendations": self.enable_pbt_recommendations,
            "enable_curriculum_recommendations": self.enable_curriculum_recommendations,
        }


class ConvergenceMonitor:
    """Monitors training convergence and provides recommendations.

    This class tracks training metrics over time, detects stalls and
    divergence, and provides recommendations for interventions like
    learning rate reduction, PBT mutations, or curriculum changes.

    Attributes:
        config: Convergence monitor configuration.
    """

    def __init__(self, config: ConvergenceMonitorConfig | None = None):
        """Initialize convergence monitor.

        Args:
            config: Convergence monitor configuration.
        """
        self.config = config or ConvergenceMonitorConfig()

        # Metric history
        self._history: deque[MetricSnapshot] = deque(maxlen=self.config.window_size * 2)

        # Best values seen
        self._best_loss: float | None = None
        self._best_win_rate: float | None = None
        self._best_iteration: int = 0

        # Stall tracking
        self._iterations_since_improvement = 0
        self._last_improvement_iteration = 0

        # Callbacks
        self._on_stall: list[Callable[[ConvergenceStatus], None]] = []
        self._on_improvement: list[Callable[[MetricSnapshot], None]] = []

        # Statistics
        self._total_recommendations = 0
        self._stalls_detected = 0

        logger.info(
            f"ConvergenceMonitor initialized: "
            f"window={self.config.window_size}, "
            f"stall_threshold={self.config.stall_threshold_iterations}"
        )

    def record_metrics(
        self,
        iteration: int,
        policy_loss: float | None = None,
        value_loss: float | None = None,
        total_loss: float | None = None,
        win_rate: float | None = None,
        elo_rating: float | None = None,
        exploration_rate: float | None = None,
        learning_rate: float | None = None,
        **extra_metrics: float,
    ) -> None:
        """Record training metrics.

        Args:
            iteration: Training iteration number.
            policy_loss: Policy network loss.
            value_loss: Value network loss.
            total_loss: Total combined loss.
            win_rate: Win rate against reference.
            elo_rating: ELO rating estimate.
            exploration_rate: Current exploration rate.
            learning_rate: Current learning rate.
            **extra_metrics: Additional custom metrics.
        """
        snapshot = MetricSnapshot(
            iteration=iteration,
            policy_loss=policy_loss,
            value_loss=value_loss,
            total_loss=total_loss,
            win_rate=win_rate,
            elo_rating=elo_rating,
            exploration_rate=exploration_rate,
            learning_rate=learning_rate,
            extra_metrics=dict(extra_metrics),
        )

        self._history.append(snapshot)

        # Check for improvement
        improved = False

        if total_loss is not None:
            threshold = self.config.improvement_threshold
            if self._best_loss is None or total_loss < self._best_loss - threshold:
                self._best_loss = total_loss
                improved = True

        if win_rate is not None:
            threshold = self.config.improvement_threshold
            if self._best_win_rate is None or win_rate > self._best_win_rate + threshold:
                self._best_win_rate = win_rate
                improved = True

        if improved:
            self._iterations_since_improvement = 0
            self._last_improvement_iteration = iteration
            self._best_iteration = iteration
            for callback in self._on_improvement:
                try:
                    callback(snapshot)
                except Exception as e:
                    logger.error(f"Error in improvement callback: {e}")
        else:
            if len(self._history) > 1:
                self._iterations_since_improvement = iteration - self._last_improvement_iteration

    def get_status(self) -> ConvergenceStatus:
        """Get current convergence status.

        Returns:
            ConvergenceStatus indicating current training state.
        """
        if len(self._history) < self.config.window_size:
            return ConvergenceStatus.STABLE  # Not enough data

        # Check for stall
        if self._iterations_since_improvement >= self.config.stall_threshold_iterations:
            return ConvergenceStatus.STALLED

        # Calculate recent trend
        recent = list(self._history)[-self.config.window_size:]
        losses = [s.total_loss for s in recent if s.total_loss is not None]
        win_rates = [s.win_rate for s in recent if s.win_rate is not None]

        # Check for divergence (loss increasing significantly)
        if len(losses) >= 5:
            first_half = losses[:len(losses)//2]
            second_half = losses[len(losses)//2:]
            if first_half and second_half:
                first_avg = statistics.mean(first_half)
                second_avg = statistics.mean(second_half)
                if second_avg > first_avg * (1 + self.config.divergence_threshold):
                    return ConvergenceStatus.DIVERGING

        # Check for improvement
        if len(losses) >= 5:
            first_half = losses[:len(losses)//2]
            second_half = losses[len(losses)//2:]
            if first_half and second_half:
                first_avg = statistics.mean(first_half)
                second_avg = statistics.mean(second_half)
                if second_avg < first_avg * (1 - self.config.improvement_threshold):
                    return ConvergenceStatus.IMPROVING

        # Check win rate trend
        if len(win_rates) >= 5:
            first_half = win_rates[:len(win_rates)//2]
            second_half = win_rates[len(win_rates)//2:]
            if first_half and second_half:
                first_avg = statistics.mean(first_half)
                second_avg = statistics.mean(second_half)
                if second_avg > first_avg + self.config.improvement_threshold:
                    return ConvergenceStatus.IMPROVING

        # Check if improvement is slowing
        if self._iterations_since_improvement > self.config.stall_threshold_iterations // 2:
            return ConvergenceStatus.SLOWING

        return ConvergenceStatus.STABLE

    def is_stalled(self) -> bool:
        """Check if training is stalled.

        Returns:
            True if training appears stalled.
        """
        status = self.get_status()
        if status == ConvergenceStatus.STALLED:
            self._stalls_detected += 1
            for callback in self._on_stall:
                try:
                    callback(status)
                except Exception as e:
                    logger.error(f"Error in stall callback: {e}")
            return True
        return False

    def is_diverging(self) -> bool:
        """Check if training is diverging.

        Returns:
            True if metrics are getting worse.
        """
        return self.get_status() == ConvergenceStatus.DIVERGING

    def get_recommendations(self) -> list[Recommendation]:
        """Get recommendations for improving training.

        Returns:
            List of recommendations sorted by priority.
        """
        recommendations: list[Recommendation] = []
        status = self.get_status()

        if status == ConvergenceStatus.IMPROVING:
            recommendations.append(Recommendation(
                type=RecommendationType.CONTINUE,
                priority=1,
                reason="Training is improving, continue current approach",
                confidence=0.9,
            ))
            return recommendations

        # Get recent metrics
        recent = list(self._history)[-self.config.window_size:] if self._history else []

        # Learning rate recommendation
        if status in (ConvergenceStatus.STALLED, ConvergenceStatus.SLOWING):
            current_lr = recent[-1].learning_rate if recent and recent[-1].learning_rate else None
            if current_lr:
                new_lr = current_lr * self.config.lr_reduction_factor
                recommendations.append(Recommendation(
                    type=RecommendationType.REDUCE_LR,
                    priority=8 if status == ConvergenceStatus.STALLED else 5,
                    reason="Learning rate may be too high for current state",
                    suggested_value=new_lr,
                    confidence=0.7,
                ))

        # Exploration recommendation
        if status == ConvergenceStatus.STALLED:
            has_exploration = recent and recent[-1].exploration_rate
            current_exploration = recent[-1].exploration_rate if has_exploration else None
            if current_exploration:
                boost = self.config.exploration_boost_factor
                new_exploration = min(1.0, current_exploration * boost)
                recommendations.append(Recommendation(
                    type=RecommendationType.INCREASE_EXPLORATION,
                    priority=6,
                    reason="Increase exploration to escape local minimum",
                    suggested_value=new_exploration,
                    confidence=0.6,
                ))

        # MCTS simulations recommendation
        if status == ConvergenceStatus.SLOWING:
            recommendations.append(Recommendation(
                type=RecommendationType.INCREASE_SIMULATIONS,
                priority=4,
                reason="More MCTS simulations may improve policy quality",
                confidence=0.5,
            ))

        # PBT mutation recommendation
        if self.config.enable_pbt_recommendations and status == ConvergenceStatus.STALLED:
            recommendations.append(Recommendation(
                type=RecommendationType.PBT_MUTATE,
                priority=7,
                reason="Consider PBT mutation to explore new hyperparameters",
                confidence=0.65,
            ))

        # Curriculum recommendations
        if self.config.enable_curriculum_recommendations:
            win_rates = [s.win_rate for s in recent if s.win_rate is not None]
            if win_rates:
                avg_win_rate = statistics.mean(win_rates)

                if avg_win_rate >= self.config.target_win_rate:
                    recommendations.append(Recommendation(
                        type=RecommendationType.CURRICULUM_ADVANCE,
                        priority=9,
                        reason=f"Win rate {avg_win_rate:.1%} exceeds target, advance curriculum",
                        confidence=0.8,
                    ))
                is_stalled = status == ConvergenceStatus.STALLED
                if avg_win_rate < self.config.min_win_rate and is_stalled:
                    recommendations.append(Recommendation(
                        type=RecommendationType.CURRICULUM_RETREAT,
                        priority=8,
                        reason=f"Win rate {avg_win_rate:.1%} too low, retreat curriculum",
                        confidence=0.7,
                    ))

        # Divergence recommendation
        if status == ConvergenceStatus.DIVERGING:
            recommendations.append(Recommendation(
                type=RecommendationType.REDUCE_LR,
                priority=10,
                reason="Training is diverging, reduce learning rate urgently",
                suggested_value=(recent[-1].learning_rate or 0.001) * 0.1,
                confidence=0.85,
            ))

        # Early stopping recommendation
        if self._iterations_since_improvement > self.config.stall_threshold_iterations * 2:
            recommendations.append(Recommendation(
                type=RecommendationType.EARLY_STOP,
                priority=3,
                reason="No improvement for extended period, consider early stopping",
                confidence=0.6,
            ))

        # Sort by priority (higher = more important)
        recommendations.sort(key=lambda r: r.priority, reverse=True)

        self._total_recommendations += len(recommendations)
        return recommendations

    def get_improvement_rate(self) -> float | None:
        """Calculate rate of improvement.

        Returns:
            Improvement rate (negative = improving) or None.
        """
        if len(self._history) < self.config.window_size:
            return None

        recent = list(self._history)[-self.config.window_size:]
        losses = [s.total_loss for s in recent if s.total_loss is not None]

        if len(losses) < 2:
            return None

        first_avg = statistics.mean(losses[:len(losses)//2])
        second_avg = statistics.mean(losses[len(losses)//2:])

        if first_avg == 0:
            return None

        return (second_avg - first_avg) / first_avg

    def get_best_metrics(self) -> dict[str, Any]:
        """Get best metrics seen during training.

        Returns:
            Dictionary with best metrics.
        """
        return {
            "best_loss": self._best_loss,
            "best_win_rate": self._best_win_rate,
            "best_iteration": self._best_iteration,
            "iterations_since_improvement": self._iterations_since_improvement,
        }

    def register_stall_callback(
        self, callback: Callable[[ConvergenceStatus], None]
    ) -> None:
        """Register callback for training stalls."""
        self._on_stall.append(callback)

    def register_improvement_callback(
        self, callback: Callable[[MetricSnapshot], None]
    ) -> None:
        """Register callback for improvements."""
        self._on_improvement.append(callback)

    def get_statistics(self) -> dict[str, Any]:
        """Get convergence monitor statistics."""
        return {
            "history_size": len(self._history),
            "best_loss": self._best_loss,
            "best_win_rate": self._best_win_rate,
            "best_iteration": self._best_iteration,
            "iterations_since_improvement": self._iterations_since_improvement,
            "stalls_detected": self._stalls_detected,
            "total_recommendations": self._total_recommendations,
            "current_status": self.get_status().value,
        }

    def reset(self) -> None:
        """Reset convergence monitor state."""
        self._history.clear()
        self._best_loss = None
        self._best_win_rate = None
        self._best_iteration = 0
        self._iterations_since_improvement = 0
        self._last_improvement_iteration = 0
        self._stalls_detected = 0
        self._total_recommendations = 0
        logger.info("ConvergenceMonitor state reset")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ConvergenceMonitor("
            f"status={self.get_status().value}, "
            f"best_loss={self._best_loss}, "
            f"iterations_since_improvement={self._iterations_since_improvement})"
        )
