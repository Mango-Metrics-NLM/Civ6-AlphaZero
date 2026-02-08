"""
Worker Autoscaler for Civ6 AlphaZero Cloud Training.

Dynamically scales self-play workers based on multiple factors:
- Spot VM availability patterns
- Cost budget constraints
- Training phase requirements
- Current worker utilization

All values configurable via environment variables with CIV6AZ_AUTOSCALE_ prefix.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from alpha_zero.cloud.gcp.exceptions import ConfigurationError

if TYPE_CHECKING:
    from alpha_zero.cloud.gcp.cost_monitor import CostMonitor
    from alpha_zero.cloud.gcp.spot_availability import SpotAvailabilityTracker

logger = logging.getLogger(__name__)


class ScalingAction(str, Enum):
    """Scaling action types."""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


@dataclass
class ScalingDecision:
    """A scaling decision with reasoning.

    Attributes:
        target_workers: Recommended number of workers.
        current_workers: Current number of workers.
        action: Scale up, down, or maintain.
        reason: Human-readable reason for decision.
        confidence: Confidence in decision (0-1).
        factors: Breakdown of factors that influenced decision.
    """

    target_workers: int
    current_workers: int
    action: ScalingAction
    reason: str
    confidence: float = 1.0
    factors: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target_workers": self.target_workers,
            "current_workers": self.current_workers,
            "action": self.action.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "factors": self.factors,
        }


@dataclass
class WorkerMetrics:
    """Metrics for a single worker.

    Attributes:
        worker_id: Worker identifier.
        games_completed: Total games completed.
        avg_game_duration_seconds: Average game duration.
        preempted: Whether worker was preempted.
        last_update: Last metrics update timestamp.
    """

    worker_id: int
    games_completed: int = 0
    avg_game_duration_seconds: float = 0.0
    preempted: bool = False
    last_update: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worker_id": self.worker_id,
            "games_completed": self.games_completed,
            "avg_game_duration_seconds": self.avg_game_duration_seconds,
            "preempted": self.preempted,
            "last_update": self.last_update,
        }


@dataclass
class WorkerAutoscalerConfig:
    """Configuration for worker autoscaling.

    Environment Variables:
        CIV6AZ_AUTOSCALE_ENABLED: Enable autoscaling (default: true)
        CIV6AZ_AUTOSCALE_BASE_WORKERS: Base worker count (default: 10)
        CIV6AZ_AUTOSCALE_MIN_WORKERS: Minimum workers (default: 2)
        CIV6AZ_AUTOSCALE_MAX_WORKERS: Maximum workers (default: 50)
        CIV6AZ_AUTOSCALE_SCALE_UP_THRESHOLD: Utilization for scale up (default: 0.8)
        CIV6AZ_AUTOSCALE_SCALE_DOWN_THRESHOLD: Utilization for scale down (default: 0.3)
        CIV6AZ_AUTOSCALE_COOLDOWN_SECONDS: Cooldown between decisions (default: 300)
    """

    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_AUTOSCALE_ENABLED", "true"
        ).lower() == "true"
    )
    base_workers: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_AUTOSCALE_BASE_WORKERS", "10"))
    )
    min_workers: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_AUTOSCALE_MIN_WORKERS", "2"))
    )
    max_workers: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_AUTOSCALE_MAX_WORKERS", "50"))
    )
    scale_up_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_AUTOSCALE_SCALE_UP_THRESHOLD", "0.8")
        )
    )
    scale_down_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_AUTOSCALE_SCALE_DOWN_THRESHOLD", "0.3")
        )
    )
    cooldown_seconds: int = field(
        default_factory=lambda: int(
            os.environ.get("CIV6AZ_AUTOSCALE_COOLDOWN_SECONDS", "300")
        )
    )

    # Factors for scaling decision
    use_spot_availability: bool = True
    use_cost_budget: bool = True
    use_training_phase: bool = True

    # Scaling step sizes
    scale_up_percent: float = 0.25  # Scale up by 25%
    scale_down_percent: float = 0.20  # Scale down by 20%

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        errors: list[str] = []

        if self.base_workers < 1:
            errors.append("base_workers must be at least 1")

        if self.min_workers < 0:
            errors.append("min_workers cannot be negative")

        if self.max_workers < self.min_workers:
            errors.append("max_workers must be >= min_workers")

        if not 0 < self.scale_up_threshold <= 1:
            errors.append("scale_up_threshold must be between 0 and 1")

        if not 0 <= self.scale_down_threshold < 1:
            errors.append("scale_down_threshold must be between 0 and 1")

        if self.scale_down_threshold >= self.scale_up_threshold:
            errors.append("scale_down_threshold must be < scale_up_threshold")

        if self.cooldown_seconds < 0:
            errors.append("cooldown_seconds cannot be negative")

        if errors:
            raise ConfigurationError(
                f"Invalid autoscaler configuration: {'; '.join(errors)}",
                details={"errors": errors},
            )

    @classmethod
    def from_env(cls) -> WorkerAutoscalerConfig:
        """Create from environment variables."""
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorkerAutoscalerConfig:
        """Load from YAML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Autoscaler config file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkerAutoscalerConfig:
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            base_workers=data.get("base_workers", 10),
            min_workers=data.get("min_workers", 2),
            max_workers=data.get("max_workers", 50),
            scale_up_threshold=data.get("scale_up_threshold", 0.8),
            scale_down_threshold=data.get("scale_down_threshold", 0.3),
            cooldown_seconds=data.get("cooldown_seconds", 300),
            use_spot_availability=data.get("use_spot_availability", True),
            use_cost_budget=data.get("use_cost_budget", True),
            use_training_phase=data.get("use_training_phase", True),
            scale_up_percent=data.get("scale_up_percent", 0.25),
            scale_down_percent=data.get("scale_down_percent", 0.20),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "base_workers": self.base_workers,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
            "cooldown_seconds": self.cooldown_seconds,
            "use_spot_availability": self.use_spot_availability,
            "use_cost_budget": self.use_cost_budget,
            "use_training_phase": self.use_training_phase,
            "scale_up_percent": self.scale_up_percent,
            "scale_down_percent": self.scale_down_percent,
        }


class WorkerAutoscaler:
    """Autoscales self-play workers based on multiple factors.

    Considers:
    - Spot VM availability patterns
    - Cost budget constraints
    - Training phase requirements
    - Current worker utilization

    Usage:
        config = WorkerAutoscalerConfig()
        autoscaler = WorkerAutoscaler(
            config,
            spot_tracker=spot_tracker,
            cost_monitor=cost_monitor,
        )

        decision = autoscaler.get_scaling_decision(current_workers=10)
        if decision.action == ScalingAction.SCALE_UP:
            launch_workers(decision.target_workers - decision.current_workers)
    """

    def __init__(
        self,
        config: WorkerAutoscalerConfig | None = None,
        spot_tracker: SpotAvailabilityTracker | None = None,
        cost_monitor: CostMonitor | None = None,
    ) -> None:
        """Initialize autoscaler.

        Args:
            config: Configuration (defaults to from env).
            spot_tracker: Optional spot availability tracker.
            cost_monitor: Optional cost monitor.
        """
        self.config = config or WorkerAutoscalerConfig.from_env()
        self._spot_tracker = spot_tracker
        self._cost_monitor = cost_monitor
        self._worker_metrics: dict[int, WorkerMetrics] = {}
        self._last_scaling_time: float = 0.0
        self._scaling_history: list[ScalingDecision] = []

        logger.info(
            f"WorkerAutoscaler initialized: enabled={self.config.enabled}, "
            f"base={self.config.base_workers}, "
            f"range=[{self.config.min_workers}, {self.config.max_workers}]"
        )

    def get_scaling_decision(
        self,
        current_workers: int,
        iteration: int = 0,
        total_iterations: int = 1,
        utilization: float = 1.0,
    ) -> ScalingDecision:
        """Get scaling decision based on current state.

        Args:
            current_workers: Current number of workers.
            iteration: Current training iteration.
            total_iterations: Total iterations planned.
            utilization: Current utilization (0-1).

        Returns:
            ScalingDecision with recommendation.
        """
        if not self.config.enabled:
            return ScalingDecision(
                target_workers=current_workers,
                current_workers=current_workers,
                action=ScalingAction.MAINTAIN,
                reason="Autoscaling disabled",
            )

        # Check cooldown
        if self._in_cooldown():
            return ScalingDecision(
                target_workers=current_workers,
                current_workers=current_workers,
                action=ScalingAction.MAINTAIN,
                reason="In cooldown period",
            )

        factors: dict[str, float] = {}
        target = float(self.config.base_workers)

        # Factor 1: Spot availability
        if self.config.use_spot_availability and self._spot_tracker:
            spot_mult = self._get_spot_multiplier()
            factors["spot_availability"] = spot_mult
            target *= spot_mult

        # Factor 2: Cost budget
        if self.config.use_cost_budget and self._cost_monitor:
            cost_mult = self._get_cost_multiplier()
            factors["cost_budget"] = cost_mult
            target *= cost_mult

        # Factor 3: Training phase
        if self.config.use_training_phase and total_iterations > 0:
            phase_mult = self._get_phase_multiplier(iteration, total_iterations)
            factors["training_phase"] = phase_mult
            target *= phase_mult

        # Factor 4: Utilization
        if utilization < self.config.scale_down_threshold:
            util_mult = 0.8  # Scale down
            factors["utilization"] = util_mult
            target *= util_mult
        elif utilization > self.config.scale_up_threshold:
            util_mult = 1.2  # Scale up
            factors["utilization"] = util_mult
            target *= util_mult
        else:
            factors["utilization"] = 1.0

        # Apply bounds
        target = round(target)
        target = max(self.config.min_workers, target)
        target = min(self.config.max_workers, target)

        # Determine action
        if target > current_workers:
            action = ScalingAction.SCALE_UP
            reason = self._build_reason("scale up", factors)
        elif target < current_workers:
            action = ScalingAction.SCALE_DOWN
            reason = self._build_reason("scale down", factors)
        else:
            action = ScalingAction.MAINTAIN
            reason = "No scaling needed"

        decision = ScalingDecision(
            target_workers=target,
            current_workers=current_workers,
            action=action,
            reason=reason,
            confidence=self._calculate_confidence(factors),
            factors=factors,
        )

        # Record decision
        if action != ScalingAction.MAINTAIN:
            self._last_scaling_time = time.time()
            self._scaling_history.append(decision)

        logger.info(
            f"Scaling decision: {action.value} from {current_workers} to {target} "
            f"({reason})"
        )

        return decision

    def _in_cooldown(self) -> bool:
        """Check if we're in cooldown period."""
        return time.time() - self._last_scaling_time < self.config.cooldown_seconds

    def _get_spot_multiplier(self) -> float:
        """Get multiplier based on spot availability."""
        if not self._spot_tracker:
            return 1.0

        from alpha_zero.cloud.gcp.spot_availability import AvailabilityLevel

        level = self._spot_tracker.get_current_availability()

        if level == AvailabilityLevel.HIGH:
            return 1.5
        elif level == AvailabilityLevel.MEDIUM:
            return 1.0
        elif level == AvailabilityLevel.LOW:
            return 0.5
        else:
            return 1.0

    def _get_cost_multiplier(self) -> float:
        """Get multiplier based on cost budget."""
        if not self._cost_monitor:
            return 1.0

        try:
            status = self._cost_monitor.get_cost_status()
            remaining = status.total_remaining_usd

            if remaining <= 0:
                return 0.0  # Budget exceeded
            elif status.alert_level.value == "critical":
                return 0.3  # Critical budget
            elif status.alert_level.value == "warning":
                return 0.6  # Warning budget
            else:
                return 1.0  # Normal budget
        except Exception as e:
            logger.warning(f"Failed to get cost status: {e}")
            return 1.0

    def _get_phase_multiplier(self, iteration: int, total_iterations: int) -> float:
        """Get multiplier based on training phase."""
        if total_iterations <= 0:
            return 1.0

        progress = iteration / total_iterations

        # Ramp up at start, maintain in middle, taper at end
        if progress < 0.1:
            return 0.5 + (progress / 0.1) * 0.5  # Ramp up
        elif progress > 0.9:
            return 1.0 - ((progress - 0.9) / 0.1) * 0.3  # Taper
        else:
            return 1.0  # Full capacity

    def _build_reason(self, action: str, factors: dict[str, float]) -> str:
        """Build human-readable reason string."""
        parts = [action]
        for name, value in factors.items():
            if value != 1.0:
                parts.append(f"{name}={value:.2f}")
        return " due to " + ", ".join(parts[1:]) if len(parts) > 1 else parts[0]

    def _calculate_confidence(self, factors: dict[str, float]) -> float:
        """Calculate confidence in scaling decision."""
        # Higher variance in factors = lower confidence
        values = list(factors.values())
        if not values:
            return 1.0

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)

        # Map variance to confidence (higher variance = lower confidence)
        confidence = max(0.5, 1.0 - variance)
        return round(confidence, 2)

    def get_optimal_worker_count(
        self,
        iteration: int = 0,
        total_iterations: int = 1,
    ) -> int:
        """Get optimal worker count for current conditions.

        Convenience method that doesn't require current worker count.

        Args:
            iteration: Current training iteration.
            total_iterations: Total iterations planned.

        Returns:
            Optimal worker count.
        """
        decision = self.get_scaling_decision(
            current_workers=self.config.base_workers,
            iteration=iteration,
            total_iterations=total_iterations,
        )
        return decision.target_workers

    def record_worker_metrics(
        self,
        worker_id: int,
        games_completed: int,
        avg_game_duration: float,
        preempted: bool = False,
    ) -> None:
        """Record worker performance metrics.

        Args:
            worker_id: Worker identifier.
            games_completed: Number of games completed.
            avg_game_duration: Average game duration in seconds.
            preempted: Whether worker was preempted.
        """
        self._worker_metrics[worker_id] = WorkerMetrics(
            worker_id=worker_id,
            games_completed=games_completed,
            avg_game_duration_seconds=avg_game_duration,
            preempted=preempted,
        )

        if preempted and self._spot_tracker:
            self._spot_tracker.record_preemption(
                region="us-central1",
                gpu_type="NVIDIA_TESLA_T4",
                worker_id=worker_id,
            )

    def get_statistics(self) -> dict[str, Any]:
        """Get autoscaler statistics.

        Returns:
            Dictionary with autoscaler statistics.
        """
        now = time.time()
        cooldown_remaining = max(
            0, self.config.cooldown_seconds - (now - self._last_scaling_time)
        )

        recent_decisions = [d.to_dict() for d in self._scaling_history[-10:]]

        return {
            "enabled": self.config.enabled,
            "base_workers": self.config.base_workers,
            "min_workers": self.config.min_workers,
            "max_workers": self.config.max_workers,
            "in_cooldown": self._in_cooldown(),
            "cooldown_remaining_seconds": cooldown_remaining,
            "total_scaling_decisions": len(self._scaling_history),
            "recent_decisions": recent_decisions,
            "tracked_workers": len(self._worker_metrics),
        }

    def reset(self) -> None:
        """Reset autoscaler state."""
        self._worker_metrics.clear()
        self._last_scaling_time = 0.0
        self._scaling_history.clear()
        logger.info("WorkerAutoscaler state reset")


def get_worker_autoscaler(
    spot_tracker: SpotAvailabilityTracker | None = None,
    cost_monitor: CostMonitor | None = None,
) -> WorkerAutoscaler:
    """Get a worker autoscaler instance.

    Convenience function for getting an autoscaler.

    Args:
        spot_tracker: Optional spot availability tracker.
        cost_monitor: Optional cost monitor.

    Returns:
        WorkerAutoscaler instance.
    """
    return WorkerAutoscaler(
        WorkerAutoscalerConfig.from_env(),
        spot_tracker=spot_tracker,
        cost_monitor=cost_monitor,
    )
