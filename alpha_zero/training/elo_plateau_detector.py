"""
Elo Rating Plateau Detection for Civ6 AlphaZero Training.

Monitors Elo rating progression and detects when training has plateaued
(no improvement for >N iterations). Integrates with the alerting system
to notify when intervention may be needed.

Features:
- Elo rating tracking over iterations
- Plateau detection with configurable threshold
- Trend analysis
- Alert integration
- Callbacks for plateau/improvement events

All values configurable via environment variables with CIV6AZ_ELO_ prefix.
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
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from alpha_zero.cloud.gcp.alert_manager import AlertManager

logger = logging.getLogger(__name__)


class EloStatus(str, Enum):
    """Elo progression status."""

    IMPROVING = "improving"
    STABLE = "stable"
    PLATEAU = "plateau"
    REGRESSING = "regressing"


@dataclass
class EloSnapshot:
    """Snapshot of Elo rating at a point in time.

    Attributes:
        iteration: Training iteration number.
        elo_rating: Current Elo rating.
        win_rate: Win rate in evaluation games.
        games_played: Number of evaluation games played.
        timestamp: When this snapshot was recorded.
        opponent_type: Type of opponent (self, random, previous).
    """

    iteration: int
    elo_rating: float
    win_rate: float
    games_played: int
    timestamp: float = field(default_factory=time.time)
    opponent_type: str = "self"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "elo_rating": self.elo_rating,
            "win_rate": self.win_rate,
            "games_played": self.games_played,
            "timestamp": self.timestamp,
            "opponent_type": self.opponent_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EloSnapshot:
        """Create from dictionary."""
        return cls(
            iteration=data["iteration"],
            elo_rating=data["elo_rating"],
            win_rate=data["win_rate"],
            games_played=data["games_played"],
            timestamp=data.get("timestamp", time.time()),
            opponent_type=data.get("opponent_type", "self"),
        )


@dataclass
class EloPlateauDetectorConfig:
    """Configuration for Elo plateau detection.

    Environment Variables:
        CIV6AZ_ELO_PLATEAU_THRESHOLD: Iterations before plateau (default: 5)
        CIV6AZ_ELO_IMPROVEMENT_THRESHOLD: Elo points for improvement (default: 10.0)
        CIV6AZ_ELO_REGRESSION_THRESHOLD: Elo points for regression (default: 50.0)
        CIV6AZ_ELO_WINDOW_SIZE: Window size for trend analysis (default: 10)
        CIV6AZ_ELO_MIN_ITERATIONS: Minimum iterations before detection (default: 3)
        CIV6AZ_ELO_ALERT_ON_PLATEAU: Send alerts on plateau (default: true)
    """

    plateau_threshold_iterations: int = field(
        default_factory=lambda: int(
            os.environ.get("CIV6AZ_ELO_PLATEAU_THRESHOLD", "5")
        )
    )
    improvement_threshold_elo: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_ELO_IMPROVEMENT_THRESHOLD", "10.0")
        )
    )
    regression_threshold_elo: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_ELO_REGRESSION_THRESHOLD", "50.0")
        )
    )
    window_size: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_ELO_WINDOW_SIZE", "10"))
    )
    min_iterations: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_ELO_MIN_ITERATIONS", "3"))
    )
    alert_on_plateau: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_ELO_ALERT_ON_PLATEAU", "true"
        ).lower() == "true"
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values."""
        errors: list[str] = []

        if self.plateau_threshold_iterations < 1:
            errors.append("plateau_threshold_iterations must be at least 1")

        if self.improvement_threshold_elo <= 0:
            errors.append("improvement_threshold_elo must be positive")

        if self.regression_threshold_elo <= 0:
            errors.append("regression_threshold_elo must be positive")

        if self.window_size < 2:
            errors.append("window_size must be at least 2")

        if self.min_iterations < 1:
            errors.append("min_iterations must be at least 1")

        if errors:
            from alpha_zero.cloud.gcp.exceptions import ConfigurationError
            raise ConfigurationError(
                f"Invalid Elo detector configuration: {'; '.join(errors)}",
                details={"errors": errors},
            )

    @classmethod
    def from_env(cls) -> EloPlateauDetectorConfig:
        """Create from environment variables."""
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> EloPlateauDetectorConfig:
        """Load from YAML file."""
        path = Path(path)
        if not path.exists():
            from alpha_zero.cloud.gcp.exceptions import ConfigurationError
            raise ConfigurationError(f"Elo detector config file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            from alpha_zero.cloud.gcp.exceptions import ConfigurationError
            raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EloPlateauDetectorConfig:
        """Create from dictionary."""
        return cls(
            plateau_threshold_iterations=data.get("plateau_threshold_iterations", 5),
            improvement_threshold_elo=data.get("improvement_threshold_elo", 10.0),
            regression_threshold_elo=data.get("regression_threshold_elo", 50.0),
            window_size=data.get("window_size", 10),
            min_iterations=data.get("min_iterations", 3),
            alert_on_plateau=data.get("alert_on_plateau", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plateau_threshold_iterations": self.plateau_threshold_iterations,
            "improvement_threshold_elo": self.improvement_threshold_elo,
            "regression_threshold_elo": self.regression_threshold_elo,
            "window_size": self.window_size,
            "min_iterations": self.min_iterations,
            "alert_on_plateau": self.alert_on_plateau,
        }


class EloPlateauDetector:
    """Detects Elo rating plateaus during training.

    Monitors Elo ratings over iterations and detects when improvement
    has stalled beyond a configurable threshold.

    Usage:
        config = EloPlateauDetectorConfig()
        detector = EloPlateauDetector(config)

        # Record evaluation results
        detector.record_evaluation(
            iteration=100,
            elo_rating=1550.0,
            win_rate=0.55,
            games_played=200,
        )

        # Check for plateau
        if detector.is_plateau():
            # Trigger intervention (hyperparameter sweep, etc.)
            pass
    """

    def __init__(
        self,
        config: EloPlateauDetectorConfig | None = None,
        alert_manager: AlertManager | None = None,
    ) -> None:
        """Initialize Elo plateau detector.

        Args:
            config: Configuration for detection.
            alert_manager: Optional alert manager for notifications.
        """
        self.config = config or EloPlateauDetectorConfig.from_env()
        self._alert_manager = alert_manager

        # Elo history
        self._history: deque[EloSnapshot] = deque(maxlen=self.config.window_size * 2)

        # Best values tracking
        self._best_elo: float | None = None
        self._best_elo_iteration: int = 0
        self._iterations_since_improvement: int = 0

        # Callbacks
        self._plateau_callbacks: list[Callable[[EloSnapshot], None]] = []
        self._improvement_callbacks: list[Callable[[EloSnapshot], None]] = []

        # Statistics
        self._total_evaluations: int = 0
        self._plateau_count: int = 0
        self._alert_sent_for_current_plateau: bool = False

        logger.info(
            f"EloPlateauDetector initialized: "
            f"plateau_threshold={self.config.plateau_threshold_iterations}, "
            f"improvement_threshold={self.config.improvement_threshold_elo}"
        )

    def record_evaluation(
        self,
        iteration: int,
        elo_rating: float,
        win_rate: float,
        games_played: int,
        opponent_type: str = "self",
    ) -> None:
        """Record an evaluation result.

        Args:
            iteration: Training iteration number.
            elo_rating: Current Elo rating.
            win_rate: Win rate in evaluation games.
            games_played: Number of games played.
            opponent_type: Type of opponent (self, random, previous).
        """
        snapshot = EloSnapshot(
            iteration=iteration,
            elo_rating=elo_rating,
            win_rate=win_rate,
            games_played=games_played,
            opponent_type=opponent_type,
        )

        self._history.append(snapshot)
        self._total_evaluations += 1

        # Check for improvement
        improved = self._check_improvement(elo_rating, iteration)

        if improved:
            self._alert_sent_for_current_plateau = False
            for callback in self._improvement_callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    logger.error(f"Improvement callback error: {e}")

        # Check for plateau and send alert
        if self.is_plateau() and not self._alert_sent_for_current_plateau:
            self._plateau_count += 1
            self._alert_sent_for_current_plateau = True

            if self.config.alert_on_plateau and self._alert_manager:
                self._alert_manager.elo_plateau(
                    current_elo=elo_rating,
                    iterations_stalled=self._iterations_since_improvement,
                    threshold=self.config.plateau_threshold_iterations,
                )

            for callback in self._plateau_callbacks:
                try:
                    callback(snapshot)
                except Exception as e:
                    logger.error(f"Plateau callback error: {e}")

    def _check_improvement(self, elo_rating: float, iteration: int) -> bool:
        """Check if Elo rating improved significantly.

        Args:
            elo_rating: Current Elo rating.
            iteration: Current iteration.

        Returns:
            True if significant improvement detected.
        """
        if self._best_elo is None:
            self._best_elo = elo_rating
            self._best_elo_iteration = iteration
            self._iterations_since_improvement = 0
            logger.info(f"Initial Elo: {elo_rating:.0f}")
            return True

        improvement = elo_rating - self._best_elo

        if improvement >= self.config.improvement_threshold_elo:
            old_best = self._best_elo
            self._best_elo = elo_rating
            self._best_elo_iteration = iteration
            self._iterations_since_improvement = 0

            logger.info(
                f"Elo improvement: {old_best:.0f} -> {elo_rating:.0f} "
                f"(+{improvement:.0f})"
            )
            return True
        else:
            self._iterations_since_improvement += 1

            logger.debug(
                f"Elo: {elo_rating:.0f}, best: {self._best_elo:.0f}, "
                f"iterations without improvement: {self._iterations_since_improvement}"
            )
            return False

    def is_plateau(self) -> bool:
        """Check if Elo rating has plateaued.

        Returns:
            True if iterations without improvement exceeds threshold.
        """
        if len(self._history) < self.config.min_iterations:
            return False

        return (
            self._iterations_since_improvement
            >= self.config.plateau_threshold_iterations
        )

    def get_status(self) -> EloStatus:
        """Get current Elo progression status.

        Returns:
            EloStatus indicating current state.
        """
        if len(self._history) < self.config.min_iterations:
            return EloStatus.STABLE

        if self._iterations_since_improvement == 0:
            return EloStatus.IMPROVING

        if (
            self._iterations_since_improvement
            >= self.config.plateau_threshold_iterations
        ):
            return EloStatus.PLATEAU

        # Check for regression
        recent = list(self._history)[-self.config.window_size :]
        if len(recent) >= 2:
            half = len(recent) // 2
            first_half = [s.elo_rating for s in recent[:half]]
            second_half = [s.elo_rating for s in recent[half:]]

            if first_half and second_half:
                first_mean = statistics.mean(first_half)
                second_mean = statistics.mean(second_half)

                if (
                    first_mean - second_mean
                    >= self.config.regression_threshold_elo
                ):
                    return EloStatus.REGRESSING

        return EloStatus.STABLE

    def get_trend(self) -> float | None:
        """Calculate Elo trend (positive = improving).

        Uses linear regression to determine trend direction.

        Returns:
            Trend value (Elo points per iteration) or None if insufficient data.
        """
        if len(self._history) < self.config.window_size:
            return None

        recent = list(self._history)[-self.config.window_size :]
        ratings = [s.elo_rating for s in recent]

        if len(ratings) < 2:
            return None

        # Simple linear regression slope
        n = len(ratings)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(ratings)

        numerator = sum(
            (i - x_mean) * (r - y_mean) for i, r in enumerate(ratings)
        )
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return None

        return numerator / denominator

    def get_recent_elo(self, count: int = 5) -> list[float]:
        """Get recent Elo ratings.

        Args:
            count: Number of recent ratings to return.

        Returns:
            List of recent Elo ratings.
        """
        recent = list(self._history)[-count:]
        return [s.elo_rating for s in recent]

    def get_statistics(self) -> dict[str, Any]:
        """Get detector statistics.

        Returns:
            Dictionary with detector statistics.
        """
        recent = list(self._history)[-self.config.window_size :]
        recent_ratings = [s.elo_rating for s in recent]

        return {
            "total_evaluations": self._total_evaluations,
            "best_elo": self._best_elo,
            "best_elo_iteration": self._best_elo_iteration,
            "iterations_since_improvement": self._iterations_since_improvement,
            "current_status": self.get_status().value,
            "plateau_count": self._plateau_count,
            "current_elo": self._history[-1].elo_rating if self._history else None,
            "recent_mean_elo": (
                statistics.mean(recent_ratings) if recent_ratings else None
            ),
            "recent_std_elo": (
                statistics.stdev(recent_ratings) if len(recent_ratings) > 1 else None
            ),
            "trend": self.get_trend(),
            "is_plateau": self.is_plateau(),
        }

    def register_plateau_callback(
        self, callback: Callable[[EloSnapshot], None]
    ) -> None:
        """Register callback for plateau detection.

        Args:
            callback: Function to call when plateau detected.
        """
        self._plateau_callbacks.append(callback)

    def register_improvement_callback(
        self, callback: Callable[[EloSnapshot], None]
    ) -> None:
        """Register callback for improvements.

        Args:
            callback: Function to call when improvement detected.
        """
        self._improvement_callbacks.append(callback)

    def reset(self) -> None:
        """Reset detector state."""
        self._history.clear()
        self._best_elo = None
        self._best_elo_iteration = 0
        self._iterations_since_improvement = 0
        self._plateau_count = 0
        self._alert_sent_for_current_plateau = False
        logger.info("EloPlateauDetector state reset")


def get_elo_detector(
    alert_manager: AlertManager | None = None,
) -> EloPlateauDetector:
    """Get an Elo plateau detector instance.

    Convenience function for getting a detector.

    Args:
        alert_manager: Optional alert manager for notifications.

    Returns:
        EloPlateauDetector instance.
    """
    return EloPlateauDetector(
        EloPlateauDetectorConfig.from_env(),
        alert_manager=alert_manager,
    )
