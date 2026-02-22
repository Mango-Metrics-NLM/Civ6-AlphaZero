"""
Spot VM Availability Tracking for Civ6 AlphaZero Cloud Training.

Tracks Spot VM availability patterns and recommends worker counts based on
time-of-day patterns (more workers during off-peak US hours) and historical
preemption data.

Features:
- Time-based availability patterns
- Preemption event tracking
- Worker count recommendations
- Preemption rate analysis

All values configurable via environment variables with CIV6AZ_SPOT_ prefix.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from alpha_zero.cloud.gcp.exceptions import ConfigurationError
from alpha_zero.utils.compat import UTC

logger = logging.getLogger(__name__)


class AvailabilityLevel(str, Enum):
    """Spot VM availability levels."""

    HIGH = "high"  # Off-peak hours, high availability
    MEDIUM = "medium"  # Shoulder hours
    LOW = "low"  # Peak hours, low availability
    UNKNOWN = "unknown"


@dataclass
class SpotPattern:
    """Pattern of Spot availability for a specific time period.

    Attributes:
        hour_start: Start hour (0-23, UTC).
        hour_end: End hour (0-23, UTC).
        days: Days of week (0=Monday, 6=Sunday).
        level: Expected availability level.
        worker_multiplier: Multiplier for worker count.
    """

    hour_start: int
    hour_end: int
    days: list[int]  # 0=Monday, 6=Sunday
    level: AvailabilityLevel
    worker_multiplier: float

    def matches(self, dt: datetime) -> bool:
        """Check if datetime matches this pattern.

        Args:
            dt: Datetime to check (should be UTC).

        Returns:
            True if datetime matches pattern.
        """
        # Check day of week
        if dt.weekday() not in self.days:
            return False

        # Check hour range
        hour = dt.hour
        if self.hour_start <= self.hour_end:
            return self.hour_start <= hour < self.hour_end
        else:
            # Handle wrap-around (e.g., 22:00 to 06:00)
            return hour >= self.hour_start or hour < self.hour_end

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hour_start": self.hour_start,
            "hour_end": self.hour_end,
            "days": self.days,
            "level": self.level.value,
            "worker_multiplier": self.worker_multiplier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpotPattern:
        """Create from dictionary."""
        return cls(
            hour_start=data["hour_start"],
            hour_end=data["hour_end"],
            days=data["days"],
            level=AvailabilityLevel(data["level"]),
            worker_multiplier=data["worker_multiplier"],
        )


@dataclass
class PreemptionEvent:
    """Record of a Spot VM preemption event.

    Attributes:
        timestamp: When preemption occurred (ISO format).
        region: GCP region.
        gpu_type: GPU type.
        worker_id: Worker that was preempted.
        job_duration_hours: How long the job ran before preemption.
    """

    timestamp: str
    region: str
    gpu_type: str
    worker_id: int | None = None
    job_duration_hours: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "region": self.region,
            "gpu_type": self.gpu_type,
            "worker_id": self.worker_id,
            "job_duration_hours": self.job_duration_hours,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreemptionEvent:
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            region=data["region"],
            gpu_type=data["gpu_type"],
            worker_id=data.get("worker_id"),
            job_duration_hours=data.get("job_duration_hours", 0.0),
        )


@dataclass
class SpotAvailabilityConfig:
    """Configuration for Spot VM availability tracking.

    Environment Variables:
        CIV6AZ_SPOT_HISTORY_HOURS: Hours of history to track (default: 168)
        CIV6AZ_SPOT_HIGH_AVAILABILITY_MULTIPLIER: Worker multiplier for high (default: 1.5)
        CIV6AZ_SPOT_LOW_AVAILABILITY_MULTIPLIER: Worker multiplier for low (default: 0.5)
        CIV6AZ_SPOT_PEAK_HOURS_START: US peak hours start UTC (default: 16, 9am PT)
        CIV6AZ_SPOT_PEAK_HOURS_END: US peak hours end UTC (default: 1, 5pm PT)
        CIV6AZ_SPOT_MIN_WORKERS: Minimum workers regardless of availability (default: 2)
        CIV6AZ_SPOT_MAX_WORKERS: Maximum workers during high availability (default: 50)
    """

    history_hours: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_SPOT_HISTORY_HOURS", "168"))
    )
    high_availability_multiplier: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_SPOT_HIGH_AVAILABILITY_MULTIPLIER", "1.5")
        )
    )
    low_availability_multiplier: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_SPOT_LOW_AVAILABILITY_MULTIPLIER", "0.5")
        )
    )
    # US Pacific peak hours converted to UTC (PT is UTC-8 or UTC-7 depending on DST)
    # Using approximate values: 9am PT = 16/17 UTC, 5pm PT = 1/2 UTC
    peak_hours_start_utc: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_SPOT_PEAK_HOURS_START", "16"))
    )
    peak_hours_end_utc: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_SPOT_PEAK_HOURS_END", "1"))
    )
    min_workers: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_SPOT_MIN_WORKERS", "2"))
    )
    max_workers: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_SPOT_MAX_WORKERS", "50"))
    )

    # Default patterns (initialized in __post_init__)
    patterns: list[SpotPattern] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize default patterns if not provided."""
        if not self.patterns:
            self.patterns = self._default_patterns()
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        errors: list[str] = []

        if self.history_hours < 1:
            errors.append("history_hours must be at least 1")

        if self.high_availability_multiplier <= 0:
            errors.append("high_availability_multiplier must be positive")

        if self.low_availability_multiplier <= 0:
            errors.append("low_availability_multiplier must be positive")

        if self.min_workers < 0:
            errors.append("min_workers cannot be negative")

        if self.max_workers < self.min_workers:
            errors.append("max_workers must be >= min_workers")

        if errors:
            raise ConfigurationError(
                f"Invalid spot availability configuration: {'; '.join(errors)}",
                details={"errors": errors},
            )

    def _default_patterns(self) -> list[SpotPattern]:
        """Create default availability patterns based on US business hours."""
        weekdays = [0, 1, 2, 3, 4]  # Monday to Friday
        weekends = [5, 6]  # Saturday, Sunday

        return [
            # Weekday off-peak (midnight to 9am PT = 8 UTC to 16 UTC)
            SpotPattern(
                hour_start=8,
                hour_end=self.peak_hours_start_utc,
                days=weekdays,
                level=AvailabilityLevel.HIGH,
                worker_multiplier=self.high_availability_multiplier,
            ),
            # Weekday peak (9am to 5pm PT = 16 UTC to 1 UTC next day)
            SpotPattern(
                hour_start=self.peak_hours_start_utc,
                hour_end=self.peak_hours_end_utc,
                days=weekdays,
                level=AvailabilityLevel.LOW,
                worker_multiplier=self.low_availability_multiplier,
            ),
            # Weekday late night (5pm PT to midnight = 1 UTC to 8 UTC)
            SpotPattern(
                hour_start=self.peak_hours_end_utc,
                hour_end=8,
                days=weekdays,
                level=AvailabilityLevel.MEDIUM,
                worker_multiplier=1.0,
            ),
            # Weekends - all day high availability
            SpotPattern(
                hour_start=0,
                hour_end=24,
                days=weekends,
                level=AvailabilityLevel.HIGH,
                worker_multiplier=self.high_availability_multiplier,
            ),
        ]

    @classmethod
    def from_env(cls) -> SpotAvailabilityConfig:
        """Create from environment variables."""
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> SpotAvailabilityConfig:
        """Load from YAML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Spot availability config file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpotAvailabilityConfig:
        """Create from dictionary."""
        patterns_data = data.get("patterns", [])
        patterns = [SpotPattern.from_dict(p) for p in patterns_data] if patterns_data else []

        return cls(
            history_hours=data.get("history_hours", 168),
            high_availability_multiplier=data.get("high_availability_multiplier", 1.5),
            low_availability_multiplier=data.get("low_availability_multiplier", 0.5),
            peak_hours_start_utc=data.get("peak_hours_start_utc", 16),
            peak_hours_end_utc=data.get("peak_hours_end_utc", 1),
            min_workers=data.get("min_workers", 2),
            max_workers=data.get("max_workers", 50),
            patterns=patterns,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "history_hours": self.history_hours,
            "high_availability_multiplier": self.high_availability_multiplier,
            "low_availability_multiplier": self.low_availability_multiplier,
            "peak_hours_start_utc": self.peak_hours_start_utc,
            "peak_hours_end_utc": self.peak_hours_end_utc,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "patterns": [p.to_dict() for p in self.patterns],
        }


class SpotAvailabilityTracker:
    """Tracks Spot VM availability patterns and recommends worker counts.

    Uses historical preemption data and time-of-day patterns to predict
    Spot availability and adjust self-play worker counts accordingly.

    Usage:
        config = SpotAvailabilityConfig()
        tracker = SpotAvailabilityTracker(config)

        # Record preemption
        tracker.record_preemption(region="us-central1", gpu_type="NVIDIA_TESLA_T4")

        # Get recommended worker count
        workers = tracker.get_recommended_worker_count(base_workers=10)
    """

    def __init__(self, config: SpotAvailabilityConfig | None = None) -> None:
        """Initialize tracker.

        Args:
            config: Configuration (defaults to from env).
        """
        self.config = config or SpotAvailabilityConfig.from_env()
        self._preemption_history: deque[PreemptionEvent] = deque(maxlen=10000)

        logger.info(
            f"SpotAvailabilityTracker initialized: "
            f"history={self.config.history_hours}h, "
            f"high_mult={self.config.high_availability_multiplier}, "
            f"low_mult={self.config.low_availability_multiplier}"
        )

    def get_current_availability(
        self,
        region: str = "us-central1",
        gpu_type: str = "NVIDIA_TESLA_T4",
        dt: datetime | None = None,
    ) -> AvailabilityLevel:
        """Get current Spot availability level.

        Based on time-of-day patterns and recent preemption history.

        Args:
            region: GCP region.
            gpu_type: GPU type.
            dt: Datetime to check (default: now UTC).

        Returns:
            AvailabilityLevel.
        """
        if dt is None:
            dt = datetime.now(UTC)

        # Check patterns
        for pattern in self.config.patterns:
            if pattern.matches(dt):
                base_level = pattern.level
                break
        else:
            base_level = AvailabilityLevel.MEDIUM

        # Adjust based on recent preemption rate
        preemption_rate = self.get_preemption_rate(region, gpu_type, hours=24)

        if preemption_rate > 0.5:
            # High preemption rate - downgrade availability
            if base_level == AvailabilityLevel.HIGH:
                return AvailabilityLevel.MEDIUM
            elif base_level == AvailabilityLevel.MEDIUM:
                return AvailabilityLevel.LOW
        elif preemption_rate < 0.1:
            # Low preemption rate - upgrade availability
            if base_level == AvailabilityLevel.LOW:
                return AvailabilityLevel.MEDIUM
            elif base_level == AvailabilityLevel.MEDIUM:
                return AvailabilityLevel.HIGH

        return base_level

    def get_recommended_worker_count(
        self,
        base_workers: int,
        region: str = "us-central1",
        gpu_type: str = "NVIDIA_TESLA_T4",
        dt: datetime | None = None,
    ) -> int:
        """Get recommended worker count based on availability.

        Args:
            base_workers: Base worker count to scale.
            region: GCP region.
            gpu_type: GPU type.
            dt: Datetime to check (default: now UTC).

        Returns:
            Recommended worker count.
        """
        if dt is None:
            dt = datetime.now(UTC)

        # Find matching pattern
        multiplier = 1.0
        for pattern in self.config.patterns:
            if pattern.matches(dt):
                multiplier = pattern.worker_multiplier
                break

        # Adjust for preemption rate
        preemption_rate = self.get_preemption_rate(region, gpu_type, hours=24)
        if preemption_rate > 0.3:
            # Reduce workers if seeing high preemption
            multiplier *= 1.0 - (preemption_rate * 0.5)

        recommended = int(base_workers * multiplier)

        # Apply min/max bounds
        recommended = max(self.config.min_workers, recommended)
        recommended = min(self.config.max_workers, recommended)

        logger.debug(
            f"Worker recommendation: base={base_workers}, mult={multiplier:.2f}, "
            f"recommended={recommended}, preemption_rate={preemption_rate:.2f}"
        )

        return recommended

    def record_preemption(
        self,
        region: str,
        gpu_type: str,
        worker_id: int | None = None,
        job_duration_hours: float = 0.0,
    ) -> None:
        """Record a preemption event.

        Args:
            region: GCP region.
            gpu_type: GPU type.
            worker_id: Worker that was preempted.
            job_duration_hours: How long job ran before preemption.
        """
        event = PreemptionEvent(
            timestamp=datetime.now(UTC).isoformat(),
            region=region,
            gpu_type=gpu_type,
            worker_id=worker_id,
            job_duration_hours=job_duration_hours,
        )
        self._preemption_history.append(event)

        logger.info(
            f"Preemption recorded: region={region}, gpu={gpu_type}, "
            f"worker={worker_id}, duration={job_duration_hours:.2f}h"
        )

    def get_preemption_rate(
        self,
        region: str,
        gpu_type: str,
        hours: int = 24,
    ) -> float:
        """Get preemption rate for given region/GPU in recent hours.

        Args:
            region: GCP region.
            gpu_type: GPU type.
            hours: Number of hours to look back.

        Returns:
            Preemption rate (preemptions per hour, normalized 0-1).
        """
        cutoff_time = datetime.now(UTC).timestamp() - (hours * 3600)

        matching_events = [
            e
            for e in self._preemption_history
            if e.region == region
            and e.gpu_type == gpu_type
            and datetime.fromisoformat(e.timestamp).timestamp() > cutoff_time
        ]

        if not matching_events:
            return 0.0

        # Preemptions per hour, capped at 1.0
        rate = len(matching_events) / hours
        return min(1.0, rate)

    def get_best_launch_times(
        self,
        count: int = 3,
    ) -> list[tuple[int, int]]:
        """Get best hours to launch Spot workers (UTC).

        Args:
            count: Number of time windows to return.

        Returns:
            List of (start_hour, end_hour) tuples in UTC.
        """
        # Find patterns with HIGH availability
        high_patterns = [p for p in self.config.patterns if p.level == AvailabilityLevel.HIGH]

        if not high_patterns:
            # No high availability patterns, return early morning
            return [(6, 9), (0, 3), (3, 6)][:count]

        times = []
        for pattern in high_patterns[:count]:
            times.append((pattern.hour_start, pattern.hour_end))

        return times

    def get_statistics(self) -> dict[str, Any]:
        """Get availability statistics.

        Returns:
            Dictionary with tracker statistics.
        """
        now = datetime.now(UTC)
        current_level = self.get_current_availability()

        # Preemption stats by GPU type
        preemption_by_gpu: dict[str, int] = {}
        for event in self._preemption_history:
            preemption_by_gpu[event.gpu_type] = preemption_by_gpu.get(event.gpu_type, 0) + 1

        return {
            "current_availability": current_level.value,
            "current_hour_utc": now.hour,
            "current_day": now.strftime("%A"),
            "total_preemptions_tracked": len(self._preemption_history),
            "preemptions_by_gpu": preemption_by_gpu,
            "24h_preemption_rate": self.get_preemption_rate(
                "us-central1", "NVIDIA_TESLA_T4", hours=24
            ),
            "best_launch_times_utc": self.get_best_launch_times(),
            "pattern_count": len(self.config.patterns),
        }

    def reset_history(self) -> None:
        """Clear preemption history."""
        count = len(self._preemption_history)
        self._preemption_history.clear()
        logger.info(f"Cleared {count} preemption events from history")


def get_spot_tracker() -> SpotAvailabilityTracker:
    """Get a spot availability tracker instance.

    Convenience function for getting a tracker.

    Returns:
        SpotAvailabilityTracker instance.
    """
    return SpotAvailabilityTracker(SpotAvailabilityConfig.from_env())
