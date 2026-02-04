"""
Cost Monitoring for GCP Cloud Training.

Provides real-time cost tracking, budget enforcement, and alerting
for Vertex AI training jobs.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from alpha_zero.cloud.gcp.exceptions import BudgetExceededError, ConfigurationError

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Cost alert severity levels."""

    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    BUDGET_EXCEEDED = "budget_exceeded"


@dataclass
class CostBreakdown:
    """Breakdown of current costs.

    Attributes:
        total_today_usd: Total cost accumulated today.
        total_project_usd: Total project cost since tracking began.
        by_gpu_type: Cost breakdown by GPU type.
        by_job_type: Cost breakdown by job type (selfplay, training).
        timestamp: When this breakdown was calculated.
    """

    total_today_usd: float
    total_project_usd: float
    by_gpu_type: dict[str, float]
    by_job_type: dict[str, float]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_today_usd": self.total_today_usd,
            "total_project_usd": self.total_project_usd,
            "by_gpu_type": self.by_gpu_type,
            "by_job_type": self.by_job_type,
            "timestamp": self.timestamp,
        }


@dataclass
class CostStatus:
    """Current cost monitoring status.

    Attributes:
        breakdown: Detailed cost breakdown.
        alert_level: Current alert level based on budget usage.
        daily_remaining_usd: Remaining daily budget.
        total_remaining_usd: Remaining total budget.
        estimated_daily_rate_usd: Estimated daily spending rate.
    """

    breakdown: CostBreakdown
    alert_level: AlertLevel
    daily_remaining_usd: float
    total_remaining_usd: float
    estimated_daily_rate_usd: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "breakdown": self.breakdown.to_dict(),
            "alert_level": self.alert_level.value,
            "daily_remaining_usd": self.daily_remaining_usd,
            "total_remaining_usd": self.total_remaining_usd,
            "estimated_daily_rate_usd": self.estimated_daily_rate_usd,
        }


@dataclass
class CostMonitorConfig:
    """Configuration for cost monitoring and budget enforcement.

    All values can be overridden via environment variables with the CIV6AZ_ prefix.

    Environment Variables:
        CIV6AZ_COST_BUDGET_DAILY: Daily budget in USD (default: 50.0)
        CIV6AZ_COST_BUDGET_TOTAL: Total budget limit (default: 1000.0)
        CIV6AZ_COST_ALERT_THRESHOLD: Alert at this percentage (default: 0.8)
        CIV6AZ_COST_CRITICAL_THRESHOLD: Critical at percentage (default: 0.95)
        CIV6AZ_COST_CHECK_INTERVAL_S: How often to check costs (default: 300)
        CIV6AZ_COST_BILLING_API_ENABLED: Use GCP Billing API (default: False)
    """

    daily_budget_usd: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_COST_BUDGET_DAILY", "50.0"))
    )
    total_budget_usd: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_COST_BUDGET_TOTAL", "1000.0"))
    )
    alert_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_COST_ALERT_THRESHOLD", "0.8"))
    )
    critical_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_COST_CRITICAL_THRESHOLD", "0.95"))
    )
    check_interval_seconds: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_COST_CHECK_INTERVAL_S", "300"))
    )
    billing_api_enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_COST_BILLING_API_ENABLED", "false"
        ).lower() == "true"
    )

    # GPU pricing estimates (hourly USD) - used when Billing API unavailable
    gpu_prices: dict[str, float] = field(
        default_factory=lambda: {
            "NVIDIA_TESLA_T4": 0.35,
            "NVIDIA_TESLA_T4_SPOT": 0.11,
            "NVIDIA_L4": 0.81,
            "NVIDIA_L4_SPOT": 0.24,
            "NVIDIA_TESLA_V100": 2.48,
            "NVIDIA_TESLA_V100_SPOT": 0.74,
            "NVIDIA_TESLA_A100": 3.67,
            "NVIDIA_TESLA_A100_SPOT": 1.10,
        }
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigurationError: If values are invalid.
        """
        if self.daily_budget_usd <= 0:
            raise ConfigurationError(
                "Daily budget must be positive",
                details={"daily_budget_usd": self.daily_budget_usd},
            )
        if self.total_budget_usd <= 0:
            raise ConfigurationError(
                "Total budget must be positive",
                details={"total_budget_usd": self.total_budget_usd},
            )
        if not 0 < self.alert_threshold < 1:
            raise ConfigurationError(
                "Alert threshold must be between 0 and 1",
                details={"alert_threshold": self.alert_threshold},
            )
        if not 0 < self.critical_threshold <= 1:
            raise ConfigurationError(
                "Critical threshold must be between 0 and 1",
                details={"critical_threshold": self.critical_threshold},
            )
        if self.alert_threshold >= self.critical_threshold:
            raise ConfigurationError(
                "Alert threshold must be less than critical threshold",
                details={
                    "alert_threshold": self.alert_threshold,
                    "critical_threshold": self.critical_threshold,
                },
            )

    @classmethod
    def from_env(cls) -> CostMonitorConfig:
        """Create configuration from environment variables.

        Returns:
            CostMonitorConfig instance with values from environment.
        """
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> CostMonitorConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            CostMonitorConfig instance.

        Raises:
            ConfigurationError: If file is invalid or missing.
        """
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e

        return cls.from_dict(data.get("cost_monitor", data))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostMonitorConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            CostMonitorConfig instance.
        """
        # Extract gpu_prices separately to handle nested dict
        gpu_prices = data.pop("gpu_prices", None)
        config = cls(**{k: v for k, v in data.items() if k != "gpu_prices"})
        if gpu_prices:
            config.gpu_prices = gpu_prices
        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "daily_budget_usd": self.daily_budget_usd,
            "total_budget_usd": self.total_budget_usd,
            "alert_threshold": self.alert_threshold,
            "critical_threshold": self.critical_threshold,
            "check_interval_seconds": self.check_interval_seconds,
            "billing_api_enabled": self.billing_api_enabled,
            "gpu_prices": self.gpu_prices,
        }


class CostMonitor:
    """Monitors cloud costs and enforces budget limits.

    Provides real-time cost tracking, budget enforcement, and alerting
    for GCP Vertex AI training jobs. Supports both GCP Billing API
    (when available) and estimation based on job configurations.

    Usage:
        config = CostMonitorConfig()
        monitor = CostMonitor(config)

        # Check before launching jobs
        if monitor.can_afford(estimated_cost=5.0):
            launch_job()

        # Periodic monitoring
        status = monitor.get_cost_status()
        if status.alert_level == AlertLevel.CRITICAL:
            stop_all_jobs()
    """

    def __init__(self, config: CostMonitorConfig | None = None) -> None:
        """Initialize cost monitor.

        Args:
            config: Cost monitor configuration. Uses default if None.
        """
        self.config = config or CostMonitorConfig()

        # Cost tracking state
        self._costs_by_gpu: dict[str, float] = {}
        self._costs_by_job: dict[str, float] = {}
        self._total_cost_usd: float = 0.0
        self._daily_cost_usd: float = 0.0
        self._tracking_start_time: float = time.time()
        self._current_day_start: float = self._get_day_start()

        # Alert callbacks
        self._alert_callbacks: list[Callable[[AlertLevel, CostStatus], None]] = []
        self._last_alert_level: AlertLevel = AlertLevel.NORMAL

        # Background monitoring
        self._monitoring: bool = False
        self._monitor_thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()
        self._lock: threading.Lock = threading.Lock()

        logger.info(
            f"CostMonitor initialized: daily_budget=${self.config.daily_budget_usd:.2f}, "
            f"total_budget=${self.config.total_budget_usd:.2f}"
        )

    def _get_day_start(self) -> float:
        """Get timestamp for start of current day (UTC)."""
        now = datetime.utcnow()
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return day_start.timestamp()

    def _reset_daily_if_needed(self) -> None:
        """Reset daily cost tracking if a new day has started."""
        current_day = self._get_day_start()
        if current_day > self._current_day_start:
            with self._lock:
                logger.info(
                    f"New day started, resetting daily cost "
                    f"(previous: ${self._daily_cost_usd:.2f})"
                )
                self._daily_cost_usd = 0.0
                self._current_day_start = current_day

    def record_cost(
        self,
        amount_usd: float,
        gpu_type: str | None = None,
        job_type: str | None = None,
    ) -> None:
        """Record a cost incurred.

        Args:
            amount_usd: Cost amount in USD.
            gpu_type: GPU type (e.g., NVIDIA_TESLA_T4).
            job_type: Job type (e.g., selfplay, training).
        """
        self._reset_daily_if_needed()

        with self._lock:
            self._total_cost_usd += amount_usd
            self._daily_cost_usd += amount_usd

            if gpu_type:
                self._costs_by_gpu[gpu_type] = (
                    self._costs_by_gpu.get(gpu_type, 0.0) + amount_usd
                )
            if job_type:
                self._costs_by_job[job_type] = (
                    self._costs_by_job.get(job_type, 0.0) + amount_usd
                )

        logger.debug(
            f"Recorded cost: ${amount_usd:.4f} "
            f"(gpu={gpu_type}, job={job_type}, "
            f"daily_total=${self._daily_cost_usd:.2f})"
        )

        # Check for alerts
        self._check_alerts()

    def estimate_job_cost(
        self,
        gpu_type: str,
        duration_hours: float,
        gpu_count: int = 1,
        use_spot: bool = False,
    ) -> float:
        """Estimate cost for a job.

        Args:
            gpu_type: GPU type (e.g., NVIDIA_TESLA_T4).
            duration_hours: Expected job duration in hours.
            gpu_count: Number of GPUs.
            use_spot: Whether using spot/preemptible VMs.

        Returns:
            Estimated cost in USD.
        """
        price_key = f"{gpu_type}_SPOT" if use_spot else gpu_type
        hourly_rate = self.config.gpu_prices.get(
            price_key, self.config.gpu_prices.get(gpu_type)
        )

        if hourly_rate is None:
            logger.warning(
                f"GPU price for '{gpu_type}' (spot={use_spot}) not found in config. "
                f"Defaulting to $1.00/hr. Cost estimation may be inaccurate."
            )
            hourly_rate = 1.0

        return hourly_rate * duration_hours * gpu_count

    def get_current_costs(self) -> CostBreakdown:
        """Get current cost breakdown.

        Returns:
            CostBreakdown with current cost information.
        """
        self._reset_daily_if_needed()

        with self._lock:
            return CostBreakdown(
                total_today_usd=self._daily_cost_usd,
                total_project_usd=self._total_cost_usd,
                by_gpu_type=dict(self._costs_by_gpu),
                by_job_type=dict(self._costs_by_job),
                timestamp=datetime.utcnow().isoformat(),
            )

    def get_cost_status(self) -> CostStatus:
        """Get comprehensive cost status.

        Returns:
            CostStatus with current status and alerts.
        """
        breakdown = self.get_current_costs()

        # Calculate remaining budgets
        daily_remaining = max(0.0, self.config.daily_budget_usd - breakdown.total_today_usd)
        total_remaining = max(0.0, self.config.total_budget_usd - breakdown.total_project_usd)

        # Calculate estimated daily rate
        elapsed_hours = (time.time() - self._tracking_start_time) / 3600.0
        estimated_rate = (
            breakdown.total_project_usd / elapsed_hours * 24
            if elapsed_hours > 0
            else 0.0
        )

        # Determine alert level
        alert_level = self._calculate_alert_level(breakdown)

        return CostStatus(
            breakdown=breakdown,
            alert_level=alert_level,
            daily_remaining_usd=daily_remaining,
            total_remaining_usd=total_remaining,
            estimated_daily_rate_usd=estimated_rate,
        )

    def _calculate_alert_level(self, breakdown: CostBreakdown) -> AlertLevel:
        """Calculate alert level based on cost breakdown.

        Args:
            breakdown: Current cost breakdown.

        Returns:
            Appropriate alert level.
        """
        # Check daily budget
        daily_usage = breakdown.total_today_usd / self.config.daily_budget_usd
        # Check total budget
        total_usage = breakdown.total_project_usd / self.config.total_budget_usd

        max_usage = max(daily_usage, total_usage)

        if max_usage >= 1.0:
            return AlertLevel.BUDGET_EXCEEDED
        elif max_usage >= self.config.critical_threshold:
            return AlertLevel.CRITICAL
        elif max_usage >= self.config.alert_threshold:
            return AlertLevel.WARNING
        else:
            return AlertLevel.NORMAL

    def can_afford(self, estimated_cost: float) -> bool:
        """Check if estimated cost is within budget.

        Args:
            estimated_cost: Estimated cost in USD.

        Returns:
            True if cost is affordable within both daily and total budgets.
        """
        status = self.get_cost_status()

        # Check both daily and total remaining
        can_afford_daily = estimated_cost <= status.daily_remaining_usd
        can_afford_total = estimated_cost <= status.total_remaining_usd

        if not can_afford_daily:
            logger.warning(
                f"Cannot afford ${estimated_cost:.2f}: "
                f"daily remaining=${status.daily_remaining_usd:.2f}"
            )
        if not can_afford_total:
            logger.warning(
                f"Cannot afford ${estimated_cost:.2f}: "
                f"total remaining=${status.total_remaining_usd:.2f}"
            )

        return can_afford_daily and can_afford_total

    def check_budget(self) -> None:
        """Check budget and raise if exceeded.

        Raises:
            BudgetExceededError: If budget is exceeded.
        """
        status = self.get_cost_status()

        if status.alert_level == AlertLevel.BUDGET_EXCEEDED:
            breakdown = status.breakdown
            if breakdown.total_today_usd >= self.config.daily_budget_usd:
                raise BudgetExceededError(
                    "Daily budget exceeded",
                    budget_type="daily",
                    budget_limit=self.config.daily_budget_usd,
                    current_cost=breakdown.total_today_usd,
                )
            else:
                raise BudgetExceededError(
                    "Total budget exceeded",
                    budget_type="total",
                    budget_limit=self.config.total_budget_usd,
                    current_cost=breakdown.total_project_usd,
                )

    def register_alert_callback(
        self,
        callback: Callable[[AlertLevel, CostStatus], None],
    ) -> None:
        """Register callback for cost alerts.

        Callback is called when alert level changes.

        Args:
            callback: Function to call with (alert_level, cost_status).
        """
        self._alert_callbacks.append(callback)
        logger.debug(f"Registered alert callback: {callback}")

    def _check_alerts(self) -> None:
        """Check for alert conditions and notify callbacks."""
        status = self.get_cost_status()

        if status.alert_level != self._last_alert_level:
            old_level = self._last_alert_level
            self._last_alert_level = status.alert_level

            logger.info(
                f"Cost alert level changed: {old_level.value} -> {status.alert_level.value}"
            )

            for callback in self._alert_callbacks:
                try:
                    callback(status.alert_level, status)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

    def start_monitoring(self) -> None:
        """Start background cost monitoring thread."""
        if self._monitoring:
            logger.warning("Cost monitoring already running")
            return

        self._stop_event.clear()
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="CostMonitor",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info(
            f"Started cost monitoring "
            f"(interval={self.config.check_interval_seconds}s)"
        )

    def stop_monitoring(self) -> None:
        """Stop background cost monitoring thread."""
        if not self._monitoring:
            return

        self._stop_event.set()
        self._monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

        logger.info("Stopped cost monitoring")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Check and log status periodically
                status = self.get_cost_status()
                logger.info(
                    f"Cost status: daily=${status.breakdown.total_today_usd:.2f}/"
                    f"${self.config.daily_budget_usd:.2f}, "
                    f"total=${status.breakdown.total_project_usd:.2f}/"
                    f"${self.config.total_budget_usd:.2f}, "
                    f"alert={status.alert_level.value}"
                )

                # Check alerts
                self._check_alerts()

            except Exception as e:
                logger.error(f"Error in cost monitoring loop: {e}")

            # Wait for next check
            self._stop_event.wait(timeout=self.config.check_interval_seconds)

    def get_statistics(self) -> dict[str, Any]:
        """Get cost monitoring statistics for logging.

        Returns:
            Dictionary with statistics.
        """
        status = self.get_cost_status()
        elapsed_hours = (time.time() - self._tracking_start_time) / 3600.0

        return {
            "daily_cost_usd": status.breakdown.total_today_usd,
            "total_cost_usd": status.breakdown.total_project_usd,
            "daily_budget_usd": self.config.daily_budget_usd,
            "total_budget_usd": self.config.total_budget_usd,
            "daily_usage_pct": status.breakdown.total_today_usd
            / self.config.daily_budget_usd
            * 100,
            "total_usage_pct": status.breakdown.total_project_usd
            / self.config.total_budget_usd
            * 100,
            "alert_level": status.alert_level.value,
            "estimated_daily_rate_usd": status.estimated_daily_rate_usd,
            "tracking_elapsed_hours": elapsed_hours,
            "costs_by_gpu": status.breakdown.by_gpu_type,
            "costs_by_job": status.breakdown.by_job_type,
        }

    def reset(self) -> None:
        """Reset all cost tracking state.

        Use with caution - this clears all accumulated cost data.
        """
        with self._lock:
            self._costs_by_gpu.clear()
            self._costs_by_job.clear()
            self._total_cost_usd = 0.0
            self._daily_cost_usd = 0.0
            self._tracking_start_time = time.time()
            self._current_day_start = self._get_day_start()
            self._last_alert_level = AlertLevel.NORMAL

        logger.info("Cost tracking reset")

    def __repr__(self) -> str:
        return (
            f"CostMonitor(daily_budget=${self.config.daily_budget_usd:.2f}, "
            f"total_budget=${self.config.total_budget_usd:.2f}, "
            f"current_daily=${self._daily_cost_usd:.2f}, "
            f"current_total=${self._total_cost_usd:.2f})"
        )
