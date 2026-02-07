"""
Idle Instance Manager for GCP Cloud Training.

Provides automatic detection and termination of idle compute instances
to achieve 100% cost savings when not actively training.
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
from typing import TYPE_CHECKING, Any

import yaml

from alpha_zero.cloud.gcp.exceptions import ConfigurationError

if TYPE_CHECKING:
    from alpha_zero.cloud.gcp.preemption_handler import PreemptionHandler

logger = logging.getLogger(__name__)


class IdleAction(str, Enum):
    """Actions for idle instances."""

    NONE = "none"
    WARN = "warn"
    CHECKPOINT = "checkpoint"
    STOP = "stop"


@dataclass
class IdlePolicy:
    """Policy for handling idle state of a specific job type.

    Attributes:
        job_type: Type of job (selfplay, training, evaluation).
        auto_stop: Whether to automatically stop idle instances.
        idle_threshold_minutes: Minutes before considered idle.
        checkpoint_before_stop: Whether to checkpoint before stopping.
        notify_before_stop: Whether to notify before stopping.
        min_instances: Minimum instances to keep running.
    """

    job_type: str
    auto_stop: bool = True
    idle_threshold_minutes: int = 15
    checkpoint_before_stop: bool = True
    notify_before_stop: bool = True
    min_instances: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_type": self.job_type,
            "auto_stop": self.auto_stop,
            "idle_threshold_minutes": self.idle_threshold_minutes,
            "checkpoint_before_stop": self.checkpoint_before_stop,
            "notify_before_stop": self.notify_before_stop,
            "min_instances": self.min_instances,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdlePolicy:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class IdleStatus:
    """Idle status for a job.

    Attributes:
        job_name: Name of the job.
        job_type: Type of job.
        is_idle: Whether the job is considered idle.
        idle_duration_seconds: Duration of idle state.
        last_activity_time: Timestamp of last activity.
        gpu_utilization: GPU utilization percentage (0-1).
        should_stop: Whether the job should be stopped.
        reason: Human-readable reason for status.
    """

    job_name: str
    job_type: str
    is_idle: bool
    idle_duration_seconds: float
    last_activity_time: str
    gpu_utilization: float | None
    should_stop: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_name": self.job_name,
            "job_type": self.job_type,
            "is_idle": self.is_idle,
            "idle_duration_seconds": self.idle_duration_seconds,
            "last_activity_time": self.last_activity_time,
            "gpu_utilization": self.gpu_utilization,
            "should_stop": self.should_stop,
            "reason": self.reason,
        }


@dataclass
class IdleInstanceConfig:
    """Configuration for idle instance detection and management.

    All values can be overridden via environment variables with the CIV6AZ_ prefix.

    Environment Variables:
        CIV6AZ_IDLE_THRESHOLD_MINUTES: Minutes before considered idle (default: 15)
        CIV6AZ_IDLE_GPU_UTILIZATION_THRESHOLD: GPU util below this is idle (default: 0.1)
        CIV6AZ_IDLE_CHECK_INTERVAL_S: Check interval (default: 60)
        CIV6AZ_IDLE_AUTO_STOP_ENABLED: Enable auto-stop (default: true)
        CIV6AZ_IDLE_GRACE_PERIOD_S: Grace period before stopping (default: 300)
    """

    idle_threshold_minutes: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_IDLE_THRESHOLD_MINUTES", "15"))
    )
    gpu_utilization_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("CIV6AZ_IDLE_GPU_UTILIZATION_THRESHOLD", "0.1")
        )
    )
    check_interval_seconds: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_IDLE_CHECK_INTERVAL_S", "60"))
    )
    auto_stop_enabled: bool = field(
        default_factory=lambda: os.environ.get("CIV6AZ_IDLE_AUTO_STOP_ENABLED", "true").lower()
        == "true"
    )
    grace_period_seconds: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_IDLE_GRACE_PERIOD_S", "300"))
    )

    # Checkpoint before stop
    checkpoint_before_stop: bool = field(
        default_factory=lambda: os.environ.get("CIV6AZ_IDLE_CHECKPOINT_BEFORE_STOP", "true").lower()
        == "true"
    )

    # Minimum instances to keep
    min_instances: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_IDLE_MIN_INSTANCES", "0"))
    )

    # Default policies for different job types
    policies: dict[str, IdlePolicy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize default policies and validate configuration."""
        # Set up default policies if not provided
        if not self.policies:
            self.policies = {
                "selfplay": IdlePolicy(
                    job_type="selfplay",
                    auto_stop=True,
                    idle_threshold_minutes=self.idle_threshold_minutes,
                    checkpoint_before_stop=False,  # Self-play is stateless
                ),
                "training": IdlePolicy(
                    job_type="training",
                    auto_stop=True,
                    idle_threshold_minutes=self.idle_threshold_minutes,
                    checkpoint_before_stop=True,
                ),
                "evaluation": IdlePolicy(
                    job_type="evaluation",
                    auto_stop=True,
                    idle_threshold_minutes=self.idle_threshold_minutes,
                    checkpoint_before_stop=False,
                ),
            }
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigurationError: If values are invalid.
        """
        if self.idle_threshold_minutes < 1:
            raise ConfigurationError(
                "idle_threshold_minutes must be at least 1",
                details={"idle_threshold_minutes": self.idle_threshold_minutes},
            )
        if not 0 <= self.gpu_utilization_threshold <= 1:
            raise ConfigurationError(
                "gpu_utilization_threshold must be between 0 and 1",
                details={"gpu_utilization_threshold": self.gpu_utilization_threshold},
            )
        if self.check_interval_seconds < 1:
            raise ConfigurationError(
                "check_interval_seconds must be at least 1",
                details={"check_interval_seconds": self.check_interval_seconds},
            )
        if self.grace_period_seconds < 0:
            raise ConfigurationError(
                "grace_period_seconds must be non-negative",
                details={"grace_period_seconds": self.grace_period_seconds},
            )

    @classmethod
    def from_env(cls) -> IdleInstanceConfig:
        """Create configuration from environment variables.

        Returns:
            IdleInstanceConfig instance with values from environment.
        """
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> IdleInstanceConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            IdleInstanceConfig instance.

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

        return cls.from_dict(data.get("idle_manager", data))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdleInstanceConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            IdleInstanceConfig instance.
        """
        # Handle policies separately
        policies_data = data.pop("policies", None)
        config = cls(**{k: v for k, v in data.items() if k != "policies"})

        if policies_data:
            config.policies = {k: IdlePolicy.from_dict(v) for k, v in policies_data.items()}

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "idle_threshold_minutes": self.idle_threshold_minutes,
            "gpu_utilization_threshold": self.gpu_utilization_threshold,
            "check_interval_seconds": self.check_interval_seconds,
            "auto_stop_enabled": self.auto_stop_enabled,
            "grace_period_seconds": self.grace_period_seconds,
            "checkpoint_before_stop": self.checkpoint_before_stop,
            "min_instances": self.min_instances,
            "policies": {k: v.to_dict() for k, v in self.policies.items()},
        }

    def get_policy(self, job_type: str) -> IdlePolicy:
        """Get policy for a job type.

        Args:
            job_type: Type of job.

        Returns:
            IdlePolicy for the job type.
        """
        return self.policies.get(
            job_type,
            IdlePolicy(
                job_type=job_type,
                idle_threshold_minutes=self.idle_threshold_minutes,
            ),
        )


class IdleInstanceManager:
    """Manages idle instance detection and auto-stop.

    Monitors jobs for idle state based on activity timestamps and
    GPU utilization. Implements grace periods and checkpoint-before-stop.

    Usage:
        config = IdleInstanceConfig()
        manager = IdleInstanceManager(config)

        # Register job activity
        manager.mark_active("job-123", job_type="training")

        # Start monitoring
        manager.start_monitoring()

        # Check status
        status = manager.check_job_idle("job-123")
        if status.should_stop:
            manager.stop_job("job-123")

        # Cleanup
        manager.stop_monitoring()
    """

    def __init__(
        self,
        config: IdleInstanceConfig | None = None,
        preemption_handler: PreemptionHandler | None = None,
    ) -> None:
        """Initialize idle manager.

        Args:
            config: Idle manager configuration. Uses default if None.
            preemption_handler: Optional handler for checkpoint before stop.
        """
        self.config = config or IdleInstanceConfig()
        self.preemption_handler = preemption_handler

        # Job tracking
        self._jobs: dict[str, dict[str, Any]] = {}
        self._grace_period_starts: dict[str, float] = {}
        self._stopped_jobs: set[str] = set()

        # Callbacks
        self._stop_callbacks: list[Callable[[str, IdleStatus], None]] = []
        self._warning_callbacks: list[Callable[[str, IdleStatus], None]] = []

        # Background monitoring
        self._monitoring: bool = False
        self._monitor_thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()
        self._lock: threading.Lock = threading.Lock()

        logger.info(
            f"IdleInstanceManager initialized: "
            f"threshold={self.config.idle_threshold_minutes}min, "
            f"grace_period={self.config.grace_period_seconds}s"
        )

    def register_job(
        self,
        job_name: str,
        job_type: str = "training",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a job for idle tracking.

        Args:
            job_name: Unique name of the job.
            job_type: Type of job (selfplay, training, evaluation).
            metadata: Optional job metadata.
        """
        with self._lock:
            self._jobs[job_name] = {
                "job_type": job_type,
                "last_activity": time.time(),
                "gpu_utilization": None,
                "metadata": metadata or {},
                "registered_time": time.time(),
            }
        logger.debug(f"Registered job: {job_name} (type={job_type})")

    def unregister_job(self, job_name: str) -> None:
        """Unregister a job from idle tracking.

        Args:
            job_name: Name of the job to unregister.
        """
        with self._lock:
            self._jobs.pop(job_name, None)
            self._grace_period_starts.pop(job_name, None)
            self._stopped_jobs.discard(job_name)
        logger.debug(f"Unregistered job: {job_name}")

    def mark_active(
        self,
        job_name: str,
        job_type: str | None = None,
        gpu_utilization: float | None = None,
    ) -> None:
        """Mark a job as active.

        Args:
            job_name: Name of the job.
            job_type: Optional job type (registers if not exists).
            gpu_utilization: Optional GPU utilization (0-1).
        """
        with self._lock:
            if job_name not in self._jobs:
                if job_type:
                    self._jobs[job_name] = {
                        "job_type": job_type,
                        "last_activity": time.time(),
                        "gpu_utilization": gpu_utilization,
                        "metadata": {},
                        "registered_time": time.time(),
                    }
                else:
                    logger.warning(f"Unknown job: {job_name}")
                    return
            else:
                self._jobs[job_name]["last_activity"] = time.time()
                if gpu_utilization is not None:
                    self._jobs[job_name]["gpu_utilization"] = gpu_utilization

            # Clear grace period if active
            self._grace_period_starts.pop(job_name, None)

        logger.debug(f"Job marked active: {job_name}")

    def check_job_idle(self, job_name: str) -> IdleStatus:
        """Check if a specific job is idle.

        Args:
            job_name: Name of the job.

        Returns:
            IdleStatus for the job.
        """
        with self._lock:
            if job_name not in self._jobs:
                return IdleStatus(
                    job_name=job_name,
                    job_type="unknown",
                    is_idle=False,
                    idle_duration_seconds=0,
                    last_activity_time="",
                    gpu_utilization=None,
                    should_stop=False,
                    reason="Job not found",
                )

            job = self._jobs[job_name]
            job_type = job["job_type"]
            policy = self.config.get_policy(job_type)

            # Calculate idle duration
            last_activity = job["last_activity"]
            idle_duration = time.time() - last_activity
            idle_threshold = policy.idle_threshold_minutes * 60

            # Check GPU utilization
            gpu_util = job.get("gpu_utilization")
            is_gpu_idle = gpu_util is not None and gpu_util < self.config.gpu_utilization_threshold

            # Determine if idle
            is_idle = idle_duration >= idle_threshold or is_gpu_idle

            # Determine if should stop
            should_stop = False
            reason = "Active"

            if is_idle:
                reason = f"Idle for {idle_duration / 60:.1f} minutes"
                if is_gpu_idle:
                    reason += f" (GPU util: {gpu_util:.1%})"

                # Check if auto-stop is enabled
                if self.config.auto_stop_enabled and policy.auto_stop:
                    # Check grace period
                    grace_start = self._grace_period_starts.get(job_name)
                    if grace_start is None:
                        # Start grace period
                        self._grace_period_starts[job_name] = time.time()
                        reason += f"; grace period started ({self.config.grace_period_seconds}s)"
                    else:
                        grace_elapsed = time.time() - grace_start
                        if grace_elapsed >= self.config.grace_period_seconds:
                            should_stop = True
                            reason += "; grace period expired"
                        else:
                            remaining = self.config.grace_period_seconds - grace_elapsed
                            reason += f"; grace period: {remaining:.0f}s remaining"

            return IdleStatus(
                job_name=job_name,
                job_type=job_type,
                is_idle=is_idle,
                idle_duration_seconds=idle_duration,
                last_activity_time=datetime.fromtimestamp(last_activity).isoformat(),
                gpu_utilization=gpu_util,
                should_stop=should_stop,
                reason=reason,
            )

    def get_all_idle_jobs(self) -> list[IdleStatus]:
        """Get all idle jobs.

        Returns:
            List of IdleStatus for idle jobs.
        """
        idle_jobs = []
        with self._lock:
            job_names = list(self._jobs.keys())

        for job_name in job_names:
            status = self.check_job_idle(job_name)
            if status.is_idle:
                idle_jobs.append(status)

        return idle_jobs

    def stop_idle_jobs(self, dry_run: bool = False) -> list[str]:
        """Stop all idle jobs that should be stopped.

        Args:
            dry_run: If True, don't actually stop jobs.

        Returns:
            List of job names that were (or would be) stopped.
        """
        stopped = []
        with self._lock:
            job_names = list(self._jobs.keys())

        for job_name in job_names:
            status = self.check_job_idle(job_name)
            if status.should_stop:
                if dry_run:
                    logger.info(f"[DRY RUN] Would stop job: {job_name}")
                else:
                    try:
                        self._stop_job(job_name, status)
                    except Exception as e:
                        logger.error(f"Failed to stop job {job_name}: {e}")
                        continue
                stopped.append(job_name)

        return stopped

    def _stop_job(self, job_name: str, status: IdleStatus) -> bool:
        """Stop a job with proper handling.

        Args:
            job_name: Name of the job to stop.
            status: Current idle status.

        Returns:
            True if job was successfully stopped.
        """
        policy = self.config.get_policy(status.job_type)

        # Checkpoint before stop if configured
        if policy.checkpoint_before_stop and self.preemption_handler:
            logger.info(f"Creating checkpoint before stopping job: {job_name}")
            try:
                self.preemption_handler.set_job_info(job_name, status.job_type)
                self.preemption_handler.handle_preemption()
            except Exception as e:
                logger.warning(f"Checkpoint before stop failed: {e}")

        # Notify callbacks
        for callback in self._stop_callbacks:
            try:
                callback(job_name, status)
            except Exception as e:
                logger.error(f"Stop callback failed: {e}")

        # Mark as stopped
        with self._lock:
            self._stopped_jobs.add(job_name)
            self._jobs.pop(job_name, None)
            self._grace_period_starts.pop(job_name, None)

        logger.info(
            f"Stopped idle job: {job_name} (idle for {status.idle_duration_seconds / 60:.1f} min)"
        )

        return True

    def register_stop_callback(
        self,
        callback: Callable[[str, IdleStatus], None],
    ) -> None:
        """Register callback for when jobs are stopped.

        Args:
            callback: Function to call with (job_name, status).
        """
        self._stop_callbacks.append(callback)
        logger.debug(f"Registered stop callback: {callback}")

    def register_warning_callback(
        self,
        callback: Callable[[str, IdleStatus], None],
    ) -> None:
        """Register callback for idle warnings.

        Args:
            callback: Function to call with (job_name, status).
        """
        self._warning_callbacks.append(callback)
        logger.debug(f"Registered warning callback: {callback}")

    def start_monitoring(self) -> None:
        """Start background idle monitoring thread."""
        if self._monitoring:
            logger.warning("Idle monitoring already running")
            return

        self._stop_event.clear()
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="IdleMonitor",
            daemon=True,
        )
        self._monitor_thread.start()
        logger.info(f"Started idle monitoring (interval={self.config.check_interval_seconds}s)")

    def stop_monitoring(self) -> None:
        """Stop background idle monitoring thread."""
        if not self._monitoring:
            return

        self._stop_event.set()
        self._monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

        logger.info("Stopped idle monitoring")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_event.is_set():
            try:
                # Check all jobs
                idle_jobs = self.get_all_idle_jobs()

                if idle_jobs:
                    logger.debug(f"Found {len(idle_jobs)} idle jobs")

                    # Issue warnings for jobs in grace period
                    for status in idle_jobs:
                        if not status.should_stop:
                            # Still in grace period
                            for callback in self._warning_callbacks:
                                try:
                                    callback(status.job_name, status)
                                except Exception as e:
                                    logger.error(f"Warning callback failed: {e}")

                # Stop jobs that should be stopped
                if self.config.auto_stop_enabled:
                    try:
                        self.stop_idle_jobs()
                    except Exception as e:
                        logger.error(f"Error stopping idle jobs: {e}")

            except Exception as e:
                logger.error(f"Error in idle monitoring loop: {e}")

            # Wait for next check
            self._stop_event.wait(timeout=self.config.check_interval_seconds)

    def get_statistics(self) -> dict[str, Any]:
        """Get idle manager statistics.

        Returns:
            Dictionary with manager statistics.
        """
        with self._lock:
            job_count = len(self._jobs)
            grace_count = len(self._grace_period_starts)
            stopped_count = len(self._stopped_jobs)

        idle_jobs = self.get_all_idle_jobs()

        return {
            "registered_jobs": job_count,
            "idle_jobs": len(idle_jobs),
            "in_grace_period": grace_count,
            "stopped_jobs": stopped_count,
            "auto_stop_enabled": self.config.auto_stop_enabled,
            "monitoring_active": self._monitoring,
            "config": self.config.to_dict(),
        }

    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            self._jobs.clear()
            self._grace_period_starts.clear()
            self._stopped_jobs.clear()
        logger.info("Idle manager state reset")

    def __repr__(self) -> str:
        return (
            f"IdleInstanceManager("
            f"threshold={self.config.idle_threshold_minutes}min, "
            f"grace={self.config.grace_period_seconds}s, "
            f"jobs={len(self._jobs)})"
        )
