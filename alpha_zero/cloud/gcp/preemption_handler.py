"""
Preemption Handler for GCP Spot VM Training.

Provides automatic checkpointing and recovery from Spot VM preemptions
with configurable checkpoint intervals and signal handling.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml

from alpha_zero.cloud.gcp.exceptions import ConfigurationError, PreemptionError

if TYPE_CHECKING:
    from alpha_zero.cloud.gcp.storage import GCSCheckpointStorage

logger = logging.getLogger(__name__)


@dataclass
class PreemptionState:
    """State saved during preemption for recovery.

    Attributes:
        job_name: Name of the preempted job.
        job_type: Type of job (selfplay, training).
        iteration: Training iteration at preemption.
        checkpoint_path: Path to saved checkpoint.
        preemption_time: ISO timestamp of preemption.
        retry_count: Number of retry attempts.
        metadata: Additional job-specific metadata.
    """

    job_name: str
    job_type: str
    iteration: int
    checkpoint_path: str
    preemption_time: str
    retry_count: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_name": self.job_name,
            "job_type": self.job_type,
            "iteration": self.iteration,
            "checkpoint_path": self.checkpoint_path,
            "preemption_time": self.preemption_time,
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreemptionState:
        """Create from dictionary."""
        return cls(
            job_name=data["job_name"],
            job_type=data["job_type"],
            iteration=data["iteration"],
            checkpoint_path=data["checkpoint_path"],
            preemption_time=data["preemption_time"],
            retry_count=data["retry_count"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class PreemptionHandlerConfig:
    """Configuration for preemption handling and checkpointing.

    All values can be overridden via environment variables with the CIV6AZ_ prefix.

    Environment Variables:
        CIV6AZ_CHECKPOINT_INTERVAL_S: Checkpoint interval in seconds (default: 300 = 5 min)
        CIV6AZ_CHECKPOINT_MIN_INTERVAL_S: Minimum between checkpoints (default: 60)
        CIV6AZ_PREEMPTION_MAX_RETRIES: Max auto-resume attempts (default: 5)
        CIV6AZ_PREEMPTION_RETRY_DELAY_S: Delay between retries (default: 60)
        CIV6AZ_PREEMPTION_BACKOFF_MULTIPLIER: Exponential backoff (default: 2.0)
        CIV6AZ_PREEMPTION_MAX_BACKOFF_S: Maximum backoff delay (default: 3600)
    """

    checkpoint_interval_seconds: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_CHECKPOINT_INTERVAL_S", "300"))
    )
    min_checkpoint_interval_seconds: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_CHECKPOINT_MIN_INTERVAL_S", "60"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_PREEMPTION_MAX_RETRIES", "5"))
    )
    retry_delay_seconds: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_PREEMPTION_RETRY_DELAY_S", "60"))
    )
    backoff_multiplier: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_PREEMPTION_BACKOFF_MULTIPLIER", "2.0"))
    )
    max_backoff_seconds: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_PREEMPTION_MAX_BACKOFF_S", "3600"))
    )

    # Signal handling
    handle_sigterm: bool = field(
        default_factory=lambda: os.environ.get("CIV6AZ_PREEMPTION_HANDLE_SIGTERM", "true").lower()
        == "true"
    )
    checkpoint_on_signal: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_PREEMPTION_CHECKPOINT_ON_SIGNAL", "true"
        ).lower()
        == "true"
    )

    # State preservation options
    preserve_optimizer_state: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_PREEMPTION_PRESERVE_OPTIMIZER", "true"
        ).lower()
        == "true"
    )
    preserve_scheduler_state: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_PREEMPTION_PRESERVE_SCHEDULER", "true"
        ).lower()
        == "true"
    )

    # GCS paths for preemption state
    preemption_state_prefix: str = field(
        default_factory=lambda: os.environ.get("CIV6AZ_PREEMPTION_STATE_PREFIX", "preemption_state")
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigurationError: If values are invalid.
        """
        if self.checkpoint_interval_seconds < 1:
            raise ConfigurationError(
                "checkpoint_interval_seconds must be at least 1",
                details={"checkpoint_interval_seconds": self.checkpoint_interval_seconds},
            )
        if self.min_checkpoint_interval_seconds < 1:
            raise ConfigurationError(
                "min_checkpoint_interval_seconds must be at least 1",
                details={"min_checkpoint_interval_seconds": self.min_checkpoint_interval_seconds},
            )
        if self.max_retries < 0:
            raise ConfigurationError(
                "max_retries must be non-negative",
                details={"max_retries": self.max_retries},
            )
        if self.backoff_multiplier < 1.0:
            raise ConfigurationError(
                "backoff_multiplier must be at least 1.0",
                details={"backoff_multiplier": self.backoff_multiplier},
            )

    @classmethod
    def from_env(cls) -> PreemptionHandlerConfig:
        """Create configuration from environment variables.

        Returns:
            PreemptionHandlerConfig instance with values from environment.
        """
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> PreemptionHandlerConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            PreemptionHandlerConfig instance.

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

        return cls.from_dict(data.get("preemption_handler", data))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreemptionHandlerConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            PreemptionHandlerConfig instance.
        """
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "checkpoint_interval_seconds": self.checkpoint_interval_seconds,
            "min_checkpoint_interval_seconds": self.min_checkpoint_interval_seconds,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "backoff_multiplier": self.backoff_multiplier,
            "max_backoff_seconds": self.max_backoff_seconds,
            "handle_sigterm": self.handle_sigterm,
            "checkpoint_on_signal": self.checkpoint_on_signal,
            "preserve_optimizer_state": self.preserve_optimizer_state,
            "preserve_scheduler_state": self.preserve_scheduler_state,
            "preemption_state_prefix": self.preemption_state_prefix,
        }


class PreemptionHandler:
    """Handles Spot VM preemptions with checkpoint-based recovery.

    Implements 5-minute (configurable) checkpoint intervals and automatic
    resume logic. Works with both self-play workers and training jobs.

    Features:
    - Periodic checkpoint creation
    - SIGTERM signal handling for preemption notice
    - Exponential backoff for retries
    - GCS-based state persistence

    Usage:
        config = PreemptionHandlerConfig()
        handler = PreemptionHandler(config)

        # In worker initialization
        handler.install_signal_handlers()
        handler.start_checkpoint_timer(model, optimizer, state_getter)

        # In training loop (optional manual checkpoints)
        if handler.should_checkpoint():
            handler.checkpoint_now(model, optimizer, iteration)

        # On job startup - check for resume
        resume_state = handler.get_resume_state(job_name)
        if resume_state:
            model.load_state_dict(torch.load(resume_state.checkpoint_path))
    """

    def __init__(
        self,
        config: PreemptionHandlerConfig | None = None,
        checkpoint_dir: str | Path | None = None,
        gcs_storage: GCSCheckpointStorage | None = None,
    ) -> None:
        """Initialize preemption handler.

        Args:
            config: Handler configuration. Uses default if None.
            checkpoint_dir: Local directory for checkpoints.
            gcs_storage: Optional GCS storage for cloud checkpoints.
        """
        self.config = config or PreemptionHandlerConfig()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.gcs_storage = gcs_storage

        # Create local checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint timing
        self._last_checkpoint_time: float = 0.0
        self._checkpoint_count: int = 0

        # Timer state
        self._timer_running: bool = False
        self._timer_thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()

        # State for checkpointing
        self._model: torch.nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._state_getter: Callable[[], dict[str, Any]] | None = None

        # Job tracking
        self._job_name: str | None = None
        self._job_type: str = "unknown"
        self._current_iteration: int = 0

        # Signal handling
        self._original_sigterm_handler: Any = None
        self._preemption_detected: bool = False

        # Retry tracking
        self._retry_counts: dict[str, int] = {}

        # Thread safety
        self._lock: threading.Lock = threading.Lock()

        logger.info(
            f"PreemptionHandler initialized: "
            f"interval={self.config.checkpoint_interval_seconds}s, "
            f"max_retries={self.config.max_retries}"
        )

    def set_job_info(
        self,
        job_name: str,
        job_type: str = "training",
    ) -> None:
        """Set current job information.

        Args:
            job_name: Name of the current job.
            job_type: Type of job (selfplay, training).
        """
        self._job_name = job_name
        self._job_type = job_type
        logger.debug(f"Job info set: name={job_name}, type={job_type}")

    def install_signal_handlers(self) -> None:
        """Install SIGTERM and SIGINT handlers for preemption detection.

        Note: On Windows, only SIGINT is available.
        """
        if not self.config.handle_sigterm:
            logger.info("Signal handling disabled by configuration")
            return

        def signal_handler(signum: int, frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            logger.warning(f"Received {sig_name} signal - preemption detected")
            self._preemption_detected = True

            if self.config.checkpoint_on_signal:
                logger.info("Creating emergency checkpoint...")
                try:
                    self._emergency_checkpoint()
                except Exception as e:
                    logger.error(f"Emergency checkpoint failed: {e}")

            # Call original handler if it exists
            if self._original_sigterm_handler and callable(self._original_sigterm_handler):
                self._original_sigterm_handler(signum, frame)

        # Install handlers
        try:
            self._original_sigterm_handler = signal.signal(signal.SIGTERM, signal_handler)
            logger.info("Installed SIGTERM handler")
        except (ValueError, OSError) as e:
            # May fail on Windows or non-main thread
            logger.warning(f"Could not install SIGTERM handler: {e}")

        try:
            signal.signal(signal.SIGINT, signal_handler)
            logger.info("Installed SIGINT handler")
        except (ValueError, OSError) as e:
            logger.warning(f"Could not install SIGINT handler: {e}")

    def uninstall_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigterm_handler:
            try:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
                logger.debug("Restored original SIGTERM handler")
            except (ValueError, OSError) as e:
                logger.warning(f"Could not restore original SIGTERM handler: {e}")
            self._original_sigterm_handler = None

    def should_checkpoint(self) -> bool:
        """Check if it's time for a checkpoint.

        Returns:
            True if checkpoint interval has elapsed.
        """
        elapsed = time.time() - self._last_checkpoint_time
        return elapsed >= self.config.checkpoint_interval_seconds

    def _can_checkpoint(self) -> bool:
        """Check if minimum interval has passed since last checkpoint.

        Returns:
            True if minimum interval has elapsed.
        """
        if self._last_checkpoint_time == 0:
            return True
        elapsed = time.time() - self._last_checkpoint_time
        return elapsed >= self.config.min_checkpoint_interval_seconds

    def checkpoint_now(
        self,
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        iteration: int | None = None,
        extra: dict[str, Any] | None = None,
        reason: str = "periodic",
    ) -> str:
        """Create checkpoint immediately.

        Args:
            model: Model to checkpoint. If None, uses internal self._model.
            optimizer: Optional optimizer to save. If None, uses internal self._optimizer.
            iteration: Current training iteration. If None, uses internal self._current_iteration.
            extra: Additional data to include.
            reason: Reason for checkpoint.

        Returns:
            Path to saved checkpoint, or empty string if no model available.
        """
        # Use internal state if not provided
        model = model or self._model
        optimizer = optimizer or self._optimizer
        iteration = iteration if iteration is not None else self._current_iteration

        # Return empty string if no model available
        if model is None:
            logger.warning("No model available for checkpoint")
            return ""

        self._current_iteration = iteration

        checkpoint_data: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "iteration": iteration,
            "checkpoint_time": datetime.utcnow().isoformat(),
            "reason": reason,
            "checkpoint_number": self._checkpoint_count,
        }

        if optimizer and self.config.preserve_optimizer_state:
            checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()

        if extra:
            checkpoint_data["extra"] = extra

        # Generate checkpoint filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{timestamp}_iter_{iteration}.pt"
        local_path = self.checkpoint_dir / filename

        # Save locally
        torch.save(checkpoint_data, local_path)
        logger.info(f"Checkpoint saved: {local_path} (iteration={iteration}, reason={reason})")

        # Upload to GCS if configured
        gcs_path = None
        if self.gcs_storage:
            try:
                blob_path = f"checkpoints/{filename}"
                data = local_path.read_bytes()
                gcs_path = self.gcs_storage.backend.upload_blob(
                    self.gcs_storage.bucket,
                    blob_path,
                    data,
                )
                logger.info(f"Checkpoint uploaded to GCS: {gcs_path}")
            except Exception as e:
                logger.error(f"Failed to upload checkpoint to GCS: {e}")

        # Update tracking
        with self._lock:
            self._last_checkpoint_time = time.time()
            self._checkpoint_count += 1

        return gcs_path or str(local_path)

    def _emergency_checkpoint(self) -> str | None:
        """Create emergency checkpoint during preemption.

        Returns:
            Path to checkpoint or None if not possible.
        """
        if not self._model:
            logger.warning("No model registered for emergency checkpoint")
            return None

        if not self._can_checkpoint():
            logger.debug("Minimum checkpoint interval not elapsed, skipping")
            return None

        # Get current state if getter available
        extra = {}
        if self._state_getter:
            try:
                extra = self._state_getter()
            except Exception as e:
                logger.warning(f"State getter failed: {e}")

        try:
            return self.checkpoint_now(
                model=self._model,
                optimizer=self._optimizer,
                iteration=self._current_iteration,
                extra=extra,
                reason="preemption",
            )
        except Exception as e:
            logger.error(f"Emergency checkpoint failed: {e}")
            return None

    def start_checkpoint_timer(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        state_getter: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        """Start background checkpoint timer.

        Args:
            model: Model to checkpoint periodically.
            optimizer: Optional optimizer to include.
            state_getter: Optional callable returning current state.
        """
        if self._timer_running:
            logger.warning("Checkpoint timer already running")
            return

        self._model = model
        self._optimizer = optimizer
        self._state_getter = state_getter

        self._stop_event.clear()
        self._timer_running = True
        self._timer_thread = threading.Thread(
            target=self._checkpoint_loop,
            name="PreemptionCheckpoint",
            daemon=True,
        )
        self._timer_thread.start()

        logger.info(
            f"Started checkpoint timer (interval={self.config.checkpoint_interval_seconds}s)"
        )

    def stop_checkpoint_timer(self) -> None:
        """Stop background checkpoint timer."""
        if not self._timer_running:
            return

        self._stop_event.set()
        self._timer_running = False

        if self._timer_thread:
            self._timer_thread.join(timeout=5.0)
            self._timer_thread = None

        logger.info("Stopped checkpoint timer")

    def _checkpoint_loop(self) -> None:
        """Background checkpoint loop."""
        while not self._stop_event.is_set():
            # Wait for interval
            self._stop_event.wait(timeout=self.config.checkpoint_interval_seconds)

            if self._stop_event.is_set():
                break

            if self._preemption_detected:
                logger.info("Preemption detected, stopping checkpoint loop")
                break

            # Create checkpoint
            if self._model:
                try:
                    extra = {}
                    if self._state_getter:
                        extra = self._state_getter()

                    self.checkpoint_now(
                        model=self._model,
                        optimizer=self._optimizer,
                        iteration=self._current_iteration,
                        extra=extra,
                        reason="periodic",
                    )
                except Exception as e:
                    logger.error(f"Periodic checkpoint failed: {e}")

    def update_iteration(self, iteration: int) -> None:
        """Update current iteration for checkpoint tracking.

        Args:
            iteration: Current training iteration.
        """
        self._current_iteration = iteration

    def save_preemption_state(
        self,
        checkpoint_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save preemption state for recovery.

        Args:
            checkpoint_path: Path to the last checkpoint.
            metadata: Additional metadata to save.
        """
        if not self._job_name:
            logger.warning("No job name set, cannot save preemption state")
            return

        state = PreemptionState(
            job_name=self._job_name,
            job_type=self._job_type,
            iteration=self._current_iteration,
            checkpoint_path=checkpoint_path,
            preemption_time=datetime.utcnow().isoformat(),
            retry_count=self._retry_counts.get(self._job_name, 0),
            metadata=metadata or {},
        )

        # Save locally (sanitize job_name to prevent path traversal)
        safe_job_name = os.path.basename(self._job_name)
        state_path = self.checkpoint_dir / f"preemption_state_{safe_job_name}.yaml"
        with open(state_path, "w") as f:
            yaml.dump(state.to_dict(), f)

        logger.info(f"Preemption state saved: {state_path}")

        # Upload to GCS if configured
        if self.gcs_storage:
            try:
                gcs_path = f"{self.config.preemption_state_prefix}/{self._job_name}.yaml"
                state_bytes = yaml.dump(state.to_dict()).encode("utf-8")
                self.gcs_storage.backend.upload_blob(
                    self.gcs_storage.bucket,
                    gcs_path,
                    state_bytes,
                )
                logger.info(f"Preemption state uploaded to GCS: {gcs_path}")
            except Exception as e:
                logger.error(f"Failed to upload preemption state to GCS: {e}")

    def get_resume_state(self, job_name: str) -> PreemptionState | None:
        """Get preemption state for resuming a job.

        Args:
            job_name: Name of the job to resume.

        Returns:
            PreemptionState if found, None otherwise.
        """
        # Try local first (sanitize job_name to prevent path traversal)
        safe_job_name = os.path.basename(job_name)
        state_path = self.checkpoint_dir / f"preemption_state_{safe_job_name}.yaml"
        if state_path.exists():
            try:
                with open(state_path) as f:
                    data = yaml.safe_load(f)
                return PreemptionState.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load local preemption state: {e}")

        # Try GCS
        if self.gcs_storage:
            try:
                gcs_path = f"{self.config.preemption_state_prefix}/{job_name}.yaml"
                data_bytes = self.gcs_storage.backend.download_blob(
                    self.gcs_storage.bucket,
                    gcs_path,
                )
                data = yaml.safe_load(data_bytes.decode("utf-8"))
                return PreemptionState.from_dict(data)
            except Exception as e:
                logger.debug(f"No preemption state in GCS: {e}")

        return None

    def clear_resume_state(self, job_name: str) -> None:
        """Clear preemption state after successful resume.

        Args:
            job_name: Name of the job.
        """
        # Clear local (sanitize job_name to prevent path traversal)
        safe_job_name = os.path.basename(job_name)
        state_path = self.checkpoint_dir / f"preemption_state_{safe_job_name}.yaml"
        if state_path.exists():
            state_path.unlink()
            logger.debug(f"Cleared local preemption state: {state_path}")

        # Clear GCS
        if self.gcs_storage:
            try:
                gcs_path = f"{self.config.preemption_state_prefix}/{job_name}.yaml"
                self.gcs_storage.backend.delete_blob(
                    self.gcs_storage.bucket,
                    gcs_path,
                )
                logger.debug(f"Cleared GCS preemption state: {gcs_path}")
            except Exception as e:
                logger.debug(f"Failed to clear GCS preemption state: {e}")

        # Clear retry count
        self._retry_counts.pop(job_name, None)

    def should_retry(self, job_name: str) -> bool:
        """Check if job should be retried after preemption.

        Args:
            job_name: Name of the job.

        Returns:
            True if retry count is under limit.
        """
        count = self._retry_counts.get(job_name, 0)
        return count < self.config.max_retries

    def get_retry_delay(self, job_name: str) -> float:
        """Calculate retry delay with exponential backoff.

        Args:
            job_name: Name of the job.

        Returns:
            Delay in seconds before retry.
        """
        retry_count = self._retry_counts.get(job_name, 0)
        delay = self.config.retry_delay_seconds * (self.config.backoff_multiplier**retry_count)
        return min(delay, self.config.max_backoff_seconds)

    def increment_retry_count(self, job_name: str) -> int:
        """Increment and return retry count for a job.

        Args:
            job_name: Name of the job.

        Returns:
            New retry count.
        """
        self._retry_counts[job_name] = self._retry_counts.get(job_name, 0) + 1
        return self._retry_counts[job_name]

    def handle_preemption(
        self,
        error: PreemptionError | Exception | None = None,
    ) -> PreemptionState | None:
        """Handle a preemption event.

        Args:
            error: Optional preemption error with details.

        Returns:
            PreemptionState if checkpoint was created, None otherwise.
        """
        logger.warning(f"Handling preemption: {error}")

        # Create emergency checkpoint
        checkpoint_path = self._emergency_checkpoint()

        if checkpoint_path and self._job_name:
            # Save preemption state
            self.save_preemption_state(
                checkpoint_path=checkpoint_path,
                metadata={"error": str(error) if error else None},
            )

            return self.get_resume_state(self._job_name)

        return None

    @property
    def preemption_detected(self) -> bool:
        """Check if preemption has been detected."""
        return self._preemption_detected

    def get_statistics(self) -> dict[str, Any]:
        """Get handler statistics for logging.

        Returns:
            Dictionary with handler statistics.
        """
        return {
            "checkpoint_count": self._checkpoint_count,
            "last_checkpoint_time": self._last_checkpoint_time,
            "timer_running": self._timer_running,
            "preemption_detected": self._preemption_detected,
            "job_name": self._job_name,
            "job_type": self._job_type,
            "current_iteration": self._current_iteration,
            "retry_counts": dict(self._retry_counts),
            "config": self.config.to_dict(),
        }

    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop_checkpoint_timer()
        self.uninstall_signal_handlers()
        logger.info("PreemptionHandler cleaned up")

    def __repr__(self) -> str:
        return (
            f"PreemptionHandler("
            f"interval={self.config.checkpoint_interval_seconds}s, "
            f"max_retries={self.config.max_retries}, "
            f"checkpoints={self._checkpoint_count})"
        )
