"""Memory monitoring and protection utilities.

This module provides memory monitoring for GPU and CPU, automatic
batch size reduction, and gradient checkpointing activation to
prevent OOM errors during training.

Usage:
    from alpha_zero.utils.memory_guard import (
        MemoryGuardConfig,
        MemoryGuard,
    )

    config = MemoryGuardConfig()
    guard = MemoryGuard(config)

    # Check memory before operation
    if guard.can_allocate(memory_mb=1024):
        # Perform operation
        pass

    # Get recommended batch size
    batch_size = guard.get_safe_batch_size(current_batch=32)
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Try to import torch for GPU memory tracking
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

# Try to import psutil for CPU memory tracking
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False


class MemoryStatus(str, Enum):
    """Memory status levels."""

    NORMAL = "normal"  # Memory usage is fine
    WARNING = "warning"  # Approaching limit
    CRITICAL = "critical"  # Very high usage
    OOM_RISK = "oom_risk"  # Imminent OOM


@dataclass
class MemoryInfo:
    """Memory usage information.

    Attributes:
        total_mb: Total memory in MB.
        used_mb: Used memory in MB.
        free_mb: Free memory in MB.
        usage_ratio: Usage ratio (0-1).
        device: Device identifier (e.g., "gpu:0", "cpu").
        timestamp: Time of measurement.
    """

    total_mb: float
    used_mb: float
    free_mb: float
    usage_ratio: float
    device: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_mb": self.total_mb,
            "used_mb": self.used_mb,
            "free_mb": self.free_mb,
            "usage_ratio": self.usage_ratio,
            "device": self.device,
            "timestamp": self.timestamp,
        }


@dataclass
class MemoryGuardConfig:
    """Configuration for memory guard.

    Attributes:
        warning_threshold: Usage ratio triggering warning (0-1).
        critical_threshold: Usage ratio triggering critical status (0-1).
        oom_threshold: Usage ratio indicating OOM risk (0-1).
        min_batch_size: Minimum batch size to reduce to.
        batch_reduction_factor: Factor to reduce batch size by.
        enable_gradient_checkpointing: Auto-enable gradient checkpointing.
        gc_on_warning: Run garbage collection on warning.
        monitor_interval_seconds: Interval between memory checks.
        memory_reserve_mb: Reserved memory buffer in MB.
        enable_cuda_memory_management: Use CUDA memory management.
    """

    warning_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_MEMORY_WARNING_THRESHOLD", "0.7"))
    )
    critical_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_MEMORY_CRITICAL_THRESHOLD", "0.85"))
    )
    oom_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_MEMORY_OOM_THRESHOLD", "0.95"))
    )
    min_batch_size: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_MEMORY_MIN_BATCH", "1"))
    )
    batch_reduction_factor: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_MEMORY_BATCH_REDUCTION", "0.5"))
    )
    enable_gradient_checkpointing: bool = field(
        default_factory=lambda: (
            os.environ.get("CIV6AZ_MEMORY_GRADIENT_CHECKPOINT", "true").lower() == "true"
        )
    )
    gc_on_warning: bool = field(
        default_factory=lambda: (
            os.environ.get("CIV6AZ_MEMORY_GC_ON_WARNING", "true").lower() == "true"
        )
    )
    monitor_interval_seconds: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_MEMORY_MONITOR_INTERVAL", "10.0"))
    )
    memory_reserve_mb: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_MEMORY_RESERVE_MB", "512.0"))
    )
    enable_cuda_memory_management: bool = field(
        default_factory=lambda: (
            os.environ.get("CIV6AZ_MEMORY_CUDA_MANAGEMENT", "true").lower() == "true"
        )
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if not 0 <= self.warning_threshold <= 1:
            raise ValueError("warning_threshold must be between 0 and 1")
        if not 0 <= self.critical_threshold <= 1:
            raise ValueError("critical_threshold must be between 0 and 1")
        if not 0 <= self.oom_threshold <= 1:
            raise ValueError("oom_threshold must be between 0 and 1")
        if self.warning_threshold >= self.critical_threshold:
            raise ValueError("warning_threshold must be less than critical_threshold")
        if self.critical_threshold >= self.oom_threshold:
            raise ValueError("critical_threshold must be less than oom_threshold")
        if self.min_batch_size < 1:
            raise ValueError("min_batch_size must be >= 1")
        if not 0 < self.batch_reduction_factor < 1:
            raise ValueError("batch_reduction_factor must be between 0 and 1")
        if self.monitor_interval_seconds <= 0:
            raise ValueError("monitor_interval_seconds must be positive")
        if self.memory_reserve_mb < 0:
            raise ValueError("memory_reserve_mb must be non-negative")

    @classmethod
    def from_env(cls) -> MemoryGuardConfig:
        """Create configuration from environment variables."""
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> MemoryGuardConfig:
        """Load configuration from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if "memory_guard" in data:
            data = data["memory_guard"]

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryGuardConfig:
        """Create configuration from dictionary."""
        kwargs: dict[str, Any] = {}

        if "warning_threshold" in data:
            kwargs["warning_threshold"] = float(data["warning_threshold"])
        if "critical_threshold" in data:
            kwargs["critical_threshold"] = float(data["critical_threshold"])
        if "oom_threshold" in data:
            kwargs["oom_threshold"] = float(data["oom_threshold"])
        if "min_batch_size" in data:
            kwargs["min_batch_size"] = int(data["min_batch_size"])
        if "batch_reduction_factor" in data:
            kwargs["batch_reduction_factor"] = float(data["batch_reduction_factor"])
        if "enable_gradient_checkpointing" in data:
            kwargs["enable_gradient_checkpointing"] = bool(data["enable_gradient_checkpointing"])
        if "gc_on_warning" in data:
            kwargs["gc_on_warning"] = bool(data["gc_on_warning"])
        if "monitor_interval_seconds" in data:
            kwargs["monitor_interval_seconds"] = float(data["monitor_interval_seconds"])
        if "memory_reserve_mb" in data:
            kwargs["memory_reserve_mb"] = float(data["memory_reserve_mb"])
        if "enable_cuda_memory_management" in data:
            kwargs["enable_cuda_memory_management"] = bool(data["enable_cuda_memory_management"])

        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "oom_threshold": self.oom_threshold,
            "min_batch_size": self.min_batch_size,
            "batch_reduction_factor": self.batch_reduction_factor,
            "enable_gradient_checkpointing": self.enable_gradient_checkpointing,
            "gc_on_warning": self.gc_on_warning,
            "monitor_interval_seconds": self.monitor_interval_seconds,
            "memory_reserve_mb": self.memory_reserve_mb,
            "enable_cuda_memory_management": self.enable_cuda_memory_management,
        }


class MemoryGuard:
    """Guards against memory exhaustion during training.

    This class monitors GPU and CPU memory usage, provides automatic
    batch size reduction, and activates gradient checkpointing when
    memory pressure is high.

    Attributes:
        config: Memory guard configuration.
    """

    def __init__(self, config: MemoryGuardConfig | None = None) -> None:
        """Initialize memory guard.

        Args:
            config: Memory guard configuration.
        """
        self.config = config or MemoryGuardConfig()
        self._lock = threading.RLock()

        # Current state
        self._current_batch_size: int | None = None
        self._gradient_checkpointing_enabled = False

        # Memory history
        self._memory_history: list[MemoryInfo] = []
        self._max_history = 100

        # Monitoring
        self._monitoring = False
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Callbacks
        self._on_warning: list[Callable[[MemoryInfo], None]] = []
        self._on_critical: list[Callable[[MemoryInfo], None]] = []
        self._on_oom_risk: list[Callable[[MemoryInfo], None]] = []

        # Statistics
        self._gc_runs = 0
        self._batch_reductions = 0
        self._oom_prevented = 0

        logger.info(
            f"MemoryGuard initialized: "
            f"warning={self.config.warning_threshold:.0%}, "
            f"critical={self.config.critical_threshold:.0%}, "
            f"torch={TORCH_AVAILABLE}, psutil={PSUTIL_AVAILABLE}"
        )

    def get_gpu_memory(self, device_id: int = 0) -> MemoryInfo | None:
        """Get GPU memory information.

        Args:
            device_id: GPU device ID.

        Returns:
            MemoryInfo or None if not available.
        """
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return None

        try:
            with self._lock:
                props = torch.cuda.get_device_properties(device_id)
                total_mb = props.total_memory / (1024 * 1024)

                # Get reserved memory (includes fragmentation overhead)
                reserved = torch.cuda.memory_reserved(device_id) / (1024 * 1024)
                used_mb = reserved
                free_mb = total_mb - used_mb

                return MemoryInfo(
                    total_mb=total_mb,
                    used_mb=used_mb,
                    free_mb=free_mb,
                    usage_ratio=used_mb / total_mb if total_mb > 0 else 0,
                    device=f"gpu:{device_id}",
                )
        except Exception as e:
            logger.warning(f"Failed to get GPU memory: {e}")
            return None

    def get_cpu_memory(self) -> MemoryInfo | None:
        """Get CPU memory information.

        Returns:
            MemoryInfo or None if not available.
        """
        if not PSUTIL_AVAILABLE:
            return None

        try:
            mem = psutil.virtual_memory()
            total_mb = mem.total / (1024 * 1024)
            used_mb = mem.used / (1024 * 1024)
            free_mb = mem.available / (1024 * 1024)

            return MemoryInfo(
                total_mb=total_mb,
                used_mb=used_mb,
                free_mb=free_mb,
                usage_ratio=mem.percent / 100,
                device="cpu",
            )
        except Exception as e:
            logger.warning(f"Failed to get CPU memory: {e}")
            return None

    def get_memory_status(self, device: str = "gpu:0") -> MemoryStatus:
        """Get current memory status.

        Args:
            device: Device to check ("gpu:N" or "cpu").

        Returns:
            MemoryStatus indicating current status.
        """
        if device.startswith("gpu:"):
            device_id = int(device.split(":")[1])
            mem_info = self.get_gpu_memory(device_id)
        else:
            mem_info = self.get_cpu_memory()

        if mem_info is None:
            return MemoryStatus.NORMAL  # Can't determine, assume OK

        if mem_info.usage_ratio >= self.config.oom_threshold:
            return MemoryStatus.OOM_RISK
        if mem_info.usage_ratio >= self.config.critical_threshold:
            return MemoryStatus.CRITICAL
        if mem_info.usage_ratio >= self.config.warning_threshold:
            return MemoryStatus.WARNING
        return MemoryStatus.NORMAL

    def can_allocate(self, memory_mb: float, device: str = "gpu:0") -> bool:
        """Check if memory can be allocated safely.

        Args:
            memory_mb: Memory needed in MB.
            device: Target device.

        Returns:
            True if allocation is safe.
        """
        if device.startswith("gpu:"):
            device_id = int(device.split(":")[1])
            mem_info = self.get_gpu_memory(device_id)
        else:
            mem_info = self.get_cpu_memory()

        if mem_info is None:
            return True  # Can't determine, allow

        available = mem_info.free_mb - self.config.memory_reserve_mb
        return available >= memory_mb

    def get_safe_batch_size(
        self,
        current_batch: int,
        memory_per_sample_mb: float | None = None,
        device: str = "gpu:0",
    ) -> int:
        """Get safe batch size based on memory conditions.

        Args:
            current_batch: Current batch size.
            memory_per_sample_mb: Estimated memory per sample.
            device: Target device.

        Returns:
            Recommended batch size.
        """
        status = self.get_memory_status(device)

        if status == MemoryStatus.NORMAL:
            return current_batch

        # Reduce batch size based on status
        if status == MemoryStatus.WARNING:
            reduction = self.config.batch_reduction_factor
        elif status == MemoryStatus.CRITICAL:
            reduction = self.config.batch_reduction_factor**2
        else:  # OOM_RISK
            reduction = self.config.batch_reduction_factor**3

        new_batch = max(
            self.config.min_batch_size,
            int(current_batch * reduction),
        )

        if new_batch < current_batch:
            self._batch_reductions += 1
            logger.warning(
                f"Reducing batch size from {current_batch} to {new_batch} "
                f"due to {status.value} memory status"
            )

        return new_batch

    def cleanup_memory(self, force_gc: bool = False) -> None:
        """Clean up memory (garbage collection and CUDA cache).

        Args:
            force_gc: Force garbage collection regardless of settings.
        """
        if force_gc or self.config.gc_on_warning:
            gc.collect()
            self._gc_runs += 1

        if TORCH_AVAILABLE and torch.cuda.is_available():
            if self.config.enable_cuda_memory_management:
                torch.cuda.empty_cache()

        logger.debug("Memory cleanup completed")

    def check_and_respond(self, device: str = "gpu:0") -> MemoryStatus:
        """Check memory and take appropriate action.

        Args:
            device: Device to check.

        Returns:
            Current memory status.
        """
        status = self.get_memory_status(device)

        if device.startswith("gpu:"):
            device_id = int(device.split(":")[1])
            mem_info = self.get_gpu_memory(device_id)
        else:
            mem_info = self.get_cpu_memory()

        if mem_info:
            self._memory_history.append(mem_info)
            if len(self._memory_history) > self._max_history:
                self._memory_history = self._memory_history[-self._max_history :]

        if status == MemoryStatus.WARNING:
            if self.config.gc_on_warning:
                self.cleanup_memory()
            for callback in self._on_warning:
                try:
                    if mem_info:
                        callback(mem_info)
                except Exception as e:
                    logger.error(f"Error in warning callback: {e}")

        elif status == MemoryStatus.CRITICAL:
            self.cleanup_memory(force_gc=True)
            for callback in self._on_critical:
                try:
                    if mem_info:
                        callback(mem_info)
                except Exception as e:
                    logger.error(f"Error in critical callback: {e}")

        elif status == MemoryStatus.OOM_RISK:
            self.cleanup_memory(force_gc=True)
            self._oom_prevented += 1
            for callback in self._on_oom_risk:
                try:
                    if mem_info:
                        callback(mem_info)
                except Exception as e:
                    logger.error(f"Error in OOM risk callback: {e}")

            logger.error(
                f"OOM risk detected on {device}! "
                f"Memory: {mem_info.used_mb:.0f}/{mem_info.total_mb:.0f} MB"
                if mem_info
                else f"OOM risk detected on {device}!"
            )

        return status

    def enable_gradient_checkpointing(self, model: Any, enable: bool = True) -> None:
        """Enable or disable gradient checkpointing on a model.

        Args:
            model: PyTorch model.
            enable: Whether to enable checkpointing.
        """
        if not TORCH_AVAILABLE:
            return

        if not self.config.enable_gradient_checkpointing:
            return

        try:
            if hasattr(model, "gradient_checkpointing_enable"):
                if enable:
                    model.gradient_checkpointing_enable()
                    self._gradient_checkpointing_enabled = True
                    logger.info("Gradient checkpointing enabled")
                else:
                    if hasattr(model, "gradient_checkpointing_disable"):
                        model.gradient_checkpointing_disable()
                    self._gradient_checkpointing_enabled = False
                    logger.info("Gradient checkpointing disabled")
        except Exception as e:
            logger.warning(f"Failed to set gradient checkpointing: {e}")

    def start_monitoring(self, device: str = "gpu:0") -> None:
        """Start background memory monitoring.

        Args:
            device: Device to monitor.
        """
        if self._monitoring:
            logger.warning("Memory monitoring already running")
            return

        self._monitoring = True
        self._stop_event.clear()

        def monitor_loop() -> None:
            while not self._stop_event.wait(self.config.monitor_interval_seconds):
                try:
                    self.check_and_respond(device)
                except Exception as e:
                    logger.error(f"Error in memory monitoring: {e}")

        self._monitor_thread = threading.Thread(
            target=monitor_loop,
            daemon=True,
            name="MemoryGuard-Monitor",
        )
        self._monitor_thread.start()
        logger.info(
            f"Started memory monitoring with {self.config.monitor_interval_seconds}s interval"
        )

    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        if not self._monitoring:
            return

        self._monitoring = False
        self._stop_event.set()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

        logger.info("Stopped memory monitoring")

    def register_warning_callback(self, callback: Callable[[MemoryInfo], None]) -> None:
        """Register callback for memory warnings."""
        self._on_warning.append(callback)

    def register_critical_callback(self, callback: Callable[[MemoryInfo], None]) -> None:
        """Register callback for critical memory."""
        self._on_critical.append(callback)

    def register_oom_risk_callback(self, callback: Callable[[MemoryInfo], None]) -> None:
        """Register callback for OOM risk."""
        self._on_oom_risk.append(callback)

    def get_statistics(self) -> dict[str, Any]:
        """Get memory guard statistics."""
        return {
            "gc_runs": self._gc_runs,
            "batch_reductions": self._batch_reductions,
            "oom_prevented": self._oom_prevented,
            "gradient_checkpointing_enabled": self._gradient_checkpointing_enabled,
            "monitoring_active": self._monitoring,
            "history_size": len(self._memory_history),
        }

    def get_memory_trend(self, window_size: int = 10) -> float | None:
        """Get memory usage trend (positive = increasing).

        Args:
            window_size: Number of samples for trend calculation.

        Returns:
            Trend coefficient or None if insufficient data.
        """
        if len(self._memory_history) < window_size:
            return None

        recent = self._memory_history[-window_size:]
        first_usage = recent[0].usage_ratio
        last_usage = recent[-1].usage_ratio

        return last_usage - first_usage

    def reset(self) -> None:
        """Reset memory guard state."""
        with self._lock:
            self._memory_history.clear()
            self._gc_runs = 0
            self._batch_reductions = 0
            self._oom_prevented = 0
            self._current_batch_size = None
        logger.info("MemoryGuard state reset")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MemoryGuard("
            f"warning={self.config.warning_threshold:.0%}, "
            f"critical={self.config.critical_threshold:.0%}, "
            f"monitoring={self._monitoring})"
        )
