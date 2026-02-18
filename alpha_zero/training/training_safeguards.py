"""Training safeguards for monitoring training health and triggering automated interventions.

This module provides real-time monitoring of training metrics (loss, gradients, memory, convergence)
and triggers interventions when issues are detected (NaN values, loss explosion, gradient issues,
memory pressure, convergence problems).

Key features:
- NaN detection in loss and gradients with consecutive tracking
- Loss explosion detection using rolling window statistics
- Gradient norm monitoring (high/exploding)
- Memory pressure monitoring via MemoryGuard integration
- Convergence monitoring via ConvergenceMonitor integration
- Checkpoint validation (NaN/Inf detection)
- Automated intervention recommendations (LR reduction, rollback, batch size reduction, etc.)
- Fully configurable via environment variables (prefix: CIV6AZ_SAFEGUARD_)

Example usage:
    config = TrainingSafeguardConfig.from_env()
    safeguard = TrainingSafeguard(
        config,
        convergence_monitor=ConvergenceMonitor(),
        memory_guard=MemoryGuard()
    )

    for step in range(num_steps):
        # ... training step ...
        report = safeguard.check_step(loss, grad_norm, step, device="gpu:0")

        if report.intervention:
            if report.intervention.action == InterventionAction.ROLLBACK:
                # Rollback to previous checkpoint
                load_checkpoint(prev_checkpoint)
            elif report.intervention.action == InterventionAction.REDUCE_LR:
                # Reduce learning rate
                lr *= 0.5

Environment variables:
    CIV6AZ_SAFEGUARD_NAN_DETECTION: Enable NaN detection (default: true)
    CIV6AZ_SAFEGUARD_LOSS_EXPLOSION_THRESHOLD: Loss explosion multiplier (default: 10.0)
    CIV6AZ_SAFEGUARD_LOSS_EXPLOSION_WINDOW: Rolling window size for loss history (default: 50)
    CIV6AZ_SAFEGUARD_GRAD_CLIP: Gradient norm clipping threshold (default: 1.0)
    CIV6AZ_SAFEGUARD_GRAD_ALERT: Gradient norm alert threshold (default: 5.0)
    CIV6AZ_SAFEGUARD_MEMORY_BUDGET: Memory budget fraction (default: 0.85)
    CIV6AZ_SAFEGUARD_CHECKPOINT_VALIDATION: Enable checkpoint validation (default: true)
    CIV6AZ_SAFEGUARD_AUTO_INTERVENTION: Enable auto intervention (default: true)
    CIV6AZ_SAFEGUARD_MAX_NAN_STEPS: Max consecutive NaN steps before rollback (default: 3)
"""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from alpha_zero.training.convergence_monitor import ConvergenceMonitor, ConvergenceStatus
from alpha_zero.utils.compat import UTC
from alpha_zero.utils.memory_guard import MemoryGuard, MemoryStatus

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class GradientStatus(str, Enum):
    """Gradient health status."""

    NORMAL = "normal"
    HIGH = "high"
    EXPLODING = "exploding"


class InterventionAction(str, Enum):
    """Automated intervention actions."""

    REDUCE_LR = "reduce_lr"
    ROLLBACK = "rollback"
    REDUCE_BATCH_SIZE = "reduce_batch_size"
    EARLY_STOP = "early_stop"
    INCREASE_EXPLORATION = "increase_exploration"


@dataclass
class TrainingSafeguardConfig:
    """Configuration for training safeguards.

    All fields can be configured via environment variables with CIV6AZ_SAFEGUARD_ prefix.
    """

    nan_detection_enabled: bool = True
    loss_explosion_threshold: float = 10.0
    loss_explosion_window: int = 50
    gradient_norm_clip: float = 1.0
    gradient_norm_alert_threshold: float = 5.0
    memory_budget_fraction: float = 0.85
    checkpoint_validation_enabled: bool = True
    auto_intervention_enabled: bool = True
    max_consecutive_nan_steps: int = 3

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.loss_explosion_threshold <= 1.0:
            raise ValueError("loss_explosion_threshold must be > 1.0")
        if self.loss_explosion_window <= 0:
            raise ValueError("loss_explosion_window must be > 0")
        if self.gradient_norm_clip <= 0:
            raise ValueError("gradient_norm_clip must be > 0")
        if self.gradient_norm_alert_threshold <= 0:
            raise ValueError("gradient_norm_alert_threshold must be > 0")
        if not 0 < self.memory_budget_fraction <= 1.0:
            raise ValueError("memory_budget_fraction must be in (0, 1]")
        if self.max_consecutive_nan_steps <= 0:
            raise ValueError("max_consecutive_nan_steps must be > 0")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nan_detection_enabled": self.nan_detection_enabled,
            "loss_explosion_threshold": self.loss_explosion_threshold,
            "loss_explosion_window": self.loss_explosion_window,
            "gradient_norm_clip": self.gradient_norm_clip,
            "gradient_norm_alert_threshold": self.gradient_norm_alert_threshold,
            "memory_budget_fraction": self.memory_budget_fraction,
            "checkpoint_validation_enabled": self.checkpoint_validation_enabled,
            "auto_intervention_enabled": self.auto_intervention_enabled,
            "max_consecutive_nan_steps": self.max_consecutive_nan_steps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingSafeguardConfig:
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path | str) -> TrainingSafeguardConfig:
        """Load from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_env(cls) -> TrainingSafeguardConfig:
        """Create from environment variables with CIV6AZ_SAFEGUARD_ prefix."""
        return cls(
            nan_detection_enabled=os.environ.get("CIV6AZ_SAFEGUARD_NAN_DETECTION", "true").lower()
            == "true",
            loss_explosion_threshold=float(
                os.environ.get("CIV6AZ_SAFEGUARD_LOSS_EXPLOSION_THRESHOLD", "10.0")
            ),
            loss_explosion_window=int(
                os.environ.get("CIV6AZ_SAFEGUARD_LOSS_EXPLOSION_WINDOW", "50")
            ),
            gradient_norm_clip=float(os.environ.get("CIV6AZ_SAFEGUARD_GRAD_CLIP", "1.0")),
            gradient_norm_alert_threshold=float(
                os.environ.get("CIV6AZ_SAFEGUARD_GRAD_ALERT", "5.0")
            ),
            memory_budget_fraction=float(os.environ.get("CIV6AZ_SAFEGUARD_MEMORY_BUDGET", "0.85")),
            checkpoint_validation_enabled=os.environ.get(
                "CIV6AZ_SAFEGUARD_CHECKPOINT_VALIDATION", "true"
            ).lower()
            == "true",
            auto_intervention_enabled=os.environ.get(
                "CIV6AZ_SAFEGUARD_AUTO_INTERVENTION", "true"
            ).lower()
            == "true",
            max_consecutive_nan_steps=int(os.environ.get("CIV6AZ_SAFEGUARD_MAX_NAN_STEPS", "3")),
        )


@dataclass
class Intervention:
    """Recommended intervention action."""

    action: InterventionAction
    reason: str
    severity: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate severity."""
        if not 1 <= self.severity <= 5:
            raise ValueError("severity must be in [1, 5]")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action": self.action.value,
            "reason": self.reason,
            "severity": self.severity,
            "metadata": {
                k: v.value if isinstance(v, Enum) else v for k, v in self.metadata.items()
            },
        }


@dataclass
class CheckpointValidation:
    """Checkpoint validation result."""

    valid: bool
    path: str
    num_parameters: int = 0
    has_nan: bool = False
    has_inf: bool = False
    dtype_issues: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "path": self.path,
            "num_parameters": self.num_parameters,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "dtype_issues": self.dtype_issues,
            "error": self.error,
        }


@dataclass
class SafeguardReport:
    """Training safeguard check report."""

    step: int
    loss: float
    grad_norm: float
    nan_detected: bool = False
    loss_exploding: bool = False
    gradient_status: GradientStatus = GradientStatus.NORMAL
    memory_status: MemoryStatus | None = None
    convergence_status: ConvergenceStatus | None = None
    intervention: Intervention | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step": self.step,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "nan_detected": self.nan_detected,
            "loss_exploding": self.loss_exploding,
            "gradient_status": self.gradient_status.value if self.gradient_status else None,
            "memory_status": self.memory_status.value if self.memory_status else None,
            "convergence_status": (
                self.convergence_status.value if self.convergence_status else None
            ),
            "intervention": self.intervention.to_dict() if self.intervention else None,
            "timestamp": self.timestamp,
        }


class TrainingSafeguard:
    """Training safeguard monitor with automated intervention recommendations.

    Monitors training health metrics and recommends interventions when issues are detected.
    Integrates with ConvergenceMonitor for convergence tracking and MemoryGuard for
    memory pressure monitoring.
    """

    def __init__(
        self,
        config: TrainingSafeguardConfig | None = None,
        *,
        convergence_monitor: ConvergenceMonitor | None = None,
        memory_guard: MemoryGuard | None = None,
    ) -> None:
        """Initialize training safeguard.

        Args:
            config: Safeguard configuration (uses default if None)
            convergence_monitor: Optional convergence monitor for convergence tracking
            memory_guard: Optional memory guard for memory pressure monitoring
        """
        self.config = config or TrainingSafeguardConfig()
        self.convergence_monitor = convergence_monitor
        self.memory_guard = memory_guard

        # Internal state
        self._loss_history: deque[float] = deque(maxlen=self.config.loss_explosion_window)
        self._consecutive_nan_count = 0
        self._step_count = 0
        self._intervention_count = 0
        self._last_valid_loss: float | None = None

        logger.info(f"TrainingSafeguard initialized with config: {self.config.to_dict()}")

    def check_step(
        self,
        loss: float,
        grad_norm: float,
        step: int,
        *,
        device: str = "gpu:0",
        **extra_metrics: float,
    ) -> SafeguardReport:
        """Check training step for issues and recommend interventions.

        Args:
            loss: Current training loss
            grad_norm: Current gradient norm
            step: Current training step
            device: Device for memory monitoring (default: "gpu:0")
            **extra_metrics: Additional metrics to log (not used for intervention logic)

        Returns:
            SafeguardReport with detected issues and optional intervention
        """
        self._step_count = step

        # Run all checks
        nan_detected = (
            self.check_nan(loss, grad_norm) if self.config.nan_detection_enabled else False
        )
        loss_exploding = self.check_loss_explosion(loss, step)
        gradient_status = self.check_gradient_health(grad_norm)
        memory_status = self.check_memory(device)
        convergence_status = (
            self.convergence_monitor.get_status() if self.convergence_monitor else None
        )

        # Build report
        report = SafeguardReport(
            step=step,
            loss=loss,
            grad_norm=grad_norm,
            nan_detected=nan_detected,
            loss_exploding=loss_exploding,
            gradient_status=gradient_status,
            memory_status=memory_status,
            convergence_status=convergence_status,
        )

        # Compute intervention if auto-intervention enabled
        if self.config.auto_intervention_enabled:
            intervention = self.get_intervention(report)
            if intervention:
                report.intervention = intervention
                self._intervention_count += 1

        # Log appropriately
        if report.intervention:
            logger.warning(
                f"Step {step}: Intervention recommended: {report.intervention.action} "
                f"(severity {report.intervention.severity}) - {report.intervention.reason}"
            )
        elif nan_detected or loss_exploding or gradient_status != GradientStatus.NORMAL:
            logger.warning(
                f"Step {step}: Issues detected - NaN: {nan_detected}, "
                f"Loss exploding: {loss_exploding}, Gradient: {gradient_status}"
            )
        else:
            logger.debug(
                f"Step {step}: All checks passed - loss: {loss:.4f}, grad_norm: {grad_norm:.4f}"
            )

        return report

    def check_nan(self, loss: float, grad_norm: float) -> bool:
        """Check for NaN in loss or gradient norm.

        Args:
            loss: Current training loss
            grad_norm: Current gradient norm

        Returns:
            True if NaN detected, False otherwise
        """
        has_nan = math.isnan(loss) or math.isnan(grad_norm)

        if has_nan:
            self._consecutive_nan_count += 1
            logger.warning(
                f"NaN detected (loss: {loss}, grad_norm: {grad_norm}) - "
                f"consecutive count: {self._consecutive_nan_count}"
            )
        else:
            self._consecutive_nan_count = 0

        return has_nan

    def check_loss_explosion(self, loss: float, step: int) -> bool:
        """Check if loss is exploding compared to recent history.

        Args:
            loss: Current training loss
            step: Current training step

        Returns:
            True if loss is exploding, False otherwise
        """
        # Skip if NaN
        if math.isnan(loss):
            return False

        # Update history
        self._loss_history.append(loss)
        self._last_valid_loss = loss

        # Need enough history to detect explosion
        if len(self._loss_history) < 10:
            return False

        # Check if loss exceeds threshold * mean
        mean_loss = sum(self._loss_history) / len(self._loss_history)
        threshold = mean_loss * self.config.loss_explosion_threshold

        if loss > threshold:
            logger.warning(
                f"Loss explosion detected at step {step}: "
                f"loss={loss:.4f} > threshold={threshold:.4f} "
                f"(mean={mean_loss:.4f} * {self.config.loss_explosion_threshold})"
            )
            return True

        return False

    def check_gradient_health(self, grad_norm: float) -> GradientStatus:
        """Check gradient health based on norm.

        Args:
            grad_norm: Current gradient norm

        Returns:
            GradientStatus (NORMAL, HIGH, or EXPLODING)
        """
        if math.isnan(grad_norm):
            return GradientStatus.EXPLODING

        exploding_threshold = self.config.gradient_norm_alert_threshold * 10
        if grad_norm > exploding_threshold:
            logger.warning(f"Gradient exploding: grad_norm={grad_norm:.4f}")
            return GradientStatus.EXPLODING

        if grad_norm > self.config.gradient_norm_alert_threshold:
            logger.warning(f"Gradient high: grad_norm={grad_norm:.4f}")
            return GradientStatus.HIGH

        return GradientStatus.NORMAL

    def check_memory(self, device: str = "gpu:0") -> MemoryStatus | None:
        """Check memory status via MemoryGuard.

        Args:
            device: Device to check memory for (default: "gpu:0")

        Returns:
            MemoryStatus if memory_guard available, None otherwise
        """
        if self.memory_guard is None:
            return None

        try:
            return self.memory_guard.get_memory_status(device=device)
        except Exception as e:
            logger.warning(f"Failed to check memory status: {e}")
            return None

    def validate_checkpoint(self, path: Path | str) -> CheckpointValidation:
        """Validate checkpoint for NaN/Inf and other issues.

        Args:
            path: Path to checkpoint file

        Returns:
            CheckpointValidation with validation results
        """
        if not self.config.checkpoint_validation_enabled:
            return CheckpointValidation(valid=True, path=str(path), error="Validation disabled")

        if not TORCH_AVAILABLE:
            return CheckpointValidation(valid=False, path=str(path), error="PyTorch not available")

        path_str = str(path)

        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location="cpu")

            # Check for state_dict (common checkpoint format)
            state_dict = checkpoint
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            # Count parameters and check for NaN/Inf
            num_parameters = 0
            has_nan = False
            has_inf = False
            dtype_issues = []

            for name, param in state_dict.items():
                if not isinstance(param, torch.Tensor):
                    continue

                num_parameters += param.numel()

                if torch.isnan(param).any():
                    has_nan = True
                    logger.warning(f"NaN detected in checkpoint parameter: {name}")

                if torch.isinf(param).any():
                    has_inf = True
                    logger.warning(f"Inf detected in checkpoint parameter: {name}")

                # Check for unexpected dtypes
                if param.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                    dtype_issues.append(f"{name}: {param.dtype}")

            valid = not (has_nan or has_inf)

            return CheckpointValidation(
                valid=valid,
                path=path_str,
                num_parameters=num_parameters,
                has_nan=has_nan,
                has_inf=has_inf,
                dtype_issues=dtype_issues,
            )

        except Exception as e:
            logger.error(f"Failed to validate checkpoint {path_str}: {e}")
            return CheckpointValidation(valid=False, path=path_str, error=str(e))

    def get_intervention(self, report: SafeguardReport) -> Intervention | None:
        """Compute intervention recommendation based on safeguard report.

        Intervention priority (highest to lowest):
        1. ROLLBACK: Consecutive NaN steps >= threshold (severity 5)
        2. REDUCE_LR: Loss exploding, gradient exploding, or convergence diverging (severity 4)
        3. REDUCE_BATCH_SIZE: Memory OOM risk or critical (severity 3-4)
        4. INCREASE_EXPLORATION: Convergence stalled (severity 2)

        Args:
            report: SafeguardReport from check_step

        Returns:
            Intervention if action recommended, None if training healthy
        """
        # Priority 1: Rollback on consecutive NaN
        if (
            report.nan_detected
            and self._consecutive_nan_count >= self.config.max_consecutive_nan_steps
        ):
            return Intervention(
                action=InterventionAction.ROLLBACK,
                reason=f"Consecutive NaN steps: {self._consecutive_nan_count} >= "
                f"{self.config.max_consecutive_nan_steps}",
                severity=5,
                metadata={
                    "consecutive_nan_count": self._consecutive_nan_count,
                    "loss": report.loss,
                    "grad_norm": report.grad_norm,
                },
            )

        # Priority 2: Reduce LR for loss/gradient issues
        if report.loss_exploding:
            return Intervention(
                action=InterventionAction.REDUCE_LR,
                reason=f"Loss exploding: {report.loss:.4f}",
                severity=4,
                metadata={"loss": report.loss},
            )

        if report.gradient_status == GradientStatus.EXPLODING:
            return Intervention(
                action=InterventionAction.REDUCE_LR,
                reason=f"Gradient exploding: {report.grad_norm:.4f}",
                severity=4,
                metadata={"grad_norm": report.grad_norm},
            )

        if report.convergence_status == ConvergenceStatus.DIVERGING:
            return Intervention(
                action=InterventionAction.REDUCE_LR,
                reason="Convergence diverging",
                severity=4,
                metadata={"convergence_status": report.convergence_status},
            )

        # Priority 3: Reduce batch size for memory issues
        if report.memory_status == MemoryStatus.OOM_RISK:
            return Intervention(
                action=InterventionAction.REDUCE_BATCH_SIZE,
                reason="Memory OOM risk",
                severity=4,
                metadata={"memory_status": report.memory_status},
            )

        if report.memory_status == MemoryStatus.CRITICAL:
            return Intervention(
                action=InterventionAction.REDUCE_BATCH_SIZE,
                reason="Memory critical",
                severity=3,
                metadata={"memory_status": report.memory_status},
            )

        # Priority 4: Increase exploration for stalled convergence
        if report.convergence_status == ConvergenceStatus.STALLED:
            return Intervention(
                action=InterventionAction.INCREASE_EXPLORATION,
                reason="Convergence stalled",
                severity=2,
                metadata={"convergence_status": report.convergence_status},
            )

        return None

    def reset(self) -> None:
        """Reset internal state."""
        self._loss_history.clear()
        self._consecutive_nan_count = 0
        self._step_count = 0
        self._intervention_count = 0
        self._last_valid_loss = None
        logger.info("TrainingSafeguard state reset")

    @property
    def step_count(self) -> int:
        """Get current step count."""
        return self._step_count

    @property
    def intervention_count(self) -> int:
        """Get total intervention count."""
        return self._intervention_count

    @property
    def consecutive_nan_count(self) -> int:
        """Get consecutive NaN count."""
        return self._consecutive_nan_count
