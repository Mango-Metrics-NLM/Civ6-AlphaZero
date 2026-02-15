"""
Dynamic GPU Scheduler for GCP Cloud Training.

Provides intelligent GPU type selection based on training phase,
implementing T4 → V100 upgrade strategy for final training phases.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from alpha_zero.cloud.gcp.config import get_gpu_costs
from alpha_zero.cloud.gcp.cost_monitor import AlertLevel
from alpha_zero.cloud.gcp.exceptions import ConfigurationError

if TYPE_CHECKING:
    from alpha_zero.cloud.gcp.cost_monitor import CostMonitor

logger = logging.getLogger(__name__)


def _get_gpu_costs_with_env_overrides() -> dict[str, float]:
    """Get GPU costs using centralized config with env-var overrides."""
    return get_gpu_costs()


class TrainingPhase(str, Enum):
    """Training phase for GPU selection."""

    EARLY = "early"  # 0-50% of iterations
    MIDDLE = "middle"  # 50-80% of iterations
    LATE = "late"  # 80-90% of iterations
    FINAL = "final"  # 90-100% of iterations


@dataclass
class GPUSelection:
    """Result of GPU selection for a job.

    Attributes:
        gpu_type: Selected GPU type (e.g., NVIDIA_TESLA_T4).
        machine_type: GCE machine type for the GPU.
        use_spot: Whether to use spot/preemptible instances.
        reason: Human-readable reason for selection.
        estimated_hourly_cost: Estimated hourly cost in USD.
    """

    gpu_type: str
    machine_type: str
    use_spot: bool
    reason: str
    estimated_hourly_cost: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "gpu_type": self.gpu_type,
            "machine_type": self.machine_type,
            "use_spot": self.use_spot,
            "reason": self.reason,
            "estimated_hourly_cost": self.estimated_hourly_cost,
        }


@dataclass
class GPUSchedulerConfig:
    """Configuration for dynamic GPU selection.

    All values can be overridden via environment variables with the CIV6AZ_ prefix.

    Environment Variables:
        CIV6AZ_GPU_DEFAULT: Default GPU type (default: NVIDIA_TESLA_T4)
        CIV6AZ_GPU_FINAL_PHASE: GPU for final phase (default: NVIDIA_TESLA_V100)
        CIV6AZ_GPU_FINAL_PHASE_THRESHOLD: Iteration ratio to switch (default: 0.9)
        CIV6AZ_GPU_PREFER_SPOT: Prefer spot instances (default: true)
        CIV6AZ_GPU_UPGRADE_ENABLED: Enable auto GPU upgrade (default: true)
    """

    default_gpu: str = field(
        default_factory=lambda: os.environ.get("CIV6AZ_GPU_DEFAULT", "NVIDIA_TESLA_T4")
    )
    final_phase_gpu: str = field(
        default_factory=lambda: os.environ.get("CIV6AZ_GPU_FINAL_PHASE", "NVIDIA_TESLA_V100")
    )
    final_phase_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_GPU_FINAL_PHASE_THRESHOLD", "0.9"))
    )
    prefer_spot: bool = field(
        default_factory=lambda: os.environ.get("CIV6AZ_GPU_PREFER_SPOT", "true").lower() == "true"
    )
    upgrade_enabled: bool = field(
        default_factory=lambda: (
            os.environ.get("CIV6AZ_GPU_UPGRADE_ENABLED", "true").lower() == "true"
        )
    )

    # Phase thresholds (percentage of total iterations)
    early_phase_end: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_GPU_EARLY_PHASE_END", "0.5"))
    )
    middle_phase_end: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_GPU_MIDDLE_PHASE_END", "0.8"))
    )
    late_phase_end: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_GPU_LATE_PHASE_END", "0.9"))
    )

    # GPU tier mappings for different training phases
    phase_gpu_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "early": os.environ.get("CIV6AZ_GPU_PHASE_EARLY", "NVIDIA_TESLA_T4"),
            "middle": os.environ.get("CIV6AZ_GPU_PHASE_MIDDLE", "NVIDIA_TESLA_T4"),
            "late": os.environ.get("CIV6AZ_GPU_PHASE_LATE", "NVIDIA_L4"),
            "final": os.environ.get("CIV6AZ_GPU_PHASE_FINAL", "NVIDIA_TESLA_V100"),
        }
    )

    # Machine types for each GPU
    gpu_machine_types: dict[str, str] = field(
        default_factory=lambda: {
            "NVIDIA_TESLA_T4": "n1-standard-8",
            "NVIDIA_L4": "g2-standard-8",
            "NVIDIA_TESLA_V100": "n1-standard-8",
            "NVIDIA_TESLA_A100": "a2-highgpu-1g",
        }
    )

    # GPU hourly costs (USD) for cost-aware selection
    # Defaults can be overridden via CIV6AZ_GPU_COST_* environment variables
    gpu_costs: dict[str, float] = field(default_factory=_get_gpu_costs_with_env_overrides)

    # Spot VM settings by job type
    spot_for_selfplay: bool = field(
        default_factory=lambda: os.environ.get("CIV6AZ_GPU_SPOT_SELFPLAY", "true").lower() == "true"
    )
    spot_for_training: bool = field(
        default_factory=lambda: (
            os.environ.get("CIV6AZ_GPU_SPOT_TRAINING", "false").lower() == "true"
        )
    )

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigurationError: If values are invalid.
        """
        if not 0 < self.early_phase_end < 1:
            raise ConfigurationError(
                "early_phase_end must be between 0 and 1",
                details={"early_phase_end": self.early_phase_end},
            )
        if not self.early_phase_end < self.middle_phase_end < 1:
            raise ConfigurationError(
                "middle_phase_end must be between early_phase_end and 1",
                details={
                    "early_phase_end": self.early_phase_end,
                    "middle_phase_end": self.middle_phase_end,
                },
            )
        if not self.middle_phase_end < self.late_phase_end <= 1:
            raise ConfigurationError(
                "late_phase_end must be between middle_phase_end and 1",
                details={
                    "middle_phase_end": self.middle_phase_end,
                    "late_phase_end": self.late_phase_end,
                },
            )
        if not 0 < self.final_phase_threshold <= 1:
            raise ConfigurationError(
                "final_phase_threshold must be between 0 and 1",
                details={"final_phase_threshold": self.final_phase_threshold},
            )

    @classmethod
    def from_env(cls) -> GPUSchedulerConfig:
        """Create configuration from environment variables.

        Returns:
            GPUSchedulerConfig instance with values from environment.
        """
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> GPUSchedulerConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            GPUSchedulerConfig instance.

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

        return cls.from_dict(data.get("gpu_scheduler", data))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GPUSchedulerConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            GPUSchedulerConfig instance.
        """
        # Handle nested dictionaries separately
        phase_gpu_mapping = data.pop("phase_gpu_mapping", None)
        gpu_machine_types = data.pop("gpu_machine_types", None)
        gpu_costs = data.pop("gpu_costs", None)

        # Filter out unknown fields to prevent TypeError
        known_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        config = cls(**filtered_data)

        if phase_gpu_mapping:
            config.phase_gpu_mapping = phase_gpu_mapping
        if gpu_machine_types:
            config.gpu_machine_types = gpu_machine_types
        if gpu_costs:
            config.gpu_costs = gpu_costs

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "default_gpu": self.default_gpu,
            "final_phase_gpu": self.final_phase_gpu,
            "final_phase_threshold": self.final_phase_threshold,
            "prefer_spot": self.prefer_spot,
            "upgrade_enabled": self.upgrade_enabled,
            "early_phase_end": self.early_phase_end,
            "middle_phase_end": self.middle_phase_end,
            "late_phase_end": self.late_phase_end,
            "phase_gpu_mapping": self.phase_gpu_mapping,
            "gpu_machine_types": self.gpu_machine_types,
            "gpu_costs": self.gpu_costs,
            "spot_for_selfplay": self.spot_for_selfplay,
            "spot_for_training": self.spot_for_training,
        }


class DynamicGPUScheduler:
    """Dynamically selects GPU type based on training phase and cost.

    Implements T4 -> V100 upgrade strategy for final training phases.
    Considers cost budget, availability, and training requirements.

    Usage:
        config = GPUSchedulerConfig()
        scheduler = DynamicGPUScheduler(config)

        # Get GPU for current phase
        selection = scheduler.select_gpu(
            job_type="training",
            current_iteration=9000,
            total_iterations=10000
        )

        # Apply to job spec
        job_spec.accelerator_type = selection.gpu_type
        job_spec.machine_type = selection.machine_type
        job_spec.spot = selection.use_spot
    """

    def __init__(
        self,
        config: GPUSchedulerConfig | None = None,
        cost_monitor: CostMonitor | None = None,
    ) -> None:
        """Initialize GPU scheduler.

        Args:
            config: GPU scheduler configuration. Uses default if None.
            cost_monitor: Optional cost monitor for budget-aware decisions.
        """
        self.config = config or GPUSchedulerConfig()
        self.cost_monitor = cost_monitor

        logger.info(
            f"DynamicGPUScheduler initialized: "
            f"default={self.config.default_gpu}, "
            f"final={self.config.final_phase_gpu}, "
            f"threshold={self.config.final_phase_threshold}"
        )

    def get_training_phase(
        self,
        current_iteration: int,
        total_iterations: int,
    ) -> TrainingPhase:
        """Determine training phase based on iteration progress.

        Args:
            current_iteration: Current training iteration.
            total_iterations: Total planned iterations.

        Returns:
            TrainingPhase enum value.
        """
        if total_iterations <= 0:
            return TrainingPhase.EARLY

        progress = current_iteration / total_iterations

        if progress < self.config.early_phase_end:
            return TrainingPhase.EARLY
        elif progress < self.config.middle_phase_end:
            return TrainingPhase.MIDDLE
        elif progress < self.config.late_phase_end:
            return TrainingPhase.LATE
        else:
            return TrainingPhase.FINAL

    def should_upgrade(
        self,
        current_iteration: int,
        total_iterations: int,
    ) -> bool:
        """Check if GPU should be upgraded for final phase.

        Args:
            current_iteration: Current training iteration.
            total_iterations: Total planned iterations.

        Returns:
            True if GPU should be upgraded to final phase GPU.
        """
        if not self.config.upgrade_enabled:
            return False

        if total_iterations <= 0:
            return False

        progress = current_iteration / total_iterations
        return progress >= self.config.final_phase_threshold

    def select_gpu(
        self,
        job_type: str,
        current_iteration: int = 0,
        total_iterations: int = 1,
        budget_remaining: float | None = None,
    ) -> GPUSelection:
        """Select optimal GPU for a job.

        Args:
            job_type: Type of job ("selfplay" or "training").
            current_iteration: Current training iteration.
            total_iterations: Total planned iterations.
            budget_remaining: Optional remaining budget in USD.

        Returns:
            GPUSelection with selected GPU configuration.
        """
        phase = self.get_training_phase(current_iteration, total_iterations)

        # Determine GPU type based on phase
        if phase == TrainingPhase.FINAL and self.config.upgrade_enabled:
            gpu_type = self.config.final_phase_gpu
            reason = f"Final training phase ({phase.value}): using high-performance GPU"
        else:
            gpu_type = self.config.phase_gpu_mapping.get(phase.value, self.config.default_gpu)
            reason = f"Training phase {phase.value}: using configured GPU"

        # Determine spot usage
        if job_type == "selfplay":
            use_spot = self.config.spot_for_selfplay
        elif job_type == "training":
            # Training in final phase should not use spot
            if phase == TrainingPhase.FINAL:
                use_spot = False
                reason += "; spot disabled for final training"
            else:
                use_spot = self.config.spot_for_training
        else:
            use_spot = self.config.prefer_spot

        # Get machine type
        machine_type = self.config.gpu_machine_types.get(gpu_type, "n1-standard-8")

        # Calculate cost
        cost_key = f"{gpu_type}_SPOT" if use_spot else gpu_type
        hourly_cost = self.config.gpu_costs.get(cost_key, self.config.gpu_costs.get(gpu_type, 1.0))

        # Budget-aware selection
        if budget_remaining is not None and budget_remaining < hourly_cost:
            # Downgrade to cheaper GPU if budget is low
            cheaper_gpu = self._find_cheaper_gpu(hourly_cost, use_spot)
            if cheaper_gpu:
                gpu_type, hourly_cost = cheaper_gpu
                machine_type = self.config.gpu_machine_types.get(gpu_type, "n1-standard-8")
                reason = f"Budget constraint: downgraded to {gpu_type}"

        # Cost monitor integration
        if self.cost_monitor:
            status = self.cost_monitor.get_cost_status()

            if status.alert_level == AlertLevel.CRITICAL:
                # Force cheapest option in critical budget situation
                cheapest = self._get_cheapest_gpu(use_spot=True)
                if cheapest:
                    gpu_type, hourly_cost = cheapest
                    machine_type = self.config.gpu_machine_types.get(gpu_type, "n1-standard-8")
                    use_spot = True
                    reason = "Budget critical: using cheapest GPU with spot"

        selection = GPUSelection(
            gpu_type=gpu_type,
            machine_type=machine_type,
            use_spot=use_spot,
            reason=reason,
            estimated_hourly_cost=hourly_cost,
        )

        logger.info(
            f"GPU selection: job_type={job_type}, "
            f"iteration={current_iteration}/{total_iterations}, "
            f"phase={phase.value}, gpu={gpu_type}, "
            f"spot={use_spot}, cost=${hourly_cost:.2f}/hr"
        )

        return selection

    def _find_cheaper_gpu(
        self,
        max_cost: float,
        use_spot: bool,
    ) -> tuple[str, float] | None:
        """Find a cheaper GPU option.

        Args:
            max_cost: Maximum acceptable hourly cost.
            use_spot: Whether to consider spot pricing.

        Returns:
            Tuple of (gpu_type, cost) or None if no cheaper option.
        """
        candidates = []
        for gpu, cost in self.config.gpu_costs.items():
            if "_SPOT" in gpu and not use_spot:
                continue
            if "_SPOT" not in gpu and use_spot:
                continue
            if cost < max_cost:
                base_gpu = gpu.replace("_SPOT", "")
                candidates.append((base_gpu, cost))

        if candidates:
            # Return cheapest option
            return min(candidates, key=lambda x: x[1])
        return None

    def _get_cheapest_gpu(self, use_spot: bool = True) -> tuple[str, float] | None:
        """Get the cheapest available GPU.

        Args:
            use_spot: Whether to use spot pricing.

        Returns:
            Tuple of (gpu_type, cost) or None.
        """
        candidates = []
        for gpu, cost in self.config.gpu_costs.items():
            is_spot = "_SPOT" in gpu
            if is_spot == use_spot:
                base_gpu = gpu.replace("_SPOT", "")
                candidates.append((base_gpu, cost))

        if candidates:
            return min(candidates, key=lambda x: x[1])
        return None

    def get_gpu_for_phase(self, phase: TrainingPhase) -> str:
        """Get the configured GPU type for a training phase.

        Args:
            phase: Training phase.

        Returns:
            GPU type string.
        """
        return self.config.phase_gpu_mapping.get(phase.value, self.config.default_gpu)

    def get_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics for logging.

        Returns:
            Dictionary with scheduler configuration and state.
        """
        return {
            "default_gpu": self.config.default_gpu,
            "final_phase_gpu": self.config.final_phase_gpu,
            "final_phase_threshold": self.config.final_phase_threshold,
            "upgrade_enabled": self.config.upgrade_enabled,
            "prefer_spot": self.config.prefer_spot,
            "spot_for_selfplay": self.config.spot_for_selfplay,
            "spot_for_training": self.config.spot_for_training,
            "phase_gpu_mapping": self.config.phase_gpu_mapping,
            "has_cost_monitor": self.cost_monitor is not None,
        }

    def __repr__(self) -> str:
        return (
            f"DynamicGPUScheduler("
            f"default={self.config.default_gpu}, "
            f"final={self.config.final_phase_gpu}, "
            f"threshold={self.config.final_phase_threshold})"
        )
