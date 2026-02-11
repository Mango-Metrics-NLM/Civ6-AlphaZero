"""
Budget Tier Scaling Configuration for Civ6 AlphaZero Cloud Training.

Provides configurable scaling knobs for different budget tiers:
- Budget ($650/mo): Cost-optimized with Spot T4 GPUs
- Standard ($1,200/mo): Balanced performance and cost
- Burst (on-demand): Maximum performance for exploration

All values configurable via environment variables with CIV6AZ_SCALING_ prefix.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from alpha_zero.cloud.gcp.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class BudgetTier(str, Enum):
    """Budget tier enumeration for scaling configuration."""

    BUDGET = "budget"  # $650/mo - Cost-optimized
    STANDARD = "standard"  # $1,200/mo - Balanced
    BURST = "burst"  # On-demand - Maximum performance


@dataclass
class GPUSpec:
    """GPU specification for a tier.

    Attributes:
        accelerator_type: GPU type (e.g., NVIDIA_TESLA_T4, NVIDIA_L4).
        accelerator_count: Number of GPUs.
        use_spot: Whether to use Spot VMs for cost savings.
    """

    accelerator_type: str
    accelerator_count: int = 1
    use_spot: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accelerator_type": self.accelerator_type,
            "accelerator_count": self.accelerator_count,
            "use_spot": self.use_spot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GPUSpec:
        """Create from dictionary."""
        return cls(
            accelerator_type=data["accelerator_type"],
            accelerator_count=data.get("accelerator_count", 1),
            use_spot=data.get("use_spot", False),
        )


@dataclass
class SelfPlayTierConfig:
    """Self-play worker configuration per tier.

    Attributes:
        worker_count: Number of parallel self-play workers.
        gpu_spec: GPU specification for workers.
        games_per_cycle: Total games to generate per training cycle.
        machine_type: GCE machine type for workers.
        max_retries: Maximum job retries on failure.
    """

    worker_count: int
    gpu_spec: GPUSpec
    games_per_cycle: int
    machine_type: str = "n1-standard-8"
    max_retries: int = 3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "worker_count": self.worker_count,
            "gpu_spec": self.gpu_spec.to_dict(),
            "games_per_cycle": self.games_per_cycle,
            "machine_type": self.machine_type,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelfPlayTierConfig:
        """Create from dictionary."""
        gpu_spec_data = data.get("gpu_spec", {})
        gpu_spec = GPUSpec.from_dict(gpu_spec_data) if gpu_spec_data else GPUSpec("NVIDIA_TESLA_T4")
        return cls(
            worker_count=data["worker_count"],
            gpu_spec=gpu_spec,
            games_per_cycle=data["games_per_cycle"],
            machine_type=data.get("machine_type", "n1-standard-8"),
            max_retries=data.get("max_retries", 3),
        )


@dataclass
class TrainingTierConfig:
    """Training configuration per tier.

    Attributes:
        gpu_spec: GPU specification for training.
        training_steps_per_cycle: Training steps per cycle.
        machine_type: GCE machine type for training.
        batch_size: Training batch size.
    """

    gpu_spec: GPUSpec
    training_steps_per_cycle: int
    machine_type: str = "g2-standard-8"
    batch_size: int = 256

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu_spec": self.gpu_spec.to_dict(),
            "training_steps_per_cycle": self.training_steps_per_cycle,
            "machine_type": self.machine_type,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingTierConfig:
        """Create from dictionary."""
        gpu_spec_data = data.get("gpu_spec", {})
        gpu_spec = GPUSpec.from_dict(gpu_spec_data) if gpu_spec_data else GPUSpec("NVIDIA_L4")
        return cls(
            gpu_spec=gpu_spec,
            training_steps_per_cycle=data["training_steps_per_cycle"],
            machine_type=data.get("machine_type", "g2-standard-8"),
            batch_size=data.get("batch_size", 256),
        )


@dataclass
class EvaluationTierConfig:
    """Evaluation configuration per tier.

    Attributes:
        eval_games: Number of evaluation games per cycle.
        parallel_games: Number of games to run in parallel.
        early_stop_threshold: Games threshold for early stopping decision.
    """

    eval_games: int
    parallel_games: int = 4
    early_stop_threshold: int = 200

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "eval_games": self.eval_games,
            "parallel_games": self.parallel_games,
            "early_stop_threshold": self.early_stop_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationTierConfig:
        """Create from dictionary."""
        return cls(
            eval_games=data["eval_games"],
            parallel_games=data.get("parallel_games", 4),
            early_stop_threshold=data.get("early_stop_threshold", 200),
        )


@dataclass
class CycleTierConfig:
    """Cycle timing configuration per tier.

    Attributes:
        frequency_hours: Hours between training cycles.
        min_games_before_training: Minimum games before triggering training.
        max_queue_size: Maximum trajectory queue size.
    """

    frequency_hours: float
    min_games_before_training: int
    max_queue_size: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frequency_hours": self.frequency_hours,
            "min_games_before_training": self.min_games_before_training,
            "max_queue_size": self.max_queue_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CycleTierConfig:
        """Create from dictionary."""
        return cls(
            frequency_hours=data["frequency_hours"],
            min_games_before_training=data["min_games_before_training"],
            max_queue_size=data["max_queue_size"],
        )


@dataclass
class ScalingTierConfig:
    """Complete scaling configuration for a budget tier.

    Provides all scaling knobs for self-play, training, evaluation, and
    cycle timing based on the selected budget tier.

    Environment Variables:
        CIV6AZ_SCALING_TIER: Budget tier (budget, standard, burst)
        CIV6AZ_SCALING_MONTHLY_BUDGET: Monthly budget in USD
        CIV6AZ_SCALING_SELFPLAY_WORKERS: Override self-play worker count
        CIV6AZ_SCALING_GAMES_PER_CYCLE: Override games per cycle
        CIV6AZ_SCALING_TRAINING_STEPS: Override training steps per cycle
        CIV6AZ_SCALING_EVAL_GAMES: Override evaluation games
        CIV6AZ_SCALING_CYCLE_HOURS: Override cycle frequency in hours
    """

    tier: BudgetTier
    monthly_budget_usd: float
    self_play: SelfPlayTierConfig
    training: TrainingTierConfig
    evaluation: EvaluationTierConfig
    cycle: CycleTierConfig

    # Alert thresholds (as fraction of monthly budget)
    warning_threshold: float = 0.8
    critical_threshold: float = 1.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        errors: list[str] = []

        if self.monthly_budget_usd <= 0 and self.tier != BudgetTier.BURST:
            errors.append("monthly_budget_usd must be positive")

        if self.self_play.worker_count < 1:
            errors.append("self_play.worker_count must be at least 1")

        if self.self_play.games_per_cycle < 1:
            errors.append("self_play.games_per_cycle must be at least 1")

        if self.training.training_steps_per_cycle < 1:
            errors.append("training.training_steps_per_cycle must be at least 1")

        if self.evaluation.eval_games < 1:
            errors.append("evaluation.eval_games must be at least 1")

        if self.cycle.frequency_hours <= 0:
            errors.append("cycle.frequency_hours must be positive")

        if not 0 < self.warning_threshold <= 1:
            errors.append("warning_threshold must be between 0 and 1")

        if not 0 < self.critical_threshold <= 2:
            errors.append("critical_threshold must be between 0 and 2")

        if errors:
            raise ConfigurationError(
                f"Invalid scaling configuration: {'; '.join(errors)}",
                details={"errors": errors},
            )

    @classmethod
    def budget_preset(cls) -> ScalingTierConfig:
        """Create $650/month budget tier preset.

        Optimized for cost with Spot T4 GPUs and reduced throughput.
        """
        return cls(
            tier=BudgetTier.BUDGET,
            monthly_budget_usd=650.0,
            self_play=SelfPlayTierConfig(
                worker_count=3,
                gpu_spec=GPUSpec("NVIDIA_TESLA_T4", 1, use_spot=True),
                games_per_cycle=300,
                machine_type="n1-standard-8",
            ),
            training=TrainingTierConfig(
                gpu_spec=GPUSpec("NVIDIA_L4", 1, use_spot=False),
                training_steps_per_cycle=500,
                machine_type="g2-standard-8",
            ),
            evaluation=EvaluationTierConfig(
                eval_games=200,
                parallel_games=4,
                early_stop_threshold=200,
            ),
            cycle=CycleTierConfig(
                frequency_hours=8.0,
                min_games_before_training=300,
                max_queue_size=1000,
            ),
        )

    @classmethod
    def standard_preset(cls) -> ScalingTierConfig:
        """Create $1,200/month standard tier preset.

        Balanced performance and cost with more workers and throughput.
        """
        return cls(
            tier=BudgetTier.STANDARD,
            monthly_budget_usd=1200.0,
            self_play=SelfPlayTierConfig(
                worker_count=8,
                gpu_spec=GPUSpec("NVIDIA_TESLA_T4", 1, use_spot=True),
                games_per_cycle=800,
                machine_type="n1-standard-8",
            ),
            training=TrainingTierConfig(
                gpu_spec=GPUSpec("NVIDIA_L4", 2, use_spot=False),
                training_steps_per_cycle=1000,
                machine_type="g2-standard-24",
            ),
            evaluation=EvaluationTierConfig(
                eval_games=400,
                parallel_games=4,
                early_stop_threshold=200,
            ),
            cycle=CycleTierConfig(
                frequency_hours=6.0,
                min_games_before_training=500,
                max_queue_size=2000,
            ),
        )

    @classmethod
    def burst_preset(cls) -> ScalingTierConfig:
        """Create burst/on-demand tier preset.

        Maximum performance with A100 GPU and large self-play fleet.
        No strict budget cap - use for exploration and rapid iteration.
        """
        return cls(
            tier=BudgetTier.BURST,
            monthly_budget_usd=float("inf"),
            self_play=SelfPlayTierConfig(
                worker_count=15,
                gpu_spec=GPUSpec("NVIDIA_L4", 1, use_spot=True),
                games_per_cycle=1500,
                machine_type="g2-standard-8",
            ),
            training=TrainingTierConfig(
                gpu_spec=GPUSpec("NVIDIA_TESLA_A100", 1, use_spot=False),
                training_steps_per_cycle=2000,
                machine_type="a2-highgpu-1g",
                batch_size=512,
            ),
            evaluation=EvaluationTierConfig(
                eval_games=400,
                parallel_games=8,
                early_stop_threshold=200,
            ),
            cycle=CycleTierConfig(
                frequency_hours=4.0,
                min_games_before_training=800,
                max_queue_size=5000,
            ),
            warning_threshold=0.9,  # Less strict for burst mode
            critical_threshold=1.5,
        )

    @classmethod
    def from_env(cls) -> ScalingTierConfig:
        """Create configuration from environment variables.

        Loads base tier preset and applies any environment variable overrides.

        Returns:
            ScalingTierConfig with environment overrides applied.
        """
        tier_name = os.environ.get("CIV6AZ_SCALING_TIER", "budget").lower()
        try:
            tier = BudgetTier(tier_name)
        except ValueError:
            logger.warning(f"Unknown tier '{tier_name}', defaulting to budget")
            tier = BudgetTier.BUDGET

        # Load base preset
        if tier == BudgetTier.BUDGET:
            config = cls.budget_preset()
        elif tier == BudgetTier.STANDARD:
            config = cls.standard_preset()
        else:
            config = cls.burst_preset()

        # Apply environment overrides
        config = config._apply_env_overrides()

        return config

    def _apply_env_overrides(self) -> ScalingTierConfig:
        """Apply environment variable overrides to configuration.

        Returns:
            New configuration with overrides applied.
        """
        # Monthly budget override
        if budget_str := os.environ.get("CIV6AZ_SCALING_MONTHLY_BUDGET"):
            self.monthly_budget_usd = float(budget_str)

        # Self-play overrides
        if workers_str := os.environ.get("CIV6AZ_SCALING_SELFPLAY_WORKERS"):
            self.self_play.worker_count = int(workers_str)

        if games_str := os.environ.get("CIV6AZ_SCALING_GAMES_PER_CYCLE"):
            self.self_play.games_per_cycle = int(games_str)

        # Training overrides
        if steps_str := os.environ.get("CIV6AZ_SCALING_TRAINING_STEPS"):
            self.training.training_steps_per_cycle = int(steps_str)

        # Evaluation overrides
        if eval_str := os.environ.get("CIV6AZ_SCALING_EVAL_GAMES"):
            self.evaluation.eval_games = int(eval_str)

        # Cycle overrides
        if cycle_str := os.environ.get("CIV6AZ_SCALING_CYCLE_HOURS"):
            self.cycle.frequency_hours = float(cycle_str)

        # Threshold overrides
        if warn_str := os.environ.get("CIV6AZ_SCALING_WARNING_THRESHOLD"):
            self.warning_threshold = float(warn_str)

        if crit_str := os.environ.get("CIV6AZ_SCALING_CRITICAL_THRESHOLD"):
            self.critical_threshold = float(crit_str)

        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ScalingTierConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            ScalingTierConfig instance.

        Raises:
            ConfigurationError: If file is invalid or missing.
        """
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Scaling configuration file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScalingTierConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            ScalingTierConfig instance.
        """
        tier_str = data.get("tier", "budget")
        tier = BudgetTier(tier_str) if isinstance(tier_str, str) else tier_str

        return cls(
            tier=tier,
            monthly_budget_usd=data.get("monthly_budget_usd", 650.0),
            self_play=SelfPlayTierConfig.from_dict(data["self_play"]),
            training=TrainingTierConfig.from_dict(data["training"]),
            evaluation=EvaluationTierConfig.from_dict(data["evaluation"]),
            cycle=CycleTierConfig.from_dict(data["cycle"]),
            warning_threshold=data.get("warning_threshold", 0.8),
            critical_threshold=data.get("critical_threshold", 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "tier": self.tier.value,
            "monthly_budget_usd": self.monthly_budget_usd,
            "self_play": self.self_play.to_dict(),
            "training": self.training.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "cycle": self.cycle.to_dict(),
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save configuration.
        """
        path = Path(path)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False)

    def get_estimated_monthly_cost(self) -> dict[str, float]:
        """Estimate monthly cost breakdown based on configuration.

        Returns:
            Dictionary with cost estimates by component.
        """
        # GPU hourly costs (approximate)
        gpu_costs = {
            "NVIDIA_TESLA_T4": 0.35,
            "NVIDIA_TESLA_T4_SPOT": 0.11,
            "NVIDIA_L4": 0.81,
            "NVIDIA_L4_SPOT": 0.24,
            "NVIDIA_TESLA_A100": 3.67,
            "NVIDIA_TESLA_A100_SPOT": 1.10,
        }

        # Calculate hours per month based on cycle frequency
        cycles_per_month = 30 * 24 / self.cycle.frequency_hours

        # Self-play cost (estimate ~3 hours per cycle for all workers combined)
        selfplay_hours = 3.0 * cycles_per_month
        selfplay_gpu = self.self_play.gpu_spec.accelerator_type
        if self.self_play.gpu_spec.use_spot:
            selfplay_gpu += "_SPOT"
        selfplay_cost = (
            selfplay_hours
            * self.self_play.worker_count
            * self.self_play.gpu_spec.accelerator_count
            * gpu_costs.get(selfplay_gpu, 0.5)
        )

        # Training cost (estimate ~3 hours per cycle)
        training_hours = 3.0 * cycles_per_month
        training_gpu = self.training.gpu_spec.accelerator_type
        if self.training.gpu_spec.use_spot:
            training_gpu += "_SPOT"
        training_cost = (
            training_hours
            * self.training.gpu_spec.accelerator_count
            * gpu_costs.get(training_gpu, 1.0)
        )

        # Evaluation cost (estimate ~1.5 hours per cycle)
        eval_hours = 1.5 * cycles_per_month
        eval_cost = eval_hours * gpu_costs.get("NVIDIA_L4_SPOT", 0.24)

        # Storage cost (estimate 500GB at $0.02/GB)
        storage_cost = 500 * 0.02

        # Misc costs (pipelines, networking)
        misc_cost = 20.0

        total = selfplay_cost + training_cost + eval_cost + storage_cost + misc_cost

        return {
            "self_play": round(selfplay_cost, 2),
            "training": round(training_cost, 2),
            "evaluation": round(eval_cost, 2),
            "storage": round(storage_cost, 2),
            "misc": round(misc_cost, 2),
            "total": round(total, 2),
            "budget": self.monthly_budget_usd,
            "within_budget": total <= self.monthly_budget_usd,
        }

    def __repr__(self) -> str:
        return (
            f"ScalingTierConfig(tier={self.tier.value!r}, "
            f"budget=${self.monthly_budget_usd:.0f}/mo, "
            f"workers={self.self_play.worker_count}, "
            f"games={self.self_play.games_per_cycle})"
        )


def get_scaling_config() -> ScalingTierConfig:
    """Get the current scaling configuration from environment.

    Convenience function for getting the current scaling configuration.

    Returns:
        ScalingTierConfig loaded from environment.
    """
    return ScalingTierConfig.from_env()
