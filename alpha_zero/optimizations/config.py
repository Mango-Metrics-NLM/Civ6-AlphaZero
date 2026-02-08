"""Unified configuration for performance optimizations.

Provides a single configuration class that combines all optimization
settings with preset configurations for common use cases.

Usage:
    from alpha_zero.optimizations.config import PerformanceOptimizationsConfig

    # Use preset
    config = PerformanceOptimizationsConfig.fast_preset()

    # Custom configuration
    config = PerformanceOptimizationsConfig(
        enable_playout_cap=True,
        enable_policy_pruning=True,
        playout_cap=PlayoutCapConfig(min_playouts=25),
    )

    # Load from YAML
    config = PerformanceOptimizationsConfig.from_yaml("configs/optimizations.yaml")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from alpha_zero.optimizations.auxiliary_targets import AuxiliaryTargetsConfig
from alpha_zero.optimizations.cyclic_lr import CyclicLRConfig
from alpha_zero.optimizations.playout_cap import PlayoutCapConfig
from alpha_zero.optimizations.policy_pruning import PolicyPruningConfig

logger = logging.getLogger(__name__)


@dataclass
class PerformanceOptimizationsConfig:
    """Unified configuration for all performance optimizations.

    This config combines settings for:
    - Playout cap randomization (MCTS speedup)
    - Policy target pruning (action space reduction)
    - Cyclic learning rates (training stability)
    - Auxiliary policy targets (richer training signals)

    All settings can be overridden via environment variables with
    the CIV6AZ_OPT_ prefix.

    Attributes:
        enable_playout_cap: Enable playout cap randomization
        enable_policy_pruning: Enable policy target pruning
        enable_cyclic_lr: Enable cyclic learning rate scheduler
        enable_auxiliary_targets: Enable auxiliary training targets
        playout_cap: Playout cap configuration
        policy_pruning: Policy pruning configuration
        cyclic_lr: Cyclic LR configuration
        auxiliary_targets: Auxiliary targets configuration
        log_optimization_stats: Whether to log optimization statistics
        stats_log_interval: How often to log stats (in steps)
    """

    enable_playout_cap: bool = True
    enable_policy_pruning: bool = True
    enable_cyclic_lr: bool = True
    enable_auxiliary_targets: bool = True
    playout_cap: PlayoutCapConfig = field(default_factory=PlayoutCapConfig)
    policy_pruning: PolicyPruningConfig = field(default_factory=PolicyPruningConfig)
    cyclic_lr: CyclicLRConfig = field(default_factory=CyclicLRConfig)
    auxiliary_targets: AuxiliaryTargetsConfig = field(
        default_factory=AuxiliaryTargetsConfig
    )
    log_optimization_stats: bool = True
    stats_log_interval: int = 1000

    def __post_init__(self) -> None:
        """Validate configuration and apply environment overrides."""
        self._apply_env_overrides()
        self._validate()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        prefix = "CIV6AZ_OPT_"

        # Boolean enables
        if os.environ.get(f"{prefix}PLAYOUT_CAP_ENABLED"):
            self.enable_playout_cap = (
                os.environ[f"{prefix}PLAYOUT_CAP_ENABLED"].lower() == "true"
            )
        if os.environ.get(f"{prefix}POLICY_PRUNING_ENABLED"):
            self.enable_policy_pruning = (
                os.environ[f"{prefix}POLICY_PRUNING_ENABLED"].lower() == "true"
            )
        if os.environ.get(f"{prefix}CYCLIC_LR_ENABLED"):
            self.enable_cyclic_lr = (
                os.environ[f"{prefix}CYCLIC_LR_ENABLED"].lower() == "true"
            )
        if os.environ.get(f"{prefix}AUXILIARY_TARGETS_ENABLED"):
            self.enable_auxiliary_targets = (
                os.environ[f"{prefix}AUXILIARY_TARGETS_ENABLED"].lower() == "true"
            )

        # Playout cap overrides
        if os.environ.get(f"{prefix}PLAYOUT_MIN"):
            self.playout_cap.min_playouts = int(os.environ[f"{prefix}PLAYOUT_MIN"])
        if os.environ.get(f"{prefix}PLAYOUT_MAX"):
            self.playout_cap.max_playouts = int(os.environ[f"{prefix}PLAYOUT_MAX"])

        # Cyclic LR overrides
        if os.environ.get(f"{prefix}CYCLIC_LR_TYPE"):
            self.cyclic_lr.schedule_type = os.environ[f"{prefix}CYCLIC_LR_TYPE"]
        if os.environ.get(f"{prefix}CYCLIC_LR_BASE"):
            self.cyclic_lr.base_lr = float(os.environ[f"{prefix}CYCLIC_LR_BASE"])
        if os.environ.get(f"{prefix}CYCLIC_LR_MAX"):
            self.cyclic_lr.max_lr = float(os.environ[f"{prefix}CYCLIC_LR_MAX"])

    def _validate(self) -> None:
        """Validate configuration consistency."""
        if self.stats_log_interval < 1:
            raise ValueError(
                f"stats_log_interval must be >= 1, got {self.stats_log_interval}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PerformanceOptimizationsConfig:
        """Create from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            PerformanceOptimizationsConfig instance
        """
        config_data = {}

        # Handle simple fields
        for key in [
            "enable_playout_cap",
            "enable_policy_pruning",
            "enable_cyclic_lr",
            "enable_auxiliary_targets",
            "log_optimization_stats",
            "stats_log_interval",
        ]:
            if key in data:
                config_data[key] = data[key]

        # Handle nested configs
        if "playout_cap" in data:
            config_data["playout_cap"] = PlayoutCapConfig.from_dict(data["playout_cap"])
        if "policy_pruning" in data:
            config_data["policy_pruning"] = PolicyPruningConfig.from_dict(
                data["policy_pruning"]
            )
        if "cyclic_lr" in data:
            config_data["cyclic_lr"] = CyclicLRConfig.from_dict(data["cyclic_lr"])
        if "auxiliary_targets" in data:
            config_data["auxiliary_targets"] = AuxiliaryTargetsConfig.from_dict(
                data["auxiliary_targets"]
            )

        return cls(**config_data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> PerformanceOptimizationsConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            PerformanceOptimizationsConfig instance
        """
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "enable_playout_cap": self.enable_playout_cap,
            "enable_policy_pruning": self.enable_policy_pruning,
            "enable_cyclic_lr": self.enable_cyclic_lr,
            "enable_auxiliary_targets": self.enable_auxiliary_targets,
            "playout_cap": self.playout_cap.to_dict(),
            "policy_pruning": self.policy_pruning.to_dict(),
            "cyclic_lr": self.cyclic_lr.to_dict(),
            "auxiliary_targets": self.auxiliary_targets.to_dict(),
            "log_optimization_stats": self.log_optimization_stats,
            "stats_log_interval": self.stats_log_interval,
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def fast_preset(cls) -> PerformanceOptimizationsConfig:
        """Create configuration optimized for speed.

        This preset maximizes training throughput at some cost to
        final model quality. Useful for rapid iteration and debugging.

        Returns:
            Fast optimization configuration
        """
        return cls(
            enable_playout_cap=True,
            enable_policy_pruning=True,
            enable_cyclic_lr=True,
            enable_auxiliary_targets=False,  # Disabled for speed
            playout_cap=PlayoutCapConfig(
                min_playouts=25,
                max_playouts=200,
                cap_distribution="geometric",
                geometric_p=0.4,  # More aggressive
                use_position_complexity=False,  # Disabled for speed
            ),
            policy_pruning=PolicyPruningConfig(
                min_probability=0.005,  # More aggressive
                max_actions=50,
                preserve_top_k=3,
                use_entropy_threshold=False,  # Disabled for speed
            ),
            cyclic_lr=CyclicLRConfig(
                schedule_type="triangular",  # Simpler schedule
                base_lr=1e-5,
                max_lr=1e-3,
                step_size_up=2000,  # Faster cycles
                warmup_steps=500,
            ),
            log_optimization_stats=False,  # Disabled for speed
        )

    @classmethod
    def quality_preset(cls) -> PerformanceOptimizationsConfig:
        """Create configuration optimized for quality.

        This preset prioritizes final model quality over training speed.
        Useful for final training runs and production models.

        Returns:
            Quality optimization configuration
        """
        return cls(
            enable_playout_cap=True,
            enable_policy_pruning=True,
            enable_cyclic_lr=True,
            enable_auxiliary_targets=True,
            playout_cap=PlayoutCapConfig(
                min_playouts=100,
                max_playouts=800,
                cap_distribution="geometric",
                geometric_p=0.2,  # More conservative
                use_position_complexity=True,
            ),
            policy_pruning=PolicyPruningConfig(
                min_probability=0.001,
                max_actions=100,
                preserve_top_k=5,
                use_entropy_threshold=True,
            ),
            cyclic_lr=CyclicLRConfig(
                schedule_type="warm_restarts",
                base_lr=1e-6,
                max_lr=5e-4,
                step_size_up=10000,
                warmup_steps=2000,
                t_mult=2.0,
            ),
            auxiliary_targets=AuxiliaryTargetsConfig(
                enable_attention_targets=True,
                enable_opponent_modeling=True,
                enable_phase_prediction=True,
                enable_future_value=True,
            ),
            log_optimization_stats=True,
            stats_log_interval=500,
        )

    @classmethod
    def balanced_preset(cls) -> PerformanceOptimizationsConfig:
        """Create balanced configuration.

        This preset provides a good balance between speed and quality.
        Recommended as a starting point for most use cases.

        Returns:
            Balanced optimization configuration
        """
        return cls(
            enable_playout_cap=True,
            enable_policy_pruning=True,
            enable_cyclic_lr=True,
            enable_auxiliary_targets=True,
            playout_cap=PlayoutCapConfig(
                min_playouts=50,
                max_playouts=400,
                cap_distribution="geometric",
                geometric_p=0.3,
                use_position_complexity=True,
            ),
            policy_pruning=PolicyPruningConfig(
                min_probability=0.002,
                max_actions=75,
                preserve_top_k=4,
                use_entropy_threshold=True,
            ),
            cyclic_lr=CyclicLRConfig(
                schedule_type="cosine_annealing",
                base_lr=1e-5,
                max_lr=1e-3,
                step_size_up=5000,
                warmup_steps=1000,
            ),
            auxiliary_targets=AuxiliaryTargetsConfig(
                enable_attention_targets=True,
                enable_opponent_modeling=True,
                enable_phase_prediction=False,  # Simpler
                enable_future_value=False,  # Simpler
            ),
            log_optimization_stats=True,
            stats_log_interval=1000,
        )

    @classmethod
    def minimal_preset(cls) -> PerformanceOptimizationsConfig:
        """Create minimal configuration with all optimizations disabled.

        Useful for baseline comparisons or debugging.

        Returns:
            Minimal configuration with no optimizations
        """
        return cls(
            enable_playout_cap=False,
            enable_policy_pruning=False,
            enable_cyclic_lr=False,
            enable_auxiliary_targets=False,
            log_optimization_stats=False,
        )

    def get_enabled_optimizations(self) -> list[str]:
        """Get list of enabled optimization names.

        Returns:
            List of enabled optimization names
        """
        enabled = []
        if self.enable_playout_cap:
            enabled.append("playout_cap")
        if self.enable_policy_pruning:
            enabled.append("policy_pruning")
        if self.enable_cyclic_lr:
            enabled.append("cyclic_lr")
        if self.enable_auxiliary_targets:
            enabled.append("auxiliary_targets")
        return enabled

    def get_summary(self) -> str:
        """Get human-readable configuration summary.

        Returns:
            Summary string
        """
        enabled = self.get_enabled_optimizations()
        lines = [
            "Performance Optimizations Configuration",
            "=" * 40,
            f"Enabled: {', '.join(enabled) if enabled else 'None'}",
            "",
        ]

        if self.enable_playout_cap:
            lines.extend(
                [
                    "Playout Cap:",
                    f"  Range: {self.playout_cap.min_playouts}-{self.playout_cap.max_playouts}",
                    f"  Distribution: {self.playout_cap.cap_distribution}",
                    f"  Complexity scaling: {self.playout_cap.use_position_complexity}",
                    "",
                ]
            )

        if self.enable_policy_pruning:
            lines.extend(
                [
                    "Policy Pruning:",
                    f"  Min probability: {self.policy_pruning.min_probability}",
                    f"  Max actions: {self.policy_pruning.max_actions}",
                    f"  Top-k preserved: {self.policy_pruning.preserve_top_k}",
                    "",
                ]
            )

        if self.enable_cyclic_lr:
            lines.extend(
                [
                    "Cyclic LR:",
                    f"  Schedule: {self.cyclic_lr.schedule_type}",
                    f"  LR range: {self.cyclic_lr.base_lr} - {self.cyclic_lr.max_lr}",
                    f"  Step size: {self.cyclic_lr.step_size_up}",
                    "",
                ]
            )

        if self.enable_auxiliary_targets:
            aux = self.auxiliary_targets
            aux_enabled = []
            if aux.enable_attention_targets:
                aux_enabled.append("attention")
            if aux.enable_opponent_modeling:
                aux_enabled.append("opponent")
            if aux.enable_phase_prediction:
                aux_enabled.append("phase")
            if aux.enable_future_value:
                aux_enabled.append("future_value")
            lines.extend(
                [
                    "Auxiliary Targets:",
                    f"  Enabled: {', '.join(aux_enabled)}",
                    f"  Total weight: {aux.get_total_weight():.3f}",
                    "",
                ]
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        enabled = self.get_enabled_optimizations()
        return f"PerformanceOptimizationsConfig(enabled={enabled})"
