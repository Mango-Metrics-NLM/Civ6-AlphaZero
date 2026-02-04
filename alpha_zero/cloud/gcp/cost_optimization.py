"""Unified cost optimization manager for GCP cloud training.

This module provides a unified interface for all cost optimization
components, coordinating spot VMs, GPU scheduling, idle management,
preemption handling, and budget enforcement.

Usage:
    from alpha_zero.cloud.gcp.cost_optimization import (
        CostOptimizationConfig,
        CostOptimizationManager,
    )

    config = CostOptimizationConfig()
    manager = CostOptimizationManager(config)

    # Initialize all components
    await manager.initialize()

    # Get optimized job configuration
    job_config = manager.get_optimized_job_config(
        job_type="self_play",
        iteration=500,
        total_iterations=1000,
    )

    # Start monitoring
    await manager.start_monitoring()
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from alpha_zero.cloud.gcp.cost_monitor import CostMonitor, CostMonitorConfig
from alpha_zero.cloud.gcp.gpu_scheduler import DynamicGPUScheduler, GPUSchedulerConfig
from alpha_zero.cloud.gcp.idle_manager import IdleInstanceConfig, IdleInstanceManager
from alpha_zero.cloud.gcp.preemption_handler import PreemptionHandler, PreemptionHandlerConfig
from alpha_zero.cloud.gcp.quota_manager import QuotaManager, QuotaManagerConfig

logger = logging.getLogger(__name__)


@dataclass
class JobSpec:
    """Specification for an optimized job.

    Attributes:
        gpu_type: GPU type to use.
        gpu_count: Number of GPUs.
        use_spot: Whether to use spot/preemptible VMs.
        region: GCP region.
        machine_type: Machine type.
        checkpoint_interval: Checkpoint interval in seconds.
        estimated_cost_per_hour: Estimated cost per hour.
        reason: Reason for this configuration.
    """

    gpu_type: str
    gpu_count: int = 1
    use_spot: bool = True
    region: str = "us-central1"
    machine_type: str = "n1-highmem-8"
    checkpoint_interval: int = 300
    estimated_cost_per_hour: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "use_spot": self.use_spot,
            "region": self.region,
            "machine_type": self.machine_type,
            "checkpoint_interval": self.checkpoint_interval,
            "estimated_cost_per_hour": self.estimated_cost_per_hour,
            "reason": self.reason,
        }


@dataclass
class SavingsReport:
    """Report of cost savings from optimization.

    Attributes:
        total_cost: Total cost incurred.
        estimated_full_price: Estimated cost at full price.
        total_savings: Total savings achieved.
        savings_percentage: Savings as percentage.
        spot_savings: Savings from spot VMs.
        idle_savings: Savings from idle management.
        gpu_optimization_savings: Savings from GPU optimization.
        hours_run: Total hours of compute.
        preemptions_handled: Number of preemptions handled.
        checkpoints_saved: Number of checkpoints saved.
    """

    total_cost: float = 0.0
    estimated_full_price: float = 0.0
    total_savings: float = 0.0
    savings_percentage: float = 0.0
    spot_savings: float = 0.0
    idle_savings: float = 0.0
    gpu_optimization_savings: float = 0.0
    hours_run: float = 0.0
    preemptions_handled: int = 0
    checkpoints_saved: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_cost": self.total_cost,
            "estimated_full_price": self.estimated_full_price,
            "total_savings": self.total_savings,
            "savings_percentage": self.savings_percentage,
            "spot_savings": self.spot_savings,
            "idle_savings": self.idle_savings,
            "gpu_optimization_savings": self.gpu_optimization_savings,
            "hours_run": self.hours_run,
            "preemptions_handled": self.preemptions_handled,
            "checkpoints_saved": self.checkpoints_saved,
        }


@dataclass
class CostOptimizationConfig:
    """Configuration for the unified cost optimization manager.

    Attributes:
        enabled: Whether cost optimization is enabled.
        cost_monitor: Cost monitor configuration.
        gpu_scheduler: GPU scheduler configuration.
        idle_manager: Idle instance manager configuration.
        preemption_handler: Preemption handler configuration.
        quota_manager: Quota manager configuration.
        default_region: Default GCP region.
        enable_spot_vms: Enable spot/preemptible VMs.
        fallback_to_on_demand: Fallback to on-demand if spot unavailable.
    """

    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_COST_OPT_ENABLED", "true"
        ).lower() == "true"
    )
    cost_monitor: CostMonitorConfig = field(default_factory=CostMonitorConfig)
    gpu_scheduler: GPUSchedulerConfig = field(default_factory=GPUSchedulerConfig)
    idle_manager: IdleInstanceConfig = field(default_factory=IdleInstanceConfig)
    preemption_handler: PreemptionHandlerConfig = field(
        default_factory=PreemptionHandlerConfig
    )
    quota_manager: QuotaManagerConfig = field(default_factory=QuotaManagerConfig)
    default_region: str = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_DEFAULT_REGION", "us-central1"
        )
    )
    enable_spot_vms: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_ENABLE_SPOT_VMS", "true"
        ).lower() == "true"
    )
    fallback_to_on_demand: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_FALLBACK_ON_DEMAND", "true"
        ).lower() == "true"
    )

    @classmethod
    def from_env(cls) -> CostOptimizationConfig:
        """Create configuration from environment variables."""
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> CostOptimizationConfig:
        """Load configuration from YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if "cost_optimization" in data:
            data = data["cost_optimization"]

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostOptimizationConfig:
        """Create configuration from dictionary."""
        kwargs: dict[str, Any] = {}

        if "enabled" in data:
            kwargs["enabled"] = bool(data["enabled"])
        if "default_region" in data:
            kwargs["default_region"] = str(data["default_region"])
        if "enable_spot_vms" in data:
            kwargs["enable_spot_vms"] = bool(data["enable_spot_vms"])
        if "fallback_to_on_demand" in data:
            kwargs["fallback_to_on_demand"] = bool(data["fallback_to_on_demand"])

        # Sub-configurations
        if "cost_monitor" in data:
            kwargs["cost_monitor"] = CostMonitorConfig.from_dict(data["cost_monitor"])
        if "gpu_scheduler" in data:
            kwargs["gpu_scheduler"] = GPUSchedulerConfig.from_dict(data["gpu_scheduler"])
        if "idle_manager" in data:
            kwargs["idle_manager"] = IdleInstanceConfig.from_dict(data["idle_manager"])
        if "preemption_handler" in data:
            kwargs["preemption_handler"] = PreemptionHandlerConfig.from_dict(
                data["preemption_handler"]
            )
        if "quota_manager" in data:
            kwargs["quota_manager"] = QuotaManagerConfig.from_dict(data["quota_manager"])

        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "default_region": self.default_region,
            "enable_spot_vms": self.enable_spot_vms,
            "fallback_to_on_demand": self.fallback_to_on_demand,
            "cost_monitor": self.cost_monitor.to_dict(),
            "gpu_scheduler": self.gpu_scheduler.to_dict(),
            "idle_manager": self.idle_manager.to_dict(),
            "preemption_handler": self.preemption_handler.to_dict(),
            "quota_manager": self.quota_manager.to_dict(),
        }


class CostOptimizationManager:
    """Unified manager for all cost optimization components.

    This class coordinates cost monitoring, GPU scheduling, idle management,
    preemption handling, and quota management to achieve optimal cost
    efficiency while maintaining training quality.

    Attributes:
        config: Cost optimization configuration.
        cost_monitor: Cost monitoring component.
        gpu_scheduler: GPU scheduling component.
        idle_manager: Idle instance management component.
        preemption_handler: Preemption handling component.
        quota_manager: Quota management component.
    """

    def __init__(
        self,
        config: CostOptimizationConfig | None = None,
        billing_client: Any | None = None,
        storage_backend: Any | None = None,
    ):
        """Initialize cost optimization manager.

        Args:
            config: Cost optimization configuration.
            billing_client: Optional billing client (for testing).
            storage_backend: Optional storage backend (for testing).
        """
        self.config = config or CostOptimizationConfig()
        self._billing_client = billing_client
        self._storage_backend = storage_backend

        # Initialize components (lazy initialization)
        self._cost_monitor: CostMonitor | None = None
        self._gpu_scheduler: DynamicGPUScheduler | None = None
        self._idle_manager: IdleInstanceManager | None = None
        self._preemption_handler: PreemptionHandler | None = None
        self._quota_manager: QuotaManager | None = None

        # State tracking
        self._initialized = False
        self._monitoring = False
        self._start_time: float = 0.0

        # Savings tracking
        self._spot_hours: float = 0.0
        self._on_demand_hours: float = 0.0
        self._idle_hours_saved: float = 0.0
        self._preemptions: int = 0
        self._checkpoints: int = 0

        logger.info(
            f"CostOptimizationManager created: enabled={self.config.enabled}, "
            f"spot_vms={self.config.enable_spot_vms}"
        )

    @property
    def cost_monitor(self) -> CostMonitor:
        """Get cost monitor instance."""
        if self._cost_monitor is None:
            self._cost_monitor = CostMonitor(
                self.config.cost_monitor,
            )
        return self._cost_monitor

    @property
    def gpu_scheduler(self) -> DynamicGPUScheduler:
        """Get GPU scheduler instance."""
        if self._gpu_scheduler is None:
            self._gpu_scheduler = DynamicGPUScheduler(
                self.config.gpu_scheduler,
                cost_monitor=self.cost_monitor,
            )
        return self._gpu_scheduler

    @property
    def idle_manager(self) -> IdleInstanceManager:
        """Get idle manager instance."""
        if self._idle_manager is None:
            self._idle_manager = IdleInstanceManager(self.config.idle_manager)
        return self._idle_manager

    @property
    def preemption_handler(self) -> PreemptionHandler:
        """Get preemption handler instance."""
        if self._preemption_handler is None:
            self._preemption_handler = PreemptionHandler(
                self.config.preemption_handler,
            )
        return self._preemption_handler

    @property
    def quota_manager(self) -> QuotaManager:
        """Get quota manager instance."""
        if self._quota_manager is None:
            self._quota_manager = QuotaManager(self.config.quota_manager)
        return self._quota_manager

    async def initialize(self) -> bool:
        """Initialize all cost optimization components.

        Returns:
            True if initialization successful.
        """
        if not self.config.enabled:
            logger.info("Cost optimization is disabled")
            return True

        if self._initialized:
            logger.warning("Already initialized")
            return True

        try:
            logger.info("Initializing cost optimization components...")
            self._start_time = time.time()

            # Initialize cost monitor
            self.cost_monitor.start_monitoring()
            logger.info("Cost monitor initialized")

            # Register preemption signal handlers
            self.preemption_handler.install_signal_handlers()
            logger.info("Preemption handler initialized")

            # Register callbacks
            self._register_callbacks()

            self._initialized = True
            logger.info("Cost optimization manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize cost optimization: {e}")
            return False

    def _register_callbacks(self) -> None:
        """Register internal callbacks between components."""
        # Notify idle manager when job completes checkpoint
        def on_checkpoint(path: str) -> None:
            self._checkpoints += 1
            logger.debug(f"Checkpoint saved: {path}")

        self.preemption_handler.register_checkpoint_callback(on_checkpoint)

        # Notify cost monitor of idle savings
        def on_job_stopped(job_name: str) -> None:
            self._idle_hours_saved += 0.25  # Assume ~15 min saved per stop
            logger.debug(f"Job stopped for idle: {job_name}")

        self.idle_manager.register_stop_callback(on_job_stopped)

    def get_optimized_job_config(
        self,
        job_type: str,
        iteration: int,
        total_iterations: int,
        gpu_count: int = 1,
    ) -> JobSpec:
        """Get optimized job configuration.

        Args:
            job_type: Type of job ("self_play", "training", "evaluation").
            iteration: Current training iteration.
            total_iterations: Total planned iterations.
            gpu_count: Required GPU count.

        Returns:
            JobSpec with optimized configuration.
        """
        if not self.config.enabled:
            return JobSpec(
                gpu_type="NVIDIA_TESLA_T4",
                gpu_count=gpu_count,
                use_spot=False,
                region=self.config.default_region,
                reason="Cost optimization disabled",
            )

        # Get GPU selection from scheduler
        gpu_selection = self.gpu_scheduler.select_gpu(
            job_type=job_type,
            iteration=iteration,
            total_iterations=total_iterations,
        )

        gpu_type = gpu_selection.gpu_type
        use_spot = self.config.enable_spot_vms and gpu_selection.prefer_spot

        # Check quota availability
        region = self.config.default_region
        if not self.quota_manager.can_launch(gpu_type, gpu_count, region):
            # Try fallback
            fallback = self.quota_manager.get_fallback_config(
                gpu_type, region, gpu_count
            )
            if fallback:
                if fallback.gpu_type:
                    gpu_type = fallback.gpu_type
                if fallback.region:
                    region = fallback.region
                if fallback.use_spot is not None:
                    use_spot = fallback.use_spot

        # Check budget
        if not self.cost_monitor.can_afford(estimated_cost=1.0):
            # Reduce to minimum config
            gpu_type = "NVIDIA_TESLA_T4"
            use_spot = True
            logger.warning("Budget limited, using minimum GPU config")

        # Determine checkpoint interval
        checkpoint_interval = self.config.preemption_handler.checkpoint_interval_seconds
        if use_spot:
            # More frequent checkpoints for spot VMs
            checkpoint_interval = min(checkpoint_interval, 300)

        # Estimate cost using centralized cost_monitor (1 hour baseline)
        estimated_cost = self.cost_monitor.estimate_job_cost(
            gpu_type=gpu_type,
            duration_hours=1.0,
            gpu_count=gpu_count,
            use_spot=use_spot,
        )

        return JobSpec(
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            use_spot=use_spot,
            region=region,
            checkpoint_interval=checkpoint_interval,
            estimated_cost_per_hour=estimated_cost,
            reason=f"Phase: {gpu_selection.phase.value}, Spot: {use_spot}",
        )

    async def start_monitoring(self) -> None:
        """Start all monitoring tasks."""
        if not self.config.enabled:
            return

        if self._monitoring:
            logger.warning("Monitoring already started")
            return

        self._monitoring = True

        # Start component monitoring
        self.idle_manager.start_monitoring()
        self.preemption_handler.start_checkpoint_timer(
            model=None,  # Will be set by trainer
            optimizer=None,
            state_getter=lambda: {},
        )

        logger.info("Cost optimization monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop all monitoring tasks."""
        if not self._monitoring:
            return

        self._monitoring = False

        # Stop component monitoring
        self.cost_monitor.stop_monitoring()
        self.idle_manager.stop_monitoring()
        self.preemption_handler.stop_checkpoint_timer()

        logger.info("Cost optimization monitoring stopped")

    def record_job_start(self, job_name: str, job_type: str, use_spot: bool) -> None:
        """Record job start for tracking.

        Args:
            job_name: Name of the job.
            job_type: Type of job.
            use_spot: Whether using spot VM.
        """
        self.idle_manager.register_job(job_name, job_type)
        if use_spot:
            self._spot_hours += 0.0  # Will accumulate
        else:
            self._on_demand_hours += 0.0

    def record_job_activity(self, job_name: str, gpu_utilization: float = 1.0) -> None:
        """Record job activity to prevent idle shutdown.

        Args:
            job_name: Name of the job.
            gpu_utilization: Current GPU utilization.
        """
        self.idle_manager.mark_active(job_name, gpu_utilization=gpu_utilization)

    def record_job_end(self, job_name: str, hours: float, use_spot: bool) -> None:
        """Record job completion.

        Args:
            job_name: Name of the job.
            hours: Hours run.
            use_spot: Whether used spot VM.
        """
        self.idle_manager.unregister_job(job_name)
        if use_spot:
            self._spot_hours += hours
        else:
            self._on_demand_hours += hours

    def record_preemption(self) -> None:
        """Record a preemption event."""
        self._preemptions += 1

    def get_savings_report(self) -> SavingsReport:
        """Get comprehensive savings report.

        Returns:
            SavingsReport with all savings details.
        """
        # Get prices from config if available
        prices = self._cost_monitor.config.gpu_prices if self._cost_monitor else {}
        t4_on_demand = prices.get("NVIDIA_TESLA_T4", 0.35)
        t4_spot = prices.get("NVIDIA_TESLA_T4_SPOT", 0.11)
        v100_on_demand = prices.get("NVIDIA_TESLA_V100", 2.48)

        spot_cost = self._spot_hours * t4_spot
        spot_full_price = self._spot_hours * t4_on_demand
        spot_savings = spot_full_price - spot_cost

        on_demand_cost = self._on_demand_hours * t4_on_demand

        # Idle savings (estimated 15 min per stopped job)
        idle_savings = self._idle_hours_saved * t4_on_demand

        # GPU optimization savings (T4 vs V100 differential)
        # Simplified: estimate 50% savings from using T4 for early phases
        gpu_optimization_savings = (self._spot_hours + self._on_demand_hours) * 0.5 * (v100_on_demand - t4_on_demand)

        total_cost = spot_cost + on_demand_cost
        estimated_full_price = spot_full_price + on_demand_cost + gpu_optimization_savings
        total_savings = spot_savings + idle_savings + gpu_optimization_savings

        savings_percentage = (
            (total_savings / estimated_full_price * 100)
            if estimated_full_price > 0
            else 0.0
        )

        return SavingsReport(
            total_cost=total_cost,
            estimated_full_price=estimated_full_price,
            total_savings=total_savings,
            savings_percentage=savings_percentage,
            spot_savings=spot_savings,
            idle_savings=idle_savings,
            gpu_optimization_savings=gpu_optimization_savings,
            hours_run=self._spot_hours + self._on_demand_hours,
            preemptions_handled=self._preemptions,
            checkpoints_saved=self._checkpoints,
        )

    def get_status(self) -> dict[str, Any]:
        """Get current status of all components.

        Returns:
            Dictionary with component statuses.
        """
        return {
            "enabled": self.config.enabled,
            "initialized": self._initialized,
            "monitoring": self._monitoring,
            "uptime_hours": (time.time() - self._start_time) / 3600 if self._start_time else 0,
            "cost_monitor": self.cost_monitor.get_statistics() if self._cost_monitor else None,
            "gpu_scheduler": {
                "current_config": self.gpu_scheduler.config.to_dict()
            } if self._gpu_scheduler else None,
            "idle_manager": self.idle_manager.get_statistics() if self._idle_manager else None,
            "preemption_handler": self.preemption_handler.get_statistics() if self._preemption_handler else None,
            "quota_manager": self.quota_manager.get_statistics() if self._quota_manager else None,
            "savings": self.get_savings_report().to_dict(),
        }

    async def shutdown(self) -> None:
        """Shutdown all components gracefully."""
        logger.info("Shutting down cost optimization manager...")

        await self.stop_monitoring()

        # Final checkpoint
        if self._preemption_handler:
            try:
                self.preemption_handler.checkpoint_now(reason="shutdown")
            except Exception as e:
                logger.warning(f"Failed to save final checkpoint: {e}")

        # Log final savings
        savings = self.get_savings_report()
        logger.info(
            f"Final savings report: "
            f"total_cost=${savings.total_cost:.2f}, "
            f"savings=${savings.total_savings:.2f} ({savings.savings_percentage:.1f}%)"
        )

        self._initialized = False
        logger.info("Cost optimization manager shutdown complete")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CostOptimizationManager("
            f"enabled={self.config.enabled}, "
            f"initialized={self._initialized}, "
            f"monitoring={self._monitoring})"
        )
