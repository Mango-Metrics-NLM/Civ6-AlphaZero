"""GCP cost estimation utilities for Vertex AI jobs."""

import logging

from alpha_zero.cloud.gcp.config import GCPConfig, get_gpu_costs

logger = logging.getLogger(__name__)

# Default spot discount factor (conservative 70% discount → pay 30% of on-demand)
_DEFAULT_SPOT_DISCOUNT = 0.3


class CostEstimator:
    """Estimates GCP costs for Vertex AI jobs."""

    def __init__(self, config: GCPConfig | None = None) -> None:
        """Initialize with optional config; falls back to env-based defaults."""
        self.config = config or GCPConfig.from_env(validate=False)
        self.gpu_costs = get_gpu_costs()

        # Standard machine type costs (approximate us-central1 on-demand $/hr)
        self.machine_costs: dict[str, float] = {
            "n1-standard-4": 0.19,
            "n1-standard-8": 0.38,
            "n1-standard-16": 0.76,
            "n1-standard-32": 1.52,
            "n1-highmem-2": 0.12,
            "n1-highmem-4": 0.24,
            "n1-highmem-8": 0.48,
            "a2-highgpu-1g": 3.67,
            "a2-highgpu-2g": 7.34,
            "a2-highgpu-4g": 14.68,
            "a2-highgpu-8g": 29.36,
        }

    def estimate_job_cost(
        self,
        machine_type: str,
        accelerator_type: str,
        accelerator_count: int,
        duration_hours: float,
        is_spot: bool = False,
    ) -> float:
        """Estimate the cost of a single Vertex AI job run.

        Args:
            machine_type: GCE machine type (e.g., ``n1-standard-8``).
            accelerator_type: GPU type (e.g., ``NVIDIA_TESLA_T4``), or ``""`` for CPU-only.
            accelerator_count: Number of GPUs (0 for CPU-only jobs).
            duration_hours: Duration of the job in hours.
            is_spot: Whether to use Spot pricing (~70% discount).

        Returns:
            Estimated cost in USD.
        """
        # --- Machine cost ---
        machine_cost_hourly = self.machine_costs.get(machine_type, 0.0)
        if machine_cost_hourly == 0.0:
            logger.warning(
                "Unknown machine type: %s. Cost estimation may be inaccurate.",
                machine_type,
            )

        # --- GPU cost ---
        gpu_cost_hourly = 0.0
        if accelerator_count > 0 and accelerator_type:
            gpu_cost_hourly = self._resolve_gpu_cost(accelerator_type, is_spot)

        total_gpu_cost = gpu_cost_hourly * accelerator_count

        # --- Spot discount on machine (a2-* bundles GPU cost, skip) ---
        if is_spot and not machine_type.startswith("a2-"):
            machine_cost_hourly *= _DEFAULT_SPOT_DISCOUNT

        total_hourly_rate = machine_cost_hourly + total_gpu_cost
        return total_hourly_rate * duration_hours

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_gpu_cost(self, accelerator_type: str, is_spot: bool) -> float:
        """Resolve per-GPU hourly cost, checking for an explicit SPOT key first."""
        if is_spot:
            spot_key = f"{accelerator_type}_SPOT"
            if spot_key in self.gpu_costs:
                return self.gpu_costs[spot_key]
            # Fallback: apply default spot discount to on-demand rate
            return self.gpu_costs.get(accelerator_type, 0.0) * _DEFAULT_SPOT_DISCOUNT
        return self.gpu_costs.get(accelerator_type, 0.0)
