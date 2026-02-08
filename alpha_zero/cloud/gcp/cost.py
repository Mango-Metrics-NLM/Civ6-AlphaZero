import logging
from typing import Optional

from alpha_zero.cloud.gcp.config import GCPConfig, get_gpu_costs

logger = logging.getLogger(__name__)


class CostEstimator:
    """Estimates GCP costs for Vertex AI jobs."""

    def __init__(self, config: Optional[GCPConfig] = None):
        """Initialize with optional config manually, otherwise loads from env."""
        self.config = config or GCPConfig.from_env(validate=False)
        self.gpu_costs = get_gpu_costs()

        # Standard machine type costs (approximate us-central1 on-demand)
        # TODO: Move these to config or fetch dynamically if needed
        self.machine_costs = {
            "n1-standard-4": 0.19,
            "n1-standard-8": 0.38,
            "n1-standard-16": 0.76,
            "n1-standard-32": 1.52,
            "n1-highmem-2": 0.12,
            "n1-highmem-4": 0.24,
            "n1-highmem-8": 0.48,
            "a2-highgpu-1g": 3.67,  # Includes A100 GPU cost usually, but we separate for clarity
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
        """Estimate the cost of a single job run.

        Args:
            machine_type: GCE machine type (e.g., 'n1-standard-8')
            accelerator_type: GPU type (e.g., 'NVIDIA_TESLA_T4')
            accelerator_count: Number of GPUs
            duration_hours: Duration of the job in hours
            is_spot: Whether to use Spot pricing (approx 60-91% discount, using conservative 60%)

        Returns:
            Estimated cost in USD.
        """
        # Base machine cost
        machine_cost_hourly = self.machine_costs.get(machine_type, 0.0)
        if machine_cost_hourly == 0.0:
            logger.warning(f"Unknown machine type: {machine_type}. Cost estimation may be inaccurate.")
        
        # GPU cost
        # Config provides separate spot/on-demand keys
        gpu_key = accelerator_type
        if is_spot:
            # Check if specific SPOT key exists, otherwise use base and apply discount
            spot_key = f"{accelerator_type}_SPOT"
            if spot_key in self.gpu_costs:
                gpu_cost_hourly = self.gpu_costs[spot_key]
            else:
                # Fallback: manually apply discount if specific key missing logic not in config
                # But config.py logic should handle it. 
                # Let's check how get_gpu_costs works in config.py
                # It returns dictionary with keys like NVIDIA_TESLA_T4 and NVIDIA_TESLA_T4_SPOT
                if spot_key in self.gpu_costs:
                     gpu_cost_hourly = self.gpu_costs[spot_key]
                else:
                     # Fallback to logic: 30% of on-demand if not found
                     gpu_cost_hourly = self.gpu_costs.get(accelerator_type, 0.0) * 0.3
        else:
             gpu_cost_hourly = self.gpu_costs.get(accelerator_type, 0.0)

        total_gpu_cost = gpu_cost_hourly * accelerator_count

        # Total hourly rate
        total_hourly_rate = machine_cost_hourly + total_gpu_cost
        
        # Apply spot discount to machine if needed (simple heuristic if not using a2 which bundles)
        if is_spot and "a2-" not in machine_type:
             total_hourly_rate = (machine_cost_hourly * 0.3) + total_gpu_cost # GPU cost already spot-adjusted

        return total_hourly_rate * duration_hours
