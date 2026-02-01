"""Cloud configuration for Vertex AI deployment.

Provides configuration classes for:
- Machine profiles (GPU types, memory, pricing)
- Cloud-specific settings
- Environment variable integration
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class MachineProfile:
    """Configuration for a Vertex AI machine type.

    Attributes:
        name: Profile name (e.g., "selfplay_t4_spot").
        machine_type: GCP machine type (e.g., "n1-standard-8").
        accelerator_type: GPU accelerator type.
        accelerator_count: Number of GPUs.
        spot: Whether to use Spot/Preemptible instances.
        vcpus: Number of vCPUs.
        memory_gb: RAM in GB.
        gpu_memory_gb: VRAM per GPU in GB.
        hourly_rate_spot: Estimated spot pricing ($/hr).
        hourly_rate_ondemand: On-demand pricing ($/hr).
        description: Human-readable description.
    """

    name: str
    machine_type: str
    accelerator_type: str
    accelerator_count: int
    spot: bool
    vcpus: int = 8
    memory_gb: float = 30.0
    gpu_memory_gb: float = 16.0
    hourly_rate_spot: float = 0.0
    hourly_rate_ondemand: float = 0.0
    description: str = ""

    def to_vertex_spec(self) -> dict[str, Any]:
        """Convert to Vertex AI worker pool spec format."""
        machine_spec: dict[str, Any] = {
            "machine_type": self.machine_type,
        }

        if self.accelerator_type and self.accelerator_count > 0:
            machine_spec["accelerator_type"] = self.accelerator_type
            machine_spec["accelerator_count"] = self.accelerator_count

        return {
            "machine_spec": machine_spec,
            "replica_count": 1,
        }


# Pre-defined machine profiles for budget deployment
MACHINE_PROFILES: dict[str, MachineProfile] = {
    # Self-play workers (use Spot for cost savings)
    "selfplay_t4_spot": MachineProfile(
        name="selfplay_t4_spot",
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        spot=True,
        vcpus=8,
        memory_gb=30.0,
        gpu_memory_gb=16.0,
        hourly_rate_spot=0.11,
        hourly_rate_ondemand=0.35,
        description="Budget self-play with T4 GPU (Spot)",
    ),
    "selfplay_l4_spot": MachineProfile(
        name="selfplay_l4_spot",
        machine_type="g2-standard-8",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        spot=True,
        vcpus=8,
        memory_gb=32.0,
        gpu_memory_gb=24.0,
        hourly_rate_spot=0.24,
        hourly_rate_ondemand=0.80,
        description="Self-play with L4 GPU (Spot)",
    ),
    # Training (prefer on-demand for reliability)
    "train_l4_ondemand": MachineProfile(
        name="train_l4_ondemand",
        machine_type="g2-standard-24",
        accelerator_type="NVIDIA_L4",
        accelerator_count=2,
        spot=False,
        vcpus=24,
        memory_gb=96.0,
        gpu_memory_gb=48.0,  # 2x24GB
        hourly_rate_spot=0.0,
        hourly_rate_ondemand=2.68,
        description="Training with 2x L4 GPUs (on-demand)",
    ),
    "train_l4_single": MachineProfile(
        name="train_l4_single",
        machine_type="g2-standard-8",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        spot=False,
        vcpus=8,
        memory_gb=32.0,
        gpu_memory_gb=24.0,
        hourly_rate_spot=0.24,
        hourly_rate_ondemand=0.80,
        description="Training with single L4 GPU (budget)",
    ),
    "train_a100_burst": MachineProfile(
        name="train_a100_burst",
        machine_type="a2-highgpu-1g",
        accelerator_type="NVIDIA_TESLA_A100",
        accelerator_count=1,
        spot=False,
        vcpus=12,
        memory_gb=85.0,
        gpu_memory_gb=40.0,
        hourly_rate_spot=0.0,
        hourly_rate_ondemand=3.67,
        description="Burst training with A100 GPU",
    ),
    # Evaluation (use Spot, can tolerate preemption)
    "eval_l4_spot": MachineProfile(
        name="eval_l4_spot",
        machine_type="g2-standard-8",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        spot=True,
        vcpus=8,
        memory_gb=32.0,
        gpu_memory_gb=24.0,
        hourly_rate_spot=0.24,
        hourly_rate_ondemand=0.80,
        description="Evaluation with L4 GPU (Spot)",
    ),
    # CPU-only for testing
    "cpu_only": MachineProfile(
        name="cpu_only",
        machine_type="n1-standard-8",
        accelerator_type="",
        accelerator_count=0,
        spot=True,
        vcpus=8,
        memory_gb=30.0,
        gpu_memory_gb=0.0,
        hourly_rate_spot=0.05,
        hourly_rate_ondemand=0.20,
        description="CPU-only for testing",
    ),
}


def get_machine_profiles() -> dict[str, MachineProfile]:
    """Get all available machine profiles.

    Returns:
        Dictionary mapping profile names to MachineProfile objects.
    """
    return MACHINE_PROFILES.copy()


def get_machine_profile(name: str) -> MachineProfile:
    """Get a specific machine profile by name.

    Args:
        name: Profile name.

    Returns:
        MachineProfile for the given name.

    Raises:
        KeyError: If profile doesn't exist.
    """
    if name not in MACHINE_PROFILES:
        available = ", ".join(MACHINE_PROFILES.keys())
        raise KeyError(f"Unknown machine profile: {name}. Available: {available}")
    return MACHINE_PROFILES[name]


@dataclass
class CloudConfig:
    """Configuration for cloud deployment.

    Attributes:
        project_id: GCP project ID.
        region: GCP region (e.g., "us-central1").
        bucket: GCS bucket for storage.
        prefix: Path prefix within bucket.
        artifact_registry: Artifact Registry repository URI.
        service_account: Service account email (optional).
        network: VPC network name (optional).
        wandb_project: Weights & Biases project name.
        wandb_api_key_secret: Secret Manager secret name for W&B API key.
    """

    project_id: str = ""
    region: str = "us-central1"
    bucket: str = ""
    prefix: str = "civ6-alphazero"
    artifact_registry: str = ""
    service_account: str | None = None
    network: str | None = None
    wandb_project: str = "civ6-alphazero"
    wandb_api_key_secret: str = "wandb-api-key"

    # Component-specific settings
    selfplay_profile: str = "selfplay_t4_spot"
    train_profile: str = "train_l4_single"
    eval_profile: str = "eval_l4_spot"

    # Training cycle settings
    selfplay_games_per_worker: int = 100
    num_selfplay_workers: int = 3
    training_steps_per_iteration: int = 1000
    eval_games: int = 400
    promotion_threshold: float = 0.55

    def __post_init__(self) -> None:
        """Apply environment variable overrides."""
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        self.project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", self.project_id)
        self.region = os.environ.get("CIV6_REGION", self.region)
        self.bucket = os.environ.get("CIV6_GCS_BUCKET", self.bucket)
        self.prefix = os.environ.get("CIV6_GCS_PREFIX", self.prefix)

        # Construct artifact registry from project if not set
        if not self.artifact_registry and self.project_id:
            self.artifact_registry = (
                f"{self.region}-docker.pkg.dev/{self.project_id}/civ6-alphazero"
            )

        # Training settings
        if val := os.environ.get("CIV6_SELFPLAY_PROFILE"):
            self.selfplay_profile = val
        if val := os.environ.get("CIV6_TRAIN_PROFILE"):
            self.train_profile = val
        if val := os.environ.get("CIV6_EVAL_PROFILE"):
            self.eval_profile = val

        if val := os.environ.get("CIV6_NUM_SELFPLAY_WORKERS"):
            self.num_selfplay_workers = int(val)
        if val := os.environ.get("CIV6_GAMES_PER_WORKER"):
            self.selfplay_games_per_worker = int(val)
        if val := os.environ.get("CIV6_TRAINING_STEPS"):
            self.training_steps_per_iteration = int(val)
        if val := os.environ.get("CIV6_EVAL_GAMES"):
            self.eval_games = int(val)

    @classmethod
    def from_env(cls) -> CloudConfig:
        """Create configuration from environment variables."""
        return cls()

    def get_container_uri(self, component: str) -> str:
        """Get container image URI for a component.

        Args:
            component: Component name ("selfplay", "trainer", "evaluator").

        Returns:
            Full container image URI.
        """
        if not self.artifact_registry:
            raise ValueError("artifact_registry not configured")
        return f"{self.artifact_registry}/civ6-{component}:latest"

    def get_selfplay_profile(self) -> MachineProfile:
        """Get machine profile for self-play workers."""
        return get_machine_profile(self.selfplay_profile)

    def get_train_profile(self) -> MachineProfile:
        """Get machine profile for training."""
        return get_machine_profile(self.train_profile)

    def get_eval_profile(self) -> MachineProfile:
        """Get machine profile for evaluation."""
        return get_machine_profile(self.eval_profile)

    def estimate_iteration_cost(self) -> dict[str, float]:
        """Estimate cost for one training iteration.

        Returns:
            Dictionary with cost breakdown by component.
        """
        selfplay_profile = self.get_selfplay_profile()
        train_profile = self.get_train_profile()
        eval_profile = self.get_eval_profile()

        # Estimate durations (hours)
        selfplay_hours = 1.5  # ~90 min for 100 games per worker
        train_hours = 1.0  # ~60 min for 1000 steps
        eval_hours = 0.5  # ~30 min for 400 games

        # Calculate costs
        selfplay_rate = (
            selfplay_profile.hourly_rate_spot
            if selfplay_profile.spot
            else selfplay_profile.hourly_rate_ondemand
        )
        train_rate = (
            train_profile.hourly_rate_spot
            if train_profile.spot
            else train_profile.hourly_rate_ondemand
        )
        eval_rate = (
            eval_profile.hourly_rate_spot
            if eval_profile.spot
            else eval_profile.hourly_rate_ondemand
        )

        selfplay_cost = selfplay_rate * selfplay_hours * self.num_selfplay_workers
        train_cost = train_rate * train_hours
        eval_cost = eval_rate * eval_hours

        return {
            "selfplay": round(selfplay_cost, 2),
            "training": round(train_cost, 2),
            "evaluation": round(eval_cost, 2),
            "total": round(selfplay_cost + train_cost + eval_cost, 2),
        }


__all__ = [
    "MachineProfile",
    "CloudConfig",
    "get_machine_profiles",
    "get_machine_profile",
    "MACHINE_PROFILES",
]
