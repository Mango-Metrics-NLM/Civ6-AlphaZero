"""
GCP Configuration for Civ6 AlphaZero Cloud Training.

Provides dataclass-based configuration with:
- Environment variable support (no hardcoded values)
- Preset configurations for budget/on-demand tiers
- Validation and serialization utilities
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from alpha_zero.cloud.gcp.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Default GPU pricing (USD/hour) - can be overridden via CIV6AZ_GPU_COST_* env vars
DEFAULT_GPU_COSTS: dict[str, float] = {
    "NVIDIA_TESLA_T4": 0.35,
    "NVIDIA_TESLA_T4_SPOT": 0.11,
    "NVIDIA_L4": 0.70,
    "NVIDIA_L4_SPOT": 0.22,
    "NVIDIA_TESLA_V100": 2.48,
    "NVIDIA_TESLA_V100_SPOT": 0.74,
    "NVIDIA_TESLA_P100": 1.46,
    "NVIDIA_TESLA_P100_SPOT": 0.43,
    "NVIDIA_TESLA_A100": 3.67,
    "NVIDIA_TESLA_A100_SPOT": 1.10,
}


def get_gpu_costs() -> dict[str, float]:
    """Get GPU costs, allowing env-var overrides.

    Each GPU type can be overridden via CIV6AZ_GPU_COST_{NAME} env var,
    where NAME is the GPU type with NVIDIA_ and TESLA_ prefixes stripped.
    Example: CIV6AZ_GPU_COST_T4=0.40 overrides NVIDIA_TESLA_T4 pricing.
    """
    costs = {}
    for gpu_type, default_cost in DEFAULT_GPU_COSTS.items():
        env_key = f"CIV6AZ_GPU_COST_{gpu_type.replace('NVIDIA_', '').replace('TESLA_', '')}"
        costs[gpu_type] = float(os.environ.get(env_key, str(default_cost)))
    return costs


class VertexTier(str, Enum):
    """Vertex AI pricing tier."""

    BUDGET = "budget"  # T4 for both self-play and training
    ON_DEMAND = "ondemand"  # A100 for both


@dataclass
class SelfPlayWorkerConfig:
    """Configuration for self-play worker jobs.

    Attributes:
        machine_type: GCE machine type (e.g., n1-standard-8).
        accelerator_type: GPU type (e.g., NVIDIA_TESLA_T4).
        accelerator_count: Number of GPUs per worker.
        replica_count: Number of parallel workers.
        spot: Use Spot VMs for cost savings.
        games_per_worker: Number of games each worker generates.
        max_retries: Maximum job retries on failure.
    """

    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    replica_count: int = 10
    spot: bool = True
    games_per_worker: int = 100
    max_retries: int = 3

    @classmethod
    def budget_preset(cls) -> SelfPlayWorkerConfig:
        """Create budget tier configuration (T4 Spot VMs)."""
        return cls(
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            replica_count=10,
            spot=True,
        )

    @classmethod
    def ondemand_preset(cls) -> SelfPlayWorkerConfig:
        """Create on-demand tier configuration (A100 GPUs)."""
        return cls(
            machine_type="a2-highgpu-1g",
            accelerator_type="NVIDIA_TESLA_A100",
            accelerator_count=1,
            replica_count=20,
            spot=False,
        )


@dataclass
class TrainingWorkerConfig:
    """Configuration for training jobs.

    Attributes:
        machine_type: GCE machine type.
        accelerator_type: GPU type.
        accelerator_count: Number of GPUs.
        spot: Use Spot VMs (not recommended for training).
        checkpoint_interval: Steps between checkpoints.
        epochs_per_iteration: Training epochs per iteration.
    """

    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    spot: bool = False
    checkpoint_interval: int = 1000
    epochs_per_iteration: int = 10

    @classmethod
    def budget_preset(cls) -> TrainingWorkerConfig:
        """Create budget tier configuration (T4 GPU for better quota availability)."""
        return cls(
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1,
            spot=False,
        )

    @classmethod
    def ondemand_preset(cls) -> TrainingWorkerConfig:
        """Create on-demand tier configuration (8x A100)."""
        return cls(
            machine_type="a2-highgpu-8g",
            accelerator_type="NVIDIA_TESLA_A100",
            accelerator_count=8,
            spot=False,
        )


@dataclass
class GCSPathConfig:
    """GCS path configuration.

    Defines the bucket structure for all artifacts.
    """

    models_prefix: str = "models/civ6az"
    trajectories_prefix: str = "trajectories/civ6az"
    checkpoints_prefix: str = "checkpoints/civ6az"
    logs_prefix: str = "logs/civ6az"

    def model_path(self, version: int) -> str:
        """Get path for a model version."""
        return f"{self.models_prefix}/v{version:03d}/model.pt"

    def metadata_path(self, version: int) -> str:
        """Get path for model metadata."""
        return f"{self.models_prefix}/v{version:03d}/metadata.json"

    def trajectory_path(self, iteration: int, worker_id: int) -> str:
        """Get path for trajectory shard."""
        return f"{self.trajectories_prefix}/iter_{iteration:03d}/shard_worker_{worker_id}.pt"

    def checkpoint_path(self, step: int) -> str:
        """Get path for training checkpoint."""
        return f"{self.checkpoints_prefix}/training_step_{step}.pt"

    def latest_checkpoint_path(self) -> str:
        """Get path for latest checkpoint marker."""
        return f"{self.checkpoints_prefix}/latest.pt"


@dataclass
class GCPConfig:
    """Main GCP configuration for Vertex AI training.

    All values are loaded from environment variables by default,
    ensuring no hardcoded project IDs or bucket names.

    Environment Variables:
        CIV6AZ_VERTEX_PROJECT: GCP project ID
        CIV6AZ_VERTEX_REGION: GCP region (default: us-central1)
        CIV6AZ_GCS_BUCKET: GCS bucket name (without gs:// prefix)
        CIV6AZ_VERTEX_SERVICE_ACCOUNT: Service account email (optional)
        CIV6AZ_CONTAINER_IMAGE_URI: Container image URI (optional)
    """

    project_id: str = field(default_factory=lambda: os.environ.get("CIV6AZ_VERTEX_PROJECT", ""))
    region: str = field(
        default_factory=lambda: os.environ.get("CIV6AZ_VERTEX_REGION", "us-central1")
    )
    bucket: str = field(default_factory=lambda: os.environ.get("CIV6AZ_GCS_BUCKET", ""))
    service_account: str | None = field(
        default_factory=lambda: os.environ.get("CIV6AZ_VERTEX_SERVICE_ACCOUNT")
    )
    container_image_uri: str | None = field(
        default_factory=lambda: os.environ.get("CIV6AZ_CONTAINER_IMAGE_URI")
    )

    # Worker configurations
    self_play: SelfPlayWorkerConfig = field(default_factory=SelfPlayWorkerConfig)
    training: TrainingWorkerConfig = field(default_factory=TrainingWorkerConfig)

    # Path configuration
    paths: GCSPathConfig = field(default_factory=GCSPathConfig)

    # Tier selection
    tier: VertexTier = VertexTier.BUDGET

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Normalize bucket name (remove gs:// prefix if present)
        if self.bucket.startswith("gs://"):
            self.bucket = self.bucket[5:]

    @classmethod
    def from_env(cls, validate: bool = True) -> GCPConfig:
        """Create configuration from environment variables.

        Args:
            validate: Whether to validate required fields.

        Returns:
            GCPConfig instance.

        Raises:
            ConfigurationError: If required variables are missing.
        """
        config = cls()
        if validate:
            config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> GCPConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            GCPConfig instance.

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

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GCPConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            GCPConfig instance.
        """
        self_play_data = data.pop("self_play", {})
        training_data = data.pop("training", {})
        paths_data = data.pop("paths", {})

        self_play = (
            SelfPlayWorkerConfig(**self_play_data) if self_play_data else SelfPlayWorkerConfig()
        )
        training = (
            TrainingWorkerConfig(**training_data) if training_data else TrainingWorkerConfig()
        )
        paths = GCSPathConfig(**paths_data) if paths_data else GCSPathConfig()

        tier_str = data.pop("tier", "budget")
        tier = VertexTier(tier_str) if isinstance(tier_str, str) else tier_str

        return cls(
            self_play=self_play,
            training=training,
            paths=paths,
            tier=tier,
            **data,
        )

    @classmethod
    def with_tier(cls, tier: VertexTier | str) -> GCPConfig:
        """Create configuration with specified tier presets.

        Args:
            tier: Pricing tier (budget or ondemand).

        Returns:
            GCPConfig with appropriate worker configs.
        """
        if isinstance(tier, str):
            tier = VertexTier(tier)

        if tier == VertexTier.BUDGET:
            return cls(
                tier=tier,
                self_play=SelfPlayWorkerConfig.budget_preset(),
                training=TrainingWorkerConfig.budget_preset(),
            )
        else:
            return cls(
                tier=tier,
                self_play=SelfPlayWorkerConfig.ondemand_preset(),
                training=TrainingWorkerConfig.ondemand_preset(),
            )

    def validate(self) -> None:
        """Validate that all required configuration is present.

        Raises:
            ConfigurationError: If required values are missing.
        """
        missing: list[str] = []

        if not self.project_id:
            missing.append("CIV6AZ_VERTEX_PROJECT")
        if not self.bucket:
            missing.append("CIV6AZ_GCS_BUCKET")

        if missing:
            raise ConfigurationError(
                "Missing required GCP configuration",
                missing_vars=missing,
            )

    @property
    def bucket_uri(self) -> str:
        """Get full GCS bucket URI."""
        return f"gs://{self.bucket}"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {
            "project_id": self.project_id,
            "region": self.region,
            "bucket": self.bucket,
            "service_account": self.service_account,
            "container_image_uri": self.container_image_uri,
            "tier": self.tier.value,
            "self_play": {
                "machine_type": self.self_play.machine_type,
                "accelerator_type": self.self_play.accelerator_type,
                "accelerator_count": self.self_play.accelerator_count,
                "replica_count": self.self_play.replica_count,
                "spot": self.self_play.spot,
                "games_per_worker": self.self_play.games_per_worker,
                "max_retries": self.self_play.max_retries,
            },
            "training": {
                "machine_type": self.training.machine_type,
                "accelerator_type": self.training.accelerator_type,
                "accelerator_count": self.training.accelerator_count,
                "spot": self.training.spot,
                "checkpoint_interval": self.training.checkpoint_interval,
                "epochs_per_iteration": self.training.epochs_per_iteration,
            },
            "paths": {
                "models_prefix": self.paths.models_prefix,
                "trajectories_prefix": self.paths.trajectories_prefix,
                "checkpoints_prefix": self.paths.checkpoints_prefix,
                "logs_prefix": self.paths.logs_prefix,
            },
        }

    def __repr__(self) -> str:
        return (
            f"GCPConfig(project={self.project_id!r}, region={self.region!r}, "
            f"bucket={self.bucket!r}, tier={self.tier.value})"
        )
