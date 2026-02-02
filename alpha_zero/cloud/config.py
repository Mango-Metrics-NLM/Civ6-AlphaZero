"""Cloud configuration with environment-based loading.

This module provides base configuration classes for cloud providers,
with support for environment-based configuration and multiple deployment
environments (dev, staging, prod).

All configuration values are loaded from environment variables to avoid
hardcoding sensitive information like project IDs and credentials.

Example:
    from alpha_zero.cloud.config import CloudConfig, CloudEnvironment

    # Load from environment
    config = CloudConfig.from_env()

    # Or create explicitly
    config = CloudConfig(
        provider=CloudProvider.GCP,
        environment=CloudEnvironment.PRODUCTION,
        project_id=os.environ["GCP_PROJECT_ID"],
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""

    GCP = "gcp"
    AWS = "aws"  # Future
    AZURE = "azure"  # Future
    LOCAL = "local"  # For testing


class CloudEnvironment(Enum):
    """Deployment environments."""

    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"
    TEST = "test"

    @classmethod
    def from_string(cls, value: str) -> CloudEnvironment:
        """Create from string, case-insensitive."""
        value_lower = value.lower()
        for env in cls:
            if env.value == value_lower or env.name.lower() == value_lower:
                return env
        raise ValueError(f"Unknown environment: {value}")


@dataclass
class MachineProfile:
    """Profile for cloud machine configuration.

    Defines machine type, GPU configuration, and pricing for different
    training workloads.

    Attributes:
        name: Profile identifier.
        machine_type: Cloud machine type (e.g., n1-standard-8).
        accelerator_type: GPU type or None for CPU-only.
        accelerator_count: Number of GPUs.
        spot: Whether to use spot/preemptible instances.
        hourly_rate: On-demand hourly rate in USD.
        hourly_rate_spot: Spot instance hourly rate in USD.
    """

    name: str
    machine_type: str
    accelerator_type: str | None = None
    accelerator_count: int = 0
    spot: bool = False
    hourly_rate: float = 0.0
    hourly_rate_spot: float = 0.0

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


# Predefined machine profiles for common workloads
MACHINE_PROFILES: dict[str, MachineProfile] = {
    "selfplay_t4_spot": MachineProfile(
        name="selfplay_t4_spot",
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        spot=True,
        hourly_rate=0.95,
        hourly_rate_spot=0.35,
    ),
    "selfplay_l4_spot": MachineProfile(
        name="selfplay_l4_spot",
        machine_type="g2-standard-8",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        spot=True,
        hourly_rate=0.98,
        hourly_rate_spot=0.29,
    ),
    "train_l4_single": MachineProfile(
        name="train_l4_single",
        machine_type="g2-standard-16",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        spot=True,
        hourly_rate=1.39,
        hourly_rate_spot=0.42,
    ),
    "train_l4_ondemand": MachineProfile(
        name="train_l4_ondemand",
        machine_type="g2-standard-16",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        spot=False,
        hourly_rate=1.39,
        hourly_rate_spot=0.42,
    ),
    "train_a100_burst": MachineProfile(
        name="train_a100_burst",
        machine_type="a2-highgpu-1g",
        accelerator_type="NVIDIA_TESLA_A100",
        accelerator_count=1,
        spot=True,
        hourly_rate=3.67,
        hourly_rate_spot=1.10,
    ),
    "eval_l4_spot": MachineProfile(
        name="eval_l4_spot",
        machine_type="g2-standard-8",
        accelerator_type="NVIDIA_L4",
        accelerator_count=1,
        spot=True,
        hourly_rate=0.98,
        hourly_rate_spot=0.29,
    ),
}


def get_machine_profile(name: str) -> MachineProfile:
    """Get a predefined machine profile by name.

    Args:
        name: Profile name (e.g., "selfplay_t4_spot").

    Returns:
        MachineProfile for the requested configuration.

    Raises:
        KeyError: If profile name is not found.
    """
    if name not in MACHINE_PROFILES:
        raise KeyError(
            f"Unknown machine profile: {name}. "
            f"Available profiles: {list(MACHINE_PROFILES.keys())}"
        )
    return MACHINE_PROFILES[name]


@dataclass
class CloudConfig:
    """Base configuration for cloud providers.

    All configuration values can be loaded from environment variables
    to avoid hardcoding sensitive information.

    Environment Variables:
        CLOUD_PROVIDER: Cloud provider (gcp, aws, azure, local)
        CLOUD_ENVIRONMENT: Deployment environment (dev, staging, prod)
        CLOUD_PROJECT_ID: Project/account identifier
        CLOUD_REGION: Default region for resources
        CLOUD_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
    """

    provider: CloudProvider = CloudProvider.LOCAL
    environment: CloudEnvironment = CloudEnvironment.DEVELOPMENT
    project_id: str = ""
    region: str = ""
    log_level: str = "INFO"

    # Resource limits (can be overridden per-job)
    default_machine_type: str = ""
    default_accelerator_type: str = ""
    default_accelerator_count: int = 0

    # Storage defaults
    default_bucket: str = ""
    checkpoint_prefix: str = "checkpoints"
    artifact_prefix: str = "artifacts"

    # Additional metadata
    labels: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, prefix: str = "CLOUD") -> CloudConfig:
        """Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix (default: CLOUD)

        Returns:
            CloudConfig populated from environment.

        Example:
            # With CLOUD_PROJECT_ID=my-project, CLOUD_REGION=us-central1
            config = CloudConfig.from_env()
            assert config.project_id == "my-project"
        """

        def get_env(key: str, default: str = "") -> str:
            return os.environ.get(f"{prefix}_{key}", default)

        provider_str = get_env("PROVIDER", "local")
        try:
            provider = CloudProvider(provider_str.lower())
        except ValueError:
            logger.warning(f"Unknown provider '{provider_str}', using LOCAL")
            provider = CloudProvider.LOCAL

        env_str = get_env("ENVIRONMENT", "dev")
        try:
            environment = CloudEnvironment.from_string(env_str)
        except ValueError:
            logger.warning(f"Unknown environment '{env_str}', using DEVELOPMENT")
            environment = CloudEnvironment.DEVELOPMENT

        # Parse labels from comma-separated key=value pairs
        labels_str = get_env("LABELS", "")
        labels = {}
        if labels_str:
            for pair in labels_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    labels[k.strip()] = v.strip()

        # Parse tags from comma-separated values
        tags_str = get_env("TAGS", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()]

        return cls(
            provider=provider,
            environment=environment,
            project_id=get_env("PROJECT_ID"),
            region=get_env("REGION"),
            log_level=get_env("LOG_LEVEL", "INFO"),
            default_machine_type=get_env("DEFAULT_MACHINE_TYPE"),
            default_accelerator_type=get_env("DEFAULT_ACCELERATOR_TYPE"),
            default_accelerator_count=int(get_env("DEFAULT_ACCELERATOR_COUNT", "0")),
            default_bucket=get_env("DEFAULT_BUCKET"),
            checkpoint_prefix=get_env("CHECKPOINT_PREFIX", "checkpoints"),
            artifact_prefix=get_env("ARTIFACT_PREFIX", "artifacts"),
            labels=labels,
            tags=tags,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider.value,
            "environment": self.environment.value,
            "project_id": self.project_id,
            "region": self.region,
            "log_level": self.log_level,
            "default_machine_type": self.default_machine_type,
            "default_accelerator_type": self.default_accelerator_type,
            "default_accelerator_count": self.default_accelerator_count,
            "default_bucket": self.default_bucket,
            "checkpoint_prefix": self.checkpoint_prefix,
            "artifact_prefix": self.artifact_prefix,
            "labels": self.labels,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CloudConfig:
        """Create from dictionary."""
        provider = CloudProvider(data.get("provider", "local"))
        environment = CloudEnvironment.from_string(data.get("environment", "dev"))

        return cls(
            provider=provider,
            environment=environment,
            project_id=data.get("project_id", ""),
            region=data.get("region", ""),
            log_level=data.get("log_level", "INFO"),
            default_machine_type=data.get("default_machine_type", ""),
            default_accelerator_type=data.get("default_accelerator_type", ""),
            default_accelerator_count=data.get("default_accelerator_count", 0),
            default_bucket=data.get("default_bucket", ""),
            checkpoint_prefix=data.get("checkpoint_prefix", "checkpoints"),
            artifact_prefix=data.get("artifact_prefix", "artifacts"),
            labels=data.get("labels", {}),
            tags=data.get("tags", []),
        )

    def validate(self) -> list[str]:
        """Validate configuration, returning list of errors."""
        errors = []

        if self.provider != CloudProvider.LOCAL:
            if not self.project_id:
                errors.append("project_id is required for cloud providers")
            if not self.region:
                errors.append("region is required for cloud providers")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0

    def get_storage_path(self, *parts: str) -> str:
        """Build a storage path from parts.

        Args:
            *parts: Path components to join.

        Returns:
            Full storage path (e.g., gs://bucket/prefix/parts).
        """
        if self.provider == CloudProvider.GCP:
            base = f"gs://{self.default_bucket}"
        elif self.provider == CloudProvider.LOCAL:
            base = str(Path.cwd() / "storage")
        else:
            base = self.default_bucket

        return "/".join([base] + list(parts))

    def get_checkpoint_path(self, checkpoint_name: str) -> str:
        """Get full path for a checkpoint."""
        return self.get_storage_path(self.checkpoint_prefix, checkpoint_name)

    def get_artifact_path(self, artifact_name: str) -> str:
        """Get full path for an artifact."""
        return self.get_storage_path(self.artifact_prefix, artifact_name)
