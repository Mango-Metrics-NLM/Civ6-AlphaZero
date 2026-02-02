"""Cloud configuration for distributed training.

Provides:
- CloudConfig: Main configuration dataclass for cloud settings
- Environment-based configuration loading
- YAML configuration loading
- Validation for required settings

All configuration values can be overridden via environment variables
with the CIV6AZ_ prefix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from alpha_zero.cloud.exceptions import CloudConfigError


@dataclass
class CloudConfig:
    """Configuration for cloud training infrastructure.

    Attributes:
        project_id: GCP project ID (from GCP_PROJECT_ID env var).
        region: GCP region (from GCP_REGION env var, default us-central1).
        zone: GCP zone (from GCP_ZONE env var, optional).
        bucket_name: GCS bucket name (from GCS_BUCKET_NAME env var).
        checkpoint_prefix: Prefix for checkpoint paths in bucket.
        artifact_prefix: Prefix for training artifacts in bucket.
        enable_cloud_logging: Whether to use Cloud Logging.
        enable_cloud_monitoring: Whether to use Cloud Monitoring.
        service_account: Service account email (optional, uses default if not set).
        network: VPC network name for Vertex AI jobs.
        machine_type: Default machine type for training.
        accelerator_type: Default accelerator type (e.g., NVIDIA_TESLA_T4).
        accelerator_count: Number of accelerators per machine.
        use_spot: Whether to use spot/preemptible instances.
        max_run_duration_seconds: Maximum job run duration.
    """

    # Core GCP settings
    project_id: str = ""
    region: str = "us-central1"
    zone: str = ""
    bucket_name: str = ""

    # Storage paths
    checkpoint_prefix: str = "checkpoints/"
    artifact_prefix: str = "artifacts/"

    # Observability
    enable_cloud_logging: bool = True
    enable_cloud_monitoring: bool = True
    log_level: str = "INFO"

    # Authentication
    service_account: str = ""

    # Networking
    network: str = ""
    subnetwork: str = ""

    # Compute configuration
    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    replica_count: int = 1

    # Spot/preemptible settings
    use_spot: bool = True
    max_run_duration_seconds: int = 86400  # 24 hours

    # Preemption handling
    preemption_check_interval: float = 5.0
    checkpoint_on_preemption: bool = True

    # Additional custom settings
    custom_labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Normalize paths to use forward slashes
        if self.checkpoint_prefix and not self.checkpoint_prefix.endswith("/"):
            object.__setattr__(self, "checkpoint_prefix", self.checkpoint_prefix + "/")
        if self.artifact_prefix and not self.artifact_prefix.endswith("/"):
            object.__setattr__(self, "artifact_prefix", self.artifact_prefix + "/")

    @classmethod
    def from_env(cls, prefix: str = "GCP_") -> CloudConfig:
        """Load configuration from environment variables.

        Loads settings from environment variables with the given prefix:
        - GCP_PROJECT_ID -> project_id
        - GCP_REGION -> region
        - GCP_ZONE -> zone
        - GCS_BUCKET_NAME -> bucket_name
        - And CIV6AZ_ overrides for all other settings

        Args:
            prefix: Environment variable prefix (default: GCP_).

        Returns:
            CloudConfig instance with values from environment.
        """
        # Map environment variables to config fields
        env_mapping = {
            f"{prefix}PROJECT_ID": "project_id",
            f"{prefix}REGION": "region",
            f"{prefix}ZONE": "zone",
            "GCS_BUCKET_NAME": "bucket_name",
            "CIV6AZ_CHECKPOINT_PREFIX": "checkpoint_prefix",
            "CIV6AZ_ARTIFACT_PREFIX": "artifact_prefix",
            "CIV6AZ_ENABLE_CLOUD_LOGGING": "enable_cloud_logging",
            "CIV6AZ_ENABLE_CLOUD_MONITORING": "enable_cloud_monitoring",
            "CIV6AZ_LOG_LEVEL": "log_level",
            f"{prefix}SERVICE_ACCOUNT": "service_account",
            f"{prefix}NETWORK": "network",
            f"{prefix}SUBNETWORK": "subnetwork",
            "CIV6AZ_MACHINE_TYPE": "machine_type",
            "CIV6AZ_ACCELERATOR_TYPE": "accelerator_type",
            "CIV6AZ_ACCELERATOR_COUNT": "accelerator_count",
            "CIV6AZ_REPLICA_COUNT": "replica_count",
            "CIV6AZ_USE_SPOT": "use_spot",
            "CIV6AZ_MAX_RUN_DURATION": "max_run_duration_seconds",
            "CIV6AZ_PREEMPTION_CHECK_INTERVAL": "preemption_check_interval",
        }

        # Boolean fields that need conversion
        bool_fields = {"enable_cloud_logging", "enable_cloud_monitoring", "use_spot"}
        # Integer fields that need conversion
        int_fields = {"accelerator_count", "replica_count", "max_run_duration_seconds"}
        # Float fields that need conversion
        float_fields = {"preemption_check_interval"}

        kwargs: dict[str, Any] = {}
        for env_var, field_name in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                if field_name in bool_fields:
                    kwargs[field_name] = value.lower() in ("true", "1", "yes")
                elif field_name in int_fields:
                    kwargs[field_name] = int(value)
                elif field_name in float_fields:
                    kwargs[field_name] = float(value)
                else:
                    kwargs[field_name] = value

        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> CloudConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            CloudConfig instance.

        Raises:
            CloudConfigError: If file cannot be read or parsed.
        """
        path = Path(path)
        if not path.exists():
            raise CloudConfigError(f"Configuration file not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise CloudConfigError(f"Failed to parse YAML: {path}", cause=e) from e

        # Extract cloud/gcp section if present
        if "cloud" in data:
            data = data["cloud"]
        if "gcp" in data:
            data = data["gcp"]

        # Filter to valid fields only
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)

    def validate(self, require_project: bool = True) -> None:
        """Validate configuration settings.

        Args:
            require_project: Whether project_id is required.

        Raises:
            CloudConfigError: If validation fails.
        """
        if require_project and not self.project_id:
            raise CloudConfigError(
                "project_id is required. Set GCP_PROJECT_ID environment variable."
            )

        if self.use_spot and self.max_run_duration_seconds < 60:
            raise CloudConfigError(
                "max_run_duration_seconds must be at least 60 for spot instances"
            )

        if self.accelerator_count < 0:
            raise CloudConfigError("accelerator_count must be non-negative")

        if self.replica_count < 1:
            raise CloudConfigError("replica_count must be at least 1")

    def get_checkpoint_uri(self, checkpoint_name: str) -> str:
        """Get full GCS URI for a checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint file.

        Returns:
            Full gs:// URI for the checkpoint.
        """
        if not self.bucket_name:
            raise CloudConfigError("bucket_name is required for checkpoint URIs")
        return f"gs://{self.bucket_name}/{self.checkpoint_prefix}{checkpoint_name}"

    def get_artifact_uri(self, artifact_name: str) -> str:
        """Get full GCS URI for an artifact.

        Args:
            artifact_name: Name of the artifact.

        Returns:
            Full gs:// URI for the artifact.
        """
        if not self.bucket_name:
            raise CloudConfigError("bucket_name is required for artifact URIs")
        return f"gs://{self.bucket_name}/{self.artifact_prefix}{artifact_name}"

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            "project_id": self.project_id,
            "region": self.region,
            "zone": self.zone,
            "bucket_name": self.bucket_name,
            "checkpoint_prefix": self.checkpoint_prefix,
            "artifact_prefix": self.artifact_prefix,
            "enable_cloud_logging": self.enable_cloud_logging,
            "enable_cloud_monitoring": self.enable_cloud_monitoring,
            "log_level": self.log_level,
            "service_account": self.service_account,
            "network": self.network,
            "subnetwork": self.subnetwork,
            "machine_type": self.machine_type,
            "accelerator_type": self.accelerator_type,
            "accelerator_count": self.accelerator_count,
            "replica_count": self.replica_count,
            "use_spot": self.use_spot,
            "max_run_duration_seconds": self.max_run_duration_seconds,
            "preemption_check_interval": self.preemption_check_interval,
            "checkpoint_on_preemption": self.checkpoint_on_preemption,
            "custom_labels": self.custom_labels,
        }
