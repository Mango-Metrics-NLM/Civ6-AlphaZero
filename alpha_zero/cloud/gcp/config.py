"""GCP-specific configuration classes.

This module provides configuration dataclasses for GCP services,
with full environment variable support for all settings.

Environment Variables (all optional, with sensible defaults):
    GCP_PROJECT_ID: GCP project ID
    GCP_REGION: Default region (e.g., us-central1)
    GCP_SERVICE_ACCOUNT: Service account email
    GCP_BUCKET: Default Cloud Storage bucket
    VERTEX_MACHINE_TYPE: Default Vertex AI machine type
    VERTEX_ACCELERATOR_TYPE: Default GPU type
    VERTEX_ACCELERATOR_COUNT: Number of GPUs
    VERTEX_CONTAINER_URI: Training container image

Example:
    from alpha_zero.cloud.gcp.config import GCPConfig

    # Load all config from environment
    config = GCPConfig.from_env()

    # Or create programmatically
    config = GCPConfig(
        project_id="my-project",
        region="us-central1",
        vertex=VertexConfig(
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_TESLA_T4",
        ),
    )
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Default values as constants (not hardcoded in code paths)
DEFAULT_REGION = "us-central1"
DEFAULT_MACHINE_TYPE = "n1-standard-4"
DEFAULT_CONTAINER_URI = "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest"


@dataclass
class VertexConfig:
    """Configuration for Vertex AI training.

    All values can be overridden at job submission time.
    """

    # Machine configuration
    machine_type: str = DEFAULT_MACHINE_TYPE
    accelerator_type: str = ""
    accelerator_count: int = 0

    # Container configuration
    container_uri: str = DEFAULT_CONTAINER_URI
    python_module: str = ""  # e.g., "alpha_zero.training_civ6"
    script_path: str = ""  # Alternative: direct script path

    # Resources
    replica_count: int = 1
    boot_disk_size_gb: int = 100
    timeout_seconds: int = 86400  # 24 hours default

    # Networking
    network: str = ""
    enable_web_access: bool = False

    # Experiment tracking
    tensorboard_resource_name: str = ""
    experiment_name: str = ""

    @classmethod
    def from_env(cls, prefix: str = "VERTEX") -> VertexConfig:
        """Load from environment variables."""

        def get_env(key: str, default: str = "") -> str:
            return os.environ.get(f"{prefix}_{key}", default)

        return cls(
            machine_type=get_env("MACHINE_TYPE", DEFAULT_MACHINE_TYPE),
            accelerator_type=get_env("ACCELERATOR_TYPE"),
            accelerator_count=int(get_env("ACCELERATOR_COUNT", "0")),
            container_uri=get_env("CONTAINER_URI", DEFAULT_CONTAINER_URI),
            python_module=get_env("PYTHON_MODULE"),
            script_path=get_env("SCRIPT_PATH"),
            replica_count=int(get_env("REPLICA_COUNT", "1")),
            boot_disk_size_gb=int(get_env("BOOT_DISK_SIZE_GB", "100")),
            timeout_seconds=int(get_env("TIMEOUT_SECONDS", "86400")),
            network=get_env("NETWORK"),
            enable_web_access=get_env("ENABLE_WEB_ACCESS", "false").lower() == "true",
            tensorboard_resource_name=get_env("TENSORBOARD_RESOURCE"),
            experiment_name=get_env("EXPERIMENT_NAME"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "machine_type": self.machine_type,
            "accelerator_type": self.accelerator_type,
            "accelerator_count": self.accelerator_count,
            "container_uri": self.container_uri,
            "python_module": self.python_module,
            "script_path": self.script_path,
            "replica_count": self.replica_count,
            "boot_disk_size_gb": self.boot_disk_size_gb,
            "timeout_seconds": self.timeout_seconds,
            "network": self.network,
            "enable_web_access": self.enable_web_access,
            "tensorboard_resource_name": self.tensorboard_resource_name,
            "experiment_name": self.experiment_name,
        }


@dataclass
class StorageConfig:
    """Configuration for Cloud Storage."""

    bucket: str = ""
    checkpoint_prefix: str = "checkpoints"
    artifact_prefix: str = "artifacts"
    data_prefix: str = "data"
    logs_prefix: str = "logs"

    # Upload settings
    chunk_size_mb: int = 8
    retry_count: int = 3
    retry_delay_seconds: float = 1.0

    @classmethod
    def from_env(cls, prefix: str = "GCS") -> StorageConfig:
        """Load from environment variables."""

        def get_env(key: str, default: str = "") -> str:
            return os.environ.get(f"{prefix}_{key}", default)

        return cls(
            bucket=get_env("BUCKET"),
            checkpoint_prefix=get_env("CHECKPOINT_PREFIX", "checkpoints"),
            artifact_prefix=get_env("ARTIFACT_PREFIX", "artifacts"),
            data_prefix=get_env("DATA_PREFIX", "data"),
            logs_prefix=get_env("LOGS_PREFIX", "logs"),
            chunk_size_mb=int(get_env("CHUNK_SIZE_MB", "8")),
            retry_count=int(get_env("RETRY_COUNT", "3")),
            retry_delay_seconds=float(get_env("RETRY_DELAY_SECONDS", "1.0")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bucket": self.bucket,
            "checkpoint_prefix": self.checkpoint_prefix,
            "artifact_prefix": self.artifact_prefix,
            "data_prefix": self.data_prefix,
            "logs_prefix": self.logs_prefix,
            "chunk_size_mb": self.chunk_size_mb,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
        }

    def get_checkpoint_path(self, name: str) -> str:
        """Get full GCS path for a checkpoint."""
        return f"gs://{self.bucket}/{self.checkpoint_prefix}/{name}"

    def get_artifact_path(self, name: str) -> str:
        """Get full GCS path for an artifact."""
        return f"gs://{self.bucket}/{self.artifact_prefix}/{name}"


@dataclass
class GCPConfig:
    """Complete GCP configuration.

    Combines project-level settings with service-specific configurations
    for Vertex AI and Cloud Storage.
    """

    # Project settings
    project_id: str = ""
    region: str = DEFAULT_REGION
    service_account: str = ""

    # Service configurations
    vertex: VertexConfig = field(default_factory=VertexConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Labels for all resources
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> GCPConfig:
        """Load complete GCP configuration from environment."""

        def get_env(key: str, default: str = "") -> str:
            return os.environ.get(f"GCP_{key}", default)

        # Parse labels
        labels_str = get_env("LABELS", "")
        labels = {}
        if labels_str:
            for pair in labels_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    labels[k.strip()] = v.strip()

        return cls(
            project_id=get_env("PROJECT_ID"),
            region=get_env("REGION", DEFAULT_REGION),
            service_account=get_env("SERVICE_ACCOUNT"),
            vertex=VertexConfig.from_env(),
            storage=StorageConfig.from_env(),
            labels=labels,
        )

    def validate(self) -> list[str]:
        """Validate configuration, returning list of errors."""
        errors = []

        if not self.project_id:
            errors.append("GCP_PROJECT_ID is required")
        if not self.region:
            errors.append("GCP_REGION is required")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "project_id": self.project_id,
            "region": self.region,
            "service_account": self.service_account,
            "vertex": self.vertex.to_dict(),
            "storage": self.storage.to_dict(),
            "labels": self.labels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GCPConfig:
        """Create from dictionary."""
        vertex_data = data.get("vertex", {})
        storage_data = data.get("storage", {})

        return cls(
            project_id=data.get("project_id", ""),
            region=data.get("region", DEFAULT_REGION),
            service_account=data.get("service_account", ""),
            vertex=VertexConfig(**vertex_data) if vertex_data else VertexConfig(),
            storage=StorageConfig(**storage_data) if storage_data else StorageConfig(),
            labels=data.get("labels", {}),
        )
