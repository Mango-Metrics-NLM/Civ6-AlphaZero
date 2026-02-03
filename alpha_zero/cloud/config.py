"""Cloud configuration for distributed training.

Provides:
- CloudConfig: Main configuration dataclass for cloud settings
- CloudProvider: Enum for cloud provider selection
- CloudEnvironment: Enum for deployment environment
- Environment-based configuration loading
- YAML configuration loading
- Validation for required settings

All configuration values can be overridden via environment variables
with the CIV6AZ_ prefix.

Also provides nested config classes for Phase 3 cloud infrastructure:
- GCSConfig: Google Cloud Storage settings
- VertexConfig: Vertex AI job settings
- PreemptionConfig: Preemption handling settings
- CloudLoggingConfig: Cloud logging settings
- CloudMonitoringConfig: Cloud monitoring settings
- TrainingConfig: Training hyperparameters
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from alpha_zero.cloud.exceptions import CloudConfigError

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


# =============================================================================
# Phase 3 Nested Configuration Classes
# =============================================================================


@dataclass
class GCSConfig:
    """Google Cloud Storage configuration.

    Attributes:
        bucket_name: GCS bucket for checkpoints and artifacts.
        checkpoint_prefix: Prefix path for checkpoint storage.
        artifact_prefix: Prefix path for artifact storage.
        logs_prefix: Prefix path for log storage.
        enable_compression: Enable checkpoint compression.
        max_checkpoints: Maximum checkpoints to retain (0 = unlimited).
    """

    bucket_name: str = ""
    checkpoint_prefix: str = "checkpoints"
    artifact_prefix: str = "artifacts"
    logs_prefix: str = "logs"
    enable_compression: bool = True
    max_checkpoints: int = 10

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GCSConfig:
        """Create from dictionary."""
        return cls(
            bucket_name=str(data.get("bucket_name", "")),
            checkpoint_prefix=str(data.get("checkpoint_prefix", "checkpoints")),
            artifact_prefix=str(data.get("artifact_prefix", "artifacts")),
            logs_prefix=str(data.get("logs_prefix", "logs")),
            enable_compression=bool(data.get("enable_compression", True)),
            max_checkpoints=int(data.get("max_checkpoints", 10)),
        )

    @classmethod
    def from_env(cls) -> GCSConfig:
        """Create from environment variables."""
        return cls(
            bucket_name=os.environ.get("CIV6AZ_GCS_BUCKET", ""),
            checkpoint_prefix=os.environ.get(
                "CIV6AZ_GCS_CHECKPOINT_PREFIX", "checkpoints"
            ),
            artifact_prefix=os.environ.get("CIV6AZ_GCS_ARTIFACT_PREFIX", "artifacts"),
            logs_prefix=os.environ.get("CIV6AZ_GCS_LOGS_PREFIX", "logs"),
            enable_compression=os.environ.get(
                "CIV6AZ_GCS_ENABLE_COMPRESSION", "true"
            ).lower()
            == "true",
            max_checkpoints=int(os.environ.get("CIV6AZ_GCS_MAX_CHECKPOINTS", "10")),
        )


@dataclass
class VertexConfig:
    """Vertex AI job configuration.

    Attributes:
        machine_type: GCE machine type for training.
        accelerator_type: GPU accelerator type.
        accelerator_count: Number of GPUs per node.
        boot_disk_size_gb: Boot disk size in GB.
        container_uri: Container image URI for training.
        service_account: GCP service account for the job.
        network: VPC network for the job.
        max_run_duration_seconds: Maximum job runtime (0 = unlimited).
        enable_web_access: Enable Vertex AI web access for debugging.
    """

    machine_type: str = "n1-standard-8"
    accelerator_type: str = "NVIDIA_TESLA_T4"
    accelerator_count: int = 1
    boot_disk_size_gb: int = 100
    container_uri: str = "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest"
    service_account: str = ""
    network: str = ""
    max_run_duration_seconds: int = 172800  # 48 hours
    enable_web_access: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VertexConfig:
        """Create from dictionary."""
        return cls(
            machine_type=str(data.get("machine_type", "n1-standard-8")),
            accelerator_type=str(data.get("accelerator_type", "NVIDIA_TESLA_T4")),
            accelerator_count=int(data.get("accelerator_count", 1)),
            boot_disk_size_gb=int(data.get("boot_disk_size_gb", 100)),
            container_uri=str(
                data.get(
                    "container_uri",
                    "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest",
                )
            ),
            service_account=str(data.get("service_account", "")),
            network=str(data.get("network", "")),
            max_run_duration_seconds=int(
                data.get("max_run_duration_seconds", 172800)
            ),
            enable_web_access=bool(data.get("enable_web_access", False)),
        )

    @classmethod
    def from_env(cls) -> VertexConfig:
        """Create from environment variables."""
        return cls(
            machine_type=os.environ.get("CIV6AZ_VERTEX_MACHINE_TYPE", "n1-standard-8"),
            accelerator_type=os.environ.get(
                "CIV6AZ_VERTEX_ACCELERATOR_TYPE", "NVIDIA_TESLA_T4"
            ),
            accelerator_count=int(
                os.environ.get("CIV6AZ_VERTEX_ACCELERATOR_COUNT", "1")
            ),
            boot_disk_size_gb=int(
                os.environ.get("CIV6AZ_VERTEX_BOOT_DISK_SIZE_GB", "100")
            ),
            container_uri=os.environ.get(
                "CIV6AZ_VERTEX_CONTAINER_URI",
                "gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest",
            ),
            service_account=os.environ.get("CIV6AZ_VERTEX_SERVICE_ACCOUNT", ""),
            network=os.environ.get("CIV6AZ_VERTEX_NETWORK", ""),
            max_run_duration_seconds=int(
                os.environ.get("CIV6AZ_VERTEX_MAX_RUN_DURATION", "172800")
            ),
            enable_web_access=os.environ.get(
                "CIV6AZ_VERTEX_ENABLE_WEB_ACCESS", "false"
            ).lower()
            == "true",
        )


@dataclass
class PreemptionConfig:
    """Preemption handling configuration.

    Attributes:
        enabled: Enable preemption handling for spot instances.
        check_interval_seconds: Interval to check for preemption signals.
        warning_seconds: Expected warning time before preemption.
        checkpoint_on_preemption: Save checkpoint when preempted.
        max_retry_count: Maximum retries after preemption.
        backoff_multiplier: Backoff multiplier for retries.
    """

    enabled: bool = True
    check_interval_seconds: float = 5.0
    warning_seconds: float = 30.0
    checkpoint_on_preemption: bool = True
    max_retry_count: int = 3
    backoff_multiplier: float = 2.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreemptionConfig:
        """Create from dictionary."""
        return cls(
            enabled=bool(data.get("enabled", True)),
            check_interval_seconds=float(data.get("check_interval_seconds", 5.0)),
            warning_seconds=float(data.get("warning_seconds", 30.0)),
            checkpoint_on_preemption=bool(data.get("checkpoint_on_preemption", True)),
            max_retry_count=int(data.get("max_retry_count", 3)),
            backoff_multiplier=float(data.get("backoff_multiplier", 2.0)),
        )

    @classmethod
    def from_env(cls) -> PreemptionConfig:
        """Create from environment variables."""
        return cls(
            enabled=os.environ.get("CIV6AZ_PREEMPTION_ENABLED", "true").lower()
            == "true",
            check_interval_seconds=float(
                os.environ.get("CIV6AZ_PREEMPTION_CHECK_INTERVAL", "5.0")
            ),
            warning_seconds=float(
                os.environ.get("CIV6AZ_PREEMPTION_WARNING_SECONDS", "30.0")
            ),
            checkpoint_on_preemption=os.environ.get(
                "CIV6AZ_PREEMPTION_CHECKPOINT", "true"
            ).lower()
            == "true",
            max_retry_count=int(
                os.environ.get("CIV6AZ_PREEMPTION_MAX_RETRY", "3")
            ),
            backoff_multiplier=float(
                os.environ.get("CIV6AZ_PREEMPTION_BACKOFF", "2.0")
            ),
        )


@dataclass
class CloudLoggingConfig:
    """Cloud logging configuration.

    Attributes:
        enabled: Enable cloud logging.
        log_name: Cloud Logging log name.
        structured: Use structured logging format.
        buffer_size: Log buffer size before flush.
        flush_interval_seconds: Interval to flush logs.
    """

    enabled: bool = True
    log_name: str = "civ6-alphazero-training"
    structured: bool = True
    buffer_size: int = 100
    flush_interval_seconds: float = 30.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CloudLoggingConfig:
        """Create from dictionary."""
        return cls(
            enabled=bool(data.get("enabled", True)),
            log_name=str(data.get("log_name", "civ6-alphazero-training")),
            structured=bool(data.get("structured", True)),
            buffer_size=int(data.get("buffer_size", 100)),
            flush_interval_seconds=float(data.get("flush_interval_seconds", 30.0)),
        )

    @classmethod
    def from_env(cls) -> CloudLoggingConfig:
        """Create from environment variables."""
        return cls(
            enabled=os.environ.get("CIV6AZ_CLOUD_LOGGING_ENABLED", "true").lower()
            == "true",
            log_name=os.environ.get(
                "CIV6AZ_CLOUD_LOGGING_NAME", "civ6-alphazero-training"
            ),
            structured=os.environ.get(
                "CIV6AZ_CLOUD_LOGGING_STRUCTURED", "true"
            ).lower()
            == "true",
            buffer_size=int(os.environ.get("CIV6AZ_CLOUD_LOGGING_BUFFER_SIZE", "100")),
            flush_interval_seconds=float(
                os.environ.get("CIV6AZ_CLOUD_LOGGING_FLUSH_INTERVAL", "30.0")
            ),
        )


@dataclass
class CloudMonitoringConfig:
    """Cloud monitoring configuration.

    Attributes:
        enabled: Enable cloud monitoring.
        metric_prefix: Prefix for custom metrics.
        export_interval_seconds: Interval to export metrics.
        export_metrics: List of metrics to export.
    """

    enabled: bool = True
    metric_prefix: str = "custom.googleapis.com/alphazero"
    export_interval_seconds: float = 60.0
    export_metrics: list[str] = field(
        default_factory=lambda: [
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "learning_rate",
            "games_played",
            "training_step",
        ]
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CloudMonitoringConfig:
        """Create from dictionary."""
        default_metrics = [
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
            "learning_rate",
            "games_played",
            "training_step",
        ]
        return cls(
            enabled=bool(data.get("enabled", True)),
            metric_prefix=str(
                data.get("metric_prefix", "custom.googleapis.com/alphazero")
            ),
            export_interval_seconds=float(data.get("export_interval_seconds", 60.0)),
            export_metrics=list(data.get("export_metrics", default_metrics)),
        )

    @classmethod
    def from_env(cls) -> CloudMonitoringConfig:
        """Create from environment variables."""
        return cls(
            enabled=os.environ.get("CIV6AZ_CLOUD_MONITORING_ENABLED", "true").lower()
            == "true",
            metric_prefix=os.environ.get(
                "CIV6AZ_CLOUD_MONITORING_PREFIX", "custom.googleapis.com/alphazero"
            ),
            export_interval_seconds=float(
                os.environ.get("CIV6AZ_CLOUD_MONITORING_INTERVAL", "60.0")
            ),
        )


@dataclass
class TrainingConfig:
    """Training configuration for cloud runs.

    Attributes:
        batch_size: Training batch size.
        learning_rate: Initial learning rate.
        games_per_iteration: Self-play games per iteration.
        num_iterations: Total training iterations.
        checkpoint_interval: Steps between checkpoints.
        eval_interval: Steps between evaluations.
    """

    batch_size: int = 256
    learning_rate: float = 0.001
    games_per_iteration: int = 100
    num_iterations: int = 1000
    checkpoint_interval: int = 100
    eval_interval: int = 500

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Create from dictionary."""
        return cls(
            batch_size=int(data.get("batch_size", 256)),
            learning_rate=float(data.get("learning_rate", 0.001)),
            games_per_iteration=int(data.get("games_per_iteration", 100)),
            num_iterations=int(data.get("num_iterations", 1000)),
            checkpoint_interval=int(data.get("checkpoint_interval", 100)),
            eval_interval=int(data.get("eval_interval", 500)),
        )

    @classmethod
    def from_env(cls) -> TrainingConfig:
        """Create from environment variables."""
        return cls(
            batch_size=int(os.environ.get("CIV6AZ_TRAINING_BATCH_SIZE", "256")),
            learning_rate=float(os.environ.get("CIV6AZ_TRAINING_LR", "0.001")),
            games_per_iteration=int(
                os.environ.get("CIV6AZ_TRAINING_GAMES_PER_ITER", "100")
            ),
            num_iterations=int(os.environ.get("CIV6AZ_TRAINING_NUM_ITERATIONS", "1000")),
            checkpoint_interval=int(
                os.environ.get("CIV6AZ_TRAINING_CHECKPOINT_INTERVAL", "100")
            ),
            eval_interval=int(os.environ.get("CIV6AZ_TRAINING_EVAL_INTERVAL", "500")),
        )


# =============================================================================
# Main CloudConfig (Flat structure for backward compatibility)
# =============================================================================


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

    # Phase 3 nested configs (optional)
    gcs: GCSConfig = field(default_factory=GCSConfig)
    vertex: VertexConfig = field(default_factory=VertexConfig)
    preemption: PreemptionConfig = field(default_factory=PreemptionConfig)
    logging: CloudLoggingConfig = field(default_factory=CloudLoggingConfig)
    monitoring: CloudMonitoringConfig = field(default_factory=CloudMonitoringConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Normalize paths to use forward slashes
        if self.checkpoint_prefix and not self.checkpoint_prefix.endswith("/"):
            object.__setattr__(self, "checkpoint_prefix", self.checkpoint_prefix + "/")
        if self.artifact_prefix and not self.artifact_prefix.endswith("/"):
            object.__setattr__(self, "artifact_prefix", self.artifact_prefix + "/")

        # Sync flat config to nested configs if bucket_name set but gcs not
        if self.bucket_name and not self.gcs.bucket_name:
            self.gcs = GCSConfig(
                bucket_name=self.bucket_name,
                checkpoint_prefix=self.checkpoint_prefix.rstrip("/"),
                artifact_prefix=self.artifact_prefix.rstrip("/"),
            )

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

        # Also load nested configs from environment
        kwargs["gcs"] = GCSConfig.from_env()
        kwargs["vertex"] = VertexConfig.from_env()
        kwargs["preemption"] = PreemptionConfig.from_env()
        kwargs["logging"] = CloudLoggingConfig.from_env()
        kwargs["monitoring"] = CloudMonitoringConfig.from_env()
        kwargs["training"] = TrainingConfig.from_env()

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
            cloud_data = data["cloud"]
        elif "gcp" in data:
            cloud_data = data["gcp"]
        else:
            cloud_data = data

        # Filter to valid fields only (flat structure)
        flat_fields = {
            "project_id", "region", "zone", "bucket_name",
            "checkpoint_prefix", "artifact_prefix",
            "enable_cloud_logging", "enable_cloud_monitoring", "log_level",
            "service_account", "network", "subnetwork",
            "machine_type", "accelerator_type", "accelerator_count", "replica_count",
            "use_spot", "max_run_duration_seconds",
            "preemption_check_interval", "checkpoint_on_preemption",
            "custom_labels",
        }
        kwargs = {k: v for k, v in cloud_data.items() if k in flat_fields}

        # Load nested configs if present
        if "gcs" in cloud_data:
            kwargs["gcs"] = GCSConfig.from_dict(cloud_data["gcs"])
        if "vertex" in cloud_data:
            kwargs["vertex"] = VertexConfig.from_dict(cloud_data["vertex"])
        if "preemption" in cloud_data:
            kwargs["preemption"] = PreemptionConfig.from_dict(cloud_data["preemption"])
        if "logging" in cloud_data:
            kwargs["logging"] = CloudLoggingConfig.from_dict(cloud_data["logging"])
        if "monitoring" in cloud_data:
            kwargs["monitoring"] = CloudMonitoringConfig.from_dict(
                cloud_data["monitoring"]
            )
        if "training" in data:
            kwargs["training"] = TrainingConfig.from_dict(data["training"])

        return cls(**kwargs)

    def validate(self, require_project: bool = True) -> list[str]:
        """Validate configuration settings.

        Args:
            require_project: Whether project_id is required.

        Returns:
            List of validation error messages (empty if valid).
        """
        issues: list[str] = []

        if require_project and not self.project_id:
            issues.append(
                "project_id is required. Set GCP_PROJECT_ID environment variable."
            )

        if self.use_spot and self.max_run_duration_seconds < 60:
            issues.append(
                "max_run_duration_seconds must be at least 60 for spot instances"
            )

        if self.accelerator_count < 0:
            issues.append("accelerator_count must be non-negative")

        if self.replica_count < 1:
            issues.append("replica_count must be at least 1")

        return issues

    def is_valid(self, require_project: bool = True) -> bool:
        """Check if configuration is valid.

        Args:
            require_project: Whether project_id is required.

        Returns:
            True if configuration is valid, False otherwise.
        """
        return len(self.validate(require_project=require_project)) == 0

    def get_checkpoint_uri(self, checkpoint_name: str) -> str:
        """Get full GCS URI for a checkpoint.

        Args:
            checkpoint_name: Name of the checkpoint file.

        Returns:
            Full gs:// URI for the checkpoint.
        """
        bucket = self.bucket_name or self.gcs.bucket_name
        if not bucket:
            raise CloudConfigError("bucket_name is required for checkpoint URIs")
        prefix = self.checkpoint_prefix or self.gcs.checkpoint_prefix
        return f"gs://{bucket}/{prefix}{checkpoint_name}"

    def get_artifact_uri(self, artifact_name: str) -> str:
        """Get full GCS URI for an artifact.

        Args:
            artifact_name: Name of the artifact.

        Returns:
            Full gs:// URI for the artifact.
        """
        bucket = self.bucket_name or self.gcs.bucket_name
        if not bucket:
            raise CloudConfigError("bucket_name is required for artifact URIs")
        prefix = self.artifact_prefix or self.gcs.artifact_prefix
        return f"gs://{bucket}/{prefix}{artifact_name}"

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
