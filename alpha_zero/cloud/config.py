"""Cloud training configuration system.

This module provides configuration dataclasses for cloud training:
- CloudConfig: Main cloud configuration
- VertexConfig: Vertex AI job settings
- GCSConfig: Google Cloud Storage settings
- PreemptionConfig: Preemption handling settings

All values can be:
- Loaded from YAML files
- Overridden via environment variables (CIV6AZ_ prefix)
- Set programmatically

Usage:
    from alpha_zero.cloud.config import CloudConfig

    # Load from file
    config = CloudConfig.from_yaml("configs/cloud_training.yaml")

    # Load from environment
    config = CloudConfig.from_env()

    # Environment variable overrides
    # CIV6AZ_CLOUD_PROJECT_ID=my-project will set cloud.project_id
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class CloudProvider(str, Enum):
    """Supported cloud providers."""

    GCP = "gcp"
    AWS = "aws"
    AZURE = "azure"
    LOCAL = "local"


class CloudEnvironment(str, Enum):
    """Cloud environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


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


@dataclass
class CloudConfig:
    """Main cloud training configuration.

    Combines all cloud-related configuration:
    - GCP project and region settings
    - Vertex AI job configuration
    - GCS storage configuration
    - Preemption handling
    - Cloud logging and monitoring
    - Training parameters

    Attributes:
        provider: Cloud provider (gcp, aws, azure, local).
        environment: Environment type (development, staging, production, test).
        project_id: GCP project ID.
        region: GCP region.
        zone: GCP zone.
        gcs: GCS storage configuration.
        vertex: Vertex AI configuration.
        preemption: Preemption handling configuration.
        logging: Cloud logging configuration.
        monitoring: Cloud monitoring configuration.
        training: Training configuration.
        labels: Custom labels for resources.
    """

    provider: str = CloudProvider.GCP.value
    environment: str = CloudEnvironment.DEVELOPMENT.value
    project_id: str = ""
    region: str = "us-central1"
    zone: str = "us-central1-a"
    gcs: GCSConfig = field(default_factory=GCSConfig)
    vertex: VertexConfig = field(default_factory=VertexConfig)
    preemption: PreemptionConfig = field(default_factory=PreemptionConfig)
    logging: CloudLoggingConfig = field(default_factory=CloudLoggingConfig)
    monitoring: CloudMonitoringConfig = field(default_factory=CloudMonitoringConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CloudConfig:
        """Create from dictionary."""
        cloud_data = data.get("cloud", data)
        return cls(
            provider=str(cloud_data.get("provider", CloudProvider.GCP.value)),
            environment=str(
                cloud_data.get("environment", CloudEnvironment.DEVELOPMENT.value)
            ),
            project_id=str(cloud_data.get("project_id", "")),
            region=str(cloud_data.get("region", "us-central1")),
            zone=str(cloud_data.get("zone", "us-central1-a")),
            gcs=GCSConfig.from_dict(cloud_data.get("gcs", {})),
            vertex=VertexConfig.from_dict(cloud_data.get("vertex", {})),
            preemption=PreemptionConfig.from_dict(cloud_data.get("preemption", {})),
            logging=CloudLoggingConfig.from_dict(cloud_data.get("logging", {})),
            monitoring=CloudMonitoringConfig.from_dict(
                cloud_data.get("monitoring", {})
            ),
            training=TrainingConfig.from_dict(data.get("training", {})),
            labels=dict(cloud_data.get("labels", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> CloudConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            CloudConfig instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML parsing fails.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls.from_dict(data)

        # Apply environment variable overrides
        return config.apply_env_overrides()

    @classmethod
    def from_env(cls) -> CloudConfig:
        """Create configuration from environment variables."""
        return cls(
            provider=os.environ.get("CIV6AZ_CLOUD_PROVIDER", CloudProvider.GCP.value),
            environment=os.environ.get(
                "CIV6AZ_CLOUD_ENVIRONMENT", CloudEnvironment.DEVELOPMENT.value
            ),
            project_id=os.environ.get("CIV6AZ_GCP_PROJECT_ID", ""),
            region=os.environ.get("CIV6AZ_GCP_REGION", "us-central1"),
            zone=os.environ.get("CIV6AZ_GCP_ZONE", "us-central1-a"),
            gcs=GCSConfig.from_env(),
            vertex=VertexConfig.from_env(),
            preemption=PreemptionConfig.from_env(),
            logging=CloudLoggingConfig.from_env(),
            monitoring=CloudMonitoringConfig.from_env(),
            training=TrainingConfig.from_env(),
        )

    def apply_env_overrides(self) -> CloudConfig:
        """Apply environment variable overrides to configuration.

        Environment variables follow the pattern:
        CIV6AZ_<SECTION>_<FIELD>=value

        Returns:
            Updated CloudConfig with environment overrides applied.
        """
        # Override top-level fields
        if "CIV6AZ_GCP_PROJECT_ID" in os.environ:
            self.project_id = os.environ["CIV6AZ_GCP_PROJECT_ID"]
        if "CIV6AZ_GCP_REGION" in os.environ:
            self.region = os.environ["CIV6AZ_GCP_REGION"]
        if "CIV6AZ_GCP_ZONE" in os.environ:
            self.zone = os.environ["CIV6AZ_GCP_ZONE"]
        if "CIV6AZ_CLOUD_PROVIDER" in os.environ:
            self.provider = os.environ["CIV6AZ_CLOUD_PROVIDER"]
        if "CIV6AZ_CLOUD_ENVIRONMENT" in os.environ:
            self.environment = os.environ["CIV6AZ_CLOUD_ENVIRONMENT"]

        # Override nested configs from environment
        self.gcs = GCSConfig.from_env() if os.environ.get("CIV6AZ_GCS_BUCKET") else self.gcs
        if os.environ.get("CIV6AZ_VERTEX_MACHINE_TYPE"):
            self.vertex = VertexConfig.from_env()

        return self

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid).
        """
        issues: list[str] = []

        if not self.project_id:
            issues.append("project_id is required for GCP operations")

        if not self.gcs.bucket_name:
            issues.append("gcs.bucket_name is required for checkpoint storage")

        if self.vertex.accelerator_count < 0:
            issues.append("vertex.accelerator_count must be non-negative")

        if self.training.batch_size <= 0:
            issues.append("training.batch_size must be positive")

        if self.training.learning_rate <= 0:
            issues.append("training.learning_rate must be positive")

        return issues

    def is_valid(self) -> bool:
        """Check if configuration is valid.

        Returns:
            True if configuration is valid, False otherwise.
        """
        return len(self.validate()) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "cloud": {
                "provider": self.provider,
                "environment": self.environment,
                "project_id": self.project_id,
                "region": self.region,
                "zone": self.zone,
                "gcs": {
                    "bucket_name": self.gcs.bucket_name,
                    "checkpoint_prefix": self.gcs.checkpoint_prefix,
                    "artifact_prefix": self.gcs.artifact_prefix,
                    "logs_prefix": self.gcs.logs_prefix,
                    "enable_compression": self.gcs.enable_compression,
                    "max_checkpoints": self.gcs.max_checkpoints,
                },
                "vertex": {
                    "machine_type": self.vertex.machine_type,
                    "accelerator_type": self.vertex.accelerator_type,
                    "accelerator_count": self.vertex.accelerator_count,
                    "boot_disk_size_gb": self.vertex.boot_disk_size_gb,
                    "container_uri": self.vertex.container_uri,
                    "service_account": self.vertex.service_account,
                    "network": self.vertex.network,
                    "max_run_duration_seconds": self.vertex.max_run_duration_seconds,
                    "enable_web_access": self.vertex.enable_web_access,
                },
                "preemption": {
                    "enabled": self.preemption.enabled,
                    "check_interval_seconds": self.preemption.check_interval_seconds,
                    "warning_seconds": self.preemption.warning_seconds,
                    "checkpoint_on_preemption": self.preemption.checkpoint_on_preemption,
                    "max_retry_count": self.preemption.max_retry_count,
                    "backoff_multiplier": self.preemption.backoff_multiplier,
                },
                "logging": {
                    "enabled": self.logging.enabled,
                    "log_name": self.logging.log_name,
                    "structured": self.logging.structured,
                    "buffer_size": self.logging.buffer_size,
                    "flush_interval_seconds": self.logging.flush_interval_seconds,
                },
                "monitoring": {
                    "enabled": self.monitoring.enabled,
                    "metric_prefix": self.monitoring.metric_prefix,
                    "export_interval_seconds": self.monitoring.export_interval_seconds,
                    "export_metrics": self.monitoring.export_metrics,
                },
                "labels": self.labels,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "games_per_iteration": self.training.games_per_iteration,
                "num_iterations": self.training.num_iterations,
                "checkpoint_interval": self.training.checkpoint_interval,
                "eval_interval": self.training.eval_interval,
            },
        }

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save the configuration.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
