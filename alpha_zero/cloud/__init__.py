"""Cloud training infrastructure for Civ6 AlphaZero.

This module provides cloud training capabilities including:
- GCP/Vertex AI integration
- GCS checkpoint storage
- Preemption handling
- Cloud logging and monitoring

Usage:
    from alpha_zero.cloud import CloudConfig, GCSCheckpointManager

    config = CloudConfig.from_yaml("configs/cloud_training.yaml")
    checkpoint_manager = GCSCheckpointManager(
        bucket_name=config.gcs.bucket_name,
        prefix=config.gcs.checkpoint_prefix,
    )
"""

from __future__ import annotations

from alpha_zero.cloud.cloud_logging import CloudLogger, CloudMetricsExporter
from alpha_zero.cloud.config import (
    CloudConfig,
    CloudEnvironment,
    CloudLoggingConfig,
    CloudMonitoringConfig,
    CloudProvider,
    GCSConfig,
    PreemptionConfig,
    TrainingConfig,
    VertexConfig,
)
from alpha_zero.cloud.gcs_storage import (
    CheckpointMetadata,
    GCSCheckpointManager,
    GCSModelStorage,
)
from alpha_zero.cloud.preemption import (
    PreemptionHandler,
    PreemptionState,
    create_preemption_handler,
)
from alpha_zero.cloud.vertex_launcher import (
    JobInfo,
    JobStatus,
    VertexJobLauncher,
    create_launcher_from_config,
)

__all__ = [
    "CheckpointMetadata",
    "CloudConfig",
    "CloudEnvironment",
    "CloudLogger",
    "CloudLoggingConfig",
    "CloudMetricsExporter",
    "CloudMonitoringConfig",
    "CloudProvider",
    "GCSCheckpointManager",
    "GCSConfig",
    "GCSModelStorage",
    "JobInfo",
    "JobStatus",
    "PreemptionConfig",
    "PreemptionHandler",
    "PreemptionState",
    "TrainingConfig",
    "VertexConfig",
    "VertexJobLauncher",
    "create_launcher_from_config",
    "create_preemption_handler",
]
