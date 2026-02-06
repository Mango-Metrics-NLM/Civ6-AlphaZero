"""Cloud infrastructure for distributed training.

This module provides cloud provider integrations for training
AlphaZero models on cloud infrastructure. Supports multiple providers
with pluggable backends for storage, training jobs, monitoring, and preemption handling.

Supported Providers:
- Google Cloud Platform (GCP) with Vertex AI

Example:
    from alpha_zero.cloud import CloudConfig, CloudProvider
    from alpha_zero.cloud.gcp import GCPConfig, VertexTrainer

    # Configure cloud provider
    config = CloudConfig.from_env()
    trainer = VertexTrainer(config)

    # Submit training job
    job = trainer.submit_training_job(
        script_path="train.py",
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
    )

Phase 3 Usage:
    from alpha_zero.cloud import CloudConfig, GCSCheckpointManager

    config = CloudConfig.from_yaml("configs/cloud_training.yaml")
    checkpoint_manager = GCSCheckpointManager(
        bucket_name=config.gcs.bucket_name,
        prefix=config.gcs.checkpoint_prefix,
    )
"""

from alpha_zero.cloud.config import (
    CloudConfig,
    CloudEnvironment,
    CloudProvider,
)
from alpha_zero.cloud.protocols import (
    CloudMonitoringProtocol,
    CloudStorageProtocol,
    CloudTrainerProtocol,
    PreemptionHandlerProtocol,
    TrainingJobInfo,
)

# Try to import exceptions, they may not exist yet
try:
    from alpha_zero.cloud.exceptions import (  # noqa: F401
        CloudConfigError,
        CloudConnectionError,
        CloudStorageError,
        PreemptionError,
    )

    _EXCEPTIONS_AVAILABLE = True
except ImportError:
    _EXCEPTIONS_AVAILABLE = False

# Try to import Phase 3 cloud components
try:
    from alpha_zero.cloud.cloud_logging import (  # noqa: F401
        CloudLogger,
        CloudMetricsExporter,
    )
    from alpha_zero.cloud.gcs_storage import (  # noqa: F401
        CheckpointMetadata,
        GCSCheckpointManager,
        GCSModelStorage,
    )
    from alpha_zero.cloud.preemption import (  # noqa: F401
        PreemptionHandler,
        PreemptionState,
        create_preemption_handler,
    )
    from alpha_zero.cloud.vertex_launcher import (  # noqa: F401
        JobInfo,
        JobStatus,
        VertexJobLauncher,
        create_launcher_from_config,
    )

    _PHASE3_AVAILABLE = True
except ImportError:
    _PHASE3_AVAILABLE = False

__all__ = [
    # Config (sorted)
    "CloudConfig",
    "CloudEnvironment",
    # Protocols (sorted)
    "CloudMonitoringProtocol",
    "CloudProvider",
    "CloudStorageProtocol",
    "CloudTrainerProtocol",
    "PreemptionHandlerProtocol",
    "TrainingJobInfo",
]

# Add exceptions to __all__ if available (sorted)
if _EXCEPTIONS_AVAILABLE:
    __all__.extend(
        [
            "CloudConfigError",
            "CloudConnectionError",
            "CloudStorageError",
            "PreemptionError",
        ]
    )

# Add Phase 3 components to __all__ if available (sorted)
if _PHASE3_AVAILABLE:
    __all__.extend(
        [
            "CheckpointMetadata",
            "CloudLogger",
            "CloudMetricsExporter",
            "GCSCheckpointManager",
            "GCSModelStorage",
            "JobInfo",
            "JobStatus",
            "PreemptionHandler",
            "PreemptionState",
            "VertexJobLauncher",
            "create_launcher_from_config",
            "create_preemption_handler",
        ]
    )
