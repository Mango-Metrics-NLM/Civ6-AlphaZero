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

__all__ = [
    # Config
    "CloudConfig",
    "CloudEnvironment",
    "CloudProvider",
    # Protocols
    "CloudStorageProtocol",
    "CloudTrainerProtocol",
    "CloudMonitoringProtocol",
    "PreemptionHandlerProtocol",
    # Protocol support types
    "TrainingJobInfo",
]

# Add exceptions to __all__ if available
if _EXCEPTIONS_AVAILABLE:
    __all__.extend(
        [
            "CloudConfigError",
            "CloudStorageError",
            "CloudConnectionError",
            "PreemptionError",
        ]
    )
