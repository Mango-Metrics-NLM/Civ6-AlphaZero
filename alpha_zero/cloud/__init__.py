"""Cloud infrastructure for distributed training.

This module provides cloud provider integrations for training
AlphaZero models on cloud infrastructure.

Supported Providers:
- Google Cloud Platform (GCP) with Vertex AI

Example:
    from alpha_zero.cloud import CloudConfig, get_cloud_provider
    from alpha_zero.cloud.gcp import GCPConfig, VertexTrainer

    # Configure cloud provider
    config = GCPConfig.from_env()
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
    CloudStorageProtocol,
    CloudTrainerProtocol,
)

__all__ = [
    # Config
    "CloudConfig",
    "CloudEnvironment",
    "CloudProvider",
    # Protocols
    "CloudStorageProtocol",
    "CloudTrainerProtocol",
]
