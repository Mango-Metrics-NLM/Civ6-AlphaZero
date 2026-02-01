"""Google Cloud Platform integration for AlphaZero training.

This module provides GCP-specific implementations of cloud training
and storage protocols, including:

- Vertex AI custom training jobs
- Cloud Storage for checkpoints and artifacts
- Authentication helpers

Example:
    from alpha_zero.cloud.gcp import GCPConfig, VertexTrainer, CloudStorageClient

    # Configure GCP
    config = GCPConfig.from_env()

    # Submit training job
    trainer = VertexTrainer(config)
    job_id = trainer.submit_training_job(
        script_path="train.py",
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
    )

    # Upload checkpoint
    storage = CloudStorageClient(config)
    storage.upload_file("model.pt", "checkpoints/model.pt")
"""

from alpha_zero.cloud.gcp.config import GCPConfig, StorageConfig, VertexConfig
from alpha_zero.cloud.gcp.storage import CloudStorageClient
from alpha_zero.cloud.gcp.vertex_trainer import VertexTrainer

__all__ = [
    # Config
    "GCPConfig",
    "VertexConfig",
    "StorageConfig",
    # Implementations
    "CloudStorageClient",
    "VertexTrainer",
]
