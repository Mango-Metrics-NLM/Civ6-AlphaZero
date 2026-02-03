"""
GCP Cloud Integration Module for Civ6 AlphaZero

Provides cloud infrastructure components for distributed training:
- GCS storage for models, trajectories, and checkpoints
- Vertex AI job management for self-play and training
- Configuration with environment variable support

Usage:
    from alpha_zero.cloud.gcp import GCPConfig, GCSModelStorage

    config = GCPConfig.from_env()
    storage = GCSModelStorage(config)
    storage.save_model(model, version=1, metadata={"elo": 1000})
"""

from alpha_zero.cloud.gcp.config import (
    GCPConfig,
    SelfPlayWorkerConfig,
    TrainingWorkerConfig,
    VertexTier,
)
from alpha_zero.cloud.gcp.exceptions import (
    ConfigurationError,
    GCPError,
    GCSStorageError,
    PreemptionError,
    QuotaExceededError,
    VertexJobError,
)
from alpha_zero.cloud.gcp.storage import (
    GCSBackend,
    GCSCheckpointStorage,
    GCSModelStorage,
    GCSTrajectoryStorage,
    ModelMetadata,
)
from alpha_zero.cloud.gcp.vertex_jobs import (
    JobResult,
    JobSpec,
    JobState,
    VertexAIClient,
    VertexSelfPlayJob,
    VertexTrainingJob,
)

__all__ = [
    # Config
    "GCPConfig",
    "SelfPlayWorkerConfig",
    "TrainingWorkerConfig",
    "VertexTier",
    # Exceptions
    "GCPError",
    "GCSStorageError",
    "VertexJobError",
    "ConfigurationError",
    "PreemptionError",
    "QuotaExceededError",
    # Storage
    "GCSBackend",
    "GCSModelStorage",
    "GCSTrajectoryStorage",
    "GCSCheckpointStorage",
    "ModelMetadata",
    # Vertex AI Jobs
    "JobSpec",
    "JobResult",
    "JobState",
    "VertexAIClient",
    "VertexSelfPlayJob",
    "VertexTrainingJob",
]
