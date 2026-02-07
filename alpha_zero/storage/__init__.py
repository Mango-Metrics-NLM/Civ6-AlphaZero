"""Storage backends for Civ6 AlphaZero.

This module provides abstract storage interfaces and implementations for:
- Model weights storage and retrieval
- Training trajectory persistence
- Checkpoint management
- Evaluation results storage

Supports both local filesystem and cloud storage (GCS) backends with
a unified interface for seamless switching between development and production.

Usage:
    from alpha_zero.storage import create_storage_backend, StorageConfig

    # Local storage (default)
    config = StorageConfig(backend_type="local", base_path="./checkpoints")
    storage = create_storage_backend(config)

    # GCS storage
    config = StorageConfig(
        backend_type="gcs",
        gcs_bucket="my-bucket",
        gcs_prefix="civ6-alphazero",
    )
    storage = create_storage_backend(config)

    # Save and load models
    storage.save_model(model_state, iteration=100)
    model_state = storage.load_champion()

    # Save trajectories
    storage.save_trajectory(trajectory, worker_id=0, game_id=42)
"""

from alpha_zero.storage.backend import StorageBackend
from alpha_zero.storage.config import GCSConfig, StorageConfig
from alpha_zero.storage.factory import create_storage_backend, get_storage_backend_from_env
from alpha_zero.storage.local_backend import LocalStorageBackend
from alpha_zero.storage.models import (
    CheckpointMetadata,
    EvaluationResult,
    ModelMetadata,
    TrajectoryMetadata,
)
from alpha_zero.storage.replay_buffer import (
    InMemoryReplayBuffer,
    ReplayBufferConfig,
    StorageReplayBuffer,
)

__all__ = [
    "CheckpointMetadata",
    "EvaluationResult",
    "GCSConfig",
    "InMemoryReplayBuffer",
    # Implementations
    "LocalStorageBackend",
    # Models
    "ModelMetadata",
    # Replay Buffer
    "ReplayBufferConfig",
    # Core interfaces
    "StorageBackend",
    "StorageConfig",
    "StorageReplayBuffer",
    "TrajectoryMetadata",
    # Factory
    "create_storage_backend",
    "get_storage_backend_from_env",
]

# Lazy imports for optional GCS backend
def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "GCSStorageBackend":
        from alpha_zero.storage.gcs_backend import GCSStorageBackend
        return GCSStorageBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
