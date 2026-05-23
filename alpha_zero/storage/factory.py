"""Factory functions for creating storage backends.

Provides a unified factory for creating storage backends based on
configuration, with support for environment variable overrides.

Usage:
    from alpha_zero.storage import create_storage_backend, StorageConfig

    # Automatic backend selection based on environment
    storage = create_storage_backend()

    # Explicit local storage
    config = StorageConfig.for_local("./data")
    storage = create_storage_backend(config)

    # Explicit GCS storage
    config = StorageConfig.for_gcs("my-bucket", "civ6-alphazero")
    storage = create_storage_backend(config)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from alpha_zero.storage.config import StorageConfig
from alpha_zero.storage.local_backend import LocalStorageBackend

if TYPE_CHECKING:
    from alpha_zero.storage.backend import StorageBackend

logger = logging.getLogger(__name__)


def create_storage_backend(
    config: StorageConfig | None = None,
    backend_type: str | None = None,
) -> StorageBackend:
    """Create a storage backend based on configuration.

    This factory function creates the appropriate storage backend
    based on the configuration or environment variables.

    Priority for backend type selection:
    1. Explicit backend_type parameter
    2. Config's backend_type
    3. CIV6_STORAGE_BACKEND environment variable
    4. AIP_CHECKPOINT_DIR presence (implies GCS on Vertex AI)
    5. Default to "local"

    Args:
        config: Storage configuration. If None, creates from environment.
        backend_type: Explicit backend type override ("local" or "gcs").

    Returns:
        Configured storage backend instance.

    Raises:
        ValueError: If backend type is invalid or configuration is missing.
        ImportError: If GCS backend is requested but dependencies missing.

    Example:
        # Auto-detect from environment
        storage = create_storage_backend()

        # Explicit local
        storage = create_storage_backend(backend_type="local")

        # Explicit GCS
        config = StorageConfig.for_gcs("my-bucket")
        storage = create_storage_backend(config)
    """
    # Create config from environment if not provided
    if config is None:
        config = StorageConfig.from_env()

    # Determine backend type
    effective_backend = backend_type or config.backend_type

    # Check for Vertex AI environment
    if effective_backend == "local" and os.environ.get("AIP_CHECKPOINT_DIR"):
        aip_dir = os.environ["AIP_CHECKPOINT_DIR"]
        if aip_dir.startswith("gs://"):
            logger.info("Detected Vertex AI environment, switching to GCS backend")
            effective_backend = "gcs"
            config._parse_gcs_path(aip_dir)
            config.backend_type = "gcs"

    # Create appropriate backend
    if effective_backend == "local":
        logger.info(f"Creating LocalStorageBackend at {config.base_path}")
        return LocalStorageBackend(config)

    elif effective_backend == "gcs":
        # Import GCS backend only when needed
        try:
            from alpha_zero.storage.gcs_backend import GCSStorageBackend
        except ImportError as e:
            raise ImportError(
                "GCS backend requires google-cloud-storage. "
                "Install with: pip install google-cloud-storage"
            ) from e

        if not config.gcs.bucket:
            raise ValueError(
                "GCS bucket name is required for GCS backend. "
                "Set CIV6_GCS_BUCKET environment variable or configure gcs.bucket."
            )

        logger.info(f"Creating GCSStorageBackend: gs://{config.gcs.bucket}/{config.gcs.prefix}")
        return GCSStorageBackend(config)

    elif effective_backend == "spacetimedb":
        # Import SpacetimeDB backend only when needed
        try:
            from alpha_zero.storage.spacetimedb_backend import SpacetimeDBStorageBackend
        except ImportError as e:
            raise ImportError(
                "SpacetimeDB backend requires spacetimedb-sdk. "
                "Install with: pip install spacetimedb-sdk"
            ) from e

        stdb_config = getattr(config, "spacetimedb", None)
        host = stdb_config.host if stdb_config else "http://localhost:3000"
        module = stdb_config.module_address if stdb_config else ""
        logger.info("Creating SpacetimeDBStorageBackend (host=%s, module=%s)", host, module)
        return SpacetimeDBStorageBackend(config)

    else:
        raise ValueError(
            f"Invalid backend type: {effective_backend}. Must be 'local', 'gcs', or 'spacetimedb'."
        )


def get_storage_backend_from_env() -> StorageBackend:
    """Create storage backend purely from environment variables.

    This is a convenience function for use in containerized environments
    where all configuration comes from environment variables.

    Environment Variables:
        CIV6_STORAGE_BACKEND: Backend type ("local" or "gcs")
        CIV6_STORAGE_BASE_PATH: Local storage base path
        CIV6_GCS_BUCKET: GCS bucket name
        CIV6_GCS_PREFIX: GCS path prefix
        AIP_CHECKPOINT_DIR: Vertex AI checkpoint directory

    Returns:
        Configured storage backend instance.
    """
    return create_storage_backend(StorageConfig.from_env())


__all__ = [
    "create_storage_backend",
    "get_storage_backend_from_env",
]
