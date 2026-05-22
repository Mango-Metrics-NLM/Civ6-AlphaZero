"""Configuration for storage backends.

Provides a unified configuration class that supports both local
and GCS storage backends with environment variable overrides.

Environment Variables:
    CIV6_STORAGE_BACKEND: Backend type ("local" or "gcs")
    CIV6_GCS_BUCKET: GCS bucket name
    CIV6_GCS_PREFIX: GCS path prefix
    CIV6_STORAGE_BASE_PATH: Local storage base path
    AIP_CHECKPOINT_DIR: Vertex AI checkpoint directory (auto-set)
    AIP_MODEL_DIR: Vertex AI model directory (auto-set)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from alpha_zero.spacetimedb.config import SpacetimeDBConfig

from alpha_zero.utils.constants import PATHS


@dataclass
class GCSConfig:
    """Configuration specific to GCS storage.

    Attributes:
        bucket: GCS bucket name.
        prefix: Path prefix within bucket.
        project: GCP project ID (optional, uses default).
        credentials_path: Path to service account key file.
        timeout: Request timeout in seconds.
        retry_max_attempts: Maximum retry attempts for failed operations.
        retry_initial_delay: Initial retry delay in seconds.
        retry_max_delay: Maximum retry delay in seconds.
    """

    bucket: str = ""
    prefix: str = "civ6-alphazero"
    project: str | None = None
    credentials_path: str | None = None
    timeout: float = 60.0
    retry_max_attempts: int = 5
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 30.0

    @classmethod
    def from_env(cls) -> GCSConfig:
        """Create from environment variables."""
        return cls(
            bucket=os.environ.get("CIV6_GCS_BUCKET", ""),
            prefix=os.environ.get("CIV6_GCS_PREFIX", "civ6-alphazero"),
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            credentials_path=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
            timeout=float(os.environ.get("CIV6_GCS_TIMEOUT", "60.0")),
            retry_max_attempts=int(os.environ.get("CIV6_GCS_RETRY_MAX", "5")),
        )


@dataclass
class StorageConfig:
    """Unified storage configuration.

    Supports both local filesystem and GCS backends with automatic
    environment variable overrides for cloud deployment.

    Attributes:
        backend_type: Storage backend ("local" or "gcs").
        base_path: Base path for local storage.
        gcs: GCS-specific configuration.
        compress_weights: Whether to compress model weights.
        compress_trajectories: Whether to compress trajectories.
        models_subdir: Subdirectory for model checkpoints.
        games_subdir: Subdirectory for game trajectories.
        eval_subdir: Subdirectory for evaluation results.
        checkpoints_subdir: Subdirectory for training checkpoints.
        champion_subdir: Subdirectory for champion model.
    """

    backend_type: Literal["local", "gcs", "spacetimedb"] = "local"
    base_path: str = PATHS.DEFAULT_LOCAL_STORAGE_ROOT
    gcs: GCSConfig = field(default_factory=GCSConfig)
    spacetimedb: SpacetimeDBConfig | None = None

    # Compression settings
    compress_weights: bool = True
    compress_trajectories: bool = False

    # Directory structure
    models_subdir: str = "models"
    games_subdir: str = "games"
    eval_subdir: str = "eval"
    checkpoints_subdir: str = "checkpoints"
    champion_subdir: str = "champion"
    configs_subdir: str = "configs"
    logs_subdir: str = "logs"

    # Vertex AI integration
    use_vertex_ai_paths: bool = True

    def __post_init__(self) -> None:
        """Apply environment overrides after initialization."""
        self._apply_env_overrides()
        self._apply_vertex_ai_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        if env_backend := os.environ.get("CIV6_STORAGE_BACKEND"):
            if env_backend in ("local", "gcs", "spacetimedb"):
                self.backend_type = env_backend  # type: ignore[assignment]

        if env_path := os.environ.get("CIV6_STORAGE_BASE_PATH"):
            self.base_path = env_path

        # Update GCS config from environment only if bucket env var is set
        # This avoids overwriting programmatically set GCS config
        if self.backend_type == "gcs" and os.environ.get("CIV6_GCS_BUCKET"):
            self.gcs = GCSConfig.from_env()

    def _apply_vertex_ai_overrides(self) -> None:
        """Apply Vertex AI environment variable overrides.

        Vertex AI sets these environment variables automatically:
        - AIP_CHECKPOINT_DIR: Directory for saving/loading checkpoints
        - AIP_MODEL_DIR: Directory for final model output
        - AIP_TENSORBOARD_LOG_DIR: Directory for TensorBoard logs
        """
        if not self.use_vertex_ai_paths:
            return

        # Check for Vertex AI checkpoint directory
        if aip_checkpoint := os.environ.get("AIP_CHECKPOINT_DIR"):
            # Vertex AI paths are GCS paths (gs://bucket/path)
            if aip_checkpoint.startswith("gs://"):
                self.backend_type = "gcs"
                self._parse_gcs_path(aip_checkpoint)
            else:
                self.base_path = aip_checkpoint

    def _parse_gcs_path(self, gcs_path: str) -> None:
        """Parse a gs:// path and update GCS config."""
        if not gcs_path.startswith("gs://"):
            return

        # Remove gs:// prefix
        path = gcs_path[5:]
        parts = path.split("/", 1)
        self.gcs.bucket = parts[0]
        if len(parts) > 1:
            self.gcs.prefix = parts[1]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StorageConfig:
        """Create from dictionary."""
        gcs_data = data.get("gcs", {})
        gcs_config = GCSConfig(
            bucket=gcs_data.get("bucket", ""),
            prefix=gcs_data.get("prefix", "civ6-alphazero"),
            project=gcs_data.get("project"),
            credentials_path=gcs_data.get("credentials_path"),
            timeout=gcs_data.get("timeout", 60.0),
            retry_max_attempts=gcs_data.get("retry_max_attempts", 5),
        )

        spacetimedb_config = None
        if stdb_data := data.get("spacetimedb"):
            from alpha_zero.spacetimedb.config import SpacetimeDBConfig

            spacetimedb_config = SpacetimeDBConfig.from_dict(stdb_data)

        return cls(
            backend_type=data.get("backend_type", "local"),
            base_path=data.get("base_path", PATHS.DEFAULT_LOCAL_STORAGE_ROOT),
            gcs=gcs_config,
            spacetimedb=spacetimedb_config,
            compress_weights=data.get("compress_weights", True),
            compress_trajectories=data.get("compress_trajectories", False),
            models_subdir=data.get("models_subdir", "models"),
            games_subdir=data.get("games_subdir", "games"),
            eval_subdir=data.get("eval_subdir", "eval"),
            checkpoints_subdir=data.get("checkpoints_subdir", "checkpoints"),
            champion_subdir=data.get("champion_subdir", "champion"),
            configs_subdir=data.get("configs_subdir", "configs"),
            logs_subdir=data.get("logs_subdir", "logs"),
            use_vertex_ai_paths=data.get("use_vertex_ai_paths", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {
            "backend_type": self.backend_type,
            "base_path": self.base_path,
            "gcs": {
                "bucket": self.gcs.bucket,
                "prefix": self.gcs.prefix,
                "project": self.gcs.project,
                "timeout": self.gcs.timeout,
                "retry_max_attempts": self.gcs.retry_max_attempts,
            },
            "compress_weights": self.compress_weights,
            "compress_trajectories": self.compress_trajectories,
            "models_subdir": self.models_subdir,
            "games_subdir": self.games_subdir,
            "eval_subdir": self.eval_subdir,
            "checkpoints_subdir": self.checkpoints_subdir,
            "champion_subdir": self.champion_subdir,
            "configs_subdir": self.configs_subdir,
            "logs_subdir": self.logs_subdir,
            "use_vertex_ai_paths": self.use_vertex_ai_paths,
        }
        if self.spacetimedb is not None:
            result["spacetimedb"] = self.spacetimedb.to_dict()
        return result

    @classmethod
    def for_local(
        cls,
        base_path: str = PATHS.DEFAULT_LOCAL_STORAGE_ROOT,
        compress_weights: bool = True,
    ) -> StorageConfig:
        """Create configuration for local storage."""
        return cls(
            backend_type="local",
            base_path=base_path,
            compress_weights=compress_weights,
        )

    @classmethod
    def for_gcs(
        cls,
        bucket: str,
        prefix: str = "civ6-alphazero",
        project: str | None = None,
    ) -> StorageConfig:
        """Create configuration for GCS storage."""
        return cls(
            backend_type="gcs",
            gcs=GCSConfig(
                bucket=bucket,
                prefix=prefix,
                project=project,
            ),
        )

    @classmethod
    def for_spacetimedb(
        cls,
        module_address: str,
        host: str = "http://localhost:3000",
        compress_weights: bool = True,
    ) -> StorageConfig:
        """Create configuration for SpacetimeDB storage.

        Args:
            module_address: SpacetimeDB module address.
            host: SpacetimeDB server host URL.
            compress_weights: Whether to compress model weights.

        Returns:
            StorageConfig configured for SpacetimeDB backend.
        """
        from alpha_zero.spacetimedb.config import SpacetimeDBConfig

        return cls(
            backend_type="spacetimedb",
            spacetimedb=SpacetimeDBConfig(
                enabled=True,
                host=host,
                module_address=module_address,
            ),
            compress_weights=compress_weights,
        )

    @classmethod
    def from_env(cls) -> StorageConfig:
        """Create from environment variables only."""
        config = cls()
        # __post_init__ already applies env overrides
        return config

    def get_models_path(self) -> str:
        """Get full path for models directory."""
        if self.backend_type == "gcs":
            return f"gs://{self.gcs.bucket}/{self.gcs.prefix}/{self.models_subdir}"
        return f"{self.base_path}/{self.models_subdir}"

    def get_games_path(self) -> str:
        """Get full path for games directory."""
        if self.backend_type == "gcs":
            return f"gs://{self.gcs.bucket}/{self.gcs.prefix}/{self.games_subdir}"
        return f"{self.base_path}/{self.games_subdir}"

    def get_checkpoints_path(self) -> str:
        """Get full path for checkpoints directory."""
        if self.backend_type == "gcs":
            return f"gs://{self.gcs.bucket}/{self.gcs.prefix}/{self.checkpoints_subdir}"
        return f"{self.base_path}/{self.checkpoints_subdir}"


__all__ = [
    "GCSConfig",
    "StorageConfig",
]
