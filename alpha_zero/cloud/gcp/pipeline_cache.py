"""
Vertex AI Pipeline Caching for Civ6 AlphaZero Cloud Training.

Provides caching for pipeline components to skip unchanged work and reduce costs.
Uses content-addressable storage based on component signatures (model version,
config hash, code hash).

Features:
- Automatic cache key generation from component inputs
- TTL-based cache expiration
- GCS-backed cache storage
- Cache statistics and management

All values configurable via environment variables with CIV6AZ_CACHE_ prefix.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from alpha_zero.cloud.gcp.exceptions import ConfigurationError, GCSStorageError

if TYPE_CHECKING:
    from google.cloud import storage as gcs_lib

logger = logging.getLogger(__name__)


class ComponentType(str, Enum):
    """Types of pipeline components that can be cached."""

    SELF_PLAY = "self_play"
    TRAINING = "training"
    EVALUATION = "evaluation"
    MODEL_EXPORT = "model_export"


@dataclass
class ComponentSignature:
    """Signature for a pipeline component for cache key generation.

    A signature uniquely identifies a component execution based on its
    inputs (model version, configuration, code).

    Attributes:
        component_type: Type of component.
        model_version: Model version used.
        config_hash: Hash of relevant configuration.
        code_hash: Hash of component code (optional).
        input_data_hash: Hash of input data paths/versions.
    """

    component_type: ComponentType
    model_version: int
    config_hash: str
    code_hash: str | None = None
    input_data_hash: str | None = None

    def to_cache_key(self) -> str:
        """Generate cache key from signature.

        Returns:
            A unique cache key string.
        """
        parts = [
            self.component_type.value,
            f"v{self.model_version:03d}",
            self.config_hash[:16],
        ]
        if self.code_hash:
            parts.append(self.code_hash[:8])
        if self.input_data_hash:
            parts.append(self.input_data_hash[:8])

        return "_".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "component_type": self.component_type.value,
            "model_version": self.model_version,
            "config_hash": self.config_hash,
            "code_hash": self.code_hash,
            "input_data_hash": self.input_data_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComponentSignature:
        """Create from dictionary."""
        return cls(
            component_type=ComponentType(data["component_type"]),
            model_version=data["model_version"],
            config_hash=data["config_hash"],
            code_hash=data.get("code_hash"),
            input_data_hash=data.get("input_data_hash"),
        )


@dataclass
class CacheEntry:
    """A cached pipeline component result.

    Attributes:
        signature: Component signature.
        output_path: GCS path to cached output.
        created_at: Timestamp of cache creation.
        valid_until: Optional expiration timestamp.
        metadata: Additional metadata.
    """

    signature: ComponentSignature
    output_path: str
    created_at: str
    valid_until: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.valid_until is None:
            return False
        expiry = datetime.fromisoformat(self.valid_until)
        return datetime.now(timezone.utc) > expiry

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signature": self.signature.to_dict(),
            "output_path": self.output_path,
            "created_at": self.created_at,
            "valid_until": self.valid_until,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        """Create from dictionary."""
        return cls(
            signature=ComponentSignature.from_dict(data["signature"]),
            output_path=data["output_path"],
            created_at=data["created_at"],
            valid_until=data.get("valid_until"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PipelineCacheConfig:
    """Configuration for pipeline caching.

    Environment Variables:
        CIV6AZ_CACHE_ENABLED: Enable caching (default: true)
        CIV6AZ_CACHE_TTL_HOURS: Cache TTL in hours (default: 168 = 7 days)
        CIV6AZ_CACHE_GCS_PREFIX: GCS prefix for cache (default: cache/pipeline)
        CIV6AZ_CACHE_INCLUDE_CODE_HASH: Include code hash (default: true)
        CIV6AZ_CACHE_MAX_ENTRIES: Max cache entries (default: 1000)
    """

    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_CACHE_ENABLED", "true"
        ).lower() == "true"
    )
    ttl_hours: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_CACHE_TTL_HOURS", "168"))
    )
    gcs_prefix: str = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_CACHE_GCS_PREFIX", "cache/pipeline"
        )
    )
    include_code_hash: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_CACHE_INCLUDE_CODE_HASH", "true"
        ).lower() == "true"
    )
    max_entries: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_CACHE_MAX_ENTRIES", "1000"))
    )

    # Component-specific caching settings
    cache_self_play: bool = True
    cache_training: bool = False  # Training typically shouldn't be cached
    cache_evaluation: bool = True
    cache_model_export: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        errors: list[str] = []

        if self.ttl_hours < 0:
            errors.append("ttl_hours cannot be negative")

        if self.max_entries < 1:
            errors.append("max_entries must be at least 1")

        if errors:
            raise ConfigurationError(
                f"Invalid cache configuration: {'; '.join(errors)}",
                details={"errors": errors},
            )

    @classmethod
    def from_env(cls) -> PipelineCacheConfig:
        """Create from environment variables."""
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineCacheConfig:
        """Load from YAML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Cache configuration file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineCacheConfig:
        """Create from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            ttl_hours=data.get("ttl_hours", 168),
            gcs_prefix=data.get("gcs_prefix", "cache/pipeline"),
            include_code_hash=data.get("include_code_hash", True),
            max_entries=data.get("max_entries", 1000),
            cache_self_play=data.get("cache_self_play", True),
            cache_training=data.get("cache_training", False),
            cache_evaluation=data.get("cache_evaluation", True),
            cache_model_export=data.get("cache_model_export", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "ttl_hours": self.ttl_hours,
            "gcs_prefix": self.gcs_prefix,
            "include_code_hash": self.include_code_hash,
            "max_entries": self.max_entries,
            "cache_self_play": self.cache_self_play,
            "cache_training": self.cache_training,
            "cache_evaluation": self.cache_evaluation,
            "cache_model_export": self.cache_model_export,
        }

    def should_cache(self, component_type: ComponentType) -> bool:
        """Check if component type should be cached."""
        cache_map = {
            ComponentType.SELF_PLAY: self.cache_self_play,
            ComponentType.TRAINING: self.cache_training,
            ComponentType.EVALUATION: self.cache_evaluation,
            ComponentType.MODEL_EXPORT: self.cache_model_export,
        }
        return self.enabled and cache_map.get(component_type, False)


class PipelineCache:
    """Manages caching for Vertex AI pipeline components.

    Provides cache key generation, lookup, and storage for pipeline
    components to skip unchanged work.

    Usage:
        config = PipelineCacheConfig()
        cache = PipelineCache(config, bucket="my-bucket")

        # Check cache before running
        signature = cache.compute_signature(
            component_type=ComponentType.SELF_PLAY,
            model_version=5,
            config=self_play_config,
        )

        if cache.is_cached(signature):
            output = cache.get_cached_output(signature)
        else:
            output = run_component()
            cache.store(signature, output)
    """

    def __init__(
        self,
        config: PipelineCacheConfig | None = None,
        bucket: str | None = None,
        project_id: str | None = None,
    ) -> None:
        """Initialize pipeline cache.

        Args:
            config: Cache configuration.
            bucket: GCS bucket name (without gs:// prefix).
            project_id: GCP project ID.
        """
        self.config = config or PipelineCacheConfig.from_env()
        self._bucket = (bucket or os.environ.get("CIV6AZ_GCS_BUCKET", "")).removeprefix(
            "gs://"
        )
        self._project_id = project_id or os.environ.get("CIV6AZ_VERTEX_PROJECT")
        self._client: gcs_lib.Client | None = None
        self._cache_index: dict[str, CacheEntry] = {}
        self._index_loaded = False

        logger.info(
            f"PipelineCache initialized: bucket={self._bucket}, "
            f"enabled={self.config.enabled}, ttl={self.config.ttl_hours}h"
        )

    @property
    def client(self) -> gcs_lib.Client:
        """Lazy-initialize GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage as gcs_lib
                self._client = gcs_lib.Client(project=self._project_id)
            except ImportError as err:
                raise ImportError(
                    "google-cloud-storage is required for pipeline caching. "
                    "Install with: pip install google-cloud-storage"
                ) from err
        return self._client

    def _load_index(self) -> None:
        """Load cache index from GCS."""
        if self._index_loaded:
            return

        try:
            bucket = self.client.bucket(self._bucket)
            index_blob = bucket.blob(f"{self.config.gcs_prefix}/index.json")

            if index_blob.exists():
                data = json.loads(index_blob.download_as_text())
                for key, entry_data in data.items():
                    self._cache_index[key] = CacheEntry.from_dict(entry_data)
                logger.info(f"Loaded cache index with {len(self._cache_index)} entries")
            else:
                logger.info("No existing cache index found, starting fresh")

            self._index_loaded = True

        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
            self._index_loaded = True

    def _save_index(self, max_retries: int = 3) -> None:
        """Save cache index to GCS with optimistic locking.

        Uses GCS generation preconditions to prevent race conditions when
        multiple processes attempt to update the index concurrently.

        Args:
            max_retries: Maximum retry attempts on conflict.
        """
        for attempt in range(max_retries):
            try:
                bucket = self.client.bucket(self._bucket)
                index_blob = bucket.blob(f"{self.config.gcs_prefix}/index.json")

                # Get current generation for optimistic locking
                current_generation = None
                try:
                    index_blob.reload()
                    current_generation = index_blob.generation
                except Exception:
                    # Blob doesn't exist yet, that's OK
                    pass

                data = {key: entry.to_dict() for key, entry in self._cache_index.items()}

                # Use if_generation_match for optimistic locking
                # If generation is None, use if_generation_match=0 to require blob doesn't exist
                if current_generation is not None:
                    index_blob.upload_from_string(
                        json.dumps(data, indent=2),
                        content_type="application/json",
                        if_generation_match=current_generation,
                    )
                else:
                    index_blob.upload_from_string(
                        json.dumps(data, indent=2),
                        content_type="application/json",
                        if_generation_match=0,  # Require blob doesn't exist
                    )

                logger.debug(f"Saved cache index with {len(self._cache_index)} entries")
                return

            except Exception as e:
                error_str = str(e).lower()
                if "precondition" in error_str or "generation" in error_str:
                    # Conflict - reload and retry
                    logger.warning(
                        f"Cache index conflict on attempt {attempt + 1}/{max_retries}, "
                        "reloading and retrying..."
                    )
                    self._load_index()
                    continue
                else:
                    logger.error(f"Failed to save cache index: {e}")
                    return

        logger.error(f"Failed to save cache index after {max_retries} attempts")

    @staticmethod
    def compute_config_hash(config: dict[str, Any]) -> str:
        """Compute hash of configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            SHA-256 hash string.
        """
        # Sort keys for deterministic hashing
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()

    @staticmethod
    def compute_code_hash(code_path: str | Path) -> str:
        """Compute hash of code file or directory.

        Args:
            code_path: Path to code file or directory.

        Returns:
            SHA-256 hash string.
        """
        code_path = Path(code_path)

        if code_path.is_file():
            content = code_path.read_bytes()
            return hashlib.sha256(content).hexdigest()

        elif code_path.is_dir():
            hasher = hashlib.sha256()
            for py_file in sorted(code_path.rglob("*.py")):
                hasher.update(py_file.read_bytes())
            return hasher.hexdigest()

        return hashlib.sha256(b"unknown").hexdigest()

    def compute_signature(
        self,
        component_type: ComponentType,
        model_version: int,
        config: dict[str, Any],
        input_paths: list[str] | None = None,
        code_path: str | Path | None = None,
    ) -> ComponentSignature:
        """Compute signature for a component.

        Args:
            component_type: Type of component.
            model_version: Model version to use.
            config: Configuration dictionary.
            input_paths: Optional list of input data paths.
            code_path: Optional path to component code.

        Returns:
            ComponentSignature for cache lookup.
        """
        config_hash = self.compute_config_hash(config)

        code_hash = None
        if self.config.include_code_hash and code_path:
            code_hash = self.compute_code_hash(code_path)

        input_data_hash = None
        if input_paths:
            paths_str = ",".join(sorted(input_paths))
            input_data_hash = hashlib.sha256(paths_str.encode()).hexdigest()

        return ComponentSignature(
            component_type=component_type,
            model_version=model_version,
            config_hash=config_hash,
            code_hash=code_hash,
            input_data_hash=input_data_hash,
        )

    def is_cached(self, signature: ComponentSignature) -> bool:
        """Check if component output is cached and valid.

        Args:
            signature: Component signature.

        Returns:
            True if valid cache entry exists.
        """
        if not self.config.should_cache(signature.component_type):
            return False

        self._load_index()

        cache_key = signature.to_cache_key()
        entry = self._cache_index.get(cache_key)

        if entry is None:
            return False

        if entry.is_expired():
            logger.debug(f"Cache entry expired: {cache_key}")
            return False

        # Verify output still exists in GCS
        try:
            bucket = self.client.bucket(self._bucket)
            blob = bucket.blob(entry.output_path)
            if not blob.exists():
                logger.debug(f"Cache output not found in GCS: {entry.output_path}")
                return False
        except Exception as e:
            logger.warning(f"Failed to verify cache output: {e}")
            return False

        logger.info(f"Cache hit: {cache_key}")
        return True

    def get_cached_output(self, signature: ComponentSignature) -> str | None:
        """Get cached output path if available.

        Args:
            signature: Component signature.

        Returns:
            GCS path to cached output, or None if not cached.
        """
        if not self.is_cached(signature):
            return None

        cache_key = signature.to_cache_key()
        entry = self._cache_index.get(cache_key)
        return entry.output_path if entry else None

    def store(
        self,
        signature: ComponentSignature,
        output_path: str,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        """Store component output in cache.

        Args:
            signature: Component signature.
            output_path: GCS path to output (must exist).
            metadata: Optional metadata to store.

        Returns:
            Created CacheEntry.

        Raises:
            GCSStorageError: If output path doesn't exist.
        """
        if not self.config.should_cache(signature.component_type):
            logger.debug(f"Caching disabled for {signature.component_type.value}")
            return CacheEntry(
                signature=signature,
                output_path=output_path,
                created_at=datetime.now(timezone.utc).isoformat(),
            )

        self._load_index()

        # Verify output exists
        try:
            bucket = self.client.bucket(self._bucket)
            blob = bucket.blob(output_path)
            if not blob.exists():
                raise GCSStorageError(
                    f"Output path does not exist: {output_path}",
                    bucket=self._bucket,
                    blob_path=output_path,
                    operation="cache_store",
                )
        except GCSStorageError:
            raise
        except Exception as e:
            logger.warning(f"Failed to verify output path: {e}")

        # Create cache entry
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(hours=self.config.ttl_hours) if self.config.ttl_hours > 0 else None

        entry = CacheEntry(
            signature=signature,
            output_path=output_path,
            created_at=now.isoformat(),
            valid_until=expiry.isoformat() if expiry else None,
            metadata=metadata or {},
        )

        cache_key = signature.to_cache_key()
        self._cache_index[cache_key] = entry

        # Enforce max entries
        if len(self._cache_index) > self.config.max_entries:
            self._evict_oldest()

        self._save_index()

        logger.info(f"Cached: {cache_key} -> {output_path}")
        return entry

    def _evict_oldest(self) -> int:
        """Evict oldest cache entries to enforce max_entries.

        Returns:
            Number of entries evicted.
        """
        if len(self._cache_index) <= self.config.max_entries:
            return 0

        # Sort by creation time
        sorted_entries = sorted(
            self._cache_index.items(),
            key=lambda x: x[1].created_at,
        )

        # Remove oldest entries
        num_to_remove = len(self._cache_index) - self.config.max_entries
        for i in range(num_to_remove):
            key = sorted_entries[i][0]
            del self._cache_index[key]
            logger.debug(f"Evicted cache entry: {key}")

        return num_to_remove

    def invalidate(self, signature: ComponentSignature) -> bool:
        """Invalidate a cache entry.

        Args:
            signature: Component signature.

        Returns:
            True if entry was invalidated.
        """
        self._load_index()

        cache_key = signature.to_cache_key()
        if cache_key in self._cache_index:
            del self._cache_index[cache_key]
            self._save_index()
            logger.info(f"Invalidated cache entry: {cache_key}")
            return True

        return False

    def invalidate_by_model_version(self, model_version: int) -> int:
        """Invalidate all entries for a model version.

        Args:
            model_version: Model version to invalidate.

        Returns:
            Number of entries invalidated.
        """
        self._load_index()

        keys_to_remove = [
            key
            for key, entry in self._cache_index.items()
            if entry.signature.model_version == model_version
        ]

        for key in keys_to_remove:
            del self._cache_index[key]

        if keys_to_remove:
            self._save_index()
            logger.info(
                f"Invalidated {len(keys_to_remove)} cache entries for model v{model_version}"
            )

        return len(keys_to_remove)

    def invalidate_by_component_type(self, component_type: ComponentType) -> int:
        """Invalidate all entries for a component type.

        Args:
            component_type: Component type to invalidate.

        Returns:
            Number of entries invalidated.
        """
        self._load_index()

        keys_to_remove = [
            key
            for key, entry in self._cache_index.items()
            if entry.signature.component_type == component_type
        ]

        for key in keys_to_remove:
            del self._cache_index[key]

        if keys_to_remove:
            self._save_index()
            logger.info(
                f"Invalidated {len(keys_to_remove)} cache entries for {component_type.value}"
            )

        return len(keys_to_remove)

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed.
        """
        self._load_index()

        keys_to_remove = [
            key for key, entry in self._cache_index.items() if entry.is_expired()
        ]

        for key in keys_to_remove:
            del self._cache_index[key]

        if keys_to_remove:
            self._save_index()
            logger.info(f"Removed {len(keys_to_remove)} expired cache entries")

        return len(keys_to_remove)

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        self._load_index()

        by_component: dict[str, int] = {}
        for entry in self._cache_index.values():
            ct = entry.signature.component_type.value
            by_component[ct] = by_component.get(ct, 0) + 1

        expired_count = sum(1 for e in self._cache_index.values() if e.is_expired())

        return {
            "enabled": self.config.enabled,
            "total_entries": len(self._cache_index),
            "max_entries": self.config.max_entries,
            "expired_entries": expired_count,
            "by_component_type": by_component,
            "ttl_hours": self.config.ttl_hours,
            "bucket": self._bucket,
            "gcs_prefix": self.config.gcs_prefix,
        }

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        self._load_index()
        count = len(self._cache_index)
        self._cache_index.clear()
        self._save_index()
        logger.info(f"Cleared {count} cache entries")
        return count


def get_pipeline_cache(bucket: str | None = None) -> PipelineCache:
    """Get a pipeline cache instance.

    Convenience function for getting a pipeline cache.

    Args:
        bucket: GCS bucket name (optional, defaults to env).

    Returns:
        PipelineCache instance.
    """
    return PipelineCache(PipelineCacheConfig.from_env(), bucket=bucket)
