"""
GCS Data Lifecycle Management for Civ6 AlphaZero Cloud Training.

Provides automated retention policies, archival to cold storage, and cleanup
of old training artifacts while preserving all model checkpoints indefinitely.

Features:
- Iteration-based cleanup (keep last N iterations)
- Native GCS lifecycle rule generation and application
- Storage class transitions (STANDARD -> NEARLINE -> COLDLINE)
- Dry-run mode for safe testing

All values configurable via environment variables with CIV6AZ_LIFECYCLE_ prefix.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from alpha_zero.cloud.gcp.exceptions import ConfigurationError, GCSStorageError

if TYPE_CHECKING:
    from google.cloud import storage as gcs_lib

logger = logging.getLogger(__name__)


class StorageClass(str, Enum):
    """GCS storage classes ordered by cost (high to low)."""

    STANDARD = "STANDARD"
    NEARLINE = "NEARLINE"
    COLDLINE = "COLDLINE"
    ARCHIVE = "ARCHIVE"


@dataclass
class RetentionPolicyConfig:
    """Configuration for a single retention policy rule.

    Attributes:
        prefix: GCS path prefix this rule applies to (e.g., "trajectories/").
        keep_recent_iterations: Number of recent iterations to keep in STANDARD.
            Set to 0 to keep all iterations.
        archive_storage_class: Storage class to transition older data to.
        archive_after_days: Days after creation to transition to archive class.
            Set to 0 to skip time-based archival.
        delete_after_days: Days after creation to delete.
            Set to 0 to never delete.
        exclude_patterns: List of patterns to exclude from this rule.
    """

    prefix: str
    keep_recent_iterations: int = 20
    archive_storage_class: StorageClass = StorageClass.NEARLINE
    archive_after_days: int = 30
    delete_after_days: int = 0
    exclude_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prefix": self.prefix,
            "keep_recent_iterations": self.keep_recent_iterations,
            "archive_storage_class": self.archive_storage_class.value,
            "archive_after_days": self.archive_after_days,
            "delete_after_days": self.delete_after_days,
            "exclude_patterns": self.exclude_patterns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetentionPolicyConfig:
        """Create from dictionary."""
        storage_class = data.get("archive_storage_class", "NEARLINE")
        if isinstance(storage_class, str):
            storage_class = StorageClass(storage_class)

        return cls(
            prefix=data["prefix"],
            keep_recent_iterations=data.get("keep_recent_iterations", 20),
            archive_storage_class=storage_class,
            archive_after_days=data.get("archive_after_days", 30),
            delete_after_days=data.get("delete_after_days", 0),
            exclude_patterns=data.get("exclude_patterns", []),
        )


@dataclass
class LifecycleRule:
    """A single GCS lifecycle rule for bucket configuration.

    Maps to google.cloud.storage.bucket.LifecycleRuleConditions/Action.
    """

    action_type: str  # "SetStorageClass" or "Delete"
    storage_class: StorageClass | None = None
    age_days: int | None = None
    matches_prefix: list[str] | None = None
    matches_suffix: list[str] | None = None

    def to_gcs_rule(self) -> dict[str, Any]:
        """Convert to GCS lifecycle rule format."""
        condition: dict[str, Any] = {}
        if self.age_days is not None:
            condition["age"] = self.age_days
        if self.matches_prefix:
            condition["matchesPrefix"] = self.matches_prefix
        if self.matches_suffix:
            condition["matchesSuffix"] = self.matches_suffix

        action: dict[str, Any] = {"type": self.action_type}
        if self.storage_class and self.action_type == "SetStorageClass":
            action["storageClass"] = self.storage_class.value

        return {"action": action, "condition": condition}


@dataclass
class CleanupResult:
    """Result of a cleanup operation.

    Attributes:
        archived_count: Number of objects archived (storage class changed).
        deleted_count: Number of objects deleted.
        bytes_archived: Total bytes archived.
        bytes_deleted: Total bytes deleted.
        errors: List of error messages.
        dry_run: Whether this was a dry run (no changes made).
    """

    archived_count: int = 0
    deleted_count: int = 0
    bytes_archived: int = 0
    bytes_deleted: int = 0
    errors: list[str] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "archived_count": self.archived_count,
            "deleted_count": self.deleted_count,
            "bytes_archived": self.bytes_archived,
            "bytes_deleted": self.bytes_deleted,
            "errors": self.errors,
            "dry_run": self.dry_run,
        }

    def merge(self, other: CleanupResult) -> CleanupResult:
        """Merge another result into this one."""
        return CleanupResult(
            archived_count=self.archived_count + other.archived_count,
            deleted_count=self.deleted_count + other.deleted_count,
            bytes_archived=self.bytes_archived + other.bytes_archived,
            bytes_deleted=self.bytes_deleted + other.bytes_deleted,
            errors=self.errors + other.errors,
            dry_run=self.dry_run or other.dry_run,
        )


@dataclass
class LifecycleConfig:
    """Configuration for GCS data lifecycle management.

    Environment Variables:
        CIV6AZ_LIFECYCLE_ENABLED: Enable lifecycle management (default: true)
        CIV6AZ_LIFECYCLE_DRY_RUN: Enable dry-run mode (default: false)
        CIV6AZ_TRAJECTORY_KEEP_ITERATIONS: Iterations to keep (default: 20)
        CIV6AZ_APPLY_GCS_LIFECYCLE: Apply native GCS rules (default: true)
        CIV6AZ_LIFECYCLE_CLEANUP_INTERVAL: Iterations between cleanups (default: 10)
    """

    enabled: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_LIFECYCLE_ENABLED", "true"
        ).lower() == "true"
    )
    dry_run: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_LIFECYCLE_DRY_RUN", "false"
        ).lower() == "true"
    )

    # Trajectory retention: Keep last N iterations in STANDARD, archive older
    trajectories_retention: RetentionPolicyConfig = field(
        default_factory=lambda: RetentionPolicyConfig(
            prefix="trajectories/",
            keep_recent_iterations=int(
                os.environ.get("CIV6AZ_TRAJECTORY_KEEP_ITERATIONS", "20")
            ),
            archive_storage_class=StorageClass.NEARLINE,
            archive_after_days=30,
            delete_after_days=0,  # Never delete trajectories
        )
    )

    # Model retention: Keep ALL models indefinitely in STANDARD
    models_retention: RetentionPolicyConfig = field(
        default_factory=lambda: RetentionPolicyConfig(
            prefix="models/",
            keep_recent_iterations=0,  # Keep all models
            archive_storage_class=StorageClass.STANDARD,
            archive_after_days=0,  # Never archive models
            delete_after_days=0,  # Never delete models
        )
    )

    # Checkpoint retention: Keep ALL checkpoints, archive after 90 days
    checkpoints_retention: RetentionPolicyConfig = field(
        default_factory=lambda: RetentionPolicyConfig(
            prefix="checkpoints/",
            keep_recent_iterations=0,  # Keep all checkpoints
            archive_storage_class=StorageClass.NEARLINE,
            archive_after_days=90,
            delete_after_days=0,  # Never delete checkpoints
        )
    )

    # Logs retention: Keep 50 iterations, archive, delete after 1 year
    logs_retention: RetentionPolicyConfig = field(
        default_factory=lambda: RetentionPolicyConfig(
            prefix="logs/",
            keep_recent_iterations=50,
            archive_storage_class=StorageClass.COLDLINE,
            archive_after_days=30,
            delete_after_days=365,  # Delete logs after 1 year
        )
    )

    # Evaluation retention: Keep 30 iterations, archive after 60 days
    eval_retention: RetentionPolicyConfig = field(
        default_factory=lambda: RetentionPolicyConfig(
            prefix="eval/",
            keep_recent_iterations=30,
            archive_storage_class=StorageClass.NEARLINE,
            archive_after_days=60,
            delete_after_days=0,
        )
    )

    apply_gcs_lifecycle_rules: bool = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_APPLY_GCS_LIFECYCLE", "true"
        ).lower() == "true"
    )

    iteration_cleanup_interval: int = field(
        default_factory=lambda: int(
            os.environ.get("CIV6AZ_LIFECYCLE_CLEANUP_INTERVAL", "10")
        )
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if self.iteration_cleanup_interval < 1:
            raise ConfigurationError(
                "iteration_cleanup_interval must be at least 1",
                details={"value": self.iteration_cleanup_interval},
            )

    @classmethod
    def from_env(cls) -> LifecycleConfig:
        """Create configuration from environment variables."""
        return cls()

    @classmethod
    def from_yaml(cls, path: str | Path) -> LifecycleConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            LifecycleConfig instance.
        """
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"Lifecycle configuration file not found: {path}")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {path}: {e}") from e

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LifecycleConfig:
        """Create configuration from dictionary."""
        config = cls(
            enabled=data.get("enabled", True),
            dry_run=data.get("dry_run", False),
            apply_gcs_lifecycle_rules=data.get("apply_gcs_lifecycle_rules", True),
            iteration_cleanup_interval=data.get("iteration_cleanup_interval", 10),
        )

        if "trajectories_retention" in data:
            config.trajectories_retention = RetentionPolicyConfig.from_dict(
                data["trajectories_retention"]
            )
        if "models_retention" in data:
            config.models_retention = RetentionPolicyConfig.from_dict(
                data["models_retention"]
            )
        if "checkpoints_retention" in data:
            config.checkpoints_retention = RetentionPolicyConfig.from_dict(
                data["checkpoints_retention"]
            )
        if "logs_retention" in data:
            config.logs_retention = RetentionPolicyConfig.from_dict(
                data["logs_retention"]
            )
        if "eval_retention" in data:
            config.eval_retention = RetentionPolicyConfig.from_dict(
                data["eval_retention"]
            )

        return config

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "trajectories_retention": self.trajectories_retention.to_dict(),
            "models_retention": self.models_retention.to_dict(),
            "checkpoints_retention": self.checkpoints_retention.to_dict(),
            "logs_retention": self.logs_retention.to_dict(),
            "eval_retention": self.eval_retention.to_dict(),
            "apply_gcs_lifecycle_rules": self.apply_gcs_lifecycle_rules,
            "iteration_cleanup_interval": self.iteration_cleanup_interval,
        }

    def get_all_policies(self) -> list[RetentionPolicyConfig]:
        """Get all retention policies."""
        return [
            self.trajectories_retention,
            self.models_retention,
            self.checkpoints_retention,
            self.logs_retention,
            self.eval_retention,
        ]


class GCSLifecycleManager:
    """Manages GCS data lifecycle for training artifacts.

    Provides:
    - Iteration-based cleanup (keep last N iterations)
    - Native GCS lifecycle rule generation and application
    - Storage class transitions (STANDARD -> NEARLINE -> COLDLINE)
    - Dry-run mode for safe testing

    Thread-safe for concurrent operations.

    Example:
        from alpha_zero.cloud.gcp import GCPConfig
        from alpha_zero.cloud.gcp.lifecycle import GCSLifecycleManager, LifecycleConfig

        gcp_config = GCPConfig.from_env()
        lifecycle_config = LifecycleConfig.from_env()

        manager = GCSLifecycleManager(gcp_config.bucket, lifecycle_config)

        # Apply native GCS lifecycle rules
        manager.apply_bucket_lifecycle_rules()

        # Run iteration-based cleanup
        result = manager.cleanup_old_iterations(current_iteration=100)
        print(f"Archived {result.archived_count} objects")
    """

    # Pattern to extract iteration numbers from paths
    ITERATION_PATTERN = re.compile(r"iter(?:ation)?[_-]?(\d+)")

    def __init__(
        self,
        bucket: str,
        config: LifecycleConfig | None = None,
        project_id: str | None = None,
    ) -> None:
        """Initialize lifecycle manager.

        Args:
            bucket: GCS bucket name (without gs:// prefix).
            config: Lifecycle configuration (defaults to from env).
            project_id: GCP project ID (defaults to env).
        """
        self._bucket = bucket.removeprefix("gs://")
        self.config = config or LifecycleConfig.from_env()
        self._project_id = project_id or os.environ.get("CIV6AZ_VERTEX_PROJECT")
        self._client: gcs_lib.Client | None = None

        logger.info(
            f"GCSLifecycleManager initialized: bucket={self._bucket}, "
            f"enabled={self.config.enabled}, dry_run={self.config.dry_run}"
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
                    "google-cloud-storage is required for lifecycle management. "
                    "Install with: pip install google-cloud-storage"
                ) from err
        return self._client

    # =========================================================================
    # Iteration-Based Cleanup
    # =========================================================================

    def cleanup_old_iterations(
        self,
        current_iteration: int,
        policy: RetentionPolicyConfig | None = None,
    ) -> CleanupResult:
        """Clean up old iterations based on retention policy.

        Archives or deletes data from iterations older than the retention
        threshold.

        Args:
            current_iteration: Current training iteration number.
            policy: Specific policy to apply (default: all policies).

        Returns:
            CleanupResult with cleanup statistics.
        """
        if not self.config.enabled:
            logger.info("Lifecycle management disabled, skipping cleanup")
            return CleanupResult(dry_run=True)

        result = CleanupResult(dry_run=self.config.dry_run)
        policies = [policy] if policy else self.config.get_all_policies()

        for pol in policies:
            if pol.keep_recent_iterations <= 0:
                logger.debug(f"Skipping {pol.prefix}: keep_recent_iterations=0")
                continue  # Skip policies that keep everything

            try:
                policy_result = self._cleanup_policy(
                    pol, current_iteration, self.config.dry_run
                )
                result = result.merge(policy_result)
            except Exception as e:
                error_msg = f"Policy {pol.prefix}: {e}"
                result.errors.append(error_msg)
                logger.error(f"Cleanup failed for {pol.prefix}: {e}")

        return result

    def _cleanup_policy(
        self,
        policy: RetentionPolicyConfig,
        current_iteration: int,
        dry_run: bool,
    ) -> CleanupResult:
        """Apply cleanup for a single retention policy."""
        result = CleanupResult(dry_run=dry_run)

        # Find all iterations for this prefix
        iterations = self._get_iterations_for_prefix(policy.prefix)

        if not iterations:
            logger.debug(f"No iterations found for {policy.prefix}")
            return result

        # Determine which iterations to archive
        keep_threshold = current_iteration - policy.keep_recent_iterations
        iterations_to_archive = [i for i in iterations if i <= keep_threshold]

        if not iterations_to_archive:
            logger.debug(
                f"No iterations to archive for {policy.prefix} "
                f"(current={current_iteration}, keep={policy.keep_recent_iterations})"
            )
            return result

        logger.info(
            f"Cleanup for {policy.prefix}: "
            f"current={current_iteration}, keep_last={policy.keep_recent_iterations}, "
            f"archiving {len(iterations_to_archive)} iterations: {iterations_to_archive}"
        )

        for iteration in iterations_to_archive:
            try:
                iter_result = self._archive_iteration(
                    policy.prefix,
                    iteration,
                    policy.archive_storage_class,
                    dry_run,
                )
                result = result.merge(iter_result)
            except Exception as e:
                result.errors.append(f"Iteration {iteration}: {e}")
                logger.error(f"Failed to archive iteration {iteration}: {e}")

        return result

    def _get_iterations_for_prefix(self, prefix: str) -> list[int]:
        """Get all iteration numbers for a given prefix."""
        bucket = self.client.bucket(self._bucket)
        blobs = bucket.list_blobs(prefix=prefix)

        iterations = set()
        for blob in blobs:
            match = self.ITERATION_PATTERN.search(blob.name)
            if match:
                iterations.add(int(match.group(1)))

        return sorted(iterations)

    def _archive_iteration(
        self,
        prefix: str,
        iteration: int,
        storage_class: StorageClass,
        dry_run: bool,
    ) -> CleanupResult:
        """Archive a single iteration to a different storage class."""
        result = CleanupResult(dry_run=dry_run)

        # Build iteration prefix patterns (handles both iter_001 and iteration_1)
        iter_patterns = [
            f"{prefix}iter_{iteration:03d}/",
            f"{prefix}iter_{iteration}/",
            f"{prefix}iteration_{iteration:03d}/",
            f"{prefix}iteration_{iteration}/",
        ]

        bucket = self.client.bucket(self._bucket)

        for iter_prefix in iter_patterns:
            blobs = bucket.list_blobs(prefix=iter_prefix)

            for blob in blobs:
                try:
                    # Skip if already in target storage class
                    if blob.storage_class == storage_class.value:
                        continue

                    if dry_run:
                        logger.debug(
                            f"[DRY RUN] Would archive {blob.name} "
                            f"from {blob.storage_class} to {storage_class.value}"
                        )
                        result.archived_count += 1
                        result.bytes_archived += blob.size or 0
                    else:
                        # Update storage class
                        blob.update_storage_class(storage_class.value)
                        result.archived_count += 1
                        result.bytes_archived += blob.size or 0
                        logger.debug(f"Archived {blob.name} to {storage_class.value}")

                except Exception as e:
                    result.errors.append(f"{blob.name}: {e}")
                    logger.error(f"Failed to archive {blob.name}: {e}")

        return result

    def should_run_cleanup(self, iteration: int) -> bool:
        """Check if cleanup should run at this iteration.

        Args:
            iteration: Current training iteration.

        Returns:
            True if cleanup should run.
        """
        if not self.config.enabled:
            return False
        return iteration % self.config.iteration_cleanup_interval == 0

    # =========================================================================
    # GCS Native Lifecycle Rules
    # =========================================================================

    def generate_lifecycle_rules(self) -> list[LifecycleRule]:
        """Generate GCS lifecycle rules from retention policies.

        Returns:
            List of LifecycleRule objects.
        """
        rules: list[LifecycleRule] = []

        for policy in self.config.get_all_policies():
            # Add archive rule if time-based archival is configured
            if policy.archive_after_days > 0:
                rules.append(
                    LifecycleRule(
                        action_type="SetStorageClass",
                        storage_class=policy.archive_storage_class,
                        age_days=policy.archive_after_days,
                        matches_prefix=[policy.prefix],
                    )
                )

            # Add delete rule if configured
            if policy.delete_after_days > 0:
                rules.append(
                    LifecycleRule(
                        action_type="Delete",
                        age_days=policy.delete_after_days,
                        matches_prefix=[policy.prefix],
                    )
                )

        return rules

    def apply_bucket_lifecycle_rules(self) -> bool:
        """Apply lifecycle rules to the GCS bucket.

        Returns:
            True if successful.

        Raises:
            GCSStorageError: If rule application fails.
        """
        if not self.config.apply_gcs_lifecycle_rules:
            logger.info("GCS lifecycle rule application is disabled")
            return True

        rules = self.generate_lifecycle_rules()

        if self.config.dry_run:
            logger.info("[DRY RUN] Would apply lifecycle rules:")
            for rule in rules:
                logger.info(f"  {rule.to_gcs_rule()}")
            return True

        try:
            bucket = self.client.bucket(self._bucket)
            bucket.reload()

            # Convert to GCS format
            gcs_rules = [rule.to_gcs_rule() for rule in rules]

            bucket.lifecycle_rules = gcs_rules
            bucket.patch()

            logger.info(f"Applied {len(rules)} lifecycle rules to {self._bucket}")
            return True

        except Exception as e:
            logger.error(f"Failed to apply lifecycle rules: {e}")
            raise GCSStorageError(
                f"Failed to apply lifecycle rules: {e}",
                bucket=self._bucket,
                operation="apply_lifecycle",
            ) from e

    def get_current_lifecycle_rules(self) -> list[dict[str, Any]]:
        """Get current lifecycle rules from bucket.

        Returns:
            List of current rules as dictionaries.
        """
        try:
            bucket = self.client.bucket(self._bucket)
            bucket.reload()
            return list(bucket.lifecycle_rules) if bucket.lifecycle_rules else []
        except Exception as e:
            logger.error(f"Failed to get lifecycle rules: {e}")
            return []

    # =========================================================================
    # Storage Analysis
    # =========================================================================

    def get_storage_summary(self) -> dict[str, Any]:
        """Get summary of storage usage by prefix and storage class.

        Returns:
            Dictionary with storage statistics.
        """
        summary: dict[str, dict[str, int | float]] = {}

        for policy in self.config.get_all_policies():
            bucket = self.client.bucket(self._bucket)
            blobs = bucket.list_blobs(prefix=policy.prefix)

            prefix_stats: dict[str, int | float] = {
                "STANDARD": 0,
                "NEARLINE": 0,
                "COLDLINE": 0,
                "ARCHIVE": 0,
                "total_count": 0,
                "total_bytes": 0,
            }

            for blob in blobs:
                storage_class = blob.storage_class or "STANDARD"
                size = blob.size or 0

                prefix_stats[storage_class] = prefix_stats.get(storage_class, 0) + size
                prefix_stats["total_count"] += 1
                prefix_stats["total_bytes"] += size

            summary[policy.prefix] = prefix_stats

        return summary

    def estimate_cost_savings(self, current_iteration: int) -> dict[str, float]:
        """Estimate potential cost savings from lifecycle policies.

        Args:
            current_iteration: Current training iteration.

        Returns:
            Dictionary with estimated savings.
        """
        # GCS pricing (approximate USD per GB per month)
        prices = {
            "STANDARD": 0.020,
            "NEARLINE": 0.010,
            "COLDLINE": 0.004,
            "ARCHIVE": 0.0012,
        }

        summary = self.get_storage_summary()
        current_cost = 0.0
        optimized_cost = 0.0

        for _prefix, stats in summary.items():
            for storage_class in ["STANDARD", "NEARLINE", "COLDLINE", "ARCHIVE"]:
                gb = stats.get(storage_class, 0) / (1024**3)
                current_cost += gb * prices[storage_class]

        # Estimate optimized cost after applying policies
        for policy in self.config.get_all_policies():
            prefix_stats = summary.get(policy.prefix, {})
            total_gb = prefix_stats.get("total_bytes", 0) / (1024**3)

            if policy.keep_recent_iterations > 0:
                # Estimate fraction that would be archived
                archived_fraction = 0.8  # Rough estimate
                standard_gb = total_gb * (1 - archived_fraction)
                archived_gb = total_gb * archived_fraction
            else:
                standard_gb = total_gb
                archived_gb = 0

            optimized_cost += standard_gb * prices["STANDARD"]
            optimized_cost += archived_gb * prices[policy.archive_storage_class.value]

        savings = current_cost - optimized_cost

        return {
            "current_monthly_cost_usd": round(current_cost, 2),
            "estimated_optimized_cost_usd": round(optimized_cost, 2),
            "estimated_monthly_savings_usd": round(savings, 2),
            "savings_percentage": round(
                (savings / current_cost * 100) if current_cost > 0 else 0, 1
            ),
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get lifecycle manager statistics.

        Returns:
            Dictionary with manager statistics.
        """
        return {
            "bucket": self._bucket,
            "enabled": self.config.enabled,
            "dry_run": self.config.dry_run,
            "cleanup_interval": self.config.iteration_cleanup_interval,
            "policies": {
                "trajectories": self.config.trajectories_retention.to_dict(),
                "models": self.config.models_retention.to_dict(),
                "checkpoints": self.config.checkpoints_retention.to_dict(),
                "logs": self.config.logs_retention.to_dict(),
                "eval": self.config.eval_retention.to_dict(),
            },
        }


def get_lifecycle_manager(bucket: str) -> GCSLifecycleManager:
    """Get a lifecycle manager instance.

    Convenience function for getting a lifecycle manager.

    Args:
        bucket: GCS bucket name.

    Returns:
        GCSLifecycleManager instance.
    """
    return GCSLifecycleManager(bucket, LifecycleConfig.from_env())
