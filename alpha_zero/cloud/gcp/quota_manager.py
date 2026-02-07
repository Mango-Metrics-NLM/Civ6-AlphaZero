"""GCP quota management and fallback strategies.

This module tracks GCP quota usage, predicts quota exhaustion,
and provides automatic fallback strategies to maintain training
continuity when quotas are exceeded.

Usage:
    from alpha_zero.cloud.gcp.quota_manager import (
        QuotaManagerConfig,
        QuotaManager,
    )

    config = QuotaManagerConfig()
    manager = QuotaManager(config)

    # Check quota before launching job
    if manager.can_launch(gpu_type="NVIDIA_TESLA_T4", count=1, region="us-central1"):
        # Launch job
        pass
    else:
        # Get fallback options
        fallback = manager.get_fallback_config(
            gpu_type="NVIDIA_TESLA_T4",
            region="us-central1"
        )
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class QuotaType(str, Enum):
    """Types of GCP quotas."""

    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"


class QuotaStatus(str, Enum):
    """Quota status levels."""

    AVAILABLE = "available"  # Under threshold, can launch
    WARNING = "warning"  # Approaching limit
    CRITICAL = "critical"  # Very close to limit
    EXHAUSTED = "exhausted"  # At or over limit


@dataclass
class QuotaInfo:
    """Information about a specific quota.

    Attributes:
        quota_type: Type of quota (GPU, CPU, etc.).
        resource_name: Specific resource (e.g., "NVIDIA_TESLA_T4").
        region: GCP region.
        limit: Maximum allowed quota.
        used: Currently used quota.
        last_updated: Timestamp of last update.
    """

    quota_type: QuotaType
    resource_name: str
    region: str
    limit: float
    used: float
    last_updated: float = field(default_factory=time.time)

    @property
    def available(self) -> float:
        """Get available quota."""
        return max(0, self.limit - self.used)

    @property
    def usage_ratio(self) -> float:
        """Get usage ratio (0-1)."""
        if self.limit <= 0:
            return 1.0
        return min(1.0, self.used / self.limit)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quota_type": self.quota_type.value,
            "resource_name": self.resource_name,
            "region": self.region,
            "limit": self.limit,
            "used": self.used,
            "available": self.available,
            "usage_ratio": self.usage_ratio,
            "last_updated": self.last_updated,
        }


@dataclass
class FallbackConfig:
    """Configuration for fallback when quota is exhausted.

    Attributes:
        gpu_type: Alternative GPU type.
        region: Alternative region.
        use_spot: Whether to use spot/preemptible.
        reduce_count: Reduce GPU count by this factor.
        reason: Reason for fallback.
    """

    gpu_type: str | None = None
    region: str | None = None
    use_spot: bool | None = None
    reduce_count: float = 1.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gpu_type": self.gpu_type,
            "region": self.region,
            "use_spot": self.use_spot,
            "reduce_count": self.reduce_count,
            "reason": self.reason,
        }


@dataclass
class QuotaManagerConfig:
    """Configuration for quota manager.

    Attributes:
        warning_threshold: Usage ratio triggering warning (0-1).
        critical_threshold: Usage ratio triggering critical status (0-1).
        cache_ttl_seconds: How long cached quota info is valid.
        enable_fallback: Whether to enable automatic fallback.
        preferred_regions: Ordered list of preferred regions.
        gpu_fallback_order: Ordered list of GPU fallback preferences.
        enable_spot_fallback: Allow falling back to spot VMs.
        max_fallback_attempts: Maximum fallback attempts.
    """

    warning_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_QUOTA_WARNING_THRESHOLD", "0.7"))
    )
    critical_threshold: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_QUOTA_CRITICAL_THRESHOLD", "0.9"))
    )
    cache_ttl_seconds: float = field(
        default_factory=lambda: float(os.environ.get("CIV6AZ_QUOTA_CACHE_TTL", "300.0"))
    )
    enable_fallback: bool = field(
        default_factory=lambda: os.environ.get("CIV6AZ_QUOTA_ENABLE_FALLBACK", "true").lower()
        == "true"
    )
    preferred_regions: list[str] = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_QUOTA_PREFERRED_REGIONS", "us-central1,us-west1,us-east1"
        ).split(",")
    )
    gpu_fallback_order: list[str] = field(
        default_factory=lambda: os.environ.get(
            "CIV6AZ_QUOTA_GPU_FALLBACK",
            "NVIDIA_TESLA_T4,NVIDIA_TESLA_V100,NVIDIA_TESLA_P100,NVIDIA_L4",
        ).split(",")
    )
    enable_spot_fallback: bool = field(
        default_factory=lambda: os.environ.get("CIV6AZ_QUOTA_SPOT_FALLBACK", "true").lower()
        == "true"
    )
    max_fallback_attempts: int = field(
        default_factory=lambda: int(os.environ.get("CIV6AZ_QUOTA_MAX_FALLBACK", "5"))
    )

    def __post_init__(self) -> None:
        """Validate configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if not 0 <= self.warning_threshold <= 1:
            raise ValueError("warning_threshold must be between 0 and 1")
        if not 0 <= self.critical_threshold <= 1:
            raise ValueError("critical_threshold must be between 0 and 1")
        if self.warning_threshold >= self.critical_threshold:
            raise ValueError("warning_threshold must be less than critical_threshold")
        if self.cache_ttl_seconds <= 0:
            raise ValueError("cache_ttl_seconds must be positive")
        if self.max_fallback_attempts < 1:
            raise ValueError("max_fallback_attempts must be >= 1")

    @classmethod
    def from_env(cls) -> QuotaManagerConfig:
        """Create configuration from environment variables.

        Returns:
            QuotaManagerConfig instance.
        """
        return cls()

    @classmethod
    def from_yaml(cls, path: Path) -> QuotaManagerConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            QuotaManagerConfig instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        if "quota_manager" in data:
            data = data["quota_manager"]

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuotaManagerConfig:
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            QuotaManagerConfig instance.
        """
        kwargs: dict[str, Any] = {}

        if "warning_threshold" in data:
            kwargs["warning_threshold"] = float(data["warning_threshold"])
        if "critical_threshold" in data:
            kwargs["critical_threshold"] = float(data["critical_threshold"])
        if "cache_ttl_seconds" in data:
            kwargs["cache_ttl_seconds"] = float(data["cache_ttl_seconds"])
        if "enable_fallback" in data:
            kwargs["enable_fallback"] = bool(data["enable_fallback"])
        if "preferred_regions" in data:
            kwargs["preferred_regions"] = list(data["preferred_regions"])
        if "gpu_fallback_order" in data:
            kwargs["gpu_fallback_order"] = list(data["gpu_fallback_order"])
        if "enable_spot_fallback" in data:
            kwargs["enable_spot_fallback"] = bool(data["enable_spot_fallback"])
        if "max_fallback_attempts" in data:
            kwargs["max_fallback_attempts"] = int(data["max_fallback_attempts"])

        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "warning_threshold": self.warning_threshold,
            "critical_threshold": self.critical_threshold,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "enable_fallback": self.enable_fallback,
            "preferred_regions": self.preferred_regions,
            "gpu_fallback_order": self.gpu_fallback_order,
            "enable_spot_fallback": self.enable_spot_fallback,
            "max_fallback_attempts": self.max_fallback_attempts,
        }


class QuotaManager:
    """Manages GCP quota tracking and fallback strategies.

    This class tracks quota usage across regions and GPU types,
    predicts quota exhaustion, and provides fallback strategies
    when quotas are exceeded.

    Attributes:
        config: Quota manager configuration.
    """

    def __init__(
        self,
        config: QuotaManagerConfig | None = None,
        quota_client: Any | None = None,
    ):
        """Initialize quota manager.

        Args:
            config: Quota manager configuration.
            quota_client: Optional GCP quota client (for testing).
        """
        self.config = config or QuotaManagerConfig()
        self._quota_client = quota_client
        self._lock = threading.RLock()

        # Cached quota info: (quota_type, resource_name, region) -> QuotaInfo
        self._cache: dict[tuple[str, str, str], QuotaInfo] = {}

        # Track recent usage for prediction
        self._usage_history: dict[tuple[str, str, str], list[tuple[float, float]]] = {}

        # Callbacks
        self._on_quota_warning: list[Callable[[QuotaInfo], None]] = []
        self._on_quota_exhausted: list[Callable[[QuotaInfo], None]] = []

        # Statistics
        self._fallback_count = 0
        self._quota_checks = 0
        self._cache_hits = 0

        logger.info(
            f"QuotaManager initialized: "
            f"warning={self.config.warning_threshold:.0%}, "
            f"critical={self.config.critical_threshold:.0%}"
        )

    def get_quota_status(
        self,
        quota_type: QuotaType,
        resource_name: str,
        region: str,
    ) -> QuotaStatus:
        """Get current quota status.

        Args:
            quota_type: Type of quota.
            resource_name: Specific resource name.
            region: GCP region.

        Returns:
            QuotaStatus indicating current status.
        """
        quota_info = self.get_quota_info(quota_type, resource_name, region)

        if quota_info.available <= 0:
            return QuotaStatus.EXHAUSTED
        if quota_info.usage_ratio >= self.config.critical_threshold:
            return QuotaStatus.CRITICAL
        if quota_info.usage_ratio >= self.config.warning_threshold:
            return QuotaStatus.WARNING
        return QuotaStatus.AVAILABLE

    def get_quota_info(
        self,
        quota_type: QuotaType,
        resource_name: str,
        region: str,
    ) -> QuotaInfo:
        """Get quota information.

        Args:
            quota_type: Type of quota.
            resource_name: Specific resource name.
            region: GCP region.

        Returns:
            QuotaInfo with current quota details.
        """
        self._quota_checks += 1
        cache_key = (quota_type.value, resource_name, region)

        with self._lock:
            # Check cache
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                age = time.time() - cached.last_updated
                if age < self.config.cache_ttl_seconds:
                    self._cache_hits += 1
                    return cached

            # Fetch from API or use defaults
            quota_info = self._fetch_quota_info(quota_type, resource_name, region)
            self._cache[cache_key] = quota_info

            # Record history
            if cache_key not in self._usage_history:
                self._usage_history[cache_key] = []
            history = self._usage_history[cache_key]
            history.append((time.time(), quota_info.usage_ratio))
            # Keep last 100 entries
            if len(history) > 100:
                self._usage_history[cache_key] = history[-100:]

            return quota_info

    def _fetch_quota_info(
        self,
        quota_type: QuotaType,
        resource_name: str,
        region: str,
    ) -> QuotaInfo:
        """Fetch quota info from GCP API.

        Args:
            quota_type: Type of quota.
            resource_name: Specific resource name.
            region: GCP region.

        Returns:
            QuotaInfo with quota details.
        """
        # Default quotas (used when API not available)
        default_gpu_quotas = {
            "NVIDIA_TESLA_T4": 8,
            "NVIDIA_TESLA_V100": 4,
            "NVIDIA_TESLA_P100": 4,
            "NVIDIA_TESLA_A100": 2,
            "NVIDIA_L4": 8,
        }

        if self._quota_client is not None:
            try:
                # Use injected client for actual API calls
                result = self._quota_client.get_quota(
                    quota_type=quota_type.value,
                    resource_name=resource_name,
                    region=region,
                )
                return QuotaInfo(
                    quota_type=quota_type,
                    resource_name=resource_name,
                    region=region,
                    limit=result.get("limit", 8),
                    used=result.get("used", 0),
                )
            except Exception as e:
                logger.warning(f"Failed to fetch quota from API: {e}")

        # Use defaults
        limit = default_gpu_quotas.get(resource_name, 8) if quota_type == QuotaType.GPU else 100

        return QuotaInfo(
            quota_type=quota_type,
            resource_name=resource_name,
            region=region,
            limit=limit,
            used=0,  # Conservative: assume no usage
        )

    def can_launch(
        self,
        gpu_type: str,
        count: int,
        region: str,
    ) -> bool:
        """Check if resources can be launched within quota.

        Args:
            gpu_type: GPU type to launch.
            count: Number of GPUs.
            region: Target region.

        Returns:
            True if launch is possible within quota.
        """
        quota_info = self.get_quota_info(QuotaType.GPU, gpu_type, region)
        can_launch = quota_info.available >= count

        if not can_launch:
            logger.warning(
                f"Cannot launch {count}x {gpu_type} in {region}: "
                f"available={quota_info.available}, limit={quota_info.limit}"
            )

        return can_launch

    def reserve_quota(
        self,
        quota_type: QuotaType,
        resource_name: str,
        region: str,
        amount: float,
    ) -> bool:
        """Reserve quota (update usage tracking).

        Args:
            quota_type: Type of quota.
            resource_name: Specific resource name.
            region: GCP region.
            amount: Amount to reserve.

        Returns:
            True if reservation successful.
        """
        cache_key = (quota_type.value, resource_name, region)

        with self._lock:
            quota_info = self.get_quota_info(quota_type, resource_name, region)

            if quota_info.available < amount:
                return False

            # Update cache
            quota_info.used += amount
            quota_info.last_updated = time.time()
            self._cache[cache_key] = quota_info

            # Check thresholds and notify
            if quota_info.usage_ratio >= self.config.critical_threshold:
                for callback in self._on_quota_exhausted:
                    try:
                        callback(quota_info)
                    except Exception as e:
                        logger.error(f"Error in quota exhausted callback: {e}")
            elif quota_info.usage_ratio >= self.config.warning_threshold:
                for callback in self._on_quota_warning:
                    try:
                        callback(quota_info)
                    except Exception as e:
                        logger.error(f"Error in quota warning callback: {e}")

            logger.debug(
                f"Reserved {amount} {resource_name} in {region}, "
                f"usage now {quota_info.usage_ratio:.1%}"
            )

            return True

    def release_quota(
        self,
        quota_type: QuotaType,
        resource_name: str,
        region: str,
        amount: float,
    ) -> None:
        """Release quota (update usage tracking).

        Args:
            quota_type: Type of quota.
            resource_name: Specific resource name.
            region: GCP region.
            amount: Amount to release.
        """
        cache_key = (quota_type.value, resource_name, region)

        with self._lock:
            if cache_key in self._cache:
                quota_info = self._cache[cache_key]
                quota_info.used = max(0, quota_info.used - amount)
                quota_info.last_updated = time.time()

                logger.debug(
                    f"Released {amount} {resource_name} in {region}, "
                    f"usage now {quota_info.usage_ratio:.1%}"
                )

    def get_fallback_config(
        self,
        gpu_type: str,
        region: str,
        count: int = 1,
    ) -> FallbackConfig | None:
        """Get fallback configuration when quota exhausted.

        Args:
            gpu_type: Original GPU type.
            region: Original region.
            count: Number of GPUs needed.

        Returns:
            FallbackConfig or None if no fallback available.
        """
        if not self.config.enable_fallback:
            return None

        self._fallback_count += 1
        attempts = 0

        # Try alternative GPU types in same region
        for alt_gpu in self.config.gpu_fallback_order:
            if alt_gpu == gpu_type:
                continue
            if attempts >= self.config.max_fallback_attempts:
                break
            attempts += 1

            if self.can_launch(alt_gpu, count, region):
                logger.info(f"Fallback: {gpu_type} -> {alt_gpu} in {region}")
                return FallbackConfig(
                    gpu_type=alt_gpu,
                    region=region,
                    reason=f"Quota exhausted for {gpu_type}, using {alt_gpu}",
                )

        # Try same GPU in alternative regions
        for alt_region in self.config.preferred_regions:
            if alt_region == region:
                continue
            if attempts >= self.config.max_fallback_attempts:
                break
            attempts += 1

            if self.can_launch(gpu_type, count, alt_region):
                logger.info(f"Fallback: {region} -> {alt_region} for {gpu_type}")
                return FallbackConfig(
                    gpu_type=gpu_type,
                    region=alt_region,
                    reason=f"Quota exhausted in {region}, using {alt_region}",
                )

        # Try spot VMs if enabled
        if self.config.enable_spot_fallback:
            logger.info(f"Fallback: enabling spot VMs for {gpu_type}")
            return FallbackConfig(
                gpu_type=gpu_type,
                region=region,
                use_spot=True,
                reason="Quota limited, enabling spot VMs",
            )

        # Try reducing count
        if count > 1:
            reduced = count // 2
            if reduced >= 1 and self.can_launch(gpu_type, reduced, region):
                logger.info(f"Fallback: reducing GPU count from {count} to {reduced}")
                return FallbackConfig(
                    gpu_type=gpu_type,
                    region=region,
                    reduce_count=reduced / count,
                    reason="Reducing GPU count to fit quota",
                )

        logger.warning(f"No fallback available for {count}x {gpu_type} in {region}")
        return None

    def predict_exhaustion_time(
        self,
        quota_type: QuotaType,
        resource_name: str,
        region: str,
    ) -> float | None:
        """Predict time until quota exhaustion based on usage trends.

        Args:
            quota_type: Type of quota.
            resource_name: Specific resource name.
            region: GCP region.

        Returns:
            Estimated seconds until exhaustion, or None if not predictable.
        """
        cache_key = (quota_type.value, resource_name, region)

        with self._lock:
            history = self._usage_history.get(cache_key, [])

        if len(history) < 2:
            return None

        # Calculate usage rate from recent history
        recent = history[-10:]  # Last 10 data points
        if len(recent) < 2:
            return None

        time_delta = recent[-1][0] - recent[0][0]
        if time_delta <= 0:
            return None

        usage_delta = recent[-1][1] - recent[0][1]
        if usage_delta <= 0:
            return None  # Usage decreasing or stable

        # Calculate rate (usage ratio per second)
        rate = usage_delta / time_delta

        # Calculate remaining capacity
        current_usage = recent[-1][1]
        remaining = 1.0 - current_usage

        if rate <= 0:
            return None

        return remaining / rate

    def register_warning_callback(self, callback: Callable[[QuotaInfo], None]) -> None:
        """Register callback for quota warnings.

        Args:
            callback: Function called when quota reaches warning threshold.
        """
        self._on_quota_warning.append(callback)

    def register_exhausted_callback(self, callback: Callable[[QuotaInfo], None]) -> None:
        """Register callback for quota exhaustion.

        Args:
            callback: Function called when quota is exhausted.
        """
        self._on_quota_exhausted.append(callback)

    def invalidate_cache(
        self,
        quota_type: QuotaType | None = None,
        resource_name: str | None = None,
        region: str | None = None,
    ) -> None:
        """Invalidate cached quota information.

        Args:
            quota_type: Optional type filter.
            resource_name: Optional resource filter.
            region: Optional region filter.
        """
        with self._lock:
            if quota_type is None and resource_name is None and region is None:
                self._cache.clear()
                logger.debug("Cleared all quota cache")
            else:
                to_remove = []
                for key in self._cache:
                    qt, rn, r = key
                    if quota_type is not None and qt != quota_type.value:
                        continue
                    if resource_name is not None and rn != resource_name:
                        continue
                    if region is not None and r != region:
                        continue
                    to_remove.append(key)

                for key in to_remove:
                    del self._cache[key]
                logger.debug(f"Invalidated {len(to_remove)} cache entries")

    def get_all_quotas(self) -> list[QuotaInfo]:
        """Get all cached quota information.

        Returns:
            List of QuotaInfo for all cached quotas.
        """
        with self._lock:
            return list(self._cache.values())

    def get_statistics(self) -> dict[str, Any]:
        """Get quota manager statistics.

        Returns:
            Dictionary with statistics.
        """
        with self._lock:
            return {
                "quota_checks": self._quota_checks,
                "cache_hits": self._cache_hits,
                "cache_hit_ratio": (
                    self._cache_hits / self._quota_checks if self._quota_checks > 0 else 0.0
                ),
                "fallback_count": self._fallback_count,
                "cached_quotas": len(self._cache),
            }

    def reset(self) -> None:
        """Reset quota manager state."""
        with self._lock:
            self._cache.clear()
            self._usage_history.clear()
            self._fallback_count = 0
            self._quota_checks = 0
            self._cache_hits = 0
        logger.info("QuotaManager state reset")

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"QuotaManager("
            f"warning={self.config.warning_threshold:.0%}, "
            f"critical={self.config.critical_threshold:.0%}, "
            f"cached={len(self._cache)})"
        )
