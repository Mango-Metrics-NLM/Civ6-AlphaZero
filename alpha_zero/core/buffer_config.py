"""Centralized buffer and storage configuration.

This module provides configuration classes for replay buffers and shared storage.
It consolidates buffer-related settings that were previously scattered or hardcoded.

Example:
    >>> from alpha_zero.core.buffer_config import ReplayBufferConfig
    >>>
    >>> # Use defaults
    >>> config = ReplayBufferConfig()
    >>>
    >>> # Use small buffer for testing
    >>> config = ReplayBufferConfig.small()
    >>>
    >>> # Override via environment
    >>> import os
    >>> os.environ["CIV6AZ_BUFFER_MAX_SIZE"] = "1000000"
    >>> config = ReplayBufferConfig().apply_env_overrides()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from alpha_zero.core.config import ConfigMixin


@dataclass
class ReplayBufferConfig(ConfigMixin):
    """Configuration for distributed replay buffer.

    This config consolidates buffer-related settings from shared_storage.py
    and adds additional parameters that were previously hardcoded.

    Environment variables use prefix: CIV6AZ_BUFFER_
    Example: CIV6AZ_BUFFER_MAX_SIZE=1000000

    Attributes:
        max_size: Maximum number of examples in buffer.
        min_size: Minimum examples before training can start.
        use_prioritized: Whether to use prioritized experience replay.
        priority_alpha: Priority exponent for prioritized replay.
        priority_beta: Importance sampling exponent for prioritized replay.
        compress: Whether to compress stored examples.
        trajectory_queue_size: Size of trajectory submission queue.
        sample_batch_size: Default batch size for sampling.
        max_pending_trajectories: Maximum trajectories waiting to be processed.
        max_loss_history: Maximum length of loss history to keep.
        max_eval_history: Maximum length of evaluation history to keep.
    """

    ENV_PREFIX: ClassVar[str] = "CIV6AZ_BUFFER_"

    # Buffer capacity
    max_size: int = 500_000
    min_size: int = 10_000

    # Prioritized replay
    use_prioritized: bool = False
    priority_alpha: float = 0.6
    priority_beta: float = 0.4

    # Compression
    compress: bool = False

    # Queue settings
    trajectory_queue_size: int = 1000
    sample_batch_size: int = 256
    max_pending_trajectories: int = 100

    # History limits (previously hardcoded in shared_storage.py)
    max_loss_history: int = 1000
    max_eval_history: int = 100

    @classmethod
    def small(cls) -> ReplayBufferConfig:
        """Return small buffer config for testing.

        Returns:
            ReplayBufferConfig with reduced sizes for fast testing.
        """
        return cls(
            max_size=10_000,
            min_size=1_000,
            trajectory_queue_size=100,
            max_pending_trajectories=10,
            max_loss_history=100,
            max_eval_history=10,
        )

    @classmethod
    def large(cls) -> ReplayBufferConfig:
        """Return large buffer config for production.

        Returns:
            ReplayBufferConfig with increased sizes for production.
        """
        return cls(
            max_size=1_000_000,
            min_size=50_000,
            trajectory_queue_size=5000,
            max_pending_trajectories=500,
            max_loss_history=5000,
            max_eval_history=500,
        )

    @classmethod
    def prioritized(cls) -> ReplayBufferConfig:
        """Return config with prioritized experience replay enabled.

        Returns:
            ReplayBufferConfig with PER enabled.
        """
        return cls(
            use_prioritized=True,
            priority_alpha=0.6,
            priority_beta=0.4,
        )


@dataclass
class SharedStorageConfig(ConfigMixin):
    """Configuration for shared storage.

    This config controls how model weights and training statistics
    are shared between processes.

    Environment variables use prefix: CIV6AZ_STORAGE_
    Example: CIV6AZ_STORAGE_MAX_MODEL_HISTORY=20

    Attributes:
        mode: Storage mode ('multiprocessing' for local, 'distributed' for remote).
        max_model_history: Number of model checkpoints to keep in memory.
        compress_weights: Whether to compress model weights.
        compress_trajectories: Whether to compress trajectories.
        track_statistics: Whether to track training statistics.
    """

    ENV_PREFIX: ClassVar[str] = "CIV6AZ_STORAGE_"

    # Storage mode
    mode: str = "multiprocessing"

    # Model update settings
    max_model_history: int = 10

    # Compression settings
    compress_weights: bool = True
    compress_trajectories: bool = False

    # Statistics tracking
    track_statistics: bool = True

    @classmethod
    def minimal(cls) -> SharedStorageConfig:
        """Return minimal storage config for testing.

        Returns:
            SharedStorageConfig with minimal overhead.
        """
        return cls(
            max_model_history=2,
            compress_weights=False,
            compress_trajectories=False,
            track_statistics=False,
        )

    @classmethod
    def production(cls) -> SharedStorageConfig:
        """Return production storage config.

        Returns:
            SharedStorageConfig optimized for production use.
        """
        return cls(
            max_model_history=20,
            compress_weights=True,
            compress_trajectories=True,
            track_statistics=True,
        )
