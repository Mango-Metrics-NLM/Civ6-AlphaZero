"""Policy target pruning for efficient training.

Prunes low-probability actions from policy targets to reduce the effective
action space during training, improving both training speed and policy quality.

Usage:
    from alpha_zero.optimizations.policy_pruning import PolicyTargetPruner, PolicyPruningConfig

    config = PolicyPruningConfig(min_probability=0.01, max_actions=100)
    pruner = PolicyTargetPruner(config)

    # During training data generation
    pruned_policy = pruner.prune(raw_policy, action_mask)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PolicyPruningConfig:
    """Configuration for policy target pruning.

    Attributes:
        min_probability: Minimum probability threshold to keep an action
        max_actions: Maximum number of actions to keep
        preserve_top_k: Always keep top-k actions regardless of probability
        temperature_scaling: Apply temperature before pruning (higher = more uniform)
        use_entropy_threshold: Use entropy-based dynamic thresholding
        entropy_min_threshold: Minimum threshold when entropy is low
        entropy_max_threshold: Maximum threshold when entropy is high
    """

    min_probability: float = 0.001
    max_actions: int = 100
    preserve_top_k: int = 5
    temperature_scaling: float = 1.0
    use_entropy_threshold: bool = False
    entropy_min_threshold: float = 0.0005
    entropy_max_threshold: float = 0.01

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.min_probability < 0:
            raise ValueError(f"min_probability must be >= 0, got {self.min_probability}")
        if self.max_actions < 1:
            raise ValueError(f"max_actions must be >= 1, got {self.max_actions}")
        if self.preserve_top_k < 0:
            raise ValueError(f"preserve_top_k must be >= 0, got {self.preserve_top_k}")
        if self.temperature_scaling <= 0:
            raise ValueError(f"temperature_scaling must be > 0, got {self.temperature_scaling}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyPruningConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_probability": self.min_probability,
            "max_actions": self.max_actions,
            "preserve_top_k": self.preserve_top_k,
            "temperature_scaling": self.temperature_scaling,
            "use_entropy_threshold": self.use_entropy_threshold,
            "entropy_min_threshold": self.entropy_min_threshold,
            "entropy_max_threshold": self.entropy_max_threshold,
        }


@dataclass
class PruningStats:
    """Statistics about policy pruning operations."""

    total_pruned: int = 0
    total_policies: int = 0
    sum_original_actions: int = 0
    sum_pruned_actions: int = 0
    sum_kl_divergence: float = 0.0

    @property
    def avg_original_actions(self) -> float:
        """Average actions before pruning."""
        if self.total_policies == 0:
            return 0.0
        return self.sum_original_actions / self.total_policies

    @property
    def avg_pruned_actions(self) -> float:
        """Average actions after pruning."""
        if self.total_policies == 0:
            return 0.0
        return self.sum_pruned_actions / self.total_policies

    @property
    def avg_kl_divergence(self) -> float:
        """Average KL divergence from pruning."""
        if self.total_policies == 0:
            return 0.0
        return self.sum_kl_divergence / self.total_policies

    @property
    def reduction_ratio(self) -> float:
        """Average reduction ratio."""
        if self.sum_original_actions == 0:
            return 0.0
        return 1.0 - (self.sum_pruned_actions / self.sum_original_actions)

    def record(
        self,
        original_actions: int,
        pruned_actions: int,
        kl_divergence: float,
    ) -> None:
        """Record a pruning operation."""
        self.total_policies += 1
        self.sum_original_actions += original_actions
        self.sum_pruned_actions += pruned_actions
        self.sum_kl_divergence += kl_divergence

    def reset(self) -> None:
        """Reset statistics."""
        self.total_pruned = 0
        self.total_policies = 0
        self.sum_original_actions = 0
        self.sum_pruned_actions = 0
        self.sum_kl_divergence = 0.0


class PolicyTargetPruner:
    """Prunes policy targets for efficient training.

    Reduces the effective action space by removing low-probability actions,
    while preserving the top-k actions and respecting minimum probability
    thresholds.

    Example:
        >>> config = PolicyPruningConfig(min_probability=0.01, preserve_top_k=5)
        >>> pruner = PolicyTargetPruner(config)
        >>>
        >>> pruned_policy = pruner.prune(raw_policy, action_mask)
        >>> # Use pruned_policy as training target
    """

    def __init__(self, config: PolicyPruningConfig):
        """Initialize policy pruner.

        Args:
            config: Pruning configuration
        """
        self.config = config
        self._stats = PruningStats()

    def prune(
        self,
        policy: np.ndarray,
        action_mask: np.ndarray | None = None,
        record_stats: bool = True,
    ) -> np.ndarray:
        """Prune low-probability actions from policy.

        Args:
            policy: Raw policy probabilities (sum to 1)
            action_mask: Optional mask of valid actions
            record_stats: Whether to record statistics

        Returns:
            Pruned and renormalized policy
        """
        policy = policy.copy().astype(np.float64)
        original_actions = int((policy > 0).sum())

        # Apply temperature scaling
        if self.config.temperature_scaling != 1.0:
            policy = self._apply_temperature(policy)

        # Apply action mask
        if action_mask is not None:
            policy = policy * action_mask

        # Get threshold
        if self.config.use_entropy_threshold:
            threshold = self._compute_dynamic_threshold(policy)
        else:
            threshold = self.config.min_probability

        # Get indices sorted by probability (descending)
        sorted_indices = np.argsort(policy)[::-1]

        # Determine which actions to keep
        keep_mask = np.zeros_like(policy, dtype=bool)

        # Always keep top-k
        top_k = min(self.config.preserve_top_k, len(sorted_indices))
        keep_mask[sorted_indices[:top_k]] = True

        # Keep actions above threshold up to max_actions
        kept_count = top_k
        for idx in sorted_indices[top_k:]:
            if kept_count >= self.config.max_actions:
                break
            if policy[idx] >= threshold:
                keep_mask[idx] = True
                kept_count += 1

        # Apply mask and renormalize
        pruned = policy * keep_mask
        total = pruned.sum()
        if total > 0:
            pruned = pruned / total
        else:
            # Fallback: keep original distribution
            pruned = policy / policy.sum() if policy.sum() > 0 else policy

        pruned_actions = int((pruned > 0).sum())

        # Record statistics
        if record_stats:
            kl_div = self._compute_kl_divergence(policy, pruned)
            self._stats.record(original_actions, pruned_actions, kl_div)

        return pruned.astype(np.float32)

    def _apply_temperature(self, policy: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to policy."""
        if self.config.temperature_scaling == 1.0:
            return policy

        # Avoid numerical issues
        policy = np.clip(policy, 1e-10, 1.0)
        policy = np.power(policy, 1.0 / self.config.temperature_scaling)

        total = policy.sum()
        if total > 0:
            policy = policy / total

        return policy

    def _compute_dynamic_threshold(self, policy: np.ndarray) -> float:
        """Compute dynamic threshold based on policy entropy."""
        # Compute entropy
        policy_nonzero = policy[policy > 0]
        if len(policy_nonzero) <= 1:
            return self.config.entropy_min_threshold

        entropy = -np.sum(policy_nonzero * np.log2(policy_nonzero))
        max_entropy = np.log2(len(policy_nonzero))

        if max_entropy == 0:
            normalized_entropy = 0.0
        else:
            normalized_entropy = entropy / max_entropy

        # Interpolate threshold based on entropy
        # High entropy (uniform) -> higher threshold (prune more)
        # Low entropy (peaked) -> lower threshold (keep more)
        threshold = (
            self.config.entropy_min_threshold
            + normalized_entropy
            * (self.config.entropy_max_threshold - self.config.entropy_min_threshold)
        )

        return threshold

    def _compute_kl_divergence(
        self,
        original: np.ndarray,
        pruned: np.ndarray,
    ) -> float:
        """Compute KL divergence between original and pruned policy."""
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p = original + eps
        q = pruned + eps

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        # KL(P || Q)
        kl = np.sum(p * np.log(p / q))

        return float(kl)

    def compute_pruning_stats(
        self,
        policy: np.ndarray,
        pruned_policy: np.ndarray,
    ) -> dict[str, float]:
        """Compute detailed statistics about a pruning operation.

        Args:
            policy: Original policy
            pruned_policy: Pruned policy

        Returns:
            Dictionary of statistics
        """
        original_actions = int((policy > 0).sum())
        pruned_actions = int((pruned_policy > 0).sum())

        return {
            "original_actions": original_actions,
            "pruned_actions": pruned_actions,
            "reduction_ratio": 1.0 - (pruned_actions / max(original_actions, 1)),
            "kl_divergence": self._compute_kl_divergence(policy, pruned_policy),
        }

    def get_stats(self) -> PruningStats:
        """Get accumulated statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset accumulated statistics."""
        self._stats.reset()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PolicyTargetPruner(min_prob={self.config.min_probability}, "
            f"max_actions={self.config.max_actions}, top_k={self.config.preserve_top_k})"
        )
