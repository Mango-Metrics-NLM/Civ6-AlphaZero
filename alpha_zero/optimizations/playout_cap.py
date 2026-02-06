"""Playout cap randomization for efficient MCTS.

Implements playout cap randomization technique for significant speedups
during self-play without sacrificing training quality.

Key insight: Most positions don't need the full simulation budget.
Randomly varying playout caps provides similar training signal
with significantly less compute (up to 50x speedup reported).

Usage:
    from alpha_zero.optimizations.playout_cap import PlayoutCapRandomizer, PlayoutCapConfig

    config = PlayoutCapConfig(min_playouts=50, max_playouts=800)
    randomizer = PlayoutCapRandomizer(config)

    # During self-play
    num_sims = randomizer.sample_playout_cap(observation)
    action_probs = mcts.search(env, num_simulations=num_sims)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CapDistribution(str, Enum):
    """Distribution types for sampling playout caps."""

    UNIFORM = "uniform"
    GEOMETRIC = "geometric"
    EXPONENTIAL = "exponential"
    LEARNED = "learned"


@dataclass
class PlayoutCapConfig:
    """Configuration for playout cap randomization.

    Attributes:
        min_playouts: Minimum number of MCTS simulations
        max_playouts: Maximum number of MCTS simulations
        cap_distribution: Distribution type for sampling
        geometric_p: Parameter for geometric distribution (probability of stopping)
        exponential_lambda: Parameter for exponential distribution
        use_position_complexity: Scale caps based on estimated position complexity
        early_game_multiplier: Multiplier for early game positions
        late_game_multiplier: Multiplier for late game positions
        early_game_turns: Number of turns considered "early game"
    """

    min_playouts: int = 50
    max_playouts: int = 800
    cap_distribution: str = "geometric"
    geometric_p: float = 0.3
    exponential_lambda: float = 0.005
    use_position_complexity: bool = True
    early_game_multiplier: float = 1.2
    late_game_multiplier: float = 0.8
    early_game_turns: int = 10

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.min_playouts < 1:
            raise ValueError(f"min_playouts must be >= 1, got {self.min_playouts}")
        if self.max_playouts < self.min_playouts:
            raise ValueError(
                f"max_playouts ({self.max_playouts}) must be >= min_playouts ({self.min_playouts})"
            )
        if not 0 < self.geometric_p < 1:
            raise ValueError(f"geometric_p must be in (0, 1), got {self.geometric_p}")
        if self.exponential_lambda <= 0:
            raise ValueError(f"exponential_lambda must be > 0, got {self.exponential_lambda}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayoutCapConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min_playouts": self.min_playouts,
            "max_playouts": self.max_playouts,
            "cap_distribution": self.cap_distribution,
            "geometric_p": self.geometric_p,
            "exponential_lambda": self.exponential_lambda,
            "use_position_complexity": self.use_position_complexity,
            "early_game_multiplier": self.early_game_multiplier,
            "late_game_multiplier": self.late_game_multiplier,
            "early_game_turns": self.early_game_turns,
        }


@dataclass
class PlayoutCapStats:
    """Statistics about playout cap sampling.

    Attributes:
        total_samples: Total number of caps sampled
        sum_caps: Sum of all caps for average calculation
        min_sampled: Minimum cap actually sampled
        max_sampled: Maximum cap actually sampled
        cap_histogram: Histogram of sampled caps
    """

    total_samples: int = 0
    sum_caps: int = 0
    min_sampled: int = 0
    max_sampled: int = 0
    cap_histogram: dict[int, int] = field(default_factory=dict)

    @property
    def average_cap(self) -> float:
        """Average cap sampled."""
        if self.total_samples == 0:
            return 0.0
        return self.sum_caps / self.total_samples

    def record(self, cap: int) -> None:
        """Record a sampled cap."""
        self.total_samples += 1
        self.sum_caps += cap
        if self.total_samples == 1:
            self.min_sampled = cap
            self.max_sampled = cap
        else:
            self.min_sampled = min(self.min_sampled, cap)
            self.max_sampled = max(self.max_sampled, cap)

        # Bucket into groups of 50 for histogram
        bucket = (cap // 50) * 50
        self.cap_histogram[bucket] = self.cap_histogram.get(bucket, 0) + 1

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_samples = 0
        self.sum_caps = 0
        self.min_sampled = 0
        self.max_sampled = 0
        self.cap_histogram.clear()


class PlayoutCapRandomizer:
    """Randomizes playout caps during self-play.

    Provides significant speedups by varying the number of MCTS simulations
    per position based on configurable distributions.

    Example:
        >>> config = PlayoutCapConfig(min_playouts=50, max_playouts=400)
        >>> randomizer = PlayoutCapRandomizer(config)
        >>>
        >>> for position in game:
        ...     num_sims = randomizer.sample_playout_cap(observation, turn=turn)
        ...     action_probs = mcts.search(env, num_simulations=num_sims)
    """

    def __init__(
        self,
        config: PlayoutCapConfig,
        rng: np.random.Generator | None = None,
        complexity_fn: Callable[[dict[str, np.ndarray]], float] | None = None,
    ) -> None:
        """Initialize playout cap randomizer.

        Args:
            config: Playout cap configuration
            rng: Random number generator (uses default if None)
            complexity_fn: Optional function to estimate position complexity
        """
        self.config = config
        self._rng = rng or np.random.default_rng()
        self._complexity_fn = complexity_fn
        self._stats = PlayoutCapStats()

    def sample_playout_cap(
        self,
        observation: dict[str, np.ndarray] | None = None,
        turn: int = 0,
    ) -> int:
        """Sample a playout cap for the current position.

        Args:
            observation: Current game observation (for complexity scaling)
            turn: Current turn number (for early/late game scaling)

        Returns:
            Number of MCTS simulations to run
        """
        # Sample base cap from distribution
        base_cap = self._sample_from_distribution()

        # Apply complexity scaling if enabled
        if self.config.use_position_complexity and observation is not None:
            complexity = self._estimate_complexity(observation, turn)
            base_cap = int(base_cap * complexity)

        # Clamp to valid range
        cap = max(self.config.min_playouts, min(self.config.max_playouts, base_cap))

        # Record statistics
        self._stats.record(cap)

        return cap

    def _sample_from_distribution(self) -> int:
        """Sample from the configured distribution."""
        dist = CapDistribution(self.config.cap_distribution)

        if dist == CapDistribution.UNIFORM:
            return int(
                self._rng.integers(
                    self.config.min_playouts,
                    self.config.max_playouts + 1,
                )
            )

        elif dist == CapDistribution.GEOMETRIC:
            # Geometric distribution favors lower values
            # Each doubling has probability (1 - p) of occurring
            k = int(self._rng.geometric(self.config.geometric_p))
            return min(
                self.config.min_playouts * (2 ** (k - 1)),
                self.config.max_playouts,
            )

        elif dist == CapDistribution.EXPONENTIAL:
            # Exponential decay from max
            sample = self._rng.exponential(1.0 / self.config.exponential_lambda)
            return int(
                max(
                    self.config.min_playouts,
                    self.config.max_playouts - sample,
                )
            )

        else:  # LEARNED or unknown - use max
            return self.config.max_playouts

    def _estimate_complexity(
        self,
        observation: dict[str, np.ndarray],
        turn: int,
    ) -> float:
        """Estimate position complexity for scaling.

        Args:
            observation: Current observation
            turn: Current turn number

        Returns:
            Complexity multiplier (typically 0.5 to 1.5)
        """
        # Use custom complexity function if provided
        if self._complexity_fn is not None:
            return self._complexity_fn(observation)

        multiplier = 1.0

        # Apply early/late game multipliers
        if turn < self.config.early_game_turns:
            multiplier *= self.config.early_game_multiplier
        elif turn > self.config.early_game_turns * 3:
            multiplier *= self.config.late_game_multiplier

        # Use action mask density as complexity proxy
        if "action_mask" in observation:
            mask = observation["action_mask"]
            valid_ratio = mask.sum() / len(mask)
            # More valid actions = more complex position
            # Scale from 0.7 to 1.3 based on valid action ratio
            multiplier *= 0.7 + 0.6 * valid_ratio

        return multiplier

    def get_effective_speedup(self) -> float:
        """Compute effective speedup over using max_playouts always.

        Returns:
            Speedup factor (e.g., 2.0 means 2x faster)
        """
        if self._stats.total_samples == 0:
            return 1.0

        avg_cap = self._stats.average_cap
        if avg_cap == 0:
            return 1.0

        return self.config.max_playouts / avg_cap

    def get_stats(self) -> PlayoutCapStats:
        """Get sampling statistics.

        Returns:
            PlayoutCapStats object
        """
        return self._stats

    def reset_stats(self) -> None:
        """Reset sampling statistics."""
        self._stats.reset()

    def set_complexity_function(
        self,
        fn: Callable[[dict[str, np.ndarray]], float],
    ) -> None:
        """Set custom complexity estimation function.

        Args:
            fn: Function that takes observation and returns complexity multiplier
        """
        self._complexity_fn = fn

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PlayoutCapRandomizer(min={self.config.min_playouts}, "
            f"max={self.config.max_playouts}, dist={self.config.cap_distribution})"
        )
