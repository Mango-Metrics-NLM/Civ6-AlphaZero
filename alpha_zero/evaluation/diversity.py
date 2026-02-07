"""Game diversity tracking and analysis.

This module tracks various metrics to measure how diverse the model's gameplay is,
including opening move sequences, game length distribution, and action entropy.

High diversity indicates the model is exploring different strategies rather than
converging to a single dominant approach.

Usage:
    from alpha_zero.evaluation.diversity import DiversityTracker, DiversityMetrics
    from alpha_zero.evaluation.config import DiversityConfig

    config = DiversityConfig(enabled=True, opening_depth=10)
    tracker = DiversityTracker(config)

    # Record games during evaluation
    for game in games:
        tracker.record_game(game.history, baseline_name="random")

    # Get aggregated metrics
    metrics = tracker.compute_metrics()
    print(f"Opening entropy: {metrics.opening_entropy:.2f}")
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from alpha_zero.evaluation.config import DiversityConfig

logger = logging.getLogger(__name__)


@dataclass
class DiversityMetrics:
    """Aggregated diversity metrics from evaluation games.

    Attributes:
        unique_openings: Number of unique opening move sequences
        opening_entropy: Shannon entropy of opening distribution (higher = more diverse)
        avg_game_length: Average number of turns per game
        game_length_std: Standard deviation of game lengths
        game_length_min: Minimum game length observed
        game_length_max: Maximum game length observed
        victory_type_distribution: Distribution of victory types (if tracked)
        action_entropy_by_turn: Entropy of action distribution at each turn
        avg_action_entropy: Average action entropy across all turns
        total_games: Total number of games recorded
    """

    unique_openings: int = 0
    opening_entropy: float = 0.0
    avg_game_length: float = 0.0
    game_length_std: float = 0.0
    game_length_min: int = 0
    game_length_max: int = 0
    victory_type_distribution: dict[str, float] = field(default_factory=dict)
    action_entropy_by_turn: list[float] = field(default_factory=list)
    avg_action_entropy: float = 0.0
    total_games: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiversityMetrics:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Diversity Metrics (from {self.total_games} games):",
            f"  Unique Openings: {self.unique_openings}",
            f"  Opening Entropy: {self.opening_entropy:.3f}",
            f"  Avg Game Length: {self.avg_game_length:.1f} (+/-{self.game_length_std:.1f})",
            f"  Game Length Range: [{self.game_length_min}, {self.game_length_max}]",
            f"  Avg Action Entropy: {self.avg_action_entropy:.3f}",
        ]

        if self.victory_type_distribution:
            lines.append("  Victory Distribution:")
            for vtype, freq in sorted(
                self.victory_type_distribution.items(), key=lambda x: -x[1]
            ):
                lines.append(f"    {vtype}: {freq:.1%}")

        return "\n".join(lines)


class DiversityTracker:
    """Tracks game diversity metrics during evaluation.

    Monitors various aspects of gameplay diversity:
    - Opening move sequences and their frequency
    - Game length distribution
    - Victory type distribution
    - Action entropy over time (how predictable are actions)
    """

    def __init__(self, config: DiversityConfig) -> None:
        """Initialize diversity tracker.

        Args:
            config: Diversity configuration
        """
        self.config = config
        self._opening_counts: dict[tuple[int, ...], int] = defaultdict(int)
        self._game_lengths: list[int] = []
        self._victory_types: dict[str, int] = defaultdict(int)
        self._actions_by_turn: dict[int, list[int]] = defaultdict(list)
        self._baseline_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"game_lengths": [], "openings": defaultdict(int), "games": 0}
        )
        self._total_games = 0

    def record_game(
        self,
        game_history: list[tuple[int, dict[str, Any]]],
        baseline_name: str = "unknown",
        victory_type: str | None = None,
        game_length: int | None = None,
    ) -> None:
        """Record a completed game for diversity analysis.

        Args:
            game_history: List of (action, info) tuples from the game
            baseline_name: Name of the baseline opponent
            victory_type: Type of victory (e.g., "elimination", "timeout")
            game_length: Explicit game length (uses history length if not provided)
        """
        if not self.config.enabled or not game_history:
            return

        self._total_games += 1

        # Extract model's actions
        model_actions = [
            action for action, info in game_history if info.get("is_model", True)
        ]

        # Track opening sequence
        if self.config.track_opening_moves and model_actions:
            opening = tuple(model_actions[: self.config.opening_depth])
            self._opening_counts[opening] += 1
            self._baseline_data[baseline_name]["openings"][opening] += 1

        # Track game length
        length = game_length if game_length is not None else len(game_history)
        if self.config.track_game_lengths:
            self._game_lengths.append(length)
            self._baseline_data[baseline_name]["game_lengths"].append(length)

        # Track victory type
        if self.config.track_victory_types and victory_type:
            self._victory_types[victory_type] += 1

        # Track actions by turn for entropy
        if self.config.track_action_entropy:
            for action, info in game_history:
                if info.get("is_model", True):
                    turn = info.get("turn", 0)
                    self._actions_by_turn[turn].append(action)

        self._baseline_data[baseline_name]["games"] += 1

    def compute_metrics(self) -> DiversityMetrics:
        """Compute aggregated diversity metrics."""
        metrics = DiversityMetrics(total_games=self._total_games)

        # Opening diversity
        if self._opening_counts:
            metrics.unique_openings = len(self._opening_counts)
            metrics.opening_entropy = self._compute_entropy(
                list(self._opening_counts.values())
            )

        # Game length statistics
        if self._game_lengths:
            lengths = np.array(self._game_lengths)
            metrics.avg_game_length = float(np.mean(lengths))
            metrics.game_length_std = float(np.std(lengths))
            metrics.game_length_min = int(np.min(lengths))
            metrics.game_length_max = int(np.max(lengths))

        # Victory distribution
        if self._victory_types:
            total = sum(self._victory_types.values())
            metrics.victory_type_distribution = {
                k: v / total for k, v in self._victory_types.items()
            }

        # Action entropy by turn
        if self._actions_by_turn:
            entropies = []
            for turn in sorted(self._actions_by_turn.keys()):
                actions = self._actions_by_turn[turn]
                if len(actions) >= 2:
                    entropy = self._compute_action_entropy(actions)
                    entropies.append(entropy)
                    metrics.action_entropy_by_turn.append(entropy)
            if entropies:
                metrics.avg_action_entropy = float(np.mean(entropies))

        return metrics

    def _compute_entropy(self, counts: list[int]) -> float:
        """Compute Shannon entropy from counts."""
        if not counts:
            return 0.0
        total = sum(counts)
        if total == 0:
            return 0.0
        probs = np.array(counts) / total
        probs = probs[probs > 0]
        if len(probs) <= 1:
            return 0.0
        return float(-np.sum(probs * np.log2(probs)))

    def _compute_action_entropy(self, actions: list[int]) -> float:
        """Compute entropy of an action distribution."""
        if len(actions) < 2:
            return 0.0
        action_counts: dict[int, int] = defaultdict(int)
        for a in actions:
            action_counts[a] += 1
        return self._compute_entropy(list(action_counts.values()))

    def get_baseline_metrics(self, baseline_name: str) -> dict[str, Any]:
        """Get metrics specific to a single baseline."""
        data = self._baseline_data.get(baseline_name, {})
        if not data:
            return {"games": 0}
        result = {
            "games": data.get("games", 0),
            "unique_openings": len(data.get("openings", {})),
        }
        lengths = data.get("game_lengths", [])
        if lengths:
            result["avg_game_length"] = float(np.mean(lengths))
            result["game_length_std"] = float(np.std(lengths))
        return result

    def get_opening_distribution(
        self, top_k: int = 10
    ) -> list[tuple[tuple[int, ...], int, float]]:
        """Get the most common opening sequences."""
        if not self._opening_counts:
            return []
        total = sum(self._opening_counts.values())
        sorted_openings = sorted(
            self._opening_counts.items(), key=lambda x: -x[1]
        )[:top_k]
        return [(opening, count, count / total) for opening, count in sorted_openings]

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self._opening_counts.clear()
        self._game_lengths.clear()
        self._victory_types.clear()
        self._actions_by_turn.clear()
        self._baseline_data.clear()
        self._total_games = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert tracker state to dictionary."""
        return {
            "config": self.config.to_dict(),
            "opening_counts": {str(k): v for k, v in self._opening_counts.items()},
            "game_lengths": self._game_lengths,
            "victory_types": dict(self._victory_types),
            "total_games": self._total_games,
        }
