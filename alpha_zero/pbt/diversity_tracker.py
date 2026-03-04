"""Policy diversity measurement for population-based training.

Tracks action distributions for each agent in the population and
computes diversity metrics including entropy, pairwise KL divergence,
and a combined behavioural diversity score. Agents with more unique
strategies receive a fitness bonus to encourage population diversity.

Usage::

    from alpha_zero.pbt.diversity_tracker import (
        PolicyDiversityTracker,
        DiversityMetrics,
    )

    tracker = PolicyDiversityTracker(action_space_size=356)
    tracker.record_actions("agent_1", [0, 1, 2, 0, 1])
    tracker.record_actions("agent_2", [3, 4, 5, 3, 4])
    metrics = tracker.compute_diversity()
    bonus = tracker.get_diversity_bonus("agent_1", weight=0.1)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiversityMetrics:
    """Diversity measurements for a population.

    Attributes:
        action_entropy: Average entropy of per-agent action distributions.
            Higher values indicate agents explore more of the action space.
        pairwise_kl_divergence: Average KL divergence between all pairs
            of agent action distributions. Higher values indicate more
            diverse strategies across the population.
        behavioral_diversity: Combined diversity score (weighted average
            of entropy and KL divergence).
    """

    action_entropy: float = 0.0
    pairwise_kl_divergence: float = 0.0
    behavioral_diversity: float = 0.0


class PolicyDiversityTracker:
    """Tracks behavioural diversity across a population of agents.

    Maintains action frequency histograms for each agent and computes
    various diversity metrics from those distributions.

    Args:
        action_space_size: Size of the discrete action space.
    """

    def __init__(self, action_space_size: int) -> None:
        if action_space_size < 1:
            raise ValueError(
                f"action_space_size must be >= 1, got {action_space_size}"
            )
        self._action_space_size = action_space_size
        self._action_counts: dict[str, np.ndarray] = {}

    @property
    def action_space_size(self) -> int:
        """Size of the discrete action space."""
        return self._action_space_size

    @property
    def tracked_agents(self) -> list[str]:
        """List of agent IDs currently being tracked."""
        return list(self._action_counts.keys())

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_actions(self, agent_id: str, actions: list[int]) -> None:
        """Record actions taken by an agent to build its action distribution.

        Args:
            agent_id: Identifier for the agent.
            actions: List of action indices taken by the agent.
        """
        if agent_id not in self._action_counts:
            self._action_counts[agent_id] = np.zeros(
                self._action_space_size, dtype=np.float64
            )

        counts = self._action_counts[agent_id]
        for action in actions:
            if 0 <= action < self._action_space_size:
                counts[action] += 1

    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------

    def _get_distribution(self, agent_id: str) -> np.ndarray:
        """Get normalised action distribution for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            Probability distribution over actions. If no actions have
            been recorded, returns a uniform distribution.
        """
        if agent_id not in self._action_counts:
            return np.ones(self._action_space_size) / self._action_space_size

        counts = self._action_counts[agent_id]
        total = counts.sum()
        if total < 1e-9:
            return np.ones(self._action_space_size) / self._action_space_size

        result: np.ndarray = counts / total
        return result

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_diversity(self) -> DiversityMetrics:
        """Compute diversity metrics across all tracked agents.

        Returns:
            DiversityMetrics with entropy, KL divergence, and combined
            behavioural diversity scores.
        """
        agent_ids = list(self._action_counts.keys())

        if len(agent_ids) == 0:
            return DiversityMetrics()

        # Compute per-agent entropies
        entropies: list[float] = []
        distributions: dict[str, np.ndarray] = {}
        for aid in agent_ids:
            dist = self._get_distribution(aid)
            distributions[aid] = dist
            entropies.append(self._entropy(dist))

        avg_entropy = float(np.mean(entropies)) if entropies else 0.0

        # Compute pairwise KL divergence
        avg_kl = 0.0
        if len(agent_ids) >= 2:
            kl_values: list[float] = []
            for i, aid_i in enumerate(agent_ids):
                for j, aid_j in enumerate(agent_ids):
                    if i < j:
                        kl_ij = self._kl_divergence(
                            distributions[aid_i], distributions[aid_j]
                        )
                        kl_ji = self._kl_divergence(
                            distributions[aid_j], distributions[aid_i]
                        )
                        # Symmetric KL (Jensen-Shannon-like)
                        kl_values.append((kl_ij + kl_ji) / 2.0)
            avg_kl = float(np.mean(kl_values)) if kl_values else 0.0

        # Combined metric: weighted average of normalised entropy and KL
        max_entropy = float(np.log(self._action_space_size)) if self._action_space_size > 1 else 1.0
        norm_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0.0
        # KL is unbounded; use sigmoid-like saturation
        norm_kl = 1.0 - 1.0 / (1.0 + avg_kl)

        behavioral_diversity = 0.5 * norm_entropy + 0.5 * norm_kl

        logger.debug(
            "Diversity metrics: entropy=%.4f, kl=%.4f, diversity=%.4f "
            "(agents=%d)",
            avg_entropy,
            avg_kl,
            behavioral_diversity,
            len(agent_ids),
        )

        return DiversityMetrics(
            action_entropy=avg_entropy,
            pairwise_kl_divergence=avg_kl,
            behavioral_diversity=behavioral_diversity,
        )

    def get_diversity_bonus(self, agent_id: str, weight: float = 0.1) -> float:
        """Compute diversity bonus for an agent's fitness.

        Agents with action distributions that differ more from the
        population mean receive a higher bonus, encouraging the
        population to maintain strategic variety.

        Args:
            agent_id: Agent to compute bonus for.
            weight: Scaling factor for the bonus.

        Returns:
            Non-negative diversity bonus.
        """
        agent_ids = list(self._action_counts.keys())

        if len(agent_ids) < 2 or agent_id not in self._action_counts:
            return 0.0

        agent_dist = self._get_distribution(agent_id)

        # Compute mean distribution across all other agents
        other_dists = [
            self._get_distribution(aid) for aid in agent_ids if aid != agent_id
        ]
        if not other_dists:
            return 0.0

        mean_dist = np.mean(other_dists, axis=0)

        # KL divergence from the mean
        kl_from_mean = self._kl_divergence(agent_dist, mean_dist)

        # Saturate using sigmoid-like function and scale by weight
        bonus = weight * (1.0 - 1.0 / (1.0 + kl_from_mean))
        return float(bonus)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all tracked distributions."""
        logger.debug(
            "Resetting diversity tracker (%d agents cleared)",
            len(self._action_counts),
        )
        self._action_counts.clear()

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy(dist: np.ndarray) -> float:
        """Compute Shannon entropy of a probability distribution.

        Args:
            dist: Probability distribution (sums to 1).

        Returns:
            Entropy in nats.
        """
        # Filter out zeros to avoid log(0)
        nonzero = dist[dist > 0]
        return float(-np.sum(nonzero * np.log(nonzero)))

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence KL(p || q).

        Adds a small epsilon to avoid log(0) and division by zero.

        Args:
            p: First distribution.
            q: Second distribution.

        Returns:
            KL divergence (non-negative).
        """
        eps = 1e-10
        p_safe = np.clip(p, eps, 1.0)
        q_safe = np.clip(q, eps, 1.0)
        return float(np.sum(p_safe * np.log(p_safe / q_safe)))
