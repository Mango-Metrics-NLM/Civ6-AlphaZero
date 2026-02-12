"""MCTS child-selection policies (PUCT, Gumbel, etc.)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.core.types import DEFAULT_C_PUCT


class PUCTPolicy:
    """PUCT (Polynomial Upper Confidence Trees) selection policy.

    Formula: score(a) = Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))

    Args:
        c_puct: Exploration constant.
    """

    def __init__(self, c_puct: float = DEFAULT_C_PUCT) -> None:
        self._c_puct = c_puct

    @property
    def c_puct(self) -> float:
        """Return the PUCT exploration constant."""
        return self._c_puct

    def select_child(
        self,
        parent_visit_count: int,
        children_priors: NDArray[np.float32],
        children_values: NDArray[np.float32],
        children_visit_counts: NDArray[np.int64],
        valid_mask: NDArray[np.bool_],
    ) -> int:
        """Select child using PUCT formula.

        Q values are negated (negamax convention).

        Args:
            parent_visit_count: Total visit count of the parent node.
            children_priors: Prior probabilities for each child.
            children_values: Mean action-value (Q) for each child.
            children_visit_counts: Visit count for each child.
            valid_mask: Boolean mask indicating which children are valid.

        Returns:
            Index of the selected child.
        """
        sqrt_parent = np.sqrt(float(parent_visit_count))

        # PUCT score: -Q(a) + c * P(a) * sqrt(N) / (1 + n(a))
        exploration = (
            self._c_puct * children_priors * sqrt_parent / (1.0 + children_visit_counts)
        )
        scores = -children_values + exploration  # negamax: negate Q

        # Mask invalid children
        scores = np.where(valid_mask, scores, -np.inf)

        return int(np.argmax(scores))
