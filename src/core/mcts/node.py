"""MCTS tree node -- domain-agnostic."""

from __future__ import annotations

__all__ = ["MCTSNode"]

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class MCTSNode:
    """A single node in the MCTS search tree.

    Attributes:
        prior: Network's prior probability for the action that led here.
        visit_count: Number of times this node has been visited.
        value_sum: Cumulative backpropagated value.
        children: Mapping from action index to child node.
        parent: Parent node (None for root).
        action: Action from parent that led to this node (None for root).
        virtual_loss: Virtual loss counter for parallel MCTS.
    """

    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, MCTSNode] = field(default_factory=dict)
    parent: MCTSNode | None = field(default=None, repr=False)
    action: int | None = None
    virtual_loss: int = 0

    @property
    def value(self) -> float:
        """Mean value Q(s,a) = W/N. Returns 0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def effective_value(self) -> float:
        """Value adjusted for virtual loss (parallel MCTS)."""
        total = self.visit_count + self.virtual_loss
        if total == 0:
            return 0.0
        adjusted = self.value_sum - self.virtual_loss
        return adjusted / total

    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded (has children)."""
        return len(self.children) > 0

    @property
    def is_root(self) -> bool:
        """Whether this is the root node."""
        return self.parent is None

    def expand(
        self,
        policy: NDArray[np.float32],
        action_mask: NDArray[np.bool_],
    ) -> None:
        """Create child nodes for all valid actions.

        Args:
            policy: Raw policy distribution over all actions.
            action_mask: Boolean mask where True = valid action.

        The policy is normalized over valid actions only.
        """
        valid_actions = np.where(action_mask)[0]
        if len(valid_actions) == 0:
            return

        valid_policy = policy[valid_actions].astype(np.float64)
        policy_sum = valid_policy.sum()
        if policy_sum > 0:
            valid_policy = valid_policy / policy_sum
        else:
            valid_policy = np.ones_like(valid_policy) / len(valid_policy)

        for i, action in enumerate(valid_actions):
            self.children[int(action)] = MCTSNode(
                prior=float(valid_policy[i]),
                parent=self,
                action=int(action),
            )

    def backpropagate(self, value: float) -> None:
        """Backpropagate a value up to the root, flipping sign for alternating players.

        Args:
            value: Value to backpropagate (from current player's perspective).
        """
        node: MCTSNode | None = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # negamax: flip for opponent
            node = node.parent

    def add_virtual_loss(self) -> None:
        """Add virtual loss for parallel MCTS (discourages re-selection)."""
        node: MCTSNode | None = self
        while node is not None:
            node.virtual_loss += 1
            node = node.parent

    def remove_virtual_loss(self) -> None:
        """Remove virtual loss after backpropagation."""
        node: MCTSNode | None = self
        while node is not None:
            node.virtual_loss = max(0, node.virtual_loss - 1)
            node = node.parent

    def child_visit_counts(self, action_space_size: int) -> NDArray[np.float32]:
        """Get visit counts for all actions as a flat array.

        Args:
            action_space_size: Total number of actions.

        Returns:
            Float array of shape (action_space_size,) with visit counts.
        """
        counts = np.zeros(action_space_size, dtype=np.float32)
        for action, child in self.children.items():
            counts[action] = child.visit_count
        return counts
