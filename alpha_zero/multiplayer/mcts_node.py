"""N-player MCTS node with per-player value tracking.

Unlike the standard MCTSNode which uses negamax-style value flipping
(suitable only for 2-player zero-sum games), this node stores a value
vector with one component per player. This enables Max-N search where
each player maximizes their own value component.

Backward compatible: for num_players=2, behavior degenerates to the
standard negamax case when using the paranoid search strategy.

Example:
    >>> node = NPlayerMCTSNode(num_players=4)
    >>> node.expand(policy, action_mask)
    >>> values = {0: 0.8, 1: -0.2, 2: 0.1, 3: -0.7}
    >>> node.backpropagate(values)
    >>> node.get_value(player_id=0)
    0.8
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from alpha_zero.core.mcts_constants import EPSILON

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NPlayerMCTSNode:
    """MCTS node with per-player value storage.

    Stores a value sum for each player, enabling multi-player search
    strategies (Max-N, Paranoid, Best-Reply).

    Attributes:
        num_players: Number of players in the game.
        prior: Prior probability from neural network policy.
        visit_count: Number of times this node was visited.
        value_sums: Per-player accumulated value sums.
        children: Child nodes keyed by action index.
        parent: Parent node reference for backpropagation.
        action: Action that led to this node from parent.
        player_to_move: Which player acts at this node.
    """

    num_players: int = 2
    prior: float = 0.0
    visit_count: int = 0
    value_sums: dict[int, float] = field(default_factory=dict)
    children: dict[int, NPlayerMCTSNode] = field(default_factory=dict)
    parent: NPlayerMCTSNode | None = field(default=None, repr=False)
    action: int | None = None
    player_to_move: int = 0

    def get_value(self, player_id: int) -> float:
        """Get mean value from a specific player's perspective.

        Args:
            player_id: Player whose value to retrieve.

        Returns:
            Mean value for the player, or 0.0 if unvisited.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sums.get(player_id, 0.0) / self.visit_count

    def get_all_values(self) -> dict[int, float]:
        """Get mean values for all players.

        Returns:
            Dict mapping player_id -> mean value.
        """
        if self.visit_count == 0:
            return {p: 0.0 for p in range(self.num_players)}
        return {
            p: self.value_sums.get(p, 0.0) / self.visit_count
            for p in range(self.num_players)
        }

    @property
    def total_visit_value(self) -> float:
        """Sum of absolute values across all players (for normalization)."""
        if self.visit_count == 0:
            return 0.0
        total = sum(abs(v) for v in self.value_sums.values())
        return total / self.visit_count

    def is_expanded(self) -> bool:
        """Check if node has been expanded with children."""
        return len(self.children) > 0

    def select_child(
        self,
        c_puct: float,
        selecting_player: int,
    ) -> tuple[int, NPlayerMCTSNode]:
        """Select child with highest UCB score for the selecting player.

        Uses PUCT formula: score = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        In Max-N search, the selecting player picks the child that maximizes
        their own value component. This is the key difference from negamax.

        Args:
            c_puct: Exploration constant.
            selecting_player: Player ID making the selection.

        Returns:
            Tuple of (action, child_node).

        Raises:
            AssertionError: If called on a node with no children.
        """
        assert self.children, "select_child called on node with no children"

        best_score = -float("inf")
        best_action = -1
        best_child: NPlayerMCTSNode | None = None

        sqrt_parent_visits = np.sqrt(self.visit_count) if self.visit_count > 0 else 0.0

        for action, child in self.children.items():
            # Q-value: mean value for the selecting player
            q_value = child.get_value(selecting_player)

            # Exploration bonus (PUCT)
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)

            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_child is not None
        return best_action, best_child

    def select_child_paranoid(
        self,
        c_puct: float,
        root_player: int,
    ) -> tuple[int, NPlayerMCTSNode]:
        """Select child using Paranoid search strategy.

        In Paranoid search, non-root players are assumed to collude
        against the root player. This effectively reduces N-player
        to 2-player (root vs alliance), enabling deeper search.

        Args:
            c_puct: Exploration constant.
            root_player: The player from whose perspective we search.

        Returns:
            Tuple of (action, child_node).
        """
        assert self.children, "select_child_paranoid called on node with no children"

        best_score = -float("inf")
        best_action = -1
        best_child: NPlayerMCTSNode | None = None

        sqrt_parent_visits = np.sqrt(self.visit_count) if self.visit_count > 0 else 0.0

        is_root_players_turn = self.player_to_move == root_player

        for action, child in self.children.items():
            q_value = child.get_value(root_player)

            # If it's an opponent's turn, they try to minimize root player's value
            if not is_root_players_turn:
                q_value = -q_value

            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_child is not None
        return best_action, best_child

    def expand(
        self,
        policy: np.ndarray,
        action_mask: np.ndarray,
        next_player: int = 0,
    ) -> None:
        """Expand node with children for valid actions.

        Args:
            policy: Policy probabilities for all actions.
            action_mask: Mask of valid actions (1=valid, 0=invalid).
            next_player: Player who will act at child nodes.
        """
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            logger.warning("No valid actions during expansion")
            return

        # Normalize policy over valid actions
        valid_policy = policy[valid_actions].copy()
        policy_sum = valid_policy.sum()
        if policy_sum > EPSILON:
            valid_policy = valid_policy / policy_sum
        else:
            valid_policy = np.ones_like(valid_policy) / len(valid_policy)

        for i, action in enumerate(valid_actions):
            self.children[int(action)] = NPlayerMCTSNode(
                num_players=self.num_players,
                prior=float(valid_policy[i]),
                parent=self,
                action=int(action),
                player_to_move=next_player,
            )

    def backpropagate(self, values: dict[int, float]) -> None:
        """Backpropagate per-player values up the tree.

        Unlike negamax which flips the value at each level, we store
        the actual value for each player at every node. This is the
        Max-N approach.

        Args:
            values: Dict mapping player_id -> value for this evaluation.
        """
        node: NPlayerMCTSNode | None = self
        while node is not None:
            node.visit_count += 1
            for player_id, value in values.items():
                node.value_sums[player_id] = (
                    node.value_sums.get(player_id, 0.0) + value
                )
            node = node.parent

    def backpropagate_paranoid(
        self,
        root_player_value: float,
        root_player: int,
    ) -> None:
        """Backpropagate using paranoid assumption.

        Root player gets the actual value; all opponents share
        the negated value (they are treated as a coalition).

        Args:
            root_player_value: Value from root player's perspective.
            root_player: ID of the root player.
        """
        values = {}
        per_opponent_value = -root_player_value / max(self.num_players - 1, 1)
        for p in range(self.num_players):
            if p == root_player:
                values[p] = root_player_value
            else:
                values[p] = per_opponent_value
        self.backpropagate(values)

    def get_child_visit_counts(self) -> dict[int, int]:
        """Get visit counts for all children.

        Returns:
            Dict mapping action -> visit_count.
        """
        return {action: child.visit_count for action, child in self.children.items()}

    def get_action_probabilities(
        self,
        action_space_size: int,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Compute action probabilities from visit counts.

        Args:
            action_space_size: Total size of the action space.
            temperature: Sampling temperature (0 = argmax, 1 = proportional).

        Returns:
            Probability distribution over all actions.
        """
        action_probs = np.zeros(action_space_size)

        if not self.children:
            return action_probs

        if temperature < EPSILON:
            # Greedy: pick most visited
            best_action = max(self.children, key=lambda a: self.children[a].visit_count)
            action_probs[best_action] = 1.0
        else:
            visits = np.array([
                self.children[a].visit_count for a in sorted(self.children.keys())
            ], dtype=np.float64)
            actions = sorted(self.children.keys())

            if temperature == 1.0:
                total = visits.sum()
                if total > 0:
                    probs = visits / total
                else:
                    probs = np.ones(len(visits)) / len(visits)
            else:
                # Apply temperature
                log_visits = np.log(visits + EPSILON)
                log_visits = log_visits / temperature
                log_visits -= log_visits.max()  # Numerical stability
                probs = np.exp(log_visits)
                probs = probs / (probs.sum() + EPSILON)

            for i, action in enumerate(actions):
                action_probs[action] = probs[i]

        return action_probs
