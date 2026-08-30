"""Shared PUCT search state: configuration and the classic tree node."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alpha_zero.core.mcts_constants import EPSILON
from alpha_zero.utils.constants import MCTS


@dataclass(slots=True)
class MCTSConfig:
    """Configuration for MCTS.

    Defaults sourced from alpha_zero.utils.constants.MCTS to keep hyperparameter
    values consistent across all MCTS variants and the orchestrator config.
    """

    num_simulations: int = MCTS.CLASSIC_NUM_SIMULATIONS
    c_puct: float = MCTS.DEFAULT_C_PUCT
    dirichlet_alpha: float = MCTS.DEFAULT_DIRICHLET_ALPHA
    dirichlet_epsilon: float = MCTS.DEFAULT_DIRICHLET_EPSILON
    temperature: float = MCTS.DEFAULT_TEMPERATURE
    temperature_threshold: int = MCTS.DEFAULT_TEMPERATURE_THRESHOLD

    def __post_init__(self) -> None:
        if not MCTS.MIN_SIMULATIONS <= self.num_simulations <= MCTS.MAX_SIMULATIONS:
            raise ValueError(
                f"num_simulations must be in [{MCTS.MIN_SIMULATIONS}, "
                f"{MCTS.MAX_SIMULATIONS}]; got {self.num_simulations}"
            )
        if self.c_puct <= 0:
            raise ValueError(f"c_puct must be > 0; got {self.c_puct}")
        if self.dirichlet_alpha <= 0:
            raise ValueError(f"dirichlet_alpha must be > 0; got {self.dirichlet_alpha}")
        if not 0.0 <= self.dirichlet_epsilon <= 1.0:
            raise ValueError(f"dirichlet_epsilon must be in [0, 1]; got {self.dirichlet_epsilon}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0; got {self.temperature}")
        if self.temperature_threshold < 0:
            raise ValueError(
                f"temperature_threshold must be >= 0; got {self.temperature_threshold}"
            )


def _env_current_player(env: object) -> int | None:
    """Return the env's acting player, or None when it does not expose one.

    Stub environments in tests often have no current_player; those keep the
    historical per-ply alternation.
    """
    player = getattr(env, "current_player", None)
    return None if player is None else int(player)


@dataclass(slots=True)
class MCTSNode:
    """Node in the MCTS tree.

    Attributes:
        prior: Prior probability from neural network.
        visit_count: Number of times node was visited.
        value_sum: Sum of values from all visits.
        children: Child nodes keyed by action.
        parent: Parent node.
        action: Action that led to this node.
        virtual_loss: Virtual loss for parallel MCTS (threads in flight).
        to_play: Player to move in this node's state, or None when the caller
            does not track it. Values are stored from the acting player's own
            perspective, so a sign flip is correct only across a change of
            player — and in MicroCiv a player makes many moves per turn.
    """

    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, MCTSNode] = field(default_factory=dict)
    parent: MCTSNode | None = None
    action: int | None = None
    virtual_loss: int = 0
    to_play: int | None = None

    @property
    def value(self) -> float:
        """Mean value estimate."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def effective_value(self) -> float:
        """Value adjusted for virtual loss (for parallel MCTS)."""
        total_visits = self.visit_count + self.virtual_loss
        if total_visits == 0:
            return 0.0
        # Virtual loss assumes worst case (-1)
        adjusted_sum = self.value_sum - self.virtual_loss
        return adjusted_sum / total_visits

    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0

    def add_virtual_loss(self, loss: int = 1) -> None:
        """Add virtual loss for parallel MCTS.

        Virtual loss discourages other threads from exploring the same path.

        Args:
            loss: Amount of virtual loss to add.
        """
        self.virtual_loss += loss

    def remove_virtual_loss(self, loss: int = 1) -> None:
        """Remove virtual loss after backpropagation.

        Args:
            loss: Amount of virtual loss to remove.
        """
        self.virtual_loss = max(0, self.virtual_loss - loss)

    def select_child(self, c_puct: float, use_virtual_loss: bool = False) -> tuple[int, MCTSNode]:
        """Select child with highest UCB score.

        Args:
            c_puct: Exploration constant.
            use_virtual_loss: Whether to use virtual loss in selection.

        Returns:
            Tuple of (action, child_node).
        """
        best_score = -float("inf")
        best_action = -1
        best_child = None

        sqrt_parent_visits = np.sqrt(
            self.visit_count + self.virtual_loss if use_virtual_loss else self.visit_count
        )

        for action, child in self.children.items():
            # Negate only when the child is a different player's decision.
            # Unconditional negation would pair badly with the to_play-aware
            # backup: across a same-player edge it would make PUCT prefer the
            # child that *minimises* the mover's own value.
            same_player = (
                self.to_play is not None
                and child.to_play is not None
                and child.to_play == self.to_play
            )
            sign = 1.0 if same_player else -1.0

            # PUCT formula with optional virtual loss
            if use_virtual_loss:
                q_value = sign * child.effective_value  # Use virtual-loss-adjusted value
                effective_visits = child.visit_count + child.virtual_loss
            else:
                q_value = sign * child.value
                effective_visits = child.visit_count
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + effective_visits)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_child is not None, "select_child called on node with no children"
        return best_action, best_child

    def expand(self, policy: np.ndarray, action_mask: np.ndarray) -> None:
        """Expand node with children for valid actions.

        Args:
            policy: Policy probabilities for all actions.
            action_mask: Mask of valid actions.
        """
        # Get valid actions
        valid_actions = np.where(action_mask > 0)[0]

        # Normalize policy over valid actions
        valid_policy = policy[valid_actions]
        valid_policy = valid_policy / (valid_policy.sum() + EPSILON)

        # Create child nodes
        for i, action in enumerate(valid_actions):
            self.children[int(action)] = MCTSNode(
                prior=float(valid_policy[i]),
                parent=self,
                action=int(action),
            )

    def backpropagate(self, value: float) -> None:
        """Backpropagate ``value`` (from this node's perspective) to the root.

        The sign flips only where the acting player changes. Flipping on every
        edge is correct in a game with one ply per turn, but wrong in MicroCiv,
        where a player moves each unit before ending their turn.

        Nodes with an unknown ``to_play`` fall back to per-ply alternation so
        callers that never set it keep their previous behaviour.

        Args:
            value: Value from this node's perspective.
        """
        node: MCTSNode | None = self
        reference = self.to_play
        while node is not None:
            if reference is None or node.to_play is None:
                signed = value
                value = -value
            else:
                signed = value if node.to_play == reference else -value
            node.visit_count += 1
            node.value_sum += signed
            node = node.parent
