"""MuZero's search tree: nodes, the mover model, Q normalisation and the signed backup.

Why this module exists
----------------------
``MuZeroMCTS`` searches a *learned* latent model, so past the root there is no
environment to ask who is to move. Until 2026-11-12 its nodes carried no
``to_play`` at all and its backup summed rewards with no sign flip, as if the
game had one player. MuZero is self-played on two-player MicroCiv, whose
rewards are relative to a player, so an opponent's gain counted as the root
player's gain (the T18 forced-win oracle failed every even-depth alternating
case).

The mover does not have to be learned. ``MicroCivEnv`` changes
``current_player`` only inside ``_end_turn``, which only the END_TURN action
reaches, and the search knows every action on its path. A :class:`TurnModel`
encodes that rule; :class:`EndTurnModel` is MicroCiv's.

Frames
------
* A node's ``value`` is from the perspective of its own ``to_play``.
* An edge's ``reward`` is from the perspective of the player to move *after*
  the transition -- the child's ``to_play``. This is MicroCiv's frame:
  ``step`` scores the terminal outcome against ``current_player`` after the
  action, and a turn-limit end only follows END_TURN, which has already passed
  the move.
* So ``reward + discount * child.value`` is a single quantity in the child's
  frame, and :func:`edge_q` signs it into the parent's with
  :func:`~alpha_zero.core.mcts.backup.perspective_sign` -- the same rule every
  other search variant uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from alpha_zero.core.mcts.backup import perspective_sign

__all__ = [
    "MICROCIV_END_TURN_ACTION",
    "EndTurnModel",
    "MinMaxStats",
    "MuZeroNode",
    "TurnModel",
    "backpropagate_with_rewards",
    "edge_q",
]

#: Flat index of END_TURN in ``MicroCivActionSpace`` (``end_turn_offset``).
#: ``tests/unit/test_muzero_turn_model.py`` holds this against the action space
#: and against a played game, so the two cannot drift apart silently.
MICROCIV_END_TURN_ACTION = 0


class TurnModel(Protocol):
    """Who moves after ``action`` is played by ``to_play``."""

    def __call__(self, to_play: int, action: int) -> int: ...


@dataclass(frozen=True)
class EndTurnModel:
    """Two-player games in which exactly one action passes the move (MicroCiv).

    A scalar value flipped between two perspectives cannot express a game with
    more than two players -- "not mine" is not "the opponent's" -- so this model
    is two-player only; ``alpha_zero.multiplayer.mcts.NPlayerMCTS`` keeps a real
    per-player value vector.
    """

    end_turn_action: int = MICROCIV_END_TURN_ACTION

    def __call__(self, to_play: int, action: int) -> int:
        if to_play not in (0, 1):
            # ``1 - to_play`` would return -1 for player 2: refuse loudly rather
            # than sign a >2-player game wrongly.
            raise ValueError(
                f"EndTurnModel is two-player only; got to_play={to_play}. "
                "Use NPlayerMCTS for more players."
            )
        return 1 - to_play if action == self.end_turn_action else to_play


class MinMaxStats:
    """Running bounds of the in-tree Q values, for normalising selection (MuZero).

    Replaces a ``[0, 1]`` normaliser that was fixed per simulation and never
    updated, so it normalised nothing once rewards and signed values left that
    range.
    """

    def __init__(self) -> None:
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, value: float) -> None:
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        # With no spread yet the value passes through unnormalised, as in
        # MuZero's reference. A constant here (0.5 was tried) scores every
        # visited child above an unvisited one at equal Q, so selection keeps
        # following the branch it already visited.
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


@dataclass
class MuZeroNode:
    """Node in the MuZero search tree, holding a learned hidden state.

    Attributes:
        hidden_state: Hidden state tensor from the representation/dynamics function.
        reward: Predicted reward on the edge into this node, in this node's
            ``to_play`` frame (see the module docstring).
        prior: Prior probability from the policy network.
        visit_count: Number of times the node was visited.
        value_sum: Sum of backed-up values, in this node's ``to_play`` frame.
        to_play: Player to move in this node's state. ``MuZeroMCTS`` sets it on
            every node it creates; ``None`` only for nodes built by hand, which
            then fall back to per-ply alternation like every other variant.
        children: Child nodes keyed by action.
        parent: Parent node.
        action: Action that led to this node.
    """

    hidden_state: torch.Tensor | None = None
    reward: float = 0.0
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    to_play: int | None = None
    children: dict[int, MuZeroNode] = field(default_factory=dict)
    parent: MuZeroNode | None = None
    action: int | None = None

    @property
    def value(self) -> float:
        """Mean value estimate, in this node's ``to_play`` frame."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        """Check if node has been expanded."""
        return len(self.children) > 0

    def select_child(
        self,
        c_puct: float,
        min_q: float,
        max_q: float,
        discount: float = 1.0,
    ) -> tuple[int, MuZeroNode | None]:
        """Select the child with the highest PUCT score.

        The exploitation term is the edge's Q in *this* node's frame,
        ``sign * (child.reward + discount * child.value)``, normalised into
        ``[0, 1]`` by ``(min_q, max_q)`` -- the bounds of a :class:`MinMaxStats` --
        and left raw while there is no spread, so equal Q values score equally.
        It scored ``child.value`` alone until 2026-11-12: unsigned, so a
        player-change edge was ranked from the opponent's side, and without the
        edge reward, so a game ending on that edge looked worth nothing.

        Args:
            c_puct: Exploration constant.
            min_q: Lower bound for normalisation.
            max_q: Upper bound for normalisation.
            discount: Discount applied to the child's value.

        Returns:
            Tuple of (action, child_node).
        """
        best_score = -float("inf")
        best_action = -1
        best_child: MuZeroNode | None = None

        sqrt_parent_visits = np.sqrt(self.visit_count)

        for action, child in self.children.items():
            if child.visit_count > 0:
                q_value = edge_q(self, child, discount)
                if max_q > min_q:
                    q_value = (q_value - min_q) / (max_q - min_q)
            else:
                q_value = 0.0

            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_child is not None, "select_child called on node with no children"
        return best_action, best_child

    def expand(
        self,
        actions: list[int],
        priors: np.ndarray,
        turn_model: TurnModel | None = None,
    ) -> None:
        """Expand node with children for the given actions.

        Args:
            actions: Valid action indices.
            priors: Prior probability of each action.
            turn_model: Who moves after each action. ``None`` leaves the
                children's ``to_play`` unset (per-ply alternation).
        """
        for i, action in enumerate(actions):
            child_to_play = (
                None
                if turn_model is None or self.to_play is None
                else turn_model(self.to_play, action)
            )
            self.children[action] = MuZeroNode(
                prior=float(priors[i]),
                parent=self,
                action=action,
                to_play=child_to_play,
            )


def edge_q(parent: MuZeroNode, child: MuZeroNode, discount: float) -> float:
    """Q of the edge ``parent -> child`` in the parent's frame.

    ``child.reward`` and ``child.value`` are both in the child's frame, so they
    combine first and are signed once. Unvisited children are worth 0, as in
    MuZero's reference ``ucb_score``.
    """
    if child.visit_count == 0:
        return 0.0
    sign = perspective_sign(parent.to_play, child.to_play)
    return sign * (child.reward + discount * child.value)


def backpropagate_with_rewards(
    path: Sequence[MuZeroNode],
    leaf_value: float,
    discount: float,
    stats: MinMaxStats | None = None,
) -> None:
    """Back a leaf evaluation up ``path``, adding and signing each edge's reward.

    Args:
        path: Nodes root-first, leaf-last.
        leaf_value: The leaf's value, in the leaf's ``to_play`` frame.
        discount: Per-edge discount.
        stats: Updated with every edge's Q (from the updated means), so
            selection normalises against the values the tree actually holds.
    """
    value = leaf_value
    for index in range(len(path) - 1, -1, -1):
        node = path[index]
        node.visit_count += 1
        node.value_sum += value
        if index == 0:
            break
        parent = path[index - 1]
        if stats is not None:
            stats.update(edge_q(parent, node, discount))
        value = perspective_sign(parent.to_play, node.to_play) * (node.reward + discount * value)
