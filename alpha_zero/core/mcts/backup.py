"""Sign-correct value backup, shared by every MCTS variant.

Why this module exists
----------------------
A node's value is stored from the perspective of the player who is to move in
that node's state. Propagating a leaf evaluation to the root therefore flips the
sign only where the acting player *changes*.

Plain negamax -- ``value = -value`` on every edge -- assumes exactly one ply per
turn. MicroCiv does not work that way: ``MicroCivEnv`` advances
``current_player`` only inside ``_end_turn``, so a player moves each unit before
yielding, and most tree edges are *same-player* edges (measured: 71% of steps
over 200 seeded random actions). Under plain negamax the sign of a node's mean
value becomes a function of the depth at which its leaves were evaluated, and
``value_sum`` averages ``+v`` and ``-v`` estimates together.

``MCTSNode.backpropagate`` in :mod:`alpha_zero.core.mcts.base` was corrected for
this on 2026-08-25. The correction was never propagated to the other search
variants, which is what this module fixes: one implementation, called by all of
them, instead of five copies of a subtly wrong loop.

Backwards compatibility
-----------------------
A node whose ``to_play`` is ``None`` -- and any path whose leaf has no
``to_play`` -- falls back to per-ply alternation, i.e. exactly the previous
behaviour. Callers that never populate ``to_play`` are therefore unaffected, and
variants that cannot observe the acting player (MuZero searches a learned latent
model, where the environment is not available past the root) keep working
unchanged and say so at their call site.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "BackupNode",
    "backpropagate_path",
    "backup_edge_summary",
    "env_current_player",
    "log_backup_diagnostics",
    "perspective_sign",
    "should_track_to_play",
    "signed_value",
]


def env_current_player(env: object) -> int | None:
    """Return the environment's acting player, or ``None`` when it exposes none.

    Lives here rather than in :mod:`alpha_zero.core.mcts.base` because the
    ``gumbel`` and ``muzero`` variants deliberately depend on neither ``base``
    nor each other (CLAUDE.md 2026-08-30), and all of them need to read the
    acting player to populate ``to_play``. ``base`` re-exports it under its
    historical private name.

    Stub environments in tests often have no ``current_player``; those keep the
    historical per-ply alternation.
    """
    player = getattr(env, "current_player", None)
    return None if player is None else int(player)


@runtime_checkable
class BackupNode(Protocol):
    """The minimal surface :func:`backpropagate_path` needs from a tree node.

    Every MCTS node type in this package satisfies it structurally, so no
    variant has to inherit from a shared base class to use the shared backup.
    """

    visit_count: int
    value_sum: float
    to_play: int | None


def perspective_sign(parent_to_play: int | None, child_to_play: int | None) -> float:
    """Return the sign converting a child's value into the parent's perspective.

    Selection and backup must agree on this rule. Negating unconditionally --
    the plain-negamax habit -- pairs badly with a ``to_play``-aware backup:
    across a same-player edge it makes the selector prefer the child that
    *minimises* the mover's own value.

    Args:
        parent_to_play: Player to move in the parent's state, or ``None``.
        child_to_play: Player to move in the child's state, or ``None``.

    Returns:
        ``+1.0`` when both players are known and equal, ``-1.0`` otherwise.
        The unknown case returns ``-1.0`` so callers that never populate
        ``to_play`` keep the historical alternating behaviour.
    """
    same_player = (
        parent_to_play is not None and child_to_play is not None and child_to_play == parent_to_play
    )
    return 1.0 if same_player else -1.0


def signed_value(value: float, node_to_play: int | None, reference: int | None) -> float:
    """Return ``value`` signed for a node, given the perspective it was measured in.

    Args:
        value: The leaf evaluation, from ``reference``'s perspective.
        node_to_play: The player to move in the node's state, or ``None`` when
            the caller does not track it.
        reference: The player the evaluation is expressed from, or ``None``.

    Returns:
        ``value`` when the node shares the reference player's perspective, and
        ``-value`` when it does not. Returns ``value`` unchanged when either
        player is unknown; the alternation fallback is applied by the caller,
        which is the only place that knows the traversal order.
    """
    if reference is None or node_to_play is None:
        return value
    return value if node_to_play == reference else -value


def backpropagate_path(
    path: Sequence[BackupNode],
    leaf_value: float,
    *,
    reference: int | None = None,
) -> None:
    """Backpropagate ``leaf_value`` along ``path``, flipping only on a player change.

    Args:
        path: Nodes ordered root-first, leaf-last -- the ``search_path`` every
            variant in this package already builds. Typed as a ``Sequence``
            rather than a ``list`` because ``list`` is invariant: a
            ``list[GumbelMCTSNode]`` is not a ``list[BackupNode]``, and every
            caller passes its own concrete node type. Only node *attributes* are
            mutated here, never the sequence, so covariance is sound.
        leaf_value: Value of the leaf state, from the leaf player's perspective.
        reference: Player the value is expressed from. Defaults to the leaf
            node's ``to_play``, which is the convention every caller here uses:
            both the network evaluation and the terminal ``winner`` comparison
            are taken from the simulated environment's ``current_player`` at the
            leaf.

    Notes:
        Mirrors :meth:`alpha_zero.core.mcts.base.MCTSNode.backpropagate` exactly,
        including its alternation fallback, so delegating to it is behaviour
        preserving for callers that already populate ``to_play``.
    """
    if not path:
        return

    if reference is None:
        reference = getattr(path[-1], "to_play", None)

    # Two regimes, chosen once from the leaf rather than per node.
    #
    # When the leaf's player is unknown the caller is not tracking perspective at
    # all, so the whole path alternates -- exactly the historical behaviour, and
    # what MuZero (a learned latent model with no observable mover) still gets.
    #
    # When it IS known, each node is signed independently from its own player. A
    # per-node fallback that flipped a running value would make one unknown node
    # invert every node above it, so a partially-populated path would corrupt the
    # root rather than merely lose information about one node. Signing
    # independently keeps the damage local to the node that cannot be signed.
    if reference is None:
        value = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
        return

    for node in reversed(path):
        node_to_play = getattr(node, "to_play", None)
        if node_to_play is None:
            # Unknown for this node only: credit the leaf's own perspective
            # rather than guessing, and leave its ancestors untouched.
            signed = leaf_value
        else:
            signed = signed_value(leaf_value, node_to_play, reference)
        node.visit_count += 1
        node.value_sum += signed


def backup_edge_summary(path: Sequence[BackupNode]) -> tuple[int, int, int]:
    """Summarise a path's edges for diagnostics.

    Returns:
        ``(same_player_edges, player_change_edges, unknown_edges)``.

    This exists because the defect it guards against is invisible in aggregate
    metrics -- a parity-scrambled value still produces plausible-looking win
    rates. Call it at DEBUG level once per search, never per node: the backup
    loop is the hottest path in the searcher.
    """
    same = changed = unknown = 0
    for parent, child in itertools.pairwise(path):
        parent_player = getattr(parent, "to_play", None)
        child_player = getattr(child, "to_play", None)
        if parent_player is None or child_player is None:
            unknown += 1
        elif parent_player == child_player:
            same += 1
        else:
            changed += 1
    return same, changed, unknown


def log_backup_diagnostics(path: Sequence[BackupNode], *, search_name: str) -> None:
    """Emit a one-line DEBUG summary of a path's edge composition.

    Guarded on the logger's effective level so the ``zip`` in
    :func:`backup_edge_summary` is not paid for when DEBUG is off.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    same, changed, unknown = backup_edge_summary(path)
    logger.debug(
        "%s backup path: depth=%d same_player_edges=%d player_change_edges=%d unknown=%d",
        search_name,
        len(path),
        same,
        changed,
        unknown,
    )


def should_track_to_play(env: object) -> bool:
    """Whether sign-correct backup is valid for this environment.

    A scalar value flipped between two perspectives cannot express a game with
    more than two players: "not mine" is not the same as "the opponent's" once
    there are several opponents. Above two players the variants therefore leave
    ``to_play`` unset and fall back to per-ply alternation, which keeps
    >2-player behaviour exactly as it was before sign-correct backup existed.
    ``alpha_zero.multiplayer.mcts.NPlayerMCTS`` is the search that keeps a real
    per-player value vector.

    Args:
        env: The environment being searched.

    Returns:
        ``True`` when the environment has at most two players (or does not say).
    """
    return int(getattr(env, "num_players", 2) or 2) <= 2
