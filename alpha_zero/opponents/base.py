"""Opponent Protocol — uniform interface for anything that picks actions.

This is intentionally minimal: the contract is one method, ``get_action``,
plus lifecycle hooks for engines that hold external resources (UCI
subprocesses, GPU sessions, sockets).

Unlike :class:`alpha_zero.multiplayer.opponent_model.OpponentModel`, which
*predicts* what an opponent will do from observed history, this protocol
*plays* — it's used in self-play, evaluation, and tournament harnesses
where one player slot is filled by a non-learning entity (Stockfish, a
frozen checkpoint, a rule-based bot, etc.).

Because the protocol is :func:`runtime_checkable`, callers can use
``isinstance(obj, Opponent)`` for duck-type validation in tests.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Opponent(Protocol):
    """Anything that selects an action for a given environment state.

    Implementations must respect the action mask supplied by the environment
    and may consult ``env`` for richer context (e.g., a chess opponent reads
    ``env.board``). The action returned must be a valid index — i.e.,
    ``action_mask[action] > 0`` — or implementations must raise.
    """

    #: Human-readable identifier used by logs, tournament tables, and config.
    name: str

    def get_action(self, env: Any, action_mask: np.ndarray) -> int:
        """Select an action index for the current state of ``env``.

        Args:
            env: The environment instance (e.g. ``ChessEnv``). Implementations
                may rely on env-specific attributes such as ``env.board`` for
                rich state; generic implementations should treat ``env`` as
                opaque and use only ``action_mask``.
            action_mask: 1.0 at legal action indices, 0.0 elsewhere.

        Returns:
            The chosen action index.
        """
        ...

    def reset(self) -> None:
        """Reset any per-episode internal state. No-op by default."""
        ...

    def close(self) -> None:
        """Release external resources (subprocesses, sockets, etc.). Idempotent."""
        ...
