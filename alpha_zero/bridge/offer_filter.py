"""What may be offered to the policy, given what this game turn's ledger rules out.

Split from ``agent_loop.py`` (GM-B7c-2 review round) for size. The loop asks
the policy to choose from a mask; a command the game refused this turn, a
choice already made, or an ``end_turn`` the game is blocking is still in that
mask, because the mod offers it again. :func:`sendable` takes those out
before the policy is asked, so its choices are not spent on them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from alpha_zero.bridge.turn_ledger import UNBLOCKING_ACTIONS, TurnLedger, action_of, is_end_turn

__all__ = ["sendable", "vetoed"]


def vetoed(ledger: TurnLedger, command: Any, unblocking_only: bool, others: bool) -> bool:
    """Whether ``command`` must not be sent now.

    Args:
        ledger: This game turn's ledger.
        command: The converted command.
        unblocking_only: Past the turn's cap: only commands that can answer
            what blocks the end of turn may go out.
        others: Whether something else is on offer (a blocked ``end_turn`` is
            only held back while it is).
    """
    if ledger.is_suppressed(command):
        return True
    if is_end_turn(command):
        return ledger.end_turn_withheld() and others
    return unblocking_only and action_of(command) not in UNBLOCKING_ACTIONS


def sendable(
    ledger: TurnLedger,
    convert: Callable[[int], Any],
    legal_mask: np.ndarray,
    unblocking_only: bool,
) -> tuple[np.ndarray, dict[int, Any]]:
    """The mask without what the ledger rules out, and the commands already converted.

    Only when the ledger rules anything out: otherwise the mask is returned as
    it is and nothing is converted.

    Args:
        ledger: This game turn's ledger.
        convert: Action index to command; raises for an index that cannot be sent.
        legal_mask: The mask built from the mod's offers.
        unblocking_only: See :func:`vetoed`.

    Returns:
        ``(mask, commands by action index)``. A blocked ``end_turn`` is left
        out while anything else remains.
    """
    if not (ledger.has_vetoes() or unblocking_only):
        return legal_mask, {}
    try:
        offered = [int(i) for i in np.flatnonzero(legal_mask)]
    except Exception:  # a mask that cannot be read is left to the policy
        return legal_mask, {}
    ready: dict[int, Any] = {}
    end_turns: list[int] = []
    mask = legal_mask.copy()
    for index in offered:
        try:
            command = convert(index)
        except Exception:  # not sendable; the policy need not be offered it
            mask[index] = 0
            continue
        if is_end_turn(command):
            end_turns.append(index)
            ready[index] = command
        elif vetoed(ledger, command, unblocking_only, True):
            mask[index] = 0
        else:
            ready[index] = command
    if ledger.end_turn_withheld() and len(ready) > len(end_turns):
        for index in end_turns:
            mask[index] = 0
            del ready[index]
    return mask, ready
