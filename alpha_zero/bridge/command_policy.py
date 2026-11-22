"""Policies that answer with a command, and how the agent loop asks them (GM-B6b).

``PolicyEngine`` answers with an action index in the bridge's own action
space. A search that plays in the simulator cannot: its answer has to be
translated (``bridge/search_translator.py``), and what comes out is a ready
command. ``CommandPolicy`` is the protocol for that, and :func:`choose` is the
one place the loop asks it: it hands over the turn ledger's veto and a seed for
this decision, and reports a policy that named nothing or failed instead of
letting either pass as an ``end_turn``.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from alpha_zero.bridge.offer_filter import vetoed
from alpha_zero.bridge.protocol import LuaCommand
from alpha_zero.bridge.turn_ledger import TurnLedger
from alpha_zero.utils.seeding import derive_seed

logger = logging.getLogger(__name__)

SEED_STREAM = "bridge_search"


class CommandPolicy(Protocol):
    """A policy whose answer is a live command."""

    async def select_command(
        self,
        converted_state: dict[str, Any],
        *,
        allowed: Callable[[dict[str, Any]], bool] | None = None,
        decision_seed: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """``(command, record)``; ``record["fallback"]`` says why when nothing was chosen."""
        ...


def is_command_policy(policy: Any) -> bool:
    """Whether ``policy`` answers with a command.

    ``select_command`` has to be a coroutine function defined on the policy's
    class or set on the instance itself: a mock answers "yes" to any attribute,
    and a mocked index policy must not be taken for a command policy. A wrapper
    that only forwards attributes (``__getattr__``) is therefore not one; it
    has to define ``select_command``.
    """
    own = getattr(type(policy), "select_command", None)
    if own is None:
        own = getattr(policy, "__dict__", {}).get("select_command")
    return inspect.iscoroutinefunction(own)


def allowed_by(ledger: TurnLedger, unblocking_only: bool) -> Callable[[dict[str, Any]], bool]:
    """The turn ledger as a veto on commands: False for what must not be sent now."""

    def allowed(command: dict[str, Any]) -> bool:
        return not vetoed(ledger, command, unblocking_only, True)

    return allowed


def decision_seed(base_seed: int | None, turn_id: Any, command_number: int) -> int | None:
    """The seed of one decision: a game turn and the command's number in it."""
    return derive_seed(base_seed, SEED_STREAM, turn_id, command_number)


@dataclass(frozen=True)
class Choice:
    """What a command policy answered.

    Attributes:
        command: The command to send, or None when the policy named nothing
            that may be sent.
        record: The policy's record of the decision (empty when it raised).
        failed: The policy raised.
        reason: Why ``command`` is None; None otherwise.
    """

    command: LuaCommand | None
    record: dict[str, Any] = field(default_factory=dict)
    failed: bool = False
    reason: str | None = None


def _base_seed(policy: Any) -> int | None:
    seed = getattr(getattr(policy, "config", None), "seed", None)
    return seed if isinstance(seed, int) and not isinstance(seed, bool) else None


async def choose(
    policy: Any,
    ledger: TurnLedger,
    converted_state: dict[str, Any],
    *,
    unblocking_only: bool,
    turn_id: Any,
) -> Choice:
    """Ask a command policy for this state's command. Never raises."""
    allowed = allowed_by(ledger, unblocking_only)
    seed = decision_seed(_base_seed(policy), turn_id, ledger.commands_this_turn)
    try:
        command, record = await policy.select_command(
            converted_state, allowed=allowed, decision_seed=seed
        )
    except Exception as exc:
        logger.error("Command policy failed: %s", exc, exc_info=True)
        return Choice(None, {}, failed=True, reason=f"policy: {exc}")
    record = record if isinstance(record, dict) else {}
    fallback = record.get("fallback")
    if fallback:
        return Choice(None, record, reason=str(fallback))
    try:
        lua_command = LuaCommand.from_dict(command)
    except Exception as exc:
        return Choice(None, record, reason=f"not a command: {exc}")
    try:
        if not allowed(command):
            return Choice(None, record, reason="ruled out for this turn")
    except Exception as exc:
        return Choice(None, record, reason=f"could not be checked: {exc}")
    return Choice(lua_command, record)


def diagnostics_of(choice: Choice) -> dict[str, float]:
    """A decision as the numbers the eval harness logs."""
    try:
        value = float(choice.record.get("root_value", 0.0))
    except (TypeError, ValueError):
        value = 0.0
    return {"value_estimate": value, "policy_kind": 1.0}
