"""What the game did with the commands sent in the current game turn.

The agent loop sends several commands per game turn and gets a fresh state
after each. Without a memory it repeats itself: a command the mod refused is
offered again in the next state and a deterministic policy picks it again; an
``end_turn`` the game blocked is chosen again while the choice that would
unblock it sits untouched. :class:`TurnLedger` is that memory. It is pure
(no asyncio, no torch): the loop tells it what it saw, sent and was answered,
and asks it what not to send.

A command is remembered by what the mod was asked to do (:func:`command_key`),
never by its action index: an index is a unit's place in a list or an item's
place among the offered ones, and both move inside a turn.

"No expected effect" is a suspicion, not a verdict. It reads what the mod
exports two frames after a request; nobody has measured in a real game whether
that is long enough, and a change the export does not carry reads as none. A
command is therefore suppressed only after it has had no effect
``no_effect_limit`` times.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Any

from alpha_zero.bridge.protocol import END_TURN_BLOCKED
from alpha_zero.bridge.state_diff import expected_effect, summarize

logger = logging.getLogger(__name__)

__all__ = ["LedgerLimits", "Outcome", "TurnLedger", "command_key", "is_end_turn"]

_END_TURN = "end_turn"

#: Choices made once per game turn: research (per player) and what a city
#: builds (per city). The game never offers the item already in progress, so
#: "the best of what is offered" names a different item after every choice and
#: a deterministic policy would change its mind until the turn's cap.
_ONCE_PER_TURN = frozenset({"research_tech", "research_civic", "city_produce"})


def _as_dict(command: Any) -> dict[str, Any]:
    if isinstance(command, dict):
        return command
    try:
        return dict(command.to_dict())
    except Exception:  # not a command (a test double): it has no identity
        return {}


def command_key(command: Any) -> str:
    """The identity of a command: what the mod was asked to do.

    Args:
        command: A ``LuaCommand`` or a dict in the shape of its ``to_dict()``.

    Returns:
        A string equal for two commands that ask the same thing, whatever the
        key order, the metadata, or whether an absent field is missing or None.
    """
    data = _as_dict(command)
    target = data.get("target")
    if isinstance(target, dict):
        where = f"{target.get('x')},{target.get('y')}"
    elif isinstance(target, (tuple, list)) and len(target) == 2:
        where = f"{target[0]},{target[1]}"
    else:
        where = ""
    action = data.get("action_type")
    action = getattr(action, "value", action)
    item = data.get("item_id")
    parts = (
        str(action),
        "" if data.get("unit_id") is None else str(data["unit_id"]),
        "" if data.get("city_id") is None else str(data["city_id"]),
        where,
        "" if item is None else str(item),
        "" if data.get("item_type") is None else str(data["item_type"]),
    )
    return "|".join(parts)


def _choice_slot(command: Any) -> str | None:
    """The once-per-turn slot ``command`` fills, or None for any other command."""
    data = _as_dict(command)
    action = data.get("action_type")
    action = getattr(action, "value", action)
    if action not in _ONCE_PER_TURN:
        return None
    return f"{action}|{'' if data.get('city_id') is None else data['city_id']}"


def is_end_turn(command: Any) -> bool:
    """Whether ``command`` (a ``LuaCommand`` or its dict) ends the turn."""
    action = _as_dict(command).get("action_type")
    return bool(getattr(action, "value", action) == _END_TURN)


class Outcome(enum.Enum):
    """What the mod answered to one command."""

    CARRIED_OUT = "carried_out"
    BLOCKED_END_TURN = "blocked_end_turn"
    REFUSED = "refused"


@dataclass(frozen=True)
class LedgerLimits:
    """Bounds of one game turn.

    Attributes:
        max_commands_per_turn: Past this many commands only ``end_turn`` is sent.
        no_effect_limit: A command with no expected effect this many times is
            suppressed for the rest of the turn.
        max_blocked_end_turns: This many blocked ``end_turn`` answers in a row,
            with no change in the state between them, is a stall.
    """

    max_commands_per_turn: int = 100
    no_effect_limit: int = 2
    max_blocked_end_turns: int = 3

    def __post_init__(self) -> None:
        if self.max_commands_per_turn < 1:
            raise ValueError(
                f"max_commands_per_turn must be >= 1, got {self.max_commands_per_turn}"
            )
        if self.no_effect_limit < 1:
            raise ValueError(f"no_effect_limit must be >= 1, got {self.no_effect_limit}")
        if self.max_blocked_end_turns < 1:
            raise ValueError(
                f"max_blocked_end_turns must be >= 1, got {self.max_blocked_end_turns}"
            )


def _summary(state: Any) -> dict[str, Any] | None:
    """The comparable part of ``state``; None for one that cannot be summarised."""
    try:
        return summarize(state)
    except Exception:  # a test double, or a state too broken to compare
        return None


class TurnLedger:
    """Per-game-turn memory of commands sent and how the game answered."""

    def __init__(self, limits: LedgerLimits | None = None) -> None:
        self.limits = limits if limits is not None else LedgerLimits()
        self.turn_id: Any = None
        self.game_turns_finished = 0
        self.commands_this_turn = 0
        self._seen_a_state = False
        self._refused: dict[str, str | None] = {}
        self._no_effect: dict[str, int] = {}
        self._chosen: set[str] = set()
        self._blocked_streak = 0
        self._withheld_on: dict[str, Any] | None = None
        self._end_turn_withheld = False
        self._last_summary: dict[str, Any] | None = None
        # (command, the state it was sent on, whether the mod answered it)
        self._pending: tuple[dict[str, Any], Any, bool] | None = None
        self._refused_total = 0
        self._no_effect_total = 0
        self._blocked_total = 0

    # --- what the loop reports -----------------------------------------
    def observe(self, state: Any) -> None:
        """Take in a state received from the game, before a command is chosen."""
        turn_id = getattr(state, "turn_id", None)
        summary = _summary(state)
        if self._seen_a_state and turn_id != self.turn_id:
            self.game_turns_finished += 1
            self._new_turn()
        elif self._seen_a_state:
            self._judge_pending(state)
            if summary is not None and summary != self._last_summary:
                # Something moved: whatever blocked the end of turn may be gone.
                self._blocked_streak = 0
            if self._end_turn_withheld and summary is not None and summary != self._withheld_on:
                self._end_turn_withheld = False
        self._seen_a_state = True
        self.turn_id = turn_id
        self._last_summary = summary
        self._pending = None

    def sent(self, command: Any, state: Any) -> None:
        """A command went out, chosen on ``state``."""
        self.commands_this_turn += 1
        self._pending = (_as_dict(command), state, False)

    def answered(self, command: Any, ack: Any) -> Outcome:
        """The mod's answer to ``command``."""
        data = _as_dict(command)
        if bool(getattr(ack, "success", False)):
            if not is_end_turn(data):
                self._blocked_streak = 0
            if self._pending is not None:
                self._pending = (self._pending[0], self._pending[1], True)
            slot = _choice_slot(data)
            if slot is not None:
                self._chosen.add(slot)
            return Outcome.CARRIED_OUT

        self._pending = None
        code = _command_code(ack)
        if code == END_TURN_BLOCKED:
            self._record_blocked()
            return Outcome.BLOCKED_END_TURN
        if is_end_turn(data):
            # end_turn is never suppressed; a refusal of it is only counted.
            self._refused_total += 1
            return Outcome.REFUSED
        self._refused[command_key(data)] = code
        self._refused_total += 1
        return Outcome.REFUSED

    # --- what the loop asks ----------------------------------------------
    def is_suppressed(self, command: Any) -> bool:
        """Whether ``command`` should not be sent again in this game turn."""
        if is_end_turn(command):
            return False
        if _choice_slot(command) in self._chosen:
            return True  # research, or this city's production, was chosen this turn
        key = command_key(command)
        return key in self._refused or self._no_effect.get(key, 0) >= self.limits.no_effect_limit

    def end_turn_withheld(self) -> bool:
        """Whether the game blocked ``end_turn`` and nothing has changed since."""
        return self._end_turn_withheld

    def over_cap(self) -> bool:
        """Whether this game turn has used up its commands."""
        return self.commands_this_turn >= self.limits.max_commands_per_turn

    def stalled(self) -> bool:
        """Whether the turn can neither be ended nor, it seems, unblocked."""
        return self._blocked_streak >= self.limits.max_blocked_end_turns

    def stats(self) -> dict[str, int]:
        """Counters for the session."""
        return {
            "game_turns_played": self.game_turns_finished,
            "refused_commands": self._refused_total,
            "no_effect_commands": self._no_effect_total,
            "blocked_end_turns": self._blocked_total,
        }

    # --- internals --------------------------------------------------------
    def _new_turn(self) -> None:
        self.commands_this_turn = 0
        self._refused.clear()
        self._no_effect.clear()
        self._chosen.clear()
        self._blocked_streak = 0
        self._end_turn_withheld = False
        self._withheld_on = None

    def _record_blocked(self) -> None:
        self._blocked_streak += 1
        self._blocked_total += 1
        self._end_turn_withheld = True
        self._withheld_on = self._last_summary

    def _judge_pending(self, state: Any) -> None:
        if self._pending is None:
            return
        command, sent_on, was_answered = self._pending
        if is_end_turn(command):
            blocking = getattr(state, "end_turn_blocking", None)
            if not was_answered and isinstance(blocking, str) and blocking:
                # The fallback end_turn is sent without reading its answer;
                # the same turn coming back with a blocker is that answer.
                self._record_blocked()
            return
        try:
            verdict = expected_effect(command, sent_on, state)
        except Exception:  # a state that cannot be compared is not judged
            return
        if verdict is False:
            # A choice the game did not take may be made again.
            self._chosen.discard(_choice_slot(command) or "")
            key = command_key(command)
            self._no_effect[key] = self._no_effect.get(key, 0) + 1
            self._no_effect_total += 1
            logger.warning(
                "Command had no visible effect (%d of %d before it is left alone this turn): %s",
                self._no_effect[key],
                self.limits.no_effect_limit,
                key,
            )


def _command_code(ack: Any) -> str | None:
    """The error code of the failed command in ``ack``, else the ack's own."""
    executed = getattr(ack, "executed", None)
    if isinstance(executed, list):
        for entry in executed:
            if (
                isinstance(entry, dict)
                and entry.get("success") is False
                and entry.get("error_code")
            ):
                return str(entry["error_code"])
        for entry in executed:
            if isinstance(entry, dict) and entry.get("error_code"):
                return str(entry["error_code"])
    code = getattr(ack, "error_code", None)
    return code if isinstance(code, str) else None
