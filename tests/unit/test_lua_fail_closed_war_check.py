"""GM-B9 / B2: a war check the mod cannot read refuses the move or attack.

``CombatManager.IsAttackChangeWarState`` answers nil or an empty table (no war), or a non-empty
table (war). Any other answer (``true``, a number, a string) is not the game's documented shape:
the move or attack is refused with ``EXECUTION_FAILED``, the message says the answer could not
be read, nothing is requested, and the action is not offered in the exported state.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import gameinfo_snapshot as g
from tests.fixtures import lua_commands as c
from tests.fixtures import lua_harness as h
from tests.fixtures import lua_others as o

MOVE = c.unit_move(102, 0, 1)
ATTACK = o.attack_command("unit_attack")
RANGED = o.attack_command("unit_ranged_attack")
COMMANDS = {"move": MOVE, "attack": ATTACK, "ranged": RANGED}
#: Where each command points, and the offered action it must not appear in.
OFFERED = {
    "move": ("move", (0, 1)),
    "attack": ("attack", (2, 0)),
    "ranged": ("ranged_attack", (2, 0)),
}


def _answer(expression: str) -> str:
    return c.ARCHER + f"CombatManager.IsAttackChangeWarState = function() return {expression} end\n"


# GM-B9 (B4): attacks are offered only with OBSERVE_ALL_PLAYERS on, so the "is it offered" halves
# run with it on by default; the executor is not affected by the flag.
def _runtime(expression: str, *, observe: bool = True) -> Any:
    runtime = h.load_mod(setup=_answer(expression))
    if observe:
        runtime.execute(o.OBSERVE)
    return runtime


def _offered_to(state: dict[str, Any], kind: str) -> bool:
    action, tile = OFFERED[kind]
    for entry in state["legal_actions"]:
        if entry.get("unit_id") == 102 and entry.get("action") == action:
            if tile in {(t[0], t[1]) for t in entry.get("targets", [])}:
                return True
    return False


def _nothing_requested(runtime: Any) -> bool:
    return (
        c.queued(runtime) == 0
        and not c.requests(runtime, "UnitManager.RequestOperation")
        and not c.requests(runtime, "UnitManager.RequestCommand")
    )


def _war_refused(kind: str, result: tuple[Any, Any, Any]) -> bool:
    """A war answer refuses with WOULD_START_WAR; a move onto a tile with nothing seen on it
    answers the blind OPERATION_REFUSED instead (GM-B8a: the refusal must not give it away)."""
    allowed = {"WOULD_START_WAR"} | ({"OPERATION_REFUSED"} if kind == "move" else set())
    return result[0] is False and result[1] in allowed


UNREADABLE = ["true", "false", "1", "42", "'war'", "'peace'"]


@pytest.mark.parametrize("kind", list(COMMANDS))
@pytest.mark.parametrize("expression", UNREADABLE)
class TestUnreadableAnswer:
    def test_the_command_is_refused_naming_the_unreadable_answer(
        self, kind: str, expression: str
    ) -> None:
        runtime = _runtime(expression)
        success, code, message = h.execute(runtime, COMMANDS[kind])
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert "could not be read" in str(message).lower()

    def test_nothing_is_requested_and_nothing_moves_or_dies(
        self, kind: str, expression: str
    ) -> None:
        runtime = _runtime(expression)
        h.execute(runtime, COMMANDS[kind])
        c.frame(runtime)
        assert _nothing_requested(runtime)
        assert runtime.eval("Civ6Stub.unit(102).x") == 1
        assert runtime.eval("Civ6Stub.unit(102).y") == 0
        assert not runtime.eval("Civ6Stub.unit(201).damage")

    def test_the_action_is_not_offered(self, kind: str, expression: str) -> None:
        state = h.export_state(_runtime(expression))
        assert state["legal_actions"], "premise: a state with actions was exported"
        assert not _offered_to(state, kind)


@pytest.mark.parametrize("kind", list(COMMANDS))
class TestReadableAnswers:
    """The documented shapes keep their meaning (these pass today)."""

    @pytest.mark.parametrize("expression", ["nil", "{}"])
    def test_no_war_goes_through_and_is_offered(self, kind: str, expression: str) -> None:
        runtime = _runtime(expression)
        assert _offered_to(h.export_state(runtime), kind)
        assert h.execute(runtime, COMMANDS[kind])[:2] == (True, None)
        assert not _nothing_requested(runtime)

    @pytest.mark.parametrize("expression", ["nil", "{}"])
    def test_with_the_flag_off_an_attack_is_not_offered_but_still_goes_through(
        self, kind: str, expression: str
    ) -> None:
        runtime = _runtime(expression, observe=False)
        offered = _offered_to(h.export_state(runtime), kind)
        assert offered is (kind == "move")  # GM-B9: a move is offered, an attack is not
        assert h.execute(runtime, COMMANDS[kind])[:2] == (True, None)

    def test_a_non_empty_list_is_a_war_and_is_refused_and_not_offered(self, kind: str) -> None:
        runtime = _runtime("{ 1 }")
        assert not _offered_to(h.export_state(runtime), kind)
        assert _war_refused(kind, h.execute(runtime, COMMANDS[kind]))
        assert _nothing_requested(runtime)


@pytest.mark.parametrize("kind", list(COMMANDS))
def test_a_non_empty_table_without_list_entries_is_a_war_too(kind: str) -> None:
    # The stand-in's header documents the answer as "{} or { defender }": a table with a key
    # and no list part is non-empty. The mod counts list entries only (``#results > 0``), so it
    # reads this as "no war". The contract says "non-empty table: WOULD_START_WAR".
    runtime = _runtime("{ defender = 1 }")
    assert _war_refused(kind, h.execute(runtime, COMMANDS[kind]))
    assert _nothing_requested(runtime)
    assert not _offered_to(h.export_state(runtime), kind)


WARNING = "[AlphaZero] WARN: IsAttackChangeWarState answered"


def _warnings(runtime: Any) -> list[str]:
    return [line for line in g.printed(runtime) if line.startswith(WARNING)]


class TestTheUnreadableAnswerIsSaidOnce:
    """M4: ``print`` (not the log: the offer check runs with it off), once per session."""

    @pytest.mark.parametrize("expression", UNREADABLE)
    def test_one_line_across_several_exports_and_commands(self, expression: str) -> None:
        runtime = g.load_printing(_answer(expression))
        runtime.execute(o.OBSERVE)
        for _ in range(3):
            h.export_state(runtime)
        for command in COMMANDS.values():
            h.execute(runtime, command)
        h.export_state(runtime)
        lines = _warnings(runtime)
        assert len(lines) == 1, lines
        assert "answered" in lines[0]

    def test_the_line_says_what_the_game_answered(self) -> None:
        runtime = g.load_printing(_answer("'peace'"))
        runtime.execute(o.OBSERVE)
        h.export_state(runtime)
        (line,) = _warnings(runtime)
        assert "string" in line and "peace" in line

    @pytest.mark.parametrize("expression", ["nil", "{}", "{ 1 }"])
    def test_a_readable_answer_prints_nothing(self, expression: str) -> None:
        runtime = g.load_printing(_answer(expression))
        runtime.execute(o.OBSERVE)
        h.export_state(runtime)
        for command in COMMANDS.values():
            h.execute(runtime, command)
        assert _warnings(runtime) == []

    def test_a_new_session_says_it_again(self) -> None:
        first = g.load_printing(_answer("true"))
        h.export_state(first)
        second = g.load_printing(_answer("true"))
        h.export_state(second)
        assert len(_warnings(first)) == len(_warnings(second)) == 1
