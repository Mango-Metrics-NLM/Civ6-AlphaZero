"""GM-B9: the command cap and a game that cannot be asked what blocks the turn.

Past ``MAX_COMMANDS_PER_TURN`` only ``end_turn`` is carried out, except that while the game blocks
the end of turn the commands that can answer the blocker are still carried out, up to twice the
cap (otherwise the cap would trap the turn). When the game cannot be asked (no
``NotificationManager``) ``end_turn`` is refused as ``ENDTURN_BLOCKING_NOT_ASKABLE``, so the same
exception must apply: the turn is just as trapped. With the manager present and nothing blocking
the cap holds.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import lua_commands as c

CAP = 3
SKIP = {"action_type": "unit_skip", "unit_id": 102}
RESEARCH_WRITING = {"action_type": "research_tech", "item_id": 2}  # TECH_WRITING, index 2
MOVE = c.unit_move(102, 0, 1)
CAP_REACHED = "COMMAND_CAP_REACHED"
CANNOT_ASK = "NotificationManager = nil\n"


def _exchange(runtime: Any, commands: list[dict[str, Any]], msg_id: int) -> dict[str, Any]:
    c.send(runtime, commands, msg_id)
    c.tick(runtime, 5)
    return c.acks(runtime)[-1]


def _at_the_cap(setup: str) -> Any:
    runtime = c.loaded_turn(setup)
    runtime.execute(f"AlphaZeroConfig.Set('MAX_COMMANDS_PER_TURN', {CAP})")
    for msg_id in (1,):
        assert _exchange(runtime, [SKIP] * CAP, msg_id)["success"] is True, "premise"
    assert runtime.eval("Civ6Stub.players[0].tech.current") == 1, "premise: researching mining"
    return runtime


def _researching(runtime: Any) -> int:
    return int(runtime.eval("Civ6Stub.players[0].tech.current"))


class TestGameCannotBeAsked:
    def test_an_unblocking_command_is_still_carried_out_and_shows_in_the_game(self) -> None:
        runtime = _at_the_cap(CANNOT_ASK)
        ack = _exchange(runtime, [RESEARCH_WRITING], 2)
        assert ack.get("error_code") != CAP_REACHED
        assert ack["success"] is True
        assert _researching(runtime) == 2

    def test_the_allowance_runs_to_twice_the_cap_and_no_further(self) -> None:
        runtime = _at_the_cap(CANNOT_ASK)
        for msg_id in range(2, 2 + CAP):  # handled 3 -> 4, 5, 6: up to twice the cap
            assert _exchange(runtime, [RESEARCH_WRITING], msg_id).get("error_code") != CAP_REACHED
        runtime.execute("Civ6Stub.players[0].tech.current = 1")
        ack = _exchange(runtime, [RESEARCH_WRITING], 2 + CAP)  # 7 > 6
        assert (ack["success"], ack["error_code"]) == (False, CAP_REACHED)
        assert _researching(runtime) == 1

    def test_end_turn_is_refused_as_unknown_so_the_exception_is_needed(self) -> None:
        runtime = _at_the_cap(CANNOT_ASK)
        ack = _exchange(runtime, [{"action_type": "end_turn"}], 2)
        assert ack["success"] is False
        assert "ENDTURN_BLOCKING_NOT_ASKABLE" in str(ack)
        assert runtime.eval("Civ6Stub.turnEnded") is False
        assert runtime.eval("Civ6Stub.turnDeactivated") is False

    def test_a_command_that_cannot_answer_a_blocker_is_refused_past_the_cap(self) -> None:
        runtime = _at_the_cap(CANNOT_ASK)
        ack = _exchange(runtime, [MOVE], 2)
        assert ack["error_code"] == CAP_REACHED
        assert (runtime.eval("Civ6Stub.unit(102).x"), runtime.eval("Civ6Stub.unit(102).y")) == (
            1,
            0,
        )


class TestControlTheGameCanBeAskedAndNothingBlocks:
    def test_the_same_command_is_refused_past_the_cap(self) -> None:
        runtime = _at_the_cap("")
        ack = _exchange(runtime, [RESEARCH_WRITING], 2)
        assert (ack["success"], ack["error_code"]) == (False, CAP_REACHED)
        assert _researching(runtime) == 1

    def test_a_command_that_cannot_answer_a_blocker_is_refused_past_the_cap(self) -> None:
        runtime = _at_the_cap("")
        assert _exchange(runtime, [MOVE], 2)["error_code"] == CAP_REACHED
        assert runtime.eval("Civ6Stub.unit(102).x") == 1


@pytest.mark.parametrize("setup", [CANNOT_ASK, ""], ids=["cannot ask", "nothing blocks"])
def test_below_the_cap_everything_is_carried_out(setup: str) -> None:
    runtime = c.loaded_turn(setup)
    runtime.execute(f"AlphaZeroConfig.Set('MAX_COMMANDS_PER_TURN', {CAP})")
    assert _exchange(runtime, [MOVE], 1)["success"] is True
