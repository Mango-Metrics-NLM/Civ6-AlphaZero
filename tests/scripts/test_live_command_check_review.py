"""A command given with an id written as text is judged as it is sent.

``LuaCommand.from_dict`` turns ``"unit_id": "1"`` into the number 1 before the
command goes out, and the states carry numbers. Judging the JSON as typed made
a successful command read as "no effect" (exit 1).
"""

from __future__ import annotations

from typing import Any

from tests.scripts.test_live_command_check import MOVE, FakeBridge, _run, make_state


async def test_a_text_id_is_judged_as_the_number_that_is_sent() -> None:
    as_typed: dict[str, Any] = {**MOVE, "unit_id": str(MOVE["unit_id"])}
    bridge = FakeBridge(current=make_state(), after=make_state(units=[(1, 3, 3, 1)]))
    code, lines = await _run(bridge, as_typed)
    assert [c.to_dict() for c in bridge.sent[0].commands] == [MOVE]
    assert code == 0, lines


async def test_control_the_same_command_with_a_number_exits_0() -> None:
    bridge = FakeBridge(current=make_state(), after=make_state(units=[(1, 3, 3, 1)]))
    code, _ = await _run(bridge, MOVE)
    assert code == 0
