"""The mod keeps working when the game raises no event batch (an idle game).

The mod polled only on ``GameCoreEventPublishComplete``, which the game raises
after a batch of game events. After a request that produces one batch and then
nothing, the two-batch wait before the next state never ended: no state, and
no further command was read. The mod now also counts frames
(``ContextPtr:SetUpdate``). The stand-in plays the two apart: ``tick`` is a
frame with an event batch, ``Civ6Stub.idleFrames(n)`` is frames with none.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import lua_commands as c

SKIP = {"action_type": "unit_skip", "unit_id": 102}
MAX_FRAMES = 30


def idle(runtime: Any, count: int = 1) -> None:
    runtime.execute(f"Civ6Stub.idleFrames({int(count)})")


def test_state_after_a_command_arrives_in_an_idle_game() -> None:
    rt = c.loaded_turn()
    before = len(c.states(rt))
    c.send(rt, [SKIP], msg_id=1)
    idle(rt, 1)  # the command is read and answered on a frame
    assert [a["msg_id"] for a in c.acks(rt)] == [1]
    assert len(c.states(rt)) == before  # not yet: the wait has begun
    idle(rt, MAX_FRAMES)
    assert len(c.states(rt)) == before + 1


def test_the_wait_in_frames_is_not_cut_short() -> None:
    rt = c.loaded_turn()
    before = len(c.states(rt))
    c.send(rt, [SKIP], msg_id=1)
    idle(rt, 1)
    idle(rt, MAX_FRAMES - 1)
    assert len(c.states(rt)) == before
    idle(rt, 1)
    assert len(c.states(rt)) == before + 1


def test_event_batches_still_end_the_wait_after_two() -> None:
    rt = c.loaded_turn()
    before = len(c.states(rt))
    c.send(rt, [SKIP], msg_id=1)
    c.tick(rt)  # read and answered
    c.tick(rt)
    assert len(c.states(rt)) == before
    c.tick(rt)
    assert len(c.states(rt)) == before + 1


def test_one_event_batch_then_idle_still_exports() -> None:
    """The reported case: one batch after the request, then nothing."""
    rt = c.loaded_turn()
    before = len(c.states(rt))
    c.send(rt, [SKIP], msg_id=1)
    c.tick(rt)
    c.tick(rt)  # one batch of the two the wait asks for
    assert len(c.states(rt)) == before
    idle(rt, MAX_FRAMES)
    assert len(c.states(rt)) == before + 1


def test_no_command_is_read_during_the_wait_on_frames() -> None:
    rt = c.loaded_turn()
    c.send(rt, [SKIP], msg_id=1)
    idle(rt, 1)
    c.send(rt, [{"action_type": "unit_fortify", "unit_id": 102}], msg_id=2)
    idle(rt, MAX_FRAMES - 1)
    assert [a["msg_id"] for a in c.acks(rt)] == [1]
    idle(rt, 2)  # the state goes out, then the next command is read
    assert [a["msg_id"] for a in c.acks(rt)] == [1, 2]


def test_a_command_is_read_in_an_idle_game_after_the_state() -> None:
    rt = c.loaded_turn()
    c.send(rt, [SKIP], msg_id=1)
    idle(rt, 1 + MAX_FRAMES)
    c.send(rt, [{"action_type": "end_turn"}], msg_id=2)
    idle(rt, 1)
    assert [a["msg_id"] for a in c.acks(rt)] == [1, 2]


def test_the_frame_limit_is_configurable() -> None:
    rt = c.loaded_turn(extra='AlphaZeroConfig.Set("STATE_EXPORT_MAX_FRAMES", 3)')
    before = len(c.states(rt))
    c.send(rt, [SKIP], msg_id=1)
    idle(rt, 1 + 2)
    assert len(c.states(rt)) == before
    idle(rt, 1)
    assert len(c.states(rt)) == before + 1


def test_a_context_without_frame_updates_works_on_events_alone() -> None:
    rt = c.loaded_turn(setup="ContextPtr = nil\n")
    before = len(c.states(rt))
    c.send(rt, [SKIP], msg_id=1)
    c.tick(rt, 3)
    assert len(c.states(rt)) == before + 1


def test_the_frame_update_is_cleared_when_the_turn_ends() -> None:
    rt = c.loaded_turn()
    assert bool(rt.eval("Civ6Stub.frameUpdate ~= nil"))
    rt.execute("AlphaZeroBridge.OnTurnEnd(0)")
    assert not bool(rt.eval("Civ6Stub.frameUpdate ~= nil"))


@pytest.mark.parametrize("frames", [0, -5])
def test_a_frame_limit_below_one_means_one(frames: int) -> None:
    rt = c.loaded_turn(extra=f'AlphaZeroConfig.Set("STATE_EXPORT_MAX_FRAMES", {frames})')
    before = len(c.states(rt))
    c.send(rt, [SKIP], msg_id=1)
    idle(rt, 2)
    assert len(c.states(rt)) == before + 1
