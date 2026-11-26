"""GM-B7a (M1-M3): the mod's main loop, driven through its own entry points.

``Civ6Stub.tick()`` is the game's frame: it carries out the requests made in earlier frames and
then publishes ``GameCoreEventPublishComplete``, the event the mod polls on. A command message
is the JSON text in ``ExposedMembers.AlphaZeroCommand``; the states and acknowledgements the
mod writes (no file access in the stand-in) are read back from the lines it prints.

Tests marked CONTROL pass on the tree before the fix.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import lua_commands as c

SKIP = {"action_type": "unit_skip", "unit_id": 102}
END_TURN = {"action_type": "end_turn"}
ENEMY_MOVE = c.unit_move(102, 2, 0)  # refused: a move never attacks
PRODUCTION_BLOCKER = (
    "Civ6Stub.notifications[0] = { { id = 7,"
    " blocking = EndTurnBlockingTypes.ENDTURN_BLOCKING_PRODUCTION,"
    " message = 'Choose production', typeName = 'NOTIFICATION_PRODUCTION', dismissed = false } }"
)


def exports(runtime: Any) -> int:
    return len(c.states(runtime))


def end_turns_requested(runtime: Any) -> int:
    return int(
        runtime.eval(
            "(function() local n = 0 for _, call in ipairs(Civ6Stub.calls) do"
            " if call.name == 'UI.RequestAction' then n = n + 1 end end return n end)()"
        )
    )


def exports_per_frame(runtime: Any, frames: int) -> list[int]:
    """New exports after each of ``frames`` frames, counted from now."""
    base = exports(runtime)
    seen = []
    for _ in range(frames):
        c.tick(runtime)
        seen.append(exports(runtime) - base)
    return seen


def exchange(
    runtime: Any, commands: list[dict[str, Any]], msg_id: int, at: int | None = None
) -> None:
    """One message at clock ``at`` (seconds into the turn), then enough frames to settle."""
    if at is not None:
        c.at(runtime, at)
    c.send(runtime, commands, msg_id)
    c.tick(runtime, 5)


@pytest.fixture
def turn() -> Any:
    return c.loaded_turn()


# ---------------------------------------------------------------------------
# Config keys
# ---------------------------------------------------------------------------
class TestConfigKeys:
    @pytest.mark.parametrize(
        ("key", "value"),
        [("STATE_EXPORT_DELAY_TICKS", 2), ("MAX_COMMANDS_PER_TURN", 200), ("COMMAND_TIMEOUT", 30)],
    )
    def test_the_defaults(self, turn: Any, key: str, value: int) -> None:
        assert turn.eval(f"AlphaZeroConfig.Get('{key}')") == value


# ---------------------------------------------------------------------------
# M1: the next state is exported on a later frame
# ---------------------------------------------------------------------------
class TestExportComesAfterTheGameHasActed:
    def test_the_state_after_a_move_is_exported_two_frames_later_and_shows_the_move(
        self, turn: Any
    ) -> None:
        c.send(turn, [c.unit_move(102, 0, 1)])
        assert exports_per_frame(turn, 5) == [0, 0, 1, 1, 1]
        warrior = c.unit_in(c.states(turn)[-1], 102)
        assert (warrior["x"], warrior["y"]) == (0, 1)

    @pytest.mark.parametrize("delay", [1, 4])
    def test_the_delay_is_the_configured_number_of_frames(self, turn: Any, delay: int) -> None:
        turn.execute(f"AlphaZeroConfig.Set('STATE_EXPORT_DELAY_TICKS', {delay})")
        c.send(turn, [c.unit_move(102, 0, 1)])
        expected = [0] * delay + [1, 1]
        assert exports_per_frame(turn, delay + 2) == expected

    def test_a_message_set_while_an_export_is_pending_is_not_read_until_after_it(
        self, turn: Any
    ) -> None:
        c.send(turn, [c.unit_move(102, 0, 1)], msg_id=1)
        base = exports(turn)
        c.tick(turn)  # frame 1: the message is processed
        c.send(turn, [SKIP], msg_id=2)
        c.tick(turn)  # frame 2: export pending
        assert c.pending(turn), "a command was read while an export was pending"
        assert [a["msg_id"] for a in c.acks(turn)] == [1]
        c.tick(turn, 2)  # frames 3 and 4: the export, then the second message
        assert exports(turn) - base >= 1
        assert not c.pending(turn)
        assert [a["msg_id"] for a in c.acks(turn)] == [1, 2]
        c.tick(turn, 6)
        assert exports(turn) - base == 2, "one export per processed message"
        assert c.unit_in(c.states(turn)[-1], 102)["x"] == 0
        assert turn.eval("Civ6Stub.unit(102).skipped") is True

    def test_a_message_whose_commands_were_all_refused_is_followed_by_an_export(
        self, turn: Any
    ) -> None:
        c.send(turn, [ENEMY_MOVE])
        assert exports_per_frame(turn, 4) == [0, 0, 1, 1]
        (ack,) = c.acks(turn)
        assert ack["executed"][0]["success"] is False
        assert ack["executed"][0]["error_code"] == "OPERATION_REFUSED"

    def test_control_a_successful_end_turn_is_followed_by_no_export(self, turn: Any) -> None:
        c.send(turn, [END_TURN])
        assert exports_per_frame(turn, 6) == [0] * 6
        assert end_turns_requested(turn) == 1
        assert c.acks(turn)[0]["executed"][0]["success"] is True

    def test_a_refused_end_turn_is_followed_by_an_export_on_the_later_frame(self) -> None:
        game = c.loaded_turn(extra=PRODUCTION_BLOCKER)
        c.send(game, [END_TURN])
        assert exports_per_frame(game, 4) == [0, 0, 1, 1]
        executed = c.acks(game)[0]["executed"][0]
        assert (executed["success"], executed["error_code"]) == (False, "END_TURN_BLOCKED")
        assert end_turns_requested(game) == 0

    def test_the_exported_state_after_a_production_choice_shows_it(self) -> None:
        game = c.loaded_turn(extra=PRODUCTION_BLOCKER)
        assert c.states(game)[-1].get("end_turn_blocking") == "ENDTURN_BLOCKING_PRODUCTION"
        c.send(game, [{"action_type": "city_produce", "city_id": 301, "item_id": "UNIT_WARRIOR"}])
        c.tick(game, 4)
        assert "end_turn_blocking" not in c.states(game)[-1]


# ---------------------------------------------------------------------------
# M2: the command window runs from the last command
# ---------------------------------------------------------------------------
class TestCommandWindow:
    def test_a_message_every_twenty_seconds_keeps_the_window_open(self, turn: Any) -> None:
        for number, at in enumerate((0, 20, 40, 60), start=1):
            exchange(turn, [SKIP], number, at)
            assert [a["msg_id"] for a in c.acks(turn)] == list(range(1, number + 1)), at

    def test_silence_longer_than_the_timeout_closes_it(self, turn: Any) -> None:
        for number, at in enumerate((0, 20, 40, 60), start=1):
            exchange(turn, [SKIP], number, at)
        exchange(turn, [SKIP], 5, 95)
        assert c.pending(turn), "the message was read after 35 s of silence"
        assert len(c.acks(turn)) == 4

    def test_a_gap_just_inside_the_timeout_is_still_read(self, turn: Any) -> None:
        for number, at in enumerate((0, 20, 40, 60), start=1):
            exchange(turn, [SKIP], number, at)
        exchange(turn, [SKIP], 5, 88)
        assert not c.pending(turn)
        assert len(c.acks(turn)) == 5

    def test_control_the_window_opens_at_the_turn_start(self) -> None:
        late = c.loaded_turn()
        exchange(late, [SKIP], 1, 31)
        assert c.pending(late) and not c.acks(late)
        early = c.loaded_turn()
        exchange(early, [SKIP], 1, 29)
        assert not c.pending(early) and len(c.acks(early)) == 1

    def test_the_window_is_measured_from_the_last_message_not_the_first(self, turn: Any) -> None:
        exchange(turn, [SKIP], 1, 25)
        exchange(turn, [SKIP], 2, 50)
        assert [a["msg_id"] for a in c.acks(turn)] == [1, 2]


# ---------------------------------------------------------------------------
# M3: a cap on commands per turn
# ---------------------------------------------------------------------------
@pytest.fixture
def capped() -> Any:
    game = c.loaded_turn()
    game.execute("AlphaZeroConfig.Set('MAX_COMMANDS_PER_TURN', 3)")
    return game


class TestCommandCap:
    def test_commands_up_to_the_cap_are_executed(self, capped: Any) -> None:
        for number in (1, 2, 3):
            exchange(capped, [SKIP], number)
        acks = c.acks(capped)
        assert [a["success"] for a in acks] == [True, True, True]

    def test_the_next_message_is_answered_without_executing_anything(self, capped: Any) -> None:
        for number in (1, 2, 3):
            exchange(capped, [SKIP], number)
        exchange(capped, [c.unit_move(102, 0, 1)], 4)
        ack = c.acks(capped)[-1]
        assert ack["msg_id"] == 4
        assert (ack["success"], ack["error_code"]) == (False, "COMMAND_CAP_REACHED")
        assert (capped.eval("Civ6Stub.unit(102).x"), capped.eval("Civ6Stub.unit(102).y")) == (1, 0)
        assert c.queued(capped) == 0 and c.dropped(capped) == 0

    def test_the_cap_counts_commands_not_messages(self, capped: Any) -> None:
        exchange(capped, [SKIP, SKIP, SKIP], 1)
        exchange(capped, [c.unit_move(102, 0, 1)], 2)
        assert c.acks(capped)[-1]["error_code"] == "COMMAND_CAP_REACHED"
        assert capped.eval("Civ6Stub.unit(102).x") == 1

    def test_a_message_that_mixes_a_command_with_end_turn_executes_nothing(
        self, capped: Any
    ) -> None:
        exchange(capped, [SKIP, SKIP, SKIP], 1)
        exchange(capped, [c.unit_move(102, 0, 1), END_TURN], 2)
        assert c.acks(capped)[-1]["error_code"] == "COMMAND_CAP_REACHED"
        assert end_turns_requested(capped) == 0
        assert capped.eval("Civ6Stub.unit(102).x") == 1

    def test_a_message_holding_only_end_turn_is_still_executed(self, capped: Any) -> None:
        exchange(capped, [SKIP, SKIP, SKIP], 1)
        exchange(capped, [END_TURN], 2)
        ack = c.acks(capped)[-1]
        assert ack["success"] is True
        assert end_turns_requested(capped) == 1

    def test_the_count_starts_again_at_the_next_turn(self, capped: Any) -> None:
        exchange(capped, [SKIP, SKIP, SKIP], 1)
        capped.execute("AlphaZeroBridge.OnTurnEnd(0)\nAlphaZeroBridge.OnTurnStart(0)")
        exchange(capped, [c.unit_move(102, 0, 1)], 2)
        ack = c.acks(capped)[-1]
        assert ack["success"] is True
        assert (capped.eval("Civ6Stub.unit(102).x"), capped.eval("Civ6Stub.unit(102).y")) == (0, 1)

    def test_control_below_the_cap_a_larger_message_is_executed_whole(self) -> None:
        game = c.loaded_turn()
        exchange(game, [SKIP, SKIP, SKIP, SKIP], 1)
        assert c.acks(game)[-1]["success"] is True


# ---------------------------------------------------------------------------
# Review of GM-B7a (H1): the cap must not trap a turn behind an end-of-turn blocker
# ---------------------------------------------------------------------------
PRODUCE_WARRIOR = {"action_type": "city_produce", "city_id": 301, "item_id": "UNIT_WARRIOR"}


class TestTheCapAndAnEndOfTurnBlocker:
    def _blocked_at_the_cap(self) -> Any:
        game = c.loaded_turn(PRODUCTION_BLOCKER)
        game.execute("AlphaZeroConfig.Set('MAX_COMMANDS_PER_TURN', 3)")
        exchange(game, [SKIP, SKIP, SKIP], 1)
        exchange(game, [END_TURN], 2)
        ack = c.acks(game)[-1]
        assert ack["success"] is False, "premise: the turn cannot end, production is not chosen"
        assert end_turns_requested(game) == 0
        return game

    def test_a_command_that_clears_the_blocker_is_still_carried_out(self) -> None:
        game = self._blocked_at_the_cap()
        exchange(game, [PRODUCE_WARRIOR], 3)
        assert c.acks(game)[-1]["success"] is True
        assert game.eval("Civ6Stub.city(301).building") == "UNIT_WARRIOR"
        exchange(game, [END_TURN], 4)
        assert c.acks(game)[-1]["success"] is True
        assert end_turns_requested(game) == 1

    def test_any_other_command_is_still_refused_while_blocked(self) -> None:
        game = self._blocked_at_the_cap()
        exchange(game, [c.unit_move(102, 0, 1)], 3)
        assert c.acks(game)[-1]["error_code"] == "COMMAND_CAP_REACHED"
        assert game.eval("Civ6Stub.unit(102).x") == 1

    def test_without_a_blocker_the_same_command_is_refused_past_the_cap(self, capped: Any) -> None:
        exchange(capped, [SKIP, SKIP, SKIP], 1)
        exchange(capped, [PRODUCE_WARRIOR], 2)
        assert c.acks(capped)[-1]["error_code"] == "COMMAND_CAP_REACHED"
        assert capped.eval("Civ6Stub.city(301).building") is None

    def test_the_allowance_is_bounded_too(self) -> None:
        # A blocker the commands never clear must not make the turn endless: past twice
        # the cap nothing but end_turn is carried out.
        game = c.loaded_turn(PRODUCTION_BLOCKER)
        game.execute("AlphaZeroConfig.Set('MAX_COMMANDS_PER_TURN', 3)")
        game.execute("Civ6Stub.cities[301].canProduce = { UNIT_WARRIOR = 'later' }")
        exchange(game, [SKIP, SKIP, SKIP], 1)
        for number in (2, 3, 4):
            exchange(game, [PRODUCE_WARRIOR], number)
            assert c.acks(game)[-1]["error_code"] != "COMMAND_CAP_REACHED"
        exchange(game, [PRODUCE_WARRIOR], 5)
        assert c.acks(game)[-1]["error_code"] == "COMMAND_CAP_REACHED"

    def test_a_message_the_cap_refused_does_not_keep_the_window_open(self, capped: Any) -> None:
        exchange(capped, [SKIP, SKIP, SKIP], 1, at=0)
        exchange(capped, [c.unit_move(102, 0, 1)], 2, at=20)  # refused by the cap
        exchange(capped, [c.unit_move(102, 0, 1)], 3, at=40)  # more than 30 s after message 1
        assert c.pending(capped), "the mod went on listening on the strength of a refused message"


class TestABatchCannotCrossTheCap:
    """Review of #158: the cap looked at the count before the batch, so a batch could pass it."""

    def test_a_batch_that_would_cross_the_cap_executes_nothing(self, capped: Any) -> None:
        exchange(capped, [SKIP, SKIP], 1)
        exchange(capped, [c.unit_move(102, 0, 1), SKIP], 2)  # 2 handled + 2 > 3
        ack = c.acks(capped)[-1]
        assert (ack["success"], ack["error_code"]) == (False, "COMMAND_CAP_REACHED")
        assert (capped.eval("Civ6Stub.unit(102).x"), capped.eval("Civ6Stub.unit(102).y")) == (1, 0)

    def test_control_a_batch_that_ends_exactly_at_the_cap_is_executed(self, capped: Any) -> None:
        exchange(capped, [SKIP, SKIP], 1)
        exchange(capped, [c.unit_move(102, 0, 1)], 2)  # 2 handled + 1 == 3
        assert c.acks(capped)[-1]["success"] is True
        assert (capped.eval("Civ6Stub.unit(102).x"), capped.eval("Civ6Stub.unit(102).y")) == (0, 1)

    def test_a_blocker_clearing_batch_cannot_cross_twice_the_cap(self) -> None:
        game = c.loaded_turn(PRODUCTION_BLOCKER)
        game.execute("AlphaZeroConfig.Set('MAX_COMMANDS_PER_TURN', 3)")
        game.execute("Civ6Stub.cities[301].canProduce = { UNIT_WARRIOR = 'later' }")
        exchange(game, [SKIP, SKIP, SKIP], 1)
        exchange(
            game, [PRODUCE_WARRIOR, PRODUCE_WARRIOR], 2
        )  # 3 + 2 <= 6: allowed (refused by the game)
        assert c.acks(game)[-1]["error_code"] != "COMMAND_CAP_REACHED"
        exchange(game, [PRODUCE_WARRIOR, PRODUCE_WARRIOR], 3)  # 5 + 2 > 6
        assert c.acks(game)[-1]["error_code"] == "COMMAND_CAP_REACHED"


class TestAckErrorFields:
    """Successful AZ_ACK must not populate error_code / error_message."""

    def test_successful_command_omits_error_code_while_failure_sets_it(self, turn: Any) -> None:
        exchange(turn, [SKIP], 1)
        ok_ack = c.acks(turn)[-1]
        assert ok_ack["success"] is True
        assert "error_code" not in ok_ack
        assert "error_message" not in ok_ack

        exchange(turn, [ENEMY_MOVE], 2)
        err_ack = c.acks(turn)[-1]
        assert err_ack["success"] is False
        assert err_ack["error_code"] == "EXECUTION_ERROR"
