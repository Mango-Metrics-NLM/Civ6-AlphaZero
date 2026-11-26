"""GM-B7c-2 (A): the turn ledger, the loop's memory of what the game did with its commands.

Pure: real ``GameStateData`` built with ``from_dict``, real ``LuaCommand.to_dict()`` dicts and
real ``AckMessage`` objects. A command is remembered by what the mod was asked to do, never by
an action index; a game turn is the unit of forgetting.

Each "stays" test has a control that shows the same setup does change what it should.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from alpha_zero.bridge.protocol import AckMessage, GameStateData, LuaCommand
from alpha_zero.bridge.turn_ledger import LedgerLimits, Outcome, TurnLedger, command_key
from tests.fixtures.ledger_states import (
    BLOCKER,
    END_TURN,
    MOVE_101,
    MOVE_102,
    PRODUCE,
    blocked,
    exchange,
    no_effect_once,
    ok,
    refused,
    state,
)


# ---------------------------------------------------------------------------
# command_key
# ---------------------------------------------------------------------------
class TestCommandKey:
    def test_a_dict_and_the_command_it_came_from_have_one_key(self) -> None:
        command = LuaCommand.from_dict(MOVE_101)
        assert command_key(command) == command_key(command.to_dict()) == command_key(MOVE_101)

    def test_key_order_does_not_matter(self) -> None:
        shuffled = {"target": {"y": 1, "x": 0}, "unit_id": 101, "action_type": "unit_move"}
        assert command_key(shuffled) == command_key(MOVE_101)

    def test_metadata_does_not_matter(self) -> None:
        with_metadata = {**MOVE_101, "metadata": {"why": "policy"}}
        assert command_key(with_metadata) == command_key(MOVE_101)

    def test_a_missing_field_and_a_none_field_are_the_same(self) -> None:
        explicit = {**MOVE_101, "city_id": None, "item_id": None, "item_type": None}
        assert command_key(explicit) == command_key(MOVE_101)

    def test_an_item_id_is_compared_as_text(self) -> None:
        research = {"action_type": "research_tech", "item_id": 2}
        assert command_key(research) == command_key({**research, "item_id": "2"})

    @pytest.mark.parametrize(
        "other",
        [
            {**MOVE_101, "unit_id": 102},
            {**MOVE_101, "target": {"x": 0, "y": 2}},
            {**MOVE_101, "target": {"x": 1, "y": 1}},
            {"action_type": "unit_skip", "unit_id": 101},
            {**PRODUCE, "item_id": "BUILDING_GRANARY"},
            {**PRODUCE, "city_id": 302},
            {**PRODUCE, "item_type": "unit"},
            END_TURN,
        ],
    )
    def test_commands_that_ask_something_else_have_another_key(self, other: dict[str, Any]) -> None:
        reference = PRODUCE if other.get("action_type") == "city_produce" else MOVE_101
        assert command_key(other) != command_key(reference)

    def test_the_key_is_a_string(self) -> None:
        assert isinstance(command_key(MOVE_101), str)


# ---------------------------------------------------------------------------
# observe: game turns
# ---------------------------------------------------------------------------
class TestGameTurns:
    def test_before_the_first_state_nothing_is_known(self) -> None:
        ledger = TurnLedger()
        assert ledger.turn_id is None
        assert ledger.game_turns_finished == 0
        assert ledger.commands_this_turn == 0

    def test_the_first_state_sets_the_turn_and_finishes_nothing(self) -> None:
        ledger = TurnLedger()
        ledger.observe(state(12))
        assert (ledger.turn_id, ledger.game_turns_finished) == (12, 0)

    def test_the_same_turn_again_finishes_nothing(self) -> None:
        ledger = TurnLedger()
        for _ in range(3):
            ledger.observe(state(12))
        assert ledger.game_turns_finished == 0

    def test_each_new_turn_id_finishes_one_turn(self) -> None:
        ledger = TurnLedger()
        for turn in (12, 13, 13, 14):
            ledger.observe(state(turn))
        assert (ledger.turn_id, ledger.game_turns_finished) == (14, 2)
        assert ledger.stats()["game_turns_played"] == ledger.game_turns_finished == 2

    def test_a_turn_id_that_is_not_an_int_is_compared_by_equality(self) -> None:
        ledger = TurnLedger()
        for turn in ("a", "a", "b"):
            ledger.observe(SimpleNamespace(turn_id=turn))
        assert (ledger.turn_id, ledger.game_turns_finished) == ("b", 1)


# ---------------------------------------------------------------------------
# What a new turn clears, and that only a new turn does
# ---------------------------------------------------------------------------
class TestMemoryLivesForTheTurn:
    def _refused_and_ignored(self) -> TurnLedger:
        ledger = TurnLedger()
        first = state(12)
        ledger.observe(first)
        exchange(ledger, MOVE_101, first, refused())
        no_effect_once(ledger, PRODUCE)
        no_effect_once(ledger, PRODUCE)
        return ledger

    def test_premise_both_kinds_of_memory_are_there(self) -> None:
        ledger = self._refused_and_ignored()
        assert ledger.is_suppressed(MOVE_101)
        assert ledger.is_suppressed(PRODUCE)

    def test_a_changed_state_in_the_same_turn_forgets_nothing(self) -> None:
        ledger = self._refused_and_ignored()
        ledger.observe(state(12, units={101: (0, 0, 1.0), 102: (1, 0, 2.0)}))
        assert ledger.is_suppressed(MOVE_101)
        assert ledger.is_suppressed(PRODUCE)
        assert ledger.commands_this_turn == 3

    def test_a_new_turn_forgets_refusals_and_ignored_commands_and_the_count(self) -> None:
        ledger = self._refused_and_ignored()
        ledger.observe(state(13))
        assert not ledger.is_suppressed(MOVE_101)
        assert not ledger.is_suppressed(PRODUCE)
        assert ledger.commands_this_turn == 0

    def test_a_new_turn_restarts_the_no_effect_count_from_zero(self) -> None:
        ledger = self._refused_and_ignored()
        ledger.observe(state(13))
        no_effect_once(ledger, PRODUCE, turn=13)
        assert not ledger.is_suppressed(PRODUCE), "one miss in the new turn is not two"

    def test_a_new_turn_forgets_the_block_and_the_streak(self) -> None:
        ledger = TurnLedger()
        first = state(12, blocking=BLOCKER)
        ledger.observe(first)
        for _ in range(3):
            exchange(ledger, END_TURN, first, blocked())
            ledger.observe(state(12, blocking=BLOCKER))
        assert ledger.stalled() and ledger.end_turn_withheld()
        ledger.observe(state(13))
        assert not ledger.stalled()
        assert not ledger.end_turn_withheld()

    def test_the_session_counters_survive_a_new_turn(self) -> None:
        ledger = self._refused_and_ignored()
        ledger.observe(state(13))
        stats = ledger.stats()
        assert stats["refused_commands"] == 1
        assert stats["no_effect_commands"] == 2


# ---------------------------------------------------------------------------
# answered: outcomes
# ---------------------------------------------------------------------------
class TestAnswered:
    def test_success_is_carried_out(self) -> None:
        ledger = TurnLedger()
        first = state()
        ledger.observe(first)
        assert exchange(ledger, MOVE_101, first, ok()) is Outcome.CARRIED_OUT
        assert not ledger.is_suppressed(MOVE_101)

    def test_a_blocked_end_turn_in_the_per_command_result_is_its_own_outcome(self) -> None:
        ledger = TurnLedger()
        first = state(blocking=BLOCKER)
        ledger.observe(first)
        assert exchange(ledger, END_TURN, first, blocked()) is Outcome.BLOCKED_END_TURN
        stats = ledger.stats()
        assert stats["blocked_end_turns"] == 1
        assert stats["refused_commands"] == 0

    def test_any_other_failure_is_a_refusal_and_is_remembered(self) -> None:
        ledger = TurnLedger()
        first = state()
        ledger.observe(first)
        assert exchange(ledger, MOVE_101, first, refused()) is Outcome.REFUSED
        assert ledger.is_suppressed(MOVE_101)
        assert ledger.stats()["refused_commands"] == 1

    def test_a_failure_without_any_per_command_result_is_a_refusal(self) -> None:
        ledger = TurnLedger()
        first = state()
        ledger.observe(first)
        bare = AckMessage(msg_id=1, success=False, error_code="COMMAND_FAILED")
        assert exchange(ledger, MOVE_101, first, bare) is Outcome.REFUSED

    def test_a_duck_typed_ack_is_read_too(self) -> None:
        ledger = TurnLedger()
        first = state()
        ledger.observe(first)
        duck = SimpleNamespace(success=False, executed=[], error_code="X")
        assert exchange(ledger, MOVE_101, first, duck) is Outcome.REFUSED
        assert exchange(ledger, MOVE_102, first, SimpleNamespace(success=True)) is (
            Outcome.CARRIED_OUT
        )

    def test_a_refusal_is_remembered_by_command_not_by_unit_list_position(self) -> None:
        ledger = TurnLedger()
        first = state(units={101: (0, 0, 2.0), 102: (1, 0, 2.0)})
        ledger.observe(first)
        exchange(ledger, MOVE_101, first, refused())
        # the unit list is reordered and a unit is gone: the refusal is about unit 101's order
        ledger.observe(state(units={102: (1, 0, 2.0), 101: (0, 0, 2.0), 103: (3, 3, 2.0)}))
        assert ledger.is_suppressed(MOVE_101)
        ledger.observe(state(units={102: (1, 0, 2.0)}))
        assert ledger.is_suppressed(MOVE_101)

    def test_a_refusal_of_one_units_command_does_not_suppress_the_same_order_for_another(
        self,
    ) -> None:
        ledger = TurnLedger()
        first = state()
        ledger.observe(first)
        exchange(ledger, MOVE_101, first, refused())
        assert ledger.is_suppressed(MOVE_101)
        assert not ledger.is_suppressed(MOVE_102)
        assert not ledger.is_suppressed({**MOVE_101, "target": {"x": 5, "y": 5}})

    def test_the_suppression_test_uses_the_key_not_the_object(self) -> None:
        ledger = TurnLedger()
        first = state()
        ledger.observe(first)
        exchange(ledger, MOVE_101, first, refused())
        reordered = {"target": {"y": 1, "x": 0}, "action_type": "unit_move", "unit_id": 101}
        assert ledger.is_suppressed(reordered)
        assert ledger.is_suppressed(LuaCommand.from_dict(MOVE_101))


# ---------------------------------------------------------------------------
# end_turn is never suppressed
# ---------------------------------------------------------------------------
class TestEndTurnIsNeverSuppressed:
    def test_not_after_being_blocked(self) -> None:
        ledger = TurnLedger()
        first = state(blocking=BLOCKER)
        ledger.observe(first)
        exchange(ledger, END_TURN, first, blocked())
        assert not ledger.is_suppressed(END_TURN)

    def test_not_after_being_refused_for_another_reason(self) -> None:
        ledger = TurnLedger()
        first = state()
        ledger.observe(first)
        exchange(ledger, END_TURN, first, refused())
        assert not ledger.is_suppressed(END_TURN)

    def test_not_after_it_had_no_visible_effect(self) -> None:
        ledger = TurnLedger()
        for _ in range(5):
            no_effect_once(ledger, END_TURN)
        assert not ledger.is_suppressed(END_TURN)

    def test_not_when_the_cap_is_reached_or_the_turn_is_stalled(self) -> None:
        ledger = TurnLedger(LedgerLimits(max_commands_per_turn=1, max_blocked_end_turns=1))
        first = state(blocking=BLOCKER)
        ledger.observe(first)
        exchange(ledger, END_TURN, first, blocked())
        assert ledger.over_cap() and ledger.stalled()
        assert not ledger.is_suppressed(END_TURN)

    def test_control_the_same_refusal_does_suppress_another_command(self) -> None:
        ledger = TurnLedger()
        first = state()
        ledger.observe(first)
        exchange(ledger, PRODUCE, first, refused())
        assert ledger.is_suppressed(PRODUCE)


# ---------------------------------------------------------------------------
# A blocked end of turn is withheld until something changes
# ---------------------------------------------------------------------------
class TestEndTurnWithheld:
    def _after_a_block(self) -> tuple[TurnLedger, GameStateData]:
        ledger = TurnLedger()
        first = state(blocking=BLOCKER)
        ledger.observe(first)
        assert not ledger.end_turn_withheld()
        exchange(ledger, END_TURN, first, blocked())
        return ledger, first

    def test_it_is_withheld_after_a_block(self) -> None:
        ledger, _ = self._after_a_block()
        assert ledger.end_turn_withheld()

    def test_it_stays_withheld_while_the_state_is_the_same(self) -> None:
        ledger, _ = self._after_a_block()
        ledger.observe(state(blocking=BLOCKER))
        assert ledger.end_turn_withheld()

    def test_a_changed_state_releases_it(self) -> None:
        ledger, _ = self._after_a_block()
        ledger.observe(state(blocking=BLOCKER, research="TECH_B"))
        assert not ledger.end_turn_withheld()

    def test_a_moved_unit_releases_it(self) -> None:
        ledger, _ = self._after_a_block()
        ledger.observe(state(blocking=BLOCKER, units={101: (0, 1, 1.0), 102: (1, 0, 2.0)}))
        assert not ledger.end_turn_withheld()

    def test_the_blocker_going_away_releases_it(self) -> None:
        ledger, _ = self._after_a_block()
        ledger.observe(state(blocking=None))
        assert not ledger.end_turn_withheld()

    def test_a_command_that_changed_nothing_does_not_release_it(self) -> None:
        ledger, first = self._after_a_block()
        exchange(ledger, PRODUCE, first, ok())
        ledger.observe(state(blocking=BLOCKER))
        assert ledger.end_turn_withheld()

    def test_control_a_successful_end_turn_withholds_nothing(self) -> None:
        ledger = TurnLedger()
        first = state()
        ledger.observe(first)
        exchange(ledger, END_TURN, first, ok())
        assert not ledger.end_turn_withheld()
