"""GM-B7c-2 (A, part 2): the turn ledger's limits: no effect, the unanswered end_turn, cap, stall.

See ``test_bridge_turn_ledger.py`` for identity, turns and outcomes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from alpha_zero.bridge.protocol import GameStateData
from alpha_zero.bridge.turn_ledger import LedgerLimits, TurnLedger
from tests.fixtures.ledger_states import (
    BLOCKER,
    END_TURN,
    MOVE_101,
    MOVE_102,
    PRODUCE,
    UPGRADE,
    blocked,
    current_research,
    exchange,
    no_effect_once,
    ok,
    refused,
    state,
)


# ---------------------------------------------------------------------------
# "No expected effect", twice
# ---------------------------------------------------------------------------
class TestNoEffect:
    def test_one_miss_is_not_enough(self) -> None:
        ledger = TurnLedger()
        no_effect_once(ledger, PRODUCE)
        assert not ledger.is_suppressed(PRODUCE)
        assert ledger.stats()["no_effect_commands"] == 1

    def test_two_misses_suppress_the_command(self) -> None:
        ledger = TurnLedger()
        no_effect_once(ledger, PRODUCE)
        no_effect_once(ledger, PRODUCE)
        assert ledger.is_suppressed(PRODUCE)
        assert ledger.stats()["no_effect_commands"] == 2

    def test_the_limit_is_configurable(self) -> None:
        ledger = TurnLedger(LedgerLimits(no_effect_limit=3))
        no_effect_once(ledger, PRODUCE)
        no_effect_once(ledger, PRODUCE)
        assert not ledger.is_suppressed(PRODUCE)
        no_effect_once(ledger, PRODUCE)
        assert ledger.is_suppressed(PRODUCE)

    def test_the_miss_is_counted_per_command_not_per_kind(self) -> None:
        ledger = TurnLedger()
        no_effect_once(ledger, PRODUCE)
        no_effect_once(ledger, {**PRODUCE, "item_id": "BUILDING_GRANARY"})
        assert not ledger.is_suppressed(PRODUCE)
        assert not ledger.is_suppressed({**PRODUCE, "item_id": "BUILDING_GRANARY"})

    def test_control_a_command_that_moved_the_unit_is_allowed_again(self) -> None:
        ledger = TurnLedger()
        for step in range(3):
            before = state(units={101: (0, step, 2.0), 102: (1, 0, 2.0)})
            ledger.observe(before)
            command = {**MOVE_101, "target": {"x": 0, "y": step + 1}}
            exchange(ledger, command, before, ok())
            ledger.observe(state(units={101: (0, step + 1, 1.0), 102: (1, 0, 2.0)}))
            assert not ledger.is_suppressed(command)
        assert ledger.stats()["no_effect_commands"] == 0

    def test_control_the_same_move_that_did_not_move_the_unit_is_suppressed(self) -> None:
        ledger = TurnLedger()
        no_effect_once(ledger, MOVE_101)
        no_effect_once(ledger, MOVE_101)
        assert ledger.is_suppressed(MOVE_101)

    def test_a_hit_does_not_reset_the_misses(self) -> None:
        ledger = TurnLedger()
        no_effect_once(ledger, MOVE_101)
        before = state()
        ledger.observe(before)
        exchange(ledger, MOVE_101, before, ok())
        ledger.observe(state(units={101: (0, 1, 1.0), 102: (1, 0, 2.0)}))  # it did move
        assert not ledger.is_suppressed(MOVE_101)
        no_effect_once(ledger, MOVE_101)
        assert ledger.is_suppressed(MOVE_101)

    def test_a_command_the_diff_cannot_judge_is_never_counted(self) -> None:
        ledger = TurnLedger()
        for _ in range(4):
            no_effect_once(ledger, UPGRADE)
        assert not ledger.is_suppressed(UPGRADE)
        assert ledger.stats()["no_effect_commands"] == 0

    def test_a_command_is_judged_once(self) -> None:
        ledger = TurnLedger()
        no_effect_once(ledger, PRODUCE)
        ledger.observe(state(12))
        ledger.observe(state(12))
        assert ledger.stats()["no_effect_commands"] == 1

    def test_a_command_the_game_refused_is_not_also_counted_as_ignored(self) -> None:
        ledger = TurnLedger()
        first = state()
        ledger.observe(first)
        exchange(ledger, MOVE_101, first, refused())
        ledger.observe(state())
        stats = ledger.stats()
        assert (stats["refused_commands"], stats["no_effect_commands"]) == (1, 0)

    def test_a_command_never_answered_is_judged_all_the_same(self) -> None:
        ledger = TurnLedger()
        first = state()
        ledger.observe(first)
        ledger.sent(PRODUCE, first)
        ledger.observe(state())
        assert ledger.stats()["no_effect_commands"] == 1

    def test_a_new_turn_is_not_a_judgement(self) -> None:
        ledger = TurnLedger()
        first = state(12)
        ledger.observe(first)
        exchange(ledger, PRODUCE, first, ok())
        ledger.observe(state(13))
        assert ledger.stats()["no_effect_commands"] == 0


# ---------------------------------------------------------------------------
# The fallback end_turn: sent, its answer never read
# ---------------------------------------------------------------------------
class TestUnansweredEndTurn:
    def test_the_same_turn_back_with_a_blocker_counts_as_a_blocked_end_turn(self) -> None:
        ledger = TurnLedger()
        first = state(blocking=BLOCKER)
        ledger.observe(first)
        ledger.sent(END_TURN, first)  # no answered(): the fallback path
        ledger.observe(state(blocking=BLOCKER))
        assert ledger.stats()["blocked_end_turns"] == 1
        assert ledger.end_turn_withheld()

    def test_three_of_them_in_a_row_are_a_stall(self) -> None:
        ledger = TurnLedger()
        current = state(blocking=BLOCKER)
        ledger.observe(current)
        for _ in range(3):
            ledger.sent(END_TURN, current)
            current = state(blocking=BLOCKER)
            ledger.observe(current)
        assert ledger.stalled()

    def test_control_without_a_blocker_nothing_is_counted(self) -> None:
        ledger = TurnLedger()
        first = state()
        ledger.observe(first)
        ledger.sent(END_TURN, first)
        ledger.observe(state())
        assert ledger.stats()["blocked_end_turns"] == 0
        assert not ledger.end_turn_withheld()

    def test_control_a_new_turn_with_a_blocker_is_not_a_blocked_end_turn(self) -> None:
        ledger = TurnLedger()
        first = state(12)
        ledger.observe(first)
        ledger.sent(END_TURN, first)
        ledger.observe(state(13, blocking=BLOCKER))
        assert ledger.stats()["blocked_end_turns"] == 0

    def test_an_answered_blocked_end_turn_is_not_counted_twice(self) -> None:
        ledger = TurnLedger()
        first = state(blocking=BLOCKER)
        ledger.observe(first)
        exchange(ledger, END_TURN, first, blocked())
        ledger.observe(state(blocking=BLOCKER))
        assert ledger.stats()["blocked_end_turns"] == 1


# ---------------------------------------------------------------------------
# The cap on commands per game turn
# ---------------------------------------------------------------------------
class TestCap:
    def test_the_default_is_a_hundred_and_the_boundary_is_exact(self) -> None:
        ledger = TurnLedger()
        current = state()
        ledger.observe(current)
        for number in range(1, 101):
            assert not ledger.over_cap(), f"over the cap after only {number - 1} commands"
            ledger.sent(MOVE_101, current)
            assert ledger.commands_this_turn == number
        assert ledger.over_cap()

    def test_the_cap_is_configurable(self) -> None:
        ledger = TurnLedger(LedgerLimits(max_commands_per_turn=2))
        current = state()
        ledger.observe(current)
        ledger.sent(MOVE_101, current)
        assert not ledger.over_cap()
        ledger.sent(MOVE_102, current)
        assert ledger.over_cap()

    def test_a_changed_state_in_the_same_turn_does_not_lower_the_count(self) -> None:
        ledger = TurnLedger(LedgerLimits(max_commands_per_turn=2))
        current = state()
        ledger.observe(current)
        ledger.sent(MOVE_101, current)
        ledger.sent(MOVE_102, current)
        ledger.observe(state(research="TECH_Z"))
        assert ledger.over_cap()

    def test_a_new_turn_lifts_it(self) -> None:
        ledger = TurnLedger(LedgerLimits(max_commands_per_turn=1))
        current = state(12)
        ledger.observe(current)
        ledger.sent(MOVE_101, current)
        assert ledger.over_cap()
        ledger.observe(state(13))
        assert not ledger.over_cap()


# ---------------------------------------------------------------------------
# Stall: blocked end turns in a row, state unchanged
# ---------------------------------------------------------------------------
class TestStall:
    def _block(self, ledger: TurnLedger, current: GameStateData) -> GameStateData:
        exchange(ledger, END_TURN, current, blocked())
        nxt = state(blocking=BLOCKER, research=current_research(current))
        ledger.observe(nxt)
        return nxt

    def test_two_blocks_are_not_a_stall_and_three_are(self) -> None:
        ledger = TurnLedger()
        current = state(blocking=BLOCKER)
        ledger.observe(current)
        current = self._block(ledger, current)
        assert not ledger.stalled()
        current = self._block(ledger, current)
        assert not ledger.stalled()
        self._block(ledger, current)
        assert ledger.stalled()

    def test_the_limit_is_configurable(self) -> None:
        ledger = TurnLedger(LedgerLimits(max_blocked_end_turns=2))
        current = state(blocking=BLOCKER)
        ledger.observe(current)
        current = self._block(ledger, current)
        assert not ledger.stalled()
        self._block(ledger, current)
        assert ledger.stalled()

    def test_a_changed_state_resets_the_streak(self) -> None:
        ledger = TurnLedger()
        current = state(blocking=BLOCKER)
        ledger.observe(current)
        current = self._block(ledger, current)
        current = self._block(ledger, current)
        ledger.observe(state(blocking=BLOCKER, units={101: (0, 1, 1.0), 102: (1, 0, 2.0)}))
        assert not ledger.stalled()
        current = state(blocking=BLOCKER, units={101: (0, 1, 1.0), 102: (1, 0, 2.0)})
        current = self._block(ledger, current)
        assert not ledger.stalled(), "the streak restarted at one, it did not carry two over"

    def test_control_without_the_change_the_third_block_stalls(self) -> None:
        ledger = TurnLedger()
        current = state(blocking=BLOCKER)
        ledger.observe(current)
        for _ in range(3):
            current = self._block(ledger, current)
        assert ledger.stalled()

    def test_a_command_that_was_carried_out_resets_the_streak(self) -> None:
        ledger = TurnLedger()
        current = state(blocking=BLOCKER)
        ledger.observe(current)
        current = self._block(ledger, current)
        current = self._block(ledger, current)
        exchange(ledger, PRODUCE, current, ok())
        assert not ledger.stalled()
        current = self._block(ledger, current)
        assert not ledger.stalled()

    def test_an_accepted_end_turn_does_not_reset_the_streak(self) -> None:
        ledger = TurnLedger()
        current = state(blocking=BLOCKER)
        ledger.observe(current)
        current = self._block(ledger, current)
        current = self._block(ledger, current)
        exchange(ledger, END_TURN, current, ok())
        current = self._block(ledger, current)
        assert ledger.stalled()

    def test_a_refusal_alone_is_never_a_stall(self) -> None:
        ledger = TurnLedger()
        current = state()
        ledger.observe(current)
        for number in range(6):
            exchange(ledger, {**MOVE_101, "target": {"x": 0, "y": number}}, current, refused())
            current = state()
            ledger.observe(current)
        assert not ledger.stalled()


# ---------------------------------------------------------------------------
# Tolerance and stats
# ---------------------------------------------------------------------------
class TestToleranceAndStats:
    def test_a_test_double_state_does_not_raise(self) -> None:
        ledger = TurnLedger()
        double = MagicMock()
        ledger.observe(double)
        ledger.sent(PRODUCE, double)
        ledger.answered(PRODUCE, ok())
        ledger.observe(double)
        ledger.sent(END_TURN, double)
        ledger.observe(double)
        assert ledger.turn_id is double.turn_id
        assert ledger.game_turns_finished == 0

    def test_a_double_state_after_a_real_one_does_not_raise(self) -> None:
        ledger = TurnLedger()
        real = state()
        ledger.observe(real)
        ledger.sent(PRODUCE, real)
        ledger.observe(MagicMock())
        ledger.observe(real)

    def test_the_stats_have_exactly_the_documented_keys(self) -> None:
        assert set(TurnLedger().stats()) == {
            "refused_commands",
            "no_effect_commands",
            "blocked_end_turns",
            "game_turns_played",
        }

    def test_the_stats_start_at_zero(self) -> None:
        assert set(TurnLedger().stats().values()) == {0}

    def test_the_default_limits_are_the_documented_ones(self) -> None:
        limits = LedgerLimits()
        assert (limits.max_commands_per_turn, limits.no_effect_limit) == (100, 2)
        assert limits.max_blocked_end_turns == 3
        with pytest.raises(AttributeError):
            limits.no_effect_limit = 5  # type: ignore[misc]  # frozen


# ---------------------------------------------------------------------------
# Once-per-turn choices (addendum): research, civic, and production per city
# ---------------------------------------------------------------------------
RESEARCH_A = {"action_type": "research_tech", "item_id": 2}
RESEARCH_B = {"action_type": "research_tech", "item_id": 1}
CIVIC = {"action_type": "research_civic", "item_id": 2}
GRANARY = {"action_type": "city_produce", "city_id": 301, "item_id": "BUILDING_GRANARY"}
OTHER_CITY = {"action_type": "city_produce", "city_id": 302, "item_id": "UNIT_WARRIOR"}


def chosen(ledger: TurnLedger, command: dict[str, object]) -> None:
    """Send ``command`` on a state, get success, and see a state where it took effect."""
    before = state()
    ledger.observe(before)
    exchange(ledger, command, before, ok())


class TestOncePerTurnChoices:
    def test_a_carried_out_research_suppresses_any_other_tech_this_turn(self) -> None:
        ledger = TurnLedger()
        chosen(ledger, RESEARCH_A)
        ledger.observe(state(research="TECH_B"))  # it took effect
        assert ledger.is_suppressed(RESEARCH_A)
        assert ledger.is_suppressed(RESEARCH_B)

    def test_control_before_it_is_carried_out_nothing_is_suppressed(self) -> None:
        ledger = TurnLedger()
        first = state()
        ledger.observe(first)
        assert not ledger.is_suppressed(RESEARCH_A)
        exchange(ledger, RESEARCH_A, first, refused())
        assert not ledger.is_suppressed(RESEARCH_B), "a refusal of one tech is not the slot"

    def test_a_civic_is_its_own_slot(self) -> None:
        ledger = TurnLedger()
        chosen(ledger, RESEARCH_A)
        ledger.observe(state(research="TECH_B"))
        assert not ledger.is_suppressed(CIVIC)
        chosen(ledger, CIVIC)
        assert ledger.is_suppressed({**CIVIC, "item_id": 1})

    def test_production_is_one_slot_per_city(self) -> None:
        ledger = TurnLedger()
        chosen(ledger, PRODUCE)
        ledger.observe(state(producing="BUILDING_MONUMENT"))
        assert ledger.is_suppressed(GRANARY)
        assert not ledger.is_suppressed(OTHER_CITY)

    def test_the_slot_is_held_after_the_state_changes_again(self) -> None:
        ledger = TurnLedger()
        chosen(ledger, RESEARCH_A)
        ledger.observe(state(research="TECH_B"))
        ledger.observe(state(research="TECH_B", units={101: (0, 1, 1.0), 102: (1, 0, 2.0)}))
        assert ledger.is_suppressed(RESEARCH_B)

    def test_a_choice_with_no_visible_effect_releases_the_slot(self) -> None:
        ledger = TurnLedger()
        chosen(ledger, RESEARCH_A)
        ledger.observe(state())  # research unchanged: no expected effect
        assert not ledger.is_suppressed(RESEARCH_B)
        assert not ledger.is_suppressed(RESEARCH_A), "one miss is not two"
        assert ledger.stats()["no_effect_commands"] == 1

    def test_a_released_slot_still_obeys_the_per_command_miss_count(self) -> None:
        ledger = TurnLedger()
        for _ in range(2):
            chosen(ledger, RESEARCH_A)
            ledger.observe(state())
        assert ledger.is_suppressed(RESEARCH_A)
        assert not ledger.is_suppressed(RESEARCH_B)

    def test_a_new_turn_clears_the_slots(self) -> None:
        ledger = TurnLedger()
        chosen(ledger, RESEARCH_A)
        ledger.observe(state(research="TECH_B"))
        chosen(ledger, PRODUCE)
        ledger.observe(state(13))
        assert not ledger.is_suppressed(RESEARCH_B)
        assert not ledger.is_suppressed(PRODUCE)
