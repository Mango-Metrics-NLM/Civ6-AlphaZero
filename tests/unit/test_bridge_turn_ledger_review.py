"""GM-B7c-2 review round (pure): choices the game asks for again, the mod's cap, a failed send.

Each test names the defect it pins. Real ``GameStateData`` and ``AckMessage`` objects.
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha_zero.bridge.protocol import AckMessage, GameStateData
from alpha_zero.bridge.turn_ledger import UNBLOCKING_ACTIONS, LedgerLimits, TurnLedger
from tests.fixtures.ledger_states import BLOCKER, END_TURN, MOVE_101, PRODUCE, exchange, ok

RESEARCH = {"action_type": "research_tech", "item_id": 2}
RESEARCH_B = {"action_type": "research_tech", "item_id": 1}
CIVIC = {"action_type": "research_civic", "item_id": 2}
CIVIC_B = {"action_type": "research_civic", "item_id": 1}
GRANARY = {"action_type": "city_produce", "city_id": 301, "item_id": "BUILDING_GRANARY"}
SKIP = {"action_type": "unit_skip", "unit_id": 101}
CAP_ACK = AckMessage(msg_id=1, success=False, error_code="COMMAND_CAP_REACHED")


def st(
    turn: int = 12,
    *,
    research: str | None = "TECH_A",
    civic: str | None = "CIVIC_A",
    producing: str | None = "UNIT_WARRIOR",
    blocking: str | None = None,
) -> GameStateData:
    data: dict[str, Any] = {
        "turn_id": turn,
        "player_id": 0,
        "map": {"width": 4, "height": 4},
        "units": [{"id": 101, "owner": 0, "x": 0, "y": 0, "moves_remaining": 2.0}],
        "cities": [{"id": 301, "owner": 0, "x": 0, "y": 0, "current_production": producing}],
        "players": [{"id": 0, "current_research": research, "current_civic": civic}],
    }
    if blocking is not None:
        data["end_turn_blocking"] = blocking
    return GameStateData.from_dict(data)


def chose(command: dict[str, Any], **took_effect: Any) -> TurnLedger:
    """A ledger in which ``command`` was carried out and its effect seen (so the slot is held)."""
    ledger = TurnLedger()
    before = st()
    ledger.observe(before)
    exchange(ledger, command, before, ok())
    ledger.observe(st(**took_effect))
    return ledger


# ---------------------------------------------------------------------------
# 3. A held slot is released when the game asks again
# ---------------------------------------------------------------------------
class TestSlotReleasedWhenTheGameAsksAgain:
    @pytest.mark.parametrize(
        ("command", "other", "took", "reopened"),
        [
            (RESEARCH, RESEARCH_B, {"research": "TECH_B"}, {"research": None}),
            (
                RESEARCH,
                RESEARCH_B,
                {"research": "TECH_B"},
                {"blocking": "ENDTURN_BLOCKING_RESEARCH"},
            ),
            (CIVIC, CIVIC_B, {"civic": "CIVIC_B"}, {"civic": None}),
            (CIVIC, CIVIC_B, {"civic": "CIVIC_B"}, {"blocking": "ENDTURN_BLOCKING_CIVIC"}),
            (PRODUCE, GRANARY, {"producing": "BUILDING_X"}, {"producing": None}),
            (
                PRODUCE,
                GRANARY,
                {"producing": "BUILDING_X"},
                {"blocking": "ENDTURN_BLOCKING_PRODUCTION"},
            ),
        ],
    )
    def test_the_slot_is_open_again(
        self,
        command: dict[str, Any],
        other: dict[str, Any],
        took: dict[str, Any],
        reopened: dict[str, Any],
    ) -> None:
        ledger = chose(command, **took)
        assert ledger.is_suppressed(other), "premise: the slot is held"
        ledger.observe(st(**{**took, **reopened}))
        assert not ledger.is_suppressed(other)
        assert not ledger.is_suppressed(command)

    @pytest.mark.parametrize(
        ("command", "other", "took"),
        [
            (RESEARCH, RESEARCH_B, {"research": "TECH_B"}),
            (CIVIC, CIVIC_B, {"civic": "CIVIC_B"}),
            (PRODUCE, GRANARY, {"producing": "BUILDING_X"}),
        ],
    )
    def test_control_still_in_progress_and_no_blocker_keeps_it_held(
        self, command: dict[str, Any], other: dict[str, Any], took: dict[str, Any]
    ) -> None:
        ledger = chose(command, **took)
        ledger.observe(st(**took))
        assert ledger.is_suppressed(other)

    def test_a_blocker_about_something_else_keeps_it_held(self) -> None:
        ledger = chose(RESEARCH, research="TECH_B")
        ledger.observe(st(research="TECH_B", blocking="ENDTURN_BLOCKING_PRODUCTION"))
        assert ledger.is_suppressed(RESEARCH_B)

    def test_reopening_one_slot_leaves_the_others_held(self) -> None:
        ledger = TurnLedger()
        first = st()
        ledger.observe(first)
        exchange(ledger, RESEARCH, first, ok())
        exchange(ledger, CIVIC, first, ok())
        ledger.observe(st(research="TECH_B", civic="CIVIC_B"))
        ledger.observe(st(research=None, civic="CIVIC_B"))
        assert not ledger.is_suppressed(RESEARCH_B)
        assert ledger.is_suppressed(CIVIC_B)


# ---------------------------------------------------------------------------
# 4. The mod's own cap answer
# ---------------------------------------------------------------------------
class TestModCapAnswer:
    def _after_cap_answer(self) -> TurnLedger:
        ledger = TurnLedger()
        first = st()
        ledger.observe(first)
        ledger.sent(MOVE_101, first)
        ledger.answered(MOVE_101, CAP_ACK)
        return ledger

    def test_the_command_is_not_remembered_as_refused(self) -> None:
        assert not self._after_cap_answer().is_suppressed(MOVE_101)

    def test_the_turn_is_over_the_cap_whatever_our_own_count_says(self) -> None:
        ledger = self._after_cap_answer()
        assert ledger.commands_this_turn == 1
        assert ledger.over_cap()
        ledger.observe(st(research="TECH_Z"))
        assert ledger.over_cap(), "kept for the rest of the turn"

    def test_a_new_turn_clears_it(self) -> None:
        ledger = self._after_cap_answer()
        ledger.observe(st(13))
        assert not ledger.over_cap()

    def test_control_another_failure_is_a_refusal_and_not_a_cap(self) -> None:
        ledger = TurnLedger()
        first = st()
        ledger.observe(first)
        ledger.sent(MOVE_101, first)
        ledger.answered(MOVE_101, AckMessage(msg_id=1, success=False, error_code="SOMETHING"))
        assert ledger.is_suppressed(MOVE_101)
        assert not ledger.over_cap()


# ---------------------------------------------------------------------------
# The cap and unblocking: may_unblock
# ---------------------------------------------------------------------------
class TestMayUnblock:
    def _withheld(self, cap: int) -> tuple[TurnLedger, GameStateData]:
        ledger = TurnLedger(LedgerLimits(max_commands_per_turn=cap))
        current = st(blocking=BLOCKER)
        ledger.observe(current)
        from tests.fixtures.ledger_states import blocked

        exchange(ledger, END_TURN, current, blocked())
        return ledger, current

    def test_the_unblocking_kinds_are_the_documented_ones(self) -> None:
        assert UNBLOCKING_ACTIONS == {
            "research_tech",
            "research_civic",
            "city_produce",
            "unit_skip",
            "unit_fortify",
            "unit_sleep",
        }

    def test_not_while_nothing_blocks_the_end_of_turn(self) -> None:
        ledger = TurnLedger(LedgerLimits(max_commands_per_turn=1))
        current = st()
        ledger.observe(current)
        ledger.sent(MOVE_101, current)
        assert ledger.over_cap() and not ledger.may_unblock()

    def test_the_boundary_is_twice_the_cap(self) -> None:
        ledger, current = self._withheld(3)  # one end_turn sent so far
        for count in range(2, 6):
            ledger.sent(SKIP, current)
            assert ledger.commands_this_turn == count
            assert ledger.may_unblock(), count
        ledger.sent(SKIP, current)
        assert ledger.commands_this_turn == 6
        assert not ledger.may_unblock()

    def test_a_released_block_ends_it(self) -> None:
        ledger, _ = self._withheld(1)
        ledger.observe(st(research="TECH_Z"))  # the state changed: not withheld any more
        assert not ledger.may_unblock()


# ---------------------------------------------------------------------------
# 5. A send that raised is not an answer
# ---------------------------------------------------------------------------
class TestSendFailed:
    def test_the_next_state_with_a_blocker_is_not_a_blocked_end_turn(self) -> None:
        ledger = TurnLedger()
        first = st(blocking=BLOCKER)
        ledger.observe(first)
        ledger.sent(END_TURN, first)
        ledger.send_failed()
        ledger.observe(st(blocking=BLOCKER))
        assert ledger.stats()["blocked_end_turns"] == 0
        assert not ledger.end_turn_withheld()

    def test_three_failed_sends_are_not_a_stall(self) -> None:
        ledger = TurnLedger()
        current = st(blocking=BLOCKER)
        ledger.observe(current)
        for _ in range(3):
            ledger.sent(END_TURN, current)
            ledger.send_failed()
            current = st(blocking=BLOCKER)
            ledger.observe(current)
        assert not ledger.stalled()

    def test_control_without_send_failed_the_same_sequence_is_a_stall(self) -> None:
        ledger = TurnLedger()
        current = st(blocking=BLOCKER)
        ledger.observe(current)
        for _ in range(3):
            ledger.sent(END_TURN, current)
            current = st(blocking=BLOCKER)
            ledger.observe(current)
        assert ledger.stalled()

    def test_the_failed_send_still_counts_against_the_cap(self) -> None:
        ledger = TurnLedger(LedgerLimits(max_commands_per_turn=1))
        first = st()
        ledger.observe(first)
        ledger.sent(MOVE_101, first)
        ledger.send_failed()
        assert ledger.over_cap()

    def test_a_failed_send_is_not_judged_for_effect(self) -> None:
        ledger = TurnLedger()
        first = st()
        ledger.observe(first)
        ledger.sent(PRODUCE, first)
        ledger.send_failed()
        ledger.observe(st())
        assert ledger.stats()["no_effect_commands"] == 0


# ---------------------------------------------------------------------------
# 6. blocked_end_turns counts the fallback end_turn seen blocked in the next state
# ---------------------------------------------------------------------------
class TestFallbackBlockedIsCounted:
    def test_the_ledger_stat_includes_it(self) -> None:
        ledger = TurnLedger()
        first = st()
        ledger.observe(first)
        ledger.sent(END_TURN, first)
        ledger.observe(st(blocking=BLOCKER))
        assert ledger.stats()["blocked_end_turns"] == 1
