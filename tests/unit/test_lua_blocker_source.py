"""GM-B2 / GM-B9: where the end-of-turn blocker comes from (the game's query, else the list scan).

Split out of ``test_lua_exporter_parity.py`` (kept under the 1,000-line ratchet); the tests are
unchanged apart from the helpers they need.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import lua_harness as h

RESEARCH_BLOCKER = (
    "{ id = 5, blocking = EndTurnBlockingTypes.ENDTURN_BLOCKING_RESEARCH,"
    " message = 'Choose a technology', typeName = 'NOTIFICATION_CHOOSE_TECH', dismissed = false }"
)
CIVIC_BLOCKER = (
    "{ id = 6, blocking = EndTurnBlockingTypes.ENDTURN_BLOCKING_CIVIC,"
    " message = 'Choose a civic', typeName = 'NOTIFICATION_CHOOSE_CIVIC', dismissed = false }"
)
END_TURN = {"action_type": "end_turn"}


def _notify(runtime: Any, *entries: str) -> None:
    runtime.execute(f"Civ6Stub.notifications[0] = {{ {', '.join(entries)} }}")


class TestBlockerSource:
    """The game's own query wins; the list scan is only the fallback."""

    @pytest.mark.parametrize("answer", ["EndTurnBlockingTypes.NO_ENDTURN_BLOCKING", "0"])
    def test_the_query_saying_nothing_blocks_wins_over_the_list(self, answer: str) -> None:
        runtime = h.load_mod()
        _notify(runtime, RESEARCH_BLOCKER)
        runtime.execute(
            f"NotificationManager.GetFirstEndTurnBlocking = function() return {answer} end"
        )
        state = h.export_state(runtime)
        assert state["schema_version"] == 2
        assert "end_turn_blocking" not in state
        assert h.execute(runtime, END_TURN) == (True, None, "End turn requested")

    @pytest.mark.parametrize("answer", ["nil", "'none'", "{}", "true"])
    def test_a_query_answer_that_is_not_a_number_is_not_nothing_blocks(self, answer: str) -> None:
        """GM-B9 (M2) changed the rule: only a NUMBER is the game's answer. Anything else makes
        the notification list the source, so a listed blocker still blocks."""
        runtime = h.load_mod()
        _notify(runtime, RESEARCH_BLOCKER)
        runtime.execute(
            f"NotificationManager.GetFirstEndTurnBlocking = function() return {answer} end"
        )
        assert h.export_state(runtime)["end_turn_blocking"] == "ENDTURN_BLOCKING_RESEARCH"
        success, code, message = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "END_TURN_BLOCKED")
        assert message.startswith("End turn blocked: ENDTURN_BLOCKING_RESEARCH")
        assert h.stub_calls(runtime) == []
        assert runtime.eval("Civ6Stub.turnDeactivated") is False

    @pytest.mark.parametrize("answer", ["nil", "'none'", "{}"])
    def test_a_non_number_answer_with_an_empty_list_requests_the_end_of_turn(
        self, answer: str
    ) -> None:
        runtime = h.load_mod(
            setup=f"NotificationManager.GetFirstEndTurnBlocking = function() return {answer} end"
        )
        assert "end_turn_blocking" not in h.export_state(runtime)
        assert h.execute(runtime, END_TURN) == (True, None, "End turn requested")
        assert runtime.eval("Civ6Stub.turnEnded") is True

    @pytest.mark.parametrize("answer", ["nil", "'none'", "{}"])
    def test_a_non_number_answer_with_the_list_missing_is_not_askable(self, answer: str) -> None:
        runtime = h.load_mod(
            setup=(
                f"NotificationManager.GetFirstEndTurnBlocking = function() return {answer} end\n"
                "NotificationManager.GetList = nil\n"
            )
        )
        assert h.export_state(runtime)["end_turn_blocking"] == "ENDTURN_BLOCKING_NOT_ASKABLE"
        success, code, message = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "END_TURN_BLOCKED")
        assert "ENDTURN_BLOCKING_NOT_ASKABLE" in message
        assert h.stub_calls(runtime) == []
        assert runtime.eval("Civ6Stub.turnDeactivated") is False

    def test_the_query_naming_a_blocker_wins_over_an_empty_list(self) -> None:
        runtime = h.load_mod()
        runtime.execute(
            "NotificationManager.GetFirstEndTurnBlocking ="
            " function() return EndTurnBlockingTypes.ENDTURN_BLOCKING_UNITS end"
            "\nCiv6Stub.canEndTurn = false"
        )
        state = h.export_state(runtime)
        assert state["end_turn_blocking"] == "ENDTURN_BLOCKING_UNITS"
        assert "end_turn_blocking_message" not in state
        success, code, message = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "END_TURN_BLOCKED")
        assert message.startswith("End turn blocked: ENDTURN_BLOCKING_UNITS")

    def test_the_message_comes_from_find_end_turn_blocking(self) -> None:
        runtime = h.load_mod()
        _notify(runtime, RESEARCH_BLOCKER)
        runtime.execute(
            "NotificationManager.FindEndTurnBlocking = function(b, p)"
            " return { GetMessage = function() return 'from the query' end } end"
        )
        assert h.export_state(runtime)["end_turn_blocking_message"] == "from the query"

    def test_a_raising_query_falls_back_to_the_list_scan(self) -> None:
        runtime = h.load_mod(
            setup="NotificationManager.GetFirstEndTurnBlocking = function() error('boom') end"
        )
        _notify(runtime, CIVIC_BLOCKER)
        state = h.export_state(runtime)
        assert state["end_turn_blocking"] == "ENDTURN_BLOCKING_CIVIC"
        assert state["end_turn_blocking_message"] == "Choose a civic"
