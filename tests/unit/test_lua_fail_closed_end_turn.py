"""GM-B9 / B1: the end of turn is refused when the game cannot be asked what blocks it.

``GetEndTurnBlocker(playerId)`` returns ``blocker, message, unknown``. ``unknown`` is true when
there is no ``NotificationManager`` or when neither way of asking worked (the game's own
``GetFirstEndTurnBlocking`` missing or raising, and the notification-list scan missing or
raising). Then ``end_turn`` is refused with ``END_TURN_BLOCKED`` and the text
``ENDTURN_BLOCKING_NOT_ASKABLE``, nothing is requested of the game, and the exported state says so.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import lua_harness as h

END_TURN = {"action_type": "end_turn"}
UNKNOWN = "ENDTURN_BLOCKING_NOT_ASKABLE"
RESEARCH_BLOCKER = (
    "Civ6Stub.notifications[0] = { { id = 5,"
    " blocking = EndTurnBlockingTypes.ENDTURN_BLOCKING_RESEARCH,"
    " message = 'Choose a technology', dismissed = false } }\n"
)
FIRST_MISSING = "NotificationManager.GetFirstEndTurnBlocking = nil\n"
FIRST_RAISES = "NotificationManager.GetFirstEndTurnBlocking = function() error('no query') end\n"
SCAN_MISSING = "NotificationManager.GetList = nil\n"
SCAN_RAISES_LIST = "NotificationManager.GetList = function() error('no list') end\n"
SCAN_RAISES_FIND = "NotificationManager.Find = function() error('no find') end\n"

#: Setups in which the game cannot be asked at all.
CANNOT_ASK = {
    "no manager": "NotificationManager = nil\n",
    "query missing, list missing": FIRST_MISSING + SCAN_MISSING,
    "query missing, list raises": FIRST_MISSING + SCAN_RAISES_LIST,
    "query raises, list missing": FIRST_RAISES + SCAN_MISSING,
    "query raises, list raises": FIRST_RAISES + SCAN_RAISES_LIST,
}


def _blocker(runtime: Any) -> tuple[Any, Any, Any]:
    result = runtime.execute("return AlphaZeroStateExporter.GetEndTurnBlocker(0)")
    return (*result, None, None, None)[:3]


def _requested_nothing(runtime: Any) -> bool:
    return (
        "UI.RequestAction" not in h.stub_calls(runtime)
        and runtime.eval("Civ6Stub.turnEnded") is False
        and runtime.eval("Civ6Stub.turnDeactivated") is False
    )


@pytest.fixture(params=list(CANNOT_ASK), ids=list(CANNOT_ASK))
def cannot_ask(request: pytest.FixtureRequest) -> Any:
    return h.load_mod(setup=CANNOT_ASK[request.param])


class TestCannotAsk:
    def test_the_third_value_says_unknown_and_the_first_two_are_nil(self, cannot_ask: Any) -> None:
        assert _blocker(cannot_ask) == (None, None, True)

    def test_end_turn_is_refused_naming_the_unknown_blocker(self, cannot_ask: Any) -> None:
        success, code, message = h.execute(cannot_ask, END_TURN)
        assert (success, code) == (False, "END_TURN_BLOCKED")
        assert UNKNOWN in message

    def test_nothing_is_requested_and_the_turn_does_not_advance(self, cannot_ask: Any) -> None:
        h.execute(cannot_ask, END_TURN)
        assert _requested_nothing(cannot_ask)
        assert cannot_ask.eval("Civ6Stub.turn") == 12

    def test_the_exported_state_carries_the_unknown_blocker(self, cannot_ask: Any) -> None:
        assert h.export_state(cannot_ask)["end_turn_blocking"] == UNKNOWN

    def test_a_blocker_present_but_unaskable_is_still_refused(self) -> None:
        # The notification exists in the game; the mod just cannot see it. Not a pass.
        runtime = h.load_mod(setup=RESEARCH_BLOCKER + "NotificationManager = nil\n")
        assert h.execute(runtime, END_TURN)[:2] == (False, "END_TURN_BLOCKED")
        assert _requested_nothing(runtime)


class TestControls:
    """Where the game CAN be asked, behaviour is as before (these pass today)."""

    def test_nothing_blocking_requests_the_end_of_turn(self) -> None:
        runtime = h.load_mod()
        assert h.execute(runtime, END_TURN) == (True, None, "End turn requested")
        assert h.stub_calls(runtime) == ["UI.RequestAction"]
        assert "end_turn_blocking" not in h.export_state(runtime)

    def test_a_working_first_query_with_nothing_blocking_is_not_unknown(self) -> None:
        runtime = h.load_mod()
        first, second, unknown = _blocker(runtime)
        assert (first, second) == (None, None)
        assert unknown is False

    @pytest.mark.parametrize("scan_setup", [FIRST_MISSING, FIRST_RAISES], ids=["missing", "raises"])
    def test_the_list_scan_finding_nothing_requests_the_end_of_turn(self, scan_setup: str) -> None:
        runtime = h.load_mod(setup=scan_setup)
        assert h.execute(runtime, END_TURN) == (True, None, "End turn requested")
        assert runtime.eval("Civ6Stub.turnEnded") is True
        assert "end_turn_blocking" not in h.export_state(runtime)

    @pytest.mark.parametrize("scan_setup", [FIRST_MISSING, FIRST_RAISES], ids=["missing", "raises"])
    def test_the_list_scan_finding_a_blocker_refuses_it_by_name(self, scan_setup: str) -> None:
        runtime = h.load_mod(setup=scan_setup + RESEARCH_BLOCKER)
        success, code, message = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "END_TURN_BLOCKED")
        assert message.startswith("End turn blocked: ENDTURN_BLOCKING_RESEARCH")
        assert UNKNOWN not in message
        assert h.export_state(runtime)["end_turn_blocking"] == "ENDTURN_BLOCKING_RESEARCH"

    @pytest.mark.parametrize("scan_setup", [FIRST_MISSING, FIRST_RAISES], ids=["missing", "raises"])
    def test_a_working_scan_is_not_unknown(self, scan_setup: str) -> None:
        runtime = h.load_mod(setup=scan_setup)
        assert _blocker(runtime)[2] is False

    def test_a_raising_find_inside_a_working_list_is_a_failed_scan(self) -> None:
        # The scan raised, so neither way of asking worked: unknown, not "nothing blocks".
        runtime = h.load_mod(setup=FIRST_MISSING + RESEARCH_BLOCKER + SCAN_RAISES_FIND)
        assert _blocker(runtime) == (None, None, True)
        assert h.execute(runtime, END_TURN)[:2] == (False, "END_TURN_BLOCKED")
        assert _requested_nothing(runtime)
