"""GM-B9 (M3): ``ACTION_ENDTURN`` ends the LOCAL player's turn, so ``end_turn`` is refused when
the player the bridge plays is not the local player.

``Game.GetLocalPlayer()`` answering a number >= 0 that differs from the bridge's player id:
``EXECUTION_FAILED``, the message names both ids, nothing is requested, the turn does not end and
is not deactivated. Equal ids, a missing or raising ``GetLocalPlayer`` and ``-1`` leave the end of
turn requested as before.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from tests.fixtures import lua_harness as h

END_TURN = {"action_type": "end_turn"}


def _load(setup: str = "") -> Any:
    return h.load_mod(setup=setup)


def _nothing_requested(runtime: Any) -> bool:
    return (
        "UI.RequestAction" not in h.stub_calls(runtime)
        and runtime.eval("Civ6Stub.turnEnded") is False
        and runtime.eval("Civ6Stub.turnDeactivated") is False
    )


@pytest.mark.parametrize(("local", "bridge"), [(0, 1), (3, 0), (2, 5)])
class TestDifferingIds:
    def test_the_end_of_turn_is_refused_naming_both_ids(self, local: int, bridge: int) -> None:
        runtime = _load(f"Civ6Stub.localPlayerId = {local}\n")
        success, code, message = h.execute(runtime, END_TURN, player_id=bridge)
        assert (success, code) == (False, "EXECUTION_FAILED")
        for player in (local, bridge):
            assert re.search(rf"\b{player}\b", str(message)), message

    def test_nothing_is_requested_and_the_turn_does_not_end(self, local: int, bridge: int) -> None:
        runtime = _load(f"Civ6Stub.localPlayerId = {local}\n")
        h.execute(runtime, END_TURN, player_id=bridge)
        assert _nothing_requested(runtime)
        assert runtime.eval("Civ6Stub.turn") == 12


class TestControls:
    """The end of turn is requested as before (these pass today)."""

    def test_equal_ids(self) -> None:
        runtime = _load()
        assert h.execute(runtime, END_TURN, player_id=0) == (True, None, "End turn requested")
        assert runtime.eval("Civ6Stub.turnEnded") is True

    def test_equal_ids_for_a_local_player_other_than_zero(self) -> None:
        runtime = _load("Civ6Stub.localPlayerId = 3\n")
        assert h.execute(runtime, END_TURN, player_id=3)[:2] == (True, None)
        assert runtime.eval("Civ6Stub.turnEnded") is True

    @pytest.mark.parametrize(
        "setup",
        [
            "Game.GetLocalPlayer = nil",
            "Game.GetLocalPlayer = function() error('not in a game') end",
            "Game.GetLocalPlayer = function() return -1 end",
            "Game.GetLocalPlayer = function() return nil end",
            "Game.GetLocalPlayer = function() return '0' end",
        ],
        ids=["missing", "raises", "minus one", "nil", "a string"],
    )
    def test_no_usable_local_player_asking_for_a_player_other_than_it(self, setup: str) -> None:
        runtime = _load(setup)
        assert h.execute(runtime, END_TURN, player_id=1) == (True, None, "End turn requested")
        assert runtime.eval("Civ6Stub.turnEnded") is True

    def test_a_blocker_is_still_reported_first_for_equal_ids(self) -> None:
        runtime = _load(
            "Civ6Stub.notifications[0] = { { id = 5,"
            " blocking = EndTurnBlockingTypes.ENDTURN_BLOCKING_RESEARCH, dismissed = false } }"
        )
        assert h.execute(runtime, END_TURN, player_id=0)[:2] == (False, "END_TURN_BLOCKED")
        assert _nothing_requested(runtime)
