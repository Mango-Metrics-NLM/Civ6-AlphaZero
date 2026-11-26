"""GM-B8a: an attack considers only what the player sees, in the offers and in the executor.

The stand-in's strict attack check is unchanged; the hidden occupant is filtered by the mod. Unit
102 is the player's warrior at (1, 0), made a ranged unit (range 2) so both attacks are in play;
the enemy warrior 201 stands at (2, 0), next to it.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import lua_commands as c
from tests.fixtures import lua_harness as h
from tests.fixtures import lua_others as o

TARGET = (2, 0)
ACTIONS = ["unit_attack", "unit_ranged_attack"]
CITY_ON_TARGET = "Civ6Stub.addCity{ id = 401, owner = 1, x = 2, y = 0 }\n"


def _load(setup: str = "", *, observe: bool = False) -> Any:
    return o.load(c.ARCHER + setup, observe=observe)


def _offered(runtime: Any) -> set[tuple[int, int]]:
    return o.attack_targets(h.export_state(runtime), 102)


def _sent(runtime: Any, action: str) -> tuple[Any, Any, str]:
    success, code, message = h.execute(runtime, o.attack_command(action))
    return success, code, str(message)


def _nothing_requested(runtime: Any) -> bool:
    return c.queued(runtime) == 0 and not c.requests(runtime, "UnitManager.RequestOperation")


@pytest.mark.parametrize("observe", [False, True], ids=["flag off", "flag on"])
class TestControl:
    def test_a_visible_occupant_is_offered_and_the_attack_goes_through(self, observe: bool) -> None:
        # GM-B9 changed the rule: an attack is offered only with OBSERVE_ALL_PLAYERS on (the
        # state shows the foe); with it off nothing is offered but the executor still attacks.
        runtime = _load(observe=observe)
        assert (TARGET in _offered(runtime)) is observe
        for action in ACTIONS:
            assert _sent(runtime, action)[:2] == (True, None)
        c.frame(runtime)
        assert runtime.eval("Civ6Stub.unit(201).damage") > 0


@pytest.mark.parametrize("observe", [False, True], ids=["flag off", "flag on"])
@pytest.mark.parametrize(
    "hide",
    ["Civ6Stub.hideUnit(201)", "Civ6Stub.hideTile(2, 0)", "Civ6Stub.fogTile(2, 0)"],
    ids=["stealth unit", "tile never seen", "tile remembered"],
)
class TestHiddenOccupant:
    def test_it_is_not_offered(self, observe: bool, hide: str) -> None:
        # GM-B9: with the flag off nothing is offered whatever the mod does, so the assertion
        # that depends on the mod runs with the flag on; the control shows it is offered unhidden.
        assert TARGET in _offered(_load(observe=True))
        runtime = _load(hide + "\n", observe=True)
        assert TARGET not in _offered(runtime)

    @pytest.mark.parametrize("action", ACTIONS)
    def test_sent_anyway_it_is_refused_as_having_no_target(
        self, observe: bool, hide: str, action: str
    ) -> None:
        runtime = _load(hide + "\n", observe=observe)
        success, code, message = _sent(runtime, action)
        assert (success, code) == (False, "INVALID_TARGET")
        assert "Nothing to attack" in message
        assert _nothing_requested(runtime)
        c.frame(runtime)
        assert not runtime.eval("Civ6Stub.unit(201).damage")


class TestCities:
    def test_a_unit_hidden_inside_a_visible_foreign_city_is_still_attackable(self) -> None:
        # The city is a foreign occupant the player sees, whatever the unit's stealth.
        # GM-B9 changed the rule: the offer needs OBSERVE_ALL_PLAYERS on; the executor does not.
        setup = CITY_ON_TARGET + "Civ6Stub.hideUnit(201)\n"
        runtime = _load(setup, observe=True)
        assert TARGET in _offered(runtime)
        assert _sent(runtime, "unit_attack")[:2] == (True, None)
        quiet = _load(setup)
        assert TARGET not in _offered(quiet)
        assert _sent(quiet, "unit_attack")[:2] == (True, None)

    @pytest.mark.parametrize("hide", ["hideTile", "fogTile"])
    def test_a_city_on_a_tile_not_visible_now_does_not_make_it_a_target(self, hide: str) -> None:
        setup = CITY_ON_TARGET + f"Civ6Stub.hideUnit(201)\nCiv6Stub.{hide}(2, 0)\n"
        # GM-B9: flag on, or "not offered" would hold whatever the mod does with visibility.
        runtime = _load(setup, observe=True)
        assert TARGET not in _offered(runtime)
        assert TARGET in _offered(_load(CITY_ON_TARGET + "Civ6Stub.hideUnit(201)\n", observe=True))
        assert _sent(runtime, "unit_attack")[:2] == (False, "INVALID_TARGET")
        assert _nothing_requested(runtime)


class TestTheGameCannotSay:
    @pytest.mark.parametrize(
        "lost",
        [
            "Civ6Stub.removeVisibility()",
            "Civ6Stub.removeVisibilityEntry()",
            "Civ6Stub.removeVisibilityMethod('IsUnitVisible')",
            "Civ6Stub.removeVisibilityMethod('IsVisible')",
        ],
        ids=["no table", "no entry", "no IsUnitVisible", "no IsVisible"],
    )
    def test_nothing_is_offered_and_an_attack_is_refused(self, lost: str) -> None:
        # GM-B9: flag on, or "not offered" would hold whatever the mod does with visibility.
        runtime = _load(lost + "\n", observe=True)
        assert TARGET not in _offered(runtime)
        assert _sent(runtime, "unit_attack")[:2] == (False, "INVALID_TARGET")
        assert _nothing_requested(runtime)
