"""GM-B9 / B4: no attack is offered on a unit or city the exported state does not show.

With ``OBSERVE_ALL_PLAYERS`` false (the default) the state holds no foreign unit or city, so the
exporter offers no ``attack`` and no ``ranged_attack``, even with a visible barbarian or an enemy at
war next to an own unit. With the flag true, a seen hostile neighbour is offered as before. The
executor is unchanged: an explicit ``unit_attack`` on a visible enemy at war is still carried out
with the flag off. Whatever the mode, every offered attack target is the tile of a unit or city in
the same exported state.

Stand-in: own unit 102 at (1, 0), made ranged (range 2); the enemy warrior 201 (player 1, at war)
stands at (2, 0).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import lua_commands as c
from tests.fixtures import lua_harness as h
from tests.fixtures import lua_others as o

#: Hostile things next to unit 102, besides the default enemy warrior at (2, 0).
SCENARIOS = {
    "enemy warrior": "",
    "barbarian": "Civ6Stub.addUnit{ id = 212, owner = 63, x = 0, y = 1 }\n",
    "enemy city": "Civ6Stub.addCity{ id = 401, owner = 1, x = 0, y = 1 }\n",
    "city-state unit at war": "Civ6Stub.addUnit{ id = 211, owner = 10, x = 0, y = 1 }\n",
}
FLAGS = {"flag off (default)": False, "flag on": True}


def _load(scenario: str, *, observe: bool) -> Any:
    return o.load(c.ARCHER + SCENARIOS[scenario], observe=observe)


def _attack_actions(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [a for a in state["legal_actions"] if a["action"] in ("attack", "ranged_attack")]


def _occupied(state: dict[str, Any]) -> set[tuple[int, int]]:
    return {(u["x"], u["y"]) for u in state["units"]} | {(u["x"], u["y"]) for u in state["cities"]}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
class TestFlagOff:
    def test_the_state_shows_no_foreign_unit_or_city(self, scenario: str) -> None:
        state = h.export_state(_load(scenario, observe=False))
        assert state["units"], "premise: own units are exported"
        assert o.foreign_units(state) == []
        assert o.foreign_cities(state) == []

    def test_no_attack_and_no_ranged_attack_is_offered(self, scenario: str) -> None:
        state = h.export_state(_load(scenario, observe=False))
        assert state["legal_actions"], "premise: a state with actions was exported"
        assert any(a["action"] == "move" for a in state["legal_actions"]), "premise: moves offered"
        assert _attack_actions(state) == []


@pytest.mark.parametrize("scenario", list(SCENARIOS))
class TestFlagOn:
    """Offers are as today: a seen hostile neighbour is offered (these pass today)."""

    def test_the_hostile_neighbour_is_offered_to_both_attacks(self, scenario: str) -> None:
        state = h.export_state(_load(scenario, observe=True))
        offered = {a["action"] for a in _attack_actions(state) if a["unit_id"] == 102}
        assert offered == {"attack", "ranged_attack"}
        assert (2, 0) in o.attack_targets(state, 102)


@pytest.mark.parametrize("observe", FLAGS.values(), ids=list(FLAGS))
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_offered_target_is_a_unit_or_city_in_the_same_state(
    scenario: str, observe: bool
) -> None:
    state = h.export_state(_load(scenario, observe=observe))
    shown = _occupied(state)
    for action in _attack_actions(state):
        for x, y in action["targets"]:
            assert (x, y) in shown, f"{action['action']} offered on ({x}, {y}), which is not shown"


@pytest.mark.parametrize("action", ["unit_attack", "unit_ranged_attack"])
def test_control_the_executor_still_attacks_a_visible_enemy_at_war_with_the_flag_off(
    action: str,
) -> None:
    runtime = _load("enemy warrior", observe=False)
    assert h.execute(runtime, o.attack_command(action))[:2] == (True, None)
    c.frame(runtime)
    assert runtime.eval("Civ6Stub.unit(201).damage") > 0
