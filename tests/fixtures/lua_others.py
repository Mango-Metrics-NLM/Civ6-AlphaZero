"""Helpers for the GM-B8a tests: other players, visibility and diplomacy in the stand-in game.

The stand-in's hooks are in ``tests/fixtures/lua/civ6_stub_others.lua`` (see its header). The
expected key sets below are written out here, from the contract, never read back from the mod.

Stand-in map: 4 wide, 3 high. Own units 101 (0, 0) and 102 (1, 0), own city 301 at (1, 1); a
default enemy warrior 201 (player 1) at (2, 0). Players: 0 local, 1 a major, 10 a city-state,
63 barbarians.
"""

from __future__ import annotations

from typing import Any

from tests.fixtures import lua_harness as h

OBSERVE = 'AlphaZeroConfig.Set("OBSERVE_ALL_PLAYERS", true)'

#: Exactly what a foreign unit carries (the game does not show the rest).
FOREIGN_UNIT_KEYS = {
    "id",
    "type_name",
    "owner",
    "x",
    "y",
    "hp",
    "max_hp",
    "damage",
    "combat_strength",
    "ranged_strength",
    "range",
    "is_civilian",
}
#: What the exporter writes for an own unit and must not write for a foreign one.
OWN_ONLY_UNIT_KEYS = {"movement", "max_movement", "can_attack", "promotions"}
#: Exactly what a foreign city carries.
FOREIGN_CITY_KEYS = {
    "id",
    "name",
    "owner",
    "x",
    "y",
    "population",
    "is_capital",
    "buildings",
    "districts",
    "available_production",
}
FORBIDDEN_CITY_KEYS = {"current_production", "production_progress", "defense_strength"}

#: A visible foreign warrior of player 1 at (3, 0) and of the city-state at (3, 1).
MAJOR_UNIT = "Civ6Stub.addUnit{ id = 210, owner = 1, x = 3, y = 0, damage = 10, combat = 20 }\n"
MINOR_UNIT = "Civ6Stub.addUnit{ id = 211, owner = 10, x = 3, y = 1 }\n"
BARB_UNIT = "Civ6Stub.addUnit{ id = 212, owner = 63, x = 0, y = 2 }\n"


def load(setup: str = "", *, observe: bool = True) -> Any:
    """The mod in the stand-in; ``OBSERVE_ALL_PLAYERS`` on unless ``observe`` is False."""
    runtime = h.load_mod(setup=setup)
    if observe:
        runtime.execute(OBSERVE)
    return runtime


def state_of(setup: str = "", *, observe: bool = True) -> dict[str, Any]:
    return h.export_state(load(setup, observe=observe))


def foreign_units(state: dict[str, Any], own: int = 0) -> list[dict[str, Any]]:
    return [u for u in state["units"] if u["owner"] != own]


def foreign_cities(state: dict[str, Any], own: int = 0) -> list[dict[str, Any]]:
    return [c for c in state["cities"] if c["owner"] != own]


def unit_ids(state: dict[str, Any]) -> list[int]:
    return [u["id"] for u in state["units"]]


def foreign_unit_ids(state: dict[str, Any]) -> list[int]:
    return sorted(u["id"] for u in foreign_units(state))


def others(state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {entry["id"]: entry for entry in state["other_players"]}


def tile(state: dict[str, Any], x: int, y: int) -> dict[str, Any] | None:
    found = [t for t in state["map"]["tiles"] if t["x"] == x and t["y"] == y]
    return found[0] if found else None


def attack_targets(state: dict[str, Any], unit_id: int) -> set[tuple[int, int]]:
    """Every tile a melee or ranged attack is offered on for ``unit_id``."""
    found: set[tuple[int, int]] = set()
    for action in state["legal_actions"]:
        if action.get("unit_id") == unit_id and action.get("action") in ("attack", "ranged_attack"):
            found.update((t[0], t[1]) for t in action.get("targets", []))
    return found


def attack_command(action: str = "unit_attack", x: int = 2, y: int = 0) -> dict[str, Any]:
    """Unit 102 attacks tile (x, y)."""
    return {"action_type": action, "unit_id": 102, "target": {"x": x, "y": y}}
