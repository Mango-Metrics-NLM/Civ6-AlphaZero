"""Damaged-but-JSON-valid states for the census strictness tests (review of #169).

``GameStateData.from_dict`` drops what it cannot parse. Each builder here starts from the
hand-counted ``mixed_state`` as the mod writes it and adds entries the parser drops (the
rest of the state stays valid and measurable). Each added entry is named by what makes the
parser raise on it.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

from tests.fixtures.census_states import mixed_state, wire

__all__ = ["DAMAGE", "action", "damaged", "expected_loss", "plain"]


def plain(turn: int = 3) -> dict[str, Any]:
    """A valid wire state, round-tripped through JSON (tuples become lists, as in a log)."""
    return json.loads(json.dumps(wire(mixed_state(turn))))


def action(raw: dict[str, Any], kind: str, **where: Any) -> dict[str, Any]:
    """The first raw legal action of ``kind`` whose fields include ``where``."""
    for a in raw["legal_actions"]:
        if a.get("action_type") == kind and all(a.get(k) == v for k, v in where.items()):
            return a  # type: ignore[no-any-return]
    raise AssertionError(f"no {kind} action {where} in the fixture")


def _tile(raw: dict[str, Any]) -> None:
    raw["map"]["tiles"] += [{"x": "not a number", "y": 0}, 5]  # bad coordinate; not an object


def _unit(raw: dict[str, Any]) -> None:
    raw["units"].append({"id": 50, "unit_type": "UNIT_WARRIOR", "owner_id": 1, "health": -5})


def _city(raw: dict[str, Any]) -> None:
    raw["cities"].append({"id": 60, "name": "Nowhere", "owner_id": 1, "population": 0})


def _player(raw: dict[str, Any]) -> None:
    raw["players"].append({"id": 7, "gold": "plenty"})  # a gold that is not a number


def _legal_action(raw: dict[str, Any]) -> None:
    raw["legal_actions"].append(7)  # not an object


def _target(raw: dict[str, Any]) -> None:
    # a one-element pair and a string: neither is {x, y} nor [x, y]
    action(raw, "move", unit_id=1)["targets"] += [[4], "bad"]


def _item(raw: dict[str, Any]) -> None:
    # a null and a fractional number are neither an id nor a name
    action(raw, "produce", city_id=10)["items"] += [None, 1.5]


#: kind -> (builder that damages a copy of a valid state, what the parser lost, by key)
DAMAGE: dict[str, tuple[Callable[[dict[str, Any]], None], dict[str, int]]] = {
    "tile": (_tile, {"tiles": 2}),
    "unit": (_unit, {"units": 1}),
    "city": (_city, {"cities": 1}),
    "player": (_player, {"players": 1}),
    "legal_action": (_legal_action, {"legal_actions": 1}),
    "target": (_target, {"legal_action_targets": 2}),
    "item": (_item, {"legal_action_items": 2}),
}


def damaged(*kinds: str, turn: int = 3) -> dict[str, Any]:
    """A valid state with every damage in ``kinds`` applied."""
    raw = plain(turn)
    for kind in kinds:
        DAMAGE[kind][0](raw)
    return raw


def expected_loss(*kinds: str) -> dict[str, int]:
    """What ``parser_losses`` must report for ``damaged(*kinds)``."""
    out: dict[str, int] = {}
    for kind in kinds:
        out.update(DAMAGE[kind][1])
    return copy.deepcopy(out)
