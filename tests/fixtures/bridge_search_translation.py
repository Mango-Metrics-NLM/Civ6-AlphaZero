"""Shared pieces for the GM-B6a tests: a converter, a rich state and an independent checker.

``is_offered`` is written from the contract ("every command the translator returns is one
the game offered") and shares no code with ``alpha_zero.bridge.search_translator``: it reads
the raw ``LegalAction`` list directly and knows nothing about boards, scaling or the proxy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv
from alpha_zero.bridge.protocol import GameStateData, LegalAction
from alpha_zero.bridge.state_converter import StateConverter
from tests.fixtures.bridge_search_states import city, make_state, unit

__all__ = [
    "BIG_H",
    "BIG_W",
    "BOARD",
    "build",
    "convert",
    "is_offered",
    "rich_state",
]

#: A real Civ6-sized map scaled onto a small board: one cell is 10.5 x 6.75 game tiles.
BIG_W, BIG_H, BOARD = 84, 54, 8

#: Names the simulator knows: the offered production names of the rich state.
PRODUCE_NAMES = ["UNIT_ARCHER", "UNIT_SETTLER", "UNIT_WARRIOR", "BUILDING_MONUMENT"]


def convert(state: GameStateData, board: int = BOARD) -> dict[str, Any]:
    """The real converter's output for ``state``."""
    return StateConverter(board_size=board, research_registry=False).convert(state)


def build(state: GameStateData, board: int = BOARD, **env_kwargs: Any) -> tuple[Any, Any]:
    """``(converted state, BridgeMCTSEnv)`` for ``state``."""
    converted = convert(state, board)
    return converted, BridgeMCTSEnv(converted, **env_kwargs)


def is_offered(command: Mapping[str, Any], legal: Sequence[LegalAction]) -> bool:
    """Whether the game's own offer list contains ``command``. Independent of the translator."""
    kind = command.get("action_type")
    if kind == "end_turn":
        return any(a.action_type == "end_turn" for a in legal)
    if kind in ("unit_skip", "unit_found_city"):
        wanted = {"unit_skip": "skip", "unit_found_city": "found_city"}[kind]
        return any(a.action_type == wanted and a.unit_id == command.get("unit_id") for a in legal)
    if kind in ("unit_move", "unit_attack"):
        wanted = {"unit_move": "move", "unit_attack": "attack"}[kind]
        target = command.get("target") or {}
        xy = (target.get("x"), target.get("y"))
        return any(
            a.action_type == wanted and a.unit_id == command.get("unit_id") and xy in a.targets
            for a in legal
        )
    if kind == "city_produce":
        name = command.get("item_id")
        return any(
            a.action_type == "produce"
            and a.city_id == command.get("city_id")
            and name in a.items
            and isinstance(name, str)
            and a.item_types.get(name, "unit") == "unit"
            for a in legal
        )
    return False


def rich_state(*, units_first: Sequence[int] = (2, 4)) -> GameStateData:
    """A map with a builder (left out by the proxy), two warriors, a settler, a city, an enemy.

    ``units_first`` orders the two own warriors in the unit list, so the same position can be
    built with the simulator's own-unit order reversed.
    """
    warriors = {2: unit(2, "UNIT_WARRIOR", 0, 40, 27), 4: unit(4, "UNIT_WARRIOR", 0, 20, 10)}
    units = [unit(7, "UNIT_BUILDER", 0, 10, 10)]
    units += [warriors[i] for i in units_first]
    units += [unit(5, "UNIT_SETTLER", 0, 60, 40), unit(3, "UNIT_WARRIOR", 1, 80, 50)]
    legal = [
        LegalAction("end_turn"),
        LegalAction("skip", unit_id=7),
        LegalAction("skip", unit_id=2),
        LegalAction("skip", unit_id=4),
        LegalAction("skip", unit_id=5),
        LegalAction("found_city", unit_id=5),
        LegalAction(
            "move", unit_id=2, targets=[(41, 27), (39, 27), (40, 26), (41, 26), (40, 28), (41, 28)]
        ),
        LegalAction("move", unit_id=4, targets=[(21, 10), (19, 10), (20, 9), (20, 11)]),
        LegalAction("move", unit_id=5, targets=[(61, 40), (59, 40), (60, 39), (60, 41)]),
        LegalAction("move", unit_id=7, targets=[(11, 10), (9, 10)]),
        LegalAction(
            "produce",
            city_id=10,
            items=list(PRODUCE_NAMES),
            item_types={"BUILDING_MONUMENT": "building"},
        ),
    ]
    return make_state(
        width=BIG_W,
        height=BIG_H,
        units=units,
        cities=[city(10, 0, 10, 5), city(11, 1, 70, 5)],
        legal_actions=legal,
    )
