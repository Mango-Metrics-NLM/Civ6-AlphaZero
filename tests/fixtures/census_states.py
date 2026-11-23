"""Builders for the expressivity-census tests (GM-B11).

Every state is a real ``GameStateData`` on an unscaled board (``size`` equals the
converter's board size) unless the builder says otherwise, so a game coordinate and a
board cell are the same number and every expected value can be counted by hand.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from alpha_zero.bridge.protocol import GameStateData, LegalAction
from tests.fixtures.bridge_search_states import city, make_state, tile, unit
from tests.fixtures.bridge_snapshots import state_message_dict

__all__ = [
    "ATTEST",
    "SCRIPT",
    "agg",
    "cap_state",
    "cell",
    "chunk_lines",
    "end_turn_only",
    "load_census",
    "mixed_state",
    "read_json",
    "record",
    "run_cli",
    "scaled_terrain_state",
    "settler_state",
    "share",
    "small_terrain_state",
    "state_line",
    "wire",
    "wire_states",
    "write_json",
    "write_log",
]

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "census_search_expressivity.py"


def load_census(name: str = "census_search_expressivity_under_test") -> ModuleType:
    """The script as a module, imported by path as the sibling script tests do."""
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mixed_state(turn_id: int = 3) -> GameStateData:
    """Board 8, unscaled. Own: warrior 1 (3,3), builder 2 (1,1), settler 3 (5,3); city 10.

    Offered (12 atomic commands, end_turn not counted): skip x3 (units 1, 2, 3); move 1 to
    its four neighbours and move 2 to (2,1) = 5; found_city 3 = 1; produce at city 10:
    UNIT_WARRIOR and BUILDING_MONUMENT = 2; research_tech TECH_POTTERY = 1.
    The stand-in leaves the builder out (no builder type), so the builder's skip and move
    are untranslatable. Hand counts, board 8:
    E1 = 8/12 (12 minus builder skip, builder move, monument, research), U = 10/12
    (12 minus monument, research), E2 = 2/3 (units 1 and 3 of {1, 2, 3}), E4 = 1/3.
    """
    legal = [
        LegalAction("end_turn"),
        LegalAction("skip", unit_id=1),
        LegalAction("move", unit_id=1, targets=[(4, 3), (2, 3), (3, 2), (3, 4)]),
        LegalAction("skip", unit_id=2),
        LegalAction("move", unit_id=2, targets=[(2, 1)]),
        LegalAction("skip", unit_id=3),
        LegalAction("found_city", unit_id=3),
        LegalAction(
            "produce",
            city_id=10,
            items=["UNIT_WARRIOR", "BUILDING_MONUMENT"],
            item_types={"BUILDING_MONUMENT": "building"},
        ),
        LegalAction("research_tech", items=["TECH_POTTERY"]),
    ]
    units = [
        unit(1, "UNIT_WARRIOR", 0, 3, 3),
        unit(2, "UNIT_BUILDER", 0, 1, 1),
        unit(3, "UNIT_SETTLER", 0, 5, 3),
        unit(9, "UNIT_WARRIOR", 1, 7, 7),
    ]
    return make_state(
        size=8, units=units, cities=[city(10, 0, 0, 7)], legal_actions=legal, turn_id=turn_id
    )


def end_turn_only(turn_id: int = 3) -> GameStateData:
    """Nothing offered but end_turn and no units: every E has a zero denominator except E3."""
    return make_state(size=8, turn_id=turn_id)


def small_terrain_state() -> GameStateData:
    """Board 8 unscaled; five tiles, so five cells received a tile (all others did not).

    (0,0) grassland, (1,0) mountain, (2,0) hills, (3,0) coast, (4,0) ocean.
    Land cells = 3 (coast and ocean are water); mountain cells = 1; E3 = 1/3.
    The other 59 cells hold the converter's default plains and must not be counted.
    """
    tiles = [
        tile(0, 0, "TERRAIN_GRASS"),
        tile(1, 0, "TERRAIN_GRASS_MOUNTAIN"),
        tile(2, 0, "TERRAIN_GRASS_HILLS"),
        tile(3, 0, "TERRAIN_COAST"),
        tile(4, 0, "TERRAIN_OCEAN"),
    ]
    return make_state(size=8, tiles=tiles)


def scaled_terrain_state() -> GameStateData:
    """An 84x54 map on board 8 (scale 8/84 in x): round(x * 8 / 84) is the cell column.

    Tiles (11,0) and (12,0) both land in cell (1,0) (1.05 and 1.14 round to 1); (0,0) is
    cell (0,0); a mountain at (30,0) is cell (3,0) (2.86 -> 3); an ocean at (40,0) is cell
    (4,0) (3.81 -> 4). Cells with a tile: 4; land cells: 3; mountain cells: 1 -> E3 = 1/3.
    Counting tiles instead of cells would give 1/4.
    """
    tiles = [
        tile(0, 0, "TERRAIN_GRASS"),
        tile(11, 0, "TERRAIN_GRASS"),
        tile(12, 0, "TERRAIN_GRASS"),
        tile(30, 0, "TERRAIN_GRASS_MOUNTAIN"),
        tile(40, 0, "TERRAIN_OCEAN"),
    ]
    return make_state(width=84, height=54, tiles=tiles)


def cap_state(n_units: int = 22) -> GameStateData:
    """``n_units`` own warriors on an 8x8 board, each offered one move to a free tile."""
    units, legal = [], [LegalAction("end_turn")]
    for i in range(1, n_units + 1):
        x, y = (i - 1) % 8, 3 * ((i - 1) // 8)
        units.append(unit(i, "UNIT_WARRIOR", 0, x, y))
        legal.append(LegalAction("move", unit_id=i, targets=[(x, y + 1)]))
    return make_state(size=8, units=units, legal_actions=legal)


def wire(state: GameStateData, *, schema: int = 2, observes: bool | None = True) -> dict[str, Any]:
    """The bare state payload as the mod writes it (``map`` not ``map_data``)."""
    state.schema_version = schema
    state.observes_others = observes
    return dict(state_message_dict(state)["state"])


def state_line(payload: dict[str, Any], *, context: bool = True) -> str:
    """One ``AZ_STATE`` log line, with or without the game's context prefix."""
    return ("AlphaZeroMain: " if context else "") + "AZ_STATE:" + json.dumps(payload)


def chunk_lines(payload: dict[str, Any], pieces: int = 3) -> list[str]:
    """A state sent in pieces: ``AZ_STATE_CHUNK:i/n:`` lines then ``AZ_STATE_END:n``."""
    text = json.dumps(payload)
    size = len(text) // pieces + 1
    parts = [text[i : i + size] for i in range(0, len(text), size)]
    lines = [f"AZ_STATE_CHUNK:{i + 1}/{len(parts)}:{p}" for i, p in enumerate(parts)]
    return [*lines, f"AZ_STATE_END:{len(parts)}"]


def write_log(path: Path, lines: Sequence[str]) -> Path:
    """Write ``lines`` as a ``Lua.log``."""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_json(path: Path, value: Any) -> Path:
    """Write ``value`` as JSON."""
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


# -- hand-written aggregates and per-state records (no state is converted) ------------------


def share(value: float | None, numerator: int = 0, denominator: int = 0) -> dict[str, Any]:
    """A ``{"value", "numerator", "denominator"}`` share as the script stores it."""
    return {"value": value, "numerator": numerator, "denominator": denominator}


def cell(
    e1: float | None = 0.5,
    u: float | None = 0.6,
    e2: float | None = 0.5,
    e3: float | None = 0.5,
    e4: float | None = 0.5,
    by_kind: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """One board's non-void measurement with the given values."""
    kinds = by_kind if by_kind is not None else {"unit_move": (1, 2)}
    e1_cell = share(e1)
    e1_cell["upper_bound_u"] = share(u)
    e1_cell["by_kind"] = {k: share(n / d, n, d) for k, (n, d) in kinds.items()}
    return {
        "void": False,
        "E1": e1_cell,
        "E2": share(e2),
        "E3": share(e3),
        "E4": share(e4),
    }


def record(source: int, turn: int, board: str = "8", **values: Any) -> dict[str, Any]:
    """A sampled state's record; ``values`` go to :func:`cell`; ``void=True`` voids it."""
    void = values.pop("void", False)
    body: dict[str, Any] = {"void": True} if void else cell(**values)
    return {"source": source, "turn": turn, "boards": {board: body}}


def agg(
    median: float | None, low: float | None = None, high: float | None = None
) -> dict[str, Any]:
    """An aggregate as ``aggregate`` returns it, reduced to what the stop rule reads."""
    ci = None if low is None or high is None else [low, high]
    return {"median": median, "median_ci95": ci}


#: The two attestations a census result needs (the script cannot tell a real log from a fake).
ATTEST = ("--mod-version", "0.9.1", "--played-by", "greedy policy")


def wire_states(
    turns: Sequence[int], *, schema: int = 2, observes: bool | None = True
) -> list[dict[str, Any]]:
    """The hand-counted ``mixed_state`` at each of ``turns``, as the mod writes it."""
    return [wire(mixed_state(t), schema=schema, observes=observes) for t in turns]


def run_cli(mod: ModuleType, tmp: Path, *argv: str) -> tuple[int, Path]:
    """``main`` with ``--out tmp/out``; returns the exit code and the census.json path."""
    out = tmp / "out"
    return mod.main([*argv, "--out", str(out)]), out / "census.json"


def read_json(path: Path) -> dict[str, Any]:
    """A written census.json."""
    return json.loads(path.read_text(encoding="utf-8"))


def settler_state(
    turn_id: int = 1, *, builder: bool = False, own_city: bool = False, enemy_city: bool = False
) -> GameStateData:
    """Board 8, unscaled: an own settler (3,3) with nothing left out of the stand-in.

    ``builder`` adds an own builder (the stand-in has no builder: one unit left out);
    ``own_city`` an own city; ``enemy_city`` the other player's city.
    """
    units = [unit(1, "UNIT_SETTLER", 0, 3, 3)]
    legal = [LegalAction("end_turn"), LegalAction("found_city", unit_id=1)]
    if builder:
        units.append(unit(2, "UNIT_BUILDER", 0, 1, 1))
        legal.append(LegalAction("skip", unit_id=2))
    cities = []
    if own_city:
        cities.append(city(10, 0, 0, 7))
    if enemy_city:
        cities.append(city(11, 1, 7, 0))
    return make_state(size=8, units=units, cities=cities, legal_actions=legal, turn_id=turn_id)
