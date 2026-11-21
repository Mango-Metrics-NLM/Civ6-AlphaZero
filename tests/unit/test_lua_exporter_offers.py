"""GM-B7b (C1, L1-L4): the exporter offers what the game accepts, executed against the stand-in.

Lua 5.1 through lupa; the game is ``tests/fixtures/lua/civ6_stub*.lua``. The stand-in is the
only judge of "accepted": ``AlphaZeroActionExecutor.Check`` (the executor's dry run) asks it the
way ``Execute`` does, and the exporter's offer has to equal what ``Check`` accepts.

Expected neighbours are written out here in Python (odd rows sit half a tile to the right), never
read back from the stand-in. Tests marked CONTROL pass on the tree before the slice and pin what
must not change; every other test fails on that tree for the reason in its docstring.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures import lua_commands as c
from tests.fixtures import lua_harness as h

GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "exporter_default_state_pre_b7b.json"
MAP_W, MAP_H = 4, 3
MOUNTAIN = (3, 2)
TILES = [(x, y) for x in range(MAP_W) for y in range(MAP_H)]

SECOND_ENEMY = (
    "table.insert(Civ6Stub.units, { id = 202, owner = 1, x = 3, y = 0, unitType = 1,"
    " moves = 2, combat = 20, attacks = 1 })\n"
)
MOVE_102_TO_ODD_ROW = "Civ6Stub.unit(102).x, Civ6Stub.unit(102).y = 2, 1\n"
ALL_NOW = (
    "Civ6Stub.cities[301].canProduce = { UNIT_WARRIOR = 'now', UNIT_SETTLER = 'now',"
    " UNIT_ARCHER = 'now', BUILDING_MONUMENT = 'now', BUILDING_GRANARY = 'now',"
    " DISTRICT_CAMPUS = 'now', DISTRICT_CITY_CENTER = 'now' }\n"
)
GRANARY_NEEDS_A_PLOT = (
    "for r in GameInfo.Buildings() do"
    " if r.BuildingType == 'BUILDING_GRANARY' then r.RequiresPlacement = true end end\n"
)

#: Games the offer is held against: the default one and variations of it.
SCENARIOS: dict[str, str] = {
    "default": "",
    "archer": c.ARCHER,
    "archer_two_enemies": c.ARCHER + SECOND_ENEMY,
    "peace": c.ARCHER + "Civ6Stub.setPeace(0, 1, true)\n",
    "odd_row_warrior": MOVE_102_TO_ODD_ROW,
    "odd_row_archer": c.ARCHER + MOVE_102_TO_ODD_ROW + SECOND_ENEMY,
    "fortified": "Civ6Stub.unit(102).activity = 'fortified'\n",
    "warrior_without_moves": "Civ6Stub.unit(102).moves = 0\n",
    "warrior_without_attacks": "Civ6Stub.unit(102).attacks = 0\n",
    "settler_cannot_found": "Civ6Stub.canFoundCity = false\n",
    "settler_in_the_city": "Civ6Stub.unit(101).x, Civ6Stub.unit(101).y = 1, 1\n",
    "range_without_strength": "Civ6Stub.unit(102).range = 2\n",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def neighbours(x: int, y: int) -> list[tuple[int, int]]:
    """The six tiles around (x, y), on or off the map. Odd rows sit half a tile to the right."""
    if y % 2 == 0:
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x - 1, y + 1), (x, y - 1), (x - 1, y - 1)]
    return [(x + 1, y), (x - 1, y), (x + 1, y + 1), (x, y + 1), (x + 1, y - 1), (x, y - 1)]


def on_map(tile: tuple[int, int]) -> bool:
    return 0 <= tile[0] < MAP_W and 0 <= tile[1] < MAP_H


def expected_moves(
    x: int, y: int, occupied: set[tuple[int, int]], impassable: set[tuple[int, int]] | None = None
) -> set[tuple[int, int]]:
    blocked = {MOUNTAIN} if impassable is None else impassable
    return {t for t in neighbours(x, y) if on_map(t) and t not in occupied and t not in blocked}


def load(setup: str = "") -> Any:
    return h.load_mod(setup=setup)


def check(runtime: Any, command: dict[str, Any], player_id: int = 0) -> tuple[Any, Any, Any]:
    """``AlphaZeroActionExecutor.Check``: ``(success, error_code, message)``."""
    dry_run = runtime.eval("AlphaZeroActionExecutor.Check")
    assert dry_run is not None, "AlphaZeroActionExecutor.Check does not exist (GM-B7b C1)"
    decoded = runtime.eval("AlphaZeroJson.Decode")(json.dumps(command))
    table = decoded[0] if isinstance(decoded, tuple) else decoded
    result = dry_run(int(player_id), table)
    values = result if isinstance(result, tuple) else (result,)
    success, code, message = (*values, None, None)[:3]
    return success, code, message


def execute(runtime: Any, command: dict[str, Any]) -> tuple[Any, Any, Any]:
    return h.execute(runtime, command)


def actions(state: dict[str, Any], name: str, **match: Any) -> list[dict[str, Any]]:
    return [
        a
        for a in state["legal_actions"]
        if a["action"] == name and all(a.get(k) == v for k, v in match.items())
    ]


def targets_of(state: dict[str, Any], name: str, unit_id: int) -> set[tuple[int, int]]:
    found: set[tuple[int, int]] = set()
    for action in actions(state, name, unit_id=unit_id):
        found |= {(t[0], t[1]) for t in action["targets"]}
    return found


def unit_actions(state: dict[str, Any], unit_id: int) -> list[str]:
    return sorted(a["action"] for a in state["legal_actions"] if a.get("unit_id") == unit_id)


UNIT_COMMANDS = {
    "move": "unit_move",
    "attack": "unit_attack",
    "ranged_attack": "unit_ranged_attack",
    "skip": "unit_skip",
    "fortify": "unit_fortify",
    "found_city": "unit_found_city",
}
Offer = tuple[str, int, tuple[int, int] | None]


def command_for(kind: str, unit_id: int, target: tuple[int, int] | None) -> dict[str, Any]:
    command: dict[str, Any] = {"action_type": UNIT_COMMANDS[kind], "unit_id": unit_id}
    if target is not None:
        command["target"] = {"x": target[0], "y": target[1]}
    return command


def offered_unit_actions(state: dict[str, Any]) -> set[Offer]:
    offered: set[Offer] = set()
    for action in state["legal_actions"]:
        if action["action"] not in UNIT_COMMANDS:
            continue
        for target in action.get("targets") or [None]:
            tile = None if target is None else (target[0], target[1])
            offered.add((action["action"], action["unit_id"], tile))
    return offered


def candidates(state: dict[str, Any]) -> Iterator[Offer]:
    """Every unit action the stand-in could be asked about, written without the exporter."""
    for unit in state["units"]:
        if unit["owner"] != 0:
            continue
        uid, pos = unit["id"], (unit["x"], unit["y"])
        for tile in neighbours(*pos):
            if on_map(tile):
                yield ("move", uid, tile)
                yield ("attack", uid, tile)
        for tile in TILES:
            yield ("ranged_attack", uid, tile)
        for kind in ("skip", "fortify", "found_city"):
            yield (kind, uid, None)


# ---------------------------------------------------------------------------
# C1: the dry run
# ---------------------------------------------------------------------------
CHECK_ITEMS = [
    "UNIT_WARRIOR",
    "UNIT_SETTLER",
    "UNIT_ARCHER",
    "BUILDING_MONUMENT",
    "BUILDING_GRANARY",
    "DISTRICT_CAMPUS",
    "NOT_A_TYPE",
]


def property_commands() -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for unit_id in (101, 102, 201, 999):
        for x, y in TILES:
            for kind in ("move", "attack", "ranged_attack"):
                commands.append(command_for(kind, unit_id, (x, y)))
        for name in ("unit_skip", "unit_fortify", "unit_sleep", "unit_found_city"):
            commands.append({"action_type": name, "unit_id": unit_id})
    for item in CHECK_ITEMS:
        commands.append({"action_type": "city_produce", "city_id": 301, "item_id": item})
    commands.append({"action_type": "city_produce", "city_id": 301, "item_id": "UNIT_WARRIOR"})
    commands.append({"action_type": "city_produce", "city_id": 999, "item_id": "UNIT_WARRIOR"})
    commands.append({"action_type": "city_produce", "city_id": 301, "item_id": 1})
    commands.append({"action_type": "unit_move", "unit_id": 102})
    commands.append({"action_type": "city_produce", "city_id": 301})
    return commands


class TestCheckIsADryRun:
    @pytest.mark.parametrize(
        "scenario", ["default", "archer", "peace", "settler_cannot_found", "odd_row_archer"]
    )
    def test_check_agrees_with_execute_on_a_fresh_copy(self, scenario: str) -> None:
        """Fails on the current tree: ``AlphaZeroActionExecutor.Check`` does not exist."""
        setup = SCENARIOS[scenario]
        checker = load(setup)
        outcomes: set[Any] = set()
        mismatches = []
        for command in property_commands():
            got = check(checker, command)
            expected = execute(load(setup), command)
            outcomes.add((got[0], got[1]))
            if got[:2] != expected[:2]:
                mismatches.append((command, got[:2], expected[:2]))
        assert not mismatches, mismatches[:5]
        # Non-vacuity: the commands reach acceptance and several different refusals.
        assert (True, None) in outcomes
        assert len({code for ok, code in outcomes if not ok}) >= 4, outcomes

    @pytest.mark.parametrize(
        "command",
        [
            c.unit_move(102, 0, 1),
            c.unit_move(102, 2, 0),
            command_for("attack", 102, (2, 0)),
            command_for("ranged_attack", 102, (2, 0)),
            {"action_type": "unit_found_city", "unit_id": 101},
            {"action_type": "city_produce", "city_id": 301, "item_id": "UNIT_SETTLER"},
            {"action_type": "city_produce", "city_id": 301, "item_id": "UNIT_WARRIOR"},
            {"action_type": "unit_skip", "unit_id": 999},
        ],
        ids=["move", "move-onto-enemy", "attack", "ranged", "found", "settler", "warrior", "ghost"],
    )
    def test_check_returns_what_execute_returns_message_included(
        self, command: dict[str, Any]
    ) -> None:
        """Fails on the current tree: ``Check`` does not exist."""
        setup = c.ARCHER
        assert check(load(setup), command) == execute(load(setup), command)

    def test_check_requests_nothing_and_changes_nothing(self) -> None:
        """Fails on the current tree: ``Check`` does not exist."""
        runtime = load(c.ARCHER)
        before = h.export_state(runtime)
        accepted = 0
        for command in property_commands():
            accepted += 1 if check(runtime, command)[0] is True else 0
        assert accepted >= 8, "nothing was accepted; the no-request claim below is vacuous"
        assert h.stub_calls(runtime) == []
        assert c.queued(runtime) == 0
        assert int(runtime.eval("#Civ6Stub.pendingRequests")) == 0
        assert c.dropped(runtime) == 0
        assert runtime.eval("Civ6Stub.applyRequests()") == 0
        assert h.export_state(runtime) == before

    def test_an_accepted_check_is_not_a_move(self) -> None:
        """Fails on the current tree: ``Check`` does not exist."""
        runtime = load()
        assert check(runtime, c.unit_move(102, 0, 1))[:2] == (True, None)
        runtime.execute("Civ6Stub.applyRequests()")
        # (corrected: Civ6Stub.unit returns the record and its index, read the fields in Lua)
        where = runtime.eval("Civ6Stub.unit(102).x"), runtime.eval("Civ6Stub.unit(102).y")
        assert (*where, runtime.eval("Civ6Stub.unit(102).moves")) == (1, 0, 2)
        assert runtime.eval("Civ6Stub.city(301).building") is None
        assert check(
            runtime, {"action_type": "city_produce", "city_id": 301, "item_id": "UNIT_WARRIOR"}
        )[:2] == (True, None)
        runtime.execute("Civ6Stub.applyRequests()")
        assert runtime.eval("Civ6Stub.city(301).building") is None

    @pytest.mark.parametrize(
        "action",
        [
            "end_turn",
            "research_tech",
            "research_civic",
            "city_buy",
            "city_citizen",
            "unit_build",
            "unit_promote",
            "unit_delete",
            "diplomacy",
            "no_such_action",
        ],
    )
    def test_any_other_action_type_is_not_implemented(self, action: str) -> None:
        """Fails on the current tree: ``Check`` does not exist."""
        runtime = load()
        command = {"action_type": action, "unit_id": 102, "item_id": "TECH_WRITING"}
        success, code, message = check(runtime, command)
        assert (success, code) == (False, "NOT_IMPLEMENTED")
        assert isinstance(message, str) and message
        assert h.stub_calls(runtime) == [] and c.queued(runtime) == 0
        assert int(runtime.eval("#Civ6Stub.notifications[0]")) == 0

    def test_a_command_without_an_action_type_is_refused(self) -> None:
        """Fails on the current tree: ``Check`` does not exist."""
        success, code, _ = check(load(), {"unit_id": 102})
        assert success is False
        assert code == execute(load(), {"unit_id": 102})[1]


# ---------------------------------------------------------------------------
# L1: neighbours come from the game (E5)
# ---------------------------------------------------------------------------
class TestNeighboursComeFromTheGame:
    def test_the_python_adjacency_matches_the_contract(self) -> None:
        """CONTROL: pins the helper these tests use (no mod code involved)."""
        assert expected_moves(1, 0, {(2, 0), (0, 0)}) == {(1, 1), (0, 1)}
        assert expected_moves(2, 1, {(0, 0), (2, 0)}) == {(3, 1), (1, 1), (2, 2), (3, 0)}
        assert sorted(neighbours(2, 1)) == [(1, 1), (2, 0), (2, 2), (3, 0), (3, 1), (3, 2)]

    @pytest.mark.parametrize("tile", [t for t in TILES if t not in {(0, 0), (2, 0), MOUNTAIN}])
    def test_the_warriors_moves_are_the_free_passable_neighbours(
        self, tile: tuple[int, int]
    ) -> None:
        """E5. Fails on the current tree on odd rows: the exporter's own offset table is wrong
        there (it offers a tile that is not a neighbour and an occupied one)."""
        runtime = load(f"Civ6Stub.unit(102).x, Civ6Stub.unit(102).y = {tile[0]}, {tile[1]}\n")
        state = h.export_state(runtime)
        expected = expected_moves(tile[0], tile[1], {(0, 0), (2, 0)})
        assert targets_of(state, "move", 102) == expected
        assert expected, "the unit has nowhere to go; the equality above is vacuous"

    def test_a_neighbour_the_old_table_missed_is_offered(self) -> None:
        """E5. Fails on the current tree: (3,2) is a neighbour of (2,1) the hand table omits.
        The stand-in's mountain is made passable so it can be offered."""
        setup = MOVE_102_TO_ODD_ROW + "Map.GetPlot(3, 2).IsImpassable = function() return false end"
        state = h.export_state(load(setup))
        assert (3, 2) in neighbours(2, 1)
        assert targets_of(state, "move", 102) == {(3, 1), (1, 1), (2, 2), (3, 0), (3, 2)}

    def test_a_tile_that_is_not_adjacent_is_never_offered(self) -> None:
        """E5. Fails on the current tree: from (2,1) the old table offers (1,0)."""
        state = h.export_state(load(MOVE_102_TO_ODD_ROW))
        moves = targets_of(state, "move", 102)
        assert (1, 0) not in moves
        assert moves <= {t for t in neighbours(2, 1) if on_map(t)}
        assert moves

    def test_the_settlers_only_move_is_the_free_neighbour(self) -> None:
        """Fails on the current tree: it is also offered (1,1), which is not adjacent to (0,0),
        and (1,0), where the warrior stands."""
        state = h.export_state(load())
        assert targets_of(state, "move", 101) == {(0, 1)}

    def test_without_the_games_adjacency_no_move_is_offered(self) -> None:
        """E5. Fails on the current tree: a hand table is used as a fallback."""
        healthy = h.export_state(load())
        assert actions(healthy, "move"), "control: the healthy game offers moves"
        state = h.export_state(load("Map.GetAdjacentPlot = nil"))
        assert actions(state, "move") == []
        assert actions(state, "skip", unit_id=101) and actions(state, "found_city", unit_id=101)
        assert actions(state, "skip", unit_id=102)

    def test_a_game_that_answers_nil_everywhere_offers_no_move(self) -> None:
        """E5. Fails on the current tree: the hand table is used whatever the game says."""
        state = h.export_state(load("Map.GetAdjacentPlot = function() return nil end"))
        assert actions(state, "move") == []
        assert actions(state, "skip", unit_id=102)


# ---------------------------------------------------------------------------
# L2: every offered unit action is one the executor's check accepts (E6)
# ---------------------------------------------------------------------------
class TestOfferedUnitActionsAreAccepted:
    @pytest.mark.parametrize("scenario", sorted(SCENARIOS))
    def test_every_offered_action_passes_check(self, scenario: str) -> None:
        """E6. Fails on the current tree: ``Check`` does not exist."""
        runtime = load(SCENARIOS[scenario])
        state = h.export_state(runtime)
        offered = offered_unit_actions(state)
        refused = [
            (o, check(runtime, command_for(*o))[1:])
            for o in sorted(offered, key=str)
            if check(runtime, command_for(*o))[0] is not True
        ]
        assert not refused, refused
        assert offered, "nothing offered; the check above is vacuous"

    @pytest.mark.parametrize("scenario", sorted(SCENARIOS))
    def test_the_offer_is_exactly_the_accepted_set(self, scenario: str) -> None:
        """E6. Fails on the current tree: ``Check`` does not exist."""
        runtime = load(SCENARIOS[scenario])
        state = h.export_state(runtime)
        accepted = {o for o in candidates(state) if check(runtime, command_for(*o))[0] is True}
        offered = offered_unit_actions(state)
        assert offered - accepted == set(), "offered but not accepted"
        assert accepted - offered == set(), "accepted but not offered"
        assert accepted, "nothing accepted; the equality is vacuous"

    def test_the_accepted_set_covers_every_kind_somewhere(self) -> None:
        """Fails on the current tree: ``Check`` does not exist. Non-vacuity for the reverse
        direction: across the scenarios each kind of action is accepted at least once."""
        seen: set[str] = set()
        for scenario in ("default", "archer", "archer_two_enemies"):
            runtime = load(SCENARIOS[scenario])
            state = h.export_state(runtime)
            seen |= {o[0] for o in candidates(state) if check(runtime, command_for(*o))[0] is True}
        assert seen == set(UNIT_COMMANDS)

    def test_a_move_never_targets_another_unit_or_the_mountain(self) -> None:
        """E6. Fails on the current tree: the enemy's tile (2,0) and the settler's (0,0) are
        offered as moves of the warrior."""
        state = h.export_state(load())
        moves = targets_of(state, "move", 102)
        assert moves == {(1, 1), (0, 1)}
        assert not moves & {(2, 0), (0, 0)}
        odd = h.export_state(load(MOVE_102_TO_ODD_ROW))
        assert MOUNTAIN in neighbours(2, 1), "the mountain must be adjacent for this to matter"
        assert MOUNTAIN not in targets_of(odd, "move", 102)

    def test_a_unit_with_no_moves_is_offered_nothing(self) -> None:
        """E6. Fails on the current tree: it is still offered an attack."""
        state = h.export_state(load(SCENARIOS["warrior_without_moves"]))
        assert unit_actions(state, 102) == []
        assert unit_actions(state, 101), "control: the settler still has moves"

    def test_a_unit_with_no_attacks_is_offered_no_attack(self) -> None:
        """CONTROL: passes on the current tree."""
        state = h.export_state(load(SCENARIOS["warrior_without_attacks"]))
        assert unit_actions(state, 102) == ["fortify", "move", "skip"]

    def test_a_fortified_unit_is_not_offered_fortify_again(self) -> None:
        """E6. Fails on the current tree: fortify is offered to every non-civilian."""
        healthy = h.export_state(load())
        assert actions(healthy, "fortify", unit_id=102), "control: a free warrior may fortify"
        state = h.export_state(load(SCENARIOS["fortified"]))
        assert actions(state, "fortify", unit_id=102) == []
        assert unit_actions(state, 102) == ["attack", "move", "skip"]

    def test_a_settler_is_not_offered_fortify(self) -> None:
        """CONTROL: passes on the current tree."""
        assert actions(h.export_state(load()), "fortify", unit_id=101) == []

    def test_a_settler_who_cannot_found_is_not_offered_found_city(self) -> None:
        """E6. Fails on the current tree: any settler with moves is offered it."""
        assert actions(h.export_state(load()), "found_city", unit_id=101), "control: offered"
        state = h.export_state(load(SCENARIOS["settler_cannot_found"]))
        assert actions(state, "found_city") == []
        assert unit_actions(state, 101) == ["move", "skip"]

    def test_a_settler_in_a_city_is_not_offered_found_city(self) -> None:
        """E6. Fails on the current tree: a city stands on the tile."""
        state = h.export_state(load(SCENARIOS["settler_in_the_city"]))
        assert actions(state, "found_city") == []

    def test_a_warrior_is_never_offered_found_city(self) -> None:
        """CONTROL: passes on the current tree."""
        for scenario in SCENARIOS:
            state = h.export_state(load(SCENARIOS[scenario]))
            assert actions(state, "found_city", unit_id=102) == [], scenario

    def test_attack_is_offered_while_at_war(self) -> None:
        """CONTROL: passes on the current tree."""
        state = h.export_state(load())
        assert targets_of(state, "attack", 102) == {(2, 0)}
        assert actions(state, "ranged_attack") == []

    def test_no_attack_is_offered_at_peace(self) -> None:
        """E6. Fails on the current tree: the exporter never asks whether it starts a war."""
        war = h.export_state(load(c.ARCHER))
        assert actions(war, "attack") or actions(war, "ranged_attack"), "control: at war"
        state = h.export_state(load(SCENARIOS["peace"]))
        assert actions(state, "attack") == []
        assert actions(state, "ranged_attack") == []
        assert unit_actions(state, 102) == ["fortify", "move", "skip"]

    def test_a_melee_unit_is_offered_attack_not_ranged_attack(self) -> None:
        """CONTROL: passes on the current tree (range 0)."""
        state = h.export_state(load())
        assert actions(state, "attack", unit_id=102)
        assert actions(state, "ranged_attack", unit_id=102) == []

    def test_a_range_without_a_ranged_strength_is_not_a_ranged_attack(self) -> None:
        """E6. Fails on the current tree: it chooses by ``GetRange() > 1``, so this melee
        unit is offered a ranged attack (and no attack)."""
        state = h.export_state(load(SCENARIOS["range_without_strength"]))
        assert actions(state, "ranged_attack") == []
        assert targets_of(state, "attack", 102) == {(2, 0)}

    def test_a_ranged_unit_is_offered_a_ranged_attack_on_the_tiles_in_range(self) -> None:
        """E6. Fails on the current tree: the archer is offered a ranged attack only (the old
        exporter picks one kind by range), so it loses its melee attack."""
        state = h.export_state(load(SCENARIOS["archer_two_enemies"]))
        assert targets_of(state, "ranged_attack", 102) == {(2, 0), (3, 0)}
        assert targets_of(state, "attack", 102) == {(2, 0)}

    def test_ranged_targets_hold_a_foreign_unit_and_are_in_range(self) -> None:
        """E6. Enemy at (3,1), distance 3 from (1,0): out of range for range 2, so the archer
        has nothing to shoot at. Fails on the current tree: not offered (or offered by its own
        range arithmetic) differently from the stand-in; see the equality test for the rest."""
        setup = c.ARCHER + "Civ6Stub.unit(201).x, Civ6Stub.unit(201).y = 3, 1\n"
        state = h.export_state(load(setup))
        assert actions(state, "ranged_attack") == []
        assert actions(state, "attack") == []
        # ... and (0,0), the settler's tile, is in range of the archer but holds no foreigner.
        assert unit_actions(state, 102) == ["fortify", "move", "skip"]

    def test_a_ranged_unit_with_no_attacks_left_shoots_nothing(self) -> None:
        """CONTROL: passes on the current tree."""
        setup = c.ARCHER + "Civ6Stub.unit(102).attacks = 0\n"
        state = h.export_state(load(setup))
        assert actions(state, "ranged_attack") == [] and actions(state, "attack") == []


# ---------------------------------------------------------------------------
# L3: production choices by hash, strict, no placed items (E4)
# ---------------------------------------------------------------------------
PRODUCE_SCENARIOS: dict[str, tuple[str, dict[str, str]]] = {
    "default": ("", {"UNIT_WARRIOR": "unit", "BUILDING_GRANARY": "building"}),
    "all_now": (
        ALL_NOW,
        {
            "UNIT_WARRIOR": "unit",
            "UNIT_SETTLER": "unit",
            "UNIT_ARCHER": "unit",
            "BUILDING_MONUMENT": "building",
            "BUILDING_GRANARY": "building",
        },
    ),
    "granary_needs_a_plot": (
        ALL_NOW + GRANARY_NEEDS_A_PLOT,
        {
            "UNIT_WARRIOR": "unit",
            "UNIT_SETTLER": "unit",
            "UNIT_ARCHER": "unit",
            "BUILDING_MONUMENT": "building",
        },
    ),
    "loose_only": (
        "Civ6Stub.cities[301].canProduce = { UNIT_WARRIOR = 'later', BUILDING_GRANARY = 'now',"
        " DISTRICT_CAMPUS = 'later' }\n",
        {"BUILDING_GRANARY": "building"},
    ),
}


def produce_action(state: dict[str, Any]) -> dict[str, Any] | None:
    found = actions(state, "produce", city_id=301)
    assert len(found) <= 1
    return found[0] if found else None


class TestProductionChoices:
    @pytest.mark.parametrize("scenario", sorted(PRODUCE_SCENARIOS))
    def test_items_are_the_strictly_producible_unplaced_type_names(self, scenario: str) -> None:
        """E4. Fails on the current tree: it asks by row index, which the game does not know,
        so the list is empty (and it would list a district if it were not)."""
        setup, expected = PRODUCE_SCENARIOS[scenario]
        produce = produce_action(h.export_state(load(setup)))
        assert produce is not None
        assert all(isinstance(item, str) for item in produce["items"]), produce["items"]
        assert sorted(produce["items"]) == sorted(expected)
        assert produce["item_types"] == expected

    def test_default_list_is_a_warrior_and_a_granary_only(self) -> None:
        """E4. Fails on the current tree: empty. Not the settler (loose only) and not the
        campus (needs a plot)."""
        produce = produce_action(h.export_state(load()))
        assert produce is not None
        assert "UNIT_SETTLER" not in produce["items"]
        assert "DISTRICT_CAMPUS" not in produce["items"]
        assert sorted(produce["items"]) == ["BUILDING_GRANARY", "UNIT_WARRIOR"]

    @pytest.mark.parametrize("scenario", sorted(PRODUCE_SCENARIOS))
    def test_an_item_is_listed_exactly_when_check_accepts_it(self, scenario: str) -> None:
        """E4. Fails on the current tree: ``Check`` does not exist."""
        setup = PRODUCE_SCENARIOS[scenario][0]
        runtime = load(setup)
        produce = produce_action(h.export_state(runtime))
        listed = set(produce["items"]) if produce else set()
        names = [
            "UNIT_SETTLER",
            "UNIT_WARRIOR",
            "UNIT_ARCHER",
            "BUILDING_MONUMENT",
            "BUILDING_GRANARY",
            "DISTRICT_CAMPUS",
            "DISTRICT_CITY_CENTER",
        ]
        accepted = {
            name
            for name in names
            if check(runtime, {"action_type": "city_produce", "city_id": 301, "item_id": name})[0]
            is True
        }
        assert listed == accepted
        assert accepted, "nothing accepted; the equality is vacuous"

    def test_a_district_is_never_listed(self) -> None:
        """E4. Fails on the current tree (empty list, so the districts check is only reached
        through the equality above); here every item is asked for in a game that allows all."""
        produce = produce_action(h.export_state(load(ALL_NOW)))
        assert produce is not None and produce["items"], "control: something is offered"
        assert not [i for i in produce["items"] if i.startswith("DISTRICT_")]
        assert set(produce["item_types"].values()) <= {"unit", "building"}

    def test_the_action_is_offered_with_no_items_while_the_city_builds_nothing(self) -> None:
        """CONTROL: passes on the current tree."""
        setup = (
            "for _, c in pairs(Civ6Stub.cityRecords[0]) do c.building = nil end\n"
            "Civ6Stub.cities[301].canProduce = {}\n"
        )
        produce = produce_action(h.export_state(load(setup)))
        assert produce is not None
        assert produce["items"] == []

    def test_no_action_when_the_city_is_busy_and_nothing_is_producible(self) -> None:
        """CONTROL: passes on the current tree."""
        setup = (
            "Civ6Stub.city(301).building = 'UNIT_WARRIOR'\nCiv6Stub.cities[301].canProduce = {}\n"
        )
        assert produce_action(h.export_state(load(setup))) is None

    def test_the_action_is_offered_to_a_busy_city_when_something_is_producible(self) -> None:
        """E4. Fails on the current tree: nothing is producible by index, so it is not offered."""
        state = h.export_state(load("Civ6Stub.city(301).building = 'UNIT_WARRIOR'\n"))
        produce = produce_action(state)
        assert produce is not None
        assert sorted(produce["items"]) == ["BUILDING_GRANARY", "UNIT_WARRIOR"]

    def test_the_stand_in_builds_what_is_listed(self) -> None:
        """E4, against the stand-in's game state: every listed item, requested by name through
        the executor and carried out in the next frame, becomes what the city builds.
        Fails on the current tree: nothing is listed."""
        produce = produce_action(h.export_state(load()))
        assert produce is not None and produce["items"], "nothing listed"
        for item in produce["items"]:
            runtime = load()
            command = {"action_type": "city_produce", "city_id": 301, "item_id": item}
            assert execute(runtime, command)[0] is True
            runtime.execute("Civ6Stub.applyRequests()")
            assert runtime.eval("Civ6Stub.city(301).building") == item


# ---------------------------------------------------------------------------
# L4: everything else is as before
# ---------------------------------------------------------------------------
class TestEverythingElseIsUnchanged:
    DROPPED = set(UNIT_COMMANDS) | {"produce"}

    def test_the_default_state_is_as_it_was_before_this_slice(self) -> None:
        """CONTROL: passes on the current tree. The fixture is the exporter's output for the
        default game before GM-B7b, minus the unit and production actions."""
        state = h.export_state(load())
        state["legal_actions"] = [
            a for a in state["legal_actions"] if a["action"] not in self.DROPPED
        ]
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        assert [a["action"] for a in golden["legal_actions"]] == [
            "end_turn",
            "research_tech",
            "research_civic",
        ], "the golden file lost its non-unit actions"
        assert state == golden

    def test_end_turn_is_first_and_research_follows_the_unit_actions(self) -> None:
        """CONTROL: passes on the current tree."""
        names = [a["action"] for a in h.export_state(load())["legal_actions"]]
        assert names[0] == "end_turn"
        assert names[-2:] == ["research_tech", "research_civic"]
