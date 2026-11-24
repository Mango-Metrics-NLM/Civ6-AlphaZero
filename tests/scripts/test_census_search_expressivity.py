"""E1..E4 of the expressivity census (GM-B11), from the pre-registration.

Spec: ``kb/specs/20261123-expressivity-census-preregistration.md``. Every expected number
is counted by hand from the specification on a state small enough to count; the hand count
is in the comment above each assertion (builders and their counts: ``census_states``).
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha_zero.bridge.protocol import LegalAction
from alpha_zero.bridge.search_translator import SearchTranslator
from tests.fixtures.bridge_search_translation import is_offered, rich_state
from tests.fixtures.census_states import (
    cap_state,
    end_turn_only,
    load_census,
    mixed_state,
    scaled_terrain_state,
    small_terrain_state,
)


@pytest.fixture(scope="module")
def mod() -> Any:
    return load_census()


def _shares(share: dict[str, Any]) -> tuple[Any, int, int]:
    return share["value"], share["numerator"], share["denominator"]


# -- offered commands and their keys -------------------------------------------------------

BIG_OFFER = [
    LegalAction("end_turn"),
    LegalAction("move", unit_id=1, targets=[(2, 2), (2, 3), (2, 2)]),  # (2,2) twice
    LegalAction("attack", unit_id=1, targets=[(5, 5)]),
    LegalAction("ranged_attack", unit_id=2, targets=[(6, 6)]),
    LegalAction("skip", unit_id=1),
    LegalAction("skip", unit_id=1),  # a second identical offer
    LegalAction("found_city", unit_id=3),
    LegalAction(
        "produce",
        city_id=5,
        items=["UNIT_A", "BUILDING_B", "UNIT_C"],
        item_types={"BUILDING_B": "building"},  # UNIT_C has no stated type: a unit
    ),
    LegalAction("fortify", unit_id=1),
    LegalAction("research_tech", items=["TECH_1", "TECH_2"]),
    LegalAction("research_civic", items=["CIVIC_1"]),
]
# Hand count of distinct atomic commands, end_turn excluded:
#   move 2 + attack 1 + ranged 1 + skip 1 + found_city 1 + produce 3 + fortify 1
#   + research_tech 2 + research_civic 1 = 13.
# Kinds the translator handles (U): move 2 + attack 1 + ranged 1 + skip 1 + found 1
#   + the two unit items of produce = 8.
TRANSLATED = {("unit_move", 1, (2, 2)), ("unit_skip", 1), ("city_produce", 5, "UNIT_A")}


def test_offered_commands_counts_distinct_atomic_commands_without_end_turn(mod: Any) -> None:
    offered = mod.offered_commands(BIG_OFFER)
    assert len(offered) == 13
    assert all(key[0] != "end_turn" for key in offered)
    assert ("unit_move", 1, (2, 2)) in offered and ("unit_move", 1, (2, 3)) in offered
    assert ("unit_skip", 1) in offered and ("unit_found_city", 3) in offered
    assert ("unit_ranged_attack", 2, (6, 6)) in offered
    assert ("city_produce", 5, "BUILDING_B") in offered
    assert ("research_tech", None, "TECH_2") in offered and (
        "research_civic",
        None,
        "CIVIC_1",
    ) in offered


def test_offered_commands_of_only_end_turn_is_empty(mod: Any) -> None:
    assert dict(mod.offered_commands([LegalAction("end_turn")])) == {}
    assert dict(mod.offered_commands([])) == {}


def test_e1_numerator_denominator_and_upper_bound(mod: Any) -> None:
    offered = mod.offered_commands(BIG_OFFER)
    e1 = mod.e1_expressible(offered, TRANSLATED)
    # 3 of 13 offered commands are in T; U = 8/13.
    assert _shares(e1) == (pytest.approx(3 / 13), 3, 13)
    assert _shares(e1["upper_bound_u"]) == (pytest.approx(8 / 13), 8, 13)
    assert e1["value"] <= e1["upper_bound_u"]["value"]
    by_kind = {k: (v["numerator"], v["denominator"]) for k, v in e1["by_kind"].items()}
    assert by_kind == {
        "unit_move": (1, 2),
        "unit_attack": (0, 1),
        "unit_ranged_attack": (0, 1),
        "unit_skip": (1, 1),
        "unit_found_city": (0, 1),
        "city_produce": (1, 3),
        "fortify": (0, 1),
        "research_tech": (0, 2),
        "research_civic": (0, 1),
    }


def test_e1_ignores_translated_keys_that_were_not_offered(mod: Any) -> None:
    offered = mod.offered_commands(BIG_OFFER)
    stray = {("unit_move", 99, (0, 0)), ("unit_skip", 98)}
    assert mod.e1_expressible(offered, TRANSLATED | stray)["numerator"] == 3


def test_e1_zero_denominator_has_no_value_and_is_excluded_not_zero(mod: Any) -> None:
    e1 = mod.e1_expressible({}, set())
    assert e1["value"] is None and e1["numerator"] == 0 and e1["denominator"] == 0
    assert e1["upper_bound_u"]["value"] is None
    values = [0.5, e1["value"], 0.7]
    agg = mod.aggregate(values)
    # excluded, not counted as 0: median of {0.5, 0.7} is 0.6 (a 0 would give 0.5).
    assert agg["median"] == pytest.approx(0.6)
    assert (agg["states_with_value"], agg["states_excluded"]) == (2, 1)


def test_command_key_matches_the_offered_key_for_each_kind(mod: Any) -> None:
    move = {"action_type": "unit_move", "unit_id": 4, "target": {"x": 2, "y": 3}}
    assert mod.command_key(move) == ("unit_move", 4, (2, 3))
    assert mod.command_key({"action_type": "unit_skip", "unit_id": 4}) == ("unit_skip", 4)
    assert mod.command_key({"action_type": "unit_found_city", "unit_id": 4}) == (
        "unit_found_city",
        4,
    )
    produce = {"action_type": "city_produce", "city_id": 10, "item_id": "UNIT_WARRIOR"}
    assert mod.command_key(produce) == ("city_produce", 10, "UNIT_WARRIOR")
    assert mod.command_key({"action_type": "end_turn"}) is None


def test_command_key_of_a_corrupted_command_is_not_an_offered_key(mod: Any) -> None:
    offered = mod.offered_commands(BIG_OFFER)
    wrong_target = {"action_type": "unit_move", "unit_id": 1, "target": {"x": 9, "y": 9}}
    wrong_unit = {"action_type": "unit_skip", "unit_id": 2}
    assert mod.command_key(wrong_target) not in offered
    assert mod.command_key(wrong_unit) not in offered


# -- E2 ------------------------------------------------------------------------------------


def test_e2_counts_units_with_a_usable_command_and_skip_does_not_count(mod: Any) -> None:
    own = [1, 2, 3, 4, 5]
    offered = [
        ("unit_move", 1, (1, 1)),
        ("unit_skip", 2),  # skip alone: not in the denominator
        ("unit_attack", 3, (2, 2)),
        ("unit_found_city", 4),
        ("unit_ranged_attack", 5, (3, 3)),
        ("unit_move", 99, (0, 0)),  # not an own unit
        ("city_produce", 1, "UNIT_X"),  # a city that shares a unit's id: not a unit command
    ]
    translated = [
        ("unit_move", 1, (1, 1)),
        ("unit_skip", 2),
        ("unit_skip", 3),  # skip does not count for unit 3
        ("unit_found_city", 4),
        ("unit_ranged_attack", 5, (3, 3)),
    ]
    # Denominator {1, 3, 4, 5} = 4; numerator {1, 4, 5} = 3.
    assert _shares(mod.e2_usable_units(own, offered, translated)) == (0.75, 3, 4)


def test_e2_zero_denominator_has_no_value(mod: Any) -> None:
    e2 = mod.e2_usable_units([1], [("unit_skip", 1)], [("unit_skip", 1)])
    assert e2["value"] is None and e2["denominator"] == 0
    assert mod.e2_usable_units([], [], [])["value"] is None


# -- E3 ------------------------------------------------------------------------------------


def test_e3_water_and_untouched_cells_are_not_land(mod: Any) -> None:
    terrain = {
        (0, 0): {"type": "plains", "features": []},
        (1, 0): {"type": "grassland", "features": ["mountains"]},  # mountain
        (2, 0): {"type": "ocean", "features": []},  # water
        (3, 0): {"type": "coast", "features": ["mountains"]},  # water, whatever its features
        (4, 0): {"type": "grassland", "features": ["hills"]},  # passable
        (5, 0): {"type": "mountain", "features": []},  # not a cell a tile reached
        (7, 0): {"type": "mountain", "features": []},  # mountain by type
        (8, 0): {"type": "plains", "features": ["ice"]},  # impassable by the proxy's rule
    }
    cells = {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (6, 0), (7, 0), (8, 0)}  # (6,0): no terrain
    # Land: (0,0) (1,0) (4,0) (7,0) (8,0) = 5; impassable among them: (1,0) (7,0) (8,0) = 3.
    assert _shares(mod.e3_impassable_land(cells, terrain)) == (pytest.approx(0.6), 3, 5)


def test_e3_zero_denominator_has_no_value(mod: Any) -> None:
    assert mod.e3_impassable_land(set(), {(0, 0): {"type": "mountain"}})["value"] is None
    water = {(0, 0): {"type": "ocean", "features": []}, (1, 0): {"type": "coast", "features": []}}
    assert mod.e3_impassable_land({(0, 0), (1, 0)}, water)["value"] is None


# -- E4 ------------------------------------------------------------------------------------


def test_e4_is_left_out_over_the_acting_players_converted_units(mod: Any) -> None:
    units = [{"player": 0}, {"player": 0}, {"player": 0}, {"player": 1}, {"player": 1}]
    # 1 left out of the 3 units of player 0; the 2 of player 1 are not in the denominator.
    assert _shares(mod.e4_left_out(1, units, 0)) == (pytest.approx(1 / 3), 1, 3)
    assert mod.e4_left_out(0, units, 1)["value"] == 0.0  # a real zero is a value
    assert mod.e4_left_out(0, [{"player": 1}], 0)["value"] is None  # no own units: no value


# -- the whole pipeline on small states ----------------------------------------------------


def test_measure_state_on_the_hand_counted_state(mod: Any) -> None:
    result = mod.measure_state(mixed_state(), 8)
    assert result["void"] is False
    # 12 offered; translated 8 (builder skip + builder move, monument, research are not).
    assert _shares(result["E1"]) == (pytest.approx(8 / 12), 8, 12)
    assert _shares(result["E1"]["upper_bound_u"]) == (pytest.approx(10 / 12), 10, 12)
    # Units {1 warrior, 2 builder, 3 settler} offered a move/found; 1 and 3 are translated.
    assert _shares(result["E2"]) == (pytest.approx(2 / 3), 2, 3)
    # The builder is the one own unit the stand-in leaves out, of 3 own units.
    assert _shares(result["E4"]) == (pytest.approx(1 / 3), 1, 3)
    # No tile was exported: no cell is land.
    assert result["E3"]["value"] is None and result["cells_with_a_tile"] == 0


def test_a_left_out_unit_stays_in_e2s_denominator(mod: Any) -> None:
    e2 = mod.measure_state(mixed_state(), 8)["E2"]
    assert e2["denominator"] == 3 and e2["numerator"] == 2  # the builder counts, unreached


def test_state_with_only_end_turn_has_no_e1_e2_e4_value(mod: Any) -> None:
    result = mod.measure_state(end_turn_only(), 8)
    assert result["void"] is False
    for metric in ("E1", "E2", "E4"):
        assert result[metric]["value"] is None and result[metric]["denominator"] == 0


def test_e3_on_an_unscaled_map_counts_only_cells_a_tile_reached(mod: Any) -> None:
    result = mod.measure_state(small_terrain_state(), 8)
    # 5 cells have a tile; 3 are land; 1 of those is a mountain. The 59 untouched default
    # plains cells are not land: counting them would give 1/62.
    assert result["cells_with_a_tile"] == 5
    assert _shares(result["E3"]) == (pytest.approx(1 / 3), 1, 3)


def test_e3_on_a_larger_map_counts_cells_not_tiles(mod: Any) -> None:
    result = mod.measure_state(scaled_terrain_state(), 8)
    # 5 tiles in 4 cells; 3 land cells, 1 mountain: 1/3 (tiles would give 1/4).
    assert result["cells_with_a_tile"] == 4
    assert _shares(result["E3"]) == (pytest.approx(1 / 3), 1, 3)


def test_e3_with_no_tiles_has_no_value(mod: Any) -> None:
    result = mod.measure_state(end_turn_only(), 8)
    assert result["cells_with_a_tile"] == 0 and result["E3"]["value"] is None


def test_units_past_the_simulators_cap_are_in_e2s_denominator_never_its_numerator(
    mod: Any,
) -> None:
    result = mod.measure_state(cap_state(22), 8)
    # 22 own warriors each offered a move; the simulator holds 20 units.
    assert result["E2"]["denominator"] == 22
    assert result["E2"]["numerator"] <= 20
    assert result["E1"]["denominator"] == 22 and result["E1"]["numerator"] <= 20
    assert result["E4"]["denominator"] == 22


@pytest.mark.parametrize("board", [8, 16])
def test_every_value_is_a_share_with_u_at_least_e1(mod: Any, board: int) -> None:
    for state in (mixed_state(), rich_state(), cap_state(5)):
        result = mod.measure_state(state, board)
        assert result["void"] is False
        e1 = result["E1"]
        assert 0.0 <= e1["value"] <= e1["upper_bound_u"]["value"] <= 1.0
        for metric in ("E1", "E2", "E4"):
            share = result[metric]
            assert 0 <= share["numerator"] <= share["denominator"]


# -- the property the specification demands: translated is a subset of offered --------------


def _pipeline(state: Any, board: int) -> tuple[Any, Any]:
    from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv
    from alpha_zero.bridge.state_converter import StateConverter

    cs = StateConverter(board_size=board, num_players=2, research_registry=False).convert(state)
    env = BridgeMCTSEnv.from_bridge_state(cs, seed=0)
    tr = SearchTranslator(cs, env.own_unit_ids(), env.own_city_ids(), allowed=None)
    decode = env.env.action_space_handler.decode_action
    _mask, commands, _reasons = tr.root_mask(env.env.get_valid_actions(), decode)
    return commands, state.legal_actions


@pytest.mark.parametrize("make", [mixed_state, rich_state, lambda: cap_state(22)])
def test_every_translated_command_key_is_an_offered_command_key(mod: Any, make: Any) -> None:
    state = make()
    commands, legal = _pipeline(state, 8)
    offered = mod.offered_commands(legal)
    assert commands
    for command in commands.values():
        assert is_offered(command, legal), command
        key = mod.command_key(command)
        assert key is None or key in offered, command


def test_the_translated_set_of_the_hand_counted_state_is_exactly_the_expected_one(
    mod: Any,
) -> None:
    commands, _legal = _pipeline(mixed_state(), 8)
    keys = {k for k in map(mod.command_key, commands.values()) if k is not None}
    # Warrior 1: skip and four moves; settler 3: skip and found_city; one unit item.
    assert keys == {
        ("unit_skip", 1),
        ("unit_move", 1, (4, 3)),
        ("unit_move", 1, (2, 3)),
        ("unit_move", 1, (3, 2)),
        ("unit_move", 1, (3, 4)),
        ("unit_skip", 3),
        ("unit_found_city", 3),
        ("city_produce", 10, "UNIT_WARRIOR"),
    }


def _corrupt(monkeypatch: pytest.MonkeyPatch, change: Any) -> None:
    original = SearchTranslator.root_mask

    def root_mask(self: Any, valid: Any, decode: Any) -> Any:
        mask, commands, reasons = original(self, valid, decode)
        change(commands)
        return mask, commands, reasons

    monkeypatch.setattr(SearchTranslator, "root_mask", root_mask)


def test_a_translation_that_names_an_unoffered_target_voids_the_state(
    mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def retarget(commands: dict[int, dict[str, Any]]) -> None:
        move = next(c for c in commands.values() if c["action_type"] == "unit_move")
        move["target"] = {"x": 7, "y": 7}

    _corrupt(monkeypatch, retarget)
    result = mod.measure_state(mixed_state(), 8)
    assert result["void"] is True and "not offered" in result["void_reason"]
    assert "E1" not in result and "E2" not in result


def test_a_translation_that_invents_a_unit_voids_the_state(
    mod: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def invent(commands: dict[int, dict[str, Any]]) -> None:
        commands[9999] = {"action_type": "unit_skip", "unit_id": 777}  # never offered

    _corrupt(monkeypatch, invent)
    result = mod.measure_state(mixed_state(), 8)
    assert result["void"] is True and "E1" not in result
