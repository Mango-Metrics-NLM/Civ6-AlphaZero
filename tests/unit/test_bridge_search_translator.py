"""GM-B6a: the translator from a simulator action to a live command (``SearchTranslator``).

Geometry used throughout: an 84 x 54 game map on a board of 8, so one board cell is
10.5 x 6.75 game tiles and the centre of cell (cx, cy) is (cx * 10.5, cy * 6.75). Distances
in the expected values below were worked out by hand from that.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from alpha_zero.bridge.protocol import LegalAction
from alpha_zero.bridge.search_translator import SearchTranslator, Translation
from tests.fixtures.bridge_search_states import city, make_state, unit
from tests.fixtures.bridge_search_translation import (
    BIG_H,
    BIG_W,
    build,
    convert,
    rich_state,
)

OWN_X, OWN_Y = 40, 27  # own warrior 2: board cell (4, 4), centre (42.0, 27.0)


def _move(unit_idx: int, dx: int, dy: int) -> dict[str, Any]:
    return {"type": "move", "unit_idx": unit_idx, "direction": (dx, dy)}


def _warrior_state(targets: list[tuple[int, int]], x: int = OWN_X, y: int = OWN_Y) -> Any:
    """Builder 7 first (left out by the proxy), then own warrior 2 offered ``targets``."""
    return make_state(
        width=BIG_W,
        height=BIG_H,
        units=[unit(7, "UNIT_BUILDER", 0, 10, 10), unit(2, "UNIT_WARRIOR", 0, x, y)],
        legal_actions=[
            LegalAction("end_turn"),
            LegalAction("move", unit_id=2, targets=targets),
        ],
    )


def _translator(state: Any, units: list[int], cities: list[int] | None = None, **kw: Any) -> Any:
    return SearchTranslator(convert(state), units, cities or [], **kw)


def _target(command: dict[str, Any]) -> tuple[int, int]:
    return command["target"]["x"], command["target"]["y"]


class TestSimpleCommands:
    def test_end_turn_when_offered(self) -> None:
        t = _translator(rich_state(), [2, 4, 5])
        assert t.translate({"type": "end_turn"}) == Translation({"action_type": "end_turn"}, None)

    def test_end_turn_not_offered_is_not_translatable(self) -> None:
        state = make_state(width=BIG_W, height=BIG_H, legal_actions=[])
        result = _translator(state, []).translate({"type": "end_turn"})
        assert result.command is None
        assert isinstance(result.reason, str) and result.reason

    def test_skip_for_the_unit_at_that_index(self) -> None:
        t = _translator(rich_state(), [2, 4, 5])
        assert t.translate({"type": "skip", "unit_idx": 1}).command == {
            "action_type": "unit_skip",
            "unit_id": 4,
        }

    def test_skip_not_offered_for_that_unit(self) -> None:
        state = _warrior_state([(41, 27)])  # no skip offered for 2
        result = _translator(state, [2]).translate({"type": "skip", "unit_idx": 0})
        assert result.command is None and result.reason

    def test_found_city_for_the_settler(self) -> None:
        t = _translator(rich_state(), [2, 4, 5])
        assert t.translate({"type": "found_city", "unit_idx": 2}).command == {
            "action_type": "unit_found_city",
            "unit_id": 5,
        }

    def test_found_city_not_offered_for_a_warrior(self) -> None:
        t = _translator(rich_state(), [2, 4, 5])
        assert t.translate({"type": "found_city", "unit_idx": 0}).command is None

    def test_translation_is_immutable(self) -> None:
        result = Translation({"action_type": "end_turn"}, None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.reason = "x"  # type: ignore[misc]

    @pytest.mark.parametrize("idx", [1, 3, 99, -1])
    def test_index_out_of_range_is_not_translatable(self, idx: int) -> None:
        t = _translator(_warrior_state([(41, 27)]), [2])
        for kind in ("skip", "found_city"):
            assert t.translate({"type": kind, "unit_idx": idx}).command is None
        assert t.translate(_move(idx, 1, 0)).command is None
        assert (
            t.translate({"type": "produce", "city_idx": idx, "unit_type": "warrior"}).command
            is None
        )


class TestIdsNotPositions:
    def test_index_means_the_own_unit_ids_entry_not_the_list_position(self) -> None:
        """Builder 7 is listed first and offered a skip too; index 0 is warrior 2."""
        state = rich_state()
        assert state.units[0].id == 7
        _, env = build(state)
        assert env.own_unit_ids() == [2, 4, 5]  # builder left out
        t = _translator(state, env.own_unit_ids())
        assert t.translate({"type": "skip", "unit_idx": 0}).command["unit_id"] == 2

    def test_enemy_unit_with_the_same_id_does_not_confuse_the_position(self) -> None:
        """Ids are per owner in Civ6: an enemy unit 2 listed first must not be read as ours."""
        state = make_state(
            width=BIG_W,
            height=BIG_H,
            units=[unit(2, "UNIT_WARRIOR", 1, 80, 50), unit(2, "UNIT_WARRIOR", 0, OWN_X, OWN_Y)],
            legal_actions=[
                LegalAction("move", unit_id=2, targets=[(41, 27), (39, 27), (40, 26)]),
            ],
        )
        result = _translator(state, [2]).translate(_move(0, 1, 0))
        assert result.command == {
            "action_type": "unit_move",
            "unit_id": 2,
            "target": {"x": 41, "y": 27},
        }

    def test_reordering_the_unit_list_gives_the_same_commands(self) -> None:
        results = []
        for order in ((2, 4), (4, 2)):
            state = rich_state(units_first=order)
            converted, env = build(state)
            translator = SearchTranslator(converted, env.own_unit_ids(), env.own_city_ids())
            mask, commands, _ = translator.root_mask(
                env.get_valid_actions(), env.env.action_space_handler.decode_action
            )
            results.append((set(np.nonzero(mask)[0]), commands))
        (_, cmds_a), (_, cmds_b) = results
        assert cmds_a != cmds_b, "index -> command must differ, or this test is vacuous"
        assert {m["action_type"] + str(m.get("unit_id")) for m in cmds_a.values()} >= {
            "unit_skip2",
            "unit_skip4",
        }
        as_set = lambda cmds: {repr(sorted(c.items(), key=str)) for c in cmds.values()}  # noqa: E731
        assert as_set(cmds_a) == as_set(cmds_b)


class TestMove:
    @pytest.mark.parametrize(
        ("dx", "dy", "expected"),
        [
            (1, 0, (41, 27)),  # cell (5,4) centre (52.5, 27.0)
            (-1, 0, (39, 27)),  # cell (3,4) centre (31.5, 27.0)
            (0, -1, (41, 26)),  # cell (4,3) centre (42.0, 20.25)
            (0, 1, (41, 28)),  # cell (4,5) centre (42.0, 33.75)
            (1, 1, (41, 28)),  # cell (5,5) centre (52.5, 33.75)
            (-1, -1, (39, 27)),  # cell (3,3) centre (31.5, 20.25)
        ],
    )
    def test_nearest_offered_tile_to_the_destination_cell_centre(
        self, dx: int, dy: int, expected: tuple[int, int]
    ) -> None:
        offered = [(41, 27), (39, 27), (40, 26), (41, 26), (40, 28), (41, 28)]
        result = _translator(_warrior_state(offered), [2]).translate(_move(0, dx, dy))
        assert result.reason is None
        assert result.command == {
            "action_type": "unit_move",
            "unit_id": 2,
            "target": {"x": expected[0], "y": expected[1]},
        }

    def test_target_must_be_strictly_nearer_than_the_unit_s_own_tile(self) -> None:
        # Destination centre (52.5, 27.0): own tile is 12.5 away; (39, 27) is 13.5 away.
        result = _translator(_warrior_state([(39, 27)]), [2]).translate(_move(0, 1, 0))
        assert result.command is None and result.reason

    def test_a_tile_exactly_as_far_as_the_own_tile_is_not_nearer(self) -> None:
        # (65, 27) and the own tile (40, 27) are both 12.5 from (52.5, 27.0).
        result = _translator(_warrior_state([(65, 27)]), [2]).translate(_move(0, 1, 0))
        assert result.command is None

    def test_tie_between_offered_tiles_takes_smaller_y_then_x(self) -> None:
        # (41, 26) and (41, 28) are equally far from (52.5, 27.0).
        result = _translator(_warrior_state([(41, 28), (41, 26)]), [2]).translate(_move(0, 1, 0))
        assert _target(result.command) == (41, 26)
        # Same y: (42, 27) and (63, 27) are both 10.5 from the centre; the smaller x wins.
        result = _translator(_warrior_state([(63, 27), (42, 27)]), [2]).translate(_move(0, 1, 0))
        assert _target(result.command) == (42, 27)

    def test_x_wraps_around_the_map(self) -> None:
        # Unit at (0, 19): cell (0, 3). Direction (0, 1): cell (0, 4), centre (0.0, 27.0).
        # Own tile is 8.0 away; (83, 20) is sqrt(50) away across the seam; (2, 19) is sqrt(68).
        state = _warrior_state([(83, 20), (2, 19)], x=0, y=19)
        result = _translator(state, [2]).translate(_move(0, 0, 1))
        assert _target(result.command) == (83, 20)

    def test_without_the_wrap_the_seam_tile_would_not_count(self) -> None:
        # Only (2, 19) is on offer: sqrt(68) is farther than the own tile (8.0).
        result = _translator(_warrior_state([(2, 19)], x=0, y=19), [2]).translate(_move(0, 0, 1))
        assert result.command is None

    def test_destination_off_the_board_is_not_translatable(self) -> None:
        state = _warrior_state([(1, 1), (0, 1), (1, 0)], x=2, y=2)  # cell (0, 0)
        t = _translator(state, [2])
        assert t.translate(_move(0, -1, -1)).command is None
        assert t.translate(_move(0, -1, 0)).command is None
        assert t.translate(_move(0, 0, -1)).command is None

    def test_destination_past_the_far_edge_is_not_translatable(self) -> None:
        state = _warrior_state([(82, 52), (83, 52)], x=83, y=53)  # cell (7, 7)
        t = _translator(state, [2])
        assert t.translate(_move(0, 1, 1)).command is None
        assert t.translate(_move(0, 0, 1)).command is None

    def test_scale_one_board_reaches_the_exact_tile(self) -> None:
        state = make_state(
            size=16,
            units=[unit(2, "UNIT_WARRIOR", 0, 5, 5)],
            legal_actions=[LegalAction("move", unit_id=2, targets=[(5, 6), (6, 5)])],
        )
        converted = convert(state, 16)
        result = SearchTranslator(converted, [2], []).translate(_move(0, 1, 0))
        assert _target(result.command) == (6, 5)

    def test_every_returned_move_target_is_offered_and_nearer(self) -> None:
        offered = [(41, 27), (39, 27), (40, 26), (41, 26), (40, 28), (41, 28)]
        t = _translator(_warrior_state(offered), [2])
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (dx, dy) == (0, 0):
                    continue
                cmd = t.translate(_move(0, dx, dy)).command
                if cmd is None:
                    continue
                cx, cy = (4 + dx) * BIG_W / 8, (4 + dy) * BIG_H / 8
                tx, ty = _target(cmd)
                assert (tx, ty) in offered
                assert (tx - cx) ** 2 + (ty - cy) ** 2 < (OWN_X - cx) ** 2 + (OWN_Y - cy) ** 2


class TestAttack:
    @staticmethod
    def _state(attack: list[tuple[int, int]], move: list[tuple[int, int]]) -> Any:
        # Own warrior (47, 27) is cell (4, 4); enemy (48, 27) is cell (5, 4).
        legal = [LegalAction("end_turn")]
        if attack:
            legal.append(LegalAction("attack", unit_id=2, targets=attack))
        if move:
            legal.append(LegalAction("move", unit_id=2, targets=move))
        return make_state(
            width=BIG_W,
            height=BIG_H,
            units=[unit(2, "UNIT_WARRIOR", 0, 47, 27), unit(3, "UNIT_WARRIOR", 1, 48, 27)],
            legal_actions=legal,
        )

    def test_attack_into_the_destination_cell(self) -> None:
        state = self._state(attack=[(48, 27), (47, 26)], move=[])
        result = _translator(state, [2]).translate(
            {"type": "attack", "unit_idx": 0, "direction": (1, 0)}
        )
        assert result.command == {
            "action_type": "unit_attack",
            "unit_id": 2,
            "target": {"x": 48, "y": 27},
        }

    def test_attack_target_in_another_cell_is_not_chosen(self) -> None:
        # (47, 26) is in the unit's own cell (4, 4), not the destination (5, 4).
        state = self._state(attack=[(47, 26)], move=[])
        result = _translator(state, [2]).translate(
            {"type": "attack", "unit_idx": 0, "direction": (1, 0)}
        )
        assert result.command is None and result.reason

    def test_attack_nearest_the_cell_centre_when_several_share_the_cell(self) -> None:
        # Cell (5, 4) centre (52.5, 27.0): (48, 27) is 4.5 away, (49, 28) is sqrt(13.25+1).
        state = self._state(attack=[(49, 28), (48, 27)], move=[])
        result = _translator(state, [2]).translate(
            {"type": "attack", "unit_idx": 0, "direction": (1, 0)}
        )
        # Squared distances 13.25 + 1 = 14.25 for (49, 28) against 20.25 for (48, 27).
        assert _target(result.command) == (49, 28)

    def test_a_move_is_never_turned_into_an_attack(self) -> None:
        state = self._state(attack=[(48, 27)], move=[])  # an attack on offer, no move
        result = _translator(state, [2]).translate(_move(0, 1, 0))
        assert result.command is None

    def test_an_attack_is_never_turned_into_a_move(self) -> None:
        state = self._state(attack=[], move=[(48, 27), (46, 27)])
        result = _translator(state, [2]).translate(
            {"type": "attack", "unit_idx": 0, "direction": (1, 0)}
        )
        assert result.command is None

    def test_a_move_whose_only_closer_tile_is_an_attack_target_is_not_translatable(self) -> None:
        # Move targets exclude the enemy tile; nothing offered for a move gets nearer.
        state = self._state(attack=[(48, 27)], move=[(46, 27), (47, 26), (47, 28)])
        assert _translator(state, [2]).translate(_move(0, 1, 0)).command is None


class TestProduce:
    @staticmethod
    def _translate(
        items: list[Any], unit_type: str, item_types: dict[str, str] | None = None, idx: int = 0
    ) -> Translation:
        state = make_state(
            width=BIG_W,
            height=BIG_H,
            cities=[city(10, 0, 30, 20)],
            legal_actions=[
                LegalAction("produce", city_id=10, items=items, item_types=item_types or {})
            ],
        )
        return _translator(state, [], [10]).translate(
            {"type": "produce", "city_idx": idx, "unit_type": unit_type}
        )

    @pytest.mark.parametrize(
        ("unit_type", "expected"),
        [("warrior", "UNIT_WARRIOR"), ("settler", "UNIT_SETTLER"), ("archer", "UNIT_ARCHER")],
    )
    def test_preferred_names(self, unit_type: str, expected: str) -> None:
        items = [
            "UNIT_CROSSBOWMAN",
            "UNIT_SWORDSMAN",
            "UNIT_ARCHER",
            "UNIT_SETTLER",
            "UNIT_WARRIOR",
        ]
        assert self._translate(items, unit_type).command == {
            "action_type": "city_produce",
            "city_id": 10,
            "item_id": expected,
            "item_type": "unit",
        }

    def test_class_not_name_decides(self) -> None:
        # A swordsman is melee, so it stands for the simulator's warrior; a slinger is ranged.
        assert self._translate(["UNIT_SWORDSMAN"], "warrior").command["item_id"] == "UNIT_SWORDSMAN"
        assert self._translate(["UNIT_SLINGER"], "archer").command["item_id"] == "UNIT_SLINGER"
        assert self._translate(["UNIT_SWORDSMAN"], "archer").command is None

    def test_without_a_preferred_name_the_first_by_name(self) -> None:
        # Both orders agree: sorted and offered order put the crossbowman first.
        result = self._translate(["UNIT_CROSSBOWMAN", "UNIT_SLINGER"], "archer")
        assert result.command["item_id"] == "UNIT_CROSSBOWMAN"

    def test_a_settler_is_not_a_builder(self) -> None:
        assert self._translate(["UNIT_BUILDER", "UNIT_SCOUT"], "settler").command is None

    def test_numbers_are_never_chosen(self) -> None:
        result = self._translate([5, 6, 7], "warrior", item_types={"5": "unit"})
        assert result.command is None and result.reason

    def test_a_name_the_table_does_not_list_is_never_chosen(self) -> None:
        assert self._translate(["UNIT_NEW_THING"], "warrior").command is None

    def test_a_name_that_is_not_a_unit_is_never_chosen(self) -> None:
        result = self._translate(
            ["UNIT_WARRIOR", "BUILDING_MONUMENT"], "warrior", {"UNIT_WARRIOR": "wonder"}
        )
        assert result.command is None

    def test_a_unit_kind_or_no_kind_is_accepted(self) -> None:
        assert self._translate(["UNIT_WARRIOR"], "warrior", {"UNIT_WARRIOR": "unit"}).command
        assert self._translate(["UNIT_WARRIOR"], "warrior", {}).command

    def test_city_id_comes_from_own_city_ids(self) -> None:
        state = make_state(
            width=BIG_W,
            height=BIG_H,
            cities=[city(11, 1, 70, 5), city(10, 0, 30, 20)],
            legal_actions=[
                LegalAction("produce", city_id=10, items=["UNIT_WARRIOR"]),
                LegalAction("produce", city_id=11, items=["UNIT_ARCHER"]),
            ],
        )
        result = _translator(state, [], [10]).translate(
            {"type": "produce", "city_idx": 0, "unit_type": "warrior"}
        )
        assert result.command["city_id"] == 10

    def test_not_offered_for_this_city(self) -> None:
        assert self._translate(["UNIT_WARRIOR"], "warrior", idx=1).command is None
