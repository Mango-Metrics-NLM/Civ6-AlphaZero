"""GM-B6a amendment: a step is never less than one game tile.

Map 4 x 3 on a board of 8 (the Lua stand-in's size): a board cell is 0.5 x 0.375 tile, so
the destination point is the unit's cell centre plus one whole tile in the direction.
Unit at game (2, 1): cell (4, 3), centre (2.0, 1.125).
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha_zero.bridge.protocol import LegalAction
from alpha_zero.bridge.search_translator import SearchTranslator
from tests.fixtures.bridge_search_states import make_state, unit
from tests.fixtures.bridge_search_translation import convert

OFFERED = [(3, 1), (1, 1), (2, 0), (2, 2)]


def _state(attack: list[tuple[int, int]] | None = None, w: int = 4, h: int = 3) -> Any:
    legal = [LegalAction("end_turn"), LegalAction("move", unit_id=2, targets=OFFERED)]
    if attack:
        legal.append(LegalAction("attack", unit_id=2, targets=attack))
    return make_state(
        width=w, height=h, units=[unit(2, "UNIT_WARRIOR", 0, 2, 1)], legal_actions=legal
    )


def _translate(state: Any, action: dict[str, Any]) -> Any:
    return SearchTranslator(convert(state), [2], []).translate(action)


def _act(kind: str, dx: int, dy: int) -> dict[str, Any]:
    return {"type": kind, "unit_idx": 0, "direction": (dx, dy)}


class TestSmallMapMove:
    def test_converted_position_is_as_assumed(self) -> None:
        assert convert(_state())["units"][0]["position"] == (4, 3)

    @pytest.mark.parametrize(
        ("dx", "dy", "expected"),
        [(1, 0, (3, 1)), (-1, 0, (1, 1)), (0, 1, (2, 2)), (0, -1, (2, 0))],
    )
    def test_a_direction_moves_one_whole_tile(
        self, dx: int, dy: int, expected: tuple[int, int]
    ) -> None:
        result = _translate(_state(), _act("move", dx, dy))
        assert result.command == {
            "action_type": "unit_move",
            "unit_id": 2,
            "target": {"x": expected[0], "y": expected[1]},
        }

    def test_without_the_one_tile_floor_the_step_would_not_leave_the_tile(self) -> None:
        # With a half-tile step the point for (1, 0) would be (2.5, 1.125): (3, 1) and the own
        # tile (2, 1) are 0.5 / 0.5 away, so nothing is strictly nearer. The floor makes it 3.0.
        result = _translate(_state(), _act("move", 1, 0))
        assert result.command is not None and result.command["target"] == {"x": 3, "y": 1}

    def test_x_wraps_on_the_small_map(self) -> None:
        # Unit (0, 1), direction (0, 1): point (0.0, 2.125). The own tile is 1.125 away; the
        # tile (3, 2) is 1 across the seam and 0.125 down (1.0078); only the wrap makes it nearer.
        state = make_state(
            width=4,
            height=3,
            units=[unit(2, "UNIT_WARRIOR", 0, 0, 1)],
            legal_actions=[LegalAction("move", unit_id=2, targets=[(3, 2), (1, 0)])],
        )
        result = _translate(state, _act("move", 0, 1))
        assert result.command is not None and result.command["target"] == {"x": 3, "y": 2}

    def test_off_the_board_cell_is_still_refused(self) -> None:
        state = make_state(
            width=4,
            height=3,
            units=[unit(2, "UNIT_WARRIOR", 0, 0, 0)],
            legal_actions=[LegalAction("move", unit_id=2, targets=[(1, 0), (0, 1)])],
        )
        assert _translate(state, _act("move", -1, 0)).command is None
        assert _translate(state, _act("move", 0, -1)).command is None


class TestSmallMapAttack:
    def test_a_target_inside_the_box_around_the_point(self) -> None:
        # Point (3.0, 1.125), box +-0.5 x +-0.5: (3, 1) is inside.
        result = _translate(_state(attack=[(3, 1)]), _act("attack", 1, 0))
        assert result.command == {
            "action_type": "unit_attack",
            "unit_id": 2,
            "target": {"x": 3, "y": 1},
        }

    @pytest.mark.parametrize("target", [(1, 1), (2, 0), (2, 2), (2, 1)])
    def test_targets_outside_the_box_are_not_chosen(self, target: tuple[int, int]) -> None:
        assert _translate(_state(attack=[target]), _act("attack", 1, 0)).command is None

    def test_a_move_offer_is_not_an_attack(self) -> None:
        assert _translate(_state(), _act("attack", 1, 0)).command is None


class TestLargeMapUnchanged:
    def test_a_map_exactly_the_board_size_steps_one_tile(self) -> None:
        # 8 x 8 on board 8: scale 1, step 1 tile, as before the amendment.
        state = make_state(
            size=8,
            units=[unit(2, "UNIT_WARRIOR", 0, 3, 3)],
            legal_actions=[LegalAction("move", unit_id=2, targets=[(4, 3), (3, 4)])],
        )
        result = SearchTranslator(convert(state, 8), [2], []).translate(_act("move", 1, 0))
        assert result.command is not None and result.command["target"] == {"x": 4, "y": 3}
