"""The simulator's board is the converter's board, not the game's map (GM-B5 follow-up).

``StateConverter(board_size=N)`` scales every unit, city and terrain position onto an N x N
board, whatever the size of the game's map. ``BridgeMCTSEnv`` must build its simulator on that
same board. Building it on ``max(_map_size)`` gives a simulator as wide as the game's map
(84 for an 84x54 map) of which only the N x N corner has terrain: every other cell is default
plains, and the observation is 13 x 84 x 84.

All tests that name a map go through the real ``StateConverter.convert`` of a real
``GameStateData``; only the fallback tests write the converted dict by hand, because the
fallback is for dicts that have no ``_board_size``.
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv
from alpha_zero.bridge.state_converter import StateConverter
from tests.fixtures.bridge_search_states import city, make_state, tile, unit

BOARD = 16
SIM_ACTIONS = 376
DIRECTIONS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

#: Maps larger than the converter's 16x16 board, one not square, one the wrong way round.
LARGE_MAPS = [(40, 24), (84, 54), (24, 40)]


def _converted(state: Any, board: int = BOARD) -> dict[str, Any]:
    return StateConverter(board_size=board, research_registry=False).convert(state)


def _env(width: int, height: int, board: int = BOARD, **kwargs: Any) -> BridgeMCTSEnv:
    state = make_state(width=width, height=height, **kwargs)
    return BridgeMCTSEnv.from_bridge_state(_converted(state, board))


def _populated(width: int, height: int, board: int = BOARD) -> BridgeMCTSEnv:
    return _env(
        width,
        height,
        board,
        units=[
            unit(2, "UNIT_WARRIOR", 0, width - 1, height - 1),
            unit(3, "UNIT_WARRIOR", 1, 0, 0),
        ],
        cities=[],
    )


def _move_index(env: BridgeMCTSEnv, unit_idx: int, dx: int, dy: int) -> int:
    return env.env.action_space_handler.encode_action(
        {"type": "move", "unit_idx": unit_idx, "direction": (dx, dy)}
    )


class TestSimulatorBoardIsTheConvertersBoard:
    @pytest.mark.parametrize(("width", "height"), LARGE_MAPS)
    def test_board_and_observation_are_the_converters(self, width: int, height: int) -> None:
        env = _populated(width, height)

        assert env.env.board_size == BOARD
        assert env.get_observation()["spatial"].shape == (13, BOARD, BOARD)
        assert env.get_valid_actions().shape == (SIM_ACTIONS,)

    @pytest.mark.parametrize(("width", "height"), LARGE_MAPS)
    def test_everything_lies_inside_the_board_and_every_cell_has_terrain(
        self, width: int, height: int
    ) -> None:
        state = make_state(
            width=width,
            height=height,
            units=[
                unit(2, "UNIT_WARRIOR", 0, width - 1, height - 1),
                unit(3, "UNIT_WARRIOR", 1, 0, 0),
                unit(4, "UNIT_SETTLER", 0, width // 2, height // 2),
            ],
        )
        state.cities = [city(9, 1, width - 1, 0), city(10, 0, 0, height - 1)]
        env = BridgeMCTSEnv.from_bridge_state(_converted(state))

        inside = {(x, y) for x in range(BOARD) for y in range(BOARD)}
        assert len(env.env.units) == 3
        assert len(env.env.cities) == 2
        assert {u.position for u in env.env.units} <= inside
        assert {c.position for c in env.env.cities} <= inside
        assert set(env.env.terrain) == inside

    @pytest.mark.parametrize(("width", "height"), LARGE_MAPS)
    def test_a_unit_at_the_far_corner_cannot_leave_the_board(self, width: int, height: int) -> None:
        env = _populated(width, height)

        mine = [u for u in env.env.units if u.player == 0]
        assert [u.position for u in mine] == [(BOARD - 1, BOARD - 1)]
        mask = env.get_valid_actions()
        legal_inside = 0
        for dx, dy in DIRECTIONS:
            dest = (BOARD - 1 + dx, BOARD - 1 + dy)
            on_board = 0 <= dest[0] < BOARD and 0 <= dest[1] < BOARD
            legal = bool(mask[_move_index(env, 0, dx, dy)])
            assert legal == env.env._is_valid_move(dest), (dx, dy)
            if not on_board:
                assert not legal, f"a move to {dest}, off the {BOARD}x{BOARD} board, is legal"
            else:
                legal_inside += legal
        assert legal_inside == 3, "the corner unit has three open neighbours on the board"

    @pytest.mark.parametrize(("width", "height"), [(8, 8), (16, 16), (10, 16)])
    def test_a_map_no_larger_than_the_board_gets_the_converters_board(
        self, width: int, height: int
    ) -> None:
        env = _populated(width, height)
        assert env.env.board_size == BOARD
        assert env.get_observation()["spatial"].shape == (13, BOARD, BOARD)

    @pytest.mark.parametrize("board", [8, 12, 16, 24])
    def test_convert_carries_the_board_size(self, board: int) -> None:
        converted = _converted(make_state(width=40, height=24), board)
        assert converted["_board_size"] == board
        assert type(converted["_board_size"]) is int
        assert converted["_map_size"] == (40, 24)

    @pytest.mark.parametrize("board", [8, 24])
    def test_a_non_default_board_is_followed_on_a_large_map(self, board: int) -> None:
        env = _populated(84, 54, board)
        assert env.env.board_size == board
        assert env.get_observation()["spatial"].shape == (13, board, board)


class TestFallbacks:
    @staticmethod
    def _converted(**extra: Any) -> dict[str, Any]:
        return {"turn": 0, "current_player": 0, "terrain": {}, "units": [], "cities": [], **extra}

    def test_no_board_size_uses_the_map_size(self) -> None:
        env = BridgeMCTSEnv.from_bridge_state(self._converted(_map_size=(8, 8)))
        assert env.env.board_size == 8

    def test_no_board_size_uses_the_larger_map_dimension(self) -> None:
        env = BridgeMCTSEnv.from_bridge_state(self._converted(_map_size=(8, 12)))
        assert env.env.board_size == 12

    def test_neither_key_gives_sixteen(self) -> None:
        env = BridgeMCTSEnv.from_bridge_state(self._converted())
        assert env.env.board_size == 16

    @pytest.mark.parametrize("bad", [0, -1, True, "16", None, 16.0])
    def test_an_unusable_board_size_falls_back_to_the_map_size(self, bad: Any) -> None:
        env = BridgeMCTSEnv.from_bridge_state(self._converted(_map_size=(8, 8), _board_size=bad))
        assert env.env.board_size == 8

    def test_a_good_board_size_wins_over_the_map_size(self) -> None:
        env = BridgeMCTSEnv.from_bridge_state(self._converted(_map_size=(84, 54), _board_size=10))
        assert env.env.board_size == 10


class TestKnownLimitScaledMountain:
    """Characterisation of a known limit, not wanted behaviour.

    On a scaled map one board cell stands for several game tiles. The converter unions their
    features, so a single mountain among plains makes the whole cell a mountain, which the
    simulator refuses to enter.
    """

    def test_one_mountain_tile_makes_the_whole_board_cell_impassable(self) -> None:
        state = make_state(
            width=84,
            height=54,
            units=[unit(2, "UNIT_WARRIOR", 0, 40, 25), unit(3, "UNIT_WARRIOR", 1, 80, 50)],
            tiles=[tile(0, 0, "TERRAIN_GRASS"), tile(1, 0, "TERRAIN_GRASS_MOUNTAIN")],
        )
        converter = StateConverter(board_size=BOARD, research_registry=False)
        converted = converter.convert(state)  # sets the scaling for this map
        assert converter._scale_coords(0, 0) == converter._scale_coords(1, 0) == (0, 0)
        env = BridgeMCTSEnv.from_bridge_state(converted)

        assert env.env.terrain[(0, 0)].terrain_type == "mountain"
        assert env.proxy_stats["impassable_tiles"] == 1


class TestKnownLimitUnknownShips:
    """Characterisation of a known limit, not wanted behaviour.

    A ship the unit mapping does not know is read as ``melee`` (with a warning), so it reaches
    the simulator as a warrior, even standing on water, which the simulator seeds impassable.
    A ship the mapping does know (``UNIT_GALLEY``) is left out and counted instead.
    """

    def _env(self, unit_type: str) -> BridgeMCTSEnv:
        state = make_state(
            units=[unit(2, unit_type, 0, 2, 2), unit(3, "UNIT_WARRIOR", 1, 14, 14)],
            tiles=[tile(2, 2, "TERRAIN_COAST")],
        )
        return BridgeMCTSEnv.from_bridge_state(_converted(state))

    def test_the_longship_is_not_in_the_mapping(self) -> None:
        from alpha_zero.bridge.state_converter import UnitMapping

        mapping = UnitMapping()

        assert "UNIT_NORWEGIAN_LONGSHIP" not in mapping.UNITS
        assert mapping.UNITS["UNIT_GALLEY"] == "naval_melee"

    def test_an_unknown_ship_becomes_a_warrior_on_impassable_terrain(self) -> None:
        env = self._env("UNIT_NORWEGIAN_LONGSHIP")

        ships = [u for u in env.env.units if u.id == 2]
        assert [u.unit_type for u in ships] == ["warrior"]
        assert env.env.terrain[ships[0].position].terrain_type == "mountain"
        assert env.proxy_stats["units_not_represented"] == 0

    def test_a_known_ship_is_left_out_and_counted(self) -> None:
        env = self._env("UNIT_GALLEY")

        assert [u.id for u in env.env.units] == [3]
        assert env.proxy_stats["units_not_represented"] == 1
