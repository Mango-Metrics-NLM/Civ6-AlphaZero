"""GM-B8a: ``StateConverter.convert`` puts only hostile owners on the enemy side.

A state from an older mod (no ``observes_others``) converts as it always did. A state from the new
mod keeps the agent's own units and cities and the units and cities of hostile owners; the rest are
left out and counted in ``_others``.
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha_zero.bridge.protocol import GameStateData
from alpha_zero.bridge.state_converter import StateConverter
from tests.fixtures import lua_harness as h
from tests.fixtures import lua_others as o

BOARD = 8


def _converter() -> StateConverter:
    return StateConverter(board_size=BOARD, create_encoder=False, research_registry=False)


def _raw(
    units: list[tuple[int, int]],
    cities: list[tuple[int, int]],
    *,
    player: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "turn": 5,
        "player_id": player,
        "map": {"width": BOARD, "height": BOARD, "tiles": []},
        "units": [
            {"id": uid, "type_name": "UNIT_WARRIOR", "owner": owner, "x": uid % BOARD, "y": 1}
            for uid, owner in units
        ],
        "cities": [
            {"id": cid, "name": f"City {cid}", "owner": owner, "x": cid % BOARD, "y": 5}
            for cid, owner in cities
        ],
    }
    raw.update(extra)
    return raw


# own: units 1, 2 and city 90; others as below.
UNITS = [(1, 0), (2, 0), (11, 1), (12, 10), (13, 63), (14, 20), (15, 30)]
CITIES = [(90, 0), (91, 1), (92, 30)]
PLAYERS = [
    {"id": 1, "kind": "major", "at_war": True},  # hostile
    {"id": 10, "kind": "minor", "at_war": False},  # not hostile
    {"id": 63, "kind": "barbarian", "at_war": True},  # hostile
    {"id": 20, "kind": "minor"},  # unknown (no at_war)
    # 30: no entry, unknown
]


def _ids(items: list[dict[str, Any]]) -> list[int]:
    return sorted(i["_id"] for i in items)


def _owners(items: list[dict[str, Any]]) -> set[int]:
    return {i["_owner_id"] for i in items}


class TestOlderMod:
    def test_every_other_owner_is_on_the_enemy_side_as_before(self) -> None:
        converted = _converter().convert(_raw(UNITS, CITIES))
        assert _ids(converted["units"]) == [1, 2, 11, 12, 13, 14, 15]
        assert _ids(converted["cities"]) == [90, 91, 92]
        foreign = [u for u in converted["units"] if u["_owner_id"] != 0]
        assert foreign
        assert {u["player"] for u in foreign} == {1}
        assert {u["player"] for u in converted["units"] if u["_owner_id"] == 0} == {0}

    def test_other_players_alone_do_not_change_an_older_state(self) -> None:
        # No ``observes_others`` key: the entries are not consulted.
        converted = _converter().convert(_raw(UNITS, CITIES, other_players=PLAYERS))
        assert _ids(converted["units"]) == [1, 2, 11, 12, 13, 14, 15]
        assert _ids(converted["cities"]) == [90, 91, 92]

    def test_a_game_state_object_converts_the_same_as_its_dict(self) -> None:
        raw = _raw(UNITS, CITIES)
        a = _converter().convert(raw)
        b = _converter().convert(GameStateData.from_dict(raw))
        assert a["units"] == b["units"]
        assert a["cities"] == b["cities"]


class TestObserving:
    def _converted(self, **extra: Any) -> dict[str, Any]:
        raw = _raw(UNITS, CITIES, observes_others=True, other_players=PLAYERS, **extra)
        return _converter().convert(raw)

    def test_only_hostile_owners_are_on_the_enemy_side(self) -> None:
        converted = self._converted()
        assert _ids(converted["units"]) == [1, 2, 11, 13]
        assert _ids(converted["cities"]) == [90, 91]
        foreign = [u for u in converted["units"] if u["_owner_id"] != 0]
        assert _owners(foreign) == {1, 63}
        assert {u["player"] for u in foreign} == {1}

    def test_the_others_are_counted(self) -> None:
        assert self._converted()["_others"] == {
            "observes_others": True,
            # hostile: units 11 and 13, city 91
            "hostile_units": 2,
            "hostile_cities": 1,
            # owner 10 at peace: unit 12
            "left_out_not_hostile": 1,
            # owners 20 and 30: units 14, 15 and city 92
            "left_out_unknown": 3,
        }

    def test_the_own_side_is_unchanged(self) -> None:
        older = _converter().convert(_raw(UNITS, CITIES))
        observing = self._converted()

        def own(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [i for i in items if i["_owner_id"] == 0]

        assert own(observing["units"]) == own(older["units"])
        assert own(observing["cities"]) == own(older["cities"])
        for key in ("current_player", "_player_id", "_board_size", "turn", "_map_size"):
            assert observing[key] == older[key], key
        assert (observing["current_player"], observing["_player_id"], observing["_board_size"]) == (
            0,
            0,
            BOARD,
        )

    def test_an_agent_that_is_not_player_zero_keeps_its_own_side(self) -> None:
        units = [(1, 1), (2, 0), (3, 63), (4, 10)]
        raw = _raw(
            units,
            [(90, 1)],
            player=1,
            observes_others=True,
            other_players=[
                {"id": 0, "kind": "major", "at_war": True},
                {"id": 63, "kind": "barbarian", "at_war": True},
                {"id": 10, "kind": "minor", "at_war": False},
            ],
        )
        converted = _converter().convert(raw)
        assert _ids(converted["units"]) == [1, 2, 3]
        assert {u["player"] for u in converted["units"] if u["_owner_id"] == 1} == {0}
        assert {u["player"] for u in converted["units"] if u["_owner_id"] != 1} == {1}
        assert converted["_player_id"] == 1
        assert converted["_others"]["left_out_not_hostile"] == 1

    def test_observing_with_nothing_foreign_counts_zero(self) -> None:
        raw = _raw([(1, 0)], [(90, 0)], observes_others=True, other_players=[])
        converted = _converter().convert(raw)
        assert converted["_others"] == {
            "observes_others": True,
            "hostile_units": 0,
            "hostile_cities": 0,
            "left_out_not_hostile": 0,
            "left_out_unknown": 0,
        }
        assert _ids(converted["units"]) == [1]

    def test_a_state_that_says_it_does_not_observe_still_leaves_unexplained_owners_out(
        self,
    ) -> None:
        raw = _raw([(1, 0), (2, 5)], [], observes_others=False, other_players=[])
        converted = _converter().convert(raw)
        assert _ids(converted["units"]) == [1]
        assert converted["_others"]["observes_others"] is False
        assert converted["_others"]["left_out_unknown"] == 1

    def test_converting_does_not_take_units_out_of_the_state_passed_in(self) -> None:
        state = GameStateData.from_dict(
            _raw(UNITS, CITIES, observes_others=True, other_players=PLAYERS)
        )
        _converter().convert(state)
        assert len(state.units) == len(UNITS)
        assert len(state.cities) == len(CITIES)

    def test_a_hostile_unit_is_still_scaled_onto_the_board(self) -> None:
        converted = self._converted()
        (barbarian,) = [u for u in converted["units"] if u["_owner_id"] == 63]
        assert barbarian["_original_pos"] == (13 % BOARD, 1)
        assert all(0 <= c < BOARD for c in barbarian["position"])


class TestAStateTheStandInExports:
    """The real mod's state (flag on) converts, and the war decides who is the enemy."""

    def _convert(self, setup: str) -> dict[str, Any]:
        state = h.export_state(o.load(o.BARB_UNIT + setup))
        return StateConverter(
            board_size=BOARD, create_encoder=False, research_registry=False
        ).convert(state)

    def test_it_converts_without_error_and_a_player_not_at_war_is_left_out(self) -> None:
        converted = self._convert("")
        assert _ids(converted["units"]) == [101, 102, 212]
        assert converted["_others"]["observes_others"] is True
        assert converted["_others"]["hostile_units"] == 1
        # The default enemy warrior 201 (player 1): a major with no war declared.
        assert converted["_others"]["left_out_not_hostile"] == 1

    def test_a_player_at_war_is_on_the_enemy_side(self) -> None:
        converted = self._convert("Civ6Stub.setWar(0, 1, true)\n")
        assert _ids(converted["units"]) == [101, 102, 201, 212]
        assert converted["_others"]["hostile_units"] == 2
        assert converted["_others"]["left_out_not_hostile"] == 0

    def test_without_diplomacy_the_major_is_unknown_not_an_enemy(self) -> None:
        converted = self._convert("Civ6Stub.removeDiplomacy()\n")
        assert _ids(converted["units"]) == [101, 102, 212]
        assert converted["_others"]["left_out_unknown"] == 1
        assert converted["_others"]["left_out_not_hostile"] == 0

    @pytest.mark.parametrize("observe", [False, True])
    def test_a_state_of_a_game_with_the_flag_off_converts_with_no_foreign_side(
        self, observe: bool
    ) -> None:
        state = h.export_state(o.load(o.BARB_UNIT, observe=observe))
        converted = StateConverter(
            board_size=BOARD, create_encoder=False, research_registry=False
        ).convert(state)
        assert converted["_others"]["observes_others"] is observe
        if not observe:
            assert _ids(converted["units"]) == [101, 102]
