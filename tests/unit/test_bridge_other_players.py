"""GM-B8a: the protocol fields for other players and the rules in ``bridge/other_players.py``.

Expected values are written from the contract, not read back from the module.
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha_zero.bridge.other_players import is_hostile, owner_relations, partition
from alpha_zero.bridge.protocol import CityData, GameStateData, UnitData


def _unit(uid: int, owner: int) -> dict[str, Any]:
    return {"id": uid, "type_name": "UNIT_WARRIOR", "owner": owner, "x": uid % 4, "y": 0}


def _city(cid: int, owner: int) -> dict[str, Any]:
    return {"id": cid, "name": f"City {cid}", "owner": owner, "x": 3, "y": cid % 3}


def _state(
    units: list[tuple[int, int]] | None = None,
    cities: list[tuple[int, int]] | None = None,
    **extra: Any,
) -> GameStateData:
    """A state for player 0 with ``(id, owner)`` units and cities."""
    raw: dict[str, Any] = {
        "turn": 3,
        "player_id": 0,
        "map": {"width": 4, "height": 3, "tiles": []},
        "units": [_unit(*u) for u in (units or [])],
        "cities": [_city(*c) for c in (cities or [])],
    }
    raw.update(extra)
    return GameStateData.from_dict(raw)


def entry(owner: int, kind: str, **fields: Any) -> dict[str, Any]:
    return {"id": owner, "kind": kind, **fields}


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------
class TestProtocolFields:
    ENTRIES = [
        {"id": 1, "kind": "major", "at_war": True, "met": True},
        {"id": 10, "kind": "minor", "at_war": False},
        {"id": 63, "kind": "barbarian", "at_war": True},
        {"id": 20, "kind": "minor"},
    ]

    @pytest.mark.parametrize("flag", [True, False])
    def test_the_new_fields_round_trip(self, flag: bool) -> None:
        state = _state(observes_others=flag, other_players=self.ENTRIES)
        assert state.observes_others is flag
        assert state.other_players == self.ENTRIES

    def test_an_entry_is_kept_raw_a_missing_key_stays_missing(self) -> None:
        state = _state(observes_others=True, other_players=self.ENTRIES)
        assert "at_war" not in state.other_players[3]
        assert "met" not in state.other_players[1]

    def test_a_state_from_an_older_mod_has_none_and_an_empty_list(self) -> None:
        state = _state()
        assert state.observes_others is None
        assert state.other_players == []

    def test_false_is_not_none(self) -> None:
        state = _state(observes_others=False, other_players=[])
        assert state.observes_others is not None
        assert state.observes_others is False

    def test_other_players_without_the_flag_are_still_kept(self) -> None:
        state = _state(other_players=self.ENTRIES)
        assert state.observes_others is None
        assert state.other_players == self.ENTRIES

    def test_a_foreign_unit_without_the_omitted_keys_parses(self) -> None:
        unit = UnitData.from_dict(
            {
                "id": 210,
                "type_name": "UNIT_ARCHER",
                "owner": 1,
                "x": 3,
                "y": 0,
                "hp": 90,
                "max_hp": 100,
                "damage": 10,
                "combat_strength": 15,
                "ranged_strength": 25,
                "range": 2,
                "is_civilian": False,
            }
        )
        assert (unit.id, unit.unit_type, unit.owner_id, unit.position) == (
            210,
            "UNIT_ARCHER",
            1,
            (3, 0),
        )
        assert (unit.health, unit.max_health, unit.damage) == (90, 100, 10)
        assert (unit.combat_strength, unit.ranged_strength, unit.range) == (15, 25, 2)

    def test_a_foreign_city_without_the_omitted_keys_parses(self) -> None:
        city = CityData.from_dict(
            {
                "id": 401,
                "name": "Rival",
                "owner": 10,
                "x": 2,
                "y": 1,
                "population": 4,
                "is_capital": True,
                "buildings": [],
                "districts": [],
                "available_production": [],
            }
        )
        assert (city.id, city.name, city.owner_id, (city.x, city.y)) == (401, "Rival", 10, (2, 1))
        assert city.population == 4
        assert city.is_capital is True
        assert city.buildings == []
        assert city.production_progress == 0
        assert city.current_production is None

    def test_a_state_with_foreign_units_and_cities_parses_whole(self) -> None:
        state = _state(units=[(1, 0), (210, 1)], cities=[(401, 10)], observes_others=True)
        assert [u.owner_id for u in state.units] == [0, 1]
        assert [c.owner_id for c in state.cities] == [10]


# ---------------------------------------------------------------------------
# is_hostile
# ---------------------------------------------------------------------------
class TestIsHostile:
    @pytest.mark.parametrize(
        ("candidate", "hostile"),
        [
            (entry(63, "barbarian", at_war=True), True),
            (entry(63, "barbarian", at_war=False), True),
            (entry(63, "barbarian"), True),
            (entry(9, "free", at_war=False), True),
            (entry(9, "free"), True),
            (entry(1, "major", at_war=True), True),
            (entry(10, "minor", at_war=True), True),
            (entry(1, "major", at_war=False), False),
            (entry(10, "minor", at_war=False), False),
            (entry(1, "major"), False),
            (entry(10, "minor"), False),
            (entry(1, "major", at_war=False, met=True), False),
            ({"id": 5}, False),
            ({}, False),
        ],
        ids=[
            "barbarian at war",
            "barbarian at_war false",
            "barbarian no at_war",
            "free no war",
            "free no at_war",
            "major at war",
            "city-state at war",
            "major at peace",
            "city-state at peace",
            "major unknown",
            "city-state unknown",
            "major at peace and met",
            "no kind",
            "empty",
        ],
    )
    def test_rule(self, candidate: dict[str, Any], hostile: bool) -> None:
        assert is_hostile(candidate) is hostile

    @pytest.mark.parametrize("value", [1, "true", "yes", [True]])
    def test_only_the_boolean_true_is_at_war(self, value: Any) -> None:
        assert is_hostile(entry(1, "major", at_war=value)) is False


# ---------------------------------------------------------------------------
# owner_relations
# ---------------------------------------------------------------------------
class TestOwnerRelations:
    def test_every_rule_in_one_state(self) -> None:
        state = _state(
            units=[(1, 0), (2, 1), (3, 10), (4, 63), (5, 20), (6, 30), (7, 40)],
            cities=[(8, 50)],
            observes_others=True,
            other_players=[
                entry(1, "major", at_war=False),  # a major at peace
                entry(10, "minor", at_war=True),  # a city-state at war
                entry(63, "barbarian", at_war=True),
                entry(20, "minor"),  # at_war missing
                entry(40, "free"),
                # 30 has no entry at all; 50 neither
            ],
        )
        assert owner_relations(state) == {
            1: "not_hostile",
            10: "hostile",
            63: "hostile",
            20: "unknown",
            30: "unknown",
            40: "hostile",
            50: "unknown",
        }

    def test_the_players_own_id_is_not_a_foreign_owner(self) -> None:
        state = _state(units=[(1, 0)], observes_others=True, other_players=[entry(0, "major")])
        assert owner_relations(state) == {}

    def test_an_owner_of_only_a_city_is_covered(self) -> None:
        state = _state(
            cities=[(8, 10)], observes_others=True, other_players=[entry(10, "minor", at_war=False)]
        )
        assert owner_relations(state) == {10: "not_hostile"}

    def test_an_entry_for_an_owner_with_nothing_in_the_state_is_not_listed(self) -> None:
        state = _state(
            units=[(2, 1)],
            observes_others=True,
            other_players=[entry(1, "major", at_war=True), entry(63, "barbarian", at_war=True)],
        )
        assert owner_relations(state) == {1: "hostile"}

    def test_no_foreign_owner_no_relations(self) -> None:
        assert owner_relations(_state(units=[(1, 0)], observes_others=True)) == {}


# ---------------------------------------------------------------------------
# partition
# ---------------------------------------------------------------------------
class TestPartition:
    def test_hostile_owners_and_counts_of_what_is_left_out(self) -> None:
        state = _state(
            units=[(1, 0), (2, 1), (3, 1), (4, 63), (5, 20), (6, 10)],
            cities=[(7, 1), (8, 10), (9, 20)],
            observes_others=True,
            other_players=[
                entry(1, "major", at_war=False),  # 2 units + 1 city left out as not hostile
                entry(10, "minor", at_war=True),  # hostile
                entry(63, "barbarian", at_war=True),  # hostile
                entry(20, "minor"),  # unknown: 1 unit + 1 city
            ],
        )
        hostile, left_out = partition(state)
        assert set(hostile) == {10, 63}
        assert left_out == {"not_hostile": 3, "unknown": 2}

    def test_an_owner_with_no_entry_is_counted_unknown(self) -> None:
        state = _state(units=[(2, 7), (3, 7)], observes_others=True)
        hostile, left_out = partition(state)
        assert set(hostile) == set()
        assert left_out == {"not_hostile": 0, "unknown": 2}

    def test_own_units_and_cities_are_never_counted(self) -> None:
        state = _state(units=[(1, 0), (2, 0)], cities=[(3, 0)], observes_others=True)
        hostile, left_out = partition(state)
        assert set(hostile) == set()
        assert left_out == {"not_hostile": 0, "unknown": 0}

    def test_everything_hostile_leaves_nothing_out(self) -> None:
        state = _state(
            units=[(2, 63)],
            cities=[(3, 63)],
            observes_others=True,
            other_players=[entry(63, "barbarian", at_war=True)],
        )
        hostile, left_out = partition(state)
        assert set(hostile) == {63}
        assert left_out == {"not_hostile": 0, "unknown": 0}
