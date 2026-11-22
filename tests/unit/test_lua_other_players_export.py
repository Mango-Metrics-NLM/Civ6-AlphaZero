"""GM-B8a: what the mod exports of other players, run in the stand-in game (lupa, Lua 5.1).

Assertions are on the EXPORTED STATE as the Python side would receive it. Expected key sets and
rules come from the written contract (``tests/fixtures/lua_others.py``), never from the mod.
Every "not exported" test also asserts that a visible control IS exported, so a mod that exports
nothing cannot pass.
"""

from __future__ import annotations

import pytest

from tests.fixtures import lua_harness as h
from tests.fixtures import lua_others as o

# The default enemy warrior 201 (player 1, at (2, 0)) is the visible control.
CONTROL = 201


def _ids(setup: str, *, observe: bool = True) -> list[int]:
    return o.foreign_unit_ids(o.state_of(setup, observe=observe))


class TestForeignUnits:
    def test_a_visible_enemy_unit_is_exported_with_exactly_the_allowed_keys(self) -> None:
        state = o.state_of(o.MAJOR_UNIT)
        by_id = {u["id"]: u for u in o.foreign_units(state)}
        assert set(by_id) == {CONTROL, 210}
        for unit in by_id.values():
            assert set(unit) == o.FOREIGN_UNIT_KEYS, set(unit) ^ o.FOREIGN_UNIT_KEYS
        unit = by_id[210]
        assert (unit["owner"], unit["x"], unit["y"]) == (1, 3, 0)
        assert (unit["type_name"], unit["hp"], unit["max_hp"], unit["damage"]) == (
            "UNIT_WARRIOR",
            90,
            100,
            10,
        )
        assert (unit["combat_strength"], unit["ranged_strength"], unit["range"]) == (20, 0, 0)
        assert unit["is_civilian"] is False

    @pytest.mark.parametrize("key", sorted(o.OWN_ONLY_UNIT_KEYS))
    def test_a_foreign_unit_never_carries_what_the_game_does_not_show(self, key: str) -> None:
        for unit in o.foreign_units(o.state_of(o.MAJOR_UNIT)):
            assert key not in unit

    def test_a_foreign_ranged_unit_carries_its_strengths(self) -> None:
        setup = "Civ6Stub.addUnit{ id = 213, owner = 1, x = 3, y = 0, ranged = 30, range = 2 }\n"
        (unit,) = [u for u in o.foreign_units(o.state_of(setup)) if u["id"] == 213]
        assert (unit["ranged_strength"], unit["range"]) == (30, 2)

    @pytest.mark.parametrize("hide", ["hideTile", "fogTile"])
    def test_a_unit_on_a_tile_not_visible_now_is_not_exported(self, hide: str) -> None:
        # fogTile = revealed but not visible now: a remembered tile shows no unit.
        ids = _ids(o.MAJOR_UNIT + f"Civ6Stub.{hide}(3, 0)\n")
        assert ids == [CONTROL]

    def test_a_stealth_unit_on_a_visible_tile_is_not_exported(self) -> None:
        ids = _ids(o.MAJOR_UNIT + "Civ6Stub.hideUnit(210)\n")
        assert ids == [CONTROL]

    @pytest.mark.parametrize(
        "spoil",
        [
            "Civ6Stub.unit(210).dead = true",
            "Civ6Stub.unit(210).delayedDeath = true",
            "Civ6Stub.unit(210).x = -9999",
        ],
        ids=["dead", "dying", "off the map"],
    )
    def test_a_dead_dying_or_off_map_unit_is_not_exported(self, spoil: str) -> None:
        assert _ids(o.MAJOR_UNIT + spoil + "\n") == [CONTROL]

    def test_a_unit_that_cannot_be_asked_if_it_is_dead_is_still_judged_by_what_can_be_asked(
        self,
    ) -> None:
        # The game's UI skips units that are dead; a context without IsDead must not cost the
        # export every foreign unit.
        setup = o.MAJOR_UNIT + "for _, u in Players[1]:GetUnits():Members() do u.IsDead = nil end\n"
        assert _ids(setup) == [CONTROL, 210]

    def test_own_units_are_unchanged(self) -> None:
        state = o.state_of(o.MAJOR_UNIT)
        own = [u for u in state["units"] if u["owner"] == 0]
        assert [u["id"] for u in own] == [101, 102]
        assert set(own[0]) == o.FOREIGN_UNIT_KEYS | o.OWN_ONLY_UNIT_KEYS


class TestFindingEveryKindOfPlayer:
    def test_a_barbarian_and_a_city_state_are_found(self) -> None:
        ids = _ids(o.MAJOR_UNIT + o.MINOR_UNIT + o.BARB_UNIT)
        assert ids == [CONTROL, 210, 211, 212]
        owners = {
            u["id"]: u["owner"] for u in o.foreign_units(o.state_of(o.MINOR_UNIT + o.BARB_UNIT))
        }
        assert owners[211] == 10
        assert owners[212] == 63

    @pytest.mark.parametrize(
        "lost",
        ["Civ6Stub.removeGetPlayers()", "Civ6Stub.breakGetPlayers()"],
        ids=["GetPlayers missing", "GetPlayers raises"],
    )
    def test_the_fallback_loop_still_finds_them(self, lost: str) -> None:
        ids = _ids(o.MINOR_UNIT + o.BARB_UNIT + lost + "\n")
        assert ids == [CONTROL, 211, 212]

    @pytest.mark.parametrize(
        "lost",
        [
            "Civ6Stub.removeGetPlayers()\nCiv6Stub.removeMaxPlayers()\n",
            "Civ6Stub.breakGetPlayers()\nCiv6Stub.removeMaxPlayers()\n",
        ],
        ids=["GetPlayers missing", "GetPlayers raises"],
    )
    def test_when_neither_way_works_nobody_is_found_and_nothing_raises(self, lost: str) -> None:
        state = o.state_of(o.MINOR_UNIT + o.BARB_UNIT + lost)
        assert o.foreign_units(state) == []
        assert o.foreign_cities(state) == []
        assert state["other_players"] == []
        assert state["observes_others"] is True
        assert [u["id"] for u in state["units"]] == [101, 102]

    @pytest.mark.parametrize("lost", ["", "Civ6Stub.removeGetPlayers()\n"], ids=["list", "loop"])
    def test_a_player_who_is_not_alive_is_not_listed(self, lost: str) -> None:
        setup = o.MINOR_UNIT + o.BARB_UNIT + lost + "Civ6Stub.kill(10)\n"
        assert _ids(setup) == [CONTROL, 212]

    def test_the_local_player_is_never_a_foreign_owner(self) -> None:
        state = o.state_of(o.MINOR_UNIT)
        assert 0 not in o.others(state)
        assert all(u["owner"] != 0 for u in o.foreign_units(state))


class TestOtherPlayersList:
    def test_kinds(self) -> None:
        state = o.state_of(o.MINOR_UNIT + o.BARB_UNIT)
        kinds = {p["id"]: p["kind"] for p in state["other_players"]}
        assert kinds == {1: "major", 10: "minor", 63: "barbarian"}

    def test_a_free_city_is_free_when_the_game_has_the_method_and_says_so(self) -> None:
        kinds = {
            p["id"]: p["kind"]
            for p in o.state_of(o.MINOR_UNIT + "Civ6Stub.addFreeCities(10, true)\n")[
                "other_players"
            ]
        }
        assert kinds[10] == "free"

    def test_free_wins_over_major(self) -> None:
        state = o.state_of("Civ6Stub.addFreeCities(1, true)\n")
        assert o.others(state)[1]["kind"] == "free"

    def test_a_free_cities_method_that_says_no_changes_nothing(self) -> None:
        state = o.state_of(o.MINOR_UNIT + "Civ6Stub.addFreeCities(10, false)\n")
        assert o.others(state)[10]["kind"] == "minor"

    def test_barbarian_wins_over_everything(self) -> None:
        setup = o.BARB_UNIT + "Civ6Stub.addFreeCities(63, true)\n"
        assert o.others(o.state_of(setup))[63]["kind"] == "barbarian"

    def test_at_war_and_met_come_from_the_game(self) -> None:
        peace = o.others(o.state_of(o.MINOR_UNIT))
        assert peace[1]["at_war"] is False
        assert peace[1]["met"] is False
        assert peace[10]["at_war"] is False
        setup = o.MINOR_UNIT + "Civ6Stub.setWar(0, 1, true)\nCiv6Stub.setMet(0, 10, true)\n"
        war = o.others(o.state_of(setup))
        assert (war[1]["at_war"], war[1]["met"]) == (True, False)
        assert (war[10]["at_war"], war[10]["met"]) == (False, True)

    def test_a_city_state_at_war_is_at_war(self) -> None:
        state = o.state_of(o.MINOR_UNIT + "Civ6Stub.setWar(0, 10, true)\n")
        assert o.others(state)[10]["at_war"] is True

    @pytest.mark.parametrize(
        "diplomacy", ["", "Civ6Stub.removeDiplomacy()\n"], ids=["asked", "gone"]
    )
    def test_a_barbarian_is_at_war_whatever_the_diplomacy_says(self, diplomacy: str) -> None:
        state = o.state_of(o.BARB_UNIT + diplomacy)
        assert o.others(state)[63]["at_war"] is True

    def test_without_diplomacy_war_and_met_are_absent_not_false(self) -> None:
        state = o.state_of(o.MINOR_UNIT + o.BARB_UNIT + "Civ6Stub.removeDiplomacy()\n")
        entries = o.others(state)
        assert set(entries) == {1, 10, 63}
        for pid in (1, 10):
            assert "at_war" not in entries[pid]
            assert "met" not in entries[pid]
            assert set(entries[pid]) == {"id", "kind"}
        assert "met" not in entries[63]

    @pytest.mark.parametrize(
        ("method", "absent", "present"),
        [("IsAtWarWith", "at_war", "met"), ("HasMet", "met", "at_war")],
    )
    def test_a_missing_diplomacy_method_costs_only_its_own_field(
        self, method: str, absent: str, present: str
    ) -> None:
        setup = (
            "do local ask = Players[0].GetDiplomacy\n"
            "Players[0].GetDiplomacy = function(self) local d = ask(self)\n"
            f"d.{method} = nil return d end end\n"
        )
        entry = o.others(o.state_of(setup))[1]
        assert absent not in entry
        assert entry[present] is False

    def test_only_owners_that_appear_are_listed(self) -> None:
        # Player 63's unit is on a hidden tile and 10 has nothing: neither is listed.
        setup = o.BARB_UNIT + "Civ6Stub.hideTile(0, 2)\n"
        assert [p["id"] for p in o.state_of(setup)["other_players"]] == [1]

    def test_an_owner_of_only_a_visible_city_is_listed(self) -> None:
        setup = "Civ6Stub.addCity{ id = 401, owner = 10, x = 2, y = 1 }\n"
        assert [p["id"] for p in o.state_of(setup)["other_players"]] == [1, 10]

    def test_entries_are_sorted_by_id(self) -> None:
        # Added in the opposite order.
        setup = o.BARB_UNIT + o.MINOR_UNIT + o.MAJOR_UNIT
        assert [p["id"] for p in o.state_of(setup)["other_players"]] == [1, 10, 63]


class TestForeignCities:
    CITY = (
        "Civ6Stub.addCity{ id = 401, owner = 10, x = 2, y = 1, name = 'Rival', population = 4,"
        " building = 'UNIT_WARRIOR' }\n"
    )

    def test_a_city_on_a_visible_tile_is_exported_without_what_the_game_hides(self) -> None:
        (city,) = o.foreign_cities(o.state_of(self.CITY))
        assert set(city) == o.FOREIGN_CITY_KEYS, set(city) ^ o.FOREIGN_CITY_KEYS
        assert not (set(city) & o.FORBIDDEN_CITY_KEYS)
        assert (city["id"], city["name"], city["owner"], city["x"], city["y"]) == (
            401,
            "Rival",
            10,
            2,
            1,
        )
        assert city["population"] == 4
        assert city["is_capital"] is True
        assert city["buildings"] == []
        assert city["districts"] == []
        assert city["available_production"] == []

    @pytest.mark.parametrize("hide", ["hideTile", "fogTile"])
    def test_a_city_on_a_tile_not_visible_now_is_not_exported(self, hide: str) -> None:
        # fogTile is revealed-but-not-visible: the city is remembered, and not exported.
        state = o.state_of(self.CITY + f"Civ6Stub.{hide}(2, 1)\n")
        assert o.foreign_cities(state) == []
        assert 10 not in o.others(state)
        assert [c["id"] for c in state["cities"]] == [301]

    def test_the_own_city_keeps_its_detail(self) -> None:
        (own,) = [c for c in o.state_of(self.CITY)["cities"] if c["owner"] == 0]
        assert own["buildings"] == ["BUILDING_MONUMENT"]
        assert own["districts"] == ["DISTRICT_CITY_CENTER", "DISTRICT_CAMPUS"]
        assert "defense_strength" in own


class TestFlagOff:
    EXTRAS = (
        o.MAJOR_UNIT
        + o.MINOR_UNIT
        + o.BARB_UNIT
        + "Civ6Stub.addCity{ id = 401, owner = 10, x = 2, y = 1 }\n"
        + "Civ6Stub.setWar(0, 1, true)\n"
    )

    def test_the_default_is_off(self) -> None:
        assert o.load(observe=False).eval('AlphaZeroConfig.Get("OBSERVE_ALL_PLAYERS")') is False

    def test_nothing_foreign_is_exported_and_the_flag_is_reported_off(self) -> None:
        state = o.state_of(self.EXTRAS, observe=False)
        assert o.foreign_units(state) == []
        assert o.foreign_cities(state) == []
        assert state["other_players"] == []
        assert state["observes_others"] is False

    def test_the_flag_is_reported_on_when_on(self) -> None:
        assert o.state_of(observe=True)["observes_others"] is True

    def test_the_own_export_is_exactly_what_it_is_without_the_other_players(self) -> None:
        # The same game with no extra unit, city or war: with the flag off, nothing differs.
        assert o.state_of(self.EXTRAS, observe=False) == o.state_of("", observe=False)

    def test_own_units_and_cities_are_complete(self) -> None:
        state = o.state_of(self.EXTRAS, observe=False)
        assert [u["id"] for u in state["units"]] == [101, 102]
        assert set(state["units"][0]) == o.FOREIGN_UNIT_KEYS | o.OWN_ONLY_UNIT_KEYS
        assert [c["id"] for c in state["cities"]] == [301]


class TestWithoutVisibility:
    @pytest.mark.parametrize(
        "lost",
        [
            "Civ6Stub.removeVisibility()",
            "Civ6Stub.removeVisibilityEntry()",
            "Civ6Stub.removeVisibilityMethod('IsUnitVisible')",
            "Civ6Stub.removeVisibilityMethod('IsVisible')",
            "PlayersVisibility[0].IsUnitVisible = function() error('not in this context') end",
        ],
        ids=["no table", "no entry", "no IsUnitVisible", "no IsVisible", "IsUnitVisible raises"],
    )
    def test_no_foreign_unit_is_exported_and_nothing_raises(self, lost: str) -> None:
        state = o.state_of(o.MAJOR_UNIT + lost + "\n")
        assert o.foreign_units(state) == []
        assert [u["id"] for u in state["units"]] == [101, 102]

    @pytest.mark.parametrize(
        "lost",
        ["Civ6Stub.removeVisibility()", "Civ6Stub.removeVisibilityEntry()"],
        ids=["no table", "no entry"],
    )
    def test_without_a_visibility_table_no_foreign_city_either_and_nobody_is_listed(
        self, lost: str
    ) -> None:
        setup = o.MAJOR_UNIT + "Civ6Stub.addCity{ id = 401, owner = 10, x = 2, y = 1 }\n" + lost
        state = o.state_of(setup + "\n")
        assert o.foreign_cities(state) == []
        assert state["other_players"] == []

    def test_a_city_needs_only_to_be_visible_so_a_missing_unit_check_does_not_hide_it(self) -> None:
        setup = (
            "Civ6Stub.addCity{ id = 401, owner = 10, x = 2, y = 1 }\n"
            "Civ6Stub.removeVisibilityMethod('IsUnitVisible')\n"
        )
        state = o.state_of(setup)
        assert [c["id"] for c in o.foreign_cities(state)] == [401]


class TestOwnFirst:
    SETUP = (
        "Civ6Stub.addUnit{ id = 5, owner = 63, x = 0, y = 2 }\n"
        "Civ6Stub.addUnit{ id = 6, owner = 1, x = 3, y = 0 }\n"
        "Civ6Stub.addCity{ id = 5, owner = 10, x = 2, y = 1 }\n"
    )

    def test_own_units_and_cities_come_first(self) -> None:
        state = o.state_of(self.SETUP)
        assert [u["owner"] for u in state["units"]][:2] == [0, 0]
        assert next(c["owner"] for c in state["cities"]) == 0
        assert {u["id"] for u in o.foreign_units(state)} == {5, 6, CONTROL}
        assert [c["id"] for c in o.foreign_cities(state)] == [5]

    def test_own_come_first_for_a_player_who_is_not_player_zero(self) -> None:
        runtime = o.load(self.SETUP)
        state = h.export_state(runtime, player_id=1)
        units = state["units"]
        own = [u for u in units if u["owner"] == 1]
        assert own, "premise: player 1 has units"
        assert [u["owner"] for u in units[: len(own)]] == [1] * len(own)
        assert any(u["owner"] == 0 for u in units), "premise: player 0's units are seen too"
