"""What the independent review of GM-B8a found, each pinned.

* another player's unit (or a whole player) that cannot be read emptied the unit list, the
  player's own units included;
* a refused move described an occupant the player cannot see ("occupied", or which player a
  war would be with);
* a player whose kind could not be asked was reported as a city-state, so barbarians whose
  ``IsBarbarian`` raised came out as a peaceful minor;
* the stand-in kept two records of war, so the mod could be offered an attack on a unit of a
  player it was told it is at peace with.
"""

from __future__ import annotations

import pytest

from alpha_zero.bridge.other_players import owner_relations
from alpha_zero.bridge.protocol import GameStateData
from tests.fixtures import lua_harness as h
from tests.fixtures import lua_others as o

OWN = {101, 102}
MOVE = {"action_type": "unit_move", "unit_id": 102, "target": {"x": 2, "y": 0}}


def _moved(game: object) -> tuple[object, object, str]:
    ok, code, message = h.execute(game, MOVE)
    return ok, code, str(message)


def _unit_ids(state: dict) -> set[int]:
    return {u["id"] for u in state["units"]}


class TestAForeignFailureDoesNotCostTheOwnUnits:
    def test_a_foreign_unit_that_cannot_be_read(self) -> None:
        setup = o.BARB_UNIT + (
            "for _, u in Players[63]:GetUnits():Members() do\n"
            "  u.GetUnitType = function() error('not in this context') end\nend\n"
        )
        ids = _unit_ids(o.state_of(setup))
        assert OWN <= ids
        assert 201 in ids, "the other foreign player's unit is still exported"

    def test_a_foreign_player_whose_units_cannot_be_walked(self) -> None:
        setup = o.BARB_UNIT + (
            "local units = Players[63]:GetUnits()\n"
            "Players[63].GetUnits = function() return { Members = function()"
            " error('no') end } end\n"
        )
        assert OWN <= _unit_ids(o.state_of(setup))

    def test_a_foreign_player_whose_cities_cannot_be_walked(self) -> None:
        setup = (
            "Players[1].GetCities = function() return { Members = function()"
            " error('no') end } end\n"
        )
        state = o.state_of(setup)
        assert [city["owner"] for city in state["cities"]] == [0]


class TestARefusedMoveDoesNotDescribeWhatIsNotSeen:
    """Unit 201 (player 1) stands at (2, 0), next to the player's warrior."""

    @pytest.mark.parametrize("peace", ["", "Civ6Stub.setPeace(0, 1, true)\n"], ids=["war", "peace"])
    def test_a_move_onto_a_hidden_unit_says_only_that_it_was_refused(self, peace: str) -> None:
        game = o.load(peace + "Civ6Stub.hideUnit(201)\n")
        ok, code, message = _moved(game)
        assert ok is False
        assert code == "OPERATION_REFUSED"
        assert "occupied" not in message.lower()
        assert "war" not in message.lower()
        assert "player" not in message.lower()

    def test_control_a_seen_unit_at_peace_still_names_the_war(self) -> None:
        game = o.load("Civ6Stub.setPeace(0, 1, true)\n")
        ok, code, _ = _moved(game)
        assert (ok, code) == (False, "WOULD_START_WAR")


class TestAKindThatCannotBeAskedIsUnknown:
    @pytest.mark.parametrize("method", ["IsBarbarian", "IsMajor"])
    def test_a_raising_method_gives_unknown_not_minor(self, method: str) -> None:
        setup = o.BARB_UNIT + f"Players[63].{method} = function() error('no') end\n"
        entry = o.others(o.state_of(setup))[63]
        if method == "IsBarbarian":
            assert entry["kind"] == "unknown"
        else:
            assert entry["kind"] == "barbarian"

    def test_a_major_whose_kind_cannot_be_asked_is_unknown(self) -> None:
        setup = "Players[1].IsMajor = function() error('no') end\n"
        assert o.others(o.state_of(setup))[1]["kind"] == "unknown"

    def test_control_kinds_that_can_be_asked(self) -> None:
        entries = o.others(o.state_of(o.MINOR_UNIT + o.BARB_UNIT))
        assert [entries[i]["kind"] for i in (1, 10, 63)] == ["major", "minor", "barbarian"]

    def test_python_does_not_call_an_unknown_kind_peaceful(self) -> None:
        state = GameStateData.from_dict(
            {
                "turn_id": 1,
                "player_id": 0,
                "map": {"width": 4, "height": 4},
                "units": [{"id": 9, "owner": 63, "x": 1, "y": 1, "type_name": "UNIT_WARRIOR"}],
                "observes_others": True,
                "other_players": [{"id": 63, "kind": "unknown", "at_war": False}],
            }
        )
        assert owner_relations(state) == {63: "unknown"}


class TestTheStandInHasOneRecordOfWar:
    @pytest.mark.parametrize("peace", [False, True])
    def test_at_war_and_the_attack_offer_agree(self, peace: bool) -> None:
        setup = "Civ6Stub.setPeace(0, 1, true)\n" if peace else ""
        state = o.state_of(setup)
        at_war = o.others(state)[1]["at_war"]
        attacks = [a for a in state["legal_actions"] if a["action"] in ("attack", "ranged_attack")]
        assert at_war is (not peace)
        assert bool(attacks) is (not peace)
