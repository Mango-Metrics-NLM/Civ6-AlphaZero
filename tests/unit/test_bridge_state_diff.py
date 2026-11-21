"""GM-B7c-1: ``alpha_zero.bridge.state_diff`` (summarize, changes, expected_effect).

States are built through ``GameStateData.from_dict`` (the wire shape the mod
writes), so the fixtures are what the tailer would hand over.

Unit move counts are whole numbers in every fixture: the protocol carries
``moves_remaining`` as a float, and the contract says the summary holds integers.
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha_zero.bridge.protocol import GameStateData
from alpha_zero.bridge.state_diff import changes, expected_effect, summarize

ME = 0
OTHER = 1


def _unit(uid: int, x: int, y: int, moves: int = 2, owner: int = ME) -> dict[str, Any]:
    return {
        "id": uid,
        "unit_type": "UNIT_WARRIOR",
        "owner": owner,
        "x": x,
        "y": y,
        "movement": moves,
        "max_movement": 2,
    }


def _city(cid: int, producing: str | None, owner: int = ME) -> dict[str, Any]:
    return {
        "id": cid,
        "name": f"City{cid}",
        "owner": owner,
        "x": 1,
        "y": 1,
        "current_production": producing,
    }


def make_state(
    *,
    turn: int = 5,
    units: list[dict[str, Any]] | None = None,
    cities: list[dict[str, Any]] | None = None,
    research: str | None = "TECH_MINING",
    civic: str | None = "CIVIC_CODE_OF_LAWS",
    blocking: str | None = None,
    player_id: int = ME,
) -> GameStateData:
    me = {
        "id": player_id,
        "civilization": "CIVILIZATION_ROME",
        "current_research": research,
        "current_civic": civic,
    }
    other = {
        "id": 1 - player_id,
        "civilization": "CIVILIZATION_EGYPT",
        "current_research": "TECH_X",
    }
    raw: dict[str, Any] = {
        "turn_id": turn,
        "player_id": player_id,
        "map": {"width": 0, "height": 0, "tiles": []},
        "units": units if units is not None else [_unit(1, 2, 3)],
        "cities": cities if cities is not None else [_city(10, "UNIT_SETTLER")],
        # the opponent first, so "own research" cannot be read from players[0]
        "players": [other, me] if player_id == ME else [me, other],
        "end_turn_blocking": blocking,
    }
    return GameStateData.from_dict(raw)


# --- summarize ---


class TestSummarize:
    def test_shape_and_values(self) -> None:
        state = make_state(
            units=[_unit(1, 2, 3, moves=2), _unit(2, 4, 5, moves=0)],
            cities=[_city(10, "UNIT_SETTLER"), _city(11, None)],
            blocking="ENDTURN_BLOCKING_UNITS",
        )
        assert summarize(state) == {
            "turn_id": 5,
            "units": {1: {"x": 2, "y": 3, "moves": 2}, 2: {"x": 4, "y": 5, "moves": 0}},
            "cities": {10: {"producing": "UNIT_SETTLER"}, 11: {"producing": None}},
            "research": "TECH_MINING",
            "civic": "CIVIC_CODE_OF_LAWS",
            "end_turn_blocking": "ENDTURN_BLOCKING_UNITS",
        }

    def test_only_own_units_and_cities(self) -> None:
        state = make_state(
            units=[_unit(1, 2, 3), _unit(2, 8, 8, owner=OTHER)],
            cities=[_city(10, "A"), _city(20, "B", owner=OTHER)],
        )
        summary = summarize(state)
        assert set(summary["units"]) == {1}
        assert set(summary["cities"]) == {10}

    def test_own_means_the_states_player_id(self) -> None:
        state = make_state(
            player_id=1,
            units=[_unit(1, 2, 3, owner=0), _unit(2, 4, 4, owner=1)],
            cities=[_city(10, "A", owner=0), _city(20, "B", owner=1)],
        )
        summary = summarize(state)
        assert set(summary["units"]) == {2}
        assert set(summary["cities"]) == {20}

    def test_research_and_civic_are_the_own_players(self) -> None:
        summary = summarize(make_state(research="TECH_BRONZE_WORKING", civic="CIVIC_X"))
        assert summary["research"] == "TECH_BRONZE_WORKING"
        assert summary["civic"] == "CIVIC_X"

    def test_no_research_no_civic_no_blocker_are_none(self) -> None:
        summary = summarize(make_state(research=None, civic=None, blocking=None))
        assert summary["research"] is None
        assert summary["civic"] is None
        assert summary["end_turn_blocking"] is None

    def test_empty_state(self) -> None:
        summary = summarize(make_state(units=[], cities=[]))
        assert summary["units"] == {}
        assert summary["cities"] == {}

    def test_values_are_plain_numbers_and_strings(self) -> None:
        # moves_remaining is a float in the protocol; the contract says "integers", so only
        # the whole-number-ness of tile and turn is pinned and moves must be a plain number
        summary = summarize(make_state())
        unit = summary["units"][1]
        assert type(unit["x"]) is int
        assert type(unit["y"]) is int
        assert type(unit["moves"]) in (int, float)
        assert type(summary["turn_id"]) is int

    def test_summary_does_not_alias_the_state(self) -> None:
        state = make_state()
        summary = summarize(state)
        state.units[0].x = 99
        assert summary["units"][1]["x"] == 2


# --- changes ---


class TestChanges:
    def test_nothing_changed_is_empty(self) -> None:
        assert changes(make_state(), make_state()) == []

    def test_accepts_summaries_and_states_alike(self) -> None:
        before, after = make_state(), make_state(units=[_unit(1, 3, 3)])
        assert changes(summarize(before), summarize(after)) == changes(before, after)
        assert changes(summarize(before), after) == changes(before, after)

    def test_unit_moved(self) -> None:
        lines = changes(make_state(units=[_unit(41, 2, 3)]), make_state(units=[_unit(41, 3, 3)]))
        assert len(lines) == 1
        assert isinstance(lines[0], str)
        assert "41" in lines[0]

    def test_unit_moves_changed(self) -> None:
        lines = changes(
            make_state(units=[_unit(41, 2, 3, moves=2)]),
            make_state(units=[_unit(41, 2, 3, moves=0)]),
        )
        assert len(lines) == 1
        assert "41" in lines[0]

    def test_unit_gone(self) -> None:
        lines = changes(make_state(units=[_unit(41, 2, 3)]), make_state(units=[]))
        assert len(lines) == 1
        assert "41" in lines[0]

    def test_unit_new(self) -> None:
        lines = changes(make_state(units=[]), make_state(units=[_unit(41, 2, 3)]))
        assert len(lines) == 1
        assert "41" in lines[0]

    def test_other_players_units_are_not_reported(self) -> None:
        before = make_state(units=[_unit(1, 2, 3), _unit(2, 8, 8, owner=OTHER)])
        after = make_state(units=[_unit(1, 2, 3), _unit(2, 9, 9, owner=OTHER)])
        assert changes(before, after) == []

    def test_city_production_changed(self) -> None:
        lines = changes(
            make_state(cities=[_city(10, "UNIT_SETTLER")]),
            make_state(cities=[_city(10, "BUILDING_MONUMENT")]),
        )
        assert len(lines) == 1
        assert "BUILDING_MONUMENT" in lines[0]

    def test_research_changed(self) -> None:
        lines = changes(make_state(research="TECH_MINING"), make_state(research="TECH_POTTERY"))
        assert len(lines) == 1
        assert "TECH_POTTERY" in lines[0]

    def test_civic_changed(self) -> None:
        lines = changes(make_state(civic="CIVIC_A"), make_state(civic="CIVIC_B"))
        assert len(lines) == 1
        assert "CIVIC_B" in lines[0]

    def test_blocker_changed(self) -> None:
        lines = changes(make_state(blocking=None), make_state(blocking="ENDTURN_BLOCKING_RESEARCH"))
        assert len(lines) == 1
        assert "ENDTURN_BLOCKING_RESEARCH" in lines[0]

    def test_blocker_cleared(self) -> None:
        lines = changes(make_state(blocking="ENDTURN_BLOCKING_RESEARCH"), make_state(blocking=None))
        assert len(lines) == 1

    def test_turn_changed(self) -> None:
        lines = changes(make_state(turn=5), make_state(turn=12))
        assert len(lines) == 1
        assert "12" in lines[0]

    def test_one_line_per_difference(self) -> None:
        before = make_state(turn=5, research="TECH_MINING", blocking=None)
        after = make_state(turn=6, research="TECH_POTTERY", blocking="ENDTURN_BLOCKING_UNITS")
        assert len(changes(before, after)) == 3

    def test_units_are_reported_in_id_order(self) -> None:
        before = make_state(units=[_unit(97, 1, 1), _unit(41, 2, 2)])
        after = make_state(units=[_unit(97, 5, 5), _unit(41, 6, 6)])
        lines = changes(before, after)
        assert len(lines) == 2
        assert "41" in lines[0]
        assert "97" in lines[1]

    def test_order_does_not_depend_on_input_order(self) -> None:
        a = [_unit(97, 1, 1), _unit(41, 2, 2), _unit(63, 3, 3)]
        b = [_unit(97, 5, 5), _unit(41, 6, 6), _unit(63, 7, 7)]
        first = changes(make_state(units=a), make_state(units=b))
        second = changes(make_state(units=list(reversed(a))), make_state(units=list(reversed(b))))
        assert first == second

    def test_deterministic_across_calls(self) -> None:
        before = make_state(turn=5, units=[_unit(1, 2, 3)], research="TECH_A")
        after = make_state(turn=6, units=[_unit(1, 3, 3)], research="TECH_B")
        assert changes(before, after) == changes(before, after)


# --- expected_effect ---


def _cmd(action: str, **fields: Any) -> dict[str, Any]:
    return {"action_type": action, **fields}


MOVE = _cmd("unit_move", unit_id=1, target={"x": 3, "y": 3})


class TestExpectedEffectUnitMove:
    def test_tile_changed(self) -> None:
        assert expected_effect(MOVE, make_state(), make_state(units=[_unit(1, 3, 3)])) is True

    def test_moves_changed(self) -> None:
        after = make_state(units=[_unit(1, 2, 3, moves=1)])
        assert expected_effect(MOVE, make_state(), after) is True

    def test_unit_gone(self) -> None:
        assert expected_effect(MOVE, make_state(), make_state(units=[])) is True

    def test_nothing_changed(self) -> None:
        assert expected_effect(MOVE, make_state(), make_state()) is False

    def test_other_units_moving_does_not_count(self) -> None:
        before = make_state(units=[_unit(1, 2, 3), _unit(2, 4, 4)])
        after = make_state(units=[_unit(1, 2, 3), _unit(2, 5, 5)])
        assert expected_effect(MOVE, before, after) is False

    def test_accepts_summaries(self) -> None:
        after = make_state(units=[_unit(1, 3, 3)])
        assert expected_effect(MOVE, summarize(make_state()), summarize(after)) is True


@pytest.mark.parametrize("action", ["unit_attack", "unit_ranged_attack"])
class TestExpectedEffectAttack:
    def test_moves_changed(self, action: str) -> None:
        cmd = _cmd(action, unit_id=1, target={"x": 3, "y": 3})
        after = make_state(units=[_unit(1, 2, 3, moves=0)])
        assert expected_effect(cmd, make_state(), after) is True

    def test_unit_gone(self, action: str) -> None:
        cmd = _cmd(action, unit_id=1, target={"x": 3, "y": 3})
        assert expected_effect(cmd, make_state(), make_state(units=[])) is True

    def test_nothing_changed(self, action: str) -> None:
        cmd = _cmd(action, unit_id=1, target={"x": 3, "y": 3})
        assert expected_effect(cmd, make_state(), make_state()) is False

    def test_a_move_of_the_tile_alone_is_not_an_attack(self, action: str) -> None:
        cmd = _cmd(action, unit_id=1, target={"x": 3, "y": 3})
        after = make_state(units=[_unit(1, 3, 3, moves=2)])
        assert expected_effect(cmd, make_state(), after) is False


@pytest.mark.parametrize("action", ["unit_skip", "unit_fortify", "unit_sleep"])
class TestExpectedEffectUnitStance:
    def test_moves_changed(self, action: str) -> None:
        after = make_state(units=[_unit(1, 2, 3, moves=0)])
        assert expected_effect(_cmd(action, unit_id=1), make_state(), after) is True

    def test_blocker_changed(self, action: str) -> None:
        before = make_state(blocking="ENDTURN_BLOCKING_UNITS")
        after = make_state(blocking=None)
        assert expected_effect(_cmd(action, unit_id=1), before, after) is True

    def test_nothing_changed(self, action: str) -> None:
        assert expected_effect(_cmd(action, unit_id=1), make_state(), make_state()) is False


class TestExpectedEffectFoundCity:
    FOUND = _cmd("unit_found_city", unit_id=1)

    def test_unit_gone(self) -> None:
        assert expected_effect(self.FOUND, make_state(), make_state(units=[])) is True

    def test_new_own_city(self) -> None:
        after = make_state(cities=[_city(10, "UNIT_SETTLER"), _city(11, None)])
        assert expected_effect(self.FOUND, make_state(), after) is True

    def test_new_enemy_city_does_not_count(self) -> None:
        after = make_state(cities=[_city(10, "UNIT_SETTLER"), _city(30, None, owner=OTHER)])
        assert expected_effect(self.FOUND, make_state(), after) is False

    def test_nothing_changed(self) -> None:
        assert expected_effect(self.FOUND, make_state(), make_state()) is False


class TestExpectedEffectCityAndResearch:
    PRODUCE = _cmd("city_produce", city_id=10, item_id=3, item_type="unit")

    def test_city_production_changed(self) -> None:
        after = make_state(cities=[_city(10, "UNIT_WARRIOR")])
        assert expected_effect(self.PRODUCE, make_state(), after) is True

    def test_other_city_changing_does_not_count(self) -> None:
        before = make_state(cities=[_city(10, "A"), _city(11, "B")])
        after = make_state(cities=[_city(10, "A"), _city(11, "C")])
        assert expected_effect(self.PRODUCE, before, after) is False

    def test_production_unchanged(self) -> None:
        assert expected_effect(self.PRODUCE, make_state(), make_state()) is False

    def test_research_tech_changed(self) -> None:
        cmd = _cmd("research_tech", item_id="TECH_POTTERY")
        assert expected_effect(cmd, make_state(), make_state(research="TECH_POTTERY")) is True

    def test_research_tech_blocker_changed(self) -> None:
        cmd = _cmd("research_tech", item_id="TECH_POTTERY")
        before = make_state(research=None, blocking="ENDTURN_BLOCKING_RESEARCH")
        assert expected_effect(cmd, before, make_state(research=None, blocking=None)) is True

    def test_research_tech_unchanged(self) -> None:
        cmd = _cmd("research_tech", item_id="TECH_POTTERY")
        assert expected_effect(cmd, make_state(), make_state()) is False

    def test_research_tech_ignores_a_civic_change(self) -> None:
        cmd = _cmd("research_tech", item_id="TECH_POTTERY")
        assert expected_effect(cmd, make_state(), make_state(civic="CIVIC_Z")) is False

    def test_research_civic_changed(self) -> None:
        cmd = _cmd("research_civic", item_id="CIVIC_B")
        assert expected_effect(cmd, make_state(), make_state(civic="CIVIC_B")) is True

    def test_research_civic_blocker_changed(self) -> None:
        cmd = _cmd("research_civic", item_id="CIVIC_B")
        before = make_state(blocking="ENDTURN_BLOCKING_CIVIC")
        assert expected_effect(cmd, before, make_state(blocking=None)) is True

    def test_research_civic_unchanged(self) -> None:
        cmd = _cmd("research_civic", item_id="CIVIC_B")
        assert expected_effect(cmd, make_state(), make_state()) is False

    def test_research_civic_ignores_a_tech_change(self) -> None:
        cmd = _cmd("research_civic", item_id="CIVIC_B")
        assert expected_effect(cmd, make_state(), make_state(research="TECH_Z")) is False


class TestExpectedEffectEndTurn:
    END = _cmd("end_turn")

    def test_turn_changed(self) -> None:
        assert expected_effect(self.END, make_state(turn=5), make_state(turn=6)) is True

    def test_turn_unchanged(self) -> None:
        assert expected_effect(self.END, make_state(turn=5), make_state(turn=5)) is False

    def test_a_blocker_does_not_count_as_ending_the_turn(self) -> None:
        after = make_state(turn=5, blocking="ENDTURN_BLOCKING_UNITS")
        assert expected_effect(self.END, make_state(turn=5), after) is False


class TestExpectedEffectNotJudged:
    @pytest.mark.parametrize(
        "action",
        ["unit_build", "unit_promote", "unit_delete", "city_buy", "city_citizen", "diplomacy"],
    )
    def test_other_actions_are_not_judged(self, action: str) -> None:
        cmd = _cmd(action, unit_id=1, city_id=10)
        changed = make_state(turn=9, units=[_unit(1, 9, 9)])
        assert expected_effect(cmd, make_state(), changed) is None
        assert expected_effect(cmd, make_state(), make_state()) is None
