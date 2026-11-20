"""GM-B1: characterise the AlphaZeroAI mod's Lua by executing it (lupa, Lua 5.1).

This slice pins what the Lua does TODAY, known gaps included. Each gap test carries the
comment ``today's behaviour; GM-B2 changes this`` so GM-B2 flips it deliberately.

The exporter wraps every component in ``pcall`` and falls back to an empty value, so an
export that is missing a component reads as a pass unless something asserts the
component is there. Every component check below is therefore paired with a negative
control (``TestNonVacuity``): break one stub method and prove the check fails.

``lupa`` is an optional extra, so it is never imported at module scope: the harness's
``import_lua51()`` skips (or, under ``CIV6AZ_REQUIRE_LUA=1``, fails) when it is absent.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from alpha_zero.bridge.protocol import GameStateData, StateMessage
from tests.fixtures import lua_harness as h

REPO_ROOT = Path(__file__).resolve().parents[2]

# Needs lupa: every test below that touches Lua calls ``h.import_lua51()`` through
# ``new_runtime``/``load_mod``, which skips or fails per ``CIV6AZ_REQUIRE_LUA``.


@pytest.fixture
def runtime() -> Any:
    return h.load_mod()


@pytest.fixture
def state(runtime: Any) -> dict[str, Any]:
    return h.export_state(runtime)


def _tile(state: dict[str, Any], x: int, y: int) -> dict[str, Any]:
    matches = [t for t in state["map"]["tiles"] if (t["x"], t["y"]) == (x, y)]
    assert len(matches) == 1, f"tile ({x},{y}) not found exactly once"
    return matches[0]


def _action_names(state: dict[str, Any]) -> list[str]:
    return [a["action"] for a in state["legal_actions"]]


def _actions_for(state: dict[str, Any], name: str, **match: Any) -> list[dict[str, Any]]:
    return [
        a
        for a in state["legal_actions"]
        if a["action"] == name and all(a.get(k) == v for k, v in match.items())
    ]


# ---------------------------------------------------------------------------
# Non-vacuity checks: each asserts one export component is really present.
# ---------------------------------------------------------------------------
def check_map(state: dict[str, Any]) -> None:
    assert (state["map"]["width"], state["map"]["height"]) == (4, 3)
    assert len(state["map"]["tiles"]) == 12
    special = _tile(state, 1, 1)
    assert special["terrain"] == "TERRAIN_GRASS"
    assert special["features"] == ["FEATURE_FOREST", "FEATURE_HILLS"]
    assert special.get("resource") == "RESOURCE_WHEAT"
    assert special.get("improvement") == "IMPROVEMENT_FARM"
    assert special["owner_id"] == 0
    assert special["yields"]["food"] == 2
    assert special["yields"]["production"] == 1


def check_units(state: dict[str, Any]) -> None:
    by_id = {u["id"]: u for u in state["units"]}
    assert sorted(by_id) == [101, 102]
    assert (by_id[101]["type_name"], by_id[101]["x"], by_id[101]["y"]) == ("UNIT_SETTLER", 0, 0)
    assert by_id[101]["is_civilian"] is True
    assert (by_id[102]["type_name"], by_id[102]["x"], by_id[102]["y"]) == ("UNIT_WARRIOR", 1, 0)
    assert by_id[102]["is_civilian"] is False
    # Values the exporter's per-unit fallback would replace with 0 / false.
    assert (by_id[101]["movement"], by_id[102]["movement"]) == (2, 2)
    assert by_id[102]["combat_strength"] == 20
    assert by_id[102]["can_attack"] is True


def check_city(state: dict[str, Any]) -> None:
    assert [c["id"] for c in state["cities"]] == [301]
    city = state["cities"][0]
    assert city["name"] == "Stubville"
    assert city["population"] == 3
    assert (city["x"], city["y"]) == (1, 1)


def check_player_gold(state: dict[str, Any]) -> None:
    assert state["player"].get("gold") == 50


def check_player_science(state: dict[str, Any]) -> None:
    assert state["player"].get("science_per_turn") == 4


def check_player_civic(state: dict[str, Any]) -> None:
    assert state["player"].get("current_civic") == "CIVIC_CRAFTSMANSHIP"


def check_player(state: dict[str, Any]) -> None:
    check_player_gold(state)
    check_player_science(state)
    check_player_civic(state)
    assert state["player"].get("civilization") == "CIVILIZATION_STUB_A"


def check_turn(state: dict[str, Any]) -> None:
    assert state["turn_id"] == 12


def check_era(state: dict[str, Any]) -> None:
    assert state["era"] == 1


def check_legal_actions(state: dict[str, Any]) -> None:
    assert _actions_for(state, "found_city", unit_id=101)
    assert _actions_for(state, "attack", unit_id=102)
    assert _actions_for(state, "produce", city_id=301)


#: Lua 5.1: ``module(name)`` makes the loaded library ``name`` the function's
#: environment, so a removed library's functions come back as globals.
_MODULE_ESCAPE = "local function f() module('{library}'); return {function} end; return f() ~= nil"


class TestSandbox:
    def test_forbidden_globals_are_nil(self) -> None:
        runtime = h.new_runtime()
        for name in ("io", "dofile", "loadfile", "require", "package", "debug", "module"):
            assert runtime.eval(f"{name} == nil") is True, f"{name} is reachable from Lua"

    @pytest.mark.parametrize(
        ("library", "function"), [("os", "execute"), ("os", "remove"), ("io", "open")]
    )
    def test_module_cannot_bring_a_removed_library_back(self, library: str, function: str) -> None:
        runtime = h.new_runtime()
        with pytest.raises(Exception, match="module"):
            runtime.execute(_MODULE_ESCAPE.format(library=library, function=function))

    @pytest.mark.parametrize(("library", "function"), [("os", "execute"), ("io", "open")])
    def test_negative_control_module_does_bring_it_back_when_left_in(
        self, library: str, function: str
    ) -> None:
        # Removing the globals alone (the lockdown as it first shipped) is not
        # enough: with ``module`` left in, the library is one call away.
        lua51 = h.import_lua51()
        runtime = lua51.LuaRuntime(
            unpack_returned_tuples=True, register_eval=False, register_builtins=False
        )
        runtime.execute("io = nil package = nil os = { time = os.time }")
        assert runtime.execute(_MODULE_ESCAPE.format(library=library, function=function)) is True

    def test_the_loaded_library_registry_is_cleared(self) -> None:
        # Belt and braces: even with ``module`` restored by hand it finds nothing.
        lua51 = h.import_lua51()
        runtime = lua51.LuaRuntime(
            unpack_returned_tuples=True, register_eval=False, register_builtins=False
        )
        runtime.execute("saved_module = module")
        runtime.execute(h._LOCKDOWN)
        runtime.execute("module = saved_module")
        for library, function in (("os", "execute"), ("io", "open")):
            assert (
                runtime.execute(_MODULE_ESCAPE.format(library=library, function=function)) is False
            )

    def test_os_has_only_time_clock_date(self) -> None:
        runtime = h.new_runtime()
        keys = runtime.eval(
            "(function() local n = {} for k in pairs(os) do n[#n + 1] = k end"
            " table.sort(n) return table.concat(n, ',') end)()"
        )
        assert keys == "clock,date,time"

    def test_python_global_is_nil(self) -> None:
        # The harness docstring says Python is not reachable from Lua.
        # DEFECT (reported to the caller): lupa still installs a ``python`` helper table
        # when register_eval/register_builtins are False, so this fails today.
        runtime = h.new_runtime()
        assert runtime.eval("python == nil") is True

    def test_python_has_no_eval_or_builtins(self) -> None:
        # What actually matters for reachability: no way to evaluate Python or reach
        # builtins from Lua, whatever else lupa leaves in the ``python`` table.
        runtime = h.new_runtime()
        assert runtime.eval("python == nil or (python.eval == nil and python.builtins == nil)")

    def test_loading_the_mod_creates_nothing_in_the_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runtime = h.load_mod()
        h.export_state(runtime)
        h.execute(runtime, {"action_type": "end_turn"})
        assert list(tmp_path.iterdir()) == []

    def test_negative_control_without_lockdown_the_mod_writes_to_the_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Proves the cwd test above can fail: skip the lockdown and the mod creates
        # the ``.\AlphaZeroAI`` directory (a literal name on Linux) and its log files.
        lua51 = h.import_lua51()
        monkeypatch.chdir(tmp_path)
        # Unlocked, the mod reads this variable and then clears and rewrites the
        # bridge files in the directory it names. Never let it see a real one.
        monkeypatch.delenv("ALPHAZERO_CIV6_IPC_DIR", raising=False)
        unlocked = lua51.LuaRuntime(unpack_returned_tuples=True)
        unlocked.execute(h.STUB.read_text(encoding="utf-8"))
        unlocked.execute(h.BUNDLE.read_text(encoding="utf-8"))
        assert list(tmp_path.iterdir()) != []


class TestBundleLoads:
    GLOBALS = (
        "AlphaZeroStateExporter",
        "AlphaZeroActionExecutor",
        "AlphaZeroBridge",
        "AlphaZeroJson",
        "AlphaZeroConfig",
    )

    @pytest.mark.parametrize("stub", [True, False])
    def test_bundle_defines_its_globals(self, stub: bool) -> None:
        runtime = h.load_mod(stub=stub)
        for name in self.GLOBALS:
            assert runtime.eval(f"type({name})") == "table", f"{name} missing (stub={stub})"

    def test_runtime_is_lua_51(self) -> None:
        assert h.new_runtime().eval("_VERSION") == "Lua 5.1"

    def test_stub_changes_the_world(self) -> None:
        # Non-vacuity for the stub=True/False parametrisation above.
        assert h.load_mod(stub=True).eval("Civ6Stub ~= nil") is True
        assert h.load_mod(stub=False).eval("Civ6Stub == nil") is True


class TestExport:
    def test_map(self, state: dict[str, Any]) -> None:
        check_map(state)

    def test_units(self, state: dict[str, Any]) -> None:
        check_units(state)
        assert all(u["owner"] == 0 for u in state["units"])  # enemy 201 not exported

    def test_city(self, state: dict[str, Any]) -> None:
        check_city(state)

    def test_player(self, state: dict[str, Any]) -> None:
        check_player(state)
        assert state["player"]["id"] == 0
        assert state["player_id"] == 0

    def test_turn_and_era(self, state: dict[str, Any]) -> None:
        check_turn(state)
        check_era(state)

    def test_mountain_tile(self, state: dict[str, Any]) -> None:
        tile = _tile(state, 3, 2)
        assert tile["terrain"] == "TERRAIN_GRASS_MOUNTAIN"
        assert "FEATURE_MOUNTAIN" in tile["features"]

    def test_ordinary_tile_has_no_special_fields(self, state: dict[str, Any]) -> None:
        tile = _tile(state, 0, 0)
        assert tile["owner_id"] == -1
        assert tile["features"] == []
        assert "resource" not in tile and "improvement" not in tile


def _nil_getter(name: str) -> str:
    """Lua setup: player 0's ``name`` getter returns nil (the method still exists)."""
    return f"Players[0].{name} = function() return nil end"


#: Lua setup: player 0's units lose ``GetDamage``. The exporter reads a unit's
#: stats under one ``pcall``, so they all fall back together.
_UNITS_WITHOUT_GET_DAMAGE = (
    "local units = Players[0].GetUnits\n"
    "Players[0].GetUnits = function(self) local c = units(self)"
    " for _, u in c:Members() do u.GetDamage = nil end return c end"
)


class TestNonVacuity:
    """Negative controls: break one stub method, and the matching check must fail."""

    @pytest.mark.parametrize(
        ("component", "check", "setup"),
        [
            ("map grid size", check_map, "Map.GetGridSize = nil"),
            ("map plots", check_map, "Map.GetPlot = function() return nil end"),
            ("tile features", check_map, "GameInfo.Features = nil"),
            ("tile resource", check_map, "GameInfo.Resources = nil"),
            ("tile improvement", check_map, "GameInfo.Improvements = nil"),
            ("tile owner", check_map, "Map.GetPlot(1, 1).GetOwner = function() return -1 end"),
            ("units", check_units, "Players[0].GetUnits = nil"),
            ("unit type names", check_units, "GameInfo.Units = nil"),
            ("unit stats", check_units, _UNITS_WITHOUT_GET_DAMAGE),
            ("cities", check_city, "Players[0].GetCities = nil"),
            # A getter that returns nil costs only its own fields, so each of
            # these fails on the check for that field and nothing else.
            ("player treasury", check_player_gold, _nil_getter("GetTreasury")),
            ("player techs", check_player_science, _nil_getter("GetTechs")),
            ("player culture", check_player_civic, _nil_getter("GetCulture")),
            ("player civilization", check_player, "PlayerConfigurations = nil"),
            ("turn", check_turn, "Game.GetCurrentGameTurn = function() return 0 end"),
            ("era", check_era, "Game.GetEras = nil"),
            ("legal actions", check_legal_actions, "Players[0].GetUnits = nil"),
        ],
    )
    def test_a_hole_in_the_stub_fails_the_check(
        self, component: str, check: Any, setup: str
    ) -> None:
        # The healthy export passes the check ...
        check(h.export_state(h.load_mod()))
        # ... and the broken one does not, so a silent pcall fallback cannot read as a pass.
        broken = h.export_state(h.load_mod(setup=setup))
        with pytest.raises(AssertionError):
            check(broken)

    @pytest.mark.parametrize(
        ("getter", "lost", "kept"),
        [
            ("GetTreasury", check_player_gold, (check_player_science, check_player_civic)),
            ("GetTechs", check_player_science, (check_player_gold, check_player_civic)),
            ("GetCulture", check_player_civic, (check_player_gold, check_player_science)),
        ],
    )
    def test_a_nil_getter_costs_only_its_own_fields(
        self, getter: str, lost: Any, kept: tuple[Any, ...]
    ) -> None:
        state = h.export_state(h.load_mod(setup=_nil_getter(getter)))
        with pytest.raises(AssertionError):
            lost(state)
        for check in kept:
            check(state)

    @pytest.mark.parametrize("getter", ["GetTreasury", "GetTechs", "GetCulture"])
    def test_a_missing_getter_empties_the_whole_player_block(self, getter: str) -> None:
        # today's behaviour: the player block is exported under one pcall, so a
        # method that does not exist costs every field, not only its own.
        state = h.export_state(h.load_mod(setup=f"Players[0].{getter} = nil"))
        assert state["player"] == {"id": 0}

    def test_a_missing_unit_getter_zeroes_the_units_stats_silently(self) -> None:
        # today's behaviour: identity and position survive, the stats fall back.
        state = h.export_state(h.load_mod(setup=_UNITS_WITHOUT_GET_DAMAGE))
        warrior = next(u for u in state["units"] if u["id"] == 102)
        assert (warrior["type_name"], warrior["x"], warrior["y"]) == ("UNIT_WARRIOR", 1, 0)
        assert (warrior["movement"], warrior["combat_strength"], warrior["can_attack"]) == (
            0,
            0,
            False,
        )

    def test_a_missing_city_production_hook_empties_the_produce_action(self) -> None:
        broken = h.export_state(h.load_mod(setup="Players[0].GetCities = nil"))
        assert _actions_for(broken, "produce") == []


class TestKnownGaps:
    """Pinned as today's behaviour. GM-B2 flips these."""

    def test_techs_researched_is_empty(self, state: dict[str, Any]) -> None:
        # today's behaviour; GM-B2 changes this
        # The stub player HAS TECH_POTTERY (HasTech(0)); the export still says nothing.
        runtime = h.load_mod()
        assert runtime.eval("Players[0]:GetTechs():HasTech(0)") is True
        assert state["player"]["techs_researched"] == []

    def test_current_research_is_a_raw_index(self, state: dict[str, Any]) -> None:
        # today's behaviour; GM-B2 changes this
        # The stub is researching TECH_MINING (index 1); the export carries the index.
        assert state["player"]["current_research"] == 1
        assert not isinstance(state["player"]["current_research"], str)

    def test_current_civic_is_a_name_unlike_research(self, state: dict[str, Any]) -> None:
        # today's behaviour; GM-B2 changes this (contrast: civics are named, techs are not)
        assert state["player"]["current_civic"] == "CIVIC_CRAFTSMANSHIP"

    def test_city_buildings_and_districts_are_empty(self, state: dict[str, Any]) -> None:
        # today's behaviour; GM-B2 changes this
        city = state["cities"][0]
        assert city["buildings"] == []
        assert city["districts"] == []

    def test_no_legal_action_offers_research(self, state: dict[str, Any]) -> None:
        # today's behaviour; GM-B2 changes this
        # The stub says CanResearch(1) and CanResearch(2), so an offer was possible.
        runtime = h.load_mod()
        assert runtime.eval("Players[0]:GetTechs():CanResearch(1)") is True
        assert set(_action_names(state)) == {
            "end_turn",
            "move",
            "skip",
            "found_city",
            "fortify",
            "attack",
            "produce",
        }
        assert not any("research" in json.dumps(a).lower() for a in state["legal_actions"])

    def test_state_has_no_schema_version(self, state: dict[str, Any]) -> None:
        # today's behaviour; GM-B2 changes this
        assert "schema_version" not in state
        assert "schema_version" not in state["player"]

    def test_is_game_over_is_false_and_winner_is_the_sentinel(self, state: dict[str, Any]) -> None:
        # today's behaviour; GM-B2 changes this
        assert state["is_game_over"] is False
        assert state["winner_id"] == -1

    def test_player_block_lists_only_the_local_player(self, state: dict[str, Any]) -> None:
        # today's behaviour; GM-B2 changes this (player 1 exists in the stub)
        assert [p["id"] for p in state["players"]] == [0]

    def test_move_targets_include_occupied_tiles(self, state: dict[str, Any]) -> None:
        # today's behaviour; GM-B2 changes this
        # Warrior 102 may "move" onto (2,0), where enemy 201 stands (that is an attack, not
        # a move), and onto (0,0), where friendly settler 101 stands. Either would be
        # rejected by the game's own movement rules; the exporter offers them anyway.
        (move,) = _actions_for(state, "move", unit_id=102)
        targets = {tuple(t) for t in move["targets"]}
        assert (2, 0) in targets
        assert (0, 0) in targets


class TestLegalActions:
    def test_end_turn_is_first(self, state: dict[str, Any]) -> None:
        assert state["legal_actions"][0] == {"action": "end_turn"}
        assert _action_names(state).count("end_turn") == 1

    def test_settler_actions(self, state: dict[str, Any]) -> None:
        names = sorted(a["action"] for a in state["legal_actions"] if a.get("unit_id") == 101)
        assert names == ["found_city", "move", "skip"]
        assert not _actions_for(state, "fortify", unit_id=101)
        assert not _actions_for(state, "attack", unit_id=101)

    def test_warrior_actions(self, state: dict[str, Any]) -> None:
        names = sorted(a["action"] for a in state["legal_actions"] if a.get("unit_id") == 102)
        assert names == ["attack", "fortify", "move", "skip"]
        assert not _actions_for(state, "found_city", unit_id=102)
        (attack,) = _actions_for(state, "attack", unit_id=102)
        assert [tuple(t) for t in attack["targets"]] == [(2, 0)]

    def test_enemy_units_have_no_actions(self, state: dict[str, Any]) -> None:
        assert not [a for a in state["legal_actions"] if a.get("unit_id") == 201]

    def test_move_targets_are_adjacent_and_in_bounds(self, state: dict[str, Any]) -> None:
        (move,) = _actions_for(state, "move", unit_id=101)
        targets = {tuple(t) for t in move["targets"]}
        assert targets == {(1, 0), (0, 1), (1, 1)}

    def test_no_move_target_is_the_mountain(self, state: dict[str, Any]) -> None:
        for action in _actions_for(state, "move"):
            assert (3, 2) not in {tuple(t) for t in action["targets"]}

    def test_impassable_tiles_are_filtered_from_move_targets(self) -> None:
        # Non-vacuity for the mountain test: no stub unit is adjacent to (3,2), so that
        # assertion holds trivially. Make a tile next to settler 101 impassable and show
        # the exporter drops exactly that target.
        setup = "Map.GetPlot(1, 1).IsImpassable = function() return true end"
        state = h.export_state(h.load_mod(setup=setup))
        (move,) = _actions_for(state, "move", unit_id=101)
        assert {tuple(t) for t in move["targets"]} == {(1, 0), (0, 1)}

    def test_produce_lists_what_the_stub_allows(self, state: dict[str, Any]) -> None:
        (produce,) = _actions_for(state, "produce", city_id=301)
        assert produce["items"] == ["UNIT_WARRIOR", "BUILDING_GRANARY", "DISTRICT_CAMPUS"]
        assert produce["item_types"] == {
            "UNIT_WARRIOR": "unit",
            "BUILDING_GRANARY": "building",
            "DISTRICT_CAMPUS": "district",
        }

    def test_produce_follows_can_produce(self) -> None:
        # CanProduce is the only source: allow nothing and the items disappear.
        setup = (
            "for _, c in Players[0]:GetCities():Members() do"
            " c:GetBuildQueue().CanProduce = function() return false end end"
        )
        state = h.export_state(h.load_mod(setup=setup))
        assert [c["id"] for c in state["cities"]] == [301], "city vanished; control is vacuous"
        (produce,) = _actions_for(state, "produce", city_id=301)
        assert produce["items"] == []
        # today's behaviour; GM-B2 changes this
        # The action is still offered, with nothing to produce.


class TestExecute:
    def test_end_turn_requests_one_action(self, runtime: Any) -> None:
        assert h.stub_calls(runtime) == []
        result = h.execute(runtime, {"action_type": "end_turn"})
        assert result == (True, None, "Turn ended")
        assert h.stub_calls(runtime) == ["UI.RequestAction"]
        # The request is the game's end-turn action (stub ACTION_ENDTURN = 7).
        assert runtime.eval("Civ6Stub.calls[1].args[1]") == 7

    def test_end_turn_says_turn_ended_whatever_the_game_did(self, runtime: Any) -> None:
        # today's behaviour; GM-B2 changes this
        # The stub's UI.RequestAction does nothing and cannot say whether the turn really
        # ended (a real game may refuse while a research or production choice is
        # pending); the mod answers "Turn ended" unconditionally.
        assert h.execute(runtime, {"action_type": "end_turn"})[2] == "Turn ended"

    def test_a_raising_ui_call_becomes_execution_failed(self, runtime: Any) -> None:
        runtime.execute("UI.RequestAction = function() error('blocked') end")
        success, error_code, message = h.execute(runtime, {"action_type": "end_turn"})
        assert (success, error_code) == (False, "EXECUTION_FAILED")
        assert "blocked" in message

    def test_unknown_action_type(self, runtime: Any) -> None:
        success, error_code, _ = h.execute(runtime, {"action_type": "bogus"})
        assert (success, error_code) == (False, "INVALID_COMMAND")
        assert h.stub_calls(runtime) == []

    def test_missing_action_type(self, runtime: Any) -> None:
        success, error_code, _ = h.execute(runtime, {})
        assert (success, error_code) == (False, "INVALID_COMMAND")
        assert h.stub_calls(runtime) == []

    def test_end_turn_without_ui(self) -> None:
        runtime = h.load_mod(setup="UI = nil")
        result = h.execute(runtime, {"action_type": "end_turn"})
        assert result == (True, None, "End turn requested")


class TestWireCompatibility:
    def test_python_reader_accepts_the_export(
        self, state: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            parsed = GameStateData.from_dict(state)
        # The reader skips what it cannot parse with a warning; the counts below are the
        # non-vacuity guard, the warning check is the second.
        assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
        assert sorted(u.id for u in parsed.units) == [101, 102]
        assert [c.id for c in parsed.cities] == [301]
        assert len(parsed.map_data.tiles) == 12
        assert len(parsed.legal_actions) == len(state["legal_actions"])
        assert parsed.turn_id == 12
        assert parsed.era == 1
        assert parsed.is_game_over is False
        assert parsed.winner_id is None

    def test_parsed_fields_match_the_export(self, state: dict[str, Any]) -> None:
        parsed = GameStateData.from_dict(state)
        city = parsed.cities[0]
        assert (city.name, city.population, city.owner_id) == ("Stubville", 3, 0)
        settler = next(u for u in parsed.units if u.id == 101)
        assert (settler.x, settler.y) == (0, 0)
        types = [a.action_type for a in parsed.legal_actions]
        assert types[0] == "end_turn"
        assert "produce" in types and "attack" in types
        attack = next(a for a in parsed.legal_actions if a.action_type == "attack")
        assert attack.targets == [(2, 0)]

    def test_state_message_accepts_a_wrapped_export(self, state: dict[str, Any]) -> None:
        message = StateMessage.from_json(
            json.dumps({"msg_id": 7, "timestamp": 1.0, "state": state})
        )
        assert sorted(u.id for u in message.state.units) == [101, 102]

    def test_reader_is_not_vacuous(self, state: dict[str, Any]) -> None:
        # Break the export (drop the units) and the unit-id check above must notice.
        broken = dict(state, units=[])
        assert GameStateData.from_dict(broken).units == []


class TestSkipFailSwitch:
    def test_missing_lupa_skips_when_not_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "lupa", None)
        monkeypatch.delenv(h.REQUIRE_ENV, raising=False)
        with pytest.raises(pytest.skip.Exception):
            h.import_lua51()

    def test_missing_lupa_fails_when_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "lupa", None)
        monkeypatch.setenv(h.REQUIRE_ENV, "1")
        with pytest.raises(pytest.fail.Exception):
            h.import_lua51()

    @pytest.mark.parametrize("value", ["", "0", "true", "yes"])
    def test_only_the_literal_one_requires_lupa(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(h.REQUIRE_ENV, value)
        assert h.lua_required() is False
        monkeypatch.setenv(h.REQUIRE_ENV, "1")
        assert h.lua_required() is True


def _installed_extras(job: dict[str, Any]) -> set[str]:
    """Extras named by the job's ``pip install -e ".[...]"`` command lines.

    Shell comments are dropped first: a comment that mentions the extra is not
    an install.
    """
    extras: set[str] = set()
    for step in job["steps"]:
        for line in str(step.get("run", "")).splitlines():
            command = line.split("#", 1)[0]
            if "pip install -e" not in command:
                continue
            for group in re.findall(r"\[([^\]]*)\]", command):
                extras.update(name.strip() for name in group.split(",") if name.strip())
    return extras


class TestWiring:
    @pytest.mark.parametrize(
        ("run", "expected"),
        [
            ('pip install -e ".[dev,lua]"', {"dev", "lua"}),
            ("pip install -e '.[dev]'  # lua is installed elsewhere", {"dev"}),
            ('# pip install -e ".[lua]"\npip install -e ".[dev]"', {"dev"}),
            ("pip install lupa", set()),
        ],
    )
    def test_extras_are_read_from_commands_not_comments(self, run: str, expected: set[str]) -> None:
        assert _installed_extras({"steps": [{"run": run}]}) == expected

    def test_ci_test_job_requires_lua_and_installs_the_extra(self) -> None:
        workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text("utf-8"))
        job = workflow["jobs"]["test"]
        assert job["env"]["CIV6AZ_REQUIRE_LUA"] == "1"
        assert _installed_extras(job), "the test job has no editable install"
        assert "lua" in _installed_extras(job), "the test job does not install the lua extra"
        assert any("tests/unit/" in s.get("run", "") for s in job["steps"])

    def test_pyproject_defines_lua_and_dev_does_not_contain_lupa(self) -> None:
        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
        extras = data["project"]["optional-dependencies"]
        assert any(dep.lower().startswith("lupa") for dep in extras["lua"])
        assert not any("lupa" in dep.lower() for dep in extras["dev"])
        assert not any("alpha_zero[lua]" in dep for dep in extras["dev"])


class TestDeterminism:
    def test_two_fresh_runtimes_export_identical_json(self) -> None:
        first = h.export_state(h.load_mod())
        second = h.export_state(h.load_mod())
        check_map(first)
        check_units(first)
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
