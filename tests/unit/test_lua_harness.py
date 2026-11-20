"""GM-B1: characterise the AlphaZeroAI mod's Lua by executing it (lupa, Lua 5.1).

This slice pinned what the Lua did, known gaps included. GM-B2 (exporter parity) changed
the research, blocker, building and end-turn gaps; their tests moved to
``test_lua_exporter_parity.py``. The gaps GM-B2 does not touch stay pinned here with the
comment ``today's behaviour; not changed by GM-B2``.

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


def check_city(state: dict[str, Any]) -> None:
    assert [c["id"] for c in state["cities"]] == [301]
    city = state["cities"][0]
    assert city["name"] == "Stubville"
    assert city["population"] == 3
    assert (city["x"], city["y"]) == (1, 1)


def check_player(state: dict[str, Any]) -> None:
    player = state["player"]
    assert player.get("gold") == 50
    assert player.get("science_per_turn") == 4
    assert player.get("civilization") == "CIVILIZATION_STUB_A"
    assert player.get("current_civic") == "CIVIC_CRAFTSMANSHIP"


def check_turn(state: dict[str, Any]) -> None:
    assert state["turn_id"] == 12


def check_era(state: dict[str, Any]) -> None:
    assert state["era"] == 1


def check_legal_actions(state: dict[str, Any]) -> None:
    assert _actions_for(state, "found_city", unit_id=101)
    assert _actions_for(state, "attack", unit_id=102)
    assert _actions_for(state, "produce", city_id=301)


class TestSandbox:
    def test_forbidden_globals_are_nil(self) -> None:
        runtime = h.new_runtime()
        for name in ("io", "dofile", "loadfile", "require", "package", "debug"):
            assert runtime.eval(f"{name} == nil") is True, f"{name} is reachable from Lua"

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
            ("cities", check_city, "Players[0].GetCities = nil"),
            ("player treasury", check_player, "Players[0].GetTreasury = nil"),
            ("player techs", check_player, "Players[0].GetTechs = nil"),
            ("player culture", check_player, "Players[0].GetCulture = nil"),
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

    def test_a_missing_city_production_hook_empties_the_produce_action(self) -> None:
        broken = h.export_state(h.load_mod(setup="Players[0].GetCities = nil"))
        assert _actions_for(broken, "produce") == []


class TestKnownGaps:
    """Pinned as today's behaviour; GM-B2 does not change these (the ones it did change
    now live in ``test_lua_exporter_parity.py``)."""

    def test_is_game_over_is_false_and_winner_is_the_sentinel(self, state: dict[str, Any]) -> None:
        # today's behaviour; not changed by GM-B2
        assert state["is_game_over"] is False
        assert state["winner_id"] == -1

    def test_player_block_lists_only_the_local_player(self, state: dict[str, Any]) -> None:
        # today's behaviour; not changed by GM-B2 (player 1 exists in the stub)
        assert [p["id"] for p in state["players"]] == [0]

    def test_move_targets_include_occupied_tiles(self, state: dict[str, Any]) -> None:
        # today's behaviour; not changed by GM-B2
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
        # today's behaviour; not changed by GM-B2
        # The action is still offered, with nothing to produce.


class TestExecute:
    def test_end_turn_requests_one_action(self, runtime: Any) -> None:
        assert h.stub_calls(runtime) == []
        result = h.execute(runtime, {"action_type": "end_turn"})
        assert result == (True, None, "End turn requested")
        assert h.stub_calls(runtime) == ["UI.RequestAction"]
        # The request is the game's end-turn action (stub ACTION_ENDTURN = 7).
        assert runtime.eval("Civ6Stub.calls[1].args[1]") == 7

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

    def test_end_turn_without_ui_is_a_failure(self) -> None:
        # GM-B2: nothing was requested, so it no longer answers success.
        runtime = h.load_mod(setup="UI = nil")
        success, error_code, message = h.execute(runtime, {"action_type": "end_turn"})
        assert (success, error_code) == (False, "EXECUTION_FAILED")
        assert message.startswith("End turn unavailable")


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


class TestWiring:
    def test_ci_test_job_requires_lua_and_installs_the_extra(self) -> None:
        workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/ci.yml").read_text("utf-8"))
        job = workflow["jobs"]["test"]
        assert job["env"]["CIV6AZ_REQUIRE_LUA"] == "1"
        installs = [
            s["run"]
            for s in job["steps"]
            if "run" in s and "pip install -e" in s["run"] and "lua" in s["run"]
        ]
        assert installs, "the test job does not install the lua extra"
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
