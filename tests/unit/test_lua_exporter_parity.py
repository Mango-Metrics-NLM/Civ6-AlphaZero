"""GM-B2: exporter parity, executed against the invented game (lupa, Lua 5.1).

The exporter wraps every component in ``pcall`` and falls back to an empty value, so
each check below asserts a positive value from the stub, never just the absence of an
error. The stub's changeable state (``Civ6Stub.players``, ``.cities``, ``.notifications``,
``.canEndTurn``) is changed after the mod has loaded where a test needs another game.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import lua_harness as h

RESEARCH_BLOCKER = (
    "{ id = 5, blocking = EndTurnBlockingTypes.ENDTURN_BLOCKING_RESEARCH,"
    " message = 'Choose a technology', typeName = 'NOTIFICATION_CHOOSE_TECH', dismissed = false }"
)
CIVIC_BLOCKER = (
    "{ id = 6, blocking = EndTurnBlockingTypes.ENDTURN_BLOCKING_CIVIC,"
    " message = 'Choose a civic', typeName = 'NOTIFICATION_CHOOSE_CIVIC', dismissed = false }"
)
PRODUCTION_BLOCKER = (
    "{ id = 7, blocking = EndTurnBlockingTypes.ENDTURN_BLOCKING_PRODUCTION,"
    " message = 'Choose production', typeName = 'NOTIFICATION_PRODUCTION', dismissed = false }"
)


def _notify(runtime: Any, *entries: str) -> None:
    runtime.execute(f"Civ6Stub.notifications[0] = {{ {', '.join(entries)} }}")


def _without_methods(owner: str, getter: str, *methods: str) -> str:
    """Lua setup: ``Players[0]:<getter>()`` returns objects lacking ``methods``."""
    drops = " ".join(f"t.{m} = nil" for m in methods)
    return (
        f"local orig = Players[0].{getter}\n"
        f"Players[0].{getter} = function(self) local t = orig(self) {drops} return t end"
    ) + (f"  -- {owner}" if owner else "")


def _actions(state: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [a for a in state["legal_actions"] if a["action"] == name]


def _call_args(runtime: Any, index: int, count: int) -> list[Any]:
    return [runtime.eval(f"Civ6Stub.calls[{index}].args[{n}]") for n in range(1, count + 1)]


@pytest.fixture
def runtime() -> Any:
    return h.load_mod()


@pytest.fixture
def state(runtime: Any) -> dict[str, Any]:
    return h.export_state(runtime)


# ---------------------------------------------------------------------------
# Exporter: top level
# ---------------------------------------------------------------------------
QUERY_OFF = "NotificationManager.GetFirstEndTurnBlocking = nil"


class TestSchemaAndBlocker:
    # Every test here runs twice: on the game's own query (GetFirstEndTurnBlocking) and
    # on the notification-list scan used when that query is missing.
    @pytest.fixture(params=["query", "scan"])
    def runtime(self, request: pytest.FixtureRequest) -> Any:
        return h.load_mod(setup=None if request.param == "query" else QUERY_OFF)

    def test_schema_version_is_two(self, state: dict[str, Any]) -> None:
        assert state["schema_version"] == 2
        assert type(state["schema_version"]) is int

    def test_nothing_blocks_by_default(self, state: dict[str, Any]) -> None:
        assert state["schema_version"] == 2, "not a schema 2 export; the absence below is vacuous"
        assert "end_turn_blocking" not in state
        assert "end_turn_blocking_message" not in state

    @pytest.mark.parametrize(
        ("entry", "name", "message"),
        [
            (RESEARCH_BLOCKER, "ENDTURN_BLOCKING_RESEARCH", "Choose a technology"),
            (CIVIC_BLOCKER, "ENDTURN_BLOCKING_CIVIC", "Choose a civic"),
            (PRODUCTION_BLOCKER, "ENDTURN_BLOCKING_PRODUCTION", "Choose production"),
        ],
    )
    def test_blocker_name_and_message(
        self, runtime: Any, entry: str, name: str, message: str
    ) -> None:
        _notify(runtime, entry)
        state = h.export_state(runtime)
        assert state["end_turn_blocking"] == name
        assert state["end_turn_blocking_message"] == message

    def test_first_live_blocker_in_list_order_wins(self, runtime: Any) -> None:
        dismissed = RESEARCH_BLOCKER.replace("id = 5", "id = 1").replace(
            "dismissed = false", "dismissed = true"
        )
        informational = "{ id = 2, blocking = 0, message = 'FYI', dismissed = false }"
        _notify(runtime, dismissed, informational, PRODUCTION_BLOCKER, CIVIC_BLOCKER)
        state = h.export_state(runtime)
        assert state["end_turn_blocking"] == "ENDTURN_BLOCKING_PRODUCTION"
        assert state["end_turn_blocking_message"] == "Choose production"

    def test_dismissed_and_non_blocking_notifications_do_not_block(self, runtime: Any) -> None:
        dismissed = RESEARCH_BLOCKER.replace("dismissed = false", "dismissed = true")
        informational = "{ id = 2, blocking = 0, message = 'FYI', dismissed = false }"
        _notify(runtime, dismissed, informational)
        state = h.export_state(runtime)
        assert state["schema_version"] == 2, "not a schema 2 export; the absence below is vacuous"
        assert "end_turn_blocking" not in state
        assert "end_turn_blocking_message" not in state

    def test_unknown_blocking_value_gets_a_synthetic_name(self, runtime: Any) -> None:
        _notify(runtime, "{ id = 9, blocking = 99, message = 'odd', dismissed = false }")
        assert h.export_state(runtime)["end_turn_blocking"] == "ENDTURN_BLOCKING_UNKNOWN_99"

    def test_blocker_without_message_has_no_message_key(self, runtime: Any) -> None:
        _notify(runtime, "{ id = 9, blocking = 3, dismissed = false }")
        state = h.export_state(runtime)
        assert state["end_turn_blocking"] == "ENDTURN_BLOCKING_RESEARCH"
        assert "end_turn_blocking_message" not in state

    def test_a_raising_get_message_does_not_hide_the_blocker(self, runtime: Any) -> None:
        _notify(runtime, RESEARCH_BLOCKER)
        runtime.execute(
            "local find = NotificationManager.Find\n"
            "NotificationManager.Find = function(p, i) local e = find(p, i)"
            " e.GetMessage = function() error('no message') end return e end"
        )
        state = h.export_state(runtime)
        assert state["end_turn_blocking"] == "ENDTURN_BLOCKING_RESEARCH"
        assert "end_turn_blocking_message" not in state

    def test_missing_notification_manager_means_the_blocker_is_not_askable(self) -> None:
        # GM-B9: the mod cannot ask what blocks, so it says so (was: "no blocker").
        runtime = h.load_mod(setup="NotificationManager = nil")
        _notify_ok = runtime.eval("Civ6Stub.notifications ~= nil")
        state = h.export_state(runtime)
        assert _notify_ok is True
        assert state["schema_version"] == 2
        assert state["end_turn_blocking"] == "ENDTURN_BLOCKING_NOT_ASKABLE"


# ---------------------------------------------------------------------------
# Exporter: player block
# ---------------------------------------------------------------------------
class TestResearchExport:
    def test_defaults(self, state: dict[str, Any]) -> None:
        player = state["player"]
        assert player["techs_researched"] == ["TECH_POTTERY"]
        assert player["current_research"] == "TECH_MINING"
        assert player["research_status"] == {
            "type": "TECH_MINING",
            "index": 1,
            "cost": 50,
            "progress": 20,
            "turns_left": 8,
            "boost_triggered": True,
        }
        # The rest of the block is unchanged.
        assert player["gold"] == 50
        assert player["science_per_turn"] == 4

    def test_techs_researched_follow_game_info_order(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.players[0].tech.has = { [2] = true, [0] = true }")
        assert h.export_state(runtime)["player"]["techs_researched"] == [
            "TECH_POTTERY",
            "TECH_WRITING",
        ]

    def test_no_current_research_drops_name_and_status(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.players[0].tech.current = -1")
        player = h.export_state(runtime)["player"]
        assert "current_research" not in player
        assert "research_status" not in player
        assert player["techs_researched"] == ["TECH_POTTERY"]

    def test_current_research_follows_the_game(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.players[0].tech.current = 2")
        player = h.export_state(runtime)["player"]
        assert player["current_research"] == "TECH_WRITING"
        assert player["research_status"]["index"] == 2
        assert player["research_status"]["cost"] == 60
        assert player["research_status"]["turns_left"] == 15
        assert player["research_status"]["boost_triggered"] is False

    @pytest.mark.parametrize(
        ("method", "field"),
        [
            ("GetResearchCost", "cost"),
            ("GetResearchProgress", "progress"),
            ("GetTurnsToResearch", "turns_left"),
            ("HasBoostBeenTriggered", "boost_triggered"),
        ],
    )
    def test_a_missing_method_drops_only_its_field(self, method: str, field: str) -> None:
        runtime = h.load_mod(setup=_without_methods("", "GetTechs", method))
        player = h.export_state(runtime)["player"]
        status = player["research_status"]
        assert field not in status
        assert set(status) == {
            "type",
            "index",
            "cost",
            "progress",
            "turns_left",
            "boost_triggered",
        } - {field}
        assert status["type"] == "TECH_MINING"
        assert player["current_research"] == "TECH_MINING"
        assert player["gold"] == 50


class TestCivicExport:
    def test_defaults(self, state: dict[str, Any]) -> None:
        player = state["player"]
        assert player["civics_researched"] == ["CIVIC_CODE_OF_LAWS"]
        assert player["current_civic"] == "CIVIC_CRAFTSMANSHIP"
        assert player["civic_status"] == {
            "type": "CIVIC_CRAFTSMANSHIP",
            "index": 1,
            "cost": 40,
            "progress": 10,
            "turns_left": 15,
            "boost_triggered": False,
        }

    def test_civics_researched_follow_game_info_order(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.players[0].civic.has = { [2] = true, [0] = true }")
        assert h.export_state(runtime)["player"]["civics_researched"] == [
            "CIVIC_CODE_OF_LAWS",
            "CIVIC_FOREIGN_TRADE",
        ]

    def test_has_civic_is_the_source_not_the_old_list(self) -> None:
        # The old path (GetResearchedCivics) now lies; HasCivic must win.
        setup = _without_methods("", "GetCulture").replace(
            "return t end", "t.GetResearchedCivics = function() return {} end return t end"
        )
        player = h.export_state(h.load_mod(setup=setup))["player"]
        assert player["civics_researched"] == ["CIVIC_CODE_OF_LAWS"]

    def test_without_has_civic_the_old_path_gives_the_same_result(self) -> None:
        runtime = h.load_mod(setup=_without_methods("", "GetCulture", "HasCivic"))
        runtime.execute("Civ6Stub.players[0].civic.has = { [2] = true, [0] = true }")
        player = h.export_state(runtime)["player"]
        assert player["civics_researched"] == ["CIVIC_CODE_OF_LAWS", "CIVIC_FOREIGN_TRADE"]

    def test_turns_left_comes_from_the_per_civic_query(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.players[0].civic.turns[1] = 22")  # turnsLeft stays 15
        assert h.export_state(runtime)["player"]["civic_status"]["turns_left"] == 22

    def test_turns_left_falls_back_to_the_current_civic_query(self) -> None:
        runtime = h.load_mod(setup=_without_methods("", "GetCulture", "GetTurnsToProgressCivic"))
        runtime.execute("Civ6Stub.players[0].civic.turns[1] = 22")
        assert h.export_state(runtime)["player"]["civic_status"]["turns_left"] == 15

    def test_no_current_civic_drops_name_and_status(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.players[0].civic.current = -1")
        player = h.export_state(runtime)["player"]
        assert player["current_research"] == "TECH_MINING", "control: the rest is exported"
        assert "current_civic" not in player
        assert "civic_status" not in player
        assert player["civics_researched"] == ["CIVIC_CODE_OF_LAWS"]

    @pytest.mark.parametrize(
        ("methods", "field"),
        [
            (("GetCultureCost",), "cost"),
            (("GetCulturalProgress",), "progress"),
            (("GetTurnsToProgressCivic", "GetTurnsLeftOnCurrentCivic"), "turns_left"),
            (("HasBoostBeenTriggered",), "boost_triggered"),
        ],
    )
    def test_a_missing_method_drops_only_its_field(
        self, methods: tuple[str, ...], field: str
    ) -> None:
        runtime = h.load_mod(setup=_without_methods("", "GetCulture", *methods))
        player = h.export_state(runtime)["player"]
        status = player["civic_status"]
        assert field not in status
        assert set(status) == {
            "type",
            "index",
            "cost",
            "progress",
            "turns_left",
            "boost_triggered",
        } - {field}
        assert status["type"] == "CIVIC_CRAFTSMANSHIP"
        assert player["current_civic"] == "CIVIC_CRAFTSMANSHIP"


# ---------------------------------------------------------------------------
# Exporter: cities
# ---------------------------------------------------------------------------
class TestCityExport:
    def test_defaults(self, state: dict[str, Any]) -> None:
        city = state["cities"][0]
        assert city["buildings"] == ["BUILDING_MONUMENT"]
        assert city["districts"] == ["DISTRICT_CITY_CENTER", "DISTRICT_CAMPUS"]

    def test_buildings_follow_game_info_order(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.cities[301].buildings = { [1] = true, [0] = true }")
        assert h.export_state(runtime)["cities"][0]["buildings"] == [
            "BUILDING_MONUMENT",
            "BUILDING_GRANARY",
        ]

    def test_districts_keep_the_games_listing_order(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.cities[301].districts = { 1, 0 }")
        assert h.export_state(runtime)["cities"][0]["districts"] == [
            "DISTRICT_CAMPUS",
            "DISTRICT_CITY_CENTER",
        ]

    def test_an_unfinished_district_is_not_listed(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.cities[301].districts = { 0, { type = 1, complete = false } }")
        assert h.export_state(runtime)["cities"][0]["districts"] == ["DISTRICT_CITY_CENTER"]

    def test_a_finished_district_entry_is_listed(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.cities[301].districts = { 0, { type = 1, complete = true } }")
        assert h.export_state(runtime)["cities"][0]["districts"] == [
            "DISTRICT_CITY_CENTER",
            "DISTRICT_CAMPUS",
        ]

    def test_a_district_without_is_complete_is_listed(self) -> None:
        setup = (
            "for _, c in Players[0]:GetCities():Members() do\n"
            " local get = c.GetDistricts\n"
            " c.GetDistricts = function(self) local col = get(self) local members = col.Members\n"
            "  col.Members = function(me) local it = members(me)\n"
            "   return function() local i, d = it() if i then d.IsComplete = nil end return i, d end\n"
            "  end return col end\n"
            "end"
        )
        runtime = h.load_mod(setup=setup)
        runtime.execute("Civ6Stub.cities[301].districts = { 0, { type = 1, complete = false } }")
        assert h.export_state(runtime)["cities"][0]["districts"] == [
            "DISTRICT_CITY_CENTER",
            "DISTRICT_CAMPUS",
        ]

    def test_missing_get_buildings_empties_only_buildings(self) -> None:
        setup = "for _, c in Players[0]:GetCities():Members() do c.GetBuildings = nil end"
        city = h.export_state(h.load_mod(setup=setup))["cities"][0]
        assert city["buildings"] == []
        assert city["districts"] == ["DISTRICT_CITY_CENTER", "DISTRICT_CAMPUS"]

    def test_missing_get_districts_empties_only_districts(self) -> None:
        setup = "for _, c in Players[0]:GetCities():Members() do c.GetDistricts = nil end"
        city = h.export_state(h.load_mod(setup=setup))["cities"][0]
        assert city["districts"] == []
        assert city["buildings"] == ["BUILDING_MONUMENT"]


# ---------------------------------------------------------------------------
# Exporter: legal research actions
# ---------------------------------------------------------------------------
class TestResearchActions:
    def test_default_tech_offer_excludes_the_current_one(self, state: dict[str, Any]) -> None:
        (action,) = _actions(state, "research_tech")
        assert action["items"] == [2]
        assert action["item_names"] == ["TECH_WRITING"]

    def test_default_civic_offer_respects_prerequisites(self, state: dict[str, Any]) -> None:
        (action,) = _actions(state, "research_civic")
        assert action["items"] == [2]
        assert action["item_names"] == ["CIVIC_FOREIGN_TRADE"]

    def test_with_no_current_research_every_researchable_tech_is_offered(
        self, runtime: Any
    ) -> None:
        runtime.execute("Civ6Stub.players[0].tech.current = -1")
        (action,) = _actions(h.export_state(runtime), "research_tech")
        assert action["items"] == [1, 2]
        assert action["item_names"] == ["TECH_MINING", "TECH_WRITING"]

    def test_a_tech_already_had_is_not_offered(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.players[0].tech.current = -1")
        runtime.execute("Civ6Stub.players[0].tech.has[1] = true")
        (action,) = _actions(h.export_state(runtime), "research_tech")
        assert action["items"] == [2]

    def test_no_tech_action_when_nothing_is_researchable(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.players[0].tech.can = {}")
        state = h.export_state(runtime)
        assert _actions(state, "research_tech") == []
        assert _actions(state, "research_civic"), "civic offer vanished; control is vacuous"

    def test_civics_current_one_is_excluded_and_none_means_all_eligible(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.players[0].civic.current = -1")
        (action,) = _actions(h.export_state(runtime), "research_civic")
        assert action["items"] == [1, 2]
        assert action["item_names"] == ["CIVIC_CRAFTSMANSHIP", "CIVIC_FOREIGN_TRADE"]

    def test_a_civic_the_game_can_progress_is_offered(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.players[0].civic.has = { [0] = true, [2] = true }")
        runtime.execute("Civ6Stub.players[0].civic.can[3] = true")
        (action,) = _actions(h.export_state(runtime), "research_civic")
        # Craftsmanship is current, Foreign Trade is had: only Early Empire is left.
        assert action["items"] == [3]
        assert action["item_names"] == ["CIVIC_EARLY_EMPIRE"]

    def test_can_progress_beats_the_prerequisite_table_when_it_offers_more(
        self, runtime: Any
    ) -> None:
        # The table says Early Empire is out of reach (Foreign Trade is not had).
        runtime.execute("Civ6Stub.players[0].civic.can[3] = true")
        (action,) = _actions(h.export_state(runtime), "research_civic")
        assert action["items"] == [2, 3]
        assert action["item_names"] == ["CIVIC_FOREIGN_TRADE", "CIVIC_EARLY_EMPIRE"]

    def test_can_progress_beats_the_prerequisite_table_when_it_offers_less(
        self, runtime: Any
    ) -> None:
        # The table would allow Foreign Trade; the game says it cannot be progressed.
        runtime.execute("Civ6Stub.players[0].civic.can = { [1] = true }")
        state = h.export_state(runtime)
        assert _actions(state, "research_civic") == []
        assert _actions(state, "research_tech"), "tech offer vanished; control is vacuous"

    def test_without_can_progress_the_prerequisite_table_decides(self) -> None:
        runtime = h.load_mod(setup=_without_methods("", "GetCulture", "CanProgress"))
        runtime.execute("Civ6Stub.players[0].civic.can = {}")  # would empty the offer
        (action,) = _actions(h.export_state(runtime), "research_civic")
        assert action["items"] == [2]
        runtime.execute("Civ6Stub.players[0].civic.has = { [0] = true, [2] = true }")
        (action,) = _actions(h.export_state(runtime), "research_civic")
        # Early Empire needs Foreign Trade, now had; Craftsmanship is current.
        assert action["items"] == [3]
        assert action["item_names"] == ["CIVIC_EARLY_EMPIRE"]

    def test_no_civic_action_without_can_progress_and_the_prerequisite_table(self) -> None:
        setup = _without_methods("", "GetCulture", "CanProgress") + "\nGameInfo.CivicPrereqs = nil"
        state = h.export_state(h.load_mod(setup=setup))
        assert _actions(state, "research_civic") == []
        assert _actions(state, "research_tech"), "tech offer vanished; control is vacuous"

    def test_the_prerequisite_table_is_not_needed_when_can_progress_exists(self) -> None:
        state = h.export_state(h.load_mod(setup="GameInfo.CivicPrereqs = nil"))
        (action,) = _actions(state, "research_civic")
        assert action["items"] == [2]

    def test_a_raising_can_research_keeps_the_other_actions(self) -> None:
        setup = _without_methods("", "GetTechs").replace(
            "return t end", "t.CanResearch = function() error('boom') end return t end"
        )
        assert _actions(h.export_state(h.load_mod()), "research_tech"), "control: healthy offer"
        runtime = h.load_mod(setup=setup)
        # GM-B9: an attack is offered only with OBSERVE_ALL_PLAYERS on, so run with it on.
        runtime.execute('AlphaZeroConfig.Set("OBSERVE_ALL_PLAYERS", true)')
        state = h.export_state(runtime)
        names = [a["action"] for a in state["legal_actions"]]
        assert names[0] == "end_turn"
        assert "research_tech" not in names
        for kept in ("found_city", "attack", "produce"):
            assert kept in names, f"{kept} lost when research failed"

    def test_a_raising_prerequisite_table_keeps_the_other_actions(self) -> None:
        setup = _without_methods("", "GetCulture", "CanProgress") + (
            "\nGameInfo.CivicPrereqs = function() error('boom') end"
        )
        assert _actions(h.export_state(h.load_mod()), "research_civic"), "control: healthy offer"
        runtime = h.load_mod(setup=setup)
        # GM-B9: an attack is offered only with OBSERVE_ALL_PLAYERS on, so run with it on.
        runtime.execute('AlphaZeroConfig.Set("OBSERVE_ALL_PLAYERS", true)')
        state = h.export_state(runtime)
        names = [a["action"] for a in state["legal_actions"]]
        assert names[0] == "end_turn"
        assert "research_civic" not in names
        for kept in ("found_city", "attack", "produce"):
            assert kept in names


# ---------------------------------------------------------------------------
# Executor: end_turn
# ---------------------------------------------------------------------------
END_TURN = {"action_type": "end_turn"}


class TestEndTurn:
    @pytest.fixture(params=["query", "scan"])
    def runtime(self, request: pytest.FixtureRequest) -> Any:
        return h.load_mod(setup=None if request.param == "query" else QUERY_OFF)

    def test_not_blocked_requests_one_action(self, runtime: Any) -> None:
        result = h.execute(runtime, END_TURN)
        assert result == (True, None, "End turn requested")
        assert h.stub_calls(runtime) == ["UI.RequestAction"]
        assert runtime.eval("Civ6Stub.calls[1].args[1]") == 7

    def test_blocked_is_refused_with_the_blocker(self, runtime: Any) -> None:
        _notify(runtime, RESEARCH_BLOCKER)
        success, code, message = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "END_TURN_BLOCKED")
        assert message.startswith("End turn blocked: ENDTURN_BLOCKING_RESEARCH")
        assert "Choose a technology" in message
        assert h.stub_calls(runtime) == []
        assert runtime.eval("Civ6Stub.turnDeactivated") is False

    def test_blocked_without_a_message_still_names_the_blocker(self, runtime: Any) -> None:
        _notify(runtime, "{ id = 9, blocking = 4, dismissed = false }")
        success, code, message = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "END_TURN_BLOCKED")
        assert message.startswith("End turn blocked: ENDTURN_BLOCKING_CIVIC")
        assert h.stub_calls(runtime) == []
        assert runtime.eval("Civ6Stub.turnDeactivated") is False

    def test_a_listed_blocker_is_refused_even_when_can_end_turn_is_true(self, runtime: Any) -> None:
        _notify(runtime, RESEARCH_BLOCKER)
        runtime.execute("Civ6Stub.canEndTurn = true")
        success, code, message = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "END_TURN_BLOCKED")
        assert message.startswith("End turn blocked: ENDTURN_BLOCKING_RESEARCH")
        assert h.stub_calls(runtime) == []
        assert runtime.eval("Civ6Stub.turnDeactivated") is False

    def test_a_dismissed_blocker_does_not_block(self, runtime: Any) -> None:
        _notify(runtime, RESEARCH_BLOCKER.replace("dismissed = false", "dismissed = true"))
        assert h.execute(runtime, END_TURN) == (True, None, "End turn requested")
        assert h.stub_calls(runtime) == ["UI.RequestAction"]

    def test_missing_notification_manager_refuses_the_end_of_turn(self) -> None:
        # GM-B9: the mod cannot ask what blocks, so nothing is requested (was: requested).
        runtime = h.load_mod(setup="NotificationManager = nil")
        success, code, message = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "END_TURN_BLOCKED")
        assert "ENDTURN_BLOCKING_NOT_ASKABLE" in message
        assert h.stub_calls(runtime) == []
        assert runtime.eval("Civ6Stub.turnDeactivated") is False

    @pytest.mark.parametrize("setup", ["UI = nil", "UI.RequestAction = nil", "ActionTypes = nil"])
    def test_nothing_to_request_with_is_a_failure(self, setup: str) -> None:
        runtime = h.load_mod(setup=setup)
        success, code, message = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert message.startswith("End turn unavailable")
        assert h.stub_calls(runtime) == []

    def test_a_raising_request_is_execution_failed(self, runtime: Any) -> None:
        runtime.execute("UI.RequestAction = function() error('blocked') end")
        success, code, message = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert "blocked" in message


# ---------------------------------------------------------------------------
# Executor: research_tech / research_civic
# ---------------------------------------------------------------------------
KINDS = {
    "tech": {
        "action": "research_tech",
        "operation": 11,
        "param": 501,
        "setter": "SetResearchingTech",
        "blocker": RESEARCH_BLOCKER,
        "other_blocker": CIVIC_BLOCKER,
        "name": "TECH_WRITING",
        "index": 2,
        "hash": 7122,
        "requested": "Research requested: TECH_WRITING",
        "fallback": "Researching tech: TECH_WRITING",
        "getter": "GetTechs",
        "can": "CanResearch",
        "bad_name": "TECH_BOGUS",
        "have_name": "TECH_POTTERY",
        "have_index": 0,
        "locked_name": "TECH_BRONZE_WORKING",
        "locked_index": 3,
        "state": "tech",
        "current_key": "current_research",
        "rows": "Technologies",
    },
    "civic": {
        "action": "research_civic",
        "operation": 12,
        "param": 502,
        "setter": "SetProgressingCivic",
        "blocker": CIVIC_BLOCKER,
        "other_blocker": RESEARCH_BLOCKER,
        "name": "CIVIC_FOREIGN_TRADE",
        "index": 2,
        "hash": 8122,
        "requested": "Civic requested: CIVIC_FOREIGN_TRADE",
        "fallback": "Progressing civic: CIVIC_FOREIGN_TRADE",
        "getter": "GetCulture",
        "can": "CanProgress",
        "bad_name": "CIVIC_BOGUS",
        "have_name": "CIVIC_CODE_OF_LAWS",
        "have_index": 0,
        "locked_name": "CIVIC_EARLY_EMPIRE",
        "locked_index": 3,
        "state": "civic",
        "current_key": "current_civic",
        "rows": "Civics",
    },
}
ITEM_FORMS = ["index", "name"]
FALLBACK_SETUPS = [
    "UI.RequestPlayerOperation = nil",
    "PlayerOperations = nil",
    "UI.RequestPlayerOperation = function() error('refused') end",
]


def _item(kind: str, form: str) -> Any:
    spec = KINDS[kind]
    return spec["index"] if form == "index" else spec["name"]


def _strip_hash(rows: str) -> str:
    """Lua setup: the rows of ``GameInfo.<rows>`` lose their ``Hash``."""
    return (
        f"local orig = GameInfo.{rows}\n"
        f"GameInfo.{rows} = function() local it = orig() return function()"
        " local r = it() if not r then return nil end local c = {}"
        " for k, v in pairs(r) do c[k] = v end c.Hash = nil return c end end"
    )


@pytest.mark.parametrize("kind", sorted(KINDS))
class TestResearchChoice:
    @pytest.mark.parametrize("form", ITEM_FORMS)
    def test_primary_path_requests_the_operation(self, runtime: Any, kind: str, form: str) -> None:
        spec = KINDS[kind]
        result = h.execute(runtime, {"action_type": spec["action"], "item_id": _item(kind, form)})
        assert result == (True, None, spec["requested"])
        assert h.stub_calls(runtime) == ["UI.RequestPlayerOperation"]
        assert _call_args(runtime, 1, 2) == [0, spec["operation"]]
        # The game wants the row's hash, not its index, and an exclusive insert.
        assert runtime.eval(f"Civ6Stub.calls[1].args[3][{spec['param']}]") == spec["hash"]
        assert runtime.eval("Civ6Stub.calls[1].args[3][503]") == 601

    @pytest.mark.parametrize("form", ITEM_FORMS)
    def test_the_game_takes_the_request_and_the_state_follows(
        self, runtime: Any, kind: str, form: str
    ) -> None:
        spec = KINDS[kind]
        before = h.export_state(runtime)["player"][spec["current_key"]]
        h.execute(runtime, {"action_type": spec["action"], "item_id": _item(kind, form)})
        assert runtime.eval("Civ6Stub.applyRequests()") == 1
        after = h.export_state(runtime)["player"][spec["current_key"]]
        assert after == spec["name"]
        assert before != after, "the request changed nothing; control is vacuous"

    @pytest.mark.parametrize("form", ITEM_FORMS)
    @pytest.mark.parametrize("setup", FALLBACK_SETUPS)
    def test_fallback_uses_the_setter_with_the_index(
        self, kind: str, form: str, setup: str
    ) -> None:
        spec = KINDS[kind]
        runtime = h.load_mod(setup=setup)
        result = h.execute(runtime, {"action_type": spec["action"], "item_id": _item(kind, form)})
        assert result == (True, None, spec["fallback"])
        assert h.stub_calls(runtime) == [spec["setter"]]
        assert _call_args(runtime, 1, 2) == [0, spec["index"]]

    def test_a_row_without_a_hash_falls_back_to_the_setter(self, kind: str) -> None:
        spec = KINDS[kind]
        runtime = h.load_mod(setup=_strip_hash(spec["rows"]))
        result = h.execute(runtime, {"action_type": spec["action"], "item_id": spec["index"]})
        assert result == (True, None, spec["fallback"])
        assert h.stub_calls(runtime) == [spec["setter"]]
        assert _call_args(runtime, 1, 2) == [0, spec["index"]]

    def test_neither_path_available_fails(self, kind: str) -> None:
        spec = KINDS[kind]
        setup = "UI.RequestPlayerOperation = nil\n" + _without_methods(
            "", spec["getter"], spec["setter"]
        )
        runtime = h.load_mod(setup=setup)
        success, code, _ = h.execute(
            runtime, {"action_type": spec["action"], "item_id": spec["index"]}
        )
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert h.stub_calls(runtime) == []

    @pytest.mark.parametrize("item", ["BOGUS", 99])
    def test_unknown_item_is_an_invalid_target(self, runtime: Any, kind: str, item: Any) -> None:
        spec = KINDS[kind]
        item_id = spec["bad_name"] if item == "BOGUS" else item
        _notify(runtime, spec["blocker"])
        success, code, _ = h.execute(runtime, {"action_type": spec["action"], "item_id": item_id})
        assert (success, code) == (False, "INVALID_TARGET")
        assert h.stub_calls(runtime) == []

    def test_missing_item_id_is_an_invalid_command(self, runtime: Any, kind: str) -> None:
        spec = KINDS[kind]
        success, code, _ = h.execute(runtime, {"action_type": spec["action"]})
        assert (success, code) == (False, "INVALID_COMMAND")
        assert h.stub_calls(runtime) == []

    @pytest.mark.parametrize("form", ITEM_FORMS)
    @pytest.mark.parametrize("setup", [None, *FALLBACK_SETUPS[:1]])
    def test_an_item_the_player_has_is_refused(
        self, kind: str, form: str, setup: str | None
    ) -> None:
        spec = KINDS[kind]
        runtime = h.load_mod(setup=setup)
        item = spec["have_index"] if form == "index" else spec["have_name"]
        success, code, message = h.execute(
            runtime, {"action_type": spec["action"], "item_id": item}
        )
        assert (success, code) == (False, "INVALID_TARGET")
        assert "already" in message.lower()
        assert h.stub_calls(runtime) == []

    @pytest.mark.parametrize("form", ITEM_FORMS)
    @pytest.mark.parametrize("setup", [None, *FALLBACK_SETUPS[:1]])
    def test_an_item_the_game_says_cannot_be_chosen_is_refused(
        self, kind: str, form: str, setup: str | None
    ) -> None:
        spec = KINDS[kind]
        runtime = h.load_mod(setup=setup)
        item = spec["locked_index"] if form == "index" else spec["locked_name"]
        success, code, message = h.execute(
            runtime, {"action_type": spec["action"], "item_id": item}
        )
        assert (success, code) == (False, "INVALID_TARGET")
        assert "cannot" in message.lower()
        assert h.stub_calls(runtime) == []

    def test_without_the_can_method_the_gate_is_skipped(self, kind: str) -> None:
        spec = KINDS[kind]
        runtime = h.load_mod(setup=_without_methods("", spec["getter"], spec["can"]))
        result = h.execute(
            runtime, {"action_type": spec["action"], "item_id": spec["locked_index"]}
        )
        assert result[:2] == (True, None)
        assert h.stub_calls(runtime) == ["UI.RequestPlayerOperation"]

    def test_only_an_exact_false_refuses(self, kind: str) -> None:
        spec = KINDS[kind]
        setup = _without_methods("", spec["getter"]).replace(
            "return t end", f"t.{spec['can']} = function() return nil end return t end"
        )
        runtime = h.load_mod(setup=setup)
        result = h.execute(
            runtime, {"action_type": spec["action"], "item_id": spec["locked_index"]}
        )
        assert result[:2] == (True, None)
        assert h.stub_calls(runtime) == ["UI.RequestPlayerOperation"]

    @pytest.mark.parametrize("setup", [None, *FALLBACK_SETUPS[:1]])
    def test_a_choice_leaves_the_notifications_alone(self, kind: str, setup: str | None) -> None:
        spec = KINDS[kind]
        runtime = h.load_mod(setup=setup)
        _notify(runtime, spec["blocker"], spec["other_blocker"], PRODUCTION_BLOCKER)
        success, code, _ = h.execute(
            runtime, {"action_type": spec["action"], "item_id": spec["index"]}
        )
        assert (success, code) == (True, None)
        first = "UI.RequestPlayerOperation" if setup is None else spec["setter"]
        assert h.stub_calls(runtime) == [first]  # no SendActivated, no Dismiss
        for position in (1, 2, 3):
            assert runtime.eval(f"Civ6Stub.notifications[0][{position}].dismissed") is False

    def test_missing_notification_manager_is_fine(self, kind: str) -> None:
        spec = KINDS[kind]
        runtime = h.load_mod(setup="NotificationManager = nil")
        result = h.execute(runtime, {"action_type": spec["action"], "item_id": spec["index"]})
        assert result == (True, None, spec["requested"])
        assert h.stub_calls(runtime) == ["UI.RequestPlayerOperation"]
        assert runtime.eval("Civ6Stub.applyRequests()") == 1


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", sorted(KINDS))
class TestBlockedTurnScenario:
    def test_choose_research_then_the_game_takes_it_then_end_turn(
        self, runtime: Any, kind: str
    ) -> None:
        spec = KINDS[kind]
        _notify(runtime, spec["blocker"])

        before = h.export_state(runtime)
        assert before["end_turn_blocking"].startswith("ENDTURN_BLOCKING_")
        (offer,) = _actions(before, spec["action"])

        blocked = h.execute(runtime, END_TURN)
        assert blocked[:2] == (False, "END_TURN_BLOCKED")
        assert h.stub_calls(runtime) == []

        chosen = h.execute(runtime, {"action_type": spec["action"], "item_id": offer["items"][0]})
        assert chosen[:2] == (True, None)
        # Requested, not yet taken: the blocker is still reported, nothing was dismissed.
        assert h.stub_calls(runtime) == ["UI.RequestPlayerOperation"]
        assert h.export_state(runtime)["end_turn_blocking"] == before["end_turn_blocking"]
        assert h.execute(runtime, END_TURN)[:2] == (False, "END_TURN_BLOCKED")

        assert runtime.eval("Civ6Stub.applyRequests()") == 1
        after = h.export_state(runtime)
        assert "end_turn_blocking" not in after
        assert after["player"][spec["current_key"]] == offer["item_names"][0]

        assert h.execute(runtime, END_TURN) == (True, None, "End turn requested")
        assert h.stub_calls(runtime)[-1] == "UI.RequestAction"

    def test_a_request_the_game_drops_leaves_the_turn_blocked(
        self, runtime: Any, kind: str
    ) -> None:
        spec = KINDS[kind]
        _notify(runtime, spec["blocker"])
        (offer,) = _actions(h.export_state(runtime), spec["action"])
        chosen = h.execute(runtime, {"action_type": spec["action"], "item_id": offer["items"][0]})
        assert chosen[:2] == (True, None)
        # The game refuses it by the next frame: the item can no longer be chosen.
        runtime.execute(f"Civ6Stub.players[0].{spec['state']}.can = {{}}")
        assert runtime.eval("Civ6Stub.applyRequests()") == 0

        state = h.export_state(runtime)
        assert state["end_turn_blocking"].startswith("ENDTURN_BLOCKING_")
        assert state["player"][spec["current_key"]] != offer["item_names"][0]
        success, code, _ = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "END_TURN_BLOCKED")
        assert "UI.RequestAction" not in h.stub_calls(runtime)
        assert runtime.eval("Civ6Stub.turnDeactivated") is False


class TestSpecScenarioEdges:
    """Two scenarios the OpenSpec change names that the classes above leave out."""

    def test_a_raising_civics_call_empties_the_civics_and_keeps_the_state(self) -> None:
        setup = (
            "local orig = Players[0].GetCulture\n"
            "Players[0].GetCulture = function(self) local c = orig(self)"
            " c.HasCivic = function() error('boom') end return c end"
        )
        state = h.export_state(h.load_mod(setup=setup))
        assert state["schema_version"] == 2
        assert not state["player"]["civics_researched"]
        # The rest of the player block survives the raising call.
        assert state["player"]["techs_researched"] == ["TECH_POTTERY"]
        assert state["player"]["gold"] == 50

    def test_a_blocker_without_can_end_turn_is_refused(self) -> None:
        runtime = h.load_mod(setup="UI.CanEndTurn = nil")
        _notify(runtime, RESEARCH_BLOCKER)
        success, code, message = h.execute(runtime, {"action_type": "end_turn"})
        assert (success, code) == (False, "END_TURN_BLOCKED")
        assert message.startswith("End turn blocked: ENDTURN_BLOCKING_RESEARCH")
        assert "UI.RequestAction" not in h.stub_calls(runtime)
        assert runtime.eval("Civ6Stub.turnDeactivated") is False


DECLARE_WAR = {"action_type": "diplomacy", "metadata": {"sub_action": 0, "target_player_id": 1}}


def _diplomacy(body: str) -> str:
    """Lua setup: player 0 has a diplomacy object whose ``DoAction`` runs ``body``."""
    return (
        "Players[0].GetDiplomacy = function(self) local d = {}"
        f" function d:DoAction(action, target) {body} end return d end"
    )


def _load_allowing_war(setup: str) -> Any:
    """GM-B9: declaring war is off by default; these tests are about what happens when it is on."""
    runtime = h.load_mod(setup=setup)
    runtime.execute('AlphaZeroConfig.Set("ALLOW_DECLARE_WAR", true)')
    return runtime


class TestDiplomacyFailureIsReported:
    """A piece the hand-edited bundle had lost (bundle-drift.md), brought back in GM-B2."""

    def test_a_raising_diplomacy_call_reports_its_error(self) -> None:
        runtime = _load_allowing_war(setup=_diplomacy("error('war refused')"))
        success, code, message = h.execute(runtime, DECLARE_WAR)
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert message.startswith("Diplomacy action DECLARE_WAR failed:")
        assert "war refused" in message

    def test_the_last_path_tried_is_the_error_reported(self) -> None:
        setup = _diplomacy("error('first path')") + (
            "\nUI.RequestDiplomacyMove = function() error('second path') end"
        )
        success, code, message = h.execute(_load_allowing_war(setup=setup), DECLARE_WAR)
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert "second path" in message
        assert "first path" not in message

    def test_the_ui_path_still_rescues_a_failed_first_path(self) -> None:
        setup = _diplomacy("error('first path')") + "\nUI.RequestDiplomacyMove = function() end"
        success, code, message = h.execute(_load_allowing_war(setup=setup), DECLARE_WAR)
        assert (success, code) == (True, None)
        assert "requested via UI" in message

    def test_with_no_api_to_call_the_old_message_stays(self) -> None:
        setup = "Players[0].GetDiplomacy = function(self) return {} end"
        success, code, message = h.execute(_load_allowing_war(setup=setup), DECLARE_WAR)
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert message == "No diplomacy API available for DECLARE_WAR"

    def test_a_working_call_succeeds(self) -> None:
        success, code, message = h.execute(_load_allowing_war(setup=_diplomacy("")), DECLARE_WAR)
        assert (success, code) == (True, None)
        assert message == "DECLARE_WAR against player 1"


def _heartbeats_at(runtime: Any, *times: int) -> list[int]:
    """Poll the bridge at each clock value; the heartbeat count after each poll."""
    counts = []
    for now in times:
        runtime.execute(f"Civ6Stub.now = {int(now)}; AlphaZeroBridge.PollCommands()")
        counts.append(runtime.eval("Civ6Stub.heartbeats"))
    return counts


class TestHeartbeatIntervalComesFromConfig:
    """A piece the hand-edited bundle had lost (bundle-drift.md), brought back in GM-B2."""

    @pytest.fixture
    def polling(self) -> Any:
        """A loaded mod on the AI player's turn, with a clock and a heartbeat counter."""
        runtime = h.load_mod()
        runtime.execute(
            "Civ6Stub.now = 1000\n"
            "os.time = function() return Civ6Stub.now end\n"
            "Civ6Stub.heartbeats = 0\n"
            "AlphaZeroFileIO.WriteHeartbeat = function()"
            " Civ6Stub.heartbeats = Civ6Stub.heartbeats + 1 end\n"
            "AlphaZeroBridge.OnTurnStart(0)"
        )
        return runtime

    def test_a_longer_interval_spaces_the_heartbeats(self, polling: Any) -> None:
        polling.execute("AlphaZeroConfig.Set('HEARTBEAT_INTERVAL', 5)")
        # First poll writes one; 3 s later is inside the interval; 5 s later is not.
        assert _heartbeats_at(polling, 1000, 1003, 1005, 1009, 1010) == [1, 1, 2, 2, 3]

    def test_the_default_interval_is_one_second(self, polling: Any) -> None:
        assert polling.eval("AlphaZeroConfig.Get('HEARTBEAT_INTERVAL')") == 1
        assert _heartbeats_at(polling, 1000, 1000, 1001, 1003) == [1, 1, 2, 3]

    def test_a_missing_interval_falls_back_to_one_second(self, polling: Any) -> None:
        polling.execute("AlphaZeroConfig.Set('HEARTBEAT_INTERVAL', nil)")
        assert _heartbeats_at(polling, 1000, 1000, 1001) == [1, 1, 2]

    @pytest.mark.parametrize("value", ["'soon'", "0", "-3", "false"])
    def test_an_unusable_interval_falls_back_instead_of_breaking_the_poll(
        self, polling: Any, value: str
    ) -> None:
        polling.execute(f"AlphaZeroConfig.Set('HEARTBEAT_INTERVAL', {value})")
        assert _heartbeats_at(polling, 1000, 1000, 1001) == [1, 1, 2]
