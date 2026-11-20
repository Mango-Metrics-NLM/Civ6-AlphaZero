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
class TestSchemaAndBlocker:
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

    def test_missing_notification_manager_means_no_blocker(self) -> None:
        runtime = h.load_mod(setup="NotificationManager = nil")
        _notify_ok = runtime.eval("Civ6Stub.notifications ~= nil")
        state = h.export_state(runtime)
        assert _notify_ok is True
        assert state["schema_version"] == 2
        assert "end_turn_blocking" not in state


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

    def test_no_current_civic_drops_name_and_status(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.players[0].civic.current = -1")
        player = h.export_state(runtime)["player"]
        assert player["current_research"] == "TECH_MINING", "control: the rest is exported"
        assert "current_civic" not in player
        assert "civic_status" not in player
        assert player["civics_researched"] == ["CIVIC_CODE_OF_LAWS"]

    @pytest.mark.parametrize(
        ("method", "field"),
        [
            ("GetCultureCost", "cost"),
            ("GetCulturalProgress", "progress"),
            ("GetTurnsLeftOnCurrentCivic", "turns_left"),
            ("HasBoostBeenTriggered", "boost_triggered"),
        ],
    )
    def test_a_missing_method_drops_only_its_field(self, method: str, field: str) -> None:
        runtime = h.load_mod(setup=_without_methods("", "GetCulture", method))
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

    def test_a_prerequisite_met_unlocks_the_civic(self, runtime: Any) -> None:
        runtime.execute("Civ6Stub.players[0].civic.has = { [0] = true, [2] = true }")
        (action,) = _actions(h.export_state(runtime), "research_civic")
        # Early Empire needs Foreign Trade, now had; Craftsmanship is current.
        assert action["items"] == [3]
        assert action["item_names"] == ["CIVIC_EARLY_EMPIRE"]

    def test_no_civic_action_without_the_prerequisite_table(self) -> None:
        state = h.export_state(h.load_mod(setup="GameInfo.CivicPrereqs = nil"))
        assert _actions(state, "research_civic") == []
        assert _actions(state, "research_tech"), "tech offer vanished; control is vacuous"

    def test_a_raising_can_research_keeps_the_other_actions(self) -> None:
        setup = _without_methods("", "GetTechs").replace(
            "return t end", "t.CanResearch = function() error('boom') end return t end"
        )
        assert _actions(h.export_state(h.load_mod()), "research_tech"), "control: healthy offer"
        state = h.export_state(h.load_mod(setup=setup))
        names = [a["action"] for a in state["legal_actions"]]
        assert names[0] == "end_turn"
        assert "research_tech" not in names
        for kept in ("found_city", "attack", "produce"):
            assert kept in names, f"{kept} lost when research failed"

    def test_a_raising_prerequisite_table_keeps_the_other_actions(self) -> None:
        setup = "GameInfo.CivicPrereqs = function() error('boom') end"
        assert _actions(h.export_state(h.load_mod()), "research_civic"), "control: healthy offer"
        state = h.export_state(h.load_mod(setup=setup))
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

    def test_blocked_without_a_message_still_names_the_blocker(self, runtime: Any) -> None:
        _notify(runtime, "{ id = 9, blocking = 4, dismissed = false }")
        success, code, message = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "END_TURN_BLOCKED")
        assert message.startswith("End turn blocked: ENDTURN_BLOCKING_CIVIC")
        assert h.stub_calls(runtime) == []

    def test_a_listed_blocker_with_can_end_turn_true_is_requested(self, runtime: Any) -> None:
        _notify(runtime, RESEARCH_BLOCKER)
        runtime.execute("Civ6Stub.canEndTurn = true")
        assert h.execute(runtime, END_TURN) == (True, None, "End turn requested")
        assert h.stub_calls(runtime) == ["UI.RequestAction"]

    def test_a_dismissed_blocker_does_not_block(self, runtime: Any) -> None:
        _notify(runtime, RESEARCH_BLOCKER.replace("dismissed = false", "dismissed = true"))
        assert h.execute(runtime, END_TURN) == (True, None, "End turn requested")
        assert h.stub_calls(runtime) == ["UI.RequestAction"]

    def test_missing_notification_manager_is_not_blocked(self) -> None:
        runtime = h.load_mod(setup="NotificationManager = nil")
        assert h.execute(runtime, END_TURN) == (True, None, "End turn requested")
        assert h.stub_calls(runtime) == ["UI.RequestAction"]

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
# kind -> (action_type, operation id, param key, setter, blocker entry, blocking name,
#          name of an index, index, ok message prefix, fallback prefix, getter)
KINDS = {
    "tech": {
        "action": "research_tech",
        "operation": 11,
        "param": 501,
        "setter": "SetResearchingTech",
        "blocker": RESEARCH_BLOCKER,
        "other_blocker": CIVIC_BLOCKER,
        "name": "TECH_MINING",
        "index": 1,
        "requested": "Research requested: TECH_MINING",
        "fallback": "Researching tech: TECH_MINING",
        "getter": "GetTechs",
        "bad_name": "TECH_BOGUS",
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
        "requested": "Civic requested: CIVIC_FOREIGN_TRADE",
        "fallback": "Progressing civic: CIVIC_FOREIGN_TRADE",
        "getter": "GetCulture",
        "bad_name": "CIVIC_BOGUS",
    },
}
ITEM_FORMS = ["index", "name"]


def _item(kind: str, form: str) -> Any:
    spec = KINDS[kind]
    return spec["index"] if form == "index" else spec["name"]


@pytest.mark.parametrize("kind", sorted(KINDS))
class TestResearchChoice:
    @pytest.mark.parametrize("form", ITEM_FORMS)
    def test_primary_path_requests_the_operation(self, runtime: Any, kind: str, form: str) -> None:
        spec = KINDS[kind]
        result = h.execute(runtime, {"action_type": spec["action"], "item_id": _item(kind, form)})
        assert result == (True, None, spec["requested"])
        assert h.stub_calls(runtime) == ["UI.RequestPlayerOperation"]
        player, operation, params_value = _call_args(runtime, 1, 3)[0:3]
        assert player == 0
        assert operation == spec["operation"]
        assert runtime.eval(f"Civ6Stub.calls[1].args[3][{spec['param']}]") == spec["index"]
        assert params_value is not None

    @pytest.mark.parametrize("form", ITEM_FORMS)
    @pytest.mark.parametrize(
        "setup",
        [
            "UI.RequestPlayerOperation = nil",
            "PlayerOperations = nil",
            "UI.RequestPlayerOperation = function() error('refused') end",
        ],
    )
    def test_fallback_uses_the_setter(self, kind: str, form: str, setup: str) -> None:
        spec = KINDS[kind]
        runtime = h.load_mod(setup=setup)
        result = h.execute(runtime, {"action_type": spec["action"], "item_id": _item(kind, form)})
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

    @pytest.mark.parametrize("path", ["primary", "fallback"])
    def test_choice_dismisses_only_the_matching_blockers(self, kind: str, path: str) -> None:
        spec = KINDS[kind]
        setup = "UI.RequestPlayerOperation = nil" if path == "fallback" else None
        runtime = h.load_mod(setup=setup)
        already = (
            spec["blocker"]
            .replace("id = ", "id = 1")
            .replace("dismissed = false", "dismissed = true")
        )
        _notify(runtime, spec["blocker"], spec["other_blocker"], PRODUCTION_BLOCKER, already)
        success, code, _ = h.execute(
            runtime, {"action_type": spec["action"], "item_id": spec["index"]}
        )
        assert (success, code) == (True, None)
        notification_id = 5 if kind == "tech" else 6
        calls = h.stub_calls(runtime)
        first = "UI.RequestPlayerOperation" if path == "primary" else spec["setter"]
        assert calls == [first, "NotificationManager.SendActivated", "NotificationManager.Dismiss"]
        assert _call_args(runtime, 2, 2) == [0, notification_id]
        assert _call_args(runtime, 3, 2) == [0, notification_id]
        # The other kinds stay: only the matching blocker is dismissed.
        assert runtime.eval("Civ6Stub.notifications[0][1].dismissed") is True
        assert runtime.eval("Civ6Stub.notifications[0][2].dismissed") is False
        assert runtime.eval("Civ6Stub.notifications[0][3].dismissed") is False

    def test_a_refused_choice_dismisses_nothing(self, runtime: Any, kind: str) -> None:
        spec = KINDS[kind]
        _notify(runtime, spec["blocker"])
        _, code, _ = h.execute(runtime, {"action_type": spec["action"], "item_id": 99})
        assert code == "INVALID_TARGET"
        assert "NotificationManager.Dismiss" not in h.stub_calls(runtime)
        assert runtime.eval("Civ6Stub.notifications[0][1].dismissed") is False

    def test_missing_notification_manager_is_fine(self, kind: str) -> None:
        spec = KINDS[kind]
        runtime = h.load_mod(setup="NotificationManager = nil")
        result = h.execute(runtime, {"action_type": spec["action"], "item_id": spec["index"]})
        assert result == (True, None, spec["requested"])
        assert h.stub_calls(runtime) == ["UI.RequestPlayerOperation"]


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
class TestBlockedTurnScenario:
    def test_choose_research_then_end_turn(self, runtime: Any) -> None:
        _notify(runtime, RESEARCH_BLOCKER)

        before = h.export_state(runtime)
        assert before["end_turn_blocking"] == "ENDTURN_BLOCKING_RESEARCH"
        (offer,) = _actions(before, "research_tech")

        blocked = h.execute(runtime, END_TURN)
        assert blocked[:2] == (False, "END_TURN_BLOCKED")
        assert h.stub_calls(runtime) == []

        chosen = h.execute(runtime, {"action_type": "research_tech", "item_id": offer["items"][0]})
        assert chosen[:2] == (True, None)
        assert runtime.eval("Civ6Stub.notifications[0][1].dismissed") is True

        after = h.export_state(runtime)
        assert "end_turn_blocking" not in after

        assert h.execute(runtime, END_TURN) == (True, None, "End turn requested")
        assert h.stub_calls(runtime)[-1] == "UI.RequestAction"


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


DECLARE_WAR = {"action_type": "diplomacy", "metadata": {"sub_action": 0, "target_player_id": 1}}


def _diplomacy(body: str) -> str:
    """Lua setup: player 0 has a diplomacy object whose ``DoAction`` runs ``body``."""
    return (
        "Players[0].GetDiplomacy = function(self) local d = {}"
        f" function d:DoAction(action, target) {body} end return d end"
    )


class TestDiplomacyFailureIsReported:
    """A piece the hand-edited bundle had lost (bundle-drift.md), brought back in GM-B2."""

    def test_a_raising_diplomacy_call_reports_its_error(self) -> None:
        runtime = h.load_mod(setup=_diplomacy("error('war refused')"))
        success, code, message = h.execute(runtime, DECLARE_WAR)
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert message.startswith("Diplomacy action DECLARE_WAR failed:")
        assert "war refused" in message

    def test_the_last_path_tried_is_the_error_reported(self) -> None:
        setup = _diplomacy("error('first path')") + (
            "\nUI.RequestDiplomacyMove = function() error('second path') end"
        )
        success, code, message = h.execute(h.load_mod(setup=setup), DECLARE_WAR)
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert "second path" in message
        assert "first path" not in message

    def test_the_ui_path_still_rescues_a_failed_first_path(self) -> None:
        setup = _diplomacy("error('first path')") + "\nUI.RequestDiplomacyMove = function() end"
        success, code, message = h.execute(h.load_mod(setup=setup), DECLARE_WAR)
        assert (success, code) == (True, None)
        assert "requested via UI" in message

    def test_with_no_api_to_call_the_old_message_stays(self) -> None:
        setup = "Players[0].GetDiplomacy = function(self) return {} end"
        success, code, message = h.execute(h.load_mod(setup=setup), DECLARE_WAR)
        assert (success, code) == (False, "EXECUTION_FAILED")
        assert message == "No diplomacy API available for DECLARE_WAR"

    def test_a_working_call_succeeds(self) -> None:
        success, code, message = h.execute(h.load_mod(setup=_diplomacy("")), DECLARE_WAR)
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
