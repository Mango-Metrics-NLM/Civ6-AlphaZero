"""GM-B10: the stand-in game's DEFAULTS are what the live session of 2026-11-21 showed.

Each test asks the stand-in with no hook set. The old reading stays available as a named
override (``canEndTurnHonoursBlockers``, ``currentlyBuildingAnswers``, ``useLongYieldKeys``),
and a second test per default shows the override still works, so a default cannot flip back
unnoticed and an override cannot rot.
"""

from __future__ import annotations

from tests.fixtures import lua_harness as h

RESEARCH_BLOCKER = (
    "Civ6Stub.notifications[0] = { { id = 5,"
    " blocking = EndTurnBlockingTypes.ENDTURN_BLOCKING_RESEARCH,"
    " message = 'Choose a technology', typeName = 'NOTIFICATION_CHOOSE_TECH', dismissed = false } }\n"
)


class TestCanEndTurn:
    def test_it_answers_true_while_a_blocker_is_active(self) -> None:
        runtime = h.load_mod(setup=RESEARCH_BLOCKER)
        assert runtime.eval("NotificationManager.GetFirstEndTurnBlocking(0)") != 0, "premise"
        assert runtime.eval("UI.CanEndTurn()") is True

    def test_it_answers_true_while_a_unit_needs_orders(self) -> None:
        runtime = h.load_mod(setup="Civ6Stub.unitsNeedOrders = true")
        assert runtime.eval("NotificationManager.GetFirstEndTurnBlocking(0)") != 0, "premise"
        assert runtime.eval("UI.CanEndTurn()") is True

    def test_the_game_still_does_not_end_the_turn_while_blocked(self) -> None:
        runtime = h.load_mod(setup=RESEARCH_BLOCKER)
        runtime.execute("UI.RequestAction(ActionTypes.ACTION_ENDTURN)")
        assert runtime.eval("Civ6Stub.turnEnded") is False

    def test_the_turn_ends_when_nothing_blocks(self) -> None:
        runtime = h.load_mod()
        runtime.execute("UI.RequestAction(ActionTypes.ACTION_ENDTURN)")
        assert runtime.eval("Civ6Stub.turnEnded") is True

    def test_the_old_reading_is_a_named_override(self) -> None:
        runtime = h.load_mod(setup=RESEARCH_BLOCKER + "Civ6Stub.canEndTurnHonoursBlockers = true")
        assert runtime.eval("UI.CanEndTurn()") is False

    def test_the_harm_is_modelled_a_forced_end_turn_while_blocked_deactivates_the_turn(
        self,
    ) -> None:
        """GM-B9 (T2): the live game deactivates the turn (LF-H1) instead of ending it."""
        runtime = h.load_mod(setup=RESEARCH_BLOCKER)
        assert runtime.eval("Civ6Stub.turnDeactivated") is False
        runtime.execute("UI.RequestAction(ActionTypes.ACTION_ENDTURN)")
        assert runtime.eval("Civ6Stub.turnDeactivated") is True
        assert runtime.eval("Civ6Stub.turnEnded") is False
        assert runtime.eval("Civ6Stub.turn") == 12

    def test_a_forced_end_turn_with_nothing_blocking_does_not_deactivate(self) -> None:
        runtime = h.load_mod()
        runtime.execute("UI.RequestAction(ActionTypes.ACTION_ENDTURN)")
        assert runtime.eval("Civ6Stub.turnDeactivated") is False


class TestCurrentlyBuilding:
    SETUP = "Civ6Stub.city(301).building = 'UNIT_WARRIOR'\n"

    def test_it_answers_nil_and_the_hash_names_the_item(self) -> None:
        runtime = h.load_mod(setup=self.SETUP)
        queue = "Players[0]:GetCities():FindID(301):GetBuildQueue()"
        assert runtime.eval(f"{queue}:CurrentlyBuilding()") is None
        assert runtime.eval(f"{queue}:GetCurrentProductionTypeHash()") == 6111

    def test_the_exporter_still_reads_the_item_from_the_hash(self) -> None:
        state = h.export_state(h.load_mod(setup=self.SETUP))
        assert state["cities"][0]["current_production"] == "UNIT_WARRIOR"

    def test_the_old_reading_is_a_named_override(self) -> None:
        runtime = h.load_mod(setup=self.SETUP + "Civ6Stub.currentlyBuildingAnswers = true")
        queue = "Players[0]:GetCities():FindID(301):GetBuildQueue()"
        assert runtime.eval(f"{queue}:CurrentlyBuilding()") == "UNIT_WARRIOR"

    def test_the_exporter_still_reads_the_item_when_currently_building_answers(self) -> None:
        """T3: the mod's "CurrentlyBuilding returns the item" branch. The hash answers 0 here,
        so only that branch can produce the name."""
        setup = (
            self.SETUP
            + "Civ6Stub.currentlyBuildingAnswers = true\n"
            + "Players[0]:GetCities():FindID(301):GetBuildQueue().GetCurrentProductionTypeHash"
            + " = function() return 0 end\n"
        )
        state = h.export_state(h.load_mod(setup=setup))
        assert state["cities"][0]["current_production"] == "UNIT_WARRIOR"


class TestYieldTypes:
    def test_the_keys_are_the_short_live_ones(self) -> None:
        runtime = h.load_mod()
        assert runtime.eval("YieldTypes.FOOD") == 0
        assert runtime.eval("YieldTypes.FAITH") == 5
        assert runtime.eval("YieldTypes.YIELD_FOOD") is None

    def test_the_exporter_reads_tile_yields_through_them(self) -> None:
        setup = "Map.GetPlot(1, 1).GetYield = function(_, y) return y + 10 end\n"
        state = h.export_state(h.load_mod(setup=setup))
        tile = next(t for t in state["map"]["tiles"] if (t["x"], t["y"]) == (1, 1))
        assert tile["yields"]["food"] == 10
        assert tile["yields"]["faith"] == 15

    def test_the_old_reading_is_a_named_override(self) -> None:
        runtime = h.load_mod(setup="Civ6Stub.useLongYieldKeys()")
        assert runtime.eval("YieldTypes.YIELD_FOOD") == 0
        assert runtime.eval("YieldTypes.FOOD") is None

    def test_the_exporter_still_reads_tile_yields_through_the_long_keys(self) -> None:
        """T3: the mod's long-key (YIELD_*) branch."""
        setup = (
            "Civ6Stub.useLongYieldKeys()\n"
            "Map.GetPlot(1, 1).GetYield = function(_, y) return y + 10 end\n"
        )
        state = h.export_state(h.load_mod(setup=setup))
        tile = next(t for t in state["map"]["tiles"] if (t["x"], t["y"]) == (1, 1))
        assert tile["yields"] == {
            "food": 10,
            "production": 11,
            "gold": 12,
            "science": 13,
            "culture": 14,
            "faith": 15,
        }


class TestLockedItem:
    def test_a_default_city_has_a_locked_item(self) -> None:
        runtime = h.load_mod()
        queue = "Players[0]:GetCities():FindID(301):GetBuildQueue()"
        assert runtime.eval(f"{queue}:CanProduce(6122, true)") is False
        assert runtime.eval(f"{queue}:CanProduce(6122, false, true)") is True

    def test_the_locked_item_is_not_offered(self) -> None:
        state = h.export_state(h.load_mod())
        produced = [
            i for a in state["legal_actions"] if a["action"] == "produce" for i in a["items"]
        ]
        assert "UNIT_WARRIOR" in produced, "premise: other items are offered"
        assert "UNIT_ARCHER" not in produced
