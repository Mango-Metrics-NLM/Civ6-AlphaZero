"""Regression tests for live Civ 6 StateExporter API parity.

Covers production resolution when ``CurrentlyBuilding()`` returns ``nil``
(falling back to ``GetCurrentProductionTypeHash()``) and live ``YieldTypes``
short keys (``FOOD``..``FAITH`` vs stub ``YIELD_FOOD``..``YIELD_FAITH``).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures import lua_harness as h


def _actions(state: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return [a for a in state["legal_actions"] if a["action"] == name]


class TestLiveCityProductionFallback:
    @pytest.mark.parametrize("item", ["UNIT_WARRIOR", "BUILDING_MONUMENT"])
    def test_current_production_falls_back_to_hash_when_currently_building_is_nil(
        self, item: str
    ) -> None:
        setup = (
            "local c = Players[0]:GetCities():FindID(301)\n"
            "local bq = c:GetBuildQueue()\n"
            "bq.CurrentlyBuilding = function() return nil end\n"
            f"Civ6Stub.city(301).building = '{item}'\n"
            "Civ6Stub.cities[301].canProduce = {}\n"
        )
        state = h.export_state(h.load_mod(setup=setup))
        assert state["cities"][0]["current_production"] == item
        assert _actions(state, "produce") == []

    def test_unknown_production_hash_falls_back_to_synthetic_prefix(self) -> None:
        setup = (
            "local c = Players[0]:GetCities():FindID(301)\n"
            "local bq = c:GetBuildQueue()\n"
            "bq.CurrentlyBuilding = function() return nil end\n"
            "bq.GetCurrentProductionTypeHash = function() return 424242 end\n"
            "Civ6Stub.cities[301].canProduce = {}\n"
        )
        state = h.export_state(h.load_mod(setup=setup))
        assert state["cities"][0]["current_production"] == "PRODUCTION_HASH_424242"
        assert _actions(state, "produce") == []


class TestLiveTileYieldTypes:
    def test_tile_yields_support_live_short_yield_type_keys(self) -> None:
        setup = (
            "YieldTypes = {"
            " FOOD = 0, PRODUCTION = 1, GOLD = 2, SCIENCE = 3, CULTURE = 4, FAITH = 5 }\n"
            "local plot = Map.GetPlot(1, 1)\n"
            "plot.GetYield = function(_, y) return y + 10 end\n"
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

    def test_tile_yields_fall_back_to_gameinfo_yields_when_yieldtypes_nil(self) -> None:
        setup = (
            "YieldTypes = nil\n"
            "GameInfo.Yields = function()\n"
            "  local rows = {\n"
            "    { Index = 0, YieldType = 'YIELD_FOOD' },\n"
            "    { Index = 1, YieldType = 'YIELD_PRODUCTION' },\n"
            "    { Index = 2, YieldType = 'YIELD_GOLD' },\n"
            "    { Index = 3, YieldType = 'YIELD_SCIENCE' },\n"
            "    { Index = 4, YieldType = 'YIELD_CULTURE' },\n"
            "    { Index = 5, YieldType = 'YIELD_FAITH' },\n"
            "  }\n"
            "  local i = 0\n"
            "  return function() i = i + 1; return rows[i] end\n"
            "end\n"
            "local plot = Map.GetPlot(1, 1)\n"
            "plot.GetYield = function(_, y) return y + 20 end\n"
        )
        state = h.export_state(h.load_mod(setup=setup))
        tile = next(t for t in state["map"]["tiles"] if (t["x"], t["y"]) == (1, 1))
        assert tile["yields"] == {
            "food": 20,
            "production": 21,
            "gold": 22,
            "science": 23,
            "culture": 24,
            "faith": 25,
        }
