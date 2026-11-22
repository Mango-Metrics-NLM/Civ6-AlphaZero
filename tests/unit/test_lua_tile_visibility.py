"""GM-B8a: the map export tells only what the player sees now, whatever the observe flag.

Run in the stand-in game (lupa, Lua 5.1). Tile (1, 1) is the stand-in's one special tile: a
forested hill owned by player 0 with wheat and a farm.
"""

from __future__ import annotations

import pytest

from tests.fixtures import lua_harness as h
from tests.fixtures import lua_others as o

SEEN_ONLY_KEYS = ("owner_id", "improvement", "resource")
SPECIAL = (1, 1)


def _state(setup: str = "", *, observe: bool, fog: bool = True) -> dict:
    runtime = o.load(setup, observe=observe)
    runtime.execute(f'AlphaZeroConfig.Set("INCLUDE_FOG_OF_WAR", {"true" if fog else "false"})')
    return h.export_state(runtime)


@pytest.mark.parametrize("observe", [False, True], ids=["flag off", "flag on"])
class TestWhatATileTellsDependsOnSeeingIt:
    def test_control_a_visible_tile_tells_everything(self, observe: bool) -> None:
        tile = o.tile(_state(observe=observe), *SPECIAL)
        assert tile is not None
        assert tile["owner_id"] == 0
        assert tile["improvement"] == "IMPROVEMENT_FARM"
        assert tile["resource"] == "RESOURCE_WHEAT"
        assert tile["yields"]["food"] == 2
        assert (tile["is_visible"], tile["is_revealed"]) == (True, True)

    def test_a_tile_not_visible_now_has_no_owner_improvement_resource_or_yield(
        self, observe: bool
    ) -> None:
        tile = o.tile(_state("Civ6Stub.fogTile(1, 1)\n", observe=observe), *SPECIAL)
        assert tile is not None, "a revealed tile is still exported"
        for key in SEEN_ONLY_KEYS:
            assert key not in tile, key
        assert not tile.get("yields")
        assert (tile["is_visible"], tile["is_revealed"]) == (False, True)

    def test_the_terrain_and_features_of_a_remembered_tile_stay(self, observe: bool) -> None:
        seen = o.tile(_state(observe=observe), *SPECIAL)
        remembered = o.tile(_state("Civ6Stub.fogTile(1, 1)\n", observe=observe), *SPECIAL)
        assert seen is not None
        assert remembered is not None
        assert remembered["terrain"] == seen["terrain"]
        assert remembered["features"] == seen["features"] == ["FEATURE_FOREST", "FEATURE_HILLS"]

    def test_a_tile_never_seen_is_not_exported(self, observe: bool) -> None:
        state = _state("Civ6Stub.hideTile(1, 1)\n", observe=observe)
        assert o.tile(state, *SPECIAL) is None
        assert len(state["map"]["tiles"]) == 11

    def test_with_fog_not_asked_for_a_never_seen_tile_is_exported_and_tells_nothing(
        self, observe: bool
    ) -> None:
        state = _state("Civ6Stub.hideTile(1, 1)\n", observe=observe, fog=False)
        tile = o.tile(state, *SPECIAL)
        assert tile is not None
        assert (tile["is_visible"], tile["is_revealed"]) == (False, False)
        for key in SEEN_ONLY_KEYS:
            assert key not in tile
        assert not tile.get("yields")


@pytest.mark.parametrize("observe", [False, True], ids=["flag off", "flag on"])
class TestWhenTheGameCannotSay:
    # No table, or none for this player: the game cannot be asked about any tile.
    LOST = ["Civ6Stub.removeVisibility()", "Civ6Stub.removeVisibilityEntry()"]
    IDS = ["no table", "no entry"]

    @pytest.mark.parametrize("lost", LOST, ids=IDS)
    def test_with_fog_on_no_tile_is_exported(self, observe: bool, lost: str) -> None:
        assert _state(lost + "\n", observe=observe)["map"]["tiles"] == []

    @pytest.mark.parametrize("lost", LOST, ids=IDS)
    def test_with_fog_off_every_tile_is_unseen_and_tells_nothing(
        self, observe: bool, lost: str
    ) -> None:
        state = _state(lost + "\n", observe=observe, fog=False)
        tiles = state["map"]["tiles"]
        assert len(tiles) == 12, "premise: fog off exports every tile"
        for tile in tiles:
            assert (tile["is_visible"], tile["is_revealed"]) == (False, False), tile
            for key in SEEN_ONLY_KEYS:
                assert key not in tile, (key, tile)
            assert not tile.get("yields")
        # Terrain is still told.
        assert all(tile["terrain"].startswith("TERRAIN_") for tile in tiles)

    # One method missing: the contract's wording ("IsVisible/IsRevealed missing: every tile is
    # not visible and not revealed") is ambiguous about the other method's answer, so only what
    # holds on either reading is asserted.
    def test_without_isvisible_no_tile_is_seen_and_none_tells_what_only_seeing_gives(
        self, observe: bool
    ) -> None:
        for fog in (True, False):
            state = _state(
                "Civ6Stub.removeVisibilityMethod('IsVisible')\n", observe=observe, fog=fog
            )
            for tile in state["map"]["tiles"]:
                assert tile["is_visible"] is False, tile
                for key in SEEN_ONLY_KEYS:
                    assert key not in tile, (key, tile)
                assert not tile.get("yields")

    def test_without_isrevealed_no_tile_is_revealed_and_with_fog_on_none_is_exported(
        self, observe: bool
    ) -> None:
        setup = "Civ6Stub.removeVisibilityMethod('IsRevealed')\n"
        assert _state(setup, observe=observe)["map"]["tiles"] == []
        for tile in _state(setup, observe=observe, fog=False)["map"]["tiles"]:
            assert tile["is_revealed"] is False, tile
