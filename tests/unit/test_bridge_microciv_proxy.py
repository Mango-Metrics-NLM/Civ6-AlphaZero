"""The simulator proxy: what MicroCiv stands in for a Civ6 unit or tile (GM-B5, Part B).

``BridgeMCTSEnv`` seeds a ``MicroCivEnv`` from a converted bridge state. MicroCiv knows three
unit classes and refuses only the terrain ``mountain``; the converter speaks Civ6
(``civilian``, ``melee``, ``grassland`` with a ``mountains`` feature, ``ocean``). The module
``alpha_zero.bridge.microciv_proxy`` is the one place that says what stands for what, and the
tests below hold it to a written table, to the simulator it targets, and (through the real
``StateConverter``, un-mocked) to what ``BridgeMCTSEnv`` then does with it.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from alpha_zero.bridge.mcts_policy import BridgeMCTSEnv
from alpha_zero.bridge.state_converter import StateConverter, UnitMapping
from alpha_zero.core.civ6_mcts import GumbelMCTS, GumbelMCTSConfig
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.envs.microciv import MicroCivEnv, Tile
from tests.fixtures.bridge_search_states import make_state, tile, unit

ROOT = Path(__file__).resolve().parents[2]
SIM_ACTIONS = 376

#: A far enemy warrior, so that the simulated game does not end on the first step.
ENEMY = (14, 14)


@pytest.fixture(scope="module")
def proxy() -> ModuleType:
    return importlib.import_module("alpha_zero.bridge.microciv_proxy")


# ---------------------------------------------------------------------------
# simulator_unit_class
# ---------------------------------------------------------------------------

UNIT_CLASS_TABLE: list[tuple[Any, Any, str | None]] = [
    # (unit_type, raw_type, expected)
    # rule 1: a settler by its raw type, whatever the category says
    ("civilian", "UNIT_SETTLER", "settler"),
    ("melee", "UNIT_SETTLER", "settler"),
    (None, "UNIT_SETTLER", "settler"),
    ("", "UNIT_SETTLER", "settler"),
    (7, "UNIT_SETTLER", "settler"),
    ("civilian", "unit_settler", "settler"),
    ("civilian", "  UNIT_SETTLER  ", "settler"),
    ("civilian", "\tUnit_Settler\n", "settler"),
    # rule 2: by category, case and surrounding whitespace ignored
    ("warrior", None, "warrior"),
    ("settler", None, "settler"),
    ("archer", None, "archer"),
    ("melee", None, "warrior"),
    ("cavalry", None, "warrior"),
    ("ranged", None, "archer"),
    ("siege", None, "archer"),
    ("MELEE", None, "warrior"),
    ("  Ranged ", None, "archer"),
    ("Warrior", None, "warrior"),
    ("SETTLER", None, "settler"),
    ("Cavalry\n", None, "warrior"),
    ("melee", "UNIT_WARRIOR", "warrior"),
    ("ranged", "UNIT_ARCHER", "archer"),
    # a raw type that is not a settler does not override the category
    ("melee", "UNIT_BUILDER", "warrior"),
    ("melee", 5, "warrior"),
    ("melee", ["UNIT_SETTLER"], "warrior"),
    # rule 3: nothing stands for it
    ("civilian", None, None),
    ("civilian", "UNIT_BUILDER", None),
    ("civilian", "UNIT_TRADER", None),
    ("civilian", "UNIT_SCOUT", None),
    ("civilian", "UNIT_SETTLER_2", None),
    ("civilian", "SETTLER", None),
    ("civilian", 5, None),
    ("civilian", ["UNIT_SETTLER"], None),
    ("civilian", b"UNIT_SETTLER", None),
    ("naval_melee", None, None),
    ("naval_ranged", None, None),
    ("air", None, None),
    ("support", None, None),
    ("religious", None, None),
    ("great_person", None, None),
    ("bogus", None, None),
    ("", None, None),
    ("   ", None, None),
    (None, None, None),
    (5, None, None),
    (3.5, None, None),
    (True, None, None),
    (["warrior"], None, None),
    (("melee",), None, None),
    ({"warrior": 1}, None, None),
    (b"warrior", None, None),
    (object(), None, None),
]


class TestSimulatorUnitClass:
    @pytest.mark.parametrize(("unit_type", "raw_type", "expected"), UNIT_CLASS_TABLE)
    def test_table(
        self, proxy: ModuleType, unit_type: Any, raw_type: Any, expected: str | None
    ) -> None:
        assert proxy.simulator_unit_class(unit_type, raw_type) == expected

    def test_raw_type_is_optional(self, proxy: ModuleType) -> None:
        assert proxy.simulator_unit_class("melee") == "warrior"
        assert proxy.simulator_unit_class("civilian") is None

    @pytest.mark.parametrize(("unit_type", "raw_type", "expected"), UNIT_CLASS_TABLE)
    def test_answer_is_a_class_or_none(
        self, proxy: ModuleType, unit_type: Any, raw_type: Any, expected: str | None
    ) -> None:
        answer = proxy.simulator_unit_class(unit_type, raw_type)
        assert answer is None or answer in proxy.SIMULATOR_UNIT_CLASSES

    def test_every_real_civ6_unit_type_through_the_converter_table(self, proxy: ModuleType) -> None:
        """Each Civ6 type the converter knows, with the category the converter gives it."""
        by_category = {
            "melee": "warrior",
            "cavalry": "warrior",
            "ranged": "archer",
            "siege": "archer",
        }
        seen_categories: set[str] = set()
        for civ6_type, category in UnitMapping().UNITS.items():
            seen_categories.add(category)
            expected = "settler" if civ6_type == "UNIT_SETTLER" else by_category.get(category)
            assert proxy.simulator_unit_class(category, civ6_type) == expected, civ6_type
        # The table above names every category the converter can produce.
        assert seen_categories == {
            "melee",
            "ranged",
            "cavalry",
            "siege",
            "naval_melee",
            "naval_ranged",
            "air",
            "support",
            "civilian",
            "religious",
            "great_person",
        }


# ---------------------------------------------------------------------------
# simulator_terrain
# ---------------------------------------------------------------------------

TERRAIN_TABLE: list[tuple[dict[str, Any], str]] = [
    # rule 1: the only terrain MicroCiv refuses
    ({"type": "mountain", "features": []}, "mountain"),
    ({"type": "mountains", "features": []}, "mountain"),
    ({"type": "coast", "features": []}, "mountain"),
    ({"type": "ocean", "features": []}, "mountain"),
    ({"type": "Mountains", "features": []}, "mountain"),
    ({"type": "OCEAN", "features": []}, "mountain"),
    ({"type": "  Coast ", "features": []}, "mountain"),
    ({"type": "grassland", "features": ["mountain"]}, "mountain"),
    ({"type": "grassland", "features": ["mountains"]}, "mountain"),
    ({"type": "grassland", "features": ["ice"]}, "mountain"),
    ({"type": "plains", "features": ["Ice"]}, "mountain"),
    ({"type": "snow", "features": ["MOUNTAINS"]}, "mountain"),
    ({"type": "plains", "features": ["forest", "ice"]}, "mountain"),
    ({"type": "plains", "features": ["hills", "mountains"]}, "mountain"),
    # mountain wins over hills
    ({"type": "hills", "features": ["mountains"]}, "mountain"),
    ({"type": "hills", "features": ["ice"]}, "mountain"),
    ({"type": "ocean", "features": ["hills"]}, "mountain"),
    ({"type": "mountains"}, "mountain"),
    ({"type": "ocean", "features": None}, "mountain"),
    # rule 2: hills
    ({"type": "hills", "features": []}, "hills"),
    ({"type": "Hills", "features": []}, "hills"),
    ({"type": "grassland", "features": ["hills"]}, "hills"),
    ({"type": "desert", "features": ["forest", "hills"]}, "hills"),
    ({"type": "tundra", "features": ["HILLS"]}, "hills"),
    # rule 3: everything else
    ({"type": "plains", "features": []}, "plains"),
    ({"type": "grassland", "features": []}, "plains"),
    ({"type": "desert", "features": []}, "plains"),
    ({"type": "tundra", "features": []}, "plains"),
    ({"type": "snow", "features": []}, "plains"),
    ({"type": "grassland", "features": ["forest"]}, "plains"),
    ({"type": "plains", "features": ["jungle", "marsh", "floodplains", "river"]}, "plains"),
    ({"type": "plains"}, "plains"),
    ({"features": ["forest"]}, "plains"),
    ({"type": "", "features": []}, "plains"),
    ({"type": "bogus", "features": []}, "plains"),
    ({}, "plains"),
    # unreadable is plains
    ({"type": None, "features": None}, "plains"),
    ({"type": 5, "features": 3}, "plains"),
    ({"type": ["ocean"], "features": []}, "plains"),
    ({"type": "plains", "features": [None, 5]}, "plains"),
]


class TestSimulatorTerrain:
    @pytest.mark.parametrize(("tile_dict", "expected"), TERRAIN_TABLE)
    def test_table(self, proxy: ModuleType, tile_dict: dict[str, Any], expected: str) -> None:
        assert proxy.simulator_terrain(tile_dict) == expected

    @pytest.mark.parametrize(
        "unreadable",
        [None, 5, "ocean", 3.5, ["ocean"], object(), b"ocean", SimpleNamespace(type="ocean")],
    )
    def test_non_dict_is_plains(self, proxy: ModuleType, unreadable: Any) -> None:
        assert proxy.simulator_terrain(unreadable) == "plains"

    @pytest.mark.parametrize(
        "mixed", [{"type": "plains", "features": [None, 5, "ice"]}, {"features": [object()]}]
    )
    def test_mixed_feature_lists_do_not_raise(self, proxy: ModuleType, mixed: Any) -> None:
        assert proxy.simulator_terrain(mixed) in proxy.SIMULATOR_TERRAIN

    @pytest.mark.parametrize(("tile_dict", "expected"), TERRAIN_TABLE)
    def test_answer_is_a_simulator_terrain(
        self, proxy: ModuleType, tile_dict: dict[str, Any], expected: str
    ) -> None:
        assert proxy.simulator_terrain(tile_dict) in proxy.SIMULATOR_TERRAIN

    def test_input_is_not_mutated(self, proxy: ModuleType) -> None:
        tile_dict = {"type": "Ocean", "features": ["Ice", "hills"], "yields": {"food": 1}}
        before = json.dumps(tile_dict, sort_keys=True)
        proxy.simulator_terrain(tile_dict)
        assert json.dumps(tile_dict, sort_keys=True) == before


# ---------------------------------------------------------------------------
# raw_unit_types
# ---------------------------------------------------------------------------


class TestRawUnitTypes:
    def test_keys_are_owner_and_id(self, proxy: ModuleType) -> None:
        state = make_state(
            units=[
                unit(5, "UNIT_SETTLER", 0, 1, 1),
                unit(5, "UNIT_BUILDER", 1, 2, 2),
                unit(6, "UNIT_WARRIOR", 0, 3, 3),
            ]
        )
        assert proxy.raw_unit_types({"_raw_state": state}) == {
            (0, 5): "UNIT_SETTLER",
            (1, 5): "UNIT_BUILDER",
            (0, 6): "UNIT_WARRIOR",
        }

    def test_values_are_returned_unchanged(self, proxy: ModuleType) -> None:
        state = make_state(units=[unit(1, "unit_settler", 0, 1, 1)])
        assert proxy.raw_unit_types({"_raw_state": state}) == {(0, 1): "unit_settler"}

    def test_units_without_a_string_type_are_left_out(self, proxy: ModuleType) -> None:
        good = unit(1, "UNIT_WARRIOR", 0, 1, 1)
        empty = unit(2, "", 0, 2, 2)
        none_type = unit(3, "UNIT_WARRIOR", 0, 3, 3)
        none_type.unit_type = None  # type: ignore[assignment]
        number = unit(4, "UNIT_WARRIOR", 0, 4, 4)
        number.unit_type = 7  # type: ignore[assignment]
        listed = unit(5, "UNIT_WARRIOR", 0, 5, 5)
        listed.unit_type = ["UNIT_SETTLER"]  # type: ignore[assignment]
        state = make_state(units=[good, empty, none_type, number, listed])
        assert proxy.raw_unit_types({"_raw_state": state}) == {(0, 1): "UNIT_WARRIOR"}

    def test_no_units(self, proxy: ModuleType) -> None:
        assert proxy.raw_unit_types({"_raw_state": make_state(units=[])}) == {}

    @pytest.mark.parametrize(
        "converted",
        [
            {},
            {"_raw_state": None},
            {"_raw_state": 5},
            {"_raw_state": "state"},
            {"_raw_state": object()},
            {"_raw_state": SimpleNamespace()},
            {"_raw_state": SimpleNamespace(units=None)},
            {"_raw_state": SimpleNamespace(units=5)},
            {"_raw_state": SimpleNamespace(units=[object()])},
            {"_raw_state": SimpleNamespace(units=[SimpleNamespace(unit_type="UNIT_WARRIOR")])},
            None,
            5,
            "converted",
            [],
            object(),
        ],
    )
    def test_unreadable_is_empty(self, proxy: ModuleType, converted: Any) -> None:
        assert proxy.raw_unit_types(converted) == {}

    def test_raising_state_is_empty(self, proxy: ModuleType) -> None:
        class Broken:
            @property
            def units(self) -> Any:
                raise RuntimeError("no units for you")

        assert proxy.raw_unit_types({"_raw_state": Broken()}) == {}


# ---------------------------------------------------------------------------
# agreement with the simulator
# ---------------------------------------------------------------------------


class TestAgreementWithTheSimulator:
    def test_constants(self, proxy: ModuleType) -> None:
        assert isinstance(proxy.SIMULATOR_UNIT_CLASSES, frozenset)
        assert isinstance(proxy.SIMULATOR_TERRAIN, frozenset)
        assert proxy.SIMULATOR_UNIT_CLASSES == frozenset({"warrior", "settler", "archer"})
        assert proxy.SIMULATOR_TERRAIN == frozenset({"plains", "hills", "mountain"})

    def test_unit_classes_are_simulator_unit_types(self, proxy: ModuleType) -> None:
        assert proxy.SIMULATOR_UNIT_CLASSES <= set(MicroCivEnv.UNIT_MOVES)

    def test_terrain_is_what_map_generation_produces(self, proxy: ModuleType) -> None:
        produced: set[str] = set()
        for seed in range(4):
            env = MicroCivEnv(board_size=16, seed=seed)
            env.reset()
            produced.update(t.terrain_type for t in env.terrain.values())
        assert proxy.SIMULATOR_TERRAIN <= produced

    def test_only_mountain_is_refused_by_the_simulator(self, proxy: ModuleType) -> None:
        env = MicroCivEnv(board_size=4)
        env.units = []
        for terrain_type in sorted(proxy.SIMULATOR_TERRAIN):
            env.terrain = {(1, 1): Tile(terrain_type=terrain_type)}
            assert env._is_valid_move((1, 1)) is (terrain_type != "mountain"), terrain_type
        # ... which is why Civ6's ocean has to be handed over as a mountain.
        env.terrain = {(1, 1): Tile(terrain_type="ocean")}
        assert env._is_valid_move((1, 1)) is True

    def test_import_needs_only_the_standard_library(self) -> None:
        code = (
            "import json, sys\n"
            "import alpha_zero.bridge.microciv_proxy as m\n"
            "heavy = [n for n in ('torch', 'numpy', 'gymnasium') if n in sys.modules]\n"
            "out = {'heavy': heavy,\n"
            "       'unit': m.simulator_unit_class('melee'),\n"
            "       'terrain': m.simulator_terrain({'type': 'ocean', 'features': []}),\n"
            "       'raw': {str(k): v for k, v in m.raw_unit_types({}).items()}}\n"
            "print(json.dumps(out))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert json.loads(result.stdout.strip().splitlines()[-1]) == {
            "heavy": [],
            "unit": "warrior",
            "terrain": "mountain",
            "raw": {},
        }


# ---------------------------------------------------------------------------
# BridgeMCTSEnv seeds through the proxy, from a real converted state
# ---------------------------------------------------------------------------


def _seed(state: Any, *, size: int = 16) -> tuple[BridgeMCTSEnv, dict[str, Any]]:
    converter = StateConverter(board_size=size, research_registry=False)
    converted = converter.convert(state)
    return BridgeMCTSEnv.from_bridge_state(converted), converted


def _move_index(env: BridgeMCTSEnv, unit_idx: int, dx: int, dy: int) -> int:
    return env.env.action_space_handler.encode_action(
        {"type": "move", "unit_idx": unit_idx, "direction": (dx, dy)}
    )


def _sim_units(env: BridgeMCTSEnv) -> dict[int, str]:
    return {u.id: u.unit_type for u in env.env.units}


class TestSeedingUnits:
    @pytest.mark.parametrize(
        ("civ6_type", "expected"),
        [
            ("UNIT_WARRIOR", "warrior"),
            ("UNIT_SWORDSMAN", "warrior"),
            ("UNIT_HORSEMAN", "warrior"),
            ("UNIT_ARCHER", "archer"),
            ("UNIT_CATAPULT", "archer"),
            ("UNIT_SETTLER", "settler"),
            ("UNIT_BUILDER", None),
            ("UNIT_TRADER", None),
            ("UNIT_SCOUT", None),
            ("UNIT_GALLEY", None),
            ("UNIT_FIGHTER", None),
            ("UNIT_MEDIC", None),
            ("UNIT_MISSIONARY", None),
            ("UNIT_GREAT_GENERAL", None),
        ],
    )
    def test_class_of_each_civ6_unit(self, civ6_type: str, expected: str | None) -> None:
        state = make_state(units=[unit(7, civ6_type, 0, 5, 5), unit(3, "UNIT_WARRIOR", 1, *ENEMY)])
        env, _ = _seed(state)
        if expected is None:
            assert 7 not in _sim_units(env)
        else:
            assert _sim_units(env)[7] == expected
        assert _sim_units(env)[3] == "warrior"

    def test_unit_fields_are_kept(self) -> None:
        state = make_state(
            units=[
                unit(9, "UNIT_ARCHER", 0, 5, 6, moves=1.0, health=50, strength=17),
                unit(3, "UNIT_WARRIOR", 1, *ENEMY),
            ]
        )
        env, converted = _seed(state)
        sim = next(u for u in env.env.units if u.id == 9)
        assert sim.unit_type == "archer"
        assert sim.position == (5, 6)
        assert sim.player == converted["current_player"] == 0
        assert sim.health == 50
        assert sim.strength == 17
        assert sim.moves_remaining == 1.0

    def test_lookup_of_the_raw_type_is_by_owner_and_id(self) -> None:
        """Two units with one id and different owners are two units."""
        state = make_state(
            units=[
                unit(5, "UNIT_SETTLER", 0, 5, 5),
                unit(5, "UNIT_BUILDER", 1, *ENEMY),
                unit(6, "UNIT_WARRIOR", 1, 12, 12),
            ]
        )
        env, _ = _seed(state)
        by_player = {(u.player, u.id): u.unit_type for u in env.env.units}
        assert by_player == {(0, 5): "settler", (1, 6): "warrior"}

    def test_unrepresented_units_are_counted_not_seeded(self) -> None:
        state = make_state(
            units=[
                unit(8, "UNIT_BUILDER", 0, 4, 4),
                unit(9, "UNIT_WARRIOR", 0, 5, 5),
                unit(3, "UNIT_WARRIOR", 1, *ENEMY),
                unit(4, "UNIT_TRADER", 1, 13, 13),
            ]
        )
        env, converted = _seed(state)
        assert _sim_units(env) == {9: "warrior", 3: "warrior"}
        assert env.proxy_stats == {"units": 2, "units_not_represented": 2, "impassable_tiles": 0}
        assert env.proxy_stats["units"] + env.proxy_stats["units_not_represented"] == len(
            converted["units"]
        )
        # The simulator's own first unit is the warrior, not the builder before it.
        mask = env.get_valid_actions()
        assert mask[_move_index(env, 0, 1, 0)] == 1.0  # (5,5) -> (6,5), the warrior's

    def test_stats_for_an_empty_state(self) -> None:
        env, _ = _seed(make_state())
        assert env.proxy_stats == {"units": 0, "units_not_represented": 0, "impassable_tiles": 0}
        assert set(env.proxy_stats) == {"units", "units_not_represented", "impassable_tiles"}

    def test_hand_made_state_without_a_raw_state(self) -> None:
        """A converted dict that has no ``_raw_state`` is judged by its category alone."""
        converted: dict[str, Any] = {
            "turn": 1,
            "current_player": 0,
            "_map_size": (8, 8),
            "terrain": {},
            "units": [
                {"_id": 1, "player": 0, "type": "melee", "position": (0, 0), "moves_remaining": 2},
                {"_id": 2, "player": 0, "type": "civilian", "position": (1, 1)},
                {"_id": 3, "player": 1, "type": "ranged", "position": (6, 6)},
                {"_id": 4, "player": 1, "type": "naval_melee", "position": (7, 7)},
            ],
            "cities": [],
        }
        env = BridgeMCTSEnv.from_bridge_state(converted)
        assert _sim_units(env) == {1: "warrior", 3: "archer"}
        assert env.proxy_stats == {"units": 2, "units_not_represented": 2, "impassable_tiles": 0}


class TestSeedingTerrain:
    #: The tile the warrior at (5, 5) looks at, east of it, and the one it can use, west.
    EAST = (6, 5)
    WEST = (4, 5)

    def _env(self, terrain: str, features: tuple[str, ...] = ()) -> BridgeMCTSEnv:
        state = make_state(
            units=[unit(2, "UNIT_WARRIOR", 0, 5, 5), unit(3, "UNIT_WARRIOR", 1, *ENEMY)],
            tiles=[
                tile(*self.EAST, terrain, features),
                tile(*self.WEST, "TERRAIN_GRASS"),
            ],
        )
        env, _ = _seed(state)
        return env

    @pytest.mark.parametrize(
        ("terrain", "features"),
        [
            ("TERRAIN_GRASS_MOUNTAIN", ()),
            ("TERRAIN_PLAINS_MOUNTAIN", ()),
            ("TERRAIN_DESERT_MOUNTAIN", ()),
            ("TERRAIN_SNOW_MOUNTAIN", ()),
            ("TERRAIN_OCEAN", ()),
            ("TERRAIN_COAST", ()),
            ("TERRAIN_GRASS", ("FEATURE_ICE",)),
            ("TERRAIN_SNOW", ("FEATURE_ICE",)),
        ],
    )
    def test_impassable_tiles_cannot_be_entered(
        self, terrain: str, features: tuple[str, ...]
    ) -> None:
        env = self._env(terrain, features)
        mask = env.get_valid_actions()
        assert env.env.terrain[self.EAST].terrain_type == "mountain"
        assert mask[_move_index(env, 0, 1, 0)] == 0.0, "a move onto the impassable tile"
        assert mask[_move_index(env, 0, -1, 0)] == 1.0, "the neighbouring grass tile is open"
        assert env.proxy_stats["impassable_tiles"] == 1

    @pytest.mark.parametrize(
        ("terrain", "features", "expected"),
        [
            ("TERRAIN_GRASS", (), "plains"),
            ("TERRAIN_PLAINS", (), "plains"),
            ("TERRAIN_DESERT", (), "plains"),
            ("TERRAIN_TUNDRA", (), "plains"),
            ("TERRAIN_SNOW", (), "plains"),
            ("TERRAIN_GRASS", ("FEATURE_FOREST",), "plains"),
            ("TERRAIN_GRASS_HILLS", (), "hills"),
            ("TERRAIN_PLAINS_HILLS", (), "hills"),
            ("TERRAIN_DESERT_HILLS", (), "hills"),
        ],
    )
    def test_passable_tiles_can_be_entered(
        self, terrain: str, features: tuple[str, ...], expected: str
    ) -> None:
        env = self._env(terrain, features)
        mask = env.get_valid_actions()
        assert env.env.terrain[self.EAST].terrain_type == expected
        assert mask[_move_index(env, 0, 1, 0)] == 1.0
        assert env.proxy_stats["impassable_tiles"] == 0

    def test_every_tile_is_a_simulator_terrain(self, proxy: ModuleType) -> None:
        state = make_state(
            units=[unit(2, "UNIT_WARRIOR", 0, 5, 5), unit(3, "UNIT_WARRIOR", 1, *ENEMY)],
            tiles=[
                tile(1, 1, "TERRAIN_OCEAN"),
                tile(2, 2, "TERRAIN_COAST"),
                tile(3, 3, "TERRAIN_GRASS_MOUNTAIN"),
                tile(4, 4, "TERRAIN_GRASS_HILLS"),
                tile(6, 6, "TERRAIN_DESERT"),
                tile(7, 7, "TERRAIN_GRASS", ("FEATURE_ICE",)),
            ],
        )
        env, _ = _seed(state)
        kinds = {t.terrain_type for t in env.env.terrain.values()}
        assert kinds == {"plains", "hills", "mountain"}
        assert kinds <= proxy.SIMULATOR_TERRAIN
        assert env.proxy_stats["impassable_tiles"] == 4  # ocean, coast, mountain, ice

    def test_features_and_yields_are_kept(self) -> None:
        state = make_state(
            units=[unit(2, "UNIT_WARRIOR", 0, 5, 5), unit(3, "UNIT_WARRIOR", 1, *ENEMY)],
            tiles=[tile(6, 5, "TERRAIN_GRASS_HILLS", ("FEATURE_FOREST",), {"food": 1, "gold": 4})],
        )
        env, converted = _seed(state)
        seeded = env.env.terrain[(6, 5)]
        assert seeded.terrain_type == "hills"
        assert seeded.features == converted["terrain"][(6, 5)]["features"]
        assert "forest" in seeded.features
        assert seeded.yields == {"food": 1, "gold": 4}

    def test_a_tile_without_a_type_is_plains(self) -> None:
        converted: dict[str, Any] = {
            "turn": 0,
            "current_player": 0,
            "_map_size": (8, 8),
            "terrain": {(0, 0): {}, (1, 1): {"features": ["ice"]}},
            "units": [],
            "cities": [],
        }
        env = BridgeMCTSEnv.from_bridge_state(converted)
        assert env.env.terrain[(0, 0)].terrain_type == "plains"
        assert env.env.terrain[(1, 1)].terrain_type == "mountain"
        assert env.proxy_stats["impassable_tiles"] == 1


class TestSettlerFoundsACity:
    def _settler_env(self) -> tuple[BridgeMCTSEnv, dict[str, Any]]:
        state = make_state(
            units=[unit(7, "UNIT_SETTLER", 0, 5, 5), unit(3, "UNIT_WARRIOR", 1, *ENEMY)]
        )
        return _seed(state)

    def test_found_city_is_legal_for_the_settler(self) -> None:
        env, _ = self._settler_env()
        found = env.env.action_space_handler.found_city_offset + 0
        assert env.get_valid_actions()[found] == 1.0
        assert env.proxy_stats["units"] == 2

    def test_founding_puts_a_city_where_the_settler_stood(self) -> None:
        env, converted = self._settler_env()
        assert env.env.cities == []
        found = env.env.action_space_handler.found_city_offset + 0

        _obs, _reward, _terminated, _truncated, info = env.step(found)

        assert info["action_success"] is True
        assert len(env.env.cities) == 1
        city = env.env.cities[0]
        assert city.position == (5, 5)
        assert city.player == converted["current_player"] == 0
        assert 7 not in _sim_units(env), "the settler is consumed"

    def test_a_builder_cannot_found(self) -> None:
        state = make_state(
            units=[unit(7, "UNIT_BUILDER", 0, 5, 5), unit(3, "UNIT_WARRIOR", 1, *ENEMY)]
        )
        env, _ = _seed(state)
        found = env.env.action_space_handler.found_city_offset + 0
        assert env.get_valid_actions()[found] == 0.0
        assert env.proxy_stats == {"units": 1, "units_not_represented": 1, "impassable_tiles": 0}


class TestClone:
    def _env(self) -> BridgeMCTSEnv:
        state = make_state(
            units=[
                unit(7, "UNIT_SETTLER", 0, 5, 5),
                unit(2, "UNIT_BUILDER", 0, 4, 4),
                unit(3, "UNIT_WARRIOR", 1, *ENEMY),
            ],
            tiles=[tile(6, 5, "TERRAIN_OCEAN")],
        )
        env, _ = _seed(state)
        return env

    def test_clone_keeps_the_stats_and_the_simulator_apart(self) -> None:
        env = self._env()
        cloned = env.clone()

        assert (
            cloned.proxy_stats
            == env.proxy_stats
            == {
                "units": 2,
                "units_not_represented": 1,
                "impassable_tiles": 1,
            }
        )
        assert cloned.env is not env.env
        assert _sim_units(cloned) == _sim_units(env)

        found = env.env.action_space_handler.found_city_offset + 0
        cloned.step(found)
        assert len(cloned.env.cities) == 1
        assert env.env.cities == []
        assert 7 in _sim_units(env)
        assert 7 not in _sim_units(cloned)

    def test_clone_does_not_seed_again(self, proxy: ModuleType, monkeypatch: Any) -> None:
        """Seeding is paid for once; MCTS clones once per simulation."""
        from alpha_zero.bridge import mcts_policy

        calls = {"n": 0}

        def counting(original: Any) -> Any:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                calls["n"] += 1
                return original(*args, **kwargs)

            return wrapper

        for name in ("simulator_unit_class", "simulator_terrain", "raw_unit_types"):
            for module in (proxy, mcts_policy):
                original = getattr(module, name, None)
                if original is not None:
                    monkeypatch.setattr(module, name, counting(original))

        env = self._env()
        seeded_calls = calls["n"]
        assert seeded_calls > 0, "construction does not go through the proxy functions"

        env.clone().clone()
        assert calls["n"] == seeded_calls


# ---------------------------------------------------------------------------
# inside the search
# ---------------------------------------------------------------------------


def _search(env: BridgeMCTSEnv, *, size: int, k: int = 8, sims: int = 8) -> np.ndarray:
    network = MicroCivNetwork(board_size=size, action_space_size=SIM_ACTIONS)
    search = GumbelMCTS(
        network=network,
        config=GumbelMCTSConfig(num_simulations=sims, num_sampled_actions=k),
        device="cpu",
    )
    policy, _value = search.search(env)
    return np.asarray(policy)


class TestSearchHonoursTheProxy:
    def test_no_mass_on_a_move_onto_impassable_terrain(self) -> None:
        state = make_state(
            size=8,
            units=[unit(2, "UNIT_WARRIOR", 0, 3, 3), unit(3, "UNIT_WARRIOR", 1, 7, 7)],
            tiles=[
                tile(4, 3, "TERRAIN_GRASS_MOUNTAIN"),
                tile(3, 4, "TERRAIN_OCEAN"),
                tile(2, 3, "TERRAIN_COAST"),
                tile(4, 4, "TERRAIN_GRASS", ("FEATURE_ICE",)),
            ],
        )
        env, _ = _seed(state, size=8)
        blocked = [_move_index(env, 0, dx, dy) for dx, dy in ((1, 0), (0, 1), (-1, 0), (1, 1))]
        open_moves = [_move_index(env, 0, dx, dy) for dx, dy in ((-1, -1), (0, -1), (1, -1))]
        root_mask = env.get_valid_actions()
        assert [root_mask[i] for i in blocked] == [0.0] * 4
        assert [root_mask[i] for i in open_moves] == [1.0] * 3

        for _ in range(3):
            policy = _search(env, size=8)
            assert policy.shape == (SIM_ACTIONS,)
            assert float(policy.sum()) == pytest.approx(1.0, abs=1e-4)
            assert [float(policy[i]) for i in blocked] == [0.0] * 4
            assert float(policy[root_mask == 0].sum()) == 0.0

    def test_a_lone_settler_can_found_a_city_in_the_search(self) -> None:
        """Boxed in by water, rock and ice, the settler has three things to do."""
        state = make_state(
            size=8,
            units=[unit(7, "UNIT_SETTLER", 0, 0, 0), unit(3, "UNIT_WARRIOR", 1, 7, 7)],
            tiles=[
                tile(1, 0, "TERRAIN_OCEAN"),
                tile(0, 1, "TERRAIN_GRASS_MOUNTAIN"),
                tile(1, 1, "TERRAIN_GRASS", ("FEATURE_ICE",)),
            ],
        )
        env, _ = _seed(state, size=8)
        handler = env.env.action_space_handler
        found = handler.found_city_offset + 0
        root_mask = env.get_valid_actions()
        assert np.nonzero(root_mask)[0].tolist() == [
            handler.end_turn_offset,
            handler.skip_offset + 0,
            found,
        ]

        for _ in range(3):
            policy = _search(env, size=8, k=4, sims=8)
            assert float(policy[found]) > 0.0, "found-city is among the actions it can return"
            assert float(policy[root_mask == 0].sum()) == 0.0
