"""Additional tests for alpha_zero.gameplay.terrain to cover missing lines.

Targets uncovered lines:
- 113-124: Feature-based movement cost path (known features, impassable features, unknown features)
- 159-160: Unknown terrain type yields (returns zero dict)
- 165-170: Feature yield modifiers loop (applies additive modifiers, clamps to 0)
- 231:     move_range <= 0 edge case in BFS (returns only starting position)
- 246:     BFS bounds check (neighbours outside [0, board_size) are skipped)
"""

import pytest

from alpha_zero.gameplay.terrain import TerrainConfig, TerrainSystem

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def default_system() -> TerrainSystem:
    """TerrainSystem with default config for most tests."""
    return TerrainSystem()


@pytest.fixture
def custom_config() -> TerrainConfig:
    """Config with multiple features and terrain types for thorough testing."""
    return TerrainConfig(
        movement_costs={
            "plains": 1,
            "hills": 2,
            "mountain": -1,
            "forest": 2,
            "marsh": 3,
            "ice": -1,
        },
        terrain_yields={
            "plains": {"food": 2, "production": 1, "gold": 0},
            "hills": {"food": 1, "production": 2, "gold": 0},
            "mountain": {"food": 0, "production": 0, "gold": 0},
        },
        defense_bonuses={
            "hills": 0.25,
            "forest": 0.15,
            "mountain": 0.0,
        },
        feature_yield_modifiers={
            "forest": {"food": -1, "production": 1, "gold": 0},
            "floodplains": {"food": 3, "production": 0, "gold": 0},
        },
    )


@pytest.fixture
def custom_system(custom_config: TerrainConfig) -> TerrainSystem:
    return TerrainSystem(custom_config)


# ------------------------------------------------------------------
# Lines 113-124: Feature-based movement cost
# ------------------------------------------------------------------


class TestMovementCostWithFeatures:
    """Cover the feature iteration path in get_movement_cost (lines 113-124)."""

    def test_known_feature_adds_cost(self, default_system: TerrainSystem):
        """A known feature (forest, cost=2) should add its cost to the base terrain cost."""
        base_cost = default_system.config.movement_costs["plains"]
        feat_cost = default_system.config.movement_costs["forest"]
        result = default_system.get_movement_cost("plains", features=["forest"])
        assert result == base_cost + feat_cost

    def test_multiple_known_features_stack(self, custom_system: TerrainSystem):
        """Multiple known features should each add their cost."""
        base = custom_system.config.movement_costs["plains"]
        forest_cost = custom_system.config.movement_costs["forest"]
        marsh_cost = custom_system.config.movement_costs["marsh"]

        result = custom_system.get_movement_cost("plains", features=["forest", "marsh"])
        assert result == base + forest_cost + marsh_cost

    def test_impassable_feature_returns_minus_one(self, custom_system: TerrainSystem):
        """If any feature has cost -1 (impassable), total cost is -1 immediately."""
        result = custom_system.get_movement_cost("plains", features=["ice"])
        assert result == -1

    def test_impassable_feature_short_circuits(self, custom_system: TerrainSystem):
        """An impassable feature among several should still return -1."""
        result = custom_system.get_movement_cost("plains", features=["forest", "ice", "marsh"])
        assert result == -1

    def test_unknown_feature_adds_default_cost_of_one(self, default_system: TerrainSystem):
        """An unknown feature (not in movement_costs) should conservatively add 1."""
        base = default_system.config.movement_costs["plains"]
        result = default_system.get_movement_cost("plains", features=["totally_unknown"])
        assert result == base + 1

    def test_mix_of_known_and_unknown_features(self, default_system: TerrainSystem):
        """Known features use their configured cost; unknown features add 1 each."""
        base = default_system.config.movement_costs["plains"]
        forest_cost = default_system.config.movement_costs["forest"]
        # "jungle" is unknown -> +1
        result = default_system.get_movement_cost("plains", features=["forest", "jungle"])
        assert result == base + forest_cost + 1

    def test_multiple_unknown_features_each_add_one(self, default_system: TerrainSystem):
        """Each unknown feature independently adds 1."""
        unknowns = ["oasis", "reef", "kelp"]
        base = default_system.config.movement_costs["plains"]
        result = default_system.get_movement_cost("plains", features=unknowns)
        assert result == base + len(unknowns)

    def test_impassable_base_ignores_features(self, default_system: TerrainSystem):
        """If the base terrain is impassable, features are irrelevant."""
        result = default_system.get_movement_cost("mountain", features=["forest"])
        assert result == -1

    def test_empty_feature_list_same_as_no_features(self, default_system: TerrainSystem):
        """An empty features list should behave identically to features=None."""
        cost_none = default_system.get_movement_cost("plains", features=None)
        cost_empty = default_system.get_movement_cost("plains", features=[])
        assert cost_none == cost_empty


# ------------------------------------------------------------------
# Lines 159-160: Unknown terrain type for yields
# ------------------------------------------------------------------


class TestUnknownTerrainYields:
    """Cover the unknown terrain branch in get_yields (lines 159-160)."""

    def test_unknown_terrain_returns_zero_yields(self, default_system: TerrainSystem):
        """A terrain type not in terrain_yields should return all-zero yields."""
        result = default_system.get_yields("swamp")
        assert result == {"food": 0, "production": 0, "gold": 0}

    def test_unknown_terrain_yields_are_independent_of_config(self, custom_system: TerrainSystem):
        """The zero-yield fallback should contain the standard resource keys."""
        result = custom_system.get_yields("totally_made_up_terrain")
        for key in ("food", "production", "gold"):
            assert key in result
            assert result[key] == 0

    def test_unknown_terrain_with_features_still_applies_modifiers(
        self, default_system: TerrainSystem
    ):
        """Even on an unknown terrain, feature yield modifiers should be applied."""
        result = default_system.get_yields("swamp", features=["forest"])
        forest_mod = default_system.config.feature_yield_modifiers["forest"]
        # Base is all zeros; modifiers applied then clamped to >= 0
        for resource, delta in forest_mod.items():
            assert result[resource] == max(0, 0 + delta)


# ------------------------------------------------------------------
# Lines 165-170: Feature yield modifiers
# ------------------------------------------------------------------


class TestFeatureYieldModifiers:
    """Cover the feature yield modifier loop in get_yields (lines 165-170)."""

    def test_forest_modifier_on_plains(self, default_system: TerrainSystem):
        """Forest on plains should adjust yields by the configured modifier."""
        base = default_system.config.terrain_yields["plains"].copy()
        modifier = default_system.config.feature_yield_modifiers["forest"]
        result = default_system.get_yields("plains", features=["forest"])

        for resource in base:
            expected = max(0, base[resource] + modifier.get(resource, 0))
            assert result[resource] == expected, (
                f"Mismatch on {resource}: expected {expected}, got {result[resource]}"
            )

    def test_forest_modifier_on_hills(self, default_system: TerrainSystem):
        """Forest on hills should adjust yields correctly."""
        base = default_system.config.terrain_yields["hills"].copy()
        modifier = default_system.config.feature_yield_modifiers["forest"]
        result = default_system.get_yields("hills", features=["forest"])

        for resource in base:
            expected = max(0, base[resource] + modifier.get(resource, 0))
            assert result[resource] == expected

    def test_yield_never_goes_below_zero(self, default_system: TerrainSystem):
        """Yield modifier that would produce a negative value should be clamped to 0."""
        # Mountain has food=0, forest modifier has food=-1 -> should clamp to 0
        result = default_system.get_yields("mountain", features=["forest"])
        assert result["food"] >= 0

    def test_multiple_feature_modifiers_stack(self, custom_system: TerrainSystem):
        """Multiple features with modifiers should each be applied additively."""
        base = custom_system.config.terrain_yields["plains"].copy()
        forest_mod = custom_system.config.feature_yield_modifiers["forest"]
        flood_mod = custom_system.config.feature_yield_modifiers["floodplains"]

        result = custom_system.get_yields("plains", features=["forest", "floodplains"])

        for resource in base:
            expected = base[resource]
            expected += forest_mod.get(resource, 0)
            expected += flood_mod.get(resource, 0)
            expected = max(0, expected)
            assert result[resource] == expected

    def test_unknown_feature_does_not_modify_yields(self, default_system: TerrainSystem):
        """A feature not in feature_yield_modifiers should leave yields unchanged."""
        base_yields = default_system.get_yields("plains")
        modified_yields = default_system.get_yields("plains", features=["unknown_feature"])
        assert modified_yields == base_yields

    def test_feature_modifiers_do_not_mutate_config(self, default_system: TerrainSystem):
        """Calling get_yields with features must not alter the config's base yields."""
        original_plains = default_system.config.terrain_yields["plains"].copy()
        default_system.get_yields("plains", features=["forest"])
        assert default_system.config.terrain_yields["plains"] == original_plains

    def test_feature_modifier_introduces_new_resource_key(self):
        """A feature modifier with a resource key not in the base yields should add it."""
        config = TerrainConfig(
            terrain_yields={"desert": {"food": 0, "production": 0, "gold": 0}},
            feature_yield_modifiers={"oasis": {"food": 3, "faith": 1}},
        )
        system = TerrainSystem(config)
        result = system.get_yields("desert", features=["oasis"])

        assert result["food"] == 3
        assert result["faith"] == 1


# ------------------------------------------------------------------
# Line 231: move_range <= 0 edge case in BFS
# ------------------------------------------------------------------


class TestBFSZeroRange:
    """Cover the move_range <= 0 early return in get_tiles_in_range (line 231)."""

    def test_zero_range_returns_only_start(self, default_system: TerrainSystem):
        """With move_range=0, only the starting position should be returned."""
        start = (4, 4)
        result = default_system.get_tiles_in_range(start, move_range=0, terrain={}, board_size=8)
        assert result == {start}

    def test_negative_range_returns_only_start(self, default_system: TerrainSystem):
        """With move_range < 0, only the starting position should be returned."""
        start = (3, 3)
        result = default_system.get_tiles_in_range(start, move_range=-5, terrain={}, board_size=8)
        assert result == {start}

    def test_zero_range_at_corner(self, default_system: TerrainSystem):
        """Zero range at a corner tile should still return just that tile."""
        start = (0, 0)
        result = default_system.get_tiles_in_range(start, move_range=0, terrain={}, board_size=8)
        assert result == {start}

    def test_zero_range_result_is_set(self, default_system: TerrainSystem):
        """The return type for zero range should be a set containing a single tuple."""
        start = (5, 5)
        result = default_system.get_tiles_in_range(start, move_range=0, terrain={}, board_size=16)
        assert isinstance(result, set)
        assert len(result) == 1


# ------------------------------------------------------------------
# Line 246: BFS bounds check
# ------------------------------------------------------------------


class TestBFSBoundsCheck:
    """Cover the out-of-bounds continue in get_tiles_in_range (line 246)."""

    def test_top_left_corner_clips_neighbours(self, default_system: TerrainSystem):
        """Starting at (0,0) should not include tiles with negative coordinates."""
        result = default_system.get_tiles_in_range(
            position=(0, 0), move_range=1, terrain={}, board_size=8
        )
        # Only (0,0), (1,0), (0,1) should be reachable; no negative coords
        assert (0, 0) in result
        assert (1, 0) in result
        assert (0, 1) in result
        assert (-1, 0) not in result
        assert (0, -1) not in result
        assert len(result) == 3

    def test_bottom_right_corner_clips_neighbours(self, default_system: TerrainSystem):
        """Starting at (board_size-1, board_size-1) should not exceed the board."""
        board_size = 8
        pos = (board_size - 1, board_size - 1)
        result = default_system.get_tiles_in_range(
            position=pos, move_range=1, terrain={}, board_size=board_size
        )
        assert pos in result
        assert (board_size - 2, board_size - 1) in result
        assert (board_size - 1, board_size - 2) in result
        # These should be excluded (at or beyond board_size)
        assert (board_size, board_size - 1) not in result
        assert (board_size - 1, board_size) not in result
        assert len(result) == 3

    def test_edge_mid_clips_one_direction(self, default_system: TerrainSystem):
        """A position along the top edge (row=0) should not reach row=-1."""
        board_size = 8
        result = default_system.get_tiles_in_range(
            position=(0, 4), move_range=1, terrain={}, board_size=board_size
        )
        assert (0, 4) in result
        assert (0, 3) in result
        assert (0, 5) in result
        assert (1, 4) in result
        assert (-1, 4) not in result
        assert len(result) == 4

    def test_small_board_clips_extensively(self, default_system: TerrainSystem):
        """On a 1x1 board, no neighbours are valid regardless of move_range."""
        result = default_system.get_tiles_in_range(
            position=(0, 0), move_range=10, terrain={}, board_size=1
        )
        assert result == {(0, 0)}

    def test_all_reachable_within_bounds(self, default_system: TerrainSystem):
        """Every tile returned by BFS should be within [0, board_size)."""
        board_size = 6
        result = default_system.get_tiles_in_range(
            position=(0, 0), move_range=5, terrain={}, board_size=board_size
        )
        for r, c in result:
            assert 0 <= r < board_size, f"Row {r} out of bounds [0, {board_size})"
            assert 0 <= c < board_size, f"Col {c} out of bounds [0, {board_size})"
