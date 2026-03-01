"""Comprehensive tests for alpha_zero.multiplayer.fog_of_war.

Targets coverage of FogOfWarConfig, VisibilityState, and FogOfWarManager
from ~17% to 70%+. Tests cover:
  - FogOfWarConfig construction, defaults, and custom values
  - VisibilityState enum ordering
  - FogOfWarManager initialization and reset
  - Visibility computation: units, cities, sight ranges, edge clamping
  - VISIBLE -> SEEN demotion across turns
  - Observation masking: hiding/keeping units and cities
  - get_visibility / get_visibility_dict for known and unknown players
  - _reveal edge cases (invalid player, board-edge clamping)
  - Multi-player interleaved visibility
  - Disabled fog passthrough
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.multiplayer.fog_of_war import (
    FogOfWarConfig,
    FogOfWarManager,
    VisibilityState,
)

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture()
def config_enabled() -> FogOfWarConfig:
    """Fog-of-war enabled config with default sight ranges."""
    return FogOfWarConfig(enabled=True, default_sight_range=2, city_sight_range=3)


@pytest.fixture()
def config_disabled() -> FogOfWarConfig:
    """Fog-of-war disabled config."""
    return FogOfWarConfig(enabled=False)


@pytest.fixture()
def manager_8x8(config_enabled: FogOfWarConfig) -> FogOfWarManager:
    """8x8 board, 2-player manager with fog enabled."""
    return FogOfWarManager(config_enabled, board_size=8, num_players=2)


@pytest.fixture()
def manager_disabled(config_disabled: FogOfWarConfig) -> FogOfWarManager:
    """8x8 board, 2-player manager with fog disabled."""
    return FogOfWarManager(config_disabled, board_size=8, num_players=2)


@pytest.fixture()
def manager_16x16(config_enabled: FogOfWarConfig) -> FogOfWarManager:
    """16x16 board, 4-player manager with fog enabled."""
    return FogOfWarManager(config_enabled, board_size=16, num_players=4)


# =====================================================================
# FogOfWarConfig tests
# =====================================================================


class TestFogOfWarConfig:
    """Tests for the FogOfWarConfig dataclass."""

    def test_defaults(self) -> None:
        cfg = FogOfWarConfig()
        assert cfg.enabled is False
        assert cfg.default_sight_range == 2
        assert cfg.city_sight_range == 3
        assert cfg.unit_sight_ranges == {"warrior": 2, "settler": 2, "archer": 3}

    def test_enabled_flag(self) -> None:
        cfg = FogOfWarConfig(enabled=True)
        assert cfg.enabled is True

    def test_custom_sight_ranges(self) -> None:
        cfg = FogOfWarConfig(unit_sight_ranges={"scout": 5, "spy": 1})
        assert cfg.unit_sight_ranges["scout"] == 5
        assert cfg.unit_sight_ranges["spy"] == 1
        assert "warrior" not in cfg.unit_sight_ranges

    def test_frozen(self) -> None:
        cfg = FogOfWarConfig()
        with pytest.raises(AttributeError):
            cfg.enabled = True  # type: ignore[misc]

    def test_custom_default_sight_range(self) -> None:
        cfg = FogOfWarConfig(default_sight_range=5)
        assert cfg.default_sight_range == 5

    def test_custom_city_sight_range(self) -> None:
        cfg = FogOfWarConfig(city_sight_range=10)
        assert cfg.city_sight_range == 10


# =====================================================================
# VisibilityState tests
# =====================================================================


class TestVisibilityState:
    """Tests for the VisibilityState IntEnum."""

    def test_integer_values(self) -> None:
        assert int(VisibilityState.UNEXPLORED) == 0
        assert int(VisibilityState.SEEN) == 1
        assert int(VisibilityState.VISIBLE) == 2

    def test_ordering(self) -> None:
        assert VisibilityState.UNEXPLORED < VisibilityState.SEEN < VisibilityState.VISIBLE

    def test_comparison_with_int(self) -> None:
        assert VisibilityState.VISIBLE == 2
        assert VisibilityState.UNEXPLORED == 0


# =====================================================================
# FogOfWarManager — construction & reset
# =====================================================================


class TestFogOfWarManagerInit:
    """Tests for FogOfWarManager construction and reset."""

    def test_initial_grids_unexplored(self, manager_8x8: FogOfWarManager) -> None:
        """All tiles start as UNEXPLORED after construction."""
        for p in range(2):
            grid = manager_8x8.get_visibility(p)
            assert grid.shape == (8, 8)
            assert (grid == VisibilityState.UNEXPLORED).all()

    def test_board_size_stored(self, manager_8x8: FogOfWarManager) -> None:
        assert manager_8x8.board_size == 8

    def test_num_players_stored(self, manager_8x8: FogOfWarManager) -> None:
        assert manager_8x8.num_players == 2

    def test_reset_clears_all(self, manager_8x8: FogOfWarManager) -> None:
        """After computing visibility, reset brings everything back to UNEXPLORED."""
        units = [{"player": 0, "position": (3, 3), "type": "warrior"}]
        manager_8x8.compute_visibility(units, [])
        # Some tiles are now visible
        assert (manager_8x8.get_visibility(0) != VisibilityState.UNEXPLORED).any()

        manager_8x8.reset()
        for p in range(2):
            assert (manager_8x8.get_visibility(p) == VisibilityState.UNEXPLORED).all()

    def test_reset_with_new_player_count(self, manager_8x8: FogOfWarManager) -> None:
        manager_8x8.reset(num_players=5)
        assert manager_8x8.num_players == 5
        for p in range(5):
            grid = manager_8x8.get_visibility(p)
            assert grid.shape == (8, 8)
            assert (grid == VisibilityState.UNEXPLORED).all()

    def test_reset_without_num_players_keeps_count(self, manager_8x8: FogOfWarManager) -> None:
        manager_8x8.reset()
        assert manager_8x8.num_players == 2

    def test_four_player_construction(self, manager_16x16: FogOfWarManager) -> None:
        assert manager_16x16.num_players == 4
        assert manager_16x16.board_size == 16
        for p in range(4):
            grid = manager_16x16.get_visibility(p)
            assert grid.shape == (16, 16)


# =====================================================================
# FogOfWarManager — compute_visibility
# =====================================================================


class TestComputeVisibility:
    """Tests for compute_visibility with units and cities."""

    def test_single_unit_center(self, manager_8x8: FogOfWarManager) -> None:
        """A warrior at (4,4) with sight 2 reveals a Chebyshev-2 square."""
        units = [{"player": 0, "position": (4, 4), "type": "warrior"}]
        manager_8x8.compute_visibility(units, [])

        grid = manager_8x8.get_visibility(0)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y, x = 4 + dy, 4 + dx
                if 0 <= x < 8 and 0 <= y < 8:
                    assert grid[y, x] == VisibilityState.VISIBLE, f"({x},{y}) should be visible"

        # Just outside range
        assert grid[4, 7] == VisibilityState.UNEXPLORED

    def test_unit_at_corner(self, manager_8x8: FogOfWarManager) -> None:
        """Unit at (0,0) clips visibility to the board boundary."""
        units = [{"player": 0, "position": (0, 0), "type": "warrior"}]
        manager_8x8.compute_visibility(units, [])

        grid = manager_8x8.get_visibility(0)
        assert grid[0, 0] == VisibilityState.VISIBLE
        assert grid[2, 2] == VisibilityState.VISIBLE
        # Beyond range
        assert grid[3, 3] == VisibilityState.UNEXPLORED

    def test_unit_at_bottom_right_corner(self, manager_8x8: FogOfWarManager) -> None:
        """Unit at (7,7) clips correctly."""
        units = [{"player": 0, "position": (7, 7), "type": "warrior"}]
        manager_8x8.compute_visibility(units, [])

        grid = manager_8x8.get_visibility(0)
        assert grid[7, 7] == VisibilityState.VISIBLE
        assert grid[5, 5] == VisibilityState.VISIBLE
        assert grid[4, 4] == VisibilityState.UNEXPLORED

    def test_city_uses_city_sight_range(self, manager_8x8: FogOfWarManager) -> None:
        """A city with sight_range=3 reveals more than a warrior with sight_range=2."""
        cities = [{"player": 0, "position": (4, 4)}]
        manager_8x8.compute_visibility([], cities)

        grid = manager_8x8.get_visibility(0)
        # Distance 3 tiles should be visible
        assert grid[4, 7] == VisibilityState.VISIBLE  # dx=3
        assert grid[7, 4] == VisibilityState.VISIBLE  # dy=3
        assert grid[1, 1] == VisibilityState.VISIBLE  # dx=-3, dy=-3

    def test_archer_uses_configured_range(self, manager_8x8: FogOfWarManager) -> None:
        """Archer has sight range 3 from default config."""
        units = [{"player": 0, "position": (4, 4), "type": "archer"}]
        manager_8x8.compute_visibility(units, [])

        grid = manager_8x8.get_visibility(0)
        assert grid[4, 7] == VisibilityState.VISIBLE  # distance 3

    def test_unknown_unit_type_uses_default(self, manager_8x8: FogOfWarManager) -> None:
        """An unknown unit type falls back to default_sight_range."""
        units = [{"player": 0, "position": (4, 4), "type": "catapult"}]
        manager_8x8.compute_visibility(units, [])

        grid = manager_8x8.get_visibility(0)
        # default_sight_range=2
        assert grid[4, 6] == VisibilityState.VISIBLE  # distance 2
        assert grid[4, 7] == VisibilityState.UNEXPLORED  # distance 3

    def test_enemy_unit_does_not_reveal_for_other_player(
        self, manager_8x8: FogOfWarManager
    ) -> None:
        """Player 1's unit does not reveal tiles for player 0."""
        units = [{"player": 1, "position": (4, 4), "type": "warrior"}]
        manager_8x8.compute_visibility(units, [])

        grid_p0 = manager_8x8.get_visibility(0)
        grid_p1 = manager_8x8.get_visibility(1)
        assert (grid_p0 == VisibilityState.UNEXPLORED).all()
        assert grid_p1[4, 4] == VisibilityState.VISIBLE

    def test_multiple_units_same_player(self, manager_8x8: FogOfWarManager) -> None:
        """Multiple units for the same player combine their visibility."""
        units = [
            {"player": 0, "position": (1, 1), "type": "warrior"},
            {"player": 0, "position": (6, 6), "type": "warrior"},
        ]
        manager_8x8.compute_visibility(units, [])

        grid = manager_8x8.get_visibility(0)
        assert grid[1, 1] == VisibilityState.VISIBLE
        assert grid[6, 6] == VisibilityState.VISIBLE

    def test_both_units_and_cities(self, manager_8x8: FogOfWarManager) -> None:
        """Units and cities contribute to the same player's visibility."""
        units = [{"player": 0, "position": (1, 1), "type": "warrior"}]
        cities = [{"player": 0, "position": (6, 6)}]
        manager_8x8.compute_visibility(units, cities)

        grid = manager_8x8.get_visibility(0)
        assert grid[1, 1] == VisibilityState.VISIBLE
        assert grid[6, 6] == VisibilityState.VISIBLE

    def test_multi_player_visibility(self, manager_16x16: FogOfWarManager) -> None:
        """Each player gets independent visibility."""
        units = [
            {"player": 0, "position": (2, 2), "type": "warrior"},
            {"player": 1, "position": (8, 8), "type": "warrior"},
            {"player": 2, "position": (14, 14), "type": "warrior"},
            {"player": 3, "position": (2, 14), "type": "warrior"},
        ]
        manager_16x16.compute_visibility(units, [])

        for p in range(4):
            grid = manager_16x16.get_visibility(p)
            pos = [(2, 2), (8, 8), (14, 14), (2, 14)][p]
            assert grid[pos[1], pos[0]] == VisibilityState.VISIBLE

    def test_unit_with_no_type_key_uses_default(self, manager_8x8: FogOfWarManager) -> None:
        """A unit dict missing 'type' defaults to 'warrior' via .get()."""
        units = [{"player": 0, "position": (4, 4)}]
        manager_8x8.compute_visibility(units, [])
        grid = manager_8x8.get_visibility(0)
        assert grid[4, 4] == VisibilityState.VISIBLE

    def test_unit_with_no_player_key_defaults_zero(self, manager_8x8: FogOfWarManager) -> None:
        """A unit dict missing 'player' defaults to player 0."""
        units = [{"position": (4, 4), "type": "warrior"}]
        manager_8x8.compute_visibility(units, [])
        grid = manager_8x8.get_visibility(0)
        assert grid[4, 4] == VisibilityState.VISIBLE

    def test_city_with_no_player_key_defaults_zero(self, manager_8x8: FogOfWarManager) -> None:
        """A city dict missing 'player' defaults to player 0."""
        cities = [{"position": (4, 4)}]
        manager_8x8.compute_visibility([], cities)
        grid = manager_8x8.get_visibility(0)
        assert grid[4, 4] == VisibilityState.VISIBLE

    def test_empty_units_and_cities(self, manager_8x8: FogOfWarManager) -> None:
        """Empty lists don't crash and leave everything unexplored."""
        manager_8x8.compute_visibility([], [])
        for p in range(2):
            assert (manager_8x8.get_visibility(p) == VisibilityState.UNEXPLORED).all()


# =====================================================================
# FogOfWarManager — VISIBLE -> SEEN demotion
# =====================================================================


class TestVisibilityDemotion:
    """Tests for VISIBLE -> SEEN transition when units move away."""

    def test_visible_becomes_seen(self, manager_8x8: FogOfWarManager) -> None:
        """After a unit moves, previously visible tiles become SEEN."""
        # Turn 1: unit at (2,2)
        units = [{"player": 0, "position": (2, 2), "type": "warrior"}]
        manager_8x8.compute_visibility(units, [])
        assert manager_8x8.get_visibility(0)[2, 2] == VisibilityState.VISIBLE

        # Turn 2: unit moves to (6,6)
        units = [{"player": 0, "position": (6, 6), "type": "warrior"}]
        manager_8x8.compute_visibility(units, [])

        grid = manager_8x8.get_visibility(0)
        assert grid[2, 2] == VisibilityState.SEEN
        assert grid[6, 6] == VisibilityState.VISIBLE

    def test_seen_stays_seen_not_unexplored(self, manager_8x8: FogOfWarManager) -> None:
        """SEEN tiles do not revert to UNEXPLORED on subsequent updates."""
        # Turn 1: unit at (1,1)
        manager_8x8.compute_visibility(
            [{"player": 0, "position": (1, 1), "type": "warrior"}], []
        )
        # Turn 2: unit moves away
        manager_8x8.compute_visibility(
            [{"player": 0, "position": (6, 6), "type": "warrior"}], []
        )
        # Turn 3: unit moves further away
        manager_8x8.compute_visibility(
            [{"player": 0, "position": (6, 1), "type": "warrior"}], []
        )

        grid = manager_8x8.get_visibility(0)
        # (1,1) was visible in turn 1 and should still be SEEN
        assert grid[1, 1] == VisibilityState.SEEN

    def test_tile_reverts_to_visible_when_unit_returns(
        self, manager_8x8: FogOfWarManager
    ) -> None:
        """A SEEN tile becomes VISIBLE again when a unit returns."""
        # Turn 1: unit at (3,3)
        manager_8x8.compute_visibility(
            [{"player": 0, "position": (3, 3), "type": "warrior"}], []
        )
        # Turn 2: unit moves away
        manager_8x8.compute_visibility(
            [{"player": 0, "position": (7, 7), "type": "warrior"}], []
        )
        assert manager_8x8.get_visibility(0)[3, 3] == VisibilityState.SEEN

        # Turn 3: unit returns
        manager_8x8.compute_visibility(
            [{"player": 0, "position": (3, 3), "type": "warrior"}], []
        )
        assert manager_8x8.get_visibility(0)[3, 3] == VisibilityState.VISIBLE

    def test_no_units_all_visible_demoted_to_seen(
        self, manager_8x8: FogOfWarManager
    ) -> None:
        """If no units/cities remain, all VISIBLE tiles become SEEN."""
        manager_8x8.compute_visibility(
            [{"player": 0, "position": (3, 3), "type": "warrior"}], []
        )
        visible_count = np.sum(manager_8x8.get_visibility(0) == VisibilityState.VISIBLE)
        assert visible_count > 0

        # Update with no units or cities
        manager_8x8.compute_visibility([], [])
        grid = manager_8x8.get_visibility(0)
        assert np.sum(grid == VisibilityState.VISIBLE) == 0
        # Previously visible tiles are now SEEN
        assert np.sum(grid == VisibilityState.SEEN) == visible_count


# =====================================================================
# FogOfWarManager — disabled mode
# =====================================================================


class TestFogDisabled:
    """Tests for disabled fog-of-war mode."""

    def test_compute_sets_all_visible(self, manager_disabled: FogOfWarManager) -> None:
        manager_disabled.compute_visibility([], [])
        for p in range(2):
            assert (manager_disabled.get_visibility(p) == VisibilityState.VISIBLE).all()

    def test_mask_observation_passthrough(self, manager_disabled: FogOfWarManager) -> None:
        game_state = {"units": [{"player": 1, "position": (7, 7)}], "cities": []}
        result = manager_disabled.mask_observation(game_state, player=0)
        assert result is game_state

    def test_disabled_ignores_unit_positions(self, manager_disabled: FogOfWarManager) -> None:
        """Even without units, disabled mode makes everything visible."""
        manager_disabled.compute_visibility([], [])
        grid = manager_disabled.get_visibility(0)
        total = grid.shape[0] * grid.shape[1]
        assert np.sum(grid == VisibilityState.VISIBLE) == total


# =====================================================================
# FogOfWarManager — get_visibility / get_visibility_dict
# =====================================================================


class TestGetVisibility:
    """Tests for get_visibility and get_visibility_dict."""

    def test_get_visibility_valid_player(self, manager_8x8: FogOfWarManager) -> None:
        grid = manager_8x8.get_visibility(0)
        assert isinstance(grid, np.ndarray)
        assert grid.dtype == np.int8
        assert grid.shape == (8, 8)

    def test_get_visibility_unknown_player(self, manager_8x8: FogOfWarManager) -> None:
        """Requesting an invalid player returns all-UNEXPLORED grid."""
        grid = manager_8x8.get_visibility(99)
        assert grid.shape == (8, 8)
        assert (grid == VisibilityState.UNEXPLORED).all()

    def test_get_visibility_dict_structure(self, manager_8x8: FogOfWarManager) -> None:
        """get_visibility_dict returns correct string labels."""
        manager_8x8.compute_visibility(
            [{"player": 0, "position": (0, 0), "type": "warrior"}], []
        )
        vd = manager_8x8.get_visibility_dict(0)
        assert isinstance(vd, dict)
        assert len(vd) == 64  # 8*8
        assert vd[(0, 0)] == "visible"
        assert vd[(7, 7)] == "unexplored"

    def test_get_visibility_dict_seen_label(self, manager_8x8: FogOfWarManager) -> None:
        """SEEN tiles produce 'seen' string."""
        # Turn 1: reveal around (1,1)
        manager_8x8.compute_visibility(
            [{"player": 0, "position": (1, 1), "type": "warrior"}], []
        )
        # Turn 2: move away
        manager_8x8.compute_visibility(
            [{"player": 0, "position": (7, 7), "type": "warrior"}], []
        )
        vd = manager_8x8.get_visibility_dict(0)
        assert vd[(1, 1)] == "seen"

    def test_get_visibility_dict_all_states_present(
        self, manager_8x8: FogOfWarManager
    ) -> None:
        """After two turns, all three visibility strings appear."""
        manager_8x8.compute_visibility(
            [{"player": 0, "position": (1, 1), "type": "warrior"}], []
        )
        manager_8x8.compute_visibility(
            [{"player": 0, "position": (6, 6), "type": "warrior"}], []
        )
        vd = manager_8x8.get_visibility_dict(0)
        values = set(vd.values())
        assert "visible" in values
        assert "seen" in values
        assert "unexplored" in values


# =====================================================================
# FogOfWarManager — mask_observation
# =====================================================================


class TestMaskObservation:
    """Tests for mask_observation filtering."""

    @pytest.fixture()
    def game_state_two_players(self, manager_8x8: FogOfWarManager) -> dict:
        """Setup a game state where player 0 and 1 have units far apart."""
        units = [
            {"player": 0, "position": (1, 1), "type": "warrior"},
            {"player": 1, "position": (7, 7), "type": "warrior"},
        ]
        cities = [
            {"player": 0, "position": (0, 0)},
            {"player": 1, "position": (7, 6)},
        ]
        manager_8x8.compute_visibility(units, cities)
        return {
            "terrain": {},
            "units": units,
            "cities": cities,
            "current_player": 0,
            "num_players": 2,
        }

    def test_hides_distant_enemy_units(
        self, manager_8x8: FogOfWarManager, game_state_two_players: dict
    ) -> None:
        masked = manager_8x8.mask_observation(game_state_two_players, player=0)
        enemy_units = [u for u in masked["units"] if u["player"] == 1]
        assert len(enemy_units) == 0

    def test_keeps_own_units(
        self, manager_8x8: FogOfWarManager, game_state_two_players: dict
    ) -> None:
        masked = manager_8x8.mask_observation(game_state_two_players, player=0)
        own_units = [u for u in masked["units"] if u["player"] == 0]
        assert len(own_units) == 1

    def test_hides_distant_enemy_cities(
        self, manager_8x8: FogOfWarManager, game_state_two_players: dict
    ) -> None:
        masked = manager_8x8.mask_observation(game_state_two_players, player=0)
        enemy_cities = [c for c in masked["cities"] if c["player"] == 1]
        assert len(enemy_cities) == 0

    def test_keeps_own_cities(
        self, manager_8x8: FogOfWarManager, game_state_two_players: dict
    ) -> None:
        masked = manager_8x8.mask_observation(game_state_two_players, player=0)
        own_cities = [c for c in masked["cities"] if c["player"] == 0]
        assert len(own_cities) == 1

    def test_visibility_key_injected(
        self, manager_8x8: FogOfWarManager, game_state_two_players: dict
    ) -> None:
        masked = manager_8x8.mask_observation(game_state_two_players, player=0)
        assert "visibility" in masked
        assert isinstance(masked["visibility"], dict)

    def test_does_not_mutate_original(
        self, manager_8x8: FogOfWarManager, game_state_two_players: dict
    ) -> None:
        original_unit_count = len(game_state_two_players["units"])
        original_city_count = len(game_state_two_players["cities"])
        _ = manager_8x8.mask_observation(game_state_two_players, player=0)
        assert len(game_state_two_players["units"]) == original_unit_count
        assert len(game_state_two_players["cities"]) == original_city_count

    def test_keeps_visible_enemy_unit(self, manager_8x8: FogOfWarManager) -> None:
        """An enemy unit adjacent to a friendly unit is visible and kept."""
        units = [
            {"player": 0, "position": (3, 3), "type": "warrior"},
            {"player": 1, "position": (4, 3), "type": "warrior"},
        ]
        manager_8x8.compute_visibility(units, [])
        game_state = {"units": units, "cities": [], "terrain": {}}
        masked = manager_8x8.mask_observation(game_state, player=0)
        assert len(masked["units"]) == 2

    def test_keeps_visible_enemy_city(self, manager_8x8: FogOfWarManager) -> None:
        """An enemy city within sight range is kept."""
        units = [{"player": 0, "position": (3, 3), "type": "archer"}]
        cities = [{"player": 1, "position": (5, 3)}]
        manager_8x8.compute_visibility(units, cities)
        game_state = {"units": units, "cities": cities, "terrain": {}}
        masked = manager_8x8.mask_observation(game_state, player=0)
        assert len(masked["cities"]) == 1

    def test_out_of_bounds_enemy_unit_hidden(self, manager_8x8: FogOfWarManager) -> None:
        """An enemy unit with position outside the board is hidden (bounds check)."""
        units = [
            {"player": 0, "position": (1, 1), "type": "warrior"},
            {"player": 1, "position": (10, 10), "type": "warrior"},  # out of bounds
        ]
        manager_8x8.compute_visibility(units, [])
        game_state = {"units": units, "cities": [], "terrain": {}}
        masked = manager_8x8.mask_observation(game_state, player=0)
        enemy_units = [u for u in masked["units"] if u["player"] == 1]
        assert len(enemy_units) == 0

    def test_out_of_bounds_enemy_city_hidden(self, manager_8x8: FogOfWarManager) -> None:
        """An enemy city with out-of-bounds position is hidden."""
        units = [{"player": 0, "position": (1, 1), "type": "warrior"}]
        cities = [{"player": 1, "position": (-1, -1)}]
        manager_8x8.compute_visibility(units, cities)
        game_state = {"units": units, "cities": cities, "terrain": {}}
        masked = manager_8x8.mask_observation(game_state, player=0)
        assert len(masked["cities"]) == 0

    def test_empty_units_and_cities(self, manager_8x8: FogOfWarManager) -> None:
        """Masking with no units or cities doesn't crash."""
        manager_8x8.compute_visibility([], [])
        game_state = {"units": [], "cities": [], "terrain": {}}
        masked = manager_8x8.mask_observation(game_state, player=0)
        assert masked["units"] == []
        assert masked["cities"] == []


# =====================================================================
# FogOfWarManager — _reveal internal method
# =====================================================================


class TestRevealInternal:
    """Tests for the internal _reveal method edge cases."""

    def test_reveal_unknown_player_noop(self, manager_8x8: FogOfWarManager) -> None:
        """Revealing for a player not in the visibility dict is a no-op."""
        # This exercises the early return in _reveal when player not in self._visibility
        manager_8x8._reveal(player=99, center=(4, 4), sight_range=2)
        # No crash, and player 99 still returns unexplored via get_visibility
        grid = manager_8x8.get_visibility(99)
        assert (grid == VisibilityState.UNEXPLORED).all()

    def test_reveal_boundary_clipping(self, manager_8x8: FogOfWarManager) -> None:
        """Revealing near board edge clips correctly without IndexError."""
        # Center at (0,0) with sight 3 should not crash
        manager_8x8._reveal(player=0, center=(0, 0), sight_range=3)
        grid = manager_8x8.get_visibility(0)
        assert grid[0, 0] == VisibilityState.VISIBLE
        assert grid[3, 3] == VisibilityState.VISIBLE

    def test_reveal_at_max_corner(self, manager_8x8: FogOfWarManager) -> None:
        """Revealing at bottom-right corner clips correctly."""
        manager_8x8._reveal(player=0, center=(7, 7), sight_range=2)
        grid = manager_8x8.get_visibility(0)
        assert grid[7, 7] == VisibilityState.VISIBLE
        assert grid[5, 5] == VisibilityState.VISIBLE

    def test_reveal_sight_range_zero(self, manager_8x8: FogOfWarManager) -> None:
        """Sight range 0 reveals only the center tile."""
        manager_8x8._reveal(player=0, center=(4, 4), sight_range=0)
        grid = manager_8x8.get_visibility(0)
        assert grid[4, 4] == VisibilityState.VISIBLE
        # Adjacent tiles should be unexplored
        assert grid[4, 5] == VisibilityState.UNEXPLORED
        assert grid[3, 4] == VisibilityState.UNEXPLORED


# =====================================================================
# Parametrized broad tests
# =====================================================================


@pytest.mark.parametrize("board_size", [4, 8, 16])
@pytest.mark.parametrize("num_players", [2, 3, 4])
class TestFogOfWarParametrized:
    """Parametrized tests across board sizes and player counts."""

    def test_construction(self, board_size: int, num_players: int) -> None:
        config = FogOfWarConfig(enabled=True)
        mgr = FogOfWarManager(config, board_size=board_size, num_players=num_players)
        assert mgr.board_size == board_size
        assert mgr.num_players == num_players

    def test_compute_and_mask_round_trip(
        self, board_size: int, num_players: int
    ) -> None:
        config = FogOfWarConfig(enabled=True)
        mgr = FogOfWarManager(config, board_size=board_size, num_players=num_players)

        mid = board_size // 2
        units = [{"player": 0, "position": (mid, mid), "type": "warrior"}]
        enemy_units = [
            {"player": p, "position": (board_size - 1, board_size - 1), "type": "warrior"}
            for p in range(1, num_players)
        ]
        mgr.compute_visibility(units + enemy_units, [])

        game_state = {
            "terrain": {},
            "units": units + enemy_units,
            "cities": [],
        }
        masked = mgr.mask_observation(game_state, player=0)
        # Own unit always present
        own = [u for u in masked["units"] if u["player"] == 0]
        assert len(own) == 1

    def test_reset_round_trip(self, board_size: int, num_players: int) -> None:
        config = FogOfWarConfig(enabled=True)
        mgr = FogOfWarManager(config, board_size=board_size, num_players=num_players)
        mgr.compute_visibility(
            [{"player": 0, "position": (0, 0), "type": "warrior"}], []
        )
        mgr.reset()
        for p in range(num_players):
            assert (mgr.get_visibility(p) == VisibilityState.UNEXPLORED).all()
