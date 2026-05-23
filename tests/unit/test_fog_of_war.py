"""Tests for fog of war visibility management.

Tests cover:
- FogOfWarConfig defaults and validation
- VisibilityState enum values
- FogOfWarManager visibility computation
- Observation masking (hidden units removed, terrain retained)
- Reset behavior
- Integration with MicroCivEnv (fog_of_war=True/False)
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
# Config tests
# =====================================================================


class TestFogOfWarConfig:
    """Tests for FogOfWarConfig dataclass."""

    def test_defaults(self) -> None:
        config = FogOfWarConfig()
        assert config.enabled is False
        assert config.default_sight_range == 2
        assert config.city_sight_range == 3
        assert "warrior" in config.unit_sight_ranges
        assert "archer" in config.unit_sight_ranges

    def test_enabled(self) -> None:
        config = FogOfWarConfig(enabled=True)
        assert config.enabled is True

    def test_frozen(self) -> None:
        config = FogOfWarConfig()
        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore[misc]

    def test_custom_sight_ranges(self) -> None:
        config = FogOfWarConfig(
            unit_sight_ranges={"warrior": 1, "scout": 5},
        )
        assert config.unit_sight_ranges["warrior"] == 1
        assert config.unit_sight_ranges["scout"] == 5


# =====================================================================
# VisibilityState tests
# =====================================================================


class TestVisibilityState:
    """Tests for VisibilityState enum."""

    def test_values(self) -> None:
        assert VisibilityState.UNEXPLORED == 0
        assert VisibilityState.SEEN == 1
        assert VisibilityState.VISIBLE == 2

    def test_ordering(self) -> None:
        assert VisibilityState.UNEXPLORED < VisibilityState.SEEN
        assert VisibilityState.SEEN < VisibilityState.VISIBLE


# =====================================================================
# FogOfWarManager tests
# =====================================================================


class TestFogOfWarManager:
    """Tests for FogOfWarManager."""

    @pytest.fixture()
    def manager(self) -> FogOfWarManager:
        config = FogOfWarConfig(enabled=True, default_sight_range=2, city_sight_range=3)
        return FogOfWarManager(config, board_size=8, num_players=2)

    @pytest.fixture()
    def disabled_manager(self) -> FogOfWarManager:
        config = FogOfWarConfig(enabled=False)
        return FogOfWarManager(config, board_size=8, num_players=2)

    # ----- Reset -----

    def test_reset_clears_visibility(self, manager: FogOfWarManager) -> None:
        """After reset, all tiles should be UNEXPLORED."""
        manager.reset()
        for player in range(2):
            grid = manager.get_visibility(player)
            assert (grid == VisibilityState.UNEXPLORED).all()

    def test_reset_with_new_player_count(self, manager: FogOfWarManager) -> None:
        """Reset can change the number of players."""
        manager.reset(num_players=4)
        assert manager.num_players == 4
        for player in range(4):
            grid = manager.get_visibility(player)
            assert grid.shape == (8, 8)

    # ----- Visibility computation -----

    def test_compute_visibility_unit_at_center(self, manager: FogOfWarManager) -> None:
        """Unit at (4,4) with sight range 2 reveals a 5x5 diamond."""
        units = [{"player": 0, "position": (4, 4), "type": "warrior"}]
        manager.compute_visibility(units, [])

        grid = manager.get_visibility(0)
        # (4,4) itself must be visible
        assert grid[4, 4] == VisibilityState.VISIBLE

        # Tiles within Chebyshev distance 2 should be visible
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                x, y = 4 + dx, 4 + dy
                if 0 <= x < 8 and 0 <= y < 8:
                    assert grid[y, x] == VisibilityState.VISIBLE, (
                        f"({x},{y}) should be visible"
                    )

        # Tile at distance 3 should NOT be visible
        assert grid[4, 7] == VisibilityState.UNEXPLORED

    def test_compute_visibility_enemy_not_revealed_for_other(
        self, manager: FogOfWarManager
    ) -> None:
        """Player 1's units don't reveal tiles for player 0."""
        units = [{"player": 1, "position": (4, 4), "type": "warrior"}]
        manager.compute_visibility(units, [])

        grid_p0 = manager.get_visibility(0)
        grid_p1 = manager.get_visibility(1)

        # Player 0 sees nothing
        assert (grid_p0 == VisibilityState.UNEXPLORED).all()
        # Player 1 sees tiles around (4,4)
        assert grid_p1[4, 4] == VisibilityState.VISIBLE

    def test_city_sight_range(self, manager: FogOfWarManager) -> None:
        """Cities use city_sight_range (3) instead of default (2)."""
        cities = [{"player": 0, "position": (4, 4)}]
        manager.compute_visibility([], cities)

        grid = manager.get_visibility(0)
        # Chebyshev distance 3 should be visible
        assert grid[4, 7] == VisibilityState.VISIBLE  # dx=3
        assert grid[7, 4] == VisibilityState.VISIBLE  # dy=3
        assert grid[1, 1] == VisibilityState.VISIBLE  # both dx=3, dy=3

    def test_archer_sight_range(self, manager: FogOfWarManager) -> None:
        """Archers have sight range 3 (from config defaults)."""
        units = [{"player": 0, "position": (4, 4), "type": "archer"}]
        manager.compute_visibility(units, [])

        grid = manager.get_visibility(0)
        # Chebyshev distance 3 should be visible for archer
        assert grid[4, 7] == VisibilityState.VISIBLE

    def test_seen_tiles_retained(self, manager: FogOfWarManager) -> None:
        """Tiles that were VISIBLE become SEEN when unit moves away."""
        # Step 1: unit at (2,2) reveals tiles
        units = [{"player": 0, "position": (2, 2), "type": "warrior"}]
        manager.compute_visibility(units, [])

        grid = manager.get_visibility(0)
        assert grid[2, 2] == VisibilityState.VISIBLE

        # Step 2: unit moves to (6,6), old tiles become SEEN
        units = [{"player": 0, "position": (6, 6), "type": "warrior"}]
        manager.compute_visibility(units, [])

        grid = manager.get_visibility(0)
        assert grid[2, 2] == VisibilityState.SEEN
        assert grid[6, 6] == VisibilityState.VISIBLE

    def test_edge_positions(self, manager: FogOfWarManager) -> None:
        """Units at board edges don't cause index errors."""
        corners = [(0, 0), (7, 0), (0, 7), (7, 7)]
        for pos in corners:
            manager.reset()
            units = [{"player": 0, "position": pos, "type": "warrior"}]
            manager.compute_visibility(units, [])
            grid = manager.get_visibility(0)
            y, x = pos[1], pos[0]
            assert grid[y, x] == VisibilityState.VISIBLE

    # ----- Disabled mode -----

    def test_disabled_all_visible(self, disabled_manager: FogOfWarManager) -> None:
        """When disabled, compute_visibility makes everything visible."""
        disabled_manager.compute_visibility([], [])
        for player in range(2):
            grid = disabled_manager.get_visibility(player)
            assert (grid == VisibilityState.VISIBLE).all()

    # ----- Observation masking -----

    def test_mask_observation_removes_hidden_enemy(
        self, manager: FogOfWarManager
    ) -> None:
        """Enemy units on non-visible tiles are removed from observation."""
        units = [
            {"player": 0, "position": (1, 1), "type": "warrior"},
            {"player": 1, "position": (7, 7), "type": "warrior"},  # far away
        ]
        manager.compute_visibility(units, [])

        game_state = {
            "terrain": {},
            "units": units,
            "cities": [],
            "current_player": 0,
            "num_players": 2,
        }

        masked = manager.mask_observation(game_state, player=0)
        masked_units = masked["units"]

        # Own unit kept
        assert any(u["player"] == 0 for u in masked_units)
        # Enemy at (7,7) removed (not visible)
        assert not any(
            u["player"] == 1 and u["position"] == (7, 7) for u in masked_units
        )

    def test_mask_observation_keeps_visible_enemy(
        self, manager: FogOfWarManager
    ) -> None:
        """Enemy units on visible tiles are kept."""
        units = [
            {"player": 0, "position": (3, 3), "type": "warrior"},
            {"player": 1, "position": (4, 3), "type": "warrior"},  # adjacent
        ]
        manager.compute_visibility(units, [])

        game_state = {
            "terrain": {},
            "units": units,
            "cities": [],
            "current_player": 0,
            "num_players": 2,
        }

        masked = manager.mask_observation(game_state, player=0)
        # Both units should be present
        assert len(masked["units"]) == 2

    def test_mask_observation_hides_enemy_city(
        self, manager: FogOfWarManager
    ) -> None:
        """Enemy cities on non-visible tiles are removed."""
        units = [{"player": 0, "position": (1, 1), "type": "warrior"}]
        cities = [{"player": 1, "position": (7, 7)}]
        manager.compute_visibility(units, cities)

        game_state = {
            "terrain": {},
            "units": units,
            "cities": cities,
            "current_player": 0,
            "num_players": 2,
        }

        masked = manager.mask_observation(game_state, player=0)
        # Enemy city at (7,7) should be hidden (player 0 can't see it)
        assert len(masked["cities"]) == 0

    def test_mask_observation_injects_visibility(
        self, manager: FogOfWarManager
    ) -> None:
        """Masked state includes a 'visibility' dict."""
        units = [{"player": 0, "position": (3, 3), "type": "warrior"}]
        manager.compute_visibility(units, [])

        game_state = {
            "terrain": {},
            "units": units,
            "cities": [],
            "current_player": 0,
            "num_players": 2,
        }

        masked = manager.mask_observation(game_state, player=0)
        assert "visibility" in masked
        assert masked["visibility"][(3, 3)] == "visible"

    def test_mask_observation_disabled_passthrough(
        self, disabled_manager: FogOfWarManager
    ) -> None:
        """When disabled, mask_observation returns the state unchanged."""
        game_state = {"units": [{"player": 1}], "cities": []}
        result = disabled_manager.mask_observation(game_state, player=0)
        assert result is game_state  # same object

    def test_mask_observation_does_not_mutate_original(
        self, manager: FogOfWarManager
    ) -> None:
        """Masking must not mutate the original game state."""
        units = [
            {"player": 0, "position": (1, 1), "type": "warrior"},
            {"player": 1, "position": (7, 7), "type": "warrior"},
        ]
        manager.compute_visibility(units, [])

        game_state = {
            "terrain": {},
            "units": list(units),
            "cities": [],
            "current_player": 0,
            "num_players": 2,
        }
        original_unit_count = len(game_state["units"])

        _ = manager.mask_observation(game_state, player=0)
        assert len(game_state["units"]) == original_unit_count

    # ----- get_visibility_dict -----

    def test_get_visibility_dict(self, manager: FogOfWarManager) -> None:
        units = [{"player": 0, "position": (0, 0), "type": "warrior"}]
        manager.compute_visibility(units, [])

        vis_dict = manager.get_visibility_dict(0)
        assert vis_dict[(0, 0)] == "visible"
        assert vis_dict[(7, 7)] == "unexplored"

    def test_get_visibility_invalid_player(self, manager: FogOfWarManager) -> None:
        """Requesting visibility for unknown player returns unexplored."""
        grid = manager.get_visibility(99)
        assert (grid == VisibilityState.UNEXPLORED).all()


# =====================================================================
# Parametrized tests
# =====================================================================


@pytest.mark.parametrize("fog_of_war", [True, False])
@pytest.mark.parametrize("num_players", [2, 4])
class TestFogOfWarParametrized:
    """Parametrized tests across fog_of_war and num_players."""

    def test_manager_creation(self, fog_of_war: bool, num_players: int) -> None:
        config = FogOfWarConfig(enabled=fog_of_war)
        manager = FogOfWarManager(config, board_size=8, num_players=num_players)
        assert manager.num_players == num_players
        for p in range(num_players):
            grid = manager.get_visibility(p)
            assert grid.shape == (8, 8)

    def test_compute_and_mask(self, fog_of_war: bool, num_players: int) -> None:
        config = FogOfWarConfig(enabled=fog_of_war)
        manager = FogOfWarManager(config, board_size=8, num_players=num_players)

        units = [{"player": 0, "position": (2, 2), "type": "warrior"}]
        cities: list[dict] = []
        manager.compute_visibility(units, cities)

        game_state = {
            "terrain": {},
            "units": units + [
                {"player": p, "position": (7, 7), "type": "warrior"}
                for p in range(1, num_players)
            ],
            "cities": [],
            "current_player": 0,
            "num_players": num_players,
        }

        masked = manager.mask_observation(game_state, player=0)
        if fog_of_war:
            # Enemy units far away should be hidden
            visible_enemies = [u for u in masked["units"] if u["player"] != 0]
            assert len(visible_enemies) == 0
        else:
            # Everything visible
            assert len(masked["units"]) == len(game_state["units"])


# =====================================================================
# MicroCivEnv integration tests
# =====================================================================


class TestMicroCivFogOfWarIntegration:
    """Integration tests: MicroCivEnv with fog_of_war parameter."""

    def test_env_default_no_fog(self) -> None:
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=8, num_players=2)
        assert env.fog_manager is None

    def test_env_with_fog(self) -> None:
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=8, num_players=2, fog_of_war=True)
        assert env.fog_manager is not None

    def test_env_reset_with_fog(self) -> None:
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=8, num_players=2, fog_of_war=True, seed=42)
        obs, _info = env.reset()
        # Observation should still have expected keys
        assert "spatial" in obs
        assert "vector" in obs
        assert "action_mask" in obs

    def test_env_step_with_fog(self) -> None:
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=8, num_players=2, fog_of_war=True, seed=42)
        env.reset()
        mask = env.get_valid_actions()
        valid = np.where(mask > 0)[0]
        if len(valid) > 0:
            obs, _reward, _terminated, _truncated, _info = env.step(valid[0])
            assert "spatial" in obs

    def test_env_clone_preserves_fog(self) -> None:
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=8, num_players=2, fog_of_war=True, seed=42)
        env.reset()
        cloned = env.clone()
        assert cloned.fog_manager is not None
        assert cloned._fog_of_war_enabled is True

    @pytest.mark.parametrize("fog_of_war", [True, False])
    def test_env_observation_shape_consistent(self, fog_of_war: bool) -> None:
        from alpha_zero.envs.microciv import MicroCivEnv

        env = MicroCivEnv(board_size=8, num_players=2, fog_of_war=fog_of_war, seed=42)
        obs, _ = env.reset()
        assert obs["spatial"].shape[0] == 13  # MicroCivEncoder.TOTAL_CHANNELS
        assert obs["vector"].shape == (64,)
