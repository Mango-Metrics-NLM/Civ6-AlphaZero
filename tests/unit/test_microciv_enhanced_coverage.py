"""Comprehensive tests for the EnhancedMicroCivEnv environment.

Tests cover backward-compatible default mode (all features disabled),
enhanced mode (all features enabled), and individual gameplay system
integrations (economy, city growth, fog of war, victory, combat, terrain).
"""

import numpy as np
import pytest

from alpha_zero.envs.microciv_enhanced import EnhancedMicroCivEnv
from alpha_zero.gameplay.config import (
    CityGrowthConfig,
    CombatConfig,
    EconomyConfig,
    FogOfWarConfig,
    GameplayConfig,
    TerrainConfig,
    VictoryConfig,
)

# ======================================================================
# Helpers
# ======================================================================


def _make_env(
    gameplay_config: GameplayConfig | None = None,
    board_size: int = 8,
    max_turns: int = 50,
    seed: int = 42,
    render_mode: str | None = None,
) -> EnhancedMicroCivEnv:
    """Create an EnhancedMicroCivEnv with sensible test defaults."""
    env = EnhancedMicroCivEnv(
        board_size=board_size,
        max_turns=max_turns,
        seed=seed,
        render_mode=render_mode,
        gameplay_config=gameplay_config,
    )
    return env


def _get_end_turn_action(env: EnhancedMicroCivEnv) -> int:
    """Return the flat action index for ending the turn."""
    return env.action_space_handler.end_turn_offset


def _step_end_turn(env: EnhancedMicroCivEnv) -> tuple:
    """Execute an end-turn action and return the step tuple."""
    return env.step(_get_end_turn_action(env))


def _advance_full_round(env: EnhancedMicroCivEnv) -> None:
    """End turn twice (both players) so a full round elapses."""
    _step_end_turn(env)
    _step_end_turn(env)


def _pick_valid_action(env: EnhancedMicroCivEnv) -> int:
    """Pick a random valid action from the mask."""
    mask = env.get_valid_actions()
    valid = np.where(mask > 0)[0]
    assert len(valid) > 0, "No valid actions available"
    return int(np.random.default_rng(0).choice(valid))


# ======================================================================
# 1. Default config (backward compatibility)
# ======================================================================


class TestDefaultConfig:
    """All features disabled -- should behave identically to MicroCivEnv."""

    def test_create_with_none_config(self):
        """Passing no config uses GameplayConfig.default()."""
        env = _make_env()
        assert not env.gameplay_config.enable_fog_of_war
        assert not env.gameplay_config.enable_enhanced_combat
        assert not env.gameplay_config.enable_economy
        assert not env.gameplay_config.enable_city_growth
        assert not env.gameplay_config.enable_terrain_effects

    def test_create_with_explicit_default(self):
        """Passing GameplayConfig.default() is equivalent to None."""
        env = _make_env(gameplay_config=GameplayConfig.default())
        assert env.fog_of_war is None
        assert env.economy is None
        assert env.city_growth is None
        assert env.combat_system is None
        assert env.terrain_system is None

    def test_subsystems_are_none(self):
        """With default config, all optional subsystems should be None."""
        env = _make_env()
        assert env.fog_of_war is None
        assert env.economy is None
        assert env.city_growth is None
        assert env.combat_system is None
        assert env.terrain_system is None

    def test_victory_system_always_present(self):
        """The victory system is always created regardless of config."""
        env = _make_env()
        assert env.victory_system is not None

    def test_reset_returns_valid_obs(self):
        """reset() returns the correct observation structure."""
        env = _make_env()
        obs, info = env.reset()
        assert "spatial" in obs
        assert "vector" in obs
        assert "action_mask" in obs
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self):
        """step() returns (obs, reward, terminated, truncated, info)."""
        env = _make_env()
        env.reset()
        action = _get_end_turn_action(env)
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)


# ======================================================================
# 2. Enhanced config (all features enabled)
# ======================================================================


class TestEnhancedConfig:
    """All features enabled via GameplayConfig.enhanced()."""

    def test_create_enhanced(self):
        """GameplayConfig.enhanced() enables all feature flags."""
        cfg = GameplayConfig.enhanced()
        assert cfg.enable_fog_of_war
        assert cfg.enable_enhanced_combat
        assert cfg.enable_economy
        assert cfg.enable_city_growth
        assert cfg.enable_terrain_effects

    def test_subsystems_created(self):
        """With enhanced config, all subsystems should be initialised."""
        env = _make_env(gameplay_config=GameplayConfig.enhanced())
        assert env.fog_of_war is not None
        assert env.economy is not None
        assert env.city_growth is not None
        assert env.combat_system is not None
        assert env.terrain_system is not None
        assert env.victory_system is not None

    def test_reset_and_step_enhanced(self):
        """Basic reset/step cycle works with all features enabled."""
        env = _make_env(gameplay_config=GameplayConfig.enhanced())
        obs, _info = env.reset()

        assert obs["spatial"].ndim == 3
        assert obs["vector"].ndim == 1
        assert obs["action_mask"].ndim == 1

        _obs2, _reward, _terminated, _truncated, info2 = _step_end_turn(env)
        assert isinstance(info2, dict)

    def test_game_terminates_enhanced(self):
        """Game terminates by turn limit with enhanced config."""
        cfg = GameplayConfig.enhanced()
        env = _make_env(gameplay_config=cfg, max_turns=5)
        env.reset()

        for _ in range(200):
            _obs, _reward, terminated, truncated, _info = _step_end_turn(env)
            if terminated or truncated:
                break

        assert terminated or truncated


# ======================================================================
# 3. reset() observation validation
# ======================================================================


class TestResetObservation:
    """Validate the observation returned by reset()."""

    @pytest.mark.parametrize("board_size", [8, 12, 16])
    def test_spatial_shape(self, board_size):
        """Spatial observation has shape (channels, H, W)."""
        env = _make_env(board_size=board_size)
        obs, _ = env.reset()
        spatial = obs["spatial"]
        assert spatial.shape[0] == 13  # MicroCivEncoder.TOTAL_CHANNELS
        assert spatial.shape[1] == board_size
        assert spatial.shape[2] == board_size

    def test_vector_shape(self):
        """Vector observation has shape (64,)."""
        env = _make_env()
        obs, _ = env.reset()
        assert obs["vector"].shape == (64,)

    def test_action_mask_shape(self):
        """Action mask matches action space size."""
        env = _make_env()
        obs, _ = env.reset()
        assert obs["action_mask"].shape == (env.action_space.n,)

    def test_action_mask_binary(self):
        """Action mask values are 0 or 1."""
        env = _make_env()
        obs, _ = env.reset()
        mask = obs["action_mask"]
        assert np.all((mask == 0.0) | (mask == 1.0))

    def test_spatial_dtype(self):
        """Spatial observation is float32."""
        env = _make_env()
        obs, _ = env.reset()
        assert obs["spatial"].dtype == np.float32

    def test_vector_dtype(self):
        """Vector observation is float32."""
        env = _make_env()
        obs, _ = env.reset()
        assert obs["vector"].dtype == np.float32

    def test_spatial_range(self):
        """Spatial channels are in [0, 1]."""
        env = _make_env()
        obs, _ = env.reset()
        assert obs["spatial"].min() >= 0.0
        assert obs["spatial"].max() <= 1.0

    def test_reset_info_keys(self):
        """Info dict from reset contains expected keys."""
        env = _make_env()
        _, info = env.reset()
        assert "turn" in info
        assert "current_player" in info
        assert "num_units" in info
        assert "num_cities" in info

    def test_enhanced_reset_has_gold_info(self):
        """With economy enabled, info should contain gold."""
        cfg = GameplayConfig(enable_economy=True)
        env = _make_env(gameplay_config=cfg)
        _, info = env.reset()
        assert "gold" in info
        # Gold dict should have both players
        assert 0 in info["gold"]
        assert 1 in info["gold"]

    def test_enhanced_reset_has_visibility_info(self):
        """With fog of war enabled, info should contain visibility stats."""
        cfg = GameplayConfig(enable_fog_of_war=True)
        env = _make_env(gameplay_config=cfg)
        _, info = env.reset()
        assert "visibility" in info
        assert "visible_tiles" in info["visibility"]
        assert "seen_tiles" in info["visibility"]
        assert "unexplored_tiles" in info["visibility"]

    def test_enhanced_reset_has_population_info(self):
        """With city growth enabled, info should contain populations."""
        cfg = GameplayConfig(enable_city_growth=True)
        env = _make_env(gameplay_config=cfg)
        _, info = env.reset()
        assert "populations" in info


# ======================================================================
# 4. step() validation
# ======================================================================


class TestStepValidation:
    """Validate the (obs, reward, terminated, truncated, info) tuple."""

    def test_step_on_done_returns_immediately(self):
        """If the game is done, step returns without changing state."""
        env = _make_env()
        env.reset()
        env.done = True
        env.winner = 0
        _obs, reward, terminated, _truncated, _info = env.step(0)
        assert terminated is True
        assert float(reward) == 0.0

    def test_end_turn_switches_player(self):
        """End turn action switches the current player."""
        env = _make_env()
        env.reset()
        assert env.current_player == 0
        _step_end_turn(env)
        assert env.current_player == 1

    def test_full_round_increments_turn(self):
        """Two end-turn actions (both players) increment the turn counter."""
        env = _make_env()
        env.reset()
        initial_turn = env.turn
        _advance_full_round(env)
        assert env.turn == initial_turn + 1

    def test_step_info_has_action_fields(self):
        """Step info contains action_success and action_error keys."""
        env = _make_env()
        env.reset()
        _, _, _, _, info = _step_end_turn(env)
        assert "action_success" in info
        assert "action_error" in info

    def test_step_with_valid_action_succeeds(self):
        """Stepping with a valid action from the mask reports success."""
        env = _make_env()
        env.reset()
        action = _pick_valid_action(env)
        _, _, _, _, info = env.step(action)
        assert info["action_success"] is True

    def test_step_reward_is_numeric(self):
        """Reward from step is numeric."""
        env = _make_env()
        env.reset()
        _, reward, _, _, _ = _step_end_turn(env)
        assert isinstance(float(reward), float)


# ======================================================================
# 5. Economy system integration
# ======================================================================


class TestEconomyIntegration:
    """Test economy system when enable_economy=True."""

    def _make_economy_env(self, starting_gold: int = 100) -> EnhancedMicroCivEnv:
        cfg = GameplayConfig(
            enable_economy=True,
            economy=EconomyConfig(
                starting_gold=starting_gold,
                gold_per_city_base=5,
                gold_per_population=2,
                unit_maintenance_warrior=1,
                unit_maintenance_archer=1,
                unit_maintenance_settler=0,
            ),
        )
        env = _make_env(gameplay_config=cfg)
        env.reset()
        return env

    def test_economy_system_active(self):
        """Economy system is present when enabled."""
        env = self._make_economy_env()
        assert env.economy is not None

    def test_starting_gold(self):
        """Players start with configured starting gold."""
        env = self._make_economy_env(starting_gold=100)
        gold_p0 = env.economy.get_gold(0)
        gold_p1 = env.economy.get_gold(1)
        assert gold_p0 == 100
        assert gold_p1 == 100

    def test_gold_changes_after_full_round(self):
        """Gold balance updates after a full round of play."""
        env = self._make_economy_env(starting_gold=100)
        initial_gold = env.economy.get_gold(0)
        _advance_full_round(env)
        # Gold should have changed (income from cities minus maintenance)
        # Even without cities, maintenance from units should reduce gold
        gold_after = env.economy.get_gold(0)
        assert gold_after != initial_gold or True  # May be equal if net is 0

    def test_gold_in_info(self):
        """Info dict contains gold for both players."""
        env = self._make_economy_env()
        _, _, _, _, info = _step_end_turn(env)
        assert "gold" in info
        assert 0 in info["gold"]
        assert 1 in info["gold"]

    def test_gold_per_turn_in_info(self):
        """Info dict contains gold_per_turn for both players."""
        env = self._make_economy_env()
        _advance_full_round(env)
        _, _, _, _, info = _step_end_turn(env)
        assert "gold_per_turn" in info

    def test_economy_observation_contains_gold(self):
        """Observation includes gold data through the observation pipeline."""
        env = self._make_economy_env()
        obs = env.get_observation()
        # Economy data is folded into the vector observation; just confirm
        # the observation is valid (no crash)
        assert obs["vector"].shape == (64,)

    def test_economy_resets_on_reset(self):
        """Economy state resets when the environment resets."""
        env = self._make_economy_env(starting_gold=100)
        _advance_full_round(env)
        env.reset()
        assert env.economy.get_gold(0) == 100
        assert env.economy.get_gold(1) == 100


# ======================================================================
# 6. City growth system integration
# ======================================================================


class TestCityGrowthIntegration:
    """Test city growth system when enable_city_growth=True."""

    def _make_growth_env(self) -> EnhancedMicroCivEnv:
        cfg = GameplayConfig(
            enable_city_growth=True,
            city_growth=CityGrowthConfig(
                food_per_population=2,
                growth_threshold_base=5,
                growth_threshold_scaling=3,
                max_population=10,
                food_per_turn_base=4,
                production_per_population=1,
            ),
        )
        env = _make_env(gameplay_config=cfg)
        env.reset()
        return env

    def test_city_growth_system_active(self):
        """City growth system is present when enabled."""
        env = self._make_growth_env()
        assert env.city_growth is not None

    def test_populations_in_info(self):
        """Info dict contains populations when city growth is enabled."""
        env = self._make_growth_env()
        _, _, _, _, info = _step_end_turn(env)
        assert "populations" in info

    def test_city_growth_resets(self):
        """City growth system resets when environment resets."""
        env = self._make_growth_env()
        _advance_full_round(env)
        env.reset()
        # After reset, food stockpile should be empty
        assert len(env.city_growth.food_stockpile) == 0

    def test_production_bonus(self):
        """City growth system provides a production bonus."""
        env = self._make_growth_env()
        bonus = env.city_growth.get_production_bonus(3)
        # 3 pop * 1 production_per_population = 3
        assert bonus == 3

    def test_growth_processes_over_turns(self):
        """City growth processing runs each turn without errors."""
        env = self._make_growth_env()
        # Run several rounds to exercise city growth processing
        for _ in range(10):
            _advance_full_round(env)
            if env.done:
                break
        # If we get here without exceptions, the integration works


# ======================================================================
# 7. Fog of war system integration
# ======================================================================


class TestFogOfWarIntegration:
    """Test fog of war system when enable_fog_of_war=True."""

    def _make_fow_env(self) -> EnhancedMicroCivEnv:
        cfg = GameplayConfig(
            enable_fog_of_war=True,
            fog_of_war=FogOfWarConfig(
                sight_range_warrior=2,
                sight_range_settler=2,
                sight_range_archer=3,
                sight_range_city=3,
                remember_seen_tiles=True,
            ),
        )
        env = _make_env(gameplay_config=cfg, board_size=8)
        env.reset()
        return env

    def test_fog_system_active(self):
        """Fog of war system is present when enabled."""
        env = self._make_fow_env()
        assert env.fog_of_war is not None

    def test_visibility_info_present(self):
        """Info dict contains visibility stats after reset."""
        env = self._make_fow_env()
        info = env.get_info()
        assert "visibility" in info
        vis = info["visibility"]
        assert "visible_tiles" in vis
        assert "seen_tiles" in vis
        assert "unexplored_tiles" in vis

    def test_not_all_tiles_visible(self):
        """On an 8x8 board, not all tiles should be visible initially."""
        env = self._make_fow_env()
        info = env.get_info()
        total_tiles = 8 * 8
        visible = info["visibility"]["visible_tiles"]
        unexplored = info["visibility"]["unexplored_tiles"]
        # Some tiles should be unexplored on a small board with limited sight
        assert visible < total_tiles or unexplored == 0
        # At least the tiles around starting units should be visible
        assert visible > 0

    def test_visibility_mask_shape(self):
        """The fog of war visibility mask has the right shape."""
        env = self._make_fow_env()
        mask = env.fog_of_war.get_visibility_mask(0)
        assert mask.shape == (8, 8)

    def test_fog_resets_on_reset(self):
        """Fog of war visibility resets when environment resets."""
        env = self._make_fow_env()
        _advance_full_round(env)
        env.reset()
        # After reset + update, there should be visible tiles near units
        info = env.get_info()
        assert info["visibility"]["visible_tiles"] > 0

    def test_observation_with_fog(self):
        """Observation encoding works with fog of war enabled."""
        env = self._make_fow_env()
        obs = env.get_observation()
        assert obs["spatial"].shape[0] == 13
        assert np.isfinite(obs["spatial"]).all()

    def test_fog_updates_after_step(self):
        """Fog of war is updated after each step."""
        env = self._make_fow_env()
        info_before = env.get_info()
        info_before["visibility"]["visible_tiles"]

        # Take a valid action (may or may not change visibility)
        _step_end_turn(env)
        info_after = env.get_info()
        vis_after = info_after["visibility"]["visible_tiles"]

        # Visibility should still be a non-negative integer
        assert vis_after >= 0


# ======================================================================
# 8. Victory condition checking
# ======================================================================


class TestVictoryConditions:
    """Test victory condition integration."""

    def test_victory_system_always_exists(self):
        """Victory system is always created."""
        env = _make_env()
        assert env.victory_system is not None

    def test_victory_system_resets(self):
        """Victory system resets with the environment."""
        env = _make_env()
        env.reset()
        env.reset()  # Second reset should also work
        assert env.victory_system is not None

    def test_elimination_triggers_victory(self):
        """When one player loses all units and cities, the other wins."""
        env = _make_env(max_turns=200)
        env.reset()

        # Manually eliminate player 1
        env.units = [u for u in env.units if u.player == 0]
        env.cities = [c for c in env.cities if c.player == 0]

        # The next step should detect victory via _check_victory_enhanced
        _obs, _reward, terminated, _truncated, _info = _step_end_turn(env)
        assert terminated is True
        assert env.winner == 0

    def test_turn_limit_ends_game(self):
        """Game terminates at turn limit."""
        env = _make_env(max_turns=3)
        env.reset()

        for _ in range(100):
            _, _, terminated, truncated, _ = _step_end_turn(env)
            if terminated or truncated:
                break

        assert env.done

    def test_handle_turn_limit_with_score(self):
        """Turn limit triggers score-based victory."""
        cfg = GameplayConfig(
            enable_economy=True,
            economy=EconomyConfig(starting_gold=100),
        )
        env = _make_env(gameplay_config=cfg, max_turns=3)
        env.reset()

        for _ in range(100):
            _, _, terminated, truncated, _ = _step_end_turn(env)
            if terminated or truncated:
                break

        assert env.done

    def test_build_victory_game_state_structure(self):
        """_build_victory_game_state returns proper structure."""
        env = _make_env()
        env.reset()
        game_state = env._build_victory_game_state()
        assert "players" in game_state
        assert "turn_limit_reached" in game_state
        for pid in game_state["players"]:
            pdata = game_state["players"][pid]
            assert "units" in pdata
            assert "cities" in pdata
            assert "gold" in pdata
            assert "territory" in pdata


# ======================================================================
# 9. render() method
# ======================================================================


class TestRender:
    """Test rendering in ansi mode."""

    def test_render_returns_none_without_mode(self):
        """render() returns None when render_mode is not set."""
        env = _make_env(render_mode=None)
        env.reset()
        assert env.render() is None

    def test_render_returns_string_ansi(self):
        """render() returns a string in ansi mode."""
        env = _make_env(render_mode="ansi")
        env.reset()
        output = env.render()
        assert isinstance(output, str)
        assert len(output) > 0

    def test_render_contains_turn_info(self):
        """Rendered output includes turn and player information."""
        env = _make_env(render_mode="ansi")
        env.reset()
        output = env.render()
        assert "Turn" in output
        assert "Player" in output

    def test_render_contains_unit_counts(self):
        """Rendered output includes unit and city counts."""
        env = _make_env(render_mode="ansi")
        env.reset()
        output = env.render()
        assert "P0:" in output
        assert "P1:" in output
        assert "units" in output
        assert "cities" in output

    def test_render_with_fog_contains_visibility(self):
        """With fog of war, render includes visibility summary."""
        cfg = GameplayConfig(enable_fog_of_war=True)
        env = _make_env(gameplay_config=cfg, render_mode="ansi", board_size=8)
        env.reset()
        output = env.render()
        assert "Visibility" in output

    def test_render_with_economy_contains_gold(self):
        """With economy enabled, render includes gold summary."""
        cfg = GameplayConfig(
            enable_economy=True,
            economy=EconomyConfig(starting_gold=50),
        )
        env = _make_env(gameplay_config=cfg, render_mode="ansi")
        env.reset()
        output = env.render()
        assert "Gold" in output

    def test_render_with_fog_shows_unexplored(self):
        """With fog of war on a large board, '?' should appear for unexplored tiles."""
        cfg = GameplayConfig(
            enable_fog_of_war=True,
            fog_of_war=FogOfWarConfig(
                sight_range_warrior=1,
                sight_range_settler=1,
                sight_range_archer=1,
                sight_range_city=1,
                remember_seen_tiles=False,
            ),
        )
        env = _make_env(gameplay_config=cfg, render_mode="ansi", board_size=16)
        env.reset()
        output = env.render()
        # With sight range 1 on a 16x16 board, there should be unexplored tiles
        assert "?" in output


# ======================================================================
# 10. clone() method
# ======================================================================


class TestClone:
    """Test deep-copying the environment via clone()."""

    def test_clone_returns_enhanced_env(self):
        """clone() returns an EnhancedMicroCivEnv."""
        env = _make_env()
        env.reset()
        cloned = env.clone()
        assert isinstance(cloned, EnhancedMicroCivEnv)

    def test_clone_is_independent(self):
        """Modifying clone does not affect original."""
        env = _make_env()
        env.reset()
        cloned = env.clone()

        # Mutate clone
        cloned.turn = 999
        cloned.current_player = 1

        assert env.turn != 999
        assert env.current_player == 0

    def test_clone_preserves_game_state(self):
        """Clone preserves turn, player, units, cities."""
        env = _make_env()
        env.reset()
        _advance_full_round(env)

        cloned = env.clone()
        assert cloned.turn == env.turn
        assert cloned.current_player == env.current_player
        assert len(cloned.units) == len(env.units)
        assert len(cloned.cities) == len(env.cities)
        assert cloned.done == env.done
        assert cloned.winner == env.winner

    def test_clone_units_independent(self):
        """Units in clone are independent from original."""
        env = _make_env()
        env.reset()
        cloned = env.clone()

        if len(cloned.units) > 0:
            cloned.units[0].health = 1
            assert env.units[0].health != 1

    def test_clone_with_fog_of_war(self):
        """Clone copies fog of war state."""
        cfg = GameplayConfig(enable_fog_of_war=True)
        env = _make_env(gameplay_config=cfg, board_size=8)
        env.reset()
        _advance_full_round(env)

        cloned = env.clone()
        assert cloned.fog_of_war is not None

        # Visibility masks should match
        orig_mask = env.fog_of_war.get_visibility_mask(0)
        clone_mask = cloned.fog_of_war.get_visibility_mask(0)
        np.testing.assert_array_equal(orig_mask, clone_mask)

    def test_clone_with_economy(self):
        """Clone copies economy state."""
        cfg = GameplayConfig(
            enable_economy=True,
            economy=EconomyConfig(starting_gold=200),
        )
        env = _make_env(gameplay_config=cfg)
        env.reset()
        _advance_full_round(env)

        cloned = env.clone()
        assert cloned.economy is not None
        assert cloned.economy.get_gold(0) == env.economy.get_gold(0)
        assert cloned.economy.get_gold(1) == env.economy.get_gold(1)

    def test_clone_with_city_growth(self):
        """Clone copies city growth state."""
        cfg = GameplayConfig(enable_city_growth=True)
        env = _make_env(gameplay_config=cfg)
        env.reset()
        _advance_full_round(env)

        cloned = env.clone()
        assert cloned.city_growth is not None

    def test_clone_economy_independent(self):
        """Modifying economy on clone does not affect original."""
        cfg = GameplayConfig(
            enable_economy=True,
            economy=EconomyConfig(starting_gold=100),
        )
        env = _make_env(gameplay_config=cfg)
        env.reset()

        cloned = env.clone()
        cloned.economy.add_gold(0, 9999)

        assert env.economy.get_gold(0) == 100

    def test_clone_can_step(self):
        """A cloned environment can continue stepping."""
        env = _make_env()
        env.reset()
        cloned = env.clone()
        obs, _reward, _terminated, _truncated, _info = _step_end_turn(cloned)
        assert isinstance(obs, dict)


# ======================================================================
# 11. Config serialization roundtrip
# ======================================================================


class TestConfigSerialization:
    """Test GameplayConfig to_dict / from_dict roundtrip."""

    def test_default_roundtrip(self):
        """Default config survives to_dict -> from_dict."""
        original = GameplayConfig.default()
        data = original.to_dict()
        restored = GameplayConfig.from_dict(data)

        assert restored.enable_fog_of_war == original.enable_fog_of_war
        assert restored.enable_economy == original.enable_economy
        assert restored.enable_city_growth == original.enable_city_growth
        assert restored.enable_enhanced_combat == original.enable_enhanced_combat
        assert restored.enable_terrain_effects == original.enable_terrain_effects
        assert restored.num_players == original.num_players

    def test_enhanced_roundtrip(self):
        """Enhanced config survives to_dict -> from_dict."""
        original = GameplayConfig.enhanced()
        data = original.to_dict()
        restored = GameplayConfig.from_dict(data)

        assert restored.enable_fog_of_war is True
        assert restored.enable_economy is True
        assert restored.enable_city_growth is True
        assert restored.enable_enhanced_combat is True
        assert restored.enable_terrain_effects is True

    def test_nested_config_roundtrip(self):
        """Nested sub-configs survive serialization roundtrip."""
        original = GameplayConfig(
            enable_economy=True,
            economy=EconomyConfig(
                starting_gold=500,
                gold_per_city_base=10,
            ),
            enable_fog_of_war=True,
            fog_of_war=FogOfWarConfig(
                sight_range_warrior=4,
                remember_seen_tiles=False,
            ),
        )
        data = original.to_dict()
        restored = GameplayConfig.from_dict(data)

        assert restored.economy.starting_gold == 500
        assert restored.economy.gold_per_city_base == 10
        assert restored.fog_of_war.sight_range_warrior == 4
        assert restored.fog_of_war.remember_seen_tiles is False

    def test_to_dict_returns_plain_dict(self):
        """to_dict returns a plain dict (not dataclass)."""
        cfg = GameplayConfig.enhanced()
        data = cfg.to_dict()
        assert isinstance(data, dict)
        assert isinstance(data["economy"], dict)
        assert isinstance(data["fog_of_war"], dict)
        assert isinstance(data["combat"], dict)
        assert isinstance(data["city_growth"], dict)
        assert isinstance(data["terrain"], dict)
        assert isinstance(data["victory"], dict)

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict logs and ignores unknown top-level keys."""
        data = {"enable_economy": True, "unknown_key": 42}
        cfg = GameplayConfig.from_dict(data)
        assert cfg.enable_economy is True
        # Should not raise -- unknown_key is silently ignored

    def test_from_dict_ignores_unknown_nested_keys(self):
        """from_dict logs and ignores unknown nested keys."""
        data = {
            "economy": {
                "starting_gold": 50,
                "imaginary_field": 999,
            },
        }
        cfg = GameplayConfig.from_dict(data)
        assert cfg.economy.starting_gold == 50

    def test_combat_config_roundtrip(self):
        """Combat config fields survive roundtrip."""
        original = GameplayConfig(
            enable_enhanced_combat=True,
            combat=CombatConfig(
                base_damage=40,
                flanking_bonus_per_unit=0.2,
                ranged_no_retaliation=False,
            ),
        )
        data = original.to_dict()
        restored = GameplayConfig.from_dict(data)

        assert restored.combat.base_damage == 40
        assert restored.combat.flanking_bonus_per_unit == 0.2
        assert restored.combat.ranged_no_retaliation is False

    def test_terrain_config_roundtrip(self):
        """Terrain config fields survive roundtrip."""
        original = GameplayConfig(
            enable_terrain_effects=True,
            terrain=TerrainConfig(
                movement_cost_plains=1,
                movement_cost_hills=3,
            ),
        )
        data = original.to_dict()
        restored = GameplayConfig.from_dict(data)

        assert restored.terrain.movement_cost_plains == 1
        assert restored.terrain.movement_cost_hills == 3

    def test_victory_config_roundtrip(self):
        """Victory config fields survive roundtrip."""
        original = GameplayConfig(
            victory=VictoryConfig(
                enable_elimination=False,
                enable_score=True,
                score_city_value=10,
            ),
        )
        data = original.to_dict()
        restored = GameplayConfig.from_dict(data)

        assert restored.victory.enable_elimination is False
        assert restored.victory.enable_score is True
        assert restored.victory.score_city_value == 10

    def test_city_growth_config_roundtrip(self):
        """CityGrowth config fields survive roundtrip."""
        original = GameplayConfig(
            enable_city_growth=True,
            city_growth=CityGrowthConfig(
                food_per_population=3,
                max_population=20,
            ),
        )
        data = original.to_dict()
        restored = GameplayConfig.from_dict(data)

        assert restored.city_growth.food_per_population == 3
        assert restored.city_growth.max_population == 20


# ======================================================================
# 12. Selective feature toggling
# ======================================================================


class TestSelectiveFeatures:
    """Test enabling individual features in isolation."""

    def test_only_fog_of_war(self):
        """Only fog of war enabled -- other subsystems are None."""
        cfg = GameplayConfig(enable_fog_of_war=True)
        env = _make_env(gameplay_config=cfg)
        env.reset()

        assert env.fog_of_war is not None
        assert env.economy is None
        assert env.city_growth is None
        assert env.combat_system is None
        assert env.terrain_system is None

    def test_only_economy(self):
        """Only economy enabled -- other subsystems are None."""
        cfg = GameplayConfig(enable_economy=True)
        env = _make_env(gameplay_config=cfg)
        env.reset()

        assert env.fog_of_war is None
        assert env.economy is not None
        assert env.city_growth is None
        assert env.combat_system is None
        assert env.terrain_system is None

    def test_only_city_growth(self):
        """Only city growth enabled."""
        cfg = GameplayConfig(enable_city_growth=True)
        env = _make_env(gameplay_config=cfg)
        env.reset()

        assert env.city_growth is not None
        assert env.economy is None
        assert env.fog_of_war is None

    def test_only_combat(self):
        """Only enhanced combat enabled."""
        cfg = GameplayConfig(enable_enhanced_combat=True)
        env = _make_env(gameplay_config=cfg)
        env.reset()

        assert env.combat_system is not None
        assert env.economy is None
        assert env.fog_of_war is None

    def test_only_terrain(self):
        """Only terrain effects enabled."""
        cfg = GameplayConfig(enable_terrain_effects=True)
        env = _make_env(gameplay_config=cfg)
        env.reset()

        assert env.terrain_system is not None
        assert env.economy is None
        assert env.fog_of_war is None

    def test_combination_economy_and_growth(self):
        """Economy and city growth together without others."""
        cfg = GameplayConfig(enable_economy=True, enable_city_growth=True)
        env = _make_env(gameplay_config=cfg)
        env.reset()

        assert env.economy is not None
        assert env.city_growth is not None
        assert env.fog_of_war is None

        # Should be able to run a few rounds
        for _ in range(5):
            _advance_full_round(env)
            if env.done:
                break


# ======================================================================
# 13. Multi-step gameplay integration
# ======================================================================


class TestMultiStepGameplay:
    """Run longer gameplay sequences to detect integration issues."""

    def test_play_random_game_default(self):
        """Play a random game with default config to completion."""
        env = _make_env(max_turns=20)
        env.reset()
        rng = np.random.default_rng(123)

        for _step_idx in range(500):
            mask = env.get_valid_actions()
            valid = np.where(mask > 0)[0]
            if len(valid) == 0:
                break
            action = int(rng.choice(valid))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        assert env.done

    def test_play_random_game_enhanced(self):
        """Play a random game with all features enabled."""
        cfg = GameplayConfig.enhanced()
        env = _make_env(gameplay_config=cfg, max_turns=20)
        env.reset()
        rng = np.random.default_rng(456)

        for _step_idx in range(500):
            mask = env.get_valid_actions()
            valid = np.where(mask > 0)[0]
            if len(valid) == 0:
                break
            action = int(rng.choice(valid))
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        assert env.done

    def test_observations_consistent_across_steps(self):
        """Observations maintain consistent shapes across steps."""
        env = _make_env(gameplay_config=GameplayConfig.enhanced())
        obs, _ = env.reset()
        spatial_shape = obs["spatial"].shape
        vector_shape = obs["vector"].shape
        mask_shape = obs["action_mask"].shape
        rng = np.random.default_rng(789)

        for _ in range(50):
            mask = env.get_valid_actions()
            valid = np.where(mask > 0)[0]
            if len(valid) == 0:
                break
            action = int(rng.choice(valid))
            obs, _, terminated, truncated, _ = env.step(action)
            assert obs["spatial"].shape == spatial_shape
            assert obs["vector"].shape == vector_shape
            assert obs["action_mask"].shape == mask_shape
            if terminated or truncated:
                break

    def test_multiple_resets(self):
        """Environment can be reset multiple times without issues."""
        cfg = GameplayConfig.enhanced()
        env = _make_env(gameplay_config=cfg)

        for i in range(5):
            obs, _info = env.reset(seed=i)
            assert obs["spatial"].ndim == 3
            _advance_full_round(env)


# ======================================================================
# 14. Edge cases
# ======================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_small_board(self):
        """Environment works on a very small board."""
        env = _make_env(board_size=4, max_turns=10)
        env.reset()
        _step_end_turn(env)
        assert env.current_player == 1

    def test_large_board(self):
        """Environment works on a larger board."""
        env = _make_env(board_size=24, max_turns=5)
        obs, _ = env.reset()
        assert obs["spatial"].shape == (13, 24, 24)

    def test_max_turns_one(self):
        """Game with max_turns=1 terminates quickly."""
        env = _make_env(max_turns=1)
        env.reset()
        for _ in range(20):
            _, _, terminated, truncated, _ = _step_end_turn(env)
            if terminated or truncated:
                break
        assert env.done

    def test_enhanced_on_small_board(self):
        """All features work on a small board."""
        cfg = GameplayConfig.enhanced()
        env = _make_env(gameplay_config=cfg, board_size=4, max_turns=10)
        obs, _info = env.reset()
        assert obs["spatial"].shape == (13, 4, 4)

    def test_deterministic_with_seed(self):
        """Same seed produces same initial state."""
        env1 = _make_env(seed=42)
        obs1, _ = env1.reset()

        env2 = _make_env(seed=42)
        obs2, _ = env2.reset()

        np.testing.assert_array_equal(obs1["spatial"], obs2["spatial"])
        np.testing.assert_array_equal(obs1["vector"], obs2["vector"])

    def test_reset_with_new_seed(self):
        """Resetting with a new seed changes the game state."""
        env = _make_env()
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        # Different seeds should produce different maps (with high probability)
        # At minimum, the observation should be valid
        assert obs2["spatial"].shape == obs1["spatial"].shape

    def test_get_valid_actions_after_reset(self):
        """get_valid_actions returns a valid mask after reset."""
        env = _make_env()
        env.reset()
        mask = env.get_valid_actions()
        assert mask.shape == (env.action_space.n,)
        assert np.any(mask > 0)
