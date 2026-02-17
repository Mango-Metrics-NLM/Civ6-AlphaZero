"""Tests for the centralized environment factory.

Validates that ``create_game_env`` returns the correct environment type
for each preset, preserves observation shapes, and handles edge cases.
"""

from __future__ import annotations

import pytest

from alpha_zero.envs.env_factory import GAMEPLAY_PRESETS, create_game_env
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.envs.microciv_enhanced import EnhancedMicroCivEnv

# ---------------------------------------------------------------------------
# Preset selection
# ---------------------------------------------------------------------------


class TestPresetSelection:
    """Test that preset names map to the correct env class."""

    def test_default_preset_returns_microciv(self) -> None:
        env = create_game_env(board_size=6, max_turns=10, gameplay_preset="default")
        assert isinstance(env, MicroCivEnv)
        assert not isinstance(env, EnhancedMicroCivEnv)

    def test_enhanced_preset_returns_enhanced(self) -> None:
        env = create_game_env(board_size=6, max_turns=10, gameplay_preset="enhanced")
        assert isinstance(env, EnhancedMicroCivEnv)
        assert isinstance(env, MicroCivEnv)  # subclass

    def test_enhanced_has_all_systems_enabled(self) -> None:
        env = create_game_env(board_size=6, max_turns=10, gameplay_preset="enhanced")
        assert isinstance(env, EnhancedMicroCivEnv)
        cfg = env.gameplay_config
        assert cfg.enable_fog_of_war is True
        assert cfg.enable_enhanced_combat is True
        assert cfg.enable_economy is True
        assert cfg.enable_city_growth is True
        assert cfg.enable_terrain_effects is True

    def test_unknown_preset_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="Unknown gameplay_preset"):
            create_game_env(board_size=6, max_turns=10, gameplay_preset="nonexistent")

    def test_presets_tuple_completeness(self) -> None:
        """Ensure GAMEPLAY_PRESETS contains at least default and enhanced."""
        assert "default" in GAMEPLAY_PRESETS
        assert "enhanced" in GAMEPLAY_PRESETS
        assert isinstance(GAMEPLAY_PRESETS, tuple)


# ---------------------------------------------------------------------------
# Observation shape parity
# ---------------------------------------------------------------------------


class TestObservationShapes:
    """Both presets must produce identical observation shapes."""

    @pytest.fixture(params=GAMEPLAY_PRESETS)
    def env(self, request: pytest.FixtureRequest) -> MicroCivEnv:
        e = create_game_env(board_size=8, max_turns=10, gameplay_preset=request.param)
        e.reset()
        return e

    def test_spatial_shape(self, env: MicroCivEnv) -> None:
        obs = env.get_observation()
        # 13 spatial channels x board_size x board_size
        assert obs["spatial"].shape == (13, 8, 8)

    def test_vector_shape(self, env: MicroCivEnv) -> None:
        obs = env.get_observation()
        assert obs["vector"].shape == (64,)

    def test_action_mask_shape(self, env: MicroCivEnv) -> None:
        obs = env.get_observation()
        assert obs["action_mask"].shape == (376,)

    def test_both_presets_same_action_space_size(self) -> None:
        env_default = create_game_env(board_size=8, max_turns=10, gameplay_preset="default")
        env_enhanced = create_game_env(board_size=8, max_turns=10, gameplay_preset="enhanced")
        assert env_default.action_space.n == env_enhanced.action_space.n == 376


# ---------------------------------------------------------------------------
# Parameter propagation
# ---------------------------------------------------------------------------


class TestParameterPropagation:
    """Factory correctly threads board_size, max_turns, and seed."""

    @pytest.mark.parametrize("preset", GAMEPLAY_PRESETS)
    def test_board_size_propagation(self, preset: str) -> None:
        env = create_game_env(board_size=6, max_turns=10, gameplay_preset=preset)
        assert env.board_size == 6

    @pytest.mark.parametrize("preset", GAMEPLAY_PRESETS)
    def test_max_turns_propagation(self, preset: str) -> None:
        env = create_game_env(board_size=6, max_turns=15, gameplay_preset=preset)
        assert env.max_turns == 15

    @pytest.mark.parametrize("preset", GAMEPLAY_PRESETS)
    def test_seed_propagation(self, preset: str) -> None:
        """Two envs with same seed should produce identical initial states."""
        env1 = create_game_env(board_size=6, max_turns=10, gameplay_preset=preset, seed=42)
        env2 = create_game_env(board_size=6, max_turns=10, gameplay_preset=preset, seed=42)
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        import numpy as np

        np.testing.assert_array_equal(obs1["spatial"], obs2["spatial"])
        np.testing.assert_array_equal(obs1["vector"], obs2["vector"])

    @pytest.mark.parametrize("preset", GAMEPLAY_PRESETS)
    def test_no_seed_defaults_to_none(self, preset: str) -> None:
        """No seed arg should not crash."""
        env = create_game_env(board_size=6, max_turns=10, gameplay_preset=preset)
        env.reset()  # Should not raise
