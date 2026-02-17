"""Smoke tests for enhanced environment pipeline.

These tests validate the end-to-end integration of ``--gameplay_preset``
across the training pipeline.  They are designed to run quickly and catch
configuration / wiring issues before cloud deployment.

Run with::

    pytest tests/cloud/smoke/ -m smoke -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from alpha_zero.envs.env_factory import GAMEPLAY_PRESETS, create_game_env
from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.envs.microciv_enhanced import EnhancedMicroCivEnv
from alpha_zero.envs.vectorized_env import VectorizedEnvConfig, VectorizedMicroCivEnv

# ---------------------------------------------------------------------------
# Constants — avoid hard-coded magic numbers
# ---------------------------------------------------------------------------

SPATIAL_CHANNELS = 13
VECTOR_DIM = 64
ACTION_SPACE_SIZE = 376

# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestEnhancedEnvCreation:
    """Factory returns correct types and validates presets."""

    @pytest.mark.parametrize("board_size", [6, 8])
    def test_default_preset_returns_microciv(self, board_size: int) -> None:
        env = create_game_env(board_size=board_size, max_turns=10, gameplay_preset="default")
        assert isinstance(env, MicroCivEnv)
        assert not isinstance(env, EnhancedMicroCivEnv)

    @pytest.mark.parametrize("board_size", [6, 8])
    def test_enhanced_preset_returns_enhanced(self, board_size: int) -> None:
        env = create_game_env(board_size=board_size, max_turns=10, gameplay_preset="enhanced")
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

    def test_presets_registry_is_tuple(self) -> None:
        assert isinstance(GAMEPLAY_PRESETS, tuple)
        assert "default" in GAMEPLAY_PRESETS
        assert "enhanced" in GAMEPLAY_PRESETS


# ---------------------------------------------------------------------------
# Observation shape parity
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestObservationParity:
    """Both presets produce identical observation shapes."""

    @pytest.mark.parametrize("board_size", [6, 8])
    @pytest.mark.parametrize("preset", GAMEPLAY_PRESETS)
    def test_spatial_shape(self, board_size: int, preset: str) -> None:
        env = create_game_env(board_size=board_size, max_turns=10, gameplay_preset=preset)
        env.reset()
        obs = env.get_observation()
        assert obs["spatial"].shape == (SPATIAL_CHANNELS, board_size, board_size)

    @pytest.mark.parametrize("board_size", [6, 8])
    @pytest.mark.parametrize("preset", GAMEPLAY_PRESETS)
    def test_vector_shape(self, board_size: int, preset: str) -> None:
        env = create_game_env(board_size=board_size, max_turns=10, gameplay_preset=preset)
        env.reset()
        obs = env.get_observation()
        assert obs["vector"].shape == (VECTOR_DIM,)

    @pytest.mark.parametrize("board_size", [6, 8])
    @pytest.mark.parametrize("preset", GAMEPLAY_PRESETS)
    def test_action_mask_shape(self, board_size: int, preset: str) -> None:
        env = create_game_env(board_size=board_size, max_turns=10, gameplay_preset=preset)
        env.reset()
        obs = env.get_observation()
        assert obs["action_mask"].shape == (ACTION_SPACE_SIZE,)

    @pytest.mark.parametrize("preset", GAMEPLAY_PRESETS)
    def test_action_space_size(self, preset: str) -> None:
        env = create_game_env(board_size=8, max_turns=10, gameplay_preset=preset)
        assert env.action_space.n == ACTION_SPACE_SIZE


# ---------------------------------------------------------------------------
# Self-play dry-run with subprocess
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestEnhancedSelfPlayDryRun:
    """Verify selfplay_worker parse_args accepts --gameplay_preset."""

    def test_parse_args_accepts_enhanced(self) -> None:
        """CLI arg parser recognises --gameplay_preset=enhanced."""
        import sys
        from unittest.mock import patch as _patch

        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        with _patch.object(
            sys,
            "argv",
            [
                "prog",
                "--worker_id=0",
                "--model_version=0",
                "--iteration=1",
                "--games=2",
                "--gameplay_preset=enhanced",
            ],
        ):
            args = parse_args()
        assert args.gameplay_preset == "enhanced"

    def test_parse_args_defaults_to_default(self) -> None:
        """Omitting --gameplay_preset defaults to 'default'."""
        import sys
        from unittest.mock import patch as _patch

        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        with _patch.object(
            sys,
            "argv",
            ["prog", "--worker_id=0", "--model_version=0", "--iteration=1", "--games=2"],
        ):
            args = parse_args()
        assert args.gameplay_preset == "default"


# ---------------------------------------------------------------------------
# Arena integration (mock-based)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestEnhancedArena:
    """Arena correctly forwards gameplay_preset to create_game_env."""

    @patch("alpha_zero.core.mcts_factory.create_mcts")
    @patch("alpha_zero.envs.env_factory.create_game_env")
    def test_arena_passes_enhanced_preset(
        self, mock_create_env: MagicMock, mock_create_mcts: MagicMock
    ) -> None:
        from alpha_zero.cloud.gcp.arena import run_arena

        # Configure mock env
        mock_env = MagicMock()
        mock_env.current_player = 0
        mock_env.reset.return_value = ({"spatial": np.zeros((13, 8, 8))}, {})
        mock_env.step.return_value = (
            {"spatial": np.zeros((13, 8, 8))},
            0.0,
            True,  # terminated
            False,
            {},
        )
        mock_create_env.return_value = mock_env

        # Configure mock MCTS
        mock_mcts = MagicMock()
        action_probs = np.zeros(ACTION_SPACE_SIZE)
        action_probs[0] = 1.0
        mock_mcts.search.return_value = (action_probs, 0.5)
        mock_mcts.select_action.return_value = 0
        mock_create_mcts.return_value = mock_mcts

        model_a = MagicMock(spec=torch.nn.Module)
        model_b = MagicMock(spec=torch.nn.Module)

        run_arena(
            model_a=model_a,
            model_b=model_b,
            num_games=1,
            mcts_type="gumbel",
            mcts_sims=4,
            board_size=8,
            max_turns=10,
            gameplay_preset="enhanced",
        )

        # Verify create_game_env was called with enhanced preset
        mock_create_env.assert_called()
        call_kwargs = mock_create_env.call_args[1]
        assert call_kwargs.get("gameplay_preset") == "enhanced"
        assert call_kwargs.get("board_size") == 8


# ---------------------------------------------------------------------------
# Vectorized env with enhanced preset
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestVectorizedEnhanced:
    """VectorizedMicroCivEnv with enhanced preset creates correct instances."""

    def test_enhanced_env_types(self) -> None:
        cfg = VectorizedEnvConfig(
            num_envs=2, board_size=6, max_turns=10, gameplay_preset="enhanced"
        )
        vec_env = VectorizedMicroCivEnv(config=cfg)
        for i in range(cfg.num_envs):
            assert isinstance(vec_env.get_env(i), EnhancedMicroCivEnv)

    def test_default_env_types(self) -> None:
        cfg = VectorizedEnvConfig(num_envs=2, board_size=6, max_turns=10, gameplay_preset="default")
        vec_env = VectorizedMicroCivEnv(config=cfg)
        for i in range(cfg.num_envs):
            assert isinstance(vec_env.get_env(i), MicroCivEnv)
            assert not isinstance(vec_env.get_env(i), EnhancedMicroCivEnv)

    def test_enhanced_observation_shapes(self) -> None:
        num_envs = 3
        board_size = 6
        cfg = VectorizedEnvConfig(
            num_envs=num_envs, board_size=board_size, max_turns=10, gameplay_preset="enhanced"
        )
        vec_env = VectorizedMicroCivEnv(config=cfg)
        vec_env.reset_all()

        spatial, vector, mask = vec_env.get_observations()
        assert spatial.shape == (num_envs, SPATIAL_CHANNELS, board_size, board_size)
        assert vector.shape == (num_envs, VECTOR_DIM)
        assert mask.shape == (num_envs, ACTION_SPACE_SIZE)

    def test_enhanced_step_all(self) -> None:
        """Smoke test: reset + step works end-to-end with enhanced env."""
        num_envs = 2
        cfg = VectorizedEnvConfig(
            num_envs=num_envs, board_size=6, max_turns=10, gameplay_preset="enhanced"
        )
        vec_env = VectorizedMicroCivEnv(config=cfg)
        vec_env.reset_all()

        _, _, mask = vec_env.get_observations()
        actions = np.zeros(num_envs, dtype=np.int32)
        for i in range(num_envs):
            valid = np.where(mask[i] > 0)[0]
            if len(valid) > 0:
                actions[i] = valid[0]

        rewards, terminated, truncated, infos = vec_env.step_all(actions)
        assert rewards.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)

    def test_invalid_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown gameplay_preset"):
            VectorizedEnvConfig(num_envs=2, board_size=6, max_turns=10, gameplay_preset="invalid")


# ---------------------------------------------------------------------------
# GCS path isolation
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestGCSPathIsolation:
    """GCS paths with agent_id must not collide with production."""

    def test_agent_id_isolates_model_path(self) -> None:
        from alpha_zero.cloud.gcp.config import GCSPathConfig

        isolated = GCSPathConfig(agent_id="enhanced-smoke-v1")
        prod = GCSPathConfig()
        assert isolated.model_path(1) != prod.model_path(1)
        assert "enhanced-smoke-v1" in isolated.model_path(1)

    def test_agent_id_isolates_trajectory_path(self) -> None:
        from alpha_zero.cloud.gcp.config import GCSPathConfig

        isolated = GCSPathConfig(agent_id="enhanced-v1")
        prod = GCSPathConfig()
        assert isolated.trajectory_path(1, 0) != prod.trajectory_path(1, 0)
        assert "enhanced-v1" in isolated.trajectory_path(1, 0)

    def test_agent_id_isolates_checkpoint_path(self) -> None:
        from alpha_zero.cloud.gcp.config import GCSPathConfig

        isolated = GCSPathConfig(agent_id="enhanced-v1")
        prod = GCSPathConfig()
        assert isolated.checkpoint_path(100) != prod.checkpoint_path(100)
        assert "enhanced-v1" in isolated.checkpoint_path(100)


# ---------------------------------------------------------------------------
# Orchestrator extra-args propagation
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestOrchestratorPropagation:
    """Orchestrator propagates gameplay_preset to extra args."""

    def test_enhanced_propagated_to_selfplay(self) -> None:
        import argparse

        from alpha_zero.cloud.gcp.vertex_orchestrator import build_selfplay_extra_args

        args = argparse.Namespace(
            mcts_type="gumbel",
            mcts_simulations=8,
            gumbel_num_actions=16,
            max_turns=30,
            board_size=8,
            action_space_size=ACTION_SPACE_SIZE,
            num_res_blocks=6,
            num_filters=64,
            num_heads=8,
            policy_hidden_dim=128,
            value_head_dim=64,
            num_parallel_games=1,
            max_batch_size=64,
            reward_shaping=False,
            reward_scale=1.0,
            algorithm="alphazero",
            gameplay_preset="enhanced",
        )
        extra = build_selfplay_extra_args(args)
        assert "--gameplay_preset=enhanced" in extra

    def test_default_not_in_selfplay_args(self) -> None:
        import argparse

        from alpha_zero.cloud.gcp.vertex_orchestrator import build_selfplay_extra_args

        args = argparse.Namespace(
            mcts_type="gumbel",
            mcts_simulations=8,
            gumbel_num_actions=16,
            max_turns=30,
            board_size=8,
            action_space_size=ACTION_SPACE_SIZE,
            num_res_blocks=6,
            num_filters=64,
            num_heads=8,
            policy_hidden_dim=128,
            value_head_dim=64,
            num_parallel_games=1,
            max_batch_size=64,
            reward_shaping=False,
            reward_scale=1.0,
            algorithm="alphazero",
            gameplay_preset="default",
        )
        extra = build_selfplay_extra_args(args)
        assert not any("gameplay_preset" in a for a in extra)

    def test_enhanced_propagated_to_evaluation(self) -> None:
        import argparse

        from alpha_zero.cloud.gcp.vertex_orchestrator import build_evaluation_extra_args

        args = argparse.Namespace(
            mcts_type="gumbel",
            mcts_simulations=8,
            gumbel_num_actions=16,
            max_turns=30,
            board_size=8,
            action_space_size=ACTION_SPACE_SIZE,
            num_res_blocks=6,
            num_filters=64,
            num_heads=8,
            policy_hidden_dim=128,
            value_head_dim=64,
            gameplay_preset="enhanced",
        )
        extra = build_evaluation_extra_args(args)
        assert "--gameplay_preset=enhanced" in extra
