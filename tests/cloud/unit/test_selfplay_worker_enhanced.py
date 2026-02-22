"""Tests for gameplay_preset integration in selfplay_worker and vectorized_env."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.envs.vectorized_env import VectorizedEnvConfig, VectorizedMicroCivEnv

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestParseArgsGameplayPreset:
    """Verify --gameplay_preset is accepted by selfplay_worker.parse_args."""

    def test_parse_args_gameplay_preset_default(self) -> None:
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        with patch.object(
            sys,
            "argv",
            ["prog", "--worker_id=0", "--model_version=0", "--iteration=1", "--games=2"],
        ):
            args = parse_args()
        assert args.gameplay_preset == "default"

    def test_parse_args_gameplay_preset_enhanced(self) -> None:
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        with patch.object(
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

    def test_parse_args_gameplay_preset_invalid(self) -> None:
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "--worker_id=0",
                "--model_version=0",
                "--iteration=1",
                "--games=2",
                "--gameplay_preset=bogus",
            ],
        ):
            with pytest.raises(SystemExit):
                parse_args()


# ---------------------------------------------------------------------------
# VectorizedEnvConfig
# ---------------------------------------------------------------------------


class TestVectorizedEnvConfigEnhanced:
    """VectorizedEnvConfig accepts and validates gameplay_preset."""

    def test_default_gameplay_preset(self) -> None:
        cfg = VectorizedEnvConfig(num_envs=2, board_size=6, max_turns=10)
        assert cfg.gameplay_preset == "default"

    def test_enhanced_gameplay_preset(self) -> None:
        cfg = VectorizedEnvConfig(
            num_envs=2, board_size=6, max_turns=10, gameplay_preset="enhanced"
        )
        assert cfg.gameplay_preset == "enhanced"

    def test_invalid_gameplay_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown gameplay_preset"):
            VectorizedEnvConfig(num_envs=2, board_size=6, max_turns=10, gameplay_preset="invalid")


# ---------------------------------------------------------------------------
# VectorizedMicroCivEnv with enhanced preset
# ---------------------------------------------------------------------------


class TestVectorizedEnvEnhanced:
    """VectorizedMicroCivEnv creates EnhancedMicroCivEnv instances when preset=enhanced."""

    def test_enhanced_env_creation(self) -> None:
        from alpha_zero.envs.microciv_enhanced import EnhancedMicroCivEnv

        cfg = VectorizedEnvConfig(
            num_envs=2, board_size=6, max_turns=10, gameplay_preset="enhanced"
        )
        vec_env = VectorizedMicroCivEnv(config=cfg)
        # All individual envs should be EnhancedMicroCivEnv
        for i in range(cfg.num_envs):
            assert isinstance(vec_env.get_env(i), EnhancedMicroCivEnv)

    def test_default_env_creation(self) -> None:
        from alpha_zero.envs.microciv_enhanced import EnhancedMicroCivEnv

        cfg = VectorizedEnvConfig(num_envs=2, board_size=6, max_turns=10, gameplay_preset="default")
        vec_env = VectorizedMicroCivEnv(config=cfg)
        for i in range(cfg.num_envs):
            assert isinstance(vec_env.get_env(i), MicroCivEnv)
            assert not isinstance(vec_env.get_env(i), EnhancedMicroCivEnv)

    def test_enhanced_step_all(self) -> None:
        """Smoke test: reset + step with enhanced env works end-to-end."""
        import numpy as np

        cfg = VectorizedEnvConfig(
            num_envs=2, board_size=6, max_turns=10, gameplay_preset="enhanced"
        )
        vec_env = VectorizedMicroCivEnv(config=cfg)
        vec_env.reset_all()

        spatial, vector, mask = vec_env.get_observations()
        assert spatial.shape[0] == 2
        assert vector.shape[0] == 2
        assert mask.shape[0] == 2

        # Take a valid action for each env
        actions = np.zeros(2, dtype=np.int32)
        for i in range(2):
            valid = np.where(mask[i] > 0)[0]
            if len(valid) > 0:
                actions[i] = valid[0]

        rewards, terminated, truncated, infos = vec_env.step_all(actions)
        assert rewards.shape == (2,)
        assert terminated.shape == (2,)
        assert truncated.shape == (2,)
        assert len(infos) == 2


# ---------------------------------------------------------------------------
# Orchestrator extra args propagation
# ---------------------------------------------------------------------------


class TestOrchestratorExtraArgs:
    """build_selfplay_extra_args and build_evaluation_extra_args propagate gameplay_preset."""

    def test_selfplay_extra_args_default(self) -> None:
        """Default preset should NOT appear in extra args (backward compat)."""
        import argparse

        from alpha_zero.cloud.gcp.vertex_orchestrator import build_selfplay_extra_args

        args = argparse.Namespace(
            mcts_type="gumbel",
            mcts_simulations=8,
            gumbel_num_actions=16,
            max_turns=30,
            board_size=8,
            action_space_size=376,
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

    def test_selfplay_extra_args_enhanced(self) -> None:
        """Enhanced preset SHOULD appear in extra args."""
        import argparse

        from alpha_zero.cloud.gcp.vertex_orchestrator import build_selfplay_extra_args

        args = argparse.Namespace(
            mcts_type="gumbel",
            mcts_simulations=8,
            gumbel_num_actions=16,
            max_turns=30,
            board_size=8,
            action_space_size=376,
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

    def test_evaluation_extra_args_enhanced(self) -> None:
        """Enhanced preset propagated to evaluation extra args."""
        import argparse

        from alpha_zero.cloud.gcp.vertex_orchestrator import build_evaluation_extra_args

        args = argparse.Namespace(
            mcts_type="gumbel",
            mcts_simulations=8,
            gumbel_num_actions=16,
            max_turns=30,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            num_heads=8,
            policy_hidden_dim=128,
            value_head_dim=64,
            gameplay_preset="enhanced",
        )
        extra = build_evaluation_extra_args(args)
        assert "--gameplay_preset=enhanced" in extra

    def test_evaluation_extra_args_default_omitted(self) -> None:
        """Default preset should NOT appear in evaluation extra args."""
        import argparse

        from alpha_zero.cloud.gcp.vertex_orchestrator import build_evaluation_extra_args

        args = argparse.Namespace(
            mcts_type="gumbel",
            mcts_simulations=8,
            gumbel_num_actions=16,
            max_turns=30,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            num_heads=8,
            policy_hidden_dim=128,
            value_head_dim=64,
            gameplay_preset="default",
        )
        extra = build_evaluation_extra_args(args)
        assert not any("gameplay_preset" in a for a in extra)
