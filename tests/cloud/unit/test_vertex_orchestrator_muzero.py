"""
Unit tests for vertex_orchestrator MuZero algorithm flag.

Tests that --algorithm=muzero is correctly parsed and propagated to
the self-play and training worker extra_args builders.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

# =============================================================================
# parse_args — algorithm flag Tests
# =============================================================================


class TestParseArgsAlgorithm:
    """Test that --algorithm flag is accepted by orchestrator."""

    def test_algorithm_muzero_accepted(self) -> None:
        """Test parsing with --algorithm=muzero."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = [
            "vertex_orchestrator",
            "--algorithm=muzero",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.algorithm == "muzero"

    def test_algorithm_alphazero_accepted(self) -> None:
        """Test parsing with --algorithm=alphazero."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = [
            "vertex_orchestrator",
            "--algorithm=alphazero",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.algorithm == "alphazero"

    def test_algorithm_default_is_alphazero(self) -> None:
        """Test that default algorithm is alphazero."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.algorithm == "alphazero"

    def test_algorithm_invalid_rejected(self) -> None:
        """Test that an invalid algorithm name is rejected."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = [
            "vertex_orchestrator",
            "--algorithm=dreamer",
        ]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                parse_args()


# =============================================================================
# build_selfplay_extra_args — MuZero mcts_type override
# =============================================================================


class TestBuildSelfplayExtraArgs:
    """Test that build_selfplay_extra_args overrides mcts_type for MuZero."""

    def test_muzero_overrides_mcts_type(self) -> None:
        """When algorithm=muzero, mcts_type should be set to muzero."""
        from argparse import Namespace

        from alpha_zero.cloud.gcp.vertex_orchestrator import build_selfplay_extra_args

        args = Namespace(
            mcts_type="civ6",
            mcts_simulations=400,
            gumbel_num_actions=16,
            max_turns=30,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            reward_shaping=False,
            reward_scale=1.0,
            algorithm="muzero",
        )

        extra = build_selfplay_extra_args(args)

        # Should contain muzero, not civ6
        assert "--mcts_type=muzero" in extra
        assert "--mcts_type=civ6" not in extra

    def test_alphazero_preserves_mcts_type(self) -> None:
        """When algorithm=alphazero, mcts_type should be preserved."""
        from argparse import Namespace

        from alpha_zero.cloud.gcp.vertex_orchestrator import build_selfplay_extra_args

        args = Namespace(
            mcts_type="gumbel",
            mcts_simulations=400,
            gumbel_num_actions=16,
            max_turns=30,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            reward_shaping=False,
            reward_scale=1.0,
            algorithm="alphazero",
        )

        extra = build_selfplay_extra_args(args)

        assert "--mcts_type=gumbel" in extra
        assert "--mcts_type=muzero" not in extra

    def test_reward_shaping_preserved_with_muzero(self) -> None:
        """Reward shaping args should still work alongside muzero override."""
        from argparse import Namespace

        from alpha_zero.cloud.gcp.vertex_orchestrator import build_selfplay_extra_args

        args = Namespace(
            mcts_type="civ6",
            mcts_simulations=400,
            gumbel_num_actions=16,
            max_turns=30,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            reward_shaping=True,
            reward_scale=0.5,
            algorithm="muzero",
        )

        extra = build_selfplay_extra_args(args)

        assert "--mcts_type=muzero" in extra
        assert "--reward_shaping" in extra
        assert "--reward_scale=0.5" in extra


# =============================================================================
# build_training_extra_args — MuZero algorithm pass-through
# =============================================================================


class TestBuildTrainingExtraArgs:
    """Test that build_training_extra_args passes --algorithm=muzero."""

    def test_muzero_passes_algorithm_flag(self) -> None:
        """When algorithm=muzero, --algorithm=muzero should be in extra args."""
        from argparse import Namespace

        from alpha_zero.cloud.gcp.vertex_orchestrator import build_training_extra_args

        args = Namespace(
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            lr_schedule_type="constant",
            weight_decay=0.01,
            value_lr_multiplier=1.0,
            policy_lr_multiplier=1.0,
            algorithm="muzero",
        )

        extra = build_training_extra_args(args)

        assert "--algorithm=muzero" in extra

    def test_alphazero_no_algorithm_flag(self) -> None:
        """When algorithm=alphazero, --algorithm should NOT be in extra args."""
        from argparse import Namespace

        from alpha_zero.cloud.gcp.vertex_orchestrator import build_training_extra_args

        args = Namespace(
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            lr_schedule_type="constant",
            weight_decay=0.01,
            value_lr_multiplier=1.0,
            policy_lr_multiplier=1.0,
            algorithm="alphazero",
        )

        extra = build_training_extra_args(args)

        # No --algorithm flag for default alphazero (backwards compat)
        assert all("--algorithm" not in a for a in extra)

    def test_lr_schedule_with_muzero(self) -> None:
        """LR schedule args should still work alongside muzero."""
        from argparse import Namespace

        from alpha_zero.cloud.gcp.vertex_orchestrator import build_training_extra_args

        args = Namespace(
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
            lr_schedule_type="cosine",
            lr_initial=0.002,
            lr_final=0.0001,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
            weight_decay=0.01,
            value_lr_multiplier=1.0,
            policy_lr_multiplier=1.0,
            algorithm="muzero",
        )

        extra = build_training_extra_args(args)

        assert "--algorithm=muzero" in extra
        assert "--lr_schedule_type=cosine" in extra
        assert "--lr_initial=0.002" in extra
