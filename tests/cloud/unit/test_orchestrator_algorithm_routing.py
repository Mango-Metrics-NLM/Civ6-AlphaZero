"""Tests for vertex_orchestrator algorithm routing.

Verifies that the --algorithm flag is correctly accepted and propagated
to selfplay and training extra args. Includes backwards-compatibility
regression guards.
"""

from __future__ import annotations

import argparse
from typing import Any
from unittest.mock import patch

import pytest

from alpha_zero.cloud.gcp.vertex_orchestrator import (
    build_evaluation_extra_args,
    build_selfplay_extra_args,
    build_training_extra_args,
    parse_args,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(**overrides: Any) -> argparse.Namespace:
    """Create a Namespace with sensible defaults and any overrides."""
    defaults: dict[str, Any] = {
        "mcts_type": "gumbel",
        "mcts_simulations": 64,
        "gumbel_num_actions": 16,
        "max_turns": 50,
        "board_size": 8,
        "action_space_size": 376,
        "num_res_blocks": 6,
        "num_filters": 64,
        "num_heads": 8,
        "policy_hidden_dim": 128,
        "value_head_dim": 64,
        "num_parallel_games": 1,
        "max_batch_size": 64,
        "reward_shaping": False,
        "reward_scale": 1.0,
        "algorithm": "alphazero",
        "gameplay_preset": "default",
        "encoder_type": "patch_embed",
        # LR schedule defaults
        "lr_schedule_type": "constant",
        "lr_initial": 0.001,
        "lr_final": 0.0001,
        "lr_warmup_steps": 0,
        "lr_decay_steps": 0,
        "weight_decay": 0.01,
        "value_lr_multiplier": 1.0,
        "policy_lr_multiplier": 1.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# parse_args flag acceptance
# ---------------------------------------------------------------------------


class TestParseArgsAlgorithmFlag:
    """Verify that parse_args accepts --algorithm with valid choices."""

    def test_algorithm_muzero_accepted(self) -> None:
        with patch("sys.argv", ["prog", "--algorithm=muzero", "--dry_run"]):
            args = parse_args()
        assert args.algorithm == "muzero"

    def test_algorithm_alphazero_accepted(self) -> None:
        with patch("sys.argv", ["prog", "--algorithm=alphazero", "--dry_run"]):
            args = parse_args()
        assert args.algorithm == "alphazero"

    def test_algorithm_default_is_alphazero(self) -> None:
        with patch("sys.argv", ["prog", "--dry_run"]):
            args = parse_args()
        assert args.algorithm == "alphazero"

    def test_algorithm_invalid_rejected(self) -> None:
        with patch("sys.argv", ["prog", "--algorithm=random"]):
            with pytest.raises(SystemExit):
                parse_args()


# ---------------------------------------------------------------------------
# build_selfplay_extra_args routing
# ---------------------------------------------------------------------------


class TestBuildSelfplayAlgorithmRouting:
    """Verify selfplay extra args correctly route MuZero vs AlphaZero."""

    def test_muzero_replaces_mcts_type(self) -> None:
        """When algorithm=muzero, mcts_type should be overridden to muzero."""
        args = _make_args(algorithm="muzero")
        extra = build_selfplay_extra_args(args)
        assert "--mcts_type=muzero" in extra
        # Original mcts_type should NOT appear
        assert "--mcts_type=gumbel" not in extra

    def test_alphazero_preserves_mcts_type(self) -> None:
        """When algorithm=alphazero, original mcts_type is kept."""
        args = _make_args(algorithm="alphazero", mcts_type="gumbel")
        extra = build_selfplay_extra_args(args)
        assert "--mcts_type=gumbel" in extra
        assert "--mcts_type=muzero" not in extra

    def test_alphazero_civ6_mcts_preserved(self) -> None:
        """civ6 MCTS type is preserved for alphazero."""
        args = _make_args(algorithm="alphazero", mcts_type="civ6")
        extra = build_selfplay_extra_args(args)
        assert "--mcts_type=civ6" in extra

    def test_muzero_with_reward_shaping(self) -> None:
        """MuZero can work with reward shaping enabled."""
        args = _make_args(algorithm="muzero", reward_shaping=True, reward_scale=0.5)
        extra = build_selfplay_extra_args(args)
        assert "--mcts_type=muzero" in extra
        assert "--reward_shaping" in extra
        assert "--reward_scale=0.5" in extra

    def test_backwards_compat_no_algorithm_attr(self) -> None:
        """If algorithm attr is missing, defaults to alphazero (backward compat)."""
        args = _make_args()
        delattr(args, "algorithm")
        extra = build_selfplay_extra_args(args)
        # Should NOT have muzero
        assert "--mcts_type=muzero" not in extra


# ---------------------------------------------------------------------------
# build_training_extra_args routing
# ---------------------------------------------------------------------------


class TestBuildTrainingAlgorithmRouting:
    """Verify training extra args correctly route MuZero vs AlphaZero."""

    def test_muzero_passes_algorithm_flag(self) -> None:
        """MuZero training should include --algorithm=muzero."""
        args = _make_args(algorithm="muzero")
        extra = build_training_extra_args(args)
        assert "--algorithm=muzero" in extra

    def test_alphazero_no_algorithm_flag(self) -> None:
        """AlphaZero should NOT include --algorithm=muzero."""
        args = _make_args(algorithm="alphazero")
        extra = build_training_extra_args(args)
        assert "--algorithm=muzero" not in extra

    def test_backwards_compat_no_algorithm_attr(self) -> None:
        """If algorithm attr is missing, defaults to alphazero."""
        args = _make_args()
        delattr(args, "algorithm")
        extra = build_training_extra_args(args)
        assert "--algorithm=muzero" not in extra

    def test_muzero_with_lr_schedule(self) -> None:
        """MuZero with LR schedule should include both sets of flags."""
        args = _make_args(
            algorithm="muzero",
            lr_schedule_type="cosine",
            lr_initial=0.003,
            lr_final=0.0001,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
        )
        extra = build_training_extra_args(args)
        assert "--algorithm=muzero" in extra
        assert "--lr_schedule_type=cosine" in extra
        assert "--lr_initial=0.003" in extra


# ---------------------------------------------------------------------------
# build_evaluation_extra_args
# ---------------------------------------------------------------------------


class TestBuildEvaluationRouting:
    """Verify eval extra args include MuZero mcts_type when appropriate."""

    def test_eval_uses_original_mcts_type(self) -> None:
        """Evaluation uses the args.mcts_type directly (not overridden by algorithm)."""
        args = _make_args(algorithm="muzero", mcts_type="gumbel")
        extra = build_evaluation_extra_args(args)
        # eval currently does NOT route algorithm, it uses mcts_type as-is
        assert "--mcts_type=gumbel" in extra
