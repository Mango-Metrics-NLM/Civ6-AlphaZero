"""
Unit tests for orchestrator MuZero configuration.

Tests that:
- build_selfplay_extra_args produces --mcts_type=muzero when algorithm=muzero
- W&B env vars propagate through _wandb_env_vars
"""

from __future__ import annotations

import argparse
import os
import sys
from unittest.mock import patch

import pytest


# =============================================================================
# build_selfplay_extra_args: MuZero override
# =============================================================================


class TestBuildSelfplayExtraArgs:
    """Tests for build_selfplay_extra_args MuZero routing."""

    def _make_args(self, **overrides: object) -> argparse.Namespace:
        """Create a minimal args namespace for build_selfplay_extra_args."""
        defaults = {
            "mcts_type": "gumbel",
            "mcts_simulations": 50,
            "gumbel_num_actions": 16,
            "max_turns": 50,
            "board_size": 8,
            "action_space_size": 376,
            "num_res_blocks": 6,
            "num_filters": 64,
            "reward_shaping": False,
            "reward_scale": 1.0,
            "algorithm": "alphazero",
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_muzero_algorithm_overrides_mcts_type(self) -> None:
        """When algorithm=muzero, mcts_type should be overridden to muzero."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import build_selfplay_extra_args

        args = self._make_args(algorithm="muzero", mcts_type="gumbel")
        extra = build_selfplay_extra_args(args)

        # Should contain --mcts_type=muzero, NOT --mcts_type=gumbel
        mcts_type_args = [a for a in extra if a.startswith("--mcts_type=")]
        assert len(mcts_type_args) == 1
        assert mcts_type_args[0] == "--mcts_type=muzero"

    def test_alphazero_algorithm_keeps_gumbel_type(self) -> None:
        """When algorithm=alphazero, mcts_type should stay as configured."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import build_selfplay_extra_args

        args = self._make_args(algorithm="alphazero", mcts_type="gumbel")
        extra = build_selfplay_extra_args(args)

        mcts_type_args = [a for a in extra if a.startswith("--mcts_type=")]
        assert len(mcts_type_args) == 1
        assert mcts_type_args[0] == "--mcts_type=gumbel"

    def test_extra_args_include_mcts_simulations(self) -> None:
        """MCTS simulations count should be passed through."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import build_selfplay_extra_args

        args = self._make_args(mcts_simulations=50)
        extra = build_selfplay_extra_args(args)

        sim_args = [a for a in extra if a.startswith("--mcts_simulations=")]
        assert len(sim_args) == 1
        assert sim_args[0] == "--mcts_simulations=50"


# =============================================================================
# W&B env var propagation
# =============================================================================


class TestWandbEnvVarPropagation:
    """Tests for W&B environment variable propagation to child jobs."""

    def test_wandb_api_key_propagated_when_set(self) -> None:
        """WANDB_API_KEY should propagate when set in environment."""
        from alpha_zero.cloud.gcp.vertex_jobs import _wandb_env_vars

        with patch.dict(os.environ, {"WANDB_API_KEY": "test-key-123"}):
            env_vars = _wandb_env_vars()

        assert "WANDB_API_KEY" in env_vars
        assert env_vars["WANDB_API_KEY"] == "test-key-123"

    def test_wandb_api_key_excluded_when_empty(self) -> None:
        """WANDB_API_KEY should NOT be included when empty."""
        from alpha_zero.cloud.gcp.vertex_jobs import _wandb_env_vars

        env_without_key = {k: v for k, v in os.environ.items() if k != "WANDB_API_KEY"}
        with patch.dict(os.environ, env_without_key, clear=True):
            env_vars = _wandb_env_vars()

        assert "WANDB_API_KEY" not in env_vars

    def test_wandb_enabled_defaults_to_false(self) -> None:
        """WANDB_ENABLED should default to 'false' when not set."""
        from alpha_zero.cloud.gcp.vertex_jobs import _wandb_env_vars

        env_clean = {k: v for k, v in os.environ.items()
                     if not k.startswith("WANDB_")}
        with patch.dict(os.environ, env_clean, clear=True):
            env_vars = _wandb_env_vars()

        assert env_vars.get("WANDB_ENABLED") == "false"

    def test_wandb_project_defaults_to_civ6_alphazero(self) -> None:
        """WANDB_PROJECT should default to 'civ6az-muzero-world-model'."""
        from alpha_zero.cloud.gcp.vertex_jobs import _wandb_env_vars

        env_clean = {k: v for k, v in os.environ.items()
                     if not k.startswith("WANDB_")}
        with patch.dict(os.environ, env_clean, clear=True):
            env_vars = _wandb_env_vars()

        assert env_vars.get("WANDB_PROJECT") == "civ6az-muzero-world-model"


# =============================================================================
# --model_version_start override flag
# =============================================================================


class TestModelVersionStartFlag:
    """Tests for the `--model_version_start` orchestrator CLI flag.

    This flag was added to let MuZero runs start from a fresh random init
    (version 0) instead of falling back to ``get_initial_model_version()``,
    which auto-detects the latest checkpoint in GCS — incompatible with
    MuZero when the latest checkpoint is an AlphaZero/ViT model.
    """

    def test_flag_defaults_to_none(self) -> None:
        """When the flag is not supplied, parse_args returns ``None``.

        ``None`` is the sentinel that preserves the legacy auto-detect
        path in vertex_orchestrator.main(); changing the default would
        silently break existing runs.
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        # vertex_orchestrator.parse_args has no required positional args and
        # no --config flag; defaults populate everything.
        test_args = ["vertex_orchestrator"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.model_version_start is None

    def test_flag_accepts_zero_for_fresh_init(self) -> None:
        """``--model_version_start=0`` should round-trip as integer 0 (not falsy None)."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator", "--model_version_start=0"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.model_version_start == 0
        # Critical: must distinguish from None so the override branch fires.
        assert args.model_version_start is not None

    def test_flag_accepts_positive_int(self) -> None:
        """``--model_version_start=42`` should round-trip as integer 42."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator", "--model_version_start=42"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()
        assert args.model_version_start == 42

    def test_flag_rejects_non_integer(self) -> None:
        """Non-integer values should fail argparse type conversion."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator", "--model_version_start=latest"]
        with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
            parse_args()
