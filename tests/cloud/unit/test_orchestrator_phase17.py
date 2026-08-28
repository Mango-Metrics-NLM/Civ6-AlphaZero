"""Tests for Phase 17 orchestrator changes.

Covers:
- Architecture preset CLI arg handling
- Updated training parameter defaults
- New architecture fields in build_*_extra_args
- Vectorized env args passthrough
- W&B config logging enhancements
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from alpha_zero.cloud.gcp.vertex_orchestrator import (
    build_evaluation_extra_args,
    build_selfplay_extra_args,
    build_training_extra_args,
    parse_args,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_args() -> argparse.Namespace:
    """Create baseline orchestrator args for tests (not CLI defaults).

    Note: ``num_parallel_games`` is set to 1 (CLI default is 4) to ensure
    deterministic sequential behaviour in unit tests.
    """
    return argparse.Namespace(
        iterations=3,
        start_iteration=0,
        workers=2,
        games_per_worker=100,
        epochs=5,
        trajectory_window=5,
        max_retries=1,
        board_size=8,
        action_space_size=376,
        num_res_blocks=6,
        num_filters=64,
        num_heads=8,
        policy_hidden_dim=128,
        value_head_dim=64,
        architecture_preset=None,
        mcts_type="gumbel",
        mcts_simulations=400,
        gumbel_num_actions=16,
        max_turns=50,
        num_parallel_games=1,  # Deliberately set to 1 for sequential behavior tests
        max_batch_size=64,
        evaluation_games=20,
        evaluation_win_threshold=0.55,
        curriculum=False,
        curriculum_stages=3,
        reward_shaping=False,
        reward_scale=1.0,
        lr_schedule_type="constant",
        lr_initial=0.001,
        lr_final=0.0001,
        lr_warmup_steps=1000,
        lr_decay_steps=100000,
        weight_decay=0.01,
        value_lr_multiplier=1.0,
        policy_lr_multiplier=1.0,
        dry_run=True,
    )


@pytest.fixture
def medium_args(base_args: argparse.Namespace) -> argparse.Namespace:
    """Args with medium architecture preset applied."""
    base_args.architecture_preset = "medium"
    base_args.num_res_blocks = 8
    base_args.num_filters = 256
    base_args.num_heads = 8
    base_args.policy_hidden_dim = 256
    base_args.value_head_dim = 128
    return base_args


# =============================================================================
# Default Value Tests (Phase 17 changes)
# =============================================================================


class TestPhase17Defaults:
    """Verify Phase 17 default value changes in parse_args."""

    @staticmethod
    def _parse_defaults() -> argparse.Namespace:
        """Parse args with empty argv to get defaults."""
        with patch("sys.argv", ["orchestrator"]):
            return parse_args()

    def test_default_games_per_worker(self) -> None:
        """games_per_worker default changed from 50 to 100."""
        args = self._parse_defaults()
        assert args.games_per_worker == 100

    def test_default_max_turns(self) -> None:
        """max_turns default changed from 30 to 50."""
        args = self._parse_defaults()
        assert args.max_turns == 50

    def test_default_trajectory_window(self) -> None:
        """trajectory_window default changed from 3 to 5."""
        args = self._parse_defaults()
        assert args.trajectory_window == 5

    def test_default_evaluation_games(self) -> None:
        """evaluation_games default changed from 0 to 20."""
        args = self._parse_defaults()
        assert args.evaluation_games == 20

    def test_default_mcts_simulations(self) -> None:
        """mcts_simulations default is 400."""
        args = self._parse_defaults()
        assert args.mcts_simulations == 400

    def test_default_architecture_preset_is_none(self) -> None:
        """architecture_preset default is None (no override)."""
        args = self._parse_defaults()
        assert args.architecture_preset is None

    def test_default_num_heads(self) -> None:
        """num_heads default is 8."""
        args = self._parse_defaults()
        assert args.num_heads == 8

    def test_default_policy_hidden_dim(self) -> None:
        """policy_hidden_dim default is 128."""
        args = self._parse_defaults()
        assert args.policy_hidden_dim == 128

    def test_default_value_head_dim(self) -> None:
        """value_head_dim default is 64."""
        args = self._parse_defaults()
        assert args.value_head_dim == 64

    def test_default_num_parallel_games(self) -> None:
        """num_parallel_games defaults to 1 (sequential).

        It was 4, which combined with the default ``--mcts_type=gumbel`` to fail
        the self-play worker's batch dispatch gate: the flag was forwarded to
        every worker and then discarded, so the default run advertised
        vectorized self-play and ran sequentially. 1 is what those runs did.
        Vectorized self-play is opt-in via ``--mcts_type=batch_gumbel``.
        """
        args = self._parse_defaults()
        assert args.num_parallel_games == 1


# =============================================================================
# Architecture Fields in build_*_extra_args
# =============================================================================


class TestBuildExtraArgsArchitecture:
    """Test new architecture fields in build_*_extra_args."""

    def test_selfplay_default_omits_arch_fields(self, base_args: argparse.Namespace) -> None:
        """Default arch values should NOT emit extra args (backwards compat)."""
        extra = build_selfplay_extra_args(base_args)
        assert "--num_heads=" not in " ".join(extra)
        assert "--policy_hidden_dim=" not in " ".join(extra)
        assert "--value_head_dim=" not in " ".join(extra)

    def test_selfplay_medium_includes_arch_fields(self, medium_args: argparse.Namespace) -> None:
        """Medium preset values should be passed through."""
        extra = build_selfplay_extra_args(medium_args)
        assert "--policy_hidden_dim=256" in extra
        assert "--value_head_dim=128" in extra

    def test_training_default_omits_arch_fields(self, base_args: argparse.Namespace) -> None:
        """Default arch values should NOT emit extra args."""
        extra = build_training_extra_args(base_args)
        assert "--num_heads=" not in " ".join(extra)
        assert "--policy_hidden_dim=" not in " ".join(extra)
        assert "--value_head_dim=" not in " ".join(extra)

    def test_training_medium_includes_arch_fields(self, medium_args: argparse.Namespace) -> None:
        """Medium preset values should be passed through."""
        extra = build_training_extra_args(medium_args)
        assert "--policy_hidden_dim=256" in extra
        assert "--value_head_dim=128" in extra

    def test_evaluation_default_omits_arch_fields(self, base_args: argparse.Namespace) -> None:
        """Default arch values should NOT emit extra args."""
        extra = build_evaluation_extra_args(base_args)
        assert "--num_heads=" not in " ".join(extra)

    def test_evaluation_medium_includes_arch_fields(self, medium_args: argparse.Namespace) -> None:
        """Medium preset values should be passed through."""
        extra = build_evaluation_extra_args(medium_args)
        assert "--policy_hidden_dim=256" in extra
        assert "--value_head_dim=128" in extra


# =============================================================================
# Vectorized Env Args Passthrough
# =============================================================================


class TestVectorizedEnvArgs:
    """Test vectorized environment args in build_selfplay_extra_args."""

    def test_sequential_omits_parallel_args(self, base_args: argparse.Namespace) -> None:
        """num_parallel_games=1 should NOT emit vectorized args."""
        extra = build_selfplay_extra_args(base_args)
        assert "--num_parallel_games=" not in " ".join(extra)
        assert "--max_batch_size=" not in " ".join(extra)

    def test_parallel_includes_args(self, base_args: argparse.Namespace) -> None:
        """num_parallel_games>1 should emit both vectorized args."""
        base_args.num_parallel_games = 8
        base_args.max_batch_size = 32
        extra = build_selfplay_extra_args(base_args)
        assert "--num_parallel_games=8" in extra
        assert "--max_batch_size=32" in extra


# =============================================================================
# GumbelMCTS Default Simulation Count
# =============================================================================


class TestGumbelMCTSDefaults:
    """Test GumbelMCTSConfig default simulation count."""

    def test_gumbel_mcts_config_default_simulations(self) -> None:
        """GumbelMCTSConfig should default to 64 simulations."""
        from alpha_zero.core.civ6_mcts import GumbelMCTSConfig

        config = GumbelMCTSConfig()
        assert config.num_simulations == 64
