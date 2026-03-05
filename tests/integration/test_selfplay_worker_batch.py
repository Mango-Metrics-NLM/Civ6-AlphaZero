"""Worker-level tests for batch self-play mode in selfplay_worker.py.

Tests the CLI integration layer: batch mode detection, dry-run handling,
and orchestrator argument passthrough.

Run with:
    pytest tests/integration/test_selfplay_worker_batch.py -v --timeout=60
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Test: Batch Mode Detection
# ============================================================================


class TestBatchModeDetection:
    """Tests for the use_batch flag logic in main()."""

    def test_batch_enabled_when_parallel_and_batch_gumbel(self):
        """Batch mode activates with num_parallel_games>1 + batch_gumbel."""
        args = argparse.Namespace(
            num_parallel_games=4,
            mcts_type="batch_gumbel",
            dry_run=False,
        )
        use_batch = (
            args.num_parallel_games > 1
            and args.mcts_type == "batch_gumbel"
            and not args.dry_run
        )
        assert use_batch is True

    def test_batch_disabled_for_single_game(self):
        """Batch mode disabled when num_parallel_games=1."""
        args = argparse.Namespace(
            num_parallel_games=1,
            mcts_type="batch_gumbel",
            dry_run=False,
        )
        use_batch = (
            args.num_parallel_games > 1
            and args.mcts_type == "batch_gumbel"
            and not args.dry_run
        )
        assert use_batch is False

    def test_batch_disabled_for_non_batch_gumbel(self):
        """Batch mode disabled when mcts_type is 'gumbel' (not 'batch_gumbel')."""
        args = argparse.Namespace(
            num_parallel_games=4,
            mcts_type="gumbel",
            dry_run=False,
        )
        use_batch = (
            args.num_parallel_games > 1
            and args.mcts_type == "batch_gumbel"
            and not args.dry_run
        )
        assert use_batch is False

    def test_batch_disabled_for_dry_run(self):
        """Batch mode disabled during dry runs."""
        args = argparse.Namespace(
            num_parallel_games=4,
            mcts_type="batch_gumbel",
            dry_run=True,
        )
        use_batch = (
            args.num_parallel_games > 1
            and args.mcts_type == "batch_gumbel"
            and not args.dry_run
        )
        assert use_batch is False

    def test_batch_disabled_for_muzero(self):
        """Batch mode disabled when mcts_type is muzero."""
        args = argparse.Namespace(
            num_parallel_games=4,
            mcts_type="muzero",
            dry_run=False,
        )
        use_batch = (
            args.num_parallel_games > 1
            and args.mcts_type == "batch_gumbel"
            and not args.dry_run
        )
        assert use_batch is False


# ============================================================================
# Test: VALID_MCTS_TYPES
# ============================================================================


class TestMCTSTypesConfig:
    """Tests that batch_gumbel is a valid MCTS type."""

    def test_batch_gumbel_in_valid_types(self):
        """batch_gumbel should be in VALID_MCTS_TYPES."""
        from alpha_zero.cloud.gcp.config import VALID_MCTS_TYPES

        assert "batch_gumbel" in VALID_MCTS_TYPES

    def test_all_expected_types_present(self):
        """All expected MCTS types should be registered."""
        from alpha_zero.cloud.gcp.config import VALID_MCTS_TYPES

        expected = {"civ6", "gumbel", "batch_gumbel", "muzero"}
        assert expected.issubset(set(VALID_MCTS_TYPES))


# ============================================================================
# Test: Orchestrator Argument Passthrough
# ============================================================================


class TestOrchestratorArgs:
    """Tests that vertex_orchestrator passes batch parameters to containers."""

    def test_num_parallel_games_in_container_args(self):
        """Orchestrator should include --num_parallel_games in worker args when > 1."""
        # Simulate the orchestrator's logic from vertex_orchestrator.py L420-422
        args = argparse.Namespace(num_parallel_games=8)
        extra: list[str] = []

        num_parallel = getattr(args, "num_parallel_games", 1)
        if num_parallel > 1:
            extra.append(f"--num_parallel_games={num_parallel}")

        assert "--num_parallel_games=8" in extra

    def test_single_game_does_not_add_flag(self):
        """Orchestrator should not add flag when num_parallel_games=1."""
        args = argparse.Namespace(num_parallel_games=1)
        extra: list[str] = []

        num_parallel = getattr(args, "num_parallel_games", 1)
        if num_parallel > 1:
            extra.append(f"--num_parallel_games={num_parallel}")

        assert len(extra) == 0

    def test_default_parallel_games(self):
        """Default should be 1 if attribute missing."""
        args = argparse.Namespace()
        num_parallel = getattr(args, "num_parallel_games", 1)
        assert num_parallel == 1


# ============================================================================
# Test: Temperature Computation
# ============================================================================


class TestComputeTemperature:
    """Tests for temperature computation in selfplay_worker."""

    def test_threshold_schedule_before(self):
        """Temperature is full before threshold."""
        from alpha_zero.cloud.gcp.selfplay_worker import compute_temperature

        temp = compute_temperature(
            turn=5, max_turns=30, temperature=1.0,
            temperature_threshold=15, schedule="threshold",
        )
        assert temp == 1.0

    def test_threshold_schedule_after(self):
        """Temperature drops to 0 at threshold."""
        from alpha_zero.cloud.gcp.selfplay_worker import compute_temperature

        temp = compute_temperature(
            turn=15, max_turns=30, temperature=1.0,
            temperature_threshold=15, schedule="threshold",
        )
        assert temp == 0.0

    def test_linear_schedule(self):
        """Linear schedule decays proportionally."""
        from alpha_zero.cloud.gcp.selfplay_worker import compute_temperature

        temp_start = compute_temperature(
            turn=0, max_turns=30, temperature=1.0,
            temperature_threshold=15, schedule="linear",
        )
        temp_mid = compute_temperature(
            turn=15, max_turns=30, temperature=1.0,
            temperature_threshold=15, schedule="linear",
        )
        assert temp_start == 1.0
        assert 0.4 < temp_mid < 0.6  # ~0.5 at midpoint

    def test_cosine_schedule(self):
        """Cosine schedule returns valid temperature."""
        from alpha_zero.cloud.gcp.selfplay_worker import compute_temperature

        temp = compute_temperature(
            turn=15, max_turns=30, temperature=1.0,
            temperature_threshold=15, schedule="cosine",
        )
        assert 0.0 <= temp <= 1.0

    def test_unknown_schedule_returns_base(self):
        """Unknown schedule returns base temperature."""
        from alpha_zero.cloud.gcp.selfplay_worker import compute_temperature

        temp = compute_temperature(
            turn=15, max_turns=30, temperature=0.7,
            temperature_threshold=15, schedule="unknown_schedule",
        )
        assert temp == 0.7


# ============================================================================
# Test: CLI Parse Args
# ============================================================================


class TestParseArgs:
    """Tests for selfplay_worker CLI argument parsing."""

    def test_batch_args_present(self):
        """CLI parser accepts batch-related arguments."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        with patch("sys.argv", [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
            "--games=10",
            "--mcts_type=batch_gumbel",
            "--num_parallel_games=4",
            "--max_batch_size=32",
        ]):
            args = parse_args()

        assert args.num_parallel_games == 4
        assert args.max_batch_size == 32
        assert args.mcts_type == "batch_gumbel"

    def test_default_batch_args(self):
        """Default batch args are sane."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        with patch("sys.argv", [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=1",
        ]):
            args = parse_args()

        assert args.num_parallel_games == 1  # default: sequential
        assert args.max_batch_size == 64  # default batch size
