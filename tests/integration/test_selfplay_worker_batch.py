"""Worker-level tests for batch self-play mode in selfplay_worker.py.

Tests the CLI integration layer: batch mode detection, dry-run handling,
and orchestrator argument passthrough.

Run with:
    pytest tests/integration/test_selfplay_worker_batch.py -v --timeout=60
"""

from __future__ import annotations

import argparse
import logging
from unittest.mock import patch

from alpha_zero.cloud.gcp.config import BATCH_MCTS_TYPE, DEFAULT_MCTS_TYPE
from alpha_zero.cloud.gcp.selfplay_worker import should_use_batch

# ============================================================================
# Test: Batch Mode Detection
# ============================================================================


class TestBatchModeDetection:
    """Tests for the batch dispatch decision in the self-play worker.

    These call ``should_use_batch`` directly. They previously re-implemented
    the boolean inline in each test body, which meant they passed regardless of
    what the worker actually did -- the dispatch could have been deleted
    outright without a single failure.
    """

    def test_batch_enabled_when_parallel_and_batch_gumbel(self):
        """Batch mode activates with num_parallel_games>1 + batch_gumbel."""
        args = argparse.Namespace(
            num_parallel_games=4,
            mcts_type=BATCH_MCTS_TYPE,
            dry_run=False,
        )
        assert should_use_batch(args) is True

    def test_batch_disabled_for_single_game(self):
        """Batch mode disabled when num_parallel_games=1."""
        args = argparse.Namespace(
            num_parallel_games=1,
            mcts_type=BATCH_MCTS_TYPE,
            dry_run=False,
        )
        assert should_use_batch(args) is False

    def test_batch_disabled_for_non_batch_gumbel(self):
        """Plain gumbel does not batch, however many parallel games are asked for."""
        args = argparse.Namespace(
            num_parallel_games=4,
            mcts_type=DEFAULT_MCTS_TYPE,
            dry_run=False,
        )
        assert should_use_batch(args) is False

    def test_batch_disabled_for_dry_run(self):
        """Dry runs return synthetic trajectories without an engine."""
        args = argparse.Namespace(
            num_parallel_games=4,
            mcts_type=BATCH_MCTS_TYPE,
            dry_run=True,
        )
        assert should_use_batch(args) is False

    def test_batch_disabled_for_muzero(self):
        """MuZero has its own self-play path."""
        args = argparse.Namespace(
            num_parallel_games=4,
            mcts_type="muzero",
            dry_run=False,
        )
        assert should_use_batch(args) is False


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
    """Tests the orchestrator's real parallelism handling, not a copy of it."""

    def test_default_is_sequential_and_self_consistent(self):
        """The default invocation must not advertise batching it will not do.

        The default was ``--num_parallel_games=4`` against a default
        ``--mcts_type=gumbel``, which fails the worker's dispatch gate: every
        unflagged run forwarded a parallelism flag that was then discarded.
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        with patch("sys.argv", ["vertex_orchestrator"]):
            args = parse_args()

        assert args.num_parallel_games == 1
        assert (
            should_use_batch(
                argparse.Namespace(
                    num_parallel_games=args.num_parallel_games,
                    mcts_type=args.mcts_type,
                    dry_run=False,
                )
            )
            is False
        )

    def test_warns_when_parallelism_would_be_ignored(self, caplog):
        """Requesting parallelism without the batch variant must not be silent."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import warn_on_ignored_parallelism

        args = argparse.Namespace(num_parallel_games=4, mcts_type=DEFAULT_MCTS_TYPE)
        with caplog.at_level(logging.WARNING):
            warned = warn_on_ignored_parallelism(args)

        assert warned is True
        assert BATCH_MCTS_TYPE in caplog.text

    def test_no_warning_when_configuration_is_coherent(self, caplog):
        """A correctly configured batch run, and a sequential run, stay quiet."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import warn_on_ignored_parallelism

        with caplog.at_level(logging.WARNING):
            assert not warn_on_ignored_parallelism(
                argparse.Namespace(num_parallel_games=4, mcts_type=BATCH_MCTS_TYPE)
            )
            assert not warn_on_ignored_parallelism(
                argparse.Namespace(num_parallel_games=1, mcts_type=DEFAULT_MCTS_TYPE)
            )
        assert caplog.text == ""

    def test_warning_does_not_mutate_mcts_type(self):
        """The warning must not rewrite args: the namespace is shared.

        ``args.mcts_type`` also feeds the evaluation, arena and tournament
        phases, so silently promoting it for self-play would change the search
        algorithm those phases use as a side effect.
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import warn_on_ignored_parallelism

        args = argparse.Namespace(num_parallel_games=4, mcts_type=DEFAULT_MCTS_TYPE)
        warn_on_ignored_parallelism(args)
        assert args.mcts_type == DEFAULT_MCTS_TYPE
        assert args.num_parallel_games == 4

    def test_extra_args_include_parallelism_only_when_requested(self):
        """Worker args carry the flag only when parallelism is actually asked for.

        Driven through ``parse_args`` rather than a hand-built namespace so the
        test exercises the real argument surface and cannot drift from it.
        """
        from alpha_zero.cloud.gcp.vertex_orchestrator import (
            build_selfplay_extra_args,
            parse_args,
        )

        with patch("sys.argv", ["o", "--mcts_type", BATCH_MCTS_TYPE, "--num_parallel_games", "8"]):
            parallel_args = parse_args()
        with patch("sys.argv", ["o", "--mcts_type", BATCH_MCTS_TYPE]):
            sequential_args = parse_args()

        parallel = build_selfplay_extra_args(parallel_args)
        sequential = build_selfplay_extra_args(sequential_args)

        assert "--num_parallel_games=8" in parallel
        assert not any(a.startswith("--num_parallel_games") for a in sequential)


class TestComputeTemperature:
    """Tests for temperature computation in selfplay_worker."""

    def test_threshold_schedule_before(self):
        """Temperature is full before threshold."""
        from alpha_zero.cloud.gcp.selfplay_worker import compute_temperature

        temp = compute_temperature(
            turn=5,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,
            schedule="threshold",
        )
        assert temp == 1.0

    def test_threshold_schedule_after(self):
        """Temperature drops to 0 at threshold."""
        from alpha_zero.cloud.gcp.selfplay_worker import compute_temperature

        temp = compute_temperature(
            turn=15,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,
            schedule="threshold",
        )
        assert temp == 0.0

    def test_linear_schedule(self):
        """Linear schedule decays proportionally."""
        from alpha_zero.cloud.gcp.selfplay_worker import compute_temperature

        temp_start = compute_temperature(
            turn=0,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,
            schedule="linear",
        )
        temp_mid = compute_temperature(
            turn=15,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,
            schedule="linear",
        )
        assert temp_start == 1.0
        assert 0.4 < temp_mid < 0.6  # ~0.5 at midpoint

    def test_cosine_schedule(self):
        """Cosine schedule returns valid temperature."""
        from alpha_zero.cloud.gcp.selfplay_worker import compute_temperature

        temp = compute_temperature(
            turn=15,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,
            schedule="cosine",
        )
        assert 0.0 <= temp <= 1.0

    def test_unknown_schedule_returns_base(self):
        """Unknown schedule returns base temperature."""
        from alpha_zero.cloud.gcp.selfplay_worker import compute_temperature

        temp = compute_temperature(
            turn=15,
            max_turns=30,
            temperature=0.7,
            temperature_threshold=15,
            schedule="unknown_schedule",
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

        with patch(
            "sys.argv",
            [
                "selfplay_worker",
                "--worker_id=0",
                "--model_version=1",
                "--iteration=1",
                "--games=10",
                "--mcts_type=batch_gumbel",
                "--num_parallel_games=4",
                "--max_batch_size=32",
            ],
        ):
            args = parse_args()

        assert args.num_parallel_games == 4
        assert args.max_batch_size == 32
        assert args.mcts_type == "batch_gumbel"

    def test_default_batch_args(self):
        """Default batch args are sane."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        with patch(
            "sys.argv",
            [
                "selfplay_worker",
                "--worker_id=0",
                "--model_version=1",
                "--iteration=1",
            ],
        ):
            args = parse_args()

        assert args.num_parallel_games == 1  # default: sequential
        assert args.max_batch_size == 64  # default batch size
