"""Unit tests for Phase 14 orchestrator enhancements.

Tests for:
1. Stall detection via TrainingDiagnostics (integration pattern used by orchestrator)
2. Win rate trend computation
3. build_training_extra_args config pass-through

Usage:
    pytest tests/unit/test_phase14_orchestrator.py -v
"""

from __future__ import annotations

import argparse

from alpha_zero.cloud.gcp.vertex_orchestrator import build_training_extra_args
from alpha_zero.training.diagnostics import TrainingDiagnostics


class TestStallDetection:
    """Test stall detection via TrainingDiagnostics."""

    def test_stall_detection_improving(self) -> None:
        """Win rates [0.25, 0.30, 0.35, 0.40, 0.45] should NOT be stalled."""
        diagnostics = TrainingDiagnostics(stall_threshold=0.02)
        win_rates = [0.25, 0.30, 0.35, 0.40, 0.45]

        stalled = diagnostics.detect_learning_stall(win_rates, lookback=5, threshold=0.02)

        assert not stalled, "Improving win rates should not trigger stall detection"

    def test_stall_detection_flat(self) -> None:
        """Win rates [0.30, 0.30, 0.31, 0.30, 0.30] should be stalled (improvement < threshold)."""
        diagnostics = TrainingDiagnostics(stall_threshold=0.02)
        win_rates = [0.30, 0.30, 0.31, 0.30, 0.30]

        stalled = diagnostics.detect_learning_stall(win_rates, lookback=5, threshold=0.02)

        assert stalled, "Flat win rates (improvement=0.01) should trigger stall detection"

    def test_stall_detection_not_enough_data(self) -> None:
        """Win rates [0.25, 0.30] should NOT be stalled (< lookback entries)."""
        diagnostics = TrainingDiagnostics(stall_threshold=0.02)
        win_rates = [0.25, 0.30]

        stalled = diagnostics.detect_learning_stall(win_rates, lookback=5, threshold=0.02)

        assert not stalled, "Not enough data (<5 entries) should return False"

    def test_stall_detection_oscillating(self) -> None:
        """Win rates [0.25, 0.35, 0.25, 0.35, 0.25] should NOT be stalled (range=0.10 > threshold)."""
        diagnostics = TrainingDiagnostics(stall_threshold=0.02)
        win_rates = [0.25, 0.35, 0.25, 0.35, 0.25]

        stalled = diagnostics.detect_learning_stall(win_rates, lookback=5, threshold=0.02)

        assert not stalled, "Oscillating rates (range=0.10) should not trigger stall detection"

    def test_stall_detection_custom_threshold(self) -> None:
        """Use threshold=0.10 -> [0.25, 0.30, 0.28, 0.32, 0.27] should be stalled (range 0.07 < 0.10)."""
        diagnostics = TrainingDiagnostics(stall_threshold=0.10)
        win_rates = [0.25, 0.30, 0.28, 0.32, 0.27]

        stalled = diagnostics.detect_learning_stall(win_rates, lookback=5, threshold=0.10)

        assert stalled, "Range 0.07 < threshold 0.10 should trigger stall detection"


class TestWinRateTrend:
    """Test win rate trend computation."""

    def test_win_rate_trend_positive(self) -> None:
        """Increasing rates should have positive slope."""
        diagnostics = TrainingDiagnostics()
        win_rates = [0.30, 0.35, 0.40, 0.45, 0.50]

        slope = diagnostics.compute_win_rate_trend(win_rates, lookback=5)

        assert slope > 0, f"Increasing win rates should have positive slope, got {slope}"
        # Linear fit of [0.30, 0.35, 0.40, 0.45, 0.50] should give slope ~0.05
        assert 0.04 < slope < 0.06, f"Expected slope ~0.05, got {slope}"

    def test_win_rate_trend_negative(self) -> None:
        """Decreasing rates should have negative slope."""
        diagnostics = TrainingDiagnostics()
        win_rates = [0.50, 0.45, 0.40, 0.35, 0.30]

        slope = diagnostics.compute_win_rate_trend(win_rates, lookback=5)

        assert slope < 0, f"Decreasing win rates should have negative slope, got {slope}"
        # Linear fit should give slope ~-0.05
        assert -0.06 < slope < -0.04, f"Expected slope ~-0.05, got {slope}"

    def test_win_rate_trend_flat(self) -> None:
        """All same rates should have slope approximately 0."""
        diagnostics = TrainingDiagnostics()
        win_rates = [0.40, 0.40, 0.40, 0.40, 0.40]

        slope = diagnostics.compute_win_rate_trend(win_rates, lookback=5)

        assert abs(slope) < 1e-6, f"Flat win rates should have slope ~0, got {slope}"


class TestBuildTrainingExtraArgs:
    """Test build_training_extra_args config pass-through."""

    def test_build_training_extra_args_constant_lr(self) -> None:
        """Default args (constant LR) should NOT include LR args in output."""
        args = argparse.Namespace(
            lr_schedule_type="constant",
            lr_initial=0.001,
            lr_final=0.0001,
            lr_warmup_steps=0,
            lr_decay_steps=0,
            weight_decay=0.01,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
        )

        extra_args = build_training_extra_args(args)

        # Should only have architecture args, NOT LR args
        assert "--board_size=8" in extra_args
        assert "--action_space_size=376" in extra_args
        assert "--num_res_blocks=6" in extra_args
        assert "--num_filters=64" in extra_args

        # LR args should NOT be present for constant schedule
        lr_arg_keys = [
            "--lr_schedule_type",
            "--lr_initial",
            "--lr_final",
            "--lr_warmup_steps",
            "--lr_decay_steps",
        ]
        for key in lr_arg_keys:
            assert not any(
                key in arg for arg in extra_args
            ), f"{key} should not be in extra_args for constant schedule"

        # Weight decay should NOT be present if default (0.01)
        assert not any(
            "--weight_decay" in arg for arg in extra_args
        ), "Default weight_decay should not be in extra_args"

    def test_build_training_extra_args_warmup_cosine(self) -> None:
        """Non-default lr_schedule_type should include LR args in output."""
        args = argparse.Namespace(
            lr_schedule_type="warmup_cosine",
            lr_initial=0.002,
            lr_final=0.00005,
            lr_warmup_steps=1000,
            lr_decay_steps=50000,
            weight_decay=0.01,
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
        )

        extra_args = build_training_extra_args(args)

        # Should have architecture args
        assert "--board_size=8" in extra_args
        assert "--action_space_size=376" in extra_args

        # Should have ALL LR args for non-constant schedule
        assert "--lr_schedule_type=warmup_cosine" in extra_args
        assert "--lr_initial=0.002" in extra_args
        assert "--lr_final=5e-05" in extra_args or "--lr_final=0.00005" in extra_args
        assert "--lr_warmup_steps=1000" in extra_args
        assert "--lr_decay_steps=50000" in extra_args

        # Weight decay should NOT be present if default (0.01)
        assert not any("--weight_decay" in arg for arg in extra_args)

    def test_build_training_extra_args_custom_weight_decay(self) -> None:
        """weight_decay != 0.01 should include --weight_decay in output."""
        args = argparse.Namespace(
            lr_schedule_type="constant",
            lr_initial=0.001,
            lr_final=0.0001,
            lr_warmup_steps=0,
            lr_decay_steps=0,
            weight_decay=0.001,  # Non-default
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
        )

        extra_args = build_training_extra_args(args)

        # Should have architecture args
        assert "--board_size=8" in extra_args
        assert "--action_space_size=376" in extra_args

        # Should have weight_decay arg
        assert "--weight_decay=0.001" in extra_args

        # LR args should NOT be present for constant schedule
        lr_arg_keys = [
            "--lr_schedule_type",
            "--lr_initial",
            "--lr_final",
            "--lr_warmup_steps",
            "--lr_decay_steps",
        ]
        for key in lr_arg_keys:
            assert not any(
                key in arg for arg in extra_args
            ), f"{key} should not be in extra_args for constant schedule"


class TestBuildTrainingExtraArgsEdgeCases:
    """Additional edge case tests for build_training_extra_args."""

    def test_build_training_extra_args_all_lr_schedules(self) -> None:
        """Test that all non-constant schedule types include LR args."""
        schedules = ["linear", "cosine", "warmup_cosine", "exponential", "step"]

        for schedule_type in schedules:
            args = argparse.Namespace(
                lr_schedule_type=schedule_type,
                lr_initial=0.001,
                lr_final=0.0001,
                lr_warmup_steps=100,
                lr_decay_steps=1000,
                weight_decay=0.01,
                board_size=8,
                action_space_size=376,
                num_res_blocks=6,
                num_filters=64,
            )

            extra_args = build_training_extra_args(args)

            # All non-constant schedules should include LR args
            assert (
                f"--lr_schedule_type={schedule_type}" in extra_args
            ), f"Schedule type {schedule_type} should be in extra_args"
            assert any(
                "--lr_initial" in arg for arg in extra_args
            ), f"Schedule {schedule_type} should include --lr_initial"
            assert any(
                "--lr_final" in arg for arg in extra_args
            ), f"Schedule {schedule_type} should include --lr_final"

    def test_build_training_extra_args_zero_weight_decay(self) -> None:
        """weight_decay=0.0 (non-default) should be in output."""
        args = argparse.Namespace(
            lr_schedule_type="constant",
            lr_initial=0.001,
            lr_final=0.0001,
            lr_warmup_steps=0,
            lr_decay_steps=0,
            weight_decay=0.0,  # Non-default
            board_size=8,
            action_space_size=376,
            num_res_blocks=6,
            num_filters=64,
        )

        extra_args = build_training_extra_args(args)

        # Should have weight_decay arg
        assert "--weight_decay=0.0" in extra_args or "--weight_decay=0" in extra_args

    def test_build_training_extra_args_architecture_only(self) -> None:
        """Architecture args should always be present."""
        args = argparse.Namespace(
            lr_schedule_type="constant",
            lr_initial=0.001,
            lr_final=0.0001,
            lr_warmup_steps=0,
            lr_decay_steps=0,
            weight_decay=0.01,
            board_size=16,
            action_space_size=512,
            num_res_blocks=8,
            num_filters=128,
        )

        extra_args = build_training_extra_args(args)

        # Verify architecture args
        assert "--board_size=16" in extra_args
        assert "--action_space_size=512" in extra_args
        assert "--num_res_blocks=8" in extra_args
        assert "--num_filters=128" in extra_args

        # learning_rate is always forwarded (Phase 17); LR schedule + weight_decay omitted for defaults
        assert "--learning_rate=0.001" in extra_args
        assert (
            len(extra_args) == 5
        ), f"Expected 5 args (4 arch + learning_rate), got {len(extra_args)}"
