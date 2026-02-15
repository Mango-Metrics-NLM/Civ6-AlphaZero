"""Tests for CurriculumScheduler and StageStats."""
from __future__ import annotations

import pytest

from src.core.config import CurriculumConfig, CurriculumStage
from src.core.training.curriculum import CurriculumScheduler, StageStats

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    num_stages: int = 3,
    promotion_elo_threshold: float = 1200.0,
    promotion_window: int = 3,
    demotion_enabled: bool = True,
    demotion_elo_threshold: float = 900.0,
    min_iterations: int = 0,
) -> CurriculumConfig:
    """Create a CurriculumConfig with sensible defaults for testing."""
    stage_defs = [
        ("easy", "mini_resource"),
        ("medium", "territory_control"),
        ("hard", "tactical_combat"),
    ]
    stages = [
        CurriculumStage(
            name=name,
            environment=env,
            min_iterations=min_iterations,
        )
        for name, env in stage_defs[:num_stages]
    ]
    return CurriculumConfig(
        stages=stages,
        promotion_elo_threshold=promotion_elo_threshold,
        promotion_window=promotion_window,
        demotion_enabled=demotion_enabled,
        demotion_elo_threshold=demotion_elo_threshold,
    )


def _make_single_stage_config() -> CurriculumConfig:
    """Create a CurriculumConfig with exactly one stage."""
    return CurriculumConfig(
        stages=[CurriculumStage(name="only", environment="mini_resource")],
        promotion_elo_threshold=1200.0,
        promotion_window=3,
        demotion_enabled=True,
        demotion_elo_threshold=900.0,
    )


# ===================================================================
# StageStats tests
# ===================================================================


class TestStageStatsDefaults:
    """Tests for StageStats default values."""

    def test_default_values(self) -> None:
        stats = StageStats(stage_name="test")
        assert stats.stage_name == "test"
        assert stats.games_played == 0
        assert stats.iterations_completed == 0
        assert stats.current_elo == 1000.0
        assert stats.peak_elo == 1000.0
        assert stats.promotion_count == 0
        assert stats.demotion_count == 0
        assert stats.elo_history == []

    def test_custom_initial_values(self) -> None:
        stats = StageStats(
            stage_name="custom",
            games_played=10,
            iterations_completed=5,
            current_elo=1100.0,
            peak_elo=1200.0,
        )
        assert stats.games_played == 10
        assert stats.iterations_completed == 5
        assert stats.current_elo == 1100.0
        assert stats.peak_elo == 1200.0


class TestStageStatsRecordElo:
    """Tests for StageStats.record_elo."""

    def test_record_elo_updates_current(self) -> None:
        stats = StageStats(stage_name="test")
        stats.record_elo(1100.0)
        assert stats.current_elo == 1100.0

    def test_record_elo_updates_peak(self) -> None:
        stats = StageStats(stage_name="test")
        stats.record_elo(1200.0)
        assert stats.peak_elo == 1200.0

    def test_record_elo_peak_does_not_decrease(self) -> None:
        stats = StageStats(stage_name="test")
        stats.record_elo(1300.0)
        stats.record_elo(1100.0)
        assert stats.current_elo == 1100.0
        assert stats.peak_elo == 1300.0

    def test_record_elo_appends_to_history(self) -> None:
        stats = StageStats(stage_name="test")
        stats.record_elo(1050.0)
        stats.record_elo(1100.0)
        stats.record_elo(1150.0)
        assert stats.elo_history == [1050.0, 1100.0, 1150.0]

    def test_record_elo_below_initial_peak(self) -> None:
        stats = StageStats(stage_name="test")
        stats.record_elo(900.0)
        assert stats.current_elo == 900.0
        assert stats.peak_elo == 1000.0  # default peak unchanged


# ===================================================================
# CurriculumScheduler constructor tests
# ===================================================================


class TestCurriculumSchedulerConstruction:
    """Tests for CurriculumScheduler constructor validation."""

    def test_valid_construction(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)
        assert scheduler.current_stage_index == 0
        assert scheduler.current_stage_name == "easy"

    def test_custom_initial_stage_index(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config, initial_stage_index=1)
        assert scheduler.current_stage_index == 1
        assert scheduler.current_stage_name == "medium"

    def test_initial_stage_last(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config, initial_stage_index=2)
        assert scheduler.current_stage_index == 2
        assert scheduler.current_stage_name == "hard"

    def test_empty_stages_raises(self) -> None:
        config = CurriculumConfig(stages=[])
        with pytest.raises(ValueError, match="at least one stage"):
            CurriculumScheduler(config)

    def test_negative_initial_stage_raises(self) -> None:
        config = _make_config()
        with pytest.raises(ValueError, match="out of range"):
            CurriculumScheduler(config, initial_stage_index=-1)

    def test_initial_stage_too_large_raises(self) -> None:
        config = _make_config()
        with pytest.raises(ValueError, match="out of range"):
            CurriculumScheduler(config, initial_stage_index=3)

    def test_initial_stage_way_too_large_raises(self) -> None:
        config = _make_config()
        with pytest.raises(ValueError, match="out of range"):
            CurriculumScheduler(config, initial_stage_index=100)


# ===================================================================
# CurriculumScheduler properties
# ===================================================================


class TestCurriculumSchedulerProperties:
    """Tests for CurriculumScheduler read-only properties."""

    def test_current_stage_returns_correct_object(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)
        stage = scheduler.current_stage
        assert isinstance(stage, CurriculumStage)
        assert stage.name == "easy"
        assert stage.environment == "mini_resource"

    def test_num_stages(self) -> None:
        config = _make_config(num_stages=3)
        scheduler = CurriculumScheduler(config)
        assert scheduler.num_stages == 3

    def test_num_stages_single(self) -> None:
        config = _make_single_stage_config()
        scheduler = CurriculumScheduler(config)
        assert scheduler.num_stages == 1

    def test_is_final_stage_false(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)
        assert scheduler.is_final_stage is False

    def test_is_final_stage_true(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config, initial_stage_index=2)
        assert scheduler.is_final_stage is True

    def test_is_final_stage_single_stage(self) -> None:
        config = _make_single_stage_config()
        scheduler = CurriculumScheduler(config)
        assert scheduler.is_final_stage is True

    def test_current_stats_returns_stage_stats(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)
        stats = scheduler.current_stats
        assert isinstance(stats, StageStats)
        assert stats.stage_name == "easy"

    def test_get_stage_stats_existing(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)
        stats = scheduler.get_stage_stats("medium")
        assert stats is not None
        assert stats.stage_name == "medium"

    def test_get_stage_stats_nonexistent(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)
        stats = scheduler.get_stage_stats("impossible")
        assert stats is None


# ===================================================================
# Promotion tests
# ===================================================================


class TestPromotion:
    """Tests for promotion logic."""

    def test_promotes_when_threshold_met(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
        )
        scheduler = CurriculumScheduler(config)

        # Record 3 iterations above threshold
        scheduler.record_iteration(elo=1300.0)
        scheduler.record_iteration(elo=1250.0)
        result = scheduler.record_iteration(elo=1200.0)

        assert result == "medium"
        assert scheduler.current_stage_name == "medium"
        assert scheduler.current_stage_index == 1

    def test_promotion_increments_promotion_count(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
        )
        scheduler = CurriculumScheduler(config)

        # Get stats for easy stage before promotion
        easy_stats = scheduler.get_stage_stats("easy")
        assert easy_stats is not None
        assert easy_stats.promotion_count == 0

        for _ in range(3):
            scheduler.record_iteration(elo=1300.0)

        assert easy_stats.promotion_count == 1

    def test_does_not_promote_below_threshold(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
        )
        scheduler = CurriculumScheduler(config)

        # Record iterations with elo below threshold
        for _ in range(5):
            result = scheduler.record_iteration(elo=1100.0)
            assert result is None

        assert scheduler.current_stage_name == "easy"

    def test_does_not_promote_before_min_iterations(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
            min_iterations=10,
        )
        scheduler = CurriculumScheduler(config)

        # Record high elo but fewer than min_iterations
        for _ in range(5):
            result = scheduler.record_iteration(elo=1500.0)
            assert result is None

        assert scheduler.current_stage_name == "easy"

    def test_promotes_after_min_iterations_reached(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
            min_iterations=5,
        )
        scheduler = CurriculumScheduler(config)

        # Record enough iterations to pass min_iterations
        for i in range(5):
            result = scheduler.record_iteration(elo=1500.0)
            # Should not promote until iteration 5 (min_iterations met)
            # and at that point we have 5 elo entries >= window of 3
            if i < 4:
                assert result is None

        # On the 5th iteration (index 4), min_iterations=5 is met (5 completed)
        assert result == "medium"

    def test_does_not_promote_before_window_filled(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=5,
        )
        scheduler = CurriculumScheduler(config)

        # Only record 3 iterations (less than window of 5)
        for _ in range(3):
            result = scheduler.record_iteration(elo=1500.0)
            assert result is None

        assert scheduler.current_stage_name == "easy"

    def test_does_not_promote_from_final_stage(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
        )
        scheduler = CurriculumScheduler(config, initial_stage_index=2)
        assert scheduler.is_final_stage is True

        for _ in range(5):
            result = scheduler.record_iteration(elo=2000.0)
            assert result is None

        assert scheduler.current_stage_name == "hard"

    def test_promotion_uses_sliding_window_average(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
        )
        scheduler = CurriculumScheduler(config)

        # First two below threshold, won't have window filled
        scheduler.record_iteration(elo=800.0)
        scheduler.record_iteration(elo=800.0)
        # Now fill the window with one low and two high - avg should be below
        # Window: [800, 1100, 1100] -> avg = 1000 < 1200
        result = scheduler.record_iteration(elo=1100.0)
        assert result is None

        # Window: [1100, 1300, 1300] -> avg = 1233.33 >= 1200
        scheduler.record_iteration(elo=1300.0)
        result = scheduler.record_iteration(elo=1300.0)
        assert result == "medium"


# ===================================================================
# Demotion tests
# ===================================================================


class TestDemotion:
    """Tests for demotion logic."""

    def test_demotes_when_enabled_and_threshold_met(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
            demotion_enabled=True,
            demotion_elo_threshold=900.0,
        )
        # Start at the middle stage
        scheduler = CurriculumScheduler(config, initial_stage_index=1)

        # Record low elo to trigger demotion
        for _ in range(2):
            scheduler.record_iteration(elo=800.0)
        result = scheduler.record_iteration(elo=800.0)

        assert result == "easy"
        assert scheduler.current_stage_name == "easy"
        assert scheduler.current_stage_index == 0

    def test_demotion_increments_demotion_count(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
            demotion_enabled=True,
            demotion_elo_threshold=900.0,
        )
        scheduler = CurriculumScheduler(config, initial_stage_index=1)

        medium_stats = scheduler.get_stage_stats("medium")
        assert medium_stats is not None
        assert medium_stats.demotion_count == 0

        for _ in range(3):
            scheduler.record_iteration(elo=800.0)

        assert medium_stats.demotion_count == 1

    def test_does_not_demote_when_disabled(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
            demotion_enabled=False,
            demotion_elo_threshold=900.0,
        )
        scheduler = CurriculumScheduler(config, initial_stage_index=1)

        for _ in range(5):
            result = scheduler.record_iteration(elo=500.0)
            assert result is None

        assert scheduler.current_stage_name == "medium"

    def test_does_not_demote_from_first_stage(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
            demotion_enabled=True,
            demotion_elo_threshold=900.0,
        )
        scheduler = CurriculumScheduler(config)
        assert scheduler.current_stage_index == 0

        for _ in range(5):
            result = scheduler.record_iteration(elo=500.0)
            assert result is None

        assert scheduler.current_stage_name == "easy"
        assert scheduler.current_stage_index == 0

    def test_does_not_demote_before_window_filled(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=5,
            demotion_enabled=True,
            demotion_elo_threshold=900.0,
        )
        scheduler = CurriculumScheduler(config, initial_stage_index=1)

        # Only 3 iterations, window is 5
        for _ in range(3):
            result = scheduler.record_iteration(elo=500.0)
            assert result is None

        assert scheduler.current_stage_name == "medium"

    def test_demotion_uses_sliding_window_average(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
            demotion_enabled=True,
            demotion_elo_threshold=900.0,
        )
        scheduler = CurriculumScheduler(config, initial_stage_index=1)

        # Mix of high and low: avg above demotion threshold
        # Window: [1000, 1000, 700] -> avg = 900, not strictly < 900
        scheduler.record_iteration(elo=1000.0)
        scheduler.record_iteration(elo=1000.0)
        result = scheduler.record_iteration(elo=700.0)
        assert result is None

        # Window: [1000, 700, 800] -> avg = 833.33 < 900
        result = scheduler.record_iteration(elo=800.0)
        assert result == "easy"


# ===================================================================
# force_stage tests
# ===================================================================


class TestForceStage:
    """Tests for CurriculumScheduler.force_stage."""

    def test_force_stage_valid_index(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)

        result = scheduler.force_stage(2)
        assert result == "hard"
        assert scheduler.current_stage_index == 2
        assert scheduler.current_stage_name == "hard"

    def test_force_stage_to_first(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config, initial_stage_index=2)

        result = scheduler.force_stage(0)
        assert result == "easy"
        assert scheduler.current_stage_index == 0

    def test_force_stage_same_index(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)

        result = scheduler.force_stage(0)
        assert result == "easy"
        assert scheduler.current_stage_index == 0

    def test_force_stage_negative_raises(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)

        with pytest.raises(ValueError, match="out of range"):
            scheduler.force_stage(-1)

    def test_force_stage_too_large_raises(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)

        with pytest.raises(ValueError, match="out of range"):
            scheduler.force_stage(3)

    def test_force_stage_way_too_large_raises(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)

        with pytest.raises(ValueError, match="out of range"):
            scheduler.force_stage(999)


# ===================================================================
# record_iteration tests
# ===================================================================


class TestRecordIteration:
    """Tests for CurriculumScheduler.record_iteration."""

    def test_updates_iterations_completed(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)

        scheduler.record_iteration(elo=1000.0)
        scheduler.record_iteration(elo=1000.0)

        assert scheduler.current_stats.iterations_completed == 2

    def test_updates_games_played(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)

        scheduler.record_iteration(elo=1000.0, games_played=25)
        scheduler.record_iteration(elo=1000.0, games_played=30)

        assert scheduler.current_stats.games_played == 55

    def test_updates_elo(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)

        scheduler.record_iteration(elo=1150.0)

        assert scheduler.current_stats.current_elo == 1150.0
        assert len(scheduler.current_stats.elo_history) == 1

    def test_returns_none_when_no_transition(self) -> None:
        config = _make_config(promotion_window=3)
        scheduler = CurriculumScheduler(config)

        result = scheduler.record_iteration(elo=1000.0)
        assert result is None

    def test_returns_stage_name_on_promotion(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
        )
        scheduler = CurriculumScheduler(config)

        scheduler.record_iteration(elo=1300.0)
        scheduler.record_iteration(elo=1300.0)
        result = scheduler.record_iteration(elo=1300.0)

        assert result == "medium"

    def test_returns_stage_name_on_demotion(self) -> None:
        config = _make_config(
            promotion_window=3,
            demotion_enabled=True,
            demotion_elo_threshold=900.0,
        )
        scheduler = CurriculumScheduler(config, initial_stage_index=2)

        scheduler.record_iteration(elo=800.0)
        scheduler.record_iteration(elo=800.0)
        result = scheduler.record_iteration(elo=800.0)

        assert result == "medium"

    def test_default_games_played_is_zero(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)

        scheduler.record_iteration(elo=1000.0)

        assert scheduler.current_stats.games_played == 0

    def test_promotion_checked_before_demotion(self) -> None:
        """When both conditions could theoretically fire, promotion wins."""
        # This is an implementation detail: promotion is checked first.
        # In practice the elo can't be both >= promotion and < demotion,
        # so we just ensure logic works when only promotion applies.
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
            demotion_enabled=True,
            demotion_elo_threshold=900.0,
        )
        scheduler = CurriculumScheduler(config)

        for _ in range(3):
            scheduler.record_iteration(elo=1300.0)

        assert scheduler.current_stage_name == "medium"


# ===================================================================
# get_summary tests
# ===================================================================


class TestGetSummary:
    """Tests for CurriculumScheduler.get_summary."""

    def test_summary_keys(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)
        summary = scheduler.get_summary()

        expected_keys = {
            "current_stage",
            "current_index",
            "num_stages",
            "is_final",
            "iterations_at_stage",
            "games_at_stage",
            "elo_at_stage",
            "peak_elo_at_stage",
        }
        assert set(summary.keys()) == expected_keys

    def test_summary_initial_values(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)
        summary = scheduler.get_summary()

        assert summary["current_stage"] == "easy"
        assert summary["current_index"] == 0
        assert summary["num_stages"] == 3
        assert summary["is_final"] is False
        assert summary["iterations_at_stage"] == 0
        assert summary["games_at_stage"] == 0
        assert summary["elo_at_stage"] == 1000.0
        assert summary["peak_elo_at_stage"] == 1000.0

    def test_summary_updates_after_iterations(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)

        scheduler.record_iteration(elo=1100.0, games_played=25)
        scheduler.record_iteration(elo=1150.0, games_played=30)

        summary = scheduler.get_summary()
        assert summary["iterations_at_stage"] == 2
        assert summary["games_at_stage"] == 55
        assert summary["elo_at_stage"] == 1150.0
        assert summary["peak_elo_at_stage"] == 1150.0

    def test_summary_reflects_stage_after_promotion(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
        )
        scheduler = CurriculumScheduler(config)

        for _ in range(3):
            scheduler.record_iteration(elo=1300.0, games_played=10)

        summary = scheduler.get_summary()
        assert summary["current_stage"] == "medium"
        assert summary["current_index"] == 1
        # New stage should have fresh stats
        assert summary["iterations_at_stage"] == 0
        assert summary["games_at_stage"] == 0

    def test_summary_is_final_at_last_stage(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config, initial_stage_index=2)
        summary = scheduler.get_summary()

        assert summary["is_final"] is True


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge case tests for CurriculumScheduler."""

    def test_single_stage_no_promotion(self) -> None:
        config = _make_single_stage_config()
        scheduler = CurriculumScheduler(config)

        assert scheduler.num_stages == 1
        assert scheduler.is_final_stage is True

        for _ in range(10):
            result = scheduler.record_iteration(elo=9999.0)
            assert result is None

        assert scheduler.current_stage_name == "only"

    def test_single_stage_no_demotion(self) -> None:
        config = _make_single_stage_config()
        scheduler = CurriculumScheduler(config)

        for _ in range(10):
            result = scheduler.record_iteration(elo=0.0)
            assert result is None

        assert scheduler.current_stage_name == "only"

    def test_traverse_all_stages(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=2,
        )
        scheduler = CurriculumScheduler(config)

        assert scheduler.current_stage_name == "easy"

        # Promote easy -> medium
        scheduler.record_iteration(elo=1300.0)
        result = scheduler.record_iteration(elo=1300.0)
        assert result == "medium"
        assert scheduler.current_stage_name == "medium"

        # Promote medium -> hard
        scheduler.record_iteration(elo=1400.0)
        result = scheduler.record_iteration(elo=1400.0)
        assert result == "hard"
        assert scheduler.current_stage_name == "hard"
        assert scheduler.is_final_stage is True

        # Cannot promote further
        for _ in range(5):
            result = scheduler.record_iteration(elo=2000.0)
            assert result is None

    def test_promote_then_demote(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=2,
            demotion_enabled=True,
            demotion_elo_threshold=900.0,
        )
        scheduler = CurriculumScheduler(config)

        # Promote easy -> medium
        scheduler.record_iteration(elo=1300.0)
        result = scheduler.record_iteration(elo=1300.0)
        assert result == "medium"

        # Demote medium -> easy
        scheduler.record_iteration(elo=800.0)
        result = scheduler.record_iteration(elo=800.0)
        assert result == "easy"

        assert scheduler.current_stage_index == 0
        assert scheduler.current_stage_name == "easy"

    def test_stats_preserved_after_stage_transitions(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=2,
        )
        scheduler = CurriculumScheduler(config)

        # Record some iterations at easy stage
        scheduler.record_iteration(elo=1300.0, games_played=10)
        scheduler.record_iteration(elo=1300.0, games_played=15)

        # Now at medium, check that easy stats are still accessible
        easy_stats = scheduler.get_stage_stats("easy")
        assert easy_stats is not None
        assert easy_stats.iterations_completed == 2
        assert easy_stats.games_played == 25
        assert easy_stats.promotion_count == 1

    def test_multiple_promotions_and_demotions(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=2,
            demotion_enabled=True,
            demotion_elo_threshold=900.0,
        )
        scheduler = CurriculumScheduler(config)

        # Promote easy -> medium
        scheduler.record_iteration(elo=1300.0)
        scheduler.record_iteration(elo=1300.0)
        assert scheduler.current_stage_name == "medium"

        # Demote medium -> easy
        scheduler.record_iteration(elo=800.0)
        scheduler.record_iteration(elo=800.0)
        assert scheduler.current_stage_name == "easy"

        # Promote easy -> medium again
        scheduler.record_iteration(elo=1300.0)
        scheduler.record_iteration(elo=1300.0)
        assert scheduler.current_stage_name == "medium"

        easy_stats = scheduler.get_stage_stats("easy")
        medium_stats = scheduler.get_stage_stats("medium")
        assert easy_stats is not None
        assert medium_stats is not None
        assert easy_stats.promotion_count == 2
        assert medium_stats.demotion_count == 1

    def test_force_stage_does_not_affect_stats(self) -> None:
        config = _make_config()
        scheduler = CurriculumScheduler(config)

        scheduler.record_iteration(elo=1100.0, games_played=20)
        scheduler.force_stage(2)

        # Easy stats should be preserved
        easy_stats = scheduler.get_stage_stats("easy")
        assert easy_stats is not None
        assert easy_stats.iterations_completed == 1
        assert easy_stats.games_played == 20

        # Hard stats should be fresh
        hard_stats = scheduler.get_stage_stats("hard")
        assert hard_stats is not None
        assert hard_stats.iterations_completed == 0

    def test_elo_exactly_at_promotion_threshold(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
        )
        scheduler = CurriculumScheduler(config)

        # Average exactly at threshold: 1200.0 >= 1200.0 should promote
        for _ in range(3):
            scheduler.record_iteration(elo=1200.0)

        assert scheduler.current_stage_name == "medium"

    def test_elo_just_below_promotion_threshold(self) -> None:
        config = _make_config(
            promotion_elo_threshold=1200.0,
            promotion_window=3,
        )
        scheduler = CurriculumScheduler(config)

        for _ in range(3):
            scheduler.record_iteration(elo=1199.99)

        assert scheduler.current_stage_name == "easy"

    def test_elo_exactly_at_demotion_threshold(self) -> None:
        """Demotion requires avg < threshold, so exactly at threshold should NOT demote."""
        config = _make_config(
            promotion_window=3,
            demotion_enabled=True,
            demotion_elo_threshold=900.0,
        )
        scheduler = CurriculumScheduler(config, initial_stage_index=1)

        for _ in range(3):
            scheduler.record_iteration(elo=900.0)

        # avg == 900.0, condition is < 900.0, so no demotion
        assert scheduler.current_stage_name == "medium"

    def test_elo_just_below_demotion_threshold(self) -> None:
        config = _make_config(
            promotion_window=3,
            demotion_enabled=True,
            demotion_elo_threshold=900.0,
        )
        scheduler = CurriculumScheduler(config, initial_stage_index=1)

        for _ in range(3):
            scheduler.record_iteration(elo=899.99)

        assert scheduler.current_stage_name == "easy"
