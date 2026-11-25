"""Unit tests for decomposed helpers in ``scripts/run_enhanced_training.py``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from scripts import run_enhanced_training as ret


def test_training_presets_have_positive_budgets() -> None:
    """All registered presets exist and define positive budgets."""
    for name in ("fast", "balanced", "full"):
        assert name in ret._TRAINING_PRESETS
        preset = ret._TRAINING_PRESETS[name]
        assert preset["num_iterations"] > 0
        assert preset["games_per_iteration"] > 0
        assert preset["batch_size"] > 0
        assert preset["eval_interval"] > 0


def test_build_curriculum_and_env_disabled_vs_enabled(tmp_path: Path) -> None:
    """Both curriculum and env tuning branches produce valid configs cleanly."""
    curriculum_on, env_on = ret._build_curriculum_and_env(
        board_size=8,
        max_turns=30,
        games_per_iteration=5,
        checkpoint_dir=str(tmp_path),
        use_curriculum=True,
        use_env_tuning=True,
    )
    assert len(curriculum_on.config.stages) == 3
    assert all(s.min_games >= 10 for s in curriculum_on.config.stages)
    assert env_on.enabled is True

    curriculum_off, env_off = ret._build_curriculum_and_env(
        board_size=6,
        max_turns=25,
        games_per_iteration=4,
        checkpoint_dir=str(tmp_path),
        use_curriculum=False,
        use_env_tuning=False,
    )
    assert len(curriculum_off.config.stages) == 1
    assert curriculum_off.config.stages[0].board_size == 6
    assert curriculum_off.config.stages[0].max_turns == 25
    assert env_off.enabled is False


def test_log_selfplay_iteration_handles_empty_and_populated_stats(tmp_path: Path) -> None:
    """Empty ``game_stats`` must not divide by zero; populated stats log metrics."""
    curriculum, _ = ret._build_curriculum_and_env(
        board_size=8,
        max_turns=30,
        games_per_iteration=4,
        checkpoint_dir=str(tmp_path),
        use_curriculum=False,
        use_env_tuning=False,
    )
    buf = ret.ReplayBuffer(max_size=10)
    tracker = MagicMock()
    tracker.is_initialized = True
    progress = {
        "current_level": 0,
        "current_stage": "stage0",
        "win_rate": 0.5,
        "total_games": 2,
    }

    ret._log_selfplay_iteration(tracker, curriculum, buf, [], [], progress, iteration=0)
    tracker.log_selfplay.assert_not_called()

    ret._log_selfplay_iteration(
        tracker,
        curriculum,
        buf,
        [],
        [{"winner": 0, "steps": 10}, {"winner": None, "steps": 14}],
        progress,
        iteration=1,
    )
    tracker.log_selfplay.assert_called_once()
    tracker.log.assert_called_once()
