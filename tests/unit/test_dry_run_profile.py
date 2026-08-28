"""Tests for the ``--dry_run`` smoke-test profile.

Background: ``--dry_run`` shrank only the outer loop (iterations, games per
iteration, epochs, evaluation games) and left ``board_size``, ``max_turns`` and
``num_simulations`` at production defaults. Because cost is dominated by those
inner dimensions, the documented "training smoke test" was really 24 full-size
games and was measured running for over 3.5 hours without completing.

These tests pin the two properties that matter: the profile bounds the inner
dimensions, and it never overrides a value the caller chose deliberately.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from alpha_zero.training_civ6 import (
    DEFAULT_EVAL_GAMES,
    DRY_RUN_PROFILE,
    DryRunProfile,
    TrainingConfig,
)


class TestDryRunProfileBounds:
    """The profile must actually shrink the cost drivers."""

    @pytest.mark.parametrize("field", ["board_size", "max_turns", "num_simulations"])
    def test_inner_dimension_is_smaller_than_the_production_default(self, field: str) -> None:
        # The bug was that these three were left alone. If a future edit raises
        # one back to its production value, the smoke test silently becomes a
        # training run again -- which is exactly what this asserts against.
        assert getattr(DRY_RUN_PROFILE, field) < getattr(TrainingConfig(), field)

    def test_defaults_are_bounded_end_to_end(self) -> None:
        bounded = DRY_RUN_PROFILE.apply_to(TrainingConfig())
        assert bounded.board_size == DRY_RUN_PROFILE.board_size
        assert bounded.max_turns == DRY_RUN_PROFILE.max_turns
        assert bounded.num_simulations == DRY_RUN_PROFILE.num_simulations

    def test_outer_loop_counts_stay_small(self) -> None:
        assert DRY_RUN_PROFILE.num_iterations <= 2
        assert DRY_RUN_PROFILE.num_games_per_iteration <= 2
        assert DRY_RUN_PROFILE.num_epochs == 1
        # The dominant cost of the old dry run was 10 evaluation games per
        # iteration at full size, so this bound is the load-bearing one.
        assert DRY_RUN_PROFILE.num_eval_games < DEFAULT_EVAL_GAMES

    def test_total_work_is_bounded(self) -> None:
        """A crude proxy for runtime, to catch a profile that is nominally
        shrunk but still expensive."""
        p = DRY_RUN_PROFILE
        games = p.num_iterations * (p.num_games_per_iteration + p.num_eval_games)
        searches = games * p.max_turns * p.num_simulations
        assert searches < 2000, f"dry-run search budget {searches} is too large for a smoke test"


class TestExplicitValuesWin:
    """A deliberately-chosen value must survive the profile."""

    def test_explicit_inner_dimensions_are_preserved(self) -> None:
        config = TrainingConfig(board_size=12, max_turns=40, num_simulations=25)
        bounded = DRY_RUN_PROFILE.apply_to(config)
        assert (bounded.board_size, bounded.max_turns, bounded.num_simulations) == (12, 40, 25)

    def test_partial_override_only_touches_unset_fields(self) -> None:
        config = TrainingConfig(board_size=12)  # max_turns / num_simulations left at default
        bounded = DRY_RUN_PROFILE.apply_to(config)
        assert bounded.board_size == 12
        assert bounded.max_turns == DRY_RUN_PROFILE.max_turns
        assert bounded.num_simulations == DRY_RUN_PROFILE.num_simulations

    def test_apply_to_does_not_mutate_the_caller_config(self) -> None:
        config = TrainingConfig()
        before = config.board_size
        DRY_RUN_PROFILE.apply_to(config)
        assert config.board_size == before

    def test_fully_explicit_config_is_returned_unchanged(self) -> None:
        config = TrainingConfig(board_size=12, max_turns=40, num_simulations=25)
        assert DRY_RUN_PROFILE.apply_to(config) is config

    def test_non_inner_fields_are_never_touched(self) -> None:
        config = TrainingConfig(checkpoint_dir="/somewhere", batch_size=77, num_epochs=9)
        bounded = DRY_RUN_PROFILE.apply_to(config)
        assert bounded.checkpoint_dir == "/somewhere"
        assert bounded.batch_size == 77
        # num_epochs is an outer-loop count applied inside train(), not here.
        assert bounded.num_epochs == 9


class TestProfileShape:
    """Guards on the profile object itself."""

    def test_profile_is_immutable(self) -> None:
        # A frozen profile keeps one test from leaking a mutation into another.
        with pytest.raises(Exception):
            DRY_RUN_PROFILE.board_size = 99  # type: ignore[misc]

    def test_every_inner_field_exists_on_training_config(self) -> None:
        # apply_to uses getattr on both, so a typo would silently no-op.
        from alpha_zero.training_civ6 import _INNER_DIMENSION_FIELDS

        config_fields = {f.name for f in fields(TrainingConfig)}
        profile_fields = {f.name for f in fields(DryRunProfile)}
        for name in _INNER_DIMENSION_FIELDS:
            assert name in config_fields, f"{name} is not a TrainingConfig field"
            assert name in profile_fields, f"{name} is not a DryRunProfile field"
