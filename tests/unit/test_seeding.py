"""Seed derivation: streams stay separate, and iterations stay different.

The defect these tests exist for: self-play and evaluation seeds carried no
iteration term (``config.seed + game_idx``, ``config.seed + EVAL_SEED_OFFSET``),
while only the DataLoader's generator varied. A seeded run therefore regenerated
the same games every iteration and scored them on the same fixed maps, so the
win-rate curve measured a repeated probe rather than learning.

Negative control (``.claude/skills/negative-control``): dropping the
``iteration`` argument at a call site in ``training_civ6`` makes
``test_self_play_examples_differ_between_iterations`` fail. Confirmed before the
fix landed -- two consecutive ``run_self_play`` calls produced byte-identical
examples.
"""

from __future__ import annotations

import pytest

from alpha_zero.utils.seeding import MAX_SEED, SeedStream, derive_seed

ALL_STREAMS = [getattr(SeedStream, name) for name in dir(SeedStream) if not name.startswith("_")]


class TestDeriveSeed:
    def test_unseeded_runs_stay_unseeded(self) -> None:
        """None must propagate, or an unseeded run would silently become seeded."""
        assert derive_seed(None, SeedStream.SELF_PLAY_ENV, 0, 0) is None

    def test_same_coordinates_give_the_same_seed(self) -> None:
        first = derive_seed(42, SeedStream.SELF_PLAY_ENV, 3, 7)
        second = derive_seed(42, SeedStream.SELF_PLAY_ENV, 3, 7)
        assert first == second

    def test_seed_is_stable_across_processes(self) -> None:
        """Must not depend on PYTHONHASHSEED; that is why it is not ``hash()``."""
        import subprocess
        import sys

        code = (
            "from alpha_zero.utils.seeding import derive_seed, SeedStream;"
            "print(derive_seed(42, SeedStream.SELF_PLAY_ENV, 3, 7))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin"},
        )
        assert int(out.stdout.strip()) == derive_seed(42, SeedStream.SELF_PLAY_ENV, 3, 7)

    def test_iteration_changes_the_seed(self) -> None:
        """The whole point: iteration 0 and 1 must not replay the same games."""
        assert derive_seed(42, SeedStream.SELF_PLAY_ENV, 0, 0) != derive_seed(
            42, SeedStream.SELF_PLAY_ENV, 1, 0
        )

    def test_index_changes_the_seed(self) -> None:
        assert derive_seed(42, SeedStream.SELF_PLAY_ENV, 0, 0) != derive_seed(
            42, SeedStream.SELF_PLAY_ENV, 0, 1
        )

    def test_base_seed_changes_the_seed(self) -> None:
        assert derive_seed(42, SeedStream.SELF_PLAY_ENV, 0, 0) != derive_seed(
            43, SeedStream.SELF_PLAY_ENV, 0, 0
        )

    @pytest.mark.parametrize("stream", ALL_STREAMS)
    def test_seed_is_in_range(self, stream: str) -> None:
        seed = derive_seed(7, stream, 0, 0)
        assert seed is not None
        assert 0 <= seed <= MAX_SEED

    def test_streams_do_not_collide_at_the_same_coordinates(self) -> None:
        """Separation by name is the property that replaced offset arithmetic."""
        seeds = [derive_seed(42, stream, 0, 0) for stream in ALL_STREAMS]
        assert len(set(seeds)) == len(seeds), "two named streams produced the same seed"

    def test_self_play_and_eval_do_not_collide_across_a_grid(self) -> None:
        """The old scheme collided when offsets and indices happened to line up."""
        train = {
            derive_seed(42, SeedStream.SELF_PLAY_ENV, it, game)
            for it in range(12)
            for game in range(12)
        }
        evaluation = {
            derive_seed(42, SeedStream.EVAL_ENV, it, game) for it in range(12) for game in range(12)
        }
        assert not (train & evaluation)

    def test_empty_stream_name_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            derive_seed(42, "")


class TestTrainingLoopUsesIterationSeeds:
    def test_self_play_examples_differ_between_iterations(self) -> None:
        """Two iterations with an identical config must not produce identical data.

        This is the regression itself: before the fix, ``run_self_play`` seeded
        environments with ``config.seed + game_idx`` and the search with
        ``config.seed + num_games``, neither of which involved the iteration, so
        consecutive iterations were byte-identical.
        """
        from alpha_zero.training_civ6 import TrainingConfig, run_self_play

        config = TrainingConfig(
            board_size=5,
            max_turns=6,
            num_simulations=2,
            seed=42,
        )
        first = run_self_play(None, config, num_games=1, use_simple_mcts=True, iteration=0)
        second = run_self_play(None, config, num_games=1, use_simple_mcts=True, iteration=1)

        assert first, "self-play produced no examples; the comparison would be vacuous"
        assert second

        def fingerprint(examples: list) -> list:
            return [(ex.spatial.tobytes(), ex.policy.tobytes()) for ex in examples]

        assert fingerprint(first) != fingerprint(second), (
            "iteration 0 and iteration 1 generated identical self-play data, so a "
            "seeded run would train on the same games every iteration"
        )

    def test_the_same_iteration_still_replays_exactly(self) -> None:
        """Varying by iteration must not cost reproducibility."""
        from alpha_zero.training_civ6 import TrainingConfig, run_self_play

        config = TrainingConfig(
            board_size=5,
            max_turns=6,
            num_simulations=2,
            seed=42,
        )
        first = run_self_play(None, config, num_games=1, use_simple_mcts=True, iteration=3)
        second = run_self_play(None, config, num_games=1, use_simple_mcts=True, iteration=3)

        def fingerprint(examples: list) -> list:
            return [(ex.spatial.tobytes(), ex.policy.tobytes()) for ex in examples]

        assert fingerprint(first) == fingerprint(second)
