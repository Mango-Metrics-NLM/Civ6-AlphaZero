"""Tests for resuming a training run from a checkpoint.

A multi-hour run that cannot resume loses everything to one interruption, which
is what motivated `--resume_from`. These tests pin the three things that make it
usable rather than merely present: the weights actually transfer, a checkpoint
from a *different* architecture is refused instead of silently cross-loaded, and
the iteration numbering continues so a resumed run neither overwrites its own
earlier checkpoints nor renumbers the trend it is trying to measure.
"""

from __future__ import annotations

import pytest
import torch

from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.training_civ6 import (
    TrainingConfig,
    infer_start_iteration,
    resume_network_weights,
)

BOARD = 6
ACTIONS = 376
CHANNELS = 13


def _network(action_space_size: int = ACTIONS) -> MicroCivNetwork:
    return MicroCivNetwork(
        board_size=BOARD,
        action_space_size=action_space_size,
        num_res_blocks=1,
        num_filters=8,
        spatial_channels=CHANNELS,
    )


class TestInferStartIteration:
    """`checkpoint_<N>.pt` is written after loop index N-1, so it resumes at N."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("checkpoint_1.pt", 1),
            ("checkpoint_7.pt", 7),
            ("checkpoint_20.pt", 20),
        ],
    )
    def test_numbered_checkpoint_yields_its_number(self, name: str, expected: int) -> None:
        assert infer_start_iteration(f"/some/dir/{name}") == expected

    @pytest.mark.parametrize(
        "name",
        ["latest.pt", "checkpoint_.pt", "checkpoint_x.pt", "final.pt", "checkpoint_3.pth"],
    )
    def test_unnumbered_names_return_none_rather_than_a_guess(self, name: str) -> None:
        # None is the signal that the caller must warn; a wrong guess would
        # silently overwrite earlier checkpoints.
        assert infer_start_iteration(f"/some/dir/{name}") is None

    def test_directory_components_are_not_matched(self) -> None:
        # Only the basename decides; a path that merely contains the pattern
        # in a parent directory must not be read as an iteration number.
        assert infer_start_iteration("/runs/checkpoint_9.pt/latest.pt") is None


class TestResumeNetworkWeights:
    def test_weights_transfer(self, tmp_path) -> None:
        source = _network()
        with torch.no_grad():
            for param in source.parameters():
                param.add_(0.15)
        path = str(tmp_path / "checkpoint_3.pt")
        source.save(path)

        target = _network()
        assert not all(
            torch.equal(a, b)
            for a, b in zip(source.state_dict().values(), target.state_dict().values(), strict=True)
        ), "fixture is vacuous: the two networks already matched before resuming"

        resume_network_weights(target, path, device="cpu")

        for key, value in source.state_dict().items():
            assert torch.equal(target.state_dict()[key], value), f"{key} did not transfer"

    def test_mismatched_architecture_is_refused(self, tmp_path) -> None:
        """The negative control for silent cross-loading.

        A checkpoint with a different action space must raise, not load. This is
        the failure the deployed path has hit before (CLAUDE.md records
        `validate_encoder_state` for exactly this class of bug).
        """
        path = str(tmp_path / "checkpoint_1.pt")
        _network(action_space_size=ACTIONS).save(path)

        target = _network(action_space_size=ACTIONS + 5)
        with pytest.raises(ValueError, match="does not match the network this run builds"):
            resume_network_weights(target, path, device="cpu")

    def test_non_checkpoint_payload_is_refused(self, tmp_path) -> None:
        path = str(tmp_path / "checkpoint_1.pt")
        torch.save({"not_a_state_dict": torch.zeros(1)}, path)

        with pytest.raises(ValueError, match="not a checkpoint written by this trainer"):
            resume_network_weights(_network(), path, device="cpu")


class TestResumeConfig:
    def test_defaults_do_not_resume(self) -> None:
        config = TrainingConfig()
        assert config.resume_from is None
        assert config.start_iteration == 0

    def test_negative_start_iteration_is_refused(self, tmp_path) -> None:
        """A negative index fails silently in two ways, so it is rejected up front.

        `range(-5, 20)` would name the first checkpoint `checkpoint_-4.pt` and,
        because `use_simple = iteration == 0` never fires, would skip the
        SimpleMCTS bootstrap and run an untrained network through the neural
        search instead. Neither raises on its own.
        """
        from alpha_zero.training_civ6 import train

        config = TrainingConfig(
            board_size=BOARD,
            max_turns=4,
            num_simulations=2,
            num_games_per_iteration=1,
            num_iterations=5,
            start_iteration=-1,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
            seed=1,
        )
        with pytest.raises(ValueError, match="must be >= 0"):
            train(config)

    def test_start_iteration_at_or_past_the_total_is_an_error(self, tmp_path) -> None:
        """`--num_iterations` is the end of the run, not a count of extra ones.

        Reading it the other way would silently run zero iterations and then
        overwrite `latest.pt` with the resumed weights.
        """
        from alpha_zero.training_civ6 import train

        config = TrainingConfig(
            board_size=BOARD,
            max_turns=4,
            num_simulations=2,
            num_games_per_iteration=1,
            num_iterations=5,
            start_iteration=5,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_dir=str(tmp_path / "logs"),
            device="cpu",
            seed=1,
        )
        with pytest.raises(ValueError, match="Nothing to do"):
            train(config)
