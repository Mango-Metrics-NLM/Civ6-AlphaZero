"""A seeded run on a GPU repeats exactly -- on real hardware.

This is the GPU counterpart of ``test_training_determinism_regression.py``, which
pins the property on CPU only. Section 7.1 of ``docs/EVALUATION.md`` registers the
A1 runs on ``cuda:0`` (training, self-play) and ``cuda:1`` (arena), and its
"After a VOID" rule depends on a same-seed re-run reproducing the same games. That
claim is true only if ``utils.seeding.enforce_deterministic_backend`` actually
makes this network deterministic on this GPU, which no CPU test can show.

**It is skipped without CUDA**, so CI (CPU-only) never runs it. Run it on the
machine that will execute the cards, with the torch build the cards pin, before
T0b::

    pytest tests/regression/test_cuda_determinism_regression.py -v

A failure here means the registration's reproduction claim does not hold on that
machine. Section 7.1 may still be re-registered at that point, because T0b is not
an A1-family run. The wiring that needs no GPU is covered by
``tests/unit/test_cuda_determinism_wiring.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a CUDA device; run on the GPU machine before T0b (see module docstring)",
)

_SEED = 20261114


def _devices() -> list[str]:
    """Every visible GPU: training is registered on one card, the arena on another."""
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def _train_weights(tmp_path: Path, device: str, tag: str) -> dict[str, torch.Tensor]:
    from alpha_zero.training_civ6 import TrainingConfig, train

    out = tmp_path / tag
    config = TrainingConfig(
        board_size=6,
        max_turns=6,
        num_simulations=4,
        num_games_per_iteration=2,
        num_iterations=2,
        num_epochs=2,
        batch_size=8,
        min_replay_size=1,
        num_res_blocks=2,
        num_filters=16,
        eval_games=0,
        use_gumbel_mcts=True,
        gumbel_num_simulations=4,
        gumbel_num_sampled_actions=4,
        checkpoint_dir=str(out / "ckpt"),
        log_dir=str(out / "log"),
        device=device,
        seed=_SEED,
    )
    train(config)
    checkpoint = torch.load(out / "ckpt" / "latest.pt", map_location="cpu", weights_only=False)
    return {k: v for k, v in checkpoint["state_dict"].items() if isinstance(v, torch.Tensor)}


def _arena_games(tmp_path: Path, device: str, tag: str) -> list[Any]:
    from alpha_zero.evaluation import benchmark

    out = tmp_path / tag
    argv = ["--seed", str(_SEED), "--board-size", "6", "--max-turns", "6", "--sims", "4"]
    argv += ["--gumbel-k", "4", "--mcts-type", "gumbel", "--device", device]
    argv += ["--opponents=random", "--n-games", "4", "--out", str(out)]
    assert benchmark.main(argv) == 0
    envelope = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert envelope["evidence"]["device"] == device
    return [row["games"] for row in envelope["results"]]


@pytest.mark.parametrize("device", _devices() or ["cuda:0"])
def test_two_seeded_trainings_on_a_gpu_give_identical_weights(tmp_path: Path, device: str) -> None:
    first = _train_weights(tmp_path, device, "a")
    second = _train_weights(tmp_path, device, "b")
    assert first.keys() == second.keys() and first, "no weights were compared"
    moved = [name for name in first if not torch.equal(first[name], second[name])]
    assert not moved, (
        f"two trainings at seed {_SEED} on {device} differ in {len(moved)} of {len(first)} "
        f"tensors (first: {moved[:3]}); deterministic algorithms are not holding on this GPU"
    )


@pytest.mark.parametrize("device", _devices() or ["cuda:0"])
def test_two_seeded_arenas_on_a_gpu_play_identical_games(tmp_path: Path, device: str) -> None:
    first = _arena_games(tmp_path, device, "a")
    second = _arena_games(tmp_path, device, "b")
    assert first and first == second, f"a seeded arena on {device} did not repeat"


def test_deterministic_algorithms_are_on_after_a_seeded_gpu_run(tmp_path: Path) -> None:
    _arena_games(tmp_path, "cuda:0", "a")
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
