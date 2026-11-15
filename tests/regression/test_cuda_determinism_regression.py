"""A seeded run on a GPU repeats exactly -- a developer-side check on real hardware.

This is the GPU counterpart of ``test_training_determinism_regression.py``, which
pins the property on CPU only. It needs a CUDA device, so **CI never runs it**
(CI is CPU-only) and a skip here is *not* a pass.

**It is not the registered gate.** Section 7.1 of ``docs/EVALUATION.md`` makes GPU
determinism a precondition and names the check that counts: T0b's ``determinism``
invocation (``scripts/check_determinism.py``), which runs inside the card's pinned
environment, at A1's network and search shape, with each repeat in its own
process. This module is the quick version for a developer on the GPU machine:
same process, tiny shapes, a few seconds. Use it to find out early that
deterministic algorithms do not hold -- not to conclude that they do.

To run it against the build the cards pin::

    pip install torch==<registered version> --index-url <registered index>
    pytest tests/regression/test_cuda_determinism_regression.py -v

``CUDA_DEVICE_ORDER`` and ``CUBLAS_WORKSPACE_CONFIG`` are set below, before torch
is imported, to what ``run_card`` gives a card's children: without the first,
``cuda:1`` here need not be the GPU the cards call ``cuda:1``; the second must be
in the environment before cuBLAS starts.

What a pass does **not** show: A1's shapes, reproduction across processes, or the
pinned environment. Those are the card invocation's job.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Only on a machine with NVIDIA tooling: a CPU session's environment (CI, where this
# module is collected and skipped) must not be changed by importing a test file.
if shutil.which("nvidia-smi"):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch

from alpha_zero.evaluation.a1_protocol import (
    default_protocol_path,
    read_registered_protocol,
)
from alpha_zero.evaluation.a1_registration import registered_devices
from alpha_zero.utils.device import torch_build_matches

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a CUDA device; a skip is not a pass (see the module docstring)",
)

_SEED = 20261114


def _devices() -> list[str]:
    """Every visible GPU: training is registered on one card, the arena on another."""
    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


@pytest.fixture(autouse=True)
def fresh_backend_flags() -> Iterator[None]:
    """Each test must switch determinism on itself, and none may leave it on.

    The switches are process-global. Without the reset, a later test would pass
    under flags an earlier one set and so prove nothing about its own code path;
    without the restore, the rest of a pytest session on the GPU machine would
    run in deterministic mode.
    """
    cudnn = torch.backends.cudnn
    before = (cudnn.deterministic, cudnn.benchmark, torch.are_deterministic_algorithms_enabled())
    cudnn.deterministic, cudnn.benchmark = False, True
    torch.use_deterministic_algorithms(False)
    yield
    cudnn.deterministic, cudnn.benchmark = before[0], before[1]
    torch.use_deterministic_algorithms(before[2])


def _train(tmp_path: Path, device: str, tag: str) -> Path:
    """One seeded tiny training on ``device``; returns its checkpoint directory."""
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
        bootstrap="none",  # as A1: every iteration's self-play uses the network on the GPU
        checkpoint_dir=str(out / "ckpt"),
        log_dir=str(out / "log"),
        device=device,
        seed=_SEED,
    )
    train(config)
    return out / "ckpt"


def _weights(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False)
    return {k: v for k, v in checkpoint["state_dict"].items() if isinstance(v, torch.Tensor)}


def _arena_games(tmp_path: Path, device: str, tag: str, checkpoint: Path | None = None) -> Any:
    """A seeded arena on ``device`` (an untrained network unless ``checkpoint``) vs random."""
    from alpha_zero.evaluation import benchmark

    out = tmp_path / tag
    argv = ["--seed", str(_SEED), "--board-size", "6", "--max-turns", "6", "--sims", "4"]
    argv += ["--gumbel-k", "4", "--mcts-type", "gumbel", "--device", device]
    argv += ["--opponents=random", "--n-games", "4", "--out", str(out)]
    if checkpoint is not None:
        argv += ["--checkpoint", str(checkpoint)]
    assert benchmark.main(argv) == 0
    envelope = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert envelope["evidence"]["device"] == device
    return [row["games"] for row in envelope["results"]]


def test_this_environment_has_the_torch_build_the_cards_pin() -> None:
    """A pass on another build says nothing about the registered one, so this fails, not skips."""
    registered = registered_devices(read_registered_protocol(default_protocol_path()))
    assert registered is not None, "section 7.1 registers no devices"
    assert torch_build_matches(torch.__version__, registered.torch_version), (
        f"this environment has torch {torch.__version__}, but the cards pin "
        f"{registered.torch_spec} from {registered.torch_index_url}. Install that build "
        "first; a determinism result on another build is not evidence for the cards."
    )


@pytest.mark.parametrize("device", _devices() or ["cuda:0"])
def test_two_seeded_trainings_on_a_gpu_give_identical_weights(tmp_path: Path, device: str) -> None:
    first = _weights(_train(tmp_path, device, "a"))
    assert torch.are_deterministic_algorithms_enabled(), "train() did not enforce determinism"
    second = _weights(_train(tmp_path, device, "b"))
    assert first.keys() == second.keys() and first, "no weights were compared"
    moved = [name for name in first if not torch.equal(first[name], second[name])]
    assert not moved, (
        f"two trainings at seed {_SEED} on {device} differ in {len(moved)} of {len(first)} "
        f"tensors (first: {moved[:3]}); deterministic algorithms are not holding on this GPU"
    )


@pytest.mark.parametrize("device", _devices() or ["cuda:0"])
def test_two_seeded_arenas_on_a_gpu_play_identical_games(tmp_path: Path, device: str) -> None:
    first = _arena_games(tmp_path, device, "a")
    assert torch.are_deterministic_algorithms_enabled(), "the arena did not enforce determinism"
    assert not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic
    second = _arena_games(tmp_path, device, "b")
    assert first and first == second, f"a seeded arena on {device} did not repeat"


def test_a_checkpoint_trained_on_one_gpu_is_judged_repeatably_on_another(tmp_path: Path) -> None:
    """A1's shape: weights trained on ``cuda:0`` are loaded onto the arena's device.

    The arena device is the last visible GPU: ``cuda:1`` on the two-GPU machine the
    cards are written for, and ``cuda:0`` again where there is only one (which still
    exercises loading a saved checkpoint for the arena).
    """
    arena_device = _devices()[-1]
    checkpoint = _train(tmp_path, "cuda:0", "train") / "latest.pt"
    first = _arena_games(tmp_path, arena_device, "a", checkpoint)
    second = _arena_games(tmp_path, arena_device, "b", checkpoint)
    assert first and first == second
