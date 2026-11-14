"""A seeded run on a GPU must be made deterministic, and say which GPU it used.

The CPU determinism gates (``tests/regression/test_training_determinism_regression.py``)
cannot see this: on CPU the helper is a no-op by design. What can be tested without
a GPU is the *wiring* -- that every seeded entry point calls
``enforce_deterministic_backend`` with its device before it builds a network, that
the helper sets the switches for a CUDA device and nothing for a CPU one, and that
an explicit GPU that does not exist is refused instead of timed as CPU. The
behaviour on real hardware is ``tests/regression/test_cuda_determinism_regression.py``.

Negative controls (recorded when this file was written, then reverted):
  * deleting the ``enforce_deterministic_backend(config.device)`` line from
    ``training_civ6.train`` made ``test_seeded_training_enforces_...`` fail;
  * deleting the call from ``benchmark.main`` made ``test_seeded_benchmark_...`` fail;
  * deleting the call from ``calibrate`` made ``test_seeded_calibration_...`` fail;
  * replacing the ``DeviceUnavailableError`` raise with ``pass`` made both
    ``test_calibration_refuses_a_gpu_...`` tests fail (the run timed the CPU and
    reported success).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from alpha_zero.utils import seeding

REPO = Path(__file__).resolve().parents[2]
#: An ordinal no test machine has, so "unavailable" holds with or without a GPU.
ABSENT_GPU = "cuda:63"


class _Recorder:
    def __init__(self) -> None:
        self.devices: list[str] = []

    def __call__(self, device: object) -> bool:
        self.devices.append(str(device))
        return False


@pytest.fixture
def restore_backend_flags() -> Any:
    """The helper flips process-global switches; put them back."""
    cudnn = torch.backends.cudnn
    before = (cudnn.deterministic, cudnn.benchmark, torch.are_deterministic_algorithms_enabled())
    env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    yield
    cudnn.deterministic, cudnn.benchmark = before[0], before[1]
    torch.use_deterministic_algorithms(before[2])
    if env is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = env


# --------------------------------------------------------------------------
# The helper
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "ordinal"),
    [
        ("cuda", 0),
        ("CUDA:1", 1),
        (" cuda:12 ", 12),
        ("cpu", None),
        ("auto", None),
        ("mps", None),
        ("cuda:x", None),
        ("", None),
        (None, None),
    ],
)
def test_cuda_ordinal(device: object, ordinal: int | None) -> None:
    assert seeding.cuda_ordinal(device) == ordinal


@pytest.mark.parametrize("device", ["cpu", "auto", "mps", None])
def test_non_cuda_devices_are_left_alone(device: object, restore_backend_flags: Any) -> None:
    cudnn = torch.backends.cudnn
    before = (cudnn.deterministic, cudnn.benchmark, torch.are_deterministic_algorithms_enabled())
    assert seeding.enforce_deterministic_backend(device) is False
    after = (cudnn.deterministic, cudnn.benchmark, torch.are_deterministic_algorithms_enabled())
    assert after == before, "the CPU path must stay byte-identical"


def test_cuda_device_gets_every_determinism_switch(
    restore_backend_flags: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

    assert seeding.enforce_deterministic_backend("cuda:1") is True

    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert torch.are_deterministic_algorithms_enabled() is True
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == seeding.CUBLAS_WORKSPACE_CONFIG


def test_an_operators_cublas_setting_is_not_overwritten(
    restore_backend_flags: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    seeding.enforce_deterministic_backend("cuda")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


# --------------------------------------------------------------------------
# Wiring: every seeded entry point calls it, with its own device
# --------------------------------------------------------------------------


def _tiny_training_config(tmp_path: Path, seed: int | None) -> Any:
    from alpha_zero.training_civ6 import TrainingConfig

    return TrainingConfig(
        board_size=6,
        max_turns=4,
        num_simulations=1,
        num_games_per_iteration=1,
        num_iterations=1,
        num_epochs=1,
        batch_size=4,
        min_replay_size=1,
        num_res_blocks=1,
        num_filters=8,
        eval_games=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "log"),
        device="cpu",
        seed=seed,
    )


def test_seeded_training_enforces_determinism_on_its_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from alpha_zero import training_civ6

    recorder = _Recorder()
    monkeypatch.setattr(training_civ6, "enforce_deterministic_backend", recorder)
    training_civ6.train(_tiny_training_config(tmp_path, seed=7))
    assert recorder.devices == ["cpu"]


def test_unseeded_training_does_not(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from alpha_zero import training_civ6

    recorder = _Recorder()
    monkeypatch.setattr(training_civ6, "enforce_deterministic_backend", recorder)
    training_civ6.train(_tiny_training_config(tmp_path, seed=None))
    assert recorder.devices == []


def _benchmark_argv(out: Path, *, seed: int | None) -> list[str]:
    argv = [
        "--board-size",
        "6",
        "--max-turns",
        "4",
        "--sims",
        "2",
        "--gumbel-k",
        "2",
        "--mcts-type",
        "gumbel",
        "--device",
        "cpu",
        "--candidate",
        "heuristic",
        "--opponents=random",
        "--n-games",
        "2",
        "--out",
        str(out),
    ]
    return argv if seed is None else ["--seed", str(seed), *argv]


@pytest.mark.parametrize(("seed", "expected"), [(11, ["cpu"]), (None, [])])
def test_seeded_benchmark_enforces_determinism_on_its_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seed: int | None, expected: list[str]
) -> None:
    from alpha_zero.evaluation import benchmark

    recorder = _Recorder()
    monkeypatch.setattr(benchmark, "enforce_deterministic_backend", recorder)
    assert benchmark.main(_benchmark_argv(tmp_path / "out", seed=seed)) == 0
    assert recorder.devices == expected


def _calibrate_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "calibrate_throughput_wiring", REPO / "scripts" / "calibrate_throughput.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("seed", "expected"), [(1, ["cpu"]), (None, [])])
def test_seeded_calibration_enforces_determinism_on_its_device(
    monkeypatch: pytest.MonkeyPatch, seed: int | None, expected: list[str]
) -> None:
    module = _calibrate_module()
    recorder = _Recorder()
    monkeypatch.setattr(module, "enforce_deterministic_backend", recorder)
    module.calibrate(4, 5, "gumbel", 4, 4, "cpu", games=1, seed=seed)
    assert recorder.devices == expected


# --------------------------------------------------------------------------
# An explicit GPU that is not there is refused, not timed as CPU
# --------------------------------------------------------------------------


def test_calibration_refuses_a_gpu_this_machine_does_not_have() -> None:
    module = _calibrate_module()
    with pytest.raises(module.DeviceUnavailableError, match="cuda:63"):
        module.calibrate(4, 5, "gumbel", 4, 4, ABSENT_GPU, games=1, seed=1)


def test_calibration_refuses_a_gpu_at_the_cli_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _calibrate_module()
    out = tmp_path / "throughput.json"
    argv = ["--board_size", "4", "--max_turns", "5", "--k", "4", "--n", "4", "--games", "1"]
    code = module.main([*argv, "--device", ABSENT_GPU, "--out", str(out)])
    assert code == 2 and not out.exists()
    assert "REFUSED" in capsys.readouterr().err


def test_calibration_payload_records_the_device_it_ran_on(tmp_path: Path) -> None:
    module = _calibrate_module()
    out = tmp_path / "throughput.json"
    argv = ["--board_size", "4", "--max_turns", "5", "--k", "4", "--n", "4", "--games", "1"]
    assert module.main([*argv, "--device", "cpu", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["device"] == "cpu"


# --------------------------------------------------------------------------
# Evidence names the GPU that was used
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cuda:1", "GPU-1"),
        ("cuda", "GPU-0"),
        ("cpu", "GPU-0"),
        (None, "GPU-0"),
        ("cuda:5", "GPU-0"),
    ],
)
def test_evidence_gpu_model_is_the_device_used(
    monkeypatch: pytest.MonkeyPatch, device: object, expected: str
) -> None:
    from alpha_zero import training_civ6
    from alpha_zero.evaluation import benchmark_evidence

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: f"GPU-{index}")
    assert benchmark_evidence.hardware_summary(device)["gpu_model"] == expected
    assert training_civ6._hardware_summary(device)["gpu_model"] == expected


def test_evidence_gpu_model_is_none_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    from alpha_zero.evaluation import benchmark_evidence

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert benchmark_evidence.hardware_summary("cuda:1")["gpu_model"] is None
