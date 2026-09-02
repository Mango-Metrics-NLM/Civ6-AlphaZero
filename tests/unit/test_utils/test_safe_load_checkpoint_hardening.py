"""WS4.1 follow-up: ``alpha_zero.utils.checkpoint.safe_load_checkpoint`` no longer
downgrades to ``weights_only=False`` on its own.

This is the loader the six live sites use (``MicroCivNetwork.load``,
``Civ6Network.load`` x2, ``MuZeroNetwork.load``, ``Learner``/``DDPLearner``
resume, ``MuZeroTrainer.load_checkpoint``). Until 2026-09-02
``allow_unsafe_fallback`` defaulted to ``True`` *and* every live caller passed
it explicitly, so a payload the tensor-only unpickler rejected was always
re-unpickled with arbitrary code execution.

The tests mirror ``TestNoAutomaticUnsafeFallback`` in
``tests/unit/test_checkpoint_utils_coverage.py``. The non-tensor negative
control would have passed *silently* under the old code (the dataclass instance
came back as a value with ``loaded_safely=False``); it only holds because the
downgrade is gone.

Negative control (run 2026-09-02)::

    $ sed -i 's/    allow_unsafe_fallback: bool = False,/    allow_unsafe_fallback: bool = True,/' \\
          alpha_zero/utils/checkpoint.py
    $ python -m pytest tests/unit/test_utils/test_safe_load_checkpoint_hardening.py -q --tb=no
    FAILED ...::test_default_is_safe
    FAILED ...::test_non_tensor_payload_raises_without_flag        (Failed: DID NOT RAISE)
    FAILED ...::test_env_var_authorises_unsafe_load[1|true|YES| on ]  (x4: warning credits
                                                                    allow_unsafe_fallback, not the env var)
    FAILED ...::test_env_var_falsy_values_do_not_authorise[0|false||no] (x4: DID NOT RAISE)
    FAILED ...::test_env_var_is_read_at_call_time                   (DID NOT RAISE)
    FAILED ...::test_live_network_loader_does_not_opt_in            (DID NOT RAISE)
    12 failed, 5 passed, 12 warnings in 5.01s
    (restored from a scratchpad copy; no git state change; the WARNING lines in
    the captured log show every one of those loads went through weights_only=False)
"""

from __future__ import annotations

import inspect
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from alpha_zero.utils.checkpoint import CheckpointLoadResult, safe_load_checkpoint
from alpha_zero.utils.safe_load import UNSAFE_CHECKPOINT_ENV_VAR, UnsafeCheckpointError

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class LegacyConfig:
    """A dataclass *instance* is not a torch safe global, so it is rejected."""

    lr: float = 0.01


def _write_non_tensor_payload(path: Path) -> None:
    torch.save({"config": LegacyConfig(), "model_state_dict": nn.Linear(2, 2).state_dict()}, path)


def _write_tensor_payload(path: Path) -> None:
    torch.save({"model_state_dict": nn.Linear(2, 2).state_dict(), "config": {"lr": 0.01}}, path)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(UNSAFE_CHECKPOINT_ENV_VAR, raising=False)


def test_default_is_safe() -> None:
    """The parameter keeps its name for compatibility but now defaults to False."""
    default = inspect.signature(safe_load_checkpoint).parameters["allow_unsafe_fallback"].default
    assert default is False


def test_safe_load_works_for_tensor_payload(tmp_path: Path) -> None:
    path = tmp_path / "ok.pt"
    _write_tensor_payload(path)
    result = safe_load_checkpoint(path)
    assert isinstance(result, CheckpointLoadResult)
    assert result.loaded_safely is True
    assert result.warnings == []
    assert "model_state_dict" in result.data


def test_non_tensor_payload_raises_without_flag(tmp_path: Path) -> None:
    """Negative control: passed silently (loaded_safely=False) under the old default."""
    path = tmp_path / "legacy.pt"
    _write_non_tensor_payload(path)
    with pytest.raises(UnsafeCheckpointError) as excinfo:
        safe_load_checkpoint(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert "allow_unsafe_fallback=True" in message
    assert UNSAFE_CHECKPOINT_ENV_VAR in message
    assert excinfo.value.__cause__ is not None
    # Historical contract: callers that catch RuntimeError keep working.
    assert isinstance(excinfo.value, RuntimeError)


def test_non_tensor_payload_loads_with_flag_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "legacy.pt"
    _write_non_tensor_payload(path)
    with (
        caplog.at_level(logging.WARNING, logger="alpha_zero.utils.safe_load"),
        pytest.warns(UserWarning, match="weights_only=False"),
    ):
        result = safe_load_checkpoint(path, allow_unsafe_fallback=True)
    assert result.loaded_safely is False
    assert isinstance(result.data["config"], LegacyConfig)
    assert result.warnings and "allow_unsafe_fallback=True" in result.warnings[0]
    logged = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert logged, "the unsafe path must be logged at WARNING"
    assert "weights_only=False" in logged[0].getMessage()


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_env_var_authorises_unsafe_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    path = tmp_path / "legacy.pt"
    _write_non_tensor_payload(path)
    monkeypatch.setenv(UNSAFE_CHECKPOINT_ENV_VAR, value)
    with pytest.warns(UserWarning, match=UNSAFE_CHECKPOINT_ENV_VAR):
        result = safe_load_checkpoint(path)
    assert result.loaded_safely is False
    assert isinstance(result.data["config"], LegacyConfig)


@pytest.mark.parametrize("value", ["0", "false", "", "no"])
def test_env_var_falsy_values_do_not_authorise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    path = tmp_path / "legacy.pt"
    _write_non_tensor_payload(path)
    monkeypatch.setenv(UNSAFE_CHECKPOINT_ENV_VAR, value)
    with pytest.raises(UnsafeCheckpointError):
        safe_load_checkpoint(path)


def test_env_var_is_read_at_call_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "legacy.pt"
    _write_non_tensor_payload(path)
    with pytest.raises(UnsafeCheckpointError):
        safe_load_checkpoint(path)
    monkeypatch.setenv(UNSAFE_CHECKPOINT_ENV_VAR, "1")
    with pytest.warns(UserWarning):
        assert safe_load_checkpoint(path).loaded_safely is False


def test_strict_security_overrides_every_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``strict_security=True`` refuses even when both opt-ins are present."""
    path = tmp_path / "legacy.pt"
    _write_non_tensor_payload(path)
    monkeypatch.setenv(UNSAFE_CHECKPOINT_ENV_VAR, "1")
    with pytest.raises(UnsafeCheckpointError, match="strict_security=True"):
        safe_load_checkpoint(path, strict_security=True, allow_unsafe_fallback=True)


def test_live_network_loader_does_not_opt_in(tmp_path: Path) -> None:
    """``MicroCivNetwork.load`` -- a live caller -- no longer bypasses the gate."""
    from alpha_zero.core.civ6_network import MicroCivNetwork

    path = tmp_path / "legacy.pt"
    torch.save({"config": LegacyConfig(), "state_dict": {}}, path)
    with pytest.raises(UnsafeCheckpointError):
        MicroCivNetwork.load(path)


def test_safe_load_module_is_a_leaf() -> None:
    """Importing the shared helper must not pull in ``alpha_zero.core``.

    Checked in a fresh interpreter so this test's own imports cannot mask it.
    The leaf property is what lets ``utils.checkpoint`` share the helper
    without a ``utils -> core -> utils`` cycle.
    """
    code = (
        "import sys, alpha_zero.utils.safe_load; "
        "print(sorted(m for m in sys.modules if m.startswith('alpha_zero.core')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout


def test_utils_checkpoint_imports_first_without_cycle() -> None:
    """``utils.checkpoint`` imported before anything in ``core`` must not raise."""
    code = (
        "import alpha_zero.utils.checkpoint as c; "
        "import alpha_zero.core.checkpoint_utils as u; "
        "assert u.safe_torch_load is c.safe_torch_load; print('ok')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "ok"
