"""Security and configuration tests for ``scripts/local_muzero_launcher.py``."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from scripts import local_muzero_launcher as launcher

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = REPO_ROOT / "scripts" / "local_muzero_launcher.py"


def test_run_command_disables_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_command`` must always pass ``shell=False`` and ``check=True``."""
    mock_run = MagicMock()
    monkeypatch.setattr(launcher.subprocess, "run", mock_run)

    launcher.run_command([sys.executable, "-V"], env={"CIV6AZ_STORAGE_BACKEND": "local"})

    mock_run.assert_called_once()
    _, kwargs = mock_run.call_args
    assert kwargs["shell"] is False
    assert kwargs["check"] is True


def test_launcher_never_uses_shell_true_in_ast() -> None:
    """Every ``subprocess.*`` call in ``local_muzero_launcher`` must keep ``shell=False``."""
    tree = ast.parse(LAUNCHER_PATH.read_text(encoding="utf-8"), filename=str(LAUNCHER_PATH))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_subprocess = (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
        )
        if not is_subprocess:
            continue
        for kw in node.keywords:
            if kw.arg != "shell":
                continue
            if not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
                offenders.append(f"line {node.lineno}: shell={ast.unparse(kw.value)}")
    assert not offenders, f"Unsafe subprocess shell setting: {offenders}"


def test_main_refuses_without_allow_gcs_and_uses_env_defaults(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without ``--allow-gcs``, the CLI refuses cleanly while naming env-resolved bucket defaults."""
    monkeypatch.setenv("CIV6AZ_GCS_BUCKET", "custom-test-bucket")
    monkeypatch.setenv("CIV6AZ_VERTEX_PROJECT", "custom-test-project")
    monkeypatch.setattr(
        sys,
        "argv",
        ["local_muzero_launcher.py", "--iterations", "1", "--num_selfplay_workers", "1"],
    )

    with pytest.raises(SystemExit) as excinfo:
        launcher.main()

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "REFUSED" in stderr
    assert "custom-test-bucket" in stderr


def test_main_worker_popen_uses_shell_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker ``subprocess.Popen`` invocations must pass ``shell=False``."""
    popen_calls: list[dict[str, object]] = []

    class _FakeProc:
        def wait(self) -> int:
            return 0

    def _fake_popen(cmd: list[str], **kwargs: object) -> _FakeProc:
        popen_calls.append({"cmd": cmd, **kwargs})
        return _FakeProc()

    class _FakeStorage:
        def __init__(self, _cfg: object) -> None:
            pass

        def get_latest_version(self) -> int:
            return 0

    monkeypatch.setattr(launcher, "GCSModelStorage", _FakeStorage)
    monkeypatch.setattr(launcher.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(launcher, "run_command", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "local_muzero_launcher.py",
            "--allow-gcs",
            "--iterations",
            "1",
            "--num_selfplay_workers",
            "1",
            "--games_per_worker",
            "1",
        ],
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())

    launcher.main()

    assert len(popen_calls) == 1
    assert popen_calls[0]["shell"] is False
