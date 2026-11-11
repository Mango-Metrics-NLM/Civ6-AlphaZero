"""Pinned source and environment (LF-E6).

Negative control (when this file was written, reverted): ``assert_running_pinned``
returning early made ``test_editable_install_fails_preflight`` fail. The test
process imports ``alpha_zero`` from the worktree, which is exactly the situation
the preflight exists to refuse.
"""

from __future__ import annotations

import shutil
import subprocess
import types
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

import alpha_zero
from alpha_zero.local import pinned_env as pe

REPO = Path(alpha_zero.__file__).resolve().parents[1]


class FakeRunner:
    """Records commands; fakes venv/pip, optionally delegates git to the real thing."""

    def __init__(self, *, real_git: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.real_git = real_git

    def __call__(
        self, cmd: Sequence[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None
    ) -> str:
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        if cmd[0] == "git" and self.real_git:
            return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout
        if cmd[0] == "git":
            return "0123456789abcdef0123456789abcdef01234567\n"
        if "venv" in cmd:
            venv = Path(cmd[-1])
            (venv / "Scripts").mkdir(parents=True, exist_ok=True)
            (venv / "Scripts" / "python.exe").write_text("")
            (venv / "bin").mkdir(parents=True, exist_ok=True)
            (venv / "bin" / "python").write_text("")
        if "wheel" in cmd:
            out = Path(cmd[cmd.index("-w") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "alpha_zero-2.0.0-py3-none-any.whl").write_bytes(b"wheel-bytes")
        return ""

    def count(self, needle: str) -> int:
        return sum(1 for c in self.calls if needle in c)


def _fake_src_tree(root: Path, version: str = "2.0.0") -> Path:
    src = root / "src"
    (src / "alpha_zero").mkdir(parents=True)
    (src / "alpha_zero" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    (src / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{version}"\n')
    return src


def test_rejects_bad_sha(tmp_path: Path) -> None:
    with pytest.raises(pe.PinnedEnvError):
        pe.prepare_pinned_run("not a sha; rm -rf", tmp_path / "run", envs_root=tmp_path / "e")


def test_git_archive_of_head_is_extracted_and_read_only(tmp_path: Path) -> None:
    runner = FakeRunner(real_git=True)
    env = pe.prepare_pinned_run(
        "HEAD", tmp_path / "run", envs_root=tmp_path / "envs", runner=runner, repo_root=REPO
    )
    assert env.src == tmp_path / "run" / "src"
    assert (env.src / "pyproject.toml").is_file()
    assert (env.src / "alpha_zero" / "__init__.py").is_file()
    with pytest.raises(PermissionError):
        (env.src / "pyproject.toml").open("a").close()
    flat = [" ".join(c) for c in runner.calls]
    assert any("pip wheel" in c and "--no-deps" in c for c in flat)
    assert not any(" -e " in f" {c} " for c in flat)
    assert env.python.exists()
    assert env.sha and env.wheel_sha256


def test_venv_reused_for_same_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = FakeRunner()
    fake = _fake_src_tree(tmp_path / "fake")
    monkeypatch.setattr(
        pe, "_extract_archive", lambda sha, repo_root, dest, runner: shutil.copytree(fake, dest)
    )
    kwargs = {"envs_root": tmp_path / "envs", "runner": runner, "repo_root": tmp_path}
    pe.prepare_pinned_run("0123456789abcdef", tmp_path / "r1", **kwargs)  # type: ignore[arg-type]
    assert runner.count("venv") == 1
    env2 = pe.prepare_pinned_run("0123456789abcdef", tmp_path / "r2", **kwargs)  # type: ignore[arg-type]
    assert runner.count("venv") == 1, "second run must reuse the venv"
    assert env2.src == tmp_path / "r2" / "src" and env2.wheel_sha256


def test_editable_install_fails_preflight(tmp_path: Path) -> None:
    src = _fake_src_tree(tmp_path)
    env = pe.PinnedEnv(python=tmp_path / "venv" / "Scripts" / "python.exe", src=src)
    with pytest.raises(pe.PinnedEnvError, match="not inside"):
        pe.assert_running_pinned(env)


def test_pinned_module_passes_preflight(tmp_path: Path) -> None:
    src = _fake_src_tree(tmp_path)
    env = pe.PinnedEnv(python=tmp_path / "venv" / "Scripts" / "python.exe", src=src)
    module = types.SimpleNamespace(
        __file__=str(src / "alpha_zero" / "__init__.py"), __version__="2.0.0"
    )
    pe.assert_running_pinned(env, module=module)


def test_installed_in_venv_passes_preflight(tmp_path: Path) -> None:
    src = _fake_src_tree(tmp_path)
    venv = tmp_path / "venv"
    env = pe.PinnedEnv(python=venv / "Scripts" / "python.exe", src=src)
    module = types.SimpleNamespace(
        __file__=str(venv / "Lib" / "site-packages" / "alpha_zero" / "__init__.py"),
        __version__="2.0.0",
    )
    pe.assert_running_pinned(env, module=module)


def test_version_mismatch_fails_preflight(tmp_path: Path) -> None:
    src = _fake_src_tree(tmp_path, version="2.0.0")
    env = pe.PinnedEnv(python=tmp_path / "venv" / "Scripts" / "python.exe", src=src)
    module = types.SimpleNamespace(
        __file__=str(src / "alpha_zero" / "__init__.py"), __version__="1.8.0"
    )
    with pytest.raises(pe.PinnedEnvError, match="version"):
        pe.assert_running_pinned(env, module=module)


def test_child_env_is_isolated() -> None:
    env = pe.child_env({"PATH": "p", "PYTHONPATH": "leak"})
    assert env["PYTHONNOUSERSITE"] == "1"
    assert "PYTHONPATH" not in env
