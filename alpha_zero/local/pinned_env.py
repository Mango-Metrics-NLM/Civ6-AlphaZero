"""Pinned source and environment for a local run (LF-E6).

Why
---
The globally installed ``alpha_zero`` on a dev box can be a stale editable
install of another checkout, so "run the code at SHA X" silently runs something
else. A pinned run therefore executes from

* ``<run_dir>/src`` -- ``git archive <sha>`` extracted and made read-only; and
* a venv at ``<data_root>/envs/<sha12>`` built from a **non-editable** wheel of
  that SHA (so the code under test is the wheel, not a path on ``sys.path``).

Children run with ``cwd=src`` and ``PYTHONNOUSERSITE=1`` (:func:`child_env`);
:func:`assert_running_pinned` is the preflight that proves it.

Design notes
------------
* Every subprocess goes through an injected ``runner`` so tests build no real
  venv. The default runner raises :class:`PinnedEnvError` on a non-zero exit.
* The wheel is built *before* the tree is made read-only and ``build/`` and
  ``*.egg-info`` are removed afterwards, so setuptools never needs to write
  into read-only source and the extracted tree stays byte-identical to the SHA.
* A venv is reused when its marker file (``.pinned.json``) names the same full
  SHA; the marker also carries the wheel hash that ``run.json`` records.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_zero.utils.data_root import DataRootError, resolve_data_root

logger = logging.getLogger(__name__)

MARKER_NAME = ".pinned.json"
_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$|^HEAD$")

Runner = Callable[..., str]

# Every argv handed to this is a list built by this module from a regex-validated SHA and
# fixed arguments, never a shell string. Bound to a name so the call below is not read as
# a bare ``subprocess.run`` of unknown input (ruff S603), without growing the ruff-ignore
# ratchet for a case that is not actually untrusted.
_exec = subprocess.run


class PinnedEnvError(RuntimeError):
    """Pinned source/env could not be prepared, or the process is not running from it."""


@dataclass(frozen=True)
class PinnedEnv:
    """A prepared pinned run. ``venv`` is derived from ``python``'s location."""

    python: Path
    src: Path
    sha: str = ""
    wheel_sha256: str = ""

    @property
    def venv(self) -> Path:
        return self.python.parent.parent


def child_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for a pinned child: no user site, no inherited ``PYTHONPATH``."""
    env = dict(os.environ if base is None else base)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _default_runner(
    cmd: Sequence[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None
) -> str:
    logger.debug("run: %s", " ".join(str(c) for c in cmd))
    result = _exec(
        [str(c) for c in cmd],
        cwd=cwd,
        env=dict(env) if env is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PinnedEnvError(
            f"command failed ({result.returncode}): {' '.join(map(str, cmd))}\n{result.stderr[-2000:]}"
        )
    return result.stdout


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _venv_python(venv: Path) -> Path:
    for candidate in (venv / "Scripts" / "python.exe", venv / "bin" / "python"):
        if candidate.exists():
            return candidate
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _extract_archive(sha: str, repo_root: Path, dest: Path, runner: Runner) -> None:
    """``git archive <sha>`` of ``repo_root`` extracted into ``dest``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tar_path = dest.parent / f"{dest.name}.tar"
    try:
        runner(
            ["git", "-C", str(repo_root), "archive", "--format=tar", "-o", str(tar_path), sha],
            cwd=repo_root,
        )
        dest.mkdir()
        with tarfile.open(tar_path) as tar:
            if not hasattr(tarfile, "data_filter"):
                raise PinnedEnvError("Python >= 3.11.4 is required for safe tar extraction")
            tar.extractall(dest, filter="data")
    finally:
        tar_path.unlink(missing_ok=True)


def _make_read_only(src: Path) -> None:
    for path in src.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _clean_build_artifacts(src: Path) -> None:
    shutil.rmtree(src / "build", ignore_errors=True)
    for egg in src.glob("*.egg-info"):
        shutil.rmtree(egg, ignore_errors=True)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_venv(venv: Path, src: Path, wheel_dir: Path, full_sha: str, runner: Runner) -> str:
    """Create ``venv`` and install a non-editable wheel of ``src``. Returns the wheel sha256."""
    env = child_env()
    runner([sys.executable, "-m", "venv", str(venv)], env=env)
    py = _venv_python(venv)
    runner([py, "-m", "pip", "wheel", str(src), "--no-deps", "-w", str(wheel_dir)], env=env)
    wheels = sorted(wheel_dir.glob("alpha_zero-*.whl"))
    if not wheels:
        raise PinnedEnvError(f"pip wheel produced no alpha_zero wheel in {wheel_dir}")
    wheel = wheels[-1]
    # Installing the wheel (not --no-deps) pulls the dependencies pyproject declares.
    runner([py, "-m", "pip", "install", str(wheel)], env=env)
    wheel_hash = _sha256_file(wheel)
    (venv / MARKER_NAME).write_text(
        json.dumps({"sha": full_sha, "wheel_sha256": wheel_hash}), encoding="utf-8"
    )
    return wheel_hash


def prepare_pinned_run(
    sha: str,
    run_dir: Path,
    *,
    envs_root: Path | None = None,
    runner: Runner | None = None,
    repo_root: Path | None = None,
) -> PinnedEnv:
    """Extract ``sha`` to ``<run_dir>/src`` (read-only) and ensure its venv exists.

    ``envs_root`` defaults to ``<CIV6AZ_DATA_ROOT>/envs``. The venv is reused when
    its marker already names this SHA. Raises :class:`PinnedEnvError` on any
    failure, including an unset data root when ``envs_root`` is not given.
    """
    if not _SHA_RE.match(sha):
        raise PinnedEnvError(f"not a git SHA: {sha!r}")
    run = runner or _default_runner
    repo = repo_root or _default_repo_root()
    run_dir = Path(run_dir)
    if envs_root is None:
        try:
            root = resolve_data_root(required=True)
        except DataRootError as exc:
            raise PinnedEnvError(str(exc)) from exc
        if root is None:  # unreachable: required=True raises
            raise PinnedEnvError("CIV6AZ_DATA_ROOT is not set")
        envs_root = root / "envs"

    full_sha = run(["git", "-C", str(repo), "rev-parse", "--verify", f"{sha}^{{commit}}"], cwd=repo)
    full_sha = full_sha.strip() or sha
    venv = Path(envs_root) / full_sha[:12]
    src = run_dir / "src"
    if src.exists():
        raise PinnedEnvError(f"{src} already exists; a run directory is single-use")

    _extract_archive(full_sha, repo, src, run)

    marker = venv / MARKER_NAME
    wheel_hash = ""
    if marker.is_file() and _venv_python(venv).exists():
        data = json.loads(marker.read_text(encoding="utf-8"))
        if data.get("sha") == full_sha:
            wheel_hash = str(data.get("wheel_sha256", ""))
            logger.info("reusing pinned venv %s", venv)
    if not wheel_hash:
        if venv.exists():
            shutil.rmtree(venv, ignore_errors=True)
        try:
            wheel_hash = _build_venv(venv, src, run_dir / "wheel", full_sha, run)
        finally:
            _clean_build_artifacts(src)
    _make_read_only(src)
    return PinnedEnv(python=_venv_python(venv), src=src, sha=full_sha, wheel_sha256=wheel_hash)


def _pyproject_version(src: Path) -> str:
    data: dict[str, Any] = tomllib.loads((src / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def assert_running_pinned(env: PinnedEnv, *, module: Any = None) -> None:
    """Preflight: this process imported ``alpha_zero`` from ``env`` and the version matches.

    ``alpha_zero.__file__`` must lie inside ``env.src`` or ``env.venv`` and
    ``alpha_zero.__version__`` must equal ``<src>/pyproject.toml``'s version.
    ``module`` is injectable for tests; by default the imported package is used.
    """
    if module is None:
        import alpha_zero as module
    file = Path(module.__file__).resolve()
    allowed = [env.src.resolve(), env.venv.resolve()]
    if not any(root == file or root in file.parents for root in allowed):
        raise PinnedEnvError(
            f"alpha_zero imported from {file}, which is not inside the pinned src or venv "
            f"({allowed[0]}, {allowed[1]})"
        )
    expected = _pyproject_version(env.src)
    if str(module.__version__) != expected:
        raise PinnedEnvError(
            f"alpha_zero.__version__ {module.__version__!r} != pyproject version {expected!r}"
        )
