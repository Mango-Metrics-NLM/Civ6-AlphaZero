"""``read_git_sha`` in worktrees, pinned exports and under ``CIV6AZ_GIT_SHA`` (T10 review I1).

Before this, three ordinary situations produced ``git_sha: null`` in every
checkpoint and every ``eval_<iter>.json``:

* a git worktree -- the branch ref lives in the *common* dir named by
  ``<gitdir>/commondir``, and only the per-worktree gitdir was searched;
* a run-card run -- children execute from a ``git archive`` export, which has no
  ``.git`` at all;
* any run launched from another directory -- the walk started at the cwd, and
  could pick up an unrelated ancestor repository's HEAD instead.

The fixtures build real repositories with the ``git`` CLI inside ``tmp_path``.
"""

from __future__ import annotations

import io
import logging
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from alpha_zero.utils import checkpoint
from alpha_zero.utils.checkpoint import GIT_SHA_ENV, read_git_sha

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs the git CLI")

VALID_SHA = "0123456789abcdef0123456789abcdef01234567"


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _repo(path: Path, content: str = "x") -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main")
    (path / "f.txt").write_text(content, encoding="utf-8")
    _git(path, "add", "f.txt")
    _git(path, "commit", "-q", "-m", "init")
    return _git(path, "rev-parse", "HEAD")


@pytest.fixture(autouse=True)
def _no_env_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GIT_SHA_ENV, raising=False)


class TestEnvVar:
    def test_env_var_wins_over_the_checkout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sha = _repo(tmp_path / "repo")
        assert read_git_sha(tmp_path / "repo") == sha
        monkeypatch.setenv(GIT_SHA_ENV, VALID_SHA.upper())
        assert read_git_sha(tmp_path / "repo") == VALID_SHA

    @pytest.mark.parametrize("bad", ["", "abc123", VALID_SHA + "0", "g" * 40, "unknown"])
    def test_invalid_env_value_is_ignored_with_a_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        bad: str,
    ) -> None:
        sha = _repo(tmp_path / "repo")
        monkeypatch.setenv(GIT_SHA_ENV, bad)
        with caplog.at_level(logging.WARNING, logger=checkpoint.__name__):
            assert read_git_sha(tmp_path / "repo") == sha
        assert GIT_SHA_ENV in caplog.text


class TestWorktree:
    def test_worktree_resolves_through_commondir(self, tmp_path: Path) -> None:
        _repo(tmp_path / "main")
        _git(tmp_path / "main", "worktree", "add", "-q", "-b", "feature", str(tmp_path / "wt"))
        (tmp_path / "wt" / "g.txt").write_text("y", encoding="utf-8")
        _git(tmp_path / "wt", "add", "g.txt")
        _git(tmp_path / "wt", "commit", "-q", "-m", "feature")
        expected = _git(tmp_path / "wt", "rev-parse", "HEAD")
        assert read_git_sha(tmp_path / "wt") == expected
        assert read_git_sha(tmp_path / "wt" / "deeper" / "dir") == expected

    def test_worktree_ref_in_packed_refs(self, tmp_path: Path) -> None:
        _repo(tmp_path / "main")
        _git(tmp_path / "main", "worktree", "add", "-q", "-b", "feature", str(tmp_path / "wt"))
        _git(tmp_path / "main", "pack-refs", "--all")
        common = Path(_git(tmp_path / "main", "rev-parse", "--absolute-git-dir"))
        assert not (common / "refs" / "heads" / "feature").exists(), "the ref must be packed"
        assert read_git_sha(tmp_path / "wt") == _git(tmp_path / "wt", "rev-parse", "HEAD")

    def test_relative_gitdir_pointer_resolves_against_the_dot_git_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _repo(tmp_path / "main")
        _git(tmp_path / "main", "worktree", "add", "-q", "-b", "feature", str(tmp_path / "wt"))
        dot_git = tmp_path / "wt" / ".git"
        absolute = Path(dot_git.read_text(encoding="utf-8").split(":", 1)[1].strip())
        # Recreate rather than overwrite: git marks the file hidden on Windows, and
        # opening a hidden file for writing there raises PermissionError.
        dot_git.unlink()
        dot_git.write_text("gitdir: ../main/.git/worktrees/" + absolute.name + "\n", "utf-8")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)  # a cwd-relative resolution would miss it
        assert read_git_sha(tmp_path / "wt") == _git(tmp_path / "main", "rev-parse", "feature")


def _export(repo: Path, dest: Path) -> None:
    """``git archive HEAD`` extracted into ``dest``: the tree, with no ``.git``."""
    data = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")


class TestPinnedExport:
    def test_export_without_dot_git_is_none_without_env_and_the_sha_with_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sha = _repo(tmp_path / "repo")
        export = tmp_path / "run" / "src"
        _export(tmp_path / "repo", export)
        assert not (export / ".git").exists()
        # The default lookup is anchored on the package's own source root.
        monkeypatch.setattr(checkpoint, "_SOURCE_ROOT", export)
        assert read_git_sha() is None
        monkeypatch.setenv(GIT_SHA_ENV, sha)
        assert read_git_sha() == sha

    def test_unrelated_ancestor_repository_is_never_picked_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outer = _repo(tmp_path / "outer", content="unrelated")
        pinned = _repo(tmp_path / "pinned", content="pinned")
        export = tmp_path / "outer" / "data" / "runs" / "card" / "src"
        _export(tmp_path / "pinned", export)
        monkeypatch.setattr(checkpoint, "_SOURCE_ROOT", export)
        monkeypatch.chdir(export)
        assert read_git_sha() is None, "must not return the enclosing repository's HEAD"
        assert read_git_sha() != outer
        monkeypatch.setenv(GIT_SHA_ENV, pinned)
        assert read_git_sha() == pinned

    def test_default_start_is_the_package_not_the_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        here = _repo(tmp_path / "cwd_repo", content="cwd")
        monkeypatch.chdir(tmp_path / "cwd_repo")
        assert read_git_sha() != here
        assert checkpoint._SOURCE_ROOT == Path(checkpoint.__file__).resolve().parents[2]
