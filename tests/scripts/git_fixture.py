"""A throwaway git repository for the branch-policy and release-script tests.

Not a test module (no ``test_`` prefix), so pytest does not collect it.

Every remote-tracking ref the scripts read (``origin/dev``, ``origin/qa``,
``origin/main``) is written with ``git update-ref`` instead of being fetched from
a real remote, so a test states the exact remote state it needs in one line and
nothing here can reach GitHub. Tests that need a real ``git fetch``/``git push``
(``tag_release``) use :meth:`GitRepo.with_bare_remote`, whose remote is a bare
repository in the same temporary directory.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_IDENTITY = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
    # The developer's global config (init.defaultBranch, commit.gpgsign, hooks)
    # must not leak into a fixture.
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def git_env() -> dict[str, str]:
    """The environment every fixture git call (and the scripts under test) use."""
    return {**os.environ, **_IDENTITY}


class GitRepo:
    """A working repository with helpers for building branch topologies."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self.git("init", "-q", "-b", "main")
        self.git("config", "commit.gpgsign", "false")
        self.git("config", "tag.gpgsign", "false")
        self._counter = 0

    # -- plumbing ---------------------------------------------------------
    def git(self, *args: str, check: bool = True) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=self.path,
            env=git_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr}")
        return proc.stdout.strip()

    def sha(self, rev: str = "HEAD") -> str:
        return self.git("rev-parse", f"{rev}^{{commit}}")

    # -- building history -------------------------------------------------
    def commit(self, message: str, files: dict[str, str] | None = None) -> str:
        """Commit ``files`` (path -> content) on the current branch; return the SHA."""
        self._counter += 1
        files = files or {f"notes/{self._counter}.txt": f"{message}\n"}
        for rel, content in files.items():
            target = self.path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            self.git("add", rel)
        self.git("commit", "-q", "-m", message)
        return self.sha()

    def checkout(self, branch: str, start: str | None = None) -> None:
        if start is None:
            self.git("checkout", "-q", branch)
        else:
            self.git("checkout", "-q", "-B", branch, start)

    def merge(self, rev: str, message: str) -> str:
        """A true merge commit (``--no-ff``) of ``rev`` into the current branch."""
        self.git("merge", "-q", "--no-ff", "-m", message, rev)
        return self.sha()

    def set_remote(self, branch: str, rev: str, remote: str = "origin") -> None:
        """Point ``<remote>/<branch>`` at ``rev`` without any network."""
        self.git("update-ref", f"refs/remotes/{remote}/{branch}", self.sha(rev))

    def tag(self, name: str, rev: str = "HEAD") -> None:
        self.git("tag", "-a", name, "-m", name, self.sha(rev))

    # -- a real (local, bare) remote --------------------------------------
    @classmethod
    def with_bare_remote(cls, root: Path) -> tuple[GitRepo, Path]:
        bare = root / "remote.git"
        subprocess.run(
            ["git", "init", "-q", "--bare", "-b", "main", str(bare)],
            env=git_env(),
            check=True,
            capture_output=True,
        )
        repo = cls(root / "work")
        repo.git("remote", "add", "origin", bare.as_posix())
        return repo, bare

    def push(self, *refspecs: str) -> None:
        self.git("push", "-q", "origin", *refspecs)
