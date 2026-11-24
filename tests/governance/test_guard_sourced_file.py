"""The guard sources ``_guard_git.sh`` from its own directory; without it the guard must DENY.

The source happens after the EXIT trap is set, so a missing or unreadable file is a crash,
and a crash is a deny with exit 0 (a non-zero exit would let the tool call run).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.governance._guard_harness import (
    BASH,
    GUARD,
    HAVE_TOOLS,
    REPO_ROOT,
    base_env,
    parse,
    payload,
)

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")

SIBLING = GUARD.parent / "_guard_git.sh"


def _run(hooks: Path, command: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [BASH, (hooks / "pretooluse_guard.sh").as_posix()],
        cwd=REPO_ROOT,
        input=payload(command).encode("utf-8"),
        env=base_env(),
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_the_sourced_git_rules_file_exists() -> None:
    assert SIBLING.is_file()


def test_a_copy_with_the_sibling_works(tmp_path: Path) -> None:
    shutil.copy(GUARD, tmp_path / GUARD.name)
    shutil.copy(SIBLING, tmp_path / SIBLING.name)
    got = parse(_run(tmp_path, "git push origin HEAD:qa"))
    assert got.decision == "deny"
    assert "protected" in got.reason
    assert parse(_run(tmp_path, "echo hello")).decision is None


@pytest.mark.parametrize("command", ["echo hello", "git status", "git push origin HEAD:qa"])
def test_missing_sibling_denies_every_command(tmp_path: Path, command: str) -> None:
    shutil.copy(GUARD, tmp_path / GUARD.name)  # _guard_git.sh deliberately absent
    proc = _run(tmp_path, command)
    got = parse(proc)  # exit code 0 and valid JSON
    assert got.decision == "deny", got
    assert "failed unexpectedly" in got.reason
