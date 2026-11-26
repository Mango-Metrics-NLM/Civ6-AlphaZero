"""PR #171 round 1, R3: a ``cd`` inside ``( ... )`` or ``$( ... )`` does not survive the ``)``.

Bash restores the directory at the closing parenthesis, so ``(cd F && true) && git commit``
commits in the original checkout. A branch change inside the subshell stays (HEAD changed on
disk). PowerShell does not scope parentheses, so a cd there leaves the directory unknown.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.governance._guard_harness import HAVE_TOOLS, World, build_world, decide

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path)


def test_cd_inside_parentheses_is_undone_at_the_close(world: World) -> None:
    got = decide(world.p, f"(cd {world.f} && true) && git commit -m x")
    assert got.decision == "deny", got


def test_cd_alone_inside_parentheses_then_semicolon(world: World) -> None:
    got = decide(world.p, f"(cd {world.f}) ; git commit -m x")
    assert got.decision == "deny", got


def test_nested_parentheses_restore_each_level(world: World) -> None:
    got = decide(world.p, f"((cd {world.f}) && true) && git commit -m x")
    assert got.decision == "deny", got


def test_cd_to_main_inside_parentheses_does_not_leak_out(world: World) -> None:
    got = decide(world.f, f"(cd {world.p} && true) && git commit -m x")
    assert got.decision is None, got


def test_commit_inside_the_parentheses_uses_the_cd(world: World) -> None:
    got = decide(world.f, f"(cd {world.p} && git commit -m x)")
    assert got.decision == "deny", got


def test_command_substitution_restores_the_directory(world: World) -> None:
    got = decide(world.p, f"echo $(cd {world.f} && pwd) && git commit -m x")
    assert got.decision == "deny", got


def test_exported_git_dir_inside_parentheses_is_restored(world: World) -> None:
    got = decide(world.p, f"(export GIT_DIR={world.f}/.git) && git commit -m x")
    assert got.decision == "deny", got


def test_a_switch_inside_parentheses_persists(world: World) -> None:
    got = decide(world.f, "(git switch main) && git commit -m x")
    assert got.decision == "deny", got


def test_powershell_parentheses_do_not_scope_a_directory_change(world: World) -> None:
    got = decide(world.f, f"(Set-Location {world.p}); git commit -m x", tool_name="PowerShell")
    assert got.decision in ("ask", "deny"), got


def test_an_unmatched_close_leaves_the_directory_unknown(world: World) -> None:
    # From main to a feature checkout: keeping the cd past the unmatched `)` would allow.
    got = decide(world.p, f"cd {world.f} && case x in x) true ;; esac && git commit -m x")
    assert got.decision == "ask", got
    assert "could not tell which branch" in got.reason
