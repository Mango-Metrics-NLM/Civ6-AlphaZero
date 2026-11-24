"""PR #171 round 1, independent probe: git shapes the guard misread (silent allows).

- ``git checkout feature afile`` / ``git checkout feature .`` were read as branch changes;
  with more than one positional a checkout restores paths and moves no branch;
- ``git switch -t origin/main`` creates and switches to ``main``; it was not followed;
- ``git switch ma\\in`` is ``main`` in bash; ``--help`` switches nothing;
- a ``cd`` through a symbolic link is logical in bash (``cd fl/..`` returns to where it started),
  but the guard handed the chain to ``git -C``, which resolves it physically;
- GIT_DIR set in ways the guard does not follow (``declare -x``, ``set -a``, PowerShell's
  ``Set-Item Env:`` ...) was silently ignored; now any other mention makes later commits ask.

``p`` is on ``main`` (protected), ``f`` on ``feature``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.governance._guard_harness import (
    HAVE_TOOLS,
    World,
    build_world,
    decide,
    git,
)

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")


@pytest.fixture
def world(tmp_path: Path) -> World:
    w = build_world(tmp_path)
    git(w.p, "branch", "feature")
    return w


@pytest.fixture
def g(world: World) -> Path:
    """A checkout on `feature` with no local `main`, but `origin/main`."""
    repo = world.root / "g"
    git(world.root, "clone", "-q", str(world.origin), str(repo))
    git(repo, "checkout", "-q", "-b", "feature")
    git(repo, "branch", "-q", "-D", "main")
    return repo


# --- 3: branch changes -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git checkout feature afile && git commit -m x",
        "git checkout feature . && git commit -m x",
        "git switch feature --help && git commit -m x",
        "git switch -h feature && git commit -m x",
    ],
)
def test_no_branch_change_leaves_main(world: World, command: str) -> None:
    assert decide(world.p, command).decision == "deny"


@pytest.mark.parametrize(
    "command",
    [
        "git switch -t origin/main && git commit -m x",
        "git checkout --track origin/main && git commit -m x",
        "git switch --track=direct origin/main && git commit -m x",
        "git checkout -t origin/main && git push origin HEAD",
    ],
)
def test_tracking_a_remote_branch_switches_to_its_name(g: Path, command: str) -> None:
    assert decide(g, command).decision == "deny"


def test_tracking_a_branch_of_an_unknown_remote_asks(g: Path) -> None:
    got = decide(g, "git switch -t nosuchremote/x && git commit -m x")
    assert got.decision == "ask", got


def test_tracking_a_feature_branch_is_allowed(g: Path) -> None:
    git(g, "push", "-q", "origin", "feature:feature2")
    git(g, "fetch", "-q", "origin")
    assert decide(g, "git switch -t origin/feature2 && git commit -m x").decision is None


@pytest.mark.parametrize("sub", ["switch", "checkout"])
def test_a_backslash_in_a_branch_name_is_removed_as_bash_does(world: World, sub: str) -> None:
    assert decide(world.f, f"git {sub} ma\\in && git commit -m x").decision == "deny"


# --- 4: a logical cd through a symbolic link ------------------------------------------------------


@pytest.fixture
def linked(world: World) -> World:
    """`p/fl -> ../f/sub`: `cd fl/..` in bash returns to p, physically it is f."""
    (world.f / "sub").mkdir()
    try:
        os.symlink(Path("..") / "f" / "sub", world.p / "fl", target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            raise
        # A symbolic link needs a privilege on Windows; a directory junction does not, and a
        # logical `cd` follows it the same way. (A skip would raise the debt ratchet's count.)
        import _winapi

        _winapi.CreateJunction(str(world.f / "sub"), str(world.p / "fl"))
    return world


@pytest.mark.parametrize(
    ("where", "command"),
    [
        ("p", "cd fl/.. && git commit -m x"),
        ("p", "cd fl && cd .. && git commit -m x"),
        ("root", "cd p/fl/.. && git commit -m x"),
    ],
)
def test_cd_through_a_symlink_is_logical(linked: World, where: str, command: str) -> None:
    assert decide(getattr(linked, where), command).decision == "deny"


def test_cd_into_the_symlink_itself_is_its_target(linked: World) -> None:
    assert decide(linked.p, "cd fl && git commit -m x").decision is None


def test_cd_dash_p_is_physical(linked: World) -> None:
    assert decide(linked.p, "cd -P fl/.. && git commit -m x").decision is None


# --- 7: GIT_DIR forms the guard does not follow ---------------------------------------------------


@pytest.mark.parametrize(
    ("command", "tool"),
    [
        ("GIT_DIR=../p/.git; export GIT_DIR; git commit -m x", "Bash"),
        ("declare -x GIT_DIR=../p/.git; git commit -m x", "Bash"),
        ("set -a; GIT_DIR=../p/.git; git commit -m x", "Bash"),
        ("declare -x GIT_WORK_TREE=../p; git commit -m x", "Bash"),
        ("${env:GIT_DIR} = '../p/.git'; git commit -m x", "PowerShell"),
        ("Set-Item Env:GIT_DIR ../p/.git; git commit -m x", "PowerShell"),
        (
            "[Environment]::SetEnvironmentVariable('GIT_DIR','../p/.git'); git commit -m x",
            "PowerShell",
        ),
    ],
)
def test_an_unfollowed_git_dir_makes_later_commits_ask(
    world: World, command: str, tool: str
) -> None:
    got = decide(world.f, command, tool_name=tool)
    assert got.decision == "ask", got


@pytest.mark.parametrize(
    "command",
    [
        "export GIT_DIR=../p/.git && git commit -m x",
        "GIT_DIR=../p/.git git commit -m x",
        "env GIT_DIR=../p/.git git commit -m x",
    ],
)
def test_followed_git_dir_forms_still_deny(world: World, command: str) -> None:
    assert decide(world.f, command).decision == "deny"


def test_exporting_only_a_work_tree_is_followed(world: World) -> None:
    assert decide(world.f, "export GIT_WORK_TREE=. && git commit -m x").decision is None
