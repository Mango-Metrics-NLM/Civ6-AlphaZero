"""Review round S1/S2: other ways to point a git command at another repository.

git facts: ``--git-dir`` is not undone by a later absolute ``-C``; the environment variable
``GIT_DIR`` selects the repository the same way; ``env -C <dir>`` / ``env --chdir=<dir>``
change the directory the command runs in. ``GIT_WORK_TREE`` and ``GIT_COMMON_DIR`` alone do
not move HEAD (the git directory is still found from the cwd), so they are tested together
with ``GIT_DIR``, which does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.governance._guard_harness import HAVE_TOOLS, World, build_world, decide

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path)


def test_absolute_dash_c_does_not_discard_an_earlier_git_dir(world: World) -> None:
    got = decide(world.f, f"git --git-dir={world.p}/.git -C {world.f} commit -m x")
    assert got.decision == "deny", got
    got = decide(world.f, f"git --git-dir {world.p}/.git --work-tree {world.p} -C {world.f} commit")
    assert got.decision == "deny", got


def test_git_dir_from_a_variable_asks(world: World) -> None:
    got = decide(world.f, f"git --git-dir=$X -C {world.f} commit -m x")
    assert got.decision == "ask", got
    assert "could not tell which branch" in got.reason


@pytest.mark.parametrize(
    "template",
    [
        "GIT_DIR={p}/.git git commit -m x",
        "env GIT_DIR={p}/.git git commit -m x",
        "export GIT_DIR={p}/.git; git commit -m x",
        "export GIT_DIR={p}/.git && git commit -m x",
        "GIT_DIR={p}/.git GIT_WORK_TREE={p} git commit -m x",
        "GIT_DIR={p}/.git GIT_COMMON_DIR={p}/.git git commit -m x",
        "env GIT_DIR={p}/.git GIT_WORK_TREE={p} git push origin x",
        "env -C {p} git commit -m x",
        "env --chdir={p} git commit -m x",
        "env --chdir {p} git commit -m x",
    ],
)
def test_environment_and_env_chdir_retarget_a_protected_checkout(
    world: World, template: str
) -> None:
    got = decide(world.f, template.format(p=world.p))
    assert got.decision == "deny", got


def test_powershell_env_assignment_retargets(world: World) -> None:
    for cmd in (
        f"$env:GIT_DIR='{world.p}/.git'; git commit -m x",
        f'$env:GIT_DIR="{world.p}/.git"; git commit -m x',
        f"$env:GIT_DIR = '{world.p}/.git'; & git commit -m x",
    ):
        got = decide(world.f, cmd, tool_name="PowerShell")
        assert got.decision == "deny", (cmd, got)


@pytest.mark.parametrize(
    "template",
    [
        "GIT_DIR=$X git commit -m x",
        "env GIT_DIR=$X git commit -m x",
        "export GIT_DIR=$X; git commit -m x",
    ],
)
def test_git_dir_from_a_variable_in_the_environment_asks(world: World, template: str) -> None:
    got = decide(world.f, template)
    assert got.decision == "ask", got
    assert "could not tell which branch" in got.reason


@pytest.mark.parametrize(
    "template",
    [
        "GIT_DIR={f}/.git git commit -m x",
        "env GIT_DIR={f}/.git git commit -m x",
        "env -C {f} git commit -m x",
        "env --chdir={f} git commit -m x",
        "git --git-dir={f}/.git -C {f} commit -m x",
    ],
)
def test_feature_targets_are_allowed_from_a_protected_directory(
    world: World, template: str
) -> None:
    got = decide(world.p, template.format(f=world.f))
    assert got.decision is None, got
