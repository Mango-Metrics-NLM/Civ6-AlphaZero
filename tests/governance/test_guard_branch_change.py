"""Review round S3/S4: the branch changes earlier in the same command; ambiguous names.

``git switch main && git commit`` commits on ``main`` although the checkout is on ``feature``
when the guard looks. A switch or checkout to a protected name, followed by a commit or push,
denies; one to a name the guard cannot read asks; a path checkout is not a branch change.
S4: with a TAG named ``main`` too, ``git symbolic-ref --short HEAD`` prints ``heads/main``
(measured), which must still be read as the protected branch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.governance._guard_harness import HAVE_TOOLS, World, build_world, decide, git

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")


@pytest.fixture
def world(tmp_path: Path) -> World:
    w = build_world(tmp_path)
    git(w.f, "branch", "feature2")
    return w


@pytest.mark.parametrize(
    "command",
    [
        "git switch main && git commit -m x",
        "git checkout main; git commit -m x",
        "git checkout -q main && git commit -m x",
        "git checkout -B main; git commit -m x",
        "git switch -C main && git commit -m x",
        "git checkout main && git push origin feature",
        "git switch qa && git commit -m x",
        "git checkout 'main' && git commit -m x",
    ],
)
def test_switching_to_a_protected_name_then_committing_is_denied(
    world: World, command: str
) -> None:
    got = decide(world.f, command)
    assert got.decision == "deny", got


@pytest.mark.parametrize(
    "command",
    [
        "git switch - && git commit -m x",
        'git switch "$B" && git commit -m x',
        "git checkout $B; git commit -m x",
    ],
)
def test_switching_to_a_name_the_guard_cannot_read_then_committing_asks(
    world: World, command: str
) -> None:
    got = decide(world.f, command)
    assert got.decision == "ask", got
    assert "could not tell which branch" in got.reason


@pytest.mark.parametrize(
    "command",
    [
        "git switch feature2 && git commit -m x",
        "git checkout feature2; git commit -m x",
        "git checkout -b newbranch && git commit -m x",
        "git switch -c newbranch && git commit -m x",
        "git checkout -- file.txt && git commit -m x",
        "git checkout . && git commit -m x",
        "git switch main && git status",
    ],
)
def test_other_branch_changes_and_path_checkouts_are_allowed(world: World, command: str) -> None:
    assert decide(world.f, command).decision is None


def test_switching_back_to_a_feature_branch_clears_the_protected_one(world: World) -> None:
    got = decide(world.f, "git switch main && git switch feature2 && git commit -m x")
    assert got.decision is None, got


def test_a_commit_on_main_is_denied_when_a_tag_named_main_exists(world: World) -> None:
    git(world.p, "tag", "main")
    assert git(world.p, "symbolic-ref", "--short", "HEAD") == "heads/main"
    got = decide(world.p, "git commit -m x")
    assert got.decision == "deny", got
    got = decide(world.f, f"git -C {world.p} commit -m x")
    assert got.decision == "deny", got
    got = decide(world.p, "git push origin HEAD:feature")
    assert got.decision == "deny", got


def test_a_feature_branch_is_still_allowed_when_a_tag_shares_its_name(world: World) -> None:
    git(world.f, "tag", "feature")
    assert decide(world.f, "git commit -m x").decision is None
