"""PR #171 round 1, R1: a branch switch that may not happen must not replace the start branch.

``git switch feature; git commit`` commits on ``main`` when the switch fails. A switch is
certain for a later commit or push only when every separator between them is ``&&`` (and the
switch is not on the right of ``||`` or backgrounded). Otherwise the starting branch stays
possible beside the new one: protected new branch denies, protected start asks, else allow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.governance._guard_harness import HAVE_TOOLS, World, build_world, decide, git

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")


@pytest.fixture
def world(tmp_path: Path) -> World:
    w = build_world(tmp_path)
    git(w.f, "branch", "other-feature")
    # A branch with its own commit: `git checkout feature` is a branch, not a path, and a
    # push of it is not the "already merged" case the guard asks about.
    tip = git(w.p, "commit-tree", "HEAD^{tree}", "-p", "HEAD", "-m", "feature work")
    git(w.p, "branch", "feature", tip)
    return w


UNCERTAIN = [
    "git switch feature; git commit -m x",
    "git switch nonexistent; git commit -m x",
    "if false; then git switch feature; fi; git commit -m x",
    "git switch feature || true; git commit -m x",
    "git switch feature && echo ok; git commit -m x",
    "git checkout feature; git commit -m x",
    "git switch feature; git push origin feature",
    "git checkout feature; git push origin feature",
]


@pytest.mark.parametrize("command", UNCERTAIN)
def test_an_uncertain_switch_from_main_asks(world: World, command: str) -> None:
    got = decide(world.p, command)
    assert got.decision == "ask", got
    assert "may not have taken effect" in got.reason


@pytest.mark.parametrize(
    "command",
    [
        "git switch feature && git commit -m x",
        "git switch feature && echo ok && git commit -m x",
        "git checkout feature && git commit -m x",
        "git switch feature && git push origin feature",
    ],
)
def test_a_chain_of_and_is_certain_so_the_new_branch_decides(world: World, command: str) -> None:
    assert decide(world.p, command).decision is None


@pytest.mark.parametrize(
    "command",
    [
        "git switch main; git commit -m x",
        "git switch main && git commit -m x",
        "git checkout main; git commit -m x",
        "git switch main; git push origin feature",
    ],
)
def test_switching_to_a_protected_branch_denies_either_way(world: World, command: str) -> None:
    assert decide(world.f, command).decision == "deny"


@pytest.mark.parametrize(
    "command",
    [
        "git switch other-feature; git commit -m x",
        "git switch other-feature || true; git commit -m x",
        "git checkout other-feature; git commit -m x",
    ],
)
def test_uncertain_switch_between_unprotected_branches_is_allowed(
    world: World, command: str
) -> None:
    assert decide(world.f, command).decision is None
