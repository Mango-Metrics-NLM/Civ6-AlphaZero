"""PR #171 round 1, R2: branch changes are recorded per repository, not once for all.

``git switch main && git -C B switch feature2 && git commit`` committed on main in A, but B's
switch overwrote A's record. A switch whose target repository cannot be resolved makes every
later commit or push ask.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.governance._guard_harness import HAVE_TOOLS, World, build_world, decide, git

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path)


@pytest.fixture
def repo_b(world: World) -> Path:
    b = world.root / "b"
    git(world.root, "clone", "-q", str(world.origin), str(b))
    git(b, "checkout", "-q", "-b", "feature2")
    git(b, "commit", "-q", "--allow-empty", "-m", "b work")
    return b


def test_a_switch_in_another_repo_does_not_hide_a_switch_to_main_here(
    world: World, repo_b: Path
) -> None:
    got = decide(world.f, f"git switch main && git -C {repo_b} switch feature2 && git commit -m x")
    assert got.decision == "deny", got


def test_a_switch_in_another_repo_leaves_this_one_unchanged(world: World, repo_b: Path) -> None:
    got = decide(world.f, f"git -C {repo_b} switch main && git commit -m x")
    assert got.decision is None, got


def test_committing_in_the_repo_that_switched_to_main_denies(world: World, repo_b: Path) -> None:
    got = decide(world.f, f"git -C {repo_b} switch main && git -C {repo_b} commit -m x")
    assert got.decision == "deny", got


def test_each_repo_keeps_its_own_record(world: World, repo_b: Path) -> None:
    got = decide(
        world.f,
        f"git switch main && git -C {repo_b} switch feature2 && git -C {repo_b} commit -m x",
    )
    assert got.decision is None, got


def test_a_switch_in_an_unresolvable_directory_makes_later_commits_ask(world: World) -> None:
    got = decide(world.f, f'cd "$X" && git switch feature && cd {world.p} && git commit -m x')
    assert got.decision == "ask", got
