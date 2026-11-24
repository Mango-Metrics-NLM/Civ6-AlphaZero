"""G1: a bare tag name is pushed without a deny.

``git push origin v2.0.0`` pushes the tag when ``refs/tags/v2.0.0`` exists locally (git resolves
an unqualified source against refs). The rule: the first positional argument after the options
is the remote, never a refspec; every later refspec's SOURCE (before ``:``, or the whole token)
is denied when it is ``refs/tags/<x>`` or ``refs/tags/<src>`` exists in the repository the push
runs in.

What git does with an unqualified DESTINATION (measured, git 2.x, see the last test): with a
branch-like source such as ``HEAD`` git creates a NEW BRANCH of that name even when a local tag
has it, and when the name already exists on the remote as a tag it refuses ("matches more than
one"). It never updates a tag, so the destination case needs no deny and is pinned as allowed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.governance._guard_harness import (
    ALLOWED,
    HAVE_TOOLS,
    World,
    build_world,
    decide,
    git,
)

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")

TAG_DENY = "pushing tags is maintainer-only"


@pytest.fixture
def world(tmp_path: Path) -> World:
    w = build_world(tmp_path)
    git(w.f, "tag", "v1")  # lightweight
    git(w.f, "tag", "-a", "-m", "annotated", "v1a")
    git(w.f, "tag", "origin")  # a tag named like the remote
    git(w.f, "branch", "release-1.2")  # looks like a version, no tag of that name
    git(w.f, "branch", "v2-work")
    git(w.f, "branch", "both")
    git(w.f, "tag", "both")
    return w


@pytest.mark.parametrize(
    "command",
    [
        "git push origin v1",  # lightweight tag by bare name
        "git push origin v1a",  # annotated tag by bare name
        "git push origin refs/tags/v1",
        "git push origin v1:v1",
        "git push origin v1:refs/heads/elsewhere",  # a tag source, whatever the destination
        "git push origin feature v1",  # a tag among several refspecs
        "git push -u origin v1",  # options before the remote
        "git push --push-option ci.skip origin v1",  # an option that takes a value
        "git push origin 'v1'",
        'git push origin "v1"',
        "git push origin both",  # a name that is both a branch and a tag
        "bash -c 'git push origin v1'",
    ],
)
def test_pushing_a_tag_source_is_denied(world: World, command: str) -> None:
    got = decide(world.f, command)
    assert got.decision == "deny", got
    assert TAG_DENY in got.reason


def test_bare_tag_is_denied_in_the_directory_the_push_runs_in(world: World) -> None:
    # The hook runs in the protected checkout, which has no tag v1: the push's own
    # directory (-C) decides, so the tag must be looked up in `f`.
    got = decide(world.p, f"git -C {world.f} push origin v1")
    assert got.decision == "deny", got
    assert TAG_DENY in got.reason


def test_bare_tag_after_cd_is_denied_in_the_cd_directory(world: World) -> None:
    got = decide(world.p, f"cd {world.f} && git push origin v1")
    assert got.decision == "deny", got
    assert TAG_DENY in got.reason


def test_a_tag_that_exists_only_elsewhere_does_not_deny_a_branch_push(world: World) -> None:
    # `v1` exists in f, not in p: from p, pushing a branch named v1 is not a tag push.
    git(world.p, "checkout", "-q", "-b", "v1")
    git(world.p, "commit", "-q", "--allow-empty", "-m", "x")
    assert decide(world.p, f"git -C {world.p} push origin v1:refs/heads/v1") == ALLOWED


@pytest.mark.parametrize(
    "command",
    [
        "git push origin release-1.2",
        "git push origin v2-work",
        "git push origin release-1.2:release-1.2",
        "git push origin feature",
        "git push origin feature:feature",
    ],
)
def test_version_looking_branch_without_a_tag_is_allowed(world: World, command: str) -> None:
    assert decide(world.f, command) == ALLOWED


def test_the_remote_token_is_never_a_refspec(world: World) -> None:
    # A tag named `origin` exists (fixture). `origin` is the remote here, not a source.
    assert decide(world.f, "git push origin feature") == ALLOWED
    assert decide(world.f, "git push -u origin feature") == ALLOWED


def test_the_first_positional_is_the_remote_even_when_it_is_a_tag_name(world: World) -> None:
    # Without the remote token, `v1` would be the remote, not a refspec: no refspec, no deny.
    assert decide(world.f, "git push v1") == ALLOWED


def test_an_existing_deny_is_not_turned_into_something_else(world: World) -> None:
    got = decide(world.f, "git push origin v1 HEAD:qa")
    assert got.decision == "deny", got
    got = decide(world.f, "git push --force origin v1")
    assert got.decision == "deny", got


def test_unqualified_destination_naming_a_tag_is_pinned_as_allowed(world: World) -> None:
    """Pins what git does: ``HEAD:newtag`` with a local tag ``newtag`` creates a BRANCH."""
    git(world.f, "tag", "newtag")
    # git itself, dry run: it would create a new branch on the remote, not touch the tag.
    out = git(world.f, "push", "--dry-run", "--porcelain", "origin", "HEAD:newtag")
    assert "refs/heads/newtag" in out
    assert decide(world.f, "git push origin HEAD:newtag") == ALLOWED
