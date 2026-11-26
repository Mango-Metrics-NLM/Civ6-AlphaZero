"""A LIGHTWEIGHT tag reached by a name that is not its bare name.

``tags/lw`` resolves to ``refs/tags/lw``. For an ANNOTATED tag the guard also sees an object of
type ``tag`` (``cat-file -t``), but a lightweight tag is a commit, so only the check that asks
git for the full ref name (``rev-parse --symbolic-full-name``) can catch it. Found by a
negative control: removing that check left the whole suite green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.governance._guard_harness import HAVE_TOOLS, World, build_world, decide, git

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")


@pytest.fixture
def world(tmp_path: Path) -> World:
    w = build_world(tmp_path)
    git(w.f, "tag", "lw")  # lightweight
    return w


@pytest.mark.parametrize(
    "command",
    [
        "git push origin tags/lw",
        "git push origin tags/lw:tags/lw",
        "git push origin lw",
    ],
)
def test_lightweight_tag_by_any_name_git_resolves_is_denied(world: World, command: str) -> None:
    got = decide(world.f, command)
    assert got.decision == "deny", got
    assert "maintainer-only" in got.reason
