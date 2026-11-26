"""G3: a glob or "matching" refspec pushes protected branches.

``git push origin 'refs/heads/*:refs/heads/*'``, ``'*:*'`` and ``:`` (git's matching refspec)
push every branch that matches, protected ones included, like ``--all``. Rule: deny any push
refspec containing ``*`` and the bare ``:`` refspec with the ``--all`` message family, and deny
``--prune`` (it deletes remote branches) with any refspec.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.governance._guard_harness import ALLOWED, HAVE_TOOLS, World, build_world, decide

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")

FAMILY = "would push protected branches"


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path)


def _forms(spec: str) -> list[str]:
    """The refspec as written unquoted, single-quoted and double-quoted."""
    return [
        f"git push origin {spec}",
        f"git push origin '{spec}'",
        f'git push origin "{spec}"',
    ]


GLOBS = ["refs/heads/*:refs/heads/*", "*:*", ":", "refs/heads/*", "feature*:feature*"]


@pytest.mark.parametrize("spec", GLOBS)
@pytest.mark.parametrize("form", [0, 1, 2], ids=["bare", "single", "double"])
def test_glob_and_matching_refspecs_are_denied(world: World, spec: str, form: int) -> None:
    got = decide(world.f, _forms(spec)[form])
    assert got.decision == "deny", got
    assert FAMILY in got.reason


@pytest.mark.parametrize("spec", ["refs/heads/*:refs/heads/*", "*:*", ":"])
def test_wrapped_in_bash_c(world: World, spec: str) -> None:
    got = decide(world.f, f"bash -c \"git push origin '{spec}'\"")
    assert got.decision == "deny", got
    assert FAMILY in got.reason
    got = decide(world.f, f"bash -c 'git push origin \"{spec}\"'")
    assert got.decision == "deny", got


def test_glob_is_denied_with_other_options_and_refspecs(world: World) -> None:
    assert decide(world.f, "git push -u origin feature '*:*'").decision == "deny"
    assert decide(world.f, "git push --force-with-lease origin ':'").decision == "deny"


@pytest.mark.parametrize(
    "command",
    [
        "git push --prune origin feature",
        "git push --prune origin 'refs/heads/*:refs/heads/*'",
        "git push origin --prune feature:feature",
        "bash -c 'git push --prune origin feature'",
    ],
)
def test_prune_with_a_refspec_is_denied(world: World, command: str) -> None:
    got = decide(world.f, command)
    assert got.decision == "deny", got
    assert "prune" in got.reason.lower()


@pytest.mark.parametrize(
    "command",
    [
        "git push origin feature:feature",
        "git push origin feature",
        "git push origin HEAD:feature",
        "git push -u origin feature",
        "git push origin 'feature:feature'",
        "git push origin feature:refs/heads/feature",
    ],
)
def test_ordinary_refspecs_stay_allowed(world: World, command: str) -> None:
    assert decide(world.f, command) == ALLOWED
