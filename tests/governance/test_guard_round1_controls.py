"""PR #171 round 1: controls for the switch-certainty and parenthesis rules.

The contract tests (``test_guard_switch_certainty.py``, ``test_guard_switch_per_repo.py``,
``test_guard_subshell_cd.py``) pin the reported shapes. These pin the parts of the fix that a
planted defect showed no contract test could see: a switch on the right of ``||`` or inside
``$( ... )`` is never certain, an earlier switch stays possible after a later uncertain one, a
subshell's own parentheses keep a ``&&`` chain certain, and a parenthesis inside quotes is text.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.governance._guard_harness import (
    HAVE_TOOLS,
    World,
    build_world,
    decide,
    git,
    not_allowed,
)

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")


@pytest.fixture
def world(tmp_path: Path) -> World:
    w = build_world(tmp_path)
    git(w.p, "branch", "feature2")
    git(w.f, "branch", "other-feature")
    return w


@pytest.mark.parametrize(
    "command",
    [
        "false || git switch feature2 && git commit -m x",
        "false || (git switch feature2) && git commit -m x",
        "echo $(git switch feature2) && git commit -m x",
    ],
)
def test_a_switch_that_may_be_skipped_is_never_certain(world: World, command: str) -> None:
    got = decide(world.p, command)
    assert got.decision == "ask", got
    assert "may not have taken effect" in got.reason


def test_an_earlier_switch_stays_possible_after_a_later_uncertain_one(world: World) -> None:
    # If `git switch other-feature` fails, the repository is on main (the first switch ran).
    got = decide(world.f, "git switch main && git switch other-feature; git commit -m x")
    assert got.decision == "ask", got
    assert "main" in got.reason


@pytest.mark.parametrize(
    "command",
    [
        "(git switch feature2 && true) && git commit -m x",
        "git switch feature2 && (git commit -m x)",
    ],
)
def test_subshell_parentheses_keep_an_and_chain_certain(world: World, command: str) -> None:
    assert decide(world.p, command).decision is None


@pytest.mark.parametrize(
    "command",
    [
        'git commit -m "wip :)" && git push origin feature',
        'git commit -m "fix(guard): x" && git push origin feature',
    ],
)
def test_a_parenthesis_inside_quotes_is_text(world: World, command: str) -> None:
    assert decide(world.f, command).decision is None


def test_a_quoted_close_inside_a_subshell_does_not_restore_early(world: World) -> None:
    # The quoted `)` must not end the subshell: the commit inside it runs in p (main).
    got = decide(world.f, f'(cd {world.p}; echo ")"; git commit -m x)')
    assert not_allowed(got), got


def test_an_unmatched_close_makes_a_feature_directory_unknown_too(world: World) -> None:
    # From main. The guard does not model `case`: an unmatched `)` means its picture of the
    # parentheses is off, so it asks rather than keep f (which would allow).
    got = decide(world.p, f"cd {world.f} ; case x in x) true ;; esac ; git commit -m x")
    assert got.decision == "ask", got
    assert "could not tell which branch" in got.reason
