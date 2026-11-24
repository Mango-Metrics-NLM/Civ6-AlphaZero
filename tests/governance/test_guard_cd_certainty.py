"""PR #171 round 1, the R1 class for directory changes: a ``cd`` that may not happen.

``cd f; git commit`` commits in the starting checkout when the ``cd`` fails (a typo, a missing
directory), and ``if false; then cd f; fi; git commit`` never runs the ``cd`` at all. A change of
directory (``cd``, ``Set-Location``, ``pushd``, ``export GIT_DIR``, ``$env:GIT_DIR``) is certain for
a later git command only while every separator between them is ``&&``; ``cd f || exit 1`` is
certain too, because the shell stops when the ``cd`` fails. Otherwise the earlier directory stays
possible: a protected branch there asks. More than four possible directories ask as well.

``p`` is on ``main`` (protected), ``f`` on ``feature``, ``b`` a clone on ``feature2``.
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

MAYBE = "may not have taken effect"


@pytest.fixture
def world(tmp_path: Path) -> World:
    w = build_world(tmp_path)
    for name in ("d1", "d2", "d3", "d4", "d5"):
        (w.f / name).mkdir()
    return w


@pytest.fixture
def b(world: World) -> Path:
    repo = world.root / "b"
    git(world.root, "clone", "-q", str(world.origin), str(repo))
    git(repo, "checkout", "-q", "-b", "feature2")
    git(repo, "commit", "-q", "--allow-empty", "-m", "b work")
    return repo


@pytest.mark.parametrize(
    "template",
    [
        "if false; then cd {f}; fi; git commit -m x",
        "cd /nonexistent-typo; git commit -m x",
        "cd {f}; git commit -m x",
        "cd {f}\ngit commit -m x",
        "cd {f} || true; git commit -m x",
        "cd {f} && echo ok; git commit -m x",
        "export GIT_DIR={f}/.git; git commit -m x",
        "cd {f}; git push origin feature",
        "cd {f}; echo `pwd` && git commit -m x",
        "cd {f} && git commit -m x; git push origin feature",
    ],
)
def test_an_uncertain_cd_away_from_main_asks(world: World, template: str) -> None:
    got = decide(world.p, template.format(f=world.f))
    assert got.decision == "ask", got
    assert MAYBE in got.reason


def test_powershell_set_location_without_and_asks(world: World) -> None:
    got = decide(world.p, f"Set-Location {world.f}; git commit -m x", tool_name="PowerShell")
    assert got.decision == "ask", got
    assert MAYBE in got.reason


def test_an_earlier_directory_stays_possible_after_a_later_uncertain_cd(
    world: World, b: Path
) -> None:
    # If `cd b` fails, the commit runs in p (the first cd ran).
    got = decide(world.f, f"cd {world.p} && cd {b}; git commit -m x")
    assert got.decision == "ask", got
    assert "main" in got.reason


def test_a_switch_in_a_directory_that_may_not_be_reached_counts_for_both(
    world: World, b: Path
) -> None:
    # If `cd f` fails, `git switch main` runs in b, and the commit then runs in b on main.
    got = decide(b, f"cd {world.f}; git switch main && cd {b} && git commit -m x")
    assert not_allowed(got), got


def test_a_relative_cd_after_an_uncertain_one_is_not_guessed(world: World) -> None:
    got = decide(world.f, f"cd {world.root}; cd p && git commit -m x")
    assert not_allowed(got), got


def test_more_than_four_possible_directories_ask(world: World) -> None:
    f = world.f
    got = decide(f, f"cd {f}/d1; cd {f}/d2; cd {f}/d3; cd {f}/d4; cd {f}/d5; git commit -m x")
    assert got.decision == "ask", got


def test_a_few_possible_feature_directories_are_allowed(world: World) -> None:
    f = world.f
    assert decide(f, f"cd {f}/d1; cd {f}/d2; git commit -m x").decision is None


@pytest.mark.parametrize(
    "template",
    [
        "cd {f} && git commit -m x",
        "cd {f} || exit 1; git commit -m x",
        "cd {f} || exit 1\ngit commit -m x",
        "cd {f} || exit; git push origin feature",
        "cd {f} && cd {f}/d1 && cd {f}/d2 && cd {f}/d3 && cd {f}/d4 && git commit -m x",
        "cd {f} && echo `pwd` && git commit -m x",
        "cd {f} && git add . && git commit -m x && git push origin feature",
    ],
)
def test_a_certain_cd_away_from_main_is_allowed(world: World, template: str) -> None:
    assert decide(world.p, template.format(f=world.f)).decision is None


@pytest.mark.parametrize("template", ["cd {p} && git commit -m x", "cd {p}; git commit -m x"])
def test_a_cd_to_main_denies_either_way(world: World, template: str) -> None:
    assert decide(world.f, template.format(p=world.p)).decision == "deny"


def test_an_uncertain_cd_between_feature_checkouts_is_allowed(world: World, b: Path) -> None:
    assert decide(world.f, f"cd {b}; git commit -m x").decision is None


def test_powershell_call_operator_after_and_keeps_the_chain(world: World) -> None:
    got = decide(world.p, f"Set-Location {world.f} && & git commit -m x", tool_name="PowerShell")
    assert got.decision is None, got


@pytest.mark.parametrize("joiner", ["&&", ";"])
def test_a_relative_cd_applies_to_every_directory_that_may_be_in_force(
    world: World, joiner: str
) -> None:
    # alt/p is a feature checkout, root/p is main. If `cd alt` fails, `cd p` lands in root/p,
    # and the commit then runs on main, even though `cd p` itself is certain.
    alt = world.root / "alt"
    alt.mkdir()
    git(alt, "clone", "-q", str(world.origin), "p")
    git(alt / "p", "checkout", "-q", "-b", "feature3")
    got = decide(world.root, f"cd {alt}; cd p {joiner} git commit -m x")
    assert got.decision == "ask", got
    assert MAYBE in got.reason
    # A relative GIT_DIR is read in whichever directory is in force when git runs.
    got = decide(world.root, f"cd {alt}; export GIT_DIR=p/.git {joiner} git commit -m x")
    assert got.decision == "ask", got
    assert MAYBE in got.reason
