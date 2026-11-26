"""PR #171 round 1, re-review: issues the previous fixes introduced.

- a spent budget (200 segments, 40 git queries) asked and stopped, so a later DENY never ran;
- a backslash at the end of a comment was joined with the next line, which hid it;
- a quoted ``export "GIT_DIR=..."`` escaped both the follower and the catch-all;
- a trailing ``--`` made ``git checkout main --`` a path checkout;
- ``set -P`` / ``set -o physical`` make every later ``cd`` physical;
- with CDPATH set, ``cd name`` may land anywhere on it.

``p`` is on ``main`` (protected), ``f`` on ``feature``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.governance._guard_harness import (
    HAVE_TOOLS,
    World,
    base_env,
    build_world,
    decide,
    git,
)

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path)


# --- 1: a spent budget never hides a deny --------------------------------------------------------


def _many_queries(world: World) -> str:
    return "; ".join(f"git -C {world.f}{'/.' * i} commit -m x" for i in range(1, 25))


@pytest.mark.parametrize(
    "tail", ["git push origin HEAD:main", "git push --force origin feature", "git tag v9"]
)
def test_a_deny_after_the_git_query_budget_still_denies(world: World, tail: str) -> None:
    got = decide(world.f, f"{_many_queries(world)}; {tail}")
    assert got.decision == "deny", got


@pytest.mark.parametrize(
    "command",
    [
        "true; " * 210 + "git push origin HEAD:main",
        "git push origin HEAD:main; " + "true; " * 210,
        "true; " * 210 + "gh pr merge 1",
    ],
)
def test_a_deny_in_a_long_command_still_denies(world: World, command: str) -> None:
    start = time.monotonic()
    got = decide(world.f, command)
    assert time.monotonic() - start < 10
    assert got.decision == "deny", got


def test_a_long_harmless_command_still_asks(world: World) -> None:
    got = decide(world.f, "true; " * 210 + "git push origin feature")
    assert got.decision == "ask", got
    assert "too long" in got.reason


# --- 2: a backslash in a comment is not a continuation -------------------------------------------


@pytest.mark.parametrize(
    ("where", "command"),
    [
        ("p", "true # note \\\ngit commit -m x"),
        ("p", "echo hi # \\\ngit commit -m x"),
        ("f", "true # \\\ngit switch main && git commit -m x"),
    ],
)
def test_a_backslash_ending_a_comment_does_not_join(world: World, where: str, command: str) -> None:
    assert decide(getattr(world, where), command).decision == "deny"


def test_a_hash_inside_a_word_or_quotes_is_not_a_comment(world: World) -> None:
    assert decide(world.p, "echo a#b \\\n&& git commit -m x").decision == "deny"
    assert decide(world.p, "echo '#' \\\n&& git commit -m x").decision == "deny"


# --- 3: a quoted export ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        'export "GIT_DIR=../p/.git"; git commit -m x',
        "export 'GIT_DIR'=../p/.git && git commit -m x",
        "export GIT_D''IR=../p/.git && git commit -m x",
    ],
)
def test_a_quoted_export_of_git_dir_is_followed(world: World, command: str) -> None:
    assert decide(world.f, command).decision == "deny"


def test_a_quoted_mention_in_another_form_asks(world: World) -> None:
    got = decide(world.f, "declare -x GIT_D''IR=../p/.git; git commit -m x")
    assert got.decision == "ask", got


# --- 4: a trailing `--` ---------------------------------------------------------------------------


def test_a_trailing_double_dash_is_still_a_branch_checkout(world: World) -> None:
    assert decide(world.f, "git checkout main -- && git commit -m x").decision == "deny"


def test_a_trailing_double_dash_after_track_still_tracks(world: World) -> None:
    g = world.root / "g"
    git(world.root, "clone", "-q", str(world.origin), str(g))
    git(g, "checkout", "-q", "-b", "feature")
    git(g, "branch", "-q", "-D", "main")
    assert decide(g, "git checkout --track origin/main -- && git commit -m x").decision == "deny"


def test_a_path_after_double_dash_is_still_a_path_checkout(world: World) -> None:
    assert decide(world.p, "git checkout feature -- afile && git commit -m x").decision == "deny"


# --- 5 and 6: a cd the guard cannot place ------------------------------------------------------


@pytest.mark.parametrize("prefix", ["set -P; ", "set -o physical && ", "set -eP; "])
def test_physical_cd_mode_makes_cd_unknown(world: World, prefix: str) -> None:
    got = decide(world.p, f"{prefix}cd {world.f} && git commit -m x")
    assert got.decision == "ask", got


def test_cdpath_makes_a_bare_name_unknown(world: World) -> None:
    env = {**base_env(), "CDPATH": str(world.root)}
    got = decide(world.root, "cd f && git commit -m x", env=env)
    assert got.decision == "ask", got
    # `./f` and absolute paths are not searched on CDPATH.
    assert decide(world.root, "cd ./f && git commit -m x", env=env).decision is None


def test_past_five_hundred_segments_the_rest_is_not_judged_but_asked(world: World) -> None:
    # Judging every segment of an arbitrarily long command would outrun the hook's timeout,
    # which fails open; past 500 segments the guard stops and asks (never allows).
    start = time.monotonic()
    got = decide(world.f, "true; " * 600 + "git push origin HEAD:main")
    assert time.monotonic() - start < 10
    assert got.decision == "ask", got
    assert "not judged" in got.reason
