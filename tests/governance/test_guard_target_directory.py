"""G2: the protected-branch deny is judged in the directory the git command runs in.

``resolve_branches`` used to read ``HEAD`` of the hook's own working directory, so
``git -C <protected checkout> commit`` was allowed from a feature checkout, and a commit
in a feature checkout was denied from a protected one. The rule: the target is set by every
``-C`` in order (cumulative; a relative ``-C`` is relative to the previous one),
``--git-dir`` / ``--work-tree`` (both spellings), and a preceding ``cd <dir>`` segment joined
by ``&&`` or ``;``. When the directory is unknown, or the branch cannot be read there, a commit
or push is ASKED about (never allowed silently, never denied on a guess).

Fixture: ``p`` is checked out on ``main`` (protected by the shipped settings), ``f`` on
``feature``; both sit under ``world/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.governance._guard_harness import (
    ALLOWED,
    HAVE_TOOLS,
    World,
    base_env,
    build_world,
    decide,
    git,
)

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")

PROTECTED_DENY = "protected branch"
UNKNOWN = "could not tell which branch"


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path)


# --- -C -----------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["commit -m x", "push origin HEAD:feature", "push origin main"])
def test_dash_c_to_a_protected_checkout_is_denied(world: World, verb: str) -> None:
    got = decide(world.f, f"git -C {world.p} {verb}")
    assert got.decision == "deny", got


def test_dash_c_to_a_feature_checkout_is_allowed_from_a_protected_one(world: World) -> None:
    # The hook's own directory is on `main`; the target decides.
    assert decide(world.p, f"git -C {world.f} commit -m x") == ALLOWED
    assert decide(world.p, f"git -C {world.f} push origin feature") == ALLOWED


def test_plain_commit_in_a_protected_checkout_is_still_denied(world: World) -> None:
    got = decide(world.p, "git commit -m x")
    assert got.decision == "deny", got
    assert PROTECTED_DENY in got.reason


def test_plain_commit_in_a_feature_checkout_is_still_allowed(world: World) -> None:
    assert decide(world.f, "git commit -m x") == ALLOWED


def test_chained_dash_c_is_cumulative_and_relative(world: World) -> None:
    root = world.root
    assert decide(world.f, f"git -C {root} -C p commit -m x").decision == "deny"
    assert decide(world.p, f"git -C {root} -C f commit -m x") == ALLOWED
    # relative to the previous -C, not to the hook's directory
    assert decide(world.p, f"git -C {world.p} -C ../f commit -m x") == ALLOWED
    assert decide(world.f, f"git -C {world.f} -C ../p commit -m x").decision == "deny"


def test_dash_c_relative_to_the_hook_directory(world: World) -> None:
    assert decide(world.f, "git -C ../p commit -m x").decision == "deny"
    assert decide(world.p, "git -C ../f commit -m x") == ALLOWED


def test_dash_c_quoted_path_is_judged_too(world: World) -> None:
    assert decide(world.f, f'git -C "{world.p}" commit -m x').decision == "deny"


# --- --git-dir / --work-tree -----------------------------------------------------------


@pytest.mark.parametrize("sep", ["=", " "])
def test_git_dir_and_work_tree_pick_the_target(world: World, sep: str) -> None:
    def cmd(repo: Path) -> str:
        return f"git --git-dir{sep}{repo}/.git --work-tree{sep}{repo} commit -m x"

    assert decide(world.f, cmd(world.p)).decision == "deny"
    assert decide(world.p, cmd(world.f)) == ALLOWED


def test_git_dir_alone_picks_the_target(world: World) -> None:
    assert decide(world.f, f"git --git-dir={world.p}/.git commit -m x").decision == "deny"
    assert decide(world.p, f"git --git-dir={world.f}/.git commit -m x") == ALLOWED


# --- cd <dir> && git ----------------------------------------------------------------------


@pytest.mark.parametrize("joiner", ["&&", ";"])
def test_cd_into_a_protected_checkout_is_denied(world: World, joiner: str) -> None:
    got = decide(world.f, f"cd {world.p} {joiner} git commit -m x")
    assert got.decision == "deny", got


def test_cd_into_a_feature_checkout_is_allowed_from_a_protected_one(world: World) -> None:
    assert decide(world.p, f"cd {world.f} && git commit -m x") == ALLOWED
    # After `;` a failed cd leaves the commit on main (PR #171 round 1): it asks.
    assert decide(world.p, f"cd {world.f} ; git commit -m x").decision == "ask"


def test_cd_chain_is_relative_to_the_previous_cd(world: World) -> None:
    assert decide(world.f, f"cd {world.root} && cd p && git commit -m x").decision == "deny"
    assert decide(world.p, f"cd {world.root} && cd f && git commit -m x") == ALLOWED
    assert decide(world.f, "cd ../p && git commit -m x").decision == "deny"


def test_cd_then_dash_c_composes(world: World) -> None:
    assert decide(world.f, f"cd {world.root} && git -C p commit -m x").decision == "deny"
    assert decide(world.p, f"cd {world.root} && git -C f commit -m x") == ALLOWED


def test_cd_push_is_denied_too(world: World) -> None:
    assert decide(world.f, f"cd {world.p} && git push origin HEAD:feature").decision == "deny"


def test_a_later_absolute_cd_makes_an_unknown_directory_known(world: World) -> None:
    got = decide(world.f, f'cd "$X" && cd {world.p} && git commit -m x')
    assert got.decision == "deny", got


# --- unknown directory or unreadable branch: ask --------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        'cd "$X" && git commit -m x',
        "cd $X && git commit -m x",
        "cd - && git commit -m x",
        "cd && git commit -m x",
        "pushd /tmp && git commit -m x",
        "popd && git commit -m x",
        'cd "$X" ; git push origin feature',
    ],
)
def test_unknown_directory_asks(world: World, command: str) -> None:
    got = decide(world.f, command)  # the hook's own directory is a harmless feature checkout
    assert got.decision == "ask", got
    assert UNKNOWN in got.reason


@pytest.mark.parametrize("verb", ["commit -m x", "push origin feature"])
def test_dash_c_to_a_nonexistent_directory_asks(world: World, verb: str) -> None:
    got = decide(world.f, f"git -C /nonexistent/nowhere {verb}")
    assert got.decision == "ask", got
    assert UNKNOWN in got.reason


def test_dash_c_to_a_non_repository_asks(world: World, tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    got = decide(world.f, f"git -C {plain} commit -m x")
    assert got.decision == "ask", got
    assert UNKNOWN in got.reason


def test_unknown_directory_only_matters_for_commit_and_push(world: World) -> None:
    assert decide(world.f, 'cd "$X" && git status') == ALLOWED
    assert decide(world.f, 'cd "$X" && git log --oneline') == ALLOWED


def test_a_deny_still_wins_over_the_unknown_directory_ask(world: World) -> None:
    got = decide(world.f, 'cd "$X" && git push --force origin feature')
    assert got.decision == "deny", got


# --- detached HEAD, default branch -------------------------------------------------------------


def test_detached_head_in_the_target_is_not_a_protected_branch(world: World) -> None:
    git(world.f, "checkout", "-q", "--detach")
    # Hook directory is on `main`; the detached target decides, and it is neither deny nor ask.
    assert decide(world.p, f"git -C {world.f} commit -m x") == ALLOWED


def test_default_branch_is_resolved_in_the_target_repository(world: World) -> None:
    # No CIV6AZ_* knobs: the default comes from origin/HEAD of the repository the command runs
    # in. Make `p` default to `trunk` and sit on it; `f` (the hook's directory) keeps `main`.
    git(world.p, "checkout", "-q", "-b", "trunk")
    git(world.p, "push", "-q", "origin", "trunk")
    git(world.p, "fetch", "-q", "origin")
    git(world.p, "remote", "set-head", "origin", "trunk")
    got = decide(world.f, f"git -C {world.p} commit -m x", env=base_env(shipped=False))
    assert got.decision == "deny", got
    assert "trunk" in got.reason


def test_shipped_settings_protect_main_through_dash_c(world: World) -> None:
    # The same rule with the real settings.json env loaded (default `dev`, listed protected).
    got = decide(world.f, f"git -C {world.p} commit -m x", shipped=True)
    assert got.decision == "deny", got


def test_recreated_branch_ask_uses_the_dash_c_directory(world: World) -> None:
    # `f`'s feature work is unique; in `p`, a branch at main's tip is merged work.
    git(world.p, "checkout", "-q", "-b", "claude/merged-work")
    got = decide(world.f, f"git -C {world.p} push origin claude/merged-work")
    assert got.decision == "ask", got
    assert "merged" in got.reason.lower()
