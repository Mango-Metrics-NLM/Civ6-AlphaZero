"""Review round S5-S10: further routes to a tag push, a deletion or a protected branch.

git facts (checked against a bare origin with ``--dry-run``): pushing ``:v9`` or using
``--delete v9`` DELETES the remote tag v9 (an unqualified name matches the one remote ref),
with or without a local tag; ``tags/v9`` as a source resolves to ``refs/tags/v9``;
``v9^{tag}:v10`` is REFUSED by git (hint: use ``refs/tags/v10``), so a deny there is harmless;
``--pru``/``--prun`` are ``--prune``, ``--tag`` is ``--tags``, ``--fol`` is ``--follow-tags``,
``--br`` is ``--branches``, ``--mir``/``--al`` are ``--mirror``/``--all`` (git rejects those
three with a refspec; the guard denies them anyway). ``--pr`` is ambiguous (``--progress``),
git prints usage and pushes nothing, so it is deliberately not tested.
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
    git(w.f, "tag", "-a", "-m", "annotated", "v9")
    git(w.f, "branch", "feature-x")
    return w


# --- S5 / S6: tags by other names, deletions --------------------------------------------------


def test_tags_slash_name_source_is_denied(world: World) -> None:
    got = decide(world.f, "git push origin tags/v9")
    assert got.decision == "deny", got
    got = decide(world.f, "git push origin tags/v9:tags/v9")
    assert got.decision == "deny", got


@pytest.mark.parametrize(
    "command",
    ["git push origin :v9", "git push origin --delete v9", "git push -d origin v9"],
)
def test_deleting_an_unqualified_name_that_is_no_local_branch_is_not_allowed(
    world: World, command: str
) -> None:
    git(world.f, "tag", "-d", "v9")  # no local tag is needed for the remote one to go
    got = decide(world.f, command)
    assert not_allowed(got), got


@pytest.mark.parametrize(
    "command",
    [
        "git push origin --delete feature-x",
        "git push origin :feature-x",
        "git push origin --delete refs/heads/feature-x",
        "git push origin :refs/heads/feature-x",
    ],
)
def test_deleting_a_feature_branch_with_a_tracking_ref_stays_allowed(
    world: World, command: str
) -> None:
    # PR #171: the unqualified forms need origin/feature-x here; a local branch alone asks
    # (tests/governance/test_guard_copilot_threads.py), since origin may hold a TAG feature-x.
    git(world.f, "push", "-q", "origin", "feature-x")
    assert decide(world.f, command).decision is None


def test_a_tag_object_source_to_an_unqualified_destination_is_denied(world: World) -> None:
    got = decide(world.f, "git push origin 'v9^{tag}:v10'")
    assert got.decision == "deny", got
    got = decide(world.f, "git push origin v9^{tag}:v10")
    assert got.decision == "deny", got


# --- S7: -c / --config-env values -----------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git -c push.followTags=true push origin feature",
        "git -c remote.origin.push='refs/tags/*:refs/tags/*' push origin",
        "git -c remote.origin.push=refs/heads/*:refs/heads/* push origin",
        "git -c remote.origin.mirror=true push origin",
        "git -c push.default=matching push",
        "git -c push.default=matching push origin",
        "git --config-env=push.followTags=SOME_VAR push origin feature",
        "git --config-env push.followTags=SOME_VAR push origin feature",
        "git --config-env=remote.origin.push=SOME_VAR push origin",
        "git --config-env=remote.origin.mirror=SOME_VAR push origin",
    ],
)
def test_config_that_pushes_tags_or_everything_is_denied(world: World, command: str) -> None:
    got = decide(world.f, command)
    assert got.decision == "deny", got


@pytest.mark.parametrize(
    "command",
    [
        "git -c user.name=x push origin feature",
        "git -c user.name=x -c core.editor=true push origin feature",
        "git --config-env=user.name=SOME_VAR push origin feature",
    ],
)
def test_unrelated_config_is_allowed(world: World, command: str) -> None:
    assert decide(world.f, command).decision is None


# --- S8: abbreviated long options ------------------------------------------------------------


@pytest.mark.parametrize("opt", ["--pru", "--prun", "--mir", "--al", "--tag", "--fol", "--br"])
def test_abbreviated_long_options_deny_like_their_full_forms(world: World, opt: str) -> None:
    got = decide(world.f, f"git push {opt} origin feature")
    assert got.decision == "deny", got
    got = decide(world.f, f"git push origin {opt} feature")
    assert got.decision == "deny", got


# --- S9: quoted global options -----------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        'git "--no-pager" push origin HEAD:main',
        "git '--no-pager' push origin HEAD:main",
        "git '-c' x=y push origin HEAD:main",
        'git "-c" x=y "push" origin HEAD:main',
        'git "-C" . push origin HEAD:main',
    ],
)
def test_quoted_global_options_do_not_hide_the_subcommand(world: World, command: str) -> None:
    got = decide(world.f, command)
    assert got.decision == "deny", got


def test_quoted_global_option_with_a_feature_push_is_allowed(world: World) -> None:
    assert decide(world.f, 'git "--no-pager" push origin feature').decision is None


# --- S10: shell keywords --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "if true; then git push origin HEAD:main; fi",
        "! git push origin HEAD:main",
        "while false; do :; done; git push origin HEAD:main",
        "if true; then git push origin HEAD:main; else :; fi",
        "for x in a; do git push origin HEAD:main; done",
        "then git push origin HEAD:main",
        "do git push origin HEAD:main",
        "time git push origin HEAD:main",
    ],
)
def test_shell_keywords_do_not_hide_a_push_to_a_protected_branch(
    world: World, command: str
) -> None:
    got = decide(world.f, command)
    assert got.decision == "deny", got


@pytest.mark.parametrize(
    "command",
    ["for x in a; do git commit -m x; done", "if true; then git commit -m x; fi", "! git commit"],
)
def test_shell_keywords_do_not_hide_a_commit_on_a_protected_branch(
    world: World, command: str
) -> None:
    got = decide(world.p, command)
    assert got.decision == "deny", got


@pytest.mark.parametrize(
    "command",
    [
        "for x in a; do git status; done",
        "if true; then git commit -m x; fi",
        "! git push origin feature",
    ],
)
def test_shell_keywords_with_harmless_commands_are_allowed(world: World, command: str) -> None:
    assert decide(world.f, command).decision is None
