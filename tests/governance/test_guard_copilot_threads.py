"""PR #171, Copilot's second review of d10cfc3: one test per thread.

- subshell: a ``cd`` inside ``( ... )`` ended at the ``)`` (already fixed, confirmed here);
- ``--track origin/qa`` switches to the protected ``qa`` (already fixed, confirmed here);
- an unqualified remote deletion (``:name``, ``--delete name``) was exempt because a LOCAL
  branch of that name existed, though git matches the name against every remote ref and deletes
  a remote TAG of that name; it now needs a remote-tracking branch and no tag of that name;
- a quoted operand with a space (``git -C "/x/main checkout" commit``) was split on the space;
- ``env -S "git push ..."`` / ``--split-string=`` hid the command it runs.

``p`` is on ``main`` (protected), ``f`` on ``feature``.
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
)

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path)


# --- subshell (pretooluse_guard.sh:287, _guard_git.sh:82): already fixed -------------------------


def test_subshell_cd_does_not_survive_the_close(world: World) -> None:
    assert decide(world.p, f"(cd {world.f}) && git commit -m x").decision == "deny"
    assert decide(world.f, f"(cd {world.p}) && git commit -m x").decision is None


# --- --track origin/qa (_guard_git.sh:368): already fixed ----------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git checkout --track origin/qa && git commit -m x",
        "git switch --track origin/qa && git push origin HEAD",
    ],
)
def test_tracking_origin_qa_switches_to_protected_qa(world: World, command: str) -> None:
    git(world.p, "push", "-q", "origin", "main:qa")
    g = world.root / "g"
    git(world.root, "clone", "-q", str(world.origin), str(g))
    git(g, "checkout", "-q", "-b", "feature")
    assert decide(g, command).decision == "deny"


# --- unqualified remote deletion (_guard_git.sh:302) ---------------------------------------------


@pytest.fixture
def tag_only_remote(world: World) -> World:
    """Origin has a TAG `feature-x` and no branch; `f` has a local branch `feature-x`."""
    git(world.f, "tag", "feature-x")
    git(world.f, "push", "-q", "origin", "refs/tags/feature-x")
    git(world.f, "tag", "-d", "feature-x")
    git(world.f, "branch", "feature-x")
    return world


@pytest.mark.parametrize(
    "command", ["git push origin :feature-x", "git push origin --delete feature-x"]
)
def test_a_local_branch_does_not_exempt_an_unqualified_deletion(
    tag_only_remote: World, command: str
) -> None:
    got = decide(tag_only_remote.f, command)
    assert got.decision == "ask", got


def test_deleting_a_branch_with_a_tracking_ref_is_allowed(world: World) -> None:
    git(world.f, "push", "-q", "origin", "feature:feature-y")
    git(world.f, "fetch", "-q", "origin")
    assert decide(world.f, "git push origin --delete feature-y").decision is None
    assert decide(world.f, "git push origin :feature-y").decision is None


def test_a_tag_of_the_same_name_makes_the_deletion_ask(world: World) -> None:
    git(world.f, "push", "-q", "origin", "feature:feature-y")
    git(world.f, "fetch", "-q", "origin")
    git(world.f, "tag", "feature-y")
    assert decide(world.f, "git push origin --delete feature-y").decision == "ask"


def test_a_qualified_branch_deletion_is_still_allowed(tag_only_remote: World) -> None:
    got = decide(tag_only_remote.f, "git push origin --delete refs/heads/feature-x")
    assert got.decision is None, got


# --- quoted operands with spaces (_guard_git.sh:436) ---------------------------------------------


@pytest.fixture
def spaced(world: World) -> tuple[Path, Path]:
    """`root/prot checkout` on main, `root/feat checkout` on feature."""
    prot = world.root / "prot checkout"
    feat = world.root / "feat checkout"
    git(world.root, "clone", "-q", str(world.origin), str(prot))
    git(world.root, "clone", "-q", str(world.origin), str(feat))
    git(feat, "checkout", "-q", "-b", "feature")
    return prot, feat


@pytest.mark.parametrize(
    "template",
    [
        'git -C "{d}" commit -m x',
        "git -C '{d}' commit -m x",
        'git --git-dir="{d}/.git" commit -m x',
        'git --git-dir "{d}/.git" commit -m x',
        'env -C "{d}" git commit -m x',
        'cd "{d}" && git commit -m x',
        'GIT_DIR="{d}/.git" git commit -m x',
    ],
)
def test_a_quoted_directory_with_a_space_is_one_operand(
    world: World, spaced: tuple[Path, Path], template: str
) -> None:
    prot, feat = spaced
    assert decide(world.f, template.format(d=prot)).decision == "deny"
    assert decide(world.p, template.format(d=feat)).decision is None


def test_a_quoted_config_value_with_a_space_does_not_hide_the_subcommand(world: World) -> None:
    assert decide(world.f, 'git -c "user.name=a b" push origin HEAD:main').decision == "deny"
    assert decide(world.p, 'git -c "user.name=a commit" commit -m x').decision == "deny"


# --- env -S (pretooluse_guard.sh:360) ------------------------------------------------------------


@pytest.mark.parametrize(
    ("where", "command"),
    [
        ("f", 'env -S "git push origin HEAD:main"'),
        ("f", 'env --split-string="git push origin HEAD:main"'),
        ("f", "env -S 'git push origin HEAD:main'"),
        ("p", 'env -S "git commit -m x"'),
        ("p", 'env --split-string="git commit -m x"'),
    ],
)
def test_env_split_string_is_the_command(world: World, where: str, command: str) -> None:
    assert decide(getattr(world, where), command).decision == "deny"
