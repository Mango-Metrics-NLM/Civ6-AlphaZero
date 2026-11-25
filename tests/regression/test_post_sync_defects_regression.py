"""Regression tests for post-sync defects caught during full Windows + dual-GPU verification.

Covers D-4 (``.claude/hooks/_guard_state.sh::cd_resolve``):
On Windows, ``git.exe -C <rel>/..`` normalises ``..`` lexically before traversing an
NTFS directory junction or symbolic link. Handing a raw relative chain for ``cd -P``
therefore resolved back into the starting checkout instead of following the physical
parent link — failing closed when escaping a protected branch (``test_cd_dash_p_is_physical``),
and **failing open** in the reverse direction when ``cd -P`` enters a protected branch
from a feature checkout.

Resolving ``cd -P`` physically with ``pwd -P`` in ``cd_resolve`` makes both directions
agree with bash across POSIX symlinks and Windows junctions.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from tests.governance._guard_harness import World, build_world, decide, git


@pytest.fixture
def world(tmp_path: Path) -> World:
    """Create throwaway repos ``p`` (``main``) and ``f`` (``feature``) with ``feature`` on ``p``."""
    w = build_world(tmp_path)
    git(w.p, "branch", "feature")
    return w


def _link_dir(target_dir: Path, link_path: Path, rel_target: Path) -> None:
    """Create a directory symlink, falling back to an NTFS junction on unprivileged Windows."""
    try:
        os.symlink(rel_target, link_path, target_is_directory=True)
    except OSError:
        if sys.platform != "win32":
            raise
        import _winapi

        _winapi.CreateJunction(str(target_dir), str(link_path))


def _require_git() -> None:
    assert shutil.which("git") is not None, "git is required to test pretooluse_guard.sh"


def test_guard_cd_dash_p_from_protected_through_junction_allows(world: World) -> None:
    """D-4 case 1 (feature -> main link): physical ``cd -P`` leaves ``main``; logical stays."""
    _require_git()
    (world.f / "sub").mkdir()
    _link_dir(world.f / "sub", world.p / "fl", Path("..") / "f" / "sub")

    assert decide(world.p, "cd -P fl/.. && git commit -m x").decision is None
    assert decide(world.p, "cd fl/.. && git commit -m x").decision == "deny"


def test_guard_cd_dash_p_into_protected_through_junction_denies(world: World) -> None:
    """D-4 case 2 (main -> feature link): physical ``cd -P`` enters ``main`` and must deny."""
    _require_git()
    (world.p / "sub").mkdir()
    _link_dir(world.p / "sub", world.f / "pl", Path("..") / "p" / "sub")

    assert decide(world.f, "cd -P pl/.. && git commit -m x").decision == "deny"
    assert decide(world.f, "cd pl/.. && git commit -m x").decision is None


@pytest.mark.parametrize(
    "cmd_template",
    [
        "cd -P {f}/pl/.. && git commit -m x",
        "cd {f} && cd -P pl/.. && git commit -m x",
    ],
)
def test_guard_cd_dash_p_absolute_path_into_protected_denies(
    world: World, cmd_template: str
) -> None:
    """D-5 (#1): absolute/drive-letter ``cd -P`` into a protected repo must deny."""
    _require_git()
    (world.p / "sub").mkdir()
    _link_dir(world.p / "sub", world.f / "pl", Path("..") / "p" / "sub")
    f_posix = world.f.as_posix()
    assert " " not in f_posix and " " not in world.root.as_posix(), (
        f"unquoted test paths cannot contain spaces: {f_posix!r}"
    )

    assert decide(world.root, cmd_template.format(f=f_posix)).decision == "deny"
    assert decide(world.root, f"cd {f_posix}/pl/.. && git commit -m x").decision is None
