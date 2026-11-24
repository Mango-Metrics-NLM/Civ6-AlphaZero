"""Review round: how ``||``, redirections, ``&`` and marker bytes interact with a ``cd``.

``p`` is on ``main`` (protected), ``f`` on ``feature``; the hook runs in ``f`` unless noted.
``cd <dir> || x`` runs the ``cd`` whatever follows, so the left side of ``||`` carries its
directory; the right side runs only when the left failed, so what follows it cannot be placed.
Redirections are not arguments of ``cd``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.governance._guard_harness import (
    HAVE_TOOLS,
    World,
    build_world,
    decide,
    not_allowed,
)

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path)


# --- B1: || ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template",
    [
        "cd {p} || exit 1; git commit -m x",
        "cd {p} || echo failed; git commit -m x",
        "cd {p} || exit 1 && git commit -m x",
        "cd {p} || exit 1\ngit commit -m x",
    ],
)
def test_cd_on_the_left_of_or_carries(world: World, template: str) -> None:
    got = decide(world.f, template.format(p=world.p))
    assert got.decision == "deny", got


@pytest.mark.parametrize(
    "template",
    [
        "false || cd {p}; git commit -m x",
        "false || cd {p} && git commit -m x",
    ],
)
def test_cd_on_the_right_of_or_makes_the_directory_unknown(world: World, template: str) -> None:
    got = decide(world.f, template.format(p=world.p))
    assert got.decision == "ask", got
    assert "could not tell which branch" in got.reason


# --- B2: redirections are not cd arguments ------------------------------------------------


@pytest.mark.parametrize(
    "template",
    [
        "cd {p} &>/dev/null && git commit -m x",
        "cd {p} >/dev/null 2>&1 && git commit -m x",
        "cd {p} 2>/dev/null && git commit -m x",
        "cd {p} 2>&1; git commit -m x",
        "cd {p} > /dev/null && git commit -m x",
    ],
)
def test_redirections_after_cd_do_not_hide_the_directory(world: World, template: str) -> None:
    got = decide(world.f, template.format(p=world.p))
    assert got.decision == "deny", got  # a deny, not an ask: the directory is known


@pytest.mark.parametrize(
    "template",
    [
        'cmd /c "cd /d {p} & git commit -m x"',
        "pushd {p} & git commit -m x",
        "cd {p} & git commit -m x",
    ],
)
def test_ampersand_runs_in_sequence_on_cmd_so_it_is_not_allowed(
    world: World, template: str
) -> None:
    # cmd's `&` is a sequence; bash's is a background job. Either reading must not allow.
    got = decide(world.f, template.format(p=world.p))
    assert not_allowed(got), got


def test_a_feature_target_is_still_allowed_after_redirections(world: World) -> None:
    assert decide(world.p, f"cd {world.f} >/dev/null 2>&1 && git commit -m x").decision is None
    assert decide(world.p, f"cd {world.f} || exit 1; git commit -m x").decision is None


# --- marker bytes ----------------------------------------------------------------------------


# Every marker byte the splitter uses internally (\x02 .. \x08, \x0e .. \x10) must be stripped
# from the input.
@pytest.mark.parametrize("marker", [chr(c) for c in (*range(2, 9), *range(0x0E, 0x11))])
@pytest.mark.parametrize(
    "template",
    [
        "cd {p}{m}; git commit -m x",
        "cd {p} {m}&& git commit -m x",
        "cd {p}\n{m}\ngit commit -m x",
        "cd{m} {p} && git commit -m x",
    ],
)
def test_literal_marker_bytes_do_not_unset_the_directory(
    world: World, marker: str, template: str
) -> None:
    got = decide(world.f, template.format(p=world.p, m=marker))
    assert got.decision == "deny", got
