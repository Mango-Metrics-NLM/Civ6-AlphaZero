"""PowerShell's call operator after a ``cd``: ``cd <dir>; & git commit -m x``.

Assumption: the PowerShell tool sends its command in ``tool_input.command`` like Bash, and the
guard splits on ``;`` and ``&`` (the ``&`` ends the preceding segment and the operator itself
leaves an empty segment), so ``& git ...`` becomes a plain ``git ...`` segment. The directory
set by the ``cd`` / ``Set-Location`` segment must still reach it, because ``;`` carries it and
only a ``&`` AFTER a ``cd`` (a background job) would not. ``p`` is on ``main`` (protected),
``f`` on ``feature``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.governance._guard_harness import ALLOWED, HAVE_TOOLS, World, build_world, decide

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")

CHANGERS = ["cd", "Set-Location", "sl", "chdir", "Set-Location -Path"]
CALLS = ["& git commit -m x", '& "git" commit -m x', "& 'git' commit -m x", "& git push origin x"]


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path)


@pytest.mark.parametrize("call", CALLS)
@pytest.mark.parametrize("changer", CHANGERS)
def test_protected_target_is_denied(world: World, changer: str, call: str) -> None:
    got = decide(world.f, f"{changer} {world.p}; {call}", tool_name="PowerShell")
    assert got.decision == "deny", got


@pytest.mark.parametrize("call", CALLS[:2])
@pytest.mark.parametrize("changer", CHANGERS[:2])
def test_feature_target_is_allowed_from_a_protected_directory(
    world: World, changer: str, call: str
) -> None:
    # The hook's own directory is on `main`; the cd target decides when the change is certain
    # (PowerShell 7's `&&`). After `;` a failed change leaves the commit on main: it asks.
    got = decide(world.p, f"{changer} {world.f} && {call}", tool_name="PowerShell")
    assert got == ALLOWED
    got = decide(world.p, f"{changer} {world.f}; {call}", tool_name="PowerShell")
    assert got.decision == "ask", got
    assert "may not have taken effect" in got.reason


def test_quoted_directory_and_chained_set_location(world: World) -> None:
    assert (
        decide(
            world.f, f'Set-Location "{world.p}"; & git commit -m x', tool_name="PowerShell"
        ).decision
        == "deny"
    )
    cmd = f"Set-Location {world.root}; Set-Location p; & git commit -m x"
    assert decide(world.f, cmd, tool_name="PowerShell").decision == "deny"


@pytest.mark.parametrize(
    "template",
    [
        "cd {p};& git commit -m x",
        "cd {p}; &git commit -m x",
        "cd {p} && & git commit -m x",
        "cd {p}\n& git commit -m x",
        "cd '{p}'; & git commit -m x",
        'cd {p}; &"git.exe" commit -m x',
        "Set-Location -LiteralPath {p}; & git commit -m x",
    ],
)
def test_spelling_variants_of_the_call_operator_are_denied(world: World, template: str) -> None:
    got = decide(world.f, template.format(p=world.p), tool_name="PowerShell")
    assert got.decision == "deny", got
