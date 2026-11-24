"""PR #171 round 1, independent probe: shell shapes the guard misread (silent allows).

From a probe with real payloads and real bash:
- line continuations (``git \\<newline>commit``, PowerShell's backtick-newline) hid the subcommand;
- a ``cd`` that runs in a child process (``bash -c``, ``pwsh -c``, ``cmd /c``, a pipeline element,
  ``timeout``/``nohup``/``env``) was carried to the outer command;
- ``$'...'`` and heredocs could forge the parentheses that restore a directory;
- ``!``, ``|| return``, a quoted or commented ``|| exit`` and a redefined ``exit`` were read as
  certain;
- a very long command could run past the hook's timeout, which fails open.

``p`` is on ``main`` (protected), ``f`` on ``feature``.
"""

from __future__ import annotations

import time
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
    git(w.p, "branch", "feature")
    return w


# --- 1: line continuations -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("where", "command", "tool"),
    [
        ("p", "git \\\ncommit -m x", "Bash"),
        ("f", "git \\\npush origin HEAD:main", "Bash"),
        ("f", "git push \\\norigin HEAD:main", "Bash"),
        ("f", "git switch \\\nmain && git commit -m x", "Bash"),
        ("p", "git `\ncommit -m x", "PowerShell"),
        ("f", "git switch `\nmain && git commit -m x", "PowerShell"),
    ],
)
def test_a_line_continuation_does_not_hide_the_command(
    world: World, where: str, command: str, tool: str
) -> None:
    got = decide(getattr(world, where), command, tool_name=tool)
    assert got.decision == "deny", got


def test_an_escaped_backslash_before_a_newline_is_not_a_continuation(world: World) -> None:
    # `\\` is a literal backslash, so the newline still ends the command: the push is its own.
    got = decide(world.f, "echo x \\\\\ngit push origin HEAD:main")
    assert got.decision == "deny", got


def test_a_continued_cd_chain_is_still_certain(world: World) -> None:
    assert decide(world.p, f"cd {world.f} \\\n&& git commit -m x").decision is None


# --- 2: a cd in a child process does not carry -------------------------------------------------


@pytest.mark.parametrize(
    ("template", "tool"),
    [
        ('bash -c "cd {f} && make test" && git commit -m x', "Bash"),
        ("bash -c 'cd {f}' && git commit -m x", "Bash"),
        ('sh -c "cd {f}; true" ; git commit -m x', "Bash"),
        ('pwsh -c "cd {f}" && git commit -m x', "Bash"),
        ('pwsh -c "cd {f}"; git commit -m x', "PowerShell"),
        ('cmd /c "cd /d {f}" && git commit -m x', "Bash"),
        ("true | cd {f} && git commit -m x", "Bash"),
        ("timeout 5 cd {f} && git commit -m x", "Bash"),
        ("nohup cd {f} && git commit -m x", "Bash"),
        ("env cd {f} && git commit -m x", "Bash"),
    ],
)
def test_a_cd_in_a_child_process_does_not_move_the_commit(
    world: World, template: str, tool: str
) -> None:
    got = decide(world.p, template.format(f=world.f), tool_name=tool)
    assert got.decision == "deny", got


def test_the_child_shell_itself_runs_in_its_cd(world: World) -> None:
    assert decide(world.p, f'bash -c "cd {world.f} && git commit -m x"').decision is None
    assert decide(world.f, f'bash -c "cd {world.p} && git commit -m x"').decision == "deny"


@pytest.mark.parametrize("prefix", ["builtin", "command"])
def test_builtin_and_command_run_cd_in_this_shell(world: World, prefix: str) -> None:
    assert decide(world.p, f"{prefix} cd {world.f} && git commit -m x").decision is None


def test_powershell_set_location_in_a_pipeline_is_not_ignored(world: World) -> None:
    got = decide(
        world.f, f"Set-Location {world.p} | Out-Null; git commit -m x", tool_name="PowerShell"
    )
    assert not_allowed(got), got


# --- 5: parentheses forged by $'...' and heredocs ----------------------------------------------


@pytest.mark.parametrize(
    "template",
    [
        "echo $'x\\';(' ; cd {p} ; echo $'y\\');' ; git commit -m x",
        "cat <<'EOF'\n(\nEOF\ncd {p}\ncat <<'EOF'\n)\nEOF\ngit commit -m x",
        "cat > /dev/null <<'EOF'\nDon't forget\nEOF\necho '$(' && cd {p} && echo ')' && git commit -m x",
    ],
)
def test_forged_parentheses_do_not_restore_the_directory(world: World, template: str) -> None:
    got = decide(world.f, template.format(p=world.p))
    assert not_allowed(got), got


# --- 6: shapes misread as certain ----------------------------------------------------------------


@pytest.mark.parametrize(
    "template",
    [
        "! cd {f} && git commit -m x",
        "! git switch feature && git commit -m x",
        "! cd {f} || exit; git commit -m x",
        "cd {f} || return; git commit -m x",
        "exit() {{ :; }}; cd {f} || exit; git commit -m x",
        'cd {f} && echo "ok || exit 1"; git commit -m x',
        "cd {f} # x || exit\ngit commit -m x",
        'git switch feature && echo "a || exit 1"; git commit -m x',
        "git switch feature # || exit\ngit commit -m x",
    ],
)
def test_shapes_that_only_look_certain_ask(world: World, template: str) -> None:
    got = decide(world.p, template.format(f=world.f))
    assert got.decision == "ask", got


def test_a_plain_or_exit_is_still_certain(world: World) -> None:
    assert decide(world.p, f"cd {world.f} || exit 1; git commit -m x").decision is None


# --- 8: a command too long to judge asks ---------------------------------------------------------


def test_three_hundred_segments_ask_quickly(world: World) -> None:
    start = time.monotonic()
    got = decide(world.f, "echo a; " * 300 + "git status")
    assert time.monotonic() - start < 10
    assert got.decision == "ask", got
    assert "too long" in got.reason


def test_too_many_git_queries_ask(world: World) -> None:
    command = "; ".join(f"git -C {world.f}{'/.' * i} commit -m x" for i in range(1, 25))
    got = decide(world.f, command)
    assert got.decision == "ask", got
    assert "too long" in got.reason
