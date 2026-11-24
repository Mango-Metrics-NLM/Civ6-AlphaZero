"""G4: the decision JSON is built by a real encoder, not by hand-escaping.

``emit`` replaced control characters and escaped ``\\`` and ``"``, then used ``printf``. Bytes
that are not valid UTF-8 passed through, which makes the output invalid JSON; Claude Code
ignores an invalid decision, which is a fail-open. The rule: build it like
``readonly_shell_guard.sh``'s ``json_decision`` (``jq -n``, then python's ``json.dumps`` with
the bytes decoded as utf-8/replace, then a fixed safe string that still carries the decision),
and keep the EXIT-trap deny working with neither jq nor python.

Untrusted text reaches the reason in three ways, all used here: the protected-branch list
(``CIV6AZ_PROTECTED_BRANCHES``, shown in a deny), a runner name taken from the command (an
ask), and a remote-tracking ref name (the recreated-branch ask; git allows ``\\xff`` in one).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from tests.governance._guard_harness import (
    HAVE_TOOLS,
    World,
    base_env,
    build_world,
    git,
    parse,
    run_raw,
)

pytestmark = pytest.mark.skipif(not HAVE_TOOLS, reason="guard needs git plus jq or python")

RAW = os.fsdecode(b"a\xffb")  # b"a\xffb" on POSIX (surrogateescape); a plain str elsewhere
ODD_NAMES = {
    "control": "a\x01b",
    "quote": 'a"b',
    "backslash": "a\\b",
    "multibyte": "ä€é",
    "raw_byte": RAW,
}
# Tool sets on PATH: the guard prefers jq, then python3, then python.
TOOLSETS = {
    "jq_and_python": ("git", "cat", "jq", "python3", "python"),
    "python_only": ("git", "cat", "python3", "python"),
}


def _bin_dir(tmp_path: Path, tools: tuple[str, ...]) -> str:
    """A PATH holding exactly ``tools``; a missing real tool is simply left out."""
    d = tmp_path / ("bin_" + "_".join(tools))
    d.mkdir(exist_ok=True)
    for tool in tools:
        real = shutil.which(tool)
        if real is not None and not (d / tool).exists():
            (d / tool).symlink_to(Path(real).resolve())
    return str(d)


def _env(tmp_path: Path, tools: tuple[str, ...], **extra: str) -> dict[str, str]:
    env = base_env(shipped=False)
    env["PATH"] = _bin_dir(tmp_path, tools)
    env.update(extra)
    return env


@pytest.fixture
def world(tmp_path: Path) -> World:
    return build_world(tmp_path)


@pytest.mark.parametrize("toolset", sorted(TOOLSETS))
@pytest.mark.parametrize("kind", sorted(ODD_NAMES))
def test_deny_with_an_odd_protected_name_is_valid_json(
    world: World, tmp_path: Path, toolset: str, kind: str
) -> None:
    name = ODD_NAMES[kind]
    env = _env(tmp_path, TOOLSETS[toolset], CIV6AZ_PROTECTED_BRANCHES=name)
    got = parse(run_raw(world.f, "git push origin HEAD:main", env))
    assert got.decision == "deny"
    assert not any(ord(c) < 0x20 for c in got.reason)
    if kind in ("quote", "backslash", "multibyte"):
        assert name in got.reason  # survives the round trip unchanged
    if kind == "control":
        assert "a" in got.reason and "b" in got.reason


@pytest.mark.parametrize("toolset", sorted(TOOLSETS))
@pytest.mark.parametrize(
    "module", ["alpha_zero.train_\x01x", "alpha_zero.train_a\\b", "alpha_zero.train_ä€"]
)
def test_ask_with_an_odd_runner_name_is_valid_json(
    world: World, tmp_path: Path, toolset: str, module: str
) -> None:
    env = _env(tmp_path, TOOLSETS[toolset])
    got = parse(run_raw(world.f, f"python -m {module}", env))
    assert got.decision == "ask"
    assert not any(ord(c) < 0x20 for c in got.reason)
    assert "game runner" in got.reason


@pytest.mark.parametrize("toolset", sorted(TOOLSETS))
def test_ask_with_a_non_utf8_ref_name_is_valid_json(
    world: World, tmp_path: Path, toolset: str
) -> None:
    # A protected branch whose remote-tracking ref name holds a raw 0xff byte, and a push of
    # merged work to a new name: the recreated-branch ask names that branch in its reason.
    p = world.p
    odd = os.fsdecode(b"we\xffrd")  # str passed to subprocess is re-encoded with the same byte
    git(p, "checkout", "-q", "-b", "claude/merged-work")
    tip = git(p, "rev-parse", "HEAD")
    git(p, "update-ref", "-d", "refs/remotes/origin/main")  # leave only the odd name
    git(p, "update-ref", f"refs/remotes/origin/{odd}", tip)
    env = _env(tmp_path, TOOLSETS[toolset], CIV6AZ_PROTECTED_BRANCHES=odd)
    got = parse(run_raw(p, "git push origin claude/merged-work", env))
    assert got.decision == "ask"
    assert "merged" in got.reason.lower()


def test_ask_and_deny_still_work_without_jq_through_python(world: World, tmp_path: Path) -> None:
    env = _env(tmp_path, TOOLSETS["python_only"])
    assert parse(run_raw(world.f, "git push origin HEAD:qa", env)).decision == "deny"
    assert parse(run_raw(world.f, "python -m alpha_zero.train_x", env)).decision == "ask"


def test_without_jq_or_python_the_guard_still_answers_with_valid_json(
    world: World, tmp_path: Path
) -> None:
    env = _env(tmp_path, ("git", "cat"))
    got = parse(run_raw(world.f, "git push origin HEAD:qa", env))
    assert got.decision == "deny"  # it cannot read the command, so it denies
    assert "cannot inspect" in got.reason


def test_exit_trap_deny_works_without_jq_or_python(world: World, tmp_path: Path) -> None:
    env = _env(tmp_path, ("git", "cat"), CIV6AZ_GUARD_SELFTEST_FAIL="1")
    got = parse(run_raw(world.f, "git status", env))
    assert got.decision == "deny"
    assert "failed unexpectedly" in got.reason


def test_the_shell_guard_json_builder_is_not_hand_escaped() -> None:
    """The decision is built by an encoder: no ``printf`` that splices the reason in."""
    guard = (Path(__file__).resolve().parents[2] / ".claude/hooks/pretooluse_guard.sh").read_text(
        encoding="utf-8"
    )
    emit_start = guard.index("emit() {")
    emit_body = guard[emit_start : guard.index("\n}\n", emit_start)]
    assert 'printf \'{"hookSpecificOutput"' not in emit_body
    assert "jq" in emit_body or "json_decision" in emit_body
