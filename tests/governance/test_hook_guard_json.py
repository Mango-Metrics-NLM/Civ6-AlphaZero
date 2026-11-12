"""Every deny the two subagent guards emit must be valid JSON.

Claude Code ignores hook output it cannot parse, so a deny with malformed JSON is a
fail-open. The first version of ``tests_only_write_guard.sh`` escaped its message by
hand in shell; a Windows path (``alpha_zero\\core\\x.py``) went into that message and
broke the JSON. The decision is now built by jq / python, and these tests pin it with
the awkward inputs: backslashes, quotes, tabs, non-ASCII, ``..``.

Honours the same ``CIV6AZ_TEST_*_GUARD`` overrides as ``test_readonly_shell_guard.py``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOKS = REPO_ROOT / ".claude" / "hooks"
READONLY_GUARD = Path(
    os.environ.get("CIV6AZ_TEST_READONLY_GUARD", str(_HOOKS / "readonly_shell_guard.sh"))
)
WRITE_GUARD = Path(
    os.environ.get("CIV6AZ_TEST_WRITE_GUARD", str(_HOOKS / "tests_only_write_guard.sh"))
)
BASH = shutil.which("bash") or "bash"
BS = "\\"

pytestmark = pytest.mark.skipif(
    not any(shutil.which(t) for t in ("jq", "python3", "python")),
    reason="guards need jq or python on PATH",
)


def _stdout(
    script: Path, payload: dict[str, object], env_extra: dict[str, str] | None = None
) -> str:
    env = dict(os.environ)
    env.update(env_extra or {})
    proc = subprocess.run(
        [BASH, script.as_posix()],
        cwd=REPO_ROOT,
        input=json.dumps(payload),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _decision(out: str) -> dict[str, str]:
    parsed = json.loads(out)  # raises on malformed JSON: that is the assertion
    decision: dict[str, str] = parsed["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    return decision


WRITE_DENY_PATHS = [
    f"alpha_zero{BS}core{BS}x.py",
    f"C:{BS}repo{BS}alpha_zero{BS}x.py",
    f'a"b{BS}c.py',
    f"tests{BS}..{BS}alpha_zero{BS}x.py",
    "tab\there/x.py",
    "café/x.py",
    "scripts/x.py",
    "",
]


@pytest.mark.parametrize("file_path", WRITE_DENY_PATHS)
def test_write_guard_deny_is_valid_json(file_path: str) -> None:
    out = _stdout(WRITE_GUARD, {"tool_name": "Write", "tool_input": {"file_path": file_path}})
    decision = _decision(out)
    assert decision["permissionDecision"] == "deny"
    assert decision["permissionDecisionReason"]


def test_write_guard_deny_message_survives_the_backslashes() -> None:
    out = _stdout(
        WRITE_GUARD,
        {"tool_input": {"file_path": f"alpha_zero{BS}core{BS}x.py"}},
        {"CLAUDE_PROJECT_DIR": REPO_ROOT.as_posix()},
    )
    reason = _decision(out)["permissionDecisionReason"]
    assert "outside tests/" in reason
    assert "alpha_zero" in reason


def test_backslashed_path_inside_tests_is_normalised_and_allowed() -> None:
    """A backslash is a Windows separator, so ``tests\\unit\\x.py`` is ``tests/unit/x.py``."""
    env = {"CLAUDE_PROJECT_DIR": REPO_ROOT.as_posix()}
    allowed = _stdout(
        WRITE_GUARD, {"tool_input": {"file_path": f"tests{BS}unit{BS}test_x.py"}}, env
    )
    assert allowed.strip() == ""
    escaping = _stdout(
        WRITE_GUARD, {"tool_input": {"file_path": f"tests{BS}..{BS}scripts{BS}x.py"}}, env
    )
    assert _decision(escaping)["permissionDecision"] == "deny"
    sibling = _stdout(WRITE_GUARD, {"tool_input": {"file_path": f"testsx{BS}y.py"}}, env)
    assert _decision(sibling)["permissionDecision"] == "deny"


READONLY_DENY_COMMANDS = [
    "rm -rf x",
    f"git log --out{BS}put=x",
    'git diff "--output=/tmp/x"',
    "git log; rm -rf x",
    f"git diff {BS}alpha_zero{BS}core",
    "git café",
    "git log x,--output=/tmp/f",
    "git log --% --output=f",
    "git diff @args",
    "git diff *",
    f'git "{BS}" log',
    "x" * 4000,
]


@pytest.mark.parametrize("command", READONLY_DENY_COMMANDS)
def test_readonly_guard_deny_is_valid_json(command: str) -> None:
    out = _stdout(READONLY_GUARD, {"tool_name": "Bash", "tool_input": {"command": command}})
    decision = _decision(out)
    assert decision["permissionDecision"] == "deny"
    assert decision["permissionDecisionReason"]
