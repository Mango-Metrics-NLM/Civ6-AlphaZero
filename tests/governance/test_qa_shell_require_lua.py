"""H2: the QA shell may prefix pytest with ``CIV6AZ_REQUIRE_LUA=1`` (and only that).

Slip: the test role could not run the Lua suites with the flag that turns a missing
``lupa`` into a failure, so "a skip is not a pass" could not be enforced from inside
the role. The prefix is allowed once, in either order with ``OMP_NUM_THREADS=<digits>``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.governance._bash import resolve_bash

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / ".claude" / "hooks" / "readonly_shell_guard.sh"
BASH = resolve_bash()
PYTEST = "python -m pytest tests/unit/test_x.py -q"

pytestmark = pytest.mark.skipif(
    not any(shutil.which(t) for t in ("jq", "python3", "python")),
    reason="guard needs jq or python on PATH",
)


def _run(command: str, *args: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [BASH, GUARD.as_posix(), *args],
        cwd=REPO_ROOT,
        input=json.dumps({"tool_input": {"command": command}}),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc


def _decide(command: str, *args: str) -> str | None:
    proc = _run(command, *args)
    if not proc.stdout.strip():
        return None
    return str(json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecision"])


def qa(command: str) -> str | None:
    return _decide(command, "--qa")


@pytest.mark.parametrize(
    "prefix",
    [
        "CIV6AZ_REQUIRE_LUA=1",
        "CIV6AZ_REQUIRE_LUA=1 OMP_NUM_THREADS=2",
        "OMP_NUM_THREADS=2 CIV6AZ_REQUIRE_LUA=1",
    ],
)
def test_qa_allows_require_lua_prefix_before_pytest(prefix: str) -> None:
    assert qa(f"{prefix} {PYTEST}") is None


@pytest.mark.parametrize(
    "command",
    [
        f"CIV6AZ_REQUIRE_LUA=0 {PYTEST}",
        f"CIV6AZ_REQUIRE_LUA=2 {PYTEST}",
        f"CIV6AZ_REQUIRE_LUA=true {PYTEST}",
        f"CIV6AZ_REQUIRE_LUA= {PYTEST}",
        f"CIV6AZ_REQUIRE_LUA=11 {PYTEST}",
        f"CIV6AZ_REQUIRE_LUA=1 CIV6AZ_REQUIRE_LUA=1 {PYTEST}",
        f"CIV6AZ_REQUIRE_LUA=1 OMP_NUM_THREADS=2 OMP_NUM_THREADS=2 {PYTEST}",
        f"OMP_NUM_THREADS=2 CIV6AZ_REQUIRE_LUA=1 CIV6AZ_REQUIRE_LUA=1 {PYTEST}",
        f"CIV6AZ_REQUIRE_LUA=1 FOO=1 {PYTEST}",
        f"CIV6AZ_REQUIRE_LUA=1 PYTHONPATH=x {PYTEST}",
        f"CIV6AZ_OTHER=1 {PYTEST}",
        "CIV6AZ_REQUIRE_LUA=1 git log",
        "CIV6AZ_REQUIRE_LUA=1 python scripts/validate_kb.py",
        "CIV6AZ_REQUIRE_LUA=1 python -c print(1)",
        "CIV6AZ_REQUIRE_LUA=1 make verify",
        "CIV6AZ_REQUIRE_LUA=1 python -m pytest",
        "CIV6AZ_REQUIRE_LUA=1 python -m pytest alpha_zero/",
    ],
)
def test_qa_still_denies_everything_else(command: str) -> None:
    assert qa(command) == "deny", command


@pytest.mark.parametrize(
    "command", [f"CIV6AZ_REQUIRE_LUA=1 {PYTEST}", "CIV6AZ_REQUIRE_LUA=1 git log"]
)
def test_prefix_without_qa_mode_stays_denied(command: str) -> None:
    assert _decide(command) == "deny"


def test_deny_message_lists_the_new_prefix() -> None:
    proc = _run("git commit -m x", "--qa")
    reason = json.loads(proc.stdout)["hookSpecificOutput"]["permissionDecisionReason"]
    assert "CIV6AZ_REQUIRE_LUA=1" in reason


@pytest.mark.parametrize(
    "command",
    [
        "OMP_NUM_THREADS=2 git log",
        "OMP_NUM_THREADS=2 python scripts/validate_kb.py",
        "CIV6AZ_REQUIRE_LUA=1",
        "OMP_NUM_THREADS=2",
        "CIV6AZ_REQUIRE_LUA=1 OMP_NUM_THREADS=2",
        "CIV6AZ_REQUIRE_LUA=1 X=1 -m pytest tests/x",
        "CIV6AZ_REQUIRE_LUA=1 git -m pytest tests/x",
        "civ6az_require_lua=1 python -m pytest tests/unit/test_x.py",
        "CIV6AZ_REQUIRE_LUA=1 python -m pytest tests/governance -c tests/governance/pytest.ini",
        "CIV6AZ_REQUIRE_LUA=1 python -m pytest tests/x.py --rootdir=/",
        "CIV6AZ_REQUIRE_LUA=1 python -m pytest tests/x.py -p evil_plugin",
        "CIV6AZ_REQUIRE_LUA=1 python -m pytest tests/x.py --junitxml=out.xml",
    ],
)
def test_qa_denies_these_with_or_without_the_prefix(command: str) -> None:
    assert qa(command) == "deny", command


def test_python3_after_the_prefix_behaves_as_without_it() -> None:
    bare = "python3 -m pytest tests/unit/test_x.py -q"
    assert qa(bare) is None  # python3 is accepted without a prefix, so it is with one
    assert qa(f"CIV6AZ_REQUIRE_LUA=1 {bare}") is None
    assert qa(f"OMP_NUM_THREADS=2 CIV6AZ_REQUIRE_LUA=1 {bare}") is None
