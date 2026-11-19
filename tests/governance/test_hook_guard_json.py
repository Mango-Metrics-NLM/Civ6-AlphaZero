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

from tests.governance._bash import resolve_bash

REPO_ROOT = Path(__file__).resolve().parents[2]
_HOOKS = REPO_ROOT / ".claude" / "hooks"
READONLY_GUARD = Path(
    os.environ.get("CIV6AZ_TEST_READONLY_GUARD", str(_HOOKS / "readonly_shell_guard.sh"))
)
WRITE_GUARD = Path(
    os.environ.get("CIV6AZ_TEST_WRITE_GUARD", str(_HOOKS / "tests_only_write_guard.sh"))
)
BASH = resolve_bash()
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


def test_hooks_resolve_project_venv_outside_cwd_and_skip_windowsapps(tmp_path: Path) -> None:
    """Shared ``_resolve_python.sh`` resolves project ``.venv`` outside cwd and enforces >=3.11."""
    import sys

    proj_dir = tmp_path / "proj"
    outside_cwd = tmp_path / "outside"
    venv_bin = proj_dir / ".venv" / "bin"
    winapps_dir = tmp_path / "WindowsApps"
    old_py_dir = tmp_path / "oldpy"
    for d in (proj_dir, outside_cwd, venv_bin, winapps_dir, old_py_dir):
        d.mkdir(parents=True)

    # Plant a poisoned WindowsApps stub and a <3.11 stub that passes `import json,sys`.
    fake_stub = winapps_dir / "python3"
    fake_stub.write_text("#!/usr/bin/env bash\necho 'Store stub should be skipped' >&2\nexit 1\n")
    fake_stub.chmod(0o755)

    fake_old = old_py_dir / "python"
    fake_old.write_text(
        '#!/usr/bin/env bash\ncase "$*" in *version_info*) exit 1 ;; *) exit 0 ;; esac\n'
    )
    fake_old.chmod(0o755)

    real_py = Path(sys.executable).as_posix()
    venv_py = venv_bin / "python"
    venv_py.write_text(f'#!/usr/bin/env bash\nexec "{real_py}" "$@"\n')
    venv_py.chmod(0o755)

    bash_resolved = Path(BASH).resolve()
    path_parts = [str(winapps_dir), str(old_py_dir), str(bash_resolved.parent)]
    if len(bash_resolved.parents) > 1:
        usr_bin = bash_resolved.parents[1] / "usr" / "bin"
        if usr_bin.is_dir():
            path_parts.append(str(usr_bin))
    path_parts.extend(["/usr/bin", "/bin"])

    env = {
        "PATH": os.pathsep.join(path_parts),
        "CLAUDE_PROJECT_DIR": proj_dir.as_posix(),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "SystemRoot": os.environ.get("SystemRoot", ""),
    }
    helper = (_HOOKS / "_resolve_python.sh").as_posix()
    proc = subprocess.run(
        [BASH, "-c", f'source "{helper}" && resolve_python "{proj_dir.as_posix()}"'],
        cwd=outside_cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("/.venv/bin/python")
    assert "WindowsApps" not in proc.stdout

    # posttooluse_governance.sh invoked from outside_cwd must resolve proj/.venv
    # and run scripts/validate_agents.py (exiting 2 on a planted invalid Agent.md).
    scripts_dir = proj_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "validate_agents.py").write_text(
        "import sys; sys.exit(1)\n",
        encoding="utf-8",
    )
    payload = json.dumps({"tool_input": {"file_path": (proj_dir / "Agent.md").as_posix()}})
    gov_proc = subprocess.run(
        [BASH, (_HOOKS / "posttooluse_governance.sh").as_posix()],
        cwd=outside_cwd,
        input=payload,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert gov_proc.returncode == 2, (
        "posttooluse_governance.sh silently bypassed validator when cwd != CLAUDE_PROJECT_DIR"
    )

    # posttooluse_lint.sh must likewise reject fake <3.11 python and resolve .venv/bin/python.
    marker = proj_dir / "lint_ran.txt"
    (proj_dir / "x.py").write_text("x = 1\n", encoding="utf-8")
    venv_py.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-m" ] && [ "$2" = "ruff" ]; then\n'
        f'  printf "%s\\n" "$*" >> "{marker.as_posix()}"\n'
        "  exit 0\n"
        "fi\n"
        f'exec "{real_py}" "$@"\n'
    )
    venv_py.chmod(0o755)
    lint_payload = json.dumps({"tool_input": {"file_path": (proj_dir / "x.py").as_posix()}})
    lint_proc = subprocess.run(
        [BASH, (_HOOKS / "posttooluse_lint.sh").as_posix()],
        cwd=outside_cwd,
        input=lint_payload,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert lint_proc.returncode == 0
    assert marker.is_file() and "ruff check" in marker.read_text(encoding="utf-8")
