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
import shlex
import shutil
import subprocess
import sys
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


def _write_shell_fixture(path: Path, content: str) -> Path:
    """Write an executable bash fixture with explicit UTF-8 LF newlines."""
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(0o755)
    assert b"\r\n" not in path.read_bytes()
    return path


def _exec_python_shim(real_py: str) -> str:
    """Return a safely quoted bash wrapper invoking ``real_py``."""
    return f'#!/usr/bin/env bash\nexec {shlex.quote(real_py)} "$@"\n'


def _run_resolve_python(
    proj_dir: Path,
    *,
    cwd: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    """Source ``_resolve_python.sh`` and invoke ``resolve_python`` via positional args."""
    return subprocess.run(
        [
            BASH,
            "-c",
            'source "$1" && resolve_python "$2"',
            "_",
            (_HOOKS / "_resolve_python.sh").as_posix(),
            proj_dir.as_posix(),
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_usr_bin() -> Path | None:
    """Return Git for Windows' ``usr/bin`` directory alongside ``BASH``, if present."""
    bash_resolved = Path(BASH).resolve()
    if len(bash_resolved.parents) > 1:
        usr_bin = bash_resolved.parents[1] / "usr" / "bin"
        if usr_bin.is_dir():
            return usr_bin
    return None


def _normalize_shell_path(raw: str) -> Path:
    """Resolve POSIX or MSYS2 paths (including ``/tmp`` mounts) returned by Git Bash."""
    if sys.platform == "win32" and raw.startswith("/"):
        usr_bin = _git_usr_bin()
        cygpath_exe = usr_bin / "cygpath.exe" if usr_bin is not None else None
        cygpath = (
            str(cygpath_exe)
            if cygpath_exe is not None and cygpath_exe.is_file()
            else shutil.which("cygpath")
        )
        if cygpath is not None:
            converted = subprocess.check_output([cygpath, "-w", raw], text=True).strip()
            if converted:
                return Path(converted).resolve()
        if len(raw) >= 3 and raw[1].isalpha() and raw[2] == "/":
            return Path(f"{raw[1].upper()}:{raw[2:]}").resolve()
    return Path(raw).resolve()


def _minimal_hook_bin_dirs(tmp_path: Path) -> list[str]:
    """Return PATH directories providing shell/hook utilities without system Python."""
    if sys.platform == "win32":
        dirs = [str(Path(BASH).resolve().parent)]
        usr_bin = _git_usr_bin()
        if usr_bin is not None:
            dirs.append(str(usr_bin))
        return dirs

    safe_bin = tmp_path / "safe_bin"
    safe_bin.mkdir(parents=True, exist_ok=True)
    for tool in ("bash", "sh", "env", "dirname", "cat", "sed", "grep", "git"):
        resolved = shutil.which(tool)
        if resolved:
            link = safe_bin / tool
            if not link.exists():
                link.symlink_to(resolved)
    return [str(safe_bin)]


def test_hooks_resolve_project_venv_outside_cwd_and_skip_windowsapps(tmp_path: Path) -> None:
    """Shared ``_resolve_python.sh`` resolves project ``.venv`` outside cwd and enforces >=3.11."""
    proj_dir = tmp_path / "proj"
    outside_cwd = tmp_path / "outside"
    venv_bin = proj_dir / ".venv" / "bin"
    winapps_dir = tmp_path / "WindowsApps"
    old_py_dir = tmp_path / "oldpy"
    for d in (proj_dir, outside_cwd, venv_bin, winapps_dir, old_py_dir):
        d.mkdir(parents=True)

    # Plant a poisoned WindowsApps stub and a <3.11 stub that passes `import json,sys`.
    _write_shell_fixture(
        winapps_dir / "python3",
        "#!/usr/bin/env bash\necho 'Store stub should be skipped' >&2\nexit 1\n",
    )
    _write_shell_fixture(
        old_py_dir / "python",
        '#!/usr/bin/env bash\ncase "$*" in *version_info*) exit 1 ;; *) exit 0 ;; esac\n',
    )
    real_py = Path(sys.executable).as_posix()
    venv_py = _write_shell_fixture(
        venv_bin / "python",
        _exec_python_shim(real_py),
    )

    path_parts = [str(winapps_dir), str(old_py_dir), *_minimal_hook_bin_dirs(tmp_path)]
    env = {
        "PATH": os.pathsep.join(path_parts),
        "CLAUDE_PROJECT_DIR": proj_dir.as_posix(),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "SystemRoot": os.environ.get("SystemRoot", ""),
    }
    proc = _run_resolve_python(proj_dir, cwd=outside_cwd, env=env)
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
        newline="\n",
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
    (proj_dir / "x.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    _write_shell_fixture(
        venv_py,
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-m" ] && [ "$2" = "ruff" ]; then\n'
        f'  printf "%s\\n" "$*" >> {shlex.quote(marker.as_posix())}\n'
        "  exit 0\n"
        "fi\n"
        f'exec {shlex.quote(real_py)} "$@"\n',
    )
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


def test_resolve_python_unset_virtual_env_prefers_path_and_set_virtual_env_wins(
    tmp_path: Path,
) -> None:
    """Unset/empty VIRTUAL_ENV selects PATH over .venv & system; active VIRTUAL_ENV wins."""
    proj_dir = tmp_path / "proj"
    winapps_dir = tmp_path / "windowsapps"
    path_bin = tmp_path / "path_py"
    active_venv = tmp_path / "active_venv" / "bin"
    proj_venv = proj_dir / ".venv" / "bin"
    for d in (proj_dir, winapps_dir, path_bin, active_venv, proj_venv):
        d.mkdir(parents=True)

    real_py = Path(sys.executable).as_posix()
    _write_shell_fixture(
        winapps_dir / "python3",
        "#!/usr/bin/env bash\nexit 1\n",
    )
    path_py = _write_shell_fixture(
        path_bin / "python3",
        _exec_python_shim(real_py),
    )
    _write_shell_fixture(
        proj_venv / "python",
        _exec_python_shim(real_py),
    )
    active_py = _write_shell_fixture(
        active_venv / "python",
        _exec_python_shim(real_py),
    )

    base_env = {
        "PATH": os.pathsep.join(
            [str(winapps_dir), str(path_bin), *_minimal_hook_bin_dirs(tmp_path)]
        ),
        "CLAUDE_PROJECT_DIR": proj_dir.as_posix(),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "SystemRoot": os.environ.get("SystemRoot", ""),
    }

    for extra in ({}, {"VIRTUAL_ENV": ""}):
        proc = _run_resolve_python(proj_dir, cwd=tmp_path, env={**base_env, **extra})
        assert proc.returncode == 0, proc.stderr
        resolved = proc.stdout.strip()
        assert resolved.endswith("/path_py/python3"), f"unexpected {resolved!r} for {extra!r}"
        assert resolved != "/bin/python"
        assert _normalize_shell_path(resolved) == path_py.resolve()

    active_proc = _run_resolve_python(
        proj_dir,
        cwd=tmp_path,
        env={**base_env, "VIRTUAL_ENV": active_venv.parent.as_posix()},
    )
    assert active_proc.returncode == 0, active_proc.stderr
    active_resolved = active_proc.stdout.strip()
    assert active_resolved.endswith("/active_venv/bin/python")
    assert _normalize_shell_path(active_resolved) == active_py.resolve()
