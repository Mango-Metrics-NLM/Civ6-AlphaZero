"""Shared helpers for the push-guard gap tests (G1-G4 of the guard-gaps contract).

Drives the real ``.claude/hooks/pretooluse_guard.sh`` with a real stdin payload, in real
throwaway repositories that have a bare ``origin``. The ``env`` block of the shipped
``.claude/settings.json`` can be loaded, because a rule that only passes with those settings
removed is not a rule (the review of #170 found one).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from tests.governance._bash import resolve_bash

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARD = REPO_ROOT / ".claude" / "hooks" / "pretooluse_guard.sh"
SETTINGS = REPO_ROOT / ".claude" / "settings.json"
BASH = resolve_bash()
KNOBS = ("CIV6AZ_DEFAULT_BRANCH", "CIV6AZ_PROTECTED_BRANCHES")
GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.invalid",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}
HAVE_TOOLS = shutil.which("git") is not None and any(
    shutil.which(tool) for tool in ("jq", "python3", "python")
)


@dataclass(frozen=True)
class Decision:
    """What the hook said: ``decision`` is None when it printed nothing (no opinion)."""

    decision: str | None
    reason: str


def shipped_env() -> dict[str, str]:
    """The ``env`` block of the real settings.json, as Claude Code exports it."""
    env = json.loads(SETTINGS.read_text(encoding="utf-8"))["env"]
    return {str(k): str(v) for k, v in env.items()}


def git(cwd: Path, *args: str | bytes) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env={**os.environ, **GIT_ENV},
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def base_env(*, shipped: bool = True) -> dict[str, str]:
    """The process environment minus the guard's knobs, plus the shipped settings if asked."""
    env = {k: v for k, v in os.environ.items() if k not in KNOBS}
    if shipped:
        env.update(shipped_env())
    return env


def payload(command: str, tool_name: str = "Bash") -> str:
    return json.dumps({"tool_name": tool_name, "tool_input": {"command": command}})


def run_raw(
    cwd: Path, command: str, env: dict[str, str] | None = None, tool_name: str = "Bash"
) -> subprocess.CompletedProcess[bytes]:
    """Run the guard with a payload on stdin; bytes out, so invalid UTF-8 is visible."""
    return subprocess.run(
        [BASH, GUARD.as_posix()],
        cwd=cwd,
        input=payload(command, tool_name).encode("utf-8"),
        env=env if env is not None else base_env(),
        capture_output=True,
        timeout=30,
        check=False,
    )


def parse(proc: subprocess.CompletedProcess[bytes]) -> Decision:
    """Parse the hook's stdout as strict JSON (invalid JSON or UTF-8 raises)."""
    assert proc.returncode == 0, proc.stderr
    if not proc.stdout.strip():
        return Decision(None, "")
    out = json.loads(proc.stdout)["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    return Decision(str(out["permissionDecision"]), str(out["permissionDecisionReason"]))


def decide(
    cwd: Path,
    command: str,
    *,
    shipped: bool = True,
    env: dict[str, str] | None = None,
    tool_name: str = "Bash",
) -> Decision:
    return parse(
        run_raw(cwd, command, env if env is not None else base_env(shipped=shipped), tool_name)
    )


ALLOWED = Decision(None, "")


@dataclass(frozen=True)
class World:
    """``p`` is checked out on ``main`` (protected); ``f`` on ``feature`` with unique work."""

    root: Path
    origin: Path
    p: Path
    f: Path


def build_world(tmp_path: Path) -> World:
    root = tmp_path / "world"
    root.mkdir()
    origin = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    p = root / "p"
    git(tmp_path, "init", "-q", "-b", "main", str(p))
    git(p, "commit", "-q", "--allow-empty", "-m", "init")
    git(p, "remote", "add", "origin", str(origin))
    git(p, "push", "-q", "origin", "main")
    git(p, "fetch", "-q", "origin")
    f = root / "f"
    git(tmp_path, "clone", "-q", str(origin), str(f))
    git(f, "checkout", "-q", "-b", "feature")
    git(f, "commit", "-q", "--allow-empty", "-m", "feature work")
    return World(root=root, origin=origin, p=p, f=f)


def not_allowed(got: Decision) -> bool:
    """True when the hook denied or asked: the command does not run unattended."""
    return got.decision in ("deny", "ask")
