"""Shared shell discovery helper for Windows and POSIX governance hook tests.

On Windows, ``System32\\bash.exe`` is often the WSL launcher stub, which
precedes Git for Windows on ``PATH`` and fails with exit code 127 when
invoked against Windows repository paths without a configured WSL distro.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def find_bash(platform: str = sys.platform) -> str | None:
    """Return an executable ``bash`` path avoiding Windows' System32 WSL stub, or ``None``."""
    if platform != "win32":
        return shutil.which("bash")

    candidates: list[Path] = []
    git_exe = shutil.which("git")
    if git_exe is not None:
        git_root = Path(git_exe).resolve().parent.parent
        candidates.extend(
            [
                git_root / "bin" / "bash.exe",
                git_root / "usr" / "bin" / "bash.exe",
            ]
        )

    for env_var, suffix in (
        ("ProgramW6432", "Git/bin/bash.exe"),
        ("ProgramFiles", "Git/bin/bash.exe"),
        ("LocalAppData", "Programs/Git/bin/bash.exe"),
    ):
        base = os.environ.get(env_var)
        if base:
            candidates.append(Path(base) / suffix)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    which_bash = shutil.which("bash")
    if which_bash and not any(
        marker in str(Path(which_bash).resolve()).lower()
        for marker in ("\\system32\\", "\\windowsapps\\")
    ):
        return which_bash
    return None


def resolve_bash(platform: str = sys.platform) -> str:
    """Return an executable ``bash`` suitable for running hook scripts."""
    return find_bash(platform=platform) or "bash"
