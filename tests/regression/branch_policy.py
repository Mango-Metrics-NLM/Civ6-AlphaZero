"""The protected-branch list, read the way the PreToolUse guard reads it.

One definition shared by the regression tests that need it
(``test_workflow_pinning_regression.py``, ``test_coverage_floors_ratchet.py``),
so the workflow triggers, the coverage ratchet's base candidates and the agent
guard cannot each grow their own idea of which branches are protected.

Order of precedence matches ``.claude/hooks/pretooluse_guard.sh``:
``CIV6AZ_PROTECTED_BRANCHES`` from the environment, else the value
``.claude/settings.json`` declares, always on top of ``dev qa main`` (plan
S3.1) -- an unset variable must never mean "nothing is protected".

Not a test module (no ``test_`` prefix), so pytest does not collect it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SETTINGS = Path(".claude/settings.json")
ALWAYS_PROTECTED = ("dev", "qa", "main")


def _declared_in_settings() -> str:
    if not SETTINGS.exists():
        return ""
    try:
        env = json.loads(SETTINGS.read_text(encoding="utf-8")).get("env", {})
    except json.JSONDecodeError:
        return ""
    value = env.get("CIV6AZ_PROTECTED_BRANCHES", "") if isinstance(env, dict) else ""
    return str(value)


def protected_branches() -> tuple[str, ...]:
    """``dev qa main`` plus the configured list, de-duplicated, order kept."""
    raw = os.environ.get("CIV6AZ_PROTECTED_BRANCHES", "").strip() or _declared_in_settings()
    out: list[str] = []
    for name in (*ALWAYS_PROTECTED, *raw.split()):
        if name not in out:
            out.append(name)
    return tuple(out)
