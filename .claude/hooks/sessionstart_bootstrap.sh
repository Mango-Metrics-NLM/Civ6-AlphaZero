#!/usr/bin/env bash
# SessionStart: verify Python >= 3.11, install the dev extra, and REPORT WHAT IS
# ACTUALLY TRUE about the environment.
#
# This previously ran `pip install -e '.[dev]' || true` and then unconditionally
# printed "pytest ready". When the install failed the failure was invisible on
# every channel, and a session would proceed against a half-installed
# environment: `chess` publishes no wheel, so a source build failure there
# aborts the whole extra and leaves pytest plugins, bandit, psutil and httpx
# missing while the hook reported success.
#
# It also detects a tool/interpreter split: a `mypy` (or `ruff`) resolved from
# an isolated tool venv cannot see type stubs installed for python3, which
# surfaces as dozens of phantom import-untyped errors.
set -uo pipefail

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  printf '{"systemMessage": "WARNING: Python >= 3.11 is required for this project."}'
  exit 0
fi

install_log="$(mktemp)"
trap 'rm -f "$install_log"' EXIT
install_status="ok"
if ! pip install -e '.[dev]' -q >"$install_log" 2>&1; then
  install_status="failed"
fi

INSTALL_STATUS="$install_status" INSTALL_LOG="$install_log" python3 <<'PY'
import importlib.util
import json
import os
import shutil
import subprocess
import sys

# Import names for the dev-extra members a session actually depends on.
REQUIRED = {
    "pytest": "pytest",
    "numpy": "numpy",
    "torch": "torch",
    "yaml": "pyyaml",
    "gymnasium": "gymnasium",
}
OPTIONAL = {
    "hypothesis": "hypothesis",
    "psutil": "psutil",
    "httpx": "httpx",
    "bandit": "bandit",
    "tqdm": "tqdm",
    "pytest_asyncio": "pytest-asyncio",
    "pytest_timeout": "pytest-timeout",
}

missing_required = [pkg for mod, pkg in REQUIRED.items() if importlib.util.find_spec(mod) is None]
missing_optional = [pkg for mod, pkg in OPTIONAL.items() if importlib.util.find_spec(mod) is None]

# A tool on PATH resolved from its own venv cannot see stubs installed for
# python3; prefer `python3 -m <tool>` when that is the case.
split_tools = []
for tool, probe in (("mypy", "import yaml"), ("ruff", None)):
    path = shutil.which(tool)
    if not path or probe is None:
        continue
    try:
        result = subprocess.run([path, "-c", probe], capture_output=True, timeout=30)
        if result.returncode != 0:
            split_tools.append(tool)
    except Exception:
        pass

parts = [f"Python {sys.version.split()[0]}"]
try:
    parts.append(
        "ruff " + subprocess.run(["ruff", "--version"], capture_output=True, text=True).stdout.split()[1]
    )
except Exception:
    parts.append("ruff N/A")

if os.environ.get("INSTALL_STATUS") == "failed":
    try:
        with open(os.environ["INSTALL_LOG"]) as fh:
            tail = " / ".join(line.strip() for line in fh.readlines()[-3:] if line.strip())
    except Exception:
        tail = "(log unavailable)"
    parts.append(f"BOOTSTRAP FAILED: pip install -e '.[dev]' returned non-zero -- {tail[:300]}")

if missing_required:
    parts.append("MISSING (required): " + ", ".join(missing_required))
if missing_optional:
    parts.append("missing (optional): " + ", ".join(missing_optional))
if split_tools:
    parts.append(
        "tool/interpreter split for "
        + ", ".join(split_tools)
        + " -- these resolve to an isolated venv and cannot see installed stubs; "
        + "use `python3 -m <tool>` for accurate results"
    )
if not (missing_required or split_tools) and os.environ.get("INSTALL_STATUS") == "ok":
    parts.append("test suite runnable")

print(json.dumps({"systemMessage": "Environment: " + " | ".join(parts)}))
PY
exit 0
