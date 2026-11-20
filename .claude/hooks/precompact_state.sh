#!/usr/bin/env bash
# PreCompact: fold KB session state, recent ledger lines, and a git diff stat into
# the compacted context. The JSON is built with json.dumps so that quotes or
# newlines in the state files cannot produce invalid JSON (the previous inline
# printf implementation could).
set -uo pipefail

HOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_resolve_python.sh
. "${HOOK_DIR}/_resolve_python.sh"

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || { printf '{}'; exit 0; }
PY="$(resolve_python "${CLAUDE_PROJECT_DIR:-.}" || true)"
[ -n "$PY" ] || { printf '{}'; exit 0; }

"$PY" - <<'PY' 2>/dev/null || printf '{}'
import json
import pathlib
import subprocess


def read(path):
    try:
        return pathlib.Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""


kb = "".join(read(f"kb/sessions/{role}.state.json") for role in ("planning", "dev-sqe", "pre-pr"))
ledger_lines = read("kb/ledger/ledger.jsonl").splitlines()[-10:]
ledger = "\n".join(ledger_lines)
try:
    git = subprocess.run(
        ["git", "diff", "--stat", "HEAD"], capture_output=True, text=True
    ).stdout[-2000:]
except OSError:
    git = ""

context = f"KB_STATE: {kb} RECENT_LEDGER: {ledger} GIT_STATUS: {git}"
print(
    json.dumps(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreCompact",
                "additionalContext": context,
            }
        }
    )
)
PY
