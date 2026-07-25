#!/usr/bin/env bash
# PostToolUse (Write|Edit): auto-format edited Python files. This hook must NEVER
# fail the originating tool call. Unused-import removal (F401) is DEFERRED to the
# commit-time gates (pre-commit / CI) so that an in-progress multi-edit change is
# not broken mid-sequence -- see docs/HARNESS_ENGINEERING_ANALYSIS.md section 6,
# which documents ruff --fix stripping a not-yet-used import between edits.
# RUF100 is deferred alongside F401: with F401 disabled the existing `# noqa: F401`
# directives look "unused", and RUF100 --fix would strip them, re-exposing the
# imports to the commit-time F401 gate. Deferring both keeps intentional
# availability-probe imports intact.
set -uo pipefail

input="$(cat)"
f="$(printf '%s' "$input" | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print(d.get("tool_response",{}).get("filePath") or d.get("tool_input",{}).get("file_path") or "")' \
  2>/dev/null || true)"
[ -z "$f" ] && exit 0

case "$f" in
  *.py)
    ruff check --fix --quiet --extend-ignore F401,RUF100 "$f" 2>/dev/null || true
    ruff format --quiet "$f" 2>/dev/null || true
    ;;
esac
exit 0
