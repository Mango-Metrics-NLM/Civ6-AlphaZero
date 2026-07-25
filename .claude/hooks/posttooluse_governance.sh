#!/usr/bin/env bash
# PostToolUse (Write|Edit): run the governance validator for the edited file, in
# single-file (--path) mode so a mid-scaffold edit only sees its own file's
# problems (no cross-file completeness / cross-reference spam). A failure exits 2
# so its stderr is surfaced back to the agent without blocking unrelated work.
set -uo pipefail

input="$(cat)"
f="$(printf '%s' "$input" | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print(d.get("tool_response",{}).get("filePath") or d.get("tool_input",{}).get("file_path") or "")' \
  2>/dev/null || true)"
[ -z "$f" ] && exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
rel="${f#"${CLAUDE_PROJECT_DIR:-.}"/}"

case "$rel" in
  openspec/*)
    python3 scripts/validate_openspec.py --path "$rel" --quiet || exit 2
    ;;
  kb/*)
    python3 scripts/validate_kb.py --path "$rel" --quiet || exit 2
    ;;
  .claude/skills/*)
    python3 scripts/validate_skills.py --path "$rel" --quiet || exit 2
    ;;
esac
exit 0
