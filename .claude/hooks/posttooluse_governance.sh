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
  */Agent.md|Agent.md)
    python3 scripts/validate_agents.py --path "$rel" --quiet || exit 2
    ;;
  .claude/skills/*|.agents/skills/*)
    python3 scripts/validate_skills.py --path "$rel" --quiet || exit 2
    ;;
  CLAUDE.md|README.md|CONTRIBUTING.md|CHANGELOG.md|ROADMAP.md|docs/*|pyproject.toml|alpha_zero/__init__.py|setup.py)
    # Documentation drift is otherwise caught only by CI, so it lives in the
    # tree for a whole session. This validator has no --path mode (its checks
    # are cross-file by nature) but a whole-repo run costs ~100ms, which is
    # affordable at edit time.
    python3 scripts/check_doc_sync.py --quiet || exit 2
    ;;
esac
exit 0
