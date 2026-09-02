#!/usr/bin/env bash
# PostToolUse (Write|Edit and Bash): run the governance validator for the edited
# file, in single-file (--path) mode so a mid-scaffold edit only sees its own
# file's problems (no cross-file completeness / cross-reference spam). A failure
# exits 2 so its stderr is surfaced back to the agent without blocking unrelated
# work.
#
# It also carries the debt ratchet (2026-09-02). The ratchet
# (tests/regression/test_debt_ratchet.py) held new dead modules to the committed
# baseline only in CI and at pre-push, so a module added early in a session was
# reported dead hours later by a machine the author was no longer watching. The
# two events that can move `dead_modules` are a file APPEARING under alpha_zero/
# and a file DISAPPEARING from it -- and only the first is a Write|Edit, which is
# why this script is registered under the Bash matcher too. Ordinary edits to an
# already-tracked module never trigger it: the import graph cannot change without
# a file appearing or leaving.
set -uo pipefail

input="$(cat)"

# One python3 start extracts both payload shapes: the edited path (Write|Edit)
# and the command (Bash). Newlines inside the command are collapsed so the two
# values stay on two lines; the command is only ever pattern-matched, never run.
parsed="$(printf '%s' "$input" | python3 -c '
import json, sys
d = json.load(sys.stdin)
inp = d.get("tool_input", {})
print(d.get("tool_response", {}).get("filePath") or inp.get("file_path") or "")
print(" ".join((inp.get("command") or "").split()))
' 2>/dev/null || true)"
f="$(printf '%s\n' "$parsed" | sed -n 1p)"
cmd="$(printf '%s\n' "$parsed" | sed -n 2p)"
[ -z "$f" ] && [ -z "$cmd" ] && exit 0

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
rel="${f#"${CLAUDE_PROJECT_DIR:-.}"/}"

# Run the debt ratchet and surface its report on stderr. ~10-22 s depending on
# machine load, so every caller below gates it on an event that can actually
# change the import graph.
run_debt_ratchet() {
  if ! python3 -c 'import pytest' 2>/dev/null; then
    printf 'debt ratchet skipped: pytest is not importable (see the SessionStart report)\n' >&2
    return 0
  fi
  local out
  if ! out="$(OMP_NUM_THREADS=2 python3 -m pytest tests/regression/test_debt_ratchet.py \
      -q --no-header -p no:cacheprovider 2>&1)"; then
    printf '%s\n' "$out" >&2
    printf 'debt ratchet FAILED -- see the `debt-audit` skill: a module was added or removed and\n' >&2
    printf 'docs/evidence/tech_debt_baseline/baseline.json no longer describes this tree.\n' >&2
    return 1
  fi
  return 0
}

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
  alpha_zero/*.py)
    # A NEW module only. `git ls-files --error-unmatch` is the discriminator:
    # it exits non-zero for a path git does not track, which is exactly a file
    # this session created. Editing a tracked module cannot change the import
    # graph's node set, so the common case pays nothing (measured 42 ms).
    if ! git ls-files --error-unmatch "$rel" >/dev/null 2>&1; then
      run_debt_ratchet || exit 2
    fi
    ;;
esac

# Bash payload: a command that REMOVES or RENAMES a module. Anchored on the
# program actually being a removal (optionally behind `FOO=bar ` assignments or
# a `git` prefix) so that merely naming a path in an argument -- `grep -rn rm
# alpha_zero/`, `ls alpha_zero/rm_test.py` -- is not a match. Heuristic, in the
# same spirit as pretooluse_guard.sh: a miss costs a late CI failure, not a
# wrong answer.
if [ -n "$cmd" ]; then
  env_prefix='^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]+[[:space:]]+)*'
  # Fed by here-doc rather than a pipe, so the loop runs in THIS shell and its
  # `exit 2` is the hook's exit code. Piping into `while` puts the loop in a
  # subshell, where the exit is swallowed and the hook reports success --
  # the failure mode this repo keeps re-finding, so it is spelled out.
  segments="$(printf '%s' "$cmd" | sed -E 's/(\&\&|\|\||;|\|)/\n/g')"
  while IFS= read -r seg; do
    printf '%s' "$seg" | grep -qE "${env_prefix}(git[[:space:]]+)?(rm|mv)([[:space:]]|$)" || continue
    printf '%s' "$seg" | grep -qE '(^|[[:space:]"'"'"'])alpha_zero/[A-Za-z0-9_/.-]*\.py' || continue
    run_debt_ratchet || exit 2
    break
  done <<EOF
$segments
EOF
fi

exit 0
