#!/usr/bin/env bash
# PreToolUse guard for sdlc-qa-sdet (matcher `Edit|Write`), attached through that
# role's own frontmatter `hooks:`. The role may write tests and nothing else:
# a file_path that does not resolve to a location under `tests/` is DENIED.
#
# An accident net, not a control (identity is the control). Fails CLOSED: an
# unreadable payload, a missing path, a `..` segment or any script failure is a
# deny with exit 0 (a non-zero exit would let the edit run).
#
# Path rule, on the path with backslashes turned into slashes:
#   * any `..` segment is denied (no walking out of tests/);
#   * an absolute path must start with "$CLAUDE_PROJECT_DIR/tests/";
#   * a relative path must start with `tests/` (or `./tests/`).
set -euo pipefail
set -f

decided=0

# The decision JSON is built by jq, else python3/python -- never by hand-escaping
# shell strings. Hand escaping is how a Windows path (`alpha_zero\core\x.py`)
# once produced malformed JSON, which Claude Code ignores: a deny that fails
# open. If no builder works the reason is a fixed string with no untrusted text.
json_decision() {
  local dec="$1" reason="$2" out="" py
  reason="${reason//[[:cntrl:]]/ }"
  if command -v jq >/dev/null 2>&1 \
     && out="$(jq -n -c --arg d "$dec" --arg r "$reason" \
        '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:$d,permissionDecisionReason:$r}}' 2>/dev/null)" \
     && [ -n "$out" ]; then
    printf '%s' "$out"
    return 0
  fi
  for py in python3 python; do
    command -v "$py" >/dev/null 2>&1 || continue
    if out="$(printf '%s' "$reason" | "$py" -c 'import json,sys
r=sys.stdin.buffer.read().decode("utf-8","replace")
sys.stdout.write(json.dumps({"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":sys.argv[1],"permissionDecisionReason":r}}))' "$dec" 2>/dev/null)" \
       && [ -n "$out" ]; then
      printf '%s' "$out"
      return 0
    fi
  done
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"%s","permissionDecisionReason":"%s"}}' \
    "$dec" "guard could not build its message (no jq or python); see the guard script for the rule that fired."
}

deny() {
  decided=1
  json_decision deny "tests-only write guard: $1 sdlc-qa-sdet may edit tests/** only; return anything else as text to the caller."
  exit 0
}

# shellcheck disable=SC2329  # invoked by the EXIT trap
on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "$decided" -eq 0 ]; then
    decided=1
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}' \
      "tests-only write guard failed unexpectedly (exit ${rc}); denying."
    exit 0
  fi
}
trap on_exit EXIT

if [ -n "${CIV6AZ_GUARD_SELFTEST_FAIL:-}" ]; then
  false
fi

input="$(cat)"
PY_EXTRACT='import json,sys
d=json.load(sys.stdin)
t=d.get("tool_input") if isinstance(d,dict) else None
c=t.get("file_path") if isinstance(t,dict) else None
sys.stdout.buffer.write((c if isinstance(c,str) else "").encode("utf-8"))'

# Exactly one JSON document, as python's json.load requires: jq would otherwise read
# empty input as zero documents (exit 0) and print one file_path per document.
[ -n "${input//[[:space:]]/}" ] || deny "the hook payload is empty."

fp=""
parsed=0
if command -v jq >/dev/null 2>&1 \
   && fp="$(printf '%s' "$input" | jq -r -s 'if length==1 and (.[0]|type)=="object" then (.[0].tool_input.file_path // empty) else error("not exactly one payload") end' 2>/dev/null)"; then
  parsed=1
fi
if [ "$parsed" -eq 0 ]; then
  for py in python3 python; do
    command -v "$py" >/dev/null 2>&1 || continue
    if fp="$(printf '%s' "$input" | "$py" -c "$PY_EXTRACT" 2>/dev/null)"; then
      parsed=1
      break
    fi
  done
fi
[ "$parsed" -eq 1 ] || deny "cannot inspect this edit: no working jq or python could parse the hook payload."
[ -n "$fp" ] || deny "the edit names no file_path."

bsl=$(printf "\\")
p="${fp//"$bsl"//}"
case "/$p/" in
  */../*) deny "'$fp' contains a '..' segment." ;;
esac

# Win32 strips trailing dots and spaces from a path segment, so `.. `, `...` and
# `..<TAB>` can mean `..`. Any segment made only of dots, spaces and tabs, other than
# a bare `.`, is denied.
dot_re=$'^[. \t]+$'
IFS='/' read -r -a segs <<<"$p"
for seg in "${segs[@]}"; do
  [ "$seg" = "." ] && continue
  if [[ $seg =~ $dot_re ]]; then
    deny "'$fp' contains a dot/space-only segment ('$seg')."
  fi
done

root="${CLAUDE_PROJECT_DIR:-}"
root="${root//"$bsl"//}"
root="${root%/}"

case "$p" in
  tests/* | ./tests/*) exit 0 ;;
esac
if [ -n "$root" ]; then
  case "$p" in
    "$root"/tests/*) exit 0 ;;
  esac
fi
deny "'$fp' is outside tests/."
