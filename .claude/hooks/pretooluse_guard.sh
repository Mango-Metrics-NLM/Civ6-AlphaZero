#!/usr/bin/env bash
# PreToolUse guard: enforces the CLAUDE.md Section 3.2 PROHIBITED list that was
# previously only documented (prose), not machine-enforced.
#
# Denies:
#   * git commit / git push when it would land on `main`
#   * self-play / training invocations above the 1000-game cap
#
# Reads the Bash tool input as JSON on stdin and, on a violation, emits a
# PreToolUse deny decision; otherwise exits 0 silently (allow / fall through to
# normal permission handling).
#
# Matching is intentionally ANCHORED to the command actually being the dangerous
# invocation (optionally prefixed by `ENV=val` assignments), so that merely
# *mentioning* a trigger string inside a quoted argument (echo/printf, a test
# fixture, a doc heredoc) does NOT trip the guard. Heuristic, not a sandbox.
set -euo pipefail

input="$(cat)"
cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null || true)"
[ -z "$cmd" ] && exit 0

deny() {
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}' "$1"
  exit 0
}

# Optional leading `FOO=bar ` env assignments before the real program.
env_prefix='^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]+[[:space:]]+)*'

# Split the command into segments on shell separators (&&, ||, ;, |, newline) so
# a chained `... && git push origin main` is caught, while each segment is still
# matched ANCHORED (so a quoted `echo 'git push origin main'` stays a no-match).
segments="$(printf '%s' "$cmd" | sed -E 's/(\&\&|\|\||;|\|)/\n/g')"

branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"

while IFS= read -r seg; do
  [ -z "$seg" ] && continue

  # 1) No direct commits/pushes to main. Segment must START with git.
  if printf '%s' "$seg" | grep -qE "${env_prefix}git[[:space:]]+(commit|push)\b"; then
    if [ "$branch" = "main" ] || \
       printf '%s' "$seg" | grep -qE "${env_prefix}git[[:space:]]+push[^|&;]*[[:space:]]main(\b|\$)"; then
      deny "CLAUDE.md S3.2: direct commits/pushes to main are prohibited - use a feature branch."
    fi
  fi

  # 2) Game-count cap (>1000). Segment must be a python / alpha_zero invocation.
  if printf '%s' "$seg" | grep -qE "${env_prefix}(python|python3|.*alpha_zero)"; then
    games="$(printf '%s' "$seg" | grep -oE -- '--num[_-]games[=[:space:]]+[0-9]+' | grep -oE '[0-9]+' | head -1 || true)"
    if [ -n "$games" ] && [ "$games" -gt 1000 ]; then
      deny "CLAUDE.md S3.2: runs with >1000 games require confirmation (requested ${games})."
    fi
  fi
done <<EOF
$segments
EOF

exit 0
