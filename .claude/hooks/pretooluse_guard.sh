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

# Global git options that may sit between `git` and the subcommand, e.g.
# `git -C /repo push ...`, which otherwise bypassed every rule below.
git_opts='(-C[[:space:]]+[^[:space:]]+[[:space:]]+|--git-dir=[^[:space:]]+[[:space:]]+|--work-tree=[^[:space:]]+[[:space:]]+)*'

# Split the command into segments on shell separators (&&, ||, ;, |, newline) so
# a chained `... && git push origin main` is caught, while each segment is still
# matched ANCHORED (so a quoted `echo 'git push origin main'` stays a no-match).
segments="$(printf '%s' "$cmd" | sed -E 's/(\&\&|\|\||;|\|)/\n/g')"

branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"

while IFS= read -r seg; do
  [ -z "$seg" ] && continue

  # 1) No direct commits/pushes to main. Segment must START with git (an
  #    optional `-C <dir>` is allowed between `git` and the subcommand).
  #
  #    `main` is matched as a push DESTINATION in any refspec form, not just as
  #    a trailing bare word: `HEAD:main`, `feat:main`, `refs/heads/main` and a
  #    quoted `'main'` all previously slipped through, and `HEAD:main` is the
  #    idiomatic way to push a branch to main. Anchoring the tail on whitespace
  #    or end-of-segment (rather than \b) also stops `main-fix` matching.
  if printf '%s' "$seg" | grep -qE "${env_prefix}git[[:space:]]+${git_opts}(commit|push)\b"; then
    if [ "$branch" = "main" ] || \
       printf '%s' "$seg" | grep -qE "${env_prefix}git[[:space:]]+${git_opts}push([[:space:]]|$)" && \
       printf '%s' "$seg" | grep -qE -- "(^|[[:space:]])['\"]?(refs/heads/)?([A-Za-z0-9_/.^~-]+:)?main['\"]?([[:space:]]|\$)"; then
      deny "CLAUDE.md S3.2: direct commits/pushes to main are prohibited - use a feature branch."
    fi
  fi

  # 2) Game-count cap (>1000). Segment must START with a python / make / script
  #    invocation. The previous `.*alpha_zero` alternative defeated the anchor,
  #    so `echo "... --num_games 2000"` in prose was denied while the real flag
  #    names in this repo (--games, --games-per-worker, --games_per_worker) were
  #    not matched at all.
  if printf '%s' "$seg" | grep -qE "${env_prefix}(python3?|make|bash|sh|\./scripts/)"; then
    games="$(printf '%s' "$seg" | grep -oE -- '--(num[_-])?games([_-]per[_-]worker)?[=[:space:]]+[0-9]+' | grep -oE '[0-9]+' | head -1 || true)"
    # The env-var form used by scripts/run_cloud_training.sh and the Makefile
    # (GAMES_PER_WORKER=..., NUM_GAMES=...), which no flag regex would catch.
    if [ -z "$games" ]; then
      games="$(printf '%s' "$seg" | grep -oE -- '(^|[[:space:]])(NUM_GAMES|GAMES|GAMES_PER_WORKER)=[0-9]+' | grep -oE '[0-9]+' | head -1 || true)"
    fi
    if [ -n "$games" ] && [ "$games" -gt 1000 ]; then
      deny "CLAUDE.md S3.2: runs with >1000 games require confirmation (requested ${games})."
    fi
  fi

  # 3) No force-pushes, in any argument position (--force-with-lease is allowed).
  #    Deny bare --force / -f / a +refspec. The deny list's Bash prefix rules only
  #    catch the flag directly after `push`, so this regex is the real backstop.
  if printf '%s' "$seg" | grep -qE "${env_prefix}git[[:space:]]+${git_opts}push\b"; then
    if printf '%s' "$seg" | grep -qE -- '--force([[:space:]]|=|$)' \
       || printf '%s' "$seg" | grep -qE -- '(^|[[:space:]])-f([[:space:]]|$)' \
       || printf '%s' "$seg" | grep -qE -- '[[:space:]]\+[A-Za-z0-9_/.-]+(:|[[:space:]]|$)'; then
      deny "CLAUDE.md S3.2: force-pushes are prohibited (use --force-with-lease if truly needed)."
    fi
  fi
done <<EOF
$segments
EOF

exit 0
