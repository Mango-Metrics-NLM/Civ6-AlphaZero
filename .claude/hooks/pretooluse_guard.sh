#!/usr/bin/env bash
# PreToolUse guard: enforces the CLAUDE.md Section 3.2 PROHIBITED list that was
# previously only documented (prose), not machine-enforced.
#
# Denies:
#   * git commit / git push when it would land on `main` OR on the repository's
#     default branch (they are not the same branch in this repo)
#   * self-play / training invocations above the 1000-game cap
#   * force-pushes (--force / -f / +refspec; --force-with-lease is allowed)
#
# Reads the Bash tool input as JSON on stdin and, on a violation, emits a
# PreToolUse deny decision; otherwise exits 0 silently (allow / fall through to
# normal permission handling). Input shape (Claude Code PreToolUse for Bash):
#   {"tool_input": {"command": "git push origin main"}}
#
# Default-branch resolution, first non-empty wins:
#   1. CIV6AZ_DEFAULT_BRANCH  -- explicit override (tests; a clone whose
#                                origin/HEAD is unset or points elsewhere)
#   2. git symbolic-ref --short refs/remotes/origin/HEAD, `origin/` stripped
#   3. `main`
# The literal `main` is always protected as well, so a clone with no
# origin/HEAD (a fresh `git clone` sets it; a manual `git remote add` does not)
# keeps the historical rule.
#
# Matching is intentionally ANCHORED to the command actually being the dangerous
# invocation (optionally prefixed by `ENV=val` assignments), so that merely
# *mentioning* a trigger string inside a quoted argument (echo/printf, a test
# fixture, a doc heredoc) does NOT trip the guard. Heuristic, not a sandbox.
set -euo pipefail

input="$(cat)"

deny() {
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}' "$1"
  exit 0
}

# Command extraction, with a fallback. This read the payload through `jq`
# alone, swallowing every error into `|| true`: with jq absent, `cmd` came back
# empty and the guard exited 0 on the next line, so `git push --force origin
# main` was SILENTLY ALLOWED with no signal on any channel (measured 2026-09-02
# by running the hook with jq off PATH -- empty output, exit 0). jq is not
# installed by `pip install -e '.[dev]'`, not checked by the SessionStart hook,
# and not named in any manifest, so its absence is an ordinary environment, not
# a hypothetical. python3 is the fallback because every other hook in this
# directory already requires it.
#
# The distinction that matters: a parser that RAN and found no command means a
# non-Bash payload (allow, as before); NO parser at all means the guard cannot
# see what it is guarding, which is the one case that must fail closed.
if command -v jq >/dev/null 2>&1; then
  cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null || true)"
elif command -v python3 >/dev/null 2>&1; then
  cmd="$(printf '%s' "$input" | python3 -c \
    'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("command") or "")' \
    2>/dev/null || true)"
else
  deny "CLAUDE.md S3.2 guard cannot inspect this command: neither jq nor python3 is available, so the prohibited-action checks cannot run. Install jq or python3 rather than proceeding unguarded."
fi
[ -z "$cmd" ] && exit 0

# Optional leading `FOO=bar ` env assignments before the real program.
env_prefix='^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]+[[:space:]]+)*'

# Global git options that may sit between `git` and the subcommand, e.g.
# `git -C /repo push ...`, which otherwise bypassed every rule below.
# Any run of git's own global options before the subcommand. Enumerating a
# few by name (-C, --git-dir, --work-tree) left every other one as a bypass:
# `git --no-pager push origin main`, `git -c k=v push ...` and `git -P push ...`
# all slipped through the whole rule. Match an option run generically instead --
# a leading `-` is never a subcommand, so this cannot swallow `push`/`commit`.
git_opts='((-[A-Za-z-][^[:space:]]*|--[A-Za-z][A-Za-z0-9-]*(=[^[:space:]]+)?)[[:space:]]+([^-][^[:space:]]*[[:space:]]+)?)*'

# Split the command into segments on shell separators (&&, ||, ;, |, newline) so
# a chained `... && git push origin main` is caught, while each segment is still
# matched ANCHORED (so a quoted `echo 'git push origin main'` stays a no-match).
segments="$(printf '%s' "$cmd" | sed -E 's/(\&\&|\|\||;|\|)/\n/g')"

branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")"

# Repository default branch (see header). Resolved once per invocation.
default_branch="${CIV6AZ_DEFAULT_BRANCH:-}"
if [ -z "$default_branch" ]; then
  default_branch="$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null || true)"
  default_branch="${default_branch#origin/}"
fi
[ -z "$default_branch" ] && default_branch="main"

# Protected push destinations as an ERE alternation. Branch names may contain
# `.` (and in principle `+`), so the default branch is escaped before it is
# interpolated into the pattern below.
protected_re='main'
if [ "$default_branch" != "main" ]; then
  escaped="$(printf '%s' "$default_branch" | sed -E 's/[][\.^$*+?(){}|\\]/\\&/g')"
  protected_re="(main|${escaped})"
fi

while IFS= read -r seg; do
  [ -z "$seg" ] && continue

  # 1) No direct commits/pushes to main or the default branch. Segment must
  #    START with git (an optional `-C <dir>` is allowed between `git` and the
  #    subcommand).
  #
  #    Two independent triggers, each written as its own `if` because the
  #    earlier `A || B && C` form parsed as `(A || B) && C` in bash -- so a
  #    commit while checked out on main was denied only if the segment ALSO
  #    mentioned `main` as a destination, which a commit never does:
  #      (a) the working copy is checked out on a protected branch, so a
  #          commit or push would land there;
  #      (b) the segment is a push whose DESTINATION is a protected branch,
  #          in any refspec form, not just as a trailing bare word:
  #          `HEAD:main`, `feat:main`, `refs/heads/main`, `HEAD:refs/heads/main`
  #          and a quoted `'main'`
  #          all previously slipped through, and `HEAD:main` is the idiomatic
  #          way to push a branch to main. Anchoring the tail on whitespace or
  #          end-of-segment (rather than \b) also stops `main-fix` matching.
  if printf '%s' "$seg" | grep -qE "${env_prefix}git[[:space:]]+${git_opts}(commit|push)\b"; then
    if [ "$branch" = "main" ] || [ "$branch" = "$default_branch" ]; then
      deny "CLAUDE.md S3.2: direct commits/pushes to ${branch} (a protected branch) are prohibited - use a feature branch."
    fi
    if printf '%s' "$seg" | grep -qE "${env_prefix}git[[:space:]]+${git_opts}push([[:space:]]|$)" && \
       printf '%s' "$seg" | grep -qE -- "(^|[[:space:]])['\"]?(refs/heads/)?([A-Za-z0-9_/.^~-]+:)?(refs/heads/)?${protected_re}['\"]?([[:space:]]|\$)"; then
      deny "CLAUDE.md S3.2: direct pushes to main / the default branch (${default_branch}) are prohibited - use a feature branch."
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
