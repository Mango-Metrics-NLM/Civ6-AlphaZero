#!/usr/bin/env bash
# shellcheck disable=SC2329,SC1003  # ok_* are called indirectly by check_flags; bs is a literal backslash
# PreToolUse guard for the READ-ONLY SDLC role subagents (matcher `Bash|PowerShell`),
# attached through each role's own frontmatter `hooks:` -- never through
# settings.json, so it constrains those subagents and nobody else.
#
# An ACCIDENT NET, not a control (PROJECT_CHARTER.md S4.6): identity is the
# control. What this does is make "a read-only role" true by construction of its
# shell, not by the hope that a prompt is obeyed.
#
# Unlike pretooluse_guard.sh (a DENY list over a general shell), this is an
# ALLOW list over an almost empty one. A command is allowed only if it is ONE
# simple command, with no shell metacharacters at all, whose program and
# subcommand are on this list:
#
#   git log|show|diff|status|rev-parse|merge-base|ls-files|cat-file
#   git tag -l|--list [listing options]
#   gh pr view|list|checks
#   gh api            (path + --jq/-q, --paginate, -H X-GitHub-Api-Version:* only)
#   python scripts/{validate_,check_,verify_release_evidence}*.py
#
# Everything else is denied, which includes every wrapper (`bash -c`, `pwsh -c`,
# `cmd /c`, `env`, `sudo`...), every chain and pipe, every redirection, every
# expansion, and every program not named above. Because the first token must BE
# the program, no unwrapping is needed: nothing that wraps another command is on
# the list.
#
# Rejected characters, anywhere in the command:
#   ;  &  |  <  >  (  )  {  }  `  $  \  "  '  ,  @  #  %  and any newline
# (`,` because PowerShell splits `x,--output=f` into two arguments; `@` splats; `#`
# comments; `%` is `--%`, PowerShell's stop-parsing token.) After that EVERY token
# must match [A-Za-z0-9._/~^:+=-]+ (so no glob characters `* ? [`), and dashed
# tokens must additionally be on the per-command flag allowlist.
# That is `;`/`&&`/`||`/`|`/`&` chaining, `$( )`, backticks, `${}`, redirection
# to a file, subshells, script blocks, PowerShell call/escape syntax, and QUOTING.
# Quotes and backslashes matter because the command is tokenised on whitespace
# without shell-word processing: `git diff "--output=/tmp/x"` or
# `gh api -X 'POST'` would keep their quotes in the token and slip past every
# flag test. With none present, the token the guard sees is the word the shell
# runs. A legitimate `git log | head` or `--format="%h %s"` is therefore denied
# too; a read-only role reads the whole output / uses --format=%h instead.
#
# Options are an ALLOWLIST, not a denylist: every dashed argument to a git read
# subcommand and to `gh pr` must be on that subcommand's short list of safe
# flags, and `gh api` takes only a path plus --jq/-q, --paginate and
# -H/--header X-GitHub-Api-Version:*. Anything else (--output, --ext-diff,
# --textconv, --filters, -X, -f, --input, --web, ...) is denied by default. The
# request method of `gh api` cannot be changed, so it is gh's default, GET.
# Three scripts matching the python glob are excluded because they reach GCP or
# train: check_build.py, validate_gcp_credentials.py, validate_convergence.py;
# and an argument that is any prefix (argparse accepts abbreviations) of
# --out/--output/--write/--fix/--update/--apply/--execute is denied.
#
# Fails CLOSED: a payload no parser can read, or any unexpected script failure,
# is a DENY with exit 0 (a non-zero exit is a non-blocking error to Claude Code,
# i.e. the tool call would simply run).
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

emit() {
  decided=1
  json_decision "$1" "$2"
}

deny() {
  emit deny "read-only role shell guard: $1 Allowed: git log|show|diff|status|rev-parse|merge-base|ls-files|cat-file|tag -l; gh pr view|list|checks; gh api (GET only); python scripts/{validate_,check_,verify_release_evidence}*.py. Return the text you wanted to write to your caller instead."
  exit 0
}

# shellcheck disable=SC2329  # invoked by the EXIT trap
on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "$decided" -eq 0 ]; then
    decided=1
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}' \
      "read-only role shell guard failed unexpectedly (exit ${rc}); denying rather than letting the command run unchecked."
    exit 0
  fi
}
trap on_exit EXIT

# Test-only fault injection: proves the trap turns a crash into a deny. It can
# only make the guard deny, never allow.
if [ -n "${CIV6AZ_GUARD_SELFTEST_FAIL:-}" ]; then
  false
fi

input="$(cat)"

# jq, else python3, else python; a parser must EXIT 0 to count (see
# pretooluse_guard.sh for the failure this prevents). No parser -> deny.
PY_EXTRACT='import json,sys
d=json.load(sys.stdin)
t=d.get("tool_input") if isinstance(d,dict) else None
c=t.get("command") if isinstance(t,dict) else None
sys.stdout.buffer.write((c if isinstance(c,str) else "").encode("utf-8"))'

cmd=""
parsed=0
if command -v jq >/dev/null 2>&1 \
   && cmd="$(printf '%s' "$input" | jq -r '.tool_input.command // empty' 2>/dev/null)"; then
  parsed=1
fi
if [ "$parsed" -eq 0 ]; then
  for py in python3 python; do
    command -v "$py" >/dev/null 2>&1 || continue
    if cmd="$(printf '%s' "$input" | "$py" -c "$PY_EXTRACT" 2>/dev/null)"; then
      parsed=1
      break
    fi
  done
fi
if [ "$parsed" -eq 0 ]; then
  deny "cannot inspect this command: no working jq or python could parse the hook payload."
fi

# A trailing CR (Windows line endings) is not a command separator worth denying.
cmd="${cmd%$'\r'}"
# An empty command runs nothing.
[ -z "${cmd//[[:space:]]/}" ] && exit 0

# --- metacharacters: any occurrence, quoted or not --------------------------
if [[ $cmd == *$'\n'* || $cmd == *$'\r'* ]]; then
  deny "a multi-line command is not a single simple command."
fi
case "$cmd" in
  *';'* | *'&'* | *'|'* | *'<'* | *'>'* | *'('* | *')'* | *'{'* | *'}'* | *'`'* | *'$'* | *','* | *'@'* | *'#'* | *'%'*)
    deny "the command contains a shell metacharacter (chaining, pipe, redirection, substitution, a PowerShell array/splat/comment/--% token); only one plain command is allowed."
    ;;
esac
# Quotes and backslashes: see the header. Separate tests, because a lone quote
# or backslash inside a case pattern is easy to get wrong.
bs='\'
dq='"'
sq="'"
if [[ $cmd == *"$bs"* || $cmd == *"$dq"* || $cmd == *"$sq"* ]]; then
  deny "the command contains a quote or backslash; quoting could hide a flag from the guard."
fi

# --- tokenise on whitespace; with no quotes present this equals the shell's words
read -r -a tok <<<"$cmd"
n=${#tok[@]}
[ "$n" -eq 0 ] && exit 0

prog="${tok[0]}"
sub="${tok[1]:-}"
t=""
i=0

# Program and subcommand names match case-insensitively (Windows resolves
# `GIT`/`git.EXE`); every FLAG test below is case-sensitive.
lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }
progl="$(lower "$prog")"
subl="$(lower "$sub")"

# Every token must be a plain word. This is the ALLOWLIST applied to EVERY token,
# not only dashed ones: PowerShell splits `x,--output=/tmp/f` into two arguments,
# `@list` splats, `--%` stops parsing, and a glob (`*`, `?`, `[`) can expand to a
# file named like a flag. None of those characters is in the set below, so none
# can reach a command. The one exception is the value of `gh api --jq/-q`, which
# must be a plain dotted field path (`.name`, `.a.b`).
shape_re='^[A-Za-z0-9._/~^:+=-]+$'
jq_re='^[A-Za-z0-9._-]+$'
for ((i = 0; i < n; i++)); do
  t="${tok[$i]}"
  if [ "$i" -ge 1 ] && [ "$progl" = gh ] && [ "$subl" = api ]      && { [ "${tok[$((i - 1))]}" = -q ] || [ "${tok[$((i - 1))]}" = --jq ]; }; then
    [[ $t =~ $jq_re ]] || deny "the --jq filter '$t' is not a plain field path."
  else
    [[ $t =~ $shape_re ]] || deny "the argument '$t' contains a character outside the plain-word set (letters, digits and . _ / ~ ^ : + = -)."
  fi
done
t=""
i=0

# Check every dashed token from index $1 against the allow function named in $2.
# Non-dashed tokens (revisions, paths, patterns, values) are not flags.
check_flags() {
  local start="$1" fn="$2" k tk
  for ((k = start; k < n; k++)); do
    tk="${tok[$k]}"
    case "$tk" in
      -*)
        "$fn" "$tk" || deny "option '$tk' is not on the read-only allowlist for this command."
        ;;
    esac
  done
}

# A `--` alone (end of options) is always fine. These ok_* functions are invoked
# indirectly by check_flags.
ok_git_log() {
  case "$1" in
    -- | -n | -n[0-9]* | -[0-9] | -[0-9][0-9]* | --oneline | --graph | --decorate | --no-decorate | \
      --stat | --shortstat | --numstat | --name-only | --name-status | --summary | -p | --patch | \
      --no-patch | -s | --abbrev-commit | --no-abbrev-commit | --no-merges | --merges | --first-parent | \
      --all | --branches | --tags | --remotes | --reverse | --follow | --no-color | --topo-order | \
      --date-order | --author-date-order | --left-right | --cherry-pick | --cherry-mark | \
      --ancestry-path | --simplify-by-decoration | --boundary | --full-history | --root | -m | \
      --since=* | --until=* | --after=* | --before=* | --author=* | --committer=* | --grep=* | \
      --format=* | --pretty=* | --date=* | --max-count=* | --skip=* | --diff-filter=* | \
      --color=never | --decorate=* | --abbrev=* | -S?* | -G?*) return 0 ;;
  esac
  return 1
}
ok_git_show() {
  case "$1" in
    -- | --stat | --shortstat | --numstat | --name-only | --name-status | --summary | --oneline | \
      -s | -p | --patch | --no-patch | --abbrev-commit | --no-color | --color=never | -U[0-9]* | \
      --unified=* | --format=* | --pretty=* | --date=* | --diff-filter=* | --first-parent | -m) return 0 ;;
  esac
  return 1
}
ok_git_diff() {
  case "$1" in
    -- | --stat | --shortstat | --numstat | --name-only | --name-status | --summary | --raw | \
      --cached | --staged | -p | --patch | --no-patch | --no-color | --color=never | -U[0-9]* | \
      --unified=* | -w | --ignore-all-space | --ignore-space-change | -b | --diff-filter=* | \
      --check | --minimal | --patience | --histogram | -M | --find-renames | --no-renames | \
      --exit-code | --quiet | --no-index | --merge-base | -R | --word-diff | --abbrev | \
      --abbrev=* | --full-index | --relative | --text | -a) return 0 ;;
  esac
  return 1
}
ok_git_status() {
  case "$1" in
    -- | -s | -sb | -bs | --short | -b | --branch | --porcelain | --porcelain=* | --long | --ignored | \
      --untracked-files=* | -u | --no-renames | --show-stash | -z) return 0 ;;
  esac
  return 1
}
ok_git_revparse() {
  case "$1" in
    -- | --abbrev-ref | --abbrev-ref=* | --short | --short=* | --verify | -q | --quiet | \
      --show-toplevel | --git-dir | --git-common-dir | --is-inside-work-tree | --show-prefix | \
      --show-cdup | --symbolic | --symbolic-full-name | --is-bare-repository | --absolute-git-dir | \
      --is-shallow-repository | --sq-quote | --default | --not) return 0 ;;
  esac
  return 1
}
ok_git_mergebase() {
  case "$1" in
    -- | --is-ancestor | --fork-point | --octopus | --all | --independent) return 0 ;;
  esac
  return 1
}
ok_git_lsfiles() {
  case "$1" in
    -- | -c | --cached | -o | --others | -m | --modified | -d | --deleted | -s | --stage | -u | \
      --unmerged | --exclude-standard | --error-unmatch | --full-name | --eol | -z | -t | -v | \
      --directory | --no-empty-directory | --killed) return 0 ;;
  esac
  return 1
}
ok_git_catfile() {
  case "$1" in
    -- | -t | -s | -e | -p | --batch | --batch=* | --batch-check | --batch-check=* | \
      --allow-unknown-type | --buffer | --follow-symlinks | -Z) return 0 ;;
  esac
  return 1
}
ok_git_taglist() {
  case "$1" in
    -l | --list | --contains | --contains=* | --no-contains | --points-at | --points-at=* | \
      --merged | --merged=* | --no-merged | --no-merged=* | --sort | --sort=* | --format=* | \
      -n | -n[0-9]* | --column | --no-column | --ignore-case | -i) return 0 ;;
  esac
  return 1
}
ok_gh_pr() {
  case "$1" in
    --json | --json=* | --jq | --jq=* | -q | --repo | --repo=* | -R | --comments | --state | \
      --state=* | -s | --limit | --limit=* | -L | --label | --label=* | -l | --author | \
      --author=* | -A | --base | --base=* | -B | --head | --head=* | -H | --search | \
      --search=* | -S | --required) return 0 ;;
  esac
  return 1
}

case "$progl" in
  git | git.exe)
    case "$subl" in
      log) check_flags 2 ok_git_log ;;
      show) check_flags 2 ok_git_show ;;
      diff) check_flags 2 ok_git_diff ;;
      status) check_flags 2 ok_git_status ;;
      rev-parse) check_flags 2 ok_git_revparse ;;
      merge-base) check_flags 2 ok_git_mergebase ;;
      ls-files) check_flags 2 ok_git_lsfiles ;;
      cat-file) check_flags 2 ok_git_catfile ;;
      tag)
        # Listing only: -l/--list must be the first argument and every
        # other dashed argument must be a listing option.
        case "${tok[2]:-}" in
          -l | --list) ;;
          *) deny "'git tag' is allowed only as a listing (-l / --list)." ;;
        esac
        check_flags 2 ok_git_taglist
        ;;
      *)
        deny "'git ${sub:-<none>}' is not on the read-only list."
        ;;
    esac
    ;;
  gh | gh.exe)
    case "$subl" in
      pr)
        case "${tok[2]:-}" in
          view | list | checks) ;;
          *) deny "'gh pr ${tok[2]:-<none>}' is not on the read-only list." ;;
        esac
        check_flags 3 ok_gh_pr
        ;;
      api)
        # A path plus --jq/-q, --paginate, and -H/--header X-GitHub-Api-Version:*.
        # No method flag exists in this list, so the request is gh's default, GET;
        # no body field or --input, so it never becomes a POST.
        expect=""
        for ((i = 2; i < n; i++)); do
          t="${tok[$i]}"
          if [ -n "$expect" ]; then
            if [ "$expect" = header ]; then
              [[ $t == X-GitHub-Api-Version:* ]] || deny "'gh api' accepts only an X-GitHub-Api-Version header."
            fi
            expect=""
            continue
          fi
          case "$t" in
            --paginate) ;;
            -q | --jq) expect=jq ;;
            --jq=*) ;;
            -H | --header) expect=header ;;
            --header=X-GitHub-Api-Version:*) ;;
            -*) deny "'gh api $t' is not on the allowlist (path, --jq/-q, --paginate, -H X-GitHub-Api-Version only)." ;;
            graphql | */graphql) deny "'gh api graphql' is a POST." ;;
          esac
        done
        [ -z "$expect" ] || deny "'gh api' ends with a flag that is missing its value."
        ;;
      *)
        deny "'gh ${sub:-<none>}' is not on the read-only list."
        ;;
    esac
    ;;
  python | python3 | py | python.exe | python3.exe | py.exe)
    script="$sub"
    # Exactly scripts/<name>.py with a plain name: no ./, no .., no extra dirs.
    if ! [[ $script =~ ^scripts/(validate_|check_|verify_release_evidence)[A-Za-z0-9_]*\.py$ ]]; then
      deny "only python scripts/{validate_,check_,verify_release_evidence}*.py may run."
    fi
    case "$script" in
      scripts/check_build.py | scripts/validate_gcp_credentials.py | scripts/validate_convergence.py)
        deny "'$script' reaches GCP or trains; it is not a read-only check."
        ;;
    esac
    # argparse accepts any unambiguous prefix of a long option, so a denylist
    # of exact spellings is not enough: deny every prefix of a writing option.
    for ((i = 2; i < n; i++)); do
      t="${tok[$i]}"
      case "$t" in
        --?*)
          name="${t%%=*}"
          for w in --out --output --write --fix --update --apply --execute; do
            if [[ $w == "$name"* ]]; then
              deny "'$t' could make a checker write."
            fi
          done
          ;;
      esac
    done
    ;;
  *)
    deny "'${prog}' is not on the read-only list."
    ;;
esac

exit 0
