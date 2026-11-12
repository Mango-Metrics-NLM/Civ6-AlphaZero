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
#   gh api            (a RELATIVE api path + --jq/-q, --paginate, -H X-GitHub-Api-Version:*
#                      only; never a URL, host or scheme)
#   python scripts/<one of ten vetted validate_/check_/verify_ scripts>.py
#
# Two modes, chosen by the first argument the hook command passes:
#   (none)  the read-only roles
#   --qa    sdlc-qa-sdet: everything above PLUS `python -m pytest <paths under tests/>`
#           with a short list of safe flags (no `-c`, no plugins, no rootdir) and an optional leading
#           `OMP_NUM_THREADS=<digits>`. RESIDUAL RISK, stated plainly: pytest executes test
#           code and conftest, so a test can write anywhere. This is an accident net, not a
#           sandbox; the independent review reads the tests it ran, and identity is the control.
#           `python -c`, `git add|commit|push` and the rest stay denied.
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
# too; a read-only role reads the whole output / uses --pretty=oneline instead.
#
# Options are an ALLOWLIST, not a denylist: every dashed argument to a git read
# subcommand and to `gh pr` must be on that subcommand's short list of safe
# flags, and `gh api` takes only a path plus --jq/-q, --paginate and
# -H/--header X-GitHub-Api-Version:*. Anything else (--output, --ext-diff,
# --textconv, --filters, -X, -f, --input, --web, ...) is denied by default. The
# request method of `gh api` cannot be changed, so it is gh's default, GET.
# Scripts are an EXACT, case-sensitive allowlist of ten filenames, not a glob. The
# three scripts a glob used to admit are deliberately absent -- check_build.py (Cloud
# Build API with ADC), validate_gcp_credentials.py (GCS/Vertex/Logging) and
# validate_convergence.py (an uncapped training loop) -- and a casefold match to one
# of them is denied by name as well, because NTFS and default APFS open
# `check_BUILD.py` as the real file. check_branch_direction.py and
# verify_release_evidence.py write GITHUB_STEP_SUMMARY/GITHUB_OUTPUT when those
# variables are set; the guard cannot set them, so that is unreachable locally.
# A script option is denied if it contains out/write/fix/update/apply/exec/baseline/
# save/dump/create/delete/force/overwrite (argparse also accepts abbreviations), and
# every short option except -h.
#
# `git status`/`git diff` refresh the index stat cache and so can briefly hold
# .git/index.lock; run read-only roles with GIT_OPTIONAL_LOCKS=0 if a developer in
# the same worktree ever sees `index.lock exists`.
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
  emit deny "read-only role shell guard: $1 Allowed (one plain command, no quotes, pipes, redirection or globs): git log|show|diff|status|rev-parse|merge-base|ls-files|cat-file|tag -l; gh pr view|list|checks; gh api <one relative path> (GET only); python scripts/<one of ten vetted validate_/check_/verify_release_evidence checkers, exact options>.py; and, only for sdlc-qa-sdet (--qa), python -m pytest <paths under tests/> with safe flags and an optional OMP_NUM_THREADS=<digits> prefix. Return the text you wanted to write to your caller instead."
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

# Mode: the only accepted argument is --qa.
qa=0
case "${1:-}" in
  "") ;;
  --qa) qa=1 ;;
  *) printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"read-only role shell guard: unknown mode argument; denying."}}'; decided=1; exit 0 ;;
esac

input="$(cat)"

# jq, else python3, else python; a parser must EXIT 0 to count (see
# pretooluse_guard.sh for the failure this prevents). No parser -> deny.
# A payload whose tool_input is not an object, or whose command is missing or not a
# string, is NOT "nothing to run": the extractor exits non-zero, which counts as a
# parse failure and ends in a deny.
PY_EXTRACT='import json,sys
d=json.load(sys.stdin)
t=d.get("tool_input") if isinstance(d,dict) else None
c=t.get("command") if isinstance(t,dict) else None
if not isinstance(c,str):
    sys.exit(3)
sys.stdout.buffer.write(c.encode("utf-8"))'

cmd=""
parsed=0
if command -v jq >/dev/null 2>&1 \
   && cmd="$(printf '%s' "$input" | jq -r 'if (.tool_input|type)=="object" and (.tool_input.command|type)=="string" then .tool_input.command else error("no string command") end' 2>/dev/null)"; then
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

# --qa only: one optional leading OMP_NUM_THREADS=<digits> (the torch spin-wait
# workaround, CLAUDE.md section 5.3). Dropped here so the command is judged without it;
# a second one, or any other VAR=value, is then an unknown program and is denied.
if [ "$qa" -eq 1 ] && [[ ${tok[0]} =~ ^OMP_NUM_THREADS=[0-9]+$ ]]; then
  tok=("${tok[@]:1}")
  n=${#tok[@]}
  [ "$n" -eq 0 ] && deny "an environment prefix with no command."
fi

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
# can reach a command. The one exception is a `gh` jq filter (`--jq`/`-q` value), which
# must be a plain dotted field path (`.name`, `.a.b`) and must not name jq's `env`:
# gh's jq can print environment variables, which would put a token in the transcript.
shape_re='^[A-Za-z0-9._/~^:+=-]+$'
jq_re='^[A-Za-z0-9._-]+$'
jq_env_re='(^|\.)env(\.|$)'
jq_ok() { [[ $1 =~ $jq_re ]] && ! [[ $1 =~ $jq_env_re ]]; }
for ((i = 0; i < n; i++)); do
  t="${tok[$i]}"
  if [ "$progl" = gh ] && [ "$i" -ge 1 ] && { [ "${tok[$((i - 1))]}" = -q ] || [ "${tok[$((i - 1))]}" = --jq ]; }; then
    jq_ok "$t" || deny "the jq filter '$t' is not a plain field path (and may not name env)."
  elif [ "$progl" = gh ] && [[ $t == --jq=* ]]; then
    jq_ok "${t#--jq=}" || deny "the jq filter '$t' is not a plain field path (and may not name env)."
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

# The ten vetted checkers, EXACT and case-sensitive. A casefold match to an excluded
# script is denied by name so the reason is clear, but the allowlist is what decides.
script_check() {
  local sl k tk name w
  sl="$(lower "$sub")"
  case "$sl" in
    scripts/check_build.py | scripts/validate_gcp_credentials.py | scripts/validate_convergence.py)
      deny "'$sub' reaches GCP or trains; it is not a read-only check."
      ;;
  esac
  case "$sub" in
    scripts/validate_agents.py | scripts/validate_claude_agents.py | scripts/validate_kb.py | \
      scripts/validate_openspec.py | scripts/validate_skills.py | scripts/check_doc_sync.py | \
      scripts/check_coverage_floors.py | scripts/check_branch_direction.py | \
      scripts/check_learning_trend.py | scripts/verify_release_evidence.py) ;;
    *) deny "'${sub:-<none>}' is not one of the vetted read-only scripts." ;;
  esac
  # Layer 1: an EXACT allowlist of long options per script, read from each script's
  # argparse definitions. argparse accepts any unambiguous abbreviation (`--wr` is
  # `--write`), so a name match on the denied option is not enough -- only exact
  # members pass. The writing options (`check_learning_trend.py --out`) are absent.
  local opts
  case "$sub" in
    scripts/validate_agents.py) opts=" --path --quiet --json --verbose --max-warnings " ;;
    scripts/validate_claude_agents.py) opts=" --path --quiet --verbose " ;;
    scripts/validate_kb.py) opts=" --git --no-git --path --quiet --json " ;;
    scripts/validate_openspec.py) opts=" --change --path --quiet --json " ;;
    scripts/validate_skills.py) opts=" --path --quiet --json " ;;
    scripts/check_doc_sync.py) opts=" --root --quiet --json " ;;
    scripts/check_coverage_floors.py) opts=" --report --floors --package --allow-empty --verbose -v " ;;
    scripts/check_branch_direction.py) opts=" " ;;
    scripts/check_learning_trend.py) opts=" --run-dir --iterations --eval --benchmark --print-protocol-sha --evaluation-md " ;;
    scripts/verify_release_evidence.py) opts=" --repo-root --repo --head --scope " ;;
    *) opts=" " ;;
  esac
  for ((k = 2; k < n; k++)); do
    tk="${tok[$k]}"
    case "$tk" in
      -*)
        name="${tk%%=*}"
        if [ "$name" != -h ] && [ "$name" != --help ] && [[ $opts != *" $name "* ]]; then
          deny "option '$tk' is not an exact, vetted option of $sub."
        fi
        # Layer 2: even an exact member must not look like a writer.
        for w in out write fix update apply exec baseline save dump create delete force overwrite; do
          if [[ $name == *"$w"* ]]; then deny "'$tk' could make a checker write."; fi
        done
        ;;
    esac
  done
  return 0
}

# qa mode: `python -m pytest <paths under tests/>` with safe flags only.
tests_path_re='^tests(/[A-Za-z0-9._-]+)*(::[A-Za-z0-9_:.-]+)?$'
tests_path_ok() { [[ $1 =~ $tests_path_re ]] && [[ $1 != *..* ]]; }
pytest_check() {
  local k=3 tk havepath=0
  while [ "$k" -lt "$n" ]; do
    tk="${tok[$k]}"
    case "$tk" in
      -q | -qq | -x | -v | -vv | -s | --lf | --ff | --co | --collect-only | --no-header | -ra | \
        --tb=short | --tb=line | --tb=no | --tb=long | --tb=auto | --tb=native | --strict-markers) ;;
      --maxfail=*) [[ $tk =~ ^--maxfail=[0-9]+$ ]] || deny "'$tk' is not a number." ;;
      --confcutdir=*) tests_path_ok "${tk#--confcutdir=}" || deny "'$tk' is not a path under tests/." ;;
      -k | -m)
        k=$((k + 1))
        [[ ${tok[$k]:-} =~ ^[A-Za-z0-9_]+$ ]] || deny "'$tk' takes one plain word."
        ;;
      -p)
        k=$((k + 1))
        [ "${tok[$k]:-}" = "no:cacheprovider" ] || deny "'-p' is allowed only as '-p no:cacheprovider'."
        ;;
      # `-c <config>` is deliberately absent: a config can set addopts, plugins and rootdir,
      # which is more reach than a safe test run needs. Without it pytest still uses the
      # repo's own configuration.
      -*) deny "pytest option '$tk' is not on the allowlist." ;;
      *)
        tests_path_ok "$tk" || deny "pytest may run paths under tests/ only; '$tk' is not one."
        havepath=1
        ;;
    esac
    k=$((k + 1))
  done
  [ "$havepath" -eq 1 ] || deny "pytest needs at least one path under tests/."
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
        npath=0
        api_path_re='^/?[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$'
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
            *)
              # The endpoint: a RELATIVE api path only. A scheme (`https:`), a port
              # or any `:` fails the charset; `//host/x` has an empty segment and
              # fails the segment pattern; `gh api` would otherwise fetch a URL from
              # any host (an egress and SSRF channel).
              [[ $t =~ $api_path_re ]] || deny "'gh api $t' is not a relative GitHub API path (no scheme, host or double slash)."
              if [[ $t == *graphql* ]]; then deny "'gh api graphql' is a POST."; fi
              npath=$((npath + 1))
              ;;
          esac
        done
        [ "$npath" -eq 1 ] || deny "'gh api' takes exactly one api path."
        [ -z "$expect" ] || deny "'gh api' ends with a flag that is missing its value."
        ;;
      *)
        deny "'gh ${sub:-<none>}' is not on the read-only list."
        ;;
    esac
    ;;
  python | python3 | py | python.exe | python3.exe | py.exe)
    if [ "$qa" -eq 1 ] && [ "$sub" = -m ] && [ "${tok[2]:-}" = pytest ]; then
      pytest_check
    else
      script_check
    fi
    ;;
  *)
    deny "'${prog}' is not on the read-only list."
    ;;
esac

exit 0
