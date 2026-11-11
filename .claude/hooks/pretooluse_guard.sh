#!/usr/bin/env bash
# PreToolUse guard for the Bash and PowerShell tools (matcher `Bash|PowerShell`).
#
# An ACCIDENT NET, not a control (PROJECT_CHARTER.md S4.6, plan S3.6): the
# control is identity -- agents act as a machine user with no ruleset bypass.
# A pre-approved interpreter (`python -c`, `node -e`) can still reach the GitHub
# API directly; nothing here can stop that, and nothing here claims to.
#
# DENIES (permissionDecision "deny"):
#   * git commit / git push while checked out on a protected branch;
#   * git push whose destination is a protected branch, in any refspec form
#     (`qa`, `HEAD:qa`, `HEAD:refs/heads/qa`, `'qa'`), deletes one
#     (`:dev`, `--delete dev`, `-d`), or pushes tags (`refs/tags/*`, `--tags`,
#     `--follow-tags`) or everything (`--all`, `--mirror`);
#   * force-pushes (`--force`, `-f`, a `+refspec`); `--force-with-lease` is allowed;
#   * `git tag` unless every argument is a listing option
#     (`-l`, `--list`, `--contains`, `--points-at`, `--merged`, `--sort`,
#     `--format`, `-n`, ...); `git fetch --tags` is not a tag command;
#   * `gh pr merge`, `gh release`, `gh repo edit|delete|rename|archive`, `gh ruleset`;
#   * `gh api` with `-X`/`--method`, any body field (`-f`/`-F`/`--field`/
#     `--raw-field`/`--input`, which make gh send a POST), or the `graphql` endpoint;
#   * runs above the 1,000-game cap WITHOUT `--approval_ref`;
#   * a payload no parser can read (fail closed -- see below).
#
# ASKS (permissionDecision "ask", i.e. the user is prompted) on:
#   * `--approval_ref` (recorded by the runner, never verified, so a human
#     confirms -- the charter amendment's ruling), including over-cap runs;
#   * `--preset production`;
#   * `--config` on a game runner (the guard cannot price a config file);
#   * an UNKNOWN game runner: shaped like one (`scripts/run_*.py`, a module
#     named like a self-play/arena/training worker, ...) but not on the list of
#     runners that price themselves.
#   A deny anywhere in the command wins over an ask.
#
# Hook output schema (Claude Code hooks reference, PreToolUse decision control):
#   {"hookSpecificOutput":{"hookEventName":"PreToolUse",
#     "permissionDecision":"deny"|"ask","permissionDecisionReason":"..."}}
# printed on stdout with exit 0. Silence + exit 0 means "no opinion" (normal
# permission handling continues). Deny/ask permission RULES in settings.json
# still apply on top of this.
#
# Unwrapping: `pwsh -c`, `powershell -Command`, `cmd /c`, `bash -c`/`sh -c`,
# `iex`/`Invoke-Expression` and the `&` call operator are stripped before
# matching, so `pwsh -c "git push origin HEAD:qa"` is judged as the push it is.
#
# Protected branches: CIV6AZ_PROTECTED_BRANCHES (space-separated), plus the
# default branch (CIV6AZ_DEFAULT_BRANCH -> origin/HEAD -> `main`), plus
# `dev qa main` always. The LIST drives protection, not the default: setting
# CIV6AZ_DEFAULT_BRANCH=dev before the rename must not unprotect the old
# default, which is why the shipped settings list it explicitly.
#
# Matching is anchored to the START of each command segment, so merely
# *mentioning* a trigger inside an argument (`echo 'gh pr merge 1'`, a test
# fixture) does not trip it. Heuristic, not a sandbox.
#
# Performance: bash builtins (`[[ =~ ]]`, `read -ra`) rather than one
# grep/sed process per check. Process creation costs tens of milliseconds on
# Windows and this hook runs before every shell tool call, under a 10 s timeout.
# Portable to bash 3.2 (macOS): no `${x,,}`, and empty arrays are guarded.
set -euo pipefail
# No globbing anywhere: the guard splits untrusted command text on whitespace,
# and `git tag -l v2*` must stay the literal `v2*`, not a file listing.
set -f

# Program names and flags are matched case-insensitively: Windows resolves
# `GIT`, `Git.exe` and `gh.EXE` to the same binaries. Erring this way only
# ever adds denies (`HEAD:QA` is treated as `qa`).
shopt -s nocasematch

decided=0

# The reason is untrusted text (it can quote an argument), so every control
# character is replaced before it reaches the JSON: a raw \x01 made the output
# invalid JSON, which Claude Code ignores -- a fail-open.
emit() {
  local reason="$2"
  reason="${reason//[[:cntrl:]]/ }"
  reason="${reason//\\/\\\\}"
  reason="${reason//\"/\\\"}"
  decided=1
  printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"%s","permissionDecisionReason":"%s"}}' \
    "$1" "$reason"
}

deny() {
  emit deny "$1"
  exit 0
}

# Any unexpected failure (set -e, set -u, a bad test) must become a DENY with
# exit 0. A non-zero exit is a non-blocking error to Claude Code: the tool call
# would simply run. The trap is not inherited by `$(...)` subshells, so a
# failing parser attempt below cannot trigger it.
# shellcheck disable=SC2329  # invoked by the EXIT trap
on_exit() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "$decided" -eq 0 ]; then
    decided=1
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"%s"}}' \
      "PreToolUse guard failed unexpectedly (exit ${rc}); denying rather than letting the command run unchecked."
    exit 0
  fi
}
trap on_exit EXIT

# Test-only fault injection: proves the trap above turns a crash into a deny.
# It can only make the guard deny, never allow, so it is safe to ship.
if [ -n "${CIV6AZ_GUARD_SELFTEST_FAIL:-}" ]; then
  false
fi

input="$(cat)"

# --- payload extraction ------------------------------------------------------
# jq, else python3, else python. The previous version swallowed every parser
# error into `|| true`, so a parser that was on PATH but FAILED (for example
# the Windows Store `python3` alias on a machine without Store Python, which
# exits non-zero) left the command empty and the guard ALLOWED everything --
# found by reading the code, not observed here. Now a parser has to exit
# 0 to count; if none does, the guard cannot see what it guards and denies.
# A parser that ran and found no command means a non-shell payload: allow.
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
  deny "CLAUDE.md S3.2 guard cannot inspect this command: no working jq or python could parse the hook payload, so the prohibited-action checks cannot run. Install jq or python3 rather than proceeding unguarded."
fi
cmd="${cmd//$'\r'/}"
[ -z "$cmd" ] && exit 0

# --- segmentation ------------------------------------------------------------
# Split on shell / PowerShell / cmd separators. A single `&` covers bash
# backgrounding, cmd's command separator AND PowerShell's call operator
# (`& git push ...` leaves `git push ...` as its own segment). Brackets and
# backticks split subshells, script blocks and command substitutions.
#
# A backtick means two different things: bash command substitution
# (`` echo `git push ...` ``) and PowerShell's escape character
# (`` g`it push origin HEAD:`qa ``, which PowerShell runs as `git push origin
# HEAD:qa`). Rather than trust tool_name to pick one reading, a command
# containing a backtick is checked under BOTH: split on it, and with every
# backtick deleted. A deny under either reading denies.
split_segments() {
  local s="$1" sep
  s="${s//&&/$'\n'}"
  s="${s//||/$'\n'}"
  for sep in ';' '|' '&' '(' ')' '{' '}' '`'; do
    s="${s//"$sep"/$'\n'}"
  done
  SEGS="$s"
}
split_segments "$cmd"
segs="$SEGS"
if [[ $cmd == *'`'* ]]; then
  split_segments "${cmd//\`/}"
  segs="${segs}"$'\n'"${SEGS}"
fi

# --- protected branches (resolved lazily: git is a process spawn) ------------
resolved=0
branch=""
default_branch=""
protected=" dev qa main "
protected_display="dev qa main"

resolve_branches() {
  [ "$resolved" -eq 1 ] && return 0
  resolved=1
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  default_branch="${CIV6AZ_DEFAULT_BRANCH:-}"
  if [ -z "$default_branch" ]; then
    default_branch="$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null || true)"
    default_branch="${default_branch#origin/}"
  fi
  [ -z "$default_branch" ] && default_branch="main"
  local b out=" "
  # Unquoted on purpose: the env var is a space-separated list (noglob is set).
  for b in dev qa main $default_branch ${CIV6AZ_PROTECTED_BRANCHES:-}; do
    case "$out" in
      *" $b "*) ;;
      *) out="${out}${b} " ;;
    esac
  done
  protected="$out"
  protected_display="${out# }"
  protected_display="${protected_display% }"
}

is_protected() {
  resolve_branches
  local name="${1#refs/heads/}"
  [ -n "$name" ] || return 1
  case "$protected" in
    *" $name "*) return 0 ;;
  esac
  return 1
}

# Sets UQ -- a global, not stdout: every `$(...)` is a fork, and forks are slow
# under MSYS/Cygwin bash on Windows.
# Removes EVERY quote character, not just those at the ends: the shell
# concatenates `HEAD:"qa"` and `HEAD:'qa'` into `HEAD:qa`, and stripping only
# the ends let those through to the `git push` allow rule.
unquote() {
  UQ="${1//[\"\']/}"
}

# --- unwrapping --------------------------------------------------------------
re_assign='^[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+'
re_lead_junk='^[[:space:]"'"'"'{(]+'
# A quoted first word, spaces allowed: PowerShell's
# `& 'C:\Program Files\Git\cmd\git.exe' push ...` names its program this way.
re_q1="^[[:space:]]*'([^']*)'(.*)$"
re_q2='^[[:space:]]*"([^"]*)"(.*)$'
re_known_prog='^(git|gh|python[0-9.]*|py|make|bash|sh|zsh|dash|pwsh|powershell|cmd)(\.exe)?$'
re_path='([^[:space:]]*[/\\])?'
re_pwsh="^${re_path}(pwsh|powershell)(\\.exe)?([[:space:]]+-[A-Za-z]+([[:space:]]+[A-Za-z0-9]+)?)*[[:space:]]+-(c|command)([[:space:]]+|$)"
re_pwsh_enc="^${re_path}(pwsh|powershell)(\\.exe)?([[:space:]]+-[A-Za-z]+([[:space:]]+[A-Za-z0-9]+)?)*[[:space:]]+-(e|ec|en|enc|encodedcommand)([[:space:]]+|$)"
re_cmd="^${re_path}cmd(\\.exe)?([[:space:]]+/[A-Za-z])*[[:space:]]+/[ck]([[:space:]]+|$)"
re_sh="^${re_path}(bash|sh|zsh|dash)(\\.exe)?([[:space:]]+--?[A-Za-z-]+)*[[:space:]]+-[A-Za-z]*c[A-Za-z]*([[:space:]]+|$)"
re_iex='^(iex|invoke-expression)([[:space:]]+|$)'
# Launchers that run their argument as the command (`env git push ...`,
# `sudo -u x git push ...`), with their own options and `VAR=val` operands.
re_launcher="^${re_path}(env|command|builtin|exec|sudo|doas|nohup|nice|time|timeout)(\\.exe)?(([[:space:]]+-[ugnCk][[:space:]]+[^-[:space:]][^[:space:]]*)|([[:space:]]+-[^[:space:]]*)|([[:space:]]+[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*)|([[:space:]]+[0-9]+[smhd]?))*[[:space:]]+"

asks=""
add_ask() {
  case "$asks" in *"$1"*) return 0 ;; esac
  asks="${asks:+$asks; }$1"
}

# Normalise one segment: drop leading whitespace/quotes, env assignments and
# any wrapper, repeatedly, until nothing changes (nested wrappers unwrap too).
# Sets NS; runs in the main shell so add_ask() is not lost in a subshell.
# The stripped assignments are kept in NS_ASSIGN: the game cap reads the
# env-var form (`GAMES_PER_WORKER=2000 make ...`) from them.
normalise() {
  local s="$1" prev="" n=0
  NS_ASSIGN=""
  while [ "$s" != "$prev" ] && [ "$n" -lt 8 ]; do
    prev="$s"
    n=$((n + 1))
    # Quoted first word -> its basename (a quoted path to git.exe becomes
    # `git.exe`). Must run before the leading-quote strip below, which would
    # otherwise cut `'C:\Program` from `Files\...\git.exe'`.
    # Only when the basename IS a program this guard knows: a quoted whole
    # command (`"git push origin refs/heads/qa"`) must keep its slashes.
    if [[ $s =~ $re_q1 ]] || [[ $s =~ $re_q2 ]]; then
      local inner="${BASH_REMATCH[1]}" rest="${BASH_REMATCH[2]}"
      if [[ ${inner##*[/\\]} =~ $re_known_prog ]]; then
        s="${inner##*[/\\]}${rest}"
      else
        s="${inner}${rest}"
      fi
    fi
    if [[ $s =~ $re_lead_junk ]]; then s="${s:${#BASH_REMATCH[0]}}"; fi
    while [[ $s =~ $re_assign ]]; do
      NS_ASSIGN="${NS_ASSIGN} ${BASH_REMATCH[0]}"
      s="${s:${#BASH_REMATCH[0]}}"
    done
    if [[ $s =~ $re_pwsh_enc ]]; then
      add_ask "an encoded PowerShell command cannot be inspected by the guard"
    fi
    if [[ $s =~ $re_pwsh ]] || [[ $s =~ $re_cmd ]] || [[ $s =~ $re_sh ]] \
       || [[ $s =~ $re_iex ]] || [[ $s =~ $re_launcher ]]; then
      s="${s:${#BASH_REMATCH[0]}}"
    fi
  done
  NS="$s"
}

# --- runner classification ---------------------------------------------------
# Runners that price themselves (plan T5) or are priced by the run-card driver.
known_modules=" alpha_zero.training_civ6 alpha_zero.orchestrator alpha_zero.orchestrator.__main__ alpha_zero.evaluation.benchmark "
known_scripts=" run_baseline_eval run_encoder_ab run_chess_training run_enhanced_training run_muzero_training run_sweep post_run_evaluation validate_convergence benchmark_batch_selfplay local_muzero_launcher run_card run_civ6_ai benchmark_suite "
# Shaped like runners but play no games (or only read old runs).
exempt_scripts=" run_llm_eval run_tests_ipv4 run_cert_check compare_training_runs analyse_training_run summarize_training_log "
re_runner_script='^(run_|launch_|benchmark_|train_)|selfplay|self_play|arena|tournament|launcher|_training'
re_runner_module='train|selfplay|self_play|arena|tournament|launcher|coordinator|orchestrat|worker|benchmark'

re_python='^(python[0-9.]*|py)(\.exe)?$'
# The count is the LAST group of each pattern; every occurrence is read and the
# largest one judged (`--num_games 10 --games-per-iteration 5000` is 5000).
re_games='--((num|n|eval)[_-])?games([_-]per[_-](worker|iteration))?[=[:space:]]+([0-9]+)'
re_games_env='(^|[[:space:]])(NUM_GAMES|GAMES|GAMES_PER_WORKER)=([0-9]+)'
re_runnerish='^(python[0-9.]*|py|make|bash|sh)(\.exe)?$|^(\./)?scripts/'

# --- per-tool checks ---------------------------------------------------------
check_git() {
  # $@ = tokens after `git`
  local -a a=("$@")
  local i=0 n=${#a[@]} sub=""
  while [ "$i" -lt "$n" ]; do
    case "${a[$i]}" in
      -C|-c|--git-dir|--work-tree|--namespace|--config-env|--super-prefix) i=$((i + 2)) ;;
      -*) i=$((i + 1)) ;;
      *) unquote "${a[$i]}"; sub="$UQ"; i=$((i + 1)); break ;;
    esac
  done
  local -a args=()
  while [ "$i" -lt "$n" ]; do
    unquote "${a[$i]}"
    args+=("$UQ")
    i=$((i + 1))
  done

  case "$sub" in
    commit|push)
      resolve_branches
      if [ -n "$branch" ] && is_protected "$branch"; then
        deny "CLAUDE.md S3.2: direct commits/pushes to ${branch} (a protected branch) are prohibited - use a feature branch and a PR."
      fi
      ;;
  esac

  if [ "$sub" = "push" ]; then
    local t dst
    for t in ${args[@]+"${args[@]}"}; do
      case "$t" in
        --force-with-lease*|--force-if-includes|--no-force-if-includes) ;;
        --force|--force=*)
          deny "CLAUDE.md S3.2: force-pushes are prohibited (use --force-with-lease if truly needed)." ;;
        --tags|--follow-tags)
          deny "Branching policy: pushing tags is maintainer-only (${t}); release tags are created by the maintainer's release script." ;;
        --all|--branches|--mirror)
          deny "Branching policy: git push ${t} would push protected branches; push one feature branch instead." ;;
        --*) ;;
        -*)
          case "$t" in
            -*f*) deny "CLAUDE.md S3.2: force-pushes are prohibited (use --force-with-lease if truly needed)." ;;
          esac
          ;;
        +*)
          deny "CLAUDE.md S3.2: force-pushes are prohibited (a +refspec forces the update)." ;;
        tag)
          # `git push origin tag v9` is git's spelling of refs/tags/v9.
          deny "Branching policy: pushing tags is maintainer-only (git push ... tag <name>)." ;;
        *)
          dst="${t##*:}"
          # bash drops the backslash in `q\a`, and a ref name cannot contain
          # one, so deleting it loses nothing; `refs//heads/qa` collapses.
          dst="${dst//\\/}"
          while [[ $dst == *//* ]]; do dst="${dst//\/\//\/}"; done
          case "$dst" in
            refs/tags/*)
              deny "Branching policy: pushing to refs/tags/* is maintainer-only; release tags are created by the maintainer's release script." ;;
          esac
          if is_protected "$dst"; then
            deny "CLAUDE.md S3.2: pushing to or deleting ${dst#refs/heads/} (protected: ${protected_display}) is prohibited - use a feature branch and a PR."
          fi
          ;;
      esac
    done
  fi

  if [ "$sub" = "tag" ]; then
    local list_mode=0 t j m=${#args[@]} expect_value=0 bad=""
    for t in ${args[@]+"${args[@]}"}; do
      case "$t" in
        -l|--list|-n*|--contains*|--no-contains*|--points-at*|--merged*|--no-merged*) list_mode=1 ;;
      esac
    done
    j=0
    while [ "$j" -lt "$m" ]; do
      t="${args[$j]}"
      j=$((j + 1))
      if [ "$expect_value" -eq 1 ]; then expect_value=0; continue; fi
      case "$t" in
        -l|--list|-n*|-i|--ignore-case|--no-column|--omit-empty) ;;
        --contains=*|--no-contains=*|--points-at=*|--merged=*|--no-merged=*) ;;
        --sort=*|--format=*|--column=*|--color=*|--column|--color) ;;
        --contains|--no-contains|--merged|--no-merged) ;;
        --points-at|--sort|--format) expect_value=1 ;;
        -*) bad="$t"; break ;;
        *) [ "$list_mode" -eq 1 ] || { bad="$t"; break; } ;;
      esac
    done
    if [ -n "$bad" ]; then
      deny "Branching policy: git tag may only list tags here (offending argument: ${bad}); creating, moving or deleting tags is maintainer-only."
    fi
  fi
  return 0
}

check_gh() {
  # Positional tokens of the gh command line, flag values skipped.
  local -a a=("$@") pos=()
  local i=0 n=$# t
  while [ "$i" -lt "$n" ]; do
    unquote "${a[$i]}"
    t="$UQ"
    i=$((i + 1))
    case "$t" in
      -R|--repo|-H|--header|-q|--jq|-t|--template|--hostname|--cache|-p|--preview|-X|--method|-f|-F|--field|--raw-field|--input)
        i=$((i + 1)) ;;
      -*) ;;
      *) pos+=("$t") ;;
    esac
  done
  local p0="${pos[0]:-}" p1="${pos[1]:-}"
  case "$p0" in
    release|ruleset)
      deny "Branching policy: gh ${p0} is maintainer-only (releases and rulesets are not agent operations)." ;;
    repo)
      case "$p1" in
        edit|delete|rename|archive|unarchive)
          deny "Branching policy: gh repo ${p1} changes repository settings and is maintainer-only." ;;
      esac
      ;;
    pr)
      if [ "$p1" = "merge" ]; then
        deny "Branching policy: gh pr merge is maintainer-only; open the PR and let the maintainer merge it."
      fi
      ;;
    api)
      for t in "$@"; do
        unquote "$t"
        t="$UQ"
        case "$t" in
          -X*|--method|--method=*|-f*|-F*|--field|--field=*|--raw-field|--raw-field=*|--input|--input=*)
            deny "Branching policy: gh api with ${t%%=*} can write (a method or body field turns the call into a POST/PATCH/DELETE); only read-only GET calls are allowed." ;;
        esac
      done
      if [ "$p1" = "graphql" ]; then
        deny "Branching policy: gh api graphql can run mutations and is not allowed; use a read-only REST GET."
      fi
      ;;
  esac
  return 0
}

check_runner() {
  # $1 = the whole normalised segment; remaining = tokens.
  local seg="$1"
  shift
  local -a a=("$@")
  local prog
  unquote "${a[0]:-}"
  prog="${UQ##*[/\\]}"
  local has_help=0 has_approval=0 has_config=0 preset_prod=0 t prev=""
  for t in ${a[@]+"${a[@]}"}; do
    unquote "$t"
    t="$UQ"
    case "$t" in
      -h|--help) has_help=1 ;;
      --approval_ref|--approval_ref=*|--approval-ref|--approval-ref=*) has_approval=1 ;;
      --config|--config=*) has_config=1 ;;
      --preset=production) preset_prod=1 ;;
      production) if [ "$prev" = "--preset" ]; then preset_prod=1; fi ;;
    esac
    prev="$t"
  done

  # Game cap. An explicit count above 1000 is denied unless --approval_ref is
  # present, in which case a human is asked (the ref is never verified).
  #
  # Numbers are compared as STRINGS of digits, never with `[ -gt ]`: an integer
  # that overflows made `[` error inside the `if`, which reads as false, so
  # `--num_games 99999999999999999999` was allowed. After stripping leading
  # zeros, more than 4 digits is over the cap by construction.
  local games="" over=0 hay rest v
  for hay in "$seg" "${NS_ASSIGN} ${seg}"; do
    rest="$hay"
    while [[ $rest =~ $re_games ]] || [[ $rest =~ $re_games_env ]]; do
      v="${BASH_REMATCH[${#BASH_REMATCH[@]}-1]}"
      rest="${rest#*"${BASH_REMATCH[0]}"}"
      v="${v#"${v%%[!0]*}"}"
      [ -z "$v" ] && v=0
      if [ "${#v}" -gt 4 ] || { [ "${#v}" -eq 4 ] && [ "$v" -gt 1000 ]; }; then
        over=1
      fi
      if [ -z "$games" ] || [ "${#v}" -gt "${#games}" ] \
         || { [ "${#v}" -eq "${#games}" ] && [ "${#v}" -le 9 ] && [ "$v" -gt "$games" ]; }; then
        games="$v"
      fi
    done
  done
  if [ "$over" -eq 1 ]; then
    if [ "$has_approval" -eq 0 ]; then
      deny "CLAUDE.md S3.2: runs with >1000 games require confirmation (requested ${games}); pass --approval_ref <issue> to ask the maintainer."
    fi
  fi
  if [ "$has_approval" -eq 1 ]; then
    add_ask "--approval_ref is recorded, not verified: confirm the maintainer approved this run${games:+ (${games} games)}"
  fi
  if [ "$preset_prod" -eq 1 ]; then
    add_ask "--preset production runs an unpriced production-size job"
  fi

  # Classify python invocations as known / unknown game runners.
  [[ $prog =~ $re_python ]] || return 0
  local i=1 n=${#a[@]} module="" script=""
  while [ "$i" -lt "$n" ]; do
    unquote "${a[$i]}"
    t="$UQ"
    i=$((i + 1))
    case "$t" in
      -c) return 0 ;;
      -m) unquote "${a[$i]:-}"; module="$UQ"; break ;;
      -m*) module="${t#-m}"; break ;;
      -X|-W) i=$((i + 1)) ;;
      -*) ;;
      *) script="$t"; break ;;
    esac
  done

  local kind="" name=""
  if [ -n "$module" ]; then
    name="$module"
    case "$known_modules" in
      *" $module "*) kind=known ;;
      *) if [[ $module =~ $re_runner_module ]]; then kind=unknown; fi ;;
    esac
  elif [ -n "$script" ]; then
    name="${script##*[/\\]}"
    name="${name%.py}"
    if [[ $script == *.py ]]; then
      case "$known_scripts" in
        *" $name "*) kind=known ;;
        *)
          case "$exempt_scripts" in
            *" $name "*) ;;
            *) if [[ $name =~ $re_runner_script ]]; then kind=unknown; fi ;;
          esac
          ;;
      esac
    fi
  fi
  if [ -z "$kind" ] || [ "$has_help" -eq 1 ]; then
    return 0
  fi
  if [ "$has_config" -eq 1 ]; then
    add_ask "${name} --config cannot be priced by the guard (the game count lives in the config file)"
  fi
  if [ "$kind" = "unknown" ]; then
    add_ask "${name} looks like a game runner but is not on the list of runners that price themselves (1000-game cap)"
  fi
  return 0
}

# --- main loop ---------------------------------------------------------------
while IFS= read -r raw; do
  normalise "$raw"
  seg="$NS"
  [ -z "$seg" ] && continue
  toks=()
  read -ra toks <<<"$seg" || true
  [ "${#toks[@]}" -eq 0 ] && continue
  unquote "${toks[0]}"
  prog="$UQ"
  base="${prog##*[/\\]}"
  case "$base" in
    git|git.exe) check_git ${toks[@]+"${toks[@]:1}"} ;;
    gh|gh.exe) check_gh ${toks[@]+"${toks[@]:1}"} ;;
  esac
  if [[ $base =~ $re_runnerish ]] || [[ $prog =~ $re_runnerish ]]; then
    check_runner "$seg" "${toks[@]}"
  fi
done <<EOF
$segs
EOF

if [ -n "$asks" ]; then
  emit ask "PreToolUse guard: ${asks}."
fi
exit 0
