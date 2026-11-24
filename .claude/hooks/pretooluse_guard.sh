#!/usr/bin/env bash
# PreToolUse guard for the Bash and PowerShell tools (matcher `Bash|PowerShell`).
#
# An ACCIDENT NET, not a control (PROJECT_CHARTER.md S4.6, plan S3.6): the
# control is identity -- agents act as a machine user with no ruleset bypass.
# A pre-approved interpreter (`python -c`, `node -e`) can still reach the GitHub
# API directly; nothing here can stop that, and nothing here claims to.
#
# DENIES (permissionDecision "deny"):
#   * git commit / git push on a protected branch, judged in the repository the
#     command runs in and on the branch it would be on (see "Target directory"
#     and "Branch changes" below);
#   * git push whose destination is a protected branch, in any refspec form
#     (`qa`, `HEAD:qa`, `HEAD:refs/heads/qa`, `'qa'`), deletes one
#     (`:dev`, `--delete dev`, `-d`), or pushes tags: `refs/tags/*`, `--tags`,
#     `--follow-tags`, a source git resolves to a tag in the target repository
#     (`v2.0.0` when that tag exists, `tags/v2.0.0`, `v2.0.0^{tag}`), and a
#     deletion of a name that is neither `refs/heads/<x>` nor a local branch
#     (`git push origin :v9` deletes the remote TAG v9);
#   * git push of everything: `--all`, `--branches`, `--mirror`, a refspec
#     containing `*`, the "matching" refspec `:`, and `--prune` (it deletes
#     remote branches); long options also in git's abbreviated forms (`--al`,
#     `--b`, `--m`, `--ta`, `--fol`, `--pru`); and `-c` / `--config-env`
#     setting push.followTags, push.default=matching, remote.<x>.push or
#     remote.<x>.mirror;
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
#   * git commit / git push whose repository or branch the guard cannot tell:
#     a directory or GIT_DIR given by a variable, `~`, `-` or nothing
#     (`cd "$X"`, `cd -`, a bare `cd`, `pushd`/`popd`, `GIT_DIR=$X`), a `cd`
#     after `&` or on the right of `||`, a named directory where git cannot
#     read a repository, or an earlier `git switch -` / `git switch "$B"`. It
#     neither allows that silently nor denies on a guess;
#   * a `git push` to origin of a commit that is already on a protected branch,
#     when the destination has no remote-tracking ref or one that is fully
#     merged into that same branch: usually the name of a branch whose PR was
#     merged and deleted, pushed again by mistake. A convenience net, not a
#     control: local refs only, and any failed git query means no ask.
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
# Target directory (the git rules are in _guard_git.sh, sourced below): the
# repository a git command acts on is the one git itself would use. A command
# names another one in these ways, and each is followed:
#   * a `cd` (also `chdir`, `Set-Location`, `sl`) earlier in the command. `&&`,
#     `;` and a newline carry it to the segments after it, and so does the LEFT
#     side of `||`, which runs either way. A `cd` on the RIGHT of `||` (it runs
#     only if the left failed) or before `&` (a background job in bash, the
#     next command in sequence in cmd) leaves the directory unknown. A `cd`
#     before `|` or inside `( ... )` changes only its own pipeline element or
#     subshell, so it is ignored. A `cd` inside `( ... )` followed by `&&`
#     still carries past the `)`: that can only add a deny or an ask.
#     Redirections (`>/dev/null`, `2>&1`, `&>f`) are not arguments of `cd` and
#     their `&` is not a separator;
#   * `env -C <dir>` / `env --chdir[=]<dir>`;
#   * git's own `-C <dir>`, in order (a relative one is relative to the
#     previous), and `--git-dir`, which a later absolute `-C` does not undo;
#   * GIT_DIR: as `GIT_DIR=<d> git ...`, through `env`, by `export GIT_DIR=...`
#     or PowerShell's `$env:GIT_DIR = ...` earlier in the command.
#     GIT_WORK_TREE and GIT_COMMON_DIR alone do not change which HEAD git reads.
# The default branch is read in that repository when CIV6AZ_DEFAULT_BRANCH is
# unset, and the branch is read without --short (with a tag of the same name,
# `--short` prints `heads/main`).
#
# Branch changes: an earlier `git switch <b>` / `git checkout <b>` (also `-B`,
# `-C`, `-c`, `-b`, `--orphan`) in the same command decides the branch of a
# later commit or push in that repository. A path checkout (`checkout -- f`,
# `checkout .`) or a commit is not a branch change.
#
# Shell keywords (`then`, `do`, `else`, `elif`, `if`, `while`, `until`, `!`,
# `time`) are stripped before the command word is read.
#
# Protected branches: CIV6AZ_PROTECTED_BRANCHES (space-separated), plus the
# default branch (CIV6AZ_DEFAULT_BRANCH -> origin/HEAD of the TARGET
# repository -> `main`), plus
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

# The reason is untrusted text (it can quote a branch name, a runner name or a
# ref), so the decision JSON is built by an encoder, never by hand-escaping: a
# raw \x01 or a byte that is not UTF-8 made the old output invalid JSON, which
# Claude Code ignores -- a fail-open. jq first, then python (the bytes decoded
# as utf-8 with replacement), then a fixed string that still carries the
# decision. A COPY of `json_decision` in readonly_shell_guard.sh, kept
# identical on purpose (each hook stays one self-contained file); change both.
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

# The git rules live beside this file. Sourced after the trap, so a missing or
# broken file is a crash and therefore a deny.
case "${BASH_SOURCE[0]}" in
  */*|*\\*) GUARD_DIR="${BASH_SOURCE[0]%[/\\]*}" ;;
  *) GUARD_DIR="." ;;
esac
# shellcheck source=_guard_git.sh
. "${GUARD_DIR}/_guard_git.sh"

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
  # A fixed literal, like the EXIT trap's: with no jq and no python there is no
  # encoder either, and this text holds nothing untrusted.
  decided=1
  printf '%s' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"CLAUDE.md S3.2 guard cannot inspect this command: no working jq or python could parse the hook payload, so the prohibited-action checks cannot run. Install jq or python3 rather than proceeding unguarded."}}'
  exit 0
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
#
# How a separator moves a `cd` (see "Target directory" in the header). Marker
# bytes, never in a command (they are deleted from it first):
#   NO_CARRY (end of a segment followed by `|` or `)`): a `cd` there changes
#     only its own pipeline element or subshell, so it is ignored;
#   BG (end of a segment followed by `&`): bash runs it in the background, cmd
#     runs it in sequence, so the directory after it is unknown;
#   OR_RIGHT (start of a segment after `||`): it runs only when the left side
#     failed, so a `cd` there leaves the directory unknown. The LEFT side of
#     `||` runs either way and carries like `&&`;
#   RESET (a line of its own): the second, backtick-deleted reading starts
#     from the hook's own directory again;
#   REDIR: stands in for the `&` of a redirection (`&>`, `>&`, `<&`, `2>&1`)
#     while splitting, so a redirection is never read as a separator;
#   LBR / RBR: stand in for the braces of git's revision syntax (`v9^{tag}`,
#     `HEAD@{1}`, `@{-1}`), which would otherwise split a refspec in two.
NO_CARRY=$'\x02'
RESET=$'\x03'
REDIR=$'\x04'
BG=$'\x05'
OR_RIGHT=$'\x06'
LBR=$'\x07'
RBR=$'\x08'
cmd="${cmd//[$NO_CARRY$RESET$REDIR$BG$OR_RIGHT$LBR$RBR]/}"
re_revbrace='([@^])\{([^{}[:space:]]*)\}'
split_segments() {
  local s="$1" sep
  while [[ $s =~ $re_revbrace ]]; do
    s="${s/"${BASH_REMATCH[0]}"/"${BASH_REMATCH[1]}${LBR}${BASH_REMATCH[2]}${RBR}"}"
  done
  s="${s//&>/$REDIR>}"
  s="${s//>&/>$REDIR}"
  s="${s//<&/<$REDIR}"
  s="${s//&&/$'\n'}"
  s="${s//||/$'\n'$OR_RIGHT}"
  for sep in '|' ')'; do
    s="${s//"$sep"/$NO_CARRY$'\n'}"
  done
  s="${s//&/$BG$'\n'}"
  for sep in ';' '(' '{' '}' '`'; do
    s="${s//"$sep"/$'\n'}"
  done
  s="${s//$REDIR/"&"}"   # quoted: bash 5.2 reads a bare & as the match
  s="${s//$LBR/\{}"
  SEGS="${s//$RBR/\}}"
}
split_segments "$cmd"
segs="$SEGS"
if [[ $cmd == *'`'* ]]; then
  split_segments "${cmd//\`/}"
  segs="${segs}"$'\n'"${RESET}"$'\n'"${SEGS}"
fi


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
# Shell keywords that may start a segment once the splitter has cut at `;`:
# `if x; then git push ...` leaves `then git push ...`.
re_keyword='^(then|do|else|elif|if|while|until|!)([[:space:]]+|$)'
# `env` is unwrapped by unwrap_env (its -C / --chdir move the directory and its
# VAR=val operands set GIT_DIR); the generic launcher pattern below would
# drop `--chdir <dir>` and read the directory as the command.
re_env="^${re_path}env(\\.exe)?([[:space:]]+|$)"
# Launchers that run their argument as the command (`env git push ...`,
# `sudo -u x git push ...`), with their own options and `VAR=val` operands.
re_launcher="^${re_path}(env|command|builtin|exec|sudo|doas|nohup|nice|time|timeout)(\\.exe)?(([[:space:]]+-[ugnCk][[:space:]]+[^-[:space:]][^[:space:]]*)|([[:space:]]+-[^[:space:]]*)|([[:space:]]+[A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*)|([[:space:]]+[0-9]+[smhd]?))*[[:space:]]+"

asks=""
add_ask() {
  case "$asks" in *"$1"*) return 0 ;; esac
  asks="${asks:+$asks; }$1"
}

# `env [options] [VAR=val]... command`: sets NS_ENV_DIRS (each -C/--chdir) and
# adds the assignments to NS_ASSIGN. $1 = the text after `env`; sets ENV_REST.
unwrap_env() {
  local s="$1" t want=""
  while [[ $s =~ ^[[:space:]]*([^[:space:]]+) ]]; do
    t="${BASH_REMATCH[1]}"
    if [ -n "$want" ]; then
      [ "$want" = "dir" ] && { unquote "$t"; NS_ENV_DIRS+=("$UQ"); }
      want=""
      s="${s#*"$t"}"
      continue
    fi
    case "$t" in
      -C|--chdir) want=dir ;;
      --chdir=*) unquote "${t#*=}"; NS_ENV_DIRS+=("$UQ") ;;
      -u|--unset|-S|--split-string) want=skip ;;
      -*) ;;
      [A-Za-z_]*=*) NS_ASSIGN="${NS_ASSIGN} ${t} " ;;
      *) break ;;
    esac
    s="${s#*"$t"}"
  done
  ENV_REST="$s"
}

# Normalise one segment: drop leading whitespace/quotes, env assignments and
# any wrapper, repeatedly, until nothing changes (nested wrappers unwrap too).
# Sets NS; runs in the main shell so add_ask() is not lost in a subshell.
# The stripped assignments are kept in NS_ASSIGN: the game cap reads the
# env-var form (`GAMES_PER_WORKER=2000 make ...`) from them.
normalise() {
  local s="$1" prev="" n=0
  NS_ASSIGN=""
  NS_ENV_DIRS=()
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
    if [[ $s =~ $re_keyword ]]; then s="${s:${#BASH_REMATCH[0]}}"; fi
    while [[ $s =~ $re_assign ]]; do
      NS_ASSIGN="${NS_ASSIGN} ${BASH_REMATCH[0]}"
      s="${s:${#BASH_REMATCH[0]}}"
    done
    if [[ $s =~ $re_pwsh_enc ]]; then
      add_ask "an encoded PowerShell command cannot be inspected by the guard"
    fi
    if [[ $s =~ $re_env ]]; then
      unwrap_env "${s:${#BASH_REMATCH[0]}}"
      s="$ENV_REST"
      continue
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
# A redirection is not an argument: `cd p >/dev/null 2>&1` names one directory.
# A bare operator (`>`, `2>`, `&>`) takes the next word as its target.
re_redir='^([0-9]+|&)?(<|>)'
re_redir_bare='^([0-9]+|&)?(<|<<|>|>>|<>|>\||>&|<&|&>|&>>)$'
# PowerShell: `$env:GIT_DIR = '...'` sets the variable for the rest of the command.
re_psenv='^\$env:([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*[^[:space:]])[[:space:]]*$'
while IFS= read -r raw; do
  if [ "$raw" = "$RESET" ]; then
    reset_carried_state
    continue
  fi
  # How this segment's `cd` / `export` reaches the next ones: 1 carries,
  # 0 does not (a pipeline element or subshell), 2 leaves them unknown.
  carry=1
  if [[ $raw == "$OR_RIGHT"* ]]; then carry=2; raw="${raw#"$OR_RIGHT"}"; fi
  if [[ $raw == *"$NO_CARRY" ]]; then
    [ "$carry" -eq 2 ] || carry=0
    raw="${raw%"$NO_CARRY"}"
  elif [[ $raw == *"$BG" ]]; then
    carry=2
    raw="${raw%"$BG"}"
  fi
  normalise "$raw"
  seg="$NS"
  [ -z "$seg" ] && continue
  words=()
  read -ra words <<<"$seg" || true
  toks=()
  skip_next=0
  for w in ${words[@]+"${words[@]}"}; do
    if [ "$skip_next" -eq 1 ]; then skip_next=0; continue; fi
    if [[ $w =~ $re_redir ]]; then
      [[ $w =~ $re_redir_bare ]] && skip_next=1
      continue
    fi
    toks+=("$w")
  done
  [ "${#toks[@]}" -eq 0 ] && continue
  unquote "${toks[0]}"
  prog="$UQ"
  base="${prog##*[/\\]}"
  case "$base" in
    cd|chdir|set-location|sl)
      case "$carry" in
        1) apply_cd 1 "${toks[@]}" ;;
        2) apply_cd 0 "${toks[@]}" ;;
      esac ;;
    pushd|popd|push-location|pop-location)
      [ "$carry" -eq 0 ] || apply_cd 0 "${toks[@]}" ;;
    export)
      for w in ${toks[@]+"${toks[@]:1}"}; do
        case "$w" in
          GIT_DIR=*) [ "$carry" -eq 0 ] || apply_shell_git_dir $((carry == 1)) "${w#*=}" ;;
        esac
      done ;;
  esac
  if [[ $seg =~ $re_psenv ]] && [[ ${BASH_REMATCH[1]} == GIT_DIR ]]; then
    [ "$carry" -eq 0 ] || apply_shell_git_dir $((carry == 1)) "${BASH_REMATCH[2]}"
  fi
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
