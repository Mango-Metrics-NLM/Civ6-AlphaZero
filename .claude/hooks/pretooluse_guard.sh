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
#     deletion of any `refs/*` other than `refs/heads/*`;
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
#   * an unqualified remote deletion (`:x`, `--delete x`) unless
#     `refs/remotes/<remote>/x` exists here and no tag `x` does: git matches
#     the name against every remote ref, so it may delete a TAG x (a local
#     branch x proves nothing);
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
#     read a repository, an earlier `git switch -` / `git switch "$B"` or a
#     switch in a repository it cannot tell, or a protected starting branch
#     when an earlier switch may not have taken effect (see "Branch changes").
#     It neither allows that silently nor denies on a guess;
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
# Target directory (the git rules are in _guard_git.sh, the carried state in
# _guard_state.sh, both sourced below): the
# repository a git command acts on is the one git itself would use. A command
# names another one in these ways, and each is followed:
#   * a `cd` (also `chdir`, `Set-Location`, `sl`) earlier in the command. `&&`,
#     `;` and a newline carry it to the segments after it, and so does the LEFT
#     side of `||`, which runs either way. Only `&&` (and `X || exit`) makes it
#     CERTAIN; after anything else the cd may have failed, so the directory
#     before it stays possible and is judged too: a protected branch there
#     asks (`cd f; git commit` from main). Same for GIT_DIR. Past four
#     possible directories it asks. `X || exit` is certain too, unless a quote
#     or `#` comes before it or `exit` is redefined; `!` never is. A `cd` is
#     logical, as bash's (`cd link/..` returns; `cd -P` is physical). A `cd`
#     in a child process does not carry: `bash -c`, `pwsh -c`, `cmd /c`, a
#     bash pipeline element, `timeout`, `nohup`, `env`, `sudo` (in PowerShell
#     a pipeline element's Set-Location makes the directory unknown). A `cd` on the RIGHT of `||` (it runs
#     only if the left failed) or before `&` (a background job in bash, the
#     next command in sequence in cmd) leaves the directory unknown. A `cd`
#     before `|` changes only its own pipeline element, so it is ignored. A
#     `cd` (or `export GIT_DIR`) inside `( ... )` or `$( ... )` holds until the
#     matching `)`, where bash restores the directory; in PowerShell, which does
#     not scope parentheses, it leaves the directory unknown after the `)`, as
#     it does for a parenthesis inside quotes or where no command starts. An
#     unquoted `)` with no `(` (a `case` pattern) leaves it unknown, and with
#     `$'...'`, a heredoc or a PowerShell here-string anywhere in the command
#     no parenthesis restores. Line continuations are joined first.
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
# later commit or push IN THAT REPOSITORY (one record per git directory). It
# is certain only when every separator from it to the commit or push is `&&`
# (and it is not on the right of `||`): `git switch f; git commit` commits on
# the old branch when the switch fails, so a protected old branch then asks.
# A switch inside parentheses is not undone at the `)` (HEAD moved on disk).
# A switch in a repository the guard cannot tell makes every later commit or
# push ask. `checkout <b> <path>` and `--help` change no branch; `-t
# origin/main` switches to `main`. Any mention of GIT_DIR or GIT_WORK_TREE
# outside the followed forms makes every later commit or push ask. Past 200
# segments or 40 git queries the guard makes no more git queries but still
# judges every segment by the rules that need none (a deny still denies),
# then asks; past 500 segments the rest is not judged and it asks (the
# hook's own timeout would let it run unchecked). `set -P` / `set -o
# physical` anywhere, or CDPATH set, make a `cd` it cannot place unknown. A path checkout (`checkout -- f`, `checkout .`) or a commit is not
# a branch change.
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

# The git rules and the carried state live beside this file. Sourced after the
# trap, so a missing or broken file is a crash and therefore a deny.
case "${BASH_SOURCE[0]}" in
  */*|*\\*) GUARD_DIR="${BASH_SOURCE[0]%[/\\]*}" ;;
  *) GUARD_DIR="." ;;
esac
# shellcheck source=_guard_state.sh
. "${GUARD_DIR}/_guard_state.sh"
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
#   NO_CARRY (end of a segment followed by `|`): a `cd` there changes only
#     its own pipeline element, so it is ignored;
#   AND (end of a segment followed by `&&`): the next segment runs only if
#     this one succeeded, so a branch change in it stays certain;
#   OPEN / CLOSE (end of a segment followed by `(` / `)`): the main loop saves
#     the directory and GIT_DIR at `(` and restores them at `)`;
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
AND=$'\x0e'
OPEN=$'\x0f'
CLOSE=$'\x10'
cmd="${cmd//[$NO_CARRY$RESET$REDIR$BG$OR_RIGHT$LBR$RBR$AND$OPEN$CLOSE]/}"
re_revbrace='([@^])\{([^{}[:space:]]*)\}'
split_segments() {
  local s="$1" sep
  while [[ $s =~ $re_revbrace ]]; do
    s="${s/"${BASH_REMATCH[0]}"/"${BASH_REMATCH[1]}${LBR}${BASH_REMATCH[2]}${RBR}"}"
  done
  s="${s//&>/$REDIR>}"
  s="${s//>&/>$REDIR}"
  s="${s//<&/<$REDIR}"
  s="${s//&&/$AND$'\n'}"
  s="${s//||/$'\n'$OR_RIGHT}"
  sep='|'; s="${s//"$sep"/$NO_CARRY$'\n'}"
  sep=')'; s="${s//"$sep"/$CLOSE$'\n'}"
  s="${s//&/$BG$'\n'}"
  sep='('; s="${s//"$sep"/$OPEN$'\n'}"
  for sep in ';' '{' '}'; do
    s="${s//"$sep"/$'\n'}"
  done
  # A backtick keeps the chain: the second reading (backticks deleted) is
  # the one in which what they enclose did not take effect.
  sep='`'; s="${s//"$sep"/$AND$'\n'}"
  s="${s//$REDIR/"&"}"   # quoted: bash 5.2 reads a bare & as the match
  s="${s//$LBR/\{}"
  SEGS="${s//$RBR/\}}"
}
# A backslash continuation is bash's; PowerShell reads the backslash as text
# and the newline as a separator. The tool decides; an unknown tool gets both
# readings.
if [[ $cmd == *$'\\\n'* ]]; then
  join_continued "$cmd" '\'
  tool_kind
  case "$TOOL_KIND" in
    bash) split_segments "$JC"; segs="$SEGS" ;;
    ps) split_segments "$cmd"; segs="$SEGS" ;;
    *) split_segments "$JC"; segs="$SEGS"; split_segments "$cmd"
       segs="${segs}"$'\n'"${RESET}"$'\n'"${SEGS}" ;;
  esac
else
  split_segments "$cmd"
  segs="$SEGS"
fi
if [[ $cmd == *'`'* ]]; then
  join_continued "$cmd" '`'
  split_segments "${JC//\`/}"
  segs="${segs}"$'\n'"${RESET}"$'\n'"${SEGS}"
fi
# Quoting the splitter cannot see ($'...', heredocs, PowerShell here-strings)
# can forge a parenthesis, so none of them restores a directory.
PAREN_UNSURE=0
case "$cmd" in *"\$'"*|*'<<'*|*"@'"*|*'@"'*) PAREN_UNSURE=1 ;; esac
# `exit` redefined anywhere: `X || exit` proves nothing.
EXIT_RULE=1
if [[ $cmd =~ (^|[^A-Za-z0-9_])exit[[:space:]]*\(|function[[:space:]]+exit|alias[[:space:]]+exit ]]; then
  EXIT_RULE=0
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
# A `VAR=val` prefix; a quoted value may hold spaces (`GIT_DIR="/x y" git ...`).
re_assign='^[A-Za-z_][A-Za-z0-9_]*=("[^"]*"|'"'"'[^'"'"']*'"'"'|[^[:space:]"'"'"'])*[[:space:]]+'
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
  local t want="" k=0 rest=""
  local -a ws=()
  qsplit "$1"
  ws=(${QW[@]+"${QW[@]}"})
  while [ "$k" -lt "${#ws[@]}" ]; do
    t="${ws[$k]}"
    k=$((k + 1))
    if [ -n "$want" ]; then
      [ "$want" = "dir" ] && { unquote "$t"; NS_ENV_DIRS+=("$UQ"); }
      if [ "$want" = "split" ]; then unquote "$t"; rest="$UQ"; break; fi
      want=""
      continue
    fi
    case "$t" in
      -C|--chdir) want=dir ;;
      --chdir=*) unquote "${t#*=}"; NS_ENV_DIRS+=("$UQ") ;;
      # `-S "git push ..."`: the split string IS the command env runs.
      -S|--split-string) want=split ;;
      --split-string=*) unquote "${t#*=}"; rest="$UQ"; break ;;
      -S?*) unquote "${t#-S}"; rest="$UQ"; break ;;
      -u|--unset) want=skip ;;
      -*) ;;
      [A-Za-z_]*=*) NS_ASSIGN="${NS_ASSIGN} ${t} " ;;
      *) k=$((k - 1)); break ;;
    esac
  done
  while [ "$k" -lt "${#ws[@]}" ]; do
    rest="${rest:+$rest }${ws[$k]}"
    k=$((k + 1))
  done
  ENV_REST="$rest"
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
  NS_CHILD=0
  NS_LAUNCH_CHILD=0
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
      NS_LAUNCH_CHILD=1
      unwrap_env "${s:${#BASH_REMATCH[0]}}"
      s="$ENV_REST"
      continue
    fi
    # A child shell or a launcher that starts a child: a `cd` it runs does
    # not move this shell (`command`, `builtin`, `time` and `iex` run here).
    if [[ $s =~ $re_pwsh ]] || [[ $s =~ $re_cmd ]] || [[ $s =~ $re_sh ]]; then
      NS_CHILD=1
      s="${s:${#BASH_REMATCH[0]}}"
    elif [[ $s =~ $re_launcher ]]; then
      case "${BASH_REMATCH[2]}" in
        command|builtin|time) ;;
        *) NS_LAUNCH_CHILD=1 ;;
      esac
      s="${s:${#BASH_REMATCH[0]}}"
    elif [[ $s =~ $re_iex ]]; then
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
# A `(` opens a subshell or substitution only where a command can start, or
# after `$`, `<` or `>`; elsewhere (`-m "fix(x)"`) it is text in quotes.
re_open_cmd='^[[:space:]]*((then|do|else|elif|if|while|until|!|time)[[:space:]]*)?$'
re_open_sub='[$<>][[:space:]]*$'
# `X || exit`: only `exit` (a `return` outside a function does not stop
# the shell). A segment negated by `!` succeeds when its command fails.
re_exit='^exit([[:space:]]|$)'
re_neg='^[[:space:]]*((time|then|do|else|elif|if|while|until)[[:space:]]+)*![[:space:]]'

# A command too long to judge before the hook's own timeout (which would let
# it run unchecked) spends the budget: past 200 segments here, past 40 git
# queries in git_budget (_guard_state.sh), and past 500 segments in the loop.
nl="${segs//[!$'\n']/}"
if [ "${#nl}" -ge 200 ]; then budget_out "more than 200 segments"; fi
# `set -P` / `set -o physical` anywhere: a `cd` resolves `..` physically, so
# none is placed (case-sensitive: `set -p` is another option).
shopt -u nocasematch
if [[ $cmd =~ (^|[^A-Za-z0-9_-])set[[:space:]]+(-[A-Za-z]*P|-o[[:space:]]+physical) ]]; then CD_PHYS=1; fi
shopt -s nocasematch

pending=0
prev_pipe=0
seg_i=0
while IFS= read -r raw; do
  # Judging stays under the hook's timeout: past 500 segments it stops (a
  # deny further on becomes the ask below, never an allow).
  seg_i=$((seg_i + 1))
  if [ "$seg_i" -gt 500 ]; then
    BUDGET_OUT="more than 500 segments; the rest was not judged"
    break
  fi
  if [ "$raw" = "$RESET" ]; then
    reset_carried_state
    pending=0
    prev_pipe=0
    continue
  fi
  # How this segment's `cd` / `export` reaches the next ones: 1 carries,
  # 3 is a pipeline element (see eff_carry), 2 leaves them unknown.
  carry=1
  or_right=0
  ends=""
  if [[ $raw == "$OR_RIGHT"* ]]; then carry=2; or_right=1; raw="${raw#"$OR_RIGHT"}"; fi
  if [ "$prev_pipe" -eq 1 ] && [ "$carry" -ne 2 ]; then carry=3; fi
  prev_pipe=0
  case "$raw" in
    *"$AND") ends=and ;;
    *"$NO_CARRY") ends=pipe; prev_pipe=1; [ "$carry" -eq 2 ] || carry=3 ;;
    *"$BG") ends=bg; carry=2 ;;
    *"$OPEN") ends=open ;;
    *"$CLOSE") ends=close ;;
  esac
  [ -z "$ends" ] || raw="${raw%?}"
  # An `&` where a command starts (`cd f && & git commit`) is PowerShell's
  # call operator, not a separator: it leaves the chain as it was.
  if [ "$ends" = bg ] && [[ $raw =~ ^[[:space:]]*$ ]]; then ends=and; fi
  # A separator other than `&&` breaks the chain (CHAIN) before the next
  # segment is judged, except in `X || exit`: the shell stops when X fails,
  # so what follows the `exit` ran only if X succeeded.
  exit_guard=0
  if [ "$pending" -eq 1 ]; then
    pending=0
    # Only where nothing before it can hide the `||` (a quote, a comment)
    # and `exit` is not redefined.
    if [ "$or_right" -eq 1 ] && [ "$EXIT_RULE" -eq 1 ] && [ "$SEEN_QH" -eq 0 ] \
       && { [ -z "$ends" ] || [ "$ends" = and ]; }; then
      normalise "$raw"
      if [[ $NS =~ $re_exit ]]; then exit_guard=1; fi
    fi
    if [ "$exit_guard" -eq 0 ]; then CHAIN=$((CHAIN + 1)); fi
  fi
  NEG_CTX=0
  if [[ $raw =~ $re_neg ]]; then NEG_CTX=1; fi
  SW_UNCERTAIN_CTX=0
  if [ "$or_right" -eq 1 ] || [ "$NEG_CTX" -eq 1 ] || [ "$OR_DEPTH" -gt 0 ]; then
    SW_UNCERTAIN_CTX=1
  fi
  judge_segment "$raw"
  if [ "$GITENV_PEND" -eq 1 ]; then GITENV_UNKNOWN=1; fi
  quote_scan "$raw"
  case "$raw" in *[\"\'#]*) SEEN_QH=1 ;; esac
  # After the segment: what the separator does to the carried state, and
  # whether a branch change before it is still certain (only `&&` and the
  # parentheses of a subshell keep it so).
  case "$ends" in
    and) ;;
    open)
      unc=$((or_right || NEG_CTX))
      if [ "$QS" -ne 0 ] || [ "$PAREN_UNSURE" -eq 1 ]; then scope_open 0 "$unc"
      elif [[ $raw =~ $re_open_sub ]]; then scope_open 2 "$unc"
      elif [[ $raw =~ $re_open_cmd ]]; then scope_open 1 "$unc"
      else scope_open 0 "$unc"; fi ;;
    close)
      scope_close $((QS != 0))
      if [ "$SCOPE_KIND" -ne 0 ] && [ "$SCOPE_KIND" -ne 1 ]; then CHAIN=$((CHAIN + 1)); fi ;;
    *) if [ "$exit_guard" -eq 0 ]; then pending=1; fi ;;
  esac
  # A child shell's command ends where its quotes close.
  if [ "$CHILD_IDX" -ge 0 ] && [ "$QS" -eq 0 ]; then scope_close_child; fi
done <<EOF
$segs
EOF

if [ -n "$BUDGET_OUT" ]; then
  add_ask "this command is too long for the guard to judge before its own timeout (${BUDGET_OUT}); no deny rule fired in what was judged, but branches and directories were not checked; split it, or confirm"
fi
if [ -n "$asks" ]; then
  emit ask "PreToolUse guard: ${asks}."
fi
exit 0
