# Sourced by pretooluse_guard.sh before _guard_git.sh: the shell state carried
# from segment to segment of one command (directory, GIT_DIR, branch-change
# records, parentheses, quotes), and the judge of one segment. Split out to
# keep each file readable; it is not a hook of its own and does nothing when
# run.
#
# An ACCIDENT NET, not a control: see the header of pretooluse_guard.sh.
# Uses unquote, add_ask, deny and $input from that file; sets no options itself.
# shellcheck shell=bash

# --- state carried from segment to segment ---------------------------------------
# Set by the main loop of pretooluse_guard.sh as it meets `cd`, `export`,
# `$env:`, parentheses and git branch changes earlier in the same command:
#   CD_C / CD_KNOWN       -- `-C <dir>` entries of every carried `cd`, and 0 once
#                            a directory could not be followed;
#   SH_GITDIR / SH_GITDIR_SET / SH_GITDIR_KNOWN -- GIT_DIR from `export` or
#                            PowerShell's `$env:GIT_DIR`;
#   SW_KEYS / SW_NEW / SW_PREV / SW_EPOCH -- one entry per git directory (an
#                            absolute path) in which an earlier `git switch` or
#                            `git checkout` moved: the branch it moved to (":"
#                            detached, "?" unknown), the branches the repository
#                            may still be on if that switch did not happen
#                            (" a b "), and CHAIN when it ran (-1: never certain).
#                            Indexed arrays, not `declare -A`: the guard stays
#                            portable to bash 3.2 (macOS), where `declare -A`
#                            would crash it into a deny for every command;
#   SW_UNRESOLVED         -- 1 once a switch ran where the repository could not
#                            be told: every later commit or push asks;
#   CHAIN                 -- bumped at every separator other than `&&` (and the
#                            transparent `(` / `)` of a subshell). A switch is
#                            CERTAIN for a later commit or push only while
#                            CHAIN has not moved since it ran;
#   SW_UNCERTAIN_CTX      -- 1 for a segment on the right of `||` or inside
#                            parentheses opened there: a switch in it is never
#                            certain;
#   ST_* / ST_N / OR_DEPTH -- the stack of open parentheses: what `(` saved of
#                            the directory and GIT_DIR, restored at its `)`;
#   QS                    -- the quote state at the end of the text so far;
#   ALT_* / ALT_N / ALT_OVF / DIR_EPOCH -- directory changes, the R1 rule for
#                            `cd`: the globals above are the state after the
#                            last change, certain while CHAIN has not moved
#                            since (DIR_EPOCH). ALT_* keep the states before it
#                            (each change may not have happened: a failed `cd`,
#                            one inside `if`), at most three; ALT_OVF marks more,
#                            and then an uncertain directory asks. Entries from
#                            ALT_NEWFROM up are on the new side: a relative `cd`
#                            applied to a directory that may have been in
#                            force, possible even while the chain is certain.
#                            Branch changes are never restored: HEAD moved on disk.
# Per git command, check_git computes:
#   GIT_C        -- the git options that put a query where the command runs, in
#                   the order git applies them (`-C` entries, then --git-dir=);
#   TARGET_KNOWN -- 0 when a directory or git directory could not be followed.
GIT_C=()
TARGET_KNOWN=1

reset_carried_state() {
  CD_C=()
  CD_KNOWN=1
  SH_GITDIR=""
  SH_GITDIR_SET=0
  SH_GITDIR_KNOWN=1
  SW_KEYS=()
  SW_NEW=()
  SW_PREV=()
  SW_EPOCH=()
  SW_UNRESOLVED=0
  SW_UNCERTAIN_CTX=0
  CHAIN=0
  ST_N=0
  ST_KIND=()
  ST_OR=()
  ST_CDC=()
  ST_CDK=()
  ST_GD=()
  ST_GDS=()
  ST_GDK=()
  OR_DEPTH=0
  QS=0
  ALT_N=0
  ALT_NEWFROM=0
  ALT_OVF=0
  ALT_CDC=()
  ALT_CDK=()
  ALT_GD=()
  ALT_GDS=()
  ALT_GDK=()
  DIR_EPOCH=0
  ST_AN=()
  ST_AO=()
  ST_DE=()
  ST_NF=()
  CHILD_IDX=-1
  GITENV_UNKNOWN=0
  SEEN_QH=0
}
# CD_C is stored joined by the unit separator: an absolute directory (from the
# hook's own $PWD) may contain spaces.
US=$'\x1f'
TOOL_KIND=""
NEG_CTX=0
GIT_SPAWNS=0
BUDGET_OUT=""
CD_PHYS=0
SW_IN_ALT=0
DIR_ALT=0
DIR_INF_FROM=0
DIR_INF_TO=0
SEG_GO=()
reset_carried_state

# Sets SWI to the index of git directory $1 in SW_KEYS, or -1.
sw_find() {
  local k=0
  SWI=-1
  while [ "$k" -lt "${#SW_KEYS[@]}" ]; do
    if [ "${SW_KEYS[$k]}" = "$1" ]; then SWI=$k; return 0; fi
    k=$((k + 1))
  done
}

cdc_str() {
  local IFS="$US"
  CDC_S="${CD_C[*]+${CD_C[*]}}"
}
cdc_load() {
  CD_C=()
  [ -z "$1" ] || IFS="$US" read -ra CD_C <<<"$1" || true
}

# A command too long to judge before the hook's own timeout (which would let
# it run unchecked) spends the budget: from then on the guard makes no git
# query (gitq fails), but still judges every segment with the rules that need
# none, so a deny still denies; the end of the command asks for the rest.
# git_budget counts git queries and fails once the budget is spent (gitq
# counts its own calls in this shell; a `$(gitq ...)` site is guarded by
# git_budget, so a spent budget forks nothing).
budget_out() {
  [ -n "$BUDGET_OUT" ] || BUDGET_OUT="$1"
}
git_budget() {
  [ -z "$BUDGET_OUT" ] || return 1
  GIT_SPAWNS=$((GIT_SPAWNS + 1))
  if [ "$GIT_SPAWNS" -gt 40 ]; then budget_out "more than 40 git queries"; return 1; fi
}

# `(` opened. $1 = 1 for a subshell `( ... )`, 2 for a substitution `$( ... )`,
# `<( ... )` or `>( ... )`, 3 for a child shell (`bash -c "..."`), 0 when the
# guard cannot tell (inside quotes, where no command starts: `-m "fix(x)"`,
# `a=(1 2)`, or in a command with `$'...'` or a heredoc); $2 = 1 when it is on
# the right of `||` or negated.
scope_open() {
  cdc_str
  ST_KIND[ST_N]="$1"
  ST_OR[ST_N]="$2"
  ST_CDC[ST_N]="$CDC_S"
  ST_CDK[ST_N]="$CD_KNOWN"
  ST_GD[ST_N]="$SH_GITDIR"
  ST_GDS[ST_N]="$SH_GITDIR_SET"
  ST_GDK[ST_N]="$SH_GITDIR_KNOWN"
  ST_AN[ST_N]="$ALT_N"
  ST_AO[ST_N]="$ALT_OVF"
  ST_DE[ST_N]="$DIR_EPOCH"
  ST_NF[ST_N]="$ALT_NEWFROM"
  if [ "$2" -eq 1 ]; then OR_DEPTH=$((OR_DEPTH + 1)); fi
  ST_N=$((ST_N + 1))
}

# `)` closed; $1 = 1 when it is inside quotes. Sets SCOPE_KIND to the kind of
# its `(` (0 when either side is uncertain). Bash runs a subshell or
# substitution in a child, so the directory and GIT_DIR come back as they were
# at the `(`. Where that is not certain -- PowerShell, which does not scope
# parentheses (tool_is_powershell is in the main file, asked only when
# something changed), or a parenthesis that may be text -- a change made
# inside leaves them UNKNOWN: neither restored nor kept on a guess. A `)` with
# no `(` (a `case` pattern) leaves the directory unknown; a quoted one is text.
scope_close() {
  local cdc
  # A child shell's frame is closed only where its quotes close: a `)` on top
  # of it is one with no `(` of its own.
  if [ "$ST_N" -eq 0 ] || { [ "${2:-}" != child ] && [ "${ST_KIND[$((ST_N - 1))]}" -eq 3 ]; }; then
    SCOPE_KIND=0
    if [ "$1" -eq 0 ]; then SCOPE_KIND=-1; CD_KNOWN=0; CD_C=(); fi
    return 0
  fi
  ST_N=$((ST_N - 1))
  SCOPE_KIND="${ST_KIND[$ST_N]}"
  if [ "$1" -eq 1 ]; then SCOPE_KIND=0; fi
  if [ "${ST_OR[$ST_N]}" -eq 1 ]; then OR_DEPTH=$((OR_DEPTH - 1)); fi
  cdc_str
  cdc="$CDC_S"
  if [ "$cdc" = "${ST_CDC[$ST_N]}" ] && [ "$CD_KNOWN" = "${ST_CDK[$ST_N]}" ] \
     && [ "$SH_GITDIR" = "${ST_GD[$ST_N]}" ] && [ "$SH_GITDIR_SET" = "${ST_GDS[$ST_N]}" ] \
     && [ "$SH_GITDIR_KNOWN" = "${ST_GDK[$ST_N]}" ]; then
    return 0
  fi
  if [ "$SCOPE_KIND" -ne 3 ] && { [ "$SCOPE_KIND" -eq 0 ] || tool_is_powershell; }; then
    if [ "$cdc" != "${ST_CDC[$ST_N]}" ] || [ "$CD_KNOWN" != "${ST_CDK[$ST_N]}" ]; then
      CD_KNOWN=0
      CD_C=()
    fi
    if [ "$SH_GITDIR" != "${ST_GD[$ST_N]}" ] || [ "$SH_GITDIR_SET" != "${ST_GDS[$ST_N]}" ]; then
      SH_GITDIR_SET=1
      SH_GITDIR_KNOWN=0
    fi
    return 0
  fi
  cdc_load "${ST_CDC[$ST_N]}"
  CD_KNOWN="${ST_CDK[$ST_N]}"
  SH_GITDIR="${ST_GD[$ST_N]}"
  SH_GITDIR_SET="${ST_GDS[$ST_N]}"
  SH_GITDIR_KNOWN="${ST_GDK[$ST_N]}"
  ALT_N="${ST_AN[$ST_N]}"
  ALT_OVF="${ST_AO[$ST_N]}"
  DIR_EPOCH="${ST_DE[$ST_N]}"
  ALT_NEWFROM="${ST_NF[$ST_N]}"
}

# The child shell opened at CHILD_IDX ends: everything it changed is undone
# (it ran in another process), whatever parentheses inside it left open.
scope_close_child() {
  local k=0
  OR_DEPTH=0
  while [ "$k" -lt "$CHILD_IDX" ]; do
    if [ "${ST_OR[$k]}" -eq 1 ]; then OR_DEPTH=$((OR_DEPTH + 1)); fi
    k=$((k + 1))
  done
  ST_N=$((CHILD_IDX + 1))
  scope_close 0 child
  CHILD_IDX=-1
}

# True when the directory state is uncertain: a change since which something
# other than `&&` ran, with earlier states kept.
dir_uncertain() {
  [ "$DIR_EPOCH" != "$CHAIN" ] && { [ "$ALT_N" -gt 0 ] || [ "$ALT_OVF" -eq 1 ]; }
}

# Keep state ($1 cd entries, $2 cd known, $3-$5 GIT_DIR value, set, known)
# as possible, once among the entries from index $6 up. Past three kept
# states it is not kept: ALT_OVF, and a return of 1.
alt_add() {
  local k="$6"
  while [ "$k" -lt "$ALT_N" ]; do
    if [ "${ALT_CDC[$k]}" = "$1" ] && [ "${ALT_CDK[$k]}" = "$2" ] && [ "${ALT_GD[$k]}" = "$3" ] \
       && [ "${ALT_GDS[$k]}" = "$4" ] && [ "${ALT_GDK[$k]}" = "$5" ]; then
      return 0
    fi
    k=$((k + 1))
  done
  if [ "$ALT_N" -ge 3 ]; then ALT_OVF=1; return 1; fi
  ALT_CDC[ALT_N]="$1"; ALT_CDK[ALT_N]="$2"
  ALT_GD[ALT_N]="$3"; ALT_GDS[ALT_N]="$4"; ALT_GDK[ALT_N]="$5"
  ALT_N=$((ALT_N + 1))
}

# Before a change of directory or GIT_DIR: the state it replaces stays
# possible beside the ones kept before it (`cd a && cd b; git commit` runs in
# the start directory when `cd a` fails), and every kept state is now behind
# the change (none on the new side). Sets DIR_INF_FROM / DIR_INF_TO to the
# kept states that may have been in force: all when the directory was
# uncertain, else the new-side ones.
dir_change() {
  DIR_INF_FROM=$ALT_NEWFROM
  if dir_uncertain; then DIR_INF_FROM=0; fi
  DIR_INF_TO=$ALT_N
  cdc_str
  alt_add "$CDC_S" "$CD_KNOWN" "$SH_GITDIR" "$SH_GITDIR_SET" "$SH_GITDIR_KNOWN" 0 || true
  ALT_NEWFROM=$ALT_N
  DIR_EPOCH=$CHAIN
  # Negated by `!`: it succeeds when the change fails, so it is never certain.
  if [ "$NEG_CTX" -eq 1 ]; then DIR_EPOCH=-1; fi
}

# Put kept state $1 in the globals (the caller saved and restores its own).
dir_use() {
  cdc_load "${ALT_CDC[$1]}"
  CD_KNOWN="${ALT_CDK[$1]}"
  SH_GITDIR="${ALT_GD[$1]}"
  SH_GITDIR_SET="${ALT_GDS[$1]}"
  SH_GITDIR_KNOWN="${ALT_GDK[$1]}"
}

# The tool (TOOL_KIND: bash, ps or unknown), read only when it matters: a `)`
# that closes parentheses which changed the directory, a pipeline element that
# changes it, a backslash before a newline. A payload the parsers cannot read
# here is unknown, and unknown is treated as PowerShell where that asks.
# $input is the hook payload (main file).
tool_kind() {
  [ -z "$TOOL_KIND" ] || return 0
  local t="" py
  if ! { command -v jq >/dev/null 2>&1 \
         && t="$(printf '%s' "$input" | jq -r '.tool_name // empty' 2>/dev/null)"; }; then
    t=""
    for py in python3 python; do
      command -v "$py" >/dev/null 2>&1 || continue
      if t="$(printf '%s' "$input" | "$py" -c 'import json,sys
d=json.load(sys.stdin)
t=d.get("tool_name") if isinstance(d,dict) else None
sys.stdout.write(t if isinstance(t,str) else "")' 2>/dev/null)"; then
        break
      fi
      t=""
    done
  fi
  case "$t" in
    bash) TOOL_KIND=bash ;;
    powershell) TOOL_KIND=ps ;;
    *) TOOL_KIND=unknown ;;
  esac
}
tool_is_powershell() {
  tool_kind
  [ "$TOOL_KIND" != bash ]
}

# Quote state at the end of $1, carried from the text before it: QS is 0
# outside quotes, 1 inside '...', 2 inside "...". A backslash escapes the next
# character outside single quotes. Only parentheses read it: a misreading makes
# them uncertain (see scope_close), never a silent restore.
quote_scan() {
  local s="$1" i=0 n c
  case "$s" in *[\"\'\\]*) ;; *) return 0 ;; esac
  n=${#s}
  while [ "$i" -lt "$n" ]; do
    c="${s:$i:1}"
    if [ "$c" = '\' ] && [ "$QS" -ne 1 ]; then i=$((i + 2)); continue; fi
    case "$QS$c" in
      0\") QS=2 ;;
      0\') QS=1 ;;
      1\'|2\") QS=0 ;;
    esac
    i=$((i + 1))
  done
}

# A directory, git directory or branch name the guard can follow: no expansion
# left to the shell. A variable, `~`, `%VAR%` or a glob means "unknown".
re_dir_unknown='[$~%*?[`]'
is_abs_dir() {
  [[ $1 == /* ]] || [[ $1 == \\* ]] || [[ $1 =~ ^[A-Za-z]: ]]
}

# Lexically normalised absolute POSIX path $1: `.` and `..` resolved as
# bash's `cd -L` (and PowerShell) does, without reading the disk. Sets LN.
lex_norm() {
  local IFS='/' part
  local -a parts=() out=()
  read -ra parts <<<"$1" || true
  for part in ${parts[@]+"${parts[@]}"}; do
    case "$part" in
      ''|.) ;;
      ..) if [ "${#out[@]}" -gt 0 ]; then unset "out[$((${#out[@]} - 1))]"; fi ;;
      *) out+=("$part") ;;
    esac
  done
  LN="/${out[*]+${out[*]}}"
}

# Where `cd $2` from the state whose cd entries are $1 (joined by US; empty
# is the hook's own directory) lands, as cd entries in CR. Logical (bash's
# default, PowerShell): an absolute path normalised lexically from $PWD, never
# a relative chain for `git -C`, which resolves `..` physically and so leaves
# a symbolic link where bash does not (`cd fl/..`). $3 = 1 for `cd -P`, and
# for a base that is not a POSIX path (a Windows drive): the chain git follows.
cd_resolve() {
  local base="" rest
  if [ "$3" -eq 0 ] && [[ $2 == /* ]]; then lex_norm "$2"; CR="-C${US}${LN}"; return 0; fi
  if is_abs_dir "$2"; then CR="-C${US}$2"; return 0; fi
  if [ "$3" -eq 0 ]; then
    if [ -z "$1" ]; then
      base="${PWD:-}"
    else
      rest="${1#-C"$US"}"
      [[ $rest == *"$US"* ]] || base="$rest"
    fi
    if [[ $base == /* ]]; then lex_norm "${base}/$2"; CR="-C${US}${LN}"; return 0; fi
  fi
  CR="${1:+$1$US}-C${US}$2"
}

# `cd <dir>` (bash, cmd, PowerShell's aliases). $1 = 1 to apply it, 0 to make
# the directory unknown; the rest are the segment's tokens, program first,
# redirections already removed.
apply_cd() {
  local how="$1"
  shift
  local -a a=("$@")
  local t dir="" n=0 i=1 phys=0 k
  dir_change
  if [ "$how" -eq 0 ]; then CD_KNOWN=0; CD_C=(); return 0; fi
  while [ "$i" -lt "${#a[@]}" ]; do
    unquote "${a[$i]}"
    t="$UQ"
    i=$((i + 1))
    case "$t" in
      -P) phys=1; continue ;;
      -L|-e|-@|--|-path|-literalpath|/d) continue ;;
    esac
    n=$((n + 1))
    dir="$t"
  done
  # Not placed: `set -P` / `set -o physical` in the command (every cd then
  # resolves `..` physically), or a name CDPATH may resolve elsewhere.
  if [ "$n" -ne 1 ] || [ "$dir" = "-" ] || [[ $dir =~ $re_dir_unknown ]] || [ "$CD_PHYS" -eq 1 ] \
     || { [ -n "${CDPATH:-}" ] && ! is_abs_dir "$dir" && [[ $dir != .* ]]; }; then
    CD_KNOWN=0
    CD_C=()
    return 0
  fi
  if is_abs_dir "$dir"; then
    CD_KNOWN=1
  else
    # Relative to whichever directory was in force: each kept one that may
    # have been, plus this step, is on the new side too. Past the bound the
    # directory is unknown.
    k=$DIR_INF_FROM
    while [ "$k" -lt "$DIR_INF_TO" ]; do
      if [ "${ALT_CDK[$k]}" -eq 1 ]; then
        cd_resolve "${ALT_CDC[$k]}" "$dir" "$phys"
        alt_add "$CR" 1 "${ALT_GD[$k]}" "${ALT_GDS[$k]}" "${ALT_GDK[$k]}" "$ALT_NEWFROM" \
          || CD_KNOWN=0
      fi
      k=$((k + 1))
    done
  fi
  if [ "$CD_KNOWN" -eq 1 ]; then
    cdc_str
    cd_resolve "$CDC_S" "$dir" "$phys"
    cdc_load "$CR"
  else
    CD_C=()
  fi
}

# GIT_DIR set for the rest of the command (`export GIT_DIR=...`,
# `$env:GIT_DIR = ...`). $1 = 1 to apply, 0 for unknown; $2 = the value.
# GIT_WORK_TREE and GIT_COMMON_DIR are not tracked: alone they do not change
# which HEAD git reads.
apply_shell_git_dir() {
  dir_change
  SH_GITDIR_SET=1
  unquote "$2"
  SH_GITDIR="$UQ"
  if [ "$1" -eq 0 ] || [ -z "$SH_GITDIR" ] || [[ $SH_GITDIR =~ $re_dir_unknown ]]; then
    SH_GITDIR_KNOWN=0
  else
    SH_GITDIR_KNOWN=1
  fi
  # A relative one is read in whichever directory is in force when git runs:
  # every kept directory that may be in force gets it on the new side too.
  is_abs_dir "$SH_GITDIR" && return 0
  local k=$DIR_INF_FROM
  while [ "$k" -lt "$DIR_INF_TO" ]; do
    alt_add "${ALT_CDC[$k]}" "${ALT_CDK[$k]}" "$SH_GITDIR" 1 "$SH_GITDIR_KNOWN" "$ALT_NEWFROM" \
      || SH_GITDIR_KNOWN=0
    k=$((k + 1))
  done
}

# --- the segment judge (called by the main loop of pretooluse_guard.sh) ------

# Split $1 into words (QW) as the shell does for quoting: whitespace inside
# quotes, or escaped by a backslash, stays in the word (`git -C "/x y" commit`
# is three words before the subcommand, not four). Quotes stay in the words
# (unquote removes them). A quote left open falls back to plain splitting.
qsplit() {
  local s="$1" c i=0 n w="" inw=0 q=""
  QW=()
  case "$s" in
    *[\"\'\\]*) ;;
    *) read -ra QW <<<"$s" || true; return 0 ;;
  esac
  n=${#s}
  while [ "$i" -lt "$n" ]; do
    c="${s:$i:1}"
    i=$((i + 1))
    if [ -n "$q" ]; then
      w="${w}${c}"
      if [ "$c" = "$q" ]; then
        q=""
      elif [ "$c" = '\' ] && [ "$q" = '"' ] && [ "$i" -lt "$n" ]; then
        w="${w}${s:$i:1}"
        i=$((i + 1))
      fi
      continue
    fi
    case "$c" in
      [[:space:]]) if [ "$inw" -eq 1 ]; then QW+=("$w"); w=""; inw=0; fi ;;
      \"|\') q="$c"; w="${w}${c}"; inw=1 ;;
      \\) w="${w}${c}"; inw=1
          if [ "$i" -lt "$n" ]; then w="${w}${s:$i:1}"; i=$((i + 1)); fi ;;
      *) w="${w}${c}"; inw=1 ;;
    esac
  done
  if [ -n "$q" ]; then read -ra QW <<<"$s" || true; return 0; fi
  if [ "$inw" -eq 1 ]; then QW+=("$w"); fi
}
# Uses the regexes and check_* functions of that file, at call time.

# Line continuations: an escape character before a newline joins the two
# lines, unless it is itself escaped (an even run). Sets JC. $2 is `\` for
# bash and the backtick for PowerShell.
# Not inside a comment (a `#` that starts a word, outside quotes): there the
# escape is text and the newline ends the comment.
join_continued() {
  local s="$1" e="$2" out="" c i=0 n run=0 sq=0 dq=0 cm=0 prev=" "
  case "$s" in *"$e"$'\n'*) ;; *) JC="$s"; return 0 ;; esac
  n=${#s}
  while [ "$i" -lt "$n" ]; do
    c="${s:$i:1}"
    i=$((i + 1))
    if [ "$cm" -eq 1 ]; then
      [ "$c" != $'\n' ] || cm=0
    elif [ "$c" = "$e" ]; then
      run=$((run + 1))
      out="${out}${c}"
      prev="$c"
      continue
    elif [ "$c" = $'\n' ] && [ $((run % 2)) -eq 1 ]; then
      out="${out%?}"
      run=0
      prev=" "
      continue
    elif [ $((run % 2)) -eq 0 ]; then
      case "$c" in
        \') [ "$dq" -eq 1 ] || sq=$((1 - sq)) ;;
        \") [ "$sq" -eq 1 ] || dq=$((1 - dq)) ;;
        \#) if [ "$sq" -eq 0 ] && [ "$dq" -eq 0 ]; then
              case "$prev" in [[:space:]]|';'|'&'|'|'|'('|')') cm=1 ;; esac
            fi ;;
      esac
    fi
    run=0
    out="${out}${c}"
    prev="$c"
  done
  JC="$out"
}
# True when a GIT_DIR / GIT_WORK_TREE mention in segment $1 (normalised) is a
# form the guard follows: a `VAR=val` prefix of a command, `export NAME=...`
# or `$env:NAME = ...`. Anything else (`declare -x`, `set -a`, `Set-Item
# Env:`, `${env:...}`, a bare assignment) is not, and makes later commits ask.
gitenv_followed() {
  local s="$1" w
  local -a ws=()
  [ -n "$s" ] || return 1
  unquote "$s"
  [[ ${UQ//\\/} =~ GIT_DIR|GIT_WORK_TREE ]] || return 0
  if [[ $s =~ $re_psenv ]]; then
    [[ ${BASH_REMATCH[1]} =~ ^(GIT_DIR|GIT_WORK_TREE)$ ]]
    return
  fi
  read -ra ws <<<"$s" || true
  unquote "${ws[0]}"
  [ "$UQ" = export ] || return 1
  for w in "${ws[@]:1}"; do
    unquote "$w"
    UQ="${UQ//\\/}"
    if [[ $UQ =~ GIT_DIR|GIT_WORK_TREE ]] && ! [[ $UQ =~ ^(GIT_DIR|GIT_WORK_TREE)= ]]; then
      return 1
    fi
  done
  return 0
}

# How a directory change in this segment reaches the next ones (EC): carry 3
# is a pipeline element, which runs in a child in bash (ignored) but in the
# shell itself in PowerShell (unknown); a change run by a launcher that
# starts a child (`timeout`, `nohup`, `env`, `sudo`, ...) is ignored.
eff_carry() {
  EC=$carry
  if [ "$EC" -eq 3 ]; then
    if tool_is_powershell; then EC=2; else EC=0; fi
  fi
  if [ "$NS_LAUNCH_CHILD" -eq 1 ]; then EC=0; fi
}

# Judge one segment ($1, its markers removed); `carry` says how a `cd` or
# `export` in it reaches the segments after it.
judge_segment() {
  normalise "$1"
  seg="$NS"
  GITENV_PEND=0
  unquote "$1"
  if [[ ${UQ//\\/} =~ GIT_DIR|GIT_WORK_TREE ]] && ! gitenv_followed "$seg"; then GITENV_PEND=1; fi
  # A child shell (`bash -c`, `pwsh -c`, `cmd /c`): what it changes ends with
  # its quoted command (the main loop closes the frame when quotes close).
  if [ "$NS_CHILD" -eq 1 ] && [ "$CHILD_IDX" -lt 0 ]; then
    scope_open 3 0
    CHILD_IDX=$((ST_N - 1))
  fi
  [ -n "$seg" ] || return 0
  qsplit "$seg"
  words=(${QW[@]+"${QW[@]}"})
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
  [ "${#toks[@]}" -gt 0 ] || return 0
  unquote "${toks[0]}"
  prog="$UQ"
  base="${prog##*[/\\]}"
  case "$base" in
    cd|chdir|set-location|sl)
      eff_carry
      case "$EC" in
        1) apply_cd 1 "${toks[@]}" ;;
        2) apply_cd 0 "${toks[@]}" ;;
      esac ;;
    pushd|popd|push-location|pop-location)
      eff_carry
      [ "$EC" -eq 0 ] || apply_cd 0 "${toks[@]}" ;;
    export)
      eff_carry
      for w in ${toks[@]+"${toks[@]:1}"}; do
        unquote "$w"
        case "${UQ//\\/}" in
          GIT_DIR=*) [ "$EC" -eq 0 ] || apply_shell_git_dir $((EC == 1)) "${UQ#*=}" ;;
        esac
      done ;;
  esac
  if [[ $seg =~ $re_psenv ]] && [[ ${BASH_REMATCH[1]} == GIT_DIR ]]; then
    eff_carry
    [ "$EC" -eq 0 ] || apply_shell_git_dir $((EC == 1)) "${BASH_REMATCH[2]}"
  fi
  case "$base" in
    git|git.exe) check_git ${toks[@]+"${toks[@]:1}"} ;;
    gh|gh.exe) check_gh ${toks[@]+"${toks[@]:1}"} ;;
  esac
  if [[ $base =~ $re_runnerish ]] || [[ $prog =~ $re_runnerish ]]; then
    check_runner "$seg" "${toks[@]}"
  fi
  return 0
}
