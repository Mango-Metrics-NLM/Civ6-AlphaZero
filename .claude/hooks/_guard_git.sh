# Sourced by pretooluse_guard.sh: the git rules (which repository and branch a
# git command acts on, and what a push would send). Split out to keep each file
# readable; it is not a hook of its own and does nothing when run.
#
# An ACCIDENT NET, not a control: see the header of pretooluse_guard.sh.
# Uses unquote, add_ask and deny from that file, and sets no options itself.
# shellcheck shell=bash

# --- state carried from segment to segment ---------------------------------------
# Set by the main loop of pretooluse_guard.sh as it meets `cd`, `export`,
# `$env:` and git branch changes earlier in the same command:
#   CD_C / CD_KNOWN       -- `-C <dir>` entries of every carried `cd`, and 0 once
#                            a directory could not be followed;
#   SH_GITDIR / SH_GITDIR_SET / SH_GITDIR_KNOWN -- GIT_DIR from `export` or
#                            PowerShell's `$env:GIT_DIR`;
#   SW_STATE / SW_BRANCH / SW_GITDIR -- the branch an earlier `git switch` or
#                            `git checkout` moved to ("" none, known, unknown),
#                            and the git directory it happened in ("" any).
# Per git command, check_git computes:
#   GIT_C        -- the git options that put a query where the command runs, in
#                   the order git applies them (`-C` entries, then --git-dir=);
#   TARGET_KNOWN -- 0 when a directory or git directory could not be followed.
CD_C=()
CD_KNOWN=1
SH_GITDIR=""
SH_GITDIR_SET=0
SH_GITDIR_KNOWN=1
SW_STATE=""
SW_BRANCH=""
SW_GITDIR=""
GIT_C=()
TARGET_KNOWN=1

reset_carried_state() {
  CD_C=()
  CD_KNOWN=1
  SH_GITDIR=""
  SH_GITDIR_SET=0
  SH_GITDIR_KNOWN=1
  SW_STATE=""
  SW_BRANCH=""
  SW_GITDIR=""
}

gitq() {
  git ${GIT_C[@]+"${GIT_C[@]}"} "$@" 2>/dev/null
}

# A directory, git directory or branch name the guard can follow: no expansion
# left to the shell. A variable, `~`, `%VAR%` or a glob means "unknown".
re_dir_unknown='[$~%*?[`]'
is_abs_dir() {
  [[ $1 == /* ]] || [[ $1 == \\* ]] || [[ $1 =~ ^[A-Za-z]: ]]
}

# `cd <dir>` (bash, cmd, PowerShell's aliases). $1 = 1 to apply it, 0 to make
# the directory unknown; the rest are the segment's tokens, program first,
# redirections already removed.
apply_cd() {
  local how="$1"
  shift
  local -a a=("$@")
  local t dir="" n=0 i=1
  if [ "$how" -eq 0 ]; then CD_KNOWN=0; CD_C=(); return 0; fi
  while [ "$i" -lt "${#a[@]}" ]; do
    unquote "${a[$i]}"
    t="$UQ"
    i=$((i + 1))
    case "$t" in
      -L|-P|-e|-@|--|-path|-literalpath|/d) continue ;;
    esac
    n=$((n + 1))
    dir="$t"
  done
  if [ "$n" -ne 1 ] || [ "$dir" = "-" ] || [[ $dir =~ $re_dir_unknown ]]; then
    CD_KNOWN=0
    CD_C=()
  elif is_abs_dir "$dir"; then
    CD_KNOWN=1
    CD_C=(-C "$dir")
  elif [ "$CD_KNOWN" -eq 1 ]; then
    CD_C+=(-C "$dir")
  fi
}

# GIT_DIR set for the rest of the command (`export GIT_DIR=...`,
# `$env:GIT_DIR = ...`). $1 = 1 to apply, 0 for unknown; $2 = the value.
# GIT_WORK_TREE and GIT_COMMON_DIR are not tracked: alone they do not change
# which HEAD git reads.
apply_shell_git_dir() {
  SH_GITDIR_SET=1
  unquote "$2"
  SH_GITDIR="$UQ"
  if [ "$1" -eq 0 ] || [ -z "$SH_GITDIR" ] || [[ $SH_GITDIR =~ $re_dir_unknown ]]; then
    SH_GITDIR_KNOWN=0
  else
    SH_GITDIR_KNOWN=1
  fi
}

# --- protected branches (resolved lazily: git is a process spawn) ------------
# Resolved in the TARGET repository (GIT_C) and re-resolved when the target
# changes from one segment to the next. `repo_gitdir` is empty when git cannot
# read a repository there; `branch` is empty on a detached HEAD. The branch is
# read without --short: with a tag of the same name, --short prints
# `heads/main`, which is not the name `main`.
resolved_for=""
repo_gitdir=""
branch=""
default_branch=""
protected=" dev qa main "
protected_display="dev qa main"

resolve_branches() {
  local key="${GIT_C[*]+${GIT_C[*]}}|"
  [ "$resolved_for" = "$key" ] && return 0
  resolved_for="$key"
  repo_gitdir="$(gitq rev-parse --absolute-git-dir)" || repo_gitdir=""
  branch="$(gitq symbolic-ref --quiet HEAD)" || branch=""
  branch="${branch#refs/heads/}"
  default_branch="${CIV6AZ_DEFAULT_BRANCH:-}"
  if [ -z "$default_branch" ]; then
    default_branch="$(gitq symbolic-ref --quiet refs/remotes/origin/HEAD)" || default_branch=""
    default_branch="${default_branch#refs/remotes/origin/}"
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

# True when $1 is git's abbreviation of the long option $2: a prefix of it at
# least $3 characters long. git accepts any unique prefix; the minimum lengths
# come from `git push -h` (git 2.43): --al(l), --b(ranches), --m(irror),
# --ta(gs), --fol(low-tags), --pru(ne), --de(lete). --force has none: every
# abbreviation of it is ambiguous with --force-with-lease and git refuses it.
is_abbrev() {
  [ "${#1}" -ge "$3" ] && [[ $2 == "$1"* ]]
}

# ASK before a push that would re-create a merged branch. $@ = `git push` args.
#
# A CONVENIENCE NET, not a control. Judged from local refs alone (no network),
# in the repository the push runs in. It asks when, for
# ONE protected branch P whose refs/remotes/origin/P exists locally (dev, qa,
# main, the default branch, CIV6AZ_PROTECTED_BRANCHES; a name with no such ref
# is skipped):
#   * the pushed commit (the refspec's source; HEAD when none) is origin/P or
#     an ancestor of it, so the push carries nothing new; AND
#   * refs/remotes/origin/<dst> is absent (the push would create <dst>), or is
#     itself origin/P or an ancestor of it. The second case is the stale ref:
#     git does not prune by default, so after GitHub deletes a merged branch
#     its tracking ref is still here and would otherwise hide the slip.
# Any git query that fails means "no ask". It runs after every deny rule and
# only ever adds an ask. Not covered, on purpose (both err toward allow): a
# remote other than origin, and deletions.
check_recreated_branch() {
  local -a pos=()
  local t skip=0
  for t in "$@"; do
    if [ "$skip" -eq 1 ]; then skip=0; continue; fi
    case "$t" in
      -d|--de*) return 0 ;;
      -o|--push-option|--repo|--receive-pack|--exec) skip=1 ;;
      -*) ;;
      *) pos+=("$t") ;;
    esac
  done
  if [ "${#pos[@]}" -ge 1 ] && [ "${pos[0]}" != "origin" ]; then
    return 0
  fi

  # resolve_branches reads the default branch in the push's own repository.
  resolve_branches
  local def="$default_branch" cur="$branch"

  # Protected branches that have a remote-tracking ref here, with their tips.
  local -a pnames=() ptips=()
  local b seen=" " tip
  # Unquoted on purpose: $protected is a space-separated list (noglob is set).
  for b in $def $protected; do
    case "$seen" in *" $b "*) continue ;; esac
    seen="${seen}${b} "
    tip="$(gitq rev-parse --verify --quiet "refs/remotes/origin/${b}^{commit}")" || continue
    [ -n "$tip" ] || continue
    pnames+=("$b")
    ptips+=("$tip")
  done
  [ "${#pnames[@]}" -gt 0 ] || return 0

  local -a specs=()
  if [ "${#pos[@]}" -ge 2 ]; then
    specs=("${pos[@]:1}")
  else
    specs=(HEAD)
  fi
  local spec src dst sha dtip k hit what
  for spec in "${specs[@]}"; do
    case "$spec" in
      *:*) src="${spec%%:*}"; dst="${spec##*:}" ;;
      *) src="$spec"; dst="$spec" ;;
    esac
    [ -n "$src" ] || continue                  # `:dst` deletes
    if [ "$dst" = "HEAD" ] || [ "$dst" = "@" ]; then dst="$cur"; fi
    dst="${dst//\\/}"
    dst="${dst#refs/heads/}"
    case "$dst" in ""|refs/*) continue ;; esac
    sha="$(gitq rev-parse --verify --quiet "${src}^{commit}")" || continue
    [ -n "$sha" ] || continue
    # Empty when origin/<dst> does not exist: the push would create it.
    dtip="$(gitq rev-parse --verify --quiet "refs/remotes/origin/${dst}^{commit}")" || dtip=""
    hit=""
    k=0
    while [ "$k" -lt "${#pnames[@]}" ]; do
      if gitq merge-base --is-ancestor "$sha" "${ptips[$k]}" \
         && { [ -z "$dtip" ] || gitq merge-base --is-ancestor "$dtip" "${ptips[$k]}"; }; then
        hit="${pnames[$k]}"
        break
      fi
      k=$((k + 1))
    done
    [ -n "$hit" ] || continue
    if [ "$hit" = "$def" ]; then what="the default branch (${hit})"; else what="the protected branch ${hit}"; fi
    if [ -z "$dtip" ]; then
      add_ask "git push would create ${dst} on origin from a commit that is already on ${what}, so it carries nothing new; ${dst} may be a branch whose PR was merged and deleted - confirm before re-creating it"
    else
      add_ask "git push sends ${dst} a commit that is already on ${what}, and the local origin/${dst} is fully merged into ${hit} too, so it may be a stale ref of a branch whose PR was merged and deleted (git fetch --prune would tell) - confirm before re-creating it"
    fi
  done
  return 0
}

TAG_DENY="Branching policy: pushing tags is maintainer-only"
TAG_TAIL="release tags are created by the maintainer's release script."

# DENY a push refspec that pushes tags, every matching branch, or deletes what
# may be a tag. $@ = `git push` args. The first positional is the remote, never
# a refspec (a tag named `origin` does not make `git push origin feature` a tag
# push).
#   * a refspec containing `*`, or the bare `:` ("matching"): like --all;
#   * a SOURCE (before `:`, or the whole refspec; a leading `+` stripped) that
#     names a tag: refs/tags/<x>; a name git resolves under refs/tags/
#     (`tags/v9`, `v9` when refs/tags/v9 exists, also when a branch shares the
#     name); or an object of type tag (`v9^{tag}`). Only the source: git never
#     updates a tag through an unqualified DESTINATION (`HEAD:v2.0.0` creates
#     a branch, or is refused when the remote has that tag);
#   * a DELETION (`:dst`, or any refspec with -d/--delete) of a name that is
#     neither refs/heads/<x> nor a local branch: git matches an unqualified
#     name against every remote ref, so `git push origin :v9` deletes the
#     remote TAG v9 when there is one.
# Source lookups that fail add nothing (the push itself would fail the same way).
check_push_refspecs() {
  local -a pos=()
  local t skip=0 spec src dst k=1 deleting=0 full
  for t in "$@"; do
    if [ "$skip" -eq 1 ]; then skip=0; continue; fi
    case "$t" in
      -o|--push-option|--repo|--receive-pack|--exec) skip=1 ;;
      --*) if is_abbrev "$t" --delete 4; then deleting=1; fi ;;
      -*d*) deleting=1 ;;
      -*) ;;
      *) pos+=("$t") ;;
    esac
  done
  while [ "$k" -lt "${#pos[@]}" ]; do
    spec="${pos[$k]//\\/}"
    k=$((k + 1))
    case "$spec" in
      :|*'*'*)
        deny "Branching policy: git push with the refspec ${spec} pushes every matching branch, like --all, so it would push protected branches; push one feature branch instead." ;;
    esac
    src="${spec%%:*}"
    src="${src#+}"
    if [ "$deleting" -eq 1 ] || [ -z "$src" ]; then
      dst="${spec##*:}"
      [ "$deleting" -eq 1 ] && dst="$spec"
      case "$dst" in
        refs/heads/?*) continue ;;
        refs/*) deny "Branching policy: git push deleting ${dst} on the remote is maintainer-only; only branches (refs/heads/...) may be deleted here." ;;
      esac
      if [ "$TARGET_KNOWN" -eq 1 ] && gitq show-ref --verify --quiet "refs/heads/${dst}"; then
        continue
      fi
      deny "Branching policy: git push deleting ${dst} on the remote may delete a TAG of that name (git matches an unqualified name against every remote ref), and ${dst} is not a local branch here; spell it refs/heads/${dst} to delete a branch."
    fi
    case "$src" in
      refs/tags/*) deny "${TAG_DENY} (${src}); ${TAG_TAIL}" ;;
      HEAD|@|refs/*) continue ;;
    esac
    [ "$TARGET_KNOWN" -eq 1 ] || continue
    full="$(gitq rev-parse --symbolic-full-name "$src")" || full=""
    case "$full" in
      refs/tags/*) deny "${TAG_DENY} (${src} is ${full}); ${TAG_TAIL}" ;;
    esac
    if gitq show-ref --verify --quiet "refs/tags/${src}"; then
      deny "${TAG_DENY} (${src} is the tag refs/tags/${src} in this repository); ${TAG_TAIL}"
    fi
    if [ "$(gitq cat-file -t "$src")" = "tag" ]; then
      deny "${TAG_DENY} (${src} is a tag object); ${TAG_TAIL}"
    fi
  done
  return 0
}

# DENY a `-c` / `--config-env` setting that makes a push send tags or every
# branch. $1 = `key=value` (for --config-env, `key=ENVVAR`); $2 = the option.
# Keys are case-insensitive in git, and so is this match (nocasematch).
CFG_BAD=""
check_push_config() {
  local kv="$1" key="${1%%=*}" val=""
  [[ $kv == *=* ]] && val="${kv#*=}"
  case "$key" in
    push.followtags|remote.*.push|remote.*.mirror) ;;
    push.default)
      # With -c only `matching` pushes every branch; --config-env hides the value.
      if [ "$2" = "-c" ] && [[ $val != matching ]]; then return 0; fi ;;
    *) return 0 ;;
  esac
  CFG_BAD="${2} ${key}"
}

# Record a branch change made by `git switch` / `git checkout` earlier in the
# command. $1 = the subcommand; the rest its (unquoted) arguments. A path
# checkout (`checkout -- file`, `checkout .`, `checkout <commit>`) changes no
# branch; `switch -` / `checkout -` / `@{-1}` / a variable make it unknown.
note_branch_change() {
  local sub="$1"
  shift
  local t name="" create="" want_value=0 detach=0 paths=0 pos=""
  for t in "$@"; do
    if [ "$want_value" -eq 1 ]; then create="$t"; want_value=0; continue; fi
    case "$t" in
      --) [ "$sub" = "checkout" ] && paths=1; continue ;;
      -c|-b|--create|--force-create|--orphan) want_value=1 ;;   # -C/-B match too
      --orphan=*|--create=*|--force-create=*) create="${t#*=}" ;;
      --detach|-d) detach=1 ;;
      -) [ -z "$pos" ] && pos="-" ;;
      -*) ;;
      *) [ -z "$pos" ] && [ "$paths" -eq 0 ] && pos="$t" ;;
    esac
  done
  if [ -n "$create" ]; then
    name="$create"
  elif [ "$detach" -eq 1 ]; then
    name=""
  elif [ -z "$pos" ] || [ "$paths" -eq 1 ]; then
    return 0
  else
    name="$pos"
  fi
  if [ "$name" = "-" ] || [[ $name =~ $re_dir_unknown ]] || [[ $name == *@\{* ]]; then
    SW_STATE=unknown
  elif [ -z "$name" ]; then
    SW_STATE=known
    SW_BRANCH=""
  elif [ -n "$create" ] || [ "$sub" = "switch" ] || is_protected "$name" \
       || { [ "$TARGET_KNOWN" -eq 1 ] && gitq show-ref --verify --quiet "refs/heads/${name}"; }; then
    SW_STATE=known
    SW_BRANCH="${name#refs/heads/}"
  else
    return 0      # a path or a commit: no branch change the guard has to follow
  fi
  SW_GITDIR=""
  if [ "$TARGET_KNOWN" -eq 1 ]; then
    resolve_branches
    SW_GITDIR="$repo_gitdir"
  fi
}

check_git() {
  # $@ = tokens after `git` (redirections already removed).
  local -a a=("$@") copts=()
  local i=0 n=${#a[@]} sub="" opt v named=0 cwd_known
  local gdir="" gdir_set=0 gdir_known=1
  CFG_BAD=""
  # The directory: carried `cd`s, then `env -C` of this segment.
  copts=(${CD_C[@]+"${CD_C[@]}"})
  cwd_known=$CD_KNOWN
  local d
  for d in ${NS_ENV_DIRS[@]+"${NS_ENV_DIRS[@]}"}; do
    named=1
    if [[ $d =~ $re_dir_unknown ]]; then cwd_known=0
    elif is_abs_dir "$d"; then copts=(-C "$d"); cwd_known=1
    else copts+=(-C "$d"); fi
  done
  # The git directory: exported, then this segment's GIT_DIR=, then --git-dir.
  if [ "$SH_GITDIR_SET" -eq 1 ]; then
    gdir_set=1; gdir="$SH_GITDIR"; gdir_known=$SH_GITDIR_KNOWN
  fi
  local re_gd='(^|[[:space:]])GIT_DIR=([^[:space:]]*)'
  local rest="${NS_ASSIGN:-}"
  local whole
  while [[ $rest =~ $re_gd ]]; do
    whole="${BASH_REMATCH[0]}"        # saved: the test below overwrites BASH_REMATCH
    unquote "${BASH_REMATCH[2]}"
    gdir_set=1; gdir="$UQ"; gdir_known=1
    if [ -z "$gdir" ] || [[ $gdir =~ $re_dir_unknown ]]; then gdir_known=0; fi
    rest="${rest#*"$whole"}"
  done
  if [ "${#CD_C[@]}" -gt 0 ] || [ "$CD_KNOWN" -eq 0 ] || [ "$gdir_set" -eq 1 ]; then named=1; fi

  while [ "$i" -lt "$n" ]; do
    # Unquoted before it is read: `git "--no-pager" push` and `git '-c' x=y push`.
    unquote "${a[$i]}"
    opt="$UQ"
    # `-c` with a case-sensitive test: under nocasematch the `-C` arm below
    # would also take it.
    if [ "$opt" = "-c" ]; then
      unquote "${a[$((i + 1))]:-}"
      check_push_config "$UQ" -c
      i=$((i + 2))
      continue
    fi
    case "$opt" in
      -C|--git-dir|--work-tree|--git-dir=*|--work-tree=*|--config-env|--config-env=*)
        case "$opt" in
          *=*) v="${opt#*=}"; opt="${opt%%=*}"; i=$((i + 1)) ;;
          *) unquote "${a[$((i + 1))]:-}"; v="$UQ"; i=$((i + 2)) ;;
        esac
        unquote "$v"
        v="$UQ"
        case "$opt" in
          --config-env) check_push_config "$v" --config-env; continue ;;
          --work-tree) continue ;;      # does not change which HEAD is read
        esac
        named=1
        [ -n "$v" ] || continue          # git treats `-C ""` as no change
        if [ "$opt" = "-C" ]; then
          if [[ $v =~ $re_dir_unknown ]]; then cwd_known=0
          elif is_abs_dir "$v"; then copts=(-C "$v"); cwd_known=1
          else copts+=(-C "$v"); fi
        else
          gdir_set=1; gdir="$v"; gdir_known=1
          [[ $v =~ $re_dir_unknown ]] && gdir_known=0
        fi
        ;;
      --namespace|--super-prefix) i=$((i + 2)) ;;
      -*) i=$((i + 1)) ;;
      *) sub="$opt"; i=$((i + 1)); break ;;
    esac
  done
  # An absolute git directory names the repository whatever the directory is;
  # a relative one is relative to it (git applies every -C first).
  GIT_C=(${copts[@]+"${copts[@]}"})
  if [ "$gdir_set" -eq 1 ]; then
    if [ "$gdir_known" -eq 0 ]; then
      TARGET_KNOWN=0
    else
      GIT_C+=("--git-dir=${gdir}")
      if is_abs_dir "$gdir"; then TARGET_KNOWN=1; else TARGET_KNOWN=$cwd_known; fi
    fi
  else
    TARGET_KNOWN=$cwd_known
  fi

  local -a args=()
  while [ "$i" -lt "$n" ]; do
    unquote "${a[$i]}"
    args+=("$UQ")
    i=$((i + 1))
  done

  case "$sub" in
    switch|checkout) note_branch_change "$sub" ${args[@]+"${args[@]}"} ;;
    commit|push)
      local unknown_why="" cur
      if [ "$TARGET_KNOWN" -eq 0 ]; then
        unknown_why="a cd, pushd, popd, -C, env -C or GIT_DIR on the way names its directory with a variable, ~, - or nothing, or after & or on the right of ||"
      else
        resolve_branches
        cur="$branch"
        if [ -n "$SW_STATE" ] && { [ -z "$SW_GITDIR" ] || [ "$SW_GITDIR" = "$repo_gitdir" ]; }; then
          if [ "$SW_STATE" = "unknown" ]; then
            unknown_why="an earlier git switch or checkout in this command moves to a branch it cannot read"
          fi
          cur="$SW_BRANCH"
        fi
        if [ -z "$unknown_why" ] && [ "$named" -eq 1 ] && [ -z "$repo_gitdir" ]; then
          unknown_why="git cannot read a repository in the directory it names"
        elif [ -z "$unknown_why" ] && [ -n "$cur" ] && is_protected "$cur"; then
          deny "CLAUDE.md S3.2: direct commits/pushes to ${cur} (a protected branch) are prohibited - use a feature branch and a PR."
        fi
      fi
      if [ -n "$unknown_why" ]; then
        add_ask "the guard could not tell which branch this git ${sub} runs on (${unknown_why}); confirm it is not a protected branch"
      fi
      ;;
  esac

  if [ "$sub" = "push" ]; then
    if [ -n "$CFG_BAD" ]; then
      deny "Branching policy: git ${CFG_BAD} changes what git push sends (tags, every branch or a mirror); push one feature branch with an explicit refspec instead."
    fi
    local t dst
    for t in ${args[@]+"${args[@]}"}; do
      case "$t" in
        --force-with-lease*|--force-if-includes|--no-force-if-includes) ;;
        --force|--force=*)
          deny "CLAUDE.md S3.2: force-pushes are prohibited (use --force-with-lease if truly needed)." ;;
        --*)
          # git takes any unique prefix of a long option (see is_abbrev).
          if is_abbrev "$t" --tags 4 || is_abbrev "$t" --follow-tags 5; then
            deny "${TAG_DENY} (${t}); ${TAG_TAIL}"
          elif is_abbrev "$t" --all 4 || is_abbrev "$t" --branches 3 || is_abbrev "$t" --mirror 3; then
            deny "Branching policy: git push ${t} would push protected branches; push one feature branch instead."
          elif is_abbrev "$t" --prune 5; then
            deny "Branching policy: git push ${t} (--prune) deletes the remote branches its refspecs match that have no local counterpart, so it would push protected branches' deletion; push one feature branch instead."
          fi
          ;;
        -*)
          case "$t" in
            -*f*) deny "CLAUDE.md S3.2: force-pushes are prohibited (use --force-with-lease if truly needed)." ;;
          esac
          ;;
        +*)
          deny "CLAUDE.md S3.2: force-pushes are prohibited (a +refspec forces the update)." ;;
        tag)
          # `git push origin tag v9` is git's spelling of refs/tags/v9.
          deny "${TAG_DENY} (git push ... tag <name>); ${TAG_TAIL}" ;;
        *)
          dst="${t##*:}"
          # bash drops the backslash in `q\a`, and a ref name cannot contain
          # one, so deleting it loses nothing; `refs//heads/qa` collapses.
          dst="${dst//\\/}"
          while [[ $dst == *//* ]]; do dst="${dst//\/\//\/}"; done
          case "$dst" in
            refs/tags/*)
              deny "${TAG_DENY} (a refs/tags/* destination); ${TAG_TAIL}" ;;
          esac
          if is_protected "$dst"; then
            deny "CLAUDE.md S3.2: pushing to or deleting ${dst#refs/heads/} (protected: ${protected_display}) is prohibited - use a feature branch and a PR."
          fi
          ;;
      esac
    done
    check_push_refspecs ${args[@]+"${args[@]}"}
    # Only reached when no deny rule above fired (deny exits). It needs the
    # target's refs, so an unknown directory (already an ask) skips it.
    if [ "$TARGET_KNOWN" -eq 1 ]; then
      check_recreated_branch ${args[@]+"${args[@]}"}
    fi
  fi

  if [ "$sub" = "tag" ]; then
    local list_mode=0 j m=${#args[@]} expect_value=0 bad=""
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
