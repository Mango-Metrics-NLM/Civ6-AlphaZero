#!/usr/bin/env bash
# Merge a back-merge PR with a MERGE COMMIT (plan S3.1 / S3.2).
#
# dev's ruleset must allow both squash (features) and merge (back-merges),
# because GitHub cannot pick a merge method by head branch. A squashed
# back-merge loses main's ancestry; the promotion check and tag_release then
# refuse the next release. So back-merges are merged only through this script,
# which asserts the head is backmerge/*, the base is dev or qa, the head lives
# in this repository and the PR is open, then runs
#   gh pr merge <pr> --merge --match-head-commit <head sha>
# (--match-head-commit: nothing pushed after the check is merged unseen).
#
# MAINTAINER ONLY. Usage:
#   scripts/maintainer/merge_backmerge.sh [--execute] <pr-number>
# Without --execute the merge command is printed, not run. With it, the script
# refuses inside a Claude Code session (CLAUDECODE set).
#
# Test seams: GH names the gh executable; CIV6AZ_REPO overrides the repository.

set -euo pipefail

GH="${GH:-gh}"
EXECUTE=0
PR=""

die() {
  echo "merge_backmerge: $*" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --execute) EXECUTE=1 ;;
    -h | --help)
      sed -n '2,18p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      [[ -z $PR ]] || die "one PR number only"
      PR="$1"
      ;;
  esac
  shift
done

[[ $PR =~ ^[1-9][0-9]*$ ]] || die "usage: merge_backmerge.sh [--execute] <pr-number>"
if [[ $EXECUTE == 1 && -n ${CLAUDECODE:-} ]]; then
  die "refusing --execute inside a Claude Code session: merging into dev/qa is maintainer-only"
fi

repo_args=()
if [[ -n ${CIV6AZ_REPO:-} ]]; then
  repo_args=(--repo "$CIV6AZ_REPO")
fi

info="$("$GH" pr view "$PR" ${repo_args[@]+"${repo_args[@]}"} --json headRefName,baseRefName,isCrossRepository,state,headRefOid \
  --jq '[.headRefName, .baseRefName, (.isCrossRepository | tostring), .state, .headRefOid] | @tsv')" ||
  die "could not read PR #$PR"
IFS=$'\t' read -r head base cross state oid <<<"$info"

[[ $head == backmerge/* ]] || die "PR #$PR head is '$head', not backmerge/*: merge it through the normal UI (squash)"
[[ $base == dev || $base == qa ]] || die "PR #$PR targets '$base'; back-merges go into dev or qa"
[[ $cross == false ]] || die "PR #$PR head is in another repository"
[[ $state == OPEN ]] || die "PR #$PR is $state, not OPEN"
[[ $oid =~ ^[0-9a-f]{40}$ ]] || die "PR #$PR head sha '$oid' is not a commit sha"

cmd=("$GH" pr merge "$PR" ${repo_args[@]+"${repo_args[@]}"} --merge --match-head-commit "$oid")
printf '+ %s\n' "${cmd[*]}"
if [[ $EXECUTE == 1 ]]; then
  "${cmd[@]}"
else
  echo "dry run: pass --execute to merge"
fi
