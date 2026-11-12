#!/usr/bin/env bash
# Branch cut-over runbook (plan S3.7): old default -> dev / qa / main.
#
# MAINTAINER ONLY. Every GitHub write here needs repository admin and changes
# who can merge what. Agents never run this: with --execute it refuses inside a
# Claude Code session (CLAUDECODE is set) and when authenticated as the agent
# machine user. The refusal is an accident net; the control is identity.
#
# Usage:
#   scripts/maintainer/branch_cutover.sh [--execute] STEP [STEP...]
#
# Without --execute every write is printed ("+ gh api -X POST ...") and not
# run; reads (gh api GETs, git fetch) still run so the preview is real.
#
# Steps, in runbook order (one sitting; "phase1" then merge T6, then "phase2"):
#   snapshot              1. save repo settings + rulesets to $OUT_DIR/before*.json
#   create-branches       2. create qa and main at c214daa (== v2.0.0^{commit})
#   rename                3. rename the default branch to dev
#   rulesets              4. POST all six rulesets WITHOUT Branch Direction or Release Evidence
#   add-branch-direction  5. after T6 is merged into dev: PUT Branch Direction into dev/qa/main
#                            (main still without Release Evidence)
#                            (an accident net, not a control: a same-repo PR can rewrite
#                            its own workflow; maintainer-only merge into qa/main is the control)
#   settings              6. delete_branch_on_merge, allow_update_branch, allow_auto_merge on; rebase off
#   housekeeping          7. close #95 and #120, @dependabot rebase, update-branch #106 #69 #49 #32
#   verify                8. automated checks + the manual checklist
#   local                 9. retarget this clone (branch -m, -u, set-head, fetch --prune)
#   add-release-evidence  9a. after T14 is merged into dev: PUT the full main ruleset
#                            (adds Release Evidence); then confirm a draft PR into main
#                            shows it as Expected
#   phase1 = snapshot create-branches rename rulesets
#   phase2 = add-branch-direction settings housekeeping verify
#   rollback              delete the rulesets, rename dev back, delete qa/main, restore settings
#
# Prerequisites (step 0, checked by hand): LF-G1 key revoked; the civ6az-agent
# machine user exists; T3a and T4 (R1) are merged into the old default branch;
# v2.0.0 is tagged on c214daa.
#
# Test seams: GH, GIT and PYTHON name the executables; CIV6AZ_REPO,
# CIV6AZ_RELEASE_COMMIT, CIV6AZ_RELEASE_TAG and CIV6AZ_CUTOVER_DIR override the
# defaults. tests/scripts/test_maintainer_scripts.py drives every step against
# a fake gh and a temporary git repository.

set -euo pipefail

GH="${GH:-gh}"
GIT="${GIT:-git}"
PYTHON="${PYTHON:-python}"
OLD_DEFAULT="claude/civ6-alphazero-implementation-vMnun"
RELEASE_COMMIT="${CIV6AZ_RELEASE_COMMIT:-c214daa}"
RELEASE_TAG="${CIV6AZ_RELEASE_TAG:-v2.0.0}"
OUT_DIR="${CIV6AZ_CUTOVER_DIR:-cutover}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER="$HERE/rulesets.py"
RULESETS=(dev qa main qa-main-merge release-tags legacy-name)
BRANCH_DIRECTION_RULESETS=(dev qa main)
RELEASE_EVIDENCE="Release Evidence"
CLOSE_PRS=(95 120)
UPDATE_BRANCH_PRS=(106 69 49 32)
EXECUTE=0
REPO=""
CR="$(printf '\r')"

die() {
  echo "branch_cutover: $*" >&2
  exit 2
}
say() { printf '==> %s\n' "$*"; }

# A write: always printed, run only with --execute.
mutate() {
  printf '+ %s\n' "$*"
  if [[ $EXECUTE == 1 ]]; then
    "$@"
  fi
}

resolve_repo() {
  if [[ -n ${CIV6AZ_REPO:-} ]]; then
    REPO="$CIV6AZ_REPO"
  else
    REPO="$("$GH" repo view --json nameWithOwner --jq .nameWithOwner)"
  fi
  [[ $REPO =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die "not a repository name: '$REPO'"
}

refuse_agent() {
  if [[ -n ${CLAUDECODE:-} ]]; then
    die "refusing --execute inside a Claude Code session: the cut-over is maintainer-only. Run it from your own terminal."
  fi
  local login
  login="$("$GH" api user --jq .login)"
  if [[ $login == civ6az-agent ]]; then
    die "authenticated as the agent machine user (civ6az-agent); the cut-over needs the maintainer's own credentials"
  fi
}

maintainer_id() {
  local id
  id="$("$GH" api user --jq .id)"
  [[ $id =~ ^[1-9][0-9]*$ ]] || die "could not read the authenticated user's id (got '$id')"
  printf '%s' "$id"
}

# --- 1 ---------------------------------------------------------------------
step_snapshot() {
  say "1. snapshot repository settings and rulesets to $OUT_DIR/"
  mkdir -p "$OUT_DIR"
  "$GH" api "repos/$REPO" >"$OUT_DIR/before.json"
  "$GH" api "repos/$REPO/rulesets" >"$OUT_DIR/before-rulesets.json"
}

# --- 2 ---------------------------------------------------------------------
step_create_branches() {
  say "2. create qa and main at $RELEASE_COMMIT"
  "$GIT" fetch --quiet --tags origin
  local sha tagged
  sha="$("$GIT" rev-parse --verify "${RELEASE_COMMIT}^{commit}")" || die "$RELEASE_COMMIT is not a commit here"
  tagged="$("$GIT" rev-parse --verify "${RELEASE_TAG}^{commit}" 2>/dev/null)" ||
    die "$RELEASE_TAG is not tagged yet (prerequisite: tag $RELEASE_TAG on $RELEASE_COMMIT first)"
  [[ $sha == "$tagged" ]] || die "$RELEASE_COMMIT ($sha) is not ${RELEASE_TAG}^{commit} ($tagged)"
  local branch
  for branch in qa main; do
    if "$GH" api "repos/$REPO/branches/$branch" >/dev/null 2>&1; then
      die "branch $branch already exists on $REPO; refusing to move it"
    fi
  done
  for branch in qa main; do
    mutate "$GH" api -X POST "repos/$REPO/git/refs" -f "ref=refs/heads/$branch" -f "sha=$sha"
  done
}

# --- 3 ---------------------------------------------------------------------
step_rename() {
  say "3. rename the default branch $OLD_DEFAULT -> dev"
  local current
  current="$("$GH" api "repos/$REPO" --jq .default_branch)"
  [[ $current == "$OLD_DEFAULT" ]] || die "default branch is '$current', expected '$OLD_DEFAULT'"
  mutate "$GH" api -X POST "repos/$REPO/branches/$OLD_DEFAULT/rename" -f "new_name=dev"
}

# --- 4 ---------------------------------------------------------------------
step_rulesets() {
  say "4. POST the six rulesets (no Branch Direction, no Release Evidence)"
  local id name body
  id="$(maintainer_id)"
  mkdir -p "$OUT_DIR"
  for name in "${RULESETS[@]}"; do
    body="$OUT_DIR/ruleset-$name.json"
    "$PYTHON" "$RENDER" render "$name" --maintainer-id "$id"       --without-check "Branch Direction" --without-check "$RELEASE_EVIDENCE" >"$body"
    mutate "$GH" api -X POST "repos/$REPO/rulesets" --input "$body"
  done
}

# --- 5 ---------------------------------------------------------------------
step_add_branch_direction() {
  say "5. PUT Branch Direction into dev, qa and main (main still without Release Evidence)"
  "$GIT" fetch --quiet origin dev
  # `git ls-tree`, not `git cat-file -e origin/dev:<path>`: Git Bash's MSYS
  # layer rewrites a "<rev>:<path>" argument as a path list on Windows.
  local required
  for required in scripts/check_branch_direction.py .github/workflows/branch-policy.yml; do
    [[ -n $("$GIT" ls-tree --name-only origin/dev -- "$required") ]] ||
      die "T6 (R2) is not merged into dev yet: origin/dev has no $required"
  done
  local id listing name rid body
  id="$(maintainer_id)"
  listing="$OUT_DIR/rulesets-listing.json"
  mkdir -p "$OUT_DIR"
  "$GH" api "repos/$REPO/rulesets" >"$listing"
  for name in "${BRANCH_DIRECTION_RULESETS[@]}"; do
    rid="$("$PYTHON" "$RENDER" id-of "$name" <"$listing")"
    body="$OUT_DIR/ruleset-$name-full.json"
    # Release Evidence waits for 9a: until T14's job is on the branch a PR runs
    # from, the check never reports and main would be unmergeable.
    "$PYTHON" "$RENDER" render "$name" --maintainer-id "$id"       --without-check "$RELEASE_EVIDENCE" >"$body"
    mutate "$GH" api -X PUT "repos/$REPO/rulesets/$rid" --input "$body"
  done
}

# --- 9a --------------------------------------------------------------------
step_add_release_evidence() {
  say "9a. PUT Release Evidence into main"
  "$GIT" fetch --quiet origin dev
  local required
  for required in scripts/verify_release_evidence.py scripts/release_play_path.txt; do
    [[ -n $("$GIT" ls-tree --name-only origin/dev -- "$required") ]] ||
      die "T14 is not merged into dev yet: origin/dev has no $required"
  done
  local id listing rid body
  id="$(maintainer_id)"
  listing="$OUT_DIR/rulesets-listing.json"
  mkdir -p "$OUT_DIR"
  "$GH" api "repos/$REPO/rulesets" >"$listing"
  rid="$("$PYTHON" "$RENDER" id-of main <"$listing")"
  body="$OUT_DIR/ruleset-main-release-evidence.json"
  "$PYTHON" "$RENDER" render main --maintainer-id "$id" >"$body"
  mutate "$GH" api -X PUT "repos/$REPO/rulesets/$rid" --input "$body"
  cat <<'EOF'
Manual check: open a draft PR into main and confirm "Release Evidence" is listed
as Expected. A hotfix/* PR into main cannot report it until main carries the job
(the first release after T14); see docs/runbooks/branching-and-release.md.
EOF
}

# --- 6 ---------------------------------------------------------------------
step_settings() {
  say "6. repository settings"
  mutate "$GH" api -X PATCH "repos/$REPO" \
    -F delete_branch_on_merge=true -F allow_update_branch=true \
    -F allow_auto_merge=true -F allow_rebase_merge=false
}

# --- 7 ---------------------------------------------------------------------
step_housekeeping() {
  say "7. stale Dependabot PRs and branch updates"
  mutate "$GH" pr close 95 --repo "$REPO" --comment "codecov-action is unused: the Codecov upload was removed on 2026-09-02."
  mutate "$GH" pr close 120 --repo "$REPO" --comment "The pytorch base image was replaced by NGC 26.10."
  local pr skip
  while read -r pr; do
    pr="${pr%"${CR}"}" # gh on Windows may end lines with CRLF
    [[ -n $pr ]] || continue
    skip=0
    for closed in "${CLOSE_PRS[@]}"; do [[ $pr == "$closed" ]] && skip=1; done
    [[ $skip == 1 ]] && continue
    mutate "$GH" pr comment "$pr" --repo "$REPO" --body "@dependabot rebase"
  done < <("$GH" pr list --repo "$REPO" --state open --author app/dependabot --json number --jq '.[].number')
  for pr in "${UPDATE_BRANCH_PRS[@]}"; do
    mutate "$GH" pr update-branch "$pr" --repo "$REPO"
  done
}

# --- 8 ---------------------------------------------------------------------
step_verify() {
  say "8. verify"
  local failed=0 current stray names name
  current="$("$GH" api "repos/$REPO" --jq .default_branch)"
  if [[ $current != dev ]]; then
    echo "FAIL: default branch is '$current', not dev"
    failed=1
  fi
  stray="$("$GH" pr list --repo "$REPO" --state open --limit 500 --json number,baseRefName \
    --jq '.[] | select(.baseRefName != "dev") | "#\(.number) -> \(.baseRefName)"')"
  if [[ -n $stray ]]; then
    echo "FAIL: open PRs whose base is not dev:"
    echo "$stray"
    failed=1
  fi
  names="$("$GH" api "repos/$REPO/rulesets" --jq '.[].name')"
  for name in "${RULESETS[@]}"; do
    if ! grep -qx "$name" <<<"$names"; then
      echo "FAIL: ruleset $name is not applied"
      failed=1
    fi
  done
  cat <<'EOF'
Manual checks (each needs a second identity or a deliberate failure):
  [ ] a chore/x -> qa PR fails Branch Direction
  [ ] a direct push to dev is rejected
  [ ] civ6az-agent cannot merge into qa
  [ ] civ6az-agent cannot create a v* tag
EOF
  return "$failed"
}

# --- 9 ---------------------------------------------------------------------
step_local() {
  say "9. retarget this clone"
  if "$GIT" show-ref --verify --quiet "refs/heads/$OLD_DEFAULT"; then
    mutate "$GIT" branch -m "$OLD_DEFAULT" dev
  fi
  mutate "$GIT" fetch --prune origin
  mutate "$GIT" branch -u origin/dev dev
  mutate "$GIT" remote set-head origin -a
}

# --- rollback --------------------------------------------------------------
step_rollback() {
  say "rollback: delete rulesets, rename dev back, delete qa/main, restore settings"
  local listing name rid key value
  listing="$OUT_DIR/rulesets-listing.json"
  mkdir -p "$OUT_DIR"
  "$GH" api "repos/$REPO/rulesets" >"$listing"
  for name in "${RULESETS[@]}"; do
    if rid="$("$PYTHON" "$RENDER" id-of "$name" <"$listing" 2>/dev/null)"; then
      mutate "$GH" api -X DELETE "repos/$REPO/rulesets/$rid"
    fi
  done
  mutate "$GH" api -X POST "repos/$REPO/branches/dev/rename" -f "new_name=$OLD_DEFAULT"
  mutate "$GH" api -X DELETE "repos/$REPO/git/refs/heads/qa"
  mutate "$GH" api -X DELETE "repos/$REPO/git/refs/heads/main"
  [[ -f $OUT_DIR/before.json ]] || die "no $OUT_DIR/before.json snapshot to restore settings from"
  local args=()
  for key in delete_branch_on_merge allow_update_branch allow_auto_merge allow_rebase_merge; do
    value="$("$PYTHON" -c 'import json,sys; v=json.load(open(sys.argv[1]))[sys.argv[2]]; print(str(v).lower())' "$OUT_DIR/before.json" "$key")"
    args+=(-F "$key=$value")
  done
  mutate "$GH" api -X PATCH "repos/$REPO" "${args[@]}"
  echo "Then revert R1 (T4) with a PR into $OLD_DEFAULT."
}

run_step() {
  case "$1" in
    snapshot) step_snapshot ;;
    create-branches) step_create_branches ;;
    rename) step_rename ;;
    rulesets) step_rulesets ;;
    add-branch-direction) step_add_branch_direction ;;
    add-release-evidence) step_add_release_evidence ;;
    settings) step_settings ;;
    housekeeping) step_housekeeping ;;
    verify) step_verify ;;
    local) step_local ;;
    rollback) step_rollback ;;
    phase1) for s in snapshot create-branches rename rulesets; do run_step "$s"; done ;;
    phase2) for s in add-branch-direction settings housekeeping verify; do run_step "$s"; done ;;
    *) die "unknown step '$1' (see the header of this script)" ;;
  esac
}

main() {
  local steps=()
  while (($#)); do
    case "$1" in
      --execute) EXECUTE=1 ;;
      -h | --help)
        sed -n '2,42p' "${BASH_SOURCE[0]}"
        return 0
        ;;
      *) steps+=("$1") ;;
    esac
    shift
  done
  ((${#steps[@]})) || die "no step given (see --help)"
  resolve_repo
  if [[ $EXECUTE == 1 ]]; then
    refuse_agent
  else
    say "dry run: writes are printed, not run (pass --execute to apply)"
  fi
  local step
  for step in "${steps[@]}"; do
    run_step "$step"
  done
}

main "$@"
