# Design notes -- branching-and-release (R1)

## The list, not the default, drives protection

`CIV6AZ_DEFAULT_BRANCH` becomes `dev` in this PR, before the rename. If the
guard still derived protection from the default branch, the current default
(`claude/civ6-alphazero-implementation-vMnun`) would become unprotected the
moment the settings landed. Protection is therefore the union of `dev qa main`
(hardcoded, so an unset variable never means "nothing is protected"), the
resolved default branch, and `CIV6AZ_PROTECTED_BRANCHES`. The regression tests
read the same list through one helper (`tests/regression/branch_policy.py`).

## Deny vs ask

Deny is for actions an agent never takes (commit/push to a protected branch,
tags, merges, releases, rulesets, `gh api` writes, over-cap runs with no
approval). Ask is for actions that may be legitimate but that the guard cannot
judge: `--approval_ref` is recorded by the runner and never verified, so a
human confirms it; `--preset production`, `--config` and unknown runners cannot
be priced from the command line. A deny anywhere in a compound command wins
over an ask. Output schema: `hookSpecificOutput.permissionDecision` of `"deny"`
or `"ask"` with `permissionDecisionReason`, exit 0 (Claude Code hooks reference,
"PreToolUse decision control").

## Unwrapping

Segments are split on every separator bash, cmd and PowerShell use, including a
single `&` -- which is at once bash backgrounding, cmd's separator and
PowerShell's call operator. Because splitting ignores quotes, a wrapped
command's inner separators are already split; unwrapping then only strips the
wrapper prefix (`pwsh -c`, `cmd /c`, `bash -c`, `iex`, ...) and leading quotes,
repeatedly, so nested wrappers unwrap too. An encoded PowerShell command
cannot be inspected and asks.

## PowerShell coverage

The hook matcher is `Bash|PowerShell`; the PowerShell tool sends its command in
`tool_input.command` (the hooks reference's Windows example reads it from
there). Permission rules use the documented `PowerShell(...)` form with the
same shape as `Bash(...)`. The guard is the primary PowerShell net; the rules
are a second layer.

## Fail closed on the parser

A parser has to exit 0 to count. A parser that is on PATH but fails -- for
example the Windows Store `python3` alias on a machine without Store Python --
produced an empty command under the old `|| true`, and the guard allowed
everything (found by reading the code; not reproduced on this workstation,
whose `python3` works). jq, then python3, then python are tried; if
none parses, the guard denies.

## Performance

The hook runs before every shell tool call under a 10 s timeout, and process
creation is expensive under MSYS bash on Windows. The rewrite uses bash
builtins (`[[ =~ ]]`, `read -ra`, globals instead of `$(...)`), and resolves git
state only when a segment is a git command.

# Design notes -- R2

## Branch Direction is an accident net, not a control

A `pull_request` workflow runs the workflow file from the PR's own merge ref, so
a same-repo PR can rewrite the step, or add any job named `Branch Direction`,
and turn the required context green. `pull_request_target` does not fix that
for anyone holding Write: the job name, not the workflow file, is what the
ruleset matches. So the check is an accident net -- it stops wrong-direction PRs
opened by mistake -- and the control for qa and main is identity: only the
maintainer can merge into them (`qa-main-merge`), and `tag_release` and
`release.yml` re-check ancestry at release time.

Within that limit: the workflow runs the script from the default branch rather
than the PR's checkout, so an ordinary PR that edits the script is not judged by
its own edit, and there is no fallback -- if the default branch has no judge,
the job fails. The job runs only for open PRs; an `edited` event on a merged PR
would re-judge it against refs the merge moved.

## Path filters do not detect renames

`git diff --name-only` reports a rename as its destination only, so moving
`alpha_zero/x.py` into `docs/evidence/` would pass both the evidence-only rule
and the final-tag rule. Both use `--no-renames`.

## wait-for-ci reads the run that gated the merge

PR-scoped checks are read from the newest Actions run completed at or before
the producing PR's `merged_at`, and the PR's base must be `qa` (rc) or `main`
(final). They cannot change after the merge, so they are judged once: a missing
`Release Evidence` fails at once and names T14 / maintainer action 9a instead of
polling for 95 minutes. Push checks are polled; a transient API failure (5xx,
connection) is retried with bounded exponential backoff.

## Hotfixes refuse qa-only commits (ruling)

Plan S3.4 says a hotfix may carry no commit that comes only from dev. The check
also refuses commits reachable from `origin/qa` but not `origin/main`. This is
deliberate and stricter than the plan: qa-only commits are promotion merges,
evidence merges and rc fixes, and a hotfix must not ship unreleased rc work.

## REST API version pin

API version 2026-03-10 removed `merge_commit_sha` from pull request payloads.
`wait-for-ci` identifies the producing PR by exactly that field, so every call
pins `X-GitHub-Api-Version: 2022-11-28`, and a payload without the field fails
closed instead of matching nothing.

## Lineage is checked on content

`tag_release.py` must refuse commits older than the LF-R2 merge, but that
merge's SHA does not exist when this change is written. The check is therefore
the property itself: the tagged commit's `release.yml` gates its
`contents: write` job on `wait-for-ci` -- the workflow a tag push runs is the
one in the tagged commit. Once merged, the maintainer may record the SHA in
`CIV6AZ_R2_MERGE_SHA` and the ancestry check is added on top.

## integration_id is pinned

Every required check carries `integration_id` 15368, the GitHub Actions app,
confirmed from this repository's own check runs (`app.id=15368,
slug=github-actions`). Without it, a commit status of the same name from any
token with `statuses: write` satisfies the rule. `wait-for-ci` also counts only
runs whose `app.slug` is `github-actions`.

## Windows

The maintainer runs the shell scripts in Git Bash. Two defects found by the
tests: MSYS rewrites a `<rev>:<path>` argument as a path list (so the scripts
use `git ls-tree`), and `gh` output may end lines with CR.
