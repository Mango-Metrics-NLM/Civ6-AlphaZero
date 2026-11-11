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

## The judge comes from the default branch

On `pull_request` the checkout is the PR's merge ref, so running
`scripts/check_branch_direction.py` from the workspace would let a PR edit the
script that judges it. `branch-policy.yml` runs the copy on the default branch
(protected, reviewed). The one exception is the PR that introduces the script,
before it exists on the default branch; the check is not required until the
maintainer adds it after that merge (cut-over step 5).

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

## integration_id is omitted

A required check can be pinned to the app that produces it. The GitHub Actions
app id could not be confirmed from the documentation, and a wrong id makes
every required check unsatisfiable, so contexts carry no `integration_id`;
`wait-for-ci` instead counts only check runs whose `app.slug` is
`github-actions`.

## Windows

The maintainer runs the shell scripts in Git Bash. Two defects found by the
tests: MSYS rewrites a `<rev>:<path>` argument as a path list (so the scripts
use `git ls-tree`), and `gh` output may end lines with CR.
