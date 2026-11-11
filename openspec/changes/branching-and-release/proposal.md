# Introduce the dev / qa / main branch model and its agent-side guard

## Why

The repository has one long-lived branch (`claude/civ6-alphazero-implementation-vMnun`)
and no release line. The local-first plan
(`kb/specs/20261111-local-first-and-branching-plan.md` §3.1) introduces three:
`dev` (integration, the renamed default), `qa` (release candidates) and `main`
(tagged releases), with only the maintainer able to merge into `qa`/`main` or
create `v*` tags. Charter amendment A-1 (PROJECT_CHARTER.md §4.6) states that
this rule is enforced by **identity** (a machine user with no ruleset bypass),
and that until that identity is provisioned the PreToolUse guard is the only
barrier: an accident net, not a control.

The guard as it stood could not serve even as that net:

- it protected only `main` and the single default branch, so `qa` and the
  future `dev` were unprotected, and switching `CIV6AZ_DEFAULT_BRANCH` to `dev`
  ahead of the rename would have silently unprotected the current default;
- it ran only for the Bash tool, while this Windows workstation also gives the
  agent a PowerShell tool;
- it could be bypassed by any wrapper (`pwsh -c`, `cmd /c`, `bash -c`, `iex`,
  `&`), as the standards audit's finding S7 recorded;
- it had no rule at all for tags, `gh pr merge`, releases, rulesets or
  write-capable `gh api` calls;
- it denied every over-cap run outright, with no route for an approved run, and
  had no notion of a run it could not price;
- it failed open when its parser existed on PATH but did not work (for example
  a Windows Store `python3` alias with no Store Python behind it), and its own
  tests skipped whenever `jq` was absent --
  that is, on the very workstation the PowerShell rules are for.

CI had the same blind spot: the four gating workflows ran on push only for the
old default and `main`, and `cancel-in-progress: true` let a newer push cancel a
run on a protected branch, leaving a state with no CI record.

## What Changes

This change is delivered in four releases (plan R1-R4). **R1 (this PR):**

- **Guard** (`.claude/hooks/pretooluse_guard.sh`), registered for
  `Bash|PowerShell`: protection is driven by a protected-branch LIST
  (`CIV6AZ_PROTECTED_BRANCHES`, always including `dev qa main`), not by the
  default branch; wrappers are unwrapped before matching; tag pushes, tag
  creation, protected-branch deletion, `--all`/`--mirror`, `gh pr merge`,
  `gh release`, `gh repo edit`, `gh ruleset` and write-capable `gh api` are
  denied; `--approval_ref`, `--preset production`, `--config` on a game runner
  and unknown game runners **ask** (prompt the user) instead of being allowed or
  denied; an unparseable payload is denied.
- **Settings**: `CIV6AZ_DEFAULT_BRANCH=dev`,
  `CIV6AZ_PROTECTED_BRANCHES="dev qa main claude/civ6-alphazero-implementation-vMnun"`,
  matching `Bash(...)`/`PowerShell(...)` deny rules, and
  `enableAllProjectMcpServers: false`.
- **Workflows** (`ci`, `codeql`, `security-scan`, `scorecard`): push triggers on
  `dev`, `qa`, `main` and the old default; `cancel-in-progress` is false on every
  protected branch.
- **Gates**: the workflow-pinning regression test allows the protected list and
  asserts both trigger coverage and non-cancellation; the coverage-floor ratchet
  reads its base candidates from the protected list and, on a push to a
  protected branch, compares against that branch first.

**R2:**

- **Branch Direction** (`scripts/check_branch_direction.py`,
  `.github/workflows/branch-policy.yml`): the S3.4 rules over git ancestry --
  which heads each of dev/qa/main accepts, promotion and release need
  `origin/main` as an ancestor, back-merges carry only released commits plus
  update merges, hotfixes carry no unreleased commits, evidence touches only
  `docs/evidence/**`, rc fixes are patch-equivalent cherry-picks. The judge runs
  from the default branch, so a PR cannot rewrite its own check.
- **Rulesets** (`.github/rulesets/*.json`) in the REST export shape, applied by
  the maintainer with `scripts/maintainer/branch_cutover.sh`; back-merges are
  merged with `scripts/maintainer/merge_backmerge.sh`.
- **Release mechanics**: `release.yml` gains `verify-tag` (version, ancestry,
  final-tag rule) and `wait-for-ci` (push checks on the tagged commit, PR-only
  checks on the producing PR's head); the publish job needs both. An rc build is
  stamped PEP 440 `X.Y.ZrcN`. `scripts/maintainer/tag_release.py` pre-checks
  every rule and refuses commits without the R2 release workflow.

Later releases (tracked in `tasks.md`): R3 the cut-over (rename, rulesets
applied); R4 drop the old default from the protected list.

## Impact

- Affected specs: `agent-harness` (guard, permission governance),
  `validation-gates` (protected-branch CI coverage).
- Affected code: `.claude/hooks/pretooluse_guard.sh`, `.claude/settings.json`,
  `.github/workflows/{ci,codeql,security-scan,scorecard}.yml`,
  `tests/governance/test_pretooluse_guard.py`,
  `tests/regression/{branch_policy.py,test_workflow_pinning_regression.py,test_coverage_floors_ratchet.py}`.
- Behaviour visible to the agent: commands that previously ran silently now
  prompt (`--approval_ref`, `--config` on a runner, e.g.
  `python -m alpha_zero.orchestrator --config ... --dry_run`, and unknown runners
  such as `python -m alpha_zero.distributed.coordinator --dry_run`); `gh release`
  and `gh ruleset` are denied even for read-only subcommands.
- Not a control: a pre-approved interpreter (`python -c`) can still call the
  GitHub API directly. The machine-user identity is the control; provisioning it
  is a maintainer action recorded in `.claude/README.md`.
