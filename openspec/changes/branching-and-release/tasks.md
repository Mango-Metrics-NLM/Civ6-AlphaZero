# Tasks

## 1. R1 -- guard, settings, workflow triggers (LF-G6)
- [x] 1.1 Guard: protected-branch list drives protection (`dev qa main` always)
- [x] 1.2 Guard: unwrap `pwsh -c`, `powershell -Command`, `cmd /c`, `bash -c`, `iex`, `&`
- [x] 1.3 Guard: deny tag pushes/creation, protected-branch deletion, `--all`/`--mirror`
- [x] 1.4 Guard: deny `gh pr merge`, `gh release`, `gh repo edit`, `gh ruleset`, write-capable `gh api`
- [x] 1.5 Guard: ask on `--approval_ref`, `--preset production`, `--config` on a runner, unknown runners
- [x] 1.6 Guard: deny an unparseable payload; tests run with jq or python
- [x] 1.7 Settings: default `dev`, protected list, `Bash|PowerShell` matcher, deny rules, MCP opt-in off
- [x] 1.8 Workflows: push on protected branches, no cancellation on protected branches
- [x] 1.9 Regression gates read the protected list; ratchet prefers `GITHUB_REF_NAME` on protected pushes
- [x] 1.10 Negative controls run for the guard, the workflow checks and the ratchet
- [ ] 1.11 Maintainer: provision `civ6az-agent` and its PAT (`.claude/README.md`, "Agent identity")

## 2. R2 -- branch direction, rulesets, release scripts
- [x] 2.1 `scripts/check_branch_direction.py` + `branch-policy.yml` (judge runs from the default branch)
- [x] 2.2 `.github/rulesets/{dev,qa,main,qa-main-merge,release-tags,legacy-name}.json` (no Release Evidence until T14)
- [x] 2.3 `scripts/maintainer/{rulesets.py,branch_cutover.sh,merge_backmerge.sh}` (dry run by default)
- [x] 2.4 `scripts/release_version.py`, `scripts/release_checks.py`, `scripts/maintainer/tag_release.py`
- [x] 2.5 `release.yml`: `verify-tag` + `wait-for-ci` gate the publish job
- [x] 2.6 Gates: `test_branch_rulesets_regression.py`, `test_release_workflow_regression.py`, `tests/scripts/test_{check_branch_direction,maintainer_scripts,release_scripts}.py`; negative controls recorded in each module
- [ ] 2.7 Maintainer: record the LF-R2 merge SHA (`CIV6AZ_R2_MERGE_SHA`) after merge; release-tag lineage is checked on content until then

## 3. R3 -- cut-over
- [ ] 3.1 Rename the default branch to `dev`; apply rulesets

## 4. R4 -- retire the old default
- [ ] 4.1 Drop `claude/civ6-alphazero-implementation-vMnun` from the protected list and workflow triggers
