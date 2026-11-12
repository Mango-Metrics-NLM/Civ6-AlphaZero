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
- [x] 2.1 `scripts/check_branch_direction.py` + `branch-policy.yml` (an accident net; judge fetched from the default branch, no PR-copy fallback)
- [x] 2.2 `.github/rulesets/{dev,qa,main,qa-main-merge,release-tags,legacy-name}.json` (`main.json` gains Release Evidence in T14, task 5.3)
- [x] 2.3 `scripts/maintainer/{rulesets.py,branch_cutover.sh,merge_backmerge.sh}` (dry run by default)
- [x] 2.4 `scripts/release_version.py`, `scripts/release_checks.py`, `scripts/maintainer/tag_release.py`
- [x] 2.5 `release.yml`: `verify-tag` + `wait-for-ci` gate the publish job
- [x] 2.6a Review fix round 1: `--no-renames` path filters, `integration_id` 15368, merged_at-bounded PR checks, open-PR-only job, PR-base check, fail-fast Release Evidence, transient-API retry
- [x] 2.6 Gates: `test_branch_rulesets_regression.py`, `test_release_workflow_regression.py`, `tests/scripts/test_{check_branch_direction,maintainer_scripts,release_scripts}.py`; negative controls recorded in each module
- [ ] 2.7 Maintainer: record the LF-R2 merge SHA (`CIV6AZ_R2_MERGE_SHA`) after merge; release-tag lineage is checked on content until then

## 3. R3 -- cut-over
- [ ] 3.1 Rename the default branch to `dev`; apply rulesets

## 4. R4 -- retire the old default
- [ ] 4.1 Drop `claude/civ6-alphazero-implementation-vMnun` from the protected list and workflow triggers

## 5. T14 -- Release Evidence (LF-D8)
- [x] 5.1 `a1_protocol.compute_verdict` / `verdict_from_bounds`: the one section 7.1 verdict implementation (`tests/unit/test_a1_verdict.py`)
- [x] 5.2 `scripts/verify_release_evidence.py` + `scripts/release_play_path.txt`; bundle schema in `docs/evidence/SCHEMA.md`
- [x] 5.3 `release-evidence.yml` (pull requests only, `edited` included) job `Release Evidence` (n/a off main and on `hotfix/*`); `main.json` requires it
- [x] 5.4 `branch_cutover.sh`: steps 4 and 5 withhold Release Evidence; new step `add-release-evidence` (9a)
- [x] 5.5 Gates: `tests/scripts/test_verify_release_evidence.py`; negative controls (symmetry VOID ignored, freshness diff dropped) recorded in its module
- [x] 5.5a Review fix round 1: VOID fails (`learning_recorded` needs a measured verdict; VOID attempts listed beside the re-run; section 7.1 "After a VOID"); PR-only workflow; highest rc and a play-path diff to the release head; 9a gated on `origin/main`; stdlib-only `--scope`; `run.json` bound to section 7.1's commands at the run's commit (`a1_registration`)
- [ ] 5.6 Maintainer action 9a: after the first release carrying T14 has merged into main, PUT Release Evidence into the main ruleset (`branch_cutover.sh add-release-evidence`, which refuses until `origin/main` carries T14); confirm a draft PR into main shows it as Expected
- [x] 5.5b Review fix round 2: an attempt whose `run.json` is not COMPLETED is VOID by status and is checked for honesty, not completeness (a prefix of the registered invocations, the whole card's price, the arms that ran); section 7.1 "After a VOID" separates an interruption (reproduction) from a control VOID (a second draw, disclosed)
- [ ] 5.7 (Deferred, CONFIRM-FIRST; T14 review M-9) Re-verify Release Evidence at publish: `release.yml` re-runs `scripts/verify_release_evidence.py` from the tagged commit for a non-hotfix final, so a PR that rewrote its own job cannot carry a green check into a release. Known gap recorded in `docs/runbooks/branching-and-release.md` section 1
