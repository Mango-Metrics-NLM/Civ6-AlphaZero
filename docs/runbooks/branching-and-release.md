# Branching and release runbook

> **Status:** this model takes effect when the maintainer runs the cut-over (section 5). Until then the default branch is `claude/civ6-alphazero-implementation-vMnun`, no rulesets are applied, and the only protection is the PreToolUse guard.

How work moves through `dev`, `qa` and `main`, who may do each step, and what is and is
not enforced today. Specification: `openspec/changes/branching-and-release/`. The checks
named here are implemented in `scripts/check_branch_direction.py`,
`.github/workflows/branch-policy.yml`, `.github/workflows/release.yml`,
`.github/rulesets/*.json` and `scripts/maintainer/`.

## 1. Read this first: what is enforced

| Rule | Mechanism | Status |
|------|-----------|--------|
| No agent commits to `dev`, `qa`, `main` or a tag | PreToolUse guard (`.claude/hooks/pretooluse_guard.sh`) | **Accident net.** Heuristic, can be bypassed; see `.claude/README.md`. |
| Only the maintainer merges into `qa`/`main` and creates `v*` tags | The machine user `civ6az-agent` (repo Write, on no ruleset bypass list) plus the `qa-main-merge` and `release-tags` rulesets | **A convention until the maintainer provisions the user and applies the rulesets** (`scripts/maintainer/branch_cutover.sh`). After that it is enforced by identity. |
| A PR goes in the right direction | `Branch Direction` check (`scripts/check_branch_direction.py`) | **Accident net, not a control.** A same-repo PR runs the workflow file from its own merge ref, so any Write-holder can fake the check. The control for `qa`/`main` is the maintainer-only merge above. |
| A tag is only published if its lineage is right | `release.yml` `verify-tag` + `wait-for-ci`, and `tag_release` | Enforced by the workflow once it is on the tagged commit. Required checks are pinned to `integration_id` 15368 (GitHub Actions). |
| A release PR into `main` carries verified evidence | `Release Evidence` check (`scripts/verify_release_evidence.py`, `ci.yml`) | **Accident net, not a control,** for the same reason as Branch Direction. Required on `main` only after maintainer action 9a; `wait-for-ci` requires it for a non-hotfix final either way. |
| No run over 1,000 games | `alpha_zero/utils/game_budget.py` in every runner | Enforced by the runner. `--max_games` can only lower the cap. `--approval_ref <issue>` raises it and is **recorded, not verified**; the guard asks before running a command that carries it. |

Do not describe the repository as enforcing a row before the status in that row is true.

## 2. The branches

| Branch | Role | Accepts (same-repo heads) | Merge method | Up to date required |
|---|---|---|---|---|
| `dev` (default) | integration | any branch except `main`, `qa`, `hotfix/*`, `evidence/*`, `rcfix/*` | squash; merge commit for `backmerge/*` | yes |
| `qa` | release candidates | `dev` (promotion), `evidence/*`, `rcfix/*`, `backmerge/*` | merge commit | no: its own merge commits would make every later promotion "out of date" |
| `main` | tagged releases | `qa`, `hotfix/*` | merge commit | yes: main's tip is always in `qa`, so this cannot deadlock |

There is no linear-history rule. GitHub cannot pick a merge method by head branch, so
`dev` allows both squash and merge, and back-merges are merged only with
`scripts/maintainer/merge_backmerge.sh <pr>`, which asserts the head is `backmerge/*`.
A squashed back-merge is caught at the next gate: the promotion check and `tag_release`
both require `origin/main` to be an ancestor.

## 3. Flows

### Feature (anyone, including agents)

`feature|fix|chore|docs|claude/*` -> `dev`, squash. Run `make verify` first. Fill in the PR
template, including the target branch and the priced game count of any run.

### Release prep

The last ordinary `dev` PR before promotion. It bumps `pyproject.toml`,
`alpha_zero/__init__.py` and `CITATION.cff` to the same `X.Y.Z` and rolls `CHANGELOG.md`.
The changelog and `date-released` are final at the first rc, because the final-tag rule
forbids later edits outside `docs/evidence/`. Write claims about results neutrally, for
example "verdict in `docs/evidence/release_v2_1_0/report.md`", since the evidence does
not exist yet. An agent can prepare this PR (the `release` skill); it does not tag.

### Promote (maintainer)

`dev` -> `qa`, merge commit. Then, from the maintainer's own terminal:

```bash
python -m scripts.maintainer.tag_release vX.Y.Z-rc.N            # dry run: checks every gate
python -m scripts.maintainer.tag_release vX.Y.Z-rc.N --execute  # tag + push
```

### Evidence

Run cards execute on the rc commit (section 4). Results go on branch
`evidence/vX.Y.Z-rc.N`, cut from the rc tag and merged into `qa` with a merge commit; only
`docs/evidence/**` may change. The `VALIDATION-RUN` ledger event goes to `dev` in a normal
PR. Evidence follows `docs/evidence/SCHEMA.md` (`search_epoch: post-2026-09-04`);
determinism holds on CPU only, so confirmatory runs use CPU and GPU runs are exploratory.

The release's evidence is `docs/evidence/release_vX_Y_Z/bundle.json` (schema: `SCHEMA.md`,
"Release evidence bundle"), with `report.md` and an index row. Check it before opening the
release PR:

```bash
python scripts/verify_release_evidence.py --repo OWNER/NAME   # 0 pass, 1 fail, 2 missing
```

It recomputes every count, interval and the section 7.1 verdict from the per-game records,
re-hashes every component, reads the rc's check runs as `wait-for-ci` does, caps each run at
1,000 games without its own `approval_ref`, and refuses a run whose commit is not an ancestor
of the rc or whose play path (`scripts/release_play_path.txt`) changed since. A FAIL, VOID or
INCONCLUSIVE verdict passes when recorded honestly (`learning_claimed: false`, no "learns"
wording); a negative A1 ships the release with the claim withheld.

### Release-candidate fix

Fix upstream first on `dev` (squash). Then either:

- (a) branch `rcfix/<topic>` from `qa`, `git cherry-pick -x <dev sha>` with no conflict
  edits, PR it into `qa`, tag rc.N+1 and re-run the cards; or
- (b) promote a new rc from `dev`.

### Release (maintainer)

`qa` -> `main` through a PR titled "Release vX.Y.Z" (merge commit). The PR needs a green
`Release Evidence` check (section "Evidence"). Then `tag_release.py vX.Y.Z`. The same day, back-merge `backmerge/vX.Y.Z-to-dev` (from `main`) into
`dev` with a merge commit, using `merge_backmerge.sh`.

### Hotfix (maintainer)

`hotfix/*` from `main` -> `main`, then tag. Then `backmerge/...-to-qa` and
`backmerge/...-to-dev`. `Release Evidence` reports n/a for a `hotfix/*` head, but only once
`main` carries the job: between action 9a and the first release after T14, a hotfix PR into
`main` runs from a `ci.yml` without it, the required check never reports, and the PR cannot
merge. Either cherry-pick the T14 commit (the job, `scripts/verify_release_evidence.py`,
`scripts/release_play_path.txt`) onto the hotfix branch, or remove Release Evidence from the
`main` ruleset for that merge and put it back afterwards (`branch_cutover.sh
add-release-evidence`).

### Keeping a back-merge current

Use only *Update branch -> merge commit*. *Update with rebase* destroys ancestry; if it
happens, recreate the branch. On a version or CHANGELOG conflict keep `dev`'s newer version
and insert the hotfix's section below it.

### Never

Sideways merges, squashed back-merges, direct commits, or a tag created by an agent.

## 4. Local runs (run cards)

Training runs are local-only. A run card (`configs/run_cards/*.yaml`) names a pinned SHA and
its invocations, and `scripts/run_card.py` runs it from that source in a pinned venv under
`CIV6AZ_DATA_ROOT` (for example `E:\civ6az-data`), isolated from GCP: local mode fails closed
(`LocalModeViolation`) if a GCS path is reached.

```bash
python scripts/run_card.py configs/run_cards/smoke.yaml --price_only   # price; creates nothing
python scripts/run_card.py CARD --data_root E:\civ6az-data             # run; logs and run.json under <root>/runs/<card_id>/
```

A card is priced in total, across every invocation, using each runner's own parser. The run
directory is single-use, and an interrupted run is VOID. The 1,000-game cap applies per runner
and to the card as a whole. Pricing reminders (worst case, defaults included): a bare
`training_civ6` prices at 10,200 games and is refused; `--dry_run` prices at 8;
`training_civ6 --num_games 20 --num_iterations 20 --save_interval 5` prices at 480.

## 5. Maintainer cut-over (one sitting)

`scripts/maintainer/branch_cutover.sh STEP` is a dry run unless `--execute`; `--help` lists
the steps. Nothing here is run by an agent.

0. Prerequisites: the key found by the secret audit is revoked; the `civ6az-agent` machine
   user exists; the R1 changes (guard and game cap) are merged into the old default branch;
   `v2.0.0` is tagged on `c214daa`.
1. Snapshot the repository settings and rulesets (`snapshot`).
2. Create `qa` and `main` at `c214daa`, asserting it equals `v2.0.0^{commit}`
   (`create-branches`). `dev`'s tip carries the unreleased NGC change.
3. Rename the default branch to `dev` (`rename`).
4. Immediately POST the `dev`, `qa`, `main`, `qa-main-merge`, `release-tags` and
   `legacy-name` rulesets, without Branch Direction and without Release Evidence (`rulesets`).
5. Merge the release-gates change into `dev`, then add Branch Direction to `dev`, `qa`
   and `main` (`add-branch-direction`).
6. Patch repository settings (`settings`), then housekeeping on open Dependabot PRs
   (`housekeeping`).
7. Verify (`verify`): every PR base is `dev`; a `chore/x` -> `qa` PR fails Branch Direction; a
   direct push to `dev` is rejected; `civ6az-agent` cannot merge into `qa` or create a `v*`
   tag. Only now may the docs claim identity enforcement.
8. Update local clones and worktrees (`local`):
   `git branch -m <old> dev && git branch -u origin/dev dev && git remote set-head origin -a && git fetch --prune`.

9a. **After T14 (Release Evidence) merges into `dev`:** PATCH Release Evidence into the `main`
    ruleset (`branch_cutover.sh --execute add-release-evidence`, which refuses until
    `origin/dev` has `scripts/verify_release_evidence.py`, then PUTs the full `main.json`).
    Confirm that a draft PR into `main` shows Release Evidence as **Expected**. Steps 4 and 5
    deliberately leave it out, so applying them again never adds it early. Read the hotfix
    caveat in section 3 first.

Rollback: `rollback` deletes the rulesets, renames `dev` back, deletes the `qa` and `main`
refs and restores the settings snapshot.
