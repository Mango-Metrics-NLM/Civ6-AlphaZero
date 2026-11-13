# Local-First Gap Closure + dev/qa/main Branching — Program Plan v3 (two review rounds)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Each PR also runs `V` (the verify command below) and the repo's `dev-sqe-handoff` → `pre-pr-handoff` skills (CLAUDE.md §14).
> **`V`** — run it from Git Bash, which is what the agents' Bash tool uses: `make verify PYTHON=python GITLEAKS_BIN="$(command -v gitleaks)"`. The recipes need `sh.exe`. From PowerShell, wrap it: `& 'C:\Program Files\Git\bin\bash.exe' -lc '<same command>'`.

**Goal:** Close every recorded gap except the Vertex/GCP-only ones. Run train → self-play → evaluate → promote entirely on this Windows workstation. Deliver through dev → qa → main, with the rules enforced by identity and by GitHub rulesets; the hooks are only a backstop.

**Architecture:**
- Branches promote **code**. A champion pointer promotes **models**: `champion.json` is the source of truth, mirrored as an MLflow alias.
- The local loop is the CLI loop `run_iterations` (`vertex_orchestrator.py:1295`), completed and hardened. Vertex `main()` is frozen and dormant; no parity with it is claimed.
- The first learning evidence comes from the seeded `training_civ6` path. It runs from pinned source under a pre-registered, hash-locked protocol, through one run-card driver.

**Scope rule:** this is a program plan across 5 releases. Only v2.1 is broken into TDD tasks (§9); each later release gets its own task plan at sprint planning.

**First execution step:**
1. Commit this plan as `kb/specs/20261111-local-first-and-branching-plan.md` and Appendix A as `kb/specs/20261111-local-first-gap-register.md`.
2. Commit the review record as `docs/audits/2026-11-11-local-first-plan-review.md`.
3. Run `planning-handoff`.

## 0. Context
- **Baseline:** the cloud session's PRs #132 (v2.0.0 public-release readiness) and #134 (NGC 26.10) merged on 2026-11-11. This worktree equals origin `491cbdc`.
- **Open gaps:** the gap register has **82 rows** (Appendix A, `LF-` IDs), drawn from the project's records, the design panel, and two review rounds. The headline gaps:
  - No learning evidence exists since the 2026-09-04 search fixes.
  - The move mask disagrees with the executor.
  - 73.5% of games end in draws, which makes promotion impossible.
  - The only real promotion loop is Vertex's, and the CLI loop crashes as soon as it trains for real (`AttributeError 'lr_schedule_type'`).
- **How this plan was made:**
  - 3 explore agents mapped the gaps, the local execution path and CI/branch governance.
  - A design panel worked out the plan: architect/SDLC/release, ML scientist/AI, product owner/scrum.
  - Review round 1: 73 agents across six lenses produced 72 findings (0 refuted) plus 10 from a completeness critic.
  - Review round 2: 25 agents ran a release-flow simulation, an executor-readiness audit and a statistics re-check. They confirmed 17 findings (0 refuted) plus 8 minor notes.
  - Every finding is folded in below.

**Your decisions:**
- Branches: the default branch becomes `dev`; `qa` holds release candidates; `main` holds tagged releases.
- Data root: `E:\civ6az-data`.
- Runs: overnight, at most 1,000 games each.
- Role subagents are persisted in `.claude/agents`.
- **Agent identity: a machine user** — my recommended default. Your answer was interrupted; alternatives in §3.6.

**Machine (measured):**
- CPU and GPUs: Ryzen 9 9900X (12C/24T). GPU0 is an RTX 5060 Ti 16 GB that also drives the display; GPU1 is an RTX 5060 8 GB.
- Memory: about **15 GB RAM free**. llama-server uses 29 GB, WSL 12 GB and OneDrive 7 GB.
- E: is internal NVMe (NTFS, 1.6 TB free), and it inherits Modify for Authenticated Users.
- Windows Update may restart the machine between 01:00 and 07:00.
- The installed `alpha_zero` is an **editable v1.8.0 install of the main checkout**, from before the search fixes.
- An ADC file sits in `%APPDATA%\gcloud`, a user-scope `CIV6AZ_GCS_BUCKET` is set, and `.env` is auto-loaded.

**GitHub:**
- Private repo on the Team plan: rulesets work, merge queue does not.
- No protection, no tags, and no remote `main`.
- 14 open PRs: 10 from Dependabot, plus #106, #69, #49 and #32.
- `refs/pull/72/head` holds the leaked W&B key.

## 1. Global constraints
- **Confirm first** (CLAUDE.md §3.2): architectural changes to more than 3 modules; breaking public API; deleting more than 100 lines; distributed or coordination changes; network architecture changes; new external services; any run over 1,000 games. Record each approval in the issue and in a handoff note.
- **Maintainer-only is enforced by identity, not hooks.** Agents act as the machine user `civ6az-agent` (repo Write, no bypass anywhere). Only the maintainer can merge into qa/main or create `v*` tags (§3.3). The deny rules and the guard are an accident net, not a control.
- **No agent commits** to `dev`, `qa`, `main` or `refs/tags/*`. Work always goes through a short-lived branch and a PR.
- **Game cap:**
  - A run card is priced in total, across every invocation it makes.
  - Each runner prices itself and refuses more than `--max_games 1000` unless given `--approval_ref <issue>`.
  - The hook asks before running anything it cannot price.
- **Isolation:** local runs are fail-closed against GCP (§4.2) and run from pinned source and a pinned env (§4.3).
- **Storage:** large artifacts live only under `CIV6AZ_DATA_ROOT`, which is ACL-hardened.
- **Gates:** every new gate ships with a negative control and an invoking job.
- **Evidence:**
  - Follows `docs/evidence/SCHEMA.md`, with `search_epoch: post-2026-09-04`.
  - Evidence directory names use underscores.
  - Determinism holds on CPU only, so confirmatory runs use CPU and GPU runs are exploratory.
- **Vertex/GCP code is frozen, not deleted** (§12).

## 2. Review focus — five failure modes
1. **A local run writes to production GCS.** T2 makes `GCSBackend` itself refuse local mode, via a `BaseException`, plus an autouse conftest fixture.
2. **A child process imports the stale v1.8.0 package.** T1 and T1b run a preflight check on `alpha_zero.__file__`.
3. **A night exceeds 1,000 games.** T1b prices the whole card; T5 makes every runner price itself.
4. **A release cannot complete.** T6 tests every §3.4 direction on a temporary-repo fixture; §11 adds a sandbox dress rehearsal.
5. **A1 is unfairly voided, or over-claimed.**
   - T11/T14 use hull intervals and an ordered verdict table.
   - The symmetry control is judged at 99%.
   - T14 recomputes the verdict.
   - `learning_recorded` and `learning_claimed` are tracked separately.

---

## 3. Branching & release strategy
**Principle:** promote code by one-way merges with mandatory back-merges; promote models by pointer/alias; build once per tag. Sources: [GitLab Flow, upstream first](https://archives.docs.gitlab.com/15.8/ee/topics/gitlab_flow.html); [nkdagility](https://nkdagility.com/resources/blog/stop-promoting-branches/); [Databricks deploy-code](https://docs.databricks.com/machine-learning/mlops/deployment-patterns/); [MLflow aliases](https://mlflow.org/docs/latest/ml/model-registry/workflow).

### 3.1 Branches
| Branch | Role | Accepts (same-repo heads) | Merge method | Strict |
|---|---|---|---|---|
| `dev` (default) | integration | any branch except `main`, `qa`, `hotfix/*`, `evidence/*`, `rcfix/*` | squash; merge commit for `backmerge/*` | yes |
| `qa` | release candidates | `dev` (promotion), `evidence/*`, `rcfix/*`, `backmerge/*` | merge commit | **no** — its own merge commits would make every later promotion "out of date" |
| `main` | tagged releases | `qa`, `hotfix/*` | merge commit | yes — main's tip is always in qa, so this cannot deadlock |

- **There is no linear-history rule.**
- **GitHub cannot pick a merge method by head branch.** dev's ruleset therefore allows `["squash","merge"]`. Back-merges are merged only with `scripts/maintainer/merge_backmerge.sh <pr>`, which asserts the branch is `backmerge/*` and runs `gh pr merge --merge`.
- **A squashed back-merge is caught at the next gate:** the promotion check (§3.4) and `tag_release` both require `origin/main` to be an ancestor.

### 3.2 Flows
- **Feature:** `feature|fix|chore|docs|claude/*` → dev, squash.
- **Release prep** (the last dev PR before promotion; it lands in Sprint 3 for v2.1):
  - Bump `pyproject.toml`, `alpha_zero/__init__.py` and `CITATION.cff`, and roll the CHANGELOG.
  - The CHANGELOG and `date-released` are final at the rc, because the final-tag rule forbids later edits.
  - A1 lines are written verdict-neutral, e.g. "verdict in `docs/evidence/release_v2_1_0/report.md`".
- **Promote:** dev → qa, merge commit. Then `tag_release.py vX.Y.Z-rc.N`, run by the maintainer.
- **Evidence:** run cards execute on the rc SHA.
  - Results go on branch `evidence/vX.Y.Z-rc.N`, cut from the rc tag and merged into qa with a merge commit. Only `docs/evidence/**` may change.
  - The VALIDATION-RUN ledger event goes to dev in a normal PR.
- **RC fix:** fix upstream first on dev, as a squash merge. Then either:
  - (a) branch `rcfix/<topic>` from qa, `git cherry-pick -x <dev sha>` with no conflict edits, PR it into qa, tag rc.N+1 and re-run the cards; or
  - (b) promote a new rc.
- **Release:** qa → main via the "Release vX.Y.Z" PR, which needs Release Evidence once T14 lands. Then `tag_release.py vX.Y.Z`. The same day, back-merge `backmerge/vX.Y.Z-to-dev` (from main) into dev with a merge commit.
- **Hotfix:** `hotfix/*` from main → main, then tag. Then `backmerge/…-to-qa` and `backmerge/…-to-dev`.
- **Keeping a back-merge current:** use only *Update branch → merge commit*. *Update with rebase* destroys ancestry; if it happens, recreate the branch. On a version/CHANGELOG conflict, keep dev's newer version and insert the hotfix's section below it (D7).
- **Never:** sideways merges, squashed back-merges, or direct commits.

### 3.3 Rulesets (`.github/rulesets/*.json`, applied by the maintainer)
- **Branch rulesets dev/qa/main:**
  - `deletion`, `non_fast_forward`.
  - `pull_request`: 0 approvals, thread resolution required, merge methods per §3.1.
  - `required_status_checks`: Lint & Type Check · Test (Python 3.11) · Test (Python 3.12) · Cloud Unit Tests · Security Scan · Governance Gates · LLM Eval Harness · Coverage Report · Branch Direction.
  - **Release Evidence is added to `main.json` only in T14,** in the same PR that adds the job. Before then main requires the dev/qa set, so hotfixes stay mergeable.
  - Never required: Test 3.13, contributor-smoke, CodeQL, Scorecard, Dependency review.
- **Bypass:**
  - dev: admin, `pull_request` mode.
  - qa and main: `[]`.
- **`qa-main-merge`** (branch; includes qa and main):
  - Rule: `update` with `update_allows_fetch_and_merge: false`.
  - Bypass: the maintainer as a `User` actor, mode `always`.
  - Effect: only the maintainer can merge into qa/main, and still needs every check, because the qa/main rulesets have no bypass.
- **`release-tags`** (target tag, `refs/tags/v*`; rules creation, update, deletion, non_fast_forward): bypass is the maintainer as a `User` actor, mode `always`. `pull_request` mode is invalid on tag rulesets.
- **`legacy-name`:** blocks re-creating `claude/civ6-alphazero-implementation-vMnun`.
- **Repo settings:** `delete_branch_on_merge`, `allow_update_branch` and `allow_auto_merge` on; rebase merge off.
- **Test:** `test_branch_rulesets_regression.py` asserts that every check named in each JSON is a real PR job that is not `continue-on-error` and not path-filtered, expanding matrix legs (`ci.yml:101-116`).

### 3.4 Branch Direction (`branch-policy.yml` + `scripts/check_branch_direction.py`)
- **Trigger:** `pull_request` events opened, synchronize, reopened and edited.
- **Inputs:** base and head are read from `gh api …/pulls/$PR`, not from the event payload. Values are passed via `env:`. The job checks out with `fetch-depth: 0`, and the head must be in the same repo.
- **Rules:**
  - **dev → qa (promotion):** `origin/main` must be an ancestor of the head. On failure the job reports that the last back-merge was squashed or is missing.
  - **`backmerge/*`:** let R = `origin/main` (into qa), or `origin/main` and `origin/qa` (into dev).
    - Every non-merge commit in base..head must be reachable from R.
    - Every parent of every merge commit must be reachable from R or from `origin/<base>`, or lie inside the range. Any number of such merges is allowed, because strict mode adds one per update.
    - `git show --remerge-diff` for each merge goes to the job summary.
  - **`hotfix/*`:** `origin/main` is an ancestor of the head, and no commit in main..head comes only from dev.
  - **`evidence/*`:** only `docs/evidence/**` changes, and the rc tag is an ancestor.
  - **`rcfix/*`** (into qa): no merge commits, and `git cherry origin/dev <head> <base>` prints only `-` lines (patch-equivalent). Any `+` line is refused; fall back to a new rc.

### 3.5 Release mechanics
- **`scripts/release_version.py`:** an rc tag stamps PEP 440 `X.Y.ZrcN` into the build copy only. Release notes come from the base version's CHANGELOG section.
- **`release.yml`:**
  - `${version%%-*}` must equal pyproject.
  - An rc must be an ancestor of `origin/qa`; a final must be an ancestor of `origin/main`.
  - Final-tag rule: `git diff <highest rc>..<final>` may touch only `docs/evidence/**`. A hotfix final with no rc is allowed only if the tagged merge came from a `hotfix/*` PR.
- **`wait-for-ci` job** (timeout 100 min; `permissions: contents read, checks read, pull-requests read`; the publish job `needs` it):
  - **Push checks:** polls check-runs on `$GITHUB_REF^{commit}` for the jobs that run on **push** (the eight non-Branch-Direction checks).
  - **PR checks:** PR-only checks are verified on the PR that produced the commit. `GET /commits/$SHA/pulls` must return exactly one merged PR whose `merge_commit_sha == $SHA`. Its head must carry a successful Branch Direction, and also Release Evidence for a non-hotfix final.
- **CI triggers** (T4):
  - Push triggers cover dev, qa and main.
  - `cancel-in-progress` is false on protected branches, so a second merge cannot cancel CI on the commit about to be tagged.
- **`scripts/maintainer/tag_release.py`:**
  - Pre-checks every rule above.
  - Refuses an rc unless `origin/main` is an ancestor.
  - Refuses commits that do not descend from the LF-R2 merge, because older lineage has an unpinned `release.yml` with write permission.
- **v2.0.0:** tagged on `c214daa` (merge of #132; version and CHANGELOG are 2.0.0 there; CI green). This is a one-time exception, recorded in the ledger.

### 3.6 Agent-side enforcement (LF-G6, LF-G11)
- **Agent identity** (maintainer action 4):
  - The machine user `civ6az-agent` takes one Team seat, has repo Write, and appears on no bypass list.
  - Its fine-grained PAT is `GH_TOKEN` in `settings.local.json`.
  - Agent sessions set `GIT_CONFIG_COUNT` env overrides so `credential.helper` is empty and the PAT is used for `github.com`, never the maintainer's Git Credential Manager (GCM) login.
  - **Residual risk:** GCM still holds the maintainer's login under the same Windows profile.
  - **Alternatives:** (b) run agents under a separate Windows account, for full isolation; (c) keep the maintainer identity, in which case maintainer-only is a convention only and the plan must say so.
- **`settings.json`:**
  - `CIV6AZ_DEFAULT_BRANCH=dev`.
  - `CIV6AZ_PROTECTED_BRANCHES="dev qa main claude/civ6-alphazero-implementation-vMnun"`; R4 drops the last entry.
  - PreToolUse matcher `Bash|PowerShell`.
  - `enableAllProjectMcpServers: false`.
- **The guard:**
  - **Unwraps** `pwsh -c`, `cmd /c`, `bash -c`, `iex` and `&` before matching.
  - **Denies:**
    - commit/push on protected branches, and pushes to `refs/tags/*` or `+refs`;
    - deleting a protected branch;
    - `git tag` unless every argument is a listing option (`-l`, `--list`, `--contains`, `--points-at`, `--merged`, `--sort`, `--format`, `-n`);
    - `--tags`/`--follow-tags` on `git push` only;
    - `gh pr merge`, `gh release`, `gh repo edit`, `gh ruleset`;
    - `gh api` with `-X`/`--method`, `-f`/`-F`/`--field`/`--raw-field`/`--input`, or `graphql`.
  - **Asks** on `--approval_ref`, `--preset production`, `--config`, and unknown runners.
  - **Limit:** pre-approved interpreters (`python -c`) can still call the API directly. That is why identity is the control.
- **Cost models** (`alpha_zero/utils/game_budget.py`):
  - `training_civ6`: `G×(I−S) + eval_games×⌈(I−S)/eval_interval⌉`. The term is 0 when `--eval_games 0`. Defaults give 100×100 + 20×10 = **10,200**. `--dry_run` gives 8.
  - Orchestrators: `workers×games×iterations×(1+retries) + N_max×iterations`.
  - Benchmark/arena: `n_games × (len(opponents) + len(opponent_checkpoints))`.

### 3.7 Cut-over runbook (`scripts/maintainer/branch_cutover.sh`, one sitting)
0. Prerequisites:
   - LF-G1 key revoked.
   - Machine user created (action 4).
   - T3a and T4 (R1) merged into the old default branch.
   - `v2.0.0` tagged on `c214daa`.
1. Snapshot the repo settings: `gh api repos/… > before.json`.
2. Create `qa` and `main` at `c214daa`, asserting it equals `v2.0.0^{commit}`. dev's tip carries the unreleased NGC change.
3. Rename the default branch to `dev`.
4. **Immediately** POST the dev, qa, main, qa-main-merge, release-tags and legacy-name rulesets, without Branch Direction and without Release Evidence.
5. Merge T6 (R2) into dev, then PATCH Branch Direction into dev, qa and main. Release Evidence waits for T14 (action 9a). *(As delivered, R1 and R2 landed in one PR (#135), so this step is a precondition check, not a merge; runbook §5 is authoritative.)*
6. PATCH the repo settings.
7. Close #95 (codecov-action is unused) and #120 (the pytorch image was replaced by NGC). Comment `@dependabot rebase` on the other Dependabot PRs. Run `gh pr update-branch <n>` on #106, #69, #49 and #32.
8. Verify:
   - every PR base is dev;
   - a `chore/x→qa` PR fails Branch Direction;
   - a direct push to dev is rejected;
   - `civ6az-agent` cannot merge into qa or create a `v*` tag.
9. Update local clones and worktrees: `git branch -m <old> dev && git branch -u origin/dev dev && git remote set-head origin -a && git fetch --prune`.
- **Rollback:** delete the rulesets, rename dev back, delete the qa and main refs, restore `before.json`, revert R1.

## 4. Local pipeline (v2.2; isolation and pinning land in v2.1)
### 4.1 Seam and package
- **Seam:** `run_iterations` (`vertex_orchestrator.py:1295`, reached via `cli.py:547`), completed by LF-L0:
  - a `JobClient` Protocol;
  - fix the CLI training crash;
  - one threshold source;
  - state saves that fail loudly;
  - port resume, MLflow, curriculum and stall detection from `main()`;
  - resolve the preemption contradiction.
- **Job managers:** the `Vertex*Job` managers (`vertex_jobs.py:702/788/1012`) stay as the job-spec layer. Their timeouts (`:921`) become arguments.
- **Dormant classes:** `VertexAIClient`, `VertexTuningJob` and `VertexTournamentJob`.
- **New code goes in `alpha_zero/local/`,** because `cloud/*` is omitted from the 85% coverage gate (`pyproject.toml:377,456`). Its `__init__.py`, `Agent.md`, `MODULE_REFERENCE.md` row, and a live importer for every module land in the PR that creates it. Those are the `check_doc_sync` D2, `validate_agents` and debt-ratchet requirements.

### 4.2 GCP isolation, fail-closed (LF-E5, T2)
- **`enter_local_mode`:**
  - Pops `CIV6AZ_GCS_BUCKET`, `VERTEX_*`, `*CONTAINER_IMAGE_URI*` and `GOOGLE_*`.
  - Sets `CLOUDSDK_CONFIG=<run_dir>/gcloud-empty`, `NO_GCE_CHECK=True`, `CIV6AZ_STORAGE_BACKEND=local` and `CIV6AZ_LOCAL_RUN=1`.
  - Builds `GCPConfig(bucket=<run_id>, project_id="local")`.
- **`GCSBackend.__init__` and its lazy `client` property** (`cloud/gcp/storage.py:267,276-283`) raise `LocalModeViolation(BaseException)` under local mode. That covers `get_backend()` and all 8 direct sites. It is a `BaseException` because the loop's state calls catch `Exception` (`orchestrator_state.py:178-180,229-231`; `vertex_orchestrator.py:1202-1204`).
- **Children** get the environment minus a deny list: no `*_TOKEN`, `*_KEY`, `SSLKEYLOGFILE` or `CIV6AZ_ALLOW_UNSAFE_CHECKPOINT`, and no `.env`. (As delivered this is a deny list, not an allowlist; other variables pass through. Final review M-7.)
- **Tests:** an autouse conftest fixture strips these variables.

### 4.3 Pinned source and env (LF-E6, T1, T1b)
- The source is `git archive <sha>`, extracted read-only to `<run_dir>/src`.
- The env is a venv at `E:\civ6az-data\envs\<sha12>`, built from a non-editable wheel of that SHA.
- Children run with `cwd=src` and `PYTHONNOUSERSITE=1`.
- Preflight asserts that `alpha_zero.__file__` lies inside the venv or `src`, and that the version equals pyproject.
- `run.json` records the SHA, the wheel hash, `pip freeze`, the device plan, and environment variable **names**.
- This is the per-run "clone from GitHub".

### 4.4 Process control (LF-L4)
- **`LocalProcessClient`:**
  - Runs `[venv_python, -m, module, *args]`, with no shell.
  - Refuses `torchrun` and multi-accelerator jobs.
  - The deadline is `min(arg, spec.timeout_seconds)`. On a deadline or an error it cancels, waits, then raises.
  - Retries are priced.
- **Process lifetime:**
  - On Windows, a Job Object with `KILL_ON_JOB_CLOSE`. Pid files are not used, because PIDs are reused.
  - On POSIX, `start_new_session` plus `killpg`.
- **`alpha_zero/local/shutdown.py`** handles SIGBREAK, SIGINT and SIGTERM. Without it, CTRL_BREAK exits with 0xC000013A and skips `finally` blocks.
- **Self-play** flushes a shard every N games.

### 4.5 Storage (LF-L1)
- **Run scoping (bug 9):** `<root>/storage/<run_id>/`, with `run_id` matching `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`. Fix the agent_id asymmetry and the unscoped cleanup (`config.py:180-182`, `orchestrator_state.py:149-157`).
- **`LocalBackend`:**
  - root is `storage_root()`;
  - atomic writes go to `<name>.<pid>.tmp`, then `os.replace` with retries;
  - no directory fsync on Windows;
  - listings ignore `*.tmp`.
- **The 8 `GCSBackend(` sites** move to `get_backend()`.
- **Manifests:** `manifest.d/worker_<id>.json`, consolidated by the orchestrator.
- **Champion:** sha256 is verified on load, and loading uses `weights_only=True`.
- **Data root:**
  - created with `icacls /inheritance:r` and SID grants;
  - must be NTFS or ReFS, and not USB;
  - must not be inside the repo or OneDrive, and must be at most 120 characters;
  - warns below 50 GB free;
  - logs, TMP and TEMP live under the root.

### 4.6 Loop fixes (LF-L2/L3, confirm-first)
1. Stop passing architecture flags to the evaluation worker, with a contract test that every `build_*_extra_args` flag parses.
2. `champion.json` (atomic, written only by the orchestrator) replaces `latest.json` as the restart source.
3. `--train_init {latest,champion}`, default `latest` (AlphaGo Zero semantics).
4. An end-of-iteration checkpoint records `iteration`; `start_epoch` is honoured only for the same iteration.
5. One manifest per worker.
6. `run_iterations` resolves `--resume_from`.
7. A dry-run evaluation returns NaN and is rejected (updates `test_orchestrator_pipeline.py:251` and `test_orchestrator_state.py`).
8. `--seed`, `--device`, `--c_scale`, `--policy_target` and `--greedy_threshold` are forwarded to the workers via `derive_seed`.
9. Run scoping (§4.5).

- **Prerequisite:** harden-governance-gates 4.4 (split `training_worker.main`) lands before L3.
- **Different network:** the local loop trains `Civ6TransformerNetwork`, so A1 is **not** evidence for it. The loop gets its own record, A1-L.

### 4.7 Devices, memory, profiles
- **Devices:**
  - Self-play on CPU (`CUDA_VISIBLE_DEVICES=""`, OMP 1–2 threads).
  - Training on GPU0; watch VRAM, because GPU0 also drives the display.
  - Evaluation on GPU1 or CPU shards.
  - `make local-calibrate` may move self-play elsewhere.
- **Memory:** budget 1.3–1.6 GB per process plus 52 KB per position. Preflight refuses any phase that needs more than measured free commit minus 4 GB. The runbook stops llama-server and WSL first.
- **Profiles:** `configs/local_profiles.yaml`. A test asserts each profile's worst case is at most `--max_games`, for example 3 × (24×2 + 200) = 744.
- **LF-L6 / D2:** `EloPlateauDetector` halts the run after 5 consecutive non-promotions.
- **Targets:** `local-loop-smoke`, `local-calibrate`, `local-loop`, `mlflow-local`. The existing `local-smoke` gets its env var fixed.
- **`local_muzero_launcher.py`:** kept, marked superseded.

### 4.8 MLflow (LF-E8)
- **Server:** `mlflow server --backend-store-uri sqlite:///E:/civ6az-data/mlflow/mlflow.db --default-artifact-root file:///E:/civ6az-data/mlflow/artifacts --no-serve-artifacts --workers 1 --host 127.0.0.1`.
- It runs from the run's pinned venv, so the repo's `mlflow<3.0` pin governs.
- `champion.json` is the source of truth; the alias is a mirror.
- CI job "MLflow Integration".

## 5. ML science
### 5.1 Fix before any evidence run
- **LF-B1:** the move-mask collision (`civ6_actions.py:629-635`).
- **LF-A11** (CF: touches more than 3 modules): the Gumbel per-round floor, which exists in 3 copies:
  - `core/mcts/gumbel.py:476/498`, `core/batch_mcts.py:377/430`, `core/mcts/muzero.py:543/564`.
  - Fix: one `planned_halving_sims(k, n, num_legal)` in `gumbel_ops`, plus a `SearchStats(root_evals, sim_evals)` equality test.
  - Exact budgets: (8,24) and (16,64).
- **LF-A12:** σ missing from sequential halving. Memo first; implementation per the A5 decision.
- **LF-A14:** `--eval_interval/--eval_games`; `--eval_games 0` turns in-loop evaluation off (`training_civ6.py:1183-1191`).
- **LF-A15:** `--bootstrap {simple_mcts,none}`. Iteration 0 is hard-wired to SimpleMCTS at `:1130`. Save `checkpoint_0.pt` before training starts.
- **LF-A16** (provenance):
  - `Civ6Network.save(path, *, metadata=None)` (`core/civ6_network.py:475`, inherited by `MicroCivNetwork`) writes primitive-only metadata `{board_size, max_turns, num_players, mcts, gumbel_k, gumbel_sims, c_scale, policy_target, bootstrap, seed, git_sha, env_version}`.
  - `utils/checkpoint.py` adds `HorizonMismatch`, `build_provenance` and `check_provenance`, and is called from `load_network`, `resume_network_weights` (`training_civ6.py:159`) and the arena/benchmark loader.
  - Legacy files with no metadata get a warning and pass.
- **LF-A13** (prequential diagnostics):
  - `self_play_game`/`run_self_play` take an optional `step_sink: list[StepRecord] | None`; their return types are unchanged (55 call sites).
  - Each played `env.step` appends `StepRecord(game_idx, action, action_type, success)`. `action_type` comes from `decode_action(action)["type"]` and `success` from `info["action_success"]` (`envs/microciv.py:347`). Search rollouts are never recorded.
  - Computed after `run_self_play` and before `replay_buffer.add_game` (`:1132-1135`):
    - R²_oos against c, the mean value target in the buffer before insertion (0 at iteration 0);
    - sign accuracy on |v|=1, against the constant-predictor baseline, reusing `TrainingDiagnostics.compute_value_accuracy`;
    - target/policy entropy (`compute_policy_entropy`);
    - the action-type histogram;
    - the invalid-move count.
  - Recorded, not gated.
- **LF-D6** (benchmark producer):
  - `benchmark.py` gains `schema_version`, a valid `search_epoch`, `claims`/`non_claims`, `--shard i/N`, `--candidate {checkpoint,heuristic,random}`, `--max_games` and `--approval_ref`.
  - `ArenaResult` gains `games: [{pair, seat, result}]` (`cloud/gcp/arena.py`).
  - `scoring` gains `pair_score_interval(pair_scores, confidence)` and `alpha=0.025` support in `minimum_detectable_effect`.

### 5.2 A1 pre-registration — `docs/EVALUATION.md` "section 7.1" (appended; §7 gets a superseded banner)
- **Ordering:** lands in T13, after the LF-A5 decision (or with current defaults registered), and **before any A1 run, exploratory runs included**. The artifact records the sha256 of section 7.1.
- **Seeds:** exploratory 1001–1003; confirmatory **20261111** (A1) and **20261112** (A1-R), neither run before.
- **Interruptions:** an interrupted or resumed confirmatory run is **VOID**.

**Registered commands** (copied verbatim into section 7.1; `<D>` = `E:\civ6az-data\runs\<card>`; executed by `scripts/run_card.py`, T1b):
- **Training (480 games):** `python -m alpha_zero.training_civ6 --seed 20261111 --board_size 8 --max_turns 60 --num_iterations 20 --num_games 24 --mcts gumbel --gumbel_k 8 --gumbel_sims 24 --device cpu --bootstrap none --save_interval 5 --eval_games 0 --checkpoint_dir <D>\ckpt`
- **Arena calls** use `python scripts/benchmark_suite.py --opponents= --seed 20261111 --board-size 8 --max-turns 60 --sims 24 --gumbel-k 8 --mcts-type gumbel --device cpu` plus:

| Arm | Flags | Games |
|---|---|---|
| primary | `--checkpoint …20.pt --opponent-checkpoint …0.pt --n-games 200` | 200 |
| symmetry | `--checkpoint …0.pt --opponent-checkpoint …0.pt --n-games 40` | 40 |
| positive control | `--candidate heuristic --opponents=random --n-games 40` | 40 |
| secondary | checkpoints 5/10/15 vs …0.pt, 60 each | 180 |
| final vs random agent | `--opponents=random --n-games 40` | 40 |

- **Card total: 980**, priced across every invocation.
- **Positive control reference:** measured by T0b at 8/60 on post-B1 code (seeds 1001–1003 × 200 games), not the board-6 figure of 0.725.
  - Section 7.1 records P(ci_low ≤ 0.5) at n=40.
  - If that probability is above 0.05, the control takes 60 games and "final vs random" drops to 20, keeping the total at 980.
  - Pre-B1 reference: 0.7375, CI [0.711, 0.761].
- **Intervals:** for each arm, (a) `scoring.score_interval(w,d,l)` and (b) `pair_score_interval`. The decision bounds are the hull, L = min(lows) and U = max(highs). The symmetry control is judged at **99%**: chance of a VOID ≤ 0.8%, and it detects only gross asymmetry, about 0.20 score.
- **Verdicts** — the first matching row wins:

| Order | Verdict | Condition |
|---|---|---|
| 1 | VOID | run interrupted or resumed; OR the symmetry 99% hull excludes 0.5; OR the positive-control L ≤ 0.5 |
| 2 | PASS | primary L > 0.5 |
| 3 | FAIL | primary U < 0.5 |
| 4 | INCONCLUSIVE | otherwise; report MDE = `minimum_detectable_effect(200, d, alpha=0.025)` (≈0.072, about +50 Elo at d=0.47), widened by the hull ratio |

- **Trend:** `check_learning_trend.py` (LF-A2) is descriptive only.
- **Wall-clock:** not estimated in advance; T0b measures it. If the ETA is over 12 h, A1 runs as two cards under one registration (self-play, then a sharded arena).
- **Follow-ups:** A1-R must pass before any public "learns" claim. On FAIL, run A1-P (PUCT, same seed) to tell a search fault from a loop fault.

### 5.3 Promotion statistics (verified against `scoring.py` plus simulation)
- **`score_rate_lower_bound@0.52`** at n=1,000 has 80% power only at +30, +36 or +38 Elo, for draw rates d = 0.735, 0.5 and 0.4 respectively.
- **SPRT (LF-D7):** on score, with pentanomial pair scores recorded by T11; α=0.05, β=0.20; H0 is Elo ≤ 0, H1 is Elo ≥ E1. Expected games (ASN) at E1=+20, as H0/H1/max:
  - d=0.735: 215/306/346
  - d=0.5: 406/577/653
  - d=0.4: 487/692/784
- **qa→main gate:** E1=+20, N_max=1,000, evaluation only. A run still undecided at the cap counts as a reject.
  - Realised power at a true +20 Elo: **0.80 / 0.68 / 0.61** at d = 0.735 / 0.5 / 0.4.
  - α ≤ 0.045.
  - If d falls below 0.6, re-derive N_max.
- **In-loop gate:** E1=+35, N_max=200, truncation counts as a reject. P(promote):

| d | +5 | +10 | +20 | +35 |
|---|---|---|---|---|
| 0.735 | 0.07 | 0.13 | 0.33 | 0.72 |
| 0.5 | 0.05 | 0.08 | 0.19 | 0.47 |
| 0.4 | 0.04 | 0.06 | 0.14 | 0.38 |

  - This is a coarse filter.
- **Pooling:** LLRs pool only across runs pre-registered as continuations.
- **Draw rate:** at fixed Elo bounds, going from d=0.735 to d=0.5 costs 1.89× more games. Lowering draws is a game-design goal, not a statistical saving.
- **LF-D1:** decides the code default (CF, plus approval for runs over 1,000 games).

### 5.4 Later experiments (each ≤1,000 games, each pre-registered)
- **LF-B2 (v2.2):**
  - Trained cell: A1 final vs older checkpoints, 150 maps × 2 seats = 300 games, at the registered horizon.
  - 3-player random cell: 300 games per horizon.
  - **Rule** (H0: d ≥ 0.45, H1: d ≤ 0.38, one-sided α 0.05, power 0.80, n≈308): adopt a horizon only if draws ≤ 120/300 **and** the pair-clustered one-sided 95% upper bound ≤ 0.45. That gives P(adopt) = 1.00 / 0.78 / 0.046 at d = 0.30 / 0.38 / 0.45.
  - Order: try 60 turns, then 100, else LF-B4 enters core. The same rule gates LF-C2.
- **LF-D5:** `max_turns=60` is a protocol constant; 4 opponents × 110 games.
- **LF-C5:** runs after LF-C6.
- **LF-A9:** each arm is its own run of ≤1,000 games, with ≥3 seeds.
- **LF-A17:** the S5 seed-overlap memo.

### 5.5 Risks → detectors
| Risk | Detector |
|---|---|
| Value collapse | R²_oos and sign accuracy vs the constant baseline |
| Over-sharp target | the A5 probe |
| Horizon mismatch | provenance refusal |
| Budget not what was configured | `SearchStats` |
| Contaminated bootstrap | `--bootstrap none` plus the action histogram |
| Stale code | the pinned-source preflight |
| GCS writes | the `LocalModeViolation` guard |

## 6. Release evidence bundle (LF-D8)
- **Files:**
  - `docs/evidence/release_v2_1_0/bundle.json`, with its own evidence block;
  - `report.md`;
  - a README row.
- **Verifier:** `scripts/verify_release_evidence.py`, exit 0 (pass) / 1 (fail) / 2 (missing).
- **Integrity checks:**
  - Recomputes `scoring.summarize` and both intervals from the stored per-game records.
  - **Recomputes the verdict** using the §5.2 order, and fails if it differs from the stored one.
  - Re-hashes every component.
  - Verifies check-runs as `wait-for-ci` does.
- **Budget:** each component run (by `run_id`) must be ≤1,000 games or carry its own `approval_ref`. The bundle total is reported, not gated (A1 + A1-R = 1,960).
- **Freshness:**
  - Every component's `git_sha` must be an ancestor of `rc_sha`.
  - No diff since that SHA on the paths in `scripts/release_play_path.txt`: `core/mcts/**`, `mcts_constants`, `mcts_factory`, `core/batch_mcts.py`, `civ6_actions`, `civ6_network`, the encoders, `hexcrystal/**`, `envs/**`, `evaluation/**`, `utils/seeding.py`, `utils/constants.py`, `cloud/gcp/arena.py`, `training_civ6.py`.
  - No `MCTS_*` overrides.
- **Learning:** `learning_recorded` is required (schema-valid, protocol hash matches, verdict recomputed, README row present). `learning_claimed` (a PASS) is the only thing that permits "learns" wording. **A negative A1 therefore ships v2.1, with the claim withheld.**
- **CI job "Release Evidence":** always runs. It exits 0 as "n/a" when the base is not main or the head is `hotfix/*`. *(Superseded: it is a pull-request-only workflow, `.github/workflows/release-evidence.yml`; runbook §1 and §3.)*

## 7. Role subagents (LF-G13): `.claude/agents/sdlc-*.md`, 8 roles
| Agent | model | tools | Accountable for | Reviews | KB role |
|---|---|---|---|---|---|
| sdlc-product-owner | sonnet | Read, Grep, Glob | stories, acceptance criteria, doc and charter text | scope | Planning |
| sdlc-scrum-master | sonnet | Read, Grep, Glob, Bash† | sprint and compute queue, run cards | WIP and capacity | Planning |
| sdlc-architect | opus | Read, Grep, Glob, Bash† | designs, confirm-first triage, OpenSpec | changes touching more than 3 modules | Planning |
| sdlc-ml-scientist | opus | Read, Grep, Glob, Bash† | pre-registrations, memos, evidence audits | ML PRs and results | Planning |
| sdlc-ai-engineer | opus | Read, Grep, Glob, Bash† | LLM/RAG/advisor and agent-harness design | LLM and harness PRs | Planning |
| sdlc-developer | inherit | all | **every write, commit, push, PR and run-card execution** | — | DEV-SQE |
| sdlc-qa-sdet | sonnet | Read, Grep, Glob, Bash, Edit/Write limited to `tests/**` by hook | tests and negative controls | every DEV-SQE output | DEV-SQE + Pre-PR |
| sdlc-release-manager | sonnet | Read, Grep, Glob, Bash† | promotion, release and back-merge PR text; bundle manifest | release readiness | Pre-PR |

- **Owner vs executor:** read-only roles return text (a memo, script, doc, PR body or manifest). sdlc-developer commits it verbatim with `authored-by: sdlc-<role>`, and the authoring role then reviews the PR, so generator ≠ evaluator.
- **† `readonly_shell_guard.sh`** (matcher `Bash|PowerShell`):
  - **Allows:** `git log|show|diff|status|rev-parse|merge-base|ls-files|cat-file|tag -l`; `gh pr view|list|checks`; `gh api` GET; `python scripts/{validate_,check_,verify_release_evidence}*.py`.
  - **Denies everything else.**
- **The ledger** is written by the handoff skills.
- **Before T15 lands** (Sprints 0–1), roles are played by general-purpose subagents briefed from the draft role text in this section.
- **Wiring:**
  - `planning-handoff` dispatches product-owner, scrum-master and architect, plus ml-scientist or ai-engineer when relevant.
  - `dev-sqe-handoff` dispatches qa-sdet.
  - `pre-pr-handoff` 2b uses qa-sdet; release-manager sets the target branch and PR type.
- **Validator:** `scripts/validate_claude_agents.py` (stdlib only, exit 0/1/2), checks C1–C9:
  - C1: name = file stem with the `sdlc-` prefix.
  - C2: description is not vacuous.
  - C3: tools are known; read-only roles have no Edit/Write; no `permissionMode`, `mcpServers` or unapproved hooks.
  - C4: the model is valid.
  - C5: required sections are present.
  - C6: referenced skills exist.
  - C7: all 8 roles exist.
  - C8: each role is dispatched by at least one skill.
  - C9: coverage literals are correct.
- **Wired into:** PostToolUse, pre-commit, CI Governance and `make validate`. CLAUDE.md §15 is rewritten.

## 8. Delivery
### 8.1 Release train
| Release | Content | Gate |
|---|---|---|
| S0 maintainer | LF-G1, G2, G9, G12 (merge), G15, G16 | — |
| **v2.0.0** (tag) | tag `c214daa` | CI green; release artifacts |
| **v2.1.0** | T0a–T18: E4–E7, G6, G10, G11, G13, G14, G5, R1–R4, B1, A2–A4 (A4 via T11), A5 memo, A6, A7, A11–A17, C4, D6, D8, A1+A1-R | `learning_recorded` |
| **v2.2.0** | L0–L6, E2, E8, D2, D3, D5, D7, B2, C5, C6, A8, F1, F2, F3, F6 (loop), F9 | 3 nights, each ≤1k games; promote and reject paths; kill, reboot and resume; lineage; CPU `summary.json` hash reproduction; A1-L |
| **v2.3.0** | B3, B6, D1, D4, C1, A9, A10, H3, E3 | trained draw table; feasibility; ≥1 SPRT promotion persisted |
| **v3.0.0** | B4, B5, C2, C3 | `env_version` refusal; section 7.1 re-run |
| Continuous | F4, F5, F7, F8, G4, G7, G8, H1, H2, H4 (H5 = M5) | — |

### 8.2 Capacity & sprints (1 week each; ≤3 open agent PRs; maintainer review 6–8 h/week is the binding constraint; if under 6 h, T15 and T18 slip to v2.2)
| Sprint | Code | Overnight card |
|---|---|---|
| 0 (3–4 d) | T0a Windows baseline and harness smoke (no ETA claim); T1; T2; T3a†; T4 (R1); maintainer actions 1–6; cut-over with T6 (R2) | `s0-timing` harness card only |
| 1 | T1b run-card driver; T5; T7 (B1); T8 (CF); T9; T10 (A3 CF); T3 (R3); T15 role agents | **T0b** calibration at 8/60, k=8, n=24, CPU (GPU1 is exploratory and runs outside run_card: the pinned venv has CPU-only torch), after T7–T10 merge. Also the positive-control reference, heuristic vs random, seeds 1001–1003 × 200 games |
| 2 | T11; T12 → **your A5 decision**; T13 section 7.1 + docs + OpenSpec archives; T16; T18 | only after T11 and T13 merge: exploratory seeds 1001, then 1002 |
| 3 | T14 (+ Release Evidence in `main.json`); release prep as the last dev PR; promote rc.1 | **A1 → A1-R** → evidence PR → v2.1.0 |
| 4–6 | v2.2 (own plan) | nightly smokes; 3-night soak |
| 7–9 | v2.3 | approved D1 nights |
| 10+ | v3.0 (OpenSpec first) | from-scratch runs |

### 8.3 Working agreements
- **Definition of Ready:**
  - an issue with a milestone, citing its Appendix A ID;
  - acceptance criteria written as commands;
  - a size;
  - confirm-first approval recorded;
  - OpenSpec scaffolded if behaviour changes;
  - a run card if compute is needed;
  - dependencies done and an owner assigned.
- **Definition of Done:**
  - `V` is green;
  - coverage ≥85% with no floor lowered;
  - negative controls in place;
  - doc-sync passes and CHANGELOG `[Unreleased]` is updated;
  - the handoff skills have run;
  - OpenSpec is ticked;
  - evidence is schema-valid, with a VALIDATION-RUN entry;
  - the issue is closed.
- **Tracking:** `gh auth refresh -s project` → `scripts/bootstrap_tracking.py --apply`:
  - milestones = releases;
  - a Project with columns Backlog → Ready → In progress → Review → dev → qa → Released, plus a Compute view;
  - one issue per register row, except security items.
- **Metrics:**
  - lead time from PR to qa ≤ 7 days;
  - ≥2 RCs per month;
  - rcfix PRs per release;
  - red→green < 24 h;
  - overnight cards completed per week;
  - pre-registered runs completed;
  - primary L;
  - trained draw rate;
  - promotions per 10 iterations;
  - seeded reproduction rate.

## 9. v2.1 tasks
Each task is one PR from `feature/<task>` into dev. The cycle is: failing test → watch it fail → implement → watch it pass, plus the negative control → `V` → commit → PR. In `[accountable / executor]`, the executor is sdlc-developer unless stated otherwise.

### T0a: Windows baseline + harness (LF-E7)
- **Files:** create `scripts/calibrate_throughput.py` and `docs/evidence/windows_baseline_20261111/`.
- **Interface:** `calibrate(board_size, max_turns, mcts, k, n, device) -> {ms_per_eval, actions_per_game, s_per_game}`.
- **Test:** `tests/scripts/test_calibrate_throughput.py::test_reports_positive_rates_on_tiny_board` (board 4, 5 turns, CPU).
- **Commit:** the memory and Windows Update policy snapshot. No ETA claim yet; T0b, after B1, provides that.

### T1: Data root + pinned env (LF-E4, E6)
- **Files:**
  - create `alpha_zero/local/{__init__.py,Agent.md,pinned_env.py}` and `alpha_zero/utils/data_root.py`;
  - in `training_civ6.py`, `--checkpoint_dir` defaults to `<root>/runs/<run_id>/checkpoints`;
  - `docs/MODULE_REFERENCE.md`: add a `local` row, set the submodule count to 30 and utils to 27;
  - CLAUDE.md: change the "29 submodules" line to 30.
- **Interfaces:**
  - `resolve_data_root(*, required=False) -> Path|None`
  - `storage_root() -> Path`
  - `assert_safe_data_root(root, repo_root) -> None`
  - `prepare_pinned_run(sha, run_dir) -> PinnedEnv(python, src)`
  - `assert_running_pinned(env) -> None`
- **Tests:** `tests/unit/test_data_root.py` and `test_pinned_env.py`. The data root is refused inside the repo, inside OneDrive, or over 120 characters. The editable install fails `assert_running_pinned` (negative control).
- **Gate:** `pinned_env` is imported by T1b in the same PR series. Do not regenerate the debt baseline.

### T2: GCP isolation (LF-E5)
- **Files:**
  - create `alpha_zero/local/isolation.py` (`LocalModeViolation(BaseException)`, `enter_local_mode`);
  - guard `GCSBackend.__init__` and its `client` property (`cloud/gcp/storage.py:267,276-283`) — `storage.py` imports from `isolation`, which keeps it live;
  - add an autouse fixture in `tests/conftest.py`.
- **Interface:** `enter_local_mode(run_id, run_dir) -> dict[str,str]` (the child environment).
- **Tests:** `tests/unit/test_local_isolation.py`
  - with ADC and the bucket variable set, both `GCSBackend(...)` and `get_backend()`'s GCS branch raise under local mode;
  - a `try/except Exception` does not swallow the error;
  - the child environment contains no `GOOGLE_*`, `*_TOKEN` or `*_KEY`.

### T1b: Run-card driver (LF-E6, G11) — Sprint 1, after T1, T2, T5
- **Files:** create `scripts/run_card.py` and `configs/run_cards/` (YAML).
- **Interface:** `run_card(card: Path) -> CardResult`. It:
  - prepares the pinned env and enters local mode;
  - prices each invocation via `game_budget` and refuses a total over `--max_games` without `approval_ref`;
  - runs the invocations sequentially, with logs under `<D>`;
  - writes `run.json` and fails the card on any interruption.
- **Tests:** `tests/scripts/test_run_card.py`
  - a 2-invocation card summing to 1,040 is refused;
  - a tiny card runs from the pinned source (the `__file__` check);
  - a killed card is marked VOID.

### T3a: Charter amendment (LF-G12) [product-owner / developer; maintainer merges]
- **Files:** `PROJECT_CHARTER.md:21,58,75,145,244` (Vertex dormant; invariant 6) and ROADMAP M2 marked dormant.
- **Verify:** `check_doc_sync`, `test_claude_md_citations`.

### T4: R1 — guard, settings, triggers (LF-G6) [ai-engineer reviews]
- **Files:**
  - `.claude/settings.json`;
  - `.claude/hooks/pretooluse_guard.sh`;
  - workflows `{ci,codeql,security-scan,scorecard}.yml`: push on dev, qa and main, and `cancel-in-progress` false on protected branches;
  - `tests/regression/test_workflow_pinning_regression.py:257-270` and `test_coverage_floors_ratchet.py:31-34,87-117` (read the protected list; on push, use `GITHUB_REF_NAME` first);
  - scaffold `openspec/changes/branching-and-release/`.
- **Test:** `tests/governance/test_pretooluse_guard.py`.
  - **Denied:** a push or commit on dev, qa or main; `git push origin HEAD:refs/tags/v9`; `gh pr merge 1`; `gh api -X POST …`; `gh api repos/o/r/git/refs -f ref=…`; `gh api graphql -f query=…`; `pwsh -c "git push origin HEAD:qa"`; deleting a protected branch.
  - **Allowed:** `git tag -l 'v2.1.0-rc.*'`; `git fetch --tags origin`; `python -m alpha_zero.training_civ6 --dry_run`.

### T5: Game-budget cost models (LF-G11)
- **Files:**
  - create `alpha_zero/utils/game_budget.py` (MODULE_REFERENCE utils count becomes 28);
  - add `--max_games`/`--approval_ref` and an `enforce()` call in `main()` after resolution, in: `training_civ6.py`, `orchestrator/__main__.py`, `evaluation/benchmark.py`, `scripts/{run_baseline_eval,run_encoder_ab,run_chess_training,run_enhanced_training,run_muzero_training,run_sweep,post_run_evaluation,validate_convergence,benchmark_batch_selfplay,local_muzero_launcher}.py`;
  - `scripts/run_civ6_ai.py`: its hard-coded 100×100 becomes 10×10;
  - out of scope: `run_llm_eval`, `run_tests_ipv4`, `run_cert_check`, and the frozen Vertex launchers;
  - update the 3 existing tests that call `main()` with large defaults.
- **Interfaces:**
  - `worst_case_games(runner, resolved) -> int`
  - `enforce(runner, resolved, max_games=1000, approval_ref=None)`, which raises `GameBudgetExceeded`.
- **Tests:** `tests/unit/test_game_budget.py`
  - `training_civ6` defaults price to **10,200** and are refused;
  - `--dry_run` prices to 8 and is allowed;
  - `--eval_games 0` drops the evaluation term;
  - a benchmark of 200 games against 3 default opponents prices to 600;
  - `--preset production` is refused;
  - `--approval_ref` lets the run through and logs it.

### T6: R2 — Branch Direction, rulesets, release scripts [release-manager reviews]
- **Files:**
  - create `.github/workflows/branch-policy.yml`, `scripts/check_branch_direction.py`, `.github/rulesets/{dev,qa,main,qa-main-merge,release-tags,legacy-name}.json` (`main.json` **without** Release Evidence), `scripts/release_version.py`, `scripts/maintainer/{tag_release.py,branch_cutover.sh,merge_backmerge.sh}`;
  - modify `.github/workflows/release.yml` (`wait-for-ci` per §3.5, ancestry, final-tag rule).
- **Tests:** on a temporary git repo fixture, every §3.4 rule. Specifically:
  - promotion is denied when main is not an ancestor;
  - a backmerge with 3 update merges is allowed;
  - a clean `cherry-pick -x` rcfix is allowed, and a hand-edited one is denied;
  - an evidence change touching `alpha_zero/` is denied;
  - `tag_release` refuses a pre-R2 commit and a final whose diff touches code;
  - `wait-for-ci` resolves PR-scoped checks via the commit's PR, using a fixture;
  - `v2.1.0-rc.1` maps to `2.1.0rc1`;
  - every check named in the rulesets is a real job.

### T3: R3 — docs, skills, PR template [product-owner reviews]
- **Files:**
  - CONTRIBUTING.md;
  - CLAUDE.md §3.2, §5.2 (a row superseding `kb/specs/20261110-public-release-readiness.md:29-33`) and §15;
  - `.claude/README.md`;
  - `.claude/skills/release/SKILL.md`: drop the tag push at `:40-42`, add `CITATION.cff`, and add `tag_release.py` as a maintainer step;
  - `pre-pr-handoff/SKILL.md`: target branch and PR type;
  - `.github/PULL_REQUEST_TEMPLATE.md`;
  - `docs/runbooks/branching-and-release.md`.
- **Verify:** `check_doc_sync`, `validate_openspec`, `test_claude_md_citations`.

### T7: Move mask (LF-B1) [qa-sdet writes the test first]
- **Files:** `alpha_zero/core/civ6_actions.py:629-635`; test `tests/unit/test_action_mask_collision.py`; OpenSpec `fix-move-mask`.
- **Test:** `test_no_invalid_move_in_200_seeded_random_games` — 0 "Invalid move" records. Negative control: drop the occupancy check and the test fails.

### T8: Search budget (LF-A11, A6, A7) — CONFIRM-FIRST [ml-scientist reviews]
- **Files:**
  - `alpha_zero/core/mcts/{gumbel_ops,gumbel,muzero}.py`, `alpha_zero/core/batch_mcts.py`;
  - `training_civ6.py`: add `--gumbel_k`, `--gumbel_sims`, `--c_scale`, `--policy_target`, `--greedy_threshold`; `--mcts gumbel` honours the sims budget.
- **Interfaces:**
  - `planned_halving_sims(k, n, num_legal) -> int`
  - `SearchStats(root_evals, sim_evals)`
- **Tests:** `tests/unit/test_search_budget.py`
  - equality at (8,24), at (16,64), and when k exceeds the legal-move count, for all 3 searches;
  - the CLI flags round-trip.

### T9: Bootstrap + provenance (LF-A15, A16)
- **Files:**
  - `core/civ6_network.py:475` (`save(…, metadata=None)`);
  - `utils/checkpoint.py` (`HorizonMismatch`, `build_provenance`, `check_provenance`; `load_network(…, expected_max_turns=None)`);
  - `training_civ6.py`: `--bootstrap`, save `checkpoint_0.pt`, metadata at `:1199/:1204`, and `resume_network_weights(…, max_turns, board_size)` at `:159`.
- **Tests:** `tests/unit/test_checkpoint_provenance.py`
  - the metadata round-trips under `weights_only=True`;
  - a `max_turns` mismatch raises on resume and on `load_network`;
  - a legacy file only warns;
  - `--bootstrap none` runs no SimpleMCTS.

### T10: Eval decoupling + diagnostics (LF-A3 CF, A14, A13) [ml-scientist reviews]
- **Files:**
  - `training_civ6.py:859-997,1132-1135,1183-1191`;
  - **modify** the existing `alpha_zero/training/diagnostics.py` (`TrainingDiagnostics` at `:41` stays unchanged) by adding `StepRecord` and `prequential_value_metrics`.
- **Interfaces:**
  - `evaluate_against_random(...) -> ScoreSummary`, plus a float shim that warns of deprecation;
  - `write_eval_json(path, summary, meta)`;
  - `prequential_value_metrics(net_prev, fresh_examples, step_records, buffer_mean) -> dict`.
- **Tests:** `tests/unit/test_eval_summary_and_diagnostics.py`
  - evaluation runs on `--eval_interval`, not `--save_interval`;
  - `--eval_games 0` runs no evaluation;
  - histogram counts equal the number of played steps, with search rollouts excluded;
  - the R²_oos baseline is correct at iteration 0.

### T11: Benchmark producer (LF-D6, A4)
- **Files:**
  - `evaluation/benchmark.py:88,102-113,640-674` (the flags in §5.1);
  - `cloud/gcp/arena.py` (`ArenaResult.games`);
  - `evaluation/scoring.py` (`pair_score_interval`, `alpha=0.025`).
- **Tests:**
  - `result.json` passes `test_evidence_contract_regression.py`;
  - the union of shards equals an unsharded run on a tiny board;
  - `--candidate heuristic` runs;
  - the pair interval and hull are computed on a fixture.

### T12: A5/A12/A17 memos [ml-scientist / developer]
- **Files:** `scripts/probe_target_sharpness.py`; `kb/specs/20261111-completed-q-alignment-memo.md`; the S5 seed-overlap memo.
- **Output:** on 200 positions with 0 games — entropy, max-p, and the share of positions above `greedy_threshold`, for the current settings vs the paper's.
- **Ends with your decision.**

### T13: Section 7.1 + reconciliation (LF-A1 registration, G10, G5, A2) [ml-scientist + product-owner]
- **Files:**
  - `docs/EVALUATION.md`: append section 7.1 with the §5.2 commands, rules and verdict table;
  - CLAUDE.md §13 — refer to it in prose as "EVALUATION.md section 7.1", never a bare "§7.1" (citation gate);
  - the VLP; ROADMAP Now/Next; `docs/model_cards/hexcrystal_encoder.md:99`;
  - `scripts/check_learning_trend.py` (exit 0/1/2);
  - archive `public-release-readiness` and `reproducible-runs`.
- **Verify:** `check_doc_sync`, `validate_openspec`, `test_claude_md_citations`, `tests/scripts/test_check_learning_trend.py` (a 2-point fixture exits 2).

### T14: Evidence verifier + Release Evidence (LF-D8) [release-manager reviews]
- **Files:**
  - create `scripts/verify_release_evidence.py` and `scripts/release_play_path.txt`;
  - add the ci.yml "Release Evidence" job;
  - **`.github/rulesets/main.json`: add Release Evidence.**
- **Tests:** `tests/scripts/test_verify_release_evidence.py`
  - exit codes 0/1/2;
  - a FAIL verdict still passes `learning_recorded`;
  - a tampered count fails;
  - a tampered verdict fails;
  - two 980-game runs exit 0;
  - one 1,040-game run without `approval_ref` exits 1;
  - the check reports n/a on a non-main base.
- **Then maintainer action 9a.**

### T15: Role subagents + validator (LF-G13) [ai-engineer]
- **Files:**
  - `.claude/agents/sdlc-*.md` (8 roles);
  - `.claude/hooks/readonly_shell_guard.sh`;
  - `scripts/validate_claude_agents.py`;
  - the 3 handoff `SKILL.md` files;
  - `.pre-commit-config.yaml`, ci.yml Governance, Makefile `validate`, CLAUDE.md §15.
- **Test:** `tests/governance/test_validate_claude_agents.py`, with one fixture per check C1–C9 and a negative control.

### T16: Local exposure (LF-G14)
- **Files:** `alpha_zero/web/config.py:69` and the inference server: listen on `127.0.0.1` with loopback-only CORS.
- **Tests:** in `tests/unit/test_serving/`.

### T17: Release v2.1 [release-manager / developer; maintainer tags and merges]
1. Merge the release prep PR (Sprint 3, last).
2. Promote to qa → `tag_release v2.1.0-rc.1`.
3. Run the A1 and A1-R cards via `run_card.py` on the rc SHA.
4. Merge the `evidence/v2.1.0-rc.1` PR into qa.
5. Merge the release PR to main → `tag_release v2.1.0` → back-merge to dev with `merge_backmerge.sh`.

### T18: Forced-win oracle over batch_gumbel and MuZero (LF-C4) [qa-sdet]
- **Files:** `tests/unit/test_forced_win_oracle.py`, parametrized over `batch_gumbel` and `muzero`.
- **Negative control:** reverting the sign handling in `core/mcts/backup.py` makes it fail.

## 10. Maintainer actions
1. **LF-G1 — the leaked W&B key:**
   - revoke it in W&B; do not test it first;
   - mark it REMEDIATED in `secret-scanning.md` and fix `:47`;
   - ask GitHub Support to purge `refs/pull/72/head`;
   - run gitleaks over `refs/pull/*` (65 PR heads never scanned);
   - enable Dependabot alerts, and push protection where licensed.
2. **LF-G16:**
   - verify the 4 `allUsers` IAM bindings with gcloud, and remove them or confirm the services are gone;
   - set Terraform `invoker_members = []`;
   - remove the user-scope `CIV6AZ_GCS_BUCKET`.
3. Approve this plan.
4. Create the machine user `civ6az-agent` (one Team seat, Write role). Give its fine-grained PAT to agent sessions (§3.6). Or choose alternative (b) or (c).
5. Merge T3a, then T4 (R1). Tag **v2.0.0** on `c214daa`.
6. Run `branch_cutover.sh` (§3.7), merging T6 (R2) during the run.
7. Bootstrap tracking (§8.3).
8. Before each overnight card: pause Windows Update, and stop llama-server and WSL.
9. Decide LF-A5/A12 after T12.
   - 9a. After the first release carrying T14 merges into main (runbook §5 9a is authoritative; "after T14 merges" was too early): PATCH Release Evidence into the `main` ruleset, and confirm a draft PR into main shows it as Expected.
   - 9b. Optionally, once R2 has merged, record its merge SHA as `CIV6AZ_R2_MERGE_SHA` (or pass `tag_release --r2-merge SHA`) so `tag_release` also checks lineage by ancestry; content-based lineage covers it without (OpenSpec `branching-and-release` 2.7).
10. Approve confirm-first items as they come up: A3, A11 (T8), L0–L4, E2, B3–B6, C3, the D1 default, D4 (coordinator), E3, F4, G7, G8, and the MLflow major version.
11. Approve any run over 1,000 games (D1 continuations).
12. Do G4 and G9. Prune branches only after G1. Update the A9 Hub card.
13. Run Civ6 sessions for H1/H2 (NG-1: inference only).

## 11. Verification
- **Per PR:** `V` and `OMP_NUM_THREADS=2 pytest tests/unit -q -m "not slow"`, plus the governance, cloud and scripts suites a PR touches.
- **Isolation:** with ADC and the bucket variable present, `run_card.py configs/run_cards/smoke.yaml` makes zero calls to `*.googleapis.com` (the T2 test plus a firewall log check).
- **Pinned env:** the run's `run.json` and preflight show `src` under `E:\civ6az-data\runs\<id>\src`.
- **Branching:**
  - first a dress rehearsal of the whole §3.2 flow in a **sandbox repo**, with the rulesets applied and `v0.0.0-rc.1`;
  - then, on the real repo, after cut-over:
    - direction denials;
    - an evidence PR;
    - a final tag is refused when code changed after the rc;
    - `wait-for-ci` passes on the rc merge commit;
    - `civ6az-agent` cannot tag or merge into qa/main.
- **Evidence:** `python scripts/verify_release_evidence.py docs/evidence/release_v2_1_0/bundle.json` exits 0, and Release Evidence is green on the release PR.
- **v2.2:** `make local-loop-smoke` (CPU, under 2 minutes), `make local-calibrate`, then a 3-night soak with a forced kill and a reboot.

## 12. Out of scope (frozen)
- **Vertex classes:** `VertexAIClient`, `VertexTuningJob`, `VertexTournamentJob`.
- **Vertex loop:** the inline loop in `vertex_orchestrator.main()`.
- **Vertex infrastructure:** `infra/vertex_launcher.py`, `infra/terraform/` (except the IAM fix), `cloudbuild*.yaml`, `Dockerfile.vertex`, `pbt_orchestrator`.
- **Cloud cost and storage:** spot, cost and CostMonitor wiring; the GCS champion inventory.
- **Scripts:** `scripts/gcp_*`, `run_cloud_training.sh`, `launch_ab_run.py`.
- **Runs:** the Vertex C1 run and the T4-spot arm.
- **Parity:** no parity with Vertex is claimed.

---

## Appendix A — Gap register
Moved to [`20261111-local-first-gap-register.md`](20261111-local-first-gap-register.md) (82 rows, `LF-` IDs).

## Appendix B — Sources
- **Web:**
  - [nkdagility](https://nkdagility.com/resources/blog/stop-promoting-branches/)
  - [GitLab Flow](https://archives.docs.gitlab.com/15.8/ee/topics/gitlab_flow.html)
  - [Databricks MLOps patterns](https://docs.databricks.com/machine-learning/mlops/deployment-patterns/)
  - [OneUptime, build once / promote everywhere](https://oneuptime.com/blog/post/2026-07-28-build-once-promote-everywhere/view)
  - [Devtoolbox, branching 2026](https://devtoolbox.dedyn.io/blog/git-branching-strategies-guide)
  - [GitHub ruleset rules](https://docs.github.com/en/enterprise-cloud@latest/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)
  - [Fishtest mathematics](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)
- **Context7:**
  - `/github/docs`: ruleset schema, tag rulesets, bypass modes, Team-plan availability.
  - `/websites/github_en_actions`: `merge_group`, branch filters.
  - `/websites/mlflow`: aliases.
  - Subagents could not reach the Context7 plugin and used docs.github.com instead.
- **GitHub:** fetched to origin `491cbdc` at plan time. Runs re-clone a pinned SHA via `git archive` (§4.3).
