# Spec: Tech-Debt Reduction and Hardening Plan (2026-09-02, v2)

**Role:** Planning → DEV-SQE
**Session:** session_01EcSsbbTYiSkmZsMVyipm7G, branch `claude/tech-debt-hardening-plan-xrn9sp`, PR #116
**Status:** v2, supersedes v1 after an independent adversarial review (§0). Awaiting the
authorizations in §9.

Every number below was measured on 2026-09-02 against commit `062339e` (default-branch
head) and the rule or command that produced it is stated beside it. Where v1 and v2
disagree, v2 is the re-measurement and §0 says why.

---

## 0. What the peer review changed (v1 → v2)

v1 was built on three grep-based audits and four hand spot-checks. An independent
reviewer with no access to v1's reasoning re-derived the load-bearing claims. Five were
wrong, six were understated, and three mechanisms were replaced. In order of consequence:

| v1 claim | v2 correction | Evidence |
|---|---|---|
| ~15,000 dead lines, "zero non-test importers" by grep | **79 modules / 40,503 lines** have no live importer, by AST import graph with re-export resolution (`scripts/audit_import_graph.py`, shipped in this PR). The grep undercounted by 2.7x and also produced false positives: `training_profiler` is re-exported by `training/__init__`; `vertex.py`/`monitoring.py` "38/32 test files" were substring hits on other names, real count 1 and 1. | `python scripts/audit_import_graph.py` |
| Delete dead code, then widen coverage | **34 of the 79 dead modules (14,194 lines) are measured by the coverage gate.** Test-only code is inflating the 85% figure; deleting it will lower the reported number. WS1 and WS2 are one workstream now, not two. | audit JSON ∩ `[tool.coverage.run].omit` |
| WS1: add tests for `quota_manager.py`, `lifecycle.py`, `pipeline_cache.py` | All three are dead (784 / 800 / 793 lines, zero live importers; `quota_manager` has zero test importers too). Delete, don't test. | audit output |
| `action_space_size` 356 vs 376 is the #1 correctness bug, "checkpoint incompatibility" | 356 is a stale formula (the found-city block omitted; `utils/constants.py:417-420` documents it) but every 356 site is in dead, deferred, or docstring code. The deployed paths derive 376 from the environment. **The literal on a live path is 381**: `core/civ6_network.py:65` (`NetworkConfig` default, "MicroCiv default"), `civ6_local.py:638` (the live bridge), `core/muzero_network.py:83`, `multiplayer/network_heads.py:56`. A checkpoint "round-trip between 356 and 376" is impossible by construction (policy-head shape differs), so that acceptance test is replaced by a checkpoint inventory. | grep `381`, `microciv.py:198-206`, `civ6_actions.py:574-578` |
| WS1 mechanism: a draft PR "expected to go red" | Wrong tool. The `cloud-tests` job already runs 1,025 cloud tests on `.[dev]`; add `--cov=alpha_zero.cloud --cov-fail-under=<measured floor>` there with a separate coverage config. Global gate untouched, no red PR, floors ratchet up from a committed file. | `ci.yml:118-146` |
| WS1.4: `--ignore=tests/unit/test_distributed` hides tests | That directory does not exist; the suites are flat `tests/unit/test_distributed_*.py` and already run. The flag is a no-op to delete. | `ls tests/unit/test_distributed` |
| "≥ 12,000 lines deleted" as a success criterion | Did not follow from the list authorized: items 1–5 summed to 9,181 inside `alpha_zero/`. Replaced by tiered lists whose sums are stated (§2.3). | `wc -l` |
| `mechanics/` = 8 modules / 5,863 lines | 8 **dead** modules / 5,863 lines inside a 12-file / 8,001-line package. Both numbers now stated. | `wc -l alpha_zero/mechanics/*.py` |
| WS2.1: "two cloudbuild.yaml files" | Three: `infra/cloudbuild-gameplay.yaml` builds `dockerfiles/Dockerfile.inference`. And `docker-compose.yml` builds all three LEGACY worker Dockerfiles (`make local-test`). Neither was on v1's touch list. | `infra/`, `docker-compose.yml:56-132` |
| "One PR per day, 6–7 weeks" | 52 PR-sized units in the plan; at that cadence it is 10–12 weeks for a single maintainer who is author, reviewer and ratifier. CI is 24–27 min per run, not the bottleneck. v2 offers a Core scope that fits 6 weeks and a Full scope that does not. | run 33307885495 timings |
| 8 test files > 1,000 lines | 27. v1 printed the top 8 and reported the head as the count. | `find tests -name '*.py' \| xargs wc -l \| awk '$1>1000'` |
| "Ruff pin has one definition" | Three: `pyproject.toml:26`, `ci.yml:208`, `.pre-commit-config.yaml:30`. | grep `0.15.8` |
| `§3.2` "no commits to main" is enforced by the guard | **The default branch is not `main`** (it is `claude/civ6-alphazero-implementation-vMnun`) and `pretooluse_guard.sh` denies pushes only to the literal `main`. The rule is prose. | `git remote show origin`, guard lines 45-49 |
| 6–7 week estimate, no risk register, no PR disposition | Added §6 (risk/rollback/approver per workstream) and §7 (all 15 open PRs). | — |

Kept from v1 without change: CI is green on the default branch; 48% of `alpha_zero/`
lines are coverage-omitted and overlap the unsafe `torch.load` / `pickle.loads` sites;
the W&B key in `87cd68a2` is unrotated; `S101` and bandit `B101` are both off; the
`checkpoint_utils.py:170-173` fallback to `weights_only=False` is real.

---

## 1. Premise corrections

1. **CI is green.** Run 33307885495 on `062339e` passed all 8 jobs in 24 minutes. The
   three failed/cancelled runs before it were mid-work pushes on the #115 branch.
   "Get CI green" is not the work; "make green mean something" is.
2. **The gate measures about half the package and some of what it measures is dead.**
   157 of 367 files (43%) and 80,621 of 168,224 lines (48%) are omitted by the 45
   `alpha_zero/` omit patterns (rule: fnmatch of each pattern against the file path;
   the reviewer's rule, which also matches `*/__init__.py` and test paths, gives 183 /
   50%). Separately, 34 dead modules are measured and inflate the number.
3. **One credential is leaked and unrotated** (`87cd68a2`, head of open PR #72).
4. **The branch-protection rule is prose.** See §0, last row.
5. **This is the fourth god-file effort** (#115, PLR0915, `DryRunProfile`). Each landed
   one file per PR. The plan does not pretend otherwise.

---

## 2. Baseline (reproducible)

### 2.1 Commands that produce every figure

| Figure | Command |
|---|---|
| Dead modules, tiers | `python scripts/audit_import_graph.py [--json]` |
| Files > 1,000 lines | `find alpha_zero -name '*.py' -not -path '*/__pycache__/*' \| xargs wc -l \| awk '$1>1000'` |
| Functions > N statements | `ruff check alpha_zero scripts --select PLR0915 --config lint.pylint.max-statements=N` |
| Complexity > 15 | `ruff check alpha_zero --select C901 --config lint.mccabe.max-complexity=15` |
| Coverage-omitted files | pattern list from `[tool.coverage.run].omit`, fnmatch against file paths (to be scripted as `scripts/measure_debt.py`, WS-M) |
| Skip sites | `grep -rEc '@pytest\.mark\.(skip\|skipif\|xfail)\|pytest\.(skip\|importorskip)\(' tests/` |
| Unsafe loads | `grep -rn 'torch.load(' alpha_zero` minus lines/blocks with `weights_only` (AST test in WS4 replaces this) |

### 2.2 Size and shape

| Measure | Value |
|---|---|
| `alpha_zero/` | 367 files, 168,224 lines |
| Files > 1,000 lines | 27 (of which 11 are coverage-omitted and 5 are dead) |
| Functions > 100 statements (waived) | 3: `vertex_orchestrator.main` 248, `training_worker.main` 168, `run_enhanced_training.main` 134 |
| Functions > 60 statements | 18 |
| McCabe > 15 | 33 |
| Test files > 1,000 lines | 27 |
| `# type: ignore` / `# noqa` in `alpha_zero/` | 171 / 78 |
| ruff `per-file-ignores` | 23 |
| Skip/xfail/importorskip sites | 134 |
| Declared pytest markers | 18, no `--strict-markers` |
| `create_mcts` / `setup_logging` / Elo / `load_config` definitions | 4 / 11 / 5 / 6 |
| Dependency manifests | 6: `pyproject.toml`, `requirements.txt`, `-container`, `-inference`, `setup.py`, `uv.lock` (868 KB, referenced by nothing) |
| Open PRs | 15 (11 Dependabot, 4 human; oldest 2026-03-19) |

### 2.3 Dead code, by AST import graph, tiered

"Dead" = no non-test module imports it, after resolving `from pkg import Name` through
package `__init__` re-export tables, with a package's own re-export not counted as a
use and a side-effect `import pkg.mod` counted. Entry points and modules named in
Dockerfiles / cloudbuild / Makefile / k8s / shell / configs are excluded (12 modules,
4,564 lines; they are wired by string, not import).

**Tier A — delete under one blanket authorization (19,882 lines, 44 modules).**
Superseded, unwired, or scaffolding no document claims as a feature.

| Group | Modules (lines) |
|---|---|
| `cloud/` superseded by `cloud/gcp/` (8,343) | `gcp/lifecycle` 800, `gcs_storage` 798, `gcp/pipeline_cache` 793, `gcp/quota_manager` 784 (0 tests), `gcp/idle_manager` 720, `cloud_logging` 626 (0 tests), `vertex_launcher` 619, `gcp/scaling_config` 607, `gcp/worker_autoscaler` 579, `gcp/gpu_scheduler` 539, `gcp/vertex_trainer` 438, `gcp/vertex` 434, `gcp/monitoring` 320, `preemption` 286 (duplicate of `gcp/preemption_handler`) |
| `core/` scaffolding (2,368) | `action_pruning` 1,014, `network_heads` 542, `mixins` 348, `timeout_config` 188, `buffer_config` 181, four hexcrystal re-export shims 95 (CLAUDE.md §5.2 promised these paths; announce a removal window in CHANGELOG first) |
| `utils/` (2,248) | `training_diagnostics` 1,011, `training_config_enhanced` 733, `simple_mcts` 504 (second `SimpleMCTS`) |
| `training/` (2,197) | `curriculum_pipeline` 1,611 (orchestrator has its own `curriculum.py`; carries 3 of the 356 literals), `training_profiler` 586 (re-exported only) |
| `evaluation/` (1,012) | `sequential_testing` 598, `reports` 414 |
| `optimizations/` (1,431) | `benchmarks` 701, `config` 435, `cuda_graph_cache` 295 |
| `human_interface/` (1,042) | `input_handler` 531, `visualizer` 511 (the rest of the package is live via `web/`) |
| singles (1,241) | `tournament/wandb_dashboards` 456 (`TournamentWandbLogger`, "public" per CLAUDE.md §5.3, so CHANGELOG deprecation, then delete), `exceptions` 376 (a hierarchy nothing raises), `envs/docker_config` 245, `opponents/network_opponent` 106, `opponents/base` 58 |

**Tier B — documented-deferred features: one yes/no each (18,133 lines, 28 modules).**
Dead by graph but named in CLAUDE.md, ROADMAP, MODULE_REFERENCE or a runbook. Deleting
is a product decision; the alternative is a test that exercises the documented entry.

| Decision | Modules (lines) | Where it is claimed |
|---|---|---|
| B1 `mechanics/` | 8 modules, 5,863 (package 12 files / 8,001) | CLAUDE.md §7, MODULE_REFERENCE; charter lists mechanics as deferred |
| B2 PBT | `cloud/gcp/pbt_orchestrator` 846, `pbt/coordinator` 572, `pbt/muzero_hyperparams` 258 (1,676) | §5.2 2026-03-01; Phase 6 spec |
| B3 JAX backend | `core/jax_trainer` 427, `jax_checkpoint` 338, `jax_debug` 245, `jax_mcts_adapter` 121 (1,131); carries the wrong 381 default in `jax_network` | omit list "requires JAX" |
| B4 CivRealm | `envs/civrealm_wrapper` 1,678 | omit list |
| B5 game connector + Civ6 interface | `integration/game_connector` 1,358, `interfaces/civ6_interface` 383 (1,741) | §7 |
| B6 HF Hub mixins | `core/transformer_hub_mixin` 221, `core/muzero_hub_mixin` 167 (388) | `docs/HF_PUBLISHING.md` |
| B7 agent tooling | `utils/agent_tools` 783, `utils/test_runner` 589 (1,372) | CLAUDE.md §9.4, §15 |
| B8 imitation pretraining | `training/imitation_pretraining` 793 | — |
| B9 MuZero benchmark | `evaluation/muzero_vs_alphazero_benchmark` 601 | `docs/MUZERO_CLOUD_INTEGRATION.md:74-128`; `scripts/search_efficiency_report.py:341` reads its output |
| B10 RAG pipeline | `rag/pipeline` 534 | CLAUDE.md §11.4 usage example |
| B11 inference engine | `inference/inference_engine` 429 | `docs/INFERENCE_SERVICE.md` |
| B12 SpacetimeDB sinks | `spacetimedb/metrics_sink` 222, `weight_sync` 197 (419) | §7 |
| B13 monitoring extras | `monitoring/mlflow_evaluator` 276, `tournament_dashboards` 134 (410) | §5.2 2026-05-21 |
| B14 distributed | `distributed/parameter_server` 703, `cloud_coordinator` 395 (1,098) | §3.2 CONFIRM-FIRST "distributed training coordination" |

**Tier C — keep (2,488 lines, 7 modules).** `multiplayer/{diplomacy_agent, game_connector,
mcts, turn_manager, opponent_model, env_adapter, observation_model}`. Dead by graph, but
CLAUDE.md §13 item 2(c) names `NPlayerMCTS.search` as the dependency fog-of-war is
blocked on. Keep, and add the missing call site when that item is picked up.

Outside `alpha_zero/` (not in the 40,503): top-level `interfaces/` (5 modules, one
importer `tests/unit/test_interfaces.py`; a diverged fork), `infra/vertex_launcher.py`
(Makefile targets + **PROJECT_CHARTER §2 Mission names it**, so its removal is a
charter amendment, not a deletion), `sweep_config.yaml` (0 references), root
`Dockerfile.inference` (duplicate of `dockerfiles/Dockerfile.inference`, which
`infra/cloudbuild-gameplay.yaml` builds), `setup.py` (kept only for `Dockerfile.base`).

**Coverage interaction.** 45 of the 79 dead modules (26,309 lines) are omitted; 34
(14,194 lines) are measured. Eleven omit patterns match only dead modules and vanish for
free when they are deleted. Expect the global percentage to **fall** after Tier A;
WS1's floors are set from the post-deletion measurement, not before.

### 2.4 Hardcoded values, re-ranked

1. **`action_space_size = 381` on live paths** (`core/civ6_network.py:65`,
   `civ6_local.py:638`, `core/muzero_network.py:83`, `multiplayer/network_heads.py:56`)
   while the environment's `MicroCivActionSpace.total_size` is 376 and
   `transformer_network.py:255-262` already warns that 381 is wrong. 356 appears on 16
   lines, all in Tier A/B modules or docstrings.
2. GCP project id `hrmtrain` in 34 files, three with no env override; a second project
   id (`civ6-alphazero`) and three bucket names in use.
3. Developer home directories in `scripts/launch_prod_v1.6.0.sh:84-86`.
4. Env-var namespace: 51 distinct `CIV6_*` vs 341 `CIV6AZ_*`; promotion threshold has
   three names; `utils/env_loader.py` declares 6 keys and its defaults contradict the code.
5. `promotion_threshold = 0.55` in 13 modules and 6 YAMLs; `board_size` 34 sites / 14
   values; `max_turns` 45 / 13; channel counts 13/84 redefined 5 and 4 times.
6. `os.environ["APPDATA"]` unguarded in 3 scripts; `web/config.py` binds `0.0.0.0` by
   default. No real credentials in the tree.

### 2.5 Hardening gaps

| Gap | Where |
|---|---|
| `except Exception` → retry with `weights_only=False` | `core/checkpoint_utils.py:170-173` (coverage-omitted) |
| 7 more `torch.load` without `weights_only` | `hierarchical/network/hierarchical_network.py:579`, `evaluation/baselines.py:588`, `training/training_safeguards.py:492`, `cloud/gcp/storage.py:73`, `cloud/gcp/preemption_handler.py:272`, `civ6_local.py:394,681` |
| `pickle.loads` ×6 under `# noqa: S301` | `storage/{gcs,local}_backend.py`, `distributed/shared_storage.py` ×2, `cloud/gcp/storage.py` |
| 69 runtime `assert`s; `S101` globally ignored **and** bandit run with `-s B101` | top: `rag/stores/faiss_store.py` 10, `cloud/trainer.py` 8 |
| 35 `except ...: pass` | `training/curriculum_pipeline.py` (Tier A), `core/framework.py`, `core/civ6_network.py` |
| Guard checks literal `main`; default branch is not `main` | `.claude/hooks/pretooluse_guard.sh:45-49` |
| Bandit high/high only, JSON never diffed; Codecov `fail_ci_if_error: false`; no `permissions:` in `ci.yml`/`deploy_spaces.yml`; no timeouts in `release.yml`/`deploy_spaces.yml`; `softprops/action-gh-release@v2` mutable tag in the `contents: write` job; no protected `release` environment or `v*` tag rule | workflows |
| No `--strict-markers` with 18 markers and `-m` filtering everywhere | `pyproject.toml:310` |
| Pre-commit runs `pytest tests/unit/ -x` on every commit, a suite that cannot complete in the dev sandbox, so the hook is either skipped or dead | `.pre-commit-config.yaml:71-73` |
| Python version: CLAUDE.md §2.1 and PROJECT_CHARTER invariant 2 say ≥ 3.10; `pyproject` requires ≥ 3.11; CI and the SessionStart hook enforce 3.11 | doc drift `check_doc_sync` does not catch |
| mypy `ignore_missing_imports` global; `disallow_untyped_defs` in 9 packages only | `pyproject.toml` |
| `tests/conftest.py` imports torch at module scope | every collection |

---

## 3. Objective and success criteria

**Objective.** Make the coverage gate measure the deployed path and nothing dead; remove
the code nobody runs; make one definition of each configuration value the only one;
close the hardening gaps that let a hostile checkpoint execute code. No training
hyperparameter, threshold value, or architecture changes.

**Done when** (all mechanically verifiable, each with the gate that verifies it):

- [ ] Tier A is gone (44 modules / 19,882 lines) and every Tier B row has a recorded
      decision (delete, or a test that exercises its documented entry). Gate: the
      dead-module ratchet test (WS-M) reports 0 Tier A and no new dead module.
- [ ] `alpha_zero/cloud/gcp` has a per-package coverage floor in CI at its measured
      value and the omit list is ≤ 15 `alpha_zero/` patterns, each with a same-line
      reason. Global gate stays ≥ 85%. Gate: `coverage-floors.json` check + omit ratchet.
- [ ] Zero `torch.load` without `weights_only=True` (AST test), and the fallback in
      `checkpoint_utils` is an explicit flag, preceded by a checkpoint inventory.
- [ ] `S101` enforced on `alpha_zero/`; `-s B101` removed from bandit; zero runtime asserts.
- [ ] One `action_space_size` source (`MicroCivActionSpace.total_size` /
      `ActionSpaceConstants`), zero `381` or `356` literals (AST test), one
      `promotion_threshold` default, one env-var prefix with a deprecation shim.
- [ ] Files > 1,000 lines: 27 − 5 (Tier A/B deletions) − 7 (WS5) = **≤ 15**, with no
      decomposition child > 900 lines. Zero PLR0915 waivers.
- [ ] The branch guard reads the default branch from git, or the default branch is `main`.
- [ ] One dependency manifest plus one lock mechanism (uv **or** pip-tools, decided).
- [ ] Open PRs ≤ 3 with §7 executed.
- [ ] Every new gate merged with a recorded negative control.

---

## 4. Workstreams

### WS0 — Rotate the key, close #72, dispose of the PR backlog (human, day 0)

- Rotate the W&B key; record the date in `docs/runbooks/secret-scanning.md`.
- Close #72: its head **is** the leaked commit, it is 81 files / 20 commits against a
  base 6 months stale, and its one real fix (`--model_version` incumbent loading) shipped
  in v1.3.0. Purging the object afterwards is a force-push, which the guard denies by
  design; it is a human step and is cleanup, not remediation.
- Execute §7 for the other 14 PRs.

### WS-M — Measurement first (this PR + one follow-up)

- `scripts/audit_import_graph.py` (shipped here, 9 tests). Its output is the
  authoritative dead list; §2.3 was generated from it.
- `scripts/measure_debt.py`: emits §2.2 and §2.3 as JSON. Committed baseline
  `docs/evidence/tech_debt_baseline/2026-09-02.json`.
- `tests/regression/test_debt_ratchet.py`: every metric in the baseline may only move in
  the good direction; a PR that adds a dead module, an omit pattern, a `type: ignore`,
  or a file over 1,000 lines fails with the metric named. Negative control: reintroduce
  one dead module and show the failure.

### WS1 — Make the gate measure the deployed path

1. In the existing `cloud-tests` job add `--cov=alpha_zero.cloud
   --cov-config=.coveragerc-cloud --cov-fail-under=<F>` where `.coveragerc-cloud` has no
   `cloud/*` omit and `<F>` is the value measured **after** Tier A lands (CLAUDE.md's
   60.0% was measured with the dead modules present). Global job untouched.
2. Commit `coverage-floors.json` (`{"alpha_zero": 85, "alpha_zero/cloud/gcp": F}`) and
   `scripts/check_coverage_floors.py` reading `coverage json`; floors only move up.
3. Regression test: `[tool.coverage.run].omit` and `[tool.coverage.report].omit` are
   byte-identical (they must be edited in lockstep today), and every entry carries a
   same-line reason.
4. Delete the no-op `--ignore=tests/unit/test_distributed`.
5. Un-omit, tier by tier, with a floor each: `evaluation/{baselines,baseline_agents}.py`,
   `core/checkpoint_utils.py` (after WS4.1), `storage/gcs_backend.py` (mockable),
   `utils/llm_client.py` (has a stub path), `orchestrator/orchestrator.py` (after WS5).
6. Rewrite every remaining omit reason to be true; the "require GCP SDK" rationale was
   measured false on 2026-08-28.

Net omit count only decreases (v1's "never add an entry" was contradicted by its own
WS1.2; three entrypoint shims may be added if a whole-directory pattern is removed).

### WS2 — Delete (Tier A blanket; Tier B per decision)

**Step 0 for every deletion PR**, in this order, because each has already bitten:
`Makefile`, `docker-compose.yml`, the three `cloudbuild*.yaml`, `infra/`, `dockerfiles/`,
every `*/Agent.md` (`validate_agents` A6 fails the governance job on a dangling file
reference unless the doc is edited or marked `<!-- validate-agents: allow -->`),
`docs/*.md` outside `archive/`, the owning package `__init__.py` (re-exports and
`__all__`), `PROJECT_CHARTER.md`, and `docs/MODULE_REFERENCE.md` + CLAUDE.md §7 counts
(`check_doc_sync` D2 fails on a submodule-count change). Grep the module's dotted name
**and** its file path; the audit script's `deploy_refs` column does the build-file part.

Order, one PR per row, each stating lines removed and coverage delta:

1. Zero-risk: `sweep_config.yaml`, root `Dockerfile.inference`, `cloud/gcp/__main__.py`,
   `cloud/preemption.py` (duplicate), `exceptions.py`, `opponents/{base,network_opponent}`.
2. `cloud/` Tier A (14 modules, 8,343). Includes `cloud/vertex_launcher.py`, which
   `cloud/__init__.py:77` re-exports and nothing consumes.
3. LEGACY workers + their three Dockerfiles + `docker-compose.yml` services +
   `make local-test`/`local-smoke`/`build-*`/`push-*`/`run-*` targets + CLAUDE.md §5.1
   cloud table. This is the PR that touches the most non-Python files; do it alone.
4. `core/`, `utils/`, `training/`, `evaluation/`, `optimizations/`, `human_interface/`
   Tier A groups, one PR per package.
5. Top-level `interfaces/` + its test (supersedes #69's move of the same tree).
6. `TournamentWandbLogger` and the four hexcrystal shims: CHANGELOG deprecation entry
   first, delete one release later.
7. Tier B rows, as decided, one PR each. B14 (`distributed/`) is a §3.2 CONFIRM-FIRST
   item on its own.

Never delete a Tier C module. Run the wiring gate and the debt ratchet after each PR.

### WS3 — One definition per configuration value

1. **381 first.** Replace the four live defaults with `MicroCivActionSpace().total_size`
   (or `ACTION_SPACE.compute_microciv_action_space_size()`); fix `env_loader.py`'s
   registry to agree. Gate: `tests/regression/test_action_space_single_source.py`, an
   **AST** walk (not regex) that fails on any integer literal assigned to a name or
   keyword matching `action_space_size` outside the two source modules, and on
   `np.ones(376)`-style literals. Replace the impossible round-trip test with a
   read-only inventory: load the policy-head shape of the last N champion checkpoints in
   GCS and record which value production is actually on (`docs/evidence/`).
2. Env prefix: `CIV6AZ_` only, with a two-release shim in `env_loader.py` that warns once
   per legacy `CIV6_X` and maps it. Extend the registry to every `CIV6AZ_*` name, with a
   test that every `os.environ.get("CIV6AZ_...")` call site names a registered key.
   **External state:** the maintainer's `.env`, `configs/*.yaml` Vertex job envs, running
   jobs. Rollback is the shim itself; the shim is why this is two releases.
3. Promotion threshold: one env name, thirteen literals → `DEFAULT_PROMOTION_THRESHOLD`.
   Value unchanged; the promotion-gate runbook owns the value.
4. `hrmtrain` / buckets / region → `CloudDefaults` in `cloud/config.py`, read from env,
   **no literal fallback for project id**. This breaks every script run without the env
   var set, on purpose; the error names the variable. CHANGELOG entry. Remove the home
   paths from `launch_prod_v1.6.0.sh`.
5. `board_size` / `max_turns` / `num_simulations`: unify the *source* (each context's
   config dataclass), not the values.
6. `web/config.py` host → `127.0.0.1`; `0.0.0.0` behind a flag. User-visible; CHANGELOG.
7. Guard the three `os.environ["APPDATA"]` sites.

### WS4 — Hardening

1. **Inventory before flipping.** Read-only scan of the champion lineage: `torch.load(...,
   weights_only=True)` over the last N GCS versions, log failures. If any incumbent
   fails, ship a one-shot re-save migration first. Then `checkpoint_utils.py` raises on
   safe-load failure unless `--allow-unsafe-checkpoint` is passed (WARNING logged).
   Negative control: a non-tensor pickle must not load without the flag.
2. Other 7 `torch.load` sites → `weights_only=True`; `tests/security/test_no_unsafe_torch_load.py`
   as an AST walk.
3. `pickle.loads` ×6: one `storage/serialization.py` with a documented trust boundary
   and a size check; drop the `# noqa`s.
4. `S101` on for `alpha_zero/`; 69 asserts → `_require()`-style raises; bandit without
   `-s B101`, `--severity-level medium`, JSON diffed against a committed baseline.
5. 35 `except: pass` → narrowed type or `logger.debug`.
6. Workflows and hooks: `--strict-markers`; `permissions: contents: read` at the top of
   `ci.yml` and `deploy_spaces.yml`; job timeouts in `release.yml`/`deploy_spaces.yml`;
   pin `softprops/action-gh-release` to a SHA; protected `release` environment with a
   required reviewer and a `v*` tag-protection rule; Codecov `fail_ci_if_error: true` or
   remove the upload; **guard reads the default branch** from
   `git symbolic-ref refs/remotes/origin/HEAD` (or rename the default branch to `main`,
   which is the maintainer's call); pre-commit `pytest-check` restricted to
   `tests/regression/` (CI owns the suite).
7. Dependabot backlog: the maintainer merges (DEV-SQE cannot). The ruff 0.16 bump lands
   alone and updates all **three** pins; add a regression test that the three agree.

### WS5 — Decompose the seven live god files (Full scope only)

Prerequisites: WS0 closes #72 (it modifies 4 of the 7); WS3.1 lands first
(`civ6_network.py` carries the 381 default); WS2 has cleared the noise.

| Order | File | Lines | Note |
|---|---|---|---|
| 1 | `orchestrator/orchestrator.py` | 1,899 | omitted *because* it is one run loop; splitting is how it becomes measurable, and it has no `mock.patch` string problem |
| 2 | `training_civ6.py` | 1,108 | reference path; `ReplayBuffer`, `TrainingExample`, loop, CLI |
| 3 | `bridge/protocol.py` | 1,878 | 17 action types; PR #106 touches `bridge/` and must be closed or merged first |
| 4 | `core/civ6_network.py` | 1,685 | file boundaries only, no layer changes (§3.2 architecture gate) |
| 5–7 | `cloud/gcp/vertex_orchestrator.py` 2,166, `selfplay_worker.py` 1,635, `training_worker.py` 1,358 | | **Reverses the 2026-08-30 decision** ("not decomposed: 38 `mock.patch` targets"). New evidence required before starting: a test that enumerates every `mock.patch("alpha_zero.cloud.gcp...")` string in `tests/` and asserts each resolves and is consumed through the shim. If that test cannot be made to pass on the current tree, stop and re-record the decision. |

Method: decompose by responsibility, permanent re-export shim with explicit `__all__`,
patchability test per shim, no child > 900 lines, delete the PLR0915 waiver when the
`main` is split, never raise the threshold.

### WS6 — Consolidate duplicates

1. `setup_logging` 11 → 1; `create_mcts` 4 → 1; Elo 5 → 1; `MCTSConfig` 3 → 1.
2. Replay buffers: defer; document the mapping in MODULE_REFERENCE.
3. Dependencies: **decide uv or pip-tools**. If uv: CI installs from `uv.lock`, a test
   asserts lock ⊇ pyproject, `requirements*.txt` deleted or generated. If not: delete
   `uv.lock`. `setup.py` goes with `Dockerfile.base` (WS2.3).
4. Shims kept: `core/civ6_mcts.py` (5 live consumers), `MonitoringConfig` (4 scripts).
5. Gate: `tests/regression/test_single_definitions.py` pins each count to 1.

### WS7 — Organisation and drift

1. Supersede #69: adopt its `run_*.py` → `scripts/` move and `setup.py` removal via
   this plan's PRs; do **not** adopt its `interfaces/` → `alpha_zero/game_interfaces/`
   move (the tree is dead, WS2.5 deletes it). Close #69 as superseded with that reason.
2. Python version: charter amendment (maintainer-merged) + CLAUDE.md §2.1 → 3.11; add
   the version triple to `check_doc_sync` as D8.
3. Layering document + `tests/regression/test_import_layering.py` (stdlib import-graph
   walk; the audit script's graph builder is reusable). Current back-edges become the
   allowlist and shrink.
4. Type-check scope: `disallow_untyped_defs` to `utils`, `storage`, `distributed`,
   `multiplayer`, one package per PR; `warn_unused_ignores = true` and delete what it
   reports.
5. Skip sites: resolve the 3 genuine items CLAUDE.md §5.3 names; ratchet the count.
6. `tests/conftest.py` torch/numpy imports moved into fixtures; measure collection time.
7. `per-file-ignores` 23 → ≤ 12, each with a reason and date.

---

## 5. Scope options and estimate

52 PR-sized units at one validated PR per day, CI floor 25 minutes per run, one
maintainer reviewing. Two honest scopes:

| Scope | Workstreams | Units | Calendar |
|---|---|---|---|
| **Core** | WS0, WS-M, WS1, WS2 Tier A, WS3.1, WS3.4, WS4 | ~28 | 6 weeks |
| **Full** | Core + WS2 Tier B, WS3.2–3.7, WS5, WS6, WS7 | ~52 | 10–12 weeks |

Core delivers: the gate measures the deployed path, 19,882 dead lines gone, the live
`381` bug fixed, no unsafe checkpoint path, the guard protects the real default branch.
Full adds the decompositions and the namespace/dependency consolidation.

Sequencing (Core):

```
Day 0    WS0 (rotate, close #72, §7 dispositions)  ‖  WS-M baseline + ratchet
Week 1   WS2.1–2.3 (zero-risk, cloud Tier A, LEGACY workers)  ‖  WS4.6 (workflow/hook config)
Week 2   WS2.4–2.5  →  measure  →  WS1.1–1.4 (floors from the post-deletion number)
Week 3   WS3.1 (381 + inventory)  →  WS4.1–4.2 (checkpoint inventory, weights_only)
Week 4   WS4.3–4.5 (pickle policy, S101, except-pass)  ‖  WS3.4 (project id)
Week 5   WS1.5–1.6 (omit tiers)  ‖  WS4.7 (Dependabot, three pins)
Week 6   WS2.6–2.7 (deprecations, Tier B decisions that were yes)  →  re-measure, CLAUDE.md §5
```

Dependencies that are real: WS1 floors after WS2 Tier A (dead code inflates the number);
WS4.4 after WS2 (a third of the asserts are in Tier A modules); WS5 after WS0, WS2, WS3.1.

---

## 6. Risk register

| Workstream | External state touched | Rollback | Approver |
|---|---|---|---|
| WS0 | W&B account, PR #72 | rotate again; reopen PR | maintainer |
| WS2.3 | `docker-compose.yml`, Makefile targets operators may script against | revert PR; targets return | maintainer (CLAUDE.md §3.2 > 100 lines) |
| WS2 Tier B14 | distributed coordination code | revert | maintainer (CONFIRM-FIRST) |
| WS3.2 | maintainer `.env`, `configs/*.yaml` job envs, running Vertex jobs | shim keeps `CIV6_*` working for two releases | maintainer |
| WS3.4 | every script invocation without `CIV6AZ_VERTEX_PROJECT` | revert; documented in CHANGELOG | maintainer |
| WS4.1 | incumbent GCS checkpoints (269 versions) | inventory first; migration re-save; flag | maintainer |
| WS4.6 guard | local hook behaviour for everyone | revert hook | maintainer |
| WS4.6 branch rename | every open PR base, CI triggers, docs | do not rename unless the maintainer chooses it | maintainer |
| WS5.5–5.7 | 38 `mock.patch` targets | stop if the enumeration test cannot pass | maintainer (reverses a recorded decision) |
| WS6.3 | CI install path | keep both until the lock test passes | maintainer |
| WS7.2 | PROJECT_CHARTER invariant | charter amendment protocol | maintainer |

Not independently revertible: WS3.2, WS3.4, WS4.1. Each has a CHANGELOG entry and a
shim or flag instead.

---

## 7. Open PR disposition (15)

| PR | Head | Disposition | Reason |
|---|---|---|---|
| #72 | `feature/sprint6-…` (`87cd68a2`) | **Close** | head is the leaked-key commit; 81 files vs a stale base; the incumbent-loading fix shipped in v1.3.0 |
| #106 | `prep-full-gameplay` | Rebase or close | adds `pytest.ini` that silently overrides `[tool.pytest.ini_options]` (drops `--timeout=120`, redefines markers) and a `[1.7.0] — 2026-07-30` changelog heading on a 1.8.0 repo (`check_doc_sync` D7 fails); the Lua atomic-write change is worth salvaging as a small PR |
| #69 | `copilot/organize-project-directories` | **Close as superseded** | 17 files; its `run_*.py`→`scripts/` and `setup.py` removal are adopted by WS7.1/WS6.3; its `interfaces/` move is replaced by deletion (WS2.5) |
| #49 | `strategic-planning-sdk` (+42,586 lines) | Close | a general planning SDK; PROJECT_CHARTER NG-6 |
| #32 | `design-strategic-planning-sdk` | Close | scaffold for #49 |
| #103 | ruff 0.15.8 → 0.16.2 | Merge alone (WS4.7) | update all three pins in the same PR |
| #94–#98, #102 | actions bumps, gcp monitoring | Merge in one batch after #103 | none touches code |
| #99, #100, #101 | prometheus, tensorboard, chromadb ranges | Merge or close | these packages are in `requirements.txt` only; WS6.3 decides whether they exist at all |
| #116 | this plan | Draft until §9 answered | — |

---

## 8. Gates this plan adds

| Gate | Guards | Validators / jobs touched |
|---|---|---|
| `tests/regression/test_debt_ratchet.py` + baseline JSON | WS-M, all | `test` job |
| `scripts/check_coverage_floors.py` + `coverage-floors.json` | WS1 | `cloud-tests`, `coverage` |
| omit-lists-identical + same-line-reason test | WS1 | `test` |
| `tests/regression/test_action_space_single_source.py` (AST) | WS3.1 | `test` |
| `tests/regression/test_env_registry.py` | WS3.2 | `test` |
| `tests/security/test_no_unsafe_torch_load.py` (AST) | WS4.1–4.2 | `security` |
| bandit baseline diff; S101 on; `--strict-markers` | WS4 | `security`, `lint`, all pytest jobs |
| three-ruff-pins-agree test | WS4.7 | `governance` |
| `tests/regression/test_single_definitions.py` | WS6 | `test` |
| `tests/regression/test_import_layering.py` | WS7.3 | `test` |
| `check_doc_sync` D8 (Python version triple) | WS7.2 | `governance` |
| `mock.patch` target enumeration test | WS5.5–5.7 | `cloud-tests` |

Every gate merges with a recorded negative control (the `negative-control` skill).

---

## 9. Authorizations required (CLAUDE.md §3.2)

Enumerated by trigger, because v1 missed three:

1. **Deletions > 100 lines:** Tier A as one blanket authorization (44 modules, 19,882
   lines, §2.3). Tier B as fourteen yes/no rows (B1–B14). Top-level `interfaces/`,
   `sweep_config.yaml`, root `Dockerfile.inference`, LEGACY Dockerfiles and
   `docker-compose.yml` services.
2. **Distributed training coordination:** B14 (`distributed/parameter_server`,
   `cloud_coordinator`), and WS6.1's edits to six `cloud/gcp/*` modules.
3. **Architectural / > 3 modules:** WS3.2 (env prefix, ~8 modules), WS3.4 (project-id
   fail-loud), WS5 (seven decompositions; 5–7 also reverse a recorded decision), WS7.1
   (entry-point relocation).
4. **Breaking public API:** `TournamentWandbLogger` and the hexcrystal shims (with
   deprecation window), `web/config.py` host default.
5. **Charter amendments:** Python ≥ 3.11 (invariant 2); `infra/vertex_launcher.py`
   removal (Mission §2 names it); `mechanics/` (B1) if deleted.
6. **Maintainer-only actions:** merging Dependabot PRs, closing #72/#69/#49/#32, the
   default-branch decision, the key rotation.

Answer 1 (Tier A) and 3 (WS3.1 is not on the list; it is a bug fix) and DEV-SQE can run
Core from day 0. Without 1, the executable subset is WS0, WS-M, WS1.1–1.4 on the current
tree, WS3.1, WS4.1–4.2, WS4.6.

---

## 10. Confidence

- [Certain] every count in §2 that names a command; the tiers' membership by import
  graph; the 381 sites; the guard/default-branch mismatch; CI timings; PR file lists.
- [Likely] Tier A is safe to delete. The audit resolves re-exports, lazy maps, literal
  `importlib` calls and build-file strings, but not `getattr(module, name)` or a
  module path assembled at runtime. WS2 step 0 exists for that residue.
- [Likely] the Core scope fits 6 weeks; the Full scope does not fit fewer than 10.
- [Guessing] which Tier B rows the maintainer will keep. That is why they are rows.
- [Guessing] the post-deletion coverage number. WS1 sets floors from measurement, not
  from this document.
