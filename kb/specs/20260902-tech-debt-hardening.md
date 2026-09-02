# Spec: Tech-Debt Reduction and Hardening Plan (2026-09-02)

**Role:** Planning → DEV-SQE
**Session:** session_01EcSsbbTYiSkmZsMVyipm7G, branch `claude/tech-debt-hardening-plan-xrn9sp`
**Status:** proposed, awaiting the two authorizations in §8

Every number below was measured on commit `062339e` (default branch head, 2026-08-30)
on 2026-09-02, and the rule that produced it is stated next to it. A bare count is not
reproducible; re-measure with the rule before acting on a stale figure.

---

## 1. Premise corrections (read first)

The request asked to "get CI/CD into a green state". **CI is already green** on the
default branch: `ci.yml` run 33307885495 on `062339e` succeeded on every job
(lint, test ×2, cloud-tests, security, governance, eval-harness, coverage). The three
`failure`/`cancelled` runs immediately before it were mid-work pushes to the
`god-files-decomposition` branch; the failure was a suite collecting zero items
(exit 5) and it was fixed in-branch.

So the real problem is not a red CI. It is that a green CI **measures about half the
code**:

| Measure | Value | Rule |
|---|---|---|
| `alpha_zero/` files | 367 / 168,224 lines | `find alpha_zero -name '*.py'` excl. `__pycache__` |
| Files omitted from the coverage gate | **157 (43%)** | `[tool.coverage.run].omit` patterns applied to that file list |
| Lines omitted from the coverage gate | **80,621 (48%)** | same |
| Omit-list entries | 45 `alpha_zero/` patterns (98 incl. report mirror) | `pyproject.toml` |
| `torch.load` without `weights_only` | 8 sites | grep; 6 of the 8 are in omitted modules |
| `pickle.loads` under `# noqa: S301` | 6 sites | grep; 3 of 4 files omitted |

The security findings and the coverage omissions overlap almost exactly. The 85% gate is
true and irrelevant for the deployed path (`alpha_zero/cloud/*`), which is where every
bug fixed in #113 lived (CLAUDE.md §5.3 already says this).

Second correction: the repository has **one unremediated credential leak** (a W&B API
key in commit `87cd68a2`, CLAUDE.md §5.3). No hardening plan is credible while that
stands. It is workstream 0, and it is a human action (rotate the key), not code.

Third: this plan is the fourth "reduce god files" effort on record (#115 split
`civ6_mcts.py`, the PLR0915 gate, `DryRunProfile`). Each landed one file per PR. The
scale below (27 files over 1,000 lines) means "decompose everything" is a quarter of
work, not a sprint. The plan therefore ranks files by **live and defect-dense**, not by
size, and explicitly declines to touch dead ones (delete those instead).

---

## 2. Baseline (2026-09-02)

### Size and shape

| Measure | Value | Rule |
|---|---|---|
| Files > 1,000 lines | **27** | `wc -l` over `alpha_zero/**/*.py` |
| Files > 800 lines | 41 | same |
| Largest | `cloud/gcp/vertex_orchestrator.py` 2,166; `interfaces/unciv_interface.py` 1,935; `orchestrator/orchestrator.py` 1,899; `bridge/protocol.py` 1,878; `core/civ6_network.py` 1,685 | same |
| Functions > 100 statements (waived) | 3 (`vertex_orchestrator.main` 248, `training_worker.main` 168, `run_enhanced_training.main` 134) | `pyproject.toml` per-file-ignores |
| Functions > 60 statements | 18 (14 in `alpha_zero/`, 4 in `scripts/`) | `ruff --select PLR0915 --config lint.pylint.max-statements=60` |
| Functions with McCabe > 15 | 33 | `ruff --select C901 --config lint.mccabe.max-complexity=15` |
| Test files > 1,000 lines | 8 (largest `tests/unit/test_losses.py` 1,654) | `wc -l` over `tests/` |
| `# type: ignore` in `alpha_zero/` | 171 | grep |
| `# noqa` in `alpha_zero/` | 78 | grep |
| ruff `per-file-ignores` entries | 23 | `pyproject.toml` |
| skip/xfail/importorskip sites in `tests/` | 134 | rule in CLAUDE.md §5.3 |

### Dead and redundant code (zero non-test import sites unless stated)

| Item | Lines | Evidence |
|---|---|---|
| `alpha_zero/mechanics/` (8 modules) | 5,863 | only its own `__init__` re-exports; coverage-omitted as "deferred" |
| LEGACY cloud workers `cloud/selfplay_worker.py`, `cloud/trainer.py`, `cloud/evaluator.py` | ~1,600 | referenced only by `dockerfiles/Dockerfile.{selfplay,trainer,evaluator}`, which no pipeline builds |
| `infra/vertex_launcher.py` | — | Makefile + CLAUDE.md only, both say "legacy, not deployed" |
| `core/action_pruning.py`, `utils/training_diagnostics.py`, `utils/simple_mcts.py`, `training/training_profiler.py`, `evaluation/muzero_vs_alphazero_benchmark.py`, `utils/benchmarks.py` | 4,175 | 0 non-test importers each |
| `cloud/gcp/idle_manager.py`, `gpu_scheduler.py`, `vertex_trainer.py`, `vertex.py`, `monitoring.py`, `preemption.py`, `distributed/cloud_coordinator.py` | 3,132 | 0 non-test importers each; `vertex.py`/`monitoring.py` have 38/32 test files, so they are test-only |
| Top-level `interfaces/` (5 modules) | — | a diverged fork of `alpha_zero/interfaces/`; one importer, `tests/unit/test_interfaces.py` |
| `sweep_config.yaml` (root) | — | W&B-format sweep, 0 references; superseded by `configs/sweep_muzero.yaml` |
| `alpha_zero/cloud/gcp/__main__.py` | — | 0 references in code, docs, YAML, Dockerfiles |
| Duplicate Dockerfiles | — | only `dockerfiles/Dockerfile.vertex` is built; **two** `cloudbuild.yaml` files build it under different image names; root `Dockerfile.inference` and `dockerfiles/Dockerfile.inference` are duplicates |
| `TournamentWandbLogger`, hexcrystal re-export shims (`core/hex_conv.py`, `hex_encoder.py`, `encoder_registry.py`, `crystalline_layers.py`) | — | test-only consumers |

Duplicated implementations of one concept:

| Concept | Definitions | Canonical |
|---|---|---|
| `setup_logging` | **11** | `utils/logging_config.py:223` |
| `create_mcts` | 3 | `core/mcts_factory.py:110` |
| `SimpleMCTS` | 2 | `core/mcts/simple.py` |
| Replay buffer classes / `ReplayBufferConfig` | 4 / 3 | `storage/replay_buffer.py` |
| `load_config` | 6 | `utils/config.py:250` |
| Elo rating | 5 | `tournament/elo.py` |
| `MCTSConfig` / `MCTSNode` | 3 / 2 | `core/mcts/base.py` |
| Dependency manifests | 5 (`pyproject`, `requirements.txt`, `-container`, `-inference`, `setup.py`) | `pyproject.toml` |

`requirements.txt` pins `torch>=1.13.0`; `pyproject.toml` pins `torch>=2.0.0`. Five
packages in `requirements.txt` have no `pyproject` counterpart at all.

### Hardcoded values (ranked by consequence)

1. **`action_space_size` has five live values: 356 (9 sites), 376 (9 sites), 381 (2,
   annotated "known incorrect" in `core/transformer_network.py:259` yet live in
   `core/jax_network.py`), 1000, 8000.** `utils/constants.py:ActionSpaceConstants`
   computes it; the cloud, transformer and JAX paths bypass that. A checkpoint from
   the 356 path cannot load into the 376 path. This is a correctness bug wearing a
   style-issue costume. [Certain: verified in source]
2. **GCP project id `hrmtrain` in 34 files** (`alpha_zero/`, `scripts/`, `configs/`),
   three with no env override (`scripts/_diagnose_sa.py`, `vm_ssh.py`,
   `_check_job.py`). `scripts/build_and_push.py` uses a *different* project id
   (`civ6-alphazero`); three different bucket names are in use.
3. **Developer home directories** in `scripts/launch_prod_v1.6.0.sh:84-86`.
4. **Env-var namespace split:** 51 distinct `CIV6_*` names vs 341 `CIV6AZ_*`
   (`grep -rhoE '"CIV6(AZ)?_[A-Z_]+"' alpha_zero | sort -u`). Promotion threshold alone
   answers to three names (`CIV6_PROMOTION_THRESHOLD`, `CIV6AZ_PROMOTION_THRESHOLD`,
   `CIV6AZ_EVAL_WIN_THRESHOLD`). `utils/env_loader.py` declares 6 keys and has one
   importer; its declared defaults (`BOARD_SIZE=8`, `ACTION_SPACE_SIZE=376`) contradict
   the code (16 and 356 dominate).
5. `promotion_threshold = 0.55` restated in 13 modules and 6 YAMLs; canonical is
   `evaluation/promotion.py:83`. `board_size`: 34 sites, 14 values. `max_turns`: 45
   sites, 13 values. Channel counts 13/84 centralised in `utils/constants.py` and
   redefined 5 and 4 times respectively.
6. `os.environ["APPDATA"]` unguarded in 3 scripts (KeyError off Windows).
7. `web/config.py` binds `0.0.0.0` by default; `civ6_local.py` uses `127.0.0.1` for
   the same server. The permissive one ships.
8. No real credentials in the tree (only the `lm-studio` placeholder). [Certain]

### Hardening gaps

| Gap | Where | Severity |
|---|---|---|
| `except Exception` → retry with `weights_only=False` | `core/checkpoint_utils.py:170-173` | **High.** Any checkpoint that fails safe parsing is re-loaded with arbitrary code execution enabled. Module is coverage-omitted. |
| 7 more `torch.load` without `weights_only` | `hierarchical/network/hierarchical_network.py:579`, `evaluation/baselines.py:588`, `training/training_safeguards.py:492`, `cloud/gcp/storage.py:73`, `cloud/gcp/preemption_handler.py:272`, `civ6_local.py:394,681` | High |
| `pickle.loads` ×6 suppressed with `# noqa: S301`, not mitigated | `storage/{gcs,local}_backend.py`, `distributed/shared_storage.py` ×2, `cloud/gcp/storage.py` | Medium (trust boundary is GCS/local disk) |
| 69 runtime `assert`s in non-test code, **and** both linters that would flag them are off (`S101` in ruff's global ignore; bandit run with `-s B101`) | top: `rag/stores/faiss_store.py` 10, `cloud/trainer.py` 8, `rag/stores/chroma_store.py` 7, `integration/game_connector.py` 7 | Medium (stripped under `-O`; the orchestrator was already fixed for this on 2026-06-20) |
| 35 `except ...: pass` | `training/curriculum_pipeline.py`, `core/framework.py`, `core/civ6_network.py` ×2 each | Medium |
| Bandit gates high/high only; JSON report uploaded, never diffed against a baseline | `ci.yml:182` | Medium |
| Codecov `fail_ci_if_error: false` | `ci.yml:314` | Low (only silent-failure switch in CI) |
| No `--strict-markers` with 19 declared markers and heavy `-m` filtering | `pyproject.toml:310` | Medium (a typo silently deselects) |
| No `permissions:` block in `ci.yml`/`deploy_spaces.yml`; no job timeouts in `release.yml`/`deploy_spaces.yml`; `softprops/action-gh-release@v2` (mutable tag) in the job holding `contents: write` | workflows | Low–Medium |
| mypy `ignore_missing_imports = true` globally; `disallow_untyped_defs` only in 9 packages (`utils`, `rag`, `storage`, `distributed`, `multiplayer`, `optimizations` unchecked) | `pyproject.toml` | Low |
| `tests/conftest.py` imports torch at module scope | — | Perf (every collection of 464 files pays it) |
| 15 open PRs, 11 from Dependabot (ruff 0.16, `actions/checkout@7`, `setup-python@7`, `cache@6`, `codecov@7`, `action-gh-release@3`), oldest from 2026-08-12 | GitHub | Process |

---

## 3. Objective and success criteria

**Objective.** Make the gate cover the deployed path, remove the code nobody runs, and
make one definition of each configuration value the only definition, without changing
training behaviour.

**Done when** (all mechanically verifiable):

- [ ] Coverage omit list ≤ 15 `alpha_zero/` patterns (from 45), and `alpha_zero/cloud/gcp/*`
      is measured. Gate still ≥ 85% globally.
- [ ] `alpha_zero/` line count reduced by ≥ 12,000 through deletion, with zero behaviour
      change in `--seed 42` determinism gate.
- [ ] Zero `torch.load` without `weights_only`, enforced by a `tests/security/` test.
- [ ] `S101` enforced on `alpha_zero/` (only `tests/*` waived), zero runtime asserts remain.
- [ ] One `action_space_size` computation, one `promotion_threshold` default, one env-var
      prefix, each with a regression test that fails when a second definition appears.
- [ ] Files > 1,000 lines: 27 → ≤ 20 (live files only; see §5.3 for the seven).
- [ ] One dependency manifest (`pyproject.toml`); `requirements*.txt` generated or deleted.
- [ ] Open PR count ≤ 3; Dependabot backlog cleared or closed with reason.
- [ ] Every new gate has a recorded negative control (`negative-control` skill).

---

## 4. Workstreams

Each workstream is one OpenSpec change and one to three PRs. Order matters; §6 gives it.

### WS0 — Rotate the leaked W&B key (human, day 0)

Not code. The maintainer rotates the key at wandb.ai; the runbook
`docs/runbooks/secret-scanning.md` records the rotation date. Optional follow-up:
purge `87cd68a2` from `origin/feature/sprint6-validation-and-nplayer-foundation` (close
PR #72 first, it points at that commit). **Blocking for calling anything "hardened".**

### WS1 — Make the gate see the deployed path (coverage)

Evidence: §1 table. CLAUDE.md §5.3 already computed the path: `cloud/gcp` measures
60.0% against 66.1% needed; global drop ~89.7% → ~83.8% if un-omitted today.

1. Un-omit `alpha_zero/cloud/*` in a **draft PR that is expected to go red**, so CI
   reports the real number. Do not plan from the derived figure (its margin is under
   half a point).
2. Omit individually the argparse/entrypoint shims (`cli.py`, `__main__.py`,
   `container_healthcheck.py`) with a one-line reason each.
3. Add tests for `quota_manager.py` (287 statements, 0%, no test file),
   `lifecycle.py`, `pipeline_cache.py`.
4. Remove `--ignore=tests/unit/test_distributed` from the coverage job or document why
   the tests it holds cannot run in CI.
5. Repeat for the next omit tier: `evaluation/{baselines,baseline_agents,reports}.py`,
   `core/checkpoint_utils.py`, `storage/gcs_backend.py` (mockable), `utils/llm_client.py`
   (already has a stub path).
6. Rewrite every remaining omit reason to be true (the "require GCP SDK" rationale was
   measured false on 2026-08-28).

Acceptance: omit list ≤ 15 patterns; gate ≥ 85%; a `tests/regression/` test asserts the
omit list never grows without a reason comment on the same line.
Estimate: 1–2 weeks (CLAUDE.md's own figure). Risk: coverage arithmetic on unverified
inputs; mitigated by step 1.

### WS2 — Delete dead code (needs authorization §8.1)

Evidence: §2 dead-code table, ~15,000 lines with zero non-test importers.

Order, one PR each, each independently revertible:

1. `sweep_config.yaml`, `alpha_zero/cloud/gcp/__main__.py`, duplicate
   `Dockerfile.inference`, second `cloudbuild.yaml` (keep the root one; it is what
   `Dockerfile.vertex` documents).
2. LEGACY cloud workers + their three Dockerfiles + `infra/vertex_launcher.py` + the
   Makefile targets `build-{selfplay,trainer,evaluator}`, `push-*`, `run-*` that
   invoke them. Update CLAUDE.md §5.1 cloud table.
3. Top-level `interfaces/` and its one test file. If any function there is not in
   `alpha_zero/interfaces/`, port it first with a test.
4. The six zero-importer utility/benchmark modules (`action_pruning`,
   `training_diagnostics`, `utils/simple_mcts`, `training_profiler`,
   `muzero_vs_alphazero_benchmark`, `utils/benchmarks`) with their tests.
5. The seven zero-importer `cloud/gcp` modules **only after** WS1 measures them: a
   module with 38 test files (`vertex.py`) may be dead in production but its tests
   encode intent; read them before deleting, and record any intent worth keeping in
   the OpenSpec change.
6. `alpha_zero/mechanics/` (5,863 lines). This is a product decision, not a code one:
   PROJECT_CHARTER lists mechanics under deferred scope. Delete-or-keep goes to the
   maintainer with the line count attached; archive-not-delete (move to
   `docs/archive/` is wrong for code; a git tag `pre-mechanics-removal` is right).

Rules: never delete a module a Dockerfile or YAML still names; grep configs first. Run
`tests/regression/test_suite_wiring_regression.py` after each deletion (it will notice
an emptied test directory). Each PR states the line count removed.

Acceptance: ≥ 12,000 lines removed; determinism gate byte-identical before/after.

### WS3 — One definition per configuration value

Evidence: §2 hardcoded-values list.

1. **`action_space_size` first** (correctness). Make `ActionSpaceConstants` the only
   computation; replace the 22 literal sites with a call; delete the 381 JAX value or
   make `jax_network` compute it. Add `tests/regression/test_action_space_single_source.py`
   that greps `alpha_zero/` for `action_space_size\s*[:=]\s*\d+` outside the constants
   module and fails on any hit. Verify a checkpoint round-trips between the former 356
   and 376 paths.
2. **Env-var prefix.** Adopt `CIV6AZ_` only. Provide a 2-release deprecation shim in
   `utils/env_loader.py`: reading `CIV6_X` warns once and maps to `CIV6AZ_X`. Extend
   `env_loader.py` to be the registry it claims to be: every `CIV6AZ_*` name is
   declared there with its default, and a regression test asserts every
   `os.environ.get("CIV6AZ_...")` call site names a registered key. Fix the registry's
   own contradictory defaults while doing it.
3. Collapse the three promotion-threshold env names to `CIV6AZ_PROMOTION_THRESHOLD`
   and the 13 restated `0.55` literals to `evaluation.promotion.DEFAULT_PROMOTION_THRESHOLD`.
   Do **not** change the value; the promotion-gate runbook owns that decision.
4. `hrmtrain`, bucket names, region: one `CloudDefaults` in `cloud/config.py`, read
   from env, no literal fallback for project id (fail loud with the variable name).
   The three no-override scripts get the override. Remove developer home paths from
   `launch_prod_v1.6.0.sh` (take `REPO_ROOT` from the script's own location).
5. `board_size`/`max_turns`/`num_simulations`: do **not** unify values (they legitimately
   differ per context). Unify the *source*: each context's default lives in its config
   dataclass; ad-hoc literals in function signatures reference it.
6. `web/config.py` default host → `127.0.0.1`; `0.0.0.0` only by explicit flag.
7. Guard the three `os.environ["APPDATA"]` sites.

Acceptance: the three regression tests above exist and each has a recorded negative
control. Estimate: 1 week.

### WS4 — Hardening

1. `core/checkpoint_utils.py:170-173`: remove the automatic downgrade. Safe load fails
   → raise with a message naming `--allow-unsafe-checkpoint` (explicit opt-in flag,
   logged at WARNING). Add the test that feeds a non-tensor pickle and asserts no
   fallback.
2. The other 7 `torch.load` sites: `weights_only=True`. Add
   `tests/security/test_no_unsafe_torch_load.py` (AST walk, not regex, so a multi-line
   call is seen).
3. `pickle.loads` ×6: keep, but replace `# noqa: S301` with a documented trust boundary
   and an HMAC or size check where the source is a bucket. At minimum one shared
   `storage/serialization.py` so the policy has one home.
4. Re-enable `S101` for `alpha_zero/`; convert 69 asserts to explicit raises (the
   `_require()` pattern from the 2026-06-20 orchestrator fix). Drop `-s B101` from
   bandit and lower bandit to `--severity-level medium`, with the JSON report diffed
   against a committed baseline so new findings fail and old ones burn down visibly.
5. The 35 `except: pass` sites: each becomes `logger.debug` at minimum, or a narrowed
   exception type.
6. CI: add `--strict-markers` to `addopts`; `permissions: contents: read` at workflow
   top; job timeouts in `release.yml` and `deploy_spaces.yml`; pin
   `softprops/action-gh-release` to a SHA; `fail_ci_if_error: true` for Codecov (or
   drop Codecov if the artifact is unused).
7. Merge or close the Dependabot backlog. The ruff 0.16 bump changes rule behaviour, so
   it lands alone with the pre-commit pin updated in the same PR (the pin has one
   definition on purpose).

Acceptance: every item above has a test or a CI gate that fails when reverted.
Estimate: 1 week.

### WS5 — God-file decomposition (live files only)

Evidence: §2 size table plus the #115 lesson: moving a function does not shorten it,
and the three cloud workers carry 38 `mock.patch` string targets, so a split there
breaks tests silently unless the shim is pinned for patchability.

Do these seven, in this order, one PR each:

| File | Lines | Why this one |
|---|---|---|
| `cloud/gcp/vertex_orchestrator.py` | 2,166 | the deployed orchestrator; `main` is 248 statements, on the waiver list |
| `cloud/gcp/selfplay_worker.py` | 1,635 | deployed; four functions between 66 and 83 statements |
| `cloud/gcp/training_worker.py` | 1,358 | deployed; `main` 168 statements |
| `orchestrator/orchestrator.py` | 1,899 | coverage-omitted **because** it is one run loop; splitting is the path to measuring it |
| `core/civ6_network.py` | 1,685 | model definitions and training utilities in one file; the network-architecture confirm gate applies (§3.2), so split file boundaries only, no layer changes |
| `training_civ6.py` | 1,108 | the reference training path; owns `ReplayBuffer`, `TrainingExample`, the loop, and the CLI |
| `bridge/protocol.py` | 1,878 | live bridge; 17 action types in one module |

Method (from #115): decompose by responsibility, keep the old module as a permanent
re-export shim with explicit `__all__`, and add a patchability test (not an identity
test) per shim. Delete a `per-file-ignores` PLR0915 entry when its `main` is split;
never raise the threshold. Lower the threshold to 80 once the three waivers are gone.

Explicitly **not** in scope: `interfaces/unciv_interface.py` (1,935; secondary
interface with a planned-but-absent method), `envs/civrealm_wrapper.py` (1,678;
coverage-omitted external wrapper), `integration/game_connector.py` (test-only),
`hierarchical/*` (deferred). Delete or defer, do not refactor.

Acceptance: files > 1,000 lines ≤ 20; zero PLR0915 waivers; `--seed 42` gate
byte-identical. Estimate: 1 PR per file, ~1 day each once WS2 has cleared the noise.

### WS6 — Consolidate duplicates

Evidence: §2 duplicate table.

1. `setup_logging` 11 → 1: the six `cloud/gcp/*` copies become `from
   alpha_zero.utils.logging_config import setup_logging`. Pure deletion of copies.
2. `create_mcts` 3 → 1: `utils/factories.create_mcts` delegates to
   `core/mcts_factory.create_mcts`; the `simple_mcts` copy goes with WS2.
3. Elo 5 → 1 rating computation in `tournament/elo.py`; the plateau detector and the
   evaluator call it. Keep their own *config* classes if the semantics differ.
4. `MCTSConfig` 3 → 1 plus subclasses for hierarchical.
5. Replay buffers: **defer.** Four classes serve genuinely different storage modes and
   are on the deployed path. Document the mapping in `docs/MODULE_REFERENCE.md` instead.
6. Dependency manifests: `pyproject.toml` is the source. `requirements.txt` becomes
   `pip-compile` output (or is deleted, with the README install line updated).
   `requirements-container.txt` and `-inference.txt` map to extras.
   `setup.py` goes when `Dockerfile.base` goes (WS2).
7. Compat shims: keep `core/civ6_mcts.py` (5 live consumers) and `MonitoringConfig`
   (4 scripts). Delete `TournamentWandbLogger` and the four hexcrystal re-export shims
   after the removal-window date is recorded in CHANGELOG.

Acceptance: a `tests/regression/` test counts definitions of `setup_logging`,
`create_mcts`, `calculate_expected_score` in `alpha_zero/` and pins each to 1.
Estimate: 3–4 days.

### WS7 — Enterprise organisation

Do not reorganise directories wholesale. PR #69 ("Organize project structure to meet
modern Python standards", draft since 2026-08-12) tried this and is stale for a
reason: a move of 367 files invalidates every `mock.patch` string in 504 test files.
Instead:

1. **Layering document.** `docs/architecture/layering.md` naming the allowed import
   direction: `utils` → `envs`/`core` → `training`/`evaluation` → `orchestrator`/`cloud`
   → entrypoints. Enforce with a `tests/regression/test_import_layering.py` (import
   graph walk, stdlib only) that fails on a back-edge. Start with the current graph
   as the allowlist and shrink it.
2. **Entry points.** The four root `run_*.py` scripts become `python -m alpha_zero.<x>`
   or `[project.scripts]` console entries; `run_civ6_ai.py` is deleted (DEV-SQE
   summary already records that intent). README's run lines change with them.
3. **Type-check scope.** Extend `disallow_untyped_defs` to `utils`, `storage`,
   `distributed`, `multiplayer` (one package per PR). Turn on `warn_unused_ignores`
   and delete the `# type: ignore` sites it reports (expect a large share of the 171).
4. **Test layout.** `tests/conftest.py` moves the torch/numpy imports into the fixtures
   that need them; measure collection time before and after. Split the 8 test files
   over 1,000 lines along the same seams as the module they test.
5. **`per-file-ignores`** 23 → ≤ 12: every remaining entry has a reason and a date.

Acceptance: the layering test exists and passes; `run_*.py` gone; per-file-ignores
≤ 12. Estimate: 1 week, interleaved.

---

## 5. What this plan deliberately does not do

- Does not change any training hyperparameter, promotion threshold value, or network
  architecture. Those have their own runbooks and confirm gates.
- Does not move `alpha_zero/` subpackages to new paths. See WS7.
- Does not decompose dead files. Deleting is cheaper than refactoring.
- Does not raise `max-statements` or add omit entries. Both are ratchets and only
  tighten.
- Does not treat "the suite passes" as evidence of anything; each new gate ships with
  its negative control.

---

## 6. Sequencing

```
Week 0   WS0 (rotate key)  ─ human, blocks nothing technically, blocks the word "hardened"
Week 1   WS1.1 (red draft PR) ‖ WS2.1–2.4 (deletions) ‖ WS4.6 (CI config)  ‖ WS4.7 (Dependabot)
Week 2   WS3.1 (action space) → WS4.1–4.2 (torch.load) → WS2.5 (cloud dead modules after WS1 measures)
Week 3   WS1.2–1.6 (coverage tier 1–2) ‖ WS3.2–3.7 (env/config) ‖ WS6.1–6.2
Week 4–6 WS5 one file per PR, in the table order ‖ WS4.3–4.5 ‖ WS6.3–6.7
Week 6–7 WS7, then re-measure §2 and update CLAUDE.md §5.2/§5.3
```

Dependencies that are real: WS2.5 after WS1 (measure before deleting test-heavy
modules); WS5 after WS2 (do not decompose what will be deleted); WS4.4 (S101) after
WS2 (a third of the asserts are in modules on the deletion list).

Total: 6–7 weeks for one engineer at one PR per day, every PR small enough to revert
alone. [Likely: the #115 and coverage estimates in CLAUDE.md are the calibration]

---

## 7. Gates this plan adds (each must be shown to fail before it merges)

| Gate | Guards |
|---|---|
| `tests/security/test_no_unsafe_torch_load.py` | WS4.1–4.2 |
| `tests/regression/test_action_space_single_source.py` | WS3.1 |
| `tests/regression/test_env_registry.py` | WS3.2 |
| `tests/regression/test_single_definitions.py` | WS6 |
| `tests/regression/test_import_layering.py` | WS7.1 |
| `tests/regression/test_coverage_omit_ratchet.py` | WS1 |
| S101 on `alpha_zero/`, bandit baseline diff, `--strict-markers` | WS4 |

---

## 8. Authorizations required from the maintainer

CLAUDE.md §3.2 requires confirmation for deletions over 100 lines and for changes
touching more than three modules. This plan cannot proceed past WS1 without:

1. **Deletion authorization** for the WS2 list, items 1–5 (≈ 9,000 lines, all with
   zero non-test importers). Item 6 (`mechanics/`, 5,863 lines) is a separate yes/no.
2. **Architectural authorization** for WS3.2 (env prefix unification touches ~8
   modules), WS5 (seven decompositions), and WS7.2 (entry-point relocation).

Give both in one sentence on the PR and DEV-SQE can start on WS1 and WS2.1 the same
day. Withhold either and the plan degrades to WS0, WS1, WS4.1–4.2 and WS4.6, which
are still worth doing alone.

---

## 9. Confidence

- [Certain] every count in §2 with a stated rule; the `checkpoint_utils` fallback;
  the S101/B101 double-off; CI green on `062339e`; the 356/376/381 split.
- [Likely] the dead-module list. "Zero non-test importers" was checked by grep over
  Python, YAML, Dockerfiles and Makefile; dynamic imports by string
  (`importlib.import_module(name)`) would not be seen. WS2 step 0 for each PR is a
  grep for the module's *string* name.
- [Likely] the 6–7 week estimate.
- [Guessing] that `mechanics/` is deletable rather than deferred. That is why it is a
  separate authorization.
