# Changelog detail for 2.0.0 (archived)

The full, dated change narrative that accumulated under `[Unreleased]` between
1.8.0 (2026-07-12) and 2.0.0 (2026-11-10). `CHANGELOG.md` carries the summary; this
file is the record. Section headings are unchanged from the original entries.

### Fixed — One sign-correct value backup for every search variant (2026-09-04)

A node's value is stored from its own mover's perspective, so backing a leaf
evaluation up to the root flips the sign only on edges where the acting player
*changes*. Only `base`/`puct` did that; `gumbel`, `simple`, `muzero` and the
batch path used plain negamax, which assumes one ply per turn. MicroCiv does not
work that way: `MicroCivEnv` advances
`current_player` only inside `_end_turn`, and **145 of 200 seeded steps (72.5%)
leave the acting player unchanged**. Every one of those edges was signed wrong,
so the sign of a node's mean value depended on the parity of the depth at which
its leaves were evaluated, and `value_sum` averaged `+v` and `−v` estimates
together.

`core/mcts/base.py` was corrected for this on 2026-08-25 and the correction was
never propagated. `grep -c to_play` per module before this change: `base` 10,
`puct` 9, **`gumbel` 0, `simple` 0, `muzero` 0** — and `gumbel` is
`DEFAULT_MCTS_TYPE`, what `cloud/gcp/selfplay_worker.py` takes by default and
what `cloud/gcp/arena.py` hard-defaults to. The data generator and the promotion
judge shared the defect.

- **New `alpha_zero/core/mcts/backup.py`** (249 lines) owns the single
  implementation: `backpropagate_path`, `perspective_sign`,
  `should_track_to_play`, `env_current_player`, plus backup diagnostics. All
  five variants call it, and selection now uses the same perspective rule as
  backup — negating unconditionally there makes PUCT prefer the child that
  *minimises* the mover's own value.
- Nodes whose `to_play` is unknown fall back to per-ply alternation, so
  `MuZeroMCTS` — which searches a learned latent model and cannot observe the
  mover past the root — is byte-identical and says so at its call site. The
  fallback no longer mutates the running value, which previously let one unknown
  node invert every node above it.
- The `>2-player` guard that `puct` already carried is now on the three variants
  made `to_play`-aware: a scalar sign flip cannot express a 3-player game, and
  `--num_players 4` defaults to `gumbel`.
- Layering: `backup` is layer 0 in `MCTS_LAYERS`, below `base`, because `gumbel`
  and `muzero` must depend on neither `base` nor each other. The layering test
  grew from a flat leaf set to an explicit layer map plus a check that every
  module is assigned a layer, so a new module cannot escape the rule by omission.

### Changed — Gumbel's policy target is the completed-Q distribution, not a one-hot (2026-09-04)

`GumbelMCTS.search` returned a one-hot over the selected action (1 non-zero entry
of a 376-wide action space), stored verbatim as the policy training target. That
trains the policy head by cloning a single argmax, discards the search's own
improvement signal, and keeps the temperature schedule permanently inert, because
a delta always trips the greedy short-circuit in `select_action`.

Visit counts are the wrong replacement for this search: sequential halving
allocates the budget equally among survivors, so the root visit distribution is
near-uniform by construction and the survivors are typically tied. The signal
lives in the completed Q values and the Gumbel scores.

- `gumbel_ops.build_completed_q_policy` implements Danihelka et al.'s
  `softmax(logits + sigma(completedQ))`. This is the first use of the `c_scale`
  config field, declared since 2026 and read nowhere.
- New config fields on **both** `GumbelMCTSConfig` and `BatchMCTSConfig`:
  `policy_target` (`completed_q` — the new default — `visit_counts`, or
  `one_hot`) and `greedy_threshold` (was a fixed module constant). `one_hot`
  reproduces pre-2026-09 runs exactly. Both are validated in `__post_init__`.
- The sequential and batch self-play paths share the target, because a trainer
  reading a rolling replay window cannot tell which worker produced a shard; the
  batch path had been left on the one-hot entirely.
- Three tests that asserted the one-hot contract now assert a normalised
  distribution, with the legacy target covered explicitly through the config flag
  rather than deleted.

### Fixed — Seeds derive from a stream name, and self-play varies by iteration (2026-09-04)

`training_civ6.py` derived RNG seeds by adding hand-picked offsets to
`config.seed`. **None of the self-play or evaluation seeds carried an
`iteration` term** while the DataLoader's did — one of four streams threaded it,
which is what makes it a bug rather than a design choice. A seeded 20-iteration
run therefore regenerated the same games twenty times and scored them on the same
fixed maps, so the seeded learning-trend run the plan is built around could not
have produced a trend.

- **New `alpha_zero/utils/seeding.py`**: `derive_seed(base, stream, *coords)`
  hashes a stream *name* plus coordinates with BLAKE2b, and `SeedStream` names
  the seven streams (`selfplay.env`, `selfplay.search`, `eval.env`,
  `eval.search`, `eval.opponent`, `replay.sampler`, `data.loader`). Streams
  cannot collide unless spelled the same, iteration is a parameter rather than a
  stride, and the mapping is stable across processes (unlike Python's salted
  `hash`). `derive_seed(None, ...)` returns `None`, so unseeded runs stay
  unseeded.
- **Self-play varies with `iteration`; evaluation deliberately does not.**
  Evaluation is a held-out set: comparing iteration *N* against *N+1* is only
  meaningful if both were scored on the same positions. `run_self_play` and
  `evaluate_against_random` take an `iteration` argument, defaulting to 0.
- `EVAL_SEED_OFFSET` is **deprecated, not deleted** — nothing in the package
  reads it any more.
- Behaviour note: seeded runs now draw different streams than before, so action
  sequences differ. The determinism suite asserts the property (same seed twice
  is identical, different seeds differ), not any literal value.

### Changed — `MicroCivEnv.clone()` shares immutable terrain (2026-09-04)

`clone()` deep-copied the terrain dict on every call, and MCTS calls it once per
simulation. Terrain is generated once in `_generate_map` and never mutated
afterwards, so the clone now shares the reference. Measured over 300 clones per
configuration: **1,149 µs → 354 µs at `board_size=8`, 3,121 µs → 357 µs at
`board_size=16`.** `tests/unit/test_microciv_clone.py` plays a seeded game and
fails if any tile changed, so the invariant this relies on is enforced rather
than asserted in a comment; the guard fingerprints every `Tile` field rather than
two hand-picked ones.

### Changed — `pip install -e '.[dev]'` installs something (2026-09-04)

`chess` was in the `dev` extra, its sdist fails to build on current
Debian/Ubuntu setuptools, and a failing extra aborts the whole install — so the
command printed in README, CONTRIBUTING and CLAUDE.md installed **nothing**.

- `chess` is out of `dev`; the `[chess]` extra still provides it.
- Every chess-dependent test module now `pytest.importorskip`s it — the four
  `tests/unit/test_chess_*`, plus `test_stockfish_opponent`,
  `test_opponent_protocol`, `test_curriculum` and `test_env_factory` — with
  matching `E402` per-file ignores for the guards that precede their imports.
- `alpha_zero/opponents/__init__.py` resolves its exports lazily (PEP 562, the
  same pattern the root package uses), because eagerly importing
  `stockfish_opponent` made an optional extra mandatory for any test that merely
  touched the subpackage.
- Tracked debt moves accordingly. The baseline regenerated with this change
  records `pytest_skip_sites` 126 → 140 and `ruff_per_file_ignores` 21 → 25:
  both increases *are* the guards, i.e. the fix.

### Security — Every workflow action pinned to a SHA; the push guard was inert here (2026-09-04)

- **All 27 `uses:` refs across the three workflows are SHA-pinned** with a
  version comment (`ci.yml` 20, `deploy_spaces.yml` 4, `release.yml` 3); 26 were
  on mutable tags. `tests/regression/test_workflow_pinning_regression.py` parses
  the workflows with `yaml` and fails on any unpinned ref.
- **The push guard did not protect this repository.** The 2026-09-02 fix resolves
  the default branch at runtime, but `CIV6AZ_DEFAULT_BRANCH` was set nowhere and
  `origin/HEAD` is not a symbolic ref in an agent checkout, so it fell through to
  `main` — and this repository's default branch is
  `claude/civ6-alphazero-implementation-vMnun`. Pushing to the real default
  branch was allowed, verified by driving the hook. `.claude/settings.json` now
  sets `CIV6AZ_DEFAULT_BRANCH` (and `OMP_NUM_THREADS=2`). Same shape as the `jq`
  bug it replaced: a resolution chain that degrades silently.

### Fixed — Gates that were green while failing open (2026-09-04)

- **The dead-code audit counted a build copy as an external importer.**
  `scripts/audit_import_graph.py` skipped venvs and egg-info but not `build/`,
  `dist/`, `.tox/` or `.eggs/`. A build copy holds every package `__init__`
  importing by *absolute* name under a `build.lib.*` module name, so it is
  credited as an outside user of the very modules the audit exists to find:
  53 modules / 24,451 lines read as 30 / 15,120 on an identical tree. The
  negative control plants a `build/lib/alpha_zero/__init__.py`; its first version
  planted a *relative* import and passed either way, so it only became a control
  once the plant matched what a build actually writes.
- **CI's `pull_request` trigger filtered on `main`,** which is not this
  repository's default branch, so pull requests into the real default branch
  produced no `pull_request`-event run at all. The filter is removed rather than
  re-pointed, so a future branch rename cannot re-create the trap.
- **`coverage-floors.json` could be lowered silently** — the ratchet was a
  docstring. `tests/regression/test_coverage_floors_ratchet.py` now compares
  against the merge base and also refuses a dropped package.
- **The Makefile exports `OMP_NUM_THREADS ?= 2`.** CLAUDE.md has documented that
  cap as required for months while nothing set it, so `make verify` — the command
  the handoff skills mandate — was documented as hanging.
- Dependabot covers the `docker` ecosystem (three Dockerfiles were in no
  ecosystem), groups minor and patch bumps into one PR per ecosystem, and adds a
  release cooldown. The previous configuration produced a standing set of
  single-package PRs, none of which was ever merged.

### Added — Gates for each of the fixes above (2026-09-04)

85 tests across five files, each written and watched to fail before the fix it
guards:

- `tests/unit/test_mcts_backup_correctness.py` — a forced-win oracle over a
  synthetic environment where one player moves *k* consecutive plies and one
  action is a forced win, parameterised over depth to catch both parities.
  Reverting the backup fails 6 `gumbel` / 6 `puct` / 6 `simple` cases. The
  oracle's first version passed **vacuously for `gumbel`**, the variant it
  existed to protect: with a visit-count target the policy was exactly
  `[1/3, 1/3, 1/3]` and `np.argmax` broke the tie toward index 0, which was the
  winning action by coincidence. It now sweeps the winning index and rejects a
  policy uniform across two or more actions.
- `tests/unit/test_seeding.py`, `tests/unit/test_microciv_clone.py`,
  `tests/regression/test_workflow_pinning_regression.py`,
  `tests/regression/test_coverage_floors_ratchet.py`.

### Added — 2026 standards audit and the correctness-first re-sequencing (2026-09-04)

- [`docs/audits/2026-09-04-standards-audit.md`](docs/audits/2026-09-04-standards-audit.md)
  — six read-only audit passes over CI/supply chain, application security, code
  quality, the test suite, docs/governance and a primary-source 2026 rubric,
  with every gate-integrity and high-severity claim reproduced by hand. Its §0
  records four corrections to its own findings.
- `kb/specs/20260904-correctness-first-plan.md` — the re-sequencing the audit
  forced. It adds one workstream (**WS-S**, search correctness), moves **WS-L**
  (the seeded learning-trend run) behind it, folds 9 of the audit's 23 items into
  existing workstreams, and withdraws 2. `kb/specs/20260902-validated-learning-plan.md`
  is amended in place rather than superseded.

### Changed (BREAKING) — checkpoints load safely by default (2026-09-02)

Both checkpoint loaders retried a failed safe load with `weights_only=False`,
executing arbitrary code from the file without the caller asking. The retry now
requires an explicit opt-in.

- `alpha_zero.utils.checkpoint.safe_load_checkpoint(allow_unsafe_fallback=)`
  defaults to `False` (was `True`), and the six live call sites
  (`core/civ6_network.py` ×3, `core/muzero_network.py`, `distributed/learner.py`,
  `distributed/ddp_learner.py`, `training/muzero_trainer.py`) no longer pass
  `True`. `alpha_zero.core.checkpoint_utils.load_checkpoint` gains
  `allow_unsafe=`; `weights_only=False` there is a deprecated alias for it.
- **What breaks:** a checkpoint whose payload is not `weights_only`-clean now
  raises `UnsafeCheckpointError` instead of loading. Freshly-saved models are
  unaffected; the risk is legacy artifacts. The escape hatch is
  `allow_unsafe_fallback=True` or `CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1`.
- `MuZeroTrainer.save_checkpoint` used to pickle its config dataclass, so **its
  own** checkpoints could only be reopened unsafely. It now stores a serialized
  dict; checkpoints written before this need the env var once and a re-save.
- An I/O failure (missing file, truncated download) re-raises as `OSError`
  rather than being reported as an unsafe checkpoint — the old behaviour told
  operators to disable the security control to fix a failed download.

### Changed (BREAKING) — `action_space_size` defaults derive from the environment (2026-09-02)

`NetworkConfig`, `MuZeroConfig`, `NPlayerNetworkConfig`, `MicroCivNetwork`,
`civ6_local._build_network_for_bridge` and `factories.create_simultaneous_mcts`
defaulted to `381` (or `356`), while `MicroCivActionSpace.total_size` is `376`.
All now derive from `alpha_zero.utils.constants.default_action_space_size()`;
the argparse defaults in `cloud/gcp/evaluation_worker.py` and
`vertex_orchestrator.py` do too.

**What breaks:** a network built with the old default now has a 376-wide policy
head. Passing `action_space_size=381` explicitly still works, so checkpoints
trained on the old default remain loadable when the caller states the width.

### Removed — Tier A dead code, measured by import graph (2026-09-02)

`scripts/audit_import_graph.py` (stdlib AST, resolves package re-exports)
found 79 `alpha_zero/` modules with no live importer — 40,503 lines, a quarter
of the package — of which 34 were *measured* by the coverage gate and so
inflating the 85% figure with test-only code. The plan in
`kb/specs/20260902-tech-debt-hardening.md` tiers them; this release removes
Tier A (superseded or unwired, claimed by no document as a feature):

- **`alpha_zero/cloud/`**: the LEGACY `selfplay_worker`, `trainer`, `evaluator`
  (superseded by `cloud/gcp/`), `gcs_storage`, `cloud_logging`,
  `vertex_launcher`, `preemption` (duplicate of `gcp/preemption_handler`), and
  under `gcp/`: `lifecycle`, `pipeline_cache`, `quota_manager`, `idle_manager`,
  `scaling_config`, `worker_autoscaler`, `gpu_scheduler`, `vertex_trainer`,
  `vertex`, `monitoring`, `__main__`. `distributed/cloud_coordinator` went with
  them: it imported only deleted names and could no longer be imported.
  **Authorization note:** that module is row B14 of the plan's Tier B, which is
  a CONFIRM-FIRST item, and it was removed under the Tier A authorization. The
  technical justification stands (Tier A left it unimportable) but the process
  did not: it should have been raised, not folded in. Flagged for the maintainer
  rather than left in the commit's "Tier B untouched" claim.
- **`alpha_zero/core/`**: `action_pruning`, `network_heads`, `mixins`,
  `timeout_config`, `buffer_config`.
- **`alpha_zero/utils/`**: `training_diagnostics`, `training_config_enhanced`,
  `simple_mcts` (the second `SimpleMCTS`; `core/mcts/simple.py` is the one).
- **`alpha_zero/training/`**: `curriculum_pipeline`, `training_profiler`.
- **`alpha_zero/evaluation/`**: `sequential_testing`, `reports`.
- **`alpha_zero/optimizations/`**: `benchmarks`, `config`, `cuda_graph_cache`.
- **`alpha_zero/human_interface/`**: `input_handler`, `visualizer`.
- `alpha_zero/exceptions.py`, `alpha_zero/envs/docker_config.py`,
  `alpha_zero/opponents/{base,network_opponent}.py`.
- Build files that only served the LEGACY workers: `dockerfiles/Dockerfile.{base,
  selfplay,trainer,evaluator}`, `docker-compose.yml`, the Makefile
  `build-{base,selfplay,trainer,evaluator}` / `push-*` / `local-test` /
  `local-down` targets (`build-all` and `push` now build and push
  `dockerfiles/Dockerfile.vertex`; `local-smoke` dry-runs the deployed worker),
  the root `Dockerfile.inference` duplicate, `sweep_config.yaml` (W&B-format,
  unreferenced), and `setup.py` (kept only for `Dockerfile.base`).
- Test files that exercised only the removed modules, and the removed cases in
  files that mixed them with live tests.

Package `__init__` re-exports of the removed names are gone with them: importing
e.g. `HeadFactory` from `alpha_zero.core` or `CurriculumPipeline` from
`alpha_zero.training` now raises `ImportError`. Nothing in `alpha_zero/`,
`scripts/`, or the root entry points did so.

**Deprecated, not removed** (one release window, then deletion):
`alpha_zero.core.{hex_conv,hex_encoder,encoder_registry,crystalline_layers}`
(re-export shims for `alpha_zero.hexcrystal`) and
`alpha_zero.tournament.wandb_dashboards.TournamentWandbLogger` now emit a
`DeprecationWarning` on import.

### Added — A seed now determines the whole run (2026-08-30)

The project could not replay its own runs. Its only recorded training run says
so, and quantifies it: five 20-game probes of **one checkpoint** scored 0.25 /
0.35 / 0.375 / 0.375 / 0.40 (`docs/evidence/validation_iteration_zero/`). Every
judgement downstream inherited that spread — the promotion threshold, the encoder
A/B, any per-iteration trend claim.

`--seed` on `training_civ6.py` and the evaluation entry points now reaches every
source of randomness: environments, all five search variants, the random
opponent, replay sampling, batch order and network initialisation. Offsets keep
the streams disjoint; every new parameter defaults to `None`, so unseeded callers
are unaffected.

**Four unseeded sources, and only the first was found by reading the code:**

- Eight global `np.random.*` draws across `core/mcts/` and `core/batch_mcts.py`,
  reachable from no config. `gumbel_sample_actions` now takes an `rng`, threaded
  from all three of its callers.
- **`MicroCivEnv.clone()` dropped the RNG.** It built a fresh environment with no
  seed — and cloning is how MCTS simulates, while `step` itself draws (combat,
  settler auto-found). A seeded environment therefore produced unseeded rollouts
  in *every* variant, which no seed further up could correct. Found because
  `SimpleMCTS` stayed nondeterministic after its own fix.
- **`ReplayBuffer.sample` and `DataLoader(shuffle=True)`**, which decide what a
  batch contains and therefore the gradient. Found by diffing two `--seed 42`
  runs: they agreed on all 525 lines of self-play output and differed on the
  first epoch loss.
- **`NeuralNetworkPlayer`** — the *evaluated agent* — sampling globally while its
  siblings `RandomPlayer` and `HeuristicPlayer` in the same module each took a
  seed and owned a generator. This is why the probe spread survived environments
  that were already seeded.

**Measured, as a control rather than a demonstration**
(`docs/evidence/seeded_probe_reproducibility/`): one checkpoint, ten probes.

| | Score rates | Distinct | Range |
|---|---|---:|---:|
| Seeded | 0.325 × 5 | **1** | **0.000** |
| Unseeded | 0.45, 0.40, 0.30, 0.45, 0.40 | 3 | 0.150 |

Withholding the seed still produces the original spread, so the collapse is
attributable to the seed.

Gated by `tests/regression/test_training_determinism_regression.py` (10
assertions, ~7 s), which exercises the real `train()` entry point rather than a
stand-in — the first draft used its own network and stayed green when
`torch.manual_seed` was deleted from `train()`.

Behaviour note: seeded runs draw from a `numpy.random.Generator` rather than the
legacy global functions, so action sequences differ from an unseeded run today.
Nothing depended on the previous sequences.

This does **not** address the ~73.5% draw rate that
`docs/runbooks/promotion-gate.md` identifies as the upstream problem.
Reproducibility narrows a measurement's variance; it does not move the
measurement.


### Fixed — Second review round: three defects inside this PR's own fixes (2026-08-30)

Copilot returned 8 more comments; all 8 were correct. Three of them are defects in
fixes made earlier in the same PR, which is the honest case for the round:

- **`validate_skills.py --path <existing non-SKILL.md>` exited 0.** An `if/elif`
  with no `else` recorded nothing. `validate_agents.py` had been fixed for exactly
  this; its sibling was left with the hole. The `--path` contract is now asserted
  across every validator that accepts the flag, for both the missing and the
  misnamed case, with the positive half (a valid target still validates) so a
  validator that rejected everything could not pass.
- **The secret gate read gitignored files.** `gitleaks dir .` scans the working
  directory, and this project *requires* a gitignored `.env` holding real API keys
  — so a correctly configured developer would have had `make verify` fail on a file
  that cannot be committed, while the comment beside it claimed a "tracked tree"
  scan. It now stages `git ls-files -co --exclude-standard`. Allowlisting `.env`
  would have been the wrong fix: an accidentally *tracked* `.env` is in that set and
  must still fail. Both directions verified. Side effect: the tree scan went from
  121.8 MB / 21.3 s to 16.9 MB / 1.1 s.
- **The gitleaks pin could still be defeated.** `gitleaks-install` only fetched when
  the target was missing, so a `.tools/gitleaks` left from an earlier pin was reused
  for ever — the PATH-default bug one step further along. The version is now in the
  binary's *path*, making a pin change a cache miss.

Also: the allowlist parser recognised only triple-single-quoted TOML strings, so a
catch-all written `"^.*$"` was invisible to it while the other entries kept
`assert patterns` green — and the negative control compiled the regex directly,
bypassing the parser, so it proved the predicate discriminates and nothing about the
extraction. Measured: the old parser missed **4 of 5** ways to write a catch-all.
Replaced with `tomllib` (stdlib since 3.11, so the "stdlib-only" constraint the hand
parser existed for is met without hand-parsing), and the controls now run through the
real parser in every quoting style plus the `[[rules.allowlists]]` form.

And the RNG-hygiene inconsistency: `_seed_global_rngs` seeded the process-global
numpy and torch RNGs and left them there, so later tests in the same worker inherited
state — the coupling the `small_model` fixture uses `fork_rng` to avoid, reintroduced
two functions below it. Both helpers are now restoring context managers, with tests
that the state comes back *and* that seeding still takes effect inside the block.

The CI `security` job now calls `make security-secrets` rather than restating the
install and both scan commands. Restating them is how it came to run a scan that
contradicted the comment beside it.


### Added — AQA: a test suite that nothing runs is not a gate (2026-08-30)

The review finding that `tests/regression/` was enumerated by no CI job and no
make target was fixed for that one directory. The *category* was not, and five
more suites were in exactly the same position — present, passing by hand, gating
nothing:

| Suite | What running it revealed |
|---|---|
| `tests/scripts/` | **3 real failures.** `pandas` and `mlflow` imported directly rather than through `pytest.importorskip`, so absent optional extras were errors, not skips |
| `tests/e2e/` | 70 passing tests |
| `tests/cloud/smoke/` | 36 passing tests |
| `tests/smoke/` | 30 passing tests |
| `tests/inference/` | skip-guarded, never invoked |

All five are now wired into CI and `make test`, and the three `tests/scripts/`
failures are guarded. `tests/inference/` runs paired with `tests/smoke/` rather
than in its own step: every module in it is `pytest.importorskip`-guarded on the
`web` extra, so alone it collects nothing and pytest exits **5** — a red build
for a suite behaving exactly as designed. That was caught by CI, not locally,
because the local output read `1 skipped` and looked fine; the failure lived in
the exit code, which is the only part CI reads. The meta-gate now also refuses a
CI step whose every suite is fully skip-guarded, so the next one cannot repeat
it. `tests/test_sanity.py` — an unreferenced two-line
`assert True` at the tests root — was removed rather than wired; a test that
cannot fail cannot gate anything.

**`tests/regression/test_suite_wiring_regression.py`** turns the finding into a
standing invariant: every directory under `tests/` holding collectible tests must
be named by some CI pytest invocation, or listed in `NOT_GATED` with a reason.
It also holds `make test` to the CI jobs it claims to mirror. The workflow is
parsed with `yaml`, not regex — a hand parser over YAML is exactly how the skills
validator came to validate a folded block scalar vacuously. Enumeration mirrors
pytest's own collection rule (a `test_*` function, or a `Test*` class with a
`test_*` method) rather than the filename glob, so `tests/test_constants.py` —
a constants module holding `class TestDefaults` and no tests — does not demand a
CI wiring for a suite that does not exist.

**`tests/regression/test_selfplay_determinism_regression.py`** pins the CI
failure that prompted this: identical environment seed, global RNG seed and
network weights must reproduce identical play, and the global-RNG coupling that
made `run_batch_selfplay(seed=...)` insufficient stays asserted in both
directions.

**`tests/regression/test_validator_cli_regression.py`** generalises the
`--path` hole from the one validator that had it to every validator that accepts
the flag, and adds the regression cover that the `GITLEAKS_BIN` default and the
cross-platform installer shipped without.

### Fixed — `validate_agents.py` crashed while reporting a bad `--path` (2026-08-30)

Reporting `A0` for a path outside the repository called
`Path.relative_to(REPO_ROOT)` unguarded and died with a `ValueError` traceback
instead of printing its own diagnostic. The exit code was non-zero either way,
so a check that only asserted "exits non-zero" — the obvious test to write for
the original finding — would have called it fixed. The new CLI regression tests
assert the diagnostic, not just the code.


### Fixed — The governance gates now actually gate (2026-08-30)

Review of the previous commit returned 13 findings; all were correct. Four were the
same failure — something described as a gate that gated nothing:

- **`tests/regression/` was run by no CI job and no make target.** The ten
  repository-hygiene invariants added in that commit, and the pre-existing
  `test_hex_encoder_regression.py`, had never gated a pull request. Wired into the CI
  `test` and `coverage` jobs and into `make test` / `make test-all`.
- **`validate_agents.py --path <missing>` exited 0**, so a stale path from the
  PostToolUse hook read as a clean validation. Now reports `A0`, matching
  `validate_skills.py`.
- **The catch-all allowlist test checked four exact spellings.** `^.*$` contains none
  of them, matches every path, and left the test green. Replaced with a behavioural
  check that compiles the real patterns against representative source.
- **The A5 requirement said "every importable package"; the validator checked 29 of
  44.** Nested packages — including `alpha_zero/core/mcts/`, created by this PR — had
  no `Agent.md` and the gate passed. Spec narrowed to top-level; `core/mcts/` gets a
  persona document.

Four more where a gate was weaker than claimed: the layering test walked only
`tree.body`, so a function-local `from .base import …` could reintroduce the removed
dependency invisibly; `GITLEAKS_BIN` defaulted to whatever was on `PATH`, defeating the
pin; the installer always fetched `linux_x64`, against the cross-platform requirement in
CLAUDE.md §2.1; and `.gitleaks.toml` path-allowlisted `docs/archive/`, which is 25
tracked files rather than a build artifact.

Also the tests those changes should have shipped with. The block-scalar parser fixed a
*vacuous-validation* bug with no regression test — reconstructing the old parser shows
`description: >` yielded the literal `">"`, so an **empty** folded scalar passed the
non-empty check. And `_point_at` in the governance suite patched only `SKILLS_DIR`, so
tmp-tree tests were silently scanning the repository's real `.agents/skills/`.

### Security — Unremediated W&B credential found in git history (2026-08-30)

- The new repo-wide secret audit found **a real Weights & Biases API key** (40-char
  lowercase hex, the exact W&B format) committed on 2026-03-19 in `87cd68a2` across five
  deployment scripts. The commit is reachable only from the unmerged
  `feature/sprint6-validation-and-nplayer-foundation` branch — not the default branch, not
  any current PR — and all the files were later deleted, which is why the tree scan is
  clean. **Deletion does not remove a value from git history: the key requires rotation.**
  Triage and required actions: `docs/runbooks/secret-scanning.md`.
- The CI secret gate is scoped to the built branch's own lineage (`--log-opts="HEAD"`,
  the branch's whole lineage). An unscoped scan over a `fetch-depth: 0` checkout fails on findings
  from unmerged branches that no PR can remediate — a gate nobody can turn green.
  `make security-secrets-audit` keeps the repo-wide view available on demand.

### Added — Governance gates: secret scanning, persona validation, function-size bound (2026-08-30)

- **Secret scanning (`gitleaks`)** over the tracked tree *and* history — a secret removed
  in a later commit is still published in the one that added it. The pin lives once, in the
  Makefile (`GITLEAKS_VERSION`); CI reads it via `make print-gitleaks-version` rather than
  restating it. `make security-secrets` runs the blocking pair locally;
  `make security-secrets-audit` runs the repo-wide sweep. **The "0 findings across 525
  commits" baseline this originally cited was measured before `git fetch --all`, and was
  wrong** — see the Security entry above; the repo-wide audit has an open finding. What is
  measured, 2026-08-30: 0 findings on the tracked tree, and 0 across this branch's entire
  `HEAD` lineage (532 commits), which is the scope the blocking gate covers.
- **`scripts/validate_agents.py`** — a fifth deterministic validator (A1–A6) for the 30
  `Agent.md` persona documents, which had **no gate, no hook and no test** until now. It
  immediately found real drift: `monitoring/Agent.md` had no persona section and asserted
  that Weights & Biases "remains in place" (retired 2026-07-13), `alpha_zero/hub/` had no
  `Agent.md` at all, and several docs referenced files that no longer exist. All fixed.
- **`validate_skills.py` now scans `.agents/skills/` too.** That tree was routed to the
  validator by the PostToolUse hook but scanned by neither pre-commit nor CI, so it carried
  live violations including a coverage literal 5 points below the gate. Also gains S5
  (skill directory with no `SKILL.md`) and a frontmatter parser that understands YAML block
  scalars — a folded `description: >` previously validated as the literal `">"`.
- **`tests/regression/test_repo_hygiene_regression.py`** — pins that no tracked file is
  gitignored, that `.gitignore` has no duplicates, and that the secret-scan allowlist
  cannot become a catch-all.

### Fixed — Repository hygiene (2026-08-30)

- **Five files were tracked *and* gitignored**: `docker-compose.yml` and the four
  `kb/sessions/*.state.json` role-state files, which CLAUDE.md §14.5 calls "the ONLY
  mutable state per role". Git applies ignore rules only to untracked paths, so the
  contradiction was silent — until someone added a sibling file, which would have failed
  to stage. Fixed with targeted negations.
- **`.gitignore` had 83 duplicate entries and `.dockerignore` a 110-line verbatim
  duplicate** (~48% of the file). Both deduped, each verified behaviour-identical
  beforehand — 1,963 sample paths for `.gitignore`, 714 excluded files for `.dockerignore`.
  `.dockerignore` also now excludes `.claude/`, `kb/`, `openspec/`, `.github/` and
  `uv.lock`, which were being shipped into the build context and invalidating the layer
  cache on every documentation edit.
- **A percent-matching bug in two validators**: `(\d{2,3})\s*%` matched the decimal tail
  of `89.30%` as a bare `30%`, so a correct coverage figure read as a gate violation.
- **`.claude/settings.json` permissions**, broken by routing the Makefile through
  `$(PYTHON) -m`: the `Bash(ruff:*)` / `Bash(mypy:*)` / `Bash(pytest:*)` prefixes no longer
  matched, and `make verify` — the documented pre-PR gate — was never on the allow list.


### Changed — `core/civ6_mcts.py` decomposed; a statement-count guard replaces nothing (2026-08-30)

- **`alpha_zero/core/civ6_mcts.py` (2,443 lines) split into `alpha_zero/core/mcts/`**, one
  module per search variant: `mcts_protocols` (contracts), `gumbel_ops` (Gumbel-Top-k
  primitives), `base` (`MCTSConfig`, `MCTSNode`), `puct` (`Civ6MCTS`), `simple`
  (`SimpleMCTS`), `gumbel`, `muzero`. Largest is now 876 lines. Pure relocation — the diff
  outside the moved blocks is the shim and the docs; `mcts_factory.py` is untouched.
  `civ6_mcts` remains a re-export shim with an explicit `__all__`, so every historical
  import path resolves to the *same object*.
- **The layering is now enforced, not just intended.** `gumbel` and `muzero` carry their own
  node and configuration types and do not import `base`; the Gumbel-Top-k helpers live in
  `gumbel_ops` specifically so `muzero` need not import the `gumbel` search module.
  `tests/unit/test_mcts_split_backcompat.py` pins all of it.
- **That suite also pins patchability**, which an identity test cannot: a re-export shim
  trivially satisfies `old is new` while `mock.patch` on it becomes a silent no-op if the
  patched symbol's consumer also moves. `MuZeroTrainingBridge.create_mcts` imports from the
  shim at call time, and a test asserts the patch still reaches it.
- **ruff `PLR0915` enabled** (`max-statements = 100`) — statements per function, which is the
  pathology this repo actually has; module line count is only a proxy for it. Three `main`
  functions exceed it (248 / 168 / 134 statements) and are listed in `per-file-ignores` as an
  explicit burndown list rather than being silently accommodated by a higher threshold.


### Fixed — Self-play data correctness, promotion decisions, and honest gates (2026-08-27/28)

- **Batch self-play labelled every position 0.0.** The loop counted MCTS *action*
  steps against `max_turns`, the environment's *game-turn* limit, and a game turn
  spans ~6 actions — so games stopped at roughly a sixth of their length, no
  environment reached its own truncation, `winner` stayed `None`, and the value head
  trained on a constant. Measured: a 20-action budget left the environment at game
  turn 3 of 20. Now bounded by an action budget that exists only as a runaway guard,
  with unfinished games logged rather than silently recorded as draws.
- **Batch self-play stored no `action_masks`.** `prepare_dataset` substitutes an
  all-ones mask when the key is absent, so every batch-generated sample trained with
  masked policy loss disabled. Trajectory construction is now one shared helper
  (`new_trajectory`/`append_step`/`dummy_trajectory`) used by all self-play paths and
  both dry-run stubs; `append_step` commits every per-step field or none.
- **Batch self-play could hang forever.** The outer loop advanced only on completed
  games, so a deterministic first-search failure retried an identical batch
  indefinitely — on a cloud worker, a job burning GPU hours in silence.
- **Arena rejected a variant it documents as supported.** `run_arena` passed the
  Gumbel-family `num_sampled_actions` unconditionally, so `mcts_type="civ6"` raised
  `TypeError` on the first game of every run, in the deployed evaluation path.
- **Self-play parallelism was requested and silently ignored.**
  `--num_parallel_games` defaulted to 4 against a default `gumbel`, which fails the
  batch dispatch gate; the flag was forwarded to every worker and discarded. Default
  is now 1 — what those runs actually did — with warnings on both sides.
- **Action-space constants had drifted.**
  `compute_microciv_action_space_size()` returned 356 against a real 376 and
  `validate_constants` asserted the same wrong number, so the error validated itself
  green. `pbt_orchestrator` read an env var nothing sets, defaulting to a width
  matching no checkpoint. `create_transformer_network` defaulted to the value its own
  module warns is incorrect.
- **Promotion**: the legacy evaluator now decides through the policy registry (the
  default reproduces the historical arithmetic exactly), records draw-aware shadow
  statistics, and reports the enum *value* rather than its repr. `PromotionConfig`
  validates confidence against the z-table instead of failing mid-evaluation.

### Added — Observability and honest local gates (2026-08-27/28)

- Mask coverage, batch outcome composition and unfinished-game counts are now logged;
  that silence is what hid the missing masks.
- `run_batch_selfplay` accepts a `seed`, so batch runs are reproducible and
  regressions bisectable. The sequential path already had one.
- `make verify` reproduces every CI gate — it shipped omitting coverage, bandit and
  the eval gate while claiming otherwise. New `verify` and `negative-control` skills;
  per-job `timeout-minutes` in CI; a truthful SessionStart bootstrap; and a PreToolUse
  guard whose regexes match what they claim (five push-to-main bypasses closed).
- `docs/runbooks/promotion-gate.md`: the shipped promotion defaults cannot promote at
  the measured 73.5% draw rate, and switching policy alone still rejects a +31 Elo
  model. Documents the arithmetic and the recommended configuration.

### Added — Pre-PR Validations, Infrastructure & Validation Fixes (2026-08-11)

- **Test Coverage Gates**: Enforced 85% test coverage requirement in `.github/workflows/ci.yml` and local Makefile to enforce strict regression safety.
- **Hub & Spaces Automation**: Added `make publish-spaces` into Makefile to manage Hugging Face spaces deployment.
- **Agentic Skills**: Added `type-fortification-skill` and `hub-publish-validator` skills to `SKILL.md` library to formalize knowledge transfer.
- **Architecture Documentation**: Updated C4, README, and harness documents to reflect new integrations (SpaceTimeDB, HuggingFace Spaces).

### Added — Checkpoint validation wiring, GPU A/B evidence, and 100% mypy stabilization (2026-08-10)

- **Wired `validate_encoder_state` into production GCP worker load sites**: Integration into `evaluation_worker.py` (after load_state_dict) and `selfplay_worker.py` (both Transformer and MuZero paths) ensures checkpoint mismatches are logged as warnings rather than silently degrading training/eval metrics.
- **Completed 800-game local GPU A/B arena**: Ran controlled A/B training and arena evaluation of `patch_embed` versus `hex_crystalline` on MicroCiv (8x8 grid). Results show a statistically significant edge for `hex_crystalline` (+31.4 Elo, 142W/588D/70L, 95% CI [0.510, 0.579]).
- **Published to Hugging Face**: Released the extracted hex-crystalline encoder subpackage to Hugging Face under `ianshank/civ6-hexcrystal-encoder` with safetensors and config.json.
- **100% clean mypy type safety**: Fixed 33 remaining mypy errors across the entire repository, achieving zero type-checking errors in all 350 files.

### Added — Hex-crystalline encoder is a standalone, publishable subpackage (2026-08-09)

- **`alpha_zero/hexcrystal/`** — the hex encoder, crystalline lattice layer, patch
  embedding, and encoder registry now live in a subpackage whose only third-party
  dependency is torch. It was always torch-only *by source*, but importing it ran
  the root package's eager chain into numpy and gymnasium, so it could not be
  reused outside this repository. `tests/unit/test_hexcrystal_isolation.py` pins
  the contract by importing the package in a subprocess with both blocked.
- **Backwards compatibility is total for existing callers.** Re-export shims
  remain at all five historical import paths and resolve to the same objects;
  checkpoint keys are unchanged (`encoder.patch_embed.*` for both encoder types),
  verified against a key-set snapshot captured before the move.
- **`HexEncoderConfig` gained `to_dict`/`from_dict`/`to_json`/`from_json`.**
  Unknown fields are logged and dropped so a config from a newer version still
  loads. Every field is a JSON primitive, which is what lets the encoder publish a
  flat `config.json` next to safetensors weights.

### Added — Reproducible local encoder A/B and Hub publication (2026-08-09)

- **`scripts/run_encoder_ab.py`** — trains both encoder arms from identical code,
  seed, and budget, plays them head to head, and writes committable evidence to
  `docs/evidence/hexcrystal_ab/`. It composes the existing cloud workers
  (`create_model` already accepts `encoder_type`; `run_selfplay` needs no cloud
  storage or credentials; `run_arena` accepts two independently-configured
  models) rather than adding another training path.
- **Self-play is now seedable.** `run_selfplay` gained an optional `seed` that
  threads into environment construction. Seeding the global RNGs was never
  sufficient — `MicroCivEnv` holds its own generator — so game streams varied
  between runs under a fixed seed. The default (`None`) preserves the previous
  behavior, and a test asserts the unseeded case still varies so the rationale
  cannot silently rot.
- **`validate_encoder_state`** (`alpha_zero/utils/checkpoint.py`) — several load
  paths use `strict=False`, so a checkpoint loaded into the wrong architecture
  leaves the encoder randomly initialised and still plays, which is invisible in
  aggregate metrics and fatal precisely during an encoder comparison. Validation
  is structural and names both the expected and found encoder.
- **Evidence carries a Wilson 95% interval**, and the report prints an explicit
  warning whenever that interval spans 0.5. A consistency gate refuses to render
  a report whose counts, rate, interval, and significance flag disagree.
- **`scripts/upload_encoder_to_hub.py`** — stages the encoder for the Hub as
  safetensors plus a flat `config.json`, dry-run by default. Publishing requires
  an explicit `--publish` and a `HF_TOKEN`; pickled weights and non-hex
  checkpoints are refused. `docs/model_cards/hexcrystal_encoder.md` documents the
  architecture as implemented and states the evaluation's limits.
- **`--encoder_type`** added to `scripts/local_training_run.py` and recorded in
  its logged parameters.

### Changed — Training-run comparator reads MLflow (2026-08-09)

- `scripts/compare_training_runs.py` gained an MLflow source with per-key
  fallback, which its own docstring had requested since v1.5.0: MLflow has been
  the only tracker since v1.7.0, yet the script could read only deprecated W&B
  runs or regex-parsed logs.
- Its `matplotlib` import is now guarded, with `--no_plots` degradation.
  matplotlib was imported at module scope while being declared in no dependency
  list, so the script could not run in a default environment; it is now declared
  in the `monitoring` extra.
- The hardcoded promotion threshold and W&B project default are parameterised
  (`CIV6AZ_PROMOTION_THRESHOLD`, `CIV6AZ_WANDB_PROJECT`). W&B and log-file modes
  are unchanged.

### Changed — Root package resolves its exports lazily (2026-08-09)

- `alpha_zero/__init__.py` uses PEP 562 `__getattr__` instead of eager imports.
  Measured: `import alpha_zero` 1,600 ms → 6.5 ms. `from alpha_zero import X` is
  unaffected.
- **One deliberate surface change:** `__all__` is now static and no longer lists
  `RealCiv6Env`/`RewardConfig`. Both remain resolvable attributes; only
  `from alpha_zero import *` stops pulling them in. Deciding their availability
  eagerly would have required performing the very import being deferred.

### Fixed — Hex encoder architecture documentation (2026-08-09)

The living documentation described an encoder that was never implemented.
Corrected in `CLAUDE.md` §5.2 and the affected docstrings; the historical v0.14.1
changelog entry below is left as-is as a record of what was claimed at the time.

- **Claimed** axial coordinates; **implemented** even-q offset only —
  `HexAdjacency` raises `ValueError` for any other system. The
  `HexEncoderConfig.offset_type` docstring advertised an `"odd-q"` value that
  fails at runtime.
- **Claimed** ring-distance positional encoding; **implemented** none at all —
  position reaches the model solely through the transformer's learned `pos_embed`.
- **Claimed** multi-scale convolutions and a configurable ring radius;
  **implemented** a flat same-resolution stack with a fixed 7-tap kernel. The
  registry docstring advertised a `num_scales` kwarg that was silently discarded.
- **Claimed** classes `HexConv2d`, `CrystallineAttention`, `EncoderRegistry`;
  **implemented** `HexConvolution`, `CrystallineLatticeLayer`, and a module-level
  registry dict. The crystalline layer combines transforms across the *transform*
  axis, not spatially — there is no hex-neighbour attention masking.
- Newly documented: the `AdaptiveAvgPool2d` token bottleneck discards fine
  spatial structure. This is the encoder's most consequential undocumented
  property and is now stated in the module docstring and `Agent.md`.

### Fixed — Train → evaluate → play loop (2026-07-26)

Found while checking CLAUDE.md §13 against the code. Each defect was invisible
because the test covering it asserted the wrong thing.

- **The web UI's neural opponent never used trained weights.**
  `GameSessionManager._get_shared_network` read `result.state_dict`, but
  `safe_load_checkpoint` returns `CheckpointLoadResult` whose payload field is
  `.data` — both fallback branches were skipped, no weights loaded, and it
  logged success anyway. The two covering tests patched `safe_load_checkpoint`
  to return a plain dict, a type the real function never produces.
- **Two further blockers behind that one:** `session.py` looked for
  `"model_state_dict"` while `training_civ6` writes `"state_dict"`, and the
  network was rebuilt with hardcoded 6-block/64-filter defaults, so a
  checkpoint trained at any other size could not load regardless. New
  `utils.checkpoint.load_network` is now the single reader for every shape.
- **N-player checkpoints were unloadable.** `MicroCivNetwork` swaps in an
  `NPlayerValueHead` when `num_players > 2`, but `NetworkConfig` never recorded
  `num_players`, so reconstruction built a scalar head and `load_state_dict`
  failed on `value_head.fc2.weight`. Now a defaulted config field, so older
  checkpoints still load as 2-player.
- **A GPU-trained checkpoint could not be loaded on a CPU-only machine.**
  `safe_load_checkpoint`/`load_model_state` defaulted `map_location=None`, so
  `torch.load` restored tensors to the save-time device — a direct break of
  charter NG-4. Both now default to `"cpu"`.
- **Neither player controlled its own units in the web UI.** `GameController`
  flipped `current_player` after every action, but MicroCiv rotates it only in
  `_end_turn`. Measured on a seeded game: controller and env disagreed on 3 of
  6 plies before, 0 after. The route had the mirror bug, playing a single AI
  ply per human move.
- **The actions that win the game were unreachable.** The template rendered
  `valid_actions[:20]` in index order; `found_city` (341-360) and `produce`
  (361-375) sit above `move` (1-160), so a realistic 42-action mid-game state
  hid both behind a dead "+N more actions" label. Actions are now grouped by
  category with a per-group cap.
- **MCTS flipped the value sign on every ply** (`core/civ6_mcts.py`) while a
  MicroCiv player takes many plies per turn, corrupting visit counts (the
  policy target) and the search. Backup and PUCT selection are now `to_play`
  aware, with `to_play` recorded *after* the simulation step. Value targets on
  the training path were unaffected — they come from the game outcome.
  `>2`-player behaviour is unchanged by design; `NPlayerMCTS` remains correct
  there.

### Added
- `alpha_zero/utils/device.py` — `resolve_device` / `resolve_device_str` /
  `place_network`, the single source of truth replacing ~30 ad-hoc
  `torch.cuda.is_available()` sites across three incompatible idioms. Configs
  now *resolve* a device rather than warning and carrying on with a stale one.
- `--max_turns` and `--save_interval` on `training_civ6` — both already existed
  as config fields with no way to set them, which made a short run impossible
  to evaluate (`save_interval=10` means a 6-iteration run prints no win rate).
- Cross-device checkpoint round-trip test and a repo-wide guard against
  `.numpy()` without a preceding `.cpu()`.

### Changed
- CLAUDE.md §13 rewritten against the code: the N-player item was already done
  (2026-07-05) and the "12 xfailed DDP tests in `tests/unit/test_distributed`"
  item referenced a directory that does not exist. §13 had been contradicting
  §5.3 in the same file.

### Removed — Dead-code deletions, hygiene sweep (2026-07-25)
- `alpha_zero/cloud/gcp/cost_optimization.py` (644 ln) and
  `alpha_zero/cloud/gcp/lifecycle_cli.py` (570 ln) — zero importers; the former
  overlapped `gcp/cost.py`/`gcp/cost_monitor.py`.
- `alpha_zero/inference/api_server.py` (326 ln) — unused duplicate of
  `alpha_zero/serving/inference_server.py`; docs re-pointed at the real server.
- `run_debug_selfplay.py` (1,432 ln) — untested, unreferenced root runner.
- `scripts/analyze_training_run.py` — hardcoded-path throwaway that shadowed the
  maintained `scripts/analyse_training_run.py` by near-identical name.
- `scripts/validate_v130_training.py` — v1.3.0-era shim already superseded by
  `scripts/validate_convergence.py`.
- Two long-skipped `--model_version` test classes in
  `tests/cloud/unit/test_training_worker.py` (6 tests) whose own docstrings said
  "can be deleted in v1.7.0" (project is at v1.8.0); superseded by
  `tests/integration/test_training_resume_continuity.py`.
- `inspect_db.py` moved to `scripts/inspect_db.py` (kept; was misplaced at root).

### Changed — Gate hardening (`harden-lint-and-type-gates`, 2026-07-25)
- **Tool pins aligned at ruff 0.15.8 / mypy 1.19.1** across all five locations
  (CI lint job, CI governance job, pre-commit, pyproject `dev` extra,
  requirements.txt — the last two previously drifted/unpinned).
- **Lint gate is whole-repository**: CI runs `ruff check .` / `ruff format
  --check .`; 104 test files + 25 script/root/infra files reformatted
  (format-only); unmaintained fringe dirs excluded centrally via
  `[tool.ruff] extend-exclude` + `force-exclude`; pre-commit `&maintained`
  anchor widened to match.
- **Mypy is now a real gate**: the CI lint job installs the project + dev extra
  (torch, stubs incl. new `types-requests`), the `--ignore-missing-imports`
  CLI flag is gone, and `mypy alpha_zero/` exits 0 — 130 errors burned down
  (29 stale ignores removed, 101 semantic fixes via annotations/casts/scalar
  wrappers; 5 documented library-false-positive ignores remain).
- **eval-harness CI job flipped to enforcing** — its observation window closed
  with consecutive green runs (carried from add-enterprise-docs-and-gates
  task 6.3).
- **Never-in-CI test dirs measured** (`tests/e2e|inference|regression|scripts|
  smoke`): 263 passed / 12 env-dependent failures / 2m03s; CI wiring deferred
  until those suites carry env-required markers. `tests/inference` gained a
  pydantic `importorskip` guard so collection degrades to a skip without the
  `web` extra.

### Changed — Code hygiene sweep (2026-07-25)
- **Config truth**: dropped the orphaned per-file-ignore for deleted
  `tests/unit/test_pbt_dashboarding.py` and the stale coverage-omit glob
  `alpha_zero/orchestrator/reflection_*.py` (matched nothing) from both omit
  lists; added the missing `mlflow<3.0` upper bound to the `monitoring` extra
  (its comment always promised 2.x); corrected the `UP038` ignore comment (the
  rule is still active at the pinned ruff 0.8.0 — 27 violations — so the ignore
  stays until the pin upgrade).
- **Doc truth**: corrected CLAUDE.md §5.3's static-skip figure (~20 W&B-related →
  measured 11 skip sites / 17 tests, none W&B-related).

### Added — Enterprise documentation overhaul & governance gates (2026-07-25)
- **OpenSpec adoption**: `openspec/` spec-driven change directory (project.md,
  AGENTS.md, baseline specs for `agent-harness` / `kb-workflow`, and the
  `add-enterprise-docs-and-gates` change with capability deltas).
- **Deterministic validators** (stdlib-only, 0/1/2 exit contract, tested in
  `tests/governance/`): `validate_openspec.py`, `validate_kb.py`,
  `validate_skills.py`, `check_doc_sync.py`. Wired into Claude Code hooks,
  `.pre-commit-config.yaml`, and a new CI `governance` job.
- **Harness fixes**: deny `.env` edits and force-pushes; hook scripts replace the
  fragile inline commands; the PostToolUse autofix now defers F401 **and** RUF100
  so intentional `# noqa: F401` directives are preserved.
- **Skills**: handoff skills gained YAML frontmatter and a validation gate; new
  `openspec-propose`, `openspec-archive`, `doc-sync`, and `release` skills.
- **Docs**: README rewritten to match the application (MLflow, Optuna, real
  extras, `python -m` entry points, 85% gate); MIT `LICENSE`, `CONTRIBUTING.md`,
  `SECURITY.md`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`, `CODEOWNERS`, PR/issue
  templates, and `dependabot.yml` added; `docs/` reorganized with an index,
  numbered ADRs, a PRD status index, and ~19 stale artifacts moved to
  `docs/archive/`. Version aligned to 1.8.0 across packaging and the package.

### Changed — Experiment tracking is MLflow-only (W&B retirement sweep, 2026-07-13)
- **Toggle unification**: `resolve_tracker_enabled()` in `alpha_zero/monitoring/config.py` is the single enablement source with precedence `CIV6AZ_MLFLOW_ENABLED` > `MLFLOW_ENABLED` > legacy `WANDB_ENABLED` (deprecated, warns when deciding). All `from_env()` sites — cloud workers, experiment tracker — honor legacy deployment configs transparently.
- **Config bridge**: orchestrator `WandbConfig` is now an explicit deprecated bridge — `to_mlflow_config()` maps `project→experiment_name`, list tags → dict tags, `entity` → `wandb.entity` tag, `mode=="disabled"` → off. The orchestrator runs a single tracker (mlflow section wins; wandb section bridges with a DeprecationWarning).
- **BREAKING/behavior**: `--no-wandb` now disables ALL tracking (both `wandb:` and `mlflow:` config sections); new explicit `--no-tracking` alias added. Previously MLflow kept logging when only `--no-wandb` was passed.
- Deploy configs (`configs/vertex_*_smoke.yaml`, `cloud_training_v1.yaml`) switched from `WANDB_*` env blocks to `MLFLOW_*`.

### Fixed — W&B retirement sweep (2026-07-13)
- **`run_enhanced_training.py --wandb` / `run_muzero_training.py --wandb` crashed with `TypeError`** — they constructed the legacy config shape against the MLflow-aliased class; migrated to `MLflowConfig` + `WandbCompatAdapter` directly.
- **`WandbCompatAdapter.log()` silently dropped numpy metrics** (`np.int64`, `np.float32`, 0-d arrays); now coerced via `_coerce_metric_value`, with one level of nested-dict flattening (`parent/child`).
- **MLflow run-group propagation never worked** — `vertex_orchestrator` read `.name`/`.id` off an MLflow run object (metadata lives under `.info.*`); new `WandbCompatAdapter.run_name` property fixes `MLFLOW_RUN_GROUP`/`group=` tag propagation to worker containers.
- **`utils/factories.create_wandb_logger` mangled list tags** (`dict(list[str])`); now a deprecated wrapper delegating through `to_mlflow_config()`.
- `WandbCompatAdapter.alert()` also emits `logger.warning` so alerts are visible when MLflow is disabled; `watch_model()` records a `watch_model_requested` tag.

### Removed — W&B compatibility surfaces (BREAKING for out-of-repo importers, 2026-07-13)
- `alpha_zero/utils/wandb_logger.py` deprecation shim (removal was promised for v1.6.0). Last compat surface: lazy `alpha_zero.utils.{WandbLogger,get_wandb_logger,init_wandb}` aliases (emit DeprecationWarning).
- `vertex_jobs._wandb_env_vars` / `resolve_wandb_key_from_secret_manager` / `ensure_wandb_api_key` and the `CIV6AZ_WANDB_SECRET_*` env pathway (zero consumers; nothing reads `WANDB_API_KEY` anymore).
- Dead module-skipped W&B test suites (~2,300 lines): `test_wandb_logger`, `test_wandb_baseline_metrics`, `test_pbt_dashboarding`, `test_wandb_integration`, `test_orchestrator_wandb`, plus skipped classes in `test_wandb_coordination` (renamed `test_vertex_coordination`, 9 active tests kept) and `test_orchestrator_muzero_config`. Replacement coverage landed FIRST: numpy/`log_reflection_stats`/`run_name`/alert tests in `test_mlflow_logger_adapter`, `test_tracker_enabled_resolution` (precedence matrix), `test_pbt_mlflow_dashboards`, vertex-orchestrator lifecycle tests in `test_orchestrator_mlflow`, and `test_orchestrator_wandb_bridge`.

### Added — Optuna hyperparameter sweeps, replacing W&B sweeps (2026-07-13)
- `alpha_zero/optimizations/hyperparameter_search.py`: reusable `HyperparameterSearch` component — declarative YAML search spaces, `SweepConfig` with `CIV6AZ_SWEEP_*` env overrides (legacy `SWEEP_TIMEOUT_SECONDS` honored), injectable objective/tracker for testing, MLflow per-trial logging, `SubprocessTrainingObjective` with timeout + prune-on-failure semantics.
- `run_sweep.py` rewritten as a thin CLI over the component (`--config/--n-trials/--study-name/--storage/--dry-run`); `configs/sweep_muzero.yaml` seeds the search space from the historical W&B sweep results. New `tuning` extra (`pip install -e '.[tuning]'`).
- `run_muzero_training.py` prints a machine-readable `SWEEP_METRICS_JSON=` result line for subprocess objectives.
- `scripts/setup_secrets.sh` generalized: provisions any named secret (`CIV6AZ_SECRET_NAME`/`CIV6AZ_SECRET_VALUE`), wandb-api-key default removed.
- With this, no wandb imports remain in `alpha_zero/` or entry points. Two read-only analysis scripts (`compare_training_runs.py`, `search_efficiency_report.py`) keep lazy `wandb.Api()` readers for W&B-hosted *historical* runs — data MLflow cannot serve — and are documented as deprecated.

### Accepted debt — W&B naming (2026-07-13)
- `TournamentWandbLogger` and serialized `wandb_enabled` config fields keep their names for compat.
