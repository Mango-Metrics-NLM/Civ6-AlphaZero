# Changelog

All notable changes to the Civ6 AlphaZero project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] — 2026-05-21

## [1.5.0] — 2026-03-05 — Sprint 5: Type Safety — Breadth-First Mypy Stabilization

### Changed — Sprint 5

#### E1: Breadth-First Mypy Type Safety (58 files, ~153 errors fixed)

- **`hierarchical/`** package (32→0): `config.py`, `hierarchical_network.py`, `goal_types.py`, `hierarchical_mcts.py`, `level_controller.py`, `goal_encoder.py`, `training_integration.py` — `no-any-return`, `no-redef`, `assignment`, `arg-type`, `return-value` ignores for torch/numpy return types, union narrowing, and dict arithmetic
- **`cloud/gcp/training_worker.py`** (20→0): Added `TYPE_CHECKING` imports for `optim`, `Profiler`, `diagnostics`, `report` types; replaced `object | None` signatures with proper types
- **`training_civ6.py`** (8→0): `arg-type` ignores for protocol-conformant MCTS/env calls, `int()` cast for numpy `int64` action_space_size
- **`training/muzero_trainer.py`** (7→0): `operator` ignores for dynamic `self.network.*` method calls
- **`pbt/`** (16→0): `mutations.py`, `population.py`, `scheduler.py` — numpy/dict arithmetic returns
- **`utils/`** (41→~3): `rag_client.py`, `wandb_logger.py`, `llm_client.py`, `checkpoint.py`, `environment_tuning.py`, `logging_config.py`, `simple_mcts.py`
- **`cloud/`** stale-ignore sweep: Removed 15 unused `type: ignore` comments across 12 files (`vertex_orchestrator.py`, `vertex.py`, `gcs_backend.py`, `monitoring.py`, `gcs_storage.py`, `selfplay_worker.py`, `storage.py`, `vertex_trainer.py`, `vertex_jobs.py`)
- **`core/`** (23→~4): `civ6_mcts.py`, `civ6_network.py`, `config.py`, `mixins.py`, `network_heads.py`, `reflection_generator.py`, `reflection_memory.py`, `transformer_network.py`
- **`distributed/`** (28→~8): `actor.py`, `coordinator.py`, `ddp_learner.py`, `learner.py`, `parameter_server.py`, `shared_storage.py`
- **`optimizations/`** (9→~3): `auxiliary_targets.py`, `benchmarks.py`, `playout_cap.py`, `policy_pruning.py`
- **Additional files**: `rag/config/settings.py`, `envs/microciv_enhanced.py`, `envs/vectorized_env.py`, `gameplay/fog_of_war.py`, `inference/client.py`, `interfaces/unciv_interface.py`, `training/diagnostics.py`, `training/imitation_pretraining.py`, `training/training_safeguards.py`

#### Ruff Lint Auto-Fix

- Auto-fixed 15 unused import errors (`F401`) across test and source files

### Stats

- 58 files changed, +230 / -187 lines
- ~153 mypy errors fixed (251→~98, **61% reduction**)
- 7 commits, 0 test regressions

## [1.4.0] — 2026-03-05 — Sprint 4: Type Safety Expansion

### Added

- **Mypy strict overrides** for `alpha_zero.evaluation.*` and `alpha_zero.envs.*` packages (`pyproject.toml`)
- Explicit type annotations for `run_debug_selfplay.py` validators and config

### Changed — Sprint 4

#### E5: Type Safety Expansion (14 files)

- **`evaluation/config.py`**: `Callable` type hint for `env_mappings`
- **`evaluation/baselines.py`**: `dtype=np.float32` for noise addition, assert-narrowed MCTS, `hasattr` guard for `reset()`
- **`evaluation/baseline_agents.py`**: `int()` cast for MCTS `select_action` return
- **`evaluation/evaluator.py`**: Forward type declaration for `DiversityTracker | None`
- **`evaluation/reports.py`**: Explicit `dict[str, Any]` type annotation for comparison dict
- **`evaluation/evaluation_pipeline.py`**: Widened `_run_evaluation` opponent type, wraps MCTS opponents with `MCTSAgentWrapper`
- **`evaluation/muzero_vs_alphazero_benchmark.py`**: Removed 6 unused `type: ignore` comments
- **`envs/microciv.py`**: `int()` casts for numpy `integers()`, `type: ignore[misc]` for ClassVar
- **`envs/microciv_enhanced.py`**: `type: ignore[return-value]` for render
- **`envs/civrealm_wrapper.py`**: `type: ignore` for numpy overload, ClassVar, render override
- **`envs/docker_config.py`**: `float()` cast for `min()` return
- **`envs/env_factory.py`**: Variable rename to avoid assignment type mismatch
- **`envs/real_civ6_env.py`**: ClassVar, union-attr, override, `float()` cast
- **`envs/vectorized_env.py`**: `list()`/`int()` casts for no-any-return

#### Debug Self-Play Type Fixes

- **`run_debug_selfplay.py`**: Explicit `list[str]` annotations, renamed shadowed `expected_shape` variables, `assert` guards for Optional validators, `Callable` import

### Stats

- 15 files changed, +46 / -35 lines
- 0 mypy errors in `alpha_zero.envs.*` and `alpha_zero.evaluation.*`

## [1.3.0] — 2026-03-04 — Sprint 3: Convergence, Batch, N-Player

### Added — Sprint 3

#### Convergence Validation (Epic 2)

- **`scripts/validate_v130_training.py`**: Local validation script for v1.3.0 training fix
  - Checkpoint round-trip weight preservation
  - Loss decrease trend across training iterations
  - Win-rate improvement vs random player baseline
- **`tests/integration/test_convergence_validation.py`**: 6 integration tests

#### Batch Self-Play Cloud Validation (Epic 3)

- **`tests/integration/test_batch_selfplay_integration.py`**: 12 tests — vectorized env lifecycle, batch MCTS config, trajectory validation, schema consistency
- **`tests/integration/test_selfplay_worker_batch.py`**: 17 tests — batch mode detection, orchestrator passthrough, temperature scheduling, CLI parsing
- **`docs/prd/prd-batch-selfplay.md`**: Product requirements document

#### N-Player MicroCiv Foundation (Epic 5)

- **`num_players` parameter**: `MicroCivEnv(num_players=2..6)` — backwards-compatible, default=2
- **Angular spawn distribution**: `_compute_start_positions()` places N players evenly
- **Modular turn cycling**: `(current_player + 1) % num_players`
- **N-player victory**: Last-player-standing elimination
- **N-player truncation**: Most-cities tiebreak (None for tie)
- **`tests/integration/test_nplayer_microciv.py`**: 51 tests
- **`docs/prd/prd-nplayer-microciv.md`**: Product requirements document

#### DDP Test Stabilization (Epic 4)

- **`tests/integration/test_ddp_initialization.py`**: Fixed 17 attribute mismatches, removed 5 xfails → 22/22 pass

### Changed

- **CI/CD Hardening (Epic 1)**: Pinned `ruff==0.8.0`, `mypy==1.13.0`
- **`PLANNING.md`**: Sprint 3 epics and milestones
- Fixed lint errors in `inference_engine.py` and `inference_server.py`

### Stats

- 86 new tests, 22 fixed — 13 files changed, +1,935 / -149 lines

---

### Added

#### New Scripts (`scripts/`)

- **`scripts/analyse_training_run.py`** — Analyses policy loss trend and value head stability by reading MLflow metrics; produces per-run diagnostic summaries and trend plots
- **`scripts/post_run_evaluation.py`** — Post-run ELO tournament + baseline evaluation (random / heuristic agents) with results logged to MLflow; wraps the model in `Civ6MCTS` and calls `evaluate_vs_random(agent, n_games=n)` correctly
- **`scripts/run_baseline_eval.py`** — Standalone baseline evaluation that loads a model from GCS, auto-detects the latest run via `resolve_run_id()`, and logs results to MLflow
- **`scripts/migrate_mlflow_to_sqlite.py`** — Automates migration from the deprecated file-store (`mlruns/`) to the SQLite backend (`mlflow.db`); supports `--dry-run` mode
- **`scripts/local_training_run.py`** — Orchestrated local training run entry point; single command to run selfplay → training → evaluation locally with full MLflow instrumentation

#### MLflow Monitoring Package (`alpha_zero/monitoring/`)

- **`alpha_zero/monitoring/__init__.py`** — Public API: exports `MLflowLogger`, `MLflowMonitoringConfig`, `AlphaZeroMLflowEvaluator`
- **`alpha_zero/monitoring/config.py`** — `MLflowMonitoringConfig` dataclass: `tracking_uri`, `experiment_name`, `use_sqlite`, `tags`; fully serialisable with `to_dict()` / `from_dict()`
- **`alpha_zero/monitoring/mlflow_logger.py`** — `MLflowLogger` class:
  - Run lifecycle management (`start_run()` / `end_run()`) with correct FAILED vs FINISHED status on exit
  - `log_params()`, `log_metrics()`, `log_metric()`, `set_tag()`, `log_artifact()`
  - **`log_evaluation_result()`** — logs 14 metrics from `EvaluationResult` / `WinRateResult`: 4 counts (`wins`, `losses`, `draws`, `total_games`) + 3 rates (`win_rate`, `draw_rate`, `loss_rate` as 0–1 fractions) + 7 game-level stats
  - Param values truncated at 250 characters (MLflow limit)
- **`alpha_zero/monitoring/mlflow_evaluator.py`** — `AlphaZeroMLflowEvaluator`:
  - `compute_policy_entropy(logits)` — Shannon entropy of policy distribution
  - `compute_value_mae(predictions, targets)` — Mean Absolute Error of value head
  - `compute_policy_top1_accuracy(logits, targets)` — Top-1 accuracy of greedy policy
  - `compute_action_diversity(action_sequences)` — Unique-action ratio across a game batch
  - `compute_metrics(model, dataloader)` — Aggregates all four metrics over a dataset

#### New Utility (`alpha_zero/utils/`)

- **`alpha_zero/utils/env_loader.py`** — Zero-dependency `.env` loading (no `python-dotenv` required):
  - `load_civ6az_env(path)` — Loads `.env` file and merges into `os.environ` without overwriting existing vars
  - `require_civ6az_env(key)` — Retrieves an env var or raises `RuntimeError` with a descriptive message
  - `_parse_dotenv(text)` — Pure-Python parser handling quoted values, inline comments, and `export` prefixes

#### New Documentation (`docs/`)

- **`docs/GCP_INFRA.md`** — GCP infrastructure reference (VM sizing, Artifact Registry, IAP tunnelling, cost breakdown)
- **`docs/architecture/ADR-gcp-resource-deployment.md`** — Architecture Decision Record: rationale for Vertex AI + GCS resource deployment strategy
- **`docs/prd/prd-gcp-resource-deployment.md`** — Product Requirements Document for GCP resource deployment feature

---

### Fixed

#### `scripts/post_run_evaluation.py` — EvaluationPipeline API

- **Wrong kwargs**: was calling `evaluate_vs_random(model=model, num_games=n, device=d)` — incorrect API
- **Fix**: wraps model in `Civ6MCTS` and calls `evaluate_vs_random(agent, n_games=n)`
- **Win rate type**: `EvaluationResult.win_rate` is a `WinRateResult` object, not a scalar `float`; all access sites updated

#### `scripts/run_baseline_eval.py` — Hardcoded run_id & result access

- **Hardcoded run_id**: removed hardcoded production `run_id` default
- **Auto-detection**: added `resolve_run_id()` to locate the most recent MLflow run automatically
- **result.summary**: was treated as `dict`; it returns a `str` — fixed to use string formatting
- **result.win_rate**: was accessed as `float`; it is a `WinRateResult` object — fixed across all callsites

#### `alpha_zero/monitoring/mlflow_logger.py` — Lifecycle & param safety

- **`log_evaluation_result()`**: added missing method (14 metrics, 0–1 rate normalisation)
- **Lifecycle state machine**: `end_run()` now correctly sets status to `FAILED` on exception and `FINISHED` on clean exit
- **Param truncation**: added 250-character truncation for MLflow param values (API limit)

---

### Changed

- **Version**: 1.3.0 → 1.4.0
- **MLflow tracking**: migrated from file-store (`mlruns/`) to SQLite backend (`mlflow.db`); `migrate_mlflow_to_sqlite.py` automates the one-time migration
- **Lint** (`ruff`): fixed `E501` line-too-long in `post_run_evaluation.py`; fixed `RUF002` en-dash in `run_baseline_eval.py` docstrings

#### Test Suite (81 new tests)

| File | Tests | Covers |
|------|------:|--------|
| `tests/scripts/test_run_baseline_eval.py` | 39 | `run_baseline_eval.py` — arg parsing, `resolve_run_id`, GCS load, MLflow log, error paths |
| `tests/scripts/test_analyse_training_run.py` | 9 | `analyse_training_run.py` — metric fetch, trend analysis, edge cases |
| `tests/scripts/test_post_run_evaluation.py` | 11 | `post_run_evaluation.py` — pipeline wiring, ELO logging, random-agent baseline |
| `tests/unit/test_monitoring/test_mlflow_logger.py` | 24 | `MLflowLogger` — lifecycle, `log_params`, `log_metrics`, `log_evaluation_result`, truncation, failure state |
| `tests/unit/test_monitoring/test_mlflow_evaluator.py` | 12 | `AlphaZeroMLflowEvaluator` — entropy, MAE, top-1 accuracy, action diversity, `compute_metrics` |
| `tests/unit/test_utils/test_env_loader.py` | 18 | `env_loader` — `.env` parsing, quoting, comments, `export` prefix, `require_civ6az_env` errors |

**Total tests: 217 passing** (136 pre-existing + 81 new)

---

## [1.3.0] — 2026-02-23

### Fixed — Training Worker Loads Base Model for Fine-Tuning

#### Critical Bug Fix: Training From Scratch

The training worker (`training_worker.py`) was creating a **fresh randomly-initialized model** each iteration instead of loading the current best model from GCS and fine-tuning it. This made it impossible for newly trained models to beat the incumbent champion in evaluation.

- **Impact**: Production run v1.2.0 completed 50 iterations with **zero model promotions** — v218 remained champion for all 50 iterations (~$28.55 wasted)
- **Root cause**: `training_worker.py` called `create_model()` without loading existing weights; no `--model_version` arg existed
- **Fix**: Added `--model_version` CLI argument to `training_worker.py` that loads pre-trained weights from GCS before training begins

#### Changes

- **`training_worker.py`**: Added `--model_version` CLI arg; when set, loads `state_dict` from GCS via `GCSModelStorage.load_model()` before DDP wrapping
- **`vertex_orchestrator.py`**: `build_training_extra_args()` now accepts `model_version` kwarg and emits `--model_version=N`; `run_training_phase()` passes `model_version` through to training job; main loop passes current `model_version` to training phase
- **New tests**: 5 tests across `test_training_worker.py` and `test_orchestrator_pipeline.py` verifying arg parsing, weight loading flow, and orchestrator plumbing

#### Verification

After this fix, training iterations will:
1. Start with low loss (pre-trained weights, not ~6.0 random init)
2. Progressively improve policy quality across iterations
3. Produce models that can beat the incumbent in evaluation (>55% win rate)

---

## [1.2.0] — 2026-02-21

### Added — Phase 17: Training Readiness & Production Launch

#### Action Mask in Training Pipeline

- **5-tensor TensorDataset**: Selfplay workers now store `trajectory["action_masks"]` alongside spatial/vector/policies/values
- **Masked cross-entropy loss**: Training worker applies `target *= mask; target /= target.sum(-1, keepdim=True).clamp(1e-8)` so the network learns only over legal actions
- **Smoke-tested**: Phase 17 container validates action_mask flow from selfplay → trajectory → training_worker

#### W&B Group Coordination

- **`WANDB_GROUP` / `WANDB_RUN_NAME`** env vars added to `_wandb_env_vars()` in `vertex_jobs.py`
- **Orchestrator group propagation**: `vertex_orchestrator.py` sets `os.environ["WANDB_GROUP"] = run_name` after W&B init; each phase sets `WANDB_RUN_NAME=f"{phase}-iter{N:03d}"` with `finally` cleanup
- **Eval worker W&B**: Full `WandbLogger` init/log/finish added to `evaluation_worker.py`; logs `eval/win_rate`, `eval/draw_rate`, `eval/games_per_minute` at `step=iteration`

#### Evaluation Retry & Circuit Breaker

- **`run_evaluation_phase()` max_retries=2**: Evaluation now retries with `compute_retry_delay()` exponential backoff before counting toward `consecutive_failures`
- **Circuit breaker**: Eval failures after max_retries now increment `consecutive_failures`; `max_consecutive_failures=3` halts the entire run
- **`_EVAL_MACHINE_TYPE`**: Constant replaces hard-coded `"n1-standard-4"` string in evaluation job creation

#### Learning Rate Forwarding

- **`--learning_rate` arg**: Added to `vertex_orchestrator.py` CLI; `build_training_extra_args()` always emits `--learning_rate=N` to training workers
- **Backwards compatibility**: Uses `getattr(args, 'field', default)` pattern for all new args

#### MockStorageBackend Centralization

- **`tests/mocks/cloud_mocks.py`**: `MockStorageBackend` and `MockGCSStorage` moved from 3 test files to centralized mocks module
- Reduces test code duplication and ensures consistent mock behavior across all cloud unit tests

#### GCS Trajectory Cleanup

- **`scripts/cleanup_trajectories.py`**: CLI tool to list/dry-run/delete old trajectory batches from GCS
- **`GCSTrajectoryStorage.list_iteration_stats()`** and **`delete_iterations()`**: New storage methods for programmatic trajectory management

#### Container v1.2.0 (Cloud Build c72f98e9)

- **Artifact Registry**: `us-central1-docker.pkg.dev/hrmtrain/civ6az/civ6az-training:v1.2.0`
- **`cloudbuild.yaml` substitutions**: `_TAG_NAME` / `_IMAGE` substitution block; use `--substitutions=_TAG_NAME=vX.Y.Z` to override at submit time; all hard-coded version strings replaced with `${_IMAGE}:${_TAG_NAME}`
- **`dockerfiles/Dockerfile.vertex`**: Changed `requirements-runtime.txt` → `requirements-container.txt` (runtime file was deleted in cleanup)
- **All 8 Cloud Build steps green**: lint → build → smoke-test → test-selfplay → test-training → test-muzero-selfplay → test-orchestrator-guard → test-evaluation → push

### Changed

- **`DEFAULT_MCTS_TYPE`**: `"civ6"` → `"gumbel"` in `config.py`; GumbelMCTS is now the default for all new runs
- **`torch>=2.0.0`**: Updated from `>=1.13.0` in `requirements-container.txt`
- **`pytest-timeout=120`**: Added to `pyproject.toml` addopts + dev deps; prevents indefinite hangs on Windows
- **`conftest` fixture fix**: `np.ones(100)` → `np.ones(376)` in `sample_trajectory_data` (action_space_size alignment)
- **Version**: `0.14.1` → `1.2.0` (major bump for production readiness milestone)
- **Stale tests fixed**: All `test_default_mcts_type_is_civ6` → `_is_gumbel` across 3 test files
- **`alpha_zero.utils.compat.UTC`**: Python 3.10 compatibility shim already in place; container runs Python 3.10.12

### Removed

- **`TrajectoryDataset`**: Dead class removed from `training_worker.py`
- **`tests/stress/`**: Empty stress test directory deleted
- **`tests/cloud/integration/`**: Empty cloud integration test directory deleted
- **`scripts/validate_phase1.py`**: Obsolete phase-specific validation script deleted
- **`requirements-runtime.txt`**: Merged into `requirements-container.txt`; deleted
- **`--encoder_type` arg**: Dead arg removed from `vertex_orchestrator.py` (encoder is now saved in model_config)

### Production Run v1.2.0 (2026-02-21 — ACTIVE)

| Parameter | Value |
|-----------|-------|
| Container | `civ6az-prod-v120` on `civ6az-orchestrator` VM |
| Iterations | 50 |
| Workers | 4 × 50 games = 200 games/iter |
| MCTS | GumbelMCTS 16 sims, max_turns=30 |
| LR | warmup_cosine, 0.001→0.0001, value_lr_mult=0.1 |
| Start model | v218 (auto-detected from GCS) |
| W&B run | `dashing-firebrand-1393` |
| W&B URL | <https://wandb.ai/ianshank-none/civ6-alphazero/runs/ukxjr0uz> |

---

## [0.14.1] — 2026-02-18

### Added — Hex Crystalline Encoder & A/B Training Comparison

#### Crystalline Hex Spatial Encoder (`alpha_zero/core/hex_encoder.py`) — NEW

- **HexSpatialEncoder**: Hex-topology-aware spatial encoder as drop-in replacement for `PatchEmbed`
  - Axial coordinate embedding for native hexagonal grid representation
  - Crystalline attention layers with hex-neighbor masking (6-neighbor connectivity)
  - Multi-scale hex convolutions capturing local and regional spatial patterns
  - Ring-distance positional encoding (vs. Cartesian grid in PatchEmbed)
- **HexConv2d** (`hex_conv.py`): Hex-neighbor convolution with configurable ring radius
- **CrystallineAttention** (`crystalline_layers.py`): Attention restricted to hex-topology neighborhoods
- **EncoderRegistry** (`encoder_registry.py`): Factory for creating encoders by name (`patch_embed`, `hex_crystalline`)
- Fully integrated into `Civ6TransformerNetwork` via `TransformerConfig.encoder_type` field

#### Cloud Pipeline Encoder Propagation (5 files modified)

- **`vertex_orchestrator.py`**: Added `--encoder_type` and `--agent_id` CLI args with propagation to all worker types
- **`training_worker.py`**: `--encoder_type` arg passed to `create_model()` → `TransformerConfig`
- **`selfplay_worker.py`**: `--encoder_type` arg for fallback model loading when `model_config_dict` is absent
- **`evaluation_worker.py`**: `--encoder_type` arg for consistency (models auto-load via saved config)
- **`vertex_jobs.py`**: `CIV6AZ_ENCODER_TYPE` env var passed to all Vertex AI containers

#### A/B Comparison Infrastructure

- **`scripts/compare_training_runs.py`**: W&B-based comparison script pulling metrics for two runs
  - Loss curves (policy/value/total), win rate, ELO comparison, summary table
  - Generates markdown report and PNG plots via matplotlib
- **`scripts/launch_ab_run.py`**: Local A/B run launcher with `.env` loading and dry-run mode
- **`scripts/vm_ssh.py`**: SSH helper for orchestrator VM via IAP tunnel
- **`scripts/vm_deploy_ab.sh`**: VM deployment script for Run A (patch_embed baseline)
- **`scripts/vm_launch_run_b.sh`**: Auto-launcher for Run B after Run A completes (hex_crystalline)

#### Python 3.10 Compatibility Shim

- **`alpha_zero/utils/compat.py`**: Version-aware `UTC` re-export for Python 3.10 containers
  - `datetime.UTC` was added in Python 3.11; container base image uses 3.10
  - 40 files updated to use `from alpha_zero.utils.compat import UTC`

#### Gameplay Systems (MicroCiv Enhancements)

- **Terrain system** (`envs/micro_civ_terrain.py`): 8 terrain types with hex-grid movement costs, visibility, and defense bonuses
- **Fog of war** (`envs/micro_civ_fog.py`): Per-player visibility with sight-range mechanics and exploration state
- **Tech tree** (`envs/micro_civ_tech.py`): 15-tech tree with prerequisites, era progression, and unlock effects
- **Combat system** (`envs/micro_civ_combat.py`): Terrain-aware combat with flanking, fortification, and ranged attacks

#### Test Suite

- `tests/cloud/unit/test_encoder_pipeline_plumbing.py` — 22 tests for encoder_type CLI propagation
- `tests/unit/test_hex_encoder.py` — Hex encoder unit tests (spatial encoding, attention, registry)
- `tests/unit/test_hex_conv.py` — Hex convolution tests
- `tests/unit/test_crystalline_layers.py` — Crystalline attention tests
- `tests/unit/test_terrain_coverage.py` — Terrain system coverage tests
- 211 hex encoder tests + 792 cloud tests passing

### Changed

- Version: `0.13.0` → `0.14.1`
- Docker image: v0.13.0 → v0.14.1 (`gcr.io/hrmtrain/civ6az-training:v0.14.1`)
- Container base: Python 3.10 compatibility via `alpha_zero.utils.compat.UTC` shim (40 files)
- `.gitignore`: Added A/B comparison artifacts, inference benchmarks
- Total tests: 4,893+ → 5,100+ (hex encoder, cloud pipeline, gameplay systems)

#### A/B Training Comparison (Active)

- **Run A**: `vibrant-snowball-1038` (cbgz7tco) — patch_embed baseline, 20 iterations, agent_id=ab_patch_embed
- **Run B**: Auto-launches after Run A — hex_crystalline, 20 iterations, agent_id=ab_hex_crystalline
- **VM**: civ6az-orchestrator (e2-small, us-central1-a), Docker v0.14.1
- **Estimated cost**: ~$5.00 total for both runs

---

## [0.13.0] — 2026-02-14

### Added — Standalone Tournament Worker & Civ6 Lua Integration

#### Standalone Tournament Worker (`alpha_zero/cloud/gcp/tournament_worker.py`) — NEW

- **CLI-based tournament runner**: Runs ELO-rated round-robin tournaments across GCS model versions independently of the training pipeline
  - `--model_stride`: Evaluate every Nth model (default: 1) for faster tournaments across many versions
  - `--games_per_matchup`: Games per model pair (default: 6) with alternating sides
  - `--mcts_type`, `--mcts_simulations`: MCTS configuration for tournament play
  - `--output_path`: Save tournament results JSON locally
  - Reuses `arena.run_arena()` for fair game play with alternating starting player
- **VertexTournamentJob**: Vertex AI job wrapper for cloud-based tournament execution
- **Model loading head dimension inference**: Older models (v0-v29) missing `policy_hidden_dim`/`value_head_dim` in saved configs are now inferred from `state_dict` tensor shapes, enabling mixed-architecture tournaments

#### Civ6 Lua Integration & Python Bridge Fixes (PR #46)

##### Python Bridge Fixes

- `action_converter.py`: `int()` → `round()` for coordinate scaling (fixes off-by-one at grid boundaries); extracted `_ITEM_HASH_RANGE` constant
- `config.py`: Added `typing.Any` import; typed `kwargs` dict
- `file_bridge.py`: Extracted `_MAX_RETRY_BACKOFF_SECONDS` constant; explicit float/bool casts for robustness
- `protocol.py`: Added `to_dict()` methods on TileData, UnitData, CityData, PlayerData, LegalAction; added `CommandMessage.create_single()` factory; added `civics_researched`/`current_civic` fields to PlayerData
- `state_converter.py`: Extracted `_ESTIMATED_TOTAL_TECHS`/`_ESTIMATED_MAX_SCORE` constants; implemented `_extract_civics()` with `_normalize_civic_name()`; `int()` → `round()` in `inverse_scale_coords`

##### Lua Mod Fixes

- `ActionExecutor.lua`: `UnitManager.MoveUnit` → `RequestOperation` (correct Civ6 API); `safeGetUnit`/`safeGetCity` helpers; new handlers (sleep, promote, delete, buy, citizen, diplomacy stubs)
- `Main.lua`: Remove duplicate event registrations; re-entrance guard; fix polling handle leak; graceful shutdown
- `StateExporter.lua`: Nil guards throughout; hex-grid 6-neighbor offsets (was square 8-dir); CanProduce items; civics export; enemy unit detection
- `FileIO.lua`: `shellEscape()` for injection prevention; content-hash change detection for efficient polling
- `Config.lua`: Cross-platform PATH_SEP; path traversal prevention; JSON config loading
- `Logger.lua`: Log rotation at 5MB file size

#### Test Suite

- `tests/cloud/unit/test_tournament_worker.py` — 42 tests (model discovery, matchup generation, ELO computation, CLI args, GCS integration, error handling)
- `tests/unit/test_action_converter.py` — +448 lines (unit/city/tech action conversion, legality checking, find_unit/city_idx, valid action fallback)
- `tests/unit/test_bridge.py` — +200 lines (retry delays, heartbeat, poll_ack, errors, send_command_with_retry, config validation)
- `tests/unit/test_bridge_protocol.py` — +126 lines (unskipped 7 tests: to_dict, create_single, round-trip serialization)
- `tests/unit/test_state_converter.py` — +113 lines (coordinate scaling round-trip, civics extraction, constant values)

### Changed

- Version: `0.12.0` → `0.13.0`
- Total tests: 3,678 → 4,893+ (33% increase)
- GCS models: 30 (v0-v29) → 80 (v0-v79) from 50-iteration training run
- `.gitignore`: Added tournament worker artifacts, Lua mod IPC communication files
- `evaluation_worker.py`: Head dimension inference from state_dict for older model compatibility
- `vertex_orchestrator.py`: 6 new CLI args for tournament control, refactored `run_post_run_evaluation()`
- `config.py`: Added H100 GPU pricing to `DEFAULT_GPU_COSTS`

#### Live Training Run (Phase 15: golden-dream-542)

- **50/50 iterations** on Vertex AI with value_lr=0.1x, warmup_cosine LR schedule
- **80 model versions** (v0-v79) saved to GCS
- **Standalone tournament** running (17 models, stride=5, 136 matchups, ~17h estimated)
- **Cost**: ~$4.63 total (50 iterations, 2 workers, 50 games/worker)

---

## [0.12.0] — 2026-02-12

### Added — Phase 15: Scaled Training & Value Head Stabilization

#### Separate Policy/Value Head Learning Rates (`training_worker.py`)

- **Per-head LR multipliers**: `--value_lr_multiplier` and `--policy_lr_multiplier` CLI args (default 1.0)
  - Optimizer split into 3 parameter groups: encoder, policy_head, value_head
  - Each group has independent learning rate scaled by its multiplier
  - `base_lr_ratio` stored per group so LR scheduler preserves ratios during warmup/cosine decay
- **Per-head W&B logging**: `train/learning_rate_encoder`, `train/learning_rate_policy_head`, `train/learning_rate_value_head`
- **Orchestrator pass-through**: `vertex_orchestrator.py` accepts and forwards multiplier args to training worker (only when non-default, backwards-compatible via `getattr`)

#### Test Suite (8 new tests)

- `tests/unit/test_phase15_value_head_lr.py` — 8 tests:
  - `test_param_groups_created` — 3 param groups with correct names, LRs, and full param coverage
  - `test_default_multipliers_equivalent` — Both=1.0 produces uniform LRs
  - `test_scheduler_preserves_ratios` — After scheduler step, value head LR stays 0.1x encoder LR
  - `test_train_epoch_separate_lrs` — Mini training loop completes with separate per-head LRs
  - `test_parse_args_multiplier_flags` — CLI args parse correctly with explicit values
  - `test_parse_args_multiplier_defaults` — Multiplier defaults are 1.0
  - `test_orchestrator_passes_multiplier_args` — `build_training_extra_args` includes multipliers only when non-default
  - `test_orchestrator_skips_default_multipliers` — Omits multipliers when both are 1.0

### Changed

- Version: `0.11.0` → `0.12.0`
- Docker image: v0.11.0 → v0.12.0 (`gcr.io/hrmtrain/civ6az-training:v0.12.0`)
- Smoke test: soft-music-538 (1-iter, $0.12, passed)
- Live run: golden-dream-542 (50-iter, value_lr=0.1x, eval=25 games, trajectory_window=5, warmup_cosine)

---

## [0.11.0] — 2026-02-11

### Added — Phase 14: Learning Signal Validation & Training Diagnostics

#### Training Diagnostics (`alpha_zero/training/diagnostics.py`) — NEW

- **TrainingDiagnostics**: Real-time training health monitoring
  - `compute_policy_entropy(logits)` — Policy entropy for exploration quality tracking
  - `compute_value_accuracy(predicted, target)` — Value head accuracy (sign accuracy, correlation, MAE)
  - `compute_gradient_stats(model)` — Per-layer gradient norms and overall statistics
  - `detect_training_stall(losses, window)` — Loss plateau detection via coefficient of variation
  - Configurable `window_size` for rolling statistics

#### LR Scheduling in Training Worker (`training_worker.py`)

- 6 LR schedule types: constant, linear, cosine, warmup_cosine, exponential, step
- CLI args: `--lr_schedule_type`, `--lr_initial`, `--lr_final`, `--lr_warmup_steps`, `--lr_decay_steps`
- Integrates with `HyperparameterScheduler` from `alpha_zero.orchestrator.scheduler`
- `--weight_decay` and `--diagnostic_window` CLI args

#### Training Worker Enhancements (`training_worker.py`)

- **TrainingDiagnostics integration**: Per-step policy entropy, value accuracy, gradient stats
- **TrainingReport integration**: Periodic training reports with loss/grad/throughput summaries
- **PerformanceProfiler integration**: Timing sections for forward_pass and backward_pass
- **Spatial dimension validation**: Raises `ValueError` on spatial tensor shape mismatch
- **W&B per-step metrics**: diagnostics logged via `wandb_logger.log_training_step(**diagnostic_kwargs)`

#### Self-Play Temperature (`selfplay_worker.py`)

- Wired temperature to `select_action()` call (was hardcoded to 1.0)
- MCTS config override filtering by variant — prevents passing c_puct/dirichlet_* to GumbelMCTS

#### Orchestrator Stall Detection (`vertex_orchestrator.py`)

- **`detect_stall()`**: Checks if recent eval win rates show improvement
- **`compute_win_rate_trend()`**: Linear regression on win rate history
- CLI args: `--stall_lookback`, `--stall_threshold`
- W&B alerts on stall detection

#### TransformerConfig Defaults Aligned

- `TransformerConfig` defaults now match `NetworkArchitectureConfig`: spatial_size=8, action_space_size=376, embed_dim=64, num_layers=6
- Prevents silent architecture mismatch when creating models without explicit config

#### Test Suite (83 new tests)

- `tests/unit/test_phase14_diagnostics.py` — 35 tests (entropy, value accuracy, gradient stats, stall detection)
- `tests/unit/test_phase14_selfplay.py` — 20 tests (temperature wiring, MCTS config filtering)
- `tests/unit/test_phase14_training_worker.py` — 13 tests (train_epoch, parse_args, prepare_dataset, edge cases)
- `tests/unit/test_phase14_orchestrator.py` — 14 tests (stall detection, win rate trend, build_training_extra_args)
- `tests/unit/test_phase14_config_defaults.py` — 1 test (TransformerConfig matches NetworkArchitectureConfig)

### Changed

- Version: `0.10.2` → `0.11.0`
- Docker image: v0.10.2 → v0.11.0

#### Live Training Validation (sunny-fog-469)

- **20/20 iterations**, 377.5 min, **$2.44 total cost**
- **Promotions**: v6→v19 (iter 10, 70% WR), v19→v26 (iter 17, 60% WR)
- **Tournament**: 23 models, 253 matchups, 1518 games — v28 #1 (1529.7 ELO)
- **Policy loss**: steadily decreasing (1.21→0.63) — strongest learning signal
- **Value head**: unstable (correlation 0.07-0.91) — addressed in Phase 15 via separate LR

---

## [0.10.2] — 2026-02-10

### Fixed — Live Training Bug Fixes

- **Legacy model config fallback**: Models v0-v2 (saved before Phase 13) had no `model_config`. The fallback was using `TransformerConfig()` defaults (embed_dim=256, spatial_size=16) causing size mismatch errors. Fixed to use `NetworkArchitectureConfig` pipeline defaults (embed_dim=64, board_size=8) in both `selfplay_worker.py` and `evaluation_worker.py`
- **Training worker model_config persistence**: Training worker now saves `model_config` dict (via `dataclasses.asdict()` + Enum→value conversion) alongside model weights to GCS, enabling future models to load with correct architecture
- **Post-run tournament version range**: Fixed bug where `get_initial_model_version()` was called after the training loop (returning latest=5) instead of saving initial version before the loop. Tournament `range(5, 3)` was empty. Fixed to use `range(initial_model_version, latest_version + 1)`
- **Test fixes**: Updated `test_load_model_legacy_no_config`, `test_load_model_bad_config_falls_back`, and `test_load_model_legacy_format` to create state dicts matching pipeline architecture (embed_dim=64) instead of TransformerConfig defaults (embed_dim=256)

### Changed

- Docker image: v0.10.1 → v0.10.2 (`gcr.io/hrmtrain/civ6az-training:v0.10.2`)
- Validated with live Vertex AI run: trim-night-121 (1-iter + tournament, $0.06)

---

## [0.10.0] — 2026-02-09

### Added — Phase 13: Multi-Iteration Training & Post-Run Evaluation

#### NetworkArchitectureConfig (`alpha_zero/cloud/gcp/config.py`)

- **NetworkArchitectureConfig** dataclass: Single source of truth for architecture defaults
  - `board_size=8`, `action_space_size=376`, `spatial_channels=13`, `vector_dim=64`, `num_layers=6`, `embed_dim=64`
  - `from_env()` class method for environment variable overrides (`CIV6AZ_BOARD_SIZE`, etc.)
- Added `network: NetworkArchitectureConfig` field to `GCPConfig`

#### Shared Arena Module (`alpha_zero/cloud/gcp/arena.py`) — NEW

- **`run_arena()`**: Extracted from `evaluation_worker.py` into reusable module
  - Plays N games between two models with alternating sides
  - Returns `ArenaResult` dataclass (wins, losses, draws, win_rate)
  - Used by both evaluation_worker and post-run tournament
- **`ArenaResult`**: Dataclass with `to_dict()` serialization

#### Orchestrator W&B Logging (`vertex_orchestrator.py`)

- Orchestrator-level W&B initialization (opt-in via `WANDB_ENABLED` env var)
- Per-iteration metrics: `orchestrator/iteration`, `orchestrator/model_version`, `orchestrator/phase_*_duration`
- Cost tracking: `orchestrator/iteration_cost`, `orchestrator/cumulative_cost`
- Evaluation metrics: `orchestrator/eval_win_rate`, `orchestrator/eval_accepted`
- Final summary: total iterations, success/fail counts, total cost, total duration

#### Post-Run Tournament (`vertex_orchestrator.py`)

- **`run_post_run_evaluation()`**: Round-robin tournament across all model versions
  - Loads models from GCS, runs arena games locally on orchestrator machine
  - Computes ELO ratings (K=32, starting 1500.0)
  - Logs to W&B (`tournament/elo_*`, `tournament/best_model`, `tournament/best_elo`)
  - Saves results JSON to GCS (`tournament/post_run_results.json`)
- CLI flags: `--post_run_tournament`, `--tournament_games`, `--tournament_mcts_sims`

#### Evaluation Worker Migration

- Migrated from `MicroCivNetwork` (legacy, 381 actions) to `Civ6TransformerNetwork`
- Loads `model_config` from GCS for correct architecture reconstruction
- Handles enum string conversion (`norm_type`, `attention_type`)
- Handles legacy models without config (falls back to `NetworkArchitectureConfig` defaults)

### Fixed — 6 Critical Bugs

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `selfplay_worker.py` | `action_space_size` default `381` | Changed to `376` |
| 2 | `selfplay_worker.py` | `num_res_blocks=8`, `num_filters=256` | Changed to `6`, `64` |
| 3 | `training_worker.py` | `board_size=16`, `action_space_size=381`, etc. | Changed to `8`, `376`, `6`, `64` |
| 4 | `training_worker.py` | `prepare_dataset(trajectories, args.action_space_size)` | Pass `None` (uses TransformerConfig default) |
| 5 | `evaluation_worker.py` | Uses `MicroCivNetwork` (legacy) | Migrated to `Civ6TransformerNetwork` |
| 6 | `vertex_orchestrator.py` | `if new_version is not None` always True | Removed dead branch |

### Changed

- Docker image: v0.9.0 → v0.10.0 (`gcr.io/hrmtrain/civ6az-training:v0.10.0`)
- Version: `0.9.0` → `0.10.0`
- `container_healthcheck.py`: Added `arena` module to import check

#### Test Suite (Phase 13)

- `tests/cloud/unit/test_arena.py` — Arena module unit tests
- `tests/cloud/unit/test_config_network.py` — NetworkArchitectureConfig tests
- `tests/cloud/unit/test_orchestrator_wandb.py` — Orchestrator W&B logging tests
- `tests/cloud/unit/test_post_run_evaluation.py` — Post-run tournament tests
- Updated `test_evaluation_worker.py` — Migrated to Civ6TransformerNetwork
- Updated `test_selfplay_worker.py` — Updated legacy format test

#### Live Validation

- **wandering-yogurt-88**: 3-iteration run, 54.5 min, $0.32 — all phases succeeded
- **trim-night-121**: 1-iteration + post-run tournament, $0.06 — tournament produced ELO rankings, saved to GCS
- GCS models: v0 through v6

---

## [0.9.0] — 2026-02-10

### Changed — Phase 11: Code Quality, Tech Debt Reduction & Production Hardening

#### Dead Code Removal (~14,600 LOC)

- Deleted `agentic_framework/` directory (41 modules, 14,600 LOC) — confirmed zero imports from `alpha_zero/`
- Removed `agentic_framework/` from ruff check in CI, pyproject.toml per-file-ignores

#### Deprecated Modules (with warnings, not deleted)

- `alpha_zero/mechanics/` — 12 files, 223 tests; future Civ6 integration
- `alpha_zero/human_interface/` — 5 files, 58 tests; future Human vs AI
- `alpha_zero/integration/game_connector.py` — 38 tests; future CivRealm integration
- Each `__init__.py` emits `DeprecationWarning` on import

#### XFailed Test Fixes (16 of 27 → passes)

- Fixed 2 `test_vertex_trainer.py` tests (outdated mock constructor params)
- Fixed 1 `test_e2e_llm.py` test (AsyncOpenAI mock chain)
- Fixed 1 `test_production_resilience.py` test (mock patching path)
- Fixed 1 `test_run_muzero_training.py` test (policy sum tolerance)
- Fixed 5 `test_state_extraction.py` tests (mock dataclass alignment with StateConverter)
- Fixed 1 `test_hierarchical_network.py` test (INVALID_ACTION_LOGIT constant)
- Fixed 1 `test_action_masks.py` test (per-batch all-masked check)
- Fixed 1 `test_training.py` test (checkpoint resume API)
- Fixed 1 `test_real_civ6_env.py` test (CivRealmWrapper mock sync)
- Remaining 13 xfails: 12 DDP tests (different API) + 1 hierarchical action space (deferred to Phase 14)

#### Deprecated `datetime.utcnow()` Migration (19 files)

- Replaced `datetime.utcnow()` with `datetime.now(UTC)` across 19 files
- Added `from datetime import UTC` imports (Python 3.12+ deprecation fix)
- Files: logging_config.py, tournament (config/elo/history/league), evaluation (reports), pbt (scheduler/coordinator), cloud (gcs_storage/preemption_handler/cost_monitor), core (reflection_memory), rag (5 files), human_interface (2 files)

#### Logging Consolidation

- **`logging_config.py` is now canonical** — absorbs MetricsTracker, TrainingLogger, GameLogger, PerformanceProfiler, CircularBuffer, LogConfig from old `logging.py`
- **`logging.py` is now a thin re-export** with `DeprecationWarning`
- Standardized env vars: `CIV6AZ_LOG_LEVEL`, `CIV6AZ_LOG_FORMAT`, `CIV6AZ_LOG_JSON`
- Removed duplicate `ColoredFormatter` and `JSONFormatter` (kept `logging_config.py` versions)
- Backward-compatible `setup_logging()` accepts deprecated `config=`, `console_output=`, `file_output=` kwargs

#### Production `print()` Cleanup

- Replaced `print()` with `logger.info()` in 5 production modules (evaluator.py, reports.py, diversity.py, entrypoint.py, analysis.py)
- CLI modules (`cli.py`, `lifecycle_cli.py`) retain `print()` — intentional CLI output

#### Mypy Incremental Strictness

- Added `disallow_untyped_defs = true` + `disallow_incomplete_defs = true` for 4 packages:
  - `alpha_zero.core.*`, `alpha_zero.training.*`, `alpha_zero.tournament.*`, `alpha_zero.pbt.*`
- 0 new `no-untyped-def` errors (packages were already well-typed)

#### CI Hardening

- Pinned tool versions: `ruff==0.8.0`, `mypy==1.13.0`, `bandit==1.8.0`
- Added `ruff format --check alpha_zero/` to lint job
- Bandit now enforced (removed `|| true`): `--severity-level high --confidence-level high`
- Added `benchmark` pytest marker to `pyproject.toml`

#### Edge Case Tests (16 new tests)

- `tests/unit/test_mcts_edge_cases.py` — 4 tests (single valid action, extreme values, zero input, batch size one)
- `tests/unit/test_buffer_edge_cases.py` — 4 tests (empty buffer, overflow, concurrent access, zero batch size)
- `tests/unit/test_training_safeguards.py` — 4 edge case tests (NaN+grad, inf loss, negative loss, small gradient)
- `tests/unit/test_tournament_orchestrator.py` — 4 edge case tests (single player, two players, extreme ELO, all draws)

#### Version & Documentation

- Bumped version: `0.8.0` → `0.9.0`
- Updated MEMORY.md with Phase 11 status

---

### Added — Phases 8-10: Training Safeguards, MuZero Integration & Tournament System

#### Phase 8: Training Validation & Safeguards

##### Training Safeguards (`alpha_zero/training/training_safeguards.py`) — NEW

- **TrainingSafeguard**: Unified training health monitor wired into orchestrators
  - `check_step(loss, grad_norm, step)` — Runs all checks, returns `SafeguardReport`
  - NaN detection with configurable consecutive threshold (`max_consecutive_nan_steps`)
  - Loss explosion detection via rolling baseline comparison (`loss_explosion_threshold`)
  - Gradient health monitoring: NORMAL, HIGH, EXPLODING status levels
  - Memory pressure detection via existing `MemoryGuard` integration
  - Checkpoint validation: tensor shape, dtype, NaN presence verification
  - Automated intervention recommendations: REDUCE_LR, ROLLBACK, REDUCE_BATCH_SIZE, EARLY_STOP
- **TrainingSafeguardConfig**: Fully configurable with `to_dict()`/`from_dict()` serialization
- **SafeguardReport**, **Intervention**, **GradientStatus**, **InterventionAction** dataclasses

##### Training Reports (`alpha_zero/training/training_report.py`) — NEW

- **TrainingReport**: Periodic training report generation at configurable intervals
  - `record_step()` / `generate_report()` / `to_wandb_dict()` pipeline
  - Computes mean/min/max loss, gradient norms, throughput (steps/sec)
  - Includes safeguard events triggered during interval
- **TrainingReportConfig**, **TrainingReportData** dataclasses

##### Orchestrator Integration

- **TrainingOrchestrator**: Added `set_training_safeguard()` setter for backwards compatibility
  - Safeguard checks after each training step in `_train_epochs()`
  - NaN triggers rollback to last checkpoint + LR reduction
  - Loss explosion triggers LR reduction by factor 0.5
  - Memory pressure triggers batch size reduction via `MemoryGuard`
- **PBTOrchestrator**: Added `training_safeguard_config` keyword-only param
  - Per-agent safeguard creation, events logged to W&B

#### Phase 9: MuZero Integration Bridge

##### MuZero Training Bridge (`alpha_zero/training/muzero_integration.py`) — NEW

- **MuZeroTrainingBridge**: High-level interface connecting MuZero components to orchestrators
  - `create_network()`, `create_trainer()`, `create_mcts()` — Factory methods
  - `selfplay_step()` — Self-play via MuZeroMCTS (no env cloning needed)
  - `training_step()` — Wraps `MuZeroTrainer.train_step()` with metric extraction
  - `to_training_config()` — Converts integration config to `MuZeroTrainingConfig`
- **MuZeroIntegrationConfig**: Env-var configurable (`CIV6AZ_MUZERO_*` prefix)
  - `from_env()`, `from_yaml()`, `from_dict()`, `to_yaml()` serialization

##### MuZero PBT Hyperparameters (`alpha_zero/pbt/muzero_hyperparams.py`) — NEW

- `create_muzero_hyperparameter_space()` — PBT-compatible hyperparameter space
- `merge_with_base_space()` — Combines MuZero params with existing AlphaZero space

##### Orchestrator Integration

- **TrainingOrchestrator**: Added `set_muzero_bridge()` setter, `training_mode` dispatch
  - Default `training_mode="alphazero"` preserves existing behavior
- **PBTOrchestrator**: Added `muzero_config` keyword-only param

##### Configuration

- `configs/muzero_pbt.yaml` — MuZero PBT config (population_size=6, mutation_strength=0.15)

#### Phase 10: Tournament & Matchmaking System

##### Tournament Orchestrator (`alpha_zero/tournament/orchestrator.py`) — NEW

- **TournamentOrchestrator**: Bracket generation, match execution, standings
  - Formats: `round_robin`, `single_elimination`, `swiss`
  - Integrates with existing `ELORatingSystem`, `MatchMaker`, `MatchHistory`
- **TournamentFormat** enum, **Bracket**, **Matchup**, **PlayerStanding**, **TournamentResult**

##### Tournament Analysis (`alpha_zero/tournament/analysis.py`) — NEW

- **TournamentAnalyzer**: Post-tournament analytics
  - Head-to-head matrix, player trends, meta-analysis, upset detection
  - Cross-generation strength comparison, JSON/markdown report export

##### Tournament W&B Dashboards (`alpha_zero/tournament/wandb_dashboards.py`) — NEW

- **TournamentWandbLogger**: `tournament/` namespaced W&B metrics
  - Match results, round summaries, ELO leaderboard, head-to-head heatmap, ELO history

##### PBT Integration

- **PBTOrchestrator**: Added `tournament_interval` and `tournament_games_per_match` params
  - `_run_tournament_evaluation()` — Runs at configured intervals, logs to W&B

##### Configuration

- `configs/tournament.yaml` — Tournament config (format, ELO, matchmaking, analysis, baselines)

#### Test Suite (234 new tests, 495 total Phase 7-10)

- `tests/unit/test_training_safeguards.py` — 56 tests
- `tests/unit/test_training_report.py` — 16 tests
- `tests/integration/test_safeguard_integration.py` — 9 tests
- `tests/unit/test_muzero_integration.py` — 37 tests
- `tests/unit/test_muzero_hyperparams.py` — 32 tests
- `tests/unit/test_tournament_orchestrator.py` — 33 tests
- `tests/unit/test_tournament_analysis.py` — 15 tests
- `tests/unit/test_tournament_wandb_dashboards.py` — 15 tests

### Changed

- `.gitignore` — Added tournament DB, tournament reports, verify_ruff.py patterns
- `alpha_zero/training/__init__.py` — Added Phase 8-9 exports
- `alpha_zero/pbt/__init__.py` — Added Phase 9 exports
- `alpha_zero/tournament/__init__.py` — Added Phase 10 exports
- All new modules use graceful degradation (`try/except ImportError` with `_AVAILABLE` flags)

---

## [0.8.0] — 2026-02-08

### Added — Phase 7: Production Scaling & Training Validation

#### W&B Population Dashboards

- `WandbLogger.log_population_stats()` — Logs `pbt/` namespaced metrics (ELO, fitness, mutations) per generation
- `WandbLogger.define_pbt_charts()` — Registers W&B metric axes with summary types (max, min, last, sum)
- Per-agent ELO tracking via `pbt/agent/{agent_id}/elo` metrics
- Mutation event tables logged via `WandbLogger.log_table()`

#### GCS Checkpoint Rotation

- `PBTOrchestrator._rotate_checkpoints()` — Manifest-based checkpoint pruning in GCS
- Configurable `checkpoint_retention_generations` (default: 5, 0=unlimited)
- Manifest stored at `{prefix}/checkpoint_manifest.json` mapping generations to blob paths
- Graceful error handling — rotation failures never crash the main loop

#### GPU Quota Guard

- `PBTOrchestrator._check_quota()` — Queries active Vertex AI jobs and counts GPUs
- `PBTOrchestrator._wait_for_quota()` — Exponential backoff polling (60s→300s cap, 1800s max)
- Configurable via `CIV6AZ_GPU_QUOTA_LIMIT` and `CIV6AZ_QUOTA_WAIT_MAX_SECONDS` env vars
- Optimistic fallback on API errors (returns `True` on failure)

#### Baseline Benchmarking

- `PBTEvaluator.run_baseline_benchmarks()` — Runs population agents vs baseline agents at configurable intervals
- `EvaluationReport.baseline_results` field for storing benchmark match results
- Baselines do NOT affect ELO ratings — serves as absolute performance anchor
- Configurable via `baseline_interval` (default: 5, 0=disabled) and `baseline_agents` list

#### Dynamic Evaluation Strategy Switch

- `PBTOrchestrator._maybe_switch_strategy()` — Switches from `round_robin` to `top_k_vs_best` at configurable generation
- `evaluation_strategy_switch_gen` config field (default: 0 = never switch)
- Reduces evaluation cost as population matures

#### PBTConfig Enhancements

- 4 new fields: `baseline_interval`, `baseline_agents`, `checkpoint_retention_generations`, `evaluation_strategy_switch_gen`
- Full validation (`>= 0` checks), `to_dict()`/`from_dict()` serialization, YAML round-trip support
- Backwards compatible — all fields have defaults, existing configs unchanged

#### Production Configuration

- `configs/pbt_prod.yaml` — Production PBT config with GumbelMCTS-tuned hyperparameter ranges
  - `population_size: 8`, `evaluation_strategy_switch_gen: 11`, `baseline_interval: 5`
  - Full hyperparameter search space (learning_rate, c_puct, num_simulations, batch_size, etc.)

#### Test Suite (279 PBT tests passing)

- `tests/unit/test_pbt_dashboarding.py` — 14 tests for W&B population dashboard methods
- `tests/integration/test_production_resilience.py` — 18 tests for checkpoint rotation, quota guard, strategy switch, baseline benchmarks
- `tests/unit/test_pbt_config.py` — 12 new tests for Phase 7 config fields
- `tests/unit/test_pbt_evaluator.py` — 8 new tests for baseline benchmarking
- Fixed pre-existing `test_pbt_orchestrator.py` bugs (invalid VertexEvaluationJob patch, missing arena runner mock)

### Fixed

- `pbt_orchestrator.py` — Lambda keyword arg mismatch in `_run_evaluation_phase` fallback arena runner
- `pbt_orchestrator.py` — Pre-existing trailing whitespace lint issue in `_local_arena_runner`
- `test_pbt_orchestrator.py` — Removed invalid `VertexEvaluationJob` patch, added mock arena runner and `checkpoints_prefix` mock

### Added — Comprehensive Test Infrastructure & CI/CD (Phases 1-6)

#### Phase 1: Test Coverage Hardening

- `tests/sanity/test_import_sanity.py` — dynamic import verification for all `alpha_zero` submodules
- `tests/security/test_security_sanity.py` — secret scanning, `.gitignore`/`.dockerignore` validation
- Coverage threshold raised in `pyproject.toml` and CI configuration

#### Phase 2: CI/CD Pipeline Enhancements

- `.github/workflows/release.yml` — automated changelog generation and GitHub releases
- `.env.template` — environment variable documentation for W&B, GCP, and training config

#### Phase 3: GCP E2E Validation

- `tests/e2e/test_gcs_live.py` — live GCS storage operations (upload/download roundtrip, JSON, large payloads)
- `tests/e2e/test_vertex_submission.py` — Vertex AI job submission dry-run validation

#### Phase 4: Local Training Smoke Tests

- `tests/e2e/test_local_training_smoke.py` — end-to-end pipeline validation (self-play → training → evaluation)

#### Phase 5: Distributed Training Dry-Run

- `tests/integration/test_distributed_coordination.py` — multi-worker coordination simulation
- `tests/integration/test_preemption_recovery.py` — Spot VM preemption and checkpoint recovery

#### Phase 6: PBT Optimization & Evaluation

- `configs/pbt_config.yaml` — production PBT configuration with hyperparameter search spaces
- `tests/integration/test_pbt_lifecycle.py` — population lifecycle, exploitation, exploration, mutation tests
- `.agent/workflows/pbt-launch.md` — PBT launch workflow

#### Cross-Cutting

- `NEXT_STEPS.md` — comprehensive 6-phase development roadmap
- Vertex AI smoke config templates (`configs/vertex_selfplay_smoke.yaml`, `configs/vertex_training_smoke.yaml`)
- Sanitized W&B API key handling in config templates

### Changed

- `.gitignore` — added patterns for local config overrides, AI coding assistant directories
- Updated C4 architecture roadmap to reflect completed GCP and CI/CD phases
- Updated README test categories and project structure

### Fixed

- CI/CD pipeline stability improvements across all test tiers
- Vertex AI pipeline fixes (PR #34): resolved 3 CI failures
- Cost optimization (PR #29): addressed Gemini and Copilot review issues

---

## [Unreleased]

### Added

#### Cloud Training Enhancements (Phase 5)

- **Multi-GPU Training Support**: Implemented `DistributedDataParallel` (DDP) in `alpha_zero/cloud/gcp/training_worker.py`
  - Automatic `torchrun` execution for multi-accelerator jobs via `VertexTrainingJob`
  - Efficient `DistributedSampler` for data sharding
  - Rank-0 only logging and checkpointing
- **Training Validation Infrastructure**: Added `configs/vertex_orchestrator_smoke.yaml` for multi-iteration loop validation
- **DDP Test Suite**: Added `tests/cloud/unit/test_training_worker.py` and updated `test_vertex_jobs.py`

#### GCP Vertex AI Production Pipeline

End-to-end cloud training on Vertex AI with GumbelMCTS, model evaluation, W&B tracking, and opt-in curriculum/reward shaping. Successfully validated with a 3-phase cloud run (selfplay, training, evaluation) completing in 17 minutes.

##### MCTS Factory (`alpha_zero/core/mcts_factory.py`) — NEW

- `create_mcts()` factory function for instantiating any MCTS variant by name
- Supports `"civ6"` (standard), `"gumbel"` (5-50x faster), and `"muzero"` (future)
- Variant-specific config defaults (200 sims for civ6, 16 for gumbel, 50 for muzero)
- Pass-through kwargs for variant-specific settings (e.g., `num_sampled_actions`)

##### Evaluation Worker (`alpha_zero/cloud/gcp/evaluation_worker.py`) — NEW

- Model arena: plays new model vs. old model to measure improvement
- Alternates starting player each game for fairness
- Temperature=0 (greedy) play for evaluation consistency
- Results saved as JSON to GCS (`gs://{bucket}/evaluations/iter_{N}/results.json`)
- Win rate threshold gate for model acceptance/rejection

##### Vertex AI Orchestrator (`alpha_zero/cloud/gcp/vertex_orchestrator.py`) — NEW

- 3-phase pipeline: selfplay → training → evaluation (with model gate)
- CLI-driven with `--mcts_type`, `--evaluation_games`, `--curriculum`, `--reward_shaping`
- Dry-run mode for local validation without GCS/Vertex AI
- Curriculum learning integration via `CurriculumManager`
- W&B metrics passthrough to containers

##### Container Infrastructure — NEW

- `Dockerfile`: PyTorch 2.1 + CUDA 12.1 base, slim dependencies
- `.dockerignore`: Excludes tests, docs, data, IDE files
- `requirements-container.txt`: Minimal deps (torch, numpy, gymnasium, wandb, GCP)
- `scripts/build_container.sh`: Build and push script for GCR
- `alpha_zero/cloud/gcp/entrypoint.py`: Routes `--mode=selfplay|training|evaluation`

##### Vertex AI Job Wrappers (`alpha_zero/cloud/gcp/vertex_jobs.py`)

- `VertexEvaluationJob`: CPU-only evaluation jobs (n1-standard-4, no GPU)
- `_wandb_env_vars()` helper: Filters empty values (Vertex AI rejects empty env vars)
- W&B env var passthrough to selfplay and training containers

##### Self-Play Worker Enhancements (`alpha_zero/cloud/gcp/selfplay_worker.py`)

- GumbelMCTS support via `--mcts_type=gumbel` (~13x speedup over Civ6MCTS)
- W&B logging: per-game metrics and aggregate selfplay stats
- Opt-in reward shaping via `--reward_shaping` and `--reward_scale`
- Automatic noise control: Gumbel provides its own stochasticity

##### Training Worker Enhancements (`alpha_zero/cloud/gcp/training_worker.py`)

- W&B logging for per-epoch training metrics
- Board size and action space CLI args for multi-resolution support

##### Configuration (`alpha_zero/cloud/gcp/config.py`)

- `DEFAULT_GPU_COSTS` dict: Centralized GPU pricing (T4, L4, V100, A100 on-demand/spot)
- `get_gpu_costs()`: Env-var overridable GPU costs (`CIV6AZ_GPU_COST_*`)
- `SelfPlayWorkerConfig` / `TrainingWorkerConfig` extensions for MCTS/W&B settings

##### GPU Pricing Centralization

- `gpu_scheduler.py`, `cost_optimization.py`, `cost_monitor.py`: All use `get_gpu_costs()` from config
- Override any GPU price via `CIV6AZ_GPU_COST_T4`, `CIV6AZ_GPU_COST_V100`, etc.

##### Preemption Handler Fixes (`alpha_zero/cloud/gcp/preemption_handler.py`)

- Fixed 3 broken GCS method calls to use `self.gcs_storage.backend.*` API
- Added `_upload_to_gcs()`, `_download_from_gcs()`, `_delete_from_gcs()` helpers

##### `.env.template` Updates

- Added GCP/Vertex AI configuration section (project, region, bucket, container image)
- Added MCTS configuration (type, simulations)
- Added GPU pricing overrides with defaults
- Added cost budget settings

##### Test Suite (50 new tests)

- `tests/unit/test_mcts_factory.py`: 19 tests (type dispatch, config, functional)
- `tests/cloud/unit/test_evaluation_worker.py`: 16 tests (arena, GCS, Vertex jobs)
- `tests/cloud/unit/test_orchestrator_pipeline.py`: 15 tests (phases, args, GPU config)

### Fixed

- Removed unused `numpy` imports from evaluation_worker.py and selfplay_worker.py
- Fixed `enhanced_rewards` import path: `alpha_zero.utils` not `alpha_zero.core`
- Fixed unused `stage` variable in vertex_orchestrator.py
- Removed dead `get_output_model_version()` method from vertex_jobs.py
- Removed empty `TYPE_CHECKING` block from vertex_jobs.py
- Fixed `dict()` → `{}` literal in evaluation_worker.py

---

#### Phase 5.4: Cost Estimator & Robust Preemption (Finalization)

- **`CostEstimator` (`alpha_zero/cloud/gcp/cost.py`)**:
  - Real-time job cost estimation based on machine type and accelerator count.
  - Integration with `vertex_orchestrator.py` for cost visibility in logs.
- **Robust Preemption Handling**:
  - `SignalHandler` in `training_worker.py` for graceful `SIGTERM` handling.
  - Immediate checkpoint saving upon preemption signal.
- **Unified Dashboard**:
  - `configs/wandb_dashboard.json`: Comprehensive W&B dashboard template.

#### Phase 5.3: Cost Optimization for GCP Cloud Training

A comprehensive cost optimization system targeting 60-80% cost savings through spot VMs, dynamic GPU scheduling, auto-stop idle instances, and risk mitigation systems.

##### Cost Monitoring & Budget Enforcement (`alpha_zero/cloud/gcp/cost_monitor.py`)

- **CostMonitorConfig**: Configurable budget limits with environment variables (`CIV6AZ_COST_*`)
  - Daily and total budget limits in USD
  - Alert thresholds (warning at 80%, critical at 95%)
  - Check interval configuration
- **CostMonitor**: Real-time cost tracking and budget enforcement
  - GPU pricing estimates for T4, V100, P100, A100, L4
  - Alert callbacks for budget thresholds
  - Background monitoring thread
  - `can_afford()` for budget checks before launching jobs
- **Exceptions**: `BudgetExceededError` for budget limit handling

##### Dynamic GPU Scheduler (`alpha_zero/cloud/gcp/gpu_scheduler.py`)

- **GPUSchedulerConfig**: Training phase-aware GPU selection
  - Phase thresholds: Early (0-50%), Middle (50-80%), Late (80-90%), Final (90-100%)
  - Configurable GPU mappings per phase
  - Spot VM preference settings
- **DynamicGPUScheduler**: T4 → V100 upgrade strategy
  - Uses cheaper T4 GPUs for early/middle training phases
  - Automatically upgrades to V100 for final phase precision
  - Budget-aware selection with cost monitor integration
  - 42 unit tests

##### Preemption Handler (`alpha_zero/cloud/gcp/preemption_handler.py`)

- **PreemptionHandlerConfig**: Checkpoint and retry settings
  - 5-minute checkpoint intervals (configurable)
  - Exponential backoff with max 5 retries
  - GCS storage integration
- **PreemptionHandler**: Spot VM preemption recovery
  - SIGTERM signal handling for graceful shutdown
  - Automatic periodic checkpointing
  - Resume state persistence to GCS
  - Checkpoint callbacks for integration
- **Exceptions**: Enhanced `QuotaExceededError` with fallback configs

##### Idle Instance Manager (`alpha_zero/cloud/gcp/idle_manager.py`)

- **IdleInstanceConfig**: Idle detection settings
  - Configurable idle threshold (default 15 minutes)
  - GPU utilization threshold (default 10%)
  - Per-job-type policies (selfplay, training, evaluation)
  - Grace period before shutdown
- **IdleInstanceManager**: Auto-stop idle instances
  - Job registration and activity tracking
  - Idle detection by time and GPU utilization
  - Checkpoint-before-stop logic
  - Warning and stop callbacks
- **Exceptions**: `IdleTimeoutError` for recoverable shutdowns

##### Quota Manager (`alpha_zero/cloud/gcp/quota_manager.py`)

- **QuotaManagerConfig**: Quota tracking settings
  - Warning/critical thresholds (70%/90%)
  - Cache TTL for API responses
  - Fallback strategies configuration
- **QuotaManager**: GCP quota management and fallback
  - Quota usage tracking with caching
  - Fallback strategies: alternative GPUs, regions, spot VMs
  - Quota exhaustion prediction
  - Warning and exhaustion callbacks

##### Memory Guard (`alpha_zero/utils/memory_guard.py`)

- **MemoryGuardConfig**: Memory protection settings
  - Warning/critical/OOM thresholds (70%/85%/95%)
  - Batch reduction factor and minimum batch size
  - Gradient checkpointing settings
- **MemoryGuard**: GPU/CPU memory protection
  - Real-time memory monitoring (PyTorch CUDA + psutil)
  - Automatic batch size reduction on memory pressure
  - Gradient checkpointing activation
  - OOM prevention with garbage collection

##### Convergence Monitor (`alpha_zero/training/convergence_monitor.py`)

- **ConvergenceMonitorConfig**: Training monitoring settings
  - Window size for trend analysis
  - Stall detection threshold
  - Improvement thresholds
- **ConvergenceMonitor**: Training stall detection
  - Status tracking: IMPROVING, STABLE, SLOWING, STALLED, DIVERGING
  - PBT mutation recommendations
  - Curriculum advancement/retreat recommendations
  - Learning rate reduction suggestions

##### Bridge Health Checker (`alpha_zero/bridge/health_checker.py`)

- **BridgeHealthConfig**: Health monitoring settings
  - Check interval and timeout configuration
  - Circuit breaker thresholds
  - Exponential backoff settings
- **GameBridgeHealthChecker**: Resilient bridge communication
  - Circuit breaker pattern (CLOSED → OPEN → HALF_OPEN)
  - Exponential backoff retry logic
  - Health status: HEALTHY, DEGRADED, UNHEALTHY
  - Operation timeout handling
- **Exceptions**: `CircuitBreakerOpenError`, `OperationTimeoutError`

##### Unified Cost Optimization Manager (`alpha_zero/cloud/gcp/cost_optimization.py`)

- **CostOptimizationConfig**: Unified configuration
  - Combines all component configs
  - Environment variable support (`CIV6AZ_COST_OPT_*`)
  - YAML serialization
- **CostOptimizationManager**: Single interface for cost optimization
  - Coordinates all cost optimization components
  - `get_optimized_job_config()`: Returns optimal GPU/spot/region config
  - `get_savings_report()`: Comprehensive savings tracking
  - Job lifecycle management (start, activity, end)
  - Graceful initialization and shutdown

##### Test Suite (264 new tests)

- `tests/cloud/unit/test_cost_monitor.py`: Cost monitoring tests (42 tests)
- `tests/cloud/unit/test_gpu_scheduler.py`: GPU scheduler tests (39 tests)
- `tests/cloud/unit/test_preemption_handler.py`: Preemption tests (43 tests)
- `tests/cloud/unit/test_idle_manager.py`: Idle management tests (46 tests)
- `tests/unit/test_bridge_health.py`: Health checker tests (53 tests)
- All tests pass with comprehensive coverage

##### Configuration Pattern

All components follow consistent configuration patterns:

- Dataclass-based configs with validation
- Environment variable support with `CIV6AZ_` prefix
- YAML file loading support
- `from_dict()` and `to_dict()` serialization
- Thread-safe implementations with proper locking

#### Phase 4.1: Model Evaluation Framework

A comprehensive model evaluation system for measuring trained model performance against configurable baselines.

##### Evaluation Module (`alpha_zero/evaluation/`)

- **EvaluationConfig** (`config.py`): Unified configuration for evaluation
  - Experiment naming and output directory configuration
  - Board size and max turns settings
  - ELO rating integration settings
  - Environment variable support (`CIV6AZ_EVAL_*` prefix)

- **BaselineConfig** (`config.py`): Configuration for baseline opponents
  - Support for RANDOM, HEURISTIC, and NEURAL_NETWORK baseline types
  - Per-baseline game count and weight configuration
  - HeuristicWeights for customizable heuristic behavior

- **DiversityConfig** (`config.py`): Diversity tracking configuration
  - Opening move tracking with configurable depth
  - Game length distribution analysis

- **BaselinePlayer** (`baselines.py`): Protocol and implementations
  - RandomPlayer: Uniform random action selection with seeding
  - HeuristicPlayer: Configurable weighted heuristic evaluation
  - NeuralNetworkPlayer: Model checkpoint loading with optional MCTS
  - BaselineFactory: Factory pattern for baseline creation

- **ModelEvaluator** (`evaluator.py`): Main evaluation orchestrator
  - Integration with existing ELORatingSystem from tournament/elo.py
  - GameResult and EvaluationResult dataclasses
  - play_game(), evaluate_against_baseline(), evaluate_model() methods

- **DiversityTracker** (`diversity.py`): Game diversity analysis
  - Opening sequence tracking with entropy calculation
  - Game length statistics (mean, std, min, max)
  - Baseline-specific diversity breakdowns

- **ReportGenerator** (`reports.py`): Multi-format report generation
  - EvaluationReport dataclass with full evaluation results
  - JSON export with to_json() method
  - Markdown export with to_markdown() method

##### Evaluation Test Suite (62 new tests)

- `tests/unit/test_evaluation/test_config.py`: Configuration validation (10 tests)
- `tests/unit/test_evaluation/test_baselines.py`: Baseline player tests (15 tests)
- `tests/unit/test_evaluation/test_diversity.py`: Diversity tracking tests (12 tests)
- `tests/unit/test_evaluation/test_evaluator.py`: Evaluator tests (14 tests)
- `tests/unit/test_evaluation/test_reports.py`: Report generation tests (11 tests)
- `tests/integration/test_evaluation_pipeline.py`: Integration tests (11 tests)

##### Evaluation Configuration (`configs/evaluation.yaml`)

- Complete YAML configuration template
- Random and heuristic baseline definitions
- Diversity tracking settings
- ELO integration settings

#### Phase 4.2: PBT Enhancements

Enhanced Population-Based Training with temperature scheduling and crossover mutations.

##### Temperature Schedule (`alpha_zero/pbt/config.py`)

- **HyperparameterSpace** enhancements:
  - temperature_initial: Starting temperature for exploration
  - temperature_final: Ending temperature for exploitation
  - temperature_decay_steps: Steps over which temperature decays
  - Updated get_all_params() and serialization methods

##### Crossover Mutation (`alpha_zero/pbt/mutations.py`)

- **CrossoverMutation** class:
  - BLX-alpha blending crossover (default α=0.5)
  - Simulated Binary Crossover (SBX) with η=20
  - Uniform crossover with 50% probability
  - Automatic fallback to perturbation without parent

- **MutationOperator** enhancements:
  - CROSSOVER strategy in MutationStrategy enum
  - set_crossover_parent() method for parent injection
  - Integration with existing mutation infrastructure

#### Phase 4.3: Performance Optimizations

Research-based optimizations for faster training and improved sample efficiency.

##### Optimizations Module (`alpha_zero/optimizations/`)

- **PlayoutCapRandomizer** (`playout_cap.py`): Playout cap randomization
  - Geometric, uniform, and exponential cap distributions
  - Position complexity scaling for root vs non-root nodes
  - Statistics tracking (total samples, average cap, speedup estimates)
  - Configurable min/max playouts with geometric_p parameter

- **PolicyTargetPruner** (`policy_pruning.py`): Policy target pruning
  - Top-k preservation for important actions
  - Minimum probability threshold filtering
  - Maximum actions limiting
  - Entropy-based dynamic thresholding
  - Action reduction statistics tracking

- **CyclicLRScheduler** (`cyclic_lr.py`): Cyclic learning rate schedules
  - Five schedule types: triangular, triangular2, exp_range, cosine_annealing, warm_restarts
  - Momentum cycling with base_momentum and max_momentum
  - Warmup steps support
  - Iteration tracking and state serialization

- **AuxiliaryTargetGenerator** (`auxiliary_targets.py`): Auxiliary policy targets
  - Attention target generation from board positions
  - Opponent modeling for predicting opponent moves
  - Game phase prediction (early/mid/late classification)
  - Future value prediction for N-step lookahead
  - AuxiliaryLossCalculator for weighted loss computation

- **PerformanceOptimizationsConfig** (`config.py`): Unified configuration
  - Enable flags for each optimization type
  - Nested configs for all optimization parameters
  - fast_preset() and quality_preset() factory methods
  - YAML serialization and environment variable support

##### Optimizations Test Suite (148 new tests)

- `tests/unit/test_optimizations/test_playout_cap.py`: Playout cap tests (26 tests)
- `tests/unit/test_optimizations/test_policy_pruning.py`: Policy pruning tests (24 tests)
- `tests/unit/test_optimizations/test_cyclic_lr.py`: Cyclic LR tests (30 tests)
- `tests/unit/test_optimizations/test_auxiliary_targets.py`: Auxiliary targets tests (51 tests)
- `tests/unit/test_optimizations/test_config.py`: Config tests (17 tests)
- `tests/integration/test_optimizations_integration.py`: Integration tests (17 tests)

##### Optimizations Configuration (`configs/optimizations.yaml`)

- Complete YAML configuration template
- Playout cap settings with distribution parameters
- Policy pruning thresholds
- Cyclic LR schedule configuration
- Auxiliary target weights

#### Real Civ6 Bridge System

A complete bridge system enabling the AlphaZero agent to play actual Civilization VI through file-based IPC.

##### Python Bridge (`alpha_zero/bridge/`)

- **config.py**: `BridgeConfig` dataclass with validation
  - Communication directory configuration
  - Polling intervals, timeouts, retry settings
  - Map size configurations for coordinate scaling
  - Environment variable support (`CIV6_BRIDGE_*`)
- **protocol.py**: Message types and JSON schemas
  - `GameStateData`, `UnitData`, `CityData`, `TileData` for state representation
  - `LuaCommand`, `CommandMessage`, `AckMessage` for action communication
  - `LegalAction` for action masking
  - Serialization/deserialization with validation
- **file_bridge.py**: Async file-based IPC
  - `FileBridge`: Polls state.json, writes commands.json, reads ack.json
  - Heartbeat monitoring for connection health
  - Automatic retry with exponential backoff
  - `MockFileBridge` for testing without real files
- **state_converter.py**: State format transformation
  - Converts Civ6 Lua JSON to encoder-compatible dict
  - Coordinate scaling from Civ6 map to 16x16 board
  - Terrain, unit, city type mappings
  - Legal action preservation
- **action_converter.py**: Action index to Lua command
  - Maps ~483K hierarchical actions to `LuaCommand`
  - Action mask generation from legal actions
  - Validation with detailed error reporting

##### Gymnasium Environment (`alpha_zero/envs/real_civ6_env.py`)

- **RealCiv6Env**: Full Gymnasium-compliant environment
  - Dict observation space: spatial (84, 16, 16), vector (512,), action_mask
  - Discrete action space from hierarchical action encoding
  - Async bridge communication with configurable timeouts
- **MockCiv6Env**: Testing environment
  - Built-in mock state generation
  - No file I/O required
  - Compatible with all training code

##### Game Interface (`alpha_zero/interfaces/civ6_interface.py`)

- **Civ6Interface**: Implements `BaseGameInterface`
  - `connect()`, `disconnect()` for session management
  - `reset()`, `step()` for game control
  - `get_observation()`, `get_valid_actions()` for state access
  - `save_state()`, `load_state()` for checkpointing
  - Registered as "civ6" backend with `GameAdapter`

##### Civilization VI Lua Mod (`mod/AlphaZeroAI/`)

- **AlphaZeroAI.modinfo**: Mod metadata and configuration
  - File references and dependencies
  - Action criteria for enabling AI control
- **Config.lua**: Centralized configuration management
  - Communication directory, file names
  - Polling intervals, feature flags
  - Dynamic configuration overrides
- **Logger.lua**: Logging system
  - Log levels: DEBUG, INFO, WARN, ERROR
  - Console and file output
  - Scoped loggers per module
- **JsonEncoder.lua**: Pure Lua JSON encoder/decoder
  - No external dependencies (Civ6 Lua is sandboxed)
  - Handles all JSON types, escaping, unicode
- **FileIO.lua**: File operations
  - Read/write with error handling
  - Change tracking for efficient updates
  - Heartbeat, state, command, ack file management
- **StateExporter.lua**: Game state serialization
  - Map tiles with terrain, features, resources
  - Units with full stats and capabilities
  - Cities with production, buildings, population
  - Legal actions enumeration
- **ActionExecutor.lua**: Command execution
  - Unit actions: move, attack, fortify, sleep, etc.
  - City actions: production, purchase
  - Research: technology and civic selection
  - Validation and error reporting
- **Main.lua**: Entry point and orchestration
  - Event handlers for turn start/end
  - Command polling loop
  - Acknowledgment writing

##### Test Suite

- **tests/unit/test_bridge.py**: 93 unit tests for bridge components
- **tests/unit/test_state_converter.py**: State conversion tests
- **tests/unit/test_action_converter.py**: Action conversion tests
- **tests/integration/test_bridge_integration.py**: End-to-end pipeline tests
- **tests/integration/test_real_civ6_env.py**: Environment integration tests
- **tests/fixtures/game_states/**: Sample JSON fixtures
  - `turn_1_start.json`: Initial game state
  - `turn_10_mid.json`: Mid-game with units and cities
  - `turn_50_combat.json`: Combat scenario

#### Phase 4: Unified Training Orchestrator

A single, configurable entry point for full AlphaZero training with integrated self-play, MCTS, neural networks, reward shaping, curriculum learning, and comprehensive logging.

##### Core Orchestrator Module (`alpha_zero/orchestrator/`)

- **OrchestratorConfig** (`config.py`): Unified configuration system
  - YAML-based configuration with environment variable overrides (`CIV6AZ_` prefix)
  - Preset loading: `fast`, `balanced`, `production`
  - Full validation with warnings for invalid combinations
  - Nested configs for environment, network, MCTS, training, rewards, curriculum

- **TrainingOrchestrator** (`orchestrator.py`): Main training loop controller
  - Single entry point replacing multiple training scripts
  - Integrated self-play with MCTS-guided game generation
  - Automatic curriculum stage progression
  - Periodic evaluation and checkpointing
  - Graceful shutdown with signal handlers (SIGINT, SIGTERM)
  - Resume from checkpoint support

- **HyperparameterScheduler** (`scheduler.py`): Dynamic hyperparameter control
  - Six schedule types: constant, linear, cosine, step, warmup_cosine, exponential
  - Learning rate, batch size, temperature, and exploration scheduling
  - Step-based progression with configurable decay

- **CheckpointManager** (`checkpointer.py`): Model persistence
  - Automatic checkpoint rotation (configurable retention)
  - Best model tracking by win rate
  - Full training state save/resume
  - JSON metadata for checkpoint history

- **PaddedEnvironmentWrapper**: Curriculum board size handling
  - Pads observations for smaller curriculum boards to network's expected size
  - Consistent interface across all board sizes
  - Proper cloning support for MCTS simulations

##### Component Factories (`alpha_zero/utils/factories.py`)

- `create_network()`: Neural network instantiation with board/action size config
- `create_mcts()`: MCTS with optional SimpleMCTS fallback
- `create_environment()`: Environment creation with override support
- `create_reward_shaper()`: Reward shaping initialization
- `create_curriculum_manager()`: Curriculum system setup
- `create_replay_buffer()`: Experience replay initialization
- `create_wandb_logger()`: W&B logging setup

##### CLI Entry Point (`alpha_zero/orchestrator/__main__.py`)

- `python -m alpha_zero.orchestrator` command
- Preset selection: `--preset fast|balanced|production`
- Custom config: `--config path/to/config.yaml`
- Dry run mode: `--dry_run`
- Resume support: `--resume`

##### Configuration Files

- `configs/orchestrator.yaml`: Master configuration template
- `configs/orchestrator_fast.yaml`: Fast testing preset (6x6 board, 10 iterations)
- `configs/orchestrator_production.yaml`: Production training settings

#### RAG (Retrieval-Augmented Generation) Integration

A comprehensive RAG system for enhancing agent decision-making with external knowledge retrieval.

##### Configuration Module (`alpha_zero/rag/config/settings.py`)

- **VectorStoreConfig**: Vector database configuration
  - Support for Memory, ChromaDB, FAISS, and Pinecone backends
  - Configurable dimensions, distance metrics, and collection names
- **EmbeddingConfig**: Embedding model configuration
  - Support for OpenAI, Sentence Transformers, and Local embeddings
  - Batch size and dimension configuration
- **ProcessorConfig**: Text processing configuration
  - Chunk size and overlap settings
  - Recursive text splitting parameters
- **RAGConfig**: Unified configuration with YAML serialization
  - Environment variable loading
  - Configuration validation

##### Vector Store Abstraction (`alpha_zero/rag/stores/`)

- **VectorStore Protocol**: Abstract interface for vector databases
  - Async CRUD operations (add, search, delete, get)
  - Metadata filtering support
- **MemoryVectorStore**: In-memory implementation for testing
  - Cosine similarity and Euclidean distance metrics
  - NumPy-based efficient search
- **ChromaStore**: ChromaDB integration
  - Persistent and ephemeral collection support
- **FAISSStore**: FAISS integration
  - High-performance similarity search
- **VectorStoreFactory**: Factory pattern for store creation

##### Document Processing (`alpha_zero/rag/processors/`)

- **RecursiveTextSplitter**: Intelligent text chunking
  - Configurable separators and chunk sizes
  - Overlap for context preservation
- **TextDocumentProcessor**: Document processing pipeline
  - Async file and text processing
  - Chunk metadata tracking
- **EmbeddingGenerator Protocol**: Abstract embedding interface
- **LocalEmbeddings**: Random embeddings for testing
- **OpenAIEmbeddings**: OpenAI API integration
- **SentenceTransformerEmbeddings**: Local transformer models

##### Agent Workflow Integration (`alpha_zero/rag/agents/`)

- **RAGAgentWorkflow**: Complete workflow orchestration
  - Document ingestion and indexing
  - Query processing with context retrieval
  - Hook system for extensibility
  - Step-by-step execution tracking
- **ContextBuilder**: Context assembly strategies
  - Relevance-based, recency-based, diversity-based, and hybrid strategies
  - Deduplication and character limits
- **QueryEnhancer**: Query preprocessing
  - Keyword extraction
  - Query expansion with synonyms
  - Query normalization

##### Utilities (`alpha_zero/rag/utils/`)

- **RAGLogger**: Structured logging with correlation IDs
  - Multiple output formats (JSON, text)
  - Configurable log levels
- **MetricsCollector**: Performance metrics
  - Timer, Counter, and Gauge types
  - Context manager support for timing
- **TTLCache & LRUCache**: Caching implementations
  - Time-to-live expiration
  - Size-based eviction

##### Main Pipeline (`alpha_zero/rag/pipeline.py`)

- **RAGPipeline**: Unified interface for RAG operations
  - Text and file ingestion
  - Query with context enhancement
  - Caching and metrics collection
  - Factory methods for testing configuration
  - Async context manager support

##### RAG Test Suite (100 new tests)

- `tests/unit/test_rag/test_config.py`: Configuration tests (20 tests)
- `tests/unit/test_rag/test_vector_stores.py`: Vector store tests (22 tests)
- `tests/unit/test_rag/test_processors.py`: Processor tests (25 tests)
- `tests/unit/test_rag/test_pipeline.py`: Pipeline tests (19 tests)
- `tests/integration/test_rag/test_rag_workflow.py`: Integration tests (14 tests)

##### RAG Dependencies

- `chromadb>=0.4.0`: ChromaDB vector database
- `faiss-cpu>=1.7.0`: Facebook AI Similarity Search
- `openai>=1.0.0`: OpenAI API client
- `sentence-transformers>=2.2.0`: Local embedding models
- `pinecone-client>=2.0.0` (optional): Pinecone cloud vector database

### Fixed

#### Action Mask Indexing Bug (`alpha_zero/core/civ6_actions.py`)

- Fixed critical bug where action mask indices didn't match action execution
- Changed from using global unit/city indices to player-relative indices
- Ensures valid actions map correctly to game state modifications
- Reduced average game length from 700+ steps to 70-90 steps

#### Evaluation Environment Handling

- Fixed `'MicroCivEnv' object has no attribute '_env'` error during evaluation
- Always create fresh wrapped environment for each game
- Consistent interface between self-play and evaluation

#### Value Tensor Shape Mismatch

- Fixed shape mismatch in training loss computation
- Added `value.unsqueeze(1)` for proper [batch_size, 1] shape

#### Curriculum Manager API

- Fixed `record_game(win=...)` to `record_game(won=...)` parameter name

### Changed

#### Phase 3: Enhanced Training Pipeline

A comprehensive suite of training improvements designed to accelerate learning convergence and improve agent performance.

##### Curriculum Learning System (`alpha_zero/utils/curriculum_learning.py`)

- **CurriculumStage**: Dataclass defining difficulty stages with configurable parameters
  - Board size, max turns, opponent strength, starting resources
  - Win rate thresholds for automatic promotion/demotion
  - Minimum games requirement before stage transitions
- **CurriculumManager**: Orchestrates multi-stage progression
  - Three default stages: Beginner (6x6), Intermediate (8x8), Advanced (10x10)
  - Automatic stage promotion when win rate exceeds threshold
  - Automatic demotion when win rate drops below minimum threshold
  - Stage history tracking with detailed metrics
- **AdaptiveDifficultyController**: Fine-grained difficulty adjustments within stages
  - Dynamic opponent strength scaling
  - Configurable adjustment rates and bounds

##### Potential-Based Reward Shaping (`alpha_zero/utils/enhanced_rewards.py`)

- **WinConditionEvaluator**: Multi-factor game state evaluation
  - Score advantage assessment with configurable weights
  - City count and production evaluation
  - Technology progress tracking
  - Unit strength differential calculation
- **EnhancedRewardShaper**: Policy-preserving reward augmentation
  - Potential-based shaping (guarantees optimal policy preservation)
  - Configurable gamma discount factor
  - Composite potential from multiple game state factors
  - Smooth reward signals for faster credit assignment

##### Environment Tuning (`alpha_zero/utils/environment_tuning.py`)

- **EarlyTerminationConfig**: Configurable early game termination
  - Decisive advantage detection
  - Stalemate prevention
  - Turn limit enforcement with graceful handling
- **ProductionConfig**: Accelerated production mechanics
  - Production multipliers for faster game progression
  - Growth rate adjustments
  - Research speed modifications
- **TunedEnvironmentWrapper**: Gymnasium-compliant wrapper
  - Seamless integration with existing training loops
  - Configurable through dataclass parameters
  - Observation and action space preservation

##### Training Configuration (`alpha_zero/utils/training_config_enhanced.py`)

- **ParameterSchedule**: Six schedule types for hyperparameter annealing
  - Constant: Fixed value throughout training
  - Step: Discrete value changes at specified iterations
  - Linear: Smooth linear interpolation
  - Exponential: Exponential decay/growth
  - Cosine: Cosine annealing with smooth transitions
  - Warmup-Cosine: Linear warmup followed by cosine decay
- **ResourceManager**: System resource management
  - Automatic device selection (CUDA/MPS/CPU)
  - Memory-aware batch sizing
  - Worker count optimization

##### Weights & Biases Integration (`alpha_zero/utils/wandb_logger.py`)

- **WandbConfig**: Typed configuration for W&B initialization
  - Project, entity, run name, and tags configuration
  - Mode selection (online/offline/disabled)
  - Artifact logging configuration
- **WandbLogger**: Comprehensive experiment tracking
  - `init()`: Initialize W&B run with full config
  - `log()`: Generic metric logging with step tracking
  - `log_selfplay()`: Self-play statistics (games, buffer size, samples)
  - `log_training_step()`: Training metrics (loss, policy/value loss, LR)
  - `log_evaluation()`: Evaluation results (win rate, game stats)
  - `log_model()`: Model artifact versioning with metadata
  - `summary()`: Final run summary statistics
  - `finish()`: Graceful run completion

##### Enhanced Training Orchestration (`run_enhanced_training.py`)

- **Training Presets**: Pre-configured training profiles
  - Fast: Quick iteration for development (2 iterations, 2 games)
  - Balanced: Standard training (10 iterations, 20 games)
  - Full: Production training (50 iterations, 100 games)
- **Command-Line Interface**: Full argument parsing
  - All training parameters configurable via CLI
  - W&B integration flags (`--wandb`, `--wandb-project`, `--wandb-tags`)
  - Device selection and checkpoint management
- **History Tracking**: Comprehensive training history
  - Iteration-level metrics
  - Loss tracking with component breakdown
  - Curriculum stage progression
  - Final summary statistics

##### Simplified MCTS (`alpha_zero/utils/simple_mcts.py`)

- Lightweight MCTS implementation for evaluation
- Configurable simulation count
- Temperature-based action selection
- Compatible with enhanced training pipeline

#### New Run Scripts

- `run_enhanced_training.py`: Main training orchestration with all Phase 3 features
- `run_training_wandb.py`: W&B-focused training runner
- `run_civ6_ai.py`: Simple AI evaluation runner

#### Test Suite Expansion

##### Unit Tests (14 new files)

- `tests/unit/test_curriculum_learning.py`: Curriculum stage and manager tests
- `tests/unit/test_enhanced_rewards.py`: Reward shaping validation
- `tests/unit/test_environment_tuning.py`: Environment wrapper tests
- `tests/unit/test_training_config_enhanced.py`: Parameter schedule tests
- `tests/unit/test_wandb_logger.py`: W&B logger unit tests
- `tests/unit/test_enhanced_training.py`: Training function signature tests
- `tests/unit/test_enhanced_training_wandb.py`: W&B integration unit tests
- `tests/unit/test_simple_mcts.py`: Simplified MCTS tests
- `tests/unit/test_evaluate.py`: Evaluation function tests
- `tests/unit/test_self_play.py`: Self-play generation tests
- `tests/unit/test_training_loop.py`: Training loop tests
- `tests/unit/test_replay_buffer.py`: Replay buffer tests
- `tests/unit/test_integration_components.py`: Component integration tests
- `tests/unit/test_performance.py`: Performance benchmark tests

##### Integration Tests (6 new files)

- `tests/integration/test_enhanced_training.py`: End-to-end training tests
- `tests/integration/test_enhanced_training_wandb.py`: W&B integration tests
- `tests/integration/test_training_pipeline.py`: Full pipeline tests
- `tests/integration/test_curriculum_integration.py`: Curriculum system integration
- `tests/integration/test_reward_shaping_integration.py`: Reward shaping validation
- `tests/integration/test_full_training_cycle.py`: Complete training cycle tests

#### Documentation

- `PLAN_IMPROVE_LEARNING.md`: Implementation plan for Phase 3 improvements
- `TRAINING_IMPROVEMENTS_TEMPLATE.md`: Template for training experiments
- `.env.template`: Environment variable template for W&B configuration
- Updated `CLAUDE.md`: Comprehensive project documentation refresh

#### Modified Files

- **`.gitignore`**: Added W&B directories, training artifacts, profiling files
- **`alpha_zero/utils/__init__.py`**: Exported all new Phase 3 modules
  - CurriculumStage, CurriculumManager, AdaptiveDifficultyController
  - EnhancedRewardShaper, WinConditionEvaluator
  - TunedEnvironmentWrapper, EarlyTerminationConfig, ProductionConfig
  - ParameterSchedule, ResourceManager
  - WandbLogger, WandbConfig, WANDB_AVAILABLE flag
- **`requirements.txt`**: Added new dependencies
  - `wandb>=0.15.0`: Experiment tracking
  - `gymnasium>=0.29.0`: Environment compatibility
  - `tqdm>=4.65.0`: Progress bars
  - Additional testing dependencies

### Technical Highlights

#### Design Patterns

- **Dataclass Configuration**: All configuration uses typed dataclasses for IDE support
- **Factory Pattern**: ParameterSchedule.create() for schedule instantiation
- **Wrapper Pattern**: TunedEnvironmentWrapper for non-invasive environment modification
- **Strategy Pattern**: Multiple schedule types implementing common interface
- **Optional Dependencies**: W&B integration with graceful fallback when unavailable

#### Performance Considerations

- Potential-based reward shaping preserves optimal policy (mathematically proven)
- Curriculum learning reduces sample complexity through staged difficulty
- Early termination prevents wasted computation on decided games
- Batch-aware resource management for memory efficiency

#### Testing Philosophy

- 400+ test methods across 33 test files
- Unit tests validate individual components in isolation
- Integration tests verify component interactions
- Mock-based W&B tests avoid API calls during CI
- Parametrized tests cover edge cases and boundary conditions

## [Previous Releases]

For changes prior to the Phase 3 Enhanced Training Pipeline, please refer to the git history.

---

*This changelog documents the feature/improve-learning-pipeline branch changes.*
