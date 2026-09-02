# CLAUDE.md - Civ6 AlphaZero Agentic Coding System

> **Design Principle:** This document treats prompting as constraint programming, not instruction writing. It defines the feasible region, objective function, and search parameters for autonomous agent implementation.

---

## SECTION 1: OBJECTIVE FUNCTION

### 1.1 System Intent

```
I am building: An AlphaZero-style reinforcement learning AI for Civilization 6 using self-play,
Monte Carlo Tree Search, and deep neural networks.
```

### 1.2 Success Criteria (Mechanically Verifiable)

```
This succeeds when:
- [ ] All tests pass: `pytest tests/ -v` exits with code 0
- [ ] Coverage >= 85%: `pytest --cov=alpha_zero --cov-report=term-missing` shows >= 85% (gate enforced in CI and `pyproject.toml`; re-tightened 80→85 on 2026-07-13 at 89.69% measured; some infra/web modules are intentionally omitted — see `[tool.coverage]`)
- [ ] Type checking passes: `mypy alpha_zero/` exits with code 0
- [ ] Linting passes: `ruff check alpha_zero/` exits with code 0
- [ ] Training smoke test: `python -m alpha_zero.training_civ6 --dry_run` completes
- [ ] Distributed smoke test: `python -m alpha_zero.distributed.coordinator --dry_run` completes
- [ ] Orchestrator smoke test: `python -m alpha_zero.orchestrator --preset fast --dry_run` completes
- [ ] LLM integration test: `python -c "from alpha_zero.utils import create_llm_client"` imports
- [ ] RAG integration test: `python -c "from alpha_zero.orchestrator import RAGIntegration"` imports
```

### 1.3 Problem Description

**Paragraph 1 - Core Problem:**
This system implements AlphaZero for Civilization 6, a complex 4X strategy game with astronomical
state and action spaces (~10^200 states, ~10^6 actions per turn). The AI learns through self-play,
using a neural network to evaluate game states and guide Monte Carlo Tree Search. The coordination
logic involves: game simulation (MicroCiv/Unciv), neural network training, MCTS search, and
distributed self-play across multiple actors.

**Paragraph 2 - Data Flows:**
Data flows from game environments (MicroCiv for training, Unciv for production) through state
encoders (13-channel MicroCiv, 84-channel full Civ6) to the neural network. The network outputs
policy distributions and value estimates. MCTS uses these to build search trees. Training samples
(state, policy, value) accumulate in replay buffers. The learner samples batches for gradient
updates. State synchronized across components: network weights (SharedStorage), replay buffer
(DistributedReplayBuffer), training metrics (MetricsTracker).

**Paragraph 3 - Failure Modes:**
Critical invariants: (1) replay buffer never exceeds memory limits, (2) network weights are
consistent across all actors, (3) MCTS visits are properly backpropagated, (4) action masking
prevents invalid actions, (5) distributed training maintains ordering guarantees. Failure modes
include: OOM from buffer overflow, stale weights causing training divergence, MCTS selection errors
from numerical instability, and race conditions in distributed components.

---

## SECTION 2: FEASIBLE REGION (Constraints)

### 2.1 Hard Constraints (Violations = Failure)

```
- Language/Runtime: Python 3.10+ (type hints required)
- Required Dependencies:
  - torch >= 1.13.0 (neural networks)
  - numpy >= 1.23.0 (numerical operations)
  - gymnasium >= 0.29.0 (RL environments)
  - pyyaml >= 6.0 (configuration)
  - pytest >= 7.0.0 (testing)
- Security:
  - NO hardcoded secrets (use .env or environment variables)
  - API keys loaded from environment only
  - All user inputs validated before processing
- Compatibility:
  - Must run on CPU (GPU optional)
  - Cross-platform (Windows, Linux, macOS)
  - Python multiprocessing for distributed training
```

### 2.2 Soft Constraints (Preferences)

```
- Style: Ruff linting, 100 character line limit
- Architecture:
  - Prefer composition over inheritance
  - Dataclasses for configuration objects
  - Async I/O for LLM calls
  - Type hints on all public interfaces
- Performance:
  - Batch operations where possible
  - Minimize memory allocations in hot paths
  - Use torch.no_grad() for inference
- Testing:
  - pytest with fixtures
  - Unit tests for all public methods
  - Integration tests for cross-module flows
  - Coverage >= 85% on core modules
- Logging:
  - Use alpha_zero.utils.logging module
  - Structured logging with MetricsTracker
  - Debug-level tracing for development
```

### 2.3 Anti-Constraints (Explicit Freedoms)

```
You ARE permitted to:
- Restructure existing file organization for better modularity
- Add dependencies from PyPI (update requirements.txt)
- Refactor adjacent code for consistency
- Choose implementation patterns not specified
- Create new test files and fixtures
- Add new configuration parameters to configs/civ6_config.yaml
- Implement helper functions and utilities
- Add logging statements for debugging
- Create new submodules within alpha_zero/
```

---

## SECTION 3: PERMISSION ARCHITECTURE

### 3.1 Scope (What You Can Touch)

```
IN SCOPE:
- All files in alpha_zero/
- All files in tests/
- Configuration files in configs/
- Root-level Python files (pyproject.toml, run_*.py)
- Documentation files (CLAUDE.md, README.md)
- Environment templates (.env.template)
- Requirements files (requirements.txt)

OUT OF SCOPE:
- .git/ directory
- __pycache__/ directories
- .env file (contains secrets - read only)
- checkpoints/ (trained model files)
- External dependencies in site-packages
```

### 3.2 Autonomy Level

```
AUTONOMOUS (proceed without asking):
- File creation/deletion within scope
- Dependency installation (pip install)
- Running tests (pytest)
- Code refactoring within existing patterns
- Adding logging and debug statements
- Creating new test files
- Updating configuration defaults
- Documentation updates

CONFIRM FIRST (ask before proceeding):
- Architectural changes affecting >3 modules
- Breaking API changes to public interfaces
- Deletions of >100 lines of code
- Changes to distributed training coordination
- Modifications to neural network architecture
- New external service integrations

PROHIBITED (do not attempt):
- Committing directly to main branch
- Modifying .env with real credentials
- External API calls with side effects (except LLM)
- Deleting checkpoint files
- Running training with >1000 games without confirmation
```

### 3.3 Resource Budget

```
- Max iterations before requesting guidance: 5
- Max files to modify in single pass: 20
- Max lines in single file edit: 500
- Time-boxed exploration: ≤10 min research before asking
- Test execution timeout: 300 seconds
```

---

## SECTION 4: FEEDBACK LOOP SPECIFICATION

### 4.1 Verification Commands

```bash
# After writing code, run in this order:

# 1. Lint check (whole repo — CI gate scope as of 2026-07-25)
ruff check .

# 2. Type check
mypy alpha_zero/  # exits 0 as of 2026-07-25; flag-free — config lives in pyproject

# 3. Unit tests (fast feedback)
pytest tests/unit/ -v --tb=short

# 4. Integration tests
pytest tests/integration/ -v --tb=short

# 5. Full test suite with coverage
pytest tests/ -v --cov=alpha_zero --cov-report=term-missing

# 6. Smoke tests
python -m alpha_zero.training_civ6 --dry_run
python -m alpha_zero.distributed.coordinator --dry_run
```

### 4.2 Error Handling Protocol

```
ON LINT FAILURE:
→ Read error output
→ Fix automatically using ruff guidance
→ Re-run: ruff check alpha_zero/

ON TYPE ERROR:
→ Analyze mypy error message
→ Add/fix type annotations
→ Consider if Optional or Union needed
→ Re-run: mypy alpha_zero/

ON TEST FAILURE:
→ Read full traceback
→ Identify root cause (implementation vs test bug)
→ If implementation bug: fix source code
→ If test bug: fix test (with justification)
→ Re-run specific test: pytest tests/path/to/test.py::test_name -v

ON IMPORT ERROR:
→ Check if dependency is in requirements.txt
→ If missing: add and run pip install -r requirements.txt
→ If version conflict: resolve in requirements.txt

ON REPEATED FAILURE (same error 3x):
→ Stop attempting fix
→ Document what was tried
→ Report analysis with hypothesis
→ Request human guidance
```

### 4.3 Success Verification

```
Before reporting completion:
1. All verification commands pass (lint, type, tests)
2. Coverage check: pytest --cov=alpha_zero shows >= 85% on changed modules
3. Smoke test: python -m alpha_zero.training_civ6 --dry_run
4. Import check: python -c "from alpha_zero import <new_module>"
5. Generate brief summary of changes made
6. Update CLAUDE.md if new commands/patterns discovered
```

---

## SECTION 5: CONTEXT PERSISTENCE

### 5.1 Session Memory Updates

When making changes, update the relevant sections below:

#### Build Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/unit/ -v` | Run unit tests |
| `pytest tests/integration/ -v` | Run integration tests |
| `pytest tests/ -v --tb=short` | Run all tests |
| `pytest tests/ -v --cov=alpha_zero` | Run tests with coverage |
| `ruff check alpha_zero/` | Lint Python code |
| `mypy alpha_zero/` | Type check |
| `make verify` | **Reproduce every CI gate locally** (lint, format, types, CI's four suites, security incl. bandit, governance, validators, coverage, LLM eval gate) — use this instead of hand-assembling the sequence |
| `OMP_NUM_THREADS=2 pytest tests/unit -q -m "not slow"` | **Run the suite in this sandbox.** Without the thread cap, two tests wedge in `torch.conv2d` on the OpenMP spin-wait (§5.3) and hit the 120 s timeout. With it: 9,921 passed in 3m39s. Applies to any local suite run touching torch. |
| `make test-all` | Coverage, mirroring CI's coverage job (gate: 85%) |
| `make typecheck` | Type check as CI runs it (`python -m mypy alpha_zero/`) |
| `make typecheck-strict` | Harsher local `--strict` pass — **not** a CI gate |
| `make validate` | Run governance validators (openspec, kb, skills, doc-sync) |
| `make hooks-install` | Install pre-commit (self-installing) + git pre-commit/pre-push hooks |
| `python scripts/check_doc_sync.py` | Documentation-drift gate (version/counts/links/extras) |
| `python scripts/validate_openspec.py` | Validate the `openspec/` change directory |
| `python -m alpha_zero.training_civ6 --dry_run` | Training smoke test |
| `python -m alpha_zero.training_civ6 --seed 42` | Reproducible training run — two runs at one seed are identical |
| `python scripts/run_baseline_eval.py --seed 42 ...` | Reproducible evaluation probe (see `docs/evidence/seeded_probe_reproducibility/`) |
| `pytest tests/regression/test_training_determinism_regression.py -v` | Determinism gate (~7 s) |
| `python -m alpha_zero.distributed.coordinator --dry_run` | Distributed smoke test |
| `python -m alpha_zero.orchestrator --help` | Show orchestrator options |
| `python -m alpha_zero.orchestrator --preset fast --dry_run` | Fast preset smoke test |
| `python -m alpha_zero.orchestrator --config configs/orchestrator.yaml --dry_run` | Config file smoke test |
| `pip install -e '.[eval]'` | Install the LLM eval-harness (langfuse-eval-harness from git) |
| `make eval-llm` | Run the StrategyAdvisor LLM eval gate (offline, exit-code propagating) |
| `python scripts/run_llm_eval.py` | Run the LLM eval gate directly (both analyze + report configs) |
| `python scripts/run_llm_eval.py --online` | Run the eval gate against a real LLM provider (needs API keys) |
| `eval-harness list-plugins` | List harness scorers/datasets/targets/sinks/judges |
| `python scripts/launch_50iter_production_monitored.py --dry-run` | MLflow-monitored 50-iter production dry run |
| `python scripts/launch_50iter_production_monitored.py` | Launch full 50-iter production run with MLflow |
| `mlflow ui --backend-store-uri ./mlruns --port 5000` | Inspect MLflow runs in the browser |
| `python -m alpha_zero.civ6_local --mode civ6 --dry-run` | Validate Civ6 bridge config (no game needed) |
| `python -m alpha_zero.civ6_local --mode civ6 --mcts` | Start Civ6 bridge with GumbelMCTS policy |
| `python -m alpha_zero.civ6_local --mode civ6 --mcts --eval-harness` | Bridge + MCTS + MLflow eval harness |
| `python scripts/install_civ6_mod.py` | Install AlphaZeroAI mod to default Steam Mods dir |
| `python scripts/install_civ6_mod.py --mod-dir PATH` | Install mod to custom Civ6 Mods directory |
| `pip install -e '.[tuning]'` | Install Optuna for hyperparameter sweeps |
| `python run_sweep.py --dry-run` | Validate sweep config (configs/sweep_muzero.yaml) |
| `python run_sweep.py --n-trials 20` | Run Optuna sweep (MLflow-logged; CIV6AZ_SWEEP_* env overrides) |
| `pip install -e '.[hub]'` | Install Hugging Face Hub support for the encoder |
| `python scripts/run_encoder_ab.py train` | Train both encoder arms locally (device auto-detected, seeded) |
| `python scripts/run_encoder_ab.py arena --num_games 800` | Head-to-head arena; writes counts + Wilson 95% CI |
| `python scripts/run_encoder_ab.py report` | Render committable A/B evidence to `docs/evidence/hexcrystal_ab/` |
| `python scripts/compare_training_runs.py --mlflow_run_a A --mlflow_run_b B` | Compare two MLflow runs (add `--no_plots` without matplotlib) |
| `python scripts/upload_encoder_to_hub.py --checkpoint P --repo-id R` | Stage the encoder for the Hub (dry run; `--publish` uploads) |

#### Cloud Deployment Commands

| Command | Purpose |
|---------|---------|
| `make build-all` | Build the deployed image (`dockerfiles/Dockerfile.vertex`); the per-role base/selfplay/trainer/evaluator images, `docker-compose.yml` and `make local-test` were removed 2026-09-02 as dead |
| `make push` | Push it to Artifact Registry |
| `make local-smoke` | Dry-run the deployed self-play worker (`alpha_zero.cloud.gcp.selfplay_worker --dry_run`) |
| `python -m alpha_zero.cloud.gcp.vertex_orchestrator --help` | **The deployed orchestrator.** `cloudbuild.yaml` builds only `Dockerfile.vertex`, which entrypoints `alpha_zero.cloud.gcp.entrypoint` |
| `python -m infra.vertex_launcher --mode=selfplay --iteration=0 --dry-run` | Self-play job dry run — **legacy, not deployed**: resolves `civ6-selfplay:latest` and friends, images no build pipeline produces |
| `python -m infra.vertex_launcher --mode=train --iteration=0 --dry-run` | Training job dry run — legacy, not deployed |
| `python -m infra.vertex_launcher --mode=eval --iteration=0 --dry-run` | Evaluation job dry run — legacy, not deployed |
| `python -m alpha_zero.cloud.gcp.selfplay_worker --help` | Deployed self-play worker options (the LEGACY `alpha_zero.cloud.{selfplay_worker,trainer,evaluator}` modules were deleted 2026-09-02) |
| `python -m alpha_zero.cloud.gcp.training_worker --help` | Deployed trainer options |
| `python -m alpha_zero.cloud.gcp.arena --help` | Deployed arena/evaluator options |

#### Test Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/unit/test_environment.py -v` | Test MicroCiv environment |
| `pytest tests/unit/test_network.py -v` | Test neural networks |
| `pytest tests/unit/test_mcts.py -v` | Test MCTS algorithm |
| `pytest tests/unit/test_config.py -v` | Test configuration system |
| `pytest tests/unit/test_logging.py -v` | Test logging infrastructure |
| `pytest tests/unit/test_orchestrator_config.py -v` | Test orchestrator configuration |
| `pytest tests/unit/test_scheduler.py -v` | Test hyperparameter scheduler |
| `pytest tests/unit/test_checkpointer.py -v` | Test checkpoint manager |
| `pytest tests/unit/test_factories.py -v` | Test component factories |
| `pytest tests/unit/test_rag_integration.py -v` | Test RAG integration |
| `pytest tests/unit/test_strategy_advisor.py -v` | Test strategy advisor |
| `pytest tests/unit/test_strategy_advisor_contract.py -v` | StrategyAdvisor output contract gate (dependency-free) |
| `pytest tests/unit/test_llm_harness_target.py -v` | Eval-harness callable target adapter tests |
| `pytest tests/unit/test_storage_backend.py -v` | Test storage backends |
| `pytest tests/unit/test_orchestrator_stagnation.py -v` | Test stagnation config + orchestrator wiring |
| `pytest tests/regression/test_debt_ratchet.py -v` | Tech-debt ratchet — every tracked metric vs the committed baseline |
| `python scripts/audit_import_graph.py` | List `alpha_zero/` modules with no live importer (AST import graph) |
| `python scripts/measure_debt.py --write PATH` | Re-measure every tracked debt metric as JSON |
| `pytest tests/unit/test_replay_buffer.py -v` | Test replay buffers |
| `pytest tests/unit/test_nplayer_training_pipeline.py -v` | Test N-player training pipeline (G9: per-player values) |
| `pytest tests/unit/test_simultaneous_mcts.py -v` | Test DUCT algorithm |
| `pytest tests/unit/test_action_resolution.py -v` | Test action conflict resolution |
| `pytest tests/integration/test_e2e_nplayer_training.py -v` | E2E N-player training |
| `pytest tests/integration/test_e2e_pipeline.py -v` | E2E pipeline validation |
| `pytest tests/integration/test_e2e_simultaneous.py -v` | E2E DUCT simultaneous tests |
| `pytest tests/unit/test_monitoring/ tests/unit/test_utils/ -v` | Test monitoring + utils |
| `pytest tests/scripts/ -v` | Test eval scripts |
| `pytest -k "test_name" -v` | Run specific test by name |
| `python scripts/migrate_mlflow_to_sqlite.py --help` | MLflow migration tool |
| `mlflow ui --backend-store-uri sqlite:///mlflow.db` | Launch MLflow UI |
| `python scripts/analyse_training_run.py --help` | Training run analysis |
| `python scripts/summarize_training_log.py RUN.log --out DIR` | Turn a training log into committable evidence (JSON + report) |
| `python scripts/post_run_evaluation.py --help` | Post-run ELO + baseline eval |
| `python scripts/run_baseline_eval.py --help` | Baseline evaluation (random/heuristic) |

### 5.2 Architecture Decisions

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-01-10 | 13-channel MicroCiv encoder | Simplified encoding for faster Phase 1 training |
| 2026-01-10 | Hierarchical action space | Reduces effective branching factor from ~10^6 to ~10^3 |
| 2026-01-10 | SimpleMCTS + Civ6MCTS | SimpleMCTS for testing, Civ6MCTS for training |
| 2026-01-10 | ResNet backbone | Proven architecture from AlphaZero, good spatial reasoning |
| 2026-01-11 | Multi-head attention in network | Better spatial reasoning for complex game states |
| 2026-01-11 | Process-based distributed training | Avoid GIL, maximize CPU utilization for self-play |
| 2026-01-11 | Unciv JSON interface | Open-source Civ V clone with accessible save format |
| 2026-01-11 | YAML config with env overrides | Flexible configuration for different environments |
| 2026-01-19 | LLM client with provider fallback | Unified API for OpenAI/Anthropic with auto-failover |
| 2026-01-19 | Test runner utility | Programmatic test execution with coverage tracking |
| 2026-01-19 | Agent tools module | Reusable utilities for Claude Code sub-agents |
| 2026-01-20 | Unified Training Orchestrator | Single entry point for all training modes with dynamic config |
| 2026-01-20 | RAG-LLM integration layer | ChromaDB knowledge base with strategy advisor for training insights |
| 2026-01-20 | Hyperparameter scheduling | Multiple schedule types (warmup_cosine, step, exponential) |
| 2026-01-20 | Factory pattern for components | Testable, flexible component creation |
| 2026-02-01 | Abstract StorageBackend interface | Unified storage API for local and GCS backends |
| 2026-02-01 | GCS-backed replay buffer | Streaming IterableDataset for memory-efficient cloud training |
| 2026-02-01 | Cloud-native worker components | Self-play, trainer, evaluator containerized for Vertex AI |
| 2026-02-01 | Preemption handling | Graceful shutdown on Spot VM preemption with trajectory flush |
| 2026-02-01 | Budget machine profiles | Pre-configured profiles for T4/L4/A100 GPUs with cost estimates |
| 2026-02-28 | DUCT for simultaneous actions | Decoupled UCB per player, avoids combinatorial |A|^N explosion via lazy joint children |
| 2026-02-28 | ActionResolver with strategies | Configurable conflict resolution (independent, priority, random, simultaneous) |
| 2026-02-28 | SimultaneousMCTSConfig frozen | All DUCT params via frozen dataclass with CIV6AZ_ env overrides |
| 2026-02-17 | N-player MCTS (Max-N, Paranoid, Best-Reply) | Multiplayer support for 2-8 players, per-player value vectors |
| 2026-02-17 | Opponent modeling | Frequency-based prediction with exponential decay, Bayesian blending |
| 2026-02-17 | Turn manager (Sequential/Simultaneous/Dynamic) | Civ6-style dynamic turn modes |
| 2026-02-18 | Hex crystalline encoder | Hex-topology-aware spatial encoder: **even-q offset** 7-tap gather convolution (centre + 6 neighbours) → AdaptiveAvgPool token bottleneck → linear tokenization → optional crystalline lattice layer combining learned symmetry transforms across the *transform* axis. Corrected 2026-08-09: there are **no** axial coordinates, **no** ring-distance PE, and no spatial attention — position comes from the transformer's own `pos_embed`. |
| 2026-08-09 | Hex encoder extracted to `alpha_zero/hexcrystal/` | Torch-only subpackage (no numpy/gymnasium) so the encoder is reusable outside this repo and publishable to the HF Hub. Re-export shims keep every `alpha_zero.core.*` path working; `state_dict` keys unchanged (`encoder.patch_embed.*`). |
| 2026-08-09 | Root package uses PEP 562 lazy imports | `import alpha_zero` no longer pulls numpy/gymnasium (1,600 ms → 6.5 ms). `__all__` is now static and omits `RealCiv6Env`/`RewardConfig`, which remain resolvable attributes. |
| 2026-02-21 | GumbelMCTS as default | `DEFAULT_MCTS_TYPE` changed from `civ6` to `gumbel` |
| 2026-02-21 | Action mask in training pipeline | 5-tensor dataset: spatial, vector, mask, policy, value |
| 2026-02-23 | Training worker fine-tuning fix | `--model_version` loads incumbent from GCS before training |
| 2026-03-01 | Civ6 Lua mod bridge | File-based IPC in `alpha_zero/bridge/` (7 files) |
| 2026-03-01 | Web interface | FastAPI + WebSocket game UI in `alpha_zero/web/` (12 files) |
| 2026-03-01 | PBT module | Population-based training in `alpha_zero/pbt/` (10 files) |
| 2026-03-01 | Hierarchical planning | Goal decomposition in `alpha_zero/hierarchical/` (19 files) |
| 2026-05-21 | MLflow monitoring layer (alpha_zero/monitoring/) | Centralised experiment tracking with SQLite backend; file-store mlruns/ is deprecated. Combines PR #78 (orchestrator + tournament dashboards + stateless metric helpers) and PR #79 (log_evaluation_result + AlphaZeroMLflowEvaluator + analysis scripts). `MLflowConfig` is the single-source dataclass; `MonitoringConfig` retained as a backwards-compat alias. |
| 2026-05-21 | env_loader.py zero-dependency .env loader | No python-dotenv dep; CIV6AZ_ env var convention; graceful missing-key warnings |
| 2026-05-21 | EvaluationResult API: WinRateResult nesting | win_rate is WinRateResult object (0-100 scale); MLflow logs normalise to 0-1 |
| 2026-05-21 | log_evaluation_result() 14-metric logging | 4 counts + 3 normalised rates + 7 game stats per evaluation run |
| 2026-05-21 | TournamentMlflowLogger (PR #78) | Match / round / leaderboard / head-to-head facade for tournament telemetry |
| 2026-05-21 | Stateless monitoring helpers (PR #78) | `compute_value_correlation`, `compute_elo_spread`, `compute_policy_entropy` (NaN-safe, importable from `alpha_zero.monitoring`) |
| 2026-05-21 | W&B coexists with MLflow | Phased rollout; both backends can be enabled simultaneously. W&B remains primary in `cloud/`, `pbt/`, `evaluation/`, `optimizations/` pending follow-up sweep. Optional `[monitoring]` extra. |
| 2026-05-24 | BridgeMCTSEnv adapter (W2.2) | Snapshot-seeded `MicroCivEnv` adapter in `alpha_zero/bridge/mcts_env.py`. Seeds from `GameStateData` via `BridgeMCTSEnv.from_bridge_state()`. No `reset()` — each bridge state creates a fresh env; `clone()` pattern for MCTS rollouts. |
| 2026-05-24 | Civ6MctsPolicy (W2.2) | GumbelMCTS over `BridgeMCTSEnv`; implements `PolicyEngine` protocol. `--mcts` / `--no-mcts` CLI flags in `civ6_local.py`. Lazy import avoids torch overhead in Greedy path. |
| 2026-05-24 | BridgeEvalHarness (W2.3a) | MLflow-backed telemetry hooks (`on_start`, `on_turn_end`, `on_stop`) wired into `BridgeAgentLoop`. `--eval-harness` flag in `civ6_local.py`; env var `CIV6AZ_LOCAL_USE_EVAL_HARNESS=1`. |
| 2026-05-24 | install_civ6_mod.py (W2.3b) | `scripts/install_civ6_mod.py` copies `mod/AlphaZeroAI` → `%USERPROFILE%\Documents\My Games\Sid Meier's Civilization VI\Mods\AlphaZeroAI`. Creates dir if absent. Supports `--mod-dir` override. |
| 2026-05-24 | execute_with_retry wired (W3.1) | `GameBridgeHealthChecker.execute_with_retry` wraps both `wait_for_state` and `send_and_wait_with_retry` in `BridgeAgentLoop` when `health_checker` is supplied. `CircuitBreakerOpenError` maps to `state_timeouts` counter and asyncio.sleep(≥ retry_after). |
| 2026-05-30 (half superseded 2026-09-02) | Training-stagnation detection (harness engineering) | Reuses existing `EloPlateauDetector` in the orchestrator (`StagnationConfig`, opt-in, ELO/win-rate signal — NOT loss). Env: `CIV6AZ_STAGNATION_*`. Default OFF. **The `count_consecutive_non_promotions` half is gone:** it lived only in `cloud/evaluator.py`, which the import graph showed had no live importer (the deployed promotion path is `cloud/gcp/arena.py` + `evaluation/promotion.py`), so it was removed with Tier A on 2026-09-02 along with its `CIV6_STAGNATION_PATIENCE` env var and `tests/cloud/unit/test_evaluator_stagnation.py`. Promotion-stagnation detection on the deployed path is unimplemented — a real gap, not a relocation. |
| 2026-05-30 | PreToolUse prohibited-action guard | `.claude/hooks/pretooluse_guard.sh` enforces §3.2 (no commits/pushes to `main`, >1000-game cap) — turns prose into a machine deny. |
| 2026-05-30 | Pre-PR review in isolated subagent | Pre-PR handoff skill dispatches review to a fresh `Agent`/`code-review` for generator–evaluator separation (independent context). |
| 2026-07-05 | N-player training pipeline (G9) | `training_civ6.py` threads `num_players` end-to-end: `TrainingExample.value_vector` (per-player, relative order), per-player targets via `multiplayer.scoring` (`ScoringMethod`), `compute_nplayer_value_loss` in `train_epoch`, and `CurrentPlayerValueView` (reduces the per-player value vector to relative-index-0 so scalar `Civ6MCTS`/`GumbelMCTS` can drive an N-player network). Encoder is player-relative, so value targets and the scalar view both key on relative index 0. 2-player path byte-identical; opt-in via `--num_players`/`--scoring_method`. |
| 2026-07-13 | W&B fully retired; MLflow sole tracker | SDK swap verified complete (zero `import wandb` in `alpha_zero/`). `resolve_tracker_enabled()` unifies enablement (`CIV6AZ_MLFLOW_ENABLED` > `MLFLOW_ENABLED` > legacy `WANDB_ENABLED`, warns). Orchestrator `WandbConfig` = deprecated bridge via `to_mlflow_config()`; `--no-wandb`/`--no-tracking` disable ALL tracking. `utils/wandb_logger.py` shim deleted (lazy `alpha_zero.utils.WandbLogger` alias remains, warns). Adapter hardened: numpy/nested metric coercion, `run_name` property (fixes run-group propagation), `alert()` also warns in logs. |
| 2026-07-25 | OpenSpec governance + deterministic validators | Enterprise docs overhaul managed as one OpenSpec change (`openspec/`). Four stdlib-only validators (`scripts/validate_openspec.py`, `validate_kb.py`, `validate_skills.py`, `check_doc_sync.py`; exit 0/1/2; tests in `tests/governance/`) enforced by Claude hooks, `.pre-commit-config.yaml` (ruff pinned to CI's version — 0.15.8 as of 2026-07-25), and a CI `governance` job. `check_doc_sync.py` blocks doc drift (version triple, MODULE_REFERENCE counts, README forbidden patterns, unknown extras, links, changelog-version). Skills gained frontmatter + validation gates; added `openspec-propose`/`openspec-archive`/`doc-sync`/`release`. Stale docs archived to `docs/archive/` (archive-not-delete). |
| 2026-07-25 | PostToolUse autofix defers F401 **and** RUF100 | `--extend-ignore F401` alone made existing `# noqa: F401` directives look unused, so RUF100 `--fix` stripped them and re-exposed the imports to CI's F401 gate. Deferring both keeps intentional availability-probe imports intact (`.claude/hooks/posttooluse_lint.sh`). |
| 2026-08-25 (commit `0d5adee`, authored 2026-08-24) | Per-player value targets in cloud self-play | All three self-play paths in `cloud/gcp/selfplay_worker.py` broadcast one terminal scalar to every position. That reward is relative to whoever was `current_player` at the terminal action, and MicroCiv advances `current_player` only inside `_end_turn` (~6.8 actions/turn), so ~half of every two-player trajectory was labelled from the opponent's perspective. Now signed per position from `env.winner` (absolute). `training_civ6.py` always did this correctly via `relative_value_vector`; only the cloud path diverged. Diagnostic: `value_sign_accuracy` (`training_worker.py:754`). |
| 2026-08-25 | Promotion scoring SSOT + policy registry | Three incompatible definitions of "win rate" were live at once; the production one (`cloud/gcp/arena.py:41`) scores every draw as a loss, capping the statistic at `1 - draw_rate` so promotion is impossible above a 45% draw rate. `evaluation/scoring.py` owns the statistics (ternary variance — a binomial interval on fractional successes is too wide by exactly `draw_rate/4`); `evaluation/promotion.py` owns the decision as a registered, swappable policy. Default `legacy_win_rate` reproduces deployed behaviour exactly; opt in via `CIV6AZ_PROMOTION_POLICY`. |
| 2026-08-27 | Batch self-play bounded by the environment, not an action counter | `run_batch_selfplay`'s loop counted MCTS *action* steps against `max_turns`, the environment's *game-turn* limit. MicroCiv spans ~6 actions per turn, so games stopped at roughly a sixth of their length, no environment reached its own truncation, `winner` stayed `None`, and `outcome_value_for_player` labelled every position 0.0 — the value head trained on a constant. The loop now bounds on `max_turns * MAX_ACTIONS_PER_GAME_TURN` as a runaway guard only, logs games that hit it, and schedules temperature against the environment's own turn. Shape-only tests (keys, alignment, mask density) all pass against uniformly-zero targets, so the regression tests assert content: trajectories exceed `max_turns` steps, and value targets are not uniformly zero. |
| 2026-08-27 | One trajectory shape for every self-play path | `new_trajectory`/`append_step`/`dummy_trajectory` in `cloud/gcp/selfplay_worker.py`, adopted by the sequential, MuZero, batch and both dry-run paths (and the trainer's, which was a fourth inline builder). `append_step` commits all fields or none. The stored object stays a plain `dict`: shards are pickled and read across a rolling window by older and newer trainers, so a dataclass would be a cross-version hazard. |
| 2026-08-27 | Local gates mirror CI (`make verify`) | The Makefile had drifted from CI on four axes (coverage 80 vs 85, coverage scope, `mypy --strict` vs plain, `validate_kb --git`), so passing locally proved nothing. `make verify` is now the single definition; the DEV-SQE handoff calls it. New `verify` and `negative-control` skills; the latter exists because every defect in this branch's history shipped under a green suite. |
| 2026-08-28 | `DryRunProfile` owns every dry-run magic number | `--dry_run` bounded only the outer loop (iterations/games/epochs/eval-games) and left `board_size`/`max_turns`/`num_simulations` at production defaults, so the documented smoke test was 24 full-size games and ran 3h33m without completing. Inner dimensions are now overridden only where the caller left the `TrainingConfig` default, so explicit flags and every existing test config still win. |
| 2026-08-28 | `inference_thread_limit()` caps torch threads for batch-of-1 search | MCTS evaluates one position at a time; torch sizes its intra-op pool to the core count and the per-operator OpenMP barrier (default *active*, spinning) then dominates. Measured on a 4-vCPU box: 1.54 ms/forward at 2 threads vs 347 ms at 4 — 225x. Applied only around self-play and evaluation so batched training keeps torch's default, which does scale with cores on real hardware. Opt out with `CIV6AZ_INFERENCE_THREADS=auto`. |
| 2026-08-30 | A seed determines the run; every search owns its RNG | The project could not replay its own runs — five 20-game probes of **one** checkpoint scored 0.25/0.35/0.375/0.375/0.40 (`docs/evidence/validation_iteration_zero/`). Four unseeded sources, and **only the first was findable by reading code**: eight global `np.random.*` draws in the search; `MicroCivEnv.clone()` building a fresh env with **no seed**, so every MCTS rollout was unseeded in every variant no matter what the caller seeded; `ReplayBuffer.sample` + `DataLoader(shuffle=True)`, found by diffing two `--seed 42` runs that agreed on 525 lines of self-play and differed on the first epoch loss; and `NeuralNetworkPlayer` — the *evaluated agent* — drawing globally while its siblings each owned a seeded generator, which is why the probe spread survived already-seeded environments. **The method is the lesson: seeding is verified by diffing two runs, not by auditing call sites.** Controlled re-measure in `docs/evidence/seeded_probe_reproducibility/`: seeded 5/5 identical, unseeded still spans 0.30–0.45. Behaviour note: a `Generator` draws a different stream from the legacy global functions, so seeded action sequences differ from today's unseeded ones. |
| 2026-08-30 | The suite-wiring invariant is itself a gate (`tests/regression/test_suite_wiring_regression.py`) | Fixing `tests/regression/` fixed one directory, not the category — **five more suites were enumerated by no CI job**: `tests/e2e` (70 tests), `tests/cloud/smoke` (36), `tests/smoke` (30), `tests/scripts` (**3 real failures**, optional-extra imports written bare instead of `importorskip`), `tests/inference`. All now wired. The meta-gate requires every directory holding collectible tests to be named by a CI pytest invocation or listed in `NOT_GATED` with a reason, and holds `make test` to the jobs it claims to mirror. Two design points that decide whether it works: enumeration follows **pytest's collection rule** (a `test_*` function, or a `Test*` class with a `test_*` method), not the filename glob — `tests/test_constants.py` holds `class TestDefaults` and no tests, and a glob would demand CI wiring for a suite that does not exist; and the workflow is parsed with `yaml`, not regex, because a hand parser over YAML is how `validate_skills` came to validate a folded block scalar vacuously. |
| 2026-09-02 | Dead code is found by import graph, not grep | `scripts/audit_import_graph.py` (stdlib AST) resolves names through package `__init__` re-export tables and PEP 562 lazy maps, credits literal `importlib.import_module`, and separates entry points and build-file-referenced modules from dead ones. It found **79 modules / 40,503 lines** with no live importer where a grep-based census had found ~15,000 — and the grep also produced *false positives*: `training_profiler` looked dead but is re-exported by `training/__init__`, and "38 test files" for `cloud/gcp/vertex.py` were substring hits on `vertex_orchestrator` (real count: 1). **The distinction that makes it work:** a package's own `__init__` re-exporting a module is not a use (that is how an unused module hides), but a plain side-effect `import pkg.mod` there is (that is how `_compat` works). |
| 2026-09-02 | 34 dead modules were *inside* the coverage measurement | Deleting dead code **lowers** the reported coverage percentage when the dead code was tested — 14,194 of the 40,503 dead lines were measured by the gate. So a coverage floor must be set from a measurement taken *after* the deletion, never before. This is why WS1 follows WS2 in the plan, and why the v1 plan's "delete first, then widen coverage" order was hiding a red gate. |
| 2026-09-02 | Per-package coverage floors (`coverage-floors.json`) | A global `--cov-fail-under` cannot express "this package must not get worse" for a package the global run omits — and `alpha_zero/cloud/*`, the deployed path, was measured by **no job at all**. CI's `cloud-tests` job now measures it against a floor (58%, measured 59.7%) via a separate `.coveragerc-cloud`, because the repo-wide config omits the very package under test. **Gotcha:** `coverage json` without `--rcfile` re-applies the repo-wide omit list and reports "No data to report" — the floor check then has nothing to measure and looks green. |
| 2026-09-02 | The action-space bug on the live path was 381, not 356 | Both are wrong (`MicroCivActionSpace.total_size` is 376) but all 16 `356` sites were in dead, deferred or docstring code, while `381` was the **default** in `NetworkConfig`, `civ6_local`'s bridge builder, `MuZeroConfig` and the N-player head — and `transformer_network.py` had warned it was wrong since 2026 without anyone changing it. Ranking config drift by literal count finds the wrong one; rank by whether a *live default* carries it. |
| 2026-09-02 | Failed safe-load must not silently downgrade | Both checkpoint loaders retried with `weights_only=False` on any exception — arbitrary code execution from a malformed file, reached without the caller asking. The opt-in is now explicit (`allow_unsafe`/`allow_unsafe_fallback` or `CIV6AZ_ALLOW_UNSAFE_CHECKPOINT`) and the single unsafe call lives in `utils/safe_load.py`, a leaf module: `core/__init__` imports `utils.checkpoint`, so the shared helper cannot live under `core/` without a cycle. Found while doing it: `MuZeroTrainer.save_checkpoint` pickled its config dataclass, so **its own** checkpoints could only be reopened unsafely. |
| 2026-09-02 | The `main`-branch push guard never protected this repo | `pretooluse_guard.sh` compared against the literal `main`; the default branch here is `claude/civ6-alphazero-implementation-vMnun`, so §3.2's "no direct commits to main" was prose. It now resolves the branch at runtime (`CIV6AZ_DEFAULT_BRANCH` → `origin/HEAD` → `main`). A second latent bug in the same rule: `[ on-main ] || is_push && dest_is_main` parses as `(A \|\| B) && C`, so a commit while checked out on main was denied only if the command line also mentioned `main` — i.e. never. |
| 2026-08-30 | A gate is not a gate until something runs it | `tests/regression/` was referenced by no CI job and no make target, so ten hygiene invariants and a pre-existing regression suite had never gated a PR — they passed, which was true and irrelevant. Three sibling failures in the same review: `validate_agents --path <missing>` exited 0 (the hook calls `--path`, so a stale path read as validated), a catch-all allowlist test that checked four exact spellings and missed `^.*$`, and an A5 requirement claiming "every importable package" against a validator checking 29 of 44. **The check to apply to any new gate: prove it fails when the thing it guards is broken, and prove something actually invokes it.** |
| 2026-08-30 | The secret **gate** scans `HEAD`'s lineage; the repo-wide sweep is a separate audit | The first version scanned every ref. CI checks out with `fetch-depth: 0`, so it failed on 6 findings from an unmerged 2026-03 branch that no PR contains or can fix — a gate no PR can turn green, which trains people to ignore it. `--log-opts="HEAD"` still covers this branch's entire lineage (532 commits at the time of writing); `make security-secrets-audit` keeps the repo-wide view available. **My adoption baseline of "0 findings / 525 commits" was measured before `git fetch --all` and was therefore incomplete** — the gate was made blocking on an incomplete measurement, which is the mistake that surfaced this. |
| 2026-08-30 | Secret scanning is `gitleaks`, pinned once in the Makefile | The policy was asserted only by a hand-written pattern list in `tests/security/`, and nothing scanned **history** — where a removed secret still lives. `make security-secrets` runs `gitleaks dir` + `gitleaks git`; CI reads the version via `make print-gitleaks-version` so the pin has one definition. It is blocking on what it actually covers: 0 findings on the tracked tree and 0 across `HEAD`'s full commit lineage (532 commits when last measured, 2026-08-30 — the count moves with every push, the scope does not). **Do not read that as "the repository's history is clean"** — the repo-wide audit has an open finding (see the row above and §5.3); the "0 / 525 commits" figure originally quoted here was measured pre-`git fetch --all` and was incomplete. Gotcha found while validating it: a negative control using AWS's *documentation* example keys (`AKIAIOSFODNN7EXAMPLE`) does **not** fire — gitleaks allowlists those upstream. Use randomly generated values when testing the config. |
| 2026-08-30 | `scripts/validate_agents.py` — the 30 `Agent.md` persona docs are now gated | They had zero validator coverage: no gate, no hook, no test. Codes A1–A6 cover heading form, persona statement, required sections, coverage literals, per-package presence, and referenced-file existence, with `<!-- validate-agents: allow -->` as the escape hatch for prose that deliberately names a removed file. First run found 16 real errors. Warnings (missing conventional sections in 5 docs) are deliberately non-blocking — a visible burndown, same pattern as the `PLR0915` list. |
| 2026-08-30 | `validate_skills.py` scans `.agents/skills/` as well | That tree was routed to the validator by the PostToolUse hook but scanned by **neither** pre-commit nor CI, so it accumulated violations in silence. Also: the hand-rolled frontmatter parser reduced a YAML folded scalar (`description: >`) to the literal `">"`, which passed the non-empty check **vacuously** — worse than not checking. It now parses block scalars. |
| 2026-08-30 | `core/civ6_mcts.py` split into `alpha_zero/core/mcts/` | 2,443 lines holding five search variants, four protocols and three config dataclasses (scattered 1,300 lines from their variants). One module per variant, largest 876 lines. Layering: `mcts_protocols`/`gumbel_ops`/`base` are leaves; `puct`/`simple` build on `base`; **`gumbel` and `muzero` carry their own node and config types and depend on neither `base` nor each other** — the Gumbel-Top-k helpers live in `gumbel_ops` precisely because `MuZeroMCTS` calls them. `civ6_mcts` is a permanent re-export shim with explicit `__all__`; `mcts_factory.py` untouched (its lazy import resolves through the shim), so the diff outside the moved blocks is the shim and docs only. |
| 2026-08-30 | An identity test does not prove a shim is safe | `mock.patch("<old.module>.<Name>")` rebinds an attribute **on the named module object**. A re-export shim trivially satisfies `old_obj is new_obj` while the patch becomes a **silent no-op** whenever the patched symbol's consumer also moves to a sibling module. So `tests/unit/test_mcts_split_backcompat.py` pins patchability separately (`MuZeroTrainingBridge.create_mcts` imports from the shim at call time), plus module-scope-import layering. This is why the three cloud workers are **not** decomposed: they carry **38** `mock.patch` string targets against them. |
| 2026-08-30 | ruff `PLR0915` (`max-statements = 100`) over a module-line-count gate | The pathology here is the 623-line `main`, not the long module — and *moving* a 623-line function to another file does not shorten it, so a decomposition-by-relocation would not have fixed it. Statements-per-function targets it directly and is one line in an existing config, versus a bespoke validator needing six touchpoints. Three `main`s exceed the threshold (248/168/134 statements) and sit in `per-file-ignores` as an explicit burndown list — **delete an entry when its function is split; never raise the threshold.** |
| 2026-08-25 | `EvaluationResult` schema versioning | Added `schema_version`, `score_rate`, `ci_low`, `ci_high`. The `from_dict` read default is deliberately **asymmetric** with the field default so pre-change records (which carry no key) stay distinguishable from post-change ones; defaulting both the same would make them identical on read. |

### 5.3 Known Issues

| Issue | Workaround |
|-------|------------|
| Promotion gate cannot promote at the shipped defaults | Measured: 73.5% draws at `board_size=8`/`max_turns=30` (`docs/evidence/hexcrystal_ab/`), so `legacy_win_rate`'s ceiling is `1 - draw_rate` = 0.265 against a 0.55 threshold — arithmetically impossible. Switching policy alone is *not* enough: 0.55 read as a **score** rate demands ~+35 Elo from one iteration, and rejects a measured +31 Elo model (score_rate 0.545). Fix and arithmetic: **[`docs/runbooks/promotion-gate.md`](docs/runbooks/promotion-gate.md)** — recommends `CIV6AZ_PROMOTION_POLICY=score_rate_lower_bound` with `CIV6AZ_PROMOTION_THRESHOLD=0.52` (~+14 Elo, significance-gated). Code defaults deliberately unchanged. The upstream problem is the draw rate, not the threshold. |
| **UNREMEDIATED: a W&B API key is in git history** | Found 2026-08-30 by the new repo-wide secret audit. 5 findings, one distinct 40-hex value (real W&B key format), all in commit `87cd68a2` (2026-03-19) on `origin/feature/sprint6-validation-and-nplayer-foundation` — **not** on the default branch, **not** in any current PR, and all six files deleted from the tree. Deletion does not remove it from history. **The key must be rotated**; purging objects is cleanup, not remediation. W&B being retired (NG-5) does not make a leaked credential safe. Full triage: [`docs/runbooks/secret-scanning.md`](docs/runbooks/secret-scanning.md). |
| ~~Training and evaluation runs cannot be reproduced~~ (resolved 2026-08-30) | `--seed` on `training_civ6.py`, `run_baseline_eval.py` and `post_run_evaluation.py` now determines environments, every search, the random opponent, replay sampling, batch order and network init. Gated by `tests/regression/test_training_determinism_regression.py` (~7 s). Evidence: `docs/evidence/seeded_probe_reproducibility/`. **Does not** fix the draw rate — see the promotion-gate row above. |
| No GPU in dev environment | Training works on CPU (slower) |
| Batched CPU training is also hit by the OpenMP spin-wait pathology | `inference_thread_limit()` covers self-play/eval only; the training phase keeps torch's default because more cores genuinely help on real hardware. On an oversubscribed VM that default is ~17x slower at batch 128 (795 ms vs 47 ms). Fix per-run with the standard `OMP_NUM_THREADS=2`, or `OMP_WAIT_POLICY=passive` (which alone took a 4-thread forward from 342 ms to 5.8 ms). Must be set before torch is imported, so no library call can do it for you. |
| download.pytorch.org index blocked in web sessions | Use plain `pip install torch` (PyPI wheel works; verified 2026-07-13 with torch 2.13.0) — only the `--index-url https://download.pytorch.org/...` path is blocked. Full test suite then runs locally. |
| `PostToolUse` ruff `--fix` can strip imports mid-task | If you add an import in one edit and its first use in a later edit, the autofix removes it as unused. Re-add the import after the use exists; see guardrail audit in `docs/HARNESS_ENGINEERING_ANALYSIS.md` §6. |
| SimpleMCTS random rollout is weak | Use neural network MCTS after initial training |
| Windows multiprocessing spawn | Use `if __name__ == "__main__"` guard |
| pytest-xdist not always installed | Tests run without parallel flag if missing |
| `serving/` + `inference/` omitted from coverage gate | They require the `web` extra (fastapi/uvicorn), absent from the default-CI `[dev]` install, so they cannot be imported in CI's coverage job. `serving/` has unit tests (`tests/unit/test_serving/`, `importorskip`-guarded) but stays in the omit list until CI installs the `web` extra. |
| ~~`alpha_zero/cloud/*` unmeasured by any job~~ (partly resolved 2026-09-02) | It is still omitted from the *global* gate, but no longer unmeasured: CI's `cloud-tests` job now runs `--cov=alpha_zero.cloud --cov-config=.coveragerc-cloud` and `scripts/check_coverage_floors.py` enforces the per-package floor in `coverage-floors.json` (**58%**, measured 59.7% over 26 files / 5,485 statements on 2026-09-02, up from the 51.1% below because Tier A removed 14 untested dead modules from the denominator). Floors ratchet up only. Largest remaining gaps: `selfplay_worker` 48%, `training_worker` 52%, `vertex_orchestrator` 60%. Un-omitting from the global gate is still future work; the note below records the original analysis. |
| (original analysis) `alpha_zero/cloud/*` omitted from the coverage gate — the deployed path | **Measured 2026-08-28**, correcting the stale rationale in `pyproject.toml` ("require GCP SDK, Vertex AI, GCS"): all 33 `cloud/gcp` modules import cleanly with **no** Google SDK present (every `from google...` is function-local or `TYPE_CHECKING`-guarded), and CI's `cloud-tests` job already runs 1,025 of their tests on `.[dev]` alone. So no extra is needed — but un-omitting is **not** a one-liner: `cloud/gcp` **measures 60.0%** (cloud unit + batch integration + the cloud-touching unit tests; 51.1% from the cloud tests alone) against the **66.1%** it must hold. The implied global drop is **89.69% → ~83.8% (CI red)** — but note that 89.69% is the *documented CI anchor*, not a local measurement: `tests/unit/` cannot complete in the dev sandbox (four attempts wedged, three without coverage at all), so the global figure is arithmetic on one unverified input and the path-to-green below has **under half a point of margin — inside the error bar**. Land the un-omit as a draft PR that deliberately lets CI's coverage job report the real number *before* retuning the gate; do not plan against the derived figure. Path to green: omit the argparse/entrypoint shims individually (`cli.py`, `__main__.py`, `container_healthcheck.py`, +0.8pt), then add tests for `quota_manager.py` (287 stmts, 0%, no test file at all), `lifecycle.py` and `pipeline_cache.py`. Estimated 1–2 weeks. This is where every bug fixed in #113 lived, so the gate cannot see that whole class of defect. |
| Coverage instrumentation crashes under nightly numpy 2.4 + coverage C-tracer | Local-only sandbox issue ("cannot load module more than once"); run tests without `--cov`, or use pinned numpy. CI uses pinned versions and is unaffected. **Refined 2026-08-30:** `COVERAGE_CORE=pytrace` does **not** help for any target reached through `tests/conftest.py`, because that conftest imports numpy itself (line 20) — the double load happens at conftest import, before the `--cov` target is considered. Measured against `--cov=alpha_zero.core.mcts`: identical `ImportError: cannot load module more than once per process` with and without `pytrace`. The workaround only applies to suites that bypass the root conftest. **Partial workaround (2026-08-09):** `COVERAGE_CORE=pytrace` selects coverage's pure-Python tracer and avoids the crash — but only for `--cov` targets that do **not** import numpy. `COVERAGE_CORE=pytrace pytest tests/unit/test_hexcrystal_*.py --cov=alpha_zero.hexcrystal` works (measured 87.10%); `--cov=alpha_zero.utils.checkpoint` still crashes, because coverage importing a numpy-dependent module on top of conftest's import is the double load. CI remains the gate for numpy-importing modules. |
| ~~MicroCiv hardcoded for 2 players~~ (resolved 2026-07-05) | Env + encoder N-player (2–6); training pipeline now N-player too. **G7/G8/G9 all done**: `training_civ6.py` threads `num_players` (per-player `value_vector`, `compute_nplayer_value_loss`, `CurrentPlayerValueView` scalar-MCTS adapter). 2-player path byte-identical. Tests: `tests/unit/test_nplayer_training_pipeline.py`. Train: `python -m alpha_zero.training_civ6 --num_players 4`. |
| ~~12 DDP tests xfailed~~ (stale — corrected 2026-07-05) | DDP tests use CUDA `skipif`, not `xfail`. Repo has exactly one real `xfail` (`test_phase2.py:381`). **Re-measured 2026-08-27 with a stated rule** (`grep -rEc '@pytest\.mark\.(skip|skipif|xfail)|pytest\.(skip|importorskip)\(' tests/`): **124 sites across 68 files** — the earlier "11 sites / 17 tests" figure counted only static `@pytest.mark.skip`, which is why it looked an order of magnitude smaller. Most are legitimate environment gates (CUDA, optional extras: `hub`, `web`/fastapi, GCS, SpacetimeDB, JAX). Genuine debt is three items: the class-level skip on `UncivInterface.parse_multiplayer_save` (a planned method that does not exist), one disabled flaky concurrency test (`test_buffer_edge_cases.py`), and one "DUCT not yet implemented" skip. Quote the rule with the number — a bare count is not reproducible. |
| mlruns/ file-store deprecated | Run `scripts/migrate_mlflow_to_sqlite.py` to migrate to `sqlite:///mlflow.db` |
| test_deployment_infra.py: 6 failures | Require .env file not present locally; skip with `pytest -m 'not env_required'` |
| ~~Two divergent `WandbConfig` classes~~ (resolved 2026-07-13) | `utils/wandb_logger.py` shim deleted; orchestrator `WandbConfig` is now an explicit deprecated bridge (`to_mlflow_config()`), `MLflowConfig` single-source. |
| W&B naming debt (accepted) | The SDK swap is COMPLETE — no `import wandb` anywhere in `alpha_zero/`. Retained for compat: `TournamentWandbLogger` (public, MLflow-backed), worker-local `wandb_logger` vars, serialized `wandb_enabled` config fields, lazy deprecated `alpha_zero.utils.WandbLogger` alias. `run_sweep.py` was ported to Optuna on 2026-07-13 (`alpha_zero/optimizations/hyperparameter_search.py`, `pip install -e '.[tuning]'`) — no wandb imports remain in `alpha_zero/` or entry points; two read-only historical-run readers (`scripts/compare_training_runs.py`, `scripts/search_efficiency_report.py`) keep lazy `wandb.Api()` access to W&B-hosted history, documented as deprecated. Adapter no-ops: histograms/images/`define_metric`. |

### 5.4 Bug Fixes

| Bug | Fix | Date |
|-----|-----|------|
| Action mask indices mismatch | Changed to player-relative indices in `civ6_actions.py` | 2026-01-20 |
| Evaluation `_env` attribute error | Always create fresh wrapped environment per game | 2026-01-20 |
| Value tensor shape mismatch | Added `value.unsqueeze(1)` in training loss | 2026-01-20 |
| Curriculum `record_game` API | Changed `win=` to `won=` parameter | 2026-01-20 |
| **Training from scratch each iteration** | Added `--model_version` to load incumbent weights (v1.3.0) | 2026-02-23 |
| EvaluationPipeline wrong kwargs | Changed evaluate_vs_random(model=, device=) → (Civ6MCTS(agent), n_games=) | 2026-05-21 |
| run_baseline_eval hardcoded run_id | Added resolve_run_id() auto-detection; removed production UUID default | 2026-05-21 |
| EvaluationResult.win_rate type | win_rate is WinRateResult object, not float; fixed access to .win_rate.win_rate | 2026-05-21 |
| MLflowLogger lifecycle | FAILED vs FINISHED status now set correctly on context manager exit | 2026-05-21 |
| StrategyAdvisor `.complete()` mismatch | Added `LLMClient.complete()` alias → `generate_sync(prompt).content`; advisor's `llm_client.complete(prompt)` call path no longer `AttributeError`s against a real client | 2026-06-20 |
| Orchestrator asserts stripped under `python -O` | 24 `assert self._x is not None` init-guards → `_require()` raising `RuntimeError`; safe under `-O` | 2026-06-20 |
| `--dry_run` never completed (3h33m, measured) | Inner dimensions bounded via `DryRunProfile`; smoke test now exits 0 in 16.4s | 2026-08-28 |
| Network self-play ~200x slower than the random-rollout baseline | Torch intra-op threads capped around batch-of-1 inference (`inference_thread_limit`); the SimpleMCTS/network-MCTS cliff (10.9s vs 321s per 20 games) disappeared | 2026-08-28 |
| `serving/` had zero tests | Added `tests/unit/test_serving/test_inference_server.py` (12 tests, endpoints + `_load_model` paths, fastapi `importorskip`) | 2026-06-20 |

### 5.5 Information to Preserve

- Build/test commands that work
- Non-obvious environment setup steps
- Architectural decisions and their rationale
- Gotchas discovered during implementation
- LLM API keys are in .env (never commit)

---

## SECTION 6: EXECUTION PROTOCOL

### 6.1 Initial Actions (Always Do First)

```bash
# 1. Read this file (CLAUDE.md) and PROJECT_CHARTER.md (scope, non-goals, invariants)

# 2. Check environment
python --version  # Should be 3.10+

# 3. Install dependencies if needed
pip install -r requirements.txt
pip install -e .

# 4. Scan project structure
find . -type f -name "*.py" | grep -v __pycache__ | head -50

# 5. Run existing tests to establish baseline
pytest tests/ -v --tb=line | head -50

# 6. Check for any failing tests
pytest tests/ -v --tb=short 2>&1 | grep -E "(PASSED|FAILED|ERROR)"
```

### 6.2 Implementation Order

```
1. UNDERSTAND: Read existing code before modifying
   - Identify patterns used in similar code
   - Check for existing utilities that can be reused
   - Review related tests for expected behavior

2. IMPLEMENT: Write smallest working version first
   - Start with function signature and docstring
   - Implement core logic without edge cases
   - Add type hints

3. TEST: Write tests alongside implementation
   - Unit test for each public method
   - Test edge cases and error conditions
   - Mock external dependencies

4. VERIFY: Run verification loop
   - ruff check (lint)
   - mypy (types)
   - pytest (tests)

5. REFACTOR: Clean up only if needed
   - Only refactor if tests pass
   - Keep changes minimal
   - Preserve existing APIs

6. DOCUMENT: Update this file
   - Add new commands to build table
   - Document architectural decisions
   - Note any gotchas discovered
```

### 6.3 Completion Checklist

```
□ All success criteria from Section 1.2 met
□ All verification commands from Section 4.1 pass
□ CLAUDE.md updated with new commands/decisions
□ Summary of changes provided to user
□ Known limitations documented
□ No hardcoded values (check for API keys, paths, magic numbers)
□ Tests added for new functionality
□ Type hints on all new public interfaces
```

---

## SECTION 7: PROJECT STRUCTURE

```
Civ6-AlphaZero/
├── alpha_zero/
│   ├── __init__.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── microciv.py          # Simplified training environment
│   │   └── civ6_encoder.py      # State → Tensor encoding
│   ├── core/
│   │   ├── __init__.py
│   │   ├── civ6_network.py      # Neural networks
│   │   ├── mcts/                # MCTS variants, one module each (8 files):
│   │   │                        #   mcts_protocols, gumbel_ops, base (leaves);
│   │   │                        #   puct, simple; gumbel, muzero (independent)
│   │   ├── civ6_mcts.py         # Re-export shim for the historical import path
│   │   └── civ6_actions.py      # Action space definition
│   ├── distributed/             # Distributed training
│   │   ├── __init__.py
│   │   ├── actor.py             # Self-play actors
│   │   ├── learner.py           # Central learner
│   │   ├── coordinator.py       # Training orchestration
│   │   └── shared_storage.py    # Weight & buffer sharing
│   ├── storage/                 # Storage backends (NEW)
│   │   ├── __init__.py
│   │   ├── backend.py           # Abstract StorageBackend interface
│   │   ├── config.py            # StorageConfig with GCS settings
│   │   ├── local_backend.py     # Local filesystem implementation
│   │   ├── gcs_backend.py       # GCS implementation
│   │   ├── models.py            # Metadata dataclasses
│   │   ├── replay_buffer.py     # Streaming replay buffer
│   │   └── factory.py           # create_storage_backend
│   ├── cloud/                   # Cloud-native components
│   │   ├── __init__.py
│   │   ├── config.py            # CloudConfig, MachineProfile
│   │   └── gcp/                 # The deployed path: selfplay_worker, training_worker,
│   │                            #   arena, vertex_orchestrator, storage, preemption_handler
│   ├── orchestrator/            # Unified Training Orchestrator
│   │   ├── __init__.py
│   │   ├── config.py            # OrchestratorConfig with schedules
│   │   ├── orchestrator.py      # TrainingOrchestrator main class
│   │   ├── scheduler.py         # HyperparameterScheduler
│   │   ├── checkpointer.py      # CheckpointManager
│   │   ├── rag_integration.py   # RAG-LLM integration layer
│   │   ├── strategy_advisor.py  # LLM-based strategy advisor
│   │   └── __main__.py          # CLI entry point
│   ├── monitoring/              # MLflow experiment tracking (NEW)
│   │   ├── config.py            # MLflowConfig dataclass (env-var + YAML aware)
│   │   ├── metrics.py           # compute_value_correlation / compute_elo_spread / compute_policy_entropy
│   │   ├── mlflow_logger.py     # Hardened MLflowLogger (optional import, try/except guards)
│   │   ├── tournament_dashboards.py # TournamentMlflowLogger
│   │   └── __init__.py          # Public exports
│   ├── multiplayer/             # N-player multiplayer support
│   │   ├── mcts_node.py         # N-player MCTS node (per-player values)
│   │   ├── mcts.py              # Max-N, Paranoid, Best-Reply MCTS
│   │   ├── network_heads.py     # N-player value head
│   │   ├── env_adapter.py       # 2-player → N-player env wrapper
│   │   ├── turn_manager.py      # Sequential/Simultaneous/Dynamic turns
│   │   ├── opponent_model.py    # Frequency-based opponent modeling
│   │   ├── scoring.py           # Ranking & scoring system
│   │   ├── game_connector.py    # Unciv/Mock game connectors
│   │   └── config.py            # MultiplayerConfig
│   ├── pbt/                     # Population-Based Training (10 files)
│   │   ├── population.py        # Population management
│   │   ├── mutation.py          # Mutation operators
│   │   ├── fitness.py           # Fitness evaluation
│   │   └── muzero_hyperparams.py # MuZero-specific hyperparams
│   ├── tournament/              # ELO Tournament System (9 files)
│   ├── web/                     # Web Interface (12 files, FastAPI + WebSocket)
│   ├── bridge/                  # Civ6 Lua Mod Bridge (15 files, file IPC + FireTuner socket)
│   ├── hierarchical/            # Hierarchical Goal Planning (19 files)
│   ├── training/                # Training integrations (11 files: MuZero trainer,
│   │                           #   safeguards, diagnostics, convergence monitor)
│   ├── interfaces/              # Game interfaces (4 files, incl. unciv_interface.py)
│   ├── mechanics/              # Game mechanics: diplomacy, trade, espionage,
│   │                           #   religion, government, climate, eras (12 files)
│   ├── evaluation/             # Evaluation + LLM eval harness, llm_harness/ (14 files)
│   ├── gameplay/               # Gameplay systems: terrain, fog of war, tech, combat (8)
│   ├── optimizations/         # Training optimization strategies (10 files)
│   ├── opponents/             # Opponent types & strategies (4 files)
│   ├── spacetimedb/           # SpacetimeDB integration (9 files)
│   ├── inference/             # Inference service: client/engine/schemas (4)
│   ├── serving/               # FastAPI inference server (2 files; web extra,
│   │                           #   omitted from the default coverage gate — see §B/known issues)
│   ├── human_interface/       # Human gameplay interface (6 files)
│   ├── integration/           # Integration bridge / game connector (2 files)
│   ├── rag/                    # RAG pipeline: stores/processors/agents (22 files; see §11)
│   ├── utils/                   # Utilities
│   │   ├── __init__.py
│   │   ├── logging.py           # Logging infrastructure
│   │   ├── config.py            # Configuration management
│   │   ├── llm_client.py        # LLM API integration
│   │   ├── factories.py         # Component factory functions
│   │   ├── test_runner.py       # Test execution utilities
│   │   ├── agent_tools.py       # Claude Code agent tools
│   │   └── env_loader.py        # Zero-dependency .env loader (CIV6AZ_ convention)
│   └── training_civ6.py         # Training entry point
├── dockerfiles/                 # Container images
│   ├── Dockerfile.vertex        # The deployed image (built by cloudbuild.yaml)
│   └── Dockerfile.inference     # Inference server image (infra/cloudbuild-gameplay.yaml)
├── infra/                       # Infrastructure (NEW)
│   ├── __init__.py
│   └── vertex_launcher.py       # Vertex AI job launcher
├── tests/
│   ├── __init__.py
│   ├── unit/                    # Unit tests
│   │   ├── test_monitoring/     # MLflow monitoring tests (24)
│   │   └── test_utils/          # Env loader tests (18)
│   ├── scripts/                 # Script-level tests (39 + more)
│   │   └── test_run_baseline_eval.py
│   └── integration/             # Integration tests
├── configs/
│   ├── civ6_config.yaml         # Original configuration file
│   ├── orchestrator.yaml        # Orchestrator master config
│   ├── orchestrator_fast.yaml   # Fast testing preset
│   └── orchestrator_production.yaml  # Production preset
├── data/
│   └── knowledge/               # RAG knowledge base documents
│       ├── civ6_strategies.md
│       ├── unit_tactics.md
│       └── build_orders.md
├── .env                         # Environment variables (DO NOT COMMIT)
├── .env.template                # Environment template
├── requirements.txt             # Dependencies
├── pyproject.toml               # Package configuration
├── Makefile                     # Build and deployment targets
└── CLAUDE.md                    # This file
```

> **Authoritative module list:** `alpha_zero/` has **29 submodules** as of v1.8.0.
> File counts above are recursive `*.py` counts (excluding `__pycache__`). The
> single source of truth — one row per submodule with responsibility — lives in
> [`docs/MODULE_REFERENCE.md`](docs/MODULE_REFERENCE.md); update it when adding/removing modules.

---

## SECTION 8: QUICK START

```bash
# Setup
cd Civ6-AlphaZero
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Configure environment
cp .env.template .env
# Edit .env with your API keys

# Run smoke tests
pytest tests/unit/test_environment.py -v
python -m alpha_zero.training_civ6 --dry_run

# Full test suite
pytest tests/ -v --cov=alpha_zero

# Train MicroCiv
python -m alpha_zero.training_civ6 --num_games=100 --num_iterations=100
```

---

## SECTION 9: COMPONENT DOCUMENTATION

### 9.1 MicroCiv Environment

A simplified 4X game for Phase 1 training:
- **Board**: 16x16 grid (configurable)
- **Players**: 2 players, turn-based
- **Units**: Warrior, Settler, Archer
- **Actions**: Move, Attack, Skip, Produce, End Turn
- **Victory**: Eliminate enemy or most cities at turn limit

State encoding (13 channels):
| Channels | Feature |
|----------|---------|
| 0-2 | Terrain (plains, hills, mountain) |
| 3-5 | Own units (warrior, settler, archer) |
| 6-8 | Enemy units |
| 9 | Own cities |
| 10 | Enemy cities |
| 11 | Visibility |
| 12 | Constant (bias term) |

### 9.2 Neural Network Architecture

```
Spatial Input (13, H, W)
    │
    ▼
Conv2d (13 → 64) + BN + ReLU
    │
    ▼
ResBlock × 6 (64 filters)
    │
    ├─────────────────────┐
    │                     │
    ▼                     ▼
Policy Head          Value Head
Conv2d (64→2)        Conv2d (64→1)
FC (combined)        FC (combined)
    │                     │
    ▼                     ▼
Policy Logits (A)    Value [-1, 1]
```

### 9.3 LLM Integration

```python
from alpha_zero.utils import create_llm_client

# Create client (reads from .env)
client = create_llm_client()

# Generate response
response = await client.generate(
    prompt="Analyze this game state...", system_prompt="You are a Civ6 strategy expert."
)

# Synchronous version
response = client.generate_sync("Hello!")
```

### 9.4 Test Runner

```python
from alpha_zero.utils import TestRunner, run_tests

# Quick test
result = run_tests(with_coverage=True)
print(result.summary())

# Specific tests
runner = TestRunner.from_env()
result = runner.run_unit_tests()
result = runner.run_by_keyword("mcts")
```

### 9.5 Configuration System

```python
from alpha_zero.utils import ConfigManager, load_config

# Load from file
config = load_config("configs/civ6_config.yaml")

# With environment overrides
# Set CIV6AZ_TRAINING_BATCH_SIZE=512 in environment
manager = ConfigManager("configs/civ6_config.yaml")
training_config = manager.get_training_config()
```

### 9.6 Training Orchestrator

The unified training orchestrator provides a single entry point for all training modes.

```python
from alpha_zero.orchestrator import TrainingOrchestrator, OrchestratorConfig

# Load from preset
config = OrchestratorConfig.from_preset("fast")

# Or from YAML file
config = OrchestratorConfig.from_yaml("configs/orchestrator.yaml")

# Apply environment overrides (CIV6AZ_TRAINING_BATCH_SIZE=512)
config = config.apply_env_overrides()

# Create and run orchestrator
orchestrator = TrainingOrchestrator(config)
orchestrator.run()  # Full training loop
```

CLI usage:

```bash
# Show help
python -m alpha_zero.orchestrator --help

# Run with preset
python -m alpha_zero.orchestrator --preset fast

# Run with config file
python -m alpha_zero.orchestrator --config configs/orchestrator.yaml

# Dry run (validate configuration)
python -m alpha_zero.orchestrator --preset fast --dry_run

# Override settings
python -m alpha_zero.orchestrator --preset balanced \
    --iterations 10000 \
    --games-per-iteration 50 \
    --no-wandb --no-rag
```

### 9.7 Hyperparameter Scheduling

```python
from alpha_zero.orchestrator import HyperparameterScheduler
from alpha_zero.orchestrator.config import ScheduleConfig

# Create schedule configurations
lr_schedule = ScheduleConfig(
    schedule_type="warmup_cosine",
    initial_value=0.001,
    final_value=0.00001,
    decay_steps=100000,
    warmup_steps=5000,
)

# Create scheduler
scheduler = HyperparameterScheduler(
    lr_schedule=lr_schedule,
    batch_size_schedule=ScheduleConfig(initial_value=64),
)

# Get values at any step
lr = scheduler.get_learning_rate(step=1000)
batch_size = scheduler.get_batch_size(step=1000)
all_values = scheduler.get_all_values(step=1000)
```

### 9.8 RAG Integration

```python
from alpha_zero.orchestrator import RAGIntegration
from alpha_zero.orchestrator.config import RAGConfig

# Create RAG config
rag_config = RAGConfig(
    enabled=True,
    persist_directory="data/knowledge_db",
    knowledge_dir="data/knowledge",
    store_insights=True,
)

# Initialize RAG
rag = RAGIntegration(rag_config)
rag.initialize()  # Ingests knowledge base

# Query strategies
results = rag.query_strategy("early game expansion strategy")

# Store training insights
rag.store_training_insight("Found that aggressive expansion wins more games")

rag.close()
```

### 9.9 MLflow Monitoring

Optional MLflow tracking backend for the orchestrator. Tracks win rates,
value correlation, ELO spread, and training metrics. Coexists with W&B
(both can be enabled in parallel).

```python
from alpha_zero.monitoring import MLflowConfig, MLflowLogger

config = MLflowConfig(
    enabled=True,
    tracking_uri="file://./mlruns",
    experiment_name="civ6_50iter",
    run_name="trial_1",
    tags={"env": "prod"},
)
logger = MLflowLogger(config)
logger.init(config_dict={"lr": 0.001, "batch_size": 64})
logger.log_iteration(
    iteration=5,
    train_metrics={"train/loss": 1.2},
    game_metrics={"game/win_rate": 0.6},
    scheduler_metrics={"train/learning_rate": 0.001},
)
logger.finish()
```

Env vars (`CIV6AZ_MLFLOW_*` prefix): `ENABLED`, `TRACKING_URI`,
`EXPERIMENT_NAME`, `RUN_NAME`, `TAGS` (form `"k=v,k=v"`),
`ARTIFACT_LOCATION`, `LOG_FREQUENCY`, `LOG_MODELS`, `LOG_ARTIFACTS`,
`NESTED_RUNS`, `FAIL_SILENTLY`. Standard `MLFLOW_*` env vars also honored
as fallback.

Install: `pip install -e '.[monitoring]'`. Launch the 50-iter production
run via `scripts/launch_50iter_production_monitored.py`. Inspect via
`mlflow ui --backend-store-uri ./mlruns --port 5000`.

### 9.10 LLM Evaluation Harness

Optional integration of `langfuse-eval-harness` (github.com/ianshank/Agents) to
evaluate the LLM-driven `StrategyAdvisor`. The harness drives a `callable` target
that imports `alpha_zero.evaluation.llm_harness.target:advisor_target`, runs the
advisor, and grades the JSON-serialized result with its scorers.

**What it checks (be honest about scope):** by default the gate runs *offline*
with a deterministic stub LLM, so it is a **contract / serialization regression
gate** for the advisor's structured output — it does **not** measure LLM
quality. Real quality grading needs the online path
(`CIV6AZ_LLM_EVAL_ONLINE=1`) with a real judge.

It gates **both** advisor modes:
- `analyze` → `AnalysisResult` (`configs/eval/strategy_advisor.yaml`,
  dataset `data/eval/strategy_advisor_cases.jsonl`): scorers `structure`
  (keys), `has_suggestions` (content), `suggestions_non_empty` (regex).
- `report` → `GameReport` (`configs/eval/game_report.yaml`, dataset
  `data/eval/game_report_cases.jsonl`): scorers `structure` (keys),
  `has_recommendations` (content), `turns_is_integer` (regex).

```bash
pip install -e '.[eval]'                 # langfuse-eval-harness pinned to a SHA
make eval-llm                            # offline gate, BOTH modes; exit 1 on regression
python scripts/run_llm_eval.py --online  # real provider (needs API keys)
python scripts/run_llm_eval.py --config configs/eval/game_report.yaml  # one mode
```

- Config schema: `dataset`/`target`/`scorers`/`judge`/`sinks`/`gate`; scorers
  carry a `name`, the gate references it.
- Runner `scripts/run_llm_eval.py` takes a repeatable `--config` (defaults to
  both configs) and degrades gracefully (exit 0) when the `[eval]` extra is
  absent, so it is CI-safe unconditionally; otherwise it propagates the harness
  exit code (0 = gate pass, 1 = gate fail), failing fast on the first config.
- Dependency-free core: `tests/unit/test_strategy_advisor_contract.py` keeps the
  contract-regression coverage even without the harness installed (both modes,
  dataset round-trip, and config↔output key consistency).
- Env vars: `CIV6AZ_LLM_EVAL_ONLINE` (online switch), `CIV6AZ_EVAL_DATASET`,
  `CIV6AZ_EVAL_REPORT_DATASET`, `CIV6AZ_EVAL_OUT`, `CIV6AZ_EVAL_SAMPLE_RATE`.

---

## SECTION 10: DEBUG COMMANDS

```bash
# Check tensor shapes
python -c "
from alpha_zero.envs.microciv import MicroCivEnv
env = MicroCivEnv()
obs, _ = env.reset()
print(f'Spatial: {obs[\"spatial\"].shape}')
print(f'Vector: {obs[\"vector\"].shape}')
print(f'Action mask: {obs[\"action_mask\"].shape}')
"

# Test network forward pass
python -c "
import torch
from alpha_zero.core.civ6_network import MicroCivNetwork
net = MicroCivNetwork(board_size=16, action_space_size=356)
spatial = torch.randn(1, 13, 16, 16)
vector = torch.randn(1, 64)
policy, value = net(spatial, vector)
print(f'Policy shape: {policy.shape}')
print(f'Value: {value.item():.3f}')
"

# Check LLM configuration
python -c "
from alpha_zero.utils.llm_client import LLMConfig
config = LLMConfig.from_env()
print(f'Provider: {config.provider}')
print(f'OpenAI key set: {bool(config.openai_api_key)}')
print(f'Anthropic key set: {bool(config.anthropic_api_key)}')
"

# Run single game with rendering
python -c "
from alpha_zero.envs.microciv import MicroCivEnv
import numpy as np
env = MicroCivEnv(board_size=8, render_mode='ansi', seed=42)
env.reset()
print(env.render())
for _ in range(10):
    mask = env.get_valid_actions()
    action = np.random.choice(np.where(mask > 0)[0])
    env.step(action)
print(env.render())
"
```

---

## SECTION 11: RAG SYSTEM

### 11.1 RAG Overview
The RAG (Retrieval-Augmented Generation) system enhances agent workflows with context from a vector database.

### 11.2 RAG Architecture
```
alpha_zero/rag/
├── config/         # Configuration management (Pydantic models)
├── stores/         # Vector store backends (Memory, Chroma, FAISS)
├── processors/     # Document processing and embeddings
├── agents/         # Agent workflow integration
├── utils/          # Logging, metrics, caching
└── pipeline.py     # Unified RAG pipeline
```

### 11.3 RAG Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/unit/test_rag/ -v` | Run RAG unit tests |
| `pytest tests/integration/test_rag/ -v` | Run RAG integration tests |
| `pip install chromadb faiss-cpu sentence-transformers` | Install RAG dependencies (not a packaged extra) |
| `pip install -e ".[all]"` | Install all dependencies |

### 11.4 RAG Usage Example

```python
from alpha_zero.rag import RAGPipeline, RAGConfig

# Create pipeline
config = RAGConfig.from_env()
pipeline = await RAGPipeline.create(config)

# Ingest documents
await pipeline.ingest_text("Strategy guide content...", source="guide.txt")

# Query with context
result = await pipeline.query("How to win early game?", top_k=5)
print(result.context)

await pipeline.close()
```

### 11.5 RAG Environment Variables
- `RAG_VECTOR_STORE_TYPE` - Backend: memory, chroma, faiss
- `RAG_EMBEDDING_MODEL` - Embedding model name
- `RAG_EMBEDDING_API_KEY` - API key for OpenAI embeddings
- `RAG_LOG_LEVEL` - Logging level

---

## SECTION 12: VERTEX AI CLOUD DEPLOYMENT

### 12.1 Overview
The system supports deployment to Google Vertex AI for distributed training at scale.

### 12.2 Storage Backend
```python
from alpha_zero.storage import create_storage_backend, StorageConfig

# Local storage (default)
storage = create_storage_backend(StorageConfig.for_local("./data"))

# GCS storage (for cloud)
storage = create_storage_backend(StorageConfig.for_gcs("my-bucket", "civ6-alphazero"))

# Save/load models
storage.save_model(state_dict, iteration=100)
state_dict, metadata = storage.load_champion()

# Save trajectories
storage.save_trajectory(trajectory, iteration=1, worker_id=0, game_id="game_001")
```

### 12.3 Machine Profiles
| Profile | GPU | VRAM | Spot $/hr | On-Demand $/hr | Use For |
|---------|-----|------|-----------|----------------|---------|
| `selfplay_t4_spot` | 1× T4 | 16 GB | $0.11 | $0.35 | Self-play (budget) |
| `selfplay_l4_spot` | 1× L4 | 24 GB | $0.24 | $0.80 | Self-play |
| `train_l4_single` | 1× L4 | 24 GB | — | $0.80 | Training (budget) |
| `train_l4_ondemand` | 2× L4 | 48 GB | — | $2.68 | Training |
| `train_a100_burst` | 1× A100 | 40 GB | — | $3.67 | Burst training |
| `eval_l4_spot` | 1× L4 | 24 GB | $0.24 | $0.80 | Evaluation |

### 12.4 Cloud Workers

The deployed workers live in `alpha_zero/cloud/gcp/` and are entrypointed by
`alpha_zero.cloud.gcp.entrypoint` inside `dockerfiles/Dockerfile.vertex`. Each has a
`--help`; `--dry_run` exercises the path without GCS:

```bash
python -m alpha_zero.cloud.gcp.selfplay_worker --help
python -m alpha_zero.cloud.gcp.training_worker --help
python -m alpha_zero.cloud.gcp.arena --help
python -m alpha_zero.cloud.gcp.vertex_orchestrator --help
```

### 12.5 Environment Variables
| Variable | Purpose |
|----------|---------|
| `CIV6_STORAGE_BACKEND` | Backend type: "local" or "gcs" |
| `CIV6_GCS_BUCKET` | GCS bucket name |
| `CIV6_GCS_PREFIX` | Path prefix in bucket |
| `CIV6_ITERATION` | Current training iteration |
| `CIV6_WORKER_ID` | Self-play worker identifier |
| `CIV6_DEVICE` | PyTorch device (cuda/cpu) |
| `AIP_CHECKPOINT_DIR` | Vertex AI checkpoint directory (auto) |
| `AIP_MODEL_DIR` | Vertex AI model directory (auto) |

### 12.6 Docker Build
```bash
# Build the deployed image locally
make build-all

# Push to Artifact Registry
make setup-artifact-registry
make push

# Local smoke (no Docker): dry-run the deployed worker
make local-smoke
```

### 12.7 Vertex AI Jobs
```bash
# Submit self-play fleet
python -m infra.vertex_launcher \
    --mode=selfplay \
    --iteration=5 \
    --num-workers=3 \
    --games-per-worker=100

# Submit training job
python -m infra.vertex_launcher \
    --mode=train \
    --iteration=5 \
    --training-steps=1000

# Submit evaluation job
python -m infra.vertex_launcher \
    --mode=eval \
    --iteration=5 \
    --num-games=400

# Run full iteration
python -m infra.vertex_launcher \
    --mode=iteration \
    --iteration=5 \
    --wait
```

### 12.8 Estimated Costs (Budget Tier)
| Component | Hours/mo | Rate | Monthly |
|-----------|----------|------|---------|
| Self-play (3× T4 Spot) | 270 | $0.11/hr | ~$30 |
| Training (L4 on-demand) | 90 | $0.80/hr | ~$72 |
| Evaluation (L4 Spot) | 45 | $0.24/hr | ~$11 |
| GCS Storage (500 GB) | — | $0.02/GB | ~$10 |
| **Total** | | | **~$145/mo** |

---

## SECTION 13: NEXT STEPS

> Long-term themes: see ROADMAP.md. Scope boundaries and invariants: see PROJECT_CHARTER.md.

### Current State (v1.8.0)
Training convergence (the v1.3.0 weight-loading fix) has been validated, and the
project has since shipped through v1.8.0:
- **v1.5.0** — convergence validation + MLflow monitoring + batch self-play groundwork
- **v1.6.0** — cloud-deploy prep: Vertex launcher, runbook, GCP OAuth/VM fixes, 85%→80% CI gate
- **v1.7.0** — full-Civ6 local gameplay (`BridgeAgentLoop` wires the agent to Civ6) and the
  `langfuse-eval-harness` LLM eval gate for `StrategyAdvisor` (`make eval-llm`)
- **v1.8.0** — headless FireTuner socket bridge (`tuner_socket_bridge.py`): live Civ6
  gameplay verified over the socket, no GUI relay needed (coverage gate back to 85%
  as of 2026-07-13)

### Implemented (substantially built)
- **PBT**: `alpha_zero/pbt/` (10 files) · **Tournament**: `alpha_zero/tournament/` (9 files)
- **Web Interface**: `alpha_zero/web/` (12 files) · **Civ6 Bridge**: `alpha_zero/bridge/` (15 files)
- **Hierarchical Planning**: `alpha_zero/hierarchical/` (19 files)
- **MuZero**: `alpha_zero/core/muzero_network.py`, `training/muzero_*`
- **N-Player Multiplayer**: `alpha_zero/multiplayer/` (18 files)
- **Inference/Serving**: `alpha_zero/inference/` (4 files), `alpha_zero/serving/` (FastAPI server)

### Genuine Next Steps

> Two previously-listed items were deleted on 2026-07-25 as untrue. **N-player
> MicroCiv** (former item 1) landed on 2026-07-05 — `training_civ6.py` threads
> `num_players` end-to-end, covered by `tests/unit/test_nplayer_training_pipeline.py`
> and two integration suites; §5.3 already recorded this and §13 had not been
> updated in lockstep. **"12 xfailed DDP tests"** (former item 2) never existed:
> `tests/unit/test_distributed/` is not a directory, the DDP suites are flat
> `tests/unit/test_distributed_*.py` files, and the repo contains exactly one
> `xfail` (`tests/integration/test_phase2.py:381`) plus one CUDA `skipif`.

1. **Batch self-play in cloud**: wired and tested, but not yet the production
   path and not yet measured. `BatchGumbelMCTS` (`core/batch_mcts.py`) +
   `VectorizedMicroCivEnv` drive `cloud/gcp/selfplay_worker.py:run_batch_selfplay()`.
   **Corrected 2026-08-27:** that path emitted no `action_masks`, and
   `training_worker.prepare_dataset` substitutes an all-ones mask for a missing
   key — so every batch-generated sample trained with masked policy loss
   disabled. Fixed, with the trajectory shape now built by shared helpers
   (`new_trajectory`/`append_step`) so a path cannot omit a field, and the batch
   loop committing per-step fields atomically (it previously appended
   observations before a search that could fail, leaving ragged trajectories the
   trainer would index past). What remains: no config selects `batch_gumbel`
   (`--num_parallel_games` now defaults to 1 and warns when parallelism is
   requested with a variant that cannot use it, rather than silently running
   sequential), and the claimed 5-10x still has no benchmark artifact. (The LEGACY
   `cloud/selfplay_worker.py` and its `Dockerfile.selfplay` were deleted 2026-09-02.)
2. **Fog of war** (gap G10): partly done, and the remaining items are
   **sequential, not parallel**. Fog is implemented in `MicroCivEnv` (flag,
   default off) and the encoder; `DeterminizedMCTS`
   (`multiplayer/determinization.py`) is written and tested but has **zero call
   sites**. Remaining, in order: (a) fix the action-mask information leak
   (`envs/microciv.py` masks the observation but recomputes the mask from
   unmasked state, and `get_valid_actions()` has no fog path at all) — note the
   fogged mask becomes *over-permissive* for `found_city` next to a hidden enemy
   city, which the executor then rejects without clearing the unit's moves, so
   this needs a semantics decision and its own OpenSpec change; (b) expose a fog
   flag from `training_civ6.py` (the `fog_of_war_enabled` knob in
   `orchestrator/config.py` and four YAMLs is inert — no consumer builds an env
   from it); (c) wire `DeterminizedMCTS` into self-play — **blocked on N-player
   MCTS**: its `search` expects a per-player value dict, which only
   `NPlayerMCTS.search` returns; `Civ6MCTS`/`GumbelMCTS` return scalars.
3. **Coverage**: Bring omitted modules (`serving/`, `inference/`, cloud infra) under the gate so it
   can keep tightening (gate re-tightened 80% → 85% on 2026-07-13 at 89.69% measured; omit-list
   shrink is the remaining work); `serving/` now has unit tests but stays omitted (needs the `web` extra,
   absent from the default-CI `[dev]` install)
4. **Full Civ6 integration**: the bridge is the most complete subsystem here
   (15 files / 9,127 Python lines + 7,355 Lua, all 17 action types implemented,
   no TODOs) — not "continue scaling" from scratch. Remaining: sustained
   multi-turn live validation (only 4 turns verified, on random weights) and
   the `UncivInterface.parse_multiplayer_save` stub.

---

## SECTION 14: KNOWLEDGE BASE PROTOCOL

This project uses a structured knowledge base in `kb/` for session continuity
across a three-role workflow pipeline: Planning → DEV-SQE → Pre-PR.

### 14.1 Directory Structure
- `kb/summaries/` — Rolling summaries per workflow role
- `kb/sessions/` — Session state with resume pointers (one mutable file per role)
- `kb/ledger/` — Append-only JSONL event log (shared across all roles)
- `kb/handoffs/` — Immutable cross-role handoff documents

### 14.2 Workflow Roles

| Role | Purpose | Upstream | Downstream |
|------|---------|----------|------------|
| **Planning** | Task breakdown, arch decisions, acceptance criteria | — | DEV-SQE |
| **DEV-SQE** | Implementation, testing, quality engineering | Planning | Pre-PR |
| **Pre-PR** | Validation, PR preparation, documentation | DEV-SQE | (PR ready) |

### 14.3 Handoff Flow
- Planning → DEV-SQE: `PLANNING-HANDOFF` event + handoff doc
- DEV-SQE → Pre-PR: `DEV-SQE-HANDOFF` event + handoff doc
- DEV-SQE → Planning: `DESIGN-GAP` event (feedback loop)
- Pre-PR → DEV-SQE: `QA-GAP` event (feedback loop)
- Pre-PR → Planning: `SCOPE-CHANGE` event (feedback loop)
- Any role: `VALIDATION-RUN` event when a training/evaluation run produces
  committed evidence under `docs/evidence/` (records that the code was
  actually executed, not merely changed)

### 14.4 Handoff Skills
- Planning: `.claude/skills/planning-handoff/SKILL.md`
- DEV-SQE: `.claude/skills/dev-sqe-handoff/SKILL.md`
- Pre-PR: `.claude/skills/pre-pr-handoff/SKILL.md`

### 14.5 Rules
1. NEVER overwrite or truncate `kb/ledger/ledger.jsonl` — only append
2. Handoff files in `kb/handoffs/` are immutable once created
3. Session state files (`kb/sessions/*.state.json`) are the ONLY mutable state per role
4. Summary files are append-at-top with a cap of 20 entries
5. All JSONL entries must be valid JSON — validate after writing
6. In Agent Teams, designate ONE teammate (the team lead) as the sole writer
   to `kb/ledger/ledger.jsonl` to avoid race conditions
7. When spawning Agent Teams, each teammate should read relevant KB summaries
   before starting work
8. Every session that does meaningful work MUST run its role's handoff skill
   before stopping
9. Cross-role handoffs create BOTH a ledger event AND a handoff doc

---

## SECTION 15: CLAUDE CODE HARNESS

### 15.1 Overview

The project includes a Claude Code harness in `.claude/` that provides machine-enforceable
configuration for permissions, hooks, and environment variables. This complements the
human-readable instructions in this CLAUDE.md file.

### 15.2 Files

| File | Scope | Git | Purpose |
|------|-------|-----|---------|
| `.claude/settings.json` | Project | Committed | Team-wide permissions, hooks, env vars |
| `.claude/settings.local.json` | Personal | Gitignored | Personal overrides (API tokens, etc.) |
| `.claude/skills/*/SKILL.md` | Project | Committed | Handoff skills (KB workflow), plus `verify` and `negative-control` |
| `.claude/hooks/*.sh` | Project | Committed | The five hook scripts registered in `settings.json` (SessionStart, PreToolUse, two PostToolUse, PreCompact) |

### 15.3 What the Harness Does

**SessionStart hook:** Verifies Python >= 3.11, installs dev dependencies
(`pip install -e '.[dev]'`), and reports what is *actually* true: install failure with the
log tail, any missing required/optional package, and any tool/interpreter split. That last
one matters — a `mypy` resolved from an isolated tool venv cannot see this project's type
stubs and reports dozens of phantom `import-untyped` errors, so when the hook flags a split,
use `python3 -m mypy` (which is what `make typecheck` does). Note `.[dev]` can fail on the
`chess` source build, which aborts the whole extra; the hook now says so instead of
reporting "pytest ready" regardless.

**PostToolUse hooks (Write|Edit):** Two run. `posttooluse_lint.sh` runs `ruff check --fix`
and `ruff format` on the edited Python file (deferring F401/RUF100 — see §5.3), so manual
linting after edits is unnecessary. `posttooluse_governance.sh` runs the matching governance
validator for the edited path: `--path` mode for `openspec/`, `kb/` and skills, and a
whole-repo `check_doc_sync.py` (~95 ms) for documentation, `pyproject.toml` and
`alpha_zero/__init__.py`, so doc drift surfaces at edit time rather than at CI.

**PreCompact hook:** Before context compaction, dumps KB session state, recent ledger entries,
and git diff stats into the compacted context. Ensures workflow continuity across compactions.

**Stop hook:** Reminds the agent to run the appropriate handoff skill (Section 14) before
ending a session that did meaningful work.

**Permissions:** Common safe operations (pytest, ruff, mypy, git, pip, make) are pre-approved.
Dangerous operations are denied: `.env`/`.env.local` writes via the file tools
(`Edit(.env)`+`Write(.env)` — `Edit` rules already cover the Write tool, and the
explicit `Write` entries make the intent unambiguous), force-push (via
`pretooluse_guard.sh`, allowing `--force-with-lease`), and cloud ops
(gcloud/gsutil/kubectl/docker push). Note: a Bash-mediated write to `.env`
(e.g. `echo > .env`) is not pattern-blocked; it is covered by convention and the
guard's heuristic scope, not a sandbox.

> **A note on "agents" and "tools" in this repo.** There is no `.claude/agents/` directory —
> this project does not define custom Claude Code sub-agents. The word "agents" here refers to
> two distinct things: (1) the **KB workflow roles** (Planning / DEV-SQE / Pre-PR), implemented
> as the handoff **skills** under `.claude/skills/` (Section 14); and (2) the in-repo
> **`alpha_zero/utils/agent_tools.py`** helper library (code-explorer, file-manager,
> task-tracker, debug utilities) that sub-agents can import — documented in Section 9.4 / 5.2.
> The harness "tools" are the hooks + permissions in `.claude/settings.json` described above.

### 15.4 Personal Overrides

Copy and edit `.claude/settings.local.json` for personal settings:

```json
{
  "env": {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_your_token_here"
  }
}
```

This file is gitignored and never committed.

---

*Last updated: 2026-07-25*
*Version: v1.8.0 (enterprise docs overhaul + OpenSpec governance gates)*
*Current Focus: M4 (ROADMAP.md) — N-player training runs, batch self-play in cloud, W&B→MLflow sweep, omit-list shrink. Governance: PROJECT_CHARTER.md*
