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
- [ ] Coverage >= 80%: `pytest --cov=alpha_zero --cov-report=term-missing` shows >= 80% (gate enforced in CI and `pyproject.toml`; some infra/web modules are intentionally omitted — see `[tool.coverage]`)
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
  - Coverage >= 80% on core modules
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
- Root-level Python files (setup.py, pyproject.toml)
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

# 1. Lint check
ruff check alpha_zero/

# 2. Type check
mypy alpha_zero/ --ignore-missing-imports

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
2. Coverage check: pytest --cov=alpha_zero shows >= 80% on changed modules
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
| `python -m alpha_zero.training_civ6 --dry_run` | Training smoke test |
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

#### Cloud Deployment Commands

| Command | Purpose |
|---------|---------|
| `make build-all` | Build all Docker images |
| `make push` | Push images to Artifact Registry |
| `make local-test` | Run local docker-compose test |
| `make local-smoke` | Run smoke test (1 game + 1 step) |
| `docker-compose up -d` | Start local testing environment |
| `python -m infra.vertex_launcher --mode=selfplay --iteration=0 --dry-run` | Self-play job dry run |
| `python -m infra.vertex_launcher --mode=train --iteration=0 --dry-run` | Training job dry run |
| `python -m infra.vertex_launcher --mode=eval --iteration=0 --dry-run` | Evaluation job dry run |
| `python -m alpha_zero.cloud.selfplay_worker --help` | Self-play worker options |
| `python -m alpha_zero.cloud.trainer --help` | Trainer options |
| `python -m alpha_zero.cloud.evaluator --help` | Evaluator options |

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
| `pytest tests/cloud/unit/test_evaluator_stagnation.py -v` | Test promotion-stagnation detection |
| `pytest tests/unit/test_replay_buffer.py -v` | Test replay buffers |
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
| 2026-02-18 | Hex crystalline encoder | Hex-topology-aware spatial encoder, axial coordinates, ring-distance PE |
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
| 2026-05-30 | Training-stagnation detection (harness engineering) | Reuses existing `EloPlateauDetector` in the orchestrator (`StagnationConfig`, opt-in, ELO/win-rate signal — NOT loss) + new `count_consecutive_non_promotions` promotion check in `cloud/evaluator.py`. Addresses §13 zero-promotion failure. Env: `CIV6AZ_STAGNATION_*`, `CIV6_STAGNATION_PATIENCE`. All default OFF. |
| 2026-05-30 | PreToolUse prohibited-action guard | `.claude/hooks/pretooluse_guard.sh` enforces §3.2 (no commits/pushes to `main`, >1000-game cap) — turns prose into a machine deny. |
| 2026-05-30 | Pre-PR review in isolated subagent | Pre-PR handoff skill dispatches review to a fresh `Agent`/`code-review` for generator–evaluator separation (independent context). |

### 5.3 Known Issues

| Issue | Workaround |
|-------|------------|
| No GPU in dev environment | Training works on CPU (slower) |
| torch not installable in restricted web sessions | PyTorch index blocked by network policy; torch-dependent tests run in CI, not locally. Pure logic still verifiable via `python -m py_compile` + `ruff`. |
| `PostToolUse` ruff `--fix` can strip imports mid-task | If you add an import in one edit and its first use in a later edit, the autofix removes it as unused. Re-add the import after the use exists; see guardrail audit in `docs/HARNESS_ENGINEERING_ANALYSIS.md` §6. |
| SimpleMCTS random rollout is weak | Use neural network MCTS after initial training |
| Windows multiprocessing spawn | Use `if __name__ == "__main__"` guard |
| pytest-xdist not always installed | Tests run without parallel flag if missing |
| `serving/` + `inference/` omitted from coverage gate | They require the `web` extra (fastapi/uvicorn), absent from the default-CI `[dev]` install, so they cannot be imported in CI's coverage job. `serving/` has unit tests (`tests/unit/test_serving/`, `importorskip`-guarded) but stays in the omit list until CI installs the `web` extra. |
| Coverage instrumentation crashes under nightly numpy 2.4 + coverage C-tracer | Local-only sandbox issue ("cannot load module more than once"); run tests without `--cov`, or use pinned numpy. CI uses pinned versions and is unaffected. |
| MicroCiv hardcoded for 2 players | Multiplayer module built, but env needs G7/G8/G9 changes |
| 12 DDP tests xfailed | Need rewrite with `torch.multiprocessing.spawn` |
| mlruns/ file-store deprecated | Run `scripts/migrate_mlflow_to_sqlite.py` to migrate to `sqlite:///mlflow.db` |
| test_deployment_infra.py: 6 failures | Require .env file not present locally; skip with `pytest -m 'not env_required'` |
| Two divergent `WandbConfig` classes (`orchestrator/config.py` + `utils/wandb_logger.py`) | Unification deferred; `MLflowConfig` is single-source to avoid repeating the issue. |
| W&B -> MLflow swap is partial | `cloud/`, `pbt/`, `evaluation/`, `optimizations/` still call wandb directly. Tracked as a follow-up sweep. |

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
# 1. Read this file (CLAUDE.md)

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
│   │   ├── civ6_mcts.py         # MCTS implementation
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
│   ├── cloud/                   # Cloud-native components (NEW)
│   │   ├── __init__.py
│   │   ├── config.py            # CloudConfig, MachineProfile
│   │   ├── preemption.py        # PreemptionHandler for Spot VMs
│   │   ├── selfplay_worker.py   # Self-play worker entry point
│   │   ├── trainer.py           # Trainer entry point
│   │   └── evaluator.py         # Evaluator entry point
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
│   ├── bridge/                  # Civ6 Lua Mod Bridge (9 files, file IPC)
│   ├── hierarchical/            # Hierarchical Goal Planning (19 files)
│   ├── training/                # Training integrations (11 files: MuZero trainer,
│   │                           #   safeguards, diagnostics, convergence monitor)
│   ├── interfaces/              # Game interfaces (4 files, incl. unciv_interface.py)
│   ├── mechanics/              # Game mechanics: diplomacy, trade, espionage,
│   │                           #   religion, government, climate, eras (12 files)
│   ├── evaluation/             # Evaluation + LLM eval harness, llm_harness/ (12 files)
│   ├── gameplay/               # Gameplay systems: terrain, fog of war, tech, combat (8)
│   ├── optimizations/         # Training optimization strategies (9 files)
│   ├── opponents/             # Opponent types & strategies (4 files)
│   ├── spacetimedb/           # SpacetimeDB integration (9 files)
│   ├── inference/             # Inference service: server/client/engine/schemas (5)
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
├── dockerfiles/                 # Container images (NEW)
│   ├── Dockerfile.base          # Base image with PyTorch
│   ├── Dockerfile.selfplay      # Self-play worker image
│   ├── Dockerfile.trainer       # Trainer image
│   └── Dockerfile.evaluator     # Evaluator image
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
├── Makefile                     # Build and deployment targets (NEW)
├── docker-compose.yml           # Local testing environment (NEW)
└── CLAUDE.md                    # This file
```

> **Authoritative module list:** `alpha_zero/` has **27 submodules** as of v1.7.0.
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
    prompt="Analyze this game state...",
    system_prompt="You are a Civ6 strategy expert."
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
| `pip install -e ".[rag]"` | Install with RAG dependencies |
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
```bash
# Self-play worker
python -m alpha_zero.cloud.selfplay_worker \
    --num-games=100 \
    --iteration=5 \
    --worker-id=0 \
    --num-simulations=800

# Trainer
python -m alpha_zero.cloud.trainer \
    --training-steps=1000 \
    --iteration=5 \
    --batch-size=256

# Evaluator
python -m alpha_zero.cloud.evaluator \
    --num-games=400 \
    --iteration=5 \
    --promotion-threshold=0.55
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
# Build all images
make build-all

# Push to Artifact Registry
make setup-artifact-registry
make push

# Local testing
docker-compose up -d
docker-compose logs -f trainer
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

### Current State (v1.7.0)
Training convergence (the v1.3.0 weight-loading fix) has been validated, and the
project has since shipped through v1.7.0:
- **v1.5.0** — convergence validation + MLflow monitoring + batch self-play groundwork
- **v1.6.0** — cloud-deploy prep: Vertex launcher, runbook, GCP OAuth/VM fixes, 85%→80% CI gate
- **v1.7.0** — full-Civ6 local gameplay (`BridgeAgentLoop` wires the agent to Civ6) and the
  `langfuse-eval-harness` LLM eval gate for `StrategyAdvisor` (`make eval-llm`)

### Implemented (substantially built)
- **PBT**: `alpha_zero/pbt/` (10 files) · **Tournament**: `alpha_zero/tournament/` (9 files)
- **Web Interface**: `alpha_zero/web/` (12 files) · **Civ6 Bridge**: `alpha_zero/bridge/` (9 files)
- **Hierarchical Planning**: `alpha_zero/hierarchical/` (19 files)
- **MuZero**: `alpha_zero/core/muzero_network.py`, `training/muzero_*`
- **N-Player Multiplayer**: `alpha_zero/multiplayer/` (18 files)
- **Inference/Serving**: `alpha_zero/inference/` (5 files), `alpha_zero/serving/` (FastAPI server)

### Genuine Next Steps
1. **N-player MicroCiv**: Fix hardcoded 2-player env (gaps G7/G8/G9)
2. **DDP test fixes**: Re-enable the 12 xfailed distributed tests (`tests/unit/test_distributed`)
3. **Batch self-play in cloud**: Wire BatchGumbelMCTS into cloud containers (5-10x throughput)
4. **Fog of war**: Information set sampling for partial observability (gap G10)
5. **Coverage**: Bring omitted modules (`serving/`, `inference/`, cloud infra) under the gate so it
   can re-tighten 80% → 85%; `serving/` now has unit tests but stays omitted (needs the `web` extra,
   absent from the default-CI `[dev]` install)
6. **Full Civ6 integration**: Continue scaling the Lua mod bridge toward the real game

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
| `.claude/skills/*/SKILL.md` | Project | Committed | Handoff skills for KB workflow |

### 15.3 What the Harness Does

**SessionStart hook:** Automatically verifies Python >= 3.11 and installs dev dependencies
(`pip install -e '.[dev]'`) at the start of every session. Reports environment status.

**PostToolUse hook (Write|Edit):** After any Python file is written or edited, automatically
runs `ruff check --fix` and `ruff format` on that file. This means manual linting after
edits is no longer necessary during Claude Code sessions.

**PreCompact hook:** Before context compaction, dumps KB session state, recent ledger entries,
and git diff stats into the compacted context. Ensures workflow continuity across compactions.

**Stop hook:** Reminds the agent to run the appropriate handoff skill (Section 14) before
ending a session that did meaningful work.

**Permissions:** Common safe operations (pytest, ruff, mypy, git, pip, make) are pre-approved.
Dangerous operations (writing .env, force-push, cloud ops) are denied.

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

*Last updated: 2026-06-20*
*Version: v1.7.0 (full-Civ6 local gameplay + LLM eval harness)*
*Current Focus: N-player MicroCiv env, DDP test re-enable, and re-tightening the coverage gate*
