# Module Reference — `alpha_zero/`

**Authoritative list of `alpha_zero/` submodules.** This is the single source of truth
referenced by `README.md` and `CLAUDE.md` SECTION 7. Update it whenever a submodule is added
or removed.

- **As of:** v1.7.0, 2026-06-20
- **Submodule count:** 27
- **Counting method:** recursive `*.py` files, excluding `__pycache__`
  (`find alpha_zero/<mod> -name '*.py' -not -path '*__pycache__*' | wc -l`).

| Module | Files | Responsibility |
|--------|------:|----------------|
| `core` | 30 | Neural networks (ResNet/transformer), MCTS variants (Gumbel/DUCT/MuZero), action space |
| `utils` | 27 | Logging, config, LLM client, factories, test runner, `agent_tools`, `.env` loader |
| `rag` | 22 | Retrieval-augmented generation pipeline: vector stores, processors, agent integration (§11) |
| `hierarchical` | 19 | Hierarchical goal planning: decomposition, controllers, encoders |
| `multiplayer` | 18 | N-player MCTS (Max-N/Paranoid/Best-Reply), turn manager, opponent modeling, scoring |
| `evaluation` | 12 | MCTS evaluation + `llm_harness/` callable target for the LLM eval gate |
| `envs` | 12 | MicroCiv environment, vectorized env, state→tensor encoders |
| `mechanics` | 12 | Game mechanics: diplomacy, trade, espionage, religion, government, climate, eras |
| `training` | 11 | MuZero trainer/integration, training safeguards, diagnostics, convergence monitor |
| `cloud` | 48 | Cloud-native workers (self-play/trainer/evaluator) + `gcp/` Vertex AI orchestration |
| `orchestrator` | 10 | `TrainingOrchestrator`, scheduler, checkpointer, RAG integration, strategy advisor |
| `pbt` | 10 | Population-based training: population manager, mutation, fitness, MuZero hyperparams |
| `bridge` | 9 | Civ6 Lua mod bridge: file-based IPC protocol |
| `distributed` | 9 | Process-based distributed training: actor, learner, coordinator, shared storage |
| `optimizations` | 9 | Training optimization strategies |
| `spacetimedb` | 9 | SpacetimeDB integration |
| `storage` | 9 | Storage backends (local/GCS), streaming replay buffer, factory |
| `tournament` | 9 | ELO rating, matchmaking, analysis, tournament telemetry |
| `web` | 12 | FastAPI + WebSocket game UI: routes, static assets |
| `gameplay` | 8 | Gameplay systems: terrain, fog of war, tech tree, combat |
| `human_interface` | 6 | Human gameplay interface |
| `monitoring` | 6 | MLflow experiment tracking: `MLflowConfig`, `MLflowLogger`, metric helpers, dashboards |
| `inference` | 5 | Inference service: API server, client, engine, schemas |
| `interfaces` | 4 | Game interface contracts + Unciv JSON interface |
| `opponents` | 4 | Opponent types & strategies |
| `integration` | 2 | Integration bridge / game connector |
| `serving` | 2 | FastAPI inference server (`web` extra) — see coverage note below |

## Coverage notes

Some modules are intentionally excluded from the coverage gate (`[tool.coverage]` in
`pyproject.toml`) because they require optional extras or external services that the default
CI `[dev]` install does not provide:

- **`serving/`** — requires the `web` extra (fastapi/uvicorn). It has unit tests
  (`tests/unit/test_serving/`, guarded with `pytest.importorskip`) but stays omitted until CI
  installs the `web` extra into the coverage job.
- **`inference/`, `cloud/`, `web/routes/`, `spacetimedb/`, RAG/LLM infra** — likewise gated on
  optional extras (gcp, web, spacetimedb, llm) or live services.

The gate (80%) re-tightens toward 85% as these modules are brought under coverage.
