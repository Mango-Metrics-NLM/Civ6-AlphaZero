# Module Reference — `alpha_zero/`

**Authoritative list of `alpha_zero/` submodules.** This is the single source of truth
referenced by `README.md` and `CLAUDE.md` SECTION 7. Update it whenever a submodule is added
or removed.

- **As of:** v1.8.0, 2026-09-02 (Tier A dead-code removal)
- **Submodule count:** 29
- **Counting method:** recursive `*.py` files, excluding `__pycache__`
  (`find alpha_zero/<mod> -name '*.py' -not -path '*__pycache__*' | wc -l`).

| Module | Files | Responsibility |
|--------|------:|----------------|
| `core` | 35 | Neural networks (ResNet/transformer), MCTS variants in `mcts/` (PUCT/Simple/Gumbel/MuZero), action space |
| `utils` | 25 | Logging, config, LLM client, factories, test runner, `agent_tools`, `.env` loader, device resolution, safe checkpoint loading (`safe_load.py`) |
| `rag` | 22 | Retrieval-augmented generation pipeline: vector stores, processors, agent integration (§11) |
| `hierarchical` | 19 | Hierarchical goal planning: decomposition, controllers, encoders |
| `multiplayer` | 18 | N-player MCTS (Max-N/Paranoid/Best-Reply), turn manager, opponent modeling, scoring |
| `evaluation` | 12 | MCTS evaluation, promotion scoring SSOT (`scoring.py`) and policy registry (`promotion.py`), + `llm_harness/` callable target for the LLM eval gate |
| `hexcrystal` | 7 | Hex-topology spatial encoders + encoder registry — torch-only, standalone-reusable; optional Hub mixin (`hub` extra) |
| `hub` | 5 | Hugging Face Hub integration: model card generation, dataset exporter, upload utility, hub mixins (`hub` extra) |
| `envs` | 11 | MicroCiv environment, vectorized env, state→tensor encoders |
| `mechanics` | 12 | Game mechanics: diplomacy, trade, espionage, religion, government, climate, eras |
| `training` | 9 | MuZero trainer/integration, training safeguards, convergence monitor |
| `cloud` | 28 | `gcp/` Vertex AI orchestration and the deployed self-play / training / arena workers |
| `orchestrator` | 10 | `TrainingOrchestrator`, scheduler, checkpointer, RAG integration, strategy advisor |
| `pbt` | 10 | Population-based training: population manager, mutation, fitness, MuZero hyperparams |
| `bridge` | 15 | Civ6 Lua mod bridge: file IPC + headless FireTuner socket transport |
| `distributed` | 8 | Process-based distributed training: actor, learner, coordinator, shared storage |
| `optimizations` | 7 | Training optimization strategies + Optuna hyperparameter search |
| `spacetimedb` | 9 | SpacetimeDB integration |
| `storage` | 9 | Storage backends (local/GCS), streaming replay buffer, factory |
| `tournament` | 9 | ELO rating, matchmaking, analysis, tournament telemetry |
| `web` | 12 | FastAPI + WebSocket game UI: routes, static assets |
| `gameplay` | 8 | Gameplay systems: terrain, fog of war, tech tree, combat |
| `human_interface` | 4 | Human gameplay interface |
| `monitoring` | 6 | MLflow experiment tracking: `MLflowConfig`, `MLflowLogger`, metric helpers, dashboards |
| `inference` | 4 | Inference service: client, engine, schemas (HTTP server lives in `serving/`) |
| `interfaces` | 4 | Game interface contracts + Unciv JSON interface |
| `opponents` | 2 | Opponent types & strategies |
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

The gate (85%) re-tightens as these modules are brought under coverage.
