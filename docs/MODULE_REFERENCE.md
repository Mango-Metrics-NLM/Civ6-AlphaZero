# Module Reference — `alpha_zero/`

**Authoritative list of `alpha_zero/` submodules.** This is the single source of truth
referenced by `README.md` and `CLAUDE.md` SECTION 7. Update it whenever a submodule is added
or removed.

- **As of:** v2.0.0, 2026-11-20 (`rules` kernel added and `bridge` updated for GM-B0..GM-B5 research wiring, GameInfo capture, and `--mcts` refusal;
  module set unchanged since `alpha_zero/rules/` landed in GM-K1)
- **Submodule count:** 31
- **Counting method:** recursive `*.py` files, excluding `__pycache__`
  (`find alpha_zero/<mod> -name '*.py' -not -path '*__pycache__*' | wc -l`).

| Module | Files | Responsibility |
|--------|------:|----------------|
| `core` | 36 | Neural networks (ResNet/transformer), MCTS variants in `mcts/` (PUCT/Simple/Gumbel/MuZero) sharing one sign-correct value backup (`mcts/backup.py`), action space |
| `utils` | 30 | Logging, config, LLM client, factories, test runner, `agent_tools`, `.env` loader, device resolution, safe checkpoint loading (`safe_load.py`), per-stream seed derivation (`seeding.py`), worst-case game-count cost models and the 1,000-game cap (`game_budget.py`), the `CIV6AZ_DATA_ROOT` resolver and ACL hardening (`data_root.py`), loopback bind and CORS helpers for the local-first posture (`network.py`), atomic file publication so a killed run leaves no torn checkpoint or result (`atomic_io.py`) |
| `rag` | 22 | Retrieval-augmented generation pipeline: vector stores, processors, agent integration (§11) |
| `hierarchical` | 19 | Hierarchical goal planning: decomposition, controllers, encoders |
| `multiplayer` | 18 | N-player MCTS (Max-N/Paranoid/Best-Reply), turn manager, opponent modeling, scoring |
| `evaluation` | 17 | MCTS evaluation, promotion scoring SSOT (`scoring.py`) and policy registry (`promotion.py`), benchmark protocol + evidence producer (`benchmark.py`, `benchmark_evidence.py`), the A1 protocol hash, positive-control calibration and verdict (`a1_protocol.py`) and its registered commands (`a1_registration.py`), + `llm_harness/` callable target for the StrategyAdvisor eval harness; the reproduction check that compares an interrupted attempt with its re-run, checkpoint tensors and arena games (`reproduction.py`) |
| `hexcrystal` | 7 | Hex-topology spatial encoders + encoder registry — torch-only, standalone-reusable; optional Hub mixin (`hub` extra) |
| `hub` | 5 | Hugging Face Hub integration: model card generation, dataset exporter, upload utility, hub mixins (`hub` extra) |
| `envs` | 11 | MicroCiv environment, vectorized env, state→tensor encoders |
| `mechanics` | 12 | Game mechanics: diplomacy, trade, espionage, religion, government, climate, eras |
| `training` | 10 | MuZero trainer/integration, training safeguards, convergence monitor, diagnostics and the local run record |
| `cloud` | 28 | `gcp/` Vertex AI orchestration and the deployed self-play / training / arena workers |
| `orchestrator` | 10 | `TrainingOrchestrator`, scheduler, checkpointer, RAG integration, strategy advisor |
| `pbt` | 10 | Population-based training: population manager, mutation, fitness, MuZero hyperparams |
| `local` | 6 | Local-first pipeline: pinned source (`git archive`) and non-editable pinned venv for a run, optionally with a pinned torch build, and the running-pinned preflight (`pinned_env.py`); fail-closed GCP isolation (`isolation.py`); the device check that refuses a card whose GPUs the pinned environment cannot provide (`devices.py`); the run-card model and loader (`card.py`); and the run-card driver that prices, pins, probes and runs a `configs/run_cards/` card (`run_card.py`) |
| `bridge` | 24 | Civ6 Lua mod bridge: file IPC, headless FireTuner socket transport (states handed out by sequence number), the agent loop with its per-turn ledger (`turn_ledger.py`, `offer_filter.py`, settings in `agent_loop_config.py`), state differences for the live check tool (`state_diff.py`), research slot registry (`tech_registry.py`), production slots by position (`production_slots.py`), GameInfo snapshot capture (`gameinfo_capture.py`), simulator proxy / `--mcts` refusal (`microciv_proxy.py`, `mcts_policy.py`), and the translation of the search's answer into an offered live command (`search_translator.py`) |
| `distributed` | 8 | Process-based distributed training: actor, learner, coordinator, shared storage |
| `optimizations` | 7 | Training optimization strategies + Optuna hyperparameter search |
| `spacetimedb` | 9 | SpacetimeDB integration |
| `storage` | 9 | Storage backends (local/GCS), streaming replay buffer, factory |
| `tournament` | 9 | ELO rating, matchmaking, analysis, tournament telemetry |
| `web` | 12 | FastAPI + WebSocket game UI: routes, static assets |
| `rules` | 10 | Game rule sets as data: frozen integer-only specs, a strict YAML loader, a structural validator, queries, index registry and hash-pinned built-ins; a zero-game pacing model (`pacing.py`) and an importer that turns a GameInfo table export into a rule set outside the repository (`gameinfo.py`). CLI: `python -m alpha_zero.rules validate\|hash\|pace\|import-gameinfo`. A leaf package; no environment uses it yet (`docs/GAME_RULES.md`) |
| `gameplay` | 8 | Gameplay systems: combat, economy, fog of war, terrain, city growth, victory (no tech tree yet; one is planned as GM-S2) |
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
- **`inference/`, `web/routes/`, `spacetimedb/`, RAG/LLM infra** — likewise gated on
  optional extras (web, spacetimedb, llm) or live services.
- **`cloud/`** — still omitted from the repo-wide gate, but **no longer unmeasured**. Since
  2026-09-02, CI's `cloud-tests` job measures it separately with `.coveragerc-cloud` and holds
  it to the per-package floor in `coverage-floors.json` (**58%**, measured **60.2%** across
  26 files / 5,538 statements on 2026-11-20). A floor may only ever be raised. Reproduce with
  `make test-cloud-coverage`.

The repo-wide gate (85%) re-tightens as the remaining omitted modules are brought under
coverage.
