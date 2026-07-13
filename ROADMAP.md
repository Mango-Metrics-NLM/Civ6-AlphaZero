# Roadmap — Civ6-AlphaZero

> **Themes, not dates.** Milestone *detail* changes here; the charter
> ([`PROJECT_CHARTER.md`](PROJECT_CHARTER.md) §5) only names the themes and is amended
> far more rarely. Tactical, changeable to-dos live in [`NEXT_STEPS.md`](NEXT_STEPS.md)
> and its reconciliation, [`docs/NEXT_STEPS_ANALYSIS_2026-07.md`](docs/NEXT_STEPS_ANALYSIS_2026-07.md).
> What actually shipped is recorded in [`CHANGELOG.md`](CHANGELOG.md).

| Theme | Name | Status |
|-------|------|--------|
| M1 | Learning Core | ✅ Delivered |
| M2 | Scale & Observability | ✅ Delivered |
| M3 | Live Game Bridge | ✅ Delivered |
| M4 | Multiplayer & Throughput | 🔄 Active |
| M5 | Full Civ6 Mastery | ⏳ Future |

## M1 — Learning Core

AlphaZero on MicroCiv, proven end-to-end. *(Delivered: Phases 1–18, v0.x–v1.5.0.)*

- MicroCiv environment, ResNet + attention networks, hierarchical action space.
- Pluggable MCTS family (Simple, Civ6, Gumbel, Batch) with GumbelMCTS as default.
- Training convergence validated after the v1.3.0 weight-continuity fix
  (`scripts/validate_convergence.py`).

## M2 — Scale & Observability

Distributed cloud training with full experiment telemetry. *(Delivered: Phases 7, 19–21, v1.6.0.)*

- 3-phase Vertex AI pipeline (self-play → train → eval), Docker images, GCS-backed
  storage and replay, Spot-VM preemption handling.
- MLflow monitoring layer (`alpha_zero/monitoring/`), ELO tournament system,
  post-run evaluation and analysis scripts.
- 50-iteration production runs completed; 269 model versions in the GCS lineage.

## M3 — Live Game Bridge

A trained agent plays a real, unmodified Civilization VI. *(Delivered: v1.7.0–v1.8.0.)*

- Lua mod + file-IPC bridge, `BridgeAgentLoop`, `BridgeMCTSEnv` (v1.7.0).
- Headless FireTuner socket transport (`alpha_zero/bridge/tuner_socket_bridge.py`);
  live autonomous turns verified 2026-07-12 (v1.8.0).
- LLM eval harness gating `StrategyAdvisor` output contracts (`make eval-llm`).

## M4 — Multiplayer & Throughput

Make N-player training real and self-play fast; pay down debt. *(Active.)*

- Run and validate 3–4 player training (pipeline landed 2026-07-05:
  `--num_players` / `--scoring_method`; Max-N backprop and opponent modeling in
  self-play remain).
- Wire `BatchGumbelMCTS` + `VectorizedMicroCivEnv` into the cloud containers
  (5–10× self-play throughput).
- Land the champion-load fix on the cloud training path (PR #72) and re-run the
  convergence gate.
- ~~Finish the W&B → MLflow sweep~~ *(done 2026-07-13: SDK swap verified complete,
  configs/toggles unified with legacy env fallback, dead suites replaced by MLflow
  equivalents; `run_sweep.py`/Optuna port remains)*; add fixtures for the skipped
  hierarchical / state-extraction / training-smoke tests.
- Bring omitted modules (`serving/`, `inference/`, cloud infra) under the coverage
  gate (re-tightened 80% → 85% on 2026-07-13; omit-list shrink remains).

## M5 — Full Civ6 Mastery

Sustained, skilled play of the full game. *(Future.)*

- Fog of war / partial observability via information-set sampling (gap G10).
- Full-scale 84-channel encoder and hierarchical action space against the real game.
- Sustained multi-hour live sessions with a trained checkpoint (beyond the v1.8.0
  demo action space).
- MuZero-vs-AlphaZero comparison under PBT; LLM-assisted diplomacy — both remain
  opt-in per charter NG-2.

---

*Cross-links: [`PROJECT_CHARTER.md`](PROJECT_CHARTER.md) (governance, non-goals,
invariants) · [`NEXT_STEPS.md`](NEXT_STEPS.md) (tactical) ·
[`CHANGELOG.md`](CHANGELOG.md) (shipped).*
