# Roadmap — Civ6-AlphaZero

> **Themes, not dates.** Milestone *detail* changes here; the charter
> ([`PROJECT_CHARTER.md`](PROJECT_CHARTER.md) §5) only names the themes and is amended
> far more rarely. Tactical, changeable to-dos live in [`NEXT_STEPS.md`](NEXT_STEPS.md)
> and its reconciliation, [`docs/NEXT_STEPS_CASE_2026-08.md`](docs/NEXT_STEPS_CASE_2026-08.md).
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

- 3-phase Vertex AI pipeline (self-play → train → eval) in a single image
  (`dockerfiles/Dockerfile.vertex`, routed by `alpha_zero.cloud.gcp.entrypoint`),
  GCS-backed storage and replay, Spot-VM preemption handling.
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

> **Order of work (2026-09-04):** `kb/specs/20260904-correctness-first-plan.md`,
> which amends the sequencing of `kb/specs/20260902-validated-learning-plan.md`
> (still the source for each workstream's content) and supersedes the remediation
> list in [`docs/audits/2026-09-04-standards-audit.md`](docs/audits/2026-09-04-standards-audit.md).
> The bullets below are M4's inventory; those specs sequence them.
>
> The 2026-09-02 order put a seeded learning-trend run first, behind a draw-rate
> fix (measured: `docs/evidence/draw_rate_sweep/`). A new workstream **WS-S
> (search correctness)** now comes before both: the default search was defective on
> every deployed path — wrong-signed value backup, a one-hot policy target, and
> self-play/evaluation seeds with no iteration term — so no number produced by it
> was evidence of anything, and ~70% draws is what two agents that cannot tell a win
> from a loss would produce. S1–S4 and S6 landed on 2026-09-04
> ([`CHANGELOG.md`](CHANGELOG.md)); the learning run and the draw-rate re-measure
> follow on the fixed code.

- Wire `validate_encoder_state` into the cloud workers to prevent silent architecture cross-loading (done 2026-08-10).
- Complete 800-game local GPU A/B arena comparison and publish the hex-crystalline encoder to the Hugging Face Hub (done 2026-08-10).
- Run and validate 3–4 player training (pipeline landed 2026-07-05:
  `--num_players` / `--scoring_method`; Max-N backprop and opponent modeling in
  self-play remain).
- Wire `BatchGumbelMCTS` + `VectorizedMicroCivEnv` into the cloud containers
  (5–10× self-play throughput). The wiring exists in `cloud/gcp/selfplay_worker.py`
  and is tested; what remains is a config that selects it and a measured
  benchmark — the 5–10× figure is a target, not an observation. Batch
  trajectories omitted `action_masks` until 2026-08-27, so any data generated
  by this path before then trained with masked policy loss disabled.
- ~~Land the champion-load fix on the cloud training path (PR #72)~~ *(landed;
  the mechanism changed — `--model_version` was removed in favour of
  `--resume_from=latest`, which resolves the incumbent from GCS in
  `vertex_orchestrator._resolve_resume_token`)*. Re-running the convergence gate
  is still open, and now also needs to cover the 2026-08-24 value-sign fix.
- ~~Finish the W&B → MLflow sweep~~ *(done 2026-07-13: SDK swap verified complete,
  configs/toggles unified with legacy env fallback, dead suites replaced by MLflow
  equivalents; `run_sweep.py` ported to Optuna)*; add fixtures for the skipped
  hierarchical / state-extraction / training-smoke tests.
- Bring omitted modules under the coverage gate (re-tightened 80% → 85% on
  2026-07-13; omit-list shrink remains). **`cloud/` is now measured** — not by
  un-omitting it, but by a separate per-package floor (`coverage-floors.json`,
  `.coveragerc-cloud`) enforced in CI's `cloud-tests` job: floor 58%, measured
  59.7% on 2026-09-02. `serving/` and `inference/` still need the `web` extra in
  the coverage job before they can follow.
- Achieve 100% clean mypy type safety across all 350 files (done 2026-08-10).
- **Tech-debt reduction and hardening** (`kb/specs/20260902-tech-debt-hardening.md`).
  Core scope landed 2026-09-02: measurement-first debt metrics with a ratchet
  (`scripts/measure_debt.py`, `scripts/audit_import_graph.py`), workflow and hook
  hardening, `action_space_size` derived from one source instead of a wrong 381
  literal, safe-by-default checkpoint loading, Tier A deletion (44 modules /
  21,903 lines, plus the four per-role Dockerfiles, `docker-compose.yml`,
  `setup.py` and `sweep_config.yaml`), and the cloud coverage floor above.
  Remaining, in the maintainer's hands or a later PR:
  - **WS0** — rotate the leaked W&B key and close the issue backlog (see the
    known-issue row in `CLAUDE.md` §5.3 and `docs/runbooks/secret-scanning.md`).
  - **Tier B** — fourteen yes/no keep-or-delete decisions, 18,133 lines.
  - **WS3.2** — collapse the env prefix to `CIV6AZ_` only, behind a two-release
    warn-and-map shim in `env_loader.py`.
  - **WS5** — decompose the seven remaining god files, each behind a re-export
    shim with a patchability test.
  - **WS6** — consolidate duplicate definitions (`setup_logging`, `create_mcts`,
    Elo, `MCTSConfig`) and settle on one dependency manifest.
  - **WS7** — layering test, wider type-check scope, skip-site burndown,
    `per-file-ignores` 23 → ≤ 12.

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
