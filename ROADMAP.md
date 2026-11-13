# Roadmap — Civ6-AlphaZero

> **Themes, not dates.** Milestone *detail* changes here; the charter
> ([`PROJECT_CHARTER.md`](PROJECT_CHARTER.md) §5) only names the themes and is amended
> far more rarely. Actionable work lives in [Now / Next / Later](#now--next--later)
> below and in GitHub Issues; the scope of an in-flight change lives under
> `openspec/changes/`; the live-bridge how-to is [`docs/LIVE_CIV6.md`](docs/LIVE_CIV6.md).
> What actually shipped is recorded in [`CHANGELOG.md`](CHANGELOG.md).

**Delivered and validated are different axes.** The table records both. A
milestone is *delivered* when its code shipped and is exercised by tests; it is
*validated* only when a committed measurement under `docs/evidence/` supports the
claim the milestone makes. Every evidence artifact committed so far was produced
on code that predates the 2026-09-04 search-correctness fixes (wrong-signed value
backup, one-hot policy target, iteration-free seeds), so none of them is evidence
about learning.

| Theme | Name | Delivery status | Validation (evidence) |
|-------|------|-----------------|-----------------------|
| M1 | Learning Core | ✅ Delivered | ❌ **Not demonstrated.** All runs predate the 2026-09-04 search fixes; the last committed run scored 0–10% wins vs random with no trend ([`docs/evidence/validation_iteration_zero/`](docs/evidence/validation_iteration_zero/validation_iteration_zero_report.md)). A seeded post-fix learning run is the first item under *Next*. |
| M2 | Scale & Observability | ✅ Delivered; ⏸ **Vertex/GCP path dormant** (frozen 2026-11-11) | ⚠️ **Operationally exercised**, outputs pre-fix. 50-iteration Vertex AI runs completed; the promotion gate cannot promote at the measured 73.5% draw rate ([`docs/runbooks/promotion-gate.md`](docs/runbooks/promotion-gate.md)). |
| M3 | Live Game Bridge | ✅ Delivered | ⚠️ **Command delivery proven, demo-grade.** Four `END_TURN`s on random weights, 2026-07-12; turn-cycle stall open ([`docs/LIVE_CIV6.md`](docs/LIVE_CIV6.md)). |
| M4 | Multiplayer & Throughput | 🔄 Active | — Targets, not measurements. The batch self-play path has no benchmark artifact. |
| M5 | Full Civ6 Mastery | ⏳ Future | — |

## Now / Next / Later

Sequencing authority for v2.1: [`kb/specs/20261111-local-first-and-branching-plan.md`](kb/specs/20261111-local-first-and-branching-plan.md)
(local-first program; gap register `kb/specs/20261111-local-first-gap-register.md`). It carries forward
[`kb/specs/20260904-correctness-first-plan.md`](kb/specs/20260904-correctness-first-plan.md)
(search correctness → draw-rate re-measure → seeded learning run), which amends
[`kb/specs/20260902-validated-learning-plan.md`](kb/specs/20260902-validated-learning-plan.md).
Every *Now* bullet names the issue or OpenSpec change that tracks it
(`check_doc_sync` D9 enforces this).

### Now

- `openspec/changes/branching-and-release`: the dev / qa / main branch model,
  rulesets and release gates of the local-first plan.
- `openspec/changes/harden-governance-gates`: governance-gate hardening.
- `openspec/changes/fix-gumbel-budget`, `openspec/changes/fix-move-mask`: the
  search-budget and move-mask fixes A1 runs on.
- `openspec/changes/checkpoint-provenance`, `openspec/changes/eval-decoupling-diagnostics`:
  checkpoint provenance and `checkpoint_0`; decoupled in-loop evaluation and the
  prequential diagnostics A1's target-sharpness trigger reads.
- `openspec/changes/benchmark-evidence-producer`: the benchmark as the A1
  evidence producer (pair and hull intervals, per-arm `--out`, shards).

### Next — v2.1 "First validated learning run"

- **A1**, registered in [`docs/EVALUATION.md`](docs/EVALUATION.md) section 7.1
  (2026-11-11): 20 iterations × 24 self-play games at board 8 / 60 turns on the
  Gumbel search, then a 200-game paired arena of the final checkpoint against its
  own random initialisation, with symmetry and positive controls; 980 games
  priced by `configs/run_cards/a1.yaml`. The verdict is the registered table,
  whichever way it comes out. Before it runs, T0b measures the positive-control
  reference and the wall clock.
- **A1-R** at seed 20261112 (`configs/run_cards/a1r.yaml`); it must also pass
  before any public claim that the agent learns. A FAIL is followed by A1-P (the
  PUCT search at the same seed).
- Re-measure the draw rate under *trained* post-fix play (the committed sweep is
  random play on pre-fix code), then set `max_turns` on the deployed path from
  that number.

Done for v2.1 and no longer listed: the evaluation plumbing (seeded, paired
arena; additive draw-aware metrics; promotion feasibility), the Gumbel
target-sharpness probe and completed-Q alignment memo (decision pending with
the maintainer; A1 registers the current settings), and the seed-overlap memo
(`kb/specs/20261111-seed-overlap-memo.md`).

### Later

- M4: N-player training runs, a measured batch self-play benchmark, the remaining
  tech-debt workstreams (below).
- M5: fog of war via information-set sampling, the full-scale encoder and action
  space against the real game, sustained live play with a trained checkpoint.

---

## M1 — Learning Core

AlphaZero on MicroCiv, built end-to-end. *(Delivered: Phases 1–18, v0.x–v1.5.0.)*

- MicroCiv environment, ResNet + attention networks, hierarchical action space.
- Pluggable MCTS family (Simple, Civ6, Gumbel, Batch) with GumbelMCTS as default.
- Weight continuity across iterations fixed in v1.3.0 (`scripts/validate_convergence.py`
  exercises the loop). Whether the loop *learns* has not been demonstrated — see the
  Validation column above; the 2026-09-04 search fixes make every earlier run moot.

## M2 — Scale & Observability

Distributed cloud training with full experiment telemetry. *(Delivered: Phases 7, 19–21, v1.6.0.)*

> **DORMANT (2026-11-11):** the Vertex/GCP half of this milestone is frozen — code kept, not
> deleted, no local/cloud parity claimed; training runs locally. See
> [`kb/specs/20261111-local-first-and-branching-plan.md`](kb/specs/20261111-local-first-and-branching-plan.md)
> and charter amendment A-1. The history below is preserved.

- 3-phase Vertex AI pipeline (self-play → train → eval) in a single image
  (`dockerfiles/Dockerfile.vertex`, routed by `alpha_zero.cloud.gcp.entrypoint`),
  GCS-backed storage and replay, Spot-VM preemption handling.
- MLflow monitoring layer (`alpha_zero/monitoring/`), ELO tournament system,
  post-run evaluation and analysis scripts.
- 50-iteration production runs completed on Vertex AI. Their outputs predate the
  2026-08-24 value-sign fix and the 2026-09-04 search fixes, and the promotion
  gate they ran under scores every draw as a loss; the model lineage they produced
  is an operational record, not a success signal.

## M3 — Live Game Bridge

A trained agent plays a real, unmodified Civilization VI. *(Delivered: v1.7.0–v1.8.0.)*

- Lua mod + file-IPC bridge, `BridgeAgentLoop`, `BridgeMCTSEnv` (v1.7.0).
- Headless FireTuner socket transport (`alpha_zero/bridge/tuner_socket_bridge.py`);
  four autonomous `END_TURN`s delivered and acknowledged on 2026-07-12 with a
  randomly initialised network (v1.8.0). How to run it, and what is still open:
  [`docs/LIVE_CIV6.md`](docs/LIVE_CIV6.md).
- Offline contract gate for `StrategyAdvisor` output (`make eval-llm`) — a
  serialisation regression check, not a quality measurement.

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
- Complete 800-game local GPU A/B arena comparison and publish the hex-crystalline encoder to the Hugging Face Hub (done 2026-08-10; the A/B is superseded as a strength comparison — see `docs/evidence/README.md`).
- Run and validate 3–4 player training (pipeline landed 2026-07-05:
  `--num_players` / `--scoring_method`; Max-N backprop and opponent modeling in
  self-play remain).
- Wire `BatchGumbelMCTS` + `VectorizedMicroCivEnv` into the cloud containers as
  the production self-play path. The wiring exists in `cloud/gcp/selfplay_worker.py`
  and is tested; what remains is a config that selects it and a **measured
  benchmark** — no throughput figure is claimed until one is committed under
  `docs/evidence/`. Batch trajectories omitted `action_masks` until 2026-08-27,
  so any data generated by this path before then trained with masked policy loss
  disabled.
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
invariants) · [`docs/LIVE_CIV6.md`](docs/LIVE_CIV6.md) (live bridge how-to) ·
[`docs/evidence/README.md`](docs/evidence/README.md) (committed measurements) ·
[`CHANGELOG.md`](CHANGELOG.md) (shipped). The 2026-08 next-steps reconciliation is
archived at [`docs/archive/NEXT_STEPS_CASE_2026-08.md`](docs/archive/NEXT_STEPS_CASE_2026-08.md).*
