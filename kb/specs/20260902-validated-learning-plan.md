# Spec: Validated-Learning Plan — the next steps after the tech-debt Core scope (2026-09-02, v1)

**Role:** Planning → DEV-SQE (with maintainer-only items called out)
**Session:** session_01AQAVGXog3wGfPaqu4XHfXX, branch `claude/dev-roadmap-next-steps-fk867n`
**Status:** v1. Built from four independent verification passes over the repository, one
measurement run (`docs/evidence/draw_rate_sweep/`), a literature check, and an
adversarial peer review whose corrections are recorded in §0. Supersedes the
"Recommended Plan" of `docs/NEXT_STEPS_CASE_2026-08.md` (Phase 1 of which landed as
PR #113, Phase 2 of which has not happened) and the four overlapping next-step lists in
`CLAUDE.md` §13, `ROADMAP.md` M4, `NEXT_STEPS.md`, and that case document.

Every number below was measured on 2026-09-02 against commit `50144cc` (default-branch
head, CI run 33670154515 green on all 8 jobs) and the command, file, or artifact that
produced it is stated beside it.

---

## 0. What verification changed (first answer → v1)

The first answer to "what are the next steps" was given from the documentation alone.
Four verification agents, one measurement, and a peer review corrected it as follows.

| First-answer claim | Correction | Evidence |
|---|---|---|
| "The cheapest lever on the draw rate is game length and board size" | **Half right.** `max_turns` is the lever: random-play draws at board 8 fall 0.68 → 0.47 → 0.37 for 30 → 60 → 100 turns. **Board size at 30 turns makes draws worse** (0.68 → 0.87 at 16×16). | `docs/evidence/draw_rate_sweep/report.md` |
| A code-reading review argued the reverse: "lengthening `max_turns` alone will not help; a second city is geometrically near-impossible on 8×8" | **Refuted by measurement.** (1,2) city outcomes go from 6/60 at 30 turns to 26/60 at 100. The geometry constraint is real but not binding; the settler's ~17-turn cost is. | same |
| "The upstream problem is the draw rate" (runbook) | **Confirmed and mechanised.** At 8/30 the mass sits on a (1,1) city tie (41/60); random play draws 0.68 vs 0.735 with trained policies — trained play was not expanding better than random. And the ceiling is structural: cities can never be captured or destroyed (`_execute_attack` removes units only), so once both players found, elimination is impossible. | `envs/microciv.py:318-341`, `:652-702`, `:774-797` |
| "N-player MCTS wiring touches more than three modules and needs confirm-first" | **Wrong.** Branching self-play on `num_players > 2` to `NPlayerMCTS` is **one module** (`training_civ6.py`): `MicroCivEnv` already satisfies `MultiplayerEnvironment`, `self_play_game` discards the second return value, and `select_action` signatures match. The `--fog_of_war` flag is also one module. | `training_civ6.py:596-640`, `multiplayer/mcts.py:78-143`, `multiplayer/protocols.py:35-78` |
| "The cloud path is ready for a shadow-mode run" | **Not ready.** The verdict consults the registry, but `ci_low`/`ci_high` and the verdict record go to `log.debug` only; orchestrator state stores draw-blind `win_rate` only; there is no `--promotion_policy` CLI arg; no config sets the policy; the only budget mechanism (`CostMonitor`) has no importer; no committed YAML fits a ≤1000-game envelope; `selfplay_worker` has no `--seed` and `vertex_orchestrator` has no seed plumbing at all. | `promotion.py:542-574`, `orchestrator_state.py:45,82`, `vertex_orchestrator.py:246-255`, `cost_monitor.py:104-124`, `configs/cloud_training_v1.yaml:21-31`, `selfplay_worker.py:1537-1551` |
| "Run a longer seeded local run" | **Its artifact would still be draw-blind.** `summarize_training_log.py` parses only the `Win rate vs random` line; the draw-aware line at `training_civ6.py:845-855` has no regex; nothing writes `log_dir`; both eval scripts load from GCS by integer version and never import `evaluation/scoring.py`. | `scripts/summarize_training_log.py:49,126-128`; `scripts/run_baseline_eval.py:72-77,364-376` |
| — (not in the first answer) | **New defect:** the action mask marks moves onto occupied tiles legal; the executor rejects them and forfeits the unit's remaining moves. ~25 per game under random play (18,103 in 720 games). | `core/civ6_actions.py:629-635` vs `envs/microciv.py:807-818,549-551` |
| — (not in the first answer) | **`validate_convergence.py` cannot validate a local run**: it drives `training_worker.train_epoch` against a mock GCS backend with synthetic trajectories, is draw-blind and unseeded. | `scripts/validate_convergence.py:28-33,245-262` |
| "Open PRs #69/#49/#32 and ten Dependabot PRs" | All 14 open PRs are `dirty` or stale against the post-Tier-A tree. #95 and #98 are obsolete (codecov removed; release action SHA-pinned). #103 edits 2 of 4 ruff pin sites. #102 raises a floor for a package nothing imports. #106 has 7 content conflicts, none from Tier A. | §7 |

Kept from the first answer without change: the project has never demonstrated learning
(`docs/evidence/validation_iteration_zero/`: final checkpoint below random on 5/5
probes); the promotion default is arithmetically unreachable at the shipped
configuration; `NPlayerMCTS`/`DeterminizedMCTS` have zero production call sites
(15 + 17 test instantiations, 0 in `alpha_zero/` or `scripts/`); the fog action-mask
leak is real and no test covers the mask under fog; promotion-stagnation detection on the
deployed path is absent (the closest thing is an alert-only, draw-blind win-rate stall
check at `vertex_orchestrator.py:2046-2086`).

---

## 1. Premise

1. **CI is green and the gates now mean something.** Run 33670154515 on `50144cc`: 8/8
   jobs including the cloud coverage floor (59.7% ≥ 58%), gitleaks, governance. The
   debt ratchet has not moved (`scripts/measure_debt.py` vs
   `docs/evidence/tech_debt_baseline/baseline.json`: 18 of 19 metrics identical,
   `package_lines` +4 from a docstring commit).
2. **The loop executes but has not been shown to learn.** The only committed training
   run is 5 iterations × 20 games and reports "no learning shown" in its own words. Its
   two structural causes are known: a draw rate that compresses every statistic toward
   0.5, and evaluation plumbing that reports draw-blind numbers.
3. **The August correctness fixes are unvalidated by any run.** Value-sign labelling
   (`0d5adee`), promotion scoring SSOT, batch `action_masks`, seeding — all merged, none
   exercised end to end on the cloud path. The promoted lineage v0–v268 predates all of
   them and is not tagged as such (no `tags`/lineage field exists on `ModelMetadata`).
4. **Two of the three M4 feature bullets are blocked on evidence, not code.** Batch
   self-play is wired and tested but selected by no config and benchmarked by no
   committed artifact. N-player search is a one-module change. Fog is sequenced behind
   N-player search by `DeterminizedMCTS`'s per-player value contract.
5. **The documentation has four next-step lists.** `CLAUDE.md` §13, `ROADMAP.md` M4,
   `NEXT_STEPS.md` (bridge only) and `docs/NEXT_STEPS_CASE_2026-08.md`. The charter
   (§6) names `NEXT_STEPS.md` as "what now" and `ROADMAP.md` as "what next". This spec
   is the single "what now"; `NEXT_STEPS.md` points at it.

---

## 2. Baseline (reproducible)

### 2.1 Learning signal

| Measurement | Value | Source |
|---|---|---|
| Draw rate, trained Gumbel policies, board 8 / 30 turns, 800 games | **73.5%** (588/800) | `docs/evidence/hexcrystal_ab/arena_result.json` |
| Draw rate, random play, board 8 / 30 turns, 60 games | **0.68**; (1,1) tie 41/60 | `docs/evidence/draw_rate_sweep/` |
| Draw rate, random play, board 8 / 60 turns | **0.47** | same |
| Draw rate, random play, board 8 / 100 turns | **0.37** | same |
| Draw rate, random play, board 16 / 30 turns | **0.87** | same |
| Actions per game, random play, board 8 at 30 / 60 / 100 turns | 209 / 501 / 939 | same |
| `legacy_win_rate` ceiling at 73.5% draws | 0.265 vs threshold 0.55 | `docs/runbooks/promotion-gate.md` |
| Final checkpoint of the only training run vs random, draw-aware, 5 probes | 0.25 / 0.35 / 0.375 / 0.375 / 0.40 | `docs/evidence/validation_iteration_zero/` |
| Seeded probe reproducibility | 5/5 identical seeded; 0.30–0.45 unseeded | `docs/evidence/seeded_probe_reproducibility/` |
| Cost per iteration, board 8 / 20 turns / 12 sims / 20 games, CPU | self-play ~125 s flat; training 46 → 257 s over 5 iterations | `validation_iteration_zero_report.md` |

### 2.2 Draw-scoring sites (rule: every place a W/D/L record becomes one number)

| Site | Draw-aware? |
|---|---|
| `evaluation/scoring.py` (`score_rate`, ternary variance, `score_interval`) | yes — SSOT |
| `evaluation/promotion.py` registry | yes, but `DEFAULT_POLICY = "legacy_win_rate"` (`:78`) |
| `cloud/gcp/arena.py` `win_rate` (`:47-56`) | **no** ("preserved verbatim"); `score_rate` emitted beside it |
| `training_civ6.py` `evaluate_against_random` return (`:849`) | **no**; score rate logged at `:851-861` but not returned |
| `scripts/summarize_training_log.py` (`:49`) | **no** — the only eval regex is the draw-blind line |
| `scripts/run_baseline_eval.py`, `post_run_evaluation.py` | **no** — win/draw/loss rates, no CI, GCS-only loading |
| `scripts/validate_convergence.py` (`:251-262`) | **no**, and unseeded |
| `cloud/gcp/orchestrator_state.py` (`:45,82`) | **no** — `win_rate_history` only |
| `vertex_orchestrator.py` stall check (`:2046-2086`) | **no**, alert-only |
| `tournament/` Elo, `scripts/run_encoder_ab.py` | yes |

### 2.3 Mask/executor disagreement (new)

`core/civ6_actions.py:629-635` marks a move legal if on-board and not mountain.
`envs/microciv.py:807-818` also rejects any occupied tile; on rejection `:549-551` sets
`moves_remaining = 0`. 18,103 rejections in 720 random games (~25/game). The fog path
has the same shape of defect in the other direction (mask from unmasked state,
`:846-848`; no fog path in `get_valid_actions`, `:868-875`).

### 2.4 Cloud-path readiness for a validation run

| Needed | State |
|---|---|
| Promotion verdict via registry | ready (`vertex_orchestrator.py:1071-1082`) |
| `score_rate` in the GCS evaluation artifact | ready (`arena.py:77-90`) |
| `ci_low`/`ci_high`/verdict persisted | **missing** — `log_promotion_decision` is text + DEBUG only |
| `score_rate` in orchestrator state / MLflow | **missing** |
| `--promotion_policy` CLI | **missing** (env-only) |
| Any config setting the recommended policy | **missing** (comment only, `cloud_training_v1.yaml:54-62`) |
| Budget cap that halts | **missing** — `CostMonitor` has no importer |
| ≤1000-game committed config | **missing** — `cloud_training_v1.yaml` is 2000 games |
| `value_sign_accuracy` in an artifact | **missing** — DEBUG log + interval-gated MLflow |
| Lineage tag for pre-fix checkpoints | **missing** — no field, no writer |
| Seed plumbing in `selfplay_worker` / `vertex_orchestrator` | **missing** |
| `--resume_from=latest` | ready (`vertex_orchestrator.py:445-513`) |

### 2.5 Batch self-play

`should_use_batch` requires `num_parallel_games > 1` and `mcts_type == "batch_gumbel"`
(`selfplay_worker.py:508-528`); no YAML sets it. `scripts/benchmark_batch_selfplay.py`
writes MLflow + stdout only; `InferenceBenchmark` is unseeded and its single-env arm
passes no `max_turns` (`optimizations/inference_benchmark.py:160-166,192-198`), which is
a fairness bug in the comparison, not only a reproducibility one.

### 2.6 Repository

14 open PRs, all with `base.sha` from 2026-08-12; 0 open issues. Tech-debt Core scope
merged (PR #116). 53 dead modules / 24,451 lines remain (Tier B/C, `audit_import_graph`).
The hex-crystalline encoder is on the Hub as `ianshank/civ6-hexcrystal-encoder`
(verified 2026-09-02; the ROADMAP claim is true).

---

## 3. Objective and success criteria

**Objective.** Make the training loop *demonstrably* learn at a configuration where the
promotion gate is reachable, with every claim backed by a committed, seeded, draw-aware
artifact — then resume M4 features on that foundation. No neural-network architecture
change. No new training environment (NG-1). No LLM in the loop (NG-2).

**Done when** (each with the gate that verifies it):

- [ ] **D1.** A committed draw-rate table under a *trained* policy at the candidate
      configuration, and the shipped arena/cloud defaults changed only on that evidence.
      Gate: `docs/evidence/draw_rate_sweep/` has a trained-policy section; the changed
      default cites it in the same PR.
- [ ] **D2.** Zero mask/executor disagreements on the move action under random play.
      Gate: a regression test that plays N seeded random games and asserts the executor
      never returns `Invalid move`; negative control = reintroduce the collision check
      asymmetry.
- [ ] **E1.** Every site in §2.2 that returns a scalar returns or emits the draw-aware
      score rate with a CI, and the training log summariser records W/D/L, score rate and
      seed per iteration. Gate: `tests/scripts/test_summarize_training_log.py` on a log
      fixture containing both lines; grep-test that `wins / num_games` appears nowhere
      outside `evaluation/scoring.py` and a documented compat shim.
- [ ] **L1.** A seeded local run of ≥ 20 iterations at the D1 configuration, committed
      under `docs/evidence/`, whose draw-aware score rate vs random has a positive slope
      with a CI that excludes zero, **or** an honest report that it does not. Gate: the
      artifact exists with a `VALIDATION-RUN` ledger event; a checker over the JSON
      (not `validate_convergence.py`) exits 0/1.
- [ ] **C1.** One budget-capped (≤ $25, ≤ 1000 games, NG-7 sign-off recorded in the
      ledger) Vertex run on the fixed path with `score_rate_lower_bound` reported in
      shadow next to `legacy_win_rate`, `value_sign_accuracy` in the artifact, and a
      halting cost cap. Gate: the artifact + a decision memo on the promotion default.
- [ ] **N1.** `--num_players 3` self-play uses `NPlayerMCTS`; `CurrentPlayerValueView`
      is off the N-player path; a seeded 3-player run is committed. Gate: a test asserts
      the search class by `num_players`; the run artifact exists.
- [ ] **F1.** Fog mask derived from the fogged state, `get_valid_actions()` has a fog
      path, `--fog_of_war` reaches all three env-construction sites, the inert
      `fog_of_war_enabled` knob is removed or wired. Gate: a test that the fogged mask
      never reveals a hidden enemy; the `found_city` semantics decision recorded in an
      OpenSpec change.
- [ ] **B1.** A committed batch-vs-sequential benchmark with seed, `max_turns` parity,
      and provenance; a config that selects `batch_gumbel`. Gate: `docs/evidence/batch_selfplay/`.
- [ ] **R1.** Open PRs ≤ 3, disposed per §7.

---

## 4. Workstreams

Ordered by dependency, not by size. Sizes: S ≤ 1 day, M ≤ 1 week, L > 1 week.

### WS0 — Maintainer-only (day 0, unchanged from the tech-debt plan)

- Rotate the W&B key (`docs/runbooks/secret-scanning.md`). Unverifiable from the repo.
- Dispose of the 14 open PRs per §7.
- GCS champion-lineage inventory before the safe-load default reaches production
  (`CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1` is the escape hatch).
- Answer §9.

### WS-D — Own the draw rate (the prerequisite for every number below)

- **D0 (S, no code).** Re-run the sweep table under a trained checkpoint (the arena
  arms from `hexcrystal_ab` are not in the repo; use the Validation-Iteration-Zero
  `latest.pt` or the first L1 checkpoint). Random play is a lower bound on expansion;
  the decision must be made on trained play. Commit to `docs/evidence/draw_rate_sweep/`.
- **D1 (S).** Change the arena/cloud/A-B `max_turns` default (`arena.py:107-108`,
  `selfplay_worker.py:100-101,640,877,1065`, `run_encoder_ab.py:101-102`,
  `configs/cloud_training_v1.yaml:29`) to the smallest value at which D0 shows
  draw rate ≤ 40%. Prediction from random play: 60. Cost: ~2.4× actions per game.
  Keep board 8 — larger boards make it worse at short horizons and re-cost the network
  by `board_size²`. Note `TrainingConfig` (16/200) is *not* the arena's source; do not
  touch it. Checkpoints trained at 30 turns remain loadable; their value heads are simply
  miscalibrated for the longer horizon, so L1 starts from scratch.
- **D2 (S).** Fix the move mask: add the unit-collision check to
  `core/civ6_actions.py:629-635` so the mask and `_is_valid_move` agree, and add the
  regression test in §3 D2. This is a bug fix, not a semantics change.
- **D3 (M, CONFIRM-FIRST, OpenSpec).** Decision item, not a task: whether to add a
  secondary tie-break at the turn limit (`envs/microciv.py:322-333`) over state that
  already exists (`num_units`, unit health). This changes terminal reward signs for
  ~30% of games and must ship behind a version flag. **Do not** copy
  `gameplay/victory.py:323`, which breaks exact ties by `max()` and hands player 0 the
  win. Keep `None` for exact ties. Recommended only if D0+D1 leave the draw rate above
  40%.
- **D4 (L, CONFIRM-FIRST, OpenSpec).** City capture. The only change that makes
  elimination reachable once both players found. Invalidates every checkpoint's value
  function. Not in this plan's Core scope; recorded so the ceiling in D1 is not
  mistaken for a fix.

### WS-E — Draw-aware evaluation everywhere (parallel with WS-D)

- **E1 (S).** `evaluate_against_random` returns a `ScoreSummary`
  (`evaluation/scoring.py:226`) with a `win_rate` compat property for the two callers
  (`training_civ6.py:1003`); `train()` writes `eval_<iter>.json` next to each
  checkpoint (`:1010-1016`) so the summariser stops scraping scrollback; log a
  `Seed: N` banner.
- **E2 (S).** `summarize_training_log.py`: parse the W/D/L line and the seed; add
  score-rate, draw-rate and CI columns; prefer `eval_*.json` when present.
- **E3 (S).** Promote `docs/evidence/validation_iteration_zero/final_checkpoint_vs_random.py`
  to `scripts/probe_checkpoint.py` with `--checkpoint --seed --games --out`, using
  `scoring.summarize()`. Give `run_baseline_eval.py` / `post_run_evaluation.py` a
  `--checkpoint PATH` alternative to `--model_version` and route their rates through
  `scoring.py`.
- **E4 (S).** A `scripts/check_learning_trend.py` over E2's JSON: slope of score rate
  vs iteration with a bootstrap CI; exit 0/1. This replaces `validate_convergence.py`
  for local runs (which stays as the cloud harness, gaining `--seed` and draw-awareness
  when touched).
- **E5 (S).** `cloud/gcp/orchestrator_state.py`: add `score_rate`/`score_rate_history`;
  `vertex_orchestrator.py:2100-2103`: log `eval_score_rate`, `eval_score_ci_low`;
  persist `verdict.to_dict()` into the evaluation artifact beside `arena.to_dict()`.

### WS-L — Validation Iteration One (local, CPU, no sign-off needed)

After D1/D2/E1–E4. `python -m alpha_zero.training_civ6 --seed 42 --board_size 8
--max_turns <D1> --num_iterations 20 --num_games 20 --num_simulations 12` under
`OMP_NUM_THREADS=2`; budget ~3–6 h CPU by extrapolating the measured 46 → 257 s
training growth and the ~2.4× self-play re-cost. Commit the summariser output and the
E4 verdict to `docs/evidence/validation_iteration_one/`; `VALIDATION-RUN` ledger event.
**Report the result whichever way it goes.** If the slope is not positive, the next
step is D3, not WS-C.

### WS-C — Validation Iteration Zero, cloud (NG-7 sign-off)

After WS-L shows a trend. Prerequisites in code (all S): `--promotion_policy` CLI
(`vertex_orchestrator.py:246-255` pattern); a `configs/cloud_validation_capped.yaml`
at ≤ 1000 games with `CIV6AZ_PROMOTION_POLICY=score_rate_lower_bound`,
`CIV6AZ_PROMOTION_THRESHOLD=0.52` set in the job env; `--max_cost_usd` wired to the
already-written `CostMonitor` and breaking the loop; `value_sign_accuracy` emitted
per iteration into the training artifact; `--seed` on `selfplay_worker` threaded from
the orchestrator; a `tags` field on `ModelMetadata` (`cloud/gcp/storage.py:119-137`)
stamped `pre-value-sign-fix` on v0–v268 by a one-off script and with the commit SHA on
new versions. Then one run with `--resume_from=latest` explicitly set. Outputs: the
artifact, a decision memo on flipping `DEFAULT_POLICY`, and the runbook updated.

### WS-N — N-player search in self-play

- **N1 (S).** `training_civ6.py:607-640`: `elif config.num_players > 2` →
  `NPlayerMCTS(network, NPlayerMCTSConfig(...), num_players=...)` **without**
  `_mcts_network` wrapping (it needs the raw `(B, N)` head). Widen the union at
  `:475,:610`. Same branch on the eval path at `:806`. Optional `"nplayer"` entry in
  `mcts_factory.py:100-107`. Test asserts the class by `num_players`.
- **N2 (M).** A seeded 3-player run at the D1 configuration, committed. The literature
  this follows (Petosa & Balch 2019, arXiv 1910.13012) uses a score vector with mean
  per-player MSE and Max-n backup — which is what `NPlayerValueHead` +
  `compute_nplayer_value_loss` + `NPlayerMCTS` already implement — and reports that more
  than 2 players "needs more careful hyperparameter tuning"; budget for that.
- Opponent modelling stays out of scope (the tech-debt plan's Tier C keeps
  `multiplayer/` for this).

### WS-F — Fog of war (after N1; F1 needs its own OpenSpec change)

- **F1 (S–M, OpenSpec).** Compute the action mask from the fogged `game_state`
  (`envs/microciv.py:846-848`) and add the fog path to `get_valid_actions()`
  (`:868-875`). Decide the `found_city`-next-to-hidden-city semantics: the fogged mask is
  over-permissive there, and the executor rejects without clearing moves. Recommended:
  treat it like D2 — mask must agree with the executor **as seen by the player**, so
  the executor's rejection of a fog-hidden collision must not forfeit the unit's turn.
  Test: fogged mask never reveals a hidden enemy.
- **F2 (S).** `--fog_of_war` on `TrainingConfig` → the three `MicroCivEnv(...)` sites
  (`:596,:795,:880`); delete the inert `fog_of_war_enabled` in `orchestrator/config.py`
  and four YAMLs or wire it — do not leave a knob that does nothing.
- **F3 (M, CONFIRM-FIRST: >3 modules).** `DeterminizedMCTS(NPlayerMCTS)` in self-play.
  Fix `WorldSampler._inject_world` first: it does not set `turn`/`current_player`
  despite its docstring, injects units with `id=-1` against live id counters, and is a
  silent no-op on envs without `units`/`cities`. Throughput budget for `num_worlds`.

### WS-B — Batch self-play as a measured path

- **B1 (S).** `benchmark_batch_selfplay.py --out DIR` writing `result.json` + `report.md`
  with provenance; `InferenceBenchmark(seed=...)` passing `max_turns` to *both* arms and
  `seed_offset ≠ 0` to the vectorised env.
- **B2 (S).** A config with `mcts_type: batch_gumbel`, `num_parallel_games: 8`.
- **B3 (S).** Run B1 at the D1 configuration on CPU and (in WS-C) on the T4-spot tier;
  commit `docs/evidence/batch_selfplay/`. Until then the ROADMAP's 5–10× stays "target".

### WS-T — Continuing (parallel, no sequencing)

- Coverage: add the `web` extra to the coverage job (`ci.yml:321`) and un-omit
  `serving/`, `inference/`.
- Tech-debt: Tier B rows, WS3.2/WS5/WS6/WS7 per `kb/specs/20260902-tech-debt-hardening.md`.
  Two Tier B flags from the audit: `cloud/gcp/preemption_handler` is named as deployed in
  `CLAUDE.md` §7 yet has no live importer (its sibling `preemption.py` is also dead —
  decide which is real); `utils/agent_tools` is documented in §9.4/§15.3.
- Promotion-stagnation detection on the deployed path: feed score rates into
  `EloPlateauDetector` from `cloud/gcp/`; currently a documented gap.
- Bridge (M5 prep): `NEXT_STEPS.md` items 1–2 stay queued; item 3 (a trained checkpoint
  live) strictly after WS-C.

---

## 5. Sequencing and estimate

```
WS0 (maintainer, day 0) ─┐
WS-D D0,D1,D2 (1 wk) ────┼─► WS-L (≈1 wk incl. the run) ─► WS-C (≈1 wk, sign-off)
WS-E E1–E4 (1 wk) ───────┘         │
                                   ├─► WS-N N1 (days) ─► N2 (1–2 wk) ─► WS-F F1,F2 (1 wk) ─► F3 (2 wk)
                                   └─► WS-B B1,B2 (days) ─► B3 (with WS-C)
WS-T continuous
```

Core scope (D0–D2, E1–E5, L, C, N1, B1–B2): **~4 weeks** to the first committed
learning-trend artifact and the first validated cloud run. Full scope adds N2, F1–F3,
B3, D3: **~9–11 weeks**. D4 is out of scope. These are single-maintainer estimates at
the cadence PR #113–#116 demonstrated (one PR-sized unit per day).

---

## 6. Risk register

| Risk | Likelihood | Mitigation / rollback |
|---|---|---|
| D0 under trained play shows a *higher* draw rate than random at 60 turns (trained agents learn to turtle) | medium | Then D3 (tie-break) enters Core; D4 becomes the honest long-term answer. Record either way. |
| Raising `max_turns` re-costs every cloud figure ~2.4× | certain | WS-C's ≤1000-game cap is in games, not actions; re-estimate $ from B3's measured moves/s before sign-off. |
| D2 changes the mask density that `test_batch_selfplay_integration` asserts on | likely | The test asserts `0 < density < 1`, not a value; re-run and record. |
| WS-L shows no trend even at the D1 configuration | possible | That is a result, not a failure of the plan; commit it, escalate to D3. Do not proceed to WS-C. |
| E1's return-type change breaks a caller | low | `win_rate` compat property; two callers, both in `training_civ6.py`. |
| WS-C spend exceeds the cap | low once `--max_cost_usd` exists; **high today** (no halting cap) | Cap wiring is a prerequisite, not a nice-to-have. |
| N1's raw `(B, N)` head reaches a scalar consumer | low | Test asserts `CurrentPlayerValueView` is not applied on the N-player path. |
| F1's semantics decision stalls | medium | Ship F2 (flag) with F1's OpenSpec proposal open; do not ship the flag alone as "fog support". |
| `_inject_world` bugs make F3 results uninterpretable | high if skipped | F3's first task is fixing and testing `_inject_world`, not wiring. |

---

## 7. Open PR disposition (14)

| PR | Disposition | Reason |
|---|---|---|
| #95 codecov-action | **close** | codecov removed from CI (`ci.yml:327-331`) |
| #98 action-gh-release | **close** | `release.yml:117` is SHA-pinned; a v3 bump must be a re-resolved SHA |
| #103 ruff 0.16.2 | **close; redo as one PR touching all four pin sites** | `pyproject.toml:26`, `requirements.txt:12`, `.pre-commit-config.yaml:30`, `ci.yml:231`; CLAUDE.md §5.2 requires the pins to match; expect new rule findings |
| #94/#96/#97 checkout/setup-python/cache | **close; replace with a SHA-pin sweep** | same supply-chain argument already applied to `release.yml` |
| #99 prometheus-client, #100 tensorboard, #101 chromadb | **merge** | clean, one line each, `requirements.txt` only (CI installs from `pyproject.toml`) |
| #102 google-cloud-monitoring | **close; delete the requirement line** | `cloud/gcp/monitoring.py` was deleted in Tier A; nothing imports it |
| #106 bridge hardening | **maintainer decides: rebase or close** | 7 content conflicts (2 code: `bridge/config.py`, the two Lua files; 5 docs), none from Tier A; drop its `pytest.ini` (overrides `pyproject.toml`'s pytest config) and `.agent/AGENTS.md` (a fourth "agents" location) |
| #69 project structure | **close** | deletes `setup.py` already deleted; moves `interfaces/` which Tier B row B-x decides |
| #49, #32 Strategic Planning SDK | **close** | NG-6 (not a general game-AI framework); 42k/26k-line diffs against a tree 7 months newer |

---

## 8. Gates this plan adds (each merged with a negative control)

1. Move mask ≡ executor under seeded random play (D2).
2. `wins / num_games` grep-gate outside `scoring.py` (E1).
3. Log-summariser fixture with both eval lines (E2).
4. Learning-trend checker exit code (E4).
5. Search class by `num_players` (N1); `CurrentPlayerValueView` absent on the N-player path.
6. Fogged mask reveals nothing hidden (F1).
7. Benchmark artifact schema + provenance block (B1).
8. A `VALIDATION-RUN` ledger event per committed run (K-rules, existing).

---

## 9. Authorizations required (CLAUDE.md §3.2 / charter)

1. **Environment semantics** (D3 tie-break, D4 city capture, F1 `found_city` rule):
   CONFIRM-FIRST; each needs an OpenSpec change. D1 (a default value) and D2 (a bug
   fix) do not.
2. **Cloud spend** (WS-C, B3 on Vertex): NG-7 sign-off, recorded in the ledger with the
   cap.
3. **> 3 modules** (F3): CONFIRM-FIRST. N1, F2, E1–E5 are each ≤ 3 modules.
4. **Maintainer-only**: WS0 items, §7 dispositions, GCS lineage inventory.
5. **Not required**: no network-architecture change, no distributed-coordination change,
   no charter amendment. Nothing here touches NG-1..NG-7.

---

## 10. Literature check (what it supports, what it does not)

- **Draw rate as a training-signal problem** — Joshi 2025 (arXiv 2504.07757,
  search-contempt): in chess self-play the useful regime is (wins+losses)/draws ≈ 1,
  reached by an asymmetric search parameter or by temperature; draw-heavy self-play
  carries little information per game. Supports treating the draw rate as the
  first-order problem. Does **not** transfer directly: MicroCiv's draws are
  truncation ties, not high-quality equal play, so the lever here is the horizon and
  the tie rule, not contempt.
- **Multiplayer value vectors** — Petosa & Balch 2019 (arXiv 1910.13012): score vector
  targets, mean per-player MSE, Max-n backup, works on 3-player Tic-Tac-Mo/Connect-3x3
  and reports the need for tuning beyond 2 players. Matches WS-N's design; sets
  expectations for N2.
- **Auxiliary supervision** — Kong, Ram & Yu 2026 (arXiv 2607.08984): oracle-derived
  auxiliary policy loss closes much of the strong-vs-perfect gap on Connect Four and
  Chomp. Not applicable (no oracle for MicroCiv); recorded so nobody proposes it as a
  draw-rate fix.
- **Information-set search** — MAPLE (arXiv 2605.24139) and ISMCTS lineage: aggregate
  evaluation over sampled states, which is what `DeterminizedMCTS` does. Supports F3's
  design, and its cost model (`num_worlds` × search) is why F3 carries a throughput
  budget.

---

## 11. Confidence

- [Certain] every count that names a file:line or a command; the sweep table; CI state;
  PR states; call-site counts; the mask/executor disagreement.
- [Likely] `max_turns=60` at board 8 brings the *trained-policy* draw rate under 50%.
  Random play is a lower bound on expansion, so D0 decides.
- [Likely] the Core estimate of ~4 weeks; WS-L's run cost is extrapolated from one
  5-iteration run.
- [Guessing] whether WS-L shows a positive slope at all. That is the point of running it,
  and the plan says what happens if it does not.
- [Guessing] which of D3/D4 the maintainer will accept. That is why they are decision
  rows and not tasks.
