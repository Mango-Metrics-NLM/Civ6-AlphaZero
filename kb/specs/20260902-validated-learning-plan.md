# Spec: Validated-Learning Plan — the next steps after the tech-debt Core scope (2026-09-02, v2)

**Role:** Planning → DEV-SQE (with maintainer-only items called out)
**Session:** session_01AQAVGXog3wGfPaqu4XHfXX, branch `claude/dev-roadmap-next-steps-fk867n`, PR #117
**Status:** v2, supersedes v1 after an independent adversarial review (§0.1). Built from four
verification passes over the repository, one measurement (`docs/evidence/draw_rate_sweep/`),
a literature check, and that review, which re-ran the measurement with different seeds and
re-derived the estimates. Supersedes the "Recommended Plan" of
`docs/NEXT_STEPS_CASE_2026-08.md` (Phase 1 of which landed as PR #113, Phase 2 of which has
not happened) and the four overlapping next-step lists in `CLAUDE.md` §13, `ROADMAP.md`
M4, `NEXT_STEPS.md`, and that case document.

Every number below was measured on 2026-09-02 against commit `50144cc` (default-branch
head, CI run 33670154515 green on all 8 jobs) and the command, file, or artifact that
produced it is stated beside it.

---

## 0. What review changed

### 0.1 Peer review (v1 → v2)

An independent reviewer with no access to v1's reasoning re-ran the sweep with different
seeds (8/30: 0.77 [Wilson 0.59–0.88], 8/60: 0.50 [0.33–0.67]; actions per game 210 / 500,
exact) and re-derived the plan's numbers. Two blockers, seven majors. In order of consequence:

| v1 claim | v2 correction | Evidence |
|---|---|---|
| WS-L: a 20-iteration run yields a per-iteration trend | **It would evaluate twice.** `--save_interval` defaults to **10** and gates both evaluation and checkpointing, so the command as written produced two data points and no slope. Now `--save_interval 1`, and E4's checker exits 2 on fewer than 3 points. | `training_civ6.py:1038-1046`, `:1001` |
| N2: "a seeded 3-player run at the D1 configuration" | **Cannot be seeded, and would draw 85–100%.** `NPlayerMCTS` draws from the global `np.random` in four places (`multiplayer/mcts.py:338,379,390,393`), breaking the 2026-08-30 "every search owns its RNG" invariant — a second module for N1. And 3-player random play at board 8 draws **1.00 at 30 turns and 0.85 at 60** (reviewer, n=20): the most-cities tie rule makes (0,1,1)/(1,1,1) near-certain. N2 is gated on a 3-player D0 row ≤ 40%, else D3 enters Core before N2. | reviewer's run; `envs/microciv.py:322-333` |
| WS-L "after D1/D2/E1–E4"; ~3 weeks to the first artifact | **Mis-sequenced.** `training_civ6.py` already accepts every flag WS-L names; `TrainingConfig` is not the arena's source, so D1 does not touch the run; the W/D/L line is already logged (`:845-855`) so E2 can parse it after the fact. Only D2 changes policy targets. And D0 was **circular**: it wanted a trained checkpoint, but no `.pt` is tracked (`.gitignore:94`), so D0 needs L1's checkpoints. WS-L now starts on day 1 at `--max_turns 60`, in parallel with WS-D/WS-E; D0 runs on its checkpoints. | `training_civ6.py:1027,1029,1062-1064,1074`; `git ls-files \| grep .pt` empty |
| D1 is "a default value" at four sites | **The deployed orchestrator's default is 50, not 30**, forwarded to self-play, training and the arena (`vertex_orchestrator.py:217,538,646,1182,1283`); `run_cloud_training.sh:37` passes 30 explicitly; `cloud_training_v1.yaml` is read by no code. Eight further 30-defaults and six test assertions (§4 D1). ≥ 9 modules → §3.2 confirm-first, listed under §9. `arena.py:107-108` was the wrong line; the default is `:101`. | grep, §4 D1 |
| §9: "no distributed-coordination change" | **Wrong.** WS-C's `--max_cost_usd` halting the orchestrator loop, seed threading through it, and a `tags` field on `ModelMetadata` are changes to the distributed coordinator — CONFIRM-FIRST in CLAUDE.md §3.2. Moved to §9. | §4 WS-C |
| C1 runs "with `--resume_from=latest`"; D1 says "L1 starts from scratch" | **Contradiction.** `latest` would fine-tune champion v268 — pre-value-sign-fix, trained at 30 or 50 turns — so C1's first statistics would measure a mis-calibrated resumed model. C1 now starts from scratch with the lineage tagged; a resume run is a separate, later decision. | `vertex_orchestrator.py:445-513` |
| §2.2 lists "every" draw-blind site; E1's grep gate for `wins / num_games` | **Four live sites missing** and the gate **passes vacuously today**: `orchestrator/orchestrator.py:1515,1763`, `evaluation/evaluator.py:546`, `evaluation/muzero_vs_alphazero_benchmark.py:112` spell it `wins / max(...)`. Rows added; the gate is now a regex over `wins\s*/\s*(max\(\|num_\|total\|len\()` with a negative control. | grep |
| L1 "done when positive slope **or** an honest report" | Always satisfiable as a completion criterion. Core-done is now "artifact + checker verdict recorded"; the slope decides the *next* step, not done-ness. | §3 |
| C1 "halting cost cap ≤ $25" | `CostMonitor` halts on an **estimate** (billing API off by default, `cost_monitor.py:104-124`); the dollar figure is not verifiable from the repo. Paired with a games cap, which is enforced. | same |
| WS-L "3–6 h CPU from 46 → 257 s training growth and 2.4× self-play" | Both inputs wrong, partially cancelling. Training does **not** keep growing: `sample_size = min(len(buffer), batch_size*100)` caps it at 12,800 examples (`training_civ6.py:966`), ~260 s/iter from iteration 2. Self-play at 8/60 vs the **20-turn** baseline is ~4.3× (reviewer: 20.0 s vs 4.6 s per game, 533 vs 126 examples), not 2.4× (which is 30→60). Re-derived: 20 × (20 × 20 s + 260 s + 25–120 s eval) ≈ **3.9–4.4 h**, rising as the policy learns to reach the limit. | `training_civ6.py:966`; reviewer's timing run |
| "~25 mask rejections per game" | Averaged over 12 cells and dominated by 100-turn/16×16 games. At the shipped 8/30 it is **~6/game (3% of actions)**; 8/60 ~22. Every rejection inspected (5,700 actions) had a unit on the target tile — the mechanism claim holds. | reviewer's run |
| §7: 14 open PRs; #106 row | 15 (this PR). #106 also adds a 4,975-line `uv.lock` and silently changes `heartbeat_timeout_s` 2.0 → 120.0 in `bridge/config.py`. #103's ruff pin has a **fifth, unpinned** site: `deploy_spaces.yml:53` (`pip install ruff`) floats on its own. | GitHub API, files |
| F1/B1 gates | F1 ("fogged mask never reveals a hidden enemy") passes vacuously on a fixture where nothing is hidden; B1 was a directory-existence check. Both tightened in §3. | §3 |
| Line cites | Three drifts fixed: `training_civ6.py:849` → `:844` (draw-blind division) and `:856` (return); `:851-861` → `:845-855`; `arena.py:107-108` → `:101`. The other ~40 cites were opened and hold. | reviewer |

Kept from v1 without change: the measurement and its mechanism; the N1 API claim
(`MicroCivEnv` passes `isinstance(env, MultiplayerEnvironment)` at runtime and
`MicroCivNetwork.predict` matches `NPlayerMCTS._evaluate`, both re-confirmed); checkpoints
across `max_turns` stay loadable because the encoder normalises the turn by a constant
(`civ6_encoder.py:250`, `turn/500`) so only calibration shifts; the PR dispositions; the
literature reading.

### 0.2 Verification (first answer → v1)

The first answer to "what are the next steps" was given from the documentation alone.
Four verification agents and one measurement corrected it as follows.

| First-answer claim | Correction | Evidence |
|---|---|---|
| "The cheapest lever on the draw rate is game length and board size" | **Half right.** `max_turns` is the lever: random-play draws at board 8 fall 0.68 → 0.47 → 0.37 for 30 → 60 → 100 turns. **Board size at 30 turns makes draws worse** (0.68 → 0.87 at 16×16). | `docs/evidence/draw_rate_sweep/report.md` |
| A code-reading review argued the reverse: "lengthening `max_turns` alone will not help; a second city is geometrically near-impossible on 8×8" | **Refuted by measurement.** (1,2) city outcomes go from 6/60 at 30 turns to 26/60 at 100. The geometry constraint is real but not binding; the settler's ~17-turn cost is. | same |
| "The upstream problem is the draw rate" (runbook) | **Confirmed and mechanised.** At 8/30 the mass sits on a (1,1) city tie (41/60); random play draws 0.68 vs 0.735 with trained policies — trained play was not expanding better than random. And the ceiling is structural: cities can never be captured or destroyed (`_execute_attack` removes units only), so once both players found, elimination is impossible. | `envs/microciv.py:318-341`, `:652-702`, `:774-797` |
| "N-player MCTS wiring touches more than three modules and needs confirm-first" | **Wrong (and v2 corrects the correction).** Branching self-play on `num_players > 2` is one module; seeding `NPlayerMCTS` makes it two. Still ≤ 3. | `training_civ6.py:596-640`, `multiplayer/mcts.py:78-143`, `multiplayer/protocols.py:35-78` |
| "The cloud path is ready for a shadow-mode run" | **Not ready.** The verdict consults the registry, but `ci_low`/`ci_high` and the verdict record go to `log.debug` only; orchestrator state stores draw-blind `win_rate` only; there is no `--promotion_policy` CLI arg; no config sets the policy; the only budget mechanism (`CostMonitor`) has no importer; no committed YAML fits a ≤1000-game envelope; `selfplay_worker` has no `--seed` and `vertex_orchestrator` has no seed plumbing at all. | `promotion.py:542-574`, `orchestrator_state.py:45,82`, `vertex_orchestrator.py:246-255`, `cost_monitor.py:104-124`, `configs/cloud_training_v1.yaml:21-31`, `selfplay_worker.py:1537-1551` |
| "Run a longer seeded local run" | **Its artifact would still be draw-blind.** `summarize_training_log.py` parses only the `Win rate vs random` line; the draw-aware line at `training_civ6.py:845-855` has no regex; nothing writes `log_dir`; both eval scripts load from GCS by integer version and never import `evaluation/scoring.py`. | `scripts/summarize_training_log.py:49,126-128`; `scripts/run_baseline_eval.py:72-77,364-376` |
| — | **New defect:** the action mask marks moves onto occupied tiles legal; the executor rejects them and forfeits the unit's remaining moves. ~6 per game at 8/30, ~22 at 8/60. | `core/civ6_actions.py:629-635` vs `envs/microciv.py:807-818,549-551` |
| — | **`validate_convergence.py` cannot validate a local run**: it drives `training_worker.train_epoch` against a mock GCS backend with synthetic trajectories, is draw-blind and unseeded. | `scripts/validate_convergence.py:28-33,245-262` |
| "Open PRs #69/#49/#32 and ten Dependabot PRs" | All open PRs are `dirty` or stale against the post-Tier-A tree. #95 and #98 are obsolete (codecov removed; release action SHA-pinned). #102 raises a floor for a package nothing imports. #106 has 7 content conflicts, none from Tier A. | §7 |

Kept from the first answer: the project has never demonstrated learning
(`docs/evidence/validation_iteration_zero/`: final checkpoint below random on 5/5 probes);
the promotion default is arithmetically unreachable at the shipped configuration;
`NPlayerMCTS`/`DeterminizedMCTS` have zero production call sites (15 + 17 test
instantiations, 0 in `alpha_zero/` or `scripts/`); the fog action-mask leak is real and no
test covers the mask under fog; promotion-stagnation detection on the deployed path is
absent (the closest thing is an alert-only, draw-blind win-rate stall check at
`vertex_orchestrator.py:2046-2086`).

---

## 1. Premise

1. **CI is green and the gates now mean something.** Run 33670154515 on `50144cc`: 8/8
   jobs including the cloud coverage floor (59.7% ≥ 58%), gitleaks, governance. The debt
   ratchet has not moved (`scripts/measure_debt.py` vs
   `docs/evidence/tech_debt_baseline/baseline.json`: 18 of 19 metrics identical,
   `package_lines` +4 from a docstring commit).
2. **The loop executes but has not been shown to learn.** The only committed training run
   is 5 iterations × 20 games and reports "no learning shown" in its own words. Its two
   structural causes are known: a draw rate that compresses every statistic toward 0.5,
   and evaluation plumbing that reports draw-blind numbers.
3. **The August correctness fixes are unvalidated by any run.** Value-sign labelling
   (`0d5adee`), promotion scoring SSOT, batch `action_masks`, seeding — all merged, none
   exercised end to end on the cloud path. The promoted lineage v0–v268 predates all of
   them and is not tagged as such (no `tags`/lineage field exists on `ModelMetadata`).
4. **Two of the three M4 feature bullets are blocked on evidence, not code.** Batch
   self-play is wired and tested but selected by no config and benchmarked by no committed
   artifact. N-player search is a two-module change. Fog is sequenced behind N-player
   search by `DeterminizedMCTS`'s per-player value contract.
5. **The documentation had four next-step lists.** This spec is the single "what now";
   `NEXT_STEPS.md`, `ROADMAP.md` M4 and `CLAUDE.md` §13 point at it, and
   `docs/NEXT_STEPS_CASE_2026-08.md` carries a superseded banner.

---

## 2. Baseline (reproducible)

### 2.1 Learning signal

| Measurement | Value | Source |
|---|---|---|
| Draw rate, trained Gumbel policies, board 8 / 30 turns, 800 games | **73.5%** (588/800) | `docs/evidence/hexcrystal_ab/arena_result.json` |
| Draw rate, random play, board 8 / 30 turns, 60 games | **0.68**; (1,1) tie 41/60 — reviewer re-run (different seeds, n=30): 0.77 [0.59–0.88] | `docs/evidence/draw_rate_sweep/` |
| Draw rate, random play, board 8 / 60 turns | **0.47** — reviewer: 0.50 [0.33–0.67] | same |
| Draw rate, random play, board 8 / 100 turns | **0.37** | same |
| Draw rate, random play, board 16 / 30 turns | **0.87** | same |
| Draw rate, **3-player** random play, board 8, 30 / 60 turns (n=20) | **1.00 / 0.85**; 306 / 658 actions per game | reviewer's run; to be committed as a D0 row |
| Actions per game, random play, board 8 at 30 / 60 / 100 turns | 209 / 501 / 939 | same |
| Mask rejections per game, random play, 8/20 · 8/30 · 8/60 | 3.0 · **6.2** · 22.3 (all unit collisions) | reviewer's run; sweep log |
| `legacy_win_rate` ceiling at 73.5% draws | 0.265 vs threshold 0.55 | `docs/runbooks/promotion-gate.md` |
| Final checkpoint of the only training run vs random, draw-aware, 5 probes | 0.25 / 0.35 / 0.375 / 0.375 / 0.40 | `docs/evidence/validation_iteration_zero/` |
| Seeded probe reproducibility | 5/5 identical seeded; 0.30–0.45 unseeded | `docs/evidence/seeded_probe_reproducibility/` |
| Cost per iteration, board 8 / **20** turns / 12 sims / 20 games, CPU | self-play ~125 s flat; training 46 → 257 s over 5 iterations, then capped (see §4 WS-L) | `validation_iteration_zero_report.md` |
| Network self-play per game, 12 sims, 8/20 vs 8/60 | 4.6 s vs **20.0 s** (4.3×); 126 vs 533 examples | reviewer's timing run |

### 2.2 Draw-scoring sites (rule: every place a W/D/L record becomes one number, found by `rg 'wins\s*/\s*(max\(|num_|total|len\()'` plus reading)

| Site | Draw-aware? |
|---|---|
| `evaluation/scoring.py` (`score_rate`, ternary variance, `score_interval`) | yes — SSOT |
| `evaluation/promotion.py` registry | yes, but `DEFAULT_POLICY = "legacy_win_rate"` (`:78`) |
| `cloud/gcp/arena.py` `win_rate` (`:47-56`) | **no** ("preserved verbatim"); `score_rate` emitted beside it |
| `training_civ6.py` `evaluate_against_random` (`:844` division, `:856` return) | **no**; score rate logged at `:845-855` but not returned |
| `scripts/summarize_training_log.py` (`:49`) | **no** — the only eval regex is the draw-blind line |
| `scripts/run_baseline_eval.py`, `post_run_evaluation.py` | **no** — win/draw/loss rates, no CI, GCS-only loading |
| `scripts/validate_convergence.py` (`:251-262`) | **no**, and unseeded |
| `cloud/gcp/orchestrator_state.py` (`:45,82`) | **no** — `win_rate_history` only |
| `vertex_orchestrator.py` stall check (`:2046-2086`) | **no**, alert-only |
| `orchestrator/orchestrator.py:1515,1763` (`wins / max(...)`) | **no** — non-deployed orchestrator, still live code |
| `evaluation/evaluator.py:546` | **no** (report string) |
| `evaluation/muzero_vs_alphazero_benchmark.py:112` | **no** |
| `tournament/` Elo, `scripts/run_encoder_ab.py` | yes |

### 2.3 Mask/executor disagreement (new)

`core/civ6_actions.py:629-635` marks a move legal if on-board and not mountain.
`envs/microciv.py:807-818` also rejects any occupied tile; on rejection `:549-551` sets
`moves_remaining = 0`. ~6 rejections per game at the shipped 8/30 (3% of actions), ~22 at
8/60; 100% of inspected rejections had a unit on the target tile. The fog path has the
same shape of defect in the other direction (mask from unmasked state, `:846-848`; no fog
path in `get_valid_actions`, `:868-875`).

### 2.4 Cloud-path readiness for a validation run

| Needed | State |
|---|---|
| Promotion verdict via registry | ready (`vertex_orchestrator.py:1071-1082`) |
| `score_rate` in the GCS evaluation artifact | ready (`arena.py:77-90`) |
| `ci_low`/`ci_high`/verdict persisted | **missing** — `log_promotion_decision` is text + DEBUG only |
| `score_rate` in orchestrator state / MLflow | **missing** |
| `--promotion_policy` CLI | **missing** (env-only) |
| Any config setting the recommended policy | **missing** (comment only, `cloud_training_v1.yaml:54-62`, and that YAML is read by no code) |
| Budget cap that halts | **missing** — `CostMonitor` has no importer, and estimates unless the billing API is enabled |
| ≤1000-game committed config | **missing** — `run_cloud_training.sh` drives 2000 games |
| `value_sign_accuracy` in an artifact | **missing** — DEBUG log + interval-gated MLflow |
| Lineage tag for pre-fix checkpoints | **missing** — no field, no writer |
| Seed plumbing in `selfplay_worker` / `vertex_orchestrator` | **missing** |
| `--resume_from=latest` | ready (`vertex_orchestrator.py:445-513`) — and **not** to be used for C1 (§4) |
| Cloud `max_turns` today | **two defaults**: 50 via `vertex_orchestrator.py:217`, 30 via `run_cloud_training.sh:37` |

### 2.5 Batch self-play

`should_use_batch` requires `num_parallel_games > 1` and `mcts_type == "batch_gumbel"`
(`selfplay_worker.py:508-528`); no YAML sets it. `scripts/benchmark_batch_selfplay.py`
writes MLflow + stdout only; `InferenceBenchmark` is unseeded and its single-env arm passes
no `max_turns` (`optimizations/inference_benchmark.py:160-166,192-198`), which is a
fairness bug in the comparison, not only a reproducibility one.

### 2.6 Repository

15 open PRs (14 with `base.sha` from 2026-08-12, plus #117); 0 open issues. Tech-debt
Core scope merged (PR #116). 53 dead modules / 24,451 lines remain (Tier B/C,
`audit_import_graph`, reproduced by the reviewer). The hex-crystalline encoder is on the
Hub as `ianshank/civ6-hexcrystal-encoder` (verified 2026-09-02; the ROADMAP claim is true).
No checkpoint is tracked in git (`.gitignore:94`).

---

## 3. Objective and success criteria

**Objective.** Make the training loop *demonstrably* learn at a configuration where the
promotion gate is reachable, with every claim backed by a committed, seeded, draw-aware
artifact — then resume M4 features on that foundation. No neural-network architecture
change. No new training environment (NG-1). No LLM in the loop (NG-2).

**Done when** (each with the gate that verifies it, and how the gate could pass vacuously
ruled out):

- [ ] **L1.** A seeded local run of ≥ 20 iterations with `--save_interval 1` at board 8 /
      60 turns, committed under `docs/evidence/validation_iteration_one/` with a
      `VALIDATION-RUN` ledger event, and `scripts/check_learning_trend.py`'s verdict
      recorded (exit 0 = positive slope with a bootstrap CI excluding zero; 1 = not;
      2 = fewer than 3 evaluation points, which is an error, not a result). **Done is the
      artifact + recorded verdict; the verdict decides the next step (§5).**
- [ ] **D0.** The draw-rate table re-run under L1's checkpoints (2-player) and under random
      play for 3 players, committed to `docs/evidence/draw_rate_sweep/`. Gate: the report
      has a trained-policy section and a `num_players=3` row.
- [ ] **D1.** Every `max_turns` default on the arena/cloud path (§4 D1 list) changed to the
      smallest value at which D0's trained-policy row shows draw rate ≤ 40%, in one PR that
      cites D0 and updates the six test assertions. Gate: a grep-test that the deployed
      path has one `max_turns` default.
- [ ] **D2.** Zero mask/executor disagreements on the move action. Gate: a regression test
      that plays N seeded random games and asserts the executor never returns
      `Invalid move`; negative control = remove the collision check from the mask again.
- [ ] **E1.** Every §2.2 "no" site returns or emits the draw-aware score rate with a CI;
      the summariser records W/D/L, score rate, CI and seed per iteration. Gate:
      `tests/scripts/test_summarize_training_log.py` on a fixture holding both eval lines;
      a regex gate `wins\s*/\s*(max\(|num_|total|len\()` over `alpha_zero/` and `scripts/`
      that passes only for `evaluation/scoring.py` and a documented compat shim — with a
      negative control that plants one of the four §2.2 spellings.
- [ ] **C1.** One budget-capped Vertex run on the fixed path, **from scratch**, with
      `score_rate_lower_bound` reported in shadow next to `legacy_win_rate`,
      `value_sign_accuracy` in the artifact, a **games cap (≤ 1000, enforced)** plus an
      estimated-cost cap (≤ $25, `CostMonitor` estimate unless billing is enabled), and
      NG-7 sign-off recorded in the ledger. Gate: the artifact + a decision memo on the
      promotion default.
- [ ] **N1.** `--num_players 3` self-play uses a seeded `NPlayerMCTS`; `CurrentPlayerValueView`
      is off the N-player path. Gate: a test asserts the search class by `num_players`, a
      test asserts two searches with the same seed pick the same action, and no
      `np.random.` remains in `multiplayer/mcts.py`.
- [ ] **N2.** A seeded 3-player run committed — **only after** D0's 3-player row (or D3)
      brings the 3-player draw rate ≤ 40%.
- [ ] **F1.** Fog mask derived from the fogged state, `get_valid_actions()` has a fog path,
      `--fog_of_war` reaches all three env-construction sites, the inert
      `fog_of_war_enabled` knob is removed or wired. Gate: a test whose fixture **asserts at
      least one enemy is hidden** and then that the fogged mask reveals nothing about it;
      the `found_city` semantics decision recorded in an OpenSpec change.
- [ ] **B1.** A committed batch-vs-sequential benchmark under `docs/evidence/batch_selfplay/`
      whose `result.json` **validates against a schema** (seed, `max_turns` on both arms,
      git SHA, torch version, device, both throughput figures); a config that selects
      `batch_gumbel`.
- [ ] **R1.** Open PRs ≤ 3, disposed per §7.

---

## 4. Workstreams

Ordered by dependency, not by size. Sizes: S ≤ 1 day, M ≤ 1 week, L > 1 week.

### WS0 — Maintainer-only (day 0, unchanged from the tech-debt plan)

- Rotate the W&B key (`docs/runbooks/secret-scanning.md`). Unverifiable from the repo.
- Dispose of the 15 open PRs per §7.
- GCS champion-lineage inventory before the safe-load default reaches production
  (`CIV6AZ_ALLOW_UNSAFE_CHECKPOINT=1` is the escape hatch).
- Answer §9.

### WS-L — Validation Iteration One (local, CPU, ~~day 1~~, no sign-off needed)

> **AMENDED 2026-09-04 — WS-L is re-sequenced behind WS-S.** See
> [`20260904-correctness-first-plan.md`](20260904-correctness-first-plan.md) §1. On the code as it
> stands this run cannot produce the evidence it is designed to produce: the default search negates
> the value at every ply although 71% of MicroCiv steps do not change the player, its policy target
> is a one-hot (1 non-zero of 376), and neither the self-play nor the evaluation seed carries an
> `iteration` term — so a seeded 20-iteration run trains on the same games twenty times and scores
> them on the same fixed 20 maps. Run WS-L after S1–S4 land; the command below is unchanged.
> The same correction applies to D0: **re-measure the draw rate after the search is fixed**, since
> ~70% draws is what two agents that cannot tell a win from a loss would produce.

Depends on nothing in this plan except, ideally, D2 (it changes policy targets; if D2 is
not merged on day 1, start anyway and note it — the mask defect is 3% of actions).

```bash
OMP_NUM_THREADS=2 python -m alpha_zero.training_civ6 --seed 42 \
    --board_size 8 --max_turns 60 --num_simulations 12 \
    --num_games 20 --num_iterations 20 --save_interval 1 2>&1 | tee run.log
```

**Resume (added 2026-09-02, after a container restart killed the first attempt at 4.5
minutes).** `--resume_from checkpoints/validation_iteration_one/checkpoint_<N>.pt`
continues at iteration `N+1`; `--num_iterations` stays the end of the run. It restores
network weights only — optimizer moments and the replay buffer are not carried across, so
the first resumed iteration trains on fresh games alone and the artifact must say where
the run was interrupted. Without it, any interruption costs the entire run, which is a
worse property for WS-C (cloud spend) than for WS-L (CPU time). Restoring optimizer state
is deliberate follow-up work, not done here.

Budget, re-derived: self-play 20 games × ~20 s (measured at 8/60, 12 sims, rising as the
policy learns to reach the limit), training ~260 s/iteration once the buffer exceeds the
`batch_size × 100 = 12,800` sample cap (`training_civ6.py:966` — later iterations train on
a fixed-size sample of a growing buffer, which is a design fact worth knowing), evaluation
25–120 s at 20 games: **≈ 3.9–4.4 h**. Keep the checkpoints (they are D0's input). Commit
the summariser output (after E2, or re-parse `run.log` once E2 lands) and the E4 verdict;
`VALIDATION-RUN` ledger event. **Report the result whichever way it goes.**

### WS-D — Own the draw rate

- **D0 (S).** Re-run the sweep table (a) under L1's checkpoints at 2 players, (b) under
  random play at `num_players=3` — the reviewer's n=20 numbers (1.00 / 0.85 at 30 / 60
  turns) need committing at n=60. Random play is a lower bound on expansion; the D1
  decision is made on (a).
- **D1 (S–M, CONFIRM-FIRST: ≥ 9 modules).** One PR that sets every `max_turns` default on
  the arena/cloud/evaluation path to D0's number (prediction: 60), and says in its
  description that the cloud path today has **two** defaults. Sites: `vertex_orchestrator.py:217`
  (**= 50**, forwarded at `:538,646,1182,1283`), `run_cloud_training.sh:37` (= 30),
  `selfplay_worker.py:101` (+ `:640,877,1065`), `arena.py:101`, `evaluation_worker.py:79`,
  `tournament_worker.py:104`, `cli.py:533`, `vertex_jobs.py:981`, `vectorized_env.py:37`,
  `env_factory.py:48`, `inference_benchmark.py:146`, `run_baseline_eval.py:119`,
  `run_encoder_ab.py:102`. Tests asserting 30: `tests/cloud/unit/test_evaluation_worker.py:89`,
  `tests/scripts/test_run_baseline_eval.py:133`, `tests/unit/test_vectorized_env.py:28,562`,
  `tests/unit/test_inference_benchmark.py:196`, `tests/unit/test_environment_tuning.py:494`.
  Keep board 8 — larger boards make it worse at short horizons and re-cost the network by
  `board_size²`. `TrainingConfig` (16/200) is *not* the arena's source; do not touch it.
  Old checkpoints remain loadable (the encoder normalises turn by a constant,
  `civ6_encoder.py:250`); their value heads are simply miscalibrated for the new horizon.
- **D2 (S).** Fix the move mask: add the unit-collision check to
  `core/civ6_actions.py:629-635` so the mask and `_is_valid_move` agree; regression test
  per §3 D2. A bug fix, not a semantics change. Re-run and record the batch-integration
  test's mask-density assertion (`0 < density < 1`, a range, so it should hold).
- **D3 (M, CONFIRM-FIRST, OpenSpec).** Decision item: a secondary tie-break at the turn
  limit (`envs/microciv.py:322-333`) over state that already exists (`num_units`, unit
  health). Changes terminal reward signs for ~30% of 2-player games and most 3-player
  games; ships behind a version flag. **Do not** copy `gameplay/victory.py:323`, which
  breaks exact ties by `max()` and hands player 0 the win. Keep `None` for exact ties.
  Enters Core if D0's trained-policy row stays above 40%, **or** before N2 in any case
  unless D0's 3-player row is ≤ 40%.
- **D4 (L, CONFIRM-FIRST, OpenSpec).** City capture — the only change that makes
  elimination reachable once both players found. Invalidates every checkpoint's value
  function. Not in Core; recorded so D1's ceiling is not mistaken for a fix.

### WS-E — Draw-aware evaluation everywhere (parallel with WS-L and WS-D)

- **E1 (S).** `evaluate_against_random` returns a `ScoreSummary`
  (`evaluation/scoring.py:226`) with a `win_rate` compat property for the two callers
  (`training_civ6.py:1003`); `train()` writes `eval_<iter>.json` next to each checkpoint
  (`:1010-1016`); log a `Seed: N` banner. Then the other §2.2 "no" sites, including the
  four the v1 table missed.
- **E2 (S).** `summarize_training_log.py`: parse the W/D/L line and the seed; add
  score-rate, draw-rate and CI columns; prefer `eval_*.json` when present.
- **E3 (S).** Promote `docs/evidence/validation_iteration_zero/final_checkpoint_vs_random.py`
  to `scripts/probe_checkpoint.py` with `--checkpoint --seed --games --out`, using
  `scoring.summarize()`. Give `run_baseline_eval.py` / `post_run_evaluation.py` a
  `--checkpoint PATH` alternative to `--model_version` and route their rates through
  `scoring.py`.
- **E4 (S).** `scripts/check_learning_trend.py` over E2's JSON: slope of score rate vs
  iteration with a bootstrap CI; exit 0/1, **exit 2 below 3 points**. Replaces
  `validate_convergence.py` for local runs (which stays as the cloud harness, gaining
  `--seed` and draw-awareness when touched).
- **E5 (S, CONFIRM-FIRST: coordinator).** `cloud/gcp/orchestrator_state.py`: add
  `score_rate`/`score_rate_history`; `vertex_orchestrator.py:2100-2103`: log
  `eval_score_rate`, `eval_score_ci_low`; persist `verdict.to_dict()` into the evaluation
  artifact beside `arena.to_dict()`.

### WS-C — Validation Iteration Zero, cloud (NG-7 sign-off; coordinator edits CONFIRM-FIRST)

After L1's verdict is positive (§5). Prerequisites in code (all S): `--promotion_policy`
CLI (`vertex_orchestrator.py:246-255` pattern); a `configs/cloud_validation_capped.yaml`
**that code reads** (today no YAML is), at ≤ 1000 games with
`CIV6AZ_PROMOTION_POLICY=score_rate_lower_bound`, `CIV6AZ_PROMOTION_THRESHOLD=0.52` in the
job env; `--max_games` enforced in the loop and `--max_cost_usd` wired to `CostMonitor`
(an estimate unless `CIV6AZ_COST_BILLING_API_ENABLED`); `value_sign_accuracy` emitted per
iteration into the training artifact; `--seed` on `selfplay_worker` threaded from the
orchestrator; a `tags` field on `ModelMetadata` (`cloud/gcp/storage.py:119-137`) stamped
`pre-value-sign-fix` on v0–v268 by a one-off script and with the commit SHA on new
versions. Then one run **from scratch** (`--resume_from` unset; resuming `latest` would
fine-tune a pre-fix champion trained at a different horizon and the first statistics would
measure that, not the fixed path). Outputs: the artifact, a decision memo on flipping
`DEFAULT_POLICY`, the runbook updated. A resume-from-champion run is a separate, later
decision.

### WS-N — N-player search in self-play

- **N1 (S, 2 modules).** `multiplayer/mcts.py`: replace the four global `np.random.`
  draws (`:338,379,390,393`) with a per-instance `Generator` seeded from config, as the
  2026-08-30 work did for every other variant. `training_civ6.py:607-640`:
  `elif config.num_players > 2` → `NPlayerMCTS(network, NPlayerMCTSConfig(...),
  num_players=..., seed=...)` **without** `_mcts_network` wrapping (it needs the raw
  `(B, N)` head); widen the union at `:475,:610`; same branch on the eval path at `:806`.
  Optional `"nplayer"` entry in `mcts_factory.py:100-107`. Tests per §3 N1.
- **N2 (M).** Gated on D0's 3-player row ≤ 40% (or D3). Note 3-player games are ~1.3×
  longer in actions (the turn advances only on player wrap). The literature this follows
  (Petosa & Balch 2019, arXiv 1910.13012) uses a score vector with mean per-player MSE and
  Max-n backup — what `NPlayerValueHead` + `compute_nplayer_value_loss` + `NPlayerMCTS`
  already implement — and reports that more than 2 players "needs more careful
  hyperparameter tuning"; budget for that.
- Opponent modelling stays out of scope (Tier C keeps `multiplayer/` for this).

### WS-F — Fog of war (after N1; F1 needs its own OpenSpec change)

- **F1 (S–M, OpenSpec).** Compute the action mask from the fogged `game_state`
  (`envs/microciv.py:846-848`) and add the fog path to `get_valid_actions()` (`:868-875`).
  Decide the `found_city`-next-to-hidden-city semantics: the fogged mask is over-permissive
  there, and the executor rejects without clearing moves. Recommended: treat it like D2 —
  the mask must agree with the executor **as seen by the player**, so the executor's
  rejection of a fog-hidden collision must not forfeit the unit's turn. Test per §3 F1.
- **F2 (S).** `--fog_of_war` on `TrainingConfig` → the three `MicroCivEnv(...)` sites
  (`:596,:795,:880`); delete the inert `fog_of_war_enabled` in `orchestrator/config.py` and
  four YAMLs or wire it — do not leave a knob that does nothing.
- **F3 (M, CONFIRM-FIRST: >3 modules).** `DeterminizedMCTS(NPlayerMCTS)` in self-play.
  Fix `WorldSampler._inject_world` first: it does not set `turn`/`current_player` despite
  its docstring, injects units with `id=-1` against live id counters, and is a silent no-op
  on envs without `units`/`cities`. Throughput budget for `num_worlds`.

### WS-B — Batch self-play as a measured path

- **B1 (S).** `benchmark_batch_selfplay.py --out DIR` writing `result.json` + `report.md`
  with a provenance block; `InferenceBenchmark(seed=...)` passing `max_turns` to *both*
  arms and `seed_offset ≠ 0` to the vectorised env; a schema check per §3 B1.
- **B2 (S).** A config **that code reads** with `mcts_type: batch_gumbel`,
  `num_parallel_games: 8`.
- **B3 (S).** Run B1 at the D1 configuration on CPU and (in WS-C) on the T4-spot tier;
  commit `docs/evidence/batch_selfplay/`. Until then the ROADMAP's 5–10× stays "target".

### WS-T — Continuing (parallel, no sequencing)

- Coverage: add the `web` extra to the coverage job (`ci.yml:321`) and un-omit `serving/`,
  `inference/`.
- Tech-debt: Tier B rows, WS3.2/WS5/WS6/WS7 per `kb/specs/20260902-tech-debt-hardening.md`.
  Two Tier B flags from the audit: `cloud/gcp/preemption_handler` is named as deployed in
  `CLAUDE.md` §7 yet has no live importer (its sibling `preemption.py` is also dead — decide
  which is real); `utils/agent_tools` is documented in §9.4/§15.3.
- Promotion-stagnation detection on the deployed path: feed score rates into
  `EloPlateauDetector` from `cloud/gcp/`; currently a documented gap.
- Bridge (M5 prep): `NEXT_STEPS.md` items 1–2 stay queued; item 3 (a trained checkpoint
  live) strictly after WS-C.

---

## 5. Sequencing and estimate

```
day 1:  WS0 (maintainer) ‖ WS-L run (≈4 h CPU, then commit) ‖ WS-D D2 ‖ WS-E E1–E4
week 1: L1 verdict ──► D0 on L1's checkpoints (+ 3-player row)
          │
          ├─ positive slope ──► D1 (confirm-first) ──► WS-C (sign-off) ──► B3
          │                                   └──► N1 ──► [D0 3p ≤ 40% ? N2 : D3 first] ──► F1,F2 ──► F3
          └─ no slope / draws > 40% trained ──► D3 enters Core ──► re-run L1 ──► (as above)
WS-T continuous; B1, B2 any time.
```

Core scope (L1, D0–D2, E1–E5, C1, N1, B1–B2): **~3–4 weeks** to the first committed
learning-trend artifact (week 1) and the first validated cloud run. Full scope adds N2,
F1–F3, B3, D3: **~9–11 weeks**. D4 is out of scope. Single-maintainer estimates at the
cadence PR #113–#116 demonstrated.

---

## 6. Risk register

| Risk | Likelihood | Mitigation / rollback |
|---|---|---|
| L1 shows no trend at 8/60 | possible | That is a result; commit it; D3 enters Core; do not proceed to WS-C. |
| D0 under trained play shows a *higher* draw rate than random (agents learn to turtle) | medium | D3 enters Core; D4 becomes the honest long-term answer. Record either way. |
| 3-player draws stay ≥ 85% (reviewer's number) even at 60 turns | **likely** | N2 is gated; D3 (tie-break) is the fix, not more turns. |
| D1 re-costs every cloud figure ~4× against the 20-turn baseline (~2.4× against 30) | certain | WS-C's cap is in games (enforced) and dollars (estimated); re-estimate from B3's moves/s before sign-off. |
| D2 changes the mask density that `test_batch_selfplay_integration` asserts on | likely | The test asserts a range, not a value; re-run and record. |
| E1's return-type change breaks a caller | low | `win_rate` compat property; two callers, both in `training_civ6.py`. |
| WS-C spend exceeds the cap | low once `--max_games` exists; **high today** (no halting cap at all) | Cap wiring is a prerequisite. The dollar cap is an estimate; say so in the memo. |
| N1's raw `(B, N)` head reaches a scalar consumer | low | Test asserts `CurrentPlayerValueView` is not applied on the N-player path. |
| F1's semantics decision stalls | medium | Ship F2 (flag) with F1's OpenSpec proposal open; do not ship the flag alone as "fog support". |
| `_inject_world` bugs make F3 results uninterpretable | high if skipped | F3's first task is fixing and testing `_inject_world`, not wiring. |

---

## 7. Open PR disposition (15)

| PR | Disposition | Reason |
|---|---|---|
| #117 (this plan) | merge after review | — |
| #95 codecov-action | **close** | codecov removed from CI (`ci.yml:323-331`) |
| #98 action-gh-release | **close** | `release.yml:117` is SHA-pinned (v2.6.2); a v3 bump must be a re-resolved SHA |
| #103 ruff 0.16.2 | **close; redo as one PR touching all five sites** | `pyproject.toml:26`, `requirements.txt:12`, `.pre-commit-config.yaml:30`, `ci.yml:231`, and the **unpinned** `pip install ruff` at `deploy_spaces.yml:53`; CLAUDE.md §5.2 requires the pins to match; expect new rule findings |
| #94/#96/#97 checkout/setup-python/cache | **close; replace with a SHA-pin sweep** | same supply-chain argument already applied to `release.yml` |
| #99 prometheus-client, #100 tensorboard, #101 chromadb | **merge** | clean (stale base only), one line each, `requirements.txt` only (CI installs from `pyproject.toml`) |
| #102 google-cloud-monitoring | **close; delete `requirements.txt:39`** | `cloud/gcp/monitoring.py` was deleted in Tier A; nothing imports it |
| #106 bridge hardening | **maintainer decides: rebase or close** | 7 content conflicts (2 code: `bridge/config.py`, the two Lua files; 5 docs), none from Tier A. Drop its `pytest.ini` (overrides `pyproject.toml`'s pytest config), `.agent/AGENTS.md` (a fourth "agents" location) and the 4,975-line `uv.lock` (default already has one); **review** the silent `heartbeat_timeout_s` 2.0 → 120.0 change in `bridge/config.py` |
| #69 project structure | **close** | deletes `setup.py` already deleted; moves `interfaces/` which a Tier B row decides |
| #49, #32 Strategic Planning SDK | **close** | NG-6 (not a general game-AI framework); 42k/26k-line diffs against a tree 7 months newer |

---

## 8. Gates this plan adds (each merged with a negative control)

1. Move mask ≡ executor under seeded random play (D2); control: drop the collision check.
2. Draw-blind-division regex gate over `alpha_zero/` and `scripts/` (E1); control: plant
   `wins / max(n, 1)` in a scratch module.
3. Log-summariser fixture with both eval lines (E2); control: remove the W/D/L line.
4. Learning-trend checker exit codes 0/1/2 (E4); control: a 2-point fixture must exit 2.
5. One `max_turns` default on the deployed path (D1); control: reintroduce 50 at one site.
6. Search class by `num_players`; seeded `NPlayerMCTS` determinism; no `np.random.` in
   `multiplayer/mcts.py` (N1); control: restore one global draw.
7. Fogged mask reveals nothing about an enemy the fixture proves is hidden (F1).
8. Benchmark `result.json` schema + provenance (B1); control: drop the seed field.
9. A `VALIDATION-RUN` ledger event per committed run (existing K-rules).

---

## 9. Authorizations required (CLAUDE.md §3.2 / charter)

1. **Environment semantics** (D3 tie-break, D4 city capture, F1 `found_city` rule):
   CONFIRM-FIRST; each needs an OpenSpec change. D2 (a bug fix) does not.
2. **> 3 modules** (D1: ≥ 9 modules + 6 tests; F3): CONFIRM-FIRST. N1 (2), F2 (1),
   E1–E4 (≤ 3 each) do not.
3. **Distributed training coordination** (E5, and WS-C's `--max_games`/`--max_cost_usd`
   loop break, seed threading, `ModelMetadata.tags`): CONFIRM-FIRST.
4. **Cloud spend** (WS-C, B3 on Vertex): NG-7 sign-off, recorded in the ledger with both caps.
5. **Maintainer-only**: WS0 items, §7 dispositions, GCS lineage inventory.
6. **Not required**: no network-architecture change, no charter amendment. Nothing here
   touches NG-1..NG-7. WS-L needs nothing.

---

## 10. Literature check (what it supports, what it does not)

- **Draw rate as a training-signal problem** — Joshi 2025 (arXiv 2504.07757,
  search-contempt): in chess self-play the useful regime is (wins+losses)/draws ≈ 1,
  reached by an asymmetric search parameter or by temperature; draw-heavy self-play carries
  little information per game. Supports treating the draw rate as the first-order problem.
  Does **not** transfer directly: MicroCiv's draws are truncation ties, not high-quality
  equal play, so the lever here is the horizon and the tie rule, not contempt.
- **Multiplayer value vectors** — Petosa & Balch 2019 (arXiv 1910.13012): score vector
  targets, mean per-player MSE, Max-n backup; works on 3-player Tic-Tac-Mo/Connect-3x3 and
  reports the need for tuning beyond 2 players. Matches WS-N's design; sets expectations
  for N2.
- **Auxiliary supervision** — Kong, Ram & Yu 2026 (arXiv 2607.08984): oracle-derived
  auxiliary policy loss closes much of the strong-vs-perfect gap on Connect Four and Chomp.
  Not applicable (no oracle for MicroCiv); recorded so nobody proposes it as a draw-rate fix.
- **Information-set search** — MAPLE (arXiv 2605.24139) and the ISMCTS lineage: aggregate
  evaluation over sampled states, which is what `DeterminizedMCTS` does. Supports F3's
  design; its cost model (`num_worlds` × search) is why F3 carries a throughput budget.

---

## 11. Confidence

- [Certain] every count that names a file:line or a command; the sweep table and its
  independent re-run; CI state; PR states; call-site counts; the mask/executor
  disagreement and that every inspected rejection is a unit collision.
- [Likely] `max_turns=60` at board 8 brings the *trained-policy* 2-player draw rate under
  50%. Random play is a lower bound on expansion, so D0 decides.
- [Likely] the WS-L budget of ~4 h; derived from one measured 8/60 game cost and the
  sample cap, and it rises as the policy learns to reach the limit.
- [Likely] the 3-player draw rate will **not** come under 40% from the turn limit alone;
  the reviewer's 0.85 at 60 turns is n=20 and needs D0's n=60.
- [Guessing] whether L1 shows a positive slope at all. That is the point of running it,
  and §5 says what happens if it does not.
- [Guessing] which of D3/D4 the maintainer will accept. That is why they are decision rows
  and not tasks.
