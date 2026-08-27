# Next Steps — The Case (2026-08-27)

> **Method.** This document was produced by scanning the codebase and documentation with
> three parallel verification agents (batch self-play status; fog-of-war and N-player
> gaps; repository health and debt), synthesizing a position, and then submitting the
> draft to an independent adversarial peer review before publication. Every claim below
> carries a `file:line` citation verified on branch head `7b7112b` (2026-08-24).
> Supersedes `docs/NEXT_STEPS_ANALYSIS.md` (2026-08-11), several of whose claims are
> contradicted by the repository (see Finding F7).

---

## Executive Summary

**Recommendation: pause M4 feature work for roughly two weeks and run a
"Validation Iteration Zero" — a small, budget-capped, end-to-end training run on the
fixed code path — preceded by four small correctness fixes it depends on. Then resume
M4 features in dependency order (N-player MCTS wiring first, fog of war second).**

The reason is simple: on 2026-08-24/25 the project discovered that **every one of its
269 promoted model versions was trained on partially corrupted data** (value targets
mislabelled from the wrong player's perspective, `CLAUDE.md` §5.2 2026-08-25) **and
promoted through a gate that scores draws as losses** (`alpha_zero/cloud/gcp/arena.py:41`).
The fixes are merged but **no training run has validated them**. Meanwhile this review
found a third, undocumented defect of the same family: the batch self-play path drops
`action_masks` from trajectories, silently disabling masked policy loss for all
batch-generated data (Finding F1). Accelerating self-play throughput before validating
the fixed loop would multiply the rate at which invalid data is produced.

---

## Verified Findings (new evidence, not previously documented)

| # | Finding | Evidence | Severity |
|---|---------|----------|----------|
| F1 | **Batch self-play drops `action_masks`.** `run_batch_selfplay` initializes trajectories without the key (`alpha_zero/cloud/gcp/selfplay_worker.py:1044-1047`); the trainer silently substitutes an all-ones mask (`alpha_zero/cloud/gcp/training_worker.py:586-589`), so batch-generated samples train with masked cross-entropy disabled. The parity test that should catch it excludes `action_masks` from its asserted key set and is vacuous on empty output (`tests/integration/test_batch_selfplay_integration.py:268-271`). | verified in code | **High** — silent training-quality regression on the flagship M4 feature |
| F2 | **Orchestrator defaults silently fall through to sequential self-play.** Default `--num_parallel_games=4` (`alpha_zero/cloud/gcp/vertex_orchestrator.py:214-217`) with default `--mcts_type=gumbel` (`:195`) fails the batch dispatch gate (`selfplay_worker.py:1277-1279`) with no warning: a run that *looks* batch-configured runs sequential. | verified in code | Medium |
| F3 | **The promotion-policy registry is wired into only one of two decision sites.** `vertex_orchestrator.py:835-855` consults `PromotionVerdict`; `alpha_zero/cloud/evaluator.py:396` remains a bare `win_rate >= threshold` that never consults the registry (same raw statistic in the early-stop check at `:321`). Nothing in any config, runbook, or workflow opts into `score_rate`. | verified in code | Medium-High |
| F4 | **Fog-of-war search is structurally blocked on N-player MCTS.** `DeterminizedMCTS.search` requires its base search to return a per-player value dict (`alpha_zero/multiplayer/determinization.py:248,269`); only `NPlayerMCTS.search` does (`alpha_zero/multiplayer/mcts.py:78-82`) — `Civ6MCTS`/`GumbelMCTS` return scalars (`alpha_zero/core/civ6_mcts.py:512-516`). CLAUDE.md §13 item 2 presents these as independent work items; they are sequential. | verified in code | Medium (planning) |
| F5 | **The fog action-mask leak is confirmed and standalone.** The encoder receives the fogged state while the action mask is recomputed from a fresh unmasked state (`alpha_zero/envs/microciv.py:839-848`, comment: "action mask uses full state"); `get_valid_actions()` has no fog path (`:868-875`). Additionally, the `fog_of_war_enabled: true` knob in `orchestrator/config.py:403` and three YAMLs is dead config — no consumer constructs an env from it. | verified in code | Medium (latent — fog is default-off and unexposed from `training_civ6.py`) |
| F6 | **The 85% coverage gate measures roughly half the codebase.** 139 of 316 real modules (~48% of LOC) are omitted (`pyproject.toml` `[tool.coverage.run]`, 49 patterns), including all of `alpha_zero/cloud/` — the package where F1–F3 and both 2026-08 production bugs live. Root cause: every CI job installs only `.[dev]` (`.github/workflows/ci.yml:39,81,114,133,248`), so `web`/`gcp`/`distributed` extras can never import in the coverage job. | verified in config | High (systemic) |
| F7 | **Both forward-looking docs are stale.** `docs/NEXT_STEPS_ANALYSIS.md` (2026-08-11) claims "100% mypy --strict" (CI runs mypy *without* `--strict`, `ci.yml:49`) and "85% enforced in Makefile" (`Makefile:120` is `--cov-fail-under=80`), and recommends scaling work while unaware of the correctness bugs found two weeks later. `ROADMAP.md:56` still lists the PR #72 champion-load fix as pending; it landed under a different mechanism (`--resume_from=latest`, `vertex_orchestrator.py:428-495,616-620`). CLAUDE.md's skip-count ("11 sites / 17 tests") is stale against 92 present sites (mostly legitimate environment gates). | verified | Medium |
| F8 | **Flagship claims have no committed evidence artifacts.** `docs/evidence/` contains exactly one study (`hexcrystal_ab/`). The 5–10× batch speedup is aspiration (`docs/prd/prd-batch-selfplay.md:7`), not measurement — a benchmark harness exists (`scripts/benchmark_batch_selfplay.py`) but writes no artifact. The v1.8.0 live-bridge milestone is 4 turns, END_TURN-only, on random weights, recorded as prose only (`NEXT_STEPS.md:9-11,38-42,55-58`). | verified | Medium |
| F9 | **Checkpoint lineage v0–v268 predates the value-sign fix.** `0d5adee` (2026-08-24) is on the default branch but no run, evidence entry, or ledger event validates it. `NEXT_STEPS.md:55-58` recommends demoing a trained checkpoint from that lineage — which would demo a model trained on partially inverted labels and promoted by the draw-hostile gate. | verified | High (sequencing) |

---

## Thesis (starting argument): "Execute M4 as written — ship features"

The roadmap is explicit and the code is close. Batch self-play is ~70% wired with real
end-to-end tests (`selfplay_worker.py:953,1011,1020`; `test_batch_selfplay_integration.py`);
the N-player training pipeline landed 2026-07-05 and the remaining work (Max-N search
wiring) is a genuine integration job over components that are already written and
individually tested (`NPlayerMCTS` 393 lines, `NPlayerEnvAdapter` 244 lines). Fog of war
has an implemented env flag, a tested `DeterminizedMCTS`, and a named list of wiring
tasks. The project's momentum since 2026-08-01 (46 commits: HF Hub publishing, encoder
A/B with +31 Elo evidence, correctness sweep) shows the team ships. The fastest route to
M5 is to finish M4's three feature bullets in order and keep moving. Throughput first,
because every later experiment — N-player runs, fog runs, convergence gates — gets
5–10× cheaper once batch self-play is the production path.

## Counter-argument: "Stop. The loop is broken; features multiply the damage"

The thesis assumes the training loop produces valid data and valid promotions. The
evidence says it did neither, and may still not:

1. **The data was wrong.** All three self-play paths mislabelled ~half of every
   two-player trajectory's value targets until 2026-08-24 (CLAUDE.md §5.2). All 269
   promoted models were trained on that data. The fix is unvalidated by any run (F9).
2. **The gate was wrong.** Promotion scored draws as losses, capping the statistic at
   `1 − draw_rate` (CLAUDE.md §5.3); the corrected policy exists but is default-off,
   opted into by nothing, and one of the two decision sites bypasses the registry
   entirely (F3).
3. **The next feature is wrong too.** Batch self-play — the throughput bullet the
   thesis wants to ship first — currently produces trajectories that disable masked
   policy loss (F1), and the orchestrator's defaults silently don't run it anyway (F2).
4. **The watchdog isn't watching.** Every one of these defects lives in
   `alpha_zero/cloud/`, which the 85% coverage gate omits wholesale (F6). The gate that
   is supposed to catch this class of bug structurally cannot.

Shipping 5–10× throughput into this loop means generating invalid training data 5–10×
faster, promoting models through a biased gate, and demoing checkpoints trained on
corrupted labels (F9). The counter-position: declare a correctness freeze — no feature
work until the coverage omit list is emptied, both promotion sites are unified, and a
convergence run validates the fixes.

## Rebuttal (a third perspective): the unit of value is neither features nor hygiene — it is *validated evidence*

Both positions above optimize a proxy. The thesis optimizes shipped code; the
counter-argument optimizes process purity. This project's charter defines its actual
differentiator differently: *"engineered like a product … typed, tested, gated, and
observable"* (`PROJECT_CHARTER.md` §1), with skill that "demonstrably compounds across
training generations." The binding constraint on that vision today is not missing
features and not missing hygiene — it is that **the project cannot currently
demonstrate that its training loop works**, because its flagship claims have no
artifacts (F8) and its two August correctness fixes have no validating run (F9). A
full correctness freeze is also wrong: emptying a 49-pattern omit list and refactoring
`cloud/` to importability is weeks of work that produces no evidence either, and the
standing `$28.55-for-zero-promotions` incident (NG-7 rationale) shows the cost of
running blind is real but bounded — the cost of *never running* is unbounded.

The synthesis is a single, small, instrumented end-to-end run — **Validation Iteration
Zero** — that converts the dichotomy into a sequencing answer. One budget-capped run on
the fixed code path simultaneously: (a) re-runs the convergence gate over the
value-sign and champion-load fixes, closing F9; (b) reports `score_rate` alongside
`legacy_win_rate` in shadow mode, producing the evidence needed to flip the promotion
default deliberately rather than by fiat (F3); (c) benchmarks batch vs sequential
self-play and commits the artifact `docs/evidence/` has been missing (F8, F1-dependent).
Only four small fixes block this run, and all are days, not weeks. Features then resume
on a foundation the project can *prove* works — which is also the honest precondition
for the thesis's own goal, since an N-player or fog-of-war run on an unvalidated loop
would be uninterpretable.

---

## Recommended Plan

### Phase 1 — Unblock (≈1 week of code, all small)

| Item | Size | Closes |
|------|------|--------|
| Emit `action_masks` from `run_batch_selfplay` (mirror the sequential path at `selfplay_worker.py:667`); tighten the parity test to true key-set equality and non-vacuity | S (~15 lines + test) | F1 |
| Wire the promotion registry into `cloud/evaluator.py:396` (and the early-stop check at `:321`) so both decision sites consult one policy | S | F3 |
| Warn (or auto-promote `gumbel`→`batch_gumbel`) when `num_parallel_games>1` with a non-batch `mcts_type`; set `mcts_type: batch_gumbel` in one Vertex config | S | F2 |
| Fix the fog mask leak: compute the action mask from the same fogged `game_state` (`microciv.py:846-848`) and add a fog path to `get_valid_actions()`; delete or wire the dead `fog_of_war_enabled` config knob | S–M (semantics decision: legality of actions targeting unseen tiles) | F5 |
| Doc-sync: correct `ROADMAP.md:56`, retire `docs/NEXT_STEPS_ANALYSIS.md` to `docs/archive/`, refresh CLAUDE.md skip-count and the 2026-08-25 date | XS | F7 |

### Phase 2 — Validation Iteration Zero (≈1 week wall-clock; **requires explicit human sign-off per NG-7 before any cloud spend**)

- Budget-capped Vertex run (suggested cap: the T4-Spot tier, ≤ $25) on the fixed path:
  convergence gate re-run, `value_sign_accuracy` tracked (`training_worker.py:754`),
  `score_rate` shadow-reported next to `legacy_win_rate`.
- Batch-vs-sequential benchmark via `scripts/benchmark_batch_selfplay.py`, committed to
  `docs/evidence/batch_selfplay/` (games/hour, per-move latency, sample-quality parity).
- Outcome artifacts: one evidence directory, one ledger event, a decision memo on
  flipping `CIV6AZ_PROMOTION_POLICY` default. Tag pre-fix checkpoints (v0–v268) as
  `pre-value-sign-fix` in lineage metadata; do not demo them live (F9).

### Phase 3 — Resume M4 features, in dependency order (≈3–5 weeks)

1. **N-player MCTS wiring (Gap B first — it is the prerequisite).** Branch
   `run_self_play` on `num_players > 2` to build `NPlayerMCTS`; one absolute→relative
   value-order conversion against `relative_value_vector`
   (`training_civ6.py:295-305`); drop `CurrentPlayerValueView` from the N-player path
   and the knowingly-incorrect scalar fallback (`core/civ6_mcts.py:536-552`); expose
   `--search_strategy`; then a published 3–4 player run in `docs/evidence/`.
   Opponent-model integration is the largest, least-specified piece — schedule it as
   its own follow-up, not inside this item. (~1.5–2.5 weeks)
2. **Fog of war (Gap A — downstream of B by F4).** `--fog_of_war` flag through
   `TrainingConfig` and all three env-construction sites plus `env_factory.py:129`;
   then `DeterminizedMCTS` wiring over `NPlayerMCTS`, with `WorldSampler._inject_world`
   validation (it reconstructs units/cities but not terrain or id counters —
   `determinization.py:278-315`) and a throughput budget for `num_worlds`. (~1.5–3 weeks)

### Continuous (parallel, low-cost)

- **Coverage:** add the `web` extra to CI's coverage job install (one line in
  `ci.yml:248`) and un-omit `serving/`+`inference/` — the cheapest omit-list shrink;
  treat the `cloud/` un-omit as its own later milestone.
- **CI hardening:** add `timeout-minutes` to all jobs (none set today; a hang burns the
  6-hour default); consider deduplicating the coverage job's full re-run of suites the
  `test` matrix already executed.
- **Bridge (M5 prep):** keep `NEXT_STEPS.md` items 1–2 (sustained-turn validation,
  end-turn robustness) queued, but sequence item 3 (load a trained checkpoint) strictly
  after Phase 2's retrain per F9.

---

## Risks and governance notes

- **NG-7:** Phase 2's cloud run requires explicit human confirmation; the plan caps it
  at the budget tier and defines its stop condition (artifacts produced) up front.
- **Charter fit:** nothing here touches NG-1..NG-6; the promotion-default flip is
  deliberately deferred to a decision memo backed by Phase 2 data, preserving the
  documented "default unchanged deliberately" stance (CLAUDE.md §5.3).
- **Known limitation:** this review did not re-run the test suite or any training; all
  findings are static-analysis-verified against `7b7112b`. F1's training-quality impact
  is inferred from the loss implementation, not measured — Phase 2 measures it.
